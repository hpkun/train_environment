"""Diagnose UAV launch accounting: why Red fired=0 despite improved MAV survival."""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_diagnostic(model_path, config_path, reward_mode, episodes, max_steps,
                   device_str, opponent_policy, output_dir, stochastic):
    import torch
    from uav_env import make_env
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
    from algorithms.happo.tam_categorical_recurrent_policy import TAMCategoricalRecurrentHAPPOPolicy
    from algorithms.mappo.opponent_policy import OpponentPolicy

    device = torch.device(device_str)
    adapter = HeteroObsAdapterV2()

    # Load policy
    meta_path = Path(model_path).parent / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    policy = TAMCategoricalRecurrentHAPPOPolicy(
        entity_dim=int(meta.get("entity_dim", 19)),
        actor_obs_dim=int(meta.get("actor_obs_dim", 96)),
        critic_state_dim=int(meta.get("critic_state_dim", 480)),
        action_dim=int(meta.get("action_dim", 4)),
        action_levels=int(meta.get("action_levels", 40)),
        rnn_hidden_size=int(meta.get("rnn_hidden_size", 128)),
    ).to(device)
    policy.load(model_path, map_location=device)
    policy.eval()

    opponent = OpponentPolicy(mode=opponent_policy, seed=2000)

    totals = {
        "red_fired_agent_info": 0, "blue_fired_agent_info": 0,
        "red_launches_diag": 0, "blue_launches_diag": 0,
        "red_hits_term_delta": 0, "blue_hits_term_delta": 0,
        "red_hits_quality_done": 0, "blue_hits_quality_done": 0,
    }
    launch_gates_red = Counter()
    launch_gates_blue = Counter()
    per_agent_stats = {f"red_{i}": {
        "min_distances": [], "steps_range_ok": 0, "steps_ao_ok": 0, "steps_ta_ok": 0,
        "steps_geometry_ok": 0, "actions": [], "tam_uav_distance_max": -999,
        "tam_uav_angle_max": -999, "reward_total_max": -999,
    } for i in range(1, 3)}

    for ep in range(episodes):
        env = make_env(str(ROOT / config_path) if not Path(config_path).is_absolute() else config_path,
                       env_type="jsbsim_hetero", hetero_reward_mode=reward_mode, max_steps=max_steps)
        red_ids = env.red_ids
        roles = [0 if env.agent_roles.get(rid) == "mav" else 1 for rid in red_ids]
        rnn_h = np.zeros((len(red_ids), 128), dtype=np.float32)
        obs, info = env.reset(seed=2000 + ep * 100)
        prev_hits = {"red": 0, "blue": 0}
        ep_len = 0

        for step in range(max_steps):
            adapted = adapter.adapt_all(obs, info=info, red_ids=red_ids, blue_ids=env.blue_ids)
            ao = np.stack([adapted["actor_obs"].get(rid, np.zeros(96, dtype=np.float32)) for rid in red_ids])
            with torch.no_grad():
                out = policy.act(torch.as_tensor(ao, device=device), roles=roles,
                                deterministic=not stochastic,
                                rnn_hidden=torch.as_tensor(rnn_h, device=device))
            actions = out["action"].cpu().numpy()
            rnn_h = out["rnn_hidden"].cpu().numpy()
            act_dict = {rid: actions[i].astype(np.int64) for i, rid in enumerate(red_ids)}
            act_dict.update(opponent.act(obs, env.blue_ids, env=env))
            obs, rewards, term, trunc, info = env.step(act_dict)
            ep_len += 1

            # Agent info
            for rid in red_ids:
                ai = info.get(rid, {})
                if isinstance(ai, dict):
                    totals["red_fired_agent_info"] += int(ai.get("missiles_fired_this_step", 0))
            for bid in env.blue_ids:
                ai = info.get(bid, {})
                if isinstance(ai, dict):
                    totals["blue_fired_agent_info"] += int(ai.get("missiles_fired_this_step", 0))

            # Launch diag
            ld = info.get("__launch_diag__", {})
            for team, counter in ld.items():
                for key in counter:
                    if team == "red":
                        launch_gates_red[key] += int(counter[key])
                    else:
                        launch_gates_blue[key] += int(counter[key])

            # Missile term
            mt = info.get("__missile_term__", {})
            if isinstance(mt, dict):
                red_h = int(mt.get("red", {}).get("hit", 0))
                blue_h = int(mt.get("blue", {}).get("hit", 0))
                totals["red_hits_term_delta"] += max(red_h - prev_hits["red"], 0)
                totals["blue_hits_term_delta"] += max(blue_h - prev_hits["blue"], 0)
                prev_hits["red"] = red_h
                prev_hits["blue"] = blue_h

            # Per-agent range/AO/TA for red_1, red_2
            for i, rid in enumerate(["red_1", "red_2"]):
                sim = env.red_planes.get(rid)
                if sim is None or not sim.is_alive:
                    continue
                red_features = _make_feature(sim)
                for blue_sim in env.blue_planes.values():
                    if not blue_sim.is_alive:
                        continue
                    blue_features = _make_feature(blue_sim)
                    from uav_env.JSBSim.utils import get2d_AO_TA_R
                    ao_val, ta_val, distance = get2d_AO_TA_R(red_features, blue_features)
                    per_agent_stats[rid]["min_distances"].append(distance)
                    if distance < 10000:
                        per_agent_stats[rid]["steps_range_ok"] += 1
                    if ao_val < np.deg2rad(45):
                        per_agent_stats[rid]["steps_ao_ok"] += 1
                    if ta_val > np.pi / 2:
                        per_agent_stats[rid]["steps_ta_ok"] += 1
                    if distance < 10000 and ao_val < np.deg2rad(45) and ta_val > np.pi/2:
                        per_agent_stats[rid]["steps_geometry_ok"] += 1
                per_agent_stats[rid]["actions"].append(actions[i + 1].tolist())

            if all(term.values()) or all(trunc.values()):
                break

        env.close()

    # Summarize
    red_launches_raw = launch_gates_red.get("launches", 0)
    blue_launches_raw = launch_gates_blue.get("launches", 0)

    # Per-agent summaries
    for rid in ["red_1", "red_2"]:
        s = per_agent_stats[rid]
        s["min_distance_mean"] = float(np.mean(s["min_distances"])) if s["min_distances"] else 99999
        s["min_distance_min"] = float(np.min(s["min_distances"])) if s["min_distances"] else 99999
        if s["actions"]:
            arr = np.array(s["actions"])
            s["dominant_throttle"] = int(np.bincount(arr[:, 0]).argmax()) if len(arr) else -1
            s["dominant_elevator"] = int(np.bincount(arr[:, 2]).argmax()) if len(arr) else -1
            s["dominant_rudder"] = int(np.bincount(arr[:, 3]).argmax()) if len(arr) else -1

    return {
        "checkpoint": model_path, "reward_mode": reward_mode,
        "episodes": episodes, "stochastic": stochastic,
        "totals": totals,
        "launch_gates_red": dict(launch_gates_red),
        "launch_gates_blue": dict(launch_gates_blue),
        "per_agent": per_agent_stats,
        "fired_hit_consistent": (
            totals["red_fired_agent_info"] == red_launches_raw
            and totals["blue_fired_agent_info"] == blue_launches_raw
        ),
    }


def _make_feature(sim):
    pos = sim.get_position()
    vel = sim.get_velocity()
    return np.array([pos[0], pos[1], -pos[2], vel[0], vel[1], -vel[2]], dtype=np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml")
    p.add_argument("--model", required=True)
    p.add_argument("--reward-mode", default="tam_paper_reward_v1")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--opponent-policy", default="tam_direct_fsm")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--output-dir", default="outputs/tam_paper_reward_v1_launch_accounting")
    p.add_argument("--stochastic-eval", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir.startswith(str(ROOT)) else ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = "stoch" if args.stochastic_eval else "det"
    print(f"Launch accounting: {args.episodes}ep {mode} on {args.model}", flush=True)
    report = run_diagnostic(args.model, args.config, args.reward_mode,
                           args.episodes, args.max_steps, args.device,
                           args.opponent_policy, str(out_dir), args.stochastic_eval)

    (out_dir / "launch_accounting.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    t = report["totals"]
    lg_r = report["launch_gates_red"]
    lines = ["# Launch Accounting Diagnostic", "",
             f"Checkpoint: {report['checkpoint']}",
             f"Mode: {mode}, Episodes: {report['episodes']}",
             f"Reward: {report['reward_mode']}", "",
             "## Totals", "",
             f"red_fired_agent_info: {t['red_fired_agent_info']}",
             f"blue_fired_agent_info: {t['blue_fired_agent_info']}",
             f"red_launches_diag: {t['red_launches_diag']}",
             f"blue_launches_diag: {t['blue_launches_diag']}",
             f"red_hits_term_delta: {t['red_hits_term_delta']}",
             f"blue_hits_term_delta: {t['blue_hits_term_delta']}",
             f"red_hits_quality: {t['red_hits_quality_done']}",
             f"blue_hits_quality: {t['blue_hits_quality_done']}",
             f"fired_hit_consistent: {report['fired_hit_consistent']}", "",
             "## Red Launch Gates", ""]
    for k in ["alive_shooters", "alive_enemy_pairs", "unengaged_enemy_pairs",
              "range_ok_pairs", "ao_ok_pairs", "ta_ok_pairs", "geometry_ok_pairs",
              "lock_started", "lock_continued", "lock_lost", "lock_mature_pairs",
              "cooldown_blocked", "kill_cooldown_blocked", "engaged_blocked", "launches"]:
        lines.append(f"- {k}: {lg_r.get(k, 0)}")
    lines.append("")
    for rid in ["red_1", "red_2"]:
        s = report["per_agent"][rid]
        lines.append(f"## {rid}")
        lines.append(f"- min_distance: {s['min_distance_min']:.0f}m (mean {s['min_distance_mean']:.0f}m)")
        lines.append(f"- steps_range_ok: {s['steps_range_ok']}")
        lines.append(f"- steps_ao_ok: {s['steps_ao_ok']}")
        lines.append(f"- steps_ta_ok: {s['steps_ta_ok']}")
        lines.append(f"- steps_geometry_ok: {s['steps_geometry_ok']}")
        lines.append(f"- dominant_throttle: {s.get('dominant_throttle', '?')}")
        lines.append(f"- dominant_elevator: {s.get('dominant_elevator', '?')}")
        lines.append(f"- dominant_rudder: {s.get('dominant_rudder', '?')}")
    (out_dir / "launch_accounting.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
