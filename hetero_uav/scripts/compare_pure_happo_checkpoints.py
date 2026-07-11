"""Deterministic paired evaluation: random-init vs trained checkpoint.

Compares a trained checkpoint against a same-architecture random-init baseline
using identical episode seeds, opponent, and environment config.
"""
from __future__ import annotations

import argparse, csv, json, math, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_happo_reference import (
    _load_meta, _build_policy_from_meta, _role_ids, _team_done,
    _alive_counts, _env_type_override_kwargs, _update_missile_stats,
    _empty_stats,
)
from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
from algorithms.happo.rollout_safety import sanitize_policy_inputs, zero_inactive_actions, zero_inactive_hidden
from algorithms.mappo.opponent_policy import OpponentPolicy
from uav_env import make_env


def _safe(v, default=0.0):
    try: x = float(v); return x if math.isfinite(x) else default
    except: return default


def _alive_attack_distances(env) -> list[float]:
    attack = [
        sim for rid, sim in env.red_planes.items()
        if env.agent_roles.get(rid) == "attack_uav" and sim is not None and sim.is_alive
    ]
    blue = [sim for sim in env.blue_planes.values() if sim is not None and sim.is_alive]
    if not attack or not blue:
        return []
    nearest = []
    for red in attack:
        nearest.append(min(
            float(np.linalg.norm(np.asarray(red.get_position()) - np.asarray(target.get_position())))
            for target in blue
        ))
    return nearest


def _event_key(record: dict, fallback_step: int) -> tuple:
    missile_id = record.get("missile_id")
    if missile_id not in (None, ""):
        return ("missile", str(record.get("shooter_id", "")), str(missile_id))
    return (
        "fallback", int(fallback_step), str(record.get("shooter_id", "")),
        str(record.get("target_id", "")), str(record.get("event_type", "")),
    )


def _gate_flags(info: dict) -> tuple[bool, bool, bool, bool]:
    records = info.get("__launch_gate_diagnostics__", []) if isinstance(info, dict) else []
    attack_records = [
        row for row in records
        if isinstance(row, dict) and row.get("role") == "attack_uav"
        and str(row.get("agent_id", "")).startswith("red_")
    ]
    return (
        any(int(row.get("any_range_pass", 0) or 0) > 0 for row in attack_records),
        any(int(row.get("any_ata_pass", 0) or 0) > 0 for row in attack_records),
        any(int(row.get("any_ta_pass", 0) or 0) > 0 for row in attack_records),
        any(int(row.get("any_geometry_pass", 0) or 0) > 0 for row in attack_records),
    )


def _blue_death_record(event: dict, done_records: list[dict]) -> dict:
    aid = str(event.get("agent_id", ""))
    missile = next((
        row for row in done_records
        if str(row.get("target_id", "")) == aid
        and str(row.get("termination_reason", row.get("raw_termination_reason", ""))) == "hit"
    ), None)
    shooter = str((missile or {}).get("shooter_id", event.get("missile_owner", "")) or "")
    reason = str(event.get("death_reason", "") or "")
    return {
        "step": int(event.get("step", -1) or -1),
        "blue_agent_id": aid,
        "death_reason": reason,
        "killer_id": shooter,
        "shooter_id": shooter,
        "missile_id": str((missile or {}).get("missile_id", "") or ""),
    }


def _evaluate_model(policy, env, config, args, adapter, device, model_label, rich_logger=None):
    """Run deterministic episodes and collect per-episode metrics."""
    episodes = []
    opponent = OpponentPolicy(mode=args.opponent_policy, seed=args.seed_start + 999)
    roles = _role_ids(env)
    _rnn_hidden_size = getattr(policy, "rnn_hidden_size", 0)

    episode_seeds = list(args.seeds) if getattr(args, "seeds", None) else [
        args.seed_start + ep_idx for ep_idx in range(args.episodes)
    ]
    for seed in episode_seeds:
        # Missile hit sampling uses NumPy's module-level RNG. Reset it for each
        # paired episode so model evaluation order cannot change the seed result.
        np.random.seed(int(seed))
        obs, info = env.reset(seed=seed)
        opponent.reset_memory()

        ep_ret = 0.0
        ep_len = 0
        mstats = _empty_stats()
        prev_hits = {"red": 0, "blue": 0}
        rnn_hidden = np.zeros((len(env.red_ids), _rnn_hidden_size), dtype=np.float32) if _rnn_hidden_size > 0 else None
        gate_stats = {
            "range_gate_decisions": 0, "ao_gate_decisions": 0,
            "ta_gate_decisions": 0, "full_window_decisions": 0,
            "first_window_step": -1, "first_launch_step": -1,
            "first_hit_step": -1, "valid_decisions": 0,
            "min_distances": [], "nearest_distances": [],
            "red_launches": 0, "red_hits": 0,
            "blue_launches": 0, "blue_hits": 0,
            "launch_keys": set(), "hit_keys": set(),
            "blue_launch_keys": set(), "blue_hit_keys": set(),
            "blue_death_records": [],
        }
        terminated = {aid: False for aid in env.agent_ids}
        truncated = {aid: False for aid in env.agent_ids}

        while True:
            adapted = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
            active = np.zeros(len(env.red_ids), dtype=np.float32)
            for i, rid in enumerate(env.red_ids):
                agent_info = info.get(rid, {}) if isinstance(info, dict) else {}
                if isinstance(agent_info, dict) and "alive" in agent_info:
                    alive = bool(agent_info["alive"])
                else:
                    sim = env.red_planes.get(rid)
                    alive = bool(sim is not None and sim.is_alive)
                active[i] = 1.0 if alive else 0.0
            actor_obs_np = np.stack([adapted["actor_obs"].get(rid, np.zeros(adapter.flat_actor_obs_dim, dtype=np.float32)) for rid in env.red_ids])
            critic_np = adapted["critic_state"]
            san = sanitize_policy_inputs(actor_obs_np, active, critic_state=critic_np, rnn_hidden=rnn_hidden)
            act_kwargs = {}
            if rnn_hidden is not None:
                act_kwargs["rnn_hidden"] = torch.as_tensor(san.get("rnn_hidden", rnn_hidden), device=device)
            with torch.no_grad():
                out = policy.act(
                    torch.as_tensor(san["actor_obs"], device=device), roles=roles,
                    critic_state=torch.as_tensor(san.get("critic_state", critic_np), device=device),
                    deterministic=True, **act_kwargs)
            if rnn_hidden is not None and "rnn_hidden" in out:
                rnn_hidden = zero_inactive_hidden(out["rnn_hidden"].detach().cpu().numpy(), active)
            actions_np = zero_inactive_actions(out["action"].detach().cpu().numpy(), active)
            action_dict = {rid: actions_np[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}
            action_dict.update(opponent.act(obs, env.blue_ids, env=env))

            nearest_distances = _alive_attack_distances(env)
            valid_decision = bool(nearest_distances)
            if valid_decision:
                gate_stats["valid_decisions"] += 1
                gate_stats["nearest_distances"].extend(nearest_distances)
                gate_stats["min_distances"].append(min(nearest_distances))

            obs, rewards, terminated, truncated, info = env.step(action_dict)
            _update_missile_stats(mstats, info, env, prev_hits)

            any_range_ok, any_ao_ok, any_ta_ok, full_window = _gate_flags(info)
            if valid_decision and any_range_ok:
                gate_stats["range_gate_decisions"] += 1
            if valid_decision and any_ao_ok:
                gate_stats["ao_gate_decisions"] += 1
            if valid_decision and any_ta_ok:
                gate_stats["ta_gate_decisions"] += 1
            if valid_decision and full_window:
                gate_stats["full_window_decisions"] += 1
                if gate_stats["first_window_step"] < 0:
                    gate_stats["first_window_step"] = ep_len

            launch_records = info.get("__launch_quality_step__", []) if isinstance(info, dict) else []
            done_records = info.get("__launch_quality_done__", []) if isinstance(info, dict) else []
            for record in launch_records:
                if not isinstance(record, dict):
                    continue
                key = _event_key(record, ep_len)
                if str(record.get("shooter_id", "")).startswith("red_"):
                    gate_stats["launch_keys"].add(key)
                    if gate_stats["first_launch_step"] < 0:
                        gate_stats["first_launch_step"] = ep_len
                else:
                    gate_stats["blue_launch_keys"].add(key)
            for record in done_records:
                if not isinstance(record, dict):
                    continue
                reason = str(record.get("termination_reason", record.get("raw_termination_reason", "")))
                if reason != "hit":
                    continue
                key = _event_key(record, ep_len)
                if str(record.get("shooter_id", "")).startswith("red_"):
                    gate_stats["hit_keys"].add(key)
                    if gate_stats["first_hit_step"] < 0:
                        gate_stats["first_hit_step"] = ep_len
                else:
                    gate_stats["blue_hit_keys"].add(key)
            for event in info.get("death_events", []) if isinstance(info, dict) else []:
                if isinstance(event, dict) and str(event.get("agent_id", "")).startswith("blue_"):
                    gate_stats["blue_death_records"].append(_blue_death_record(event, done_records))

            ep_ret += sum(float(rewards.get(rid, 0.0)) for rid in env.red_ids)
            ep_len += 1

            if _team_done(terminated, truncated):
                break

        ra, ba = _alive_counts(env)
        if ba == 0 and ra > 0:
            winner, end_reason = "red", "blue_eliminated"
        elif ra == 0 and ba > 0:
            winner, end_reason = "blue", "red_eliminated"
        elif ra == 0 and ba == 0:
            winner, end_reason = "draw", "mutual_elimination"
        elif all(truncated.values()):
            winner, end_reason = "draw", "timeout"
        else:
            winner, end_reason = "draw", "ongoing"

        num_red = len(env.red_ids)
        num_blue = len(env.blue_ids)
        red_dead = num_red - ra
        blue_dead = num_blue - ba

        min_dist = float(np.min(gate_stats["min_distances"])) if gate_stats["min_distances"] else float("nan")
        mean_nearest = float(np.mean(gate_stats["nearest_distances"])) if gate_stats["nearest_distances"] else float("nan")
        gate_denom = max(gate_stats["valid_decisions"], 1)
        blue_missile_deaths = sum(record["death_reason"] == "missile_hit" for record in gate_stats["blue_death_records"])
        blue_crash_deaths = sum("crash" in record["death_reason"].lower() for record in gate_stats["blue_death_records"])
        blue_other_deaths = len(gate_stats["blue_death_records"]) - blue_missile_deaths - blue_crash_deaths
        episodes.append({
            "model_label": model_label, "episode_seed": seed,
            "episode_return": ep_ret, "episode_length": ep_len,
            "winner": winner, "end_reason": end_reason,
            "red_alive_final": ra, "blue_alive_final": ba,
            "mav_alive_final": int(bool(env.red_planes.get("red_0") and env.red_planes["red_0"].is_alive)),
            "red_loss_fraction": red_dead / max(num_red, 1),
            "blue_loss_fraction": blue_dead / max(num_blue, 1),
            "red_launch_count": len(gate_stats["launch_keys"]),
            "red_hit_count": len(gate_stats["hit_keys"]),
            "blue_launch_count": len(gate_stats["blue_launch_keys"]),
            "blue_hit_count": len(gate_stats["blue_hit_keys"]),
            "minimum_red_to_blue_distance_m": min_dist,
            "mean_nearest_target_distance_m": mean_nearest,
            "range_gate_valid_decisions": gate_stats["range_gate_decisions"],
            "ao_gate_valid_decisions": gate_stats["ao_gate_decisions"],
            "ta_gate_valid_decisions": gate_stats["ta_gate_decisions"],
            "full_launch_window_decisions": gate_stats["full_window_decisions"],
            "launch_gate_valid_decisions": gate_stats["valid_decisions"],
            "range_gate_occupancy": gate_stats["range_gate_decisions"] / gate_denom,
            "ao_gate_occupancy": gate_stats["ao_gate_decisions"] / gate_denom,
            "ta_gate_occupancy": gate_stats["ta_gate_decisions"] / gate_denom,
            "full_launch_window_occupancy": gate_stats["full_window_decisions"] / gate_denom,
            "first_full_launch_window_step": gate_stats["first_window_step"],
            "first_red_launch_step": gate_stats["first_launch_step"],
            "first_red_hit_step": gate_stats["first_hit_step"],
            "blue_death_count_missile_hit": blue_missile_deaths,
            "blue_death_count_crash": blue_crash_deaths,
            "blue_death_count_other": blue_other_deaths,
            "blue_death_records": json.dumps(gate_stats["blue_death_records"], sort_keys=True),
        })

    return episodes


def _aggregate(episodes, label):
    n = max(len(episodes), 1)
    returns = [e["episode_return"] for e in episodes]
    lengths = [e["episode_length"] for e in episodes]
    rlf = [e["red_loss_fraction"] for e in episodes]
    blf = [e["blue_loss_fraction"] for e in episodes]
    mav = [e["mav_alive_final"] for e in episodes]
    rl = [e["red_launch_count"] for e in episodes]
    rh = [e["red_hit_count"] for e in episodes]
    bl = [e["blue_launch_count"] for e in episodes]
    bh = [e["blue_hit_count"] for e in episodes]
    min_d = [e["minimum_red_to_blue_distance_m"] for e in episodes if np.isfinite(e["minimum_red_to_blue_distance_m"])]
    range_o = [e["range_gate_occupancy"] for e in episodes]
    ao_o = [e["ao_gate_occupancy"] for e in episodes]
    ta_o = [e["ta_gate_occupancy"] for e in episodes]
    fw_o = [e["full_launch_window_occupancy"] for e in episodes]
    return {
        "model_label": label,
        "episodes": n,
        "episode_return_mean": float(np.mean(returns)), "episode_return_std": float(np.std(returns)),
        "episode_return_median": float(np.median(returns)),
        "episode_length_mean": float(np.mean(lengths)), "episode_length_std": float(np.std(lengths)),
        "red_loss_fraction_mean": float(np.mean(rlf)), "blue_loss_fraction_mean": float(np.mean(blf)),
        "mav_survival_rate": float(np.mean(mav)),
        "red_launches_mean": float(np.mean(rl)), "red_hits_mean": float(np.mean(rh)),
        "blue_launches_mean": float(np.mean(bl)), "blue_hits_mean": float(np.mean(bh)),
        "blue_death_count_missile_hit": int(sum(e["blue_death_count_missile_hit"] for e in episodes)),
        "blue_death_count_crash": int(sum(e["blue_death_count_crash"] for e in episodes)),
        "blue_death_count_other": int(sum(e["blue_death_count_other"] for e in episodes)),
        "minimum_distance_mean": float(np.mean(min_d)) if min_d else float("nan"),
        "range_gate_occupancy_mean": float(np.mean(range_o)),
        "ao_gate_occupancy_mean": float(np.mean(ao_o)),
        "ta_gate_occupancy_mean": float(np.mean(ta_o)),
        "full_launch_window_occupancy_mean": float(np.mean(fw_o)),
    }


def _deltas(trained_eps, random_eps):
    """Per-seed deltas: trained - random."""
    d = []
    for te, re in zip(trained_eps, random_eps):
        d.append({
            "episode_seed": te["episode_seed"],
            "delta_episode_return": te["episode_return"] - re["episode_return"],
            "delta_episode_length": te["episode_length"] - re["episode_length"],
            "delta_red_loss_fraction": te["red_loss_fraction"] - re["red_loss_fraction"],
            "delta_blue_loss_fraction": te["blue_loss_fraction"] - re["blue_loss_fraction"],
            "delta_minimum_distance": te["minimum_red_to_blue_distance_m"] - re["minimum_red_to_blue_distance_m"],
            "delta_full_launch_window_occupancy": te["full_launch_window_occupancy"] - re["full_launch_window_occupancy"],
            "delta_red_launch_count": te["red_launch_count"] - re["red_launch_count"],
        })
    return d


def _verdict(agg_trained, agg_random):
    """Determine if actor shows task improvement."""
    ret_ok = agg_trained["episode_return_mean"] > agg_random["episode_return_mean"]
    rlf_ok = agg_trained["red_loss_fraction_mean"] <= agg_random["red_loss_fraction_mean"]

    fw_ok = agg_trained["full_launch_window_occupancy_mean"] > agg_random["full_launch_window_occupancy_mean"]
    launch_ok = agg_trained["red_launches_mean"] > agg_random["red_launches_mean"]
    blf_ok = agg_trained["blue_loss_fraction_mean"] > agg_random["blue_loss_fraction_mean"]
    dist_ok = agg_trained["minimum_distance_mean"] < agg_random["minimum_distance_mean"]

    any_task_ok = fw_ok or launch_ok or blf_ok or dist_ok

    if not ret_ok:
        return "ACTOR_TASK_PERFORMANCE_DEGRADED"
    if ret_ok and rlf_ok and any_task_ok:
        return "ACTOR_TASK_IMPROVEMENT_CONFIRMED"
    return "ACTOR_TASK_IMPROVEMENT_NOT_CONFIRMED"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained-checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--opponent-policy", default="brma_rule")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1001)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model_path = Path(args.trained_checkpoint)
    meta = _load_meta(model_path)
    print(f"Meta: policy_arch={meta.get('policy_arch')} revision={meta.get('reward_contract_revision')} flight_scale={meta.get('flight_scale')}")

    device = torch.device(args.device)
    # Build trained policy
    trained_policy = _build_policy_from_meta(meta, device)
    trained_policy.load(model_path, map_location=device)
    trained_policy.eval()

    # Build random-init policy (same architecture, no load)
    random_policy = _build_policy_from_meta(meta, device)
    # Force fresh init by re-creating (build_policy_from_meta already does random init)
    random_policy.eval()
    # Verify random is NOT trained: check parameter differences
    trained_params = list(trained_policy.parameters())
    random_params = list(random_policy.parameters())
    max_diff = max(float((tp - rp).abs().max().item()) for tp, rp in zip(trained_params, random_params))
    print(f"Max param diff between trained and random: {max_diff:.6f}")
    assert max_diff > 0.01, "Random-init params too close to trained — possible checkpoint load issue"

    adapter = HeteroObsAdapterV2(max_red=5, max_blue=4)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run both models on same seeds
    env = make_env(args.config, **_env_type_override_kwargs(args.config))
    try:
        print(f"Evaluating RANDOM model ({args.episodes} episodes)...")
        random_eps = _evaluate_model(random_policy, env, args.config, args, adapter, device, "random_init")

        print(f"Evaluating TRAINED model ({args.episodes} episodes)...")
        trained_eps = _evaluate_model(trained_policy, env, args.config, args, adapter, device, "trained_100k")
    finally:
        env.close()

    # Write paired CSV
    all_eps = random_eps + trained_eps
    fields = list(all_eps[0].keys()) if all_eps else []
    with (out_dir / "paired_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_eps)

    # Aggregate
    agg_random = _aggregate(random_eps, "random_init")
    agg_trained = _aggregate(trained_eps, "trained_100k")
    deltas = _deltas(trained_eps, random_eps)

    with (out_dir / "aggregate_comparison.json").open("w", encoding="utf-8") as f:
        json.dump({"random_init": agg_random, "trained_100k": agg_trained, "per_seed_deltas": deltas}, f, indent=2)

    # Also write deltas CSV
    with (out_dir / "per_seed_deltas.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(deltas[0].keys()) if deltas else [])
        w.writeheader()
        w.writerows(deltas)

    verdict = _verdict(agg_trained, agg_random)
    print(f"\n=== Aggregate Results ===")
    print(f"RANDOM: ret={agg_random['episode_return_mean']:.2f} len={agg_random['episode_length_mean']:.1f} "
          f"red_loss={agg_random['red_loss_fraction_mean']:.3f} blue_loss={agg_random['blue_loss_fraction_mean']:.3f} "
          f"MAV_surv={agg_random['mav_survival_rate']:.3f} "
          f"launches={agg_random['red_launches_mean']:.2f} hits={agg_random['red_hits_mean']:.2f} "
          f"min_dist={agg_random['minimum_distance_mean']:.0f}m "
          f"fw_occ={agg_random['full_launch_window_occupancy_mean']:.4f}")
    print(f"TRAINED: ret={agg_trained['episode_return_mean']:.2f} len={agg_trained['episode_length_mean']:.1f} "
          f"red_loss={agg_trained['red_loss_fraction_mean']:.3f} blue_loss={agg_trained['blue_loss_fraction_mean']:.3f} "
          f"MAV_surv={agg_trained['mav_survival_rate']:.3f} "
          f"launches={agg_trained['red_launches_mean']:.2f} hits={agg_trained['red_hits_mean']:.2f} "
          f"min_dist={agg_trained['minimum_distance_mean']:.0f}m "
          f"fw_occ={agg_trained['full_launch_window_occupancy_mean']:.4f}")
    print(f"\n=== Verdict: {verdict} ===")

    summary = {
        "verdict": verdict,
        "random_init": agg_random,
        "trained_100k": agg_trained,
        "per_seed_deltas_summary": {
            "delta_return_mean": float(np.mean([d["delta_episode_return"] for d in deltas])),
            "delta_length_mean": float(np.mean([d["delta_episode_length"] for d in deltas])),
            "delta_fw_occ_mean": float(np.mean([d["delta_full_launch_window_occupancy"] for d in deltas])),
            "delta_min_dist_mean": float(np.mean([d["delta_minimum_distance"] for d in deltas])),
            "seeds_improved": sum(1 for d in deltas if d["delta_episode_return"] > 0),
        },
    }
    with (out_dir / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
