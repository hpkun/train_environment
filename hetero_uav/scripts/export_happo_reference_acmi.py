"""Export HAPPO reference checkpoint rollout to Tacview ACMI."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.happo import (
    BRMAEntityHAPPOReferencePolicy,
    BRMARecurrentHAPPOReferencePolicy,
    BRMARecurrentMaskedHAPPOReferencePolicy,
    EntityHAPPOReferencePolicy,
    HAPPOReferencePolicy,
)
from algorithms.happo.hetero_entity_recurrent_policy import (
    HeteroEntityRecurrentPolicy,
    validate_entity_policy_meta,
)
from algorithms.mappo.opponent_policy import OpponentPolicy
from scripts.experiment_logging_schema import FILE_SCHEMAS, ensure_schema_files


DEFAULT_DIR = "outputs/happo_3v2_reference_200k"
DEFAULT_CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0.yaml"


def _rel(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _model_path(exp_dir: Path, checkpoint: str) -> Path:
    return exp_dir / checkpoint / "model.pt"


def _load_meta(model: Path) -> dict:
    meta = model.parent / "meta.json"
    return json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}


def _build_policy_from_meta(meta: dict, device: torch.device):
    policy_arch = meta.get("policy_arch", "flat")
    if policy_arch == "entity_attention":
        return EntityHAPPOReferencePolicy(
            entity_dim=int(meta.get("entity_dim", 19)),
            critic_state_dim=int(meta.get("critic_state_dim", 480)),
            action_dim=3,
        ).to(device)
    if policy_arch == "brma_entity":
        return BRMAEntityHAPPOReferencePolicy(
            entity_dim=int(meta.get("entity_dim", 19)),
            critic_state_dim=int(meta.get("critic_state_dim", 480)),
            action_dim=3,
        ).to(device)
    if policy_arch == "brma_recurrent":
        return BRMARecurrentHAPPOReferencePolicy(
            entity_dim=int(meta.get("entity_dim", 19)),
            critic_state_dim=int(meta.get("critic_state_dim", 480)),
            action_dim=3,
            rnn_hidden_size=int(meta.get("rnn_hidden_size", 128)),
        ).to(device)
    if policy_arch == "brma_recurrent_masked":
        return BRMARecurrentMaskedHAPPOReferencePolicy(
            entity_dim=int(meta.get("entity_dim", 19)),
            critic_state_dim=int(meta.get("critic_state_dim", 480)),
            action_dim=3,
            rnn_hidden_size=int(meta.get("rnn_hidden_size", 128)),
            random_scale_mask=bool(meta.get("random_scale_mask", False)),
            random_mask_prob=float(meta.get("random_mask_prob", 0.25)),
            biased_mask=bool(meta.get("biased_mask", False)),
        ).to(device)
    if policy_arch == "hetero_entity_recurrent":
        validate_entity_policy_meta(meta)
        return HeteroEntityRecurrentPolicy(
            entity_dim=int(meta.get("entity_dim", 21)),
            action_dim=3,
            hidden_dim=int(meta.get("hidden_dim", 128)),
            rnn_hidden_size=int(meta.get("rnn_hidden_size", 128)),
            num_attention_heads=int(meta.get("num_attention_heads", 4)),
        ).to(device)
    if policy_arch == "flat":
        return HAPPOReferencePolicy(
            int(meta.get("actor_obs_dim", 96)),
            int(meta.get("critic_state_dim", 480)),
        ).to(device)
    if policy_arch in {"pure_happo", "pure_happo_tanh"}:
        from algorithms.pure_happo import PureHAPPOPolicy, PureHAPPOTanhPolicy
        num_agents = int(meta.get("num_agents", 3))
        cls = PureHAPPOTanhPolicy if policy_arch == "pure_happo_tanh" else PureHAPPOPolicy
        return cls(
            actor_obs_dim=int(meta.get("actor_obs_dim", 96)),
            critic_state_dim=int(meta.get("critic_state_dim", 480)),
            action_dim=3, num_agents=num_agents,
        ).to(device)
    raise ValueError(f"unsupported checkpoint policy_arch: {policy_arch}")


def _acmi_id(agent_id: str) -> int:
    side, idx = agent_id.split("_", 1)
    return (100 if side == "red" else 200) + int(idx)


def _role_ids(env) -> list[int]:
    return [0 if env.agent_roles.get(rid) == "mav" else 1 for rid in env.red_ids]


def _alive_counts(env) -> tuple[int, int, bool]:
    red = sum(1 for sim in env.red_planes.values() if sim.is_alive)
    blue = sum(1 for sim in env.blue_planes.values() if sim.is_alive)
    mav = bool(env.red_planes.get("red_0") and env.red_planes["red_0"].is_alive)
    return red, blue, mav


def _aircraft_name(env, aid: str) -> str:
    """Produce Tacview-compatible name with role-appropriate visual label."""
    role = env.agent_roles.get(aid, "")
    if role == "mav":
        # MAV role → visual label reflects F-22, even if dynamics use F-16 surrogate
        return f"{aid}_MAV_F22_visual"
    if role == "attack_uav":
        return f"{aid}_UAV_F16"
    return aid


def _entries(env, acmi_visual=None) -> list[dict]:
    entries = []
    visual_map = acmi_visual or {}
    for aid in env.red_ids + env.blue_ids:
        sim = env.red_planes.get(aid) or env.blue_planes.get(aid)
        lon, lat, alt = sim.get_geodetic()
        roll, pitch, yaw = np.asarray(sim.get_rpy(), dtype=np.float64) * (180.0 / np.pi)
        role = env.agent_roles.get(aid, "")
        visual_model = visual_map.get(role, env.agent_models.get(aid, ""))
        entries.append({
            "acmi_id": _acmi_id(aid),
            "type": f"Air+FixedWing+{visual_model.upper()}" if visual_model else "Air+FixedWing",
            "lon": float(lon),
            "lat": float(lat),
            "alt": float(alt),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
            "name": _aircraft_name(env, aid),
            "color": "Red" if aid.startswith("red_") else "Blue",
            "alive": bool(sim.is_alive),
        })
    return entries


def _all_missiles(env) -> list:
    """Collect all missiles from aircraft launch histories (same pattern as export_hetero_tacview_acmi.py)."""
    seen = set()
    missiles = []
    for sim in list(env.red_planes.values()) + list(env.blue_planes.values()):
        for missile in getattr(sim, "launch_missiles", []):
            uid = getattr(missile, "uid", str(id(missile)))
            if uid in seen:
                continue
            seen.add(uid)
            missiles.append(missile)
    return missiles


def _missile_entries(env, missile_id_map: dict[str, int]) -> list[dict]:
    """Produce ACMI entries for alive missiles (matching original exporter pattern)."""
    entries = []
    for missile in _all_missiles(env):
        uid = getattr(missile, "uid", str(id(missile)))
        if uid not in missile_id_map:
            missile_id_map[uid] = 1000 + len(missile_id_map)
        if not bool(getattr(missile, "is_alive", False)):
            continue
        lon, lat, alt = missile.get_geodetic()
        roll, pitch, yaw = np.asarray(missile.get_rpy(), dtype=np.float64) * (180.0 / np.pi)
        entries.append({
            "acmi_id": missile_id_map[uid],
            "type": "Weapon+Missile",
            "lon": float(lon),
            "lat": float(lat),
            "alt": float(alt),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
            "name": str(getattr(missile, "model", "AIM-9L")).upper(),
            "color": str(getattr(missile, "color", "White")),
            "alive": True,
        })
    return entries


def _missile_explosions(
    env, missile_id_map: dict[str, int], logged_explosions: set[str]
) -> list[dict]:
    """Yellow explosion at hit position (matching original exporter pattern)."""
    explosions = []
    for missile in _all_missiles(env):
        uid = getattr(missile, "uid", str(id(missile)))
        if uid in logged_explosions:
            continue
        if uid not in missile_id_map:
            missile_id_map[uid] = 1000 + len(missile_id_map)
        if bool(getattr(missile, "is_done", False)) and bool(
            getattr(missile, "is_success", False)
        ):
            lon, lat, alt = missile.get_geodetic()
            explosions.append({
                "acmi_id": missile_id_map[uid],
                "lon": float(lon),
                "lat": float(lat),
                "alt": float(alt),
                "color": "Yellow",
                "radius": float(getattr(missile, "_Rc", 300.0)),
            })
            logged_explosions.add(uid)
    return explosions


def _team_done(terminated: dict, truncated: dict) -> bool:
    return all(terminated.values()) or all(truncated.values())


def _append_schema_row(directory: Path, filename: str, row: dict) -> None:
    ensure_schema_files(directory)
    columns = FILE_SCHEMAS[filename]
    with (directory / filename).open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=columns).writerow({col: row.get(col, "") for col in columns})


def _write_rich_export_rows(directory: Path, summary: dict) -> None:
    scenario = Path(str(summary.get("config", "scenario"))).stem
    run_id = Path(str(summary.get("model", "acmi_export"))).parents[1].name if summary.get("model") else "acmi_export"
    red_alive = int(summary.get("red_alive_final", 0))
    blue_alive = int(summary.get("blue_alive_final", 0))
    red_dead = max(0, 3 - red_alive)
    total_blue = 2 if "3v2" in scenario else 4
    blue_dead = max(0, total_blue - blue_alive)
    _append_schema_row(directory, "eval_episode_metrics.csv", {
        "run_id": run_id,
        "checkpoint_name": summary.get("checkpoint"),
        "eval_scenario": scenario,
        "episode_id": 0,
        "outcome": summary.get("outcome"),
        "episode_length": summary.get("steps"),
        "red_win": 1 if str(summary.get("outcome", "")).startswith("red") else 0,
        "blue_win": 1 if str(summary.get("outcome", "")).startswith("blue") else 0,
        "timeout": 1 if summary.get("outcome") == "timeout" else 0,
        "mav_alive": summary.get("mav_alive"),
        "red_alive_final": red_alive,
        "blue_alive_final": blue_alive,
        "red_missiles_fired": summary.get("red_missiles_fired"),
        "blue_missiles_fired": summary.get("blue_missiles_fired"),
        "red_missile_hits": summary.get("red_missile_hits"),
        "blue_missile_hits": summary.get("blue_missile_hits"),
        "red_dead": red_dead,
        "blue_dead": blue_dead,
    })
    _append_schema_row(directory, "aircraft_timeseries.csv", {
        "run_id": run_id,
        "scenario": scenario,
        "episode_id": 0,
        "step": summary.get("steps"),
        "agent_id": "red_0",
        "role": "mav",
        "team": "red",
        "alive": summary.get("mav_alive"),
        "is_mav": 1,
        "is_uav": 0,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description="Export HAPPO reference checkpoint to ACMI")
    parser.add_argument("--experiment-dir", default=DEFAULT_DIR)
    parser.add_argument("--checkpoint", choices=["best", "latest"], default="best")
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--opponent-policy", default="brma_rule",
                        choices=["zero", "random", "rule_nearest", "greedy_fsm", "brma_rule", "brma_rule_safe_pursuit"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--enable-rich-logging", action="store_true")
    parser.add_argument("--rich-log-dir", default=None)
    parser.add_argument("--diagnostics-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    exp_dir = _rel(args.experiment_dir)
    model = _rel(args.model) if args.model else _model_path(exp_dir, args.checkpoint)
    if not model.exists():
        print(f"checkpoint not found: {model}", file=sys.stderr)
        return 2
    from uav_env import make_env
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
    from uav_env.JSBSim.adapters.hetero_entity_set_adapter import HeteroEntitySetAdapter
    from uav_env.JSBSim.render_tacview import TacviewLogger

    # Unified output directory: all files go into --output as a folder
    out_dir = _rel(args.output) if args.output else exp_dir / "acmi" / f"{args.checkpoint}_3v2_episode0"
    out_dir.mkdir(parents=True, exist_ok=True)
    acmi_path   = out_dir / "episode.acmi"
    summary_path = out_dir / "summary.json"
    diag_dir     = out_dir / "diagnostics"

    meta = _load_meta(model)
    device = torch.device(args.device)
    policy = _build_policy_from_meta(meta, device)
    policy.load(model, map_location=device)
    policy.eval()
    entity_mode = meta.get("policy_arch") == "hetero_entity_recurrent"
    if entity_mode:
        adapter = HeteroEntitySetAdapter()
    else:
        adapter = HeteroObsAdapterV2()
    env = make_env(args.config, env_type="jsbsim_hetero")

    # Read acmi_visual_by_role from config for visual labelling
    try:
        import yaml
        with open(args.config, encoding="utf-8") as _f:
            _cfg = yaml.safe_load(_f) or {}
        acmi_visual = _cfg.get("acmi_visual_by_role", {})
    except Exception:
        acmi_visual = {}
    mav_dynamics_model = env.agent_models.get("red_0", "unknown")
    mav_visual_model = acmi_visual.get("mav", mav_dynamics_model)
    opponent = OpponentPolicy(mode=args.opponent_policy, seed=args.seed + 33)
    logger = TacviewLogger(reference_time="2026-01-01T00:00:00Z")
    missile_id_map: dict[str, int] = {}
    logged_explosions: set[str] = set()
    red_missile_objects, blue_missile_objects = 0, 0
    death_order: list[str] = []
    prev_alive: dict[str, bool] = {}
    missile_stats = {"red_fired": 0, "blue_fired": 0, "red_hits": 0, "blue_hits": 0}
    prev_hits = {"red": 0, "blue": 0}
    mav_sat, uav_sat = [], []
    red0_pitch, red0_roll, red0_alt = [], [], []
    outcome = "unknown"

    # Diagnostics setup (inside output directory)
    if args.diagnostics_dir:
        diag_dir = Path(args.diagnostics_dir)
    diag_dir.mkdir(parents=True, exist_ok=True)
    red_csv = diag_dir / "red_behavior_timeseries.csv"
    blue_csv = diag_dir / "blue_behavior_timeseries.csv"
    red_rows: list[dict] = []
    blue_rows: list[dict] = []
    max_steps_limit = args.max_steps or env.max_steps
    # Per-agent cumulative missile stats for diagnostics
    red_cum_fired = {rid: 0 for rid in env.red_ids}
    red_cum_hits = {rid: 0 for rid in env.red_ids}
    blue_cum_fired = {bid: 0 for bid in env.blue_ids}
    blue_cum_hits = {bid: 0 for bid in env.blue_ids}

    # Recurrent hidden state (required for hetero_entity_recurrent policies)
    _rnn_hidden_size = getattr(policy, "rnn_hidden_size", 0)
    eval_rnn_hidden = None
    if _rnn_hidden_size > 0:
        eval_rnn_hidden = np.zeros((len(env.red_ids), _rnn_hidden_size), dtype=np.float32)

    try:
        obs, info = env.reset(seed=args.seed)
        logger.record_frame(0.0, _entries(env, acmi_visual) + _missile_entries(env, missile_id_map),
                            _missile_explosions(env, missile_id_map, logged_explosions))
        step = 0
        while True:
            adapted = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
            act_kw = {}
            if eval_rnn_hidden is not None:
                from algorithms.happo.rollout_safety import zero_inactive_hidden
                active = np.ones(len(env.red_ids), dtype=np.float32)
                for i, rid in enumerate(env.red_ids):
                    ai = (info or {}).get(rid, {})
                    active[i] = 1.0 if ai.get("alive", True) else 0.0
                eval_rnn_hidden = zero_inactive_hidden(eval_rnn_hidden, active)
                act_kw["rnn_hidden"] = torch.as_tensor(eval_rnn_hidden, device=device)
            with torch.no_grad():
                if entity_mode:
                    out = policy.act(
                        torch.as_tensor(adapted["actor_entity_tokens"], device=device),
                        torch.as_tensor(adapted["actor_keep_mask"], device=device),
                        torch.as_tensor(adapted["role_ids"], device=device),
                        torch.as_tensor(adapted["critic_entity_tokens"], device=device),
                        torch.as_tensor(adapted["critic_keep_mask"], device=device),
                        deterministic=True,
                        critic_counts=torch.as_tensor(
                            adapted.get("critic_counts", np.zeros(4, dtype=np.float32)),
                            device=device),
                        **act_kw,
                    )
                else:
                    actor_obs = np.stack([
                        adapted["actor_obs"].get(rid, np.zeros(adapter.flat_actor_obs_dim, dtype=np.float32))
                        for rid in env.red_ids
                    ])
                    out = policy.act(
                        torch.as_tensor(actor_obs, device=device),
                        roles=_role_ids(env),
                        critic_state=torch.as_tensor(adapted["critic_state"], device=device),
                        deterministic=True,
                    )
            acts_np = out["action"].cpu().numpy()
            if eval_rnn_hidden is not None and "rnn_hidden" in out:
                from algorithms.happo.rollout_safety import zero_inactive_hidden
                active = np.ones(len(env.red_ids), dtype=np.float32)
                for i, rid in enumerate(env.red_ids):
                    ai = (info or {}).get(rid, {})
                    active[i] = 1.0 if ai.get("alive", True) else 0.0
                eval_rnn_hidden = zero_inactive_hidden(out["rnn_hidden"].cpu().numpy(), active)
            mav_sat.append(float(np.mean(np.abs(acts_np[0:1]) >= 0.999)))
            if len(env.red_ids) > 1:
                uav_sat.append(float(np.mean(np.abs(acts_np[1:]) >= 0.999)))
            actions = {rid: acts_np[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}
            actions.update(opponent.act(obs, env.blue_ids, env=env))
            obs, rewards, terminated, truncated, info = env.step(actions)
            step += 1
            for aid in env.agent_ids:
                fired = int(info.get(aid, {}).get("missiles_fired_this_step", 0)) if isinstance(info.get(aid, {}), dict) else 0
                if aid.startswith("red_"):
                    missile_stats["red_fired"] += fired
                else:
                    missile_stats["blue_fired"] += fired
            mt = info.get("__missile_term__", {})
            if isinstance(mt, dict):
                for side in ("red", "blue"):
                    total = int(mt.get(side, {}).get("hit", 0))
                    missile_stats[f"{side}_hits"] += max(total - prev_hits[side], 0)
                    prev_hits[side] = total
            for aid in env.red_ids + env.blue_ids:
                sim = env.red_planes.get(aid) or env.blue_planes.get(aid)
                alive = bool(sim.is_alive)
                if prev_alive.get(aid, True) and not alive:
                    death_order.append(aid)
                prev_alive[aid] = alive
            red0 = env.red_planes.get("red_0")
            if red0 is not None:
                r, p, _y = red0.get_rpy()
                red0_roll.append(abs(math.degrees(float(r))))
                red0_pitch.append(abs(math.degrees(float(p))))
                red0_alt.append(float(red0.get_position()[2]))

            # Count missile objects by side
            for missile in _all_missiles(env):
                uid = getattr(missile, "uid", str(id(missile)))
                if uid not in missile_id_map:
                    owner = getattr(missile, "parent_aircraft", None)
                    if owner and owner.uid.startswith("red_"):
                        red_missile_objects += 1
                    else:
                        blue_missile_objects += 1

            # Render frame with aircraft, missiles, and explosions
            all_entries = _entries(env, acmi_visual) + _missile_entries(env, missile_id_map)
            explosions = _missile_explosions(env, missile_id_map, logged_explosions)
            logger.record_frame(step * float(env.env_dt), all_entries, explosions)

            # Per-step diagnostics: red agents
            sim_time = step * float(env.env_dt)
            for i, rid in enumerate(env.red_ids):
                sim = env.red_planes.get(rid)
                role = env.agent_roles.get(rid, "")
                visual = acmi_visual.get(role, env.agent_models.get(rid, ""))
                row = {"episode_id": 0, "step": step, "sim_time_sec": round(sim_time, 2),
                       "agent_id": rid, "team": "red", "role": role,
                       "dynamics_model": env.agent_models.get(rid, ""),
                       "visual_model": visual, "alive": int(sim.is_alive if sim else 0),
                       "is_mav": int(role == "mav"), "is_uav": int(role == "attack_uav")}
                if sim and sim.is_alive:
                    pos = sim.get_position(); vel = sim.get_velocity(); rpy = sim.get_rpy()
                    row.update({"north_m": round(float(pos[0]), 1), "east_m": round(float(pos[1]), 1),
                                "altitude_m": round(float(pos[2]), 1),
                                "speed_mps": round(float(np.linalg.norm(vel)), 1),
                                "vn_mps": round(float(vel[0]), 1), "ve_mps": round(float(vel[1]), 1),
                                "vu_mps": round(float(vel[2]), 1),
                                "roll_deg": round(float(np.rad2deg(rpy[0])), 1),
                                "pitch_deg": round(float(np.rad2deg(rpy[1])), 1),
                                "yaw_deg": round(float(np.rad2deg(rpy[2])), 1),
                                "heading_deg": round(float(np.rad2deg(rpy[2])), 1)})
                    raw_a = acts_np[i]
                    row.update({"raw_action_pitch": round(float(raw_a[0]), 4),
                                "raw_action_heading": round(float(raw_a[1]), 4),
                                "raw_action_speed": round(float(raw_a[2]), 4),
                                "target_pitch_deg": round(float(raw_a[0]) * 90, 1),
                                "target_heading_deg": round(float(raw_a[1]) * 180, 1),
                                "target_speed_mps": round(102 + (float(raw_a[2]) + 1) / 2 * 306, 1)})
                    # Missile warning
                    row["missile_warning"] = int(sim.check_missile_warning() is not None)
                    red_cum_fired[rid] += int((info.get(rid, {}) or {}).get("missiles_fired_this_step", 0))
                    row["missiles_fired_this_step"] = int((info.get(rid, {}) or {}).get("missiles_fired_this_step", 0))
                    row["cumulative_missiles_fired"] = red_cum_fired[rid]
                    row["cumulative_missile_hits"] = red_cum_hits[rid]
                    row["missiles_remaining"] = int(getattr(sim, "num_left_missiles", 0))
                    # Reward components
                    rc = (info.get("reward_components", {}) or {}).get(rid, {}) or {}
                    row["step_reward"] = round(float(rc.get("total", 0)), 4)
                    row["r_boundary"] = round(float(rc.get("r_boundary", 0)), 4)
                    row["r_altitude_envelope"] = round(float(rc.get("r_altitude_envelope", 0)), 4)
                    row["r_adv_uav"] = round(float(rc.get("r_adv_uav", 0)), 4)
                    row["mav_safety"] = round(float(rc.get("mav_safety", rc.get("mav_survival", 0))), 4)
                    row["mav_safety_dist"] = round(float(rc.get("mav_safety_dist", 0)), 4)
                    row["mav_safety_threat"] = round(float(rc.get("mav_safety_threat", 0)), 4)
                    row["mav_safety_aspect"] = round(float(rc.get("mav_safety_aspect", 0)), 4)
                    row["mav_support"] = round(float(rc.get("mav_support", 0)), 4)
                    row["mav_support_position"] = round(float(rc.get("mav_support_position", 0)), 4)
                    row["mav_support_information"] = round(float(rc.get("mav_support_information", 0)), 4)
                    row["mav_death"] = round(float(rc.get("mav_death", rc.get("death_penalty", 0))), 4)
                    row["mav_out_zone"] = round(float(rc.get("mav_out_zone", 0)), 4)
                    row["mav_assist"] = round(float(rc.get("mav_assist", 0)), 4)
                    row["uav_attack"] = round(float(rc.get("uav_attack", 0)), 4)
                    row["uav_attack_distance"] = round(float(rc.get("uav_attack_distance", 0)), 4)
                    row["uav_attack_angle"] = round(float(rc.get("uav_attack_angle", 0)), 4)
                    row["uav_attack_gate"] = round(float(rc.get("uav_attack_gate", 0)), 4)
                    row["uav_attack_mav_shared_multiplier"] = int(rc.get("uav_attack_mav_shared_multiplier", 0))
                    row["uav_reward_target_id"] = str(rc.get("uav_reward_target_id", ""))
                    row["uav_reward_target_mav_shared"] = int(rc.get("uav_reward_target_mav_shared", 0))
                    row["uav_fire"] = round(float(rc.get("uav_fire", 0)), 4)
                    row["uav_hit"] = round(float(rc.get("uav_hit", 0)), 4)
                    row["uav_dodge"] = round(float(rc.get("uav_dodge", 0)), 4)
                    row["uav_death"] = round(float(rc.get("uav_death", 0)), 4)
                    row["uav_out_zone"] = round(float(rc.get("uav_out_zone", 0)), 4)
                red_rows.append(row)

            # Per-step diagnostics: blue agents
            for i, bid in enumerate(env.blue_ids):
                sim = env.blue_planes.get(bid)
                role = env.agent_roles.get(bid, "attack_uav")
                visual = acmi_visual.get(role, env.agent_models.get(bid, ""))
                row = {"episode_id": 0, "step": step, "sim_time_sec": round(sim_time, 2),
                       "agent_id": bid, "team": "blue", "role": role,
                       "dynamics_model": env.agent_models.get(bid, ""),
                       "visual_model": visual, "alive": int(sim.is_alive if sim else 0)}
                if sim and sim.is_alive:
                    pos = sim.get_position(); vel = sim.get_velocity(); rpy = sim.get_rpy()
                    row.update({"north_m": round(float(pos[0]), 1), "east_m": round(float(pos[1]), 1),
                                "altitude_m": round(float(pos[2]), 1),
                                "speed_mps": round(float(np.linalg.norm(vel)), 1),
                                "vn_mps": round(float(vel[0]), 1), "ve_mps": round(float(vel[1]), 1),
                                "vu_mps": round(float(vel[2]), 1),
                                "roll_deg": round(float(np.rad2deg(rpy[0])), 1),
                                "pitch_deg": round(float(np.rad2deg(rpy[1])), 1),
                                "yaw_deg": round(float(np.rad2deg(rpy[2])), 1),
                                "heading_deg": round(float(np.rad2deg(rpy[2])), 1)})
                    blue_cum_fired[bid] += int((info.get(bid, {}) or {}).get("missiles_fired_this_step", 0))
                    row["missiles_fired_this_step"] = int((info.get(bid, {}) or {}).get("missiles_fired_this_step", 0))
                    row["cumulative_missiles_fired"] = blue_cum_fired[bid]
                    row["cumulative_missile_hits"] = blue_cum_hits[bid]
                    row["missiles_remaining"] = int(getattr(sim, "num_left_missiles", 0))
                    row["opponent_policy"] = args.opponent_policy
                blue_rows.append(row)
            if _team_done(terminated, truncated):
                break
            if step > int(getattr(env, "max_steps", 1000)) + 5:
                break
        red_alive, blue_alive, mav_alive = _alive_counts(env)
        if blue_alive == 0 and red_alive > 0:
            outcome = "red_win_elimination"
        elif red_alive == 0 and blue_alive > 0:
            outcome = "blue_win_elimination"
        elif red_alive == 0 and blue_alive == 0:
            outcome = "mutual_elimination_draw"
        elif step >= int(getattr(env, "max_steps", 1000)):
            outcome = "timeout"
        logger.write(str(acmi_path))

        # Write diagnostic CSVs
        import csv as _csv
        if red_rows:
            with open(red_csv, "w", newline="") as _f:
                _w = _csv.DictWriter(_f, fieldnames=sorted(red_rows[0].keys()))
                _w.writeheader(); _w.writerows(red_rows)
        if blue_rows:
            with open(blue_csv, "w", newline="") as _f:
                _w = _csv.DictWriter(_f, fieldnames=sorted(blue_rows[0].keys()))
                _w.writeheader(); _w.writerows(blue_rows)

        # Extended summary
        red_behavior = {}
        if red_rows:
            for rid in env.red_ids:
                r_rows = [r for r in red_rows if r["agent_id"] == rid and r["alive"]]
                alts = [r["altitude_m"] for r in r_rows if "altitude_m" in r]
                spds = [r["speed_mps"] for r in r_rows if "speed_mps" in r]
                mw = sum(1 for r in r_rows if r.get("missile_warning")) / max(len(r_rows), 1)
                red_behavior[rid] = {
                    "mean_altitude": round(float(np.mean(alts)), 1) if alts else 0,
                    "final_altitude": round(alts[-1], 1) if alts else 0,
                    "mean_speed": round(float(np.mean(spds)), 1) if spds else 0,
                    "missile_warning_fraction": round(mw, 3),
                    "action_saturation": round(float(np.mean(mav_sat)) if rid == "red_0" else float(np.mean(uav_sat)), 3),
                }
        blue_behavior = {}
        if blue_rows:
            for bid in env.blue_ids:
                b_rows = [r for r in blue_rows if r["agent_id"] == bid and r["alive"]]
                reds = [r for r in red_rows if r["alive"]]
                if b_rows and reds:
                    nearest_ranges = []
                    for br in b_rows:
                        bn, be = br.get("north_m", 0), br.get("east_m", 0)
                        min_d = min(np.hypot(rr.get("north_m", 0) - bn, rr.get("east_m", 0) - be) for rr in reds)
                        nearest_ranges.append(min_d)
                    blue_behavior[bid] = {"mean_nearest_red_range": round(float(np.mean(nearest_ranges)), 1)}

        summary = {
            "exporter_name": "export_happo_reference_acmi",
            "exporter_version": "2.0",
            "checkpoint": args.checkpoint,
            "model": str(model),
            "model_meta_path": str(model.parent / "meta.json"),
            "config": args.config,
            "seed": args.seed,
            "opponent_policy": args.opponent_policy,
            "policy_arch": meta.get("policy_arch", ""),
            "reward_mode": meta.get("reward_mode", ""),
            "red_target_selection_mode": _cfg.get("red_target_selection_mode", "") if isinstance(_cfg, dict) else "",
            "outcome": outcome, "steps": step,
            "simulated_time_sec": round(step * float(env.env_dt), 1),
            "red_alive_final": red_alive, "blue_alive_final": blue_alive,
            "mav_alive": mav_alive,
            "death_order": death_order,
            "aircraft_visual_labels": {
                "red_0": _aircraft_name(env, "red_0"),
                "red_1": _aircraft_name(env, "red_1") if len(env.red_ids) > 1 else "",
                "blue_0": _aircraft_name(env, "blue_0"),
            },
            "red_0_visual_label": _aircraft_name(env, "red_0"),
            "mav_dynamics_model": mav_dynamics_model,
            "mav_visual_model": mav_visual_model,
            "mav_role": "mav", "mav_num_missiles": 0,
            "missile_objects_exported": len(missile_id_map),
            "red_missile_objects_exported": red_missile_objects,
            "blue_missile_objects_exported": blue_missile_objects,
            "missile_visualization_mode": "real_missile_position_from_env",
            "missiles_fired": missile_stats["red_fired"] + missile_stats["blue_fired"],
            "missile_hits": missile_stats["red_hits"] + missile_stats["blue_hits"],
            "red_missiles_fired": missile_stats["red_fired"],
            "blue_missiles_fired": missile_stats["blue_fired"],
            "red_missile_hits": missile_stats["red_hits"],
            "blue_missile_hits": missile_stats["blue_hits"],
            "red_0_max_abs_roll_deg": max(red0_roll) if red0_roll else 0.0,
            "red_0_max_abs_pitch_deg": max(red0_pitch) if red0_pitch else 0.0,
            "red_0_min_altitude": min(red0_alt) if red0_alt else None,
            "mav_action_saturation_rate": float(np.mean(mav_sat)) if mav_sat else 0.0,
            "uav_action_saturation_rate": float(np.mean(uav_sat)) if uav_sat else 0.0,
            "red_behavior_mode_summary": red_behavior,
            "blue_behavior_mode_summary": blue_behavior,
            "diagnostics_files": {
                "red_csv": str(red_csv), "blue_csv": str(blue_csv),
            },
            "acmi_metadata": {
                "checkpoint": args.checkpoint, "config": args.config,
                "policy_arch": meta.get("policy_arch", ""),
                "opponent_policy": args.opponent_policy,
                "red_target_selection_mode": _cfg.get("red_target_selection_mode", "") if isinstance(_cfg, dict) else "",
                "mav_dynamics": mav_dynamics_model, "mav_visual": mav_visual_model,
            },
            "output_acmi": str(acmi_path),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if args.enable_rich_logging:
            rich_dir = _rel(args.rich_log_dir) if args.rich_log_dir else output.parent
            _write_rich_export_rows(rich_dir, summary)
    finally:
        if hasattr(env, "close"):
            env.close()
    print(f"output_dir: {out_dir}")
    print(f"acmi: {acmi_path}")
    print(f"summary: {summary_path}")
    print(f"diagnostics: {diag_dir}")
    print(f"outcome: {outcome}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
