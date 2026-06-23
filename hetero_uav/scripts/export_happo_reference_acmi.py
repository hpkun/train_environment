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


def _entries(env) -> list[dict]:
    entries = []
    for aid in env.red_ids + env.blue_ids:
        sim = env.red_planes.get(aid) or env.blue_planes.get(aid)
        lon, lat, alt = sim.get_geodetic()
        roll, pitch, yaw = np.asarray(sim.get_rpy(), dtype=np.float64) * (180.0 / np.pi)
        entries.append({
            "acmi_id": _acmi_id(aid),
            "type": "Air+FixedWing",
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
    parser.add_argument("--opponent-policy", default="brma_rule")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--enable-rich-logging", action="store_true")
    parser.add_argument("--rich-log-dir", default=None)
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

    output = _rel(args.output) if args.output else exp_dir / "acmi" / f"{args.checkpoint}_3v2_episode0.acmi"
    summary_path = _rel(args.summary_json) if args.summary_json else output.with_name(output.stem + "_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

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

    # Recurrent hidden state (required for hetero_entity_recurrent policies)
    _rnn_hidden_size = getattr(policy, "rnn_hidden_size", 0)
    eval_rnn_hidden = None
    if _rnn_hidden_size > 0:
        eval_rnn_hidden = np.zeros((len(env.red_ids), _rnn_hidden_size), dtype=np.float32)

    try:
        obs, info = env.reset(seed=args.seed)
        logger.record_frame(0.0, _entries(env) + _missile_entries(env, missile_id_map),
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
            all_entries = _entries(env) + _missile_entries(env, missile_id_map)
            explosions = _missile_explosions(env, missile_id_map, logged_explosions)
            logger.record_frame(step * float(env.env_dt), all_entries, explosions)
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
        logger.write(str(output))
        summary = {
            "checkpoint": args.checkpoint,
            "model": str(model),
            "config": args.config,
            "outcome": outcome,
            "steps": step,
            "red_alive_final": red_alive,
            "blue_alive_final": blue_alive,
            "mav_alive": mav_alive,
            "death_order": death_order,
            "death_reason": "environment_info_not_explicit",
            "aircraft_visual_labels": {
                "red_0": _aircraft_name(env, "red_0"),
                "red_1": _aircraft_name(env, "red_1") if len(env.red_ids) > 1 else "",
                "blue_0": _aircraft_name(env, "blue_0"),
            },
            "red_0_visual_label": _aircraft_name(env, "red_0"),
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
            "output_acmi": str(output),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if args.enable_rich_logging:
            rich_dir = _rel(args.rich_log_dir) if args.rich_log_dir else output.parent
            _write_rich_export_rows(rich_dir, summary)
    finally:
        if hasattr(env, "close"):
            env.close()
    print(f"output_acmi: {output}")
    print(f"output_summary: {summary_path}")
    print(f"outcome: {outcome}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
