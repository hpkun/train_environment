"""Diagnose action heading alignment against target bearing and AO changes.

This script is read-only. It does not train, modify rewards, missile dynamics,
PID, aircraft XML, blue rules, action space, or observation dimensions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from check_launch_envelope_oracle import _diagnose_red_shooter
from red_attack_audit_utils import (
    blue_actions,
    direct_chase_action,
    safe_mav_action,
    team_done,
)


DEFAULT_CONFIG = (
    "uav_env/JSBSim/configs/"
    "hetero_mav_shared_geo_3v2_approach_fire_easy_f16_mav_surrogate.yaml"
)
DEFAULT_DATASET = "outputs/direct_chase_oracle_dataset/direct_chase_oracle_3v2.npz"

MODEL_SPECS = {
    "flat_easy_imitation": {
        "output_dir": "outputs/approach_fire_curriculum_50k/flat_easy_imitation",
        "checkpoint_name": "best",
    },
    "entity_easy_imitation": {
        "output_dir": "outputs/approach_fire_curriculum_50k/entity_easy_imitation",
        "checkpoint_name": "best",
    },
}

ROW_FIELDS = [
    "source",
    "episode_id",
    "step",
    "red_id",
    "target_id",
    "current_heading_rad",
    "target_bearing_rad",
    "heading_error_rad",
    "action_heading",
    "decoded_target_heading_rad",
    "command_error_to_bearing_rad",
    "AO_rad",
    "AO_next_rad",
    "delta_AO_rad",
    "range_m",
    "range_next_m",
    "delta_range_m",
    "launch_allowed",
    "block_reason",
    "action_pitch",
    "action_speed",
]

SUMMARY_FIELDS = [
    "source",
    "samples",
    "action_pitch_min",
    "action_pitch_max",
    "action_pitch_mean",
    "action_pitch_std",
    "action_heading_min",
    "action_heading_max",
    "action_heading_mean",
    "action_heading_std",
    "action_speed_min",
    "action_speed_max",
    "action_speed_mean",
    "action_speed_std",
    "action_saturation_rate",
    "heading_command_abs_error_mean_rad",
    "heading_command_abs_error_mean_deg",
    "ao_delta_mean_rad",
    "ao_delta_mean_deg",
    "ao_reduced_rate",
    "range_delta_mean_m",
    "range_reduced_rate",
    "launch_allowed_rate",
    "dominant_block_reason",
    "block_reason_counts_json",
]


def _as_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _safe_output_path(path: str | Path) -> Path:
    p = _as_path(path)
    if p.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = p.with_name(f"{p.stem}_{stamp}{p.suffix}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _wrap_pi(x: float) -> float:
    return float((x + math.pi) % (2.0 * math.pi) - math.pi)


def _bearing(src: np.ndarray, dst: np.ndarray) -> float:
    d = np.asarray(dst, dtype=np.float64) - np.asarray(src, dtype=np.float64)
    return float(math.atan2(float(d[1]), float(d[0])))


def _geometry_pair(shooter, target) -> tuple[float, float, float]:
    from uav_env.JSBSim.utils import get2d_AO_TA_R

    spos = np.asarray(shooter.get_position(), dtype=np.float64)
    svel = np.asarray(shooter.get_velocity(), dtype=np.float64)
    tpos = np.asarray(target.get_position(), dtype=np.float64)
    tvel = np.asarray(target.get_velocity(), dtype=np.float64)
    sfeat = np.array([spos[0], spos[1], -spos[2], svel[0], svel[1], -svel[2]], dtype=np.float64)
    tfeat = np.array([tpos[0], tpos[1], -tpos[2], tvel[0], tvel[1], -tvel[2]], dtype=np.float64)
    ao, ta, rng = get2d_AO_TA_R(sfeat, tfeat)
    return float(rng), float(ao), float(ta)


def _stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def analyze_dataset(dataset_path: str | Path) -> dict[str, Any]:
    p = _as_path(dataset_path)
    data = np.load(p, allow_pickle=True)
    actions = np.asarray(data["oracle_action"], dtype=np.float32)
    pitch = _stats(actions[:, 0])
    heading = _stats(actions[:, 1])
    speed = _stats(actions[:, 2])
    saturation = float(np.mean(np.any(np.abs(actions) > 0.95, axis=1)))
    summary = {
        "path": str(p),
        "samples": int(actions.shape[0]),
        "action_pitch": pitch,
        "action_heading": heading,
        "action_speed": speed,
        "action_saturation_rate": saturation,
        "has_world_heading": False,
        "has_target_bearing": False,
        "heading_bearing_static_check": (
            "not_available: dataset stores flat actor_obs and oracle_action, "
            "but not world heading or target bearing"
        ),
        "launch_range_flag_mean": float(np.mean(data["launch_range_flag"]))
        if "launch_range_flag" in data.files else None,
        "launch_angle_flag_mean": float(np.mean(data["launch_angle_flag"]))
        if "launch_angle_flag" in data.files else None,
        "launch_envelope_flag_mean": float(np.mean(data["launch_envelope_flag"]))
        if "launch_envelope_flag" in data.files else None,
    }
    return summary


def _resolve_model(output_dir: str, checkpoint_name: str) -> Path:
    base = _as_path(output_dir)
    for name in [checkpoint_name, "best", "latest"]:
        candidate = base / name / "model.pt"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no model.pt under {base}/best or {base}/latest")


def _load_policy(model_path: Path, device: torch.device):
    from algorithms.happo import EntityHAPPOReferencePolicy, HAPPOReferencePolicy

    meta_path = model_path.parent / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    arch = str(meta.get("policy_arch", "flat"))
    if arch == "entity_attention":
        policy = EntityHAPPOReferencePolicy(
            entity_dim=int(meta.get("entity_dim", 19)),
            critic_state_dim=int(meta.get("critic_state_dim", 480)),
            action_dim=3,
        ).to(device)
    elif arch == "flat":
        policy = HAPPOReferencePolicy(
            actor_obs_dim=int(meta.get("actor_obs_dim", 96)),
            critic_state_dim=int(meta.get("critic_state_dim", 480)),
            action_dim=3,
        ).to(device)
    else:
        raise ValueError(f"unsupported policy_arch: {arch}")
    policy.load(model_path, map_location=device)
    policy.eval()
    return policy, arch


def _role_ids(env) -> list[int]:
    return [0 if env.agent_roles.get(rid) == "mav" else 1 for rid in env.red_ids]


def _policy_actions(policy, adapter, env, obs, info, device: torch.device) -> np.ndarray:
    adapted = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
    actor_obs = np.stack([
        adapted["actor_obs"].get(rid, np.zeros(adapter.flat_actor_obs_dim, dtype=np.float32))
        for rid in env.red_ids
    ])
    with torch.no_grad():
        out = policy.act(
            torch.as_tensor(actor_obs, dtype=torch.float32, device=device),
            roles=_role_ids(env),
            critic_state=torch.as_tensor(adapted["critic_state"], dtype=torch.float32, device=device),
            deterministic=True,
        )
    return out["action"].detach().cpu().numpy().astype(np.float32)


def _target_by_id(env, target_id: str):
    if not target_id:
        return None
    return getattr(env, "blue_planes", {}).get(target_id) or getattr(env, "red_planes", {}).get(target_id)


def _row_before_step(env, rid: str, action: np.ndarray, source: str, ep: int, step: int) -> dict[str, Any] | None:
    sim = env.red_planes.get(rid)
    if sim is None or not bool(getattr(sim, "is_alive", False)):
        return None
    diag = _diagnose_red_shooter(env, rid)
    target = _target_by_id(env, str(diag.get("target_id", "")))
    if target is None:
        return None
    current_heading = float(sim.get_rpy()[2])
    target_bearing = _bearing(np.asarray(sim.get_position()), np.asarray(target.get_position()))
    decoded_heading = float(action[1]) * math.pi
    return {
        "source": source,
        "episode_id": ep,
        "step": step,
        "red_id": rid,
        "target_id": diag.get("target_id", ""),
        "current_heading_rad": current_heading,
        "target_bearing_rad": target_bearing,
        "heading_error_rad": _wrap_pi(target_bearing - current_heading),
        "action_heading": float(action[1]),
        "decoded_target_heading_rad": decoded_heading,
        "command_error_to_bearing_rad": _wrap_pi(decoded_heading - target_bearing),
        "AO_rad": diag.get("ao_rad", ""),
        "AO_next_rad": "",
        "delta_AO_rad": "",
        "range_m": diag.get("range_m", ""),
        "range_next_m": "",
        "delta_range_m": "",
        "launch_allowed": bool(diag.get("launch_allowed_predicted", False)),
        "block_reason": diag.get("launch_block_reason", ""),
        "action_pitch": float(action[0]),
        "action_speed": float(action[2]),
    }


def _fill_next_geometry(env, row: dict[str, Any]) -> None:
    sim = env.red_planes.get(str(row["red_id"]))
    target = _target_by_id(env, str(row["target_id"]))
    if sim is None or target is None:
        return
    try:
        rng_next, ao_next, _ta_next = _geometry_pair(sim, target)
    except Exception:
        return
    row["AO_next_rad"] = ao_next
    row["range_next_m"] = rng_next
    if row["AO_rad"] != "":
        row["delta_AO_rad"] = float(ao_next) - float(row["AO_rad"])
    if row["range_m"] != "":
        row["delta_range_m"] = float(rng_next) - float(row["range_m"])


def rollout_source(
    source: str,
    config: str,
    episodes: int,
    max_steps: int,
    device: torch.device,
    opponent_policy: str,
    model_path: Path | None = None,
) -> tuple[list[dict[str, Any]], str]:
    from uav_env import make_env
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2

    adapter = HeteroObsAdapterV2()
    policy = None
    arch = "oracle"
    if model_path is not None:
        policy, arch = _load_policy(model_path, device)

    rows: list[dict[str, Any]] = []
    for ep in range(episodes):
        env = make_env(config, env_type="jsbsim_hetero", suppress_jsbsim_output=True)
        try:
            obs, info = env.reset(seed=2000 + ep)
            for step in range(1, max_steps + 1):
                if source == "direct_chase_oracle":
                    action_dict = {}
                    for rid in env.red_ids:
                        if env.agent_roles.get(rid) == "mav":
                            action_dict[rid] = safe_mav_action()
                        else:
                            action_dict[rid] = direct_chase_action(env, rid)
                    actions_np = np.stack([action_dict[rid] for rid in env.red_ids])
                else:
                    assert policy is not None
                    actions_np = _policy_actions(policy, adapter, env, obs, info, device)
                    action_dict = {rid: actions_np[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}

                pending: list[dict[str, Any]] = []
                for i, rid in enumerate(env.red_ids):
                    if env.agent_roles.get(rid) == "mav":
                        continue
                    row = _row_before_step(env, rid, actions_np[i], source, ep, step)
                    if row is not None:
                        pending.append(row)

                action_dict.update(blue_actions(env, obs, opponent_policy))
                obs, _rewards, terminated, truncated, info = env.step(action_dict)
                for row in pending:
                    _fill_next_geometry(env, row)
                    rows.append(row)
                if team_done(terminated, truncated):
                    break
        finally:
            env.close()
    return rows, arch


def summarize_rows(source: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    actions = np.asarray([
        [float(r["action_pitch"]), float(r["action_heading"]), float(r["action_speed"])]
        for r in rows
    ], dtype=np.float32)
    if actions.size == 0:
        actions = np.zeros((0, 3), dtype=np.float32)
    pitch = _stats(actions[:, 0]) if len(actions) else _stats(np.asarray([]))
    heading = _stats(actions[:, 1]) if len(actions) else _stats(np.asarray([]))
    speed = _stats(actions[:, 2]) if len(actions) else _stats(np.asarray([]))

    def nums(key: str) -> np.ndarray:
        vals = [float(r[key]) for r in rows if r.get(key, "") != ""]
        return np.asarray(vals, dtype=np.float64)

    command_error = np.abs(nums("command_error_to_bearing_rad"))
    delta_ao = nums("delta_AO_rad")
    delta_range = nums("delta_range_m")
    block_counts = Counter(str(r.get("block_reason", "")) for r in rows)
    return {
        "source": source,
        "samples": len(rows),
        "action_pitch_min": pitch["min"],
        "action_pitch_max": pitch["max"],
        "action_pitch_mean": pitch["mean"],
        "action_pitch_std": pitch["std"],
        "action_heading_min": heading["min"],
        "action_heading_max": heading["max"],
        "action_heading_mean": heading["mean"],
        "action_heading_std": heading["std"],
        "action_speed_min": speed["min"],
        "action_speed_max": speed["max"],
        "action_speed_mean": speed["mean"],
        "action_speed_std": speed["std"],
        "action_saturation_rate": float(np.mean(np.any(np.abs(actions) > 0.95, axis=1))) if len(actions) else 0.0,
        "heading_command_abs_error_mean_rad": float(np.mean(command_error)) if command_error.size else 0.0,
        "heading_command_abs_error_mean_deg": float(np.degrees(np.mean(command_error))) if command_error.size else 0.0,
        "ao_delta_mean_rad": float(np.mean(delta_ao)) if delta_ao.size else 0.0,
        "ao_delta_mean_deg": float(np.degrees(np.mean(delta_ao))) if delta_ao.size else 0.0,
        "ao_reduced_rate": float(np.mean(delta_ao < 0.0)) if delta_ao.size else 0.0,
        "range_delta_mean_m": float(np.mean(delta_range)) if delta_range.size else 0.0,
        "range_reduced_rate": float(np.mean(delta_range < 0.0)) if delta_range.size else 0.0,
        "launch_allowed_rate": float(np.mean([bool(r.get("launch_allowed", False)) for r in rows])) if rows else 0.0,
        "dominant_block_reason": block_counts.most_common(1)[0][0] if block_counts else "",
        "block_reason_counts_json": json.dumps(dict(block_counts), sort_keys=True),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_markdown(path: Path, action_decode: list[dict[str, str]],
                    dataset: dict[str, Any], summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Heading / Action Alignment Diagnostics",
        "",
        "## Action Decode Chain",
        "",
        "| field | current meaning | unit/range | code location | dataset consistency |",
        "|---|---|---|---|---|",
    ]
    for row in action_decode:
        lines.append(
            f"| {row['field']} | {row['meaning']} | {row['unit_range']} | "
            f"{row['code']} | {row['dataset_consistency']} |"
        )
    lines.extend([
        "",
        "## Direct-Chase Dataset",
        "",
        f"- path: `{dataset['path']}`",
        f"- samples: `{dataset['samples']}`",
        f"- pitch min/max/mean/std: `{dataset['action_pitch']}`",
        f"- heading min/max/mean/std: `{dataset['action_heading']}`",
        f"- speed min/max/mean/std: `{dataset['action_speed']}`",
        f"- action_saturation_rate: `{dataset['action_saturation_rate']}`",
        f"- launch_range_flag_mean: `{dataset.get('launch_range_flag_mean')}`",
        f"- launch_angle_flag_mean: `{dataset.get('launch_angle_flag_mean')}`",
        f"- launch_envelope_flag_mean: `{dataset.get('launch_envelope_flag_mean')}`",
        f"- static heading/bearing check: `{dataset['heading_bearing_static_check']}`",
        "",
        "## Closed-Loop Alignment Summary",
        "",
        "| source | samples | cmd err deg | AO delta deg | AO reduced | range delta m | range reduced | block | heading mean | sat |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ])
    for row in summaries:
        lines.append(
            f"| {row['source']} | {row['samples']} | "
            f"{row['heading_command_abs_error_mean_deg']:.2f} | "
            f"{row['ao_delta_mean_deg']:.3f} | {row['ao_reduced_rate']:.3f} | "
            f"{row['range_delta_mean_m']:.1f} | {row['range_reduced_rate']:.3f} | "
            f"{row['dominant_block_reason']} | {row['action_heading_mean']:.3f} | "
            f"{row['action_saturation_rate']:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `action[1]` is an absolute target heading normalized by pi; it is not a relative turn command.",
        "- The direct-chase oracle computes `atan2(delta_east, delta_north) / pi`, which matches the environment decode.",
        "- If command error is small but AO does not decrease, the issue is likely closed-loop aircraft response or AO/TA timing.",
        "- If command error is large, the learned policy is not commanding the target bearing even if range improves.",
        "- This report does not change environment mechanics or training behavior.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _action_decode_table() -> list[dict[str, str]]:
    return [
        {
            "field": "action[0]",
            "meaning": "target pitch",
            "unit_range": "normalized [-1,1] -> [-90,+90] deg",
            "code": "uav_env/JSBSim/env.py::_parse_actions",
            "dataset_consistency": "direct_chase pitch uses clipped altitude delta / 5000",
        },
        {
            "field": "action[1]",
            "meaning": "absolute target heading",
            "unit_range": "normalized [-1,1] -> [-pi,+pi] rad",
            "code": "uav_env/JSBSim/env.py::_parse_actions",
            "dataset_consistency": "direct_chase heading uses atan2(de,dn)/pi, same absolute convention",
        },
        {
            "field": "action[2]",
            "meaning": "target velocity",
            "unit_range": "normalized [-1,1] -> [102,408] m/s",
            "code": "uav_env/JSBSim/env.py::_parse_actions",
            "dataset_consistency": "direct_chase uses fixed speed 0.8",
        },
        {
            "field": "AO",
            "meaning": "2D angle-off between ego velocity and LOS to target",
            "unit_range": "rad, launch threshold <45 deg",
            "code": "uav_env/JSBSim/utils.get2d_AO_TA_R via launch diagnostics",
            "dataset_consistency": "dataset only stores launch_angle_flag, not raw AO",
        },
    ]


def _read_fake_summary(path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    p = _as_path(path)
    rows: list[dict[str, Any]] = []
    with p.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    sources = sorted(set(str(r.get("source", "fake")) for r in rows))
    return rows, [summarize_rows(src, [r for r in rows if str(r.get("source", "fake")) == src]) for src in sources]


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze action heading alignment for oracle and learned policies")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-policy", default="brma_rule")
    parser.add_argument("--output-csv", default="outputs/heading_alignment_diagnostics_summary.csv")
    parser.add_argument("--output-md", default="outputs/heading_alignment_diagnostics_summary.md")
    parser.add_argument("--detail-csv", default="outputs/heading_alignment_diagnostics_detail.csv")
    parser.add_argument("--fake-input", default=None,
                        help="Optional CSV with ROW_FIELDS for fast summary generation in tests.")
    parser.add_argument("--skip-rollout", action="store_true")
    args = parser.parse_args()

    summary_csv = _safe_output_path(args.output_csv)
    summary_md = _safe_output_path(args.output_md)
    detail_csv = _safe_output_path(args.detail_csv)

    dataset = analyze_dataset(args.dataset)
    action_decode = _action_decode_table()

    if args.fake_input:
        detail_rows, summaries = _read_fake_summary(args.fake_input)
    else:
        detail_rows = []
        summaries = []
        if not args.skip_rollout:
            device = torch.device(args.device)
            oracle_rows, _oracle_arch = rollout_source(
                "direct_chase_oracle",
                args.config,
                args.episodes,
                args.max_steps,
                device,
                args.opponent_policy,
                None,
            )
            detail_rows.extend(oracle_rows)
            summaries.append(summarize_rows("direct_chase_oracle", oracle_rows))
            for label, spec in MODEL_SPECS.items():
                try:
                    model_path = _resolve_model(spec["output_dir"], spec["checkpoint_name"])
                except FileNotFoundError:
                    continue
                rows, _arch = rollout_source(
                    label,
                    args.config,
                    args.episodes,
                    args.max_steps,
                    device,
                    args.opponent_policy,
                    model_path,
                )
                detail_rows.extend(rows)
                summaries.append(summarize_rows(label, rows))

    _write_csv(detail_csv, detail_rows, ROW_FIELDS)
    _write_csv(summary_csv, summaries, SUMMARY_FIELDS)
    _write_markdown(summary_md, action_decode, dataset, summaries)
    print(f"summary_csv: {summary_csv}", flush=True)
    print(f"summary_md: {summary_md}", flush=True)
    print(f"detail_csv: {detail_csv}", flush=True)
    for row in summaries:
        print(
            f"{row['source']}: cmd_err_deg={row['heading_command_abs_error_mean_deg']:.2f} "
            f"ao_delta_deg={row['ao_delta_mean_deg']:.3f} "
            f"ao_reduced={row['ao_reduced_rate']:.3f} "
            f"block={row['dominant_block_reason']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
