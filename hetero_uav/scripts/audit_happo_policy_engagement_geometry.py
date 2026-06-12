"""Audit whether trained HAPPO policies enter the red launch envelope."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from red_attack_audit_utils import (
    DEFAULT_CONFIG,
    blue_actions,
    collect_step_counts,
    geometry,
    make_env,
    summarize_numbers,
    team_done,
    write_json,
    write_md,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _role_ids(env) -> list[int]:
    return [0 if env.agent_roles.get(rid) == "mav" else 1 for rid in env.red_ids]


def _load_policy(model: Path, device: str):
    from algorithms.happo import HAPPOReferencePolicy

    meta = json.loads((model.parent / "meta.json").read_text(encoding="utf-8"))
    policy = HAPPOReferencePolicy(
        int(meta.get("actor_obs_dim", 96)),
        int(meta.get("critic_state_dim", 480)),
    ).to(torch.device(device))
    policy.load(model, map_location=torch.device(device))
    policy.eval()
    return policy


def _policy_actions(policy, adapter, env, obs, info, device: str) -> np.ndarray:
    adapted = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
    actor_obs = np.stack([
        adapted["actor_obs"].get(rid, np.zeros(adapter.flat_actor_obs_dim, dtype=np.float32))
        for rid in env.red_ids
    ])
    with torch.no_grad():
        out = policy.act(
            torch.as_tensor(actor_obs, device=torch.device(device)),
            roles=_role_ids(env),
            critic_state=torch.as_tensor(adapted["critic_state"], device=torch.device(device)),
            deterministic=True,
        )
    return out["action"].cpu().numpy().astype(np.float32)


def run_checkpoint(
    name: str,
    model: Path,
    config: str,
    episodes: int,
    max_steps: int,
    opponent_policy: str,
    device: str,
) -> dict:
    if not model.exists():
        return {"checkpoint": name, "model_path": str(model), "missing": True}
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2

    policy = _load_policy(model, device)
    adapter = HeteroObsAdapterV2()
    distances, atas, aspects = [], [], []
    range_flags, angle_flags, envelope_flags = [], [], []
    action_values = []
    red_fired = red_hits = 0
    ep_records = []

    for ep in range(episodes):
        env = make_env(config)
        try:
            obs, info = env.reset(seed=ep)
            ep_min_dist = float("inf")
            ep_envelope_count = 0
            ep_steps = 0
            for step in range(1, max_steps + 1):
                acts = _policy_actions(policy, adapter, env, obs, info, device)
                actions = {rid: acts[i] for i, rid in enumerate(env.red_ids)}
                actions.update(blue_actions(env, obs, opponent_policy))
                for i, rid in enumerate(env.red_ids):
                    if env.agent_roles.get(rid) == "mav":
                        continue
                    g = geometry(env, rid)
                    if g.get("target_id"):
                        distances.append(float(g["distance_m"]))
                        atas.append(float(g["ata_rad"]))
                        aspects.append(float(g["aspect_rad"]))
                        range_flags.append(bool(g["launch_condition_distance"]))
                        angle_flags.append(bool(g["launch_condition_angle"]))
                        envelope_flags.append(bool(g["launch_condition_all"]))
                        ep_min_dist = min(ep_min_dist, float(g["distance_m"]))
                        ep_envelope_count += int(bool(g["launch_condition_all"]))
                    action_values.append(acts[i])
                obs, _rewards, terminated, truncated, info = env.step(actions)
                counts = collect_step_counts(info)
                red_fired += counts["red_fired"]
                red_hits = max(red_hits, counts["red_hits_total"])
                ep_steps = step
                if team_done(terminated, truncated):
                    break
            ep_records.append({
                "episode": ep,
                "steps": ep_steps,
                "min_distance_m": None if not np.isfinite(ep_min_dist) else float(ep_min_dist),
                "envelope_steps": ep_envelope_count,
            })
        finally:
            env.close()

    actions_np = np.asarray(action_values, dtype=np.float32) if action_values else np.zeros((0, 3), dtype=np.float32)
    launch_range_rate = float(np.mean(range_flags)) if range_flags else 0.0
    launch_angle_rate = float(np.mean(angle_flags)) if angle_flags else 0.0
    envelope_rate = float(np.mean(envelope_flags)) if envelope_flags else 0.0
    action_saturation = float(np.mean(np.any(np.abs(actions_np) > 0.95, axis=1))) if actions_np.size else 0.0
    action_mean = np.mean(actions_np, axis=0).tolist() if actions_np.size else [0.0, 0.0, 0.0]
    long_range = (summarize_numbers(distances)["min"] or float("inf")) > 10000.0
    avoids = envelope_rate < 0.01 and red_fired == 0
    if envelope_rate > 0.0 and red_fired == 0:
        conclusion = "policy_enters_envelope_but_red_does_not_fire"
    elif red_fired > 0:
        conclusion = "policy_triggers_red_fire"
    elif avoids:
        conclusion = "policy_avoids_or_never_reaches_engagement"
    else:
        conclusion = "inconclusive"

    return {
        "checkpoint": name,
        "model_path": str(model),
        "episodes": episodes,
        "distance_to_nearest_blue_m": summarize_numbers(distances),
        "ata_rad": summarize_numbers(atas),
        "aspect_rad": summarize_numbers(aspects),
        "launch_range_rate": launch_range_rate,
        "launch_angle_rate": launch_angle_rate,
        "launch_envelope_rate": envelope_rate,
        "action_mean": action_mean,
        "action_saturation_rate": action_saturation,
        "speed_action_mean": float(action_mean[2]) if action_mean else 0.0,
        "long_term_far_from_enemy": bool(long_range),
        "policy_avoids_engagement": bool(avoids),
        "red_missiles_fired_total": red_fired,
        "red_missile_hits_total": red_hits,
        "conclusion": conclusion,
        "episodes_detail": ep_records,
    }


def build_report(args) -> dict:
    exp = Path(args.experiment_dir)
    records = [
        run_checkpoint("best", exp / "best" / "model.pt", args.config, args.episodes, args.max_steps, args.opponent_policy, args.device),
        run_checkpoint("latest", exp / "latest" / "model.pt", args.config, args.episodes, args.max_steps, args.opponent_policy, args.device),
    ]
    return {
        "experiment_dir": args.experiment_dir,
        "config": args.config,
        "episodes": args.episodes,
        "records": records,
        "overall_conclusion": (
            "happo_policy_not_reliably_engaging"
            if all(r.get("red_missiles_fired_total", 0) == 0 for r in records if not r.get("missing"))
            else "happo_policy_can_trigger_some_red_fire"
        ),
    }


def write_report_md(data: dict, output_md: str) -> None:
    lines = ["# HAPPO Policy Engagement Geometry", ""]
    for rec in data["records"]:
        lines.extend([
            f"## {rec['checkpoint']}",
            f"- launch_range_rate: {rec.get('launch_range_rate')}",
            f"- launch_angle_rate: {rec.get('launch_angle_rate')}",
            f"- launch_envelope_rate: {rec.get('launch_envelope_rate')}",
            f"- red_missiles_fired_total: {rec.get('red_missiles_fired_total')}",
            f"- action_saturation_rate: {rec.get('action_saturation_rate')}",
            f"- conclusion: {rec.get('conclusion')}",
            "",
        ])
    lines.append(f"overall_conclusion: {data['overall_conclusion']}")
    write_md(output_md, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit HAPPO policy engagement geometry")
    parser.add_argument("--experiment-dir", default="outputs/happo_3v2_reference_f16_mav_surrogate_1m_fast")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--opponent-policy", choices=["zero", "brma_rule"], default="brma_rule")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-json",
        default="outputs/environment_audit/happo_engagement_geometry/happo_policy_engagement_geometry.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/environment_audit/happo_engagement_geometry/happo_policy_engagement_geometry.md",
    )
    args = parser.parse_args()
    data = build_report(args)
    out_json = write_json(args.output_json, data)
    write_report_md(data, args.output_md)
    print(f"output_json: {out_json}")
    print(f"overall_conclusion: {data['overall_conclusion']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

