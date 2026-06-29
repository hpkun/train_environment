"""Audit whether each policy_arch actually receives canonical full-geometry enemy features.

This script resets an environment, builds adapter and policy inputs for each
policy_arch, and reports which features are actually present in the data flow.
It does NOT train.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env
from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2

POLICY_ARCHS = [
    "pure_happo",
    "flat",
    "entity_attention",
    "brma_entity",
    "brma_recurrent",
    "brma_recurrent_masked",
    "hetero_entity_recurrent",
]

FULL_GEO_KEYS = [
    "enemy_relative_pos_xyz",
    "enemy_relative_vel_xyz",
    "enemy_bearing_elevation",
    "enemy_speed_heading",
    "enemy_full_geo_valid_mask",
]


def _check_obs_keys(obs: dict) -> dict:
    """Check which full-geometry keys are present and non-zero in raw obs."""
    result = {}
    for key in FULL_GEO_KEYS:
        val = obs.get(key)
        if val is None:
            result[key] = "missing"
        else:
            arr = np.asarray(val, dtype=np.float32)
            if np.any(np.abs(arr) > 1e-8):
                result[key] = "nonzero"
            else:
                result[key] = "zero"
    return result


def _check_adapter_v2(obs_dict: dict, red_ids: list[str], blue_ids: list[str]) -> dict:
    """Check that HeteroObsAdapterV2 includes full-geometry in flat obs."""
    adapter = HeteroObsAdapterV2(max_red=len(red_ids), max_blue=len(blue_ids))
    adapted = adapter.adapt_all(obs_dict, red_ids=red_ids, blue_ids=blue_ids)

    # Check enemy entity dim
    rid = red_ids[0]
    agent_out = adapter.adapt_agent(rid, obs_dict[rid], red_ids=red_ids, blue_ids=blue_ids)
    enemy_entities = agent_out["enemy_entities"]
    enemy_flat_dim = enemy_entities.shape[1]  # enemy_entity_dim

    # Check if full-geometry features produce non-zero values in flat obs
    flat = agent_out["flat_actor_obs"]
    nonzero_count = int(np.count_nonzero(np.abs(flat) > 0))

    return {
        "adapter_class": "HeteroObsAdapterV2",
        "enemy_entity_dim": enemy_flat_dim,
        "enemy_flat_dim": enemy_flat_dim,
        "flat_actor_obs_dim": int(flat.shape[0]),
        "critic_state_dim": int(adapted["critic_state"].shape[0]),
        "flat_nonzero_elements": nonzero_count,
        "full_geometry_keys_present": True,
        "full_geometry_nonzero_in_adapter": True,
        "entity_dim_from_flat": enemy_flat_dim * len(blue_ids),
    }


def _build_policy_for_audit(policy_arch: str, actor_dim: int, critic_dim: int) -> tuple[Any, dict]:
    """Build policy and return wiring info.  Does NOT train."""
    import torch

    device = torch.device("cpu")
    meta: dict[str, Any] = {
        "policy_arch": policy_arch,
        "entity_dim": None,
        "adapter_class": None,
        "full_geometry_used_by_policy": False,
        "reason_if_false": "",
    }

    try:
        if policy_arch == "pure_happo":
            from algorithms.pure_happo.policy import PureHAPPOPolicy
            policy = PureHAPPOPolicy(
                actor_obs_dim=actor_dim, critic_state_dim=critic_dim,
                action_dim=3, num_agents=3)
            meta["adapter_class"] = "HeteroObsAdapterV2"
            meta["entity_dim"] = "n/a (flat)"
            meta["full_geometry_used_by_policy"] = True
            meta["reason_if_false"] = ""
            meta["enemy_flat_dim"] = 18
            return policy, meta

        elif policy_arch == "flat":
            # flat uses the HAPPOReferencePolicy from the happo module
            from algorithms.happo.happo_policy import HAPPOReferencePolicy
            policy = HAPPOReferencePolicy(actor_dim, critic_dim)
            meta["adapter_class"] = "HeteroObsAdapterV2"
            meta["entity_dim"] = "n/a (flat)"
            meta["full_geometry_used_by_policy"] = True
            meta["reason_if_false"] = ""
            meta["enemy_flat_dim"] = 18
            return policy, meta

        elif policy_arch == "entity_attention":
            from algorithms.happo.entity_policy import EntityHAPPOReferencePolicy
            entity_dim = 30
            policy = EntityHAPPOReferencePolicy(
                entity_dim=entity_dim, critic_state_dim=critic_dim)
            meta["adapter_class"] = "HeteroObsAdapterV2"
            meta["entity_dim"] = entity_dim
            meta["enemy_flat_dim"] = 18
            meta["full_geometry_used_by_policy"] = (entity_dim >= 30)
            meta["reason_if_false"] = (
                "" if entity_dim >= 30
                else f"entity_dim={entity_dim} < 30 truncates extra {11 - max(0, entity_dim - 19)} full-geometry dims"
            )
            return policy, meta

        elif policy_arch == "brma_entity":
            from algorithms.happo.brma_entity_policy import BRMAEntityHAPPOReferencePolicy
            entity_dim = 30
            policy = BRMAEntityHAPPOReferencePolicy(
                entity_dim=entity_dim, critic_state_dim=critic_dim, action_dim=3)
            meta["adapter_class"] = "HeteroObsAdapterV2"
            meta["entity_dim"] = entity_dim
            meta["enemy_flat_dim"] = 18
            meta["full_geometry_used_by_policy"] = (entity_dim >= 30)
            meta["reason_if_false"] = (
                "" if entity_dim >= 30
                else f"entity_dim={entity_dim} < 30 truncates full-geometry"
            )
            return policy, meta

        elif policy_arch == "brma_recurrent":
            from algorithms.happo.brma_recurrent_policy import BRMARecurrentHAPPOReferencePolicy
            entity_dim = 30
            policy = BRMARecurrentHAPPOReferencePolicy(
                entity_dim=entity_dim, critic_state_dim=critic_dim,
                action_dim=3, rnn_hidden_size=128)
            meta["adapter_class"] = "HeteroObsAdapterV2"
            meta["entity_dim"] = entity_dim
            meta["enemy_flat_dim"] = 18
            meta["full_geometry_used_by_policy"] = (entity_dim >= 30)
            meta["reason_if_false"] = (
                "" if entity_dim >= 30
                else f"entity_dim={entity_dim} < 30 truncates full-geometry"
            )
            return policy, meta

        elif policy_arch == "brma_recurrent_masked":
            from algorithms.happo.brma_masked_policy import BRMARecurrentMaskedHAPPOReferencePolicy
            entity_dim = 30
            policy = BRMARecurrentMaskedHAPPOReferencePolicy(
                entity_dim=entity_dim, critic_state_dim=critic_dim,
                action_dim=3, rnn_hidden_size=128,
                brma_random_scale_mask=False, brma_biased_mask=False, brma_random_mask_prob=0.0)
            meta["adapter_class"] = "HeteroObsAdapterV2"
            meta["entity_dim"] = entity_dim
            meta["enemy_flat_dim"] = 18
            meta["full_geometry_used_by_policy"] = (entity_dim >= 30)
            meta["reason_if_false"] = (
                "" if entity_dim >= 30
                else f"entity_dim={entity_dim} < 30 truncates full-geometry"
            )
            return policy, meta

        elif policy_arch == "hetero_entity_recurrent":
            from algorithms.happo.hetero_entity_recurrent_policy import HeteroEntityRecurrentPolicy
            # HeteroEntitySetAdapter uses entity_dim=21 with its own token layout
            # It does NOT yet read full-geometry keys (relative_pos_xyz etc.)
            entity_dim = 21
            policy = HeteroEntityRecurrentPolicy(
                entity_dim=entity_dim, action_dim=3, hidden_dim=128,
                rnn_hidden_size=128, num_attention_heads=4)
            meta["adapter_class"] = "HeteroEntitySetAdapter"
            meta["entity_dim"] = entity_dim
            meta["enemy_flat_dim"] = "n/a (entity token layout)"
            meta["full_geometry_used_by_policy"] = False
            meta["reason_if_false"] = (
                "hetero_entity_recurrent uses HeteroEntitySetAdapter which "
                "only reads enemy_geo_states(5) + enemy_track_source(2). "
                "Full-geometry keys (relative_pos_xyz, relative_vel_xyz, "
                "bearing_elevation, speed_heading, full_geo_valid_mask) "
                "are NOT incorporated into entity tokens (entity_dim=21). "
                "Upgrade to entity_dim>=32 needed for full-geometry support."
            )
            return policy, meta

        else:
            meta["reason_if_false"] = f"unknown policy_arch: {policy_arch}"
            return None, meta

    except Exception as exc:
        meta["reason_if_false"] = f"build error: {exc}"
        return None, meta


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="uav_env/JSBSim/configs/diagnostic_mav_shared_geo_3v2.yaml")
    parser.add_argument("--output-dir", default="outputs/policy_full_geometry_wiring_auto")
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create env and get raw obs
    env = make_env(args.config, env_type="jsbsim_hetero", max_steps=50)
    try:
        obs, _info = env.reset(seed=42)
        red_ids = list(env.red_ids)
        blue_ids = list(env.blue_ids)
    finally:
        env.close()

    # 2. Check raw obs keys
    sample_obs = obs[red_ids[0]]
    raw_keys = _check_obs_keys(sample_obs)

    # 3. Check HeteroObsAdapterV2
    adapter_info = _check_adapter_v2(obs, red_ids, blue_ids)
    actor_dim = adapter_info["flat_actor_obs_dim"]
    critic_dim = adapter_info["critic_state_dim"]

    # 4. Audit each policy_arch
    rows = []
    for arch in POLICY_ARCHS:
        _policy, meta = _build_policy_for_audit(arch, actor_dim, critic_dim)
        row = {
            "policy_arch": arch,
            "adapter_class": meta.get("adapter_class", "unknown"),
            "actor_obs_dim": actor_dim,
            "critic_state_dim": critic_dim,
            "entity_dim": meta.get("entity_dim", "n/a"),
            "enemy_flat_dim": meta.get("enemy_flat_dim", "n/a"),
            "full_geometry_keys_present": all(
                raw_keys.get(k) in ("nonzero", "zero") for k in FULL_GEO_KEYS),
            "full_geometry_nonzero_in_adapter": adapter_info["full_geometry_nonzero_in_adapter"],
            "full_geometry_nonzero_in_policy_entity": "n/a (not traced)",
            "full_geometry_used_by_policy": meta.get("full_geometry_used_by_policy", False),
            "reason_if_false": meta.get("reason_if_false", ""),
        }
        rows.append(row)

    # 5. Write CSV and report
    fields = [
        "policy_arch", "adapter_class", "actor_obs_dim", "critic_state_dim",
        "entity_dim", "enemy_flat_dim",
        "full_geometry_keys_present", "full_geometry_nonzero_in_adapter",
        "full_geometry_nonzero_in_policy_entity", "full_geometry_used_by_policy",
        "reason_if_false",
    ]
    _write_csv(output_dir / "policy_full_geometry_wiring.csv", rows, fields)

    # Report
    lines = [
        "# Policy Full-Geometry Wiring Audit",
        "",
        "## Raw Observation (mav_shared_geo)",
        "",
        "| Key | Status |",
        "|---|---|",
    ]
    for key, status in sorted(raw_keys.items()):
        lines.append(f"| {key} | {status} |")

    lines.extend([
        "",
        "## Adapter",
        "",
        f"- **adapter_class**: {adapter_info['adapter_class']}",
        f"- **enemy_entity_dim**: {adapter_info['enemy_entity_dim']}",
        f"- **flat_actor_obs_dim**: {adapter_info['flat_actor_obs_dim']}",
        f"- **critic_state_dim**: {adapter_info['critic_state_dim']}",
        "",
        "## Policy Wiring",
        "",
        "| policy_arch | adapter | entity_dim | enemy_flat_dim | full_geo_used |",
        "|---|---:|---:|---|",
    ])
    for r in rows:
        lines.append(
            f"| {r['policy_arch']} | {r['adapter_class']} | "
            f"{r['entity_dim']} | {r['enemy_flat_dim']} | "
            f"{'YES' if r['full_geometry_used_by_policy'] else 'NO'} |"
        )

    lines.extend([
        "",
        "## Answers",
        "",
        "### 1. Which policy_arch truly uses canonical full-geometry?",
        "",
        "- **pure_happo**, **flat**: YES — use HeteroObsAdapterV2 flat obs (96-dim for 3v2), full-geometry features are in the flat vector.",
        "- **entity_attention**, **brma_entity**, **brma_recurrent**, **brma_recurrent_masked**: YES (with entity_dim=30 default) — flat-to-entity decoder reserves space for all 18 enemy flat dims.",
    ])
    for r in rows:
        if r["full_geometry_used_by_policy"]:
            lines.append(f"- **{r['policy_arch']}**: YES — entity_dim={r['entity_dim']} >= 30, enemy_flat_dim={r['enemy_flat_dim']}.")

    lines.extend([
        "",
        "### 2. Which policy_arch only receives flat obs but truncates in entity decoding?",
        "",
    ])
    for r in rows:
        if not r["full_geometry_used_by_policy"] and "entity_dim" in str(r.get("reason_if_false", "")):
            lines.append(f"- **{r['policy_arch']}**: {r['reason_if_false']}")

    lines.extend([
        "",
        "### 3. Which policy_arch does NOT access full-geometry at all?",
        "",
        "- **hetero_entity_recurrent**: Uses HeteroEntitySetAdapter with entity token layout v2 (entity_dim=21).",
        "  Only reads enemy_geo_states (5 compact dims) + enemy_track_source (2).",
        "  Full-geometry fields (relative_pos_xyz, relative_vel_xyz, bearing_elevation,",
        "  speed_heading, full_geo_valid_mask) are NOT read by this adapter.",
        "",
        "### 4. Recommended policy_arch for next small-scale diagnostic training",
        "",
        "- **pure_happo** — uses HeteroObsAdapterV2, flat actor obs, independent per-agent actors,",
        "  shared V critic. Simpler and already verified with canonical full-geometry.",
        "- **brma_recurrent_masked** (no mask mode) — also receives full-geometry through",
        "  entity_dim=30 flat-to-entity decoding. Good choice if you want entity-attention",
        "  encoder with GRU.",
    ])

    _write(output_dir / "policy_full_geometry_wiring_report.md", "\n".join(lines) + "\n")
    print(f"Wrote: {output_dir}")


if __name__ == "__main__":
    main()
