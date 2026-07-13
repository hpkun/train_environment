from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.mappo.opponent_policy import OpponentPolicy
from scripts.eval_policy_launch_diagnostics import _build_policy, _load_meta, _policy_actions
from uav_env import make_env
from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
from uav_env.JSBSim.envs.paper_formula_v5 import V5_COMPONENT_FIELDS


def _done(flags: dict) -> bool:
    return bool(flags) and all(bool(value) for value in flags.values())


def _load_checkpoint(path: str | None, device: torch.device):
    if not path:
        return None
    checkpoint = Path(path)
    policy = _build_policy(_load_meta(checkpoint), device)
    policy.load(checkpoint, map_location=device)
    policy.eval()
    return policy


def _gate_counts(info: dict, counts: dict[str, float]) -> None:
    for record in info.get("__launch_gate_diagnostics__", []) or []:
        if not isinstance(record, dict) or not str(record.get("agent_id", "")).startswith("red_"):
            continue
        counts["records"] += 1
        for name, key in {"track": "any_track_pass", "range": "any_range_pass",
                          "ao": "any_ata_pass", "ta": "any_ta_pass",
                          "lock": "any_lock_mature", "allowed": "any_launch"}.items():
            counts[name] += float(bool(record.get(key, 0)))


def collect_source(*, config: str, source: str, episodes: int, seed: int,
                   device: torch.device, checkpoint: str | None = None) -> list[dict]:
    policy = _load_checkpoint(checkpoint, device)
    adapter = HeteroObsAdapterV2()
    rows = []
    for episode in range(episodes):
        env = make_env(config, max_steps=1000, suppress_jsbsim_output=True)
        opponent = OpponentPolicy(mode="tam_greedy_rule", seed=seed + episode + 10000)
        obs, info = env.reset(seed=seed + episode)
        sums = {field: 0.0 for field in V5_COMPONENT_FIELDS}
        alive_counts = {field: 0 for field in V5_COMPONENT_FIELDS}
        gates = {key: 0.0 for key in ("records", "track", "range", "ao", "ta", "lock", "allowed")}
        hidden = None
        terminal_observed = False
        terminated = truncated = {}
        try:
            for step in range(1000):
                if source == "random":
                    red_actions = {rid: env.action_space[rid].sample() for rid in env.red_ids}
                elif source == "fixed_straight":
                    red_actions = {rid: np.zeros(3, dtype=np.float32) for rid in env.red_ids}
                else:
                    actions, hidden = _policy_actions(policy, adapter, env, obs, info, device, hidden)
                    red_actions = {rid: actions[idx] for idx, rid in enumerate(env.red_ids)}
                blue_actions = opponent.act(obs, env.blue_ids, env=env)
                obs, _reward, terminated, truncated, info = env.step({**red_actions, **blue_actions})
                _gate_counts(info, gates)
                for aid, comp in (info.get("reward_components", {}) or {}).items():
                    if not isinstance(comp, dict):
                        continue
                    active = float(comp.get("alive_before", 0.0)) > 0.5
                    for field in V5_COMPONENT_FIELDS:
                        value = comp.get(field, 0.0)
                        if isinstance(value, (int, float, np.number)):
                            if field == "identity_error":
                                sums[field] = max(abs(sums[field]), abs(float(value)))
                            elif field in {"true_final_j", "red_alive_final", "blue_alive_final", "mav_alive_final",
                                         "unique_red_launch", "unique_red_hit", "unique_blue_launch", "unique_blue_hit"}:
                                sums[field] = float(value)
                            else:
                                sums[field] += float(value)
                                if active:
                                    alive_counts[field] += 1
                if _done(terminated) or _done(truncated):
                    terminal_observed = True
                    break
            length = step + 1
            red_alive = sum(bool(env.red_planes[rid].is_alive) for rid in env.red_ids)
            blue_alive = sum(bool(env.blue_planes[bid].is_alive) for bid in env.blue_ids)
            mav_id = next((rid for rid in env.red_ids if env.agent_roles.get(rid) == "mav"), "")
            mav_alive = int(bool(mav_id and env.red_planes[mav_id].is_alive))
            environment_timeout = int(terminal_observed and length >= 1000 and red_alive > 0 and blue_alive > 0)
            censored = int(not terminal_observed)
            if blue_alive == 0 and red_alive > 0:
                outcome = "red_win"
            elif red_alive == 0 and blue_alive > 0:
                outcome = "blue_win"
            else:
                outcome = "draw"
            row = {"source": source, "episode": episode, "seed": seed + episode,
                   "episode_length": length}
            row.update(sums)
            # Episode status is authoritative; per-agent component status is
            # retained in rich logs and must not overwrite this one-row value.
            row.update({"terminal_observed": int(terminal_observed),
                        "environment_timeout": environment_timeout, "censored": censored,
                        "outcome": outcome, "red_alive_final": red_alive,
                        "blue_alive_final": blue_alive, "mav_alive_final": mav_alive})
            for field, count in alive_counts.items():
                row[f"{field}_alive_record_count"] = count
            for key, value in gates.items():
                row[f"launch_gate_{key}"] = value
            rows.append(row)
        finally:
            env.close()
    return rows


def write_outputs(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "v5_episode_components.csv", index=False)
    raw_fields = [field for field in V5_COMPONENT_FIELDS if field.endswith("_raw")]
    dist_rows = []
    for source, group in frame.groupby("source"):
        for field in raw_fields:
            values = pd.to_numeric(group[field], errors="coerce").dropna()
            dist_rows.append({"source": source, "component": field, "count": len(values),
                              "mean": values.mean(), "std": values.std(ddof=0),
                              "min": values.min(), "p50": values.quantile(0.5), "max": values.max(),
                              "abs_mean": values.abs().mean(), "nonzero_rate": (values.abs() > 1e-12).mean()})
    pd.DataFrame(dist_rows).to_csv(output_dir / "v5_raw_component_distribution.csv", index=False)
    metrics = raw_fields + ["episode_length", "red_alive_final", "blue_alive_final", "mav_alive_final",
                            "unique_red_launch", "unique_red_hit", "unique_blue_launch", "unique_blue_hit"]
    frame.groupby(["source", "outcome"], dropna=False)[metrics].mean().reset_index().to_csv(
        output_dir / "v5_outcome_conditioned_distribution.csv", index=False)
    gate_columns = [column for column in frame.columns if column.startswith("launch_gate_")]
    frame.groupby("source")[gate_columns].sum().reset_index().to_csv(
        output_dir / "v5_launch_gate_distribution.csv", index=False)
    summary = {"episodes": int(len(frame)), "sources": sorted(frame["source"].unique().tolist()),
               "censored_count": int(frame["censored"].sum()),
               "terminal_observed_count": int(frame["terminal_observed"].sum()),
               "all_identity_valid": bool(pd.to_numeric(frame["identity_error"], errors="coerce").abs().max() <= 1e-8),
               "note": "Evaluation only; no policy updates were performed."}
    (output_dir / "v5_collection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", default=[], help="label=path")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=51000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.episodes < 20:
        raise ValueError("paper calibration requires at least 20 episodes per source")
    device = torch.device(args.device)
    rows = []
    for offset, source in enumerate(("random", "fixed_straight")):
        rows.extend(collect_source(config=args.config, source=source, episodes=args.episodes,
                                   seed=args.seed + offset * 1000, device=device))
    for index, item in enumerate(args.checkpoint):
        if "=" not in item:
            raise ValueError("--checkpoint must be label=path")
        label, path = item.split("=", 1)
        rows.extend(collect_source(config=args.config, source=label, episodes=args.episodes,
                                   seed=args.seed + (index + 2) * 1000, device=device, checkpoint=path))
    write_outputs(rows, Path(args.output_dir))
    print(Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()
