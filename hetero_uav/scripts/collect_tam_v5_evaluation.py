from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.mappo.opponent_policy import OpponentPolicy
from scripts.eval_policy_launch_diagnostics import _build_policy, _load_meta, _policy_actions
from uav_env import make_env
from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
from uav_env.JSBSim.envs.paper_formula_v5 import V5_COMPONENT_FIELDS


BASELINE_COMMIT = "53c4e70dcac31af7a5f764429a5620e71962025a"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paper_pursuit_actions(env) -> dict[str, np.ndarray]:
    actions = {}
    for rid in env.red_ids:
        if env.agent_roles.get(rid) == "mav":
            actions[rid] = np.array([0.0, 0.0, 0.25], dtype=np.float32)
            continue
        sim = env.red_planes.get(rid)
        candidates = []
        if sim is not None and sim.is_alive:
            spos = np.asarray(sim.get_position(), dtype=np.float64)
            for bid in env.blue_ids:
                blue = env.blue_planes.get(bid)
                if blue is None or not blue.is_alive or not env._mav_shared_track_state(rid, bid)["observed"]:
                    continue
                delta = np.asarray(blue.get_position(), dtype=np.float64) - spos
                candidates.append((float(np.linalg.norm(delta)), delta))
        if not candidates:
            actions[rid] = np.array([0.0, 0.0, 0.4], dtype=np.float32)
            continue
        _distance, delta = min(candidates, key=lambda item: item[0])
        heading = np.clip(np.arctan2(float(delta[1]), float(delta[0])) / np.pi, -1.0, 1.0)
        pitch = np.clip(float(delta[2]) / 5000.0, -0.25, 0.25)
        actions[rid] = np.array([pitch, heading, 0.8], dtype=np.float32)
    return actions


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
        for name, key in {
            "alive_target": "any_alive_target", "unengaged": "any_unengaged_target",
            "track": "any_track_pass", "range": "any_range_pass", "ata": "any_ata_pass",
            "ta": "any_ta_pass", "geometry": "any_geometry_pass",
            "lock_mature": "any_lock_mature", "actual_launch": "any_launch",
        }.items():
            counts[name] += float(bool(record.get(key, 0)))
        counts["cooldown_ready"] += float(float(record.get("cooldown_remaining_end", 0) or 0) <= 0)


def collect_source(*, config: str, source: str, episodes: int, seed: int,
                   device: torch.device, checkpoint: str | None = None,
                   diagnostic_policy: bool = False) -> list[dict]:
    policy = _load_checkpoint(checkpoint, device)
    adapter = HeteroObsAdapterV2()
    rows = []
    for episode in range(episodes):
        env = make_env(config, max_steps=1000, suppress_jsbsim_output=True)
        opponent = OpponentPolicy(mode="tam_greedy_rule", seed=seed + episode + 10000)
        obs, info = env.reset(seed=seed + episode)
        sums = {field: 0.0 for field in V5_COMPONENT_FIELDS}
        alive_counts = {field: 0 for field in V5_COMPONENT_FIELDS}
        gates = {key: 0.0 for key in (
            "records", "alive_target", "unengaged", "track", "range", "ata", "ta",
            "geometry", "lock_mature", "cooldown_ready", "actual_launch",
        )}
        hidden = None
        terminal_observed = False
        terminated = truncated = {}
        try:
            for step in range(1000):
                if diagnostic_policy:
                    red_actions = _paper_pursuit_actions(env)
                elif source == "random":
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
            row["unique_red_kill"] = float(
                row.get("shared_kill_raw", 0.0)
                + row.get("direct_kill_raw", 0.0)
                + row.get("direct_and_shared_kill_raw", 0.0)
            )
            row["unique_shared_only_kill"] = float(row.get("shared_kill_raw", 0.0))
            for field, count in alive_counts.items():
                row[f"{field}_alive_record_count"] = count
            for key, value in gates.items():
                row[f"launch_gate_{key}"] = value
            rows.append(row)
        finally:
            env.close()
    return rows


def _discover_event_checkpoints(outputs_root: Path, limit: int) -> list[tuple[str, Path]]:
    ranked = []
    for log in outputs_root.glob("*/train_log.csv"):
        try:
            frame = pd.read_csv(log)
        except Exception:
            continue
        if frame.empty:
            continue
        fired = pd.to_numeric(frame.get("red_missiles_fired", pd.Series(0, index=frame.index)), errors="coerce").fillna(0)
        hits = pd.to_numeric(frame.get("red_missile_hits", pd.Series(0, index=frame.index)), errors="coerce").fillna(0)
        score = float(hits.max() * 1000.0 + fired.max())
        if score <= 0:
            continue
        run = log.parent
        candidates = sorted((run / "checkpoints").glob("step_*/model.pt")) if (run / "checkpoints").exists() else []
        model = candidates[-1] if candidates else run / "latest/model.pt"
        if model.exists():
            ranked.append((score, run.name, model))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [(f"event_ckpt_{name}", path) for _score, name, path in ranked[:max(limit, 0)]]


def _checkpoint_summary(label: str, path: Path) -> dict:
    meta_path = path.parent / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return {
        "label": label, "path": str(path.resolve()), "sha256": _sha256(path),
        "meta_path": str(meta_path.resolve()) if meta_path.exists() else "",
        "meta": meta,
        "training_steps": meta.get("total_env_steps", meta.get("total_env_steps_actual", "")),
        "reward_mode": meta.get("actual_reward_mode", meta.get("reward_mode", "")),
    }


def write_outputs(rows: list[dict], output_dir: Path, *, config: Path,
                  source_manifest: list[dict], command: str, baseline_commit: str) -> None:
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
                            "unique_red_launch", "unique_red_hit", "unique_red_kill",
                            "unique_blue_launch", "unique_blue_hit"]
    frame.groupby(["source", "outcome"], dropna=False)[metrics].mean().reset_index().to_csv(
        output_dir / "v5_outcome_conditioned_distribution.csv", index=False)
    gate_columns = [column for column in frame.columns if column.startswith("launch_gate_")]
    gate_frame = frame.groupby("source")[gate_columns].sum().reset_index()
    event_funnel = frame.groupby("source")[["unique_red_launch", "unique_red_hit", "unique_red_kill"]].sum().reset_index()
    gate_frame.merge(event_funnel, on="source", how="left").to_csv(
        output_dir / "v5_launch_gate_distribution.csv", index=False)
    numeric = frame.select_dtypes(include=[np.number])
    finite = bool(np.isfinite(numeric.to_numpy()).all())
    summary = {"episodes": int(len(frame)), "sources": sorted(frame["source"].unique().tolist()),
               "censored_count": int(frame["censored"].sum()),
               "terminal_observed_count": int(frame["terminal_observed"].sum()),
               "finite_check_passed": finite,
               "all_identity_valid": bool(pd.to_numeric(frame["identity_error"], errors="coerce").abs().max() <= 1e-8),
               "identity_max_abs": float(pd.to_numeric(frame["identity_error"], errors="coerce").abs().max()),
               "unique_red_launch_total": float(frame["unique_red_launch"].sum()),
               "unique_red_hit_total": float(frame["unique_red_hit"].sum()),
               "unique_red_kill_total": float(frame["unique_red_kill"].sum()),
               "unique_shared_only_kill_total": float(frame["unique_shared_only_kill"].sum()),
               "note": "Evaluation only; no policy updates were performed."}
    (output_dir / "v5_collection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    config_copy = output_dir / config.name
    shutil.copy2(config, config_copy)
    manifest = {
        "baseline_commit_expected": baseline_commit,
        "command": command,
        "config": str(config.resolve()), "config_copy": config_copy.name,
        "config_sha256": _sha256(config), "episodes_per_source": int(frame.groupby("source").size().min()),
        "sources": source_manifest,
        "seed_min": int(frame.seed.min()), "seed_max": int(frame.seed.max()),
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda, "device": str(next(iter(source_manifest), {}).get("device", "")),
        "summary": summary,
    }
    (output_dir / "v5_collection_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "v5_collection_command.txt").write_text(command + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", default=[], help="label=path")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=51000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--diagnostic-config", default="")
    parser.add_argument("--auto-discover-checkpoints", type=int, default=0)
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--baseline-commit", default=BASELINE_COMMIT)
    parser.add_argument("--audit-doc-dir", default="")
    args = parser.parse_args()
    if args.episodes < 20:
        raise ValueError("paper calibration requires at least 20 episodes per source")
    device = torch.device(args.device)
    rows = []
    source_manifest = []
    for offset, source in enumerate(("random", "fixed_straight")):
        rows.extend(collect_source(config=args.config, source=source, episodes=args.episodes,
                                   seed=args.seed + offset * 1000, device=device))
        source_manifest.append({"label": source, "kind": "evaluation_policy", "device": str(device)})
    checkpoint_items = list(args.checkpoint)
    explicit_paths = {Path(item.split("=", 1)[1]).resolve() for item in checkpoint_items if "=" in item}
    for label, path in _discover_event_checkpoints(Path(args.outputs_root), args.auto_discover_checkpoints):
        if path.resolve() not in explicit_paths:
            checkpoint_items.append(f"{label}={path}")
    for index, item in enumerate(checkpoint_items):
        if "=" not in item:
            raise ValueError("--checkpoint must be label=path")
        label, path = item.split("=", 1)
        rows.extend(collect_source(config=args.config, source=label, episodes=args.episodes,
                                   seed=args.seed + (index + 2) * 1000, device=device, checkpoint=path))
        source_manifest.append({**_checkpoint_summary(label, Path(path)), "kind": "checkpoint", "device": str(device)})
    pursuit_seed = args.seed + (len(checkpoint_items) + 2) * 1000
    rows.extend(collect_source(config=args.config, source="paper_pursuit_diagnostic",
                               episodes=args.episodes, seed=pursuit_seed, device=device,
                               diagnostic_policy=True))
    source_manifest.append({"label": "paper_pursuit_diagnostic", "kind": "diagnostic_policy",
                            "claim": "diagnostic policy, not paper policy", "device": str(device)})
    if args.diagnostic_config:
        fixture_seed = pursuit_seed + 1000
        rows.extend(collect_source(config=args.diagnostic_config, source="mav_shared_diagnostic_fixture",
                                   episodes=args.episodes, seed=fixture_seed, device=device,
                                   diagnostic_policy=True))
        source_manifest.append({"label": "mav_shared_diagnostic_fixture", "kind": "diagnostic_fixture",
                                "config": str(Path(args.diagnostic_config).resolve()),
                                "config_sha256": _sha256(Path(args.diagnostic_config)),
                                "claim": "diagnostic-only observability fixture, not training configuration",
                                "device": str(device)})
    command = " ".join(sys.argv)
    output_dir = Path(args.output_dir)
    write_outputs(rows, output_dir, config=Path(args.config), source_manifest=source_manifest,
                  command=command, baseline_commit=args.baseline_commit)
    if args.audit_doc_dir:
        audit_dir = Path(args.audit_doc_dir); audit_dir.mkdir(parents=True, exist_ok=True)
        for name in ("v5_collection_manifest.json", "v5_collection_summary.json",
                     "v5_launch_gate_distribution.csv", "v5_collection_command.txt"):
            shutil.copy2(output_dir / name, audit_dir / name)
    print(Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()
