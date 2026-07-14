from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = (
    "uav_env/JSBSim/configs/"
    "hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_happo_paper_formula_v5.yaml"
)
METRICS = [
    "avg_episode_return", "red_win_rate", "blue_win_rate", "timeout_rate",
    "mav_survival_rate", "red_alive_final_mean", "blue_alive_final_mean",
    "red_launch_total", "red_hit_total", "red_kill_total",
    "red_launch_per_episode", "red_hit_per_episode", "red_kill_per_episode",
    "track_rate", "range_rate", "ata_rate", "ta_rate", "geometry_rate",
    "lock_mature_rate", "actual_launch_rate", "reward_target_matches_lock_rate",
    "reward_target_matches_launch_rate", "finite", "nan_detected",
]


def _all_done(flags: dict) -> bool:
    return bool(flags) and all(bool(value) for value in flags.values())


def _state_dict_equal(left: Path, right: Path) -> bool:
    import torch

    a = torch.load(left, map_location="cpu", weights_only=True)
    b = torch.load(right, map_location="cpu", weights_only=True)
    return a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"command failed with exit code {code}: {' '.join(command)}")


def _training_command(arguments: list[str], conda_env: str, batch_path: Path) -> list[str]:
    launcher = [
        "-u", "-c",
        (
            "import runpy,sys; import uav_env; "
            "sys.argv=['scripts/train_happo_reference.py',*sys.argv[1:]]; "
            "runpy.run_path('scripts/train_happo_reference.py',run_name='__main__')"
        ),
        *arguments[2:],
    ]
    if conda_env == "current" or os.name != "nt":
        return [sys.executable, *launcher]
    activate = Path.home() / "anaconda3" / "Scripts" / "activate.bat"
    if not activate.exists():
        raise FileNotFoundError(f"conda activation script not found: {activate}")
    child = subprocess.list2cmdline(["python", *launcher])
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    batch_path.write_text(
        f'@call "{activate}" {conda_env}\n@{child}\n', encoding="utf-8"
    )
    return ["cmd.exe", "/d", "/c", str(batch_path)]


def _event_key(record: dict, episode_seed: int, step: int) -> tuple:
    missile_id = str(record.get("missile_id", "") or "")
    shooter = str(record.get("shooter_id", "") or "")
    if missile_id:
        return episode_seed, shooter, missile_id
    return episode_seed, step, shooter, str(record.get("target_id", "") or "")


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    config: str,
    episode_seeds: list[int],
    device_name: str,
    max_steps: int,
) -> dict[str, float]:
    # Load JSBSim before PyTorch on Windows. Loading the runtimes in the
    # opposite order can initialize incompatible OpenMP libraries.
    from uav_env import make_env
    import torch
    from algorithms.mappo.opponent_policy import OpponentPolicy
    from scripts.eval_policy_launch_diagnostics import _build_policy, _load_meta, _policy_actions
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2

    device = torch.device(
        "cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu"
    )
    meta = _load_meta(checkpoint)
    if meta.get("policy_arch") != "pure_happo":
        raise ValueError(f"expected pure_happo checkpoint, got {meta.get('policy_arch')}")
    policy = _build_policy(meta, device)
    policy.load(checkpoint, map_location=device)
    policy.eval()
    adapter = HeteroObsAdapterV2()

    returns: list[float] = []
    red_alive_final: list[float] = []
    blue_alive_final: list[float] = []
    mav_alive_final: list[float] = []
    red_wins = blue_wins = timeouts = 0
    launch_keys: set[tuple] = set()
    hit_keys: set[tuple] = set()
    red_kills = 0.0
    gate_den = 0.0
    gates = {name: 0.0 for name in (
        "track", "range", "ata", "ta", "geometry", "lock_mature", "actual_launch"
    )}
    reward_match_den = 0.0
    reward_lock_matches = reward_launch_matches = 0.0
    finite = True

    for episode_seed in episode_seeds:
        env = make_env(config, max_steps=max_steps, suppress_jsbsim_output=True)
        opponent = OpponentPolicy(mode="tam_greedy_rule", seed=episode_seed + 100_000)
        obs, info = env.reset(seed=episode_seed)
        hidden = None
        episode_return = 0.0
        terminated = truncated = {}
        try:
            for step in range(max_steps):
                actions, hidden = _policy_actions(policy, adapter, env, obs, info, device, hidden)
                finite = finite and bool(np.isfinite(actions).all())
                red_actions = {rid: actions[index] for index, rid in enumerate(env.red_ids)}
                blue_actions = opponent.act(obs, env.blue_ids, env=env)
                obs, rewards, terminated, truncated, info = env.step({**red_actions, **blue_actions})
                reward_values = np.asarray([rewards[rid] for rid in env.red_ids], dtype=np.float64)
                finite = finite and bool(np.isfinite(reward_values).all())
                episode_return += float(reward_values.mean())

                for record in info.get("__launch_gate_diagnostics__", []) or []:
                    aid = str(record.get("agent_id", "") or "")
                    if env.agent_roles.get(aid) != "attack_uav":
                        continue
                    gate_den += 1.0
                    gates["track"] += float(bool(record.get("any_track_pass", 0)))
                    gates["range"] += float(bool(record.get("any_range_pass", 0)))
                    gates["ata"] += float(bool(record.get("any_ata_pass", 0)))
                    gates["ta"] += float(bool(record.get("any_ta_pass", 0)))
                    gates["geometry"] += float(bool(record.get("any_geometry_pass", 0)))
                    gates["lock_mature"] += float(bool(record.get("any_lock_mature", 0)))
                    gates["actual_launch"] += float(bool(record.get("any_launch", 0)))

                for record in info.get("__launch_quality_step__", []) or []:
                    if str(record.get("shooter_id", "")).startswith("red_"):
                        launch_keys.add(_event_key(record, episode_seed, step))
                for record in info.get("__launch_quality_done__", []) or []:
                    reason = str(record.get("raw_termination_reason", record.get("termination_reason", ""))).lower()
                    if str(record.get("shooter_id", "")).startswith("red_") and reason == "hit":
                        hit_keys.add(_event_key(record, episode_seed, step))

                components = info.get("reward_components", {}) or {}
                mav_id = next((rid for rid in env.red_ids if env.agent_roles.get(rid) == "mav"), "")
                mav_comp = components.get(mav_id, {}) if mav_id else {}
                red_kills += sum(float(mav_comp.get(key, 0.0) or 0.0) for key in (
                    "shared_kill_raw", "direct_kill_raw", "direct_and_shared_kill_raw"
                ))
                for rid in env.red_ids:
                    if env.agent_roles.get(rid) != "attack_uav":
                        continue
                    comp = components.get(rid, {}) or {}
                    if float(comp.get("alive_before", 0.0) or 0.0) <= 0.5:
                        continue
                    reward_match_den += 1.0
                    reward_lock_matches += float(comp.get("reward_target_matches_lock", 0.0) or 0.0)
                    reward_launch_matches += float(comp.get("reward_target_matches_launch", 0.0) or 0.0)

                if _all_done(terminated) or _all_done(truncated):
                    break

            red_alive = sum(bool(env.red_planes[rid].is_alive) for rid in env.red_ids)
            blue_alive = sum(bool(env.blue_planes[bid].is_alive) for bid in env.blue_ids)
            mav_id = next((rid for rid in env.red_ids if env.agent_roles.get(rid) == "mav"), "")
            mav_alive = float(bool(mav_id and env.red_planes[mav_id].is_alive))
            red_wins += int(blue_alive == 0 and red_alive > 0)
            blue_wins += int(red_alive == 0 and blue_alive > 0)
            timeouts += int(step + 1 >= max_steps and red_alive > 0 and blue_alive > 0)
            returns.append(episode_return)
            red_alive_final.append(float(red_alive))
            blue_alive_final.append(float(blue_alive))
            mav_alive_final.append(mav_alive)
        finally:
            env.close()

    episodes = float(len(episode_seeds))
    result = {
        "avg_episode_return": float(np.mean(returns)),
        "red_win_rate": red_wins / episodes,
        "blue_win_rate": blue_wins / episodes,
        "timeout_rate": timeouts / episodes,
        "mav_survival_rate": float(np.mean(mav_alive_final)),
        "red_alive_final_mean": float(np.mean(red_alive_final)),
        "blue_alive_final_mean": float(np.mean(blue_alive_final)),
        "red_launch_total": float(len(launch_keys)),
        "red_hit_total": float(len(hit_keys)),
        "red_kill_total": float(red_kills),
        "red_launch_per_episode": len(launch_keys) / episodes,
        "red_hit_per_episode": len(hit_keys) / episodes,
        "red_kill_per_episode": red_kills / episodes,
        "track_rate": gates["track"] / max(gate_den, 1.0),
        "range_rate": gates["range"] / max(gate_den, 1.0),
        "ata_rate": gates["ata"] / max(gate_den, 1.0),
        "ta_rate": gates["ta"] / max(gate_den, 1.0),
        "geometry_rate": gates["geometry"] / max(gate_den, 1.0),
        "lock_mature_rate": gates["lock_mature"] / max(gate_den, 1.0),
        "actual_launch_rate": gates["actual_launch"] / max(gate_den, 1.0),
        "reward_target_matches_lock_rate": reward_lock_matches / max(reward_match_den, 1.0),
        "reward_target_matches_launch_rate": reward_launch_matches / max(reward_match_den, 1.0),
    }
    numeric_finite = bool(np.isfinite(np.asarray(list(result.values()), dtype=np.float64)).all())
    result["finite"] = float(finite and numeric_finite)
    result["nan_detected"] = float(not (finite and numeric_finite))
    return result


def classify(rows: list[dict], completed: list[bool]) -> tuple[str, dict]:
    behavior_improved = []
    outcome_improved = []
    for row in rows:
        behavior_improved.append(
            row["delta_red_launch_per_episode"] >= 0.10
            or row["delta_geometry_rate"] >= 0.01
        )
        outcome_improved.append(
            row["delta_red_hit_total"] > 0
            or row["delta_red_kill_total"] > 0
            or row["delta_blue_alive_final_mean"] < -1e-9
        )
    healthy = all(completed) and all(
        row["final_finite"] > 0.5 and row["final_nan_detected"] < 0.5 for row in rows
    )
    behavior_count = sum(behavior_improved)
    outcome_count = sum(outcome_improved)
    if healthy and behavior_count >= 2 and outcome_count >= 2:
        status = "PURE_HAPPO_LEARNABILITY_PASS"
    elif healthy and (behavior_count > 0 or outcome_count > 0):
        status = "PURE_HAPPO_LEARNABILITY_WEAK_SIGNAL"
    else:
        status = "PURE_HAPPO_LEARNABILITY_FAIL"
    return status, {
        "healthy_seed_count": sum(completed),
        "behavior_improved_seed_count": behavior_count,
        "outcome_improved_seed_count": outcome_count,
        "behavior_thresholds": {
            "red_launch_per_episode_delta": 0.10,
            "geometry_rate_absolute_delta": 0.01,
        },
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_outputs(output_dir: Path, rows: list[dict], completed: list[bool], args) -> str:
    _write_csv(output_dir / "per_seed_init_final.csv", rows)
    aggregate = []
    for metric in METRICS:
        initial = np.asarray([row[f"init_{metric}"] for row in rows], dtype=np.float64)
        final = np.asarray([row[f"final_{metric}"] for row in rows], dtype=np.float64)
        delta = final - initial
        aggregate.append({
            "metric": metric,
            "init_mean": float(initial.mean()), "init_std": float(initial.std(ddof=0)),
            "final_mean": float(final.mean()), "final_std": float(final.std(ddof=0)),
            "delta_mean": float(delta.mean()), "delta_std": float(delta.std(ddof=0)),
        })
    _write_csv(output_dir / "aggregate_init_final.csv", aggregate)
    status, evidence = classify(rows, completed)
    summary = {
        "status": status,
        "config": args.config,
        "reward_mode": "tam_happo_paper_formula_v5",
        "policy_arch": "pure_happo",
        "opponent_policy": "tam_greedy_rule",
        "seeds": args.seeds,
        "eval_episode_seeds": list(range(args.eval_seed_base, args.eval_seed_base + args.eval_episodes)),
        "train_steps_per_seed": args.train_steps,
        "eval_episodes_per_checkpoint": args.eval_episodes,
        "evidence": evidence,
        "per_seed": rows,
    }
    (output_dir / "learnability_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# TAM v5 Pure HAPPO Fast Learnability",
        "",
        f"**Status:** `{status}`",
        "",
        "The same deterministic evaluation seeds are used for init and final checkpoints. ",
        "This is a 3V2 environment-learnability check, not a final performance experiment.",
        "",
        "| seed | init return | final return | init/final launch per ep | init/final hit | init/final kill | init/final geometry | init/final blue alive | init/final MAV survival |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['init_avg_episode_return']:.4f} | {row['final_avg_episode_return']:.4f} | "
            f"{row['init_red_launch_per_episode']:.3f}/{row['final_red_launch_per_episode']:.3f} | "
            f"{row['init_red_hit_total']:.0f}/{row['final_red_hit_total']:.0f} | "
            f"{row['init_red_kill_total']:.0f}/{row['final_red_kill_total']:.0f} | "
            f"{row['init_geometry_rate']:.4f}/{row['final_geometry_rate']:.4f} | "
            f"{row['init_blue_alive_final_mean']:.3f}/{row['final_blue_alive_final_mean']:.3f} | "
            f"{row['init_mav_survival_rate']:.3f}/{row['final_mav_survival_rate']:.3f} |"
        )
    lines.extend([
        "",
        f"Behavior-improved seeds: {evidence['behavior_improved_seed_count']}/3.",
        f"Outcome-improved seeds: {evidence['outcome_improved_seed_count']}/3.",
        "A behavior improvement is launch/episode +0.10 or geometry rate +0.01 absolute.",
    ])
    (output_dir / "learnability_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default="outputs/tam_v5_pure_happo_fast_learnability_51k")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--train-steps", type=int, default=51_200)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed-base", type=int, default=91_000)
    parser.add_argument("--rollout-length", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--training-conda-env", default="brmamappo")
    args = parser.parse_args()
    if args.eval_episodes < 1 or args.train_steps < 1:
        raise ValueError("eval episodes and train steps must be positive")
    output_dir = ROOT / args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_seeds = list(range(args.eval_seed_base, args.eval_seed_base + args.eval_episodes))
    rows: list[dict] = []
    completed: list[bool] = []

    for seed in args.seeds:
        seed_dir = output_dir / f"seed_{seed}"
        init_dir = seed_dir / "init"
        train_start_dir = seed_dir / "train_start"
        train_dir = seed_dir / "train"
        final_dir = seed_dir / "final"
        common = [
            "-u", "scripts/train_happo_reference.py",
            "--config", args.config,
            "--reward-mode", "tam_happo_paper_formula_v5",
            "--rollout-length", str(args.rollout_length),
            "--num-envs", "1", "--max-steps", str(args.max_steps),
            "--device", args.device, "--policy-arch", "pure_happo",
            "--opponent-policy", "tam_greedy_rule", "--seed", str(seed),
        ]
        bootstrap_dir = seed_dir / "bootstrap"
        _run(_training_command(common + [
            "--output-dir", str(bootstrap_dir.relative_to(ROOT)),
            "--total-env-steps", "1",
            "--save-initial-checkpoint-dir", str(init_dir.relative_to(ROOT)),
            "--initial-checkpoint-only",
        ], args.training_conda_env, seed_dir / "init_build.bat"), seed_dir / "init_build.log")
        init_metrics = evaluate_checkpoint(
            init_dir / "model.pt", config=args.config, episode_seeds=eval_seeds,
            device_name=args.device, max_steps=args.max_steps,
        )
        (seed_dir / "init_eval.json").write_text(json.dumps(init_metrics, indent=2), encoding="utf-8")

        _run(_training_command(common + [
            "--output-dir", str(train_dir.relative_to(ROOT)),
            "--total-env-steps", str(args.train_steps),
            "--checkpoint-interval-steps", str(max(args.train_steps // 2, 1)),
            "--keep-checkpoints", "2",
            "--save-initial-checkpoint-dir", str(train_start_dir.relative_to(ROOT)),
        ], args.training_conda_env, seed_dir / "train.bat"), seed_dir / "training_stdout.log")
        if not _state_dict_equal(init_dir / "model.pt", train_start_dir / "model.pt"):
            raise RuntimeError(f"seed {seed}: evaluated init checkpoint differs from actual training start")
        final_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(train_dir / "latest/model.pt", final_dir / "model.pt")
        shutil.copy2(train_dir / "latest/meta.json", final_dir / "meta.json")
        final_metrics = evaluate_checkpoint(
            final_dir / "model.pt", config=args.config, episode_seeds=eval_seeds,
            device_name=args.device, max_steps=args.max_steps,
        )
        (seed_dir / "final_eval.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
        runner_status = json.loads((train_dir / "runner_status.json").read_text(encoding="utf-8"))
        complete = (
            runner_status.get("status") == "normal"
            and int(runner_status.get("total_env_steps_actual", -1)) == args.train_steps
            and not bool(runner_status.get("nan_detected", False))
        )
        completed.append(complete)
        row = {
            "seed": seed,
            "training_complete": int(complete),
            "init_checkpoint_sha256": _sha256(init_dir / "model.pt"),
            "train_start_matches_init": 1,
            **{f"init_{key}": init_metrics[key] for key in METRICS},
            **{f"final_{key}": final_metrics[key] for key in METRICS},
            **{f"delta_{key}": final_metrics[key] - init_metrics[key] for key in METRICS},
        }
        rows.append(row)
        _write_outputs(output_dir, rows, completed, args)

    status = _write_outputs(output_dir, rows, completed, args)
    print(status, flush=True)


if __name__ == "__main__":
    main()
