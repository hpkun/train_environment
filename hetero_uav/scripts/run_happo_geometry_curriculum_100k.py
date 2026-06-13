"""Run easy-to-medium-to-normal HAPPO geometry curriculum.

This runner does not change reward, missile, action, observation, PID, or model
logic. It only chains existing HAPPO training/evaluation scripts across two
initial-geometry configs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

MEDIUM_CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_medium_combat_f16_mav_surrogate.yaml"
NORMAL_CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0_f16_mav_surrogate.yaml"
EASY_CHECKPOINT = "outputs/happo_easy_combat_oracle_anchor_50k/latest/model.pt"
ORACLE_INIT_FALLBACK = "outputs/oracle_pretrain/uav_actor_oracle_pretrained_wrapped_normal/model.pt"
ORACLE_DATASET = "outputs/direct_chase_oracle_dataset/direct_chase_oracle_3v2.npz"
OUTPUT_DIR = "outputs/happo_geometry_curriculum_100k"


def _run(cmd: list[str], *, cwd: Path, dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=cwd, check=True)


def _checkpoint_or_fallback() -> str:
    easy = ROOT / EASY_CHECKPOINT
    if easy.exists():
        return EASY_CHECKPOINT
    return ORACLE_INIT_FALLBACK


def _train_cmd(
    *,
    config: str,
    output_dir: Path,
    init_checkpoint: str,
    imitation_coef: float,
    total_env_steps: int,
    device: str,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        "scripts/train_happo_reference.py",
        "--config",
        config,
        "--reward-mode",
        "happo_ref_v0",
        "--opponent-policy",
        "brma_rule",
        "--total-env-steps",
        str(total_env_steps),
        "--rollout-length",
        "256",
        "--max-steps",
        "1000",
        "--ppo-epochs",
        "10",
        "--entropy-coef",
        "0.02",
        "--actor-lr",
        "2e-4",
        "--critic-lr",
        "5e-4",
        "--eval-during-training",
        "--eval-interval-steps",
        "25000",
        "--train-eval-episodes",
        "2",
        "--eval-configs",
        config,
        "--init-checkpoint",
        init_checkpoint,
        "--uav-imitation-dataset",
        ORACLE_DATASET,
        "--uav-imitation-coef",
        str(imitation_coef),
        "--uav-imitation-until-steps",
        "25000",
        "--uav-imitation-batch-size",
        "1024",
        "--output-dir",
        str(output_dir),
        "--device",
        device,
    ]


def _eval_cmd(output_dir: Path, *, config: str, fast: bool, episodes: int, device: str) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/evaluate_happo_3v2_reference_checkpoints.py",
        "--output-dir",
        str(output_dir),
        "--checkpoint-mode",
        "all",
        "--configs",
        config,
        "--device",
        device,
    ]
    if fast:
        cmd.append("--fast")
    else:
        cmd.extend(["--episodes", str(episodes)])
    return cmd


def _acmi_cmd(output_dir: Path, *, config: str, checkpoint: str, device: str) -> list[str]:
    return [
        sys.executable,
        "scripts/export_happo_reference_acmi.py",
        "--experiment-dir",
        str(output_dir),
        "--checkpoint",
        checkpoint,
        "--config",
        config,
        "--output",
        str(output_dir / "acmi" / f"{checkpoint}_normal_3v2_episode0.acmi"),
        "--summary-json",
        str(output_dir / "acmi" / f"{checkpoint}_normal_3v2_episode0.json"),
        "--device",
        device,
    ]


def _load_records(eval_json: Path) -> list[dict[str, Any]]:
    if not eval_json.exists():
        return []
    data = json.loads(eval_json.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [record for record in data if isinstance(record, dict)]
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        return [record for record in data["records"] if isinstance(record, dict)]
    return []


def _num(record: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = record.get(key, default)
    if value is None:
        return default
    return float(value)


def _is_attack_signal(record: dict[str, Any]) -> bool:
    fired = _num(record, "red_missiles_fired_mean")
    hits = _num(record, "red_missile_hits_mean")
    blue_dead = _num(record, "blue_dead_mean")
    mav_survival = _num(record, "mav_survival_rate")
    blue_win = _num(record, "blue_win_rate", default=1.0)
    return fired > 0.3 and (hits > 0.1 or blue_dead > 0.1) and mav_survival >= 0.3 and blue_win < 0.9


def _best_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None

    def score(record: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            _num(record, "red_missile_hits_mean") + _num(record, "blue_dead_mean"),
            _num(record, "red_missiles_fired_mean"),
            _num(record, "mav_survival_rate"),
            _num(record, "red_win_rate"),
        )

    return max(records, key=score)


def _write_decision(output_dir: Path, *, source: str, normal_records: list[dict[str, Any]]) -> dict[str, Any]:
    best = _best_record(normal_records)
    success = any(_is_attack_signal(record) for record in normal_records)
    decision = {
        "source": source,
        "geometry_curriculum_success": success,
        "recommend_normal_geometry_200k": success,
        "criteria": {
            "red_missiles_fired_mean": "> 0.3",
            "red_missile_hits_or_blue_dead_mean": "> 0.1",
            "mav_survival_rate": ">= 0.3",
            "blue_win_rate": "< 0.9",
        },
        "best_normal_record": best,
        "next_step": (
            "continue normal geometry 200k and then 5v4 zero-shot evaluation"
            if success
            else "add one intermediate geometry before normal; do not extend normal 200k yet"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "final_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# Geometry Curriculum Final Decision",
        "",
        f"- source: {source}",
        f"- geometry_curriculum_success: {success}",
        f"- recommend_normal_geometry_200k: {success}",
        f"- next_step: {decision['next_step']}",
    ]
    if best:
        lines.extend(
            [
                "",
                "## Best Normal Record",
                "",
                f"- checkpoint: {best.get('checkpoint')}",
                f"- red_missiles_fired_mean: {best.get('red_missiles_fired_mean')}",
                f"- red_missile_hits_mean: {best.get('red_missile_hits_mean')}",
                f"- blue_dead_mean: {best.get('blue_dead_mean')}",
                f"- mav_survival_rate: {best.get('mav_survival_rate')}",
                f"- blue_win_rate: {best.get('blue_win_rate')}",
            ]
        )
    (output_dir / "final_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 100k HAPPO geometry curriculum")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--total-env-steps-per-stage", type=int, default=50000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    medium_dir = output_dir / "medium_50k"
    normal_dir = output_dir / "normal_50k"

    medium_train = _train_cmd(
        config=MEDIUM_CONFIG,
        output_dir=medium_dir,
        init_checkpoint=_checkpoint_or_fallback(),
        imitation_coef=0.1,
        total_env_steps=args.total_env_steps_per_stage,
        device=args.device,
    )
    normal_train = _train_cmd(
        config=NORMAL_CONFIG,
        output_dir=normal_dir,
        init_checkpoint=str(medium_dir / "latest" / "model.pt"),
        imitation_coef=0.05,
        total_env_steps=args.total_env_steps_per_stage,
        device=args.device,
    )

    print("[geometry] medium train", flush=True)
    _run(medium_train, cwd=ROOT, dry_run=args.dry_run)
    print("[geometry] medium fast eval", flush=True)
    _run(_eval_cmd(medium_dir, config=MEDIUM_CONFIG, fast=True, episodes=20, device=args.device), cwd=ROOT, dry_run=args.dry_run)

    print("[geometry] normal train", flush=True)
    _run(normal_train, cwd=ROOT, dry_run=args.dry_run)
    print("[geometry] normal fast eval", flush=True)
    _run(_eval_cmd(normal_dir, config=NORMAL_CONFIG, fast=True, episodes=20, device=args.device), cwd=ROOT, dry_run=args.dry_run)

    if args.dry_run:
        print("[geometry] dry run only; no final decision written", flush=True)
        return 0

    fast_json = normal_dir / "checkpoint_eval" / "happo_3v2_checkpoint_eval.json"
    fast_records = _load_records(fast_json)
    fast_success = any(_is_attack_signal(record) for record in fast_records)
    records_for_decision = fast_records
    source = "normal_fast_eval"

    if fast_success:
        print("[geometry] normal fast eval has attack signal; running 50-episode eval", flush=True)
        _run(_eval_cmd(normal_dir, config=NORMAL_CONFIG, fast=False, episodes=50, device=args.device), cwd=ROOT, dry_run=False)
        full_json = normal_dir / "checkpoint_eval" / "happo_3v2_checkpoint_eval.json"
        records_for_decision = _load_records(full_json)
        source = "normal_50_episode_eval"
        best = _best_record(records_for_decision)
        checkpoint = str(best.get("checkpoint", "best")) if best else "best"
        if checkpoint not in {"best", "latest"}:
            checkpoint = "best"
        print("[geometry] exporting ACMI for normal 3v2", flush=True)
        _run(_acmi_cmd(normal_dir, config=NORMAL_CONFIG, checkpoint=checkpoint, device=args.device), cwd=ROOT, dry_run=False)
    else:
        print("[geometry] normal fast eval has no sufficient attack signal; skipping 50-episode eval and ACMI", flush=True)

    decision = _write_decision(output_dir, source=source, normal_records=records_for_decision)
    print(f"[geometry] final decision: {output_dir / 'final_decision.json'}", flush=True)
    print(f"[geometry] geometry_curriculum_success={decision['geometry_curriculum_success']}", flush=True)
    print(f"[geometry] recommend_normal_geometry_200k={decision['recommend_normal_geometry_200k']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
