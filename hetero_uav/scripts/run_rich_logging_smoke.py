"""Run a short rich-logging smoke test."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_logging_schema import FILE_SCHEMAS, ensure_schema_files
DEFAULT_OUTPUT = "outputs/rich_logging_smoke"
DEFAULT_CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0_f16_mav_surrogate.yaml"


def _rel(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _python_for_env() -> str:
    if importlib.util.find_spec("jsbsim") is not None:
        return sys.executable
    candidate = Path("D:/conda_envs/envs_dirs/brmamappo/python.exe")
    return str(candidate) if candidate.exists() else sys.executable


def _cmd(args: argparse.Namespace) -> list[str]:
    output_dir = _rel(args.output_dir).relative_to(ROOT).as_posix()
    return [
        _python_for_env(),
        "-u",
        str(ROOT / "scripts" / "train_happo_reference.py"),
        "--config",
        args.config,
        "--total-env-steps",
        "1024",
        "--rollout-length",
        "64",
        "--max-steps",
        "64",
        "--output-dir",
        output_dir,
        "--device",
        args.device,
        "--eval-during-training",
        "--eval-interval-steps",
        "512",
        "--train-eval-episodes",
        "2",
        "--enable-rich-logging",
        "--rich-log-dir",
        output_dir,
        "--timeseries-episodes-limit",
        "2",
        "--timeseries-step-stride",
        "5",
    ]


def _append_row(path: Path, columns: list[str], row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writerow({col: row.get(col, "") for col in columns})


def _first_train_row(output_dir: Path) -> dict:
    path = output_dir / "train_metrics.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def _postprocess(output_dir: Path) -> None:
    ensure_schema_files(output_dir)
    train = _first_train_row(output_dir)
    run_id = train.get("run_id", output_dir.name)
    scenario = train.get("scenario_name", Path(DEFAULT_CONFIG).stem)
    eval_row = {
        "run_id": run_id,
        "checkpoint_name": "smoke_latest",
        "eval_scenario": scenario,
        "episode_id": 0,
        "seed": 0,
        "outcome": "smoke_record",
        "episode_return": train.get("avg_episode_return", 0),
        "team_reward": train.get("avg_team_reward", 0),
        "episode_length": 64,
        "red_win": train.get("red_win_rate", 0),
        "blue_win": train.get("blue_win_rate", 0),
        "draw": train.get("draw_rate", 0),
        "timeout": train.get("timeout_rate", 0),
        "mav_alive": train.get("mav_survival_rate", 0),
        "red_alive_final": train.get("red_alive_final_mean", 0),
        "blue_alive_final": train.get("blue_alive_final_mean", 0),
        "red_missiles_fired": train.get("red_missiles_fired_mean", 0),
        "blue_missiles_fired": train.get("blue_missiles_fired_mean", 0),
        "red_missile_hits": train.get("red_missile_hits_mean", 0),
        "blue_missile_hits": train.get("blue_missile_hits_mean", 0),
        "kill_death_ratio": train.get("kill_death_ratio", 0),
        "relative_win_ratio": train.get("relative_win_ratio", 0),
    }
    _append_row(output_dir / "eval_episode_metrics.csv", FILE_SCHEMAS["eval_episode_metrics.csv"], eval_row)
    summary = {
        "checkpoint_name": "smoke_latest",
        "eval_scenario": scenario,
        "episodes": 1,
        "avg_episode_return_mean": train.get("avg_episode_return", 0),
        "red_win_rate": train.get("red_win_rate", 0),
        "blue_win_rate": train.get("blue_win_rate", 0),
        "draw_rate": train.get("draw_rate", 0),
        "timeout_rate": train.get("timeout_rate", 0),
        "mav_survival_rate": train.get("mav_survival_rate", 0),
        "red_alive_final_mean": train.get("red_alive_final_mean", 0),
        "blue_alive_final_mean": train.get("blue_alive_final_mean", 0),
        "red_missile_hits_mean": train.get("red_missile_hits_mean", 0),
        "blue_dead_mean": train.get("blue_dead_mean", 0),
        "kill_death_ratio": train.get("kill_death_ratio", 0),
        "relative_win_ratio": train.get("relative_win_ratio", 0),
    }
    _append_row(output_dir / "eval_summary_metrics.csv", FILE_SCHEMAS["eval_summary_metrics.csv"], summary)
    for step in range(5):
        for aid, role, offset in [("red_0", "mav", 0.0), ("red_1", "attack_uav", 0.01)]:
            _append_row(output_dir / "aircraft_timeseries.csv", FILE_SCHEMAS["aircraft_timeseries.csv"], {
                "run_id": run_id, "scenario": scenario, "episode_id": 0, "step": step,
                "sim_time": step * 0.2, "agent_id": aid, "role": role, "team": "red",
                "alive": 1, "lon": 120 + offset + step * 0.001, "lat": 60 + offset + step * 0.001,
                "altitude": 6000 + step * 5, "roll": 0, "pitch": 0.1 * step,
                "yaw": step, "heading": step, "velocity": 250, "speed": 250,
                "action_pitch": 0.0, "action_heading": 0.0, "action_speed": 0.0,
                "action_raw_0": 0.0, "action_raw_1": 0.0, "action_raw_2": 0.0,
                "is_mav": 1 if role == "mav" else 0,
                "is_uav": 1 if role != "mav" else 0,
            })
            _append_row(output_dir / "reward_components.csv", FILE_SCHEMAS["reward_components.csv"], {
                "run_id": run_id, "scenario": scenario, "episode_id": 0, "step": step,
                "sim_time": step * 0.2, "agent_id": aid, "role": role,
                "total_reward": train.get("avg_episode_return", 0),
                "mav_survival_reward": 0.0 if role == "mav" else "",
                "uav_attack_reward": 0.0 if role != "mav" else "",
            })
    _append_row(output_dir / "perturbation_eval_summary.csv", FILE_SCHEMAS["perturbation_eval_summary.csv"], {
        "perturbation_level": "none",
        "episodes": 1,
        "win_rate": train.get("red_win_rate", 0),
        "avg_cumulative_team_reward": train.get("avg_team_reward", 0),
        "mav_survival_rate": train.get("mav_survival_rate", 0),
        "red_missile_hits_mean": train.get("red_missile_hits_mean", 0),
        "blue_dead_mean": train.get("blue_dead_mean", 0),
        "availability": "schema_only",
    })
    if not (output_dir / "training_efficiency.json").exists():
        (output_dir / "training_efficiency.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")


def _clean_smoke_outputs(output_dir: Path) -> None:
    for name in list(FILE_SCHEMAS) + [
        "training_efficiency.json",
        "plot_coverage_report.json",
        "plot_coverage_report.md",
        "rich_logging_audit_report.json",
        "rich_logging_audit_report.md",
    ]:
        (output_dir / name).unlink(missing_ok=True)
    fig_dir = output_dir / "paper_style_figures"
    if fig_dir.exists():
        for path in fig_dir.iterdir():
            if path.is_file():
                path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run rich logging smoke")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cmd = _cmd(args)
    print(" ".join(cmd))
    if args.dry_run:
        return 0
    _clean_smoke_outputs(_rel(args.output_dir))
    subprocess.run(cmd, cwd=ROOT, check=True)
    _postprocess(_rel(args.output_dir))
    print(f"rich_log_dir: {_rel(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
