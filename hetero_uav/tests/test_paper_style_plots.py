from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_paper_style_plot_and_coverage_help() -> None:
    for script in [
        "scripts/generate_paper_style_plots.py",
        "scripts/check_paper_plot_coverage.py",
        "scripts/audit_rich_logging_outputs.py",
    ]:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        assert "paper" in result.stdout.lower()


def test_paper_style_plots_from_fake_minimal_logs(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    out_dir = tmp_path / "figures"
    log_dir.mkdir()
    with (log_dir / "train_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "train_steps", "avg_episode_return", "red_win_rate", "blue_win_rate",
            "kill_death_ratio", "critic_loss", "entropy", "policy_gradient_norm",
        ])
        writer.writeheader()
        writer.writerow({
            "train_steps": 1,
            "avg_episode_return": 1.0,
            "red_win_rate": 0.5,
            "blue_win_rate": 0.0,
            "kill_death_ratio": 1.0,
            "critic_loss": 0.1,
            "entropy": 0.2,
            "policy_gradient_norm": 0.3,
        })
    with (log_dir / "aircraft_timeseries.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "agent_id", "lon", "lat", "altitude", "speed", "yaw", "pitch"])
        writer.writeheader()
        writer.writerow({"step": 0, "agent_id": "red_0", "lon": 120, "lat": 60, "altitude": 6000, "speed": 250, "yaw": 0, "pitch": 0})
    with (log_dir / "reward_components.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "agent_id", "role", "total_reward", "mav_survival_reward"])
        writer.writeheader()
        writer.writerow({"step": 0, "agent_id": "red_0", "role": "mav", "total_reward": 1.0, "mav_survival_reward": 0.2})
    (log_dir / "training_efficiency.json").write_text(json.dumps({"steps_per_second_mean": 10.0}), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_paper_style_plots.py",
            "--input-dir",
            str(log_dir),
            "--output-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=True,
    )
    assert (out_dir / "figure_index.md").exists()
    assert (out_dir / "reward_curve.png").exists()
