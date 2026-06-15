from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_analyze_entity_sanity_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/analyze_entity_sanity.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--output-dir" in result.stdout


def test_analyze_entity_sanity_generates_report_from_fake_output(tmp_path):
    out = tmp_path / "fake_entity_run"
    _write_csv(
        out / "train_log.csv",
        [
            {
                "iteration": 1,
                "total_steps": 256,
                "actor_loss_mav": "0.1",
                "actor_loss_uav": "0.2",
                "critic_loss": "1.0",
                "entropy_mav": "0.6",
                "entropy_uav": "0.5",
                "mav_action_saturation_rate": "0.1",
                "uav_action_saturation_rate": "0.2",
                "action_log_std_mav_min": "-1.2",
                "action_log_std_mav_max": "-1.1",
                "action_log_std_mav_mean": "-1.15",
                "action_log_std_uav_min": "-1.3",
                "action_log_std_uav_max": "-1.2",
                "action_log_std_uav_mean": "-1.25",
                "red_missiles_fired": "1",
                "missile_hits": "0",
                "nan_detected": "0",
            }
        ],
    )
    _write_csv(
        out / "eval_log.csv",
        [
            {
                "total_steps": 256,
                "iteration": 1,
                "config": "cfg",
                "red_win_rate": "0.0",
                "blue_win_rate": "1.0",
                "draw_rate": "0.0",
                "timeout_rate": "0.0",
                "mav_survival_rate": "0.0",
                "blue_dead_mean": "0.0",
                "red_missile_hits_mean": "0.0",
            }
        ],
    )
    (out / "latest").mkdir(parents=True)
    (out / "best").mkdir(parents=True)
    (out / "latest" / "model.pt").write_bytes(b"fake")
    (out / "best" / "model.pt").write_bytes(b"fake")
    (out / "latest" / "meta.json").write_text(
        json.dumps({"policy_arch": "entity_attention", "total_env_steps_actual": 256, "nan_detected": False}),
        encoding="utf-8",
    )
    (out / "best" / "meta.json").write_text(
        json.dumps({"policy_arch": "entity_attention"}),
        encoding="utf-8",
    )

    subprocess.run(
        [sys.executable, "scripts/analyze_entity_sanity.py", "--output-dir", str(out)],
        cwd=ROOT,
        check=True,
    )
    report = json.loads((out / "entity_sanity_report.json").read_text(encoding="utf-8"))
    assert report["training_completed"] is True
    assert report["final_steps"] == 256
    assert report["policy_arch"] == "entity_attention"
    assert report["latest_checkpoint_exists"] is True
    assert (out / "entity_sanity_report.md").exists()
