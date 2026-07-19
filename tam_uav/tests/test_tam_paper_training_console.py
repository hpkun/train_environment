from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import scripts.train_tam_paper_vanilla_happo as training
from uav_env.JSBSim.paper.protocol import ENVIRONMENT_FIDELITY_REVISION


ROOT = Path(__file__).parents[1]


def test_evaluation_and_console_cli_defaults():
    args = training.parse_args([])
    assert args.disable_evaluation is False
    assert args.console_log_interval == 10240


def test_console_line_contains_compact_progress_and_episode_metrics():
    line = training.format_console_training_line(
        128, 512, 1, 1,
        [{
            "red_team_episode_return": -12.5,
            "winner": "red",
            "red_survival_rate": 0.5,
            "red_hit_rate": 0.25,
            "red_crashes": 1,
            "red_initial_count": 2,
        }],
        32.0)
    assert line.startswith("[TRAIN] step=128/512 progress=25.00%")
    assert "reward100=-12.5" in line
    assert "win100=1.00" in line
    assert "speed=4.00step/s" in line


def test_disable_evaluation_skips_all_evaluation_outputs(monkeypatch, capsys):
    def forbidden_evaluation(*args, **kwargs):
        raise AssertionError("deterministic_evaluate must not be called")

    monkeypatch.setattr(training, "deterministic_evaluate", forbidden_evaluation)
    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix="pytest_no_evaluation_", dir=outputs) as directory:
        output = Path(directory)
        monkeypatch.setattr(sys, "argv", [
            "train_tam_paper_vanilla_happo.py",
            "--scenario", "2v2",
            "--seed", "9137",
            "--total-environment-steps", "1",
            "--rollout-length", "1",
            "--minibatch-size", "1",
            "--ppo-epochs", "1",
            "--checkpoint-interval", "1",
            "--console-log-interval", "1",
            "--device", "cpu",
            "--disable-evaluation",
            "--output-directory", str(output.relative_to(ROOT)),
        ])
        assert training.main() == 0

        assert (output / "summary.json").is_file()
        assert not (output / "baseline_reference.json").exists()
        assert not (output / "evaluation_0.json").exists()
        assert not (output / "best_evaluation_checkpoint.pt").exists()
        assert not list(output.glob("evaluation_*.json"))
        assert not (output / "evaluation_history.csv").exists()
        assert not (output / "evaluation_history.jsonl").exists()
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert summary["latest_evaluation"] is None
        assert summary["best_evaluation_checkpoint"] is None
        assert summary["best_evaluation_red_team_return"] is None

    stdout = capsys.readouterr().out
    assert "[CONFIG] scenario=2v2" in stdout
    assert "evaluation=false" in stdout
    assert "[TRAIN] step=1/1" in stdout
    assert "reward100=" in stdout and "win100=" in stdout
    assert "[DONE] steps=1" in stdout
    assert '"python_version"' not in stdout
    assert '"config_snapshot"' not in stdout


def test_environment_revision_remains_published_rules_simplified_v4():
    assert ENVIRONMENT_FIDELITY_REVISION == "published_rules_simplified_v4"
