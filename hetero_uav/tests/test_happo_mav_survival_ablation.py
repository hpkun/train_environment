import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(args, timeout=300):
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def test_happo_mav_survival_ablation_help_runs():
    result = _run(["scripts/audit_happo_mav_survival_ablation.py", "--help"])
    assert result.returncode == 0
    assert "mav survival ablation" in result.stdout.lower()


def test_happo_mav_survival_ablation_missing_model_exits_cleanly(tmp_path):
    result = _run([
        "scripts/audit_happo_mav_survival_ablation.py",
        "--experiment-dir", str(tmp_path / "missing"),
        "--episodes", "1",
    ])
    assert result.returncode != 0
    assert "checkpoint not found" in (result.stderr + result.stdout).lower()


def test_happo_mav_survival_ablation_fast_schema(tmp_path):
    exp_dir = ROOT / "outputs" / "happo_3v2_reference_200k"
    if not (exp_dir / "best" / "model.pt").exists():
        return
    out_json = tmp_path / "ablation.json"
    result = _run([
        "scripts/audit_happo_mav_survival_ablation.py",
        "--episodes", "1",
        "--max-steps-override", "3",
        "--checkpoints", "best",
        "--cases", "learned_all",
        "--output-json", str(out_json),
        "--output-md", str(tmp_path / "ablation.md"),
    ])
    assert result.returncode == 0, result.stderr + result.stdout
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert {"records", "summary", "conclusion"}.issubset(data)
    record = data["records"][0]
    for key in ["mav_survival_rate", "mav_death_rate", "blue_dead_mean", "red_missile_hits_mean"]:
        assert key in record
