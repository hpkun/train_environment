"""Test that manifest contains paper-listed hyperparameters."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_manifest_100k_has_actor_lr():
    mp = ROOT / "outputs/tam_paper_run_manifest/manifest.json"
    if not mp.exists():
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "scripts/prepare_tam_paper_run_manifest.py")],
                       cwd=ROOT, capture_output=True)
    assert mp.exists(), "manifest.json not found"
    m = json.loads(mp.read_text(encoding="utf-8"))
    cmd = m["commands"]["100k_val"]["command"]
    assert "--actor-lr 0.0005" in cmd, f"Missing --actor-lr 0.0005 in: {cmd[:200]}"


def test_manifest_100k_has_entropy_coef():
    mp = ROOT / "outputs/tam_paper_run_manifest/manifest.json"
    m = json.loads(mp.read_text(encoding="utf-8"))
    cmd = m["commands"]["100k_val"]["command"]
    assert "--entropy-coef 0.01" in cmd


def test_manifest_100k_has_critic_lr():
    mp = ROOT / "outputs/tam_paper_run_manifest/manifest.json"
    m = json.loads(mp.read_text(encoding="utf-8"))
    cmd = m["commands"]["100k_val"]["command"]
    assert "--critic-lr 0.0005" in cmd


def test_manifest_100k_has_tam_paper_mode():
    mp = ROOT / "outputs/tam_paper_run_manifest/manifest.json"
    m = json.loads(mp.read_text(encoding="utf-8"))
    cmd = m["commands"]["100k_val"]["command"]
    assert "--tam-paper-mode" in cmd


def test_manifest_100k_runnable_by_codex():
    mp = ROOT / "outputs/tam_paper_run_manifest/manifest.json"
    m = json.loads(mp.read_text(encoding="utf-8"))
    assert m["commands"]["100k_val"]["runnable_by_codex"] is True


def test_manifest_2M_requires_user_run():
    mp = ROOT / "outputs/tam_paper_run_manifest/manifest.json"
    m = json.loads(mp.read_text(encoding="utf-8"))
    assert m["commands"]["2M_probe"]["requires_user_run"] is True


def test_manifest_ppo_epochs_source():
    mp = ROOT / "outputs/tam_paper_run_manifest/manifest.json"
    m = json.loads(mp.read_text(encoding="utf-8"))
    assert m["paper_hyperparams"]["ppo_epochs_source"] == "implementation_default_not_paper_listed"
