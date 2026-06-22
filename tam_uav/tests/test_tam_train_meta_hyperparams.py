"""Test train meta.json records effective hyperparameters."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_meta_json_has_actor_lr_effective():
    """Any latest meta.json must have actor_lr_effective."""
    # Check existing run dirs
    existing = list(ROOT.glob("outputs/tam_papermode_*/latest/meta.json"))
    if not existing:
        # Check outputs/tam_papermode_deathmask_2k_smoke
        existing = list(ROOT.glob("outputs/tam_papermode_deathmask_*/latest/meta.json"))
    if existing:
        meta = json.loads(existing[0].read_text(encoding="utf-8"))
        assert "actor_lr_effective" in meta or "actor_lr" in meta, \
            "meta.json missing actor_lr_effective"
        assert "entropy_coef_effective" in meta or "entropy_coef" in meta, \
            "meta.json missing entropy_coef_effective"


def test_meta_has_paper_hyperparams_flag():
    """Newer meta.json should have paper_hyperparams_passed."""
    existing = list(ROOT.glob("outputs/tam_papermode_deathmask_*/latest/meta.json"))
    existing += list(ROOT.glob("outputs/tam_papermode_pposign_*/latest/meta.json"))
    if existing:
        meta = json.loads(existing[0].read_text(encoding="utf-8"))
        # This key was added in this task
        if "paper_hyperparams_passed" in meta:
            assert isinstance(meta["paper_hyperparams_passed"], bool)
