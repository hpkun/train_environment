from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _yaml(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def test_paper_aligned_configs_enable_mav_trim_only():
    cfg = _yaml("uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml")
    assert cfg["action_trim_by_role"]["mav"]["pitch"] == 0.10
    assert cfg["action_trim_by_role"]["mav"]["heading"] == 0.0
    assert cfg["action_trim_by_role"]["mav"]["speed"] == 0.0

    cfg_5v4 = _yaml("uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml")
    cfg_3v2_no_trim = _yaml("uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_no_mav_trim.yaml")
    assert cfg_5v4["action_trim_by_role"]["mav"] == cfg_3v2_no_trim["action_trim_by_role"]["mav"]

    for path in [
        "uav_env/JSBSim/configs/hetero_balanced_mav_shared_geo_3v3.yaml",
        "uav_env/JSBSim/configs/hetero_balanced_mav_shared_geo_4v4.yaml",
    ]:
        cfg = _yaml(path)
        assert "action_trim_by_role" not in cfg or "mav" not in cfg["action_trim_by_role"]


def test_action_trim_helper_keeps_default_compatible():
    from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv

    env = HeteroUavCombatEnv(
        max_num_red=1,
        max_num_blue=1,
        max_steps=1,
        suppress_jsbsim_output=False,
    )
    try:
        raw = {"red_0": np.array([0.0, 0.0, 0.0], dtype=np.float32)}
        trimmed = env._apply_action_trim(raw)
        np.testing.assert_allclose(trimmed["red_0"], raw["red_0"])
    finally:
        env.close()


def test_action_trim_helper_applies_mav_pitch_trim():
    from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv

    env = HeteroUavCombatEnv(
        max_num_red=1,
        max_num_blue=1,
        max_steps=1,
        action_trim_by_role={"mav": {"pitch": 0.10, "heading": 0.0, "speed": 0.0}},
        suppress_jsbsim_output=False,
    )
    try:
        raw = {"red_0": np.array([0.0, 0.0, 0.0], dtype=np.float32)}
        trimmed = env._apply_action_trim(raw)
        np.testing.assert_allclose(trimmed["red_0"], np.array([0.10, 0.0, 0.0], dtype=np.float32))
        assert env._last_action_trim_applied["red_0"] == [0.10, 0.0, 0.0]
    finally:
        env.close()


def test_diagnose_mav_action_trim_effect_smoke():
    output_json = ROOT / "outputs/test_environment_audit/mav_action_trim_effect.json"
    subprocess.run(
        [
            "python",
            "scripts/diagnose_mav_action_trim_effect.py",
            "--steps",
            "80",
            "--output-json",
            str(output_json),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    data = json.loads(output_json.read_text(encoding="utf-8"))
    summary = data["summary"]
    assert "trim_improves_altitude" in summary
    assert "trim_prevents_crash" in summary
    assert "recommend_keep_trim" in summary
    by_case = {record["case"]: record for record in data["records"]}
    disabled = by_case["trim_disabled_zero"]
    enabled = by_case["trim_enabled_zero"]
    assert enabled["mav_final_altitude_m"] > disabled["mav_final_altitude_m"] + 100.0


def test_export_tacview_help_has_disable_config_trim():
    result = subprocess.run(
        ["python", "scripts/export_hetero_tacview_acmi.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--disable-config-trim" in result.stdout


def test_mav_action_trim_doc_exists():
    doc = ROOT / "docs/mav_action_trim_design.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    for phrase in [
        "action_trim_by_role",
        "A-4",
        "zero action",
        "not reward shaping",
        "no aircraft XML",
    ]:
        assert phrase in text
