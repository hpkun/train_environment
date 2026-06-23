from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from uav_env.JSBSim.env import UavCombatEnv

ROOT = Path(__file__).resolve().parents[1]
MAIN_F16_DYNAMICS_F22_VISUAL_CONFIGS = [
    ROOT / "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_paper_role_reward_v1.yaml",
    ROOT / "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4_f16_dynamics_f22_visual_mav_paper_role_reward_v1.yaml",
    ROOT / "uav_env/JSBSim/configs/hetero_mav_shared_geo_7v6_f16_dynamics_f22_visual_mav_paper_role_reward_v1.yaml",
]


class _FakeSim:
    def __init__(
        self,
        uid: str,
        color: str,
        pos: tuple[float, float, float],
        vel: tuple[float, float, float],
        *,
        missiles: int = 2,
        alive: bool = True,
    ) -> None:
        self.uid = uid
        self.color = color
        self._pos = np.asarray(pos, dtype=np.float64)
        self._vel = np.asarray(vel, dtype=np.float64)
        self.num_left_missiles = missiles
        self.is_alive = alive

    def get_position(self):
        return self._pos

    def get_velocity(self):
        return self._vel


def _env_for_selection() -> UavCombatEnv:
    env = object.__new__(UavCombatEnv)
    env.MISSILE_LAUNCH_MIN_RANGE = UavCombatEnv.MISSILE_LAUNCH_MIN_RANGE
    env.MISSILE_LAUNCH_RANGE_THRESH = UavCombatEnv.MISSILE_LAUNCH_RANGE_THRESH
    env.MISSILE_LAUNCH_AO_THRESH = UavCombatEnv.MISSILE_LAUNCH_AO_THRESH
    env.MISSILE_LAUNCH_TA_THRESH = UavCombatEnv.MISSILE_LAUNCH_TA_THRESH
    env._engaged_targets = set()
    env.red_target_selection_mode = "mav_threat_rank"
    env.mav_observation_range_m = 80000.0
    env.agent_roles = {
        "red_0": "mav",
        "red_1": "attack_uav",
        "blue_0": "attack_uav",
        "blue_1": "attack_uav",
    }
    env.red_planes = {
        "red_0": _FakeSim("red_0", "Red", (6500.0, 0.0, -6000.0), (220.0, 0.0, 0.0), missiles=0),
        "red_1": _FakeSim("red_1", "Red", (0.0, 0.0, -6000.0), (250.0, 0.0, 0.0)),
    }
    return env


def test_red_mav_threat_ranking_prefers_supported_high_threat_target_over_closest():
    env = _env_for_selection()
    shooter = env.red_planes["red_1"]
    enemies = {
        "blue_0": _FakeSim("blue_0", "Blue", (4000.0, 0.0, -6000.0), (240.0, 0.0, 0.0)),
        "blue_1": _FakeSim("blue_1", "Blue", (6200.0, 0.0, -6000.0), (240.0, 0.0, 0.0)),
    }
    diag = {key: 0 for key in (
        "alive_enemy_pairs",
        "engaged_blocked",
        "unengaged_enemy_pairs",
        "range_ok_pairs",
        "ao_ok_pairs",
        "ta_ok_pairs",
        "geometry_ok_pairs",
    )}

    selected, distance, metrics, debug = env._select_missile_target("red_1", shooter, enemies, diag)

    assert selected is enemies["blue_1"]
    assert distance == metrics["range_m"]
    assert debug["target_selection_mode"] == "mav_threat_rank"
    assert debug["candidate_count"] == 2
    assert debug["selected_target_mav_support_score"] > 0.0
    assert debug["selected_target_score"] > 0.0


def test_blue_keeps_closest_target_selection_even_when_red_mode_is_ranked():
    env = _env_for_selection()
    shooter = _FakeSim("blue_0", "Blue", (0.0, 0.0, -6000.0), (250.0, 0.0, 0.0))
    enemies = {
        "red_0": _FakeSim("red_0", "Red", (7000.0, 0.0, -6000.0), (240.0, 0.0, 0.0), missiles=0),
        "red_1": _FakeSim("red_1", "Red", (3500.0, 0.0, -6000.0), (240.0, 0.0, 0.0)),
    }
    diag = {key: 0 for key in (
        "alive_enemy_pairs",
        "engaged_blocked",
        "unengaged_enemy_pairs",
        "range_ok_pairs",
        "ao_ok_pairs",
        "ta_ok_pairs",
        "geometry_ok_pairs",
    )}

    selected, _distance, _metrics, debug = env._select_missile_target("blue_0", shooter, enemies, diag)

    assert selected is enemies["red_1"]
    assert debug["target_selection_mode"] == "closest"
    assert debug["candidate_count"] == 2


def test_default_mode_is_closest_and_main_visual_configs_opt_in():
    env = UavCombatEnv(suppress_jsbsim_output=True)
    assert env.red_target_selection_mode == "closest"
    env.close()

    for config_path in MAIN_F16_DYNAMICS_F22_VISUAL_CONFIGS:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        assert cfg["red_target_selection_mode"] == "mav_threat_rank"


def test_ranked_selection_never_uses_targets_outside_existing_launch_gate():
    env = _env_for_selection()
    shooter = env.red_planes["red_1"]
    enemies = {
        "too_far": _FakeSim("too_far", "Blue", (15000.0, 0.0, -6000.0), (240.0, 0.0, 0.0)),
        "valid": _FakeSim("valid", "Blue", (5500.0, 0.0, -6000.0), (240.0, 0.0, 0.0)),
    }
    diag = {key: 0 for key in (
        "alive_enemy_pairs",
        "engaged_blocked",
        "unengaged_enemy_pairs",
        "range_ok_pairs",
        "ao_ok_pairs",
        "ta_ok_pairs",
        "geometry_ok_pairs",
    )}

    selected, _distance, _metrics, debug = env._select_missile_target("red_1", shooter, enemies, diag)

    assert selected is enemies["valid"]
    assert debug["candidate_count"] == 1


def test_launch_quality_record_includes_target_selection_diagnostics():
    env = _env_for_selection()
    env.current_step = 7
    env._physics_frame = 42
    shooter = env.red_planes["red_1"]
    target = _FakeSim("blue_1", "Blue", (6200.0, 0.0, -6000.0), (240.0, 0.0, 0.0))
    debug = {
        "target_selection_mode": "mav_threat_rank",
        "selected_target_score": 0.8,
        "selected_target_threat_score": 0.7,
        "selected_target_mav_support_score": 1.0,
        "selected_target_shot_quality_score": 0.6,
        "selected_target_range_m": 6200.0,
        "selected_target_AO_rad": 0.1,
        "selected_target_TA_rad": 2.5,
        "selected_target_is_mav_observed": True,
        "candidate_count": 2,
    }

    record = env._build_launch_quality_record(shooter, target, 6200.0, debug)

    for key, expected in debug.items():
        assert record[key] == expected


def test_red_target_selection_audit_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/audit_red_target_selection.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--episodes" in result.stdout
    assert "--max-steps" in result.stdout
