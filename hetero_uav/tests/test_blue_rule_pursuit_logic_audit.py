from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))


from scripts.audit_blue_rule_pursuit_logic import _select_target_debug  # noqa: E402
from scripts.train_happo_reference_parallel import RemoteEnvProxy  # noqa: E402
from rule_based_agent import _blue_pursuit_action_impl  # noqa: E402


def _base_obs(num_blue: int = 2, num_red: int = 3) -> dict:
    enemy_states = np.zeros((num_red, 11), dtype=np.float32)
    enemy_states[0] = np.asarray(
        [0.125, 0.0, 0.0, 0.0, 0.10, 0.125, 0.45, 0.0, 1.0, 0.0, 1.0],
        dtype=np.float32,
    )
    return {
        "ego_state": np.asarray(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.55, 0.0, 1.0, 0.0, 1.0],
            dtype=np.float32,
        ),
        "enemy_states": enemy_states,
        "death_mask": np.ones(num_blue + num_red, dtype=np.float32),
        "altitude": np.asarray([7000.0], dtype=np.float32),
        "velocity": np.asarray([300.0, 0.0, 0.0], dtype=np.float32),
    }


def test_valid_enemy_states_do_not_enter_no_target_selection():
    obs = _base_obs()
    selected = _select_target_debug(obs, num_blue=2, num_red=3)

    assert selected["target_idx"] == 0
    assert selected["reason_if_no_target"] == ""
    assert selected["target_quality"] == "radar"


def test_target_idx_none_only_all_dead_or_invalid_tracks():
    obs_dead = _base_obs()
    obs_dead["death_mask"][2:] = 0.0
    selected_dead = _select_target_debug(obs_dead, num_blue=2, num_red=3)
    assert selected_dead["target_idx"] is None
    assert selected_dead["reason_if_no_target"] == "all_red_dead"

    obs_invalid = _base_obs()
    obs_invalid["enemy_states"][:] = 0.0
    selected_invalid = _select_target_debug(obs_invalid, num_blue=2, num_red=3)
    assert selected_invalid["target_idx"] is None
    assert selected_invalid["reason_if_no_target"] == "all_alive_red_tracks_invalid"


def test_parallel_proxy_exposes_blue_own_position_and_heading():
    proxy = RemoteEnvProxy(
        {
            "red_ids": ["red_0"],
            "blue_ids": ["blue_0"],
            "agent_ids": ["red_0", "blue_0"],
            "max_steps": 10,
        },
        {
            "blue_own_positions": {"blue_0": np.asarray([1.0, 2.0, 6000.0])},
            "blue_own_kinematics": {"blue_0": {"heading": 1.25}},
        },
    )

    assert "blue_0" in proxy.get_blue_own_positions()
    assert proxy.get_blue_own_kinematics()["blue_0"]["heading"] == 1.25


def test_heading_action_is_absolute_and_aligned_with_target_bearing():
    obs = _base_obs()
    obs["enemy_states"][0, 3] = 0.0
    obs["enemy_states"][0, 4] = 0.0  # AWACS-quality track: direct bearing, no lead.
    own_heading = math.pi / 2.0

    action = _blue_pursuit_action_impl(
        obs,
        num_blue=2,
        num_red=3,
        blue_id=0,
        forced_target_idx=0,
        own_position=np.asarray([0.0, 0.0, 7000.0], dtype=np.float32),
        own_heading=own_heading,
    )

    assert np.isfinite(action).all()
    assert abs(float(action[1]) * math.pi - own_heading) < math.radians(1.0)


def test_static_red_passes_behind_blue_produces_reverse_pursuit_heading():
    own_heading = 0.0
    target_bearing = math.pi
    obs = _base_obs()
    obs["enemy_states"][0, 4] = 0.0  # AWACS branch avoids target-velocity lead ambiguity.
    for _ in range(24):
        ao = (target_bearing - own_heading + math.pi) % (2.0 * math.pi) - math.pi
        obs["enemy_states"][0, 0] = math.cos(ao) * 0.125
        obs["enemy_states"][0, 1] = math.sin(ao) * 0.125
        obs["enemy_states"][0, 3] = ao / math.pi
        action = _blue_pursuit_action_impl(
            obs,
            num_blue=2,
            num_red=3,
            blue_id=0,
            forced_target_idx=0,
            own_position=np.asarray([0.0, 0.0, 7000.0], dtype=np.float32),
            own_heading=own_heading,
        )
        own_heading = float(action[1]) * math.pi

    assert abs((target_bearing - own_heading + math.pi) % (2.0 * math.pi) - math.pi) < math.radians(20.0)


def test_audit_script_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/audit_blue_rule_pursuit_logic.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )

    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--red-mode" in result.stdout
