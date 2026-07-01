"""Tests for configurable BRMA-style missile-warning evasion."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.JSBSim.env import UavCombatEnv

CFG_3V2 = (
    "uav_env/JSBSim/configs/"
    "hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_"
    "brma_role_no_missile_reward_v8_pn_missile.yaml"
)


class FakeSim:
    def __init__(self, uid: str, pos, vel):
        self.uid = uid
        self._pos = np.asarray(pos, dtype=np.float64)
        self._vel = np.asarray(vel, dtype=np.float64)
        self.under_missiles = []

    def get_position(self):
        return self._pos

    def get_velocity(self):
        return self._vel


class FakeMissile:
    def __init__(self, uid: str, target, pos, vel, alive=True):
        self.uid = uid
        self._target_id = target.uid
        self.target_aircraft = target
        self.is_alive = alive
        self._pos = np.asarray(pos, dtype=np.float64)
        self._vel = np.asarray(vel, dtype=np.float64)

    def get_position(self):
        return self._pos

    def get_velocity(self):
        return self._vel


def _env(teams: str):
    env = object.__new__(UavCombatEnv)
    env.missile_evasion_config = {"mode": "brma_scripted", "teams": teams}
    return env


def test_missile_evasion_enabled_by_team():
    assert _env("both")._missile_evasion_enabled_for("red_0") is True
    assert _env("both")._missile_evasion_enabled_for("blue_0") is True
    assert _env("red_only")._missile_evasion_enabled_for("red_0") is True
    assert _env("red_only")._missile_evasion_enabled_for("blue_0") is False
    assert _env("blue_only")._missile_evasion_enabled_for("red_0") is False
    assert _env("blue_only")._missile_evasion_enabled_for("blue_0") is True
    assert _env("none")._missile_evasion_enabled_for("red_0") is False
    assert _env("none")._missile_evasion_enabled_for("blue_0") is False


def test_select_incoming_missile_uses_smallest_tgo_and_closing_filter():
    env = _env("both")
    target = FakeSim("blue_0", pos=[1000.0, 0.0, 0.0], vel=[250.0, 0.0, 0.0])
    closing_slow = FakeMissile(
        "m_slow", target, pos=[0.0, 0.0, 0.0], vel=[500.0, 0.0, 0.0]
    )
    closing_fast = FakeMissile(
        "m_fast", target, pos=[400.0, 0.0, 0.0], vel=[700.0, 0.0, 0.0]
    )
    opening = FakeMissile(
        "m_opening", target, pos=[1200.0, 0.0, 0.0], vel=[700.0, 0.0, 0.0]
    )
    target.under_missiles = [closing_slow, opening, closing_fast]

    incoming, diag = env._select_incoming_missile_threat("blue_0", target)

    assert incoming.uid == "m_fast"
    assert diag["incoming_closing_speed_mps"] > 0.0
    assert diag["incoming_t_go_sec"] < 2.0


def test_select_incoming_missile_ignores_non_closing_missiles():
    env = _env("both")
    target = FakeSim("red_0", pos=[1000.0, 0.0, 0.0], vel=[250.0, 0.0, 0.0])
    opening = FakeMissile(
        "m_opening", target, pos=[1200.0, 0.0, 0.0], vel=[700.0, 0.0, 0.0]
    )
    target.under_missiles = [opening]

    incoming, diag = env._select_incoming_missile_threat("red_0", target)

    assert incoming is None
    assert diag == {}


def test_pn_missile_config_loads_with_both_team_evasion():
    from uav_env import make_env

    env = make_env(CFG_3V2, max_steps=5)
    try:
        assert env.missile_guidance_config["mode"] == "pn"
        assert env.missile_guidance_config["navigation_gain"] == 3.0
        assert env.missile_guidance_config["max_overload_g"] == 30.0
        assert env.missile_guidance_config["speed_mps"] == 600.0
        assert env.missile_evasion_config["mode"] == "brma_scripted"
        assert env.missile_evasion_config["teams"] == "both"
    finally:
        env.close()
