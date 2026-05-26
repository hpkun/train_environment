from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rule_based_agent as blue


def _reset_policy_memory() -> None:
    blue._prev_heading_cmd.clear()
    blue._prev_lead_bearing.clear()
    blue._last_target_bearing.clear()
    blue._lost_target_steps.clear()


def _base_obs(enemy_vec: np.ndarray, red_alive: bool = True) -> dict:
    ego = np.zeros(11, dtype=np.float32)
    ego[6] = 0.6
    ego[8] = 1.0
    ego[10] = 1.0
    return {
        "ego_state": ego,
        "ally_states": np.zeros((0, 11), dtype=np.float32),
        "enemy_states": np.asarray([enemy_vec], dtype=np.float32),
        "death_mask": np.asarray([1.0, 1.0 if red_alive else 0.0], dtype=np.float32),
        "missile_warning": np.zeros(1, dtype=np.float32),
        "altitude": np.asarray([6000.0], dtype=np.float32),
        "velocity": np.asarray([300.0, 0.0, 0.0], dtype=np.float32),
    }


def _awacs_vec(ao_norm: float = 0.25, dz_norm: float = 0.0, r_norm: float = 0.1) -> np.ndarray:
    vec = np.zeros(11, dtype=np.float32)
    vec[0] = 0.1
    vec[1] = 0.1
    vec[2] = dz_norm
    vec[3] = ao_norm
    vec[4] = 0.0
    vec[5] = r_norm
    vec[6] = 0.0
    return vec


def _radar_vec() -> np.ndarray:
    vec = _awacs_vec(ao_norm=0.1)
    vec[4] = 0.2
    vec[6] = 0.5
    vec[8] = 1.0
    vec[10] = 1.0
    return vec


def test_track_quality() -> None:
    assert blue._target_track_quality(np.zeros(11, dtype=np.float32)) == "invalid"
    assert blue._target_track_quality(_awacs_vec()) == "awacs"
    assert blue._target_track_quality(_radar_vec()) == "radar"


def test_doomed_alt_filter_removed() -> None:
    _reset_policy_memory()
    doomed_body_z = _awacs_vec(ao_norm=0.2, dz_norm=-2.0)
    pursuit = blue.blue_pursuit_action(_base_obs(doomed_body_z), 1, 1, 0)
    cruise = blue.blue_pursuit_action(_base_obs(np.zeros(11, dtype=np.float32), red_alive=False), 1, 1, 0)
    assert pursuit.shape == (3,)
    assert not np.isclose(pursuit[1], cruise[1])


def test_awacs_pursuit_turns_to_coarse_bearing() -> None:
    _reset_policy_memory()
    action = blue.blue_pursuit_action(_base_obs(_awacs_vec(ao_norm=0.25)), 1, 1, 0)
    assert action.shape == (3,)
    assert action[1] > 0.0


def test_lost_target_reacquisition() -> None:
    _reset_policy_memory()
    first = blue.blue_pursuit_action(_base_obs(_awacs_vec(ao_norm=0.25)), 1, 1, 0)
    assert first[1] > 0.0
    invalid_alive = _base_obs(np.zeros(11, dtype=np.float32), red_alive=True)
    reacquire = blue.blue_pursuit_action(invalid_alive, 1, 1, 0)
    assert reacquire[1] > 0.0
    for _ in range(55):
        last = blue.blue_pursuit_action(invalid_alive, 1, 1, 0)
    assert np.isclose(last[1], 0.0, atol=1e-6)


def test_boundary_safety_wins_over_awacs() -> None:
    _reset_policy_memory()
    obs = _base_obs(_awacs_vec(ao_norm=0.25))
    own_position = np.asarray([0.0, 39000.0, 6000.0], dtype=np.float32)
    own_heading = np.pi / 2.0
    action = blue.blue_pursuit_action(
        obs, 1, 1, 0, own_position=own_position, own_heading=own_heading)
    # Boundary safety should steer left/inward from an eastbound heading and
    # reduce speed below the normal AWACS pursuit throttle.
    assert action[1] < own_heading / np.pi
    assert action[2] < 1.0


def main() -> None:
    test_track_quality()
    test_doomed_alt_filter_removed()
    test_awacs_pursuit_turns_to_coarse_bearing()
    test_lost_target_reacquisition()
    test_boundary_safety_wins_over_awacs()
    print("blue reacquisition policy smoke test passed")


if __name__ == "__main__":
    main()
