import numpy as np
import pytest

from aircombat_env_v1.combat import (action_to_targets, hit_event,
                                     update_attack_dwell)
from aircombat_env_v1.reward import terminal_reward


def test_action_mapping_and_heading_wrap():
    pitch, heading, speed = action_to_targets(
        np.array([1.0, 1.0, -1.0]), np.deg2rad(170.0))
    assert pitch == pytest.approx(np.deg2rad(20.0))
    assert heading == pytest.approx(np.deg2rad(-130.0))
    assert speed == pytest.approx(200.0)


def test_attack_dwell_hits_after_five_decisions_and_resets():
    dwell = 0.0
    for _ in range(5):
        dwell = update_attack_dwell(dwell, True)
    assert dwell == pytest.approx(1.0)
    assert hit_event(dwell, 0.0) == "red_hit"
    assert update_attack_dwell(dwell, False) == 0.0


def test_simultaneous_hit_is_draw():
    assert hit_event(1.0, 1.0) == "draw_simultaneous_hit"


@pytest.mark.parametrize("event, expected", [
    ("red_hit", 10.0), ("blue_crash", 10.0),
    ("blue_hit", -10.0), ("red_crash", -10.0),
    ("draw_simultaneous_hit", 0.0), ("timeout", 0.0),
])
def test_terminal_reward_sign(event, expected):
    assert terminal_reward(event) == expected
