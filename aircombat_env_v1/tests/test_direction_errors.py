import numpy as np

from aircombat_env_v1.geometry import paper_direction_errors


def test_matching_attitude_has_zero_errors():
    errors = paper_direction_errors(0.0, 0.2, -0.7, 0.2, -0.7)
    assert np.allclose(errors, (0.0, 0.0), atol=1e-12)


def test_right_and_left_targets_have_opposite_roll_signs():
    right, _ = paper_direction_errors(0.0, 0.0, 0.0, 0.0, 0.3)
    left, _ = paper_direction_errors(0.0, 0.0, 0.0, 0.0, -0.3)
    assert right > 0.0
    assert left < 0.0
    assert np.isclose(right, -left)


def test_climb_and_descent_have_opposite_pitch_signs():
    _, climb = paper_direction_errors(0.0, 0.0, 0.0, 0.2, 0.0)
    _, descent = paper_direction_errors(0.0, 0.0, 0.0, -0.2, 0.0)
    assert climb > 0.0
    assert descent < 0.0
    assert np.isclose(climb, -descent)


def test_heading_wrap_is_continuous_and_directional():
    eastward, _ = paper_direction_errors(
        0.0, 0.0, np.deg2rad(179.0), 0.0, np.deg2rad(-179.0))
    westward, _ = paper_direction_errors(
        0.0, 0.0, np.deg2rad(-179.0), 0.0, np.deg2rad(179.0))
    assert eastward > 0.0 and westward < 0.0
    assert np.isclose(abs(eastward), np.deg2rad(2.0), atol=1e-12)
    assert np.isclose(eastward, -westward, atol=1e-12)
