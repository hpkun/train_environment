import numpy as np
import pytest

from aircombat_env_v1 import geometry


def test_lla_neu_round_trip():
    if geometry.pymap3d is None:
        pytest.skip("pymap3d is not installed")
    origin = (120.0, 60.0, 100.0)
    lla = np.array([120.01, 60.005, 450.0])
    neu = geometry.LLA2NEU(*lla, *origin)
    recovered = geometry.NEU2LLA(*neu, *origin)
    assert np.allclose(recovered, lla, atol=1e-7)


def test_body_ned_matrices_are_inverses():
    matrix = geometry.body_to_ned_matrix(0.2, -0.1, 1.4)
    inverse = geometry.ned_to_body_matrix(0.2, -0.1, 1.4)
    assert np.allclose(matrix @ inverse, np.eye(3), atol=1e-12)


@pytest.mark.parametrize("angle, expected", [
    (3.0 * np.pi, np.pi),
    (-3.0 * np.pi, np.pi),
    (2.0 * np.pi + 0.2, 0.2),
])
def test_angle_wrapping(angle, expected):
    assert np.isclose(geometry.in_range_rad(angle), expected)
