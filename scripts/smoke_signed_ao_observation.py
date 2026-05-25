"""Pure smoke test for signed AO collinear fix in _make_entity_vec.

Does not import UavCombatEnv, does not create an environment, does not
trigger JSBSim.  Only exercises the new helper and verifies get2d_AO_TA_R
integration via pure 2D cases.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_uav_env.env import _signed_ao_from_unsigned_and_side
from my_uav_env.utils import get2d_AO_TA_R


def main() -> None:
    # ---- Unit tests for the helper itself ----
    # ahead: small unsigned AO → 0
    assert _signed_ao_from_unsigned_and_side(0.0, 0.0) == 0.0
    # behind: unsigned AO = pi → pi (was 0 before the fix)
    assert _signed_ao_from_unsigned_and_side(np.pi, 0.0) == np.pi
    # right: side_flag > 0 → +AO_unsigned
    assert _signed_ao_from_unsigned_and_side(np.pi / 2, 1.0) == np.pi / 2
    # left: side_flag < 0 → -AO_unsigned
    assert _signed_ao_from_unsigned_and_side(np.pi / 2, -1.0) == -np.pi / 2
    # general: side_flag > 0 preserves magnitude
    assert _signed_ao_from_unsigned_and_side(1.2, 0.5) == 1.2
    assert _signed_ao_from_unsigned_and_side(1.2, -0.5) == -1.2

    # ---- 2D cases using the same feature convention as _make_entity_vec ----
    def _feat(x, y, vx, vy):
        """Return 6-dim [north, east, down, vn, ve, vd]."""
        return np.array([x, y, 0.0, vx, vy, 0.0], dtype=np.float64)

    # Ego at origin, velocity +x (north), level flight.
    # Target ahead (+x): collinear ahead, small AO
    ego_feat = _feat(0.0, 0.0, 300.0, 0.0)
    tgt_ahead = _feat(10000.0, 0.0, -300.0, 0.0)
    ao_u, _ta, _r, side = get2d_AO_TA_R(ego_feat, tgt_ahead, return_side=True)
    ao_signed = _signed_ao_from_unsigned_and_side(ao_u, side)
    assert ao_u < 0.01, f"ahead: unsigned AO should be near 0, got {ao_u}"
    assert abs(ao_signed) < 0.01, f"ahead: signed AO should be near 0, got {ao_signed}"

    # Target behind (-x): collinear behind, AO ≈ π
    tgt_behind = _feat(-10000.0, 0.0, -300.0, 0.0)
    ao_u, _ta, _r, side = get2d_AO_TA_R(ego_feat, tgt_behind, return_side=True)
    ao_signed = _signed_ao_from_unsigned_and_side(ao_u, side)
    assert ao_u > np.pi - 0.01, f"behind: unsigned AO should be near pi, got {ao_u}"
    assert side == 0.0, f"behind: side_flag should be 0, got {side}"
    assert ao_signed > np.pi - 0.01, (
        f"behind: signed AO should be near pi (not 0), got {ao_signed}")

    # Target right (+y): should have positive signed AO
    tgt_right = _feat(0.0, 10000.0, 0.0, -300.0)
    ao_u, _ta, _r, side = get2d_AO_TA_R(ego_feat, tgt_right, return_side=True)
    ao_signed = _signed_ao_from_unsigned_and_side(ao_u, side)
    assert ao_signed > 0.0, f"right: signed AO should be positive, got {ao_signed}"
    assert ao_signed > np.pi / 4

    # Target left (-y): should have negative signed AO
    tgt_left = _feat(0.0, -10000.0, 0.0, 300.0)
    ao_u, _ta, _r, side = get2d_AO_TA_R(ego_feat, tgt_left, return_side=True)
    ao_signed = _signed_ao_from_unsigned_and_side(ao_u, side)
    assert ao_signed < 0.0, f"left: signed AO should be negative, got {ao_signed}"
    assert ao_signed < -np.pi / 4

    # Target ahead-right (diagonal): side_flag > 0, AO ≈ 45°
    tgt_diag_right = _feat(10000.0, 10000.0, -300.0, 0.0)
    ao_u, _ta, _r, side = get2d_AO_TA_R(ego_feat, tgt_diag_right, return_side=True)
    ao_signed = _signed_ao_from_unsigned_and_side(ao_u, side)
    assert side > 0, f"diag-right: side_flag should be > 0, got {side}"
    assert 0.6 < ao_signed < 1.0, (
        f"diag-right: signed AO should be ~45° (0.785 rad), got {ao_signed}")

    # Target ahead-left (diagonal): side_flag < 0, AO ≈ -45°
    tgt_diag_left = _feat(10000.0, -10000.0, -300.0, 0.0)
    ao_u, _ta, _r, side = get2d_AO_TA_R(ego_feat, tgt_diag_left, return_side=True)
    ao_signed = _signed_ao_from_unsigned_and_side(ao_u, side)
    assert side < 0, f"diag-left: side_flag should be < 0, got {side}"
    assert -1.0 < ao_signed < -0.6, (
        f"diag-left: signed AO should be ~-45° (-0.785 rad), got {ao_signed}")

    print("signed AO observation smoke test passed")


if __name__ == "__main__":
    main()
