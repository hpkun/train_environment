"""Pure geometry smoke test for AO/TA/q_LOS comparison.

No JSBSim, no env import.  Constructs four static geometry cases and prints
the diagnostic output, then asserts expected angular relationships.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_uav_env.alignment.geometry_diagnostics import (
    compute_body_los_angles,
    compute_current_ao_ta_r,
    describe_geometry_case,
)


def main() -> None:
    # ---- Case 1: head-on, target ahead ----
    ego_pos1 = np.array([0.0, 0.0, 6000.0])
    ego_vel1 = np.array([300.0, 0.0, 0.0])
    ego_rpy1 = np.array([0.0, 0.0, 0.0])
    tgt_pos1 = np.array([10000.0, 0.0, 6000.0])
    tgt_vel1 = np.array([-300.0, 0.0, 0.0])

    print(describe_geometry_case("Case 1: head-on, target ahead",
                                 ego_pos1, ego_vel1, ego_rpy1,
                                 tgt_pos1, tgt_vel1))
    print()

    # ---- Case 2: target behind ----
    ego_pos2 = np.array([0.0, 0.0, 6000.0])
    ego_vel2 = np.array([300.0, 0.0, 0.0])
    ego_rpy2 = np.array([0.0, 0.0, 0.0])
    tgt_pos2 = np.array([-10000.0, 0.0, 6000.0])
    tgt_vel2 = np.array([-300.0, 0.0, 0.0])

    print(describe_geometry_case("Case 2: target behind",
                                 ego_pos2, ego_vel2, ego_rpy2,
                                 tgt_pos2, tgt_vel2))
    print()

    # ---- Case 3: target right side ----
    ego_pos3 = np.array([0.0, 0.0, 6000.0])
    ego_vel3 = np.array([300.0, 0.0, 0.0])
    ego_rpy3 = np.array([0.0, 0.0, 0.0])
    tgt_pos3 = np.array([0.0, 10000.0, 6000.0])
    tgt_vel3 = np.array([0.0, -300.0, 0.0])

    print(describe_geometry_case("Case 3: target right side",
                                 ego_pos3, ego_vel3, ego_rpy3,
                                 tgt_pos3, tgt_vel3))
    print()

    # ---- Case 4: target above ahead ----
    ego_pos4 = np.array([0.0, 0.0, 6000.0])
    ego_vel4 = np.array([300.0, 0.0, 0.0])
    ego_rpy4 = np.array([0.0, 0.0, 0.0])
    tgt_pos4 = np.array([10000.0, 0.0, 8000.0])
    tgt_vel4 = np.array([-300.0, 0.0, 0.0])

    print(describe_geometry_case("Case 4: target above ahead",
                                 ego_pos4, ego_vel4, ego_rpy4,
                                 tgt_pos4, tgt_vel4))
    print()

    # ---- Assertions ----
    # Case 1: head-on ahead → q_los should be near 0
    los1 = compute_body_los_angles(ego_pos1, ego_rpy1, tgt_pos1)
    assert np.isfinite(los1["q_los_body_x_rad"])
    assert los1["q_los_body_x_deg"] < 5.0, f"expected near 0, got {los1['q_los_body_x_deg']}"

    # Case 2: behind → q_los should be near 180
    los2 = compute_body_los_angles(ego_pos2, ego_rpy2, tgt_pos2)
    assert los2["q_los_body_x_deg"] > 175.0

    # Case 3: right side → q_los should be near 90
    los3 = compute_body_los_angles(ego_pos3, ego_rpy3, tgt_pos3)
    assert 85.0 < los3["q_los_body_x_deg"] < 95.0

    # Case 4: above ahead → theta_los_deg > 0 (target is above)
    los4 = compute_body_los_angles(ego_pos4, ego_rpy4, tgt_pos4)
    assert los4["theta_los_deg"] > 0.0

    # All AO/TA/R finite
    for case_name, ep, ev, tp, tv in [
        ("head-on", ego_pos1, ego_vel1, tgt_pos1, tgt_vel1),
        ("behind", ego_pos2, ego_vel2, tgt_pos2, tgt_vel2),
        ("right", ego_pos3, ego_vel3, tgt_pos3, tgt_vel3),
        ("above", ego_pos4, ego_vel4, tgt_pos4, tgt_vel4),
    ]:
        ao_ta = compute_current_ao_ta_r(ep, ev, tp, tv)
        assert np.isfinite(ao_ta["AO_rad"])
        assert np.isfinite(ao_ta["TA_rad"])
        assert np.isfinite(ao_ta["R_m"])
        assert ao_ta["R_m"] > 0.0

    # Case 1 same-alt: 2D AO and body q_los should be close
    ao_ta1 = compute_current_ao_ta_r(ego_pos1, ego_vel1, tgt_pos1, tgt_vel1)
    los1_q = los1["q_los_body_x_deg"]
    ao1_unsigned = abs(ao_ta1["AO_deg"])
    assert abs(ao1_unsigned - los1_q) < 1.0, (
        f"same-alt head-on: |AO|={ao1_unsigned:.2f} vs q_los={los1_q:.2f}")

    # Case 4 altitude difference: 2D AO and body q_los SHOULD differ
    ao_ta4 = compute_current_ao_ta_r(ego_pos4, ego_vel4, tgt_pos4, tgt_vel4)
    los4_q = los4["q_los_body_x_deg"]
    ao4_unsigned = abs(ao_ta4["AO_deg"])
    # With a 2000 m altitude difference over 10000 m horizontal, the
    # 3D elevation is about arctan(2000/10000) ≈ 11.3°.  The 2D AO sees
    # pure head-on (0°) while the 3D q_los sees the vertical offset.
    assert abs(ao4_unsigned - los4_q) > 0.5, (
        f"alt-diff: 2D AO={ao4_unsigned:.2f} and 3D q_los={los4_q:.2f} "
        f"should differ due to vertical offset")

    print("geometry diagnostics smoke test passed")


if __name__ == "__main__":
    main()
