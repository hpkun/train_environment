"""Smoke test for 3D situation-reward candidate functions.

No JSBSim, no env import.  Compares current 2D AO/TA reward against
body-x 3D and velocity-LOS 3D candidates over four canonical geometry cases.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_uav_env.alignment.geometry_diagnostics import (
    compute_current_ao_ta_r,
    compute_pairwise_3d_q_los,
    describe_geometry_case,
)
from my_uav_env.alignment.situation_reward_candidates import (
    compare_situation_reward_candidates,
    situation_reward_3d_body_x_candidate,
    situation_reward_3d_velocity_candidate,
    situation_reward_current_formula,
)


def main() -> None:
    # ---- Case 1: head-on, target ahead ----
    ego_pos1 = np.array([0.0, 0.0, 6000.0])
    ego_vel1 = np.array([300.0, 0.0, 0.0])
    ego_rpy1 = np.array([0.0, 0.0, 0.0])
    tgt_pos1 = np.array([10000.0, 0.0, 6000.0])
    tgt_vel1 = np.array([-300.0, 0.0, 0.0])
    tgt_rpy1 = np.array([0.0, 0.0, np.pi])

    # ---- Case 2: target behind ----
    ego_pos2 = np.array([0.0, 0.0, 6000.0])
    ego_vel2 = np.array([300.0, 0.0, 0.0])
    ego_rpy2 = np.array([0.0, 0.0, 0.0])
    tgt_pos2 = np.array([-10000.0, 0.0, 6000.0])
    tgt_vel2 = np.array([-300.0, 0.0, 0.0])
    tgt_rpy2 = np.array([0.0, 0.0, np.pi])

    # ---- Case 3: target right side ----
    ego_pos3 = np.array([0.0, 0.0, 6000.0])
    ego_vel3 = np.array([300.0, 0.0, 0.0])
    ego_rpy3 = np.array([0.0, 0.0, 0.0])
    tgt_pos3 = np.array([0.0, 10000.0, 6000.0])
    tgt_vel3 = np.array([0.0, -300.0, 0.0])
    tgt_rpy3 = np.array([0.0, 0.0, -np.pi / 2])

    # ---- Case 4: target above ahead ----
    ego_pos4 = np.array([0.0, 0.0, 6000.0])
    ego_vel4 = np.array([300.0, 0.0, 0.0])
    ego_rpy4 = np.array([0.0, 0.0, 0.0])
    tgt_pos4 = np.array([10000.0, 0.0, 8000.0])
    tgt_vel4 = np.array([-300.0, 0.0, 0.0])
    tgt_rpy4 = np.array([0.0, 0.0, np.pi])

    cases = [
        ("Case 1: head-on, target ahead",
         ego_pos1, ego_vel1, ego_rpy1, tgt_pos1, tgt_vel1, tgt_rpy1),
        ("Case 2: target behind",
         ego_pos2, ego_vel2, ego_rpy2, tgt_pos2, tgt_vel2, tgt_rpy2),
        ("Case 3: target right side",
         ego_pos3, ego_vel3, ego_rpy3, tgt_pos3, tgt_vel3, tgt_rpy3),
        ("Case 4: target above ahead",
         ego_pos4, ego_vel4, ego_rpy4, tgt_pos4, tgt_vel4, tgt_rpy4),
    ]

    for name, ep, ev, er, tp, tv, tr in cases:
        print(describe_geometry_case(name, ep, ev, er, tp, tv, tr))
        comp = compare_situation_reward_candidates(ep, ev, er, tp, tv, tr)
        print(f"  --- situation reward candidates ---")
        print(f"  current 2D AO/TA     = {comp['current_2d_ao_ta']:+.6f}")
        print(f"  candidate 3D body-x  = {comp['candidate_3d_body_x']:+.6f}")
        print(f"  candidate 3D velocity= {comp['candidate_3d_velocity']:+.6f}")
        print()

    # ---- Assertions ----
    # All candidate outputs finite
    for name, ep, ev, er, tp, tv, tr in cases:
        comp = compare_situation_reward_candidates(ep, ev, er, tp, tv, tr)
        assert np.isfinite(comp["current_2d_ao_ta"]), f"{name}: current not finite"
        assert np.isfinite(comp["candidate_3d_body_x"]), f"{name}: body-x not finite"
        assert np.isfinite(comp["candidate_3d_velocity"]), f"{name}: velocity not finite"

    # Case 4 (above ahead): 2D AO and body-x q should differ
    ao_ta4 = compute_current_ao_ta_r(ego_pos4, ego_vel4, tgt_pos4, tgt_vel4)
    q3d4 = compute_pairwise_3d_q_los(
        ego_pos4, ego_vel4, ego_rpy4, tgt_pos4, tgt_vel4, tgt_rpy4,
    )
    ao4_unsigned = abs(ao_ta4["AO_deg"])
    assert abs(ao4_unsigned - q3d4["ego_body_x_q_deg"]) > 0.5, (
        "above ahead: 2D AO should differ from body-x q due to altitude")

    # Case 1 (head-on same alt): ego body-x q should be near 0
    q3d1 = compute_pairwise_3d_q_los(
        ego_pos1, ego_vel1, ego_rpy1, tgt_pos1, tgt_vel1, tgt_rpy1,
    )
    assert q3d1["ego_body_x_q_deg"] < 5.0
    # target facing back at ego → target body-x q near 0 too
    assert q3d1["target_body_x_q_deg"] < 5.0
    # ego velocity aligned with LOS → velocity q near 0
    assert q3d1["ego_velocity_q_deg"] < 5.0

    # Case 2 (behind): ego velocity faces away from target → velocity q near 180
    q3d2 = compute_pairwise_3d_q_los(
        ego_pos2, ego_vel2, ego_rpy2, tgt_pos2, tgt_vel2, tgt_rpy2,
    )
    assert q3d2["ego_velocity_q_deg"] > 175.0

    # Case 3 (right side): ego velocity perpendicular to LOS → velocity q near 90
    q3d3 = compute_pairwise_3d_q_los(
        ego_pos3, ego_vel3, ego_rpy3, tgt_pos3, tgt_vel3, tgt_rpy3,
    )
    assert 85.0 < q3d3["ego_velocity_q_deg"] < 95.0

    # Case 4 (above ahead): ego velocity q should have an elevation component
    # rel_pos = (10000, 0, 2000), velocity = (300, 0, 0)
    # angle = arccos(300*10000 / (300 * sqrt(10000^2+2000^2)))
    #       = arccos(10000/10198) ≈ 11.3°
    assert 10.0 < q3d4["ego_velocity_q_deg"] < 13.0

    # Sanity: current formula call matches compare output
    current_direct = situation_reward_current_formula(
        ao_ta4["AO_rad"], ao_ta4["TA_rad"], ao_ta4["R_m"])
    comp4 = compare_situation_reward_candidates(
        ego_pos4, ego_vel4, ego_rpy4, tgt_pos4, tgt_vel4, tgt_rpy4)
    assert abs(current_direct - comp4["current_2d_ao_ta"]) < 1e-9

    print("situation geometry candidates smoke test passed")


if __name__ == "__main__":
    main()
