from __future__ import annotations

import numpy as np


LAUNCH_QUALITY_FIELDS = (
    "team",
    "shooter_id",
    "target_id",
    "missile_id",
    "current_step",
    "physics_frame",
    "range_m",
    "AO_rad",
    "AO_deg",
    "TA_rad",
    "TA_deg",
    "relative_distance_3d_m",
    "horizontal_range_m",
    "altitude_diff_m",
    "shooter_speed_mps",
    "target_speed_mps",
    "closing_speed_mps",
    "shooter_alt_m",
    "target_alt_m",
    "target_alive_at_launch",
    "termination_reason",
    "is_success",
    "flight_time_sec",
    "launch_step",
    "termination_step",
    "step_delta",
    "target_alive_at_termination",
)


def nan_float() -> float:
    return float("nan")


def make_launch_quality_record(
    *,
    team: str,
    shooter_id: str,
    target_id: str,
    current_step: int,
    physics_frame: int | None,
    range_m: float,
    AO_rad: float,
    TA_rad: float,
    shooter_pos,
    shooter_vel,
    target_pos,
    target_vel,
    target_alive_at_launch: bool,
) -> dict:
    record = {field: "" for field in LAUNCH_QUALITY_FIELDS}
    record.update({
        "team": team,
        "shooter_id": shooter_id,
        "target_id": target_id,
        "missile_id": "",
        "current_step": int(current_step),
        "physics_frame": int(physics_frame) if physics_frame is not None else "",
        "range_m": float(range_m),
        "AO_rad": float(AO_rad),
        "AO_deg": float(np.rad2deg(AO_rad)),
        "TA_rad": float(TA_rad),
        "TA_deg": float(np.rad2deg(TA_rad)),
        "target_alive_at_launch": bool(target_alive_at_launch),
        "termination_reason": "",
        "is_success": False,
        "flight_time_sec": nan_float(),
        "launch_step": int(current_step),
        "termination_step": "",
        "step_delta": "",
        "target_alive_at_termination": "",
    })

    try:
        sp = np.asarray(shooter_pos, dtype=np.float64)
        tp = np.asarray(target_pos, dtype=np.float64)
        sv = np.asarray(shooter_vel, dtype=np.float64)
        tv = np.asarray(target_vel, dtype=np.float64)
        delta = tp - sp
        dist3d = float(np.linalg.norm(delta))
        horizontal = float(np.linalg.norm(delta[:2]))
        los_unit = delta / max(dist3d, 1e-8)
        rel_vel = tv - sv
        record.update({
            "relative_distance_3d_m": dist3d,
            "horizontal_range_m": horizontal,
            "altitude_diff_m": float(tp[2] - sp[2]),
            "shooter_speed_mps": float(np.linalg.norm(sv)),
            "target_speed_mps": float(np.linalg.norm(tv)),
            "closing_speed_mps": float(-np.sum(rel_vel * los_unit)),
            "shooter_alt_m": float(sp[2]),
            "target_alt_m": float(tp[2]),
        })
    except Exception:
        for key in (
            "relative_distance_3d_m",
            "horizontal_range_m",
            "altitude_diff_m",
            "shooter_speed_mps",
            "target_speed_mps",
            "closing_speed_mps",
            "shooter_alt_m",
            "target_alt_m",
        ):
            record[key] = nan_float()

    return record
