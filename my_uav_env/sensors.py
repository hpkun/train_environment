"""Paper-profile radar, RCS interpolation and coarse sensor tracks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from my_uav_env.alignment.state_extractor import (
    _rotation_inertial_to_body,
    body_angles_from_neu_vector,
)


@dataclass(frozen=True)
class SensorTrack:
    source: str
    target_id: str
    position_estimate: np.ndarray | None
    timestamp: float
    age: float
    confidence: float
    velocity_available: bool
    relative_velocity_available: bool
    valid: bool


def bilinear_rcs_m2(azimuth_rad: float, elevation_rad: float,
                     azimuth_grid_deg, elevation_grid_deg, table_m2) -> float:
    """Interpolate a replaceable target-body aspect RCS table."""
    az_grid = np.asarray(azimuth_grid_deg, dtype=np.float64)
    el_grid = np.asarray(elevation_grid_deg, dtype=np.float64)
    table = np.asarray(table_m2, dtype=np.float64)
    if table.shape != (el_grid.size, az_grid.size):
        raise ValueError("RCS table shape must be (elevation, azimuth)")
    az = float(np.clip(np.rad2deg(azimuth_rad), az_grid[0], az_grid[-1]))
    el = float(np.clip(np.rad2deg(elevation_rad), el_grid[0], el_grid[-1]))
    ai = min(max(int(np.searchsorted(az_grid, az) - 1), 0), az_grid.size - 2)
    ei = min(max(int(np.searchsorted(el_grid, el) - 1), 0), el_grid.size - 2)
    at = (az - az_grid[ai]) / max(az_grid[ai + 1] - az_grid[ai], 1e-12)
    et = (el - el_grid[ei]) / max(el_grid[ei + 1] - el_grid[ei], 1e-12)
    low = table[ei, ai] * (1.0 - at) + table[ei, ai + 1] * at
    high = table[ei + 1, ai] * (1.0 - at) + table[ei + 1, ai + 1] * at
    return float(low * (1.0 - et) + high * et)


def target_body_aspect(observer_position, target_position, target_rpy):
    """Return illumination azimuth/elevation in the target body frame."""
    illumination_neu = (np.asarray(observer_position, dtype=np.float64)
                        - np.asarray(target_position, dtype=np.float64))
    roll, pitch, heading = (float(v) for v in target_rpy)
    body = _rotation_inertial_to_body(roll, pitch, heading) @ illumination_neu
    horizontal = float(np.hypot(body[0], body[1]))
    return float(np.arctan2(body[1], body[0])), float(np.arctan2(-body[2], horizontal))


def radar_diagnostic(observer, target, radar_cfg, rcs_cfg) -> dict:
    """Evaluate paper radar FOV and Rmax=K*RCS^(1/4)."""
    rel = np.asarray(target.get_position()) - np.asarray(observer.get_position())
    _body, elevation, azimuth, _ = body_angles_from_neu_vector(
        rel, *[float(v) for v in observer.get_rpy()])
    aspect_az, aspect_el = target_body_aspect(
        observer.get_position(), target.get_position(), target.get_rpy())
    rcs = bilinear_rcs_m2(
        aspect_az, aspect_el, rcs_cfg.azimuth_grid_deg.value,
        rcs_cfg.elevation_grid_deg.value, rcs_cfg.table_m2.value)
    rmax = float(rcs_cfg.range_constant.value * np.power(max(rcs, 0.0), 0.25))
    distance = float(np.linalg.norm(rel))
    fov = (radar_cfg.azimuth_min_rad.value <= azimuth <= radar_cfg.azimuth_max_rad.value
           and radar_cfg.elevation_min_rad.value <= elevation <= radar_cfg.elevation_max_rad.value)
    range_ok = distance <= rmax
    return {
        "target_azimuth_rad": float(azimuth),
        "target_elevation_rad": float(elevation),
        "target_aspect_azimuth_rad": aspect_az,
        "target_aspect_elevation_rad": aspect_el,
        "interpolated_rcs_m2": rcs,
        "radar_max_range_m": rmax,
        "detected_by_fov": bool(fov),
        "detected_by_range": bool(range_ok),
        "radar_detected": bool(fov and range_ok),
    }


def select_most_dangerous_missile(aircraft, missiles) -> tuple[object | None, dict | None]:
    """Select targeting live missile by minimum positive TTC, then distance."""
    rows = []
    target_pos = np.asarray(aircraft.get_position(), dtype=np.float64)
    target_vel = np.asarray(aircraft.get_velocity(), dtype=np.float64)
    for missile in missiles:
        if not missile.is_alive or missile.target_aircraft is not aircraft:
            continue
        rel = target_pos - np.asarray(missile.get_position(), dtype=np.float64)
        rel_vel = target_vel - np.asarray(missile.get_velocity(), dtype=np.float64)
        if not np.all(np.isfinite(rel)) or not np.all(np.isfinite(rel_vel)):
            continue
        distance = float(np.linalg.norm(rel))
        if not np.isfinite(distance) or distance <= 0.0:
            continue
        closing = float(-np.dot(rel, rel_vel) / distance)
        rv2 = float(np.dot(rel_vel, rel_vel))
        ttc = float(-np.dot(rel, rel_vel) / rv2) if rv2 > 1e-9 else float("inf")
        if (not np.isfinite(closing) or closing <= 0.0
                or not np.isfinite(ttc) or ttc < 0.0):
            continue
        bearing = float(np.arctan2(rel[1], rel[0]))
        elevation = float(np.arctan2(rel[2], max(np.hypot(rel[0], rel[1]), 1e-9)))
        diag = {"missile_id": missile.uid, "distance_m": distance,
                "relative_velocity": rel_vel.tolist(),
                "closing_speed_mps": closing,
                "time_to_closest_approach_s": ttc,
                "candidate_is_approaching": True,
                "incoming_bearing_rad": bearing,
                "incoming_elevation_rad": elevation}
        rows.append((missile, diag))
    if not rows:
        return None, None
    rows.sort(key=lambda row: (
        row[1]["time_to_closest_approach_s"],
        row[1]["distance_m"], row[0].uid))
    return rows[0]
