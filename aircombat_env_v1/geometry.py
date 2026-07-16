"""Coordinate conversions and the paper's direction-error geometry."""

from __future__ import annotations

import numpy as np

try:
    import pymap3d
except ImportError:  # pragma: no cover - exercised only without optional dependency
    pymap3d = None


def _require_pymap3d() -> None:
    if pymap3d is None:
        raise ImportError("pymap3d is required for LLA/NEU conversion")


def LLA2NEU(lon, lat, alt, lon0=120.0, lat0=60.0, alt0=0.0):
    """Convert geodetic longitude/latitude/altitude to local NEU metres."""
    _require_pymap3d()
    north, east, down = pymap3d.geodetic2ned(lat, lon, alt, lat0, lon0, alt0)
    return np.array([north, east, -down], dtype=float)


def NEU2LLA(north, east, up, lon0=120.0, lat0=60.0, alt0=0.0):
    """Convert local NEU metres to longitude/latitude/altitude."""
    _require_pymap3d()
    lat, lon, alt = pymap3d.ned2geodetic(north, east, -up, lat0, lon0, alt0)
    return np.array([lon, lat, alt], dtype=float)


def in_range_rad(angle):
    """Normalize an angle to (-pi, pi]."""
    wrapped = angle % (2.0 * np.pi)
    return wrapped - 2.0 * np.pi if wrapped > np.pi else wrapped


def body_to_ned_matrix(roll, pitch, heading):
    """Return the aerospace Z-Y-X body-to-NED rotation matrix."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    ch, sh = np.cos(heading), np.sin(heading)
    return np.array([
        [ch * cp, ch * sp * sr - sh * cr, ch * sp * cr + sh * sr],
        [sh * cp, sh * sp * sr + ch * cr, sh * sp * cr - ch * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=float)


def ned_to_body_matrix(roll, pitch, heading):
    """Return the NED-to-body rotation matrix."""
    return body_to_ned_matrix(roll, pitch, heading).T


def paper_direction_errors(current_roll, current_pitch, current_heading,
                           target_pitch, target_heading):
    """Compute BRMA-MAPPO Section 2.4 roll and pitch direction errors.

    The paper defines ``d_des_I`` in a z-up inertial frame. This project uses
    NED (z down), so its third component changes from ``+sin(theta_cmd)`` to
    ``-sin(theta_cmd)``. After NED-to-body rotation, aerospace body z is also
    down; it is negated back to ``d_body_z_up`` before the paper's pitch
    ``atan2`` is evaluated.
    """
    c_pitch = np.cos(target_pitch)
    desired_ned = np.array([
        c_pitch * np.cos(target_heading),
        c_pitch * np.sin(target_heading),
        -np.sin(target_pitch),
    ], dtype=float)
    desired_body_down = (
        ned_to_body_matrix(current_roll, current_pitch, current_heading)
        @ desired_ned
    )
    x_body, y_body, z_body_down = desired_body_down
    e_roll = float(np.arctan2(y_body, x_body))
    e_pitch = float(np.arctan2(-z_body_down, x_body))
    return e_roll, e_pitch
