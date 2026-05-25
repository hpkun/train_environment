"""Pure ACMI battlefield-boundary debug marker helpers."""
from __future__ import annotations

import math


def _neu_to_lla_approx(
    north_m: float,
    east_m: float,
    up_m: float,
    lon0: float = 120.0,
    lat0: float = 60.0,
    alt0: float = 0.0,
) -> tuple[float, float, float]:
    """Approximate local NEU meters to lon/lat/alt for debug markers."""

    lat = lat0 + north_m / 111320.0
    lon_scale = 111320.0 * max(math.cos(math.radians(lat0)), 1e-6)
    lon = lon0 + east_m / lon_scale
    alt = alt0 + up_m
    return lon, lat, alt


def battlefield_boundary_acmi_lines(
    half_size_m: float,
    altitude_m: float = 0.0,
) -> list[str]:
    """Return static Tacview marker lines for battlefield boundary corners.

    This deliberately uses corner markers instead of a polyline because the
    project writer currently only emits object transform lines.
    """

    half = float(half_size_m)
    corners = [
        ("SW", -half, -half),
        ("NW", half, -half),
        ("NE", half, half),
        ("SE", -half, half),
    ]
    lines: list[str] = []
    for idx, (label, north_m, east_m) in enumerate(corners):
        lon, lat, alt = _neu_to_lla_approx(north_m, east_m, altitude_m)
        lines.append(
            f"{800000 + idx},T={lon:.6f}|{lat:.6f}|{alt:.1f}|0|0|0,"
            f"Type=Misc+Waypoint,Name=Battlefield Boundary {label} "
            f"({half:.0f}m),Color=Orange"
        )
    return lines


def write_battlefield_boundary_acmi(
    f,
    half_size_m: float,
    altitude_m: float = 0.0,
) -> None:
    """Write battlefield boundary debug markers to a file-like object."""

    for line in battlefield_boundary_acmi_lines(half_size_m, altitude_m):
        f.write(line + "\n")


def maybe_write_battlefield_boundary_acmi(
    f,
    draw_boundary: bool,
    half_size_m: float,
    altitude_m: float = 0.0,
) -> None:
    """Opt-in wrapper for boundary debug marker output."""

    if draw_boundary:
        write_battlefield_boundary_acmi(f, half_size_m, altitude_m)
