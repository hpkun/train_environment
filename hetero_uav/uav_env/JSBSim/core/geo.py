"""Local Cartesian to geodetic helpers for JSBSim initialization."""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6378137.0


def local_to_lla(position, reference_lat: float, reference_lon: float,
                 reference_alt: float) -> tuple[float, float, float]:
    """Convert local [north/east/up] meters to lon/lat/alt."""

    north, east, up = float(position[0]), float(position[1]), float(position[2])
    lat_rad = np.deg2rad(reference_lat)
    d_lat = north / EARTH_RADIUS_M
    d_lon = east / (EARTH_RADIUS_M * max(np.cos(lat_rad), 1e-6))
    return (
        reference_lon + np.rad2deg(d_lon),
        reference_lat + np.rad2deg(d_lat),
        reference_alt + up,
    )


def lla_to_local(lon: float, lat: float, alt: float, reference_lat: float,
                 reference_lon: float, reference_alt: float):
    lat0 = np.deg2rad(reference_lat)
    north = (lat - reference_lat) * np.pi / 180.0 * EARTH_RADIUS_M
    east = (lon - reference_lon) * np.pi / 180.0 * EARTH_RADIUS_M * max(np.cos(lat0), 1e-6)
    up = alt - reference_alt
    return np.array([north, east, up], dtype=np.float32)
