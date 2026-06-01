"""
Coordinate conversion and geometry utilities for the UAV combat environment.
"""
import os
import numpy as np
import pymap3d


def get_package_data_dir():
    """Return the absolute path to the package data/ directory."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def LLA2NEU(lon, lat, alt, lon0=120.0, lat0=60.0, alt0=0.0):
    """
    Convert geodetic (lon, lat, alt) to local NEU (North, East, Up) in meters.
    Observer at (lon0, lat0, alt0).
    """
    n, e, d = pymap3d.geodetic2ned(lat, lon, alt, lat0, lon0, alt0)
    return np.array([n, e, -d])


def NEU2LLA(n, e, u, lon0=120.0, lat0=60.0, alt0=0.0):
    """
    Convert local NEU (North, East, Up) in meters back to geodetic (lon, lat, alt).
    Observer at (lon0, lat0, alt0).
    """
    lat, lon, h = pymap3d.ned2geodetic(n, e, -u, lat0, lon0, alt0)
    return np.array([lon, lat, h])


def get2d_AO_TA_R(ego_feature, enm_feature, return_side=False):
    """
    Compute 2D (horizontal plane) Aspect Angle, Target Aspect, and range.

    Args:
        ego_feature: np.array [north, east, down, vn, ve, vd] for ego
        enm_feature: np.array [north, east, down, vn, ve, vd] for enemy/target
        return_side: if True, also return side_flag (-1=left, 0=center, 1=right)

    Returns:
        ego_AO (rad): angle between ego's velocity and the LOS to enemy [0, pi]
        ego_TA (rad): angle between enemy's velocity and LOS from enemy to ego [0, pi]
        R (m): horizontal distance between the two entities
        side_flag (optional): cross product sign
    """
    ego_x, ego_y = ego_feature[0], ego_feature[1]
    ego_vx, ego_vy = ego_feature[3], ego_feature[4]
    enm_x, enm_y = enm_feature[0], enm_feature[1]
    enm_vx, enm_vy = enm_feature[3], enm_feature[4]

    delta_x, delta_y = enm_x - ego_x, enm_y - ego_y
    R = np.linalg.norm([delta_x, delta_y])

    ego_v = np.linalg.norm([ego_vx, ego_vy])
    enm_v = np.linalg.norm([enm_vx, enm_vy])

    # Aspect Angle: angle between ego's velocity and LOS to enemy
    proj_dist_ego = delta_x * ego_vx + delta_y * ego_vy
    ego_AO = np.arccos(np.clip(proj_dist_ego / (R * ego_v + 1e-8), -1.0, 1.0))

    # Target Aspect: angle between enemy's velocity and LOS from enemy to ego
    proj_dist_enm = (-delta_x) * enm_vx + (-delta_y) * enm_vy
    ego_TA = np.arccos(np.clip(proj_dist_enm / (R * enm_v + 1e-8), -1.0, 1.0))

    if return_side:
        side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
        return ego_AO, ego_TA, R, side_flag
    return ego_AO, ego_TA, R


def in_range_rad(angle):
    """Normalize an angle in radians to (-pi, pi]."""
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle


def in_range_deg(angle):
    """Normalize an angle in degrees to (-180, 180]."""
    angle = angle % 360
    if angle > 180:
        angle -= 360
    return angle
