"""Deterministic direct and MAV-shared sensing contract."""
from __future__ import annotations

import numpy as np


def red_track_sources(env, observer_id: str) -> dict[str, dict]:
    observer = env.aircraft[observer_id]
    mav = env.aircraft["red_0"]
    observer_range = env.mav_detection_range_m if observer_id == "red_0" else env.uav_detection_range_m
    tracks = {}
    for target_id in env.blue_ids:
        target = env.aircraft[target_id]
        distance = float(np.linalg.norm(target.get_position() - observer.get_position()))
        direct = bool(observer.is_alive and target.is_alive and distance <= observer_range)
        mav_distance = float(np.linalg.norm(target.get_position() - mav.get_position()))
        shared = bool(
            observer_id != "red_0" and observer.is_alive and target.is_alive
            and mav.is_alive and mav_distance <= env.mav_detection_range_m
        )
        tracks[target_id] = {
            "observable": direct or shared,
            "direct": direct,
            "mav_shared": shared,
            "source": "direct_and_mav_shared" if direct and shared else (
                "direct" if direct else ("mav_shared" if shared else "unobserved")),
        }
    return tracks
