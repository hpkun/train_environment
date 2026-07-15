"""TAM-HAPPO Table 6 fixed 3v2 scenario."""
from __future__ import annotations

FT_PER_M = 1.0 / 0.3048

ROLES = {
    "red_0": "mav", "red_1": "attack_uav", "red_2": "attack_uav",
    "blue_0": "attack_uav", "blue_1": "attack_uav",
}
MISSILE_COUNTS = {"red_0": 0, "red_1": 2, "red_2": 2, "blue_0": 2, "blue_1": 2}
TABLE6_INITIAL_STATES = {
    "red_0": (120.02, 59.98, 6000.0, 250.0, 0.0),
    "red_1": (120.00, 60.00, 6000.0, 250.0, 0.0),
    "red_2": (120.04, 60.00, 6000.0, 250.0, 0.0),
    "blue_0": (120.00, 60.20, 6000.0, 250.0, 180.0),
    "blue_1": (120.04, 60.20, 6000.0, 250.0, 180.0),
}


def jsbsim_initial_state(agent_id: str, perturbation: dict | None = None) -> dict:
    lon, lat, altitude_m, speed_mps, yaw_deg = TABLE6_INITIAL_STATES[agent_id]
    delta = (perturbation or {}).get(agent_id, {})
    lon += float(delta.get("lon_deg", 0.0)); lat += float(delta.get("lat_deg", 0.0))
    altitude_m += float(delta.get("altitude_m", 0.0))
    speed_mps += float(delta.get("speed_mps", 0.0)); yaw_deg += float(delta.get("yaw_deg", 0.0))
    return {
        "ic/long-gc-deg": lon,
        "ic/lat-geod-deg": lat,
        "ic/h-sl-ft": altitude_m * FT_PER_M,
        "ic/psi-true-deg": yaw_deg,
        "ic/u-fps": speed_mps * FT_PER_M,
        "ic/v-fps": 0.0,
        "ic/w-fps": 0.0,
    }
