"""Paper-structured UAV reward adapted to the current 1v1 abstraction."""

from __future__ import annotations

from .opponent import angle_reward, distance_reward, height_reward, speed_reward


EVENT_REWARDS = {
    "red_hit": 200.0,
    "blue_hit": -200.0,
    "red_crash": -200.0,
    "draw_simultaneous_hit": 0.0,
    "draw_both_crash": 0.0,
    "timeout": 0.0,
    "blue_crash": 0.0,
    "red_numerical_invalid": 0.0,
    "blue_numerical_invalid": 0.0,
    "draw_both_numerical_invalid": 0.0,
    "physics_exception": 0.0,
}

REWARD_GLOBAL_SCALE = 1.0  # official TAM protocol project configuration


def reward_components(red_state, blue_state, geometry, r_dodge=0.0):
    return {
        "r_height": height_reward(red_state["altitude"]),
        "r_speed": speed_reward(
            red_state["true_airspeed"], blue_state["true_airspeed"]),
        "r_angle": angle_reward(
            geometry["ata_rad"], geometry["aa_rad"]),
        "r_distance": distance_reward(geometry["distance_m"]),
        "r_dodge": float(r_dodge),
    }


def dense_reward(red_state, blue_state, geometry, r_dodge=0.0):
    components = reward_components(red_state, blue_state, geometry, r_dodge)
    value = (
        10.0 * components["r_height"]
        + 10.0 * components["r_speed"]
        + 15.0 * components["r_angle"]
        + 10.0 * components["r_distance"]
        + 30.0 * components["r_dodge"]
    ) * REWARD_GLOBAL_SCALE
    return float(value), components


def terminal_reward(event):
    return float(EVENT_REWARDS.get(event, 0.0))


def step_reward(red_state, blue_state, geometry, event=None, r_dodge=0.0):
    dense, components = dense_reward(red_state, blue_state, geometry, r_dodge)
    if event in {
            "blue_crash", "red_numerical_invalid", "blue_numerical_invalid",
            "draw_both_numerical_invalid", "physics_exception"}:
        return 0.0, components
    return float(dense + terminal_reward(event)), components
