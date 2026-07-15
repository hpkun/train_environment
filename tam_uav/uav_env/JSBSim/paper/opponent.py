"""Observation-consistent greedy opponent using formal UAV dense reward."""

from __future__ import annotations

import numpy as np

from .reward import (paper_height_reward, uav_angle_reward, uav_distance_reward,
                     uav_speed_reward)
from .situation import assess_pair


MANOEUVRES = {
    "level": (24, 20, 20, 20),
    "accelerate_level": (34, 20, 20, 20),
    "left_turn": (27, 10, 20, 14),
    "right_turn": (27, 29, 20, 25),
    "climb": (27, 20, 14, 20),
    "dive": (24, 20, 25, 20),
    "left_climb": (27, 10, 14, 14),
    "right_climb": (27, 29, 14, 25),
    "left_dive": (25, 10, 25, 14),
    "right_dive": (25, 29, 25, 25),
}


def map_indices(indices) -> np.ndarray:
    values = np.asarray(indices, dtype=np.float64)
    return np.array([0.4 + values[0] / 39.0 * 0.5,
                     -1.0 + values[1] / 39.0 * 2.0,
                     -1.0 + values[2] / 39.0 * 2.0,
                     -1.0 + values[3] / 39.0 * 2.0])


class GreedyPaperOpponent:
    """Argmax over ten 0.2 s predictions scored by the formal dense reward."""

    def __init__(self, published: dict, inferred: dict):
        self.published = published
        self.inferred = inferred

    def act(self, agent, current_target, incoming_missiles) -> tuple[np.ndarray, dict]:
        if not agent.alive or current_target is None or not current_target.alive:
            action = np.asarray(MANOEUVRES["level"], dtype=np.int64)
            return action, {"current_target": None, "candidates": {},
                            "manoeuvre": "level", "action_indices": action.tolist()}
        candidates = {}
        decision_dt = (float(self.published["physics_frames_per_action"])
                       / float(self.published["simulation_frequency_hz"]))
        predicted_target_position = (np.asarray(current_target.position, dtype=np.float64)
                                     + np.asarray(current_target.velocity, dtype=np.float64)
                                     * decision_dt)
        predicted_target_velocity = np.asarray(current_target.velocity, dtype=np.float64)
        for name, indices in MANOEUVRES.items():
            position, velocity, pitch, heading, speed = self._predict(agent, indices)
            pair = assess_pair(position, velocity, predicted_target_position,
                               predicted_target_velocity,
                               self.published["maximum_attack_range_m"],
                               self.inferred["situation_height_norm_m"],
                               self.published["maximum_speed_mps"])
            r_height = paper_height_reward(
                position[2], self.published["minimum_safe_altitude_m"],
                self.inferred["optimal_altitude_m"], self.inferred["maximum_altitude_m"])
            r_speed = uav_speed_reward(speed, current_target.speed)
            r_angle = uav_angle_reward(pair.ata_rad, pair.aa_rad)
            r_distance = uav_distance_reward(pair.distance_m)
            r_dodge = self._predicted_dodge(
                position, velocity, incoming_missiles, decision_dt)
            total = (10.0 * r_height + 10.0 * r_speed + 15.0 * r_angle
                     + 10.0 * r_distance + 30.0 * r_dodge)
            candidates[name] = {
                "predicted_position_m": position.tolist(),
                "predicted_velocity_mps": velocity.tolist(),
                "predicted_pitch_rad": pitch, "predicted_heading_rad": heading,
                "predicted_target_position_m": predicted_target_position.tolist(),
                "predicted_target_velocity_mps": predicted_target_velocity.tolist(),
                "prediction_time_s": decision_dt,
                "threat_prediction": "constant_velocity",
                "reward_components": {"r_height": r_height, "r_speed": r_speed,
                                      "r_angle": r_angle, "r_distance": r_distance,
                                      "r_dodge": r_dodge},
                "total_dense_reward": float(total),
                "action_indices": list(indices),
            }
        manoeuvre = max(MANOEUVRES, key=lambda name: (candidates[name]["total_dense_reward"],
                                                       -list(MANOEUVRES).index(name)))
        action = np.asarray(MANOEUVRES[manoeuvre], dtype=np.int64)
        return action, {"current_target": current_target.agent_id,
                        "candidates": candidates, "manoeuvre": manoeuvre,
                        "action_indices": action.tolist()}

    def _predict(self, agent, indices):
        throttle, aileron, elevator, rudder = map_indices(indices)
        dt = (float(self.published["physics_frames_per_action"])
              / float(self.published["simulation_frequency_hz"]))
        speed = max(1.0, agent.speed + (55.0 * (throttle - 0.4) - 10.0) * dt)
        roll = float(np.clip(agent.roll + aileron * np.deg2rad(70.0) * dt,
                             -np.deg2rad(75.0), np.deg2rad(75.0)))
        pitch = float(np.clip(agent.pitch - elevator * np.deg2rad(25.0) * dt,
                              -np.deg2rad(40.0), np.deg2rad(40.0)))
        heading = float((agent.heading + (np.sin(roll) * np.deg2rad(100.0)
                                          + rudder * np.deg2rad(40.0)) * dt
                         + np.pi) % (2.0 * np.pi) - np.pi)
        cp = np.cos(pitch)
        velocity = np.array([cp * np.cos(heading), cp * np.sin(heading), np.sin(pitch)]) * speed
        position = np.asarray(agent.position, dtype=np.float64) + velocity * dt
        return position, velocity, pitch, heading, float(speed)

    def _predicted_dodge(self, position, velocity, incoming_missiles, decision_dt):
        active = [m for m in incoming_missiles if m.alive]
        if not active:
            return 0.0
        threat = min(active, key=lambda m: np.linalg.norm(
            m.position + m.velocity * decision_dt - position)
                     / max(m.speed_mps, 1.0))
        predicted_threat_position = threat.position + threat.velocity * decision_dt
        los = position - predicted_threat_position
        denom = max(float(np.linalg.norm(los) * np.linalg.norm(threat.velocity)), 1e-8)
        lam = float(np.arccos(np.clip(np.dot(los, threat.velocity) / denom, -1.0, 1.0)))
        angle = -float(np.cos(lam))
        speed = ((threat.decision_start_speed_mps - threat.speed_mps)
                 / float(self.inferred["missile_speed_reward_norm_mps"]))
        del velocity
        return angle + speed
