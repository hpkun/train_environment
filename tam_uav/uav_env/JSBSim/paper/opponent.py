"""Observation-consistent greedy opponent using formal UAV dense reward."""

from __future__ import annotations

import numpy as np

from .reward import (unpublished_height_reward_approximation, uav_angle_reward, uav_distance_reward,
                     uav_speed_reward)
from .situation import assess_pair
from .protocol import environment_values
from .action_semantics import map_action_indices


MANOEUVRES = {
    # Level approximation using nearest-positive-center control surfaces.
    "level": (24, 20, 20, 20),
    "accelerate": (34, 20, 20, 20),
    "decelerate": (14, 20, 20, 20),
    "left_turn": (27, 10, 20, 14),
    "right_turn": (27, 29, 20, 25),
    "climb": (27, 20, 14, 20),
    "dive": (24, 20, 25, 20),
}


def map_indices(indices) -> np.ndarray:
    return map_action_indices(indices)


class GreedyPaperOpponent:
    """Minimal greedy basic-manoeuvre reconstruction with paper-silent prediction."""

    def __init__(self, published: dict, unpublished: dict):
        self.published = published
        self.unpublished = unpublished
        self.derived = environment_values(unpublished)

    def act(self, agent, current_target, incoming_missiles) -> tuple[np.ndarray, dict]:
        if not agent.alive or current_target is None or not current_target.alive:
            action = np.asarray(MANOEUVRES["level"], dtype=np.int64)
            return action, {"current_target": None, "candidates": {},
                            "manoeuvre": "level", "action_indices": action.tolist()}
        candidates = {}
        decision_dt = (float(self.published["physics_frames_per_action"])
                       / float(self.published["simulation_frequency_hz"]))
        for name, indices in MANOEUVRES.items():
            clone = agent.clone_for_prediction()
            try:
                command = map_indices(indices)
                clone.apply_direct_fcs_command(command)
                for _ in range(int(self.published["physics_frames_per_action"])):
                    clone.step_physics_once(1.0 / float(self.published["simulation_frequency_hz"]))
                predicted_target_position = (
                    np.asarray(current_target.position, dtype=np.float64)
                    + np.asarray(current_target.velocity, dtype=np.float64) * decision_dt)
                predicted_target_velocity = np.asarray(current_target.velocity, dtype=np.float64)
                pair = assess_pair(
                    clone.position, clone.velocity, predicted_target_position,
                    predicted_target_velocity, self.published["maximum_attack_range_m"],
                    self.derived["situation_height_norm_m"],
                    self.published["maximum_speed_mps"])
                r_height = unpublished_height_reward_approximation(
                    clone.position[2], self.published["minimum_safe_altitude_m"],
                    self.unpublished["height_reward_approximation"]["nominal_altitude_m"])
                r_speed = uav_speed_reward(clone.speed, current_target.speed)
                r_angle = uav_angle_reward(pair.ata_rad, pair.aa_rad)
                r_distance = uav_distance_reward(pair.distance_m)
                r_dodge = self._predicted_dodge(
                    clone.position, clone.velocity, incoming_missiles, decision_dt)
                weights = self.published["uav_reward_weights"]
                total = (weights["height"] * r_height + weights["speed"] * r_speed
                         + weights["angle"] * r_angle + weights["distance"] * r_distance
                         + weights["dodge"] * r_dodge)
                candidates[name] = {
                    "predicted_position_m": clone.position.tolist(),
                    "predicted_velocity_mps": clone.velocity.tolist(),
                    "predicted_pitch_rad": clone.pitch,
                    "predicted_heading_rad": clone.heading,
                    "predicted_target_position_m": predicted_target_position.tolist(),
                    "predicted_target_velocity_mps": predicted_target_velocity.tolist(),
                    "prediction_time_s": decision_dt,
                    "candidate_backend": (
                        "independent_jsbsim_clone_12_frames"
                        if agent.__class__.__name__ == "JSBSimAircraftPlatform"
                        else "simple_backend_test_clone_12_frames"),
                    "threat_prediction": "constant_velocity_unpublished",
                    "reward_components": {"r_height_approximation": r_height,
                                          "r_speed": r_speed, "r_angle": r_angle,
                                          "r_distance": r_distance, "r_dodge": r_dodge},
                    "total_dense_reward": float(total),
                    "action_indices": list(indices),
                }
            finally:
                clone.close()
        manoeuvre = max(MANOEUVRES, key=lambda name: (candidates[name]["total_dense_reward"],
                                                       -list(MANOEUVRES).index(name)))
        action = np.asarray(MANOEUVRES[manoeuvre], dtype=np.int64)
        return action, {"current_target": current_target.agent_id,
                        "candidates": candidates, "manoeuvre": manoeuvre,
                        "action_indices": action.tolist()}

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
                 / float(self.unpublished["missile_speed_reward_norm_mps"]))
        del velocity
        return angle + speed
