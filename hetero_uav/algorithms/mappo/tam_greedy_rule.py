"""Deterministic TAM-style greedy maneuver opponent.

This is a paper-aligned protocol implementation, not an exact reproduction:
the paper does not publish every candidate magnitude or score weight.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


PROTOCOL_VERSION = "tam_greedy_rule_v2_paper_weighted"
RULE_CLAIM = "paper_aligned_not_exact_reproduction"
CANDIDATE_MANEUVERS = (
    "level_hold", "climb", "descend", "turn_left", "turn_right",
    "hard_turn_left", "hard_turn_right", "accelerate", "decelerate",
    "pursue_current_target", "break_left", "break_right",
    "return_center", "hard_deck_recovery",
)
SCORE_WEIGHTS = {
    "height": 10.0 / 75.0,
    "speed": 10.0 / 75.0,
    "angle": 15.0 / 75.0,
    "distance": 10.0 / 75.0,
    "avoidance": 30.0 / 75.0,
}


def wrap_heading(value: float) -> float:
    return float((float(value) + 1.0) % 2.0 - 1.0)


@dataclass(frozen=True)
class Candidate:
    name: str
    pitch: float
    heading_delta: float
    speed_mps: float


class TamGreedyRule:
    """Fixed candidate-set, immediate-score blue rule at the env decision rate."""

    def __init__(self):
        self.targets: dict[str, int] = {}
        self.target_ages: dict[str, int] = {}
        self.selected_counts: dict[str, int] = {}
        self.target_switch_count = 0
        self.warning_break_count = 0
        self.boundary_safety_count = 0
        self.hard_deck_recovery_count = 0
        self.last_candidate_scores: dict[str, dict[str, dict[str, float]]] = {}

    def reset(self) -> None:
        self.targets.clear()
        self.target_ages.clear()
        self.selected_counts.clear()
        self.target_switch_count = 0
        self.warning_break_count = 0
        self.boundary_safety_count = 0
        self.hard_deck_recovery_count = 0
        self.last_candidate_scores.clear()

    @staticmethod
    def _visible_targets(obs: dict) -> list[tuple[int, float, float, float]]:
        rel = np.asarray(obs.get("enemy_relative_pos_xyz", []), dtype=np.float32)
        bearing = np.asarray(obs.get("enemy_bearing_elevation", []), dtype=np.float32)
        valid = np.asarray(obs.get("enemy_full_geo_valid_mask", []), dtype=np.float32).reshape(-1)
        observed = np.asarray(obs.get("enemy_observed_mask", []), dtype=np.float32).reshape(-1)
        alive = np.asarray(obs.get("enemy_alive_mask", []), dtype=np.float32).reshape(-1)
        count = max(rel.shape[0] if rel.ndim == 2 else 0, bearing.shape[0] if bearing.ndim == 2 else 0)
        out = []
        for idx in range(count):
            if idx >= observed.size or observed[idx] <= 0.5:
                continue
            if idx < alive.size and alive[idx] <= 0.5:
                continue
            if valid.size and (idx >= valid.size or valid[idx] <= 0.5):
                continue
            distance_norm = float(np.linalg.norm(rel[idx])) if rel.ndim == 2 and idx < rel.shape[0] else 1.0
            bearing_norm = float(bearing[idx, 0]) if bearing.ndim == 2 and idx < bearing.shape[0] else 0.0
            elevation_norm = float(bearing[idx, 1]) if bearing.ndim == 2 and idx < bearing.shape[0] else 0.0
            if np.isfinite([distance_norm, bearing_norm, elevation_norm]).all():
                out.append((idx, distance_norm, bearing_norm, elevation_norm))
        return out

    def assign_targets(self, observations: dict[str, dict], blue_ids: list[str], engaged_slots: set[int]) -> dict[str, int]:
        assignments: dict[str, int] = {}
        used: set[int] = set()
        for bid in blue_ids:
            visible = self._visible_targets(observations.get(bid, {}))
            visible_ids = {item[0] for item in visible}
            old = self.targets.get(bid)
            age = self.target_ages.get(bid, 0)
            if old in visible_ids and old not in used and old not in engaged_slots and age < 15:
                chosen = old
            else:
                ranked = sorted(visible, key=lambda item: (item[0] in engaged_slots, item[0] in used, item[1], item[0]))
                chosen = ranked[0][0] if ranked else None
            if chosen is None:
                self.targets.pop(bid, None)
                self.target_ages.pop(bid, None)
                continue
            if old is not None and old != chosen:
                self.target_switch_count += 1
            assignments[bid] = chosen
            used.add(chosen)
            self.targets[bid] = chosen
            self.target_ages[bid] = age + 1 if old == chosen else 0
        return assignments

    @staticmethod
    def _candidates(current_heading: float, target_heading: float, target_elevation: float) -> list[Candidate]:
        pitch_to_target = float(np.clip(target_elevation, -0.18, 0.18))
        return [
            Candidate("level_hold", 0.0, 0.0, 270.0),
            Candidate("climb", 0.12, 0.0, 270.0),
            Candidate("descend", -0.10, 0.0, 270.0),
            Candidate("turn_left", 0.0, -10.0 / 180.0, 270.0),
            Candidate("turn_right", 0.0, 10.0 / 180.0, 270.0),
            Candidate("hard_turn_left", 0.02, -25.0 / 180.0, 285.0),
            Candidate("hard_turn_right", 0.02, 25.0 / 180.0, 285.0),
            Candidate("accelerate", 0.0, 0.0, 330.0),
            Candidate("decelerate", 0.0, 0.0, 220.0),
            Candidate("pursue_current_target", pitch_to_target, wrap_heading(target_heading - current_heading), 300.0),
            Candidate("break_left", 0.08, -25.0 / 180.0, 320.0),
            Candidate("break_right", 0.08, 25.0 / 180.0, 320.0),
        ]

    def action(self, bid: str, obs: dict, ownship: dict, target_slot: int | None,
               velocity_min: float, velocity_max: float) -> tuple[np.ndarray, str]:
        ego = np.asarray(obs.get("ego_geo_state", []), dtype=np.float32).reshape(-1)
        current_heading = float(ego[5]) if ego.size >= 6 and np.isfinite(ego[5]) else float(ownship.get("heading_norm", 0.0))
        altitude_norm = float(ego[2]) if ego.size >= 3 and np.isfinite(ego[2]) else 0.6
        current_speed_mps = float(ego[3] * 600.0) if ego.size >= 4 and np.isfinite(ego[3]) else 270.0
        warning = float(np.asarray(obs.get("missile_warning", 0.0)).reshape(-1)[0]) if np.asarray(obs.get("missile_warning", 0.0)).size else 0.0
        target_heading = current_heading
        target_elevation = 0.0
        distance_norm = 1.0
        for idx, dist, bearing, elevation in self._visible_targets(obs):
            if idx == target_slot:
                target_heading, target_elevation, distance_norm = bearing, elevation, dist
                break
        position = np.asarray(ownship.get("position", []), dtype=np.float32).reshape(-1)
        boundary_pressure = float(np.linalg.norm(position[:2]) / 40000.0) if position.size >= 2 else 0.0
        center_heading = current_heading
        if position.size >= 2 and np.linalg.norm(position[:2]) > 1.0:
            center_heading = float(math.atan2(-float(position[1]), -float(position[0])) / math.pi)

        best = None
        best_score = -float("inf")
        candidate_scores: dict[str, dict[str, float]] = {}
        candidates = self._candidates(current_heading, target_heading, target_elevation)
        if boundary_pressure > 0.8:
            candidates.append(Candidate(
                "return_center", 0.04, wrap_heading(center_heading - current_heading), 285.0
            ))
        if altitude_norm < 0.28:
            candidates.append(Candidate("hard_deck_recovery", 0.18, 0.0, 300.0))
        for candidate in candidates:
            heading = wrap_heading(current_heading + candidate.heading_delta)
            if candidate.name == "pursue_current_target":
                heading = wrap_heading(target_heading)
            angle_error = abs(wrap_heading(target_heading - heading))
            pitch_rad = float(candidate.pitch) * math.pi / 2.0
            predicted_altitude_m = altitude_norm * 10000.0 + current_speed_mps * math.sin(pitch_rad) * 0.2
            lower_margin = (predicted_altitude_m - 2500.0) / 3500.0
            upper_margin = (10000.0 - predicted_altitude_m) / 4000.0
            height_term = float(np.clip(min(lower_margin, upper_margin), -1.0, 1.0))
            predicted_speed_mps = current_speed_mps + 0.25 * (candidate.speed_mps - current_speed_mps)
            speed_term = float(np.clip(1.0 - abs(predicted_speed_mps - 275.0) / 125.0, -1.0, 1.0))
            angle_term = (1.0 - min(angle_error, 1.0)) if target_slot is not None else float(candidate.name == "level_hold")
            if target_slot is not None:
                range_m = distance_norm * 40000.0
                closure_m = predicted_speed_mps * math.cos(math.pi * angle_error) * 0.2
                distance_sign = -1.0 if range_m < 3000.0 else 1.0
                distance_term = float(np.clip(distance_sign * closure_m / 70.0, -1.0, 1.0))
            else:
                distance_term = 0.0
            avoidance_term = 1.0 if warning > 0.5 and candidate.name.startswith("break_") else 0.0
            score = (
                SCORE_WEIGHTS["height"] * height_term
                + SCORE_WEIGHTS["speed"] * speed_term
                + SCORE_WEIGHTS["angle"] * angle_term
                + SCORE_WEIGHTS["distance"] * distance_term
                + SCORE_WEIGHTS["avoidance"] * avoidance_term
            )
            if altitude_norm < 0.28 and candidate.pitch <= 0.05:
                score = -1e6
            if boundary_pressure > 0.95 and abs(wrap_heading(center_heading - heading)) > 0.25:
                score = -1e6
            candidate_scores[candidate.name] = {
                "height": height_term, "speed": speed_term,
                "angle": angle_term, "distance": distance_term,
                "avoidance": avoidance_term, "total": float(score),
            }
            if best is None or score > best_score:
                best, best_score = candidate, score
        assert best is not None
        heading = wrap_heading(target_heading if best.name == "pursue_current_target" else current_heading + best.heading_delta)
        if altitude_norm < 0.28:
            self.hard_deck_recovery_count += 1
        if boundary_pressure > 0.8:
            self.boundary_safety_count += 1
        if warning > 0.5 and best.name.startswith("break_"):
            self.warning_break_count += 1
        self.selected_counts[best.name] = self.selected_counts.get(best.name, 0) + 1
        self.last_candidate_scores[bid] = candidate_scores
        speed = 2.0 * (best.speed_mps - velocity_min) / max(velocity_max - velocity_min, 1e-6) - 1.0
        action = np.clip(np.asarray([best.pitch, heading, speed], dtype=np.float32), -1.0, 1.0)
        return action.astype(np.float32), best.name
