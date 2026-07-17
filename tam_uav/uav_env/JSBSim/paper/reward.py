"""Chen-Luo-Guo paper role rewards without legacy overlays."""

from __future__ import annotations

import math

import numpy as np

from .situation import SituationScore
from .protocol import NOMINAL_ALTITUDE_M, derived_environment_values


def uav_distance_reward(distance_m: float) -> float:
    distance_km = float(distance_m) / 1000.0
    if distance_km <= 5.0:
        return 1.0
    if distance_km < 10.0:
        return float(math.exp(-0.921 * (distance_km - 5.0)))
    return -1.0


def uav_speed_reward(ego_speed_mps: float, target_speed_mps: float) -> float:
    ego = max(float(ego_speed_mps), 1e-6)
    target = float(target_speed_mps)
    if target < 0.5 * ego:
        return 1.0
    if target <= 1.5 * ego:
        return float(2.0 - 2.0 * target / ego)
    return -1.0


def uav_angle_reward(ata_rad: float, aa_rad: float) -> float:
    return float(1.0 - (ata_rad + aa_rad) / np.pi)


def paper_height_reward(altitude_m: float, minimum_safe_altitude_m: float) -> float:
    """Paper-silent continuous simplification using 750 m and nominal 6000 m."""
    altitude = float(altitude_m)
    minimum = float(minimum_safe_altitude_m)
    if altitude <= 0.0:
        return -1.0
    if altitude < minimum:
        return float(-1.0 + altitude / minimum)
    return float(np.clip(
        (altitude - minimum) / (NOMINAL_ALTITUDE_M - minimum), 0.0, 1.0))


class PaperReward:
    def __init__(self, published: dict, inferred: dict):
        self.published = published
        self.inferred = inferred
        self.derived = derived_environment_values(published["maximum_attack_range_m"])
        self.global_scale = float(inferred.get("reward_global_scale", 1.0))

    def reset(self):
        pass

    def compute(self, agents, targets: dict[str, str | None], scores: dict,
                missiles, step_events: list[dict], out_of_zone_step: set[str],
                alive_at_step_start: dict[str, bool]):
        by_id = {a.agent_id: a for a in agents}
        hit_events_by_side = {"red": [], "blue": []}
        death_sequence = {
            event["agent_id"]: event.get("event_sequence_id", float("inf"))
            for event in step_events
            if event.get("event_type") == "aircraft_death"
        }
        killed_by: dict[str, int] = {}
        for event in step_events:
            if event.get("reason") == "hit":
                shooter = by_id.get(event.get("shooter_id"))
                if shooter is not None:
                    hit_events_by_side[shooter.side].append(event)
                    killed_by[shooter.agent_id] = killed_by.get(shooter.agent_id, 0) + 1
        rewards, components = {}, {}
        for agent in agents:
            if not alive_at_step_start.get(agent.agent_id, False):
                components[agent.agent_id] = self._zero_components(agent)
                rewards[agent.agent_id] = 0.0
                continue
            just_died = not agent.alive
            if agent.aircraft_type.role == "mav":
                mav_death_sequence = death_sequence.get(agent.agent_id, float("inf"))
                eligible_team_kills = sum(
                    event.get("event_sequence_id", -1) < mav_death_sequence
                    for event in hit_events_by_side[agent.side]
                )
                total, comp = self._mav(
                    agent, agents, missiles,
                    eligible_team_kills,
                    just_died)
            else:
                target = by_id.get(targets.get(agent.agent_id))
                pair = scores.get(agent.agent_id, {}).get(getattr(target, "agent_id", None))
                total, comp = self._uav(agent, target, pair, missiles,
                                        killed_by.get(agent.agent_id, 0),
                                        agent.agent_id in out_of_zone_step,
                                        just_died)
            rewards[agent.agent_id] = float(total * self.global_scale)
            comp["total"] = rewards[agent.agent_id]
            components[agent.agent_id] = comp
        return rewards, components

    @staticmethod
    def _zero_components(agent):
        if agent.aircraft_type.role == "mav":
            keys = ("r_dist", "r_threat", "r_aspect", "r_safety", "r_pos",
                    "r_aware", "r_support", "r_event", "total")
        else:
            keys = ("r_height", "r_speed", "r_angle", "r_distance",
                    "r_dodge_angle", "r_dodge_speed", "r_dodge", "r_event", "total")
        return {key: 0.0 for key in keys}

    def _mav(self, mav, agents, missiles, team_kills: int, just_died: bool):
        enemies = [a for a in agents if a.side != mav.side and a.alive]
        distances = [float(np.linalg.norm(a.position - mav.position)) for a in enemies]
        nearest = min(distances, default=float(self.derived["mav_d_max_m"]))
        danger, safe = float(self.derived["mav_d_danger_m"]), float(self.derived["mav_d_safe_m"])
        if nearest < danger:
            r_dist = -(1.0 - nearest / danger)
        elif nearest < safe:
            r_dist = -0.5 * (1.0 - (nearest - danger) / (safe - danger))
        else:
            r_dist = 0.2
        incoming = [m for m in missiles if m.target_id == mav.agent_id and m.alive]
        r_threat = -1.0 if incoming else 0.0
        r_aspect = 0.0
        for enemy in enemies:
            rel = mav.position - enemy.position
            denom = max(float(np.linalg.norm(rel) * np.linalg.norm(enemy.velocity)), 1e-8)
            ta = float(np.arccos(np.clip(np.dot(rel, enemy.velocity) / denom, -1.0, 1.0)))
            if ta < np.pi / 4.0:
                r_aspect -= 1.0 - ta / (np.pi / 4.0)
        r_safety = 0.5 * r_dist + 0.3 * r_threat + 0.2 * r_aspect
        friend_uavs = [a for a in agents if a.side == mav.side and a.aircraft_type.role != "mav" and a.alive]
        if friend_uavs:
            center = np.mean([a.position for a in friend_uavs], axis=0)
            db = float(np.linalg.norm(mav.position - center))
        else:
            db = float(self.derived["mav_d_max_m"])
        opt, max_d = float(self.derived["mav_d_opt_m"]), float(self.derived["mav_d_max_m"])
        if db < opt:
            r_pos = db / opt - 1.0
        elif db < max_d:
            r_pos = 1.0 - (db - opt) / (max_d - opt)
        else:
            r_pos = -0.5
        r_aware = 0.0
        for enemy in enemies:
            rel = enemy.position - mav.position
            denom = max(float(np.linalg.norm(rel) * np.linalg.norm(mav.velocity)), 1e-8)
            ao = float(np.arccos(np.clip(np.dot(rel, mav.velocity) / denom, -1.0, 1.0)))
            if ao < np.pi / 2.0:
                r_aware += 0.3 * (1.0 - ao / (np.pi / 2.0))
        r_support = 0.6 * r_pos + 0.4 * r_aware
        death = -200.0 if just_died else 0.0
        event = death + 200.0 * team_kills
        return r_safety + r_support + event, {
            "r_dist": r_dist, "r_threat": r_threat, "r_aspect": r_aspect,
            "r_safety": r_safety, "r_pos": r_pos, "r_aware": r_aware,
            "r_support": r_support, "r_event": event,
        }

    def _uav(self, agent, target, pair: SituationScore | None, missiles,
             kills: int, out_of_zone: bool, just_died: bool):
        r_height = paper_height_reward(
            agent.position[2], self.published["minimum_safe_altitude_m"])
        if target is not None and pair is not None:
            r_speed = uav_speed_reward(agent.speed, target.speed)
            r_angle = uav_angle_reward(pair.ata_rad, pair.aa_rad)
            r_distance = uav_distance_reward(pair.distance_m)
        else:
            r_speed = r_angle = r_distance = 0.0
        incoming = [m for m in missiles if m.target_id == agent.agent_id and m.alive]
        r_dodge_angle = r_dodge_speed = 0.0
        if incoming:
            threat = min(incoming, key=lambda m: np.linalg.norm(m.position - agent.position)
                         / max(m.speed_mps, 1.0))
            los = agent.position - threat.position
            denom = max(float(np.linalg.norm(los) * np.linalg.norm(threat.velocity)), 1e-8)
            lam = float(np.arccos(np.clip(np.dot(los, threat.velocity) / denom, -1.0, 1.0)))
            r_dodge_angle = -float(np.cos(lam))
            r_dodge_speed = ((threat.decision_start_speed_mps - threat.speed_mps)
                             / float(self.inferred["missile_speed_reward_norm_mps"]))
        r_dodge = r_dodge_angle + r_dodge_speed
        event = 200.0 * kills
        if just_died and not agent.out_of_boundary:
            event -= 200.0
        if out_of_zone:
            event -= 100.0
        total = 10.0 * r_height + 10.0 * r_speed + 15.0 * r_angle + 10.0 * r_distance + 30.0 * r_dodge + event
        return total, {
            "r_height": r_height, "r_speed": r_speed, "r_angle": r_angle,
            "r_distance": r_distance, "r_dodge_angle": r_dodge_angle,
            "r_dodge_speed": r_dodge_speed, "r_dodge": r_dodge,
            "r_event": event,
        }
