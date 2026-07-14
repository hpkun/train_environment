"""Chen-Luo-Guo paper role rewards without legacy overlays."""

from __future__ import annotations

import math

import numpy as np

from .situation import SituationScore


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


def paper_height_reward(altitude_m: float, minimum_safe_altitude_m: float,
                        optimal_altitude_m: float, maximum_altitude_m: float) -> float:
    """Continuous paper_undefined_height_approximation in [-1, 1]."""
    altitude = float(altitude_m)
    if altitude <= minimum_safe_altitude_m:
        return float(np.clip(-1.0 - (minimum_safe_altitude_m - altitude)
                             / max(minimum_safe_altitude_m, 1.0), -2.0, -1.0))
    span_low = max(optimal_altitude_m - minimum_safe_altitude_m, 1.0)
    span_high = max(maximum_altitude_m - optimal_altitude_m, 1.0)
    if altitude <= optimal_altitude_m:
        return float(1.0 - ((optimal_altitude_m - altitude) / span_low) ** 2)
    return float(np.clip(1.0 - ((altitude - optimal_altitude_m) / span_high) ** 2, -1.0, 1.0))


class PaperReward:
    def __init__(self, published: dict, inferred: dict):
        self.published = published
        self.inferred = inferred
        self.global_scale = float(inferred.get("reward_global_scale", 1.0))
        self._dead_seen: set[str] = set()
        self._team_kill_bonus_paid: dict[str, float] = {}

    def reset(self):
        self._dead_seen.clear()
        self._team_kill_bonus_paid.clear()

    def compute(self, agents, targets: dict[str, str | None], scores: dict,
                missiles, step_events: list[dict], out_of_zone_step: set[str]):
        by_id = {a.agent_id: a for a in agents}
        kill_by_side = {"red": 0, "blue": 0}
        killed_by: dict[str, int] = {}
        for event in step_events:
            if event.get("reason") == "hit":
                shooter = by_id.get(event.get("shooter_id"))
                if shooter is not None:
                    kill_by_side[shooter.side] += 1
                    killed_by[shooter.agent_id] = killed_by.get(shooter.agent_id, 0) + 1
        rewards, components = {}, {}
        for agent in agents:
            if agent.aircraft_type.role == "mav":
                total, comp = self._mav(agent, agents, missiles, kill_by_side[agent.side])
            else:
                target = by_id.get(targets.get(agent.agent_id))
                pair = scores.get(agent.agent_id, {}).get(getattr(target, "agent_id", None))
                total, comp = self._uav(agent, target, pair, missiles,
                                        killed_by.get(agent.agent_id, 0),
                                        agent.agent_id in out_of_zone_step)
            rewards[agent.agent_id] = float(total * self.global_scale)
            comp["total"] = rewards[agent.agent_id]
            components[agent.agent_id] = comp
            if not agent.alive:
                self._dead_seen.add(agent.agent_id)
        return rewards, components

    def _mav(self, mav, agents, missiles, team_kills: int):
        enemies = [a for a in agents if a.side != mav.side and a.alive]
        distances = [float(np.linalg.norm(a.position - mav.position)) for a in enemies]
        nearest = min(distances, default=float(self.inferred["mav_d_max_m"]))
        danger, safe = float(self.inferred["mav_d_danger_m"]), float(self.inferred["mav_d_safe_m"])
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
            db = float(self.inferred["mav_d_max_m"])
        opt, max_d = float(self.inferred["mav_d_opt_m"]), float(self.inferred["mav_d_max_m"])
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
        death = (-float(self.inferred["mav_death_penalty"])
                 if not mav.alive and mav.agent_id not in self._dead_seen else 0.0)
        paid = self._team_kill_bonus_paid.get(mav.agent_id, 0.0)
        cap = float(self.inferred["mav_team_kill_bonus_cap"])
        bonus = min(cap, paid + team_kills * float(self.inferred["mav_team_kill_bonus"])) - paid
        self._team_kill_bonus_paid[mav.agent_id] = paid + bonus
        event = death + bonus
        return r_safety + r_support + event, {
            "r_dist": r_dist, "r_threat": r_threat, "r_aspect": r_aspect,
            "r_safety": r_safety, "r_pos": r_pos, "r_aware": r_aware,
            "r_support": r_support, "r_event": event,
        }

    def _uav(self, agent, target, pair: SituationScore | None, missiles,
             kills: int, out_of_zone: bool):
        r_height = paper_height_reward(agent.position[2],
                                       self.published["minimum_safe_altitude_m"],
                                       self.inferred["optimal_altitude_m"],
                                       self.inferred["maximum_altitude_m"])
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
            r_dodge_speed = ((threat.previous_speed_mps - threat.speed_mps)
                             / float(self.inferred["missile_speed_reward_norm_mps"]))
        r_dodge = r_dodge_angle + r_dodge_speed
        event = 200.0 * kills
        if (not agent.alive and not agent.out_of_boundary
                and agent.agent_id not in self._dead_seen):
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
