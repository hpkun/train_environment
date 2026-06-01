"""Info dictionary construction."""

from __future__ import annotations

from collections import Counter

from .aircraft import AircraftPlatform


class InfoBuilder:
    def build(self, agents: list[AircraftPlatform], step_count: int,
              episode_return: dict[str, float], win_flag: str | None,
              termination_reason: str | None = None, events: list | None = None) -> dict:
        events = events or []
        red = [a for a in agents if a.side == "red"]
        blue = [a for a in agents if a.side == "blue"]
        fired_events = [e for e in events if getattr(e, "fired", False)]
        hit_events = [e for e in fired_events if getattr(e, "hit", False)]
        reason_counts = Counter(getattr(e, "reason", "unknown") for e in events)
        winner = None
        if win_flag == "red_win":
            winner = "red"
        elif win_flag == "blue_win":
            winner = "blue"
        elif win_flag == "draw":
            winner = "draw"
        info = {
            "red_alive": sum(a.alive for a in red),
            "blue_alive": sum(a.alive for a in blue),
            "mav_alive": any(a.alive and a.aircraft_type.role == "mav" for a in red),
            "mav_survival": 1.0 if any(a.alive and a.aircraft_type.role == "mav" for a in red) else 0.0,
            "red_kills": sum(1 for a in blue if not a.alive),
            "blue_kills": sum(1 for a in red if not a.alive),
            "missile_left": {a.agent_id: a.missile_left for a in agents},
            "episode_step": step_count,
            "agent_types": {a.agent_id: a.aircraft_type.role for a in agents},
            "agent_alive": {a.agent_id: a.alive for a in agents},
            "episode_return": dict(episode_return),
            "win_flag": win_flag,
            "winner": winner,
            "termination_reason": termination_reason,
            "crash_count": sum(a.crashed for a in agents),
            "boundary_count": sum(a.out_of_boundary for a in agents),
            "missile_events": [
                {
                    "shooter_id": e.shooter_id,
                    "target_id": e.target_id,
                    "shooter_side": e.shooter_side,
                    "fired": e.fired,
                    "hit": e.hit,
                    "reason": e.reason,
                    "distance": e.distance,
                    "missile_left_after": e.missile_left_after,
                }
                for e in events
            ],
            "missile_summary": {
                "launches": len(fired_events),
                "hits": len(hit_events),
                "misses": sum(1 for e in fired_events if not e.hit),
                "cooldown_blocks": reason_counts.get("cooldown", 0),
                "no_missile_blocks": reason_counts.get("no_missile", 0),
                "not_visible_blocks": reason_counts.get("not_visible", 0),
                "out_of_range_blocks": reason_counts.get("out_of_range", 0),
                "los_blocks": reason_counts.get("los_blocked", 0),
                "reason_counts": dict(reason_counts),
            },
        }
        for agent in agents:
            info[agent.agent_id] = {
                "alive": agent.alive,
                "step": step_count,
                "missiles_left": agent.missile_left,
                "type": agent.aircraft_type.role,
            }
        return info
