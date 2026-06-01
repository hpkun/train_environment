"""Info dictionary construction."""

from __future__ import annotations

from .aircraft import AircraftPlatform


class InfoBuilder:
    def build(self, agents: list[AircraftPlatform], step_count: int,
              episode_return: dict[str, float], win_flag: str | None) -> dict:
        red = [a for a in agents if a.side == "red"]
        blue = [a for a in agents if a.side == "blue"]
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
            "crash_count": sum(a.crashed for a in agents),
            "boundary_count": sum(a.out_of_boundary for a in agents),
        }
        for agent in agents:
            info[agent.agent_id] = {
                "alive": agent.alive,
                "step": step_count,
                "missiles_left": agent.missile_left,
                "type": agent.aircraft_type.role,
            }
        return info
