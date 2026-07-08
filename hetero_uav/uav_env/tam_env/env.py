"""TAM-HAPPO direct-FCS diagnostic environment.

This module reuses the maintained JSBSim heterogeneous combat environment and
only replaces the actor control interface with TAM-style direct flight-control
surface commands.  It is a diagnostic environment skeleton, not a full
TAM-HAPPO reproduction.
"""

from __future__ import annotations

import gymnasium
import numpy as np

from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv


class TamCombatEnv(HeteroUavCombatEnv):
    """JSBSim hetero combat env with direct-FCS Box(4) actions.

    Action order per agent:
    ``[throttle, aileron, elevator, rudder]``.

    ``throttle`` maps from actor ``[-1, 1]`` to JSBSim ``[0.4, 0.9]``.
    Control surfaces are clipped directly to ``[-1, 1]`` and written to JSBSim
    FCS properties.  Reward, observation, missile, blue-rule, initialization,
    termination, and logging behavior are inherited unchanged.
    """

    THROTTLE_MIN = 0.4
    THROTTLE_MAX = 0.9

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = gymnasium.spaces.Dict({
            aid: gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            for aid in self.agent_ids
        })

    def _apply_action_trim(self, actions: dict) -> dict:
        """Bypass inherited 3-D PID action trim for TAM direct-FCS actions."""

        trimmed: dict = {}
        self._last_action_trim_applied = {}
        self._last_effective_actions = {}
        for aid, action in actions.items():
            effective = np.asarray(action, dtype=np.float32).reshape(-1)
            if effective.size < 4:
                effective = np.pad(effective, (0, 4 - effective.size), mode="constant")
            effective = np.clip(effective[:4], -1.0, 1.0).astype(np.float32)
            trimmed[aid] = effective
            self._last_action_trim_applied[aid] = [0.0, 0.0, 0.0, 0.0]
            self._last_effective_actions[aid] = [
                round(float(value), 6) for value in effective
            ]
        return trimmed

    def _parse_actions(self, actions: dict) -> dict:
        """Parse TAM direct-FCS Box(4) actions into JSBSim command tuples."""

        targets = {}
        for aid in self.agent_ids:
            raw = np.asarray(actions.get(aid, np.zeros(4, dtype=np.float32)), dtype=np.float32).reshape(-1)
            if raw.size < 4:
                raw = np.pad(raw, (0, 4 - raw.size), mode="constant")
            act = np.nan_to_num(raw[:4], nan=0.0, posinf=1.0, neginf=-1.0)
            act = np.clip(act, -1.0, 1.0)

            throttle = self.THROTTLE_MIN + (float(act[0]) + 1.0) * 0.5 * (
                self.THROTTLE_MAX - self.THROTTLE_MIN
            )
            aileron = float(act[1])
            elevator = float(act[2])
            rudder = float(act[3])
            targets[aid] = (
                float(np.clip(throttle, self.THROTTLE_MIN, self.THROTTLE_MAX)),
                float(np.clip(aileron, -1.0, 1.0)),
                float(np.clip(elevator, -1.0, 1.0)),
                float(np.clip(rudder, -1.0, 1.0)),
            )
        return targets

    def _apply_pid_controls(self, targets: dict):
        """Apply direct FCS commands; PID controllers are not used in tam_env."""

        for aid, target in targets.items():
            if target is None:
                continue
            sim = self._get_sim(aid)
            if sim is None or not sim.is_alive:
                continue
            throttle, aileron, elevator, rudder = target
            sim.set_property_value("fcs/throttle-cmd-norm", float(throttle))
            sim.set_property_value("fcs/aileron-cmd-norm", float(aileron))
            sim.set_property_value("fcs/elevator-cmd-norm", float(elevator))
            sim.set_property_value("fcs/rudder-cmd-norm", float(rudder))
