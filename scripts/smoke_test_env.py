from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_uav_env import UavCombatEnv


def main() -> None:
    env = UavCombatEnv(
        max_num_blue=1,
        max_num_red=1,
        max_steps=5,
        suppress_jsbsim_output=True,
    )
    try:
        obs, _info = env.reset()
        print(f"reset agents={sorted(obs.keys())}")

        action = np.zeros(3, dtype=np.float32)
        actions = {"red_0": action, "blue_0": action}
        for step in range(3):
            _obs, rewards, terminated, truncated, _info = env.step(actions)
            print(
                f"step={step + 1} "
                f"rewards={rewards} "
                f"terminated={terminated} "
                f"truncated={truncated}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
