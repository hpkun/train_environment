"""Environment smoke test for strict paper observation extraction.

Do not run this in Codex; it triggers JSBSim. User runs locally only.
"""
from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from my_uav_env import UavCombatEnv
from paper_state_extractor import build_strict_paper_entity_observation


def main():
    env = UavCombatEnv(
        max_num_blue=1,
        max_num_red=1,
        max_steps=2,
        enable_gcas_for_blue=False,
        suppress_jsbsim_output=True,
    )
    try:
        env.reset()
        entities, mask, meta = build_strict_paper_entity_observation(env, "red_0")
        print(f"entities.shape: {entities.shape}")
        print(f"mask: {mask.tolist()}")
        print(f"meta: {meta}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
