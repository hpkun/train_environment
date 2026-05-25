"""Environment smoke test for strict paper observation API.

Do not run this in Codex; it creates UavCombatEnv and triggers JSBSim.
User runs locally only.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from my_uav_env import UavCombatEnv


def main() -> None:
    print("Creating 1v1 env (max_steps=2) ...", flush=True)
    env = UavCombatEnv(
        max_num_blue=1, max_num_red=1,
        max_steps=2,
        enable_gcas_for_blue=False,
        suppress_jsbsim_output=True,
    )
    try:
        print("Resetting ...", flush=True)
        env.reset()

        print("\n--- get_strict_entity_observation('red_0') ---")
        entities, mask, meta = env.get_strict_entity_observation("red_0")
        print(f"  entities.shape: {entities.shape}")
        print(f"  mask:           {mask.tolist()}")
        print(f"  meta.schema:    {meta.get('schema')}")
        print(f"  meta.layout:    {meta['layout']}")
        print(f"  self[0] sample: {entities[0][:6]} ...")  # first 6 values

        print("\n--- get_strict_team_observations('red') ---")
        red_obs = env.get_strict_team_observations("red")
        for aid, (ent, msk, mt) in red_obs.items():
            print(f"  {aid}: entities={ent.shape}, mask={msk.tolist()}, "
                  f"schema={mt.get('schema')}")

        print("\n--- get_strict_team_observations('blue') ---")
        blue_obs = env.get_strict_team_observations("blue")
        for aid, (ent, msk, mt) in blue_obs.items():
            print(f"  {aid}: entities={ent.shape}, mask={msk.tolist()}, "
                  f"schema={mt.get('schema')}")

        # Smoke that a Value error is raised for bad team name
        try:
            env.get_strict_team_observations("green")
        except ValueError as e:
            print(f"\n  ValueError correctly raised: {e}")

        print("\nstrict observation env smoke test passed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
