"""Print HeteroObsSpec v1 dimensions and raw obs keys.

Does NOT implement an adapter.  Does NOT run MAPPO or attention.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env


HETERO_CONFIG = "uav_env/JSBSim/configs/hetero_train_2v2_mav_attack.yaml"


def main() -> None:
    # ---- spec dimensions ----
    max_red = 5
    max_blue = 4
    max_allies = max_red - 1   # 4
    max_enemies = max_blue      # 4
    ego_feature_dim = 11 + 4 + 1 + 1 + 3   # = 20
    ally_entity_dim = 11 + 4               # = 15
    enemy_entity_dim = 11                  # = 11
    mask_dim = max_allies + max_allies + max_enemies + max_enemies  # = 16
    flat_actor_obs_dim = (ego_feature_dim
                          + max_allies * ally_entity_dim
                          + max_enemies * enemy_entity_dim
                          + mask_dim)      # = 140
    critic_state_dim = flat_actor_obs_dim * max_red  # = 700

    print("=== HeteroObsSpec v1 ===")
    print(f"  max_red:              {max_red}")
    print(f"  max_blue:             {max_blue}")
    print(f"  max_allies:           {max_allies}")
    print(f"  max_enemies:          {max_enemies}")
    print(f"  ego_feature_dim:      {ego_feature_dim}")
    print(f"  ally_entity_dim:      {ally_entity_dim}")
    print(f"  enemy_entity_dim:     {enemy_entity_dim}")
    print(f"  mask_dim:             {mask_dim}")
    print(f"  flat_actor_obs_dim:   {flat_actor_obs_dim}")
    print(f"  critic_state_dim:     {critic_state_dim}")

    # ---- raw env inspection ----
    print(f"\n=== Raw environment ({HETERO_CONFIG}) ===")
    env = None
    try:
        env = make_env(HETERO_CONFIG, env_type="jsbsim_hetero", max_steps=5)
        obs, _info = env.reset(seed=0)

        red0 = obs["red_0"]
        print(f"  red_0 raw obs keys ({len(red0)}):")
        for k in sorted(red0.keys()):
            arr = red0[k]
            print(f"    {k:20s}  shape={str(arr.shape):>12s}  dtype={arr.dtype}")

        print("\n  adapter v1 PLANNED used keys:")
        for k in ["ego_state", "ego_role", "missile_warning", "altitude",
                  "velocity", "ally_states", "ally_roles", "enemy_states"]:
            present = "YES" if k in red0 else "MISSING"
            print(f"    {k:20s}  {present}")

        print("\n  adapter v1 IGNORED keys:")
        for k in ["ego_type", "ally_types", "enemy_types", "enemy_roles",
                  "death_mask"]:
            present = "YES" if k in red0 else "MISSING"
            print(f"    {k:20s}  {present} (not used in v1)")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
