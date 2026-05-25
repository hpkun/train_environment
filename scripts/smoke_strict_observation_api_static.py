"""Static smoke test for strict observation API existence.

Does not create an environment, does not trigger JSBSim.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from my_uav_env import UavCombatEnv

    assert hasattr(UavCombatEnv, "get_strict_entity_observation"), \
        "UavCombatEnv missing get_strict_entity_observation"
    assert hasattr(UavCombatEnv, "get_strict_team_observations"), \
        "UavCombatEnv missing get_strict_team_observations"

    print("strict observation API static smoke test passed")


if __name__ == "__main__":
    main()
