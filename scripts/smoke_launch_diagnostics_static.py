from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_uav_env.env import LAUNCH_DIAG_KEYS, make_empty_launch_diag
from train_vanilla_mappo import (
    _accumulate_launch_diag_totals,
    _empty_launch_diag_totals,
    _launch_diag_metrics,
)


def test_empty_launch_diag() -> None:
    diag = make_empty_launch_diag()
    assert set(diag.keys()) == {"red", "blue"}
    for team in ("red", "blue"):
        assert set(LAUNCH_DIAG_KEYS).issubset(diag[team].keys())
        assert all(value == 0 for value in diag[team].values())

    diag["red"]["launches"] = 99
    fresh = make_empty_launch_diag()
    assert fresh["red"]["launches"] == 0


def test_train_aggregation_helpers() -> None:
    totals = _empty_launch_diag_totals()
    _accumulate_launch_diag_totals(totals, None)
    metrics = _launch_diag_metrics(totals)
    assert metrics["LaunchDiagRedLaunches"] == 0
    assert metrics["RedGeometryToLaunchRate"] == 0.0

    _accumulate_launch_diag_totals(
        totals,
        {
            "red": {
                "range_ok_pairs": 10,
                "ao_ok_pairs": 7,
                "ta_ok_pairs": 5,
                "geometry_ok_pairs": 4,
                "launches": 2,
                "lock_mature_pairs": 3,
                "cooldown_blocked": 1,
            },
            "blue": {
                "range_ok_pairs": 8,
                "geometry_ok_pairs": 2,
                "launches": 1,
                "engaged_blocked": 5,
                "kill_cooldown_blocked": 1,
            },
        },
    )
    metrics = _launch_diag_metrics(totals)
    assert metrics["LaunchDiagRedGeometryOk"] == 4
    assert metrics["LaunchDiagRedLaunches"] == 2
    assert metrics["LaunchDiagBlueGeometryOk"] == 2
    assert metrics["LaunchDiagBlueLaunches"] == 1
    assert metrics["LaunchDiagRedCooldownBlocked"] == 1
    assert metrics["LaunchDiagBlueEngagedBlocked"] == 5
    assert metrics["LaunchDiagBlueKillCooldownBlocked"] == 1
    assert metrics["RedGeometryToLaunchRate"] == 0.5
    assert metrics["BlueRangeToGeometryRate"] == 0.25


def main() -> None:
    test_empty_launch_diag()
    test_train_aggregation_helpers()
    print("launch diagnostics static smoke test passed")


if __name__ == "__main__":
    main()
