"""Pure smoke test for _actor_std_stats.  No env, no JSBSim."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from train_vanilla_mappo import VanillaActor, _actor_std_stats, _compute_obs_dim


def main() -> None:
    obs_dim = _compute_obs_dim(2, 2, is_red=True)
    actor = VanillaActor(obs_dim=obs_dim)
    stats = _actor_std_stats(actor)

    for key in ("action_std_mean", "action_std_min", "action_std_max",
                "action_log_std_mean"):
        assert key in stats, f"missing key: {key}"
        assert isinstance(stats[key], float), f"{key} not float: {type(stats[key])}"
        assert np.isfinite(stats[key]), f"{key} not finite: {stats[key]}"

    # Initial log_std = -1.204 → std ≈ exp(-1.204) ≈ 0.3
    assert 0.2 < stats["action_std_mean"] < 0.5, \
        f"expected std ~0.3, got {stats['action_std_mean']}"

    print(f"  action_std_mean:     {stats['action_std_mean']:.6f}")
    print(f"  action_std_min:      {stats['action_std_min']:.6f}")
    print(f"  action_std_max:      {stats['action_std_max']:.6f}")
    print(f"  action_log_std_mean: {stats['action_log_std_mean']:.6f}")
    print("vanilla entropy diagnostics smoke test passed")


if __name__ == "__main__":
    main()
