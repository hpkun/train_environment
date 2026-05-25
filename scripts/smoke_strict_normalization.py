"""Pure smoke test for strict entity normalization.  No env, no JSBSim."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_uav_env.alignment.global_state import build_strict_team_global_state
from my_uav_env.alignment.normalization import (
    normalize_strict_entities,
    normalize_strict_team_observations,
)


def main() -> None:
    # Build a 4-entity tensor (2v2: 1 ego + 1 ally + 2 enemies):
    #   row 0 = self, row 1 = relative, row 2 = relative, row 3 = padded
    entities = np.array([
        # self: x=0, y=0, h=6000, V=300, roll=0, pitch=0, heading=0, α=0, β=0, Vd=5
        [0.0, 0.0, 6000.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0],
        # relative: xb=10000, yb=0, zb=2000, θv=0, ψv=0, Vt=300, θ_los=0.2, ψ_los=0, q=0.2, d=10000
        [10000.0, 0.0, 2000.0, 0.0, 0.0, 300.0, 0.2, 0.0, 0.2, 10000.0],
        # relative (ally)
        [5000.0, 0.0, 1000.0, 0.0, 0.0, 280.0, 0.0, 0.0, 0.0, 5000.0],
        # padded (all zeros)
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)
    mask = np.array([0, 0, 0, 1], dtype=np.int64)

    original = entities.copy()

    out = normalize_strict_entities(entities, mask)

    # Shape unchanged
    assert out.shape == (4, 10)

    # Original not modified in-place
    assert np.allclose(entities, original)

    # Padded row stays all-zero
    assert np.allclose(out[3], 0.0)

    # Self row checks
    assert out[0, 2] == 6000.0 / 10000.0   # h
    assert out[0, 3] == 300.0 / 600.0       # V
    assert out[0, 9] == 5.0 / 600.0         # Vd

    # Relative row checks
    assert out[1, 0] == 10000.0 / 40000.0   # x_body
    assert out[1, 2] == 2000.0 / 10000.0    # z_body
    assert out[1, 5] == 300.0 / 600.0        # V_target
    assert out[1, 9] == 10000.0 / 40000.0   # d

    # All outputs finite
    assert np.isfinite(out).all()

    # Clipping at bounds
    big = np.full((1, 10), 1e9, dtype=np.float32)
    big_out = normalize_strict_entities(big)
    assert np.all(big_out >= -5.0) and np.all(big_out <= 5.0)

    # ---- normalize_strict_team_observations ----
    team = {
        "red_0": (entities.copy(), mask.copy(), {"schema": "test"}),
    }
    norm_team = normalize_strict_team_observations(team)
    assert "red_0" in norm_team
    e2, m2, meta2 = norm_team["red_0"]
    assert e2.shape == (4, 10)
    assert meta2 == {"schema": "test"}

    # ---- global_state with normalization ----
    fake_team = {}
    for i in range(2):
        e = entities.copy()
        m = mask.copy()
        fake_team[f"red_{i}"] = (e, m, {"schema": "test"})
    gs_norm = build_strict_team_global_state(fake_team, 2, 2, normalize=True)
    gs_raw = build_strict_team_global_state(fake_team, 2, 2, normalize=False)
    assert gs_norm.shape == gs_raw.shape == (88,)
    assert np.isfinite(gs_norm).all()
    # Normalized values should differ from raw (unless already 0)
    assert not np.allclose(gs_norm, gs_raw)

    print("strict normalization smoke test passed")


if __name__ == "__main__":
    main()
