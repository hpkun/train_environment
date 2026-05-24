"""Static smoke test for paper_state_extractor without JSBSim/env imports."""
from __future__ import annotations

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from paper_state_extractor import extract_relative_state, extract_self_state


class FakeSim:
    def __init__(self, position, velocity, rpy, is_alive=True):
        self._position = np.asarray(position, dtype=np.float64)
        self._velocity = np.asarray(velocity, dtype=np.float64)
        self._rpy = np.asarray(rpy, dtype=np.float64)
        self.is_alive = is_alive

    def get_position(self):
        return self._position

    def get_velocity(self):
        return self._velocity

    def get_rpy(self):
        return self._rpy


def main():
    observer = FakeSim(
        position=[0.0, 0.0, 1000.0],
        velocity=[100.0, 0.0, -5.0],
        rpy=[0.1, 0.05, 0.2],
    )
    target = FakeSim(
        position=[1000.0, 500.0, 1200.0],
        velocity=[150.0, 20.0, 0.0],
        rpy=[0.0, 0.0, 0.0],
    )

    self_state = extract_self_state(observer)
    rel_state = extract_relative_state(observer, target, radar_detected=True)
    rel_masked = extract_relative_state(observer, target, radar_detected=False)

    assert self_state.shape == (10,)
    assert rel_state.shape == (10,)
    assert rel_masked.shape == (10,)
    assert rel_masked[3] == 0.0
    assert rel_masked[4] == 0.0
    assert rel_masked[5] == 0.0
    assert np.isfinite(self_state).all()
    assert np.isfinite(rel_state).all()
    assert np.isfinite(rel_masked).all()

    print("paper state extractor static smoke test passed")


if __name__ == "__main__":
    main()
