"""Explicit per-shooter fire-control diagnostic state."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class FireControlState:
    current_target_id: str | None = None
    detection_state: str = "no_target"
    continuous_detection_frames: int = 0
    lock_mature: bool = False
    cooldown_frames_remaining: int = 0
    last_launch_frame: int | None = None
    blocked_reason: str = ""
    transition_reason: str = "reset"

    def snapshot(self) -> dict:
        return asdict(self)
