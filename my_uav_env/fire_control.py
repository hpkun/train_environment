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
        snapshot = asdict(self)
        snapshot.update({
            "tracked_target_id": self.current_target_id,
            "continuous_eo_detection_frames": self.continuous_detection_frames,
            "launch_cooldown_frames": self.cooldown_frames_remaining,
        })
        return snapshot

    @property
    def tracked_target_id(self) -> str | None:
        return self.current_target_id

    @tracked_target_id.setter
    def tracked_target_id(self, value: str | None) -> None:
        self.current_target_id = value

    @property
    def continuous_eo_detection_frames(self) -> int:
        return self.continuous_detection_frames

    @continuous_eo_detection_frames.setter
    def continuous_eo_detection_frames(self, value: int) -> None:
        self.continuous_detection_frames = int(value)

    @property
    def launch_cooldown_frames(self) -> int:
        return self.cooldown_frames_remaining

    @launch_cooldown_frames.setter
    def launch_cooldown_frames(self, value: int) -> None:
        self.cooldown_frames_remaining = int(value)
