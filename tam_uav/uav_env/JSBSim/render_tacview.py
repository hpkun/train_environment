"""
ACMI text-format logger for TacView visualization.

Strictly follows the TacView 2.1 ACMI file format specification:
  1. Header: FileType + FileVersion + ReferenceTime (once)
  2. Time markers: #<seconds>
  3. First appearance: ID,T=...,Type=Air+FixedWing,Name=...,Color=...
  4. Subsequent updates: ID,T=...
  5. Entity removal: -ID
"""
from typing import List, Dict, Set, Optional


class TacviewLogger:
    """Collects per-frame ACMI log lines and writes a valid .acmi file."""

    def __init__(self, reference_time: str = "2026-01-01T00:00:00Z"):
        self._reference_time = reference_time
        self._lines: List[str] = []
        self._frame_count = 0
        self._introduced: Set[int] = set()
        self._alive_prev: Set[int] = set()

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Discard all recorded frames (e.g. at the start of a new episode)."""
        self._lines.clear()
        self._frame_count = 0
        self._introduced.clear()
        self._alive_prev.clear()

    def record_frame(self, sim_time: float,
                     entries: List[dict],
                     explosions: Optional[List[dict]] = None):
        """
        Append one time-slice.

        Args:
            sim_time: elapsed simulation time in seconds.
            entries: list of entity dicts, each with keys:
                acmi_id, lon, lat, alt, roll, pitch, yaw, name, color, alive
            explosions: list of explosion dicts, each with keys:
                acmi_id, lon, lat, alt, color, radius
        """
        # ---- Time marker ----
        self._lines.append(f"#{sim_time}")

        alive_current: Set[int] = set()

        for e in entries:
            if not e["alive"]:
                continue

            aid = e["acmi_id"]
            alive_current.add(aid)

            if aid not in self._introduced:
                entity_type = e.get("type", "Air+FixedWing")
                self._lines.append(
                    f"{aid},T={e['lon']}|{e['lat']}|{e['alt']}|"
                    f"{e['roll']}|{e['pitch']}|{e['yaw']},"
                    f"Type={entity_type},Name={e['name']},Color={e['color']}"
                )
                self._introduced.add(aid)
            else:
                # ---- Subsequent update: T= only ----
                self._lines.append(
                    f"{aid},T={e['lon']}|{e['lat']}|{e['alt']}|"
                    f"{e['roll']}|{e['pitch']}|{e['yaw']}"
                )

        # ---- Entity removal: -ID for newly dead entities ----
        newly_dead = self._alive_prev - alive_current
        for aid in newly_dead:
            self._lines.append(f"-{aid}")

        self._alive_prev = alive_current.copy()

        # ---- Explosions (missile hits) ----
        if explosions:
            for ex in explosions:
                self._lines.append(
                    f"{ex['acmi_id']}F,T={ex['lon']}|{ex['lat']}|{ex['alt']}"
                    f"|0|0|0,"
                    f"Type=Misc+Explosion,Color={ex['color']},"
                    f"Radius={ex['radius']}"
                )

        self._frame_count += 1

    def write(self, filepath: str):
        """Flush all recorded frames to an .acmi file."""
        with open(filepath, "w", encoding="utf-8-sig", newline="\n") as f:
            f.write("FileType=text/acmi/tacview\n")
            f.write("FileVersion=2.1\n")
            f.write(f"0,ReferenceTime={self._reference_time}\n")
            for line in self._lines:
                f.write(line + "\n")

    def append_lines(self, lines: List[str]):
        """Append raw ACMI lines directly to the buffer.

        Used by eval scripts to inject fake visual-only entities (e.g. missile
        trajectories) that are not part of the simulation but aid after-action
        review.
        """
        self._lines.extend(lines)

    @property
    def frame_count(self) -> int:
        return self._frame_count
