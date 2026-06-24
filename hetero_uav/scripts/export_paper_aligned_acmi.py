"""Deprecated. Use scripts/export_happo_reference_acmi.py as the canonical exporter.

This script is kept as a thin wrapper that prints a deprecation warning
and delegates to the canonical exporter.  It must not contain its own
hard-coded model/config logic.
"""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    print(
        "DEPRECATED: export_paper_aligned_acmi.py is no longer maintained.\n"
        "Use scripts/export_happo_reference_acmi.py as the canonical exporter.\n"
        "Forwarding all arguments to the canonical exporter...",
        file=sys.stderr,
    )
    cmd = [sys.executable, str(ROOT / "scripts" / "export_happo_reference_acmi.py")] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
