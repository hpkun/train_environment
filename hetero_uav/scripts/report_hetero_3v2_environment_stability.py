"""Rebuild formal 3v2 stability reports from existing audit artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hetero_3v2_environment_stability_common import build_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir
    payload = build_report(input_dir)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
