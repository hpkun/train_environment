"""Run low-level Pure-HAPPO audit in a subprocess and persist crash context."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "audit_tam_brma_v1_pure_happo_low_level"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("audit_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_args = list(args.audit_args)
    if audit_args and audit_args[0] == "--":
        audit_args = audit_args[1:]
    if "--output-dir" not in audit_args:
        audit_args.extend(["--output-dir", str(out_dir)])
    cmd = [args.python, str(ROOT / "scripts" / "audit_pure_happo_low_level_diagnostics.py"), *audit_args]
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    payload = {
        "returncode": proc.returncode,
        "command": cmd,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    (out_dir / "live_rollout_subprocess_report.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (out_dir / "live_rollout_subprocess_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (out_dir / "live_rollout_subprocess_stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        (out_dir / "live_rollout_crash_report.md").write_text(
            "# Live Rollout Subprocess Crash Report\n\n"
            f"- returncode: {proc.returncode}\n"
            f"- command: `{cmd}`\n\n"
            "## stderr\n\n```text\n" + proc.stderr[-8000:] + "\n```\n\n"
            "## stdout\n\n```text\n" + proc.stdout[-8000:] + "\n```\n",
            encoding="utf-8",
        )
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
