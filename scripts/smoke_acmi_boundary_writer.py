"""Pure smoke test for optional ACMI battlefield boundary writer."""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acmi_boundary_utils import (
    maybe_write_battlefield_boundary_acmi,
    write_battlefield_boundary_acmi,
)


def main() -> None:
    buf = io.StringIO()
    write_battlefield_boundary_acmi(buf, 40000.0)
    text = buf.getvalue()
    assert "Battlefield Boundary" in text
    assert "40000m" in text
    assert "Boundary SW" in text
    assert "Boundary NE" in text

    off = io.StringIO()
    maybe_write_battlefield_boundary_acmi(off, False, 40000.0)
    assert "Boundary" not in off.getvalue()

    on = io.StringIO()
    maybe_write_battlefield_boundary_acmi(on, True, 40000.0)
    assert "Battlefield Boundary" in on.getvalue()

    print("acmi boundary writer smoke test passed")


if __name__ == "__main__":
    main()
