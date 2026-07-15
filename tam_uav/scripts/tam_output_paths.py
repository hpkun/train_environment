"""Output path boundary shared by TAM training, evaluation, and audit scripts."""

from __future__ import annotations

from pathlib import Path


def resolve_tam_output(root: Path, user_path, *, create_parent=False):
    """Resolve an output path and reject traversal and symlink escapes."""
    root = root.resolve()
    raw = Path(user_path)
    candidate = (raw if raw.is_absolute() else root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"output path must remain inside tam_uav: {user_path}") from exc
    if create_parent:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        if candidate.parent.resolve() != candidate.parent:
            raise ValueError(f"output parent resolves through an unsafe symlink: {user_path}")
    return candidate
