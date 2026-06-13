"""Generate cropped representative ACMI trajectory figures for progress reports."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = "outputs/progress_report_figures/representative_episode_selection.json"
DEFAULT_OUTPUT_DIR = "outputs/progress_report_figures"


def _rel(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    png = out_dir / f"{stem}.png"
    svg = out_dir / f"{stem}.svg"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return [png, svg]


def _missing(out_dir: Path, stem: str, title: str, message: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    return _save(fig, out_dir, stem)


def _parse_acmi(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, list[tuple[float, float, float, float]]]]:
    objects: dict[str, dict[str, str]] = {}
    tracks: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    if not path.exists():
        return objects, tracks
    current_t = 0.0
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if raw.startswith("#"):
            try:
                current_t = float(raw[1:])
            except ValueError:
                pass
            continue
        if not raw or "," not in raw:
            continue
        obj_id, rest = raw.split(",", 1)
        fields: dict[str, str] = {}
        for part in rest.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                fields[k] = v
        if "Type" in fields or "Name" in fields or "Color" in fields:
            meta = objects.setdefault(obj_id, {})
            for key in ("Type", "Name", "Color"):
                if key in fields:
                    meta[key] = fields[key]
        t = fields.get("T")
        if t:
            bits = t.split("|")
            if len(bits) >= 3:
                try:
                    tracks[obj_id].append((current_t, float(bits[0]), float(bits[1]), float(bits[2])))
                except ValueError:
                    pass
    return objects, tracks


def _is_missile(meta: dict[str, str]) -> bool:
    return "Missile" in meta.get("Type", "") or "Missile" in meta.get("Name", "")


def _is_red_missile(meta: dict[str, str]) -> bool:
    name = meta.get("Name", "")
    return _is_missile(meta) and ("red_" in name or meta.get("Color") == "Red")


def _window(objects: dict[str, dict[str, str]], tracks: dict[str, list[tuple[float, float, float, float]]]) -> tuple[float, float]:
    red_missile_times: list[float] = []
    for obj_id, meta in objects.items():
        if _is_red_missile(meta):
            pts = tracks.get(obj_id, [])
            if pts:
                red_missile_times.append(pts[0][0])
                red_missile_times.append(pts[-1][0])
    if red_missile_times:
        return max(0.0, min(red_missile_times) - 20.0), max(red_missile_times) + 20.0
    max_t = max((pts[-1][0] for pts in tracks.values() if pts), default=100.0)
    return 0.0, min(max_t, 100.0)


def _style(meta: dict[str, str]) -> tuple[str, float, float, str]:
    name = meta.get("Name", "")
    color = meta.get("Color", "")
    if _is_missile(meta):
        return "#111111", 1.7, 0.8, "missile"
    if color == "Blue" or "blue" in name:
        return "#1f77b4", 1.5, 0.9, "blue UAV"
    if "red_0" in name or "MAV" in name:
        return "#d62728", 2.5, 1.0, "red_0 MAV"
    return "#ff7f0e", 1.5, 0.9, "red UAV"


def _draw_arrows(ax: plt.Axes, xs: list[float], ys: list[float], color: str) -> None:
    if len(xs) < 8:
        return
    for frac in (0.35, 0.7):
        idx = max(1, min(len(xs) - 2, int(len(xs) * frac)))
        ax.annotate(
            "",
            xy=(xs[idx + 1], ys[idx + 1]),
            xytext=(xs[idx - 1], ys[idx - 1]),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=0.75),
        )


def _plot_one(out_dir: Path, stem: str, title: str, selected: dict[str, Any] | None) -> list[Path]:
    if not selected:
        return _missing(out_dir, stem, title, "No representative episode selected")
    acmi = Path(str(selected.get("acmi", "")))
    if not acmi.exists():
        return _missing(out_dir, stem, title, f"Missing ACMI:\n{acmi}")
    objects, tracks = _parse_acmi(acmi)
    if not tracks:
        return _missing(out_dir, stem, title, "ACMI has no drawable tracks")
    t0, t1 = _window(objects, tracks)

    fig, ax = plt.subplots(figsize=(9, 7))
    launch_points: list[tuple[float, float]] = []
    hit_points: list[tuple[float, float]] = []
    for obj_id, pts in tracks.items():
        meta = objects.get(obj_id, {})
        window_pts = [p for p in pts if t0 <= p[0] <= t1]
        if len(window_pts) < 2:
            continue
        color, lw, alpha, label = _style(meta)
        step = max(1, len(window_pts) // 280)
        xs = [p[1] for p in window_pts[::step]]
        ys = [p[2] for p in window_pts[::step]]
        ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, label=label, ls="--" if label == "missile" else "-")
        ax.scatter(xs[0], ys[0], color=color, marker="o", s=28)
        ax.scatter(xs[-1], ys[-1], color=color, marker="x", s=36)
        _draw_arrows(ax, xs, ys, color)
        if _is_red_missile(meta):
            launch_points.append((xs[0], ys[0]))
            hit_points.append((xs[-1], ys[-1]))

    if launch_points:
        lx, ly = zip(*launch_points)
        ax.scatter(lx, ly, marker="^", color="#111", s=80, label="red launch", zorder=5)
    if hit_points:
        hx, hy = zip(*hit_points)
        ax.scatter(hx, hy, marker="*", color="#ffd400", edgecolor="#111", s=170, label="hit/end", zorder=6)

    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), fontsize=8, loc="best")
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("longitude / local x")
    ax.set_ylabel("latitude / local y")
    ax.grid(alpha=0.25)
    textbox = "\n".join([
        f"outcome: {selected.get('outcome')}",
        f"red fired: {selected.get('red_missiles_fired')}",
        f"red hits: {selected.get('red_missile_hits')}",
        f"blue dead: {selected.get('blue_dead')}",
        f"MAV alive: {selected.get('mav_alive')}",
        f"window: {t0:.1f}s-{t1:.1f}s",
    ])
    ax.text(0.02, 0.98, textbox, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.82, edgecolor="#bbb"))
    if "5v4" in stem:
        ax.text(
            0.02,
            0.02,
            "Representative attack-transfer episode; not all 5v4 episodes are elimination wins.",
            transform=ax.transAxes,
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="#ddd"),
        )
    return _save(fig, out_dir, stem)


def _update_index(output_dir: Path) -> None:
    index = output_dir / "figure_index.md"
    block = """
## fig07_trajectory_3v2_representative
- file: `fig07_trajectory_3v2_representative.png`, `fig07_trajectory_3v2_representative.svg`
- title: 3v2 Representative Combat Trajectory
- purpose: Cropped ACMI trajectory for progress-report slides.
- one-sentence conclusion: The selected 3v2 episode shows red launch/hit behavior with MAV alive.
- report page: trajectory evidence

## fig08_trajectory_5v4_representative
- file: `fig08_trajectory_5v4_representative.png`, `fig08_trajectory_5v4_representative.svg`
- title: 5v4 Zero-Shot Representative Trajectory
- purpose: Cropped ACMI trajectory for attack-transfer visualization.
- one-sentence conclusion: The selected 5v4 episode shows attack transfer, but it is not an elimination claim for all episodes.
- report page: trajectory evidence
"""
    if index.exists():
        text = index.read_text(encoding="utf-8")
        if "fig07_trajectory_3v2_representative" not in text:
            index.write_text(text.rstrip() + "\n" + block, encoding="utf-8")
    else:
        index.write_text("# Progress Report Figure Index\n" + block, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate representative ACMI trajectory figures")
    parser.add_argument("--selection-json", default=DEFAULT_SELECTION)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    output_dir = _rel(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = _read_json(_rel(args.selection_json))
    selected = selection.get("selected", {}) if isinstance(selection, dict) else {}
    files = []
    files.extend(_plot_one(
        output_dir,
        "fig07_trajectory_3v2_representative",
        "3v2 Representative Combat Trajectory",
        selected.get("3v2"),
    ))
    files.extend(_plot_one(
        output_dir,
        "fig08_trajectory_5v4_representative",
        "5v4 Zero-Shot Representative Trajectory",
        selected.get("5v4"),
    ))
    _update_index(output_dir)
    print(f"generated {len(files)} files")
    for path in files:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
