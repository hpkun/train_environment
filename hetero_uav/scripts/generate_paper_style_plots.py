"""Generate paper-style plots from rich logs.

Missing inputs produce placeholder figures instead of errors.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def _rel(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, Any], key: str) -> float | None:
    try:
        value = row.get(key)
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _placeholder(out_dir: Path, stem: str, title: str, reason: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title(title)
    ax.text(0.5, 0.5, f"missing\n{reason}", ha="center", va="center")
    ax.set_xticks([])
    ax.set_yticks([])
    _save(fig, out_dir, stem)


def _plot_line(out_dir: Path, stem: str, title: str, rows: list[dict[str, str]], y_keys: list[tuple[str, str]]) -> bool:
    if not rows:
        _placeholder(out_dir, stem, title, "train_metrics.csv missing")
        return False
    xs = [_float(r, "total_env_steps_actual") or _float(r, "train_steps") or i for i, r in enumerate(rows)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    any_line = False
    for key, label in y_keys:
        pairs = [(x, _float(r, key)) for x, r in zip(xs, rows)]
        pairs = [(x, y) for x, y in pairs if y is not None]
        if pairs:
            px, py = zip(*pairs)
            ax.plot(px, py, label=label)
            any_line = True
    if not any_line:
        plt.close(fig)
        _placeholder(out_dir, stem, title, "required columns missing")
        return False
    ax.set_title(title)
    ax.set_xlabel("env steps")
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, out_dir, stem)
    return True


def _plot_bar(out_dir: Path, stem: str, title: str, rows: list[dict[str, str]], keys: list[tuple[str, str]]) -> bool:
    if not rows:
        _placeholder(out_dir, stem, title, "metrics missing")
        return False
    last = rows[-1]
    vals = [_float(last, k) for k, _ in keys]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(keys)), [v if v is not None else 0.0 for v in vals])
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([label for _, label in keys], rotation=20, ha="right")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out_dir, stem)
    return True


def _plot_trajectory(out_dir: Path, rows: list[dict[str, str]]) -> bool:
    if not rows:
        _placeholder(out_dir, "trajectory_2d", "2D Trajectory", "aircraft_timeseries.csv missing")
        return False
    fig, ax = plt.subplots(figsize=(7, 6))
    grouped: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        lon = _float(r, "lon")
        lat = _float(r, "lat")
        aid = r.get("agent_id", "agent")
        if lon is not None and lat is not None:
            grouped.setdefault(aid, []).append((lon, lat))
    if not grouped:
        plt.close(fig)
        _placeholder(out_dir, "trajectory_2d", "2D Trajectory", "lon/lat missing")
        return False
    for aid, pts in grouped.items():
        xs, ys = zip(*pts)
        ax.plot(xs, ys, label=aid)
    ax.set_title("2D Trajectory")
    ax.set_xlabel("lon/x")
    ax.set_ylabel("lat/y")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    _save(fig, out_dir, "trajectory_2d")
    return True


def _plot_attitude(out_dir: Path, rows: list[dict[str, str]]) -> bool:
    if not rows:
        _placeholder(out_dir, "aircraft_attitude_curves", "Aircraft Attitude Curves", "aircraft_timeseries.csv missing")
        return False
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    keys = [("altitude", "altitude"), ("speed", "speed"), ("yaw", "yaw"), ("pitch", "pitch")]
    for ax, (key, title) in zip(axes.flat, keys):
        grouped: dict[str, list[tuple[float, float]]] = {}
        for r in rows:
            step = _float(r, "step")
            val = _float(r, key)
            aid = r.get("agent_id", "agent")
            if step is not None and val is not None:
                grouped.setdefault(aid, []).append((step, val))
        for aid, pts in grouped.items():
            xs, ys = zip(*pts)
            ax.plot(xs, ys, label=aid)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Aircraft Attitude Curves")
    _save(fig, out_dir, "aircraft_attitude_curves")
    return True


def _plot_reward_components(out_dir: Path, rows: list[dict[str, str]]) -> bool:
    if not rows:
        _placeholder(out_dir, "reward_component_curves", "Reward Component Curves", "reward_components.csv missing")
        return False
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cols = [c for c in rows[0].keys() if c.endswith("_reward") or c == "total_reward"]
    any_line = False
    for col in cols[:8]:
        pairs = [(_float(r, "step"), _float(r, col)) for r in rows]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if pairs:
            xs, ys = zip(*pairs)
            ax.plot(xs, ys, label=col)
            any_line = True
    if not any_line:
        plt.close(fig)
        _placeholder(out_dir, "reward_component_curves", "Reward Component Curves", "component columns missing")
        return False
    ax.set_title("Reward Component Curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    _save(fig, out_dir, "reward_component_curves")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paper-style plots from rich experiment logs")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/paper_style_figures")
    args = parser.parse_args()

    input_dir = _rel(args.input_dir)
    output_dir = _rel(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _read_csv(input_dir / "train_metrics.csv")
    aircraft = _read_csv(input_dir / "aircraft_timeseries.csv")
    reward = _read_csv(input_dir / "reward_components.csv")
    generated: list[str] = []

    if _plot_line(output_dir, "reward_curve", "Reward Curve", train, [("avg_episode_return", "episode return"), ("avg_team_reward", "team reward")]):
        generated.append("reward_curve")
    if _plot_line(output_dir, "win_rate_curve", "Win-Rate Curve", train, [("red_win_rate", "red win"), ("blue_win_rate", "blue win")]):
        generated.append("win_rate_curve")
    if _plot_bar(output_dir, "rwr_kd_bar", "RWR / KD", train, [("relative_win_ratio", "RWR"), ("kill_death_ratio", "KD")]):
        generated.append("rwr_kd_bar")
    if _plot_bar(output_dir, "zero_shot_transfer_bar", "Scale Transfer Summary", train, [("red_win_rate", "win"), ("mav_survival_rate", "MAV survival"), ("blue_dead_mean", "blue dead")]):
        generated.append("zero_shot_transfer_bar")
    if _plot_line(output_dir, "ablation_reward_win_curve", "Ablation Reward/Win Curve", train, [("avg_episode_return", "return"), ("red_win_rate", "red win")]):
        generated.append("ablation_reward_win_curve")
    if _plot_trajectory(output_dir, aircraft):
        generated.append("trajectory_2d")
    if _plot_attitude(output_dir, aircraft):
        generated.append("aircraft_attitude_curves")
    if _plot_reward_components(output_dir, reward):
        generated.append("reward_component_curves")
    if _plot_bar(output_dir, "perturbation_generalization_bar", "Perturbation Generalization", _read_csv(input_dir / "perturbation_eval_summary.csv"), [("win_rate", "win"), ("mav_survival_rate", "MAV"), ("blue_dead_mean", "blue dead")]):
        generated.append("perturbation_generalization_bar")
    if _plot_line(output_dir, "loss_entropy_gradient_curves", "Loss / Entropy / Gradient", train, [("critic_loss", "critic loss"), ("entropy", "entropy"), ("policy_gradient_norm", "policy grad")]):
        generated.append("loss_entropy_gradient_curves")

    eff = _rel(input_dir / "training_efficiency.json")
    table = output_dir / "training_efficiency_table.md"
    if eff.exists():
        data = json.loads(eff.read_text(encoding="utf-8"))
        table.write_text("\n".join(["# Training Efficiency", ""] + [f"- {k}: {v}" for k, v in data.items()]) + "\n", encoding="utf-8")
    else:
        table.write_text("# Training Efficiency\n\nmissing: training_efficiency.json\n", encoding="utf-8")
    index = output_dir / "figure_index.md"
    index.write_text("\n".join(["# Paper-Style Figure Index", ""] + [f"- {name}" for name in generated]) + "\n", encoding="utf-8")
    print(f"figure_index: {index}")
    print(f"generated: {len(generated)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
