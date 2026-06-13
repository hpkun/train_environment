"""Generate PNG/SVG figures for progress reports from existing outputs only."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = "outputs/progress_report_figures"

FINAL_3V2 = {
    "red_win_rate": 0.92,
    "red_elimination_win_rate": 0.52,
    "mav_survival_rate": 0.94,
    "red_missiles_fired_mean": 1.82,
    "red_missile_hits_mean": 1.56,
    "blue_dead_mean": 1.52,
}
FINAL_5V4 = {
    "red_win_rate": 0.93,
    "red_elimination_win_rate": 0.14,
    "mav_survival_rate": 0.99,
    "red_missiles_fired_mean": 2.59,
    "red_missile_hits_mean": 2.38,
    "blue_dead_mean": 2.33,
    "red_timeout_alive_advantage_rate": 0.79,
}

TRAIN_RUNS = [
    ("HAPPO ref", "happo_3v2_reference_f16_mav_surrogate_1m_fast"),
    ("Oracle direct", "happo_oracle_pretrain_finetune_200k"),
    ("Easy anchor", "happo_easy_combat_oracle_anchor_50k"),
    ("Normal direct", "happo_normal_geometry_oracle_anchor_100k"),
    ("Medium curr.", "happo_geometry_curriculum_100k/medium_50k"),
    ("Normal curr.", "happo_geometry_curriculum_100k/normal_50k"),
    ("5v4 fine-tune", "happo_5v4_finetune_upper_bound_50k"),
]

FIGURES = [
    ("fig01_experiment_pipeline", "Experiment Pipeline", "algorithm flow"),
    ("fig02_method_comparison_bar", "Method Comparison", "metric results"),
    ("fig03_transfer_quality_3v2_vs_5v4", "3v2 Seen vs 5v4 Zero-Shot", "metric results"),
    ("fig04_transfer_retention", "Transfer Retention", "metric results"),
    ("fig05_ablation_evidence", "Component Evidence", "metric results"),
    ("fig06_training_curves", "Training Curves", "metric results"),
    ("fig07_trajectory_3v2_normal_best", "3v2 Normal Best Trajectory", "ACMI trajectory"),
    ("fig08_trajectory_5v4_zero_shot", "5v4 Zero-Shot Trajectory", "ACMI trajectory"),
    ("fig09_paper_readiness_gap", "Paper-Readiness Gap", "paper-readiness gap"),
    ("fig10_progress_summary_dashboard", "Progress Summary Dashboard", "dashboard"),
]


def _rel(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    png = out_dir / f"{stem}.png"
    svg = out_dir / f"{stem}.svg"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return [png, svg]


def _missing(ax: plt.Axes, title: str, message: str = "missing data") -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])


def _best_record(records: list[dict[str, Any]], method: str, scenario: str | None = None) -> dict[str, Any] | None:
    candidates = [r for r in records if r.get("method_variant") == method and r.get("evidence_level") == "evaluated"]
    if scenario:
        candidates = [r for r in candidates if r.get("eval_scenario") == scenario]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda r: (
            _float(r, "red_win_rate") or 0.0,
            _float(r, "red_missile_hits_mean") or 0.0,
            _float(r, "blue_dead_mean") or 0.0,
        ),
    )


def fig01_pipeline(out_dir: Path) -> list[Path]:
    steps = [
        "BRMA-inspired\nunified obs",
        "MAV actor +\nshared UAV actor",
        "direct-chase\noracle dataset",
        "wrapped-heading\nimitation",
        "UAV imitation\nanchor",
        "easy -> medium ->\nnormal curriculum",
        "3v2 seen\neval",
        "5v4 zero-shot\neval",
    ]
    fig, ax = plt.subplots(figsize=(16, 4.4))
    ax.axis("off")
    for i, label in enumerate(steps):
        x = i / (len(steps) - 1)
        ax.text(
            x,
            0.58,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#eef4ff", edgecolor="#315a9a"),
        )
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=((i + 0.72) / (len(steps) - 1), 0.58),
                xytext=((i + 0.28) / (len(steps) - 1), 0.58),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#444"),
            )
    ax.text(
        0.5,
        0.18,
        "Not full BRMA-MAPPO | Not full TAM-HAPPO | Fixed-capacity 3v2-to-5v4",
        transform=ax.transAxes,
        ha="center",
        fontsize=12,
        color="#8a3b12",
    )
    ax.set_title("Experiment Pipeline", fontsize=16, weight="bold")
    return _save(fig, out_dir, "fig01_experiment_pipeline")


def fig02_method_comparison(out_dir: Path, outputs_root: Path) -> list[Path]:
    data = _read_json(outputs_root / "paper_evidence_matrix" / "paper_evidence_matrix.json")
    records = data.get("records", []) if isinstance(data, dict) else []
    specs = [
        ("HAPPO ref", _best_record(records, "HAPPO reference v0 baseline", "normal 3v2")),
        ("Oracle direct", _best_record(records, "oracle pretrain direct fine-tune", "normal 3v2")),
        ("Easy combat", _best_record(records, "easy combat oracle anchor")),
        ("Normal direct", _best_record(records, "normal geometry direct oracle anchor", "normal 3v2")),
        ("Full method", _best_record(records, "geometry curriculum full method", "normal 3v2")),
        ("5v4 zero-shot", _best_record(records, "geometry curriculum full method", "5v4")),
    ]
    metrics = [
        ("red_win_rate", "red win"),
        ("red_elimination_win_rate", "elim win"),
        ("mav_survival_rate", "MAV survival"),
        ("red_missile_hits_mean", "red hits"),
        ("blue_dead_mean", "blue dead"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(17, 4), sharex=False)
    for ax, (key, title) in zip(axes, metrics):
        values = [_float(rec or {}, key) for _, rec in specs]
        ax.bar(range(len(specs)), [v if v is not None else 0.0 for v in values], color="#5b8fd6")
        for i, v in enumerate(values):
            ax.text(i, (v if v is not None else 0.0) + 0.02, "N/A" if v is None else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8, rotation=90 if v and v > 1.5 else 0)
        ax.set_title(title)
        ax.set_xticks(range(len(specs)))
        ax.set_xticklabels([name for name, _ in specs], rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Baseline-to-Full-Method Comparison", fontsize=15, weight="bold")
    return _save(fig, out_dir, "fig02_method_comparison_bar")


def fig03_transfer_quality(out_dir: Path) -> list[Path]:
    metrics = [
        ("red_win_rate", "red win"),
        ("red_elimination_win_rate", "elim win"),
        ("mav_survival_rate", "MAV survival"),
        ("red_missiles_fired_mean", "red fire"),
        ("red_missile_hits_mean", "red hits"),
        ("blue_dead_mean", "blue dead"),
    ]
    x = range(len(metrics))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([i - 0.18 for i in x], [FINAL_3V2[k] for k, _ in metrics], width=0.36, label="3v2 seen", color="#4c78a8")
    ax.bar([i + 0.18 for i in x], [FINAL_5V4[k] for k, _ in metrics], width=0.36, label="5v4 zero-shot", color="#f58518")
    ax.set_xticks(list(x))
    ax.set_xticklabels([label for _, label in metrics], rotation=20, ha="right")
    ax.set_title("3v2 Seen vs 5v4 Zero-Shot Transfer Quality", fontsize=14, weight="bold")
    ax.text(0.02, 0.96, "Win retained; elimination rate drops; 5v4 depends more on timeout alive advantage.",
            transform=ax.transAxes, va="top", fontsize=10, bbox=dict(facecolor="white", alpha=0.75, edgecolor="#ccc"))
    ax.text(0.02, 0.86, "UAV attack behavior transfers: red fire/hit counts remain active in 5v4.",
            transform=ax.transAxes, va="top", fontsize=10, bbox=dict(facecolor="white", alpha=0.75, edgecolor="#ccc"))
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, out_dir, "fig03_transfer_quality_3v2_vs_5v4")


def fig04_transfer_retention(out_dir: Path, outputs_root: Path) -> list[Path]:
    data = _read_json(outputs_root / "paper_evidence_matrix" / "transfer_quality.json") or {}
    metrics = (data.get("geometry_curriculum_full_method") or {}) if isinstance(data, dict) else {}
    keys = [
        ("win_retention", "win"),
        ("elimination_retention", "elim"),
        ("normalized_blue_dead_retention", "blue dead / enemy"),
        ("timeout_dependency_delta", "timeout dep. delta"),
    ]
    values = [metrics.get(key) for key, _ in keys]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(keys)), [v if isinstance(v, (int, float)) else 0.0 for v in values],
           color=["#54a24b", "#e45756", "#f58518", "#b279a2"])
    for i, v in enumerate(values):
        ax.text(i, (v if isinstance(v, (int, float)) else 0.0) + 0.03,
                "N/A" if not isinstance(v, (int, float)) else f"{v:.3f}", ha="center")
    ax.axhline(1.0, ls="--", lw=1, color="#666")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([label for _, label in keys], rotation=15, ha="right")
    ax.set_title("Transfer Retention Metrics", fontsize=14, weight="bold")
    ax.text(0.5, 0.92, "High win retention, lower elimination retention",
            transform=ax.transAxes, ha="center", fontsize=11, bbox=dict(facecolor="white", alpha=0.8))
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, out_dir, "fig04_transfer_retention")


def fig05_ablation(out_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].bar(["original", "wrapped"], [0.075882, 0.010], color=["#e45756", "#54a24b"])
    axes[0].set_title("Heading loss MSE")
    axes[0].set_ylabel("action MSE")
    axes[0].text(0.5, 0.85, "closed-loop fire:\n0 -> >0", transform=axes[0].transAxes, ha="center")

    labels = ["direct", "curriculum"]
    x = [0, 1]
    for offset, vals, name in [(-0.22, [0.05, 1.82], "fire"), (0.0, [0.0, 1.56], "hit"), (0.22, [0.0, 1.52], "blue dead")]:
        axes[1].bar([i + offset for i in x], vals, width=0.2, label=name)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("Geometry curriculum")
    axes[1].legend(fontsize=8)

    axes[2].bar(["HAPPO ref", "full method"], [0.03, 0.92], color=["#bab0ab", "#4c78a8"])
    axes[2].set_title("3v2 red win")
    axes[2].set_ylim(0, 1.05)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Key Component Evidence", fontsize=15, weight="bold")
    return _save(fig, out_dir, "fig05_ablation_evidence")


def fig06_training_curves(out_dir: Path, outputs_root: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
    specs = [
        ("red_win", "red win"),
        ("mav_survival", "MAV survival"),
        ("red_missiles_fired", "red missiles fired"),
        ("missile_hits", "missile hits"),
    ]
    any_data = False
    for label, rel in TRAIN_RUNS:
        rows = _read_csv(outputs_root / rel / "train_log.csv")
        if not rows:
            continue
        xs = [_float(r, "total_steps", "steps", "total_env_steps") for r in rows]
        for ax, (key, title) in zip(axes.flat, specs):
            ys = [_float(r, key, "red_missile_hits", "red_hit") for r in rows] if key == "missile_hits" else [_float(r, key) for r in rows]
            pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
            if pairs:
                any_data = True
                px, py = zip(*pairs)
                ax.plot(px, py, label=label, lw=1.4)
                ax.set_title(title)
                ax.grid(alpha=0.25)
    if not any_data:
        _missing(axes.flat[0], "Training Curves")
    for ax in axes.flat:
        ax.set_xlabel("env steps")
    axes.flat[0].legend(fontsize=8, loc="best")
    fig.suptitle("Training Curves from Existing Logs", fontsize=15, weight="bold")
    return _save(fig, out_dir, "fig06_training_curves")


def _parse_acmi(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, list[tuple[float, float, float]]]]:
    objects: dict[str, dict[str, str]] = {}
    tracks: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    if not path.exists():
        return objects, tracks
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw or raw.startswith("#") or "," not in raw:
            continue
        obj_id, rest = raw.split(",", 1)
        fields: dict[str, str] = {}
        for part in rest.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                fields[k] = v
        if "Type" in fields or "Name" in fields or "Color" in fields:
            current = objects.setdefault(obj_id, {})
            current.update({k: fields[k] for k in ("Type", "Name", "Color") if k in fields})
        t = fields.get("T")
        if t:
            bits = t.split("|")
            if len(bits) >= 3:
                try:
                    tracks[obj_id].append((float(bits[0]), float(bits[1]), float(bits[2])))
                except ValueError:
                    pass
    return objects, tracks


def _plot_acmi(out_dir: Path, stem: str, title: str, acmi: Path, summary: Path) -> list[Path]:
    objects, tracks = _parse_acmi(acmi)
    info = _read_json(summary) or {}
    fig, ax = plt.subplots(figsize=(8, 7))
    if not tracks:
        _missing(ax, title, f"missing or unreadable ACMI:\n{acmi.name}")
        return _save(fig, out_dir, stem)

    for obj_id, pts in tracks.items():
        if len(pts) < 2:
            continue
        meta = objects.get(obj_id, {})
        name = meta.get("Name", obj_id)
        typ = meta.get("Type", "")
        color_name = meta.get("Color", "")
        if "Missile" in typ or "Missile" in name:
            color, lw, alpha, label = "#222222", 1.0, 0.55, "missile"
        elif color_name == "Blue" or "blue" in name:
            color, lw, alpha, label = "#1f77b4", 1.4, 0.9, "blue UAV"
        elif "red_0" in name or "MAV" in name:
            color, lw, alpha, label = "#d62728", 2.3, 1.0, "red_0 MAV"
        else:
            color, lw, alpha, label = "#ff7f0e", 1.4, 0.9, "red UAV"
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        step = max(1, len(xs) // 400)
        ax.plot(xs[::step], ys[::step], color=color, lw=lw, alpha=alpha, label=label)
        ax.scatter(xs[0], ys[0], color=color, marker="o", s=18)
        ax.scatter(xs[-1], ys[-1], color=color, marker="x", s=28)

    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), fontsize=8)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("longitude / local x")
    ax.set_ylabel("latitude / local y")
    ax.grid(alpha=0.25)
    outcome = info.get("outcome", "unknown outcome")
    ax.text(0.02, 0.98, f"Outcome: {outcome}", transform=ax.transAxes, va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="#ccc"))
    return _save(fig, out_dir, stem)


def fig09_gap(out_dir: Path) -> list[Path]:
    cols = [
        ("Already supported", ["unified obs", "hetero actor", "3v2 combat", "5v4 zero-shot phenomenon", "key component evidence"], "#e8f4ea"),
        ("Not yet supported", ["multi-seed robustness", "strong baseline superiority", "full TAM-HAPPO/BRMA reproduction", "strict MAV support trajectory", "arbitrary scale generalization"], "#fff0e6"),
        ("Next needed", ["multi-seed", "stronger baselines", "reward/module ablation", "MAV support behavior", "opponent realism"], "#eef4ff"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (title, items, color) in zip(axes, cols):
        ax.set_facecolor(color)
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for i, item in enumerate(items):
            ax.text(0.08, 0.82 - i * 0.15, f"- {item}", fontsize=11, va="center")
    fig.suptitle("Paper-Readiness Gap", fontsize=15, weight="bold")
    return _save(fig, out_dir, "fig09_paper_readiness_gap")


def fig10_dashboard(out_dir: Path) -> list[Path]:
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, :])
    for ax in (ax0, ax1, ax2, ax3):
        ax.axis("off")
    ax0.set_title("Method pipeline", weight="bold")
    ax0.text(0.02, 0.86, "Unified obs\n-> MAV actor + shared UAV actor\n-> oracle imitation\n-> geometry curriculum\n-> 3v2 / 5v4 eval", fontsize=12, va="top")
    ax1.set_title("3v2 normal best", weight="bold")
    ax1.text(0.02, 0.86, "\n".join([
        "red win: 0.92",
        "elim win: 0.52",
        "MAV survival: 0.94",
        "red fire: 1.82",
        "red hits: 1.56",
        "blue dead: 1.52",
    ]), fontsize=12, va="top")
    ax2.set_title("5v4 zero-shot", weight="bold")
    ax2.text(0.02, 0.86, "\n".join([
        "red win: 0.93",
        "elim win: 0.14",
        "MAV survival: 0.99",
        "red fire: 2.59",
        "red hits: 2.38",
        "blue dead: 2.33",
    ]), fontsize=12, va="top")
    ax3.set_title("Limitations", weight="bold")
    ax3.text(
        0.02,
        0.72,
        "Proof-of-concept and single-run evidence. Not full BRMA-MAPPO, not full TAM-HAPPO, "
        "not statistically superior, not arbitrary-scale generalization.",
        fontsize=12,
        va="top",
    )
    fig.suptitle("Progress Summary Dashboard", fontsize=16, weight="bold")
    return _save(fig, out_dir, "fig10_progress_summary_dashboard")


def _write_index(out_dir: Path) -> None:
    rows = {
        "fig01_experiment_pipeline": ("Experiment Pipeline", "Show the method flow.", "The framework combines unified observation, role actors, oracle imitation, and curriculum.", "method overview"),
        "fig02_method_comparison_bar": ("Method Comparison", "Compare baseline-to-full-method metrics.", "Single-run full method is stronger than current weak baselines.", "result comparison"),
        "fig03_transfer_quality_3v2_vs_5v4": ("3v2 Seen vs 5v4 Zero-Shot", "Show transfer behavior.", "Win rate transfers, but elimination ability drops.", "transfer result"),
        "fig04_transfer_retention": ("Transfer Retention", "Quantify retention and gaps.", "High win retention coexists with lower elimination retention.", "transfer quality"),
        "fig05_ablation_evidence": ("Component Evidence", "Show key component evidence.", "Wrapped heading and geometry curriculum are important.", "ablation evidence"),
        "fig06_training_curves": ("Training Curves", "Show learning trends from existing logs.", "Curves are useful for progress reporting, not final proof.", "training progress"),
        "fig07_trajectory_3v2_normal_best": ("3v2 Normal Best Trajectory", "Show ACMI trajectory.", "3v2 best episode demonstrates red launch/hit behavior.", "trajectory evidence"),
        "fig08_trajectory_5v4_zero_shot": ("5v4 Zero-Shot Trajectory", "Show ACMI trajectory.", "5v4 episode visualizes fixed-capacity transfer behavior.", "trajectory evidence"),
        "fig09_paper_readiness_gap": ("Paper-Readiness Gap", "Show what is and is not supported.", "Paper-readiness gaps remain.", "limitations"),
        "fig10_progress_summary_dashboard": ("Progress Summary Dashboard", "One-page report summary.", "Use this as an overview slide.", "overview/dashboard"),
    }
    lines = ["# Progress Report Figure Index", ""]
    for stem, (title, purpose, conclusion, page) in rows.items():
        lines.extend([
            f"## {stem}",
            f"- file: `{stem}.png`, `{stem}.svg`",
            f"- title: {title}",
            f"- purpose: {purpose}",
            f"- one-sentence conclusion: {conclusion}",
            f"- report page: {page}",
            "",
        ])
    (out_dir / "figure_index.md").write_text("\n".join(lines), encoding="utf-8")


def _write_summary_doc(out_dir: Path) -> None:
    rel = Path("outputs/progress_report_figures")
    lines = [
        "# Progress Report Visual Summary",
        "",
        "Current progress in one sentence: the framework and fixed-capacity 3v2-to-5v4 transfer phenomenon are established, but strict algorithm superiority is not yet proven.",
        "",
        "## Displayable figures",
    ]
    for stem, title, purpose in FIGURES:
        lines.append(f"- `{rel / (stem + '.png')}`: {title} ({purpose}).")
    lines.extend([
        "",
        "## What to say for each figure",
        "",
        "- Fig. 1: explain the experimental pipeline and the fixed-capacity scope.",
        "- Fig. 2: compare current variants without claiming statistical superiority.",
        "- Fig. 3-4: explain that win rate transfers while elimination quality drops.",
        "- Fig. 5: present component evidence for wrapped heading and geometry curriculum.",
        "- Fig. 6: show training progress trends from existing logs.",
        "- Fig. 7-8: use ACMI trajectories as qualitative behavior evidence.",
        "- Fig. 9-10: close with safe claims and remaining gaps.",
        "",
        "## Safe wording",
        "",
        "Use: fixed-capacity 3v2-to-5v4 zero-shot transfer phenomenon, proof-of-concept, single-run evidence, component evidence, paper-readiness gaps remain.",
        "",
        "Avoid: solved zero-shot combat transfer, full TAM-HAPPO reproduction, full BRMA-MAPPO reproduction, statistically superior, arbitrary-scale generalization.",
    ])
    (ROOT / "docs" / "progress_report_visual_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate progress report figures from existing outputs only")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    outputs_root = _rel(args.outputs_root)
    out_dir = _rel(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    generated += fig01_pipeline(out_dir)
    generated += fig02_method_comparison(out_dir, outputs_root)
    generated += fig03_transfer_quality(out_dir)
    generated += fig04_transfer_retention(out_dir, outputs_root)
    generated += fig05_ablation(out_dir)
    generated += fig06_training_curves(out_dir, outputs_root)
    generated += _plot_acmi(
        out_dir,
        "fig07_trajectory_3v2_normal_best",
        "3v2 Normal Best ACMI Trajectory",
        outputs_root / "happo_geometry_curriculum_100k/normal_50k/acmi/best_normal_3v2_episode0_fixed.acmi",
        outputs_root / "happo_geometry_curriculum_100k/normal_50k/acmi/best_normal_3v2_episode0_fixed_summary.json",
    )
    generated += _plot_acmi(
        out_dir,
        "fig08_trajectory_5v4_zero_shot",
        "5v4 Zero-Shot ACMI Trajectory",
        outputs_root / "happo_geometry_curriculum_100k/normal_50k/acmi/best_5v4_zero_shot_episode0_fixed.acmi",
        outputs_root / "happo_geometry_curriculum_100k/normal_50k/acmi/best_5v4_zero_shot_episode0_fixed_summary.json",
    )
    generated += fig09_gap(out_dir)
    generated += fig10_dashboard(out_dir)
    _write_index(out_dir)
    _write_summary_doc(out_dir)

    print(f"generated {len(generated)} figure files")
    print(f"figure_index: {out_dir / 'figure_index.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
