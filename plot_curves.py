"""
plot_curves.py —— 离线绘制论文级奖励曲线与胜率曲线

用法:
    python plot_curves.py                                    # 默认读取 vanilla_training_log.csv
    python plot_curves.py --input results/my_run.csv         # 也兼容 results/ 精简格式
    python plot_curves.py --input vanilla_training_log.csv --smooth 20 --out ./figures

输出:
    ./reward_curve.png     (奖励均值 ± 标准差阴影)
    ./win_rate_curve.png   (每迭代滑动窗口胜率)
    ./combined_curves.png  (双栏合并图)
"""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---- 论文级样式 ----
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})


def _convolve(x: np.ndarray, window: int) -> np.ndarray:
    """Boxcar smoothing, matching the paper's sliding-window style."""
    if window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def load_results(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load steps, reward_mean, reward_std, win_rate from CSV.

    Auto-detects column names:
      - training log: Step, RedMeanReward, RedRewardStd, WinRateRecent
      - results log:  Step, RewardMean, RewardStd, WinRateRecent

    Returns (steps, r_mean, r_std, wr).
    """
    steps, r_mean, r_std, wr = [], [], [], []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        # Map column names to canonical keys
        col_step = "Step"
        col_mean = "RedMeanReward" if "RedMeanReward" in header else "RewardMean"
        col_std = "RedRewardStd" if "RedRewardStd" in header else "RewardStd"
        col_wr = "WinRateRecent"
        for row in reader:
            steps.append(int(row[col_step]))
            r_mean.append(float(row[col_mean]))
            r_std.append(float(row.get(col_std, 0.0)))
            wr.append(float(row.get(col_wr, 0.0)))
    return (
        np.array(steps, dtype=np.float64),
        np.array(r_mean, dtype=np.float64),
        np.array(r_std, dtype=np.float64),
        np.array(wr, dtype=np.float64),
    )


def plot_reward(steps: np.ndarray, r_mean: np.ndarray, r_std: np.ndarray,
                smooth: int, out_dir: str):
    """Reward curve with ±std shading (paper Figure 7 style)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    if smooth > 1:
        steps_s = _convolve(steps, smooth)
        mean_s = _convolve(r_mean, smooth)
        std_s = _convolve(r_std, smooth)
    else:
        steps_s, mean_s, std_s = steps, r_mean, r_std

    ax.fill_between(steps_s, mean_s - std_s, mean_s + std_s,
                    alpha=0.25, color="#2c7bb6", linewidth=0)
    ax.plot(steps_s, mean_s, color="#2c7bb6", linewidth=1.2, label="Team Reward (smoothed)")

    # Also plot raw for reference
    ax.plot(steps, r_mean, color="#2c7bb6", alpha=0.18, linewidth=0.5, label="Raw")

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Red Team Reward")
    ax.set_title("Training Reward Curve")
    ax.legend()

    path = os.path.join(out_dir, "reward_curve.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_win_rate(steps: np.ndarray, wr: np.ndarray, smooth: int, out_dir: str):
    """Win rate curve (paper Figure 11 style)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    if smooth > 1:
        steps_s = _convolve(steps, smooth)
        wr_s = _convolve(wr, smooth)
    else:
        steps_s, wr_s = steps, wr

    ax.plot(steps_s, wr_s, color="#d7191c", linewidth=1.2, label="Win Rate (smoothed)")
    ax.plot(steps, wr, color="#d7191c", alpha=0.18, linewidth=0.5, label="Raw")
    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, label="50% baseline")

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Win Rate (per-iteration sliding window)")
    ax.set_title("Training Win Rate Curve")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()

    path = os.path.join(out_dir, "win_rate_curve.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_combined(steps: np.ndarray, r_mean: np.ndarray, r_std: np.ndarray,
                  wr: np.ndarray, smooth: int, out_dir: str):
    """Combined figure: reward (top) + win rate (bottom)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    if smooth > 1:
        steps_s = _convolve(steps, smooth)
        mean_s = _convolve(r_mean, smooth)
        std_s = _convolve(r_std, smooth)
        wr_s = _convolve(wr, smooth)
    else:
        steps_s, mean_s, std_s, wr_s = steps, r_mean, r_std, wr

    # Top: reward
    ax1.fill_between(steps_s, mean_s - std_s, mean_s + std_s,
                     alpha=0.25, color="#2c7bb6", linewidth=0)
    ax1.plot(steps_s, mean_s, color="#2c7bb6", linewidth=1.2)
    ax1.plot(steps, r_mean, color="#2c7bb6", alpha=0.15, linewidth=0.4)
    ax1.set_ylabel("Red Team Reward")
    ax1.set_title("Training Curves (MAPPO 6v6)")

    # Bottom: win rate
    ax2.plot(steps_s, wr_s, color="#d7191c", linewidth=1.2)
    ax2.plot(steps, wr, color="#d7191c", alpha=0.15, linewidth=0.4)
    ax2.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8)
    ax2.set_xlabel("Environment Steps")
    ax2.set_ylabel("Win Rate")
    ax2.set_ylim(-0.02, 1.02)

    path = os.path.join(out_dir, "combined_curves.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def main(argv: Sequence[str] | None = None):
    import argparse as _argparse
    parser = _argparse.ArgumentParser(description="Plot training curves from CSV")
    parser.add_argument("--input", default="vanilla_training_log.csv",
                        help="Path to CSV (default: vanilla_training_log.csv; "
                             "also compatible with results/vanilla_mappo_results.csv)")
    parser.add_argument("--smooth", type=int, default=10,
                        help="Smoothing window size (default: 10)")
    parser.add_argument("--out", default=None,
                        help="Output directory for figures (default: same dir as --input)")
    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found.")
        return

    out_dir = args.out or os.path.dirname(args.input) or "."
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {args.input} ...")
    steps, r_mean, r_std, wr = load_results(args.input)
    print(f"  {len(steps)} data points, "
          f"steps range [{steps[0]:.0f}, {steps[-1]:.0f}]")

    plot_reward(steps, r_mean, r_std, args.smooth, out_dir)
    plot_win_rate(steps, wr, args.smooth, out_dir)
    plot_combined(steps, r_mean, r_std, wr, args.smooth, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
