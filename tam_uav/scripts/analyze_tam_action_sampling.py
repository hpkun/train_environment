"""Stream TAM action logs and compare sampled MAV bins with logged policy modes."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, deque
from pathlib import Path


AXES = ("throttle", "aileron", "elevator", "rudder")
NEUTRAL = (39, 20, 20, 20)


def _summary(counter: Counter, neutral: int) -> dict:
    count = sum(counter.values())
    if not count:
        return {"count": 0, "mean": None, "std": None, "dominant_bin": None,
                "mean_abs_deviation_from_neutral": None}
    mean = sum(value * n for value, n in counter.items()) / count
    variance = sum((value - mean) ** 2 * n for value, n in counter.items()) / count
    return {
        "count": count,
        "mean": mean,
        "std": math.sqrt(variance),
        "dominant_bin": counter.most_common(1)[0][0],
        "mean_abs_deviation_from_neutral": sum(
            abs(value - neutral) * n for value, n in counter.items()
        ) / count,
        "top5": counter.most_common(5),
    }


def _stream_mav_actions(path: Path):
    counters = [Counter() for _ in AXES]
    predeath = [Counter() for _ in AXES]
    episodes = 0
    inferred_deaths = 0
    if not path.exists():
        return counters, predeath, episodes, inferred_deaths
    with path.open(encoding="utf-8-sig", errors="replace") as handle:
        header = next(handle, "").rstrip("\r\n").split(",")
        index = {name: header.index(name) for name in (
            "episode_id", "step", "agent_id",
            *[f"action_index_{axis}" for axis in range(4)],
        ) if name in header}
        required = {"episode_id", "step", "agent_id"}
        if not required <= index.keys():
            # Small legacy fixtures may omit episode and step; parse them normally.
            handle.seek(0)
            for row in csv.DictReader(handle):
                if row.get("agent_id") != "red_0":
                    continue
                for axis in range(4):
                    value = row.get(f"action_index_{axis}")
                    if value not in (None, ""):
                        counters[axis][int(float(value))] += 1
            return counters, predeath, episodes, inferred_deaths
        max_index = max(index.values())
        current_episode = None
        tail = deque(maxlen=50)
        last_step = -1

        def finish_episode():
            nonlocal episodes, inferred_deaths
            if current_episode is None:
                return
            episodes += 1
            if last_step < 999:
                inferred_deaths += 1
                for action in tail:
                    for axis, value in enumerate(action):
                        if value is not None:
                            predeath[axis][value] += 1

        for line in handle:
            fields = line.rstrip("\r\n").split(",", max_index + 1)
            if len(fields) <= max_index or fields[index["agent_id"]] != "red_0":
                continue
            episode = fields[index["episode_id"]]
            if current_episode is not None and episode != current_episode:
                finish_episode()
                tail.clear()
            current_episode = episode
            try:
                last_step = int(float(fields[index["step"]]))
            except ValueError:
                last_step = -1
            action = []
            for axis in range(4):
                field_index = index.get(f"action_index_{axis}")
                try:
                    value = int(float(fields[field_index])) if field_index is not None else None
                except (ValueError, TypeError):
                    value = None
                action.append(value)
                if value is not None:
                    counters[axis][value] += 1
            tail.append(tuple(action))
        finish_episode()
    return counters, predeath, episodes, inferred_deaths


def _last_train_row(path: Path):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        last = {}
        for last in csv.DictReader(handle):
            pass
        return last


def analyze_action_sampling(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    counters, predeath, episodes, inferred_deaths = _stream_mav_actions(
        run_dir / "rich_logs" / "tam_action_timeseries.csv"
    )
    sampled = {
        name: _summary(counter, NEUTRAL[axis])
        for axis, (name, counter) in enumerate(zip(AXES, counters))
    }
    predeath_summary = {
        name: _summary(counter, NEUTRAL[axis])
        for axis, (name, counter) in enumerate(zip(AXES, predeath))
    }
    train = _last_train_row(run_dir / "train_log.csv")
    dominant_fields = [f"dominant_bin_mav_{axis}" for axis in AXES]
    has_all_dominant = all(train.get(field) not in (None, "") for field in dominant_fields)
    result = {
        "run_dir": str(run_dir),
        "sampled_bins": sampled,
        "argmax_comparison": {
            "available": has_all_dominant,
            "source": "last train_log aggregate dominant bins" if has_all_dominant else None,
            "argmax_bins": (
                {axis: int(float(train[field])) for axis, field in zip(AXES, dominant_fields)}
                if has_all_dominant else None
            ),
            "per_step_argmax_available": False,
        },
        "pre_death_50_steps": {
            "available": inferred_deaths > 0,
            "death_detection": "episode ended before step 999 (legacy-log inference)",
            "inferred_death_episodes": inferred_deaths,
            "axes": predeath_summary,
        },
        "episodes_seen": episodes,
        "limitations": [
            "legacy rich logs do not contain per-step argmax probabilities",
            "legacy rich logs do not contain explicit death markers",
        ],
    }
    return result


def _markdown(result: dict) -> str:
    lines = ["# TAM action sampling analysis", "", f"Run: `{result['run_dir']}`", "",
             "## Sampled MAV bins", ""]
    for axis, values in result["sampled_bins"].items():
        lines.append(
            f"- {axis}: count={values['count']}, mean={values['mean']}, "
            f"std={values['std']}, dominant={values['dominant_bin']}, "
            f"mean_abs_neutral_deviation={values['mean_abs_deviation_from_neutral']}"
        )
    lines += ["", "## Argmax comparison", "", json.dumps(
        result["argmax_comparison"], ensure_ascii=False, indent=2
    ), "", "## Pre-death 50 steps", "", json.dumps(
        result["pre_death_50_steps"], ensure_ascii=False, indent=2
    )]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = analyze_action_sampling(args.run_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "action_sampling.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "action_sampling.md").write_text(_markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
