# Evaluation Metrics and ACMI Export

## Why split red_win_rate

The current `red_win_rate` bundles two distinct outcomes:
1. Red eliminated all blue aircraft (elimination win)
2. Timeout occurred and red had more alive aircraft (alive advantage win)

These are qualitatively different. An elimination win means red actively killed blue.
A timeout + alive advantage win means red survived longer, which the current 3v2
setup favours by default (red has 3 vs blue 2).

## New Granular Metrics

- `red_elimination_win_rate`: fraction of episodes ending in `red_win_elimination`
- `blue_elimination_win_rate`: fraction ending in `blue_win_elimination`
- `red_timeout_alive_advantage_rate`: timeout, red has more alive
- `blue_timeout_alive_advantage_rate`: timeout, blue has more alive
- `timeout_draw_rate`: timeout with equal alive counts
- `kill_death_ratio`: blue_dead / red_dead

## Current Baseline Reality

The `alive_done_fix` baseline shows `red_win_rate=0.70` but
`red_elimination_win_rate=0.00` — all wins are from timeout + alive advantage.
This is not a combat victory but a survival advantage from asymmetric force size.

## ACMI Export

Generate a single-episode Tacview ACMI file:
```bash
python scripts/export_one_eval_acmi.py
```

View with Tacview:
Open `outputs/acmi/alive_done_fix_3v2_episode0.acmi`

ACMI is for visual inspection only — not a training metric. This first version
exports aircraft trajectories only (no missiles).
