# Current Baseline and Environment Status

## 1. Pipeline Baseline

**Status: WORKING.**

- Train → eval → save/load pipeline runs end-to-end.
- `train_mappo_baseline.py` + `eval_mappo_zero_shot.py` functional.
- `ExperimentSpec` framework with multiple runners.
- ACMI export (one-episode, debug) with fixed death display.
- Granular eval metrics (elimination win rate, kill/death ratio).
- Periodic eval during training with best-checkpoint tracking.
- Per-role action logging (mav/uav action stats).
- Multiple opponent modes: rule_nearest, greedy_fsm, brma_rule.

## 2. Learning Baseline

**Status: RUNNABLE but NOT YET an effective combat policy.**

| Run | Opponent | Config | Steps | red_win | Outcome |
|---|---|---|---|---|---|
| alive_done_fix 50k | rule_nearest | 3v2 | 50k | 0.70 | FALSE POSITIVE — timeout wins only, rule_nearest does not attack |
| paper-aligned 50k | brma_rule | 3v2 no_trim | 50k | 0.00 | Red eliminated from episode 3 |
| protocol-aligned 200k | brma_rule | 3v2 no_trim | 200k | 0.00 | Red eliminated from episode 3; 200+ episodes no recovery |
| **2v2 homogeneous 200k** | **brma_rule** | **2v2 F-16 only** | **200k** | **0.00 draw** | **Red alive all episodes; zero kills both sides** |

**Key finding:** shared MLP MAPPO + brma_legacy can survive against brma_rule
in 2v2 homogeneous F-16-only setting. The 3v2 heterogeneous failure is likely
related to the unarmed MAV (F-22 or F-16-surrogate) being targeted first.

The shared MLP MAPPO is the current formal baseline implementation.
It is runnable but has not yet produced a combat-effective policy.

## 3. Environment Status

- brma_rule opponent functional and fires missiles.
- F-22 fixed-action test: survives 200 steps but shows max_roll=180 deg (warning).
- no_mav_trim config available.
- 3v2 heterogeneous failure cause not yet attributed.
- Audit needed: death order, death reason, blue target preference,
  F-22 vs F-16 MAV surrogate comparison.

## 4. Next Step

Run failure mode audit, then 1M shared MLP baseline if no blocking issues.
