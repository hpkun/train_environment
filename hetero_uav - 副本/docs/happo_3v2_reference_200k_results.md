# HAPPO 3v2 Reference 200k Results

## 1. Experiment Definition

- Method: HAPPO reference v0.
- Scope: simplified HAPPO-style role-wise update, not full TAM-HAPPO.
- Action: high-level `[pitch, heading, speed]` retained.
- Missile/evasion: scripted environment mechanics retained.
- Temporal module: no GRU.
- Attention module: no attention.
- Reward: `happo_ref_v0`.
- Opponent: `brma_rule`.
- Train config: `uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0.yaml`.
- Eval configs: 3v2 HAPPO reference config and 5v4 main heterogeneous config.
- Output directory: `outputs/happo_3v2_reference_200k`.

## 2. Why This Experiment

The shared MLP 1M baseline mainly learned timeout survival rather than
effective combat. HAPPO reference v0 tests whether role-separated MAV/UAV
policies improve 3v2 behavior before adding GRU, attention, or full TAM-HAPPO
machinery.

## 3. Training Dynamics

Training completed 200000 environment steps with no NaN. The latest training
row reports:

- `avg_return=10.1644`;
- `red_win=1.0`;
- `blue_win=0.0`;
- `timeout=1.0`;
- `mav_survival=1.0`;
- `red_alive_final=3.0`;
- `blue_alive_final=2.0`;
- `red_missiles_fired=0`;
- `blue_missiles_fired=0`;
- `missile_hits=0`.

This is timeout survival, not effective air combat. During training, the log
does not show blue deaths or missile hits. The apparent red win is timeout
alive advantage rather than elimination.

The critic loss decreased from `4.531062` to `0.002849`, while MAV/UAV entropy
increased. Action saturation also increased, especially for the MAV in the
latest policy (`mav_action_saturation_rate=0.526042` in the last train row).

## 4. Best/Latest Checkpoint Evaluation

Each checkpoint was evaluated for 50 episodes.

| checkpoint | config | red_win | blue_win | timeout | MAV survival | blue_dead_mean | red_hit_mean |
|---|---|---:|---:|---:|---:|---:|---:|
| best | 3v2 | 0.00 | 0.92 | 0.16 | 0.00 | 0.22 | 0.22 |
| best | 5v4 | 0.00 | 1.00 | 0.36 | 0.00 | 0.56 | 0.56 |
| latest | 3v2 | 0.00 | 1.00 | 0.02 | 0.00 | 0.00 | 0.00 |
| latest | 5v4 | 0.00 | 0.98 | 0.04 | 0.00 | 0.14 | 0.14 |

The best checkpoint is better than latest in combat signal because it has some
blue deaths and red missile hits. It still fails the MAV survival objective and
does not produce red wins. Latest is worse: 3v2 has no blue deaths and no red
missile hits.

## 5. ACMI Observation

ACMI export succeeded for both best and latest 3v2 episode 0.

Best checkpoint:

- outcome: `blue_win_elimination`;
- final red alive: `0`;
- final blue alive: `1`;
- MAV alive: `false`;
- death order: `blue_1`, `red_0`, `red_2`, `red_1`;
- red missiles fired: `1`;
- red missile hits: `1`;
- red_0 minimum altitude: about `2489 m`;
- MAV action saturation: `0.0018`.

Latest checkpoint:

- outcome: `blue_win_elimination`;
- final red alive: `0`;
- final blue alive: `2`;
- MAV alive: `false`;
- death order: `red_0`, `red_1`, `red_2`;
- red missiles fired: `2`;
- red missile hits: `0`;
- red_0 minimum altitude: about `2488 m`;
- MAV action saturation: `0.5664`.

The ACMI summaries do not support a claim that the MAV learned stable rear
support behavior. In the exported episodes, the MAV dies and the red team loses.

## 6. Decision

Decision: **B. HAPPO v0 partially works but needs reward or observation
correction.**

Rationale:

- The 200k run is technically stable: train/eval, checkpointing, summary, and
  ACMI export work.
- It is not a complete failure because the best checkpoint has nonzero blue
  deaths and red missile hits.
- It is not strong enough to proceed to a 1M HAPPO reference run because MAV
  survival is zero under 50-episode checkpoint evaluation and red win rate is
  zero.
- The latest checkpoint appears worse than best, with higher action saturation
  and no 3v2 blue deaths.

Next work should inspect reward/observation/targeting before adding GRU,
attention, or full TAM-HAPPO.

## 7. Train/Eval Consistency Follow-Up

The latest training row and deterministic checkpoint evaluation disagree
strongly. The train row is a recent on-policy stochastic rollout window, while
the checkpoint evaluation is a saved-policy deterministic rollout. It is
therefore not valid to treat the latest train row timeout survival as an
effective combat result.

Before any 1M run, the required diagnostic sequence is:

- audit train/eval consistency;
- compare deterministic and stochastic policy modes;
- audit MAV failure modes.

If those diagnostics still show systematic MAV death or deterministic collapse,
the next step is to adjust the validation setup rather than increasing the
training horizon.

## 8. MAV Failure Gate

The post-200k MAV failure gate is complete. It adds death-event logging and
runs fixed-action, survival-ablation, and blue-target diagnostics.

Key results:

- death reasons are no longer only unknown; the dominant reason is
  `Crash_LowAlt`;
- F-22 fixed-action sweep is not stable;
- F-16 MAV surrogate is more stable in the fixed-action sweep but still fails
  the full HAPPO survival ablation;
- fixed safe MAV action does not keep the MAV alive;
- MAV action scaling does not keep the MAV alive;
- available missile metadata does not show blue systematically targeting the
  MAV;
- `run_1m_allowed = false`.

The current primary hypothesis is `f22_control_or_dynamics_instability`, with a
secondary concern that a fixed MAV can become predictable and vulnerable. The
next step is not more training; it is a narrow MAV control/dynamics stability
fix or a temporary F-16 MAV surrogate validation path.
