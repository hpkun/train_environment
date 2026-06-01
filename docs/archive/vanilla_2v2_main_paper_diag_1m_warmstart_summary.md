# Vanilla 2v2 paper-diagnostic 1M warm-start summary

This report is CSV-only. It does not modify code, run training, run evaluation,
reset an environment, or trigger JSBSim.

The second run is treated as a **500K->1M warm-start segment**: it starts from a
500K best checkpoint and adds another 500K env steps, but it is not a strict
optimizer-state checkpoint resume.

## 1. File coverage

| File | Rows | Launch diagnostics | ActionStdMean | Entropy |
|---|---:|---|---|---|
| `results/vanilla_2v2_main_paper_diag_500k_results.csv` | 250 | yes | yes | yes |
| `results/vanilla_2v2_main_paper_diag_500k_to_1m_results.csv` | 250 | yes | yes | yes |
| `logs/vanilla_2v2_main_paper_diag_500k.csv` | 250 | yes | yes | yes |
| `logs/vanilla_2v2_main_paper_diag_500k_to_1m.csv` | 250 | yes | yes | yes |

Result CSV columns:

```text
Step, Iteration, RedMeanReward, RedRewardStd, WinRateRecent, WinRateCumul,
RedMissiles, BlueMissiles, Episodes, RedWins, BlueWins, Draws,
RedAliveMean, BlueAliveMean, RedDeathsMissile, RedDeathsCrash,
BlueDeathsMissile, BlueDeathsCrash, RedMissileHits, BlueMissileHits,
RedMissileHitRate, BlueMissileHitRate, KD_Red, RWR, RewardVersion,
ActionStdMean, ActionStdMin, ActionStdMax, ActionLogStdMean, ActorLoss,
CriticLoss, Entropy, r_pitch, r_roll, r_alt, r_bound, r_vel, r_adv,
r_end, r_death, LaunchDiagRedGeometryOk, LaunchDiagBlueGeometryOk,
LaunchDiagRedLaunches, LaunchDiagBlueLaunches, LaunchDiagRedRangeOk,
LaunchDiagRedAoOk, LaunchDiagRedTaOk, LaunchDiagBlueRangeOk,
LaunchDiagBlueAoOk, LaunchDiagBlueTaOk, LaunchDiagRedEngagedBlocked,
LaunchDiagBlueEngagedBlocked, LaunchDiagRedCooldownBlocked,
LaunchDiagBlueCooldownBlocked, LaunchDiagRedKillCooldownBlocked,
LaunchDiagBlueKillCooldownBlocked, LaunchDiagRedLockMature,
LaunchDiagBlueLockMature, RedGeometryToLaunchRate,
BlueGeometryToLaunchRate, RedRangeToGeometryRate, BlueRangeToGeometryRate
```

The log CSVs contain the same diagnostic fields, with the same 250-row coverage.

## 2. 500K segment overview

From `results/vanilla_2v2_main_paper_diag_500k_results.csv`, final row:

| Metric | Value |
|---|---:|
| Final steps | 500,000 |
| Episodes | 1,554 |
| Red wins / Blue wins / Draws | 7 / 1,471 / 76 |
| Final RWR | 0.004505 |
| Final RedMeanReward | -31.5211 |
| Final RedMissileHitRate | 0.040250 |
| Final BlueMissileHitRate | 0.871547 |
| Final ActionStdMean | 0.600027 |
| Final Entropy | 0.907329 |

## 3. 500K->1M warm-start segment overview

From `results/vanilla_2v2_main_paper_diag_500k_to_1m_results.csv`, final row:

| Metric | Value |
|---|---:|
| Segment steps | 500,000 |
| Episodes | 455 |
| Red wins / Blue wins / Draws | 13 / 228 / 214 |
| Segment red win rate | 0.028571 |
| Final RedMeanReward | -1.2844 |
| Final RedMissileHitRate | 0.018178 |
| Final BlueMissileHitRate | 0.842271 |
| Final ActionStdMean | 1.117960 |
| Final Entropy | 1.528342 |
| Final M_red / M_blue | 12.32 / 1.14 |

## 4. Phase breakdown for second segment

Each phase uses rows with segment-local `Step` in `(phase_start, phase_end]`.
Mean fields are averaged over rows in the phase. Cumulative fields use
`phase_end - phase_start`, not sums.

| Phase | Mean RedMeanReward | Mean Entropy | Mean ActionStdMean | Mean ActorLoss | Mean CriticLoss | Mean RedMissiles | Mean BlueMissiles |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0-100K | -46.7773 | 0.9084 | 0.6015 | -0.0622 | 1.5445 | 8.72 | 1.32 |
| 100-200K | -35.8613 | 1.0629 | 0.7021 | -0.0666 | 1.2149 | 9.14 | 1.31 |
| 200-300K | -36.9769 | 1.1956 | 0.8016 | -0.0752 | 1.3989 | 8.76 | 1.41 |
| 300-400K | -44.9668 | 1.3204 | 0.9075 | -0.0810 | 1.5335 | 15.48 | 1.47 |
| 400-500K | -20.7122 | 1.4523 | 1.0364 | -0.0839 | 1.0553 | 11.96 | 1.34 |

| Phase | Red wins inc. | Blue wins inc. | Draws inc. | RedDeathsMissile delta | RedDeathsCrash delta | BlueDeathsMissile delta | BlueDeathsCrash delta | RedMissileHits delta | BlueMissileHits delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-100K | 5 | 54 | 32 | 118 | 11 | 23 | 0 | 23 | 118 |
| 100-200K | 1 | 40 | 42 | 90 | 10 | 14 | 0 | 14 | 90 |
| 200-300K | 4 | 51 | 44 | 119 | 10 | 26 | 0 | 26 | 119 |
| 300-400K | 2 | 56 | 39 | 128 | 13 | 18 | 0 | 18 | 128 |
| 400-500K | 1 | 27 | 57 | 79 | 8 | 12 | 0 | 12 | 79 |

## 5. Launch diagnostics

Judgment basis: launch diagnostic columns are per-iteration counts, not
cumulative counters. They frequently decrease from one CSV row to the next,
including high-count fields such as `LaunchDiagRedRangeOk`; therefore segment
totals below are sums over rows. `LaunchDiagBlueCooldownBlocked` is all zero in
the warm-start segment and is effectively both cumulative and per-iteration
zero.

| Segment | RedRangeOk | RedAoOk | RedTaOk | RedGeometryOk | RedLockMature | RedLaunches | RedCooldownBlocked | RedEngagedBlocked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-500K | 12,536,623 | 5,120,974 | 7,476,341 | 31,987 | 9,808 | 1,118 | 8,690 | 92,192 |
| 500K->1M warm-start | 5,210,363 | 2,779,490 | 9,511,451 | 165,050 | 46,153 | 5,116 | 41,014 | 210,916 |

| Segment | BlueRangeOk | BlueAoOk | BlueTaOk | BlueGeometryOk | BlueLockMature | BlueLaunches | BlueCooldownBlocked | BlueEngagedBlocked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-500K | 9,708,892 | 7,375,934 | 9,318,518 | 52,413 | 4,859 | 3,005 | 1,817 | 3,455,673 |
| 500K->1M warm-start | 4,919,817 | 5,212,001 | 10,508,299 | 10,837 | 751 | 634 | 0 | 643,827 |

| Segment | RedRangeToGeometryRate | RedGeometryToLockRate | RedLockToLaunchRate | RedGeometryToLaunchRate | BlueRangeToGeometryRate | BlueGeometryToLockRate | BlueLockToLaunchRate | BlueGeometryToLaunchRate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-500K | 0.002551 | 0.306625 | 0.113989 | 0.034952 | 0.005398 | 0.092706 | 0.618440 | 0.057333 |
| 500K->1M warm-start | 0.031677 | 0.279630 | 0.110849 | 0.030997 | 0.002203 | 0.069300 | 0.844208 | 0.058503 |

| Segment | Red/Blue launch ratio | Red/Blue hit rate ratio |
|---|---:|---:|
| 0-500K | 0.372047 | 0.046183 |
| 500K->1M warm-start | 8.069401 | 0.021582 |

Interpretation:

- Red geometry generation improved sharply in the warm-start segment:
  `RedRangeToGeometryRate` rose from 0.255% to 3.168%.
- Red launch volume also flipped from far below Blue to far above Blue, but
  conversion after geometry did not improve: `RedGeometryToLaunchRate` moved
  from 3.495% to 3.100%.
- Red hit quality worsened despite more launches: final Red hit rate dropped
  from 4.03% to 1.82%, and the Red/Blue hit-rate ratio dropped from 0.046 to
  0.022.
- Blue remains much more lethal per missile even though it launches less often
  in the warm-start segment.

## 6. Policy variance diagnosis

Action variance rose materially from the 500K endpoint to the 1M warm-start
endpoint:

- `ActionStdMean`: 0.6000 -> 1.1180, increase +0.5179, about +86%.
- `ActionStdMax`: 0.6091 -> 1.1482.
- `ActionLogStdMean`: -0.5108 -> 0.1112.

Entropy moved with the action standard deviation:

- `Entropy`: 0.9073 -> 1.5283, increase +0.6210.
- Warm-start phase means climb monotonically from 0.9084 to 1.4523.

`ActionStdMean > 1` appears in the final 100K phase mean and at the final row.
Because actions are normalized and later clamped in the policy path, std above
1 is large enough to plausibly cause frequent saturated samples. The CSV does
not include action clip/saturation counts, so this is a risk diagnosis rather
than a measured saturation rate.

There is a clear mixed-signal pattern:

- Reward improves strongly by the final warm-start row (`RedMeanReward`
  -31.52 -> -1.28).
- Red win rate improves versus the first 500K cumulative endpoint, but remains
  very low: 2.86% in the warm-start segment.
- Red creates many more launch opportunities and launches many more missiles,
  yet hit rate gets worse. This is consistent with reward/geometry progress
  without corresponding launch-quality progress.

## 7. Decision

Recommendation: **B. Pause, first do launch-quality diagnostics.**

Reasoning:

- Continuing unchanged to 1.5M / 2M may accumulate more behavior, but the key
  failure is now more specific than "not enough training": Red can generate
  geometry and launch, but launch quality and lethality are poor.
- Entropy/std is high enough to watch, but it is not yet the first thing to
  ablate. The stronger evidence is the launch-quality mismatch: high Red launch
  count, low Red hit rate, and very low Red/Blue hit-rate ratio.
- Paper mismatch audit remains important, but it will not explain the immediate
  warm-start symptom as directly as launch-quality diagnostics.

Suggested next non-training diagnostic scope:

- Add/read-only analyze action clipping or action saturation if already logged;
  otherwise plan instrumentation for a later code pass.
- Break Red launches by AO, TA, range, relative velocity, and missile
  probability at launch time.
- Compare Red launched-shot quality against Blue launched-shot quality using
  the same bins.
- Only after that decide between continuing unchanged and an entropy/std
  ablation.
