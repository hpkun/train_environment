# vanilla_2v2_main_entropy_diag_100k CSV Summary

## 1. File coverage

| Item | Value |
|---|---|
| Source file | `results/vanilla_2v2_main_entropy_diag_100k_results.csv` |
| Rows | 50 |
| Columns | 62 |
| All requested fields present? | Yes — zero missing |
| Log CSV available? | No (only results CSV found) |

## 2. Training overview

| Metric | Value |
|---|---|
| Iterations | 50 |
| Max total_env_steps | 100,000 |
| Total episodes | 390 |
| Final RWR (Red Win Rate) | **0.0000** |
| Final RewardVersion | `fixed_ta_alt_eq17_3dlos_v1` |

| Metric | First | Last | Mean | Max | Min |
|---|---|---|---|---|---|
| RedMeanReward | −74.52 | −78.67 | −76.82 | −72.83 | −80.52 |
| Entropy | 0.2177 | 0.3368 | 0.2730 | 0.3368 | 0.2177 |
| ActionStdMean | 0.3017 | 0.3395 | 0.3185 | 0.3395 | 0.3017 |
| ActionLogStdMean | −1.1984 | −1.0805 | −1.1447 | −1.0805 | −1.1984 |
| CriticLoss | 16.55 | 3.97 | 7.56 | 21.89 | 2.13 |
| ActorLoss | −0.0219 | −0.0352 | −0.0297 | −0.0219 | −0.0383 |

## 3. Entropy / std diagnostics

| Metric | First | Last | Δ | Status |
|---|---|---|---|---|
| Entropy | 0.2177 | 0.3368 | +0.1191 | Moderate rise, still in normal range for early training |
| ActionStdMean | 0.3017 | 0.3395 | +0.0378 | ~13% increase from init; not alarming yet |
| ActionLogStdMean | −1.1984 | −1.0805 | +0.1179 | Corresponding log-space drift |

**Judgment**: Std has drifted from 0.30 → 0.34 over 100K steps. This is a 13% increase — significant but not catastrophic. At this rate, projecting to 10M steps would give Std ≈ exp(−1.20 + 0.1179 × 100) ≈ extremely large. The entropy_coef = 0.05 without decay will continue pushing Std upward. **At 100K steps, no immediate intervention is required, but monitoring is essential.**

## 4. Missile launch diagnostics

### Cumulative totals over 100K steps

| Metric | Red | Blue | Red/Blue Ratio |
|---|---|---|---|
| RangeOk (agent-target pairs in range) | 3,208,665 | 2,377,482 | 1.35× |
| AoOk (AO < 45°) | 1,055,948 | 1,724,897 | 0.61× |
| TaOk (TA > 90°) | 1,056,736 | 1,698,901 | 0.62× |
| **GeometryOk** (all three) | **15** | **14,790** | **0.001×** |
| LockMature (≥ 0.25 s) | 1 | 1,533 | 0.001× |
| **Launches** | **1** | **854** | **0.001×** |
| EngagedBlocked | 57 | 962,022 | — |
| CooldownBlocked | 0 | 679 | — |
| KillCooldownBlocked | 0 | 0 | — |

### Derived rates

| Rate | Red | Blue |
|---|---|---|
| RangeToGeometryRate | 0.000005 | 0.0062 |
| GeometryToLockMatureRate | 0.067 | 0.104 |
| GeometryToLaunchRate | 0.067 | 0.058 |
| LockMatureToLaunchRate | 1.000 | 0.557 |
| RedBlueGeometryRatio | **0.0010** | — |
| RedBlueLaunchRatio | **0.0012** | — |

### Phase breakdown

| Phase | R_red | Ent | Std | CritLoss | M_red | M_blue | red_geo | red_launch | blue_geo | blue_launch |
|---|---|---|---|---|---|---|---|---|---|---|
| 0–20K | −74.8 | 0.233 | 0.306 | 14.70 | 0.0 | 2.4 | 0 | 0 | 4,173 | 242 |
| 20–40K | −78.1 | 0.247 | 0.310 | 8.42 | 0.0 | 2.4 | 0 | 0 | 4,067 | 238 |
| 40–60K | −77.4 | 0.261 | 0.315 | 6.63 | 0.0 | 2.2 | 0 | 0 | 2,977 | 167 |
| 60–80K | −74.5 | 0.293 | 0.325 | 5.33 | 0.0 | 2.0 | 0 | 0 | 1,959 | 113 |
| 80–100K | −78.8 | 0.321 | 0.334 | 3.81 | 0.0 | 1.9 | **15** | **1** | 1,423 | 86 |

## 5. Bottleneck diagnosis

### 1. Why is M_red ≈ 0?

**Answer: D then B.**

The bottleneck chain for Red is:

- **RangeOk**: 3.2M pairs — abundant. Red has no problem getting within 10 km of Blue.
- **AO < 45°**: 1.06M pairs — significant but only ~33% of range-ok pairs. Red often faces away from Blue.
- **TA > 90°**: 1.06M pairs — same magnitude as AO. Both angle conditions individually have ~1M hits.
- **GeometryOk (AO AND TA simultaneously)**: **only 15** — the AND is essentially never satisfied.

Red achieves AO and TA in ~1M pairs each, but they rarely overlap in the same physics frame for the same agent-target pair. This means Red's policy cannot simultaneously point at a Blue target while approaching from the rear hemisphere. The random/stochastic policy generates random headings/pitches that satisfy either AO or TA sporadically, but almost never both together.

In contrast, Blue's rule-based policy explicitly selects a tail-chase pursuit geometry, giving it 14,790 geometry_ok pairs — 986× higher than Red.

### 2. Where does Blue's advantage come from?

**Answer: B and C.**

Blue's key advantages:

- **RangeOk**: Blue has 2.4M vs Red's 3.2M. Red actually has MORE range-ok pairs. Distance is not the issue.
- **AO/TA conversion**: Blue converts 72.6% of range-ok pairs to AO-ok (rule-based pursuit actively points at target), while Red only converts 32.9%.
- **GeometryOk**: Blue's 14,790 vs Red's 15. Blue's coordinated tail-chase pursuit satisfies the simultaneous angle constraint far more often.
- **LockMature**: Blue has 1533 vs Red's 1. Once Blue has geometry, lock accumulates and launches follow.
- **EngagedBlocked**: Blue has 962,022 engaged-blocked events — this means Blue frequently has geometry_ok but the target is already being tracked by another Blue missile. This is NOT a bottleneck; it's a side effect of having many simultaneous launch windows.

### 3. Is entropy / std anomalous at 100K?

**No — not yet, but the trend is concerning.**

- Std went from 0.30 → 0.34, a 0.00038/iter drift rate.
- At this rate, projected to 10M steps: Std ≈ exp(−1.20 + 117.9) → astronomical (literally).
- At 100K, the policy is still in the "exploration" phase. The entropy_coef = 0.05 is pushing Std upward as expected.
- **Recommendation**: Do NOT add a std clamp yet. The policy hasn't learned anything useful. Adding a clamp now would reduce exploration before the policy finds launch windows. Instead, monitor Std at 300K, 500K, and 1M checkpoints. If Std exceeds 0.6–0.7 with no improvement in geometry_ok, then consider a decay schedule.

### 4. Are reward and win rate decoupled?

**Yes — but not in the "reward hacking" sense.**

- R_red is stable at −75 to −78. It has NOT improved. This is not "reward goes up while wins stay zero" — both reward AND wins are flat at zero.
- The dominant reward component is likely `r_end` (team terminal) and possibly `r_adv` (situation). With Red never winning, r_end is always negative. With Red unable to maintain rear-hemisphere geometry, r_adv is likely negative or near-zero.
- There is no evidence of reward hacking because the policy hasn't learned ANYTHING yet — the reward signal is dominated by terminal death penalties, not by exploitation of a flawed situation-reward formula.

### 5. Is 100K steps sufficient to judge failure?

**This is primarily "training too short" (A), with a minor structural concern (B).**

Arguments for "too short":
- 390 episodes is barely enough for the critic to converge from random init (CriticLoss dropped from 16.5 to 3.97, still far from converged).
- Red only started achieving geometry_ok in the last 20K steps (15 events). The policy is JUST beginning to stumble into launch windows.
- The stochastic exploration-driven search for simultaneous AO + TA satisfaction in a 3D continuous action space is inherently slow.

Structural concern:
- The geometry_ok count of 15 in 100K steps (vs Blue's 14,790) suggests a 986:1 asymmetry in launch-window access. The Blue rule-based policy has a massive built-in advantage in achieving tail-chase geometry.
- If this ratio persists past 500K steps without improvement, the environment's launch asymmetry may be too steep for learning.

## 6. Recommended next action

1. **Keep training to at least 300K–500K steps** (diagnostic only, no code change). Monitor whether Red's geometry_ok count rises from 15 toward the hundreds or thousands. If it stays below ~100 after 500K steps, the launch window is too narrow for stochastic exploration.

2. **Add a Red-specific launch-rate diagnostic alert** (diagnostic only, no behavior change). Print a warning to console when red geometry_ok per 100K steps is below some threshold (e.g. 50), so the operator knows the policy isn't finding launch windows.

3. **If geometry_ok stays below 100 at 500K**: consider a curriculum where missile launch conditions are relaxed during early training (AO < 90°, TA > 45°) and gradually tightened toward paper values. This IS a behavior change and should be implemented as a separate ablation, not a silent default.
