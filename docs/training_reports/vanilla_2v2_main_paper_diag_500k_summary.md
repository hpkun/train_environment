# vanilla_2v2_main_paper_diag_500k CSV Summary

**Corrected: cumulative fields (deaths, wins, episodes, hits) use phase delta; per-iteration
fields (launch diag, reward components) use phase sum/mean.  Previous version incorrectly
double-summed cumulative values.**

## 1. File coverage

| Item | Value |
|---|---|
| File | `results/vanilla_2v2_main_paper_diag_500k_results.csv` |
| Rows | 250 |
| Columns | 62 |
| All launch diag + ActionStdMean fields present? | Yes |
| RewardVersion | `fixed_ta_alt_eq17_3dlos_v1` |

### Field classification

| Type | Examples | Phase method |
|---|---|---|
| Cumulative (monotonic across iterations) | RedDeathsMissile, Episodes, RedWins, BlueWins, RedMissileHits | delta = end − start |
| Per-iteration snapshot | RedMeanReward, Entropy, ActionStdMean, CriticLoss, ActorLoss, RedMissiles, BlueMissiles | mean |
| Per-iteration diagnostic count | LaunchDiagRedRangeOk, LaunchDiagRedGeometryOk, LaunchDiagRedLaunches, LockMature, EngagedBlocked, CooldownBlocked | sum |

**Note**: `EngagedBlocked` / `CooldownBlocked` are physics-frame / pair-level diagnostic
counts.  12 physics frames × 2 Red agents × 2 Blue targets = 48 pair-checks per env step.
250 iterations × 2000 steps × 2 envs × 48 checks = 48,000,000 possible pair-checks total.
The 92,192 engaged-blocked events for Red represent ~0.19% of all pair-checks.  These
numbers should not be interpreted as episode-level launch opportunities.

## 2. Training overview

| Metric | Value |
|---|---|
| Iterations | 250 |
| Total env steps | 500,000 |
| Total episodes (final row) | 1,554 |

### Cumulative final values (last row)

| Metric | Value |
|---|---|
| Red wins / Blue wins / Draws | 7 / 1,471 / 76 |
| Final RWR | 0.0045 (0.45%) |
| Final recent WinRate | 0.0000 |
| RedDeathsMissile (cumulative) | 2,619 |
| RedDeathsCrash | 376 |
| BlueDeathsMissile | 45 |
| BlueDeathsCrash | 0 |
| RedMissileHits | 45 |
| BlueMissileHits | 2,619 |
| RedMissileHitRate | 4.03% |
| BlueMissileHitRate | 87.15% |
| KD_Red | 0.0150 |

### Learning curve metrics (first → last)

| Metric | First | Last | Mean |
|---|---|---|---|
| RedMeanReward | −75.70 | −31.52 | −65.20 |
| Entropy | 0.2163 | 0.9073 | 0.5276 |
| ActionStdMean | 0.3008 | 0.6000 | 0.4189 |
| ActionLogStdMean | −1.201 | −0.511 | −0.890 |
| CriticLoss | 18.21 | 0.21 | 5.27 |
| ActorLoss | −0.019 | −0.064 | −0.046 |

## 3. Phase breakdown

### Cumulative deltas (deaths, wins, episodes, hits)

| Phase | RedM | RedC | BlueM | BlueC | Episodes+ | RedWins+ | RedHits+ | BlueHits+ |
|---|---|---|---|---|---|---|---|---|
| 0–100K | 1,274 | 85 | 2 | 0 | 680 | 1 | 2 | 1,274 |
| 100–200K | 561 | 129 | 3 | 0 | 348 | 0 | 3 | 561 |
| 200–300K | 412 | 121 | 8 | 0 | 270 | 1 | 8 | 412 |
| 300–400K | 205 | 25 | 15 | 0 | 132 | 2 | 15 | 205 |
| 400–500K | 103 | 7 | 16 | 0 | 85 | 3 | 16 | 103 |

Red missile deaths per 100 episodes: 187 (0–100K) → 161 (100–200K) → 153 (200–300K) → 155 (300–400K) → 121 (400–500K). **Improving from 1.87/episode to 1.21/episode.**

### Per-iteration stats (means) + launch diagnostics (sums)

| Phase | R_red | Ent | Std | Crit | M_red | M_blue | red_geo | red_lock | red_launch | red_blocked | red_cool | blue_geo | blue_launch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0–100K | −74.4 | 0.26 | 0.32 | 13.1 | 0.0 | 2.3 | 60 | 4 | 4 | 1,794 | 0 | 27,743 | 1,568 |
| 100–200K | −73.4 | 0.39 | 0.36 | 5.9 | 0.0 | 1.8 | 672 | 245 | 29 | 7,562 | 216 | 11,434 | 654 |
| 200–300K | −70.6 | 0.51 | 0.40 | 4.2 | 0.2 | 1.6 | 1,148 | 322 | 54 | 16,805 | 268 | 7,646 | 447 |
| 300–400K | −61.4 | 0.66 | 0.47 | 2.1 | 1.3 | 1.7 | 5,959 | 1,925 | 216 | 29,510 | 1,709 | 3,658 | 221 |
| **400–500K** | **−47.1** | **0.81** | **0.54** | **1.3** | **4.9** | **1.4** | **24,148** | **7,312** | **815** | **36,521** | **6,497** | **1,932** | **115** |

## 4. Launch diagnostics (full 500K totals)

### Per-iteration sums

| Metric | Red | Blue | Ratio (R/B) |
|---|---|---|---|
| RangeOk | 12,536,623 | 9,708,892 | 1.29× |
| AoOk | 5,120,974 | 7,375,934 | 0.69× |
| TaOk | 7,476,341 | 9,318,518 | 0.80× |
| **GeometryOk** | **31,987** | **52,413** | **0.61×** |
| LockMature | 9,808 | 4,859 | 2.02× |
| Launches | 1,118 | 3,005 | 0.37× |
| EngagedBlocked | 92,192 | 3,455,673 | 0.03× |
| CooldownBlocked | 8,690 | 1,817 | 4.78× |

### Derived rates

| Rate | Red | Blue |
|---|---|---|
| Range → Geometry | 0.26% | 0.54% |
| Geometry → Lock | **30.7%** | 9.3% |
| Geometry → Launch | 3.5% | 5.7% |
| Lock → Launch | 11.4% | **61.8%** |

## 5. Bottleneck diagnosis

### A. RedGeometryOk: massive improvement from 15 → 31,987 (full-run total)

At 100K the Red per-iteration sum was 60. At 500K the full-run total is 31,987.
In the final 100K phase alone: 24,148.  The agent has learned to enter the launch cone.

### B. RedLaunches: up from 4 → 1,118 (full-run total)

Still only 3.5% conversion from geometry to launch, but the final-phase count of 815
launches per 100K steps is now ~8.2 launches per iteration — meaningful volume.

### C. The bottleneck is Lock → Launch (11.4% vs Blue 61.8%)

At 100K the bottleneck was Range → Geometry (essentially zero). Now:
- Range → Geometry (0.26%): Still the biggest filter in absolute terms, but geometry_ok
  is growing fast (24K per phase). This is not the tightest bottleneck anymore.
- Geometry → Lock (30.7%): Competitive with Blue's 9.3%. Red is BETTER at locking once
  geometry is satisfied.
- **Lock → Launch (11.4%)**: The critical bottleneck. Blue converts 61.8% of locks to
  launches. Red converts only 11.4%.  Primary blockers: cooldown (8,690) and
  engaged_blocked (92,192).

The engaged_blocked count is large but note it is a per-physics-frame, per-pair diagnostic.
It does not mean 92K "missed launch opportunities" — many of those blocked checks are the
same target being repeatedly checked across physics frames. The meaningful metric is
lock → launch conversion, which is the real bottleneck.

### D. Blue's advantage

- Geometry count: Blue 52,413 vs Red 31,987 (1.6×). Red is competitive.
- Lock → Launch: Blue 61.8% vs Red 11.4%. **5× gap**.  Blue's coordinated target
  allocation ensures launches happen efficiently. Red's two agents block each other.
- Blue never crashes (0 deaths). Red crashes 376 times.

### E. Reward vs win rate

R_red improved from −75 → −31 (+44 improvement) and Red missile deaths per episode
dropped from 1.87 to 1.21.  However RWR is still 0.45% (7 wins in 1554 episodes).
The reward improvement reflects better survival and positioning, not more kills.

### F. Std / Entropy

Std 0.30 → 0.60, Entropy 0.22 → 0.91.  At current drift rate (~0.0006/iter for Std),
10M steps would give Std ≈ 3.3 — pure noise.  **At 500K the policy is still learning**
(geometry_ok, launches, and R_red all still improving), so no intervention is justified
yet.  Monitor Std at 1M; if Std > 0.8, consider a decay schedule.

### G. Is 500K sufficient?

Red's geometry_ok growth from 60 → 24,148 in the final phase shows the policy is far
from converged.  7 wins in 1554 episodes is abysmal but the trend of improving
geometry_ok, improving R_red, and decreasing deaths per episode all suggest continued
learning.  This is NOT a failed run — it is a SLOW run.

## 6. Conclusion

**Continue to 1M steps with close monitoring of LockToLaunchRate.**

The key metric to watch is whether Red's `LockToLaunchRate` improves from the current
11.4%.  Blue achieves 61.8% because its rule-based policy has coordinated target
allocation that avoids self-blocking.  Red's stochastic policy cannot replicate this,
so either:
- Red learns to deconflict implicitly (watch for LockToLaunch approaching 30% at 1M), or
- Red needs explicit per-agent launch timing learned via reward shaping.

If `RedGeometryOk` continues its exponential growth trend (60 → 672 → 1,148 → 5,959 →
24,148 per phase), the agent is healthy and learning.  If it plateaus, add a curriculum.
If `LockToLaunchRate` doesn't improve above 15% by 1M, consider reducing cooldown
duration for Red or adding a bonus for successful launches.
