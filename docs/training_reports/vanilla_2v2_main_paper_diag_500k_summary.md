# vanilla_2v2_main_paper_diag_500k CSV Summary

## 1. File coverage

| Item | Value |
|---|---|
| File | `results/vanilla_2v2_main_paper_diag_500k_results.csv` |
| Rows | 250 |
| Columns | 62 |
| All launch diag fields? | Yes |
| ActionStdMean / Entropy? | Yes |
| RewardVersion | `fixed_ta_alt_eq17_3dlos_v1` |

## 2. Overview

| Metric | Value |
|---|---|
| Iterations | 250 |
| Total env steps | 500,000 |
| Total episodes | 1,554 |
| Red wins / Blue wins / Draws | 7 / 1,471 / 76 |
| Final cumulative RWR | **0.0045** (0.45%) |
| Final recent WinRate | 0.0000 |
| Final KD_Red | 0.0150 |
| RedMissileHitRate | 4.03% |
| BlueMissileHitRate | 87.15% |

| Metric | First | Last | Mean | Change |
|---|---|---|---|---|
| RedMeanReward | −75.70 | −31.52 | −65.20 | **+44.2** |
| Entropy | 0.2163 | 0.9073 | 0.5276 | +0.691 |
| ActionStdMean | 0.3008 | 0.6000 | 0.4189 | +0.299 |
| ActionLogStdMean | −1.201 | −0.511 | −0.890 | +0.691 |
| CriticLoss | 18.21 | 0.21 | 5.27 | −18.0 |
| ActorLoss | −0.019 | −0.064 | −0.046 | −0.045 |

## 3. Phase breakdown

| Phase | R_red | Ent | Std | Crit | M_red | M_blue | Wins+ | red_geo | red_launch | blue_geo | blue_launch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0–100K | −74.4 | 0.26 | 0.32 | 13.1 | 0.0 | 2.3 | 1 | 60 | 4 | 27,743 | 1,568 |
| 100–200K | −73.4 | 0.39 | 0.36 | 5.9 | 0.0 | 1.8 | 0 | 672 | 29 | 11,434 | 654 |
| 200–300K | −70.6 | 0.51 | 0.40 | 4.2 | 0.2 | 1.6 | 1 | 1,148 | 54 | 7,646 | 447 |
| 300–400K | −61.4 | 0.66 | 0.47 | 2.1 | 1.3 | 1.7 | 2 | 5,959 | 216 | 3,658 | 221 |
| **400–500K** | **−47.1** | **0.81** | **0.54** | **1.3** | **4.9** | **1.4** | **3** | **24,148** | **815** | 1,932 | 115 |

Deaths by phase (Red Missile / Red Crash / Blue Missile / Blue Crash):

| Phase | RedM | RedC | BlueM | BlueC |
|---|---|---|---|---|
| 0–100K | 33,585 | 1,404 | 14 | 0 |
| 100–200K | 81,592 | 8,318 | 175 | 0 |
| 200–300K | 104,638 | 14,670 | 369 | 0 |
| 300–400K | 121,108 | 17,975 | 1,134 | 0 |
| 400–500K | 128,500 | 18,677 | 1,765 | 0 |

## 4. Launch diagnostics (cumulative over 500K)

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

Derived rates:

| Rate | Red | Blue |
|---|---|---|
| Range → Geometry | 0.26% | 0.54% |
| Geometry → Lock | **30.7%** | 9.3% |
| Geometry → Launch | 3.5% | 5.7% |
| Lock → Launch | 11.4% | **61.8%** |

## 5. Bottleneck diagnosis

### A. RedGeometryOk: massive improvement from 15 → 31,987

At 100K RedGeometryOk was 15. At 500K it is 31,987 — a **2,132× increase**. In the final 100K phase alone there were 24,148 geometry_ok. The agent has clearly learned to position for launch windows.

### B. RedLaunches: up from 1 → 1,118

Still only 3.5% conversion from geometry to launch, but the absolute count is now meaningful at ~8.2 launches per iteration in the final phase.

### C. The bottleneck has shifted

At 100K the bottleneck was Range → Geometry (essentially zero). Now at 500K:

- **Range → Geometry (0.26%)**: Still the biggest filter — most agent-target pairs within range don't have both AO and TA in the launch cone simultaneously.
- **Geometry → Lock (30.7%)**: Once geometry is satisfied, Red achieves lock 31% of the time. This is actually higher than Blue's conversion (9.3%).
- **Lock → Launch (11.4%)**: The final filter. Red has lock-mature pairs but only 11.4% result in a launch. This is primarily due to **cooldown_blocked** (8,690) and **engaged_blocked** (92,192).

Wait — **Red engaged_blocked = 92,192 but Red launches = 1,118**. That means Red is being blocked from launching at engaged targets 92K times but only succeeds 1,118 times. This suggests that by the time Red achieves geometry+lock, another Red agent's missile is often already tracking that target.

But for 2v2 with Red firing only 1,118 times vs Blue's 3,005... the actual block ratio is 92,192 engaged_blocked vs 1,118 launches. That's 82:1 blocking vs launch ratio. This is suspicious — could it be that engaged_blocked refers to per-physics-frame checks (12 frames per env step), so the raw count is inflated?

Regardless, the key insight: Red's geometry→launch conversion is bottlenecked by cooldown and engaged deconfliction, not by lock or geometry.

### D. Blue's advantage

- Geometry count: Blue 52,413 vs Red 31,987 — **1.6× advantage**, but Red is competitive.
- Lock conversion: Red (30.7%) is **higher** than Blue (9.3%). Once Red has geometry, it achieves lock better.
- Launch conversion: Blue 61.8% vs Red 11.4%. Blue's lock→launch is **5× higher**. This is because Blue has coordinated target allocation that prevents unnecessary deconfliction, while Red's 2 agents often block each other (92K engaged_blocked vs 1,118 launches).
- Blue mostly has range and geometry at closer distances and better angles due to deterministic pursuit.

### E. Reward vs win rate

R_red improved from −75 to −31 (a +44 improvement), but RWR is still 0.45% (7 wins in 1554 episodes). The reward improvement is real — Red is dying slightly less, getting slightly more positive situation reward. But the improvement does NOT translate to wins because Red's missiles hit at 4.0% accuracy.

The situation reward rewards the positioning that should lead to kill opportunities, but the hit probability (based on 2D AO/TA satisfaction) doesn't correlate well with actual missile kill conditions. The agent maximizes reward through positioning rather than through killing.

### F. Std / Entropy status

- Std went from 0.30 → 0.60 (+100% from init)
- Entropy went from 0.22 → 0.91 (+313%)
- Drift rate: 0.0006/iter for Std, 0.0014/iter for Entropy

Std = 0.60 means σ ≈ 0.6 for all 3 action dimensions. The policy still has meaningful signal (0.6 out of [-1,1] is not pure noise), but the upward trend will continue without intervention. At 10M steps, projecting linearly: Std ≈ 0.30 + 0.0006 × 5000 ≈ 3.3 — way beyond any meaningful policy signal.

**Should we intervene now?** No. At 500K the policy is still improving (RedGeometryOk growing, R_red improving). Adding a std clamp or entropy decay now would prematurely constrain exploration. The policy needs to learn to convert geometry→launch and improve hit accuracy before we restrict its variance.

### G. 500K versus 10M

At 500K the training is clearly **not failed** — it's making real progress in positioning and launch frequency. The geometry_ok increase from 15 to 31,987 demonstrates learning. However:

1. **Hit rate problem is huge**: 4% hit rate vs 87% for Blue. Blue's rule-based policy achieves near-perfect alignment. Red's stochastic policy can't.

2. **The geometry→launch conversion is poor**: lock→launch = 11.4% vs Blue's 61.8%. The cooldown and deconfliction mechanisms are limiting Red more than Blue.

3. **7 wins in 1554 episodes = 0.45%**: At this rate, 10M steps would yield maybe 140 wins in 30,000 episodes. That's not a competitive policy.

## 6. Conclusion

**Continue to 1M steps but with close monitoring of geometry→launch conversion.** The agent is learning to enter the launch cone (geometry_ok growing exponentially). The key metric to watch is whether `LockToLaunchRate` improves from 11.4% toward 30-40% in the next 500K steps. If it does not, the launch deconfliction / cooldown mechanism is structurally preventing Red from succeeding.

Three specific observations for the operator to monitor at 1M:

1. **If RedLockMature ≥ 30K and RedLaunches ≥ 5K at 1M**: the policy is healthy, just slow. Continue to 10M.
2. **If RedLaunches stays < 3K but RedGeometryOk > 50K**: the deconfliction/cooldown is the bottleneck. Consider reducing cooldown for Red or adding launch reward.
3. **If RedGeometryOk plateaus < 50K**: the policy has converged to a local optimum. Add curriculum (relaxed launch conditions early).

For the conservative paper-alignment approach: **this is not yet a failed training run. Continue to 1M for diagnostics.**
