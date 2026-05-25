# Reward formula alignment plan

本文档用于 Paper alignment pass 21。目标是重新核对论文 §2.5 reward 公式与当前环境实现，形成后续修正计划。本 pass 不改变任何训练或环境行为。

## 1. Current reward implementation

当前 `UavCombatEnv._compute_rewards()` 的每个存活 agent reward 组合为：

- `0.01 * r_pitch`
- `0.002 * r_roll`
- `0.04 * r_alt`
- `0.04 * r_bound`
- `0.02 * r_vel`
- `0.15 * r_adv`
- `r_end`
- `r_death` if any，当前仅 crash frame 对死亡 agent 额外给 `-10`

当前 reward version 为：

```text
fixed_ta_alt_eq17_v1
```

含义：`_situation_reward()` 中的 Ta 角度优势函数已经从旧的负值/不连续分段切换为连续、非负、归一化 `[0,1]` 的工程修正版；`_altitude_reward()` 已切换为 pairwise eq.17-style curve，并保留 high-altitude `0.1` tail。旧行为仍保留在 helper 函数中，只用于审计。`fixed_ta_v1` 结果不应与 `fixed_ta_alt_eq17_v1` 混合比较。

## 2. Paper reward formulas to verify

本节基于本地 PDF 文本抽取结果和当前项目既有 pass 记录整理。PDF 抽取中部分数学符号存在字体编码损坏，因此仍需人工对照论文版面最终确认。

### Eq.15 pitch reward

论文文本可读部分显示：

- `r_theta = -1` if `|theta| > pi/3`
- `r_theta = -(|theta|/pi - 1/4) / 12` if `pi/3 > |theta| > pi/4`
- 其余情况应为 0，但 PDF 抽取片段未完整显示 else 分支。

人工核对项：`NEEDS PAPER TEXT VERIFICATION`，确认中间段符号、除以 12 的位置、else 是否为 0。

### Eq.16 roll reward

论文文本可读部分显示：

```text
r_phi = -(|phi|/pi - 1/4) * 4/3
if |phi| > pi/4 & |theta| > pi/4
```

人工核对项：确认条件中的 `&` 是否确实为双条件，而非排版误差。

### Eq.17 altitude reward

论文文本可读部分显示为相对高度 `z_i - z_j` 的二次分段：

- `0.1` for `H_max < z_i - z_j <= D_att,max`
- `h1 * (z_i - z_j - H_adv)^2 + 1` for `H_adv < z_i - z_j <= H_max`
- `1` for `H_att < z_i - z_j <= H_adv`
- `h2 * (z_i - z_j - H_att)^2 + 1` for `H_min < z_i - z_j <= H_att`
- `0` otherwise

人工核对项：`NEEDS PAPER TEXT VERIFICATION`，确认 `D_att,max`、`H_min/H_att/H_adv/H_max`、`h1/h2` 的具体数值是否在论文其它位置给出。

### Eq.18 boundary reward

论文文本可读部分显示：

```text
r_board = -10 if |x| > 4 * 10^4 or |y| > 4 * 10^4
```

### Eq.19 speed reward

论文文本可读部分显示：

- `r_V = -1` if `V < 0.2 mach`
- `r_V = -(0.3 - V) / 0.1` if `0.2 mach < V < 0.3 mach`

人工核对项：确认 `V` 在公式中是否以 Mach 数表达，而不是 m/s。

### Eq.20 angle advantage Ta

论文文本可读部分显示：

- `Angle1 = 4/180 * pi`
- `Angle2 = 15/180 * pi`
- `Angle3 = 35/180 * pi`
- `Ta = 10` if `q_Los < Angle1`
- `Ta = 1 + (2 * 15/180*pi - q_Los) / (15/180*pi)` if `Angle1 < q_Los < Angle2`
- `Ta = 1 - (q_Los - 15/180*pi) / (35/180*pi - 15/180*pi)` if `Angle2 < q_Los < Angle3`
- `Ta = 0` otherwise

人工核对项：

- `NEEDS PAPER TEXT VERIFICATION`：第二段 PDF 抽取符号存在编码/空格干扰，需要对照原式确认精确括号。
- 当前项目使用 AO/TA，而论文公式变量是 `q_Los`。AO/TA 是否等价于 `q_Los` 需要几何核对。
- 论文 Ta 第一段明显是 `10` 量级；当前 `fixed_ta_v1` 故意保持 `[0,1]` reward scale。

### Eq.21 distance advantage Td

论文文本可读部分显示：

```text
Td = 1, D <= 15
Td = exp(1 - D / 15), D > 15
```

论文文本说明 UAV attack range is set to 15 km，因此 `D` 应以 km 表达。

### Eq.22 situation reward r_adv

论文文本可读部分显示：

```text
r_adv_i = sum_j(alpha1 * Ta_j^i * Td_j^i - alpha2 * Ta_i^j * Td_i^j)
alpha1 = 1
alpha2 = 0.8
```

文字说明第一部分是我方 UAV 相对敌方 UAV 的 situation advantage，第二部分是敌方相对我方的 threat level。

人工核对项：符号上 `Ta_j^i` / `Ta_i^j` 的上下标方向容易与代码变量 `Ta_ij` / `Ta_ji` 混淆，下一轮修正前需要先统一命名。

### Eq.23 terminal reward r_end

论文文本可读部分显示：

```text
r_end = 0, N_blue = N_red
r_end = 30 * (N_red - N_blue), else
```

随后 joint reward：

```text
r_R = sum_i r_i + r_end
```

当前环境 API 返回 per-agent reward，因此把 team-level `r_end` 均分到队伍成员。

## 3. Current vs paper table

| Reward item | Paper formula / interpretation | Current implementation | Status | Risk | Recommended action |
| --- | --- | --- | --- | --- | --- |
| Pitch eq.15 | `-1` above `pi/3`; middle penalty between `pi/4` and `pi/3`; else likely 0 | `_pitch_penalty()` matches this structure: `-1`, then `-(theta/pi - 0.25)/12`, else 0 | needs verification | Medium: slope/sign error would weaken stability penalty | Verify exact PDF formula visually; add pure function test if confirmed |
| Roll eq.16 | `-(|phi|/pi - 1/4) * 4/3` if `|phi| > pi/4 & |theta| > pi/4` | `_roll_penalty()` uses same double condition and formula | aligned | Low | Keep; optionally add pure function tests later |
| Altitude eq.17 | Relative height quadratic piecewise, includes `0.1` high-altitude tail before `D_att,max` | `_altitude_reward()` now uses pairwise relative altitude over alive enemies and `altitude_reward_pairwise_mean_eq17()`; high-altitude tail is `0.1` | approximate / closer alignment | Medium: thresholds and h1/h2 still need verification | Keep current eq.17-style implementation under `fixed_ta_alt_eq17_v1`; verify exact constants before further changes |
| Boundary eq.18 | Fixed `-10` if either horizontal axis exceeds `4e4` | `_boundary_penalty()` fixed `-10` if `|x|` or `|y|` exceeds battlefield half-size | aligned | Low | Keep |
| Speed eq.19 | Low-speed penalty below Mach 0.3, severe below Mach 0.2 | `_speed_penalty()` uses `v / 340.0`; same thresholds and slope | aligned / needs unit verification | Low-Medium: Mach conversion constant approximate | Keep for now; optionally use local speed of sound if needed |
| Ta eq.20 | Paper appears to use `10` first segment and piecewise in `q_Los` radians | `fixed_ta_v1` uses normalized `[0,1]`, continuous non-negative curve over 4/15/35 deg | intentional mismatch | High: situation reward scale differs from paper | Do not silently replace; run reward-scale ablation: `fixed_ta_v1` vs paper-scale Ta |
| Td eq.21 | `D <= 15 km -> 1`; else `exp(1-D/15)` | `td_distance_advantage()` uses meters input, converts to km, same formula | aligned | Low | Keep; document D unit as km |
| r_adv eq.22 | Sum over all enemies: advantage minus threat, weights 1 and 0.8 | `_situation_reward()` sums all alive enemies, same weights; uses AO for ego advantage and TA for enemy threat | approximate / needs verification | High: AO/TA may not equal `q_Los`; Ta scale differs | First isolate geometry definitions, then ablate Ta scale |
| Terminal eq.23 | Team-level `r_end`, then joint reward sums agent rewards plus `r_end` | Environment returns per-agent reward; team-level `r_end` is divided by team size | approximate but intentional | Medium: per-agent API vs paper joint reward | Keep per-agent share for MAPPO stability; document in all result reports |
| `r_death` | Not part of eq.15-23 extracted text | `r_death=-10` for crash frame on dead agents | mismatch / engineering addition | Medium: changes terminal/crash shaping | Consider ablation after formula alignment |

## 4. Proposed correction order

1. Verify and test formula-clear, low-risk items first.
   - Pitch eq.15: confirm exact middle-segment sign and denominator.
   - Speed eq.19: confirm `V` is Mach, not m/s.
   - Altitude eq.17: confirm constants and high-altitude `0.1` tail.

2. Treat situation reward as an ablation, not a silent overwrite.
   - Treat `fixed_ta_alt_eq17_v1` as the current engineering baseline.
   - Add a separate paper-scale Ta implementation if eq.20 is visually confirmed.
   - Use explicit `RewardVersion`, for example `paper_ta_scale_v1`, before running new training.
   - Compare learning stability and reward component magnitudes before choosing default.

3. Separate geometry work from formula scale work.
   - First decide whether `q_Los` should be current 2D AO/TA, body-frame LOS angle, or another tail-aspect geometry.
   - Do not change Ta scale and geometry in the same pass.

4. Move observation/global state alignment to a separate pass.
   - Strict Table 1/Table 2 observation should be validated through `paper_state_extractor.py`.
   - Critic global state should be redesigned independently from reward formula changes.

5. Evaluate engineering additions separately.
   - `r_death` crash penalty and per-agent terminal sharing should be preserved until formula-aligned baselines are stable.
   - If removed, do it as an explicit ablation with a new reward version.

## 5. Pass22 altitude reward function audit

Pass22 added pure altitude reward helpers to `reward_utils.py`. The later paper environment alignment pass wires the eq.17-style pairwise helper into `UavCombatEnv._altitude_reward()`.

New helper functions:

- `altitude_reward_current(dz_m)`: exactly mirrors the current environment's dz-only curve. In the environment, `dz_m` is currently `ego_altitude - mean(enemy_altitudes)`.
- `altitude_reward_paper_eq17(dz_m)`: paper eq.17-style curve using current thresholds but adding the high-altitude `0.1` tail indicated by the eq.17 PDF extraction.
- `altitude_reward_paper_candidate(dz_m)`: compatibility alias for `altitude_reward_paper_eq17()`.
- `altitude_reward_pairwise_mean_eq17(ego_alt_m, enemy_altitudes_m)`: computes `altitude_reward_paper_eq17(ego_alt - enemy_alt)` for each enemy and returns the mean.
- `altitude_reward_pairwise_mean_candidate(...)`: compatibility alias for `altitude_reward_pairwise_mean_eq17()`.
- `sample_altitude_table(func)`: diagnostic sampling over fixed dz values.

Current environment status:

- `UavCombatEnv._altitude_reward()` now uses pairwise relative altitude over each alive enemy.
- It returns the mean of `altitude_reward_paper_eq17(ego_alt - enemy_alt)` values.
- It preserves the high-altitude `0.1` tail instead of returning 0 above `H_MAX=10000`.
- It uses reward version `fixed_ta_alt_eq17_v1`.

Candidate behavior:

- `dz <= 0`: 0
- `0 < dz < 2000`: quadratic rise from 0 to 1
- `2000 <= dz <= 5000`: 1
- `5000 < dz <= 10000`: quadratic fall from 1 to 0.1
- `dz > 10000`: 0.1

Remaining verification:

- Confirm exact paper constants for `H_min`, `H_att`, `H_adv`, `H_max`, `D_att,max`, `h1`, and `h2`.
- If these constants differ from current `H_ATT=2000`, `H_ADV=5000`, `H_MAX=10000`, mark the next change with a new reward version.

## 5. No-code-change statement

This pass does not change any training or environment behavior. It only adds this correction plan document and links it from the existing audit document.
