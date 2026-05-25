# Paper environment/reward audit

本文档用于 Paper alignment pass 17：对照论文 §2 的环境、观测、动作与奖励定义，审计当前项目实现状态。本文只记录差异与建议优先级，不修改环境、训练逻辑或 BRMA 相关代码。

## 结论摘要

当前项目已经完成了一批低耦合对齐项：雷达最大探测距离的 RCS 四次方根关系、导弹锁定/发射间隔、导弹命中概率、roll/altitude/boundary reward 的一部分公式修正，以及训练/评估日志指标扩展。

仍需重点处理的高优先级问题是 `_situation_reward()`。当前角度优势函数第一段为 `1.0`，而论文 eq.20 中应核对是否为 `10`；同时当前 4 到 15 度分段公式会出现负值，并在 15 度附近出现不连续跳变。该问题会直接影响态势奖励量级和训练信号，建议在下一轮优先修正。

## 审计优先级定义

- P0：会显著改变训练信号或论文关键结论，应优先修正。
- P1：与论文存在明确差异，但可作为工程 baseline 暂存。
- P2：诊断、文档或实现细节差异，短期不阻塞 vanilla / attention baseline。

## 1. Radar

| 项目 | 论文 §2 要求 | 当前实现 | 状态 | 优先级 |
| --- | --- | --- | --- | --- |
| 方位 FOV | 前向水平视场，当前按论文整理为 ±60° | `RADAR_AZIMUTH_HALF = np.deg2rad(60)`，`_is_detected_by_radar()` 检查 yaw 相对误差 | 基本对齐 | P2 |
| 高低角 FOV | elevation 范围 [-10°, +32°] | `RADAR_ELEVATION_MIN=-10°`，`RADAR_ELEVATION_MAX=32°`，按 LOS elevation 减 ego pitch 检查 | 基本对齐 | P2 |
| RCS | 论文使用 z-axis/y-axis 角度 RCS 表插值 | 当前 `_compute_radar_max_range()` 保留 front/side 简化近似：前/后低 RCS，侧向高 RCS | 未严格对齐，因论文未给表格数据 | P1 |
| Rmax | `Rmax = K * RCS^(1/4)` | 当前为 `self.RADAR_K * np.power(rcs, 0.25)` | 已对齐 | P2 |

备注：

- 当前 RCS 简化模型是合理工程占位，但不应声称严格复现论文 RCS table interpolation。
- 雷达范围使用 3D range `R_3d`，RCS 角度使用 2D `TA`。这在工程上可用，但若后续要严格复现论文，需要确认论文 RCS 表角度是否包含垂直方向姿态。

## 2. Missile

| 项目 | 论文 §2 要求 | 当前实现 | 状态 | 优先级 |
| --- | --- | --- | --- | --- |
| 连续锁定延迟 | 0.25 s | `self.missile_lock_delay_frames = int(round(0.25 * self.sim_freq))` | 已对齐 | P2 |
| 发射间隔 | 0.5 s | `self.missile_cooldown_frames = int(round(0.5 * self.sim_freq))`，每 physics frame 递减 | 已对齐 | P2 |
| 发射角/距离门限 | AO < 45°，射程内，满足 3-9 line / rear hemisphere | `_check_missile_launch()` 使用 `AO < 45°`、`500 m < R < 10000 m`、`TA > 90°` | 基本对齐；最小射距 500 m 是工程安全项 | P1 |
| 3-9 line | 目标处于敌后半球 | 当前以 `TA > π/2` 表示 rear hemisphere | 基本对齐，但需确认 `TA` 与论文几何符号完全一致 | P1 |
| 命中概率 | `0.05 + 0.95 * max(0, Vm·LOS/(|Vm||LOS|))` | `MissileSimulator._roll_hit_probability()` 使用 missile velocity 与 missile-to-target LOS dot product | 已对齐 | P2 |
| 同目标发射抑制 | 避免多个导弹同一时间重复攻击同一目标 | 当前 `_check_missile_launch()` / `_launch_missile()` 使用 `_engaged_targets`，并有 `refresh_engaged_targets()` | 工程增强项，论文未必明确 | P2 |

备注：

- 当前导弹最小射距 `MISSILE_LAUNCH_MIN_RANGE=500 m` 不是论文重点公式，但用于避免近距自毁/数值异常。
- 当前环境保留 `num_missiles_per_plane=999`，等价于不让载弹量成为主要限制因素。论文没有明确固定载弹量，因此该项暂不作为优先对齐目标。

## 3. Action space

| 项目 | 论文要求 | 当前实现 | 状态 | 优先级 |
| --- | --- | --- | --- | --- |
| Pitch command | 俯仰控制范围 | action[0] 映射到 `±90°`：`target_pitch = action[0] * PITCH_DEG` | 基本对齐，需核对论文是否使用绝对 pitch command | P1 |
| Heading command | 航向控制范围 | action[1] 映射到 `±π`：`target_heading = action[1] * π` | 基本对齐，当前是绝对 heading target | P1 |
| Velocity command | 速度范围 | action[2] 映射到 102-408 m/s，即 Mach 0.3-1.2 | 基本对齐 | P2 |
| 额外控制层 | 论文动作之外的安全/规避逻辑 | 当前 `_parse_actions()` 包含 missile evasion layer，蓝方可选 GCAS layer | 工程差异 | P1 |

备注：

- 当前动作是高级目标指令，再由 PID/控制器转成底层飞行控制，不是直接舵面控制。
- `enable_blue_gcas` 在训练、ACMI、批量评估脚本中默认应为 False；但 `UavCombatEnv` 构造函数默认参数仍是 True，调用方必须显式传入以保持实验一致。

## 4. Observation

### 4.1 当前环境 observation

当前 `UavCombatEnv` observation 仍是工程化 dict：

- `ego_state`: shape `(11,)`
- `ally_states`: shape `(N_ally, 11)`
- `enemy_states`: shape `(N_enemy, 11)`
- 额外字段：`death_mask`、`missile_warning`、`altitude`、`velocity`

11 维 entity vector 当前布局为：

```text
[dx_body, dy_body, dz_body, AO_signed, TA, R, V_tgt,
 sin_roll, cos_roll, sin_pitch, cos_pitch]
```

对齐状态：未严格对齐论文 Table 1 / Table 2。当前 11 维是工程 baseline，不应称为论文原始 observation。

优先级：P1。短期保持不变以保证 vanilla MAPPO 和 attention baseline 可运行；后续如果要复现论文消融，需要切到严格 10 维观测。

### 4.2 `paper_obs_utils.py` placeholder adapter

`paper_obs_utils.build_paper_entity_observation_from_env_obs()` 将当前 11 维 entity vector 裁剪为前 10 维：

```text
[dx_body, dy_body, dz_body, AO_signed, TA, R, V_tgt,
 sin_roll, cos_roll, sin_pitch]
```

状态：只是 10 维接口占位，不是论文 Table 1 / Table 2 的物理量复现。

优先级：P2。适合先验证 `AttentionActor(entity_dim=10)` 的工程链路，但不适合作为论文严格实验结果。

### 4.3 `paper_state_extractor.py` strict prototype

`paper_state_extractor.py` 已新增从 simulator/native state 构造 10 维观测的原型：

- self state: `[x, y, h, V, roll, pitch, heading, alpha, beta, Vd]`
- relative state: `[x_body, y_body, z_body, theta_v_body, psi_v_body, V, theta_LOS_body, psi_LOS_body, q_LOS, d]`

状态：原型存在，但未接入训练。

仍需确认：

- 机体系旋转矩阵与 JSBSim 坐标/姿态符号是否完全一致。
- alpha/beta 当前会尝试读取 JSBSim property；若不可用则为 placeholder。
- `q_LOS` 当前定义为 LOS 与 observer body x-axis 的夹角，不等同于环境中 AO/TA 或目标尾后角，接入 reward / mask ranking 前必须核对论文几何定义。

优先级：P1。

## 5. Reward

当前 `_compute_rewards()` 使用加权组件：

```text
0.01 * r_pitch
+ 0.002 * r_roll
+ 0.04 * r_alt
+ 0.04 * r_bound
+ 0.02 * r_vel
+ 0.15 * r_adv
+ r_end
```

### 5.1 Pitch reward eq.15

当前 `_pitch_penalty()`：

- `|theta| > π/3` 返回 `-1.0`
- `π/4 < |theta| <= π/3` 返回 `-(theta / π - 0.25) / 12.0`
- 否则返回 0

状态：大体按论文式分段惩罚实现，但需要再次核对 eq.15 的斜率和量级。当前中间段最大惩罚很小，约为 `-0.0069`，乘以权重 `0.01` 后更小。

优先级：P1。

### 5.2 Roll reward eq.16

当前 `_roll_penalty()`：

```python
phi = abs(sim.get_rpy()[0])
theta = abs(sim.get_rpy()[1])
if phi > np.pi / 4 and theta > np.pi / 4:
    return -(phi / np.pi - 0.25) * (4.0 / 3.0)
return 0.0
```

状态：已按论文 eq.16 的 roll + pitch double-condition 对齐。

优先级：P2。

### 5.3 Altitude reward eq.17

当前 `_altitude_reward()` 使用相对敌方平均高度 `dz = alt_ego - mean(enemy_alts)`，阈值：

- `H_MIN = 0`
- `H_ATT = 2000`
- `H_ADV = 5000`
- `H_MAX = 10000`

并使用归一化二次曲线，最后 `np.clip(reward, 0.0, 1.0)`。

状态：按论文二次分段形式做了近似，但论文未给出当前阈值/系数的完整数值依据。

优先级：P1。

### 5.4 Boundary reward eq.18

当前 `_boundary_penalty()`：

- 如果 `|x| > 4e4` 或 `|y| > 4e4`，固定返回 `-10.0`
- 不再按 x/y 双轴叠加

状态：已对齐论文 eq.18。

优先级：P2。

### 5.5 Speed reward eq.19

当前 `_speed_penalty()`：

- Mach `< 0.2` 返回 `-1.0`
- Mach `[0.2, 0.3)` 线性惩罚
- Mach `>= 0.3` 返回 0

状态：基本符合低速惩罚思路，但需核对 eq.19 的具体分段斜率。当前以 340 m/s 作为 Mach 换算常数。

优先级：P1。

### 5.6 Situation reward eq.20-22

当前 `_situation_reward()` 形式：

```text
r_adv_i = sum_j(1.0 * Ta_i^j * Td_i^j - 0.8 * Ta_j^i * Td_j^i)
```

其中：

- `Ta_i^j` 使用 ego 的 `AO`
- `Ta_j^i` 使用 enemy 的 `TA`
- `Td` 使用水平距离 `R / 1000.0` 转为 km
- `D_km <= 15` 时 `Td = 1.0`
- `D_km > 15` 时 `Td = exp(1.0 - D_km / 15.0)`

当前角度分段实现：

```python
if q_deg <= 4.0:
    Ta = 1.0
elif q_deg <= 15.0:
    Ta = 1.0 - 2.0 * (q_deg - 4.0) / 15.0
elif q_deg <= 35.0:
    Ta = 1.0 - 3.5 * (q_deg - 15.0) / 180.0
else:
    Ta = 0.0
```

审计结果：

- 当前第一段是 `1.0`，需要核对论文 eq.20 是否为 `10`。根据当前论文整理要求，这里很可能与论文不一致。
- 当前 4 到 15 度段会从 `1.0` 降到约 `-0.4667`，然后 15 度之后第三段又接近 `1.0`，存在明显不连续。
- 当前 15 到 35 度段只从 `1.0` 降到约 `0.6111`，之后 35 度直接归零，也存在大跳变。
- 距离单位当前使用 `R / 1000.0` 得到 km。如果论文 eq.21 中 `d` 的单位是 km，则当前一致；如果论文公式使用 m，则需要改。建议下一轮直接以论文公式原文重新核对。
- 当前 AO/TA 来自 `get2d_AO_TA_R()` 的水平 2D 几何，`AO` 是己机速度与 LOS 的夹角，`TA` 是敌机速度与敌机指向己机 LOS 的夹角。论文 eq.20-22 中的 `q_Los` 是否等价于当前 AO/TA 仍需进一步确认。

状态：未严格对齐，且当前实现疑似存在公式尺度/分段错误。

优先级：P0。

建议：

1. 下一轮优先从论文 eq.20 原文重新抄录 `Ta(q)` 分段，确认第一段常数、每段斜率、每段端点。
2. 为 `Ta(q)` 写纯函数与单元测试，检查 `q=0,4,15,35` 附近是否连续、是否非负。
3. 再决定 `_situation_reward()` 是否继续使用当前 AO/TA，或改为从 strict paper observation 的 `q_LOS` / LOS geometry 计算。

#### Pass18 function audit

本轮新增 `reward_utils.py`，只用于纯函数审计和测试，没有接入 `UavCombatEnv._situation_reward()`，因此不会改变当前训练奖励。

新增函数：

- `ta_angle_advantage_current(q_deg)`：完全复刻当前 `_situation_reward()` 的 Ta 分段，包括 15 度附近的负值和不连续行为。
- `td_distance_advantage_current(distance_m)`：完全复刻当前 Td 距离逻辑，先把 m 转为 km，再按 15 km 阈值计算。
- `ta_angle_advantage_candidate_continuous(q_deg)`：仅作为连续、非负参考曲线，用于对照当前实现，不代表论文最终公式。
- `sample_ta_table(func)`：输出固定角度采样表，便于文档和 smoke test 观察。

下一步仍必须根据论文 eq.20 原文决定：

- Ta 第一段到底应为 `1.0`、`10`，还是论文中的其他尺度。
- Ta 是否需要归一化后再进入 `r_adv`。
- Ta 是否允许负值；如果不允许，是否应显式 clamp。
- 当前 AO/TA 是否等价于论文 `q_Los`，或应换成 strict paper observation 中的 LOS 几何。

在完成上述核对之前，不应替换 `_situation_reward()` 的实际调用逻辑。

#### Pass19 fixed Ta update

本轮已将实际 `UavCombatEnv._situation_reward()` 使用的 Ta 函数切换为 `reward_utils.ta_angle_advantage_fixed()`：

- `q <= 4°`: `1.0`
- `4° < q <= 15°`: 从 `1.0` 线性降到 `0.5`
- `15° < q <= 35°`: 从 `0.5` 线性降到 `0.0`
- `q > 35°`: `0.0`
- 最终 clamp 到 `[0, 1]`

旧的 `ta_angle_advantage_current()` 仍保留在 `reward_utils.py`，用于追踪历史行为和对比旧训练日志。`td_distance_advantage()` 目前仍直接复用旧的 Td 距离公式。

该修正没有采用论文中可能存在的 `10` 倍 Ta 量级，而是保持当前 baseline 的归一化 reward 尺度，只修复负值和分段不连续问题。若后续确认论文原始实验使用 `10` 倍 Ta，应作为单独 reward-scale ablation，而不是混入当前 vanilla / attention baseline。

本轮仍未修改：

- AO/TA/R 的几何来源。
- `r_adv = 1.0 * Ta_i^j * Td_i^j - 0.8 * Ta_j^i * Td_j^i` 的组合公式。
- reward 权重 `0.15 * r_adv`。
- 训练脚本、观测、动作、导弹、雷达逻辑。

### 5.7 Terminal reward eq.23

当前 `_compute_rewards()`：

```text
raw_r_end_red  = 30 * (n_red_alive - n_blue_alive)
raw_r_end_blue = 30 * (n_blue_alive - n_red_alive)
```

并按队伍人数分摊到每个 agent。

状态：team-level 公式基本对齐；per-agent share 是工程实现选择，用于避免多智能体 reward 被 agent 数量重复放大。

优先级：P2。

### 5.8 额外 death penalty

当前对坠毁 agent 有 `r_death = -10.0`。

状态：工程附加项，论文 eq.23 之外的稳定训练设计。

优先级：P1。

## 6. Critic / Global State

论文 §2.3 / MAPPO 设定中 critic 使用 global state。当前代码：

- `train_vanilla_mappo.py` 的 `CentralizedCritic` 输入是所有 red agents 的 flattened observations concat。
- `train_attention_mappo.py` 仍复用相同 `CentralizedCritic`，actor 使用 entity attention，但 critic 没有切换到 strict global state。

状态：CTDE 结构形式存在，但 global state 不是论文原生全局物理状态，而是 red 视角 observation flatten concat。

优先级：P1。

建议：

- 在 strict paper observation 稳定后，再设计 critic global state schema。
- 不建议在修正 `_situation_reward()` 之前同时替换 critic，否则训练变化来源难以归因。

## 7. Blue rule policy / GCAS

当前蓝方不是学习策略，而是 `rule_based_agent.blue_coordinated_actions()`：

- 使用观测中的 AO/TA/R 等字段进行目标分配与规则制导。
- 包含 AWACS fallback、AO floor、stall protect、anti-stall、missile evasion 等工程逻辑。

GCAS 状态：

- `train_vanilla_mappo.py` 默认 `enable_blue_gcas=False`。
- `train_attention_mappo.py` 复用配置，默认 `enable_blue_gcas=False`。
- `eval_acmi.py` 显式使用 `enable_gcas_for_blue=False`。
- `evaluate_vanilla_mappo.py` 和 `evaluate_attention_mappo.py` 默认 `--enable-blue-gcas=False`。
- 但 `UavCombatEnv(..., enable_gcas_for_blue=True)` 是环境构造函数默认值；实验脚本必须显式传入 False 才与训练默认一致。

状态：蓝方规则策略与论文 learned multi-agent setting 不完全一致。它适合作为 baseline 对手，不应视作论文完整设置。

优先级：P1。

## 8. 推荐修正顺序

1. P0：修正 `_situation_reward()` 的 `Ta(q)` 分段，先做纯函数和边界点测试。
2. P1：核对 AO/TA/q_LOS 几何定义，决定 situation reward 是否继续用 2D AO/TA，或切到 strict LOS geometry。
3. P1：核对 pitch/speed reward 的精确斜率与权重量级。
4. P1：验证 `paper_state_extractor.py` 的旋转矩阵、alpha/beta、q_LOS 数值方向。
5. P1：设计真正的 paper global state critic 输入。
6. P2：继续保留当前 RCS approximation，除非获得论文 RCS 表格数据。

## 9. 本轮未做事项

- 未修改 `my_uav_env/env.py`。
- 未修改 `train_vanilla_mappo.py`。
- 未修改 `train_attention_mappo.py`。
- 未修改任何评估脚本。
- 未运行 JSBSim / 环境 reset / 训练。
- 未实现 BRMA-MAPPO、MaskVectorGenerator 或 biased random mask。
