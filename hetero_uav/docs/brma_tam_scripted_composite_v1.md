# brma_tam_scripted_composite_v1

## 定位与契约

`brma_tam_scripted_composite_v1` 是当前 scripted-launch、scripted-evasion JSBSim 环境中的组合奖励。它复用 BRMA 飞行状态项和 TAM 角色项，但不宣称完整复现 TAM-HAPPO。修订后的契约编号为 `reward_contract_revision: 2`。

模式要求：

- `observation_mode: mav_shared_geo`；
- `missile_evasion.mode: brma_scripted`，且 `teams` 为 `red_only` 或 `both`；
- `red_target_selection_mode: closest`；
- MAV `num_missiles: 0`。

## Active reward 白名单

攻击 UAV：

`R_UAV = r_pitch + r_roll + r_vel + 10 R_V + 15 R_A + 10 R_D + R_event,UAV`

MAV：

`R_MAV = r_pitch + r_roll + r_vel + 0.5 R_dist + 0.3 R_threat + 0.2 R_aspect + 0.6 R_pos + 0.4 R_aware + R_event,MAV`

只有上式中的项进入 active total。BRMA 的 altitude、boundary、advantage、terminal、death 项，以及 TAM dodge `R_DM`，均为 log-only。高度上边界 10000 m 当前也只有审计字段，没有 active altitude shaping。

## UAV 项

令 `V_r` 为 UAV 速度、`V_b` 为 reward target 速度：

- `V_b < 0.5 V_r`：`R_V = 1`；
- `0.5 V_r <= V_b <= 1.5 V_r`：`R_V = 2 - 2 V_b/V_r`；
- `V_b > 1.5 V_r`：`R_V = -1`；
- 无效或零 UAV 速度：`R_V = 0`，并记录 invalid。

三维几何定义：`D = p_blue - p_red`，`ATA = angle(D, v_red)`，`AA = angle(D, v_blue)`：

`R_A = 1 - (ATA + AA) / pi`。

无效位置、LOS 或速度时 `R_A=0` 并记录 geometry invalid。

距离 `d` 使用真实三维距离：

- `d <= 5 km`：`R_D = 1`；
- `5 km < d < 10 km`：`R_D = exp(-0.921 (d_km - 5))`；
- `d >= 10 km`：`R_D = -1`。

UAV event 仅含敌机击杀、一次性死亡/坠毁和一次性水平越界。死亡与同一步越界互斥，死亡优先。

## MAV safety

`R_dist` 使用最近存活蓝机的三维距离 `d`：

- `d < 8000`：`-(1 - d/8000)`；
- `8000 <= d < 15000`：`-0.5 (1 - (d-8000)/(15000-8000))`；
- `d >= 15000`：`0.2`；
- 无存活蓝机：`0`。

8000 m 处按论文分段保留跳变。

`R_threat=-1` 仅当存在存活且明确以 MAV 为目标的在飞导弹，否则为 0。蓝机当前潜在发射几何只记录为 `mav_prelaunch_geometry_threat_log/count_log`，不进入 total。

`R_aspect` 对每个存活蓝机求和。蓝机速度方向与蓝机到 MAV 的三维 LOS 夹角为 `alpha`；当 `alpha < pi/4` 时累加 `-(1-alpha/(pi/4))`，否则为 0。active reward 不按蓝机数归一化，只额外记录 per-blue mean。

## MAV support

动态战场中心是“存活攻击 UAV + 存活蓝机”的二维 XY 坐标均值，明确排除 MAV。没有存活攻击 UAV或没有存活蓝机时中心无效，`R_pos=0`。

令 MAV 到有效中心的水平距离为 `d`：

- `d < 8000`：`R_pos = d/8000 - 1`；
- `8000 <= d < 25000`：`R_pos = 1 - (d-8000)/(25000-8000)`；
- `d >= 25000`：`R_pos = -0.5`。

`R_aware` 只遍历由 current-state observation helper 判定为 MAV 当前 observed 的存活蓝机。使用 MAV 三维速度和 MAV 到蓝机三维 LOS 的 AO：

- `AO < pi/2`：累加 `0.3 (1 - AO/(pi/2))`；
- 其他情况：贡献 0。

active `R_aware` 是求和，不随蓝机数归一化。3V2 到 5V4 时尺度可能增加，因此同时记录 `mav_aware_per_blue_mean`，但不改变 active reward。

`mav_shared_track_slot_count_log` 是“存活攻击 UAV x 存活蓝机”槽位中 shared bit 为 1 的数量；`mav_shared_track_unique_blue_count_log` 是至少被一个攻击 UAV 通过 MAV shared track 观察到的不同蓝机数。兼容字段 `mav_shared_track_count_log` 明确映射为 slot count。

## Observation 与 track

`_mav_shared_track_state(observer, target)` 是 observation 和 reward diagnostics 的 single source of truth，不改变 observation space：

- 攻击 UAV 到目标不超过 `uav_direct_observation_range_m` 时 direct visible；
- MAV 存活且 MAV 到目标不超过 `mav_observation_range_m` 时，对攻击 UAV 为 MAV shared visible；
- MAV 自身在 `mav_observation_range_m` 内为 direct visible；
- 蓝方保持原有 direct-distance 逻辑；
- 可见性不增加 AO 条件。

## R_DM diagnostic

scripted evasion 已直接覆盖策略动作，因此 `R_DM` 只能用于 diagnostic。它只对 `_evasion_step_records` 本步实际选中的导弹计算：

- `R_AM = -cos(lambda)`，lambda 是导弹速度与“导弹到飞机”LOS 的夹角；
- `R_SM = (V_prev - V_current) / 1000`，速度缓存按 missile UID 隔离并在 reset 清空；
- `R_DM = R_AM + R_SM`。

第一条有效记录的 `R_SM=0`。缺导弹、导弹失效、无效 LOS 或速度时 geometry valid 为 0，并写入 missing reason。该项从不进入 active total。

## Reward target 与 fire-control target

reward target 是真实三维距离最近的存活蓝机；同距离按 `blue_ids` 顺序 tie-break。它不受 observation、ammo、engaged target、lock、cooldown、launch range 或发射门槛影响。

fire-control target 仍由原 launch gate 决定。诊断按同一决策步记录 current-state track、锁定目标、全部发射目标和首个发射目标。`launch_target_id` 固定取同一步记录顺序中的首个目标，`launch_target_ids` 用 `|` 连接全部目标。发射时使用 `lock_target_id_at_launch` 快照，避免锁在发射后被清空造成误报。scripted evasion 覆盖发生时 `action_source=scripted_evasion`，否则为 `policy`。

## 生命周期与聚合

环境 step 前的 alive mask 是奖励生命周期基准：dead-before-step 的 agent total 为 0；alive-before 且本步死亡的 agent 获得一次死亡事件。水平越界对每个 UAV 每 episode 最多一次；同一步死亡与越界只记死亡。

训练器口径是 alive-before active-agent mean。审计同时提供：

- 所有红方 agent reward 直接求和；
- 固定 initial-red-count mean；
- trainer active-agent mean。

高度 episode 聚合为：`above_altitude_max_steps=sum`、`max_altitude_m=max`、`above_altitude_max_episode_flag=max/any`。不生成 `max_altitude_m_sum` 或 `above_altitude_max_episode_flag_sum`。

## 参数来源

- 论文直接给定/复用语义：TAM `R_V/R_A/R_D`、连续 awareness、dodge angle/speed 形式；BRMA 飞行项。
- 当前环境映射：三维位置/速度、scripted launch/evasion、current-state MAV shared observation、真实在飞导弹 threat。
- 实现约定：reward target 最近三维蓝机、首个 launch target 快照、1000 m/s dodge speed normalization、事件去重集合、诊断聚合口径。

## 审计脚本

```bash
python scripts/audit_brma_tam_scripted_composite_v1.py --episodes 2 --max-steps 100
```

输出：

- `brma_tam_scripted_composite_v1_components.csv`：每个红方 agent 每决策步的 reward、active/log-only components、生命周期和 target diagnostics；
- `reward_target_diagnostics.csv`：每个存活攻击 UAV 每步至多一条 target/track/lock/launch/action-source 记录；
- `episode_summary.csv`：三种 team reward 聚合、折扣回报、component、outcome、发射/命中和高度汇总；
- `audit_summary.json`、`audit_summary.md`：按 red win、blue win、draw、timeout 分组的汇总和诊断行数告警。

审计使用固定零动作且不训练，仅验证 reward contract，不代表 learned-policy 性能。
