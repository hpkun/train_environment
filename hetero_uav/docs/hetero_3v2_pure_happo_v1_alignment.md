# hetero_3v2_pure_happo_v1 对齐与奖励合同

## 定位

该环境是固定 3V2 异构空战的 Pure HAPPO 正式基座，不是 TAM-HAPPO 或 BRMA-MAPPO 的完整复现。TAM 原文采用四维离散底层控制；本环境复用 BRMA 的三维连续高层动作和 PID，仅验证角色与感知异构的可学习性。

## 来源表

| 机制 | 标签 | 原文位置 | 正式实现 |
|---|---|---|---|
| 1 MAV + 2 attack UAV 对 2 UAV；MAV 无弹、UAV 各 2 弹 | TAM_EXPLICIT | TAM Table 6，PDF p.12 | 固定五机和角色 |
| 3V2 经纬度、高度、速度、航向 | TAM_EXPLICIT | TAM Table 6，PDF p.12 | 原值写入 JSBSim IC |
| 60 Hz 物理、5 Hz 决策、动作保持 12 帧、1000 steps | TAM_EXPLICIT | TAM Tables 2-4，PDF p.9 | `60/5/12/1000` |
| 高层 pitch/heading/speed + PID | BRMA_EXPLICIT | BRMA Sec. 2.4, Eq. (12)-(14)，PDF p.5 | Box(3)，统一 F16 PID |
| 目标评估变量和 `.35/.25/.20/.20` 权重 | PAPER_EQUATION | TAM Sec. 2.4, Eq. (10)-(12)，PDF p.4 | 仅用于选目标，不进入 reward |
| 蓝方候选机动即时贪心 | TAM_EXPLICIT | TAM Sec. 3、Table 4，PDF p.9 | 穷举目标乘固定候选动作 |
| 14 km、25 s、30 g | TAM_EXPLICIT | TAM Table 3，PDF p.9 | 红蓝使用同一协议 |
| 点质量比例导航 `K=3` | BRMA_EXPLICIT | BRMA Sec. 2.1.3, Eq. (7)-(11)，PDF p.4 | 确定性 PN |
| 3-9 line / rear hemisphere | BRMA_EXPLICIT | BRMA Sec. 2.1.3，PDF p.4 | ATA 60 deg、TA 90 deg |
| UAV flight/speed/angle/distance/dodge/事件目标 | TAM_EXPLICIT | TAM Table 1，PDF p.8 | 变量和角色意图来自论文，归一化见下 |
| MAV safety/support/information/event 目标 | TAM_EXPLICIT | TAM Table 1，PDF p.8 | 变量和角色意图来自论文，归一化见下 |
| UAV 10 km 直接探测 | BRMA_EXPLICIT | BRMA Table 4，PDF p.12 | 确定性距离门限 |
| MAV 80 km 探测 | LEARNABILITY_ADAPTATION | TAM 未公开范围 | 确定性支援探测，不含随机丢包或延迟 |
| 600 m/s、300 m 命中半径、60 s 寿命、0.2 s 起爆 | LEARNABILITY_ADAPTATION | 两文未完整公开 | 数值合同，不声称武器精确复现 |
| 奖励阈值、归一化和事件尺度 | LEARNABILITY_ADAPTATION | TAM 未公开完整常数 | 集中定义于 `formal_v1/reward.py` |
| 通信误差、感知噪声、控制延迟 | REMOVED_LEGACY | TAM conclusion 列为后续工作 | 不实现 |
| v5/v6、BRMA overlay、GCAS、scripted evasion、target hold/reallocation | REMOVED_LEGACY | 非正式合同 | 新路径不导入 |

## 几何定义

- ATA：射手速度向量与“射手到目标” LOS 的夹角，`ATA=0` 表示射手正指向目标。
- TA：目标速度向量与“目标到射手” LOS 的夹角，`TA=pi` 表示目标背向射手。
- 因此正式角度态势 `0.5*cos(ATA)-0.5*cos(TA)` 在 `ATA=0, TA=pi` 时为 `+1`，在 `ATA=pi, TA=0` 时为 `-1`。

## 攻击 UAV 奖励

所有分量在 `[-1,1]`，权重非负且和为 1：

| 分量 | 权重 | 适配常数与方向 |
|---|---:|---|
| flight | 0.20 | 高度 1-9 km、速度 160-360 m/s 为宽安全平台；100 m/10.5 km 和 80/460 m/s 为外界 |
| speed | 0.15 | 相对目标约 `+30 m/s` 最优，宽度 100 m/s；不安全本机速度压低得分 |
| angle | 0.25 | 使用上述 ATA/TA 单调几何 |
| distance | 0.20 | 3-10 km 最优；500 m 内碰撞风险和 14 km 外均降低 |
| dodge | 0.20 | `clip((risk_prev-risk_now)/0.20,-1,1)`；无导弹时严格为 0 |

导弹风险为最紧迫真实来袭弹的距离、闭合速度、TGO 和接近方向的有界组合。风险下降为正、风险上升为负；威胁结束后只在第一步产生正变化，随后回到 0。

## MAV 奖励

权重为 `safety=0.45`、`support_position=0.25`、`shared_information=0.30`：

- safety：自身飞行安全 0.25、最近敌机距离 0.25、敌机指向威胁 0.20、真实导弹绝对风险 0.15 和风险变化 0.15 的有界组合；无敌机不会自动得到满分。
- support position：MAV 到每架存活攻击 UAV 的 4-15 km 支援带最优；小于 750 m 或大于 30 km 均下降。
- shared information：只统计 `direct=false && mav_shared=true` 的存活 UAV-target 对。没有需要支援的目标时为 0，MAV 死亡后立即为 0。

## 事件与总尺度

- 红方导弹击落：对应射手 `+8`。
- attack UAV 死亡：该 UAV `-8`。
- MAV 死亡：MAV `-10`。
- 越界：对应飞机 `-6`。
- `global_reward_scale=1.0`，唯一总公式为 `reward_i = scale * (dense_i + event_i)`。

事件 `8` 相当于 40 个幅度为 `0.2` 的明显稠密改善步骤，位于要求的 20-50 steps 区间。该比例不依赖 `max_steps`；改变全局尺度会同时缩放 dense 和 event。

## Shared alive team mean

环境返回三个不同角色 reward，不广播也不除以 3。Pure HAPPO trainer 使用 transition 的 pre-action alive mask 计算：

`team_reward_t = sum_i(alive_before_t,i * reward_t,i) / sum_i(alive_before_t,i)`

死亡转移中该飞机的 `alive_before=1`，死亡事件进入共享 GAE；后续转移 active mask 为 0，不再产生稠密 reward，也不稀释存活飞机 reward。checkpoint meta 固定记录 `credit_mode=shared_alive_team_mean`。

## 目标评分

目标评分不进入 active reward。它保留 TAM 的四类变量和权重，但将不连续/无界部分替换为平滑函数：8 km 理想距离、7 km 距离宽度、50 m/s 理想闭合速度、150 m/s 闭合宽度、10 km 相对高度尺度。评分固定在 `[0,1]`。

## 最终接口

- actor observation：ego 11 + 2 ally x 11 + 2 enemy x 14 + incoming missile 7 = 68。
- critic：三个红方实际可用 flatten 拼接，共 204；死亡块置零，不泄漏未观测敌机。
- 动作：`[target_pitch,target_heading,target_speed]` Box(3)，无 trim、GCAS 或 scripted override。
- 终止：red win、blue win、mutual elimination、numeric anomaly 或 timeout；MAV 单独死亡不结束。

导弹仍是 paper-aligned kinematic approximation；三维 PID 动作也不是 TAM 四维直接舵面的精确复现。
