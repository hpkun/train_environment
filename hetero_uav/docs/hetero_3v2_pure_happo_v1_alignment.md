# hetero_3v2_pure_happo_v1 论文对齐说明

## 定位

该环境是固定 3V2 异构空战的独立正式基座，用于检验 Pure HAPPO 的环境可学习性。它不是 TAM-HAPPO 或 BRMA-MAPPO 的完整复现：TAM 原文采用四维离散底层控制，而本基座复用 BRMA 的三维高层连续动作和 PID，以隔离角色与感知异构。

## 原文依据

- Chen, Luo and Guo, *A deep reinforcement learning cooperative air combat method with temporal feature and attention enhancement for heterogeneous flight vehicles*, Aerospace Science and Technology 176 (2026) 112537。
- Tan et al., *Biased random masked attention MAPPO algorithm for zero-shot scale generalization of multi-UAV air combat*, Journal of Computational Design and Engineering 13(3), 2026, 46-68。

| 正式机制 | 标签 | 原文位置 | 本实现 |
|---|---|---|---|
| 1 MAV + 2 UAV 对 2 UAV，MAV 无弹、UAV 各 2 弹 | TAM_EXPLICIT | TAM Table 6，PDF p.12；场景说明 PDF p.8 | 固定五机与角色，不随机化 |
| 3V2 经纬度、高度、速度、航向 | TAM_EXPLICIT | TAM Table 6，PDF p.12 | 原值写入 JSBSim IC |
| 60 Hz 物理、5 Hz 决策、动作保持 12 帧 | TAM_EXPLICIT | TAM Tables 2-4，PDF p.9 | `60/5/12` |
| 1000 决策步 | TAM_EXPLICIT | TAM Table 4，PDF p.9 | timeout 截断 |
| 高层 pitch/heading/speed + PID | BRMA_EXPLICIT | BRMA Sec. 2.4, Eq. (12)-(14)，PDF p.5 | 连续 Box(3)，统一 F16 PID |
| 目标态势评估 | PAPER_EQUATION | TAM Sec. 2.4, Eq. (10)-(12)，PDF p.4 | 角度/距离/高度/速度，权重 `.35/.25/.20/.20` |
| 蓝方候选机动即时贪心 | TAM_EXPLICIT | TAM Sec. 3，Table 4，PDF p.9 | 固定候选集，以 `10:10:15:10:30` 评分 |
| 14 km、25 s、30 g | TAM_EXPLICIT | TAM Table 3，PDF p.9 | 红蓝一致协议 |
| 比例导航 K=3 | BRMA_EXPLICIT | BRMA Sec. 2.1.3, Eq. (7)-(11)，PDF p.4 | 单一确定性点质量 PN |
| 3-9 line / rear hemisphere | BRMA_EXPLICIT | BRMA Sec. 2.1.3，PDF p.4 | ATA 60 deg、TA 90 deg；TAM 未公开对应角阈值 |
| UAV 角色奖励类别与事件值 | TAM_EXPLICIT | TAM Table 1，PDF p.8 | height/speed/angle/distance/dodge；kill/death/out-of-zone 事件 |
| MAV safety/support/event | TAM_EXPLICIT | TAM Table 1，PDF p.8 | safety `.5/.3/.2`、support `.6/.4`；未公开常数见下 |
| 共享团队 reward/advantage | LEARNABILITY_ADAPTATION | 用于本阶段 Pure HAPPO 合同 | 角色回报先规范化，再固定求均值并广播 |
| UAV 10 km 直接探测 | BRMA_EXPLICIT | BRMA Table 4，PDF p.12 | 确定性距离门限 |
| MAV 80 km 探测 | LEARNABILITY_ADAPTATION | TAM 未公开探测距离 | 保证 Table 6 初始态存在支援信号 |
| 导弹 600 m/s、300 m 命中半径、60 s 寿命、0.2 s 起爆 | LEARNABILITY_ADAPTATION | 两文未完整公开 | 简单确定性数值合同，不声称武器复现 |
| MAV safety 距离、支援位置尺度、奖励总尺度 | LEARNABILITY_ADAPTATION | TAM Table 1 公式中的常数未公开 | 明示在 `reward.py`；稠密 raw 先除固定 1000-step horizon，再与事件一起仅做一次 `/200` 总归一化 |
| 通信误差、感知噪声、控制延迟 | REMOVED_LEGACY | TAM conclusion 明确作为后续工作 | 不实现 |
| v5/v6、BRMA overlay、GCAS、scripted evasion、target hold/reallocation | REMOVED_LEGACY | 非该正式合同 | 新路径不导入 |

## 最终合同

- **动作**：`[target_pitch, target_heading, target_speed]`，三维连续值 `[-1,1]`；红方各机仅由各自动作控制，死亡机 active mask 失效。
- **观测**：ego 11、2 个 ally slot 各 11、2 个 enemy slot 各 14、最近来袭弹 7，flatten 为 68。未观测敌机实体全零。critic 为三个红方可用 flatten 的拼接，共 204 维，死亡红方块置零。
- **共享感知**：UAV 始终保留直接探测；MAV 存活并探测目标时提供带来源标记的共享 track；MAV 死亡后立即消失。
- **火控**：可观测存活目标按 TAM Eq. (12) 即时选优；允许多 UAV 追踪同一目标；任意在途弹仅阻止对该目标的新发射，不清除追踪。
- **导弹**：单一确定性点质量 PN；状态 `launched/hit/miss`，终止原因 `hit/target_dead/timeout/out_of_bounds`。
- **奖励**：TAM 角色项、有界事件一次性计入、单处总尺度归一化，最终广播共享团队回报；无 launch/lock/hold bonus，无 BRMA overlay。
- **终止**：红全灭、蓝全灭、确认数值异常或 1000 步 timeout，互斥记录；MAV 单独死亡不结束回合。

## 声明边界

该环境保留 JSBSim 六自由度飞机动力学，但三维 PID 动作是论文对齐适配，不是 TAM 四维直接舵面的精确复现。导弹是 paper-aligned kinematic approximation，不是完整武器工程模型。通信约束、感知误差和控制延迟均未建模。
