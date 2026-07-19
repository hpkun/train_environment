# aircombat_env_v1

## recommended: SimpleTAMCombatEnv

`SimpleTAMCombatEnv` 是当前推荐的简化 JSBSim 多智能体空战环境，支持 `simple_paper_1v1`、`simple_paper_2v2` 与 `simple_paper_3v2_hetero`。飞机使用本地 F-16 六自由度模型；学习动作是 `Box(-1, 1, (3,))` 高层目标 `[pitch, relative_heading, speed]`，分别映射为俯仰 ±20°、相对航向 ±60° 和速度 200–300 m/s。每个决策持续 12 个 60 Hz 物理帧，每帧均由现有 `f16_pid_v1.yaml` 固定 PID 执行，不包含 fire 或底层舵面动作。

`temporary_learnability_abstraction`: the paper low-level direct-FCS action is replaced by a high-level pitch-heading-speed command executed by a fixed PID controller.

`temporary_learnability_abstraction`: the first heterogeneous environment represents role and armament heterogeneity while reusing the same JSBSim aircraft dynamics for MAV and UAV agents.

### Paper-aligned heterogeneous perception and support

`simple_paper_3v2_hetero` 使用 `HeterogeneousPerceptionSystem` 将红方 MAV 的战场感知支援落实到局部观测、目标选择和自动火控。默认 `hetero_perception_mode="paper_fused"`；`uav_only_ablation` 只关闭 MAV-to-UAV 信息共享，MAV 本身、初始部署、飞行、任务终止条件、动作/观测维度、奖励和武器模型均不变。因此，两种模式可以复用同一个 `(81, 243, 3)` MAPPO checkpoint，并用于隔离信息支援的因果作用，而不是创建新的训练场景。

`paper_explicit`：环境按 POMDP 建模；每机观测由7维自身状态以及友机、敌机、来袭导弹的相对速度、相对高度、距离、ATA和AA构成；MAV增强异构编队战场感知并提供敌机位置、速度等信息，承担协调、信息支援和生存任务；UAV承担协同接敌、攻击和支援。目标优先级使用论文态势评分权重 `angle=0.35`、`distance=0.25`、`altitude=0.20`、`speed=0.20`，仅用于目标选择，不进入reward。论文没有建模通信约束、感知不确定性或控制时延。

`paper_unspecified_engineering`：本项目令UAV直接探测距离为14 km、MAV探测距离为28 km，并采用理想、即时、无限通信距离的数据链。没有雷达视场、地形遮挡、探测概率、噪声、丢包、时延或航迹保持；目标在当前决策步不可见时立即清空对应敌机槽位和mask。态势评分中的14 km距离门槛、6000 m相对高度归一化和400 m/s相对速度归一化也是当前环境映射，不是论文公开参数。固定规则蓝方继续使用全局真实存活目标集合，这是保持规则对手简单的工程抽象，不代表蓝方局部观测模型。

在 `paper_fused` 中，红方UAV可见集合是“自身14 km直接探测”与“存活MAV在28 km内探测并共享”的稳定排序并集；MAV只使用自身探测，不接收其他飞机共享。`uav_only_ablation` 中，红方UAV只保留自身直接探测，MAV仍独立探测但不共享。红方友机始终可见，锁定本机的存活来袭导弹不受敌机可见距离限制。Actor仍只接收原论文状态变量：3v2观测保持81维，末尾2维role one-hot不变，direct/shared来源只进入info诊断而不扩展观测。

红方UAV只能从自身当前可见目标中按论文态势评分选靶；同分时按 `agent_id` 稳定排序。MAV也按同一评分从自身探测目标中选靶，但导弹容量和剩余弹药始终为0。目标失去可见性后会在同一决策步重新选择或清空；无可见目标时规则UAV稳定巡航，不追踪隐藏敌机，也不能自动发射。共享目标可以支持选靶和规则机动，但实际发射仍必须同时满足现有14 km最大攻击距离与25 s冷却约束。

短验收命令：

```powershell
python -m pytest aircombat_env_v1/tests/test_hetero_perception.py aircombat_env_v1/tests/test_simple_hetero_environment.py aircombat_env_v1/tests/test_simple_mappo_multiseed.py -q
python aircombat_env_v1/scripts/check_simple_hetero_environment.py
```

### Heterogeneous reward contracts

3v2环境通过`hetero_reward_mode`显式选择MAV奖励契约，UAV始终使用原有`PaperReward`。`legacy_v1`完整保留早期可学习性工程奖励：战场中心只取存活红方攻击UAV，awareness使用最大项，aspect使用最小项，Safety和Support在顶层各乘10，并保留旧事件与boundary叠加；该模式只用于历史实验对照。

默认`paper_table1_v2`采用论文Table 1的`Safety + Support + Event`结构：`Safety = 0.5 R_dist + 0.3 R_threat + 0.2 R_aspect`，`Support = 0.6 R_pos + 0.4 R_aware`。Aspect和awareness均对满足角度条件的目标求和。动态战场中心由所有存活红方攻击UAV与存活蓝方UAV的三维位置算术平均构成，明确排除MAV。MAV awareness使用12个物理帧推进后的当前MAV探测集合，而不是决策前航迹。

v2 Event只包含MAV当步死亡`-200`和红方攻击UAV真实击杀蓝方UAV的团队贡献：每次`+100`、每回合最多`+200`，额度由奖励模型跨step维护并在环境reset时清零。boundary、crash、numerical invalid和导弹命中均统一触发一次MAV死亡成本，不再叠加旧boundary处罚。v2没有Safety/Support十倍倍率、terminal overlay、alive/relay/rear/launch/hit/timeout奖励；relay-only与共享航迹只作为诊断日志。

`paper_unspecified_engineering`常数为：危险距离14 km、安全距离28 km、支援最优距离14 km、支援最大距离28 km、MAV死亡成本200、单次团队击杀额度100、回合上限200。这些数值分别映射当前导弹攻击距离、MAV工程探测距离和本项目事件尺度，不是论文公开的MAV阈值或成本。

新checkpoint保存`environment_contract`，包含场景、感知模式、奖励模式、奖励契约版本、观测/状态/动作维度和agent IDs。评估默认拒绝奖励模式不匹配；感知模式允许覆盖以执行预定消融。旧checkpoint没有该契约时仍可加载，但评估结果将`checkpoint_reward_contract_known`标为`false`。训练同时输出逐回合`episode_reward_components.csv`和`train_log.csv`最近20回合MAV/UAV分量统计。

```powershell
python aircombat_env_v1/scripts/check_simple_hetero_reward.py
```

武器由环境自动管理，直接复用论文导弹、观测槽位与奖励结构。默认红方由学习接口控制，蓝方使用包含 `level_hold`、`pursuit`、转向、升降和加减速的有限高层候选，并按论文奖励结构即时贪心选择。该蓝方是基于论文有限基本机动思想构建的简化贪心规则策略，不是论文未公开 FSM 的精确复现。

```powershell
python aircombat_env_v1/scripts/check_simple_environment.py
python aircombat_env_v1/scripts/run_simple_rule_combat.py --check all --episodes 10
python -m pytest aircombat_env_v1/tests -q
python -m pytest aircombat_env_v1/tests -q -m integration
```

### Minimal MAPPO baseline

`simple_mappo.py` 为 `SimpleTAMCombatEnv` 提供标准前馈 MAPPO：红方共享参数Actor、拼接红方局部观测的集中式Critic、tanh-squashed Gaussian、GAE、PPO裁剪和死亡智能体mask。训练与评估入口分别为 `scripts/train_simple_mappo.py` 和 `scripts/eval_simple_mappo.py`，不运行时依赖 `hetero_uav`。名义场景是确定性的，多次确定性评估主要用于流程一致性，不代表独立随机样本的统计置信度。

## experimental_or_legacy

### formal_paper_environment

`TAMPaperCombatEnv` 是正式论文对齐路径，支持 `paper_nominal_1v1` 和 `paper_nominal_2v2`。每架受控 F-16 使用 `MultiDiscrete([40,40,40,40])`，直接映射为 throttle、aileron、elevator、rudder；每个决策保持 12 个 60 Hz 物理帧，不经过 PID。

正式路径直接复制适配 TAM 的 `PaperMissile`、`PaperWeaponManager`、`PaperObservation`、`PaperReward` 和 `GreedyPaperOpponent`。攻击 UAV 自动发射，只检查双方存活、剩余弹药、距离不超过 14 km 和 25 s 冷却；没有 fire 动作、锁定角、最小射程、命中概率或“同目标仅一枚”限制。

论文显式导弹参数为 14 km、25 s、30 g、`Ky=Kz=3`、84 kg、2.87 m、0.127 m。论文未公开的正式项目假设为初速 500 m/s、动力段 3 s、加速度 110 m/s²、命中半径 60 m、有效阻力 `0.00012/m`。飞行包线为高度不低于 750 m、速度不高于 400 m/s、绝对过载不高于 9 g，仅逐物理帧记录，不附加保护。

结构化观测包含 7 维自身状态、每个相对实体 5 维状态以及固定槽位 mask；1v1 flatten 为 61 维，2v2 为 73 维。2v2 目标选择采用“最近存活敌机，距离相同时按 agent_id”，这是论文未公开的工程实现。

作战边界已启用：正式 TAM `protocol.py` 将 `combat_zone_radius_m` 明确定义为 `2 × maximum_attack_range_m`，`task.py` 在每个 60 Hz 帧执行该 28 km 半径判断并施加越界事件/奖励。本目录直接复制该正式项目定义；28 km 是 TAM 工程派生值，不是论文显式空间边界。

`weapon_enabled_agent_ids` 是仅供验收诊断的可选开关；默认 `None` 时所有攻击 UAV 武器正常启用，不改变正式行为。

正式 JSBSim 初始化显式收起起落架与襟翼、启动发动机并进入 propulsion steady state；正式过载口径为 `accelerations/Nz`。基础机动索引属于 `paper_unspecified_local_jsbsim_calibration`，不是论文公开参数。离线审计命令为 `python aircombat_env_v1/scripts/calibrate_paper_basic_manoeuvres.py`。只有双航向 1000 步 level 门槛通过后，才可将其作为 initial direct-FCS command；该初始命令是本地工程校准值，因为论文没有公开 initial FCS trim。

名义初始条件直接来自 `tam_paper_env_v1_2v2.yaml`：6000 m、250 m/s，红方航向 0°、蓝方 180°；2v2 位置为 red `(120,60)/(120.02,60)`、blue `(120,60.2)/(120.02,60.2)`，每架两枚导弹。

```powershell
python aircombat_env_v1/scripts/check_paper_environment.py
python aircombat_env_v1/scripts/solve_paper_trim.py
python aircombat_env_v1/scripts/check_paper_direct_fcs_health.py
python aircombat_env_v1/scripts/run_paper_rule_combat.py --scenario paper_nominal_2v2 --episodes 20
python -m pytest aircombat_env_v1/tests -q
python -m pytest aircombat_env_v1/tests -q -m integration
```

### legacy_debug_environment

旧 `AirCombat1v1Env` 的 Dict maneuver/fire 接口、PPO 与 recurrent PPO 仅保留为历史调试实验；当前推荐环境不调用这些实现。

`TAMPaperCombatEnv` 四维40档直接飞控、`paper_trim.py`、配平搜索和固定四维基础机动校准均保留为实验记录，不是当前推荐环境的运行路径。旧 PPO、recurrent PPO 及含 fire 动作的 `AirCombat1v1Env` 同样只归档保留。
