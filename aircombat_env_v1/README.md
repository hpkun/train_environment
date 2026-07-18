# aircombat_env_v1

## recommended: SimpleTAMCombatEnv

`SimpleTAMCombatEnv` 是当前推荐的简化 JSBSim 多智能体空战环境，支持 `simple_paper_1v1` 与 `simple_paper_2v2`。飞机使用本地 F-16 六自由度模型；学习动作是 `Box(-1, 1, (3,))` 高层目标 `[pitch, relative_heading, speed]`，分别映射为俯仰 ±20°、相对航向 ±60° 和速度 200–300 m/s。每个决策持续 12 个 60 Hz 物理帧，每帧均由现有 `f16_pid_v1.yaml` 固定 PID 执行，不包含 fire 或底层舵面动作。

`temporary_learnability_abstraction`: the paper low-level direct-FCS action is replaced by a high-level pitch-heading-speed command executed by a fixed PID controller.

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
