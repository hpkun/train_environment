# Experiment guide

## 1. Conda 环境

本项目主要在 `brmamappo` conda 环境中运行。Windows PowerShell 示例：

```powershell
conda activate D:\conda_envs\envs_dirs\brmamappo
```

也可以直接使用该环境的 Python 解释器：

```powershell
D:\conda_envs\envs_dirs\brmamappo\python.exe ...
```

## 2. 快速 smoke test

```powershell
D:\conda_envs\envs_dirs\brmamappo\python.exe train_vanilla_mappo.py --num-red 1 --num-blue 1 --num-envs 1 --total-env-steps 20 --replay-buffer-size 10 --max-episode-length 10 --device cpu --log-file smoke_train_log.csv --results-file results/smoke_results.csv --checkpoint-dir smoke_checkpoints
```

该命令只用于验证训练链路是否能启动、写日志和保存结果，不代表有效训练结果。

## 3. 当前默认训练命令

默认仍是 2v2 vanilla MAPPO baseline：

```powershell
D:\conda_envs\envs_dirs\brmamappo\python.exe train_vanilla_mappo.py
```

默认输出：

- `vanilla_training_log.csv`
- `results/vanilla_mappo_results.csv`
- `checkpoints/`

训练默认 `enable_blue_gcas=False`。

## 4. Preset-based commands

项目支持通过 `--preset` 缩短常用命令。列出所有 preset：

```powershell
conda activate brmamappo
python train_vanilla_mappo.py --list-presets
python train_attention_mappo.py --list-presets
```

常用 preset 示例：

```powershell
# vanilla 1v1 smoke (20 steps, cpu)
python train_vanilla_mappo.py --preset vanilla_1v1_smoke

# vanilla 2v2 smoke (~10k steps, see reward signals)
python train_vanilla_mappo.py --preset vanilla_2v2_smoke

# vanilla 2v2 main (10M steps, full training)
python train_vanilla_mappo.py --preset vanilla_2v2_main

# attention smoke variants
python train_attention_mappo.py --preset attention_1v1_smoke
python train_attention_mappo.py --preset attention_2v2_current_smoke
python train_attention_mappo.py --preset attention_2v2_placeholder_smoke
```

CLI 参数仍可覆盖 preset：

```powershell
python train_vanilla_mappo.py --preset vanilla_2v2_smoke --total-env-steps 2000
```

当前默认 reward version 为 `fixed_ta_alt_eq17_3dlos_v1`。详见
[docs/current_environment_alignment_status.md](current_environment_alignment_status.md)。

## 5. 论文式 6v6 训练命令模板

下面是 6v6 训练命令模板。注意：当前仍只是 vanilla MAPPO baseline，不是 BRMA-MAPPO。

```powershell
D:\conda_envs\envs_dirs\brmamappo\python.exe train_vanilla_mappo.py --num-red 6 --num-blue 6 --num-envs 8 --total-env-steps 10000000 --max-episode-length 1400 --device auto --log-file logs/vanilla_6v6.csv --results-file results/vanilla_6v6_results.csv --checkpoint-dir checkpoints_vanilla_6v6
```

## 6. 批量评估

`evaluate_vanilla_mappo.py` 不生成 ACMI，主要用于多局统计论文式指标。默认 `enable_blue_gcas=False`，与训练脚本和 ACMI 单局评估保持一致。若需要显式开启蓝方 GCAS，可添加 `--enable-blue-gcas`。

随机策略 smoke test：

```powershell
D:\conda_envs\envs_dirs\brmamappo\python.exe evaluate_vanilla_mappo.py --random --num-red 1 --num-blue 1 --episodes 2 --max-steps 10 --device cpu --output results/smoke_eval_metrics.csv
```

评估 trained 2v2 checkpoint：

```powershell
D:\conda_envs\envs_dirs\brmamappo\python.exe evaluate_vanilla_mappo.py --checkpoint checkpoints/vanilla_actor_best.pt --num-red 2 --num-blue 2 --episodes 20 --max-steps 1400 --device auto --output results/eval_2v2.csv
```

vanilla MLP baseline 的 flattened observation 维度随规模变化，因此不能直接把 2v2 checkpoint 用到 6v6、8v8 或 10v10。这不是 BRMA zero-shot 设置。

## 7. Tacview ACMI 单局可视化

`eval_acmi.py` 用于单局 Tacview 可视化，不用于批量统计。该脚本默认显式使用 `enable_gcas_for_blue=False`。

```powershell
D:\conda_envs\envs_dirs\brmamappo\python.exe eval_acmi.py --checkpoint checkpoints/vanilla_actor_best.pt --num-red 2 --num-blue 2 --max-steps 1400 --output eval_battle.acmi
```

随机策略 smoke test：

```powershell
D:\conda_envs\envs_dirs\brmamappo\python.exe eval_acmi.py --random --num-red 1 --num-blue 1 --max-steps 10 --output smoke_eval.acmi
```

## 8. 当前已对齐论文的内容

- 雷达 `Rmax = K * RCS^(1/4)`。
- 导弹 `0.25s` lock delay。
- 导弹 `0.5s` launch interval。
- 导弹命中概率使用 missile velocity 与 LOS dot product。
- boundary reward 使用 eq.18 的固定单次越界惩罚。
- roll reward 使用 eq.16 double-condition。
- altitude reward 使用二次分段近似。
- terminal reward 按 per-agent API 均分。
- 增加论文式评估指标。

## 9. 当前仍未对齐论文的内容

- 默认训练仍是 2v2，不是论文 6v6。
- 算法仍是 vanilla MAPPO，不是 BRMA-MAPPO。
- 尚未实现 EntityObservationEncoder。
- 尚未实现 MaskVectorGenerator。
- 尚未实现 biased random masked attention。
- observation 仍是当前 11 维工程化 entity vector，不是严格 Table 1 / Table 2。
- critic 仍使用 red agents flattened observations concat，不是论文 native global state。
- RCS 仍是 front/side approximation，不是论文 RCS table interpolation。
- PID 控制器含工程稳定项。
- 论文没有明确给出每架 UAV 的固定载弹量；当前环境保留默认 `num_missiles_per_plane=999`，等价于不让载弹量成为主要限制因素。由于论文没有提供具体载弹量，该项暂不作为优先对齐目标。

## 10. Git ignore 注意事项

以下文件不应提交：

- `smoke_train_log.csv`
- `smoke_checkpoints/`
- `smoke_eval.acmi`
- `results/smoke_*.csv`
- `__pycache__/`
- `*.pyc`

如果生成了上述文件，请保持它们处于 git ignored 状态，不要加入提交。

## 11. 下一阶段：EntityObservationEncoder 准备

- 已新增 `entity_obs_utils.py`，可将当前 Dict observation 转成 entity-wise tensor。
- 当前 tensor 暂时仍使用环境的 11 维工程化 entity vector。
- 该工具暂未接入训练，只用于后续实现 MAPPO-Attention / BRMA-MAPPO。
- 后续仍需决定是否严格改成论文 Table 1 / Table 2 的 10 维表示。

## 12. MAPPO-Attention 准备

- 已新增 `attention_models.py`。
- 目前包含 `EntityObservationEncoder`、`AttentionActor`、`AttentionCritic`。
- 当前模块尚未接入训练，仅通过纯 PyTorch smoke test 验证 shape。
- 下一步才会新增 `train_attention_mappo.py` 或在独立分支中接入训练。
- 当前 attention encoder 使用 11 维工程化 entity vector，不是最终论文 Table 1 / Table 2 的 10 维严格版本。
- 当前还没有实现 biased random mask 和 mask vector generator。

## 13. MAPPO-Attention baseline

- 已新增 `train_attention_mappo.py`。
- 这是 actor-side EntityObservationEncoder baseline。
- Critic 暂时仍使用 flattened red observations concat 的 centralized critic。
- 尚未实现 biased random mask 和 MaskVectorGenerator。
- 默认输出：
  - `attention_training_log.csv`
  - `results/attention_mappo_results.csv`
  - `checkpoints_attention/`
- `train_attention_mappo.py` 支持 `--obs-adapter current` 和 `--obs-adapter paper-placeholder`。
- `current` 是默认，使用当前 11 维工程化 entity vector。
- `paper-placeholder` 使用 10 维 placeholder adapter，不是 strict Table 1/Table 2 物理量。
- strict paper extractor 已在 `paper_state_extractor.py` 中作为原型存在，但尚未接入 SubprocVecEnv 训练。
- 使用 `paper-placeholder` 时应使用独立 checkpoint 目录，例如 `checkpoints_attention_paper_placeholder`。

smoke 命令：

```powershell
conda activate brmamappo
python train_attention_mappo.py --num-red 1 --num-blue 1 --num-envs 1 --total-env-steps 20 --replay-buffer-size 10 --max-episode-length 10 --device cpu --log-file smoke_attention_log.csv --results-file results/smoke_attention_results.csv --checkpoint-dir smoke_attention_checkpoints
```

这条命令会触发 JSBSim 环境 reset，Codex 不运行；由本地用户运行。

paper-placeholder smoke 命令：

```powershell
conda activate brmamappo
python train_attention_mappo.py --obs-adapter paper-placeholder --num-red 1 --num-blue 1 --num-envs 1 --total-env-steps 20 --replay-buffer-size 10 --max-episode-length 10 --device cpu --log-file smoke_attention_paper_log.csv --results-file results/smoke_attention_paper_results.csv --checkpoint-dir smoke_attention_paper_checkpoints
```

这条命令同样会触发 JSBSim 环境 reset，Codex 不运行；由本地用户运行。

## 14. 论文式 observation adapter 准备

- 已新增 `paper_obs_utils.py`。
- 当前只是把现有 11 维工程化 entity vector 转成 10 维接口占位。
- 它还不是严格论文 Table 1/Table 2 的物理量复现。
- 后续若要严格复现，需要从 simulator/native state 中构造：
  - self state: `x, y, h, V, phi, theta, psi, alpha, beta, Vd`
  - relative state: `x_body, y_body, z_body, theta_v_body, psi_v_body, V, theta_LOS_body, psi_LOS_body, q_LOS, d`
- 在完成 strict observation 前，`train_attention_mappo.py` 的结果只能视作工程 baseline，而不是论文 MAPPO-Attention 消融结果。

## 15. Strict paper observation prototype

- `UavCombatEnv` 已暴露 `get_strict_entity_observation(agent_id)` 和
  `get_strict_team_observations(team)`。
- `reset()`/`step()` 默认 observation 仍是 11 维工程 Dict，不受影响。
- strict API 后续可用于 `train_attention_mappo.py` 的 paper-strict adapter。

smoke 命令（触发 JSBSim，Codex 不运行，用户本地运行）：

```powershell
conda activate brmamappo
python scripts/smoke_strict_observation_env.py
```

- `paper_state_extractor.py` 仍在 `my_uav_env/alignment/state_extractor.py` 中。
- 它尝试从 simulator/native state 构造论文 Table 1/Table 2 的 10 维观测。
- 当前 `alpha/beta` 可能仍是 placeholder 0，除非 simulator 已提供对应属性。
- pass13 后 extractor 会尝试从 JSBSim property 读取 `aero/alpha-rad`、`aero/alpha-deg`、`aero/beta-rad`、`aero/beta-deg`。
- extractor 现在会在 meta 中记录 `alpha/beta` 的来源。
- `q_LOS` 的定义仍需和论文几何定义核对。
- 当前 `q_LOS` 是 observer body x-axis angle placeholder，不等同于 3-9 线尾后角。
- 后续接入训练前，还需要进一步验证 `q_LOS` 与现有 AO/TA 的关系。
- `radar_detected=False` 时会按论文 Table 2 Note 将速度角和目标速度置 0。
- `scripts/smoke_paper_state_extractor_env.py` 会打印每个 entity 的物理字段，用于本地检查数值方向和量级。
- Codex 不运行该脚本，用户本地运行。
- 该模块尚未接入训练；后续需要先验证数值合理性，再决定是否让 `train_attention_mappo.py` 使用它。

## 16. MAPPO-Attention 批量评估

- 已新增 `evaluate_attention_mappo.py`。
- 它评估 attention actor checkpoint，不生成 ACMI。
- 支持 `--obs-adapter current` / `--obs-adapter paper-placeholder`。
- checkpoint 的 `entity_dim` 必须和 `obs_adapter` 匹配。
- 仍未实现 BRMA mask。

current adapter 评估模板：

```powershell
conda activate brmamappo
python evaluate_attention_mappo.py --checkpoint checkpoints_attention/attention_actor_best.pt --obs-adapter current --num-red 2 --num-blue 2 --episodes 20 --max-steps 1400 --device auto --output results/eval_attention_2v2.csv
```

paper-placeholder adapter 评估模板：

```powershell
conda activate brmamappo
python evaluate_attention_mappo.py --checkpoint checkpoints_attention_paper_placeholder/attention_actor_best.pt --obs-adapter paper-placeholder --num-red 2 --num-blue 2 --episodes 20 --max-steps 1400 --device auto --output results/eval_attention_paper_placeholder_2v2.csv
```

随机 smoke 示例：

```powershell
conda activate brmamappo
python evaluate_attention_mappo.py --random --obs-adapter current --num-red 1 --num-blue 1 --episodes 2 --max-steps 10 --device cpu --output results/smoke_eval_attention.csv
```

这条 smoke 命令会触发 JSBSim 环境 reset，Codex 不运行；由本地用户运行。

## 17. Paper-style critic global state candidate

`my_uav_env/alignment/global_state.py` 提供 strict team global state flatten 工具。
当前 attention critic 仍使用 `obs_dim * num_red`（2v2 约 106 维）的 engineering flatten。
strict candidate 2v2 维度为 88（4 entities × 10 dim + 4 mask = 44 per agent × 2）。

本 pass 只做候选工具，不改变训练行为。后续将单独做 critic switch pass。

## 18. Reward version 标记

当前 reward version 为 `fixed_ta_alt_eq17_3dlos_v1`。

完整的环境对齐状态见 [docs/current_environment_alignment_status.md](current_environment_alignment_status.md)。

`fixed_ta_alt_eq17_3dlos_v1` 表示：

1. Ta 使用连续非负归一化版本；
2. altitude reward 使用 pairwise eq.17-style（含 high-altitude 0.1 tail）；
3. situation reward 已从 2D 水平面 AO/TA 切换为 3D body-x q_LOS + 3D distance。

注意：

- 不要与 `fixed_ta_v1`、`fixed_ta_alt_eq17_v1`、或 legacy reward 日志混合比较。
- 新实验建议使用带版本名的日志文件，例如 `vanilla_3dlos_v1.csv`。
- 如果后续需要 paper Ta scale (10×) ablation，应另开 reward-scale 实验，不要与 `fixed_ta_alt_eq17_3dlos_v1` baseline 混合。
## 18. Attention strict observation adapter

`train_attention_mappo.py` now supports three actor observation adapters:

- `--obs-adapter current`: default 11-dim engineering entity vector.
- `--obs-adapter paper-placeholder`: 10-dim placeholder projection from current env obs.
- `--obs-adapter strict`: 10-dim strict Table 1/Table 2 prototype observation from `UavCombatEnv.get_strict_team_observations("red")`.

Strict mode only changes the attention actor input. It does not change `reset()` / `step()` default observation, does not change `UavCombatEnv.observation_space`, and does not change the centralized critic. The critic still uses flattened 11-dim red observations concat.

Strict smoke preset:

```powershell
conda activate brmamappo
python train_attention_mappo.py --preset attention_1v1_strict_smoke
```

This command triggers JSBSim/env reset and is for local user runs only; Codex does not run it.
