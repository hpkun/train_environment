# aircombat_env_v1

这是自包含的 JSBSim F-16 飞控与单智能体 1v1 导弹空战实验目录。飞控参数已冻结；正式环境保持 60 Hz JSBSim/PID 与 5 Hz 高层目标接口。

## 论文事实与项目假设

- `paper_explicit`：最低飞行高度 750 m、最大速度 400 m/s、最大飞机过载 9 g；导弹最大攻击距离 14 km、发射间隔 25 s、最大过载 30 g、比例导引增益 `Ky=Kz=3`；奖励为 `10*r_height + 10*r_speed + 15*r_angle + 10*r_distance + 30*r_dodge + r_event`，击杀 `+200`、被击杀/坠毁 `-200`。
- `engineering safety termination`：当前飞机在高度低于 1000 m 或速度低于 120 m/s 时结束回合。这不是论文的 750 m 飞行包线下限。
- `paper_unspecified_project_assumption`：导弹初速 500 m/s、动力段 3 s、加速度 110 m/s²、命中半径 60 m、有效阻力 `0.00012/m`、寿命 56 s。工程锁定角为 10°，用于避免随机发射稳定获胜；这些都不是论文参数。
- `paper_unspecified_project_assumption`：low/medium/high 初始扰动分别复用 `tam_paper_env_v1_2v2.yaml` 的三档有界均匀扰动；论文未给出这些具体幅度。

名义场景 `paper_nominal_1v1` 取官方 TAM 2v2 配置的 `red_0` 与 `blue_0`：双方均为 F-16 攻击无人机、6000 m、250 m/s，位置 `(120.0, 60.0)` 与 `(120.0, 60.2)`，航向 0° 与 180°，各带两枚相同导弹。

## 正式 1v1 接口

动作是 Gymnasium Dict：`maneuver: float32[3]`（目标俯仰、相对航向、目标速度）与 `fire: 0/1`。策略使用 SquashedNormal 连续头和 Bernoulli 发射头，联合 log-probability/entropy 为两头之和。

观测为 26 维有限 `float32`：18 维 Eq.13 风格飞机/相对态势，加 8 维弹药量、冷却、来袭告警、距离、接近速度、LOS 角和剩余飞行时间。距离与相对位置按 14 km 归一化。正式胜负只由导弹命中、飞机坠毁、同时死亡、超时或数值异常决定，不使用旧几何驻留判杀。

蓝方采用 `paper_greedy` 有限候选机动与工程规则发射。规则要求目标存活、尚有弹药、冷却完成、距离不超过 14 km、锁定角内且同一目标没有在途友方导弹。

## 验证与实验

```powershell
python -m pytest aircombat_env_v1/tests -q
python -m pytest aircombat_env_v1/tests -q -m integration
python aircombat_env_v1/scripts/train_recurrent_ppo_1v1.py --total-steps 10000 --num-envs 4 --rollout-steps 128 --sequence-length 32 --eval-interval 5000 --seed 1
python aircombat_env_v1/scripts/train_recurrent_ppo_1v1.py --total-steps 200000 --num-envs 8 --rollout-steps 256 --sequence-length 32 --eval-interval 10000 --seed 1
python aircombat_env_v1/scripts/evaluate_recurrent_ppo_1v1.py --checkpoint path/to/best_nominal.pt --level low --output low.json
python aircombat_env_v1/scripts/evaluate_recurrent_ppo_1v1.py --checkpoint path/to/best_nominal.pt --level medium --output medium.json
python aircombat_env_v1/scripts/evaluate_recurrent_ppo_1v1.py --checkpoint path/to/best_nominal.pt --level high --output high.json
```

循环 PPO 的 Actor/Critic 都是 `Linear(128)-Tanh-GRU(128, 1 layer)`。rollout 保存两套隐藏状态与 episode-start mask，按长度 32 的连续序列更新，不把时间步随机展平。最佳名义模型只按固定名义种子集的导弹击杀率、被击杀率、命中时间和无效回合排序；扰动结果只作零样本评估。
