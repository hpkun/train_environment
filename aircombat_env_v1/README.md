# aircombat_env_v1

## 论文事实边界

| 分类 | 本目录中的内容 |
|---|---|
| `paper_explicit` | JSBSim 60 Hz；决策5 Hz；每次动作保持12个物理步；每回合1000个决策步；蓝方使用有限候选机动并在每个决策步按即时奖励贪心选择；UAV奖励结构为height、speed、angle、distance、dodge、event，训练权重为10、10、15、10、30；最低高度250 m、最大速度400 m/s、最大过载9 g；随机初始化采用各状态分量独立的有界均匀扰动。 |
| `paper_unspecified_engineering` | 8个候选机动的具体3维动作值、0.2秒高层动作前视、当前PID目标接口、尾追初始条件、几何驻留命中、随机扰动幅度、观测归一化尺度、高度奖励近似、安全候选惩罚以及1000步dense奖励归一化。TAM仓库的speed/angle/distance函数和贪心候选结构被复制适配；其height函数本身明确是论文未定义近似。 |
| `temporary_learnability_abstraction` | 3维目标俯仰/相对航向/目标速度动作；用几何攻击区代替真实导弹；单智能体PPO；1v1尾追课程。 |

`paper_greedy`来源结构为
`tam_uav/uav_env/JSBSim/paper/opponent.py`，候选动作通过本目录现有
`action_to_targets`接口执行。本目录不把单智能体MLP PPO称为TAM-HAPPO复现。

红方奖励保持论文权重比例：

`10*r_height + 10*r_speed + 15*r_angle + 10*r_distance + r_event`，

当前没有导弹，因此`r_dodge=0`。事件奖励为命中`+200`、被命中或红方坠毁
`-200`、draw/timeout/blue_crash/数值异常为0。`blue_crash`是
`opponent_failure`，不是红方胜利，也不能参与best模型排序。几何命中只是
`temporary_learnability_abstraction`。

20维观测的0至14维直接对应或等价表示论文Eq.13中的自身位置、速度、姿态、
相对速度、相对高度、距离、ATA和AA；15至19维是当前1v1所需的相对NEU和双方
攻击驻留。位置、相对量等具体归一化尺度属于工程值。

A standalone, repeatable JSBSim F-16 trim and PID experiment. The controller
structure follows BRMA-MAPPO Section 2.4, but its numerical gains, trim biases,
actuator signs, and acceptance thresholds are local engineering values—not
paper parameters.

Dependencies are `numpy`, `PyYAML`, `pymap3d`, `jsbsim`, `scipy`, and `pytest`.
All commands below run from the repository root.

## Required workflow

```powershell
python aircombat_env_v1/scripts/check_actuator_signs.py
python aircombat_env_v1/scripts/find_trim.py
python aircombat_env_v1/scripts/find_trim.py --accept
python aircombat_env_v1/scripts/tune_pid.py
python aircombat_env_v1/scripts/tune_pid.py --accept-candidate
python aircombat_env_v1/scripts/tune_pid.py --pitch-integral-only --config path/to/candidate_config.yaml --joint-duration 90
python aircombat_env_v1/scripts/validate_pid.py --mode quick
python aircombat_env_v1/scripts/validate_pid.py --mode full
python aircombat_env_v1/scripts/validate_pid.py --mode full --mark-validated
```

Inspect each generated output before using the corresponding accept flag.
Parameter status progresses from `initial_guess` to `candidate`, then to
`validated`. Quick mode runs eight representative 200-second health cases.
Full mode runs the 72-case matrix plus those eight long cases.
`--mark-validated` is rejected in quick mode and refuses to update the formal
configuration unless full validation passes.

Every closed-loop script uses the same 60 Hz physics/PID loop and 5 Hz command
interface: one high-level command is held for 12 physics frames. In the
combined task the interface is still called at 5 Hz, while deterministic target
values change only every 3, 4, or 5 seconds.

## Tests and smoke runs

```powershell
python -m pytest aircombat_env_v1/tests -q
python -m pytest aircombat_env_v1/tests -q -m integration
python aircombat_env_v1/scripts/check_actuator_signs.py --stabilization-duration 3
python aircombat_env_v1/scripts/find_trim.py --duration 5 --grid-size 5
python aircombat_env_v1/scripts/tune_pid.py --roll-duration 1 --pitch-duration 1 --speed-duration 1 --joint-duration 1 --maxiter 1 --popsize 2
python aircombat_env_v1/scripts/validate_pid.py --mode quick --quick-duration 1
```

Smoke outputs prove workflow execution only; they are not validated flight
parameters. Timestamped artifacts are written under `aircombat_env_v1/outputs`
and never overwrite prior runs. See `PID_DESIGN.md` for formulas, status rules,
and metric definitions.

## Minimal JSBSim 1v1 environment

`AirCombat1v1Env` is a single-agent Gymnasium environment: external actions
control red and an internal `straight` or pure-`pursuit` rule controls blue.
The action is a three-vector for target pitch, relative heading, and speed;
the observation is a clipped 16-vector. One environment step holds both
commands for 12 interleaved 60 Hz JSBSim/PID frames.

The fixed `tail_chase` scenario uses a geometric attack zone instead of
missiles. This is intentionally only a learnability check, with no radar,
weapons, rewards beyond the small geometric potential, or multi-agent API.

```powershell
python aircombat_env_v1/scripts/check_1v1_env.py
python aircombat_env_v1/scripts/run_rule_1v1.py --episodes 5 --opponent straight
```
