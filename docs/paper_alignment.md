# Paper alignment notes

## 本轮已对齐
- 雷达 Rmax 改为 RCS 四次方根关系。
- 导弹发射间隔改为 0.5 s。
- 导弹命中概率改为 missile velocity 与 LOS 的方向匹配公式。
- roll reward 加入 pitch 条件。
- altitude reward 改为二次分段近似。
- terminal reward 保持 per-agent share，并解释原因。

## 本轮没有修改
- 训练规模仍为当前 train_vanilla_mappo.py 中的 2v2。
- 观测仍为当前 11 维工程化 entity vector，没有改成论文 Table 1/Table 2 的 10 维形式。
- 算法仍为 vanilla MAPPO，没有实现 EntityObservationEncoder / MaskVectorGenerator / BRMA-MAPPO。
- 蓝方规则策略暂时不改。
- Tacview 评估工具暂时不改。

## 已知仍与论文不同
- 论文使用 RCS 查表，当前使用简化 RCS approximation。
- 论文训练主场景为 6v6，当前仍是 2v2 baseline。
- 论文 critic 使用 global state，当前 vanilla critic 使用 red agents flattened observations concat。
- 当前 PID 控制器包含额外工程稳定项。
- 当前导弹数量默认 999，论文没有明确无限弹药。
- 当前 observation 表达与论文 Table 1/Table 2 不完全一致。
