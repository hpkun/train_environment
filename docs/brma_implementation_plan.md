# BRMA implementation plan

## 1. Current baseline status

当前项目已经具备后续 BRMA-MAPPO 所需的部分基础模块：

- `attention_models.py`：包含 `EntityObservationEncoder`、`AttentionActor`、`AttentionCritic`。
- `train_attention_mappo.py`：支持 attention actor 训练入口。
- `--obs-adapter strict`：actor 可使用 strict 10 维 Table 1/Table 2 prototype observation。
- `--critic-state strict-global`：critic 可使用 strict-global candidate state。
- strict observation / global state normalization 已接入 attention training。
- `attention_2v2_strict_critic_smoke` 已用于本地短链路验证。

这些仍是 MAPPO-Attention / strict observation baseline，不是 BRMA-MAPPO。

## 2. PDF formula audit

本 pass 对本地论文 PDF 做了有限文本抽取审计。可确认的信息包括：

- 论文 §3.1 使用实体化 observation，顺序为 self、friendly entities、enemy entities。
- 友机或敌机死亡后，对应 observation 置为 0，作为 death masking。
- local / relative observation 维度为 10。
- entity MLP 先将每个 entity 投影到 hidden size，然后做 entity attention。
- actor 使用第一个 entity，也就是 self entity 的 attention output。
- 论文 §3.2 指出训练规模和推理规模不同时，attention weight matrix 尺寸变化会导致 out-of-domain 问题。
- 论文 §3.2 提到按实体类型删除 `max(0, N_inf - N_train)` 个实体，以把推理 observation 缩回训练规模。
- 论文讨论了 random mask，并说明纯随机 masking 可能丢失空战中的重要信息，因此提出 biased random masked attention。

未能可靠抽取或仍需原文核对的内容：

- MaskVectorGenerator 的精确输入 / 输出张量定义。
- biased random mask 的具体 bias score 或排序规则。
- mask probability 是否固定、随规模变化，或由实体类型 / 距离 / 威胁度决定。
- attention mask 是以 PyTorch `key_padding_mask` 语义作用，还是 additive attention mask 语义作用。
- self / ally / enemy / dead entity 在 biased mask 中是否有特殊优先级。
- padding 与 entity ordering 是否还有训练代码层面的额外约束。

上述项目均标记为 **NEEDS PAPER TEXT VERIFICATION**。在确认原文公式前，不应伪造 BRMA mask 规则。

## 3. This pass

本 pass 新增：

- `brma/__init__.py`
- `brma/mask_generator.py`
- `scripts/smoke_mask_generator.py`

`MaskVectorGenerator` 当前只提供：

- 0-valid / 1-invalid 的 entity mask 转 bool valid mask。
- self / ally / enemy 类型 mask。
- uniform random keep mask infrastructure。
- self / ally / enemy 的工程 keep 开关。
- invalid / padded entity 强制 mask。
- 至少保留一个 valid enemy 的工程保护选项。
- keep mask 到 `nn.MultiheadAttention` key padding mask 的转换。

这些是 infrastructure / candidate controls，不代表论文中的 biased random mask 已实现。

## 4. Not implemented yet

本 pass 没有接入：

- `AttentionActor.forward()`
- `EntityObservationEncoder.forward()`
- `train_attention_mappo.py`
- PPO rollout / update
- MaskVectorGenerator 到 actor attention 的训练链路
- biased random mask / mask vector generator 的论文精确公式

`generate_biased_random_mask()` 当前故意抛出 `NotImplementedError`，直到论文原文公式完成核对。

## 5. Recommended next steps

1. 继续核对论文 §3.2 和相关公式，确认 biased mask 的 bias rule、mask probability 与 mask application 语义。
2. 单独修改 `EntityObservationEncoder`，让其接受外部 key padding mask 或 additive attention mask，但保持默认行为不变。
3. 增加纯 PyTorch attention mask smoke test，验证 mask 后 attention 输出 finite 且 shape 不变。
4. 再修改 `train_attention_mappo.py`，在独立 CLI flag 下接入 MaskVectorGenerator。
5. 最后新增 BRMA-MAPPO preset 和正式训练日志字段，避免与 MAPPO-Attention baseline 混淆。

## 6. No training behavior change

本 pass 不修改环境、reward、observation space、AttentionActor forward、PPO 训练流程或评估脚本。新增模块目前不影响任何训练行为。
