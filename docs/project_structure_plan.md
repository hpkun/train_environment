# Project structure plan

本文档只规划项目结构整理，不移动文件、不修改 import、不改变任何代码行为。

## 1. Current layout

当前主要文件职责：

- `my_uav_env/`: JSBSim environment, simulator, reward computation, observation construction.
- `train_vanilla_mappo.py`: vanilla MAPPO training entrypoint.
- `train_attention_mappo.py`: MAPPO-Attention training entrypoint.
- `evaluate_vanilla_mappo.py`: vanilla MAPPO batch evaluation without Tacview.
- `evaluate_attention_mappo.py`: MAPPO-Attention batch evaluation without Tacview.
- `eval_acmi.py`: Tacview / ACMI single-episode evaluation.
- `reward_utils.py`: reward pure functions, reward version, reward audit helpers.
- `entity_obs_utils.py`: current 11-dim entity tensor builder.
- `paper_obs_utils.py`: temporary 10-dim observation adapter from current env obs.
- `paper_state_extractor.py`: strict 10-dim observation extractor prototype from simulator/native state.
- `attention_models.py`: attention actor/critic network modules.
- `scripts/`: smoke tests and pure utility checks.
- `docs/`: experiment guide, paper alignment notes, reward audits, and planning documents.

## 2. Naming policy

后续文件名不要以 `paper_` 开头。命名应表达程序功能，而不是论文来源。论文对齐是项目阶段，文件名应描述长期职责。

推荐重命名：

- `reward_utils.py` -> `my_uav_env/alignment/reward_utils.py`

  理由：reward 函数和 reward version 与环境行为强相关。

- `entity_obs_utils.py` -> `my_uav_env/alignment/entity_obs.py`

  理由：该模块负责构造 entity-wise observation tensor。

- `paper_obs_utils.py` -> `my_uav_env/alignment/obs_adapter.py`

  理由：它是从当前 env obs 到 10-dim observation interface 的 adapter，不应叫 `paper_obs`。

- `paper_state_extractor.py` -> `my_uav_env/alignment/state_extractor.py`

  理由：它从 simulator/native state 中提取 strict 10-dim observation，不应叫 `paper_state`。

- `attention_models.py` -> `models/attention_models.py` or keep at root.

  推荐保留根目录或后续放入 `models/`，不要放入 `my_uav_env`，因为它属于算法网络，不属于环境包。

## 3. Desired long-term layout

建议目标结构：

```text
my_uav_env/
  __init__.py
  env.py
  simulator.py
  ...
  alignment/
    __init__.py
    reward_utils.py
    entity_obs.py
    obs_adapter.py
    state_extractor.py

models/
  attention_models.py   # optional; can also remain at root

scripts/
  smoke_*.py

docs/
  experiment_guide.md
  paper_env_reward_audit.md
  reward_formula_alignment_plan.md
  project_structure_plan.md
```

## 4. Migration strategy

不要一次性破坏 imports。建议分三步迁移。

### Step A: Add new package without breaking old imports

- 创建 `my_uav_env/alignment/` 子包。
- 将以下根目录工具文件内容迁移到新文件：
  - `reward_utils.py` -> `my_uav_env/alignment/reward_utils.py`
  - `entity_obs_utils.py` -> `my_uav_env/alignment/entity_obs.py`
  - `paper_obs_utils.py` -> `my_uav_env/alignment/obs_adapter.py`
  - `paper_state_extractor.py` -> `my_uav_env/alignment/state_extractor.py`
- 根目录旧文件暂时保留为兼容 re-export，例如：

```python
from my_uav_env.alignment.reward_utils import *
```

### Step B: Update imports gradually

- 逐步把 `train_*.py`、`evaluate_*.py`、`scripts/*.py` 的 imports 改成 `my_uav_env.alignment.*`。
- 每次只改一小组文件。
- 每次迁移后运行 `compileall` 与对应纯工具 smoke test。

### Step C: Remove compatibility files later

- 确认所有 imports 已迁移后，再删除根目录旧兼容文件。
- 删除前先用 `rg` / search 确认没有旧 import。
- 删除兼容文件应作为单独 pass，避免和行为修改混在一起。

## 5. What should stay outside my_uav_env

- `train_*.py` 和 `evaluate_*.py` 是实验入口，不放入环境包。
- `eval_acmi.py` 是可视化入口，不放入环境包。
- `attention_models.py` 属于算法网络，不建议放入 `my_uav_env`。可以保留根目录，或者后续放入 `models/`。
- `docs/` 和 `scripts/` 保持当前职责。

## 6. No behavior change statement

本 pass 不移动文件，不修改 import，不改变代码行为，只生成整理计划。

## 7. Recommended next pass

下一步建议：Project structure migration pass A。

范围：

- 创建 `my_uav_env/alignment/`。
- 迁移 `reward_utils.py` 为 `my_uav_env/alignment/reward_utils.py`。
- 根目录 `reward_utils.py` 保留 re-export。
- 只更新最少量 import，或暂时不更新 import。
- 运行 compileall。

建议检查命令：

```powershell
conda activate brmamappo
python -m compileall reward_utils.py entity_obs_utils.py paper_obs_utils.py paper_state_extractor.py attention_models.py
```

## 8. Migration notes

### Pass A — reward_utils (completed)

- `reward_utils.py` implementation moved to `my_uav_env/alignment/reward_utils.py`.
- Root `reward_utils.py` is now a thin compatibility re-export:
  `from my_uav_env.alignment.reward_utils import *`.
- No behaviour changed; no imports in training scripts updated.
- `my_uav_env/alignment/__init__.py` created as the sub-package entry point.
- Next pass should move `entity_obs_utils.py` → `my_uav_env/alignment/entity_obs.py`
  following the same pattern.

### Pass B — entity_obs_utils (completed)

- `entity_obs_utils.py` implementation moved to `my_uav_env/alignment/entity_obs.py`.
- Root `entity_obs_utils.py` is now a thin compatibility re-export:
  `from my_uav_env.alignment.entity_obs import *`.
- No behaviour changed; no imports in training scripts updated.
- Next pass should move `paper_obs_utils.py` → `my_uav_env/alignment/obs_adapter.py`
  following the same pattern.
