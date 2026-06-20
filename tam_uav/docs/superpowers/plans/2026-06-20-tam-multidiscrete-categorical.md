# TAM MultiDiscrete Categorical Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the formal TAM continuous Gaussian approximation with a four-axis, forty-bin MultiDiscrete categorical actor whose sampled indices are the exact actions executed by JSBSim.

**Architecture:** Formal configs select `multidiscrete_categorical`; the environment maps validated integer indices directly to FCS commands. A new entity-attention/GRU categorical policy uses role-specific logits and a centralized slot-attention critic, while legacy continuous behavior remains available only behind `continuous_quantized`.

**Tech Stack:** Python, NumPy, Gymnasium, PyTorch categorical distributions and multi-head attention, pytest, JSBSim

---

### Task 1: Environment and blue action contracts

**Files:** `tests/test_tam_multidiscrete_env_contract.py`, `tests/test_tam_categorical_blue_opponent.py`, formal YAML configs, `uav_env/JSBSim/env.py`, `uav_env/JSBSim/envs/hetero_uav_combat_env.py`, `algorithms/mappo/opponent_policy.py`.

- [ ] Write failing tests for `MultiDiscrete([40]*4)`, exact endpoint/mid-bin mapping, invalid-index rejection, legacy Box compatibility, and blue int64 indices.
- [ ] Run the focused tests and confirm failures are caused by the absent categorical contract.
- [ ] Implement distribution config validation, separate discrete/continuous mappers, trim handling, and command-to-index conversion without changing tactical target selection.
- [ ] Re-run focused tests until green.

### Task 2: Categorical recurrent policy and attention critic

**Files:** `tests/test_tam_categorical_recurrent_policy.py`, `algorithms/happo/tam_categorical_recurrent_policy.py`, `algorithms/happo/__init__.py`.

- [ ] Write failing tests for long actions, `[B,4,40]` logits, stochastic bounds, deterministic argmax, summed log-prob/entropy, recurrent state, attention critic, save/load, and absence of Gaussian parameters.
- [ ] Implement the policy by reusing the existing BRMA entity decoder/encoder and GRU, adding role-specific categorical heads and a five-slot masked attention value network.
- [ ] Verify `act` and `evaluate_actions` agree on the same integer action indices.

### Task 3: Buffer and categorical PPO

**Files:** `tests/test_tam_categorical_ppo_contract.py`, `algorithms/happo/happo_buffer.py`, `algorithms/happo/happo_trainer.py`.

- [ ] Write failing tests for int64 storage/torch.long retrieval, categorical ratio consistency, trainer operation without log-std, and bin-probability metrics.
- [ ] Add `action_dtype`, route actor parameter groups without Gaussian parameters, preserve categorical actions through updates, and emit categorical-only diagnostics.
- [ ] Run buffer/trainer tests and existing continuous regression tests.

### Task 4: Training, evaluation, checkpoints, and logging

**Files:** `tests/test_tam_multidiscrete_train_smoke.py`, `scripts/train_tam_happo_direct.py`, `scripts/eval_tam_happo_direct.py`, `scripts/rich_logging.py`, logging schema.

- [ ] Write failing integration/static tests for config routing, int64 buffer construction, categorical checkpoint metadata, legacy-checkpoint rejection, deterministic argmax eval, and latest checkpoint reload.
- [ ] Route `brma_recurrent_masked` to the categorical policy under formal configs; keep the legacy continuous classes only for diagnostic configs.
- [ ] Record indices, normalized levels, mapped commands, distribution, action-space type, and categorical metrics.

### Task 5: Contract audit and regression suite

**Files:** `scripts/audit_tam_multidiscrete_contract.py` and the five requested tests.

- [ ] Implement runtime audit checks for configs, spaces, policy, buffer dtype, categorical log-prob agreement, exact env execution indices, blue indices, and legacy isolation.
- [ ] Generate JSON and Markdown audit outputs and run the specified pytest suite including airborne reset stabilization.

### Task 6: Runtime smoke gates

**Files:** generated `outputs/tam_happo_multidiscrete_f22_2k_smoke` and, only after success, `outputs/tam_happo_multidiscrete_f22_50k_smoke`.

- [ ] Run 2k with formal 3v2 training plus 3v2/5v4 evaluation; verify model/meta/rich logs and report the first eight rows.
- [ ] Only if 2k passes, run 50k and summarize runner status, finite-state status, weapon chain, MAV survival, blue alive, and both eval scales.
- [ ] Verify no changes outside `tam_uav`, no forbidden XML/reward/missile changes, and no staged files or commits.
