# TAM Recurrent Categorical HAPPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement neutral categorical initialization and time-ordered recurrent role-level HAPPO for the formal TAM path.

**Architecture:** Extend the categorical policy with a sequence evaluator, extend the rollout buffer with sequence boundaries, and route formal training to a dedicated trainer. Keep legacy continuous training isolated in the reference trainer.

**Tech Stack:** Python, PyTorch, NumPy, Gymnasium, pytest, JSBSim.

---

### Task 1: Neutral categorical prior

**Files:** policy and `tests/test_tam_categorical_neutral_init.py`.

- [ ] Write failing prior/entropy/exploration tests.
- [ ] Implement configurable Gaussian output biases for both role heads.
- [ ] Run focused tests.

### Task 2: Sequence rollout contract

**Files:** buffer, policy, and `tests/test_tam_categorical_sequence_replay.py`.

- [ ] Write failing buffer field and GRU reset tests.
- [ ] Store episode starts, environment step indices, alive masks, and initial hidden state.
- [ ] Implement time-ordered policy sequence evaluation.
- [ ] Run focused tests.

### Task 3: Dedicated categorical HAPPO trainer

**Files:** create `algorithms/happo/tam_categorical_happo_trainer.py`, export it, and add trainer tests.

- [ ] Write failing trainer/correction/finite-parameter tests.
- [ ] Implement grouped recurrent evaluation, critic update, role order, and detached correction.
- [ ] Add required categorical/gradient/correction metrics.
- [ ] Run focused tests.

### Task 4: Train/checkpoint integration

**Files:** training script and smoke tests.

- [ ] Write failing route/metadata/log tests.
- [ ] Route categorical configs to the new trainer and sequence buffer fields.
- [ ] Record trainer, recurrent update, correction, and neutral-prior metadata.
- [ ] Run focused tests.

### Task 5: Initial-policy flight validation

**Files:** create validation script, diagnostics compatibility updates, and flight tests.

- [ ] Write failing validation/flight tests.
- [ ] Implement fixed-level non-firing blue rollout and JSON/Markdown outputs.
- [ ] Update categorical checkpoint/telemetry diagnostics where needed.
- [ ] Run validation and focused tests.

### Task 6: Runtime gates

- [ ] Run the full specified pytest set.
- [ ] Run initial-policy flight validation.
- [ ] Run 2k CUDA smoke; inspect runner, sequence/correction metrics, and first 8 rows.
- [ ] If 2k passes, run 50k CUDA smoke and collect required metrics.
- [ ] Verify no staged files and no changes outside `tam_uav`.
