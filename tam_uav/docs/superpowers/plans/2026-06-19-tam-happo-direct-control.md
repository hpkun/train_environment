# TAM-HAPPO 4D Direct-Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the TAM-HAPPO-aligned four-dimensional direct-FCS training and evaluation pipeline in `tam_uav`.

**Architecture:** Add a configuration-selected branch to the existing JSBSim environment so legacy PID configurations remain compatible. Keep quantization and physical command mapping at the environment boundary, while policies and buffers consume the action dimension discovered from the environment.

**Tech Stack:** Python 3, NumPy, Gymnasium, PyTorch, PyYAML, JSBSim, pytest

---

### Task 1: Environment action contract

**Files:**
- Create: `tests/test_tam_direct_action_dim4.py`
- Modify: `uav_env/JSBSim/env.py`
- Modify: `uav_env/JSBSim/envs/hetero_uav_combat_env.py`

- [ ] Add failing unit tests asserting `(4,)`, exact 40-level quantization, throttle mapping bounds, and four direct JSBSim property writes.
- [ ] Run `python -m pytest tests/test_tam_direct_action_dim4.py -q` and confirm failures identify the missing interface.
- [ ] Add `action_interface`, quantization, throttle range, diagnostic command capture, and direct-FCS execution without PID.
- [ ] Gate old three-value action trim and scripted evasion so they cannot alter TAM actions.
- [ ] Re-run the test and retain the smallest implementation that passes.

### Task 2: Formal environment configurations

**Files:**
- Create: `uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml`
- Create: `uav_env/JSBSim/configs/tam_happo_f22_5v4_direct.yaml`
- Create: `tests/test_tam_direct_env_contract.py`

- [ ] Add a failing contract test for F22 MAV/F16 UAV composition, missile counts, simulation cadence, reward/observation modes, and disabled scripted evasion.
- [ ] Run the test and confirm it fails because formal configs are absent.
- [ ] Create the 3v2 and 5v4 configs from existing approved geometry with the TAM action fields.
- [ ] Run the test and confirm both environments satisfy the contract.

### Task 3: Blue direct-control rule

**Files:**
- Modify: `algorithms/mappo/opponent_policy.py`
- Create: `tests/test_tam_direct_blue_rule.py`

- [ ] Add failing tests for four finite clipped values, level cruise without a target, and distinguishable lateral/vertical target responses.
- [ ] Run the focused test and confirm `tam_direct_fsm` is rejected or returns the wrong shape.
- [ ] Preserve target selection and add only the direct-FCS maneuver conversion.
- [ ] Re-run the focused test and all action-contract tests.

### Task 4: Dynamic action dimension in training and evaluation

**Files:**
- Copy and adapt: `scripts/train_tam_happo_direct.py`
- Copy and adapt: `scripts/eval_tam_happo_direct.py`
- Copy: `scripts/rich_logging.py`
- Create: `scripts/run_tam_happo_direct_2k_smoke.py`
- Create: `scripts/run_tam_happo_direct_50k.py`
- Create: `tests/test_tam_direct_training_smoke.py`

- [ ] Add failing static/runtime tests proving action dimension comes from each red action space and reaches policy, buffer, trainer/eval, and checkpoint metadata.
- [ ] Run the focused test and confirm missing scripts/dimension propagation cause failure.
- [ ] Adapt only the necessary HAPPO reference scripts and helper imports; remove every operational default of three from the new entry points.
- [ ] Run a CPU-sized minimal smoke in the test and confirm `latest/model.pt` plus metadata with `action_dim: 4`.

### Task 5: Diagnostics and documentation

**Files:**
- Create: `scripts/audit_tam_direct_control_response.py`
- Create: `scripts/analyze_tam_training_curves.py`
- Modify: `README.md`
- Modify: `setup.py`

- [ ] Add tests or pure-function assertions for audit action definitions and training-summary calculations before implementation.
- [ ] Implement fixed-action telemetry capture and Markdown/JSON output for both F22 and F16.
- [ ] Implement curve summaries for returns, outcomes, role activity, MAV death, missile chain, action saturation, entropy/log-std, and 3v2/5v4 evaluation when present.
- [ ] Rename package metadata to `tam-uav` and document the direct-control contract and branch isolation.

### Task 6: Full verification and requested smoke runs

**Files:**
- Generated: `outputs/environment_audit/tam_direct_control_response.json`
- Generated: `outputs/environment_audit/tam_direct_control_response.md`
- Generated: `outputs/tam_happo_direct_f22_2k_smoke/`
- Generated only after 2k success: `outputs/tam_happo_direct_f22_50k_smoke/`

- [ ] Run `python -m pytest tests/test_tam_direct_action_dim4.py tests/test_tam_direct_env_contract.py tests/test_tam_direct_blue_rule.py -q`.
- [ ] Run the fixed-action audit for 600 steps and inspect all acceptance checks.
- [ ] Run the 2048-step training smoke and verify `latest/model.pt` and action-dimension metadata.
- [ ] If and only if the previous checks pass, run the requested 50k command and summarize self-control and weapon-chain metrics.
- [ ] Confirm `git diff -- hetero_uav` is empty and no aircraft, engine, or missile dynamics XML changed.
- [ ] Commit implementation with `Implement TAM-HAPPO 4D direct-control pipeline in tam_uav`.
