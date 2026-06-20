# TAM Categorical Semantics and Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct TAM categorical optimizer and architecture semantics, add reproducible environment and trend diagnostics, and validate them through a 200k GPU trend probe.

**Architecture:** Keep the policy and environment contracts intact while separating optimizer ownership and centralizing categorical architecture identity. Add standalone read-only audit scripts and a pure CSV trend summarizer so diagnostics do not affect training behavior.

**Tech Stack:** Python 3.9, PyTorch 2.8 CUDA 12.8, pytest, NumPy, JSBSim, CSV/JSON/Markdown.

---

### Task 1: Separate categorical actor optimizer ownership

**Files:**
- Modify: `algorithms/happo/tam_categorical_happo_trainer.py`
- Modify: `tests/test_tam_categorical_happo_trainer.py`

- [ ] Add tests that compare optimizer parameter identities against `actor_shared_parameters`, `mav_actor.parameters`, and `uav_actor.parameters`, assert all sets are disjoint, and assert update metrics contain finite `grad_norm_shared`, `grad_norm_mav_head`, `grad_norm_uav_head`, `grad_norm_actor`, and `grad_norm_critic`.
- [ ] Run `D:\conda_envs\envs_dirs\brmamappo\python.exe -m pytest tests/test_tam_categorical_happo_trainer.py -q` and confirm the ownership assertions fail against the current dual-owned shared parameters.
- [ ] Create `shared_actor_opt` only when shared parameters exist; restrict role optimizers to their heads; update `_update_role` to zero, clip, and step the shared and selected role optimizers in the required order.
- [ ] Run the trainer tests and confirm they pass with finite parameters and correction metrics.

### Task 2: Make categorical policy architecture identity traceable

**Files:**
- Modify: `scripts/train_tam_happo_direct.py`
- Modify: `scripts/eval_tam_happo_direct.py`
- Create: `tests/test_tam_policy_arch_metadata.py`

- [ ] Add failing tests for explicit `tam_categorical_recurrent`, the compatible `brma_recurrent_masked` alias, complete requested/effective metadata, unchanged continuous routing, and rejection of continuous or conflicting categorical checkpoint metadata.
- [ ] Run `D:\conda_envs\envs_dirs\brmamappo\python.exe -m pytest tests/test_tam_policy_arch_metadata.py -q` and verify the missing choice and metadata failures.
- [ ] Add a categorical route resolver returning requested name, effective name, and alias flag; use the effective name for construction and checkpoint validation while preserving the requested name in metadata and warning once for aliases.
- [ ] Update evaluation to recognize `effective_policy_arch=tam_categorical_recurrent` and validate categorical checkpoint contracts before loading weights.
- [ ] Run metadata and existing categorical smoke tests until green.

### Task 3: Add airborne initialization audit

**Files:**
- Create: `scripts/validate_tam_airborne_initialization.py`
- Modify: `tests/test_tam_airborne_initial_state_stabilization.py`

- [ ] Add failing tests for reset-contract comparison, explicit death classification, and finite fixed-flight summary helpers.
- [ ] Implement reset FCS snapshots, target/actual speed-altitude-yaw checks, a 120-second fixed-neutral flight trace, and three formal episodes without mutating initialization logic.
- [ ] Emit `outputs/environment_audit/tam_airborne_initialization.json` and `.md`, including the 60-second F22 speed criterion and explicit death reasons.
- [ ] Run the focused tests and audit command and inspect both artifacts.

### Task 4: Add missile threat audit

**Files:**
- Create: `scripts/validate_tam_missile_threat.py`
- Create: `tests/test_tam_missile_threat_diagnostics.py`

- [ ] Add failing pure-summary tests covering launch/hit rates, 5/10/20-second warning windows, geometry bins, post-launch survival, launch opportunities, and zero-hit termination reasons.
- [ ] Implement deterministic, stochastic, no-blue-missile, and formal missile-enabled collection with ten episodes per scenario and no environment mutation beyond the existing debug missile switch.
- [ ] Emit `outputs/environment_audit/tam_missile_threat.json` and `.md` and run the focused tests and audit command.

### Task 5: Add training trend analysis

**Files:**
- Create: `scripts/analyze_tam_training_trend.py`
- Create: `tests/test_tam_training_trend.py`

- [ ] Add failing tests for staged return values, rolling slope, finite metric aggregation, collapse detection, baseline comparison, and deterministic A/B/C/D classification.
- [ ] Implement streaming CSV readers for `train_log.csv`, `eval_log.csv`, and optional rich logs; summarize return, wins, MAV survival/death time, launches/hits, blue alive, entropy, action usage, correction, and KL.
- [ ] Emit `trend_summary.json` and `.md` under the selected run directory and preserve the supplied paper-reference note.
- [ ] Run focused trend tests with synthetic logs.

### Task 6: Correct interface documentation only

**Files:**
- Modify: `uav_env/JSBSim/env.py`
- Modify: `README.md`
- Modify: relevant files under `docs/` found by exact text search

- [ ] Search for `Box(3)`, `target_pitch`, `target_heading`, and `target_velocity`; identify statements that incorrectly describe the formal TAM route.
- [ ] Update only comments and documentation to distinguish `legacy_pid_3d` from `tam_direct_fcs_4d` and state that formal TAM uses 4D MultiDiscrete categorical actions.
- [ ] Run `git diff --check` and confirm no executable environment behavior changed in this task.

### Task 7: Run requested verification and 2k smoke

**Files:**
- Verify: requested test modules and generated environment audit artifacts

- [ ] Run the exact requested pytest modules plus the new missile and trend tests with the CUDA environment.
- [ ] Run airborne and missile audit scripts and inspect JSON for finite values and explicit outcome classifications.
- [ ] Run the specified 2048-step training command with `--policy-arch tam_categorical_recurrent --device cuda`; if CUDA probing fails at execution time, rerun only with `--device cpu`.
- [ ] Verify runner status is normal, metadata is complete, and no NaN/nonfinite marker exists.

### Task 8: Run 200k trend probe and report

**Files:**
- Generate: `outputs/tam_happo_categorical_semanticsfix_f22_200k_trend/`

- [ ] Run the specified 200000-step trend command on CUDA with heartbeat stall protection and no 1M run.
- [ ] Verify normal runner completion, checkpoint metadata, train/eval logs, and rich logging outputs.
- [ ] Run `scripts/analyze_tam_training_trend.py` with the supplied paper comparison note.
- [ ] Inspect `trend_summary.json` and `.md` and record final return, slope, launch/hit, survival/death, blue-alive, eval, entropy, action usage, correction, KL, and A/B/C/D decision.

### Task 9: Final verification and publish

**Files:**
- Verify all modified `tam_uav` files

- [ ] Run the complete requested pytest command fresh, compile new scripts, run `git diff --check`, and confirm no `hetero_uav` changes.
- [ ] Stage only `tam_uav`, inspect the cached diff, and commit with `Fix TAM categorical trainer semantics and trend diagnostics`.
- [ ] Push `main` to `origin/main`; if security approval rejects it, report the local SHA, ahead count, and required user approval without bypassing the rejection.

