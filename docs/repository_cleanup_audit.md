# Repository conservative cleanup audit

## 1. KEEP_CORE

Do not delete or modify these during the current
`attention_2v2_brma_paper_500k_probe_env6_mb1024` run:

- `train_attention_mappo.py`
- `attention_models.py`
- `brma/*`
- `my_uav_env/*`
- `rule_based_agent.py`
- `train_vanilla_mappo.py`
- `evaluate_vanilla_mappo.py`
- `evaluate_attention_mappo.py`
- `eval_acmi.py`
- `configs/experiment_presets.py`
- Paper reproduction presets and speed benchmark presets.
- Current running outputs:
  - `logs/attention_2v2_brma_paper_500k_probe_env6_mb1024.csv`
  - `results/attention_2v2_brma_paper_500k_probe_env6_mb1024_results.csv`
  - `checkpoints/attention_2v2_brma_paper_500k_probe_env6_mb1024/`
- Any file under `logs/`, `results/`, or `checkpoints/` that may be written by
  active training.

## 2. KEEP_SMOKE_TESTS_FOR_NOW

Keep these smoke tests because they cover the current BRMA reproduction path:

- `scripts/smoke_experiment_presets.py`: verifies paper and smoke presets are
  discoverable.
- `scripts/smoke_brma_train_mode_static.py`: verifies default-off BRMA train
  mode integration without env startup.
- `scripts/smoke_brma_train_step_static.py`: verifies KL-only mask-generator
  gradients through the selected soft mask path.
- `scripts/smoke_brma_losses_static.py`: verifies Gaussian KL and BRMA loss
  helper behavior.
- `scripts/smoke_brma_collection_soft_path.py`: verifies selected-set soft mask
  collection semantics and Gaussian parameter storage.
- `scripts/smoke_brma_soft_mask_actor_api.py`: verifies differentiable actor
  soft-mask API.
- `scripts/smoke_attention_eq33_encoder_static.py`: verifies Eq.33 encoder
  shape behavior.
- `scripts/smoke_attention_critic_entities_static.py`: verifies
  attention-entities critic behavior.

## 3. CANDIDATE_ARCHIVE_AFTER_500K

These can be reviewed after the running 500k probe completes, but are not
deleted in this pass:

- Early vanilla launch-quality diagnostic scripts and generated reports.
- Old battlefield boundary debug smoke scripts.
- Old dry-run transition tests that are superseded by train-mode static tests.
- Outdated training reports that refer only to earlier vanilla warm-start or
  launch-quality probes.
- Legacy adapter or reward smoke tests whose coverage is duplicated by newer
  strict observation / BRMA tests.

## 4. SAFE_DELETE_NOW

Only low-risk generated files are in scope for deletion:

- `__pycache__/` directories.
- `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/` directories if present.
- `*.pyc` bytecode files.
- Clearly temporary editor backup files: `*.bak`, `*.tmp`, `*~`.
- Temporary empty files only if they are clearly not under active
  `logs/`, `results/`, or `checkpoints/` outputs.

Current scan found Python cache directories and `.pyc` files. No `.bak`,
`.tmp`, or `*~` files were found.

## 5. DO_NOT_DELETE

Do not delete:

- Any source code file.
- Any `scripts/smoke_*.py` file.
- Any `docs/*.md` file.
- Any `configs/*.py` file.
- Any `logs/*.csv` file.
- Any `results/*.csv` file.
- Any `checkpoints/*` content.
- Current training output paths listed in KEEP_CORE.
- Core training, environment, BRMA, reward, blue policy, missile, radar, launch,
  evaluation, or paper preset files.
