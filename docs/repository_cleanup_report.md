# Repository Artifact Cleanup Report

Date: 2026-06-01

## 1. Virtual Environment Cleanup

Deleted `venv_aircombat/`.

Reason: `venv_aircombat/` was a local Python virtual environment, not project source. It contained environment artifacts such as `bin/python`, `bin/pip`, activation scripts, package console entry points, and `pyvenv.cfg`; these files should not be versioned with the repository.

Note: `git rm -r venv_aircombat` could not be completed in this workspace because Git could not create `.git/index.lock` due to permission denial. The tracked virtualenv files were removed from the working tree instead.

## 2. Logs, Results, and Checkpoints Cleanup

Protected current training output:

- `logs/attention_2v2_brma_paper_500k_probe_env6_mb1024.csv`
- `results/attention_2v2_brma_paper_500k_probe_env6_mb1024_results.csv` (protected path; file was not present at cleanup time)
- `checkpoints/attention_2v2_brma_paper_500k_probe_env6_mb1024/`

Deleted historical `logs/*.csv` except the protected current training log. Removed historical attention, BRMA smoke/probe, benchmark, probe, and vanilla log CSV files.

Deleted historical `results/*` except the protected current training result path. Removed historical result CSV files and launch-quality CSV files, including smoke, probe, benchmark, attention, BRMA, and vanilla outputs.

Deleted historical checkpoint files and checkpoint directories under `checkpoints/` except the protected current training checkpoint directory. Removed old top-level `.pt` files and obsolete experiment checkpoint directories, including old smoke, probe, benchmark, attention, BRMA, and vanilla checkpoints.

Post-cleanup state:

- `logs/` contains only `attention_2v2_brma_paper_500k_probe_env6_mb1024.csv`.
- `results/` contains no files at cleanup time.
- `checkpoints/` contains only `attention_2v2_brma_paper_500k_probe_env6_mb1024/`.

## 3. ACMI, Temporary Files, and Cache Cleanup

Deleted cache and temporary artifacts matching:

- `__pycache__/`
- `.pytest_cache/`
- `.mypy_cache/`
- `.ruff_cache/`
- `*.pyc`
- `*.acmi`
- `*.tmp`
- `*.bak`
- `*~`

No source files were deleted as part of this cache cleanup.

## 4. Archived Training Reports

Created `docs/archive/` and moved these older training reports from `docs/training_reports/`:

- `probe_best_reward_50k_launch_quality_summary.md`
- `probe_best_winrate_50k_launch_quality_summary.md`
- `probe_final_50k_launch_quality_summary.md`
- `vanilla_2v2_launch_quality_probe_50k_summary.md`
- `vanilla_2v2_main_entropy_diag_100k_summary.md`
- `vanilla_2v2_main_paper_diag_1m_warmstart_summary.md`
- `vanilla_2v2_main_paper_diag_500k_summary.md`

No `docs/*.md` file was deleted.

## 5. Gitignore Updates

Confirmed or added ignore rules for:

- `__pycache__/`
- `*.pyc`
- `.pytest_cache/`
- `.mypy_cache/`
- `.ruff_cache/`
- `*.tmp`
- `*.bak`
- `*~`
- `venv_aircombat/`
- `.venv_aircombat/`
- `logs/`
- `results/`
- `checkpoints/`
- `*.acmi`
- `runs/`

## 6. Code Safety Boundary

This cleanup did not modify training logic, environment dynamics, reward logic, blue policy, missile/radar/launch logic, or BRMA loss code.

No changes were made to:

- `train_attention_mappo.py`
- `train_vanilla_mappo.py`
- `attention_models.py`
- `brma/`
- `my_uav_env/`
- `rule_based_agent.py`
- `configs/`
- `scripts/`
- `evaluate_*.py`
- `eval_acmi.py`
