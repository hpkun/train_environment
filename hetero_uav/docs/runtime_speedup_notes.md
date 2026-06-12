# Runtime Speedup Notes

This note records runtime-only changes for the current HAPPO reference v0,
direct-chase oracle pretrain, and checkpoint evaluation workflow. These changes
do not alter reward, missile dynamics, PID, aircraft XML, action space,
observation dimensions, or the HAPPO reference trainer.

## Paper and Current Environment Settings

BRMA-MAPPO motivates entity/mask-style observation and zero-shot scale
evaluation, but the current project does not implement BRMA's biased random
masked attention. TAM-HAPPO motivates heterogeneous MAV/UAV roles, temporal
feature extraction, attention-enhanced value estimation, and 3v2-to-5v4
heterogeneous transfer, but the current HAPPO reference v0 remains a simplified
no-GRU and no-attention reference.

The current main experiment keeps the existing high-level action interface
`[pitch, heading, speed]`, JSBSim/PID dynamics, scripted missile handling, and
fixed-capacity V2 observation. The active protocol remains 3v2 training and
5v4 zero-shot evaluation.

Current relevant runtime settings:

- `sim_freq=60` and `agent_interaction_steps=12`, so the decision interval is
  0.2 seconds.
- Main HAPPO reference configs keep `max_steps=1000`.
- Formal checkpoint evaluation can still use `--episodes 100`.
- Fast checkpoint screening uses `--fast`, which reduces evaluation to 20
  episodes on the 3v2 seen config only.

## Bottlenecks

The slow components are expected to be JSBSim environment stepping, repeated
checkpoint evaluation, repeated oracle dataset regeneration, and full-dataset
behavior cloning when a valid pretrained checkpoint already exists.

## Applied Speedups

- `run_happo_oracle_pretrain_finetune_200k.py` now supports
  `--skip-existing` and `--force`.
- Dataset collection is reused when both `.npz` and summary JSON exist.
- Oracle pretrain is reused when both model and metadata exist and metadata
  marks `pretrained_from_oracle=true`.
- Fine-tuning is skipped when `latest/model.pt` exists and metadata indicates
  enough environment steps were completed.
- The oracle-pretrain runner now defaults `--train-eval-episodes` to 2.
- `collect_direct_chase_oracle_dataset.py` defaults to `--max-samples 100000`
  and stops once the sample cap is reached.
- `pretrain_uav_actor_from_oracle.py` supports validation split, sample cap,
  early stopping, target validation loss, best-checkpoint saving, and device
  reporting.
- `evaluate_happo_3v2_reference_checkpoints.py --fast` provides quick
  checkpoint screening without replacing formal evaluation.
- `run_oracle_pretrain_fast_check.py` provides a small chain-level smoke check.
- `profile_runtime_hotspots.py` records rough component wall times and writes a
  JSON/Markdown profile under `outputs/runtime_profile/`.

## What Did Not Change

- No reward changes.
- No missile launch or missile dynamics changes.
- No PID or aircraft XML changes.
- No action-space changes.
- No observation dimension changes.
- No GRU, attention, full TAM-HAPPO, or BRMA attention implementation.
- No change to the 3v2 train and 5v4 zero-shot protocol.

## Recommended Usage

For repeated development runs:

```powershell
python scripts/run_happo_oracle_pretrain_finetune_200k.py --skip-existing --dry-run
```

For quick checkpoint screening:

```powershell
python scripts/evaluate_happo_3v2_reference_checkpoints.py --experiment-dir outputs/happo_oracle_pretrain_finetune_200k --checkpoint-mode latest_only --fast
```

For a local runtime profile:

```powershell
python scripts/profile_runtime_hotspots.py
```
