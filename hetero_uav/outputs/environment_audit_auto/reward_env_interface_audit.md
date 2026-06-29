# Reward / Environment Interface Audit
- `_step_kill_count` is step-local kill accounting used by reward overlays.
- `missiles_fired_this_step` in info is step-local and reset after info generation.
- `__missile_term__` is accumulated termination counters by side.
- `reward_components` are merged into per-agent info for diagnostics.
- Rich logs are sufficient only when enabled; missing reward/missile/aircraft timeseries prevents post-hoc causality.
- `brma_paper_homogeneous_v1` is a diagnostic homogeneous baseline, not a claim of full original-paper reproduction.
- 3v2 `30*(Nred-Nblue)` terminal can encode initial team-size bias unless applied only at episode end and interpreted carefully.