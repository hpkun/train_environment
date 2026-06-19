# Paper Experiment Matrix

This matrix is intentionally small. The goal is to decide whether the new
entity-attention actor is worth longer training, not to build a full BRMA or
TAM-HAPPO reproduction.

## A. flat_baseline_long

- `policy_arch=flat`
- Run directory: `outputs/full_10m_normal_geometry_max1000_env1`
- Uses the current long-running flat-observation HAPPO reference experiment.
- Purpose: current main baseline and checkpoint-compatible reference.
- Status: still allowed to continue running; do not stop or overwrite it.

## B. flat_100k_noinit_imitation

- `policy_arch=flat`
- `total_env_steps=100000` preferred; `50000` acceptable if runtime is a concern.
- No init checkpoint.
- Uses `outputs/direct_chase_oracle_dataset/direct_chase_oracle_3v2.npz`
- `uav_imitation_coef=0.03`
- `uav_imitation_until_steps` equals the training step budget.
- Purpose: short flat-policy control under the same no-init imitation condition.

## C. entity_attention_100k_noinit_imitation

- `policy_arch=entity_attention`
- Same step budget, rollout length, max steps, opponent, and imitation settings as B.
- No init checkpoint.
- Purpose: test whether entity-attention is stable and shows early signs of
  better firing, hit, survival, or 5v4 transfer behavior.

## Comparison Criteria

The short comparison should report:

- final training return and win/draw/timeout rates;
- MAV survival;
- red missiles fired;
- missile hits;
- blue dead proxy from rich logs when available;
- entropy and log standard deviation stability;
- 3v2 seen eval;
- 5v4 zero-shot eval.

## Interpretation Boundary

The 50k/100k comparison is a screening experiment only.

It can support statements such as:

- entity-attention training is stable or unstable;
- entity-attention has or has not shown early firing/hit signals;
- entity-attention is or is not worth a longer 500k run.

It must not be presented as:

- final proof that entity-attention is better than flat HAPPO;
- a full BRMA-MAPPO reproduction;
- evidence for GRU, random mask, or biased random mask.
