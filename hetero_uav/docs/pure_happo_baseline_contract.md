# Pure HAPPO Baseline Contract

## Scope

The formal baseline uses `hetero_3v2_pure_happo_v1` with the frozen formal-v1
environment contract. The algorithm contract is `pure_happo_sequential_v2`.
It is a vanilla heterogeneous-agent PPO baseline and does not include GRU,
attention, entity encoding, masks, communication faults, or role-local critics.

## Policy And Value Functions

- Three independent actors are used for `red_0`, `red_1`, and `red_2`.
- Each actor receives only its own 68-dimensional local observation.
- Actor parameters and Adam optimizer states are disjoint.
- Each actor produces a three-dimensional raw Gaussian action. The environment
  action is `tanh(raw_action)`.
- Gaussian means are not clipped to the environment action range. Log standard
  deviations are bounded to `[-5, 2]` when constructing the distribution.
- Rollouts store both raw and bounded actions. PPO replay uses the stored raw
  action and the numerically stable tanh Jacobian correction.
- Base Gaussian entropy is logged as a stable exploration proxy; it is not the
  analytic entropy of the transformed distribution.
- One centralized critic receives the 204-dimensional state and returns one
  scalar shared-team value.

## Sequential Update

Each trainer update samples one reproducible random agent order. A shared
correction factor starts at one. For each agent in that order, all PPO epochs
run contiguously with a fixed detached factor. The agent's final policy ratio,
relative to rollout log probability, then updates the detached factor. Invalid
agent samples contribute ratio one. The centralized critic is updated only
after all actor updates.

This is the repository's vanilla sequential HAPPO contract. It is not a
role-local vector-critic variant and is not synchronous MAPPO.

## Returns And Episode Boundaries

Rewards are aggregated as the active-before shared-team mean. GAE stores and
uses separate masks:

- true termination: no bootstrap and no continuation;
- time-limit truncation: bootstrap, but no continuation into the next episode;
- ordinary transition: bootstrap and continue;
- unfinished rollout tail: bootstrap from the stored next value.

## Checkpoint Compatibility

New checkpoints record and evaluation validates:

- `algorithm_contract = pure_happo_sequential_v2`
- `policy_distribution = tanh_squashed_gaussian_raw_action`
- `critic_contract = centralized_shared_scalar_v`
- `gae_contract = separated_termination_truncation`
- actor observation dimension 68, critic state dimension 204, action dimension
  3, three agents, and the formal environment/reward contract versions.

Older checkpoints remain historical artifacts and cannot be continued or
silently evaluated as sequential-v2 results.

## Diagnostics

`train_log.csv` contains finite rollout-level summaries. `update_metrics.jsonl`
contains the fixed agent order, contiguous epoch trace, per-epoch actor metrics,
explicit mean/max/final epoch summaries, final-policy ratio statistics, factor
statistics before and after each actor, critic diagnostics, and GAE boundary
counts.
