# Formal Experiment Protocol

## Red Policy

**Canonical algorithm**: `pure_happo`

- Independent per-agent actor MLPs (no parameter sharing)
- Centralized global V critic
- Sequential HAPPO-style update with correction factor M
- **Tanh-squashed Gaussian** bounded continuous action distribution
- Tanh Jacobian log-prob correction for PPO replay consistency

CLI: `--policy-arch pure_happo`

### Canonical Hyperparameters

- Actor learning rate: `5e-4`
- Critic learning rate: `5e-4`
- PPO clip parameter: `0.2`
- Entropy coefficient: `0.01`
- Value coefficient: `0.5`
- Maximum gradient norm: `10.0`
- Actor PPO epochs: `5`
- Critic epochs: `5`
- Discount factor: `0.99`
- GAE lambda: `0.95`
- Initial action log standard deviation: `-1.204`

These defaults apply only to `policy_arch=pure_happo`; legacy/custom policy
paths retain their existing defaults.

### Evaluation Contract

`pure_happo` has one independent actor per training agent and therefore has a
fixed agent count. Formal evaluation uses exactly the training 3V2 config.
Supplying a 5V4 or other-scale eval config is rejected instead of being
silently skipped. Zero-shot scale transfer belongs to entity-set policies, not
this Pure HAPPO baseline.

Formal Pure HAPPO training starts from scratch and rejects checkpoint
initialization, imitation/BC input, and BRMA random or biased mask flags.

### Logging Contract

Summary rich logging writes `train_metrics.csv` and
`episode_reward_components.csv` only. Per-step aircraft, missile, attention,
reward-component, and target-diagnostic files are reserved for full rich
logging. Reward component identity is checked on every rollout before PPO
update; a mismatch larger than `1e-6` aborts training.

## Blue Opponent

**Canonical opponent**: `brma_rule`

- Fixed rule-based blue opponent (legacy delta-10 heading authority)
- No curriculum, no safe-pursuit extension, no easy/hard variants

CLI: `--opponent-policy brma_rule`

## Prohibited Modifications

- No reward modification
- No missile dynamics modification
- No launch gate modification
- No PID / aircraft XML / action space / observation dimension modification

## Checkpoint Compatibility

- Old `pure_happo_tanh` checkpoints load as `pure_happo` (backward-compat alias)
- Old clamp-Gaussian `pure_happo` checkpoints may still load via `LegacyClampPureHAPPOPolicy` for evaluation only
