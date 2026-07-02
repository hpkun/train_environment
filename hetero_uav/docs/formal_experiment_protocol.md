# Formal Experiment Protocol

## Red Policy

**Canonical algorithm**: `pure_happo`

- Independent per-agent actor MLPs (no parameter sharing)
- Centralized global V critic
- Sequential HAPPO-style update with correction factor M
- **Tanh-squashed Gaussian** bounded continuous action distribution
- Tanh Jacobian log-prob correction for PPO replay consistency

CLI: `--policy-arch pure_happo`

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
