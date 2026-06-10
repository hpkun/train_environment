# Baseline 1M Plan

## Formal Baseline Definition
- Shared MLP MAPPO (96¡ú256¡ú128¡ú3 actor, 480¡ú256¡ú128¡ú1 critic)
- V2 observation (mav_shared_geo)
- brma_legacy reward
- brma_rule opponent
- no_mav_trim config
- 3v2 train, 3v2 + 5v4 eval

## Why not rule_nearest
rule_nearest produces false positive results ¡ª blue does not attack, red wins
by timeout + alive advantage. Not a valid combat baseline.

## Why not GRU/attention now
The immediate goal is a reproducible shared MLP baseline. GRU/attention are
subsequent model improvements, not the baseline.

## Why 1M steps
Parent project uses 10M steps. 200k was insufficient to determine whether
shared MLP can learn at all. 1M is a cost/credibility compromise.

## Acceleration
- eval_interval_steps=100000 (reduced frequency)
- no ACMI export during training
- rollout_length=512
- Final eval and ACMI after training completes
