# MAPPO V1 vs V2 Short Trainability Comparison

## Purpose

Compare V1 (brma_sensor) and V2 (mav_shared_geo) observation modes on a
short MAPPO training run.  Used only to verify pipeline stability and
observation-mode compatibility.  **Not** a formal experiment conclusion.

## V1: brma_sensor

- BRMA radar/FOV/RCS observation path
- actor_obs_dim = 140
- critic_state_dim = 700
- Retained for compatibility and ablation

## V2: mav_shared_geo

- TAM-HAPPO-style geometric observation
- actor_obs_dim = 96
- critic_state_dim = 480
- MAV-mediated information sharing (direct > shared > unavailable)

## Fairness Caveat

- V1 and V2 use different observation dimensions and content
- Short smoke runs (20 iterations, 128 max-steps) are not sufficient
  for conclusions about which observation mode is better
- Formal experiments need multiple seeds, episodes, and fixed protocols

## Next Stages

1. If V2 smoke is stable, run longer diagnostic (200+ iterations)
2. Consider role-aware MAPPO or attention encoder
3. Do not rush to HAPPO
