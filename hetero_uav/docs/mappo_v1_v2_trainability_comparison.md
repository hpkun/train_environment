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

## Comparison Tool Behavior

- Comparison scripts are diagnostic tools, not experiment runners.
- Comparison scripts are fail-fast:
  subprocess failure, NaN detection, or dimension mismatch raises an exception
  and aborts the script.
- `trainability_summary.json` and `trainability_summary.csv` are
  generated for machine-readable downstream analysis.
- The summary files are intended for later analysis and reproducibility checks,
  not for claiming a final V1/V2 result.
- Zero-shot smoke saves stdout per version and produces
  `zero_shot_smoke_summary.json`.
- Current comparison is not formal because it uses short iterations and a
  single seed.
- The next formal stage requires multiple seeds, multiple evaluation episodes,
  a fixed training budget, and a fixed evaluation protocol.

## Next Stages

1. If V2 smoke is stable, run longer diagnostic (200+ iterations)
2. Consider role-aware MAPPO or attention encoder
3. Do not rush to HAPPO
