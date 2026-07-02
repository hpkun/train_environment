# Blue Easy Opponent Diagnostics

This note documents two script-layer blue opponent modes used only for pressure
diagnostics. They do not change reward, missile dynamics, launch gates, PID,
aircraft XML, action space, observation dimension, or red-side fire control.

## Opponent Modes

### `tam_greedy_easy`

Weak TAM-style greedy finite-state opponent.

- Selects the nearest visible red target only.
- Does not prioritize MAV by role.
- Does not keep long target persistence.
- Uses `search`, `approach`, `attack`, `extend`, and `evade` states.
- Enters short `extend` after close pass, over-close range, or target loss.
- Caps speed to `0.65`, pitch to `[-0.25, 0.25]`, and heading delta to `0.12`
  normalized heading per decision.

### `brma_rule_safe_pursuit_easy`

Intermediate mode between `brma_rule` and `brma_rule_safe_pursuit`.

- Uses delta-10 BRMA rule behavior during the first 200 steps.
- Afterward, uses safe pursuit with probability `0.6`.
- Postprocesses safe-pursuit actions with speed cap `0.7`, pitch cap
  `[-0.35, 0.35]`, and heading delta cap `0.15`.
- Uses short extend/search behavior after close pass or prolonged target loss.

## Diagnostic Script

Example:

```powershell
python scripts/diagnose_blue_pressure.py `
  --config uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_happo_ref_v0.yaml `
  --opponent-policies brma_rule tam_greedy_easy brma_rule_safe_pursuit_easy brma_rule_safe_pursuit `
  --episodes 5 `
  --max-steps 1000 `
  --device cpu `
  --summary-json outputs/blue_pressure_diagnostics/summary.json `
  --csv outputs/blue_pressure_diagnostics/summary.csv
```

If `--checkpoint` is omitted, red actions are zero and the script measures
baseline blue pressure. If `--checkpoint` is provided, it loads the red policy
through the existing HAPPO eval checkpoint loader.
