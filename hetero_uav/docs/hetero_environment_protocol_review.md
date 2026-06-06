# Hetero Environment Protocol Review

This document freezes the environment protocol decisions before training.
It is NOT a method module. No training, no algorithm changes.

## 1. Protocol Decision

### Main Paper-Aligned Protocol

This is the **main protocol**.

- **Train**: 3v2 (red = 1 MAV + 2 attack_uav, blue = 2 attack_uav)
- **Eval**: 5v4 (red = 1 MAV + 4 attack_uav, blue = 4 attack_uav)
- The extra red aircraft is the non-shooting MAV.
- Red attack UAV count == blue attack UAV count.
- Configs:
  - `hetero_mav_shared_geo_3v2.yaml`
  - `hetero_mav_shared_geo_5v4.yaml`

### Hard Ablation (Balanced)

This is a **hard ablation**, not the main protocol.

- **Train**: 3v3 (red = 1 MAV + 2 attack_uav, blue = 3 attack_uav)
- **Eval**: 4v4 (red = 1 MAV + 3 attack_uav, blue = 4 attack_uav)
- Total aircraft counts match, but red has one fewer attacking UAV
  than blue due to the non-shooting MAV. This makes the task harder.
- Configs:
  - `hetero_balanced_mav_shared_geo_3v3.yaml`
  - `hetero_balanced_mav_shared_geo_4v4.yaml`

### Optional Reward Overlay

`minimal_v1` is an optional role-aware reward overlay.

- Configs:
  - `hetero_mav_shared_geo_3v2_reward_minimal.yaml`
  - `hetero_mav_shared_geo_5v4_reward_minimal.yaml`
  - `hetero_diagnostic_close_range_mav_shared_geo_3v2_reward_minimal.yaml`

### Optional Reference

V1 `brma_sensor` configs are available as an optional reference.
They are NOT required for main protocol readiness.

## 2. Reward Decision

- `brma_legacy` is the **default baseline reward**
- `minimal_v1` is an **optional** role-aware overlay
- **No termination change** — termination logic is unchanged from BRMA

The main paper-aligned configs explicitly declare `hetero_reward_mode: "brma_legacy"`
for protocol clarity, even though this is already the default.

## 3. Opponent Decision

- `rule_nearest` remains available
- `greedy_fsm` is a diagnostic environment opponent (engineering approximation)
- `greedy_fsm` is **NOT** yet declared the final default training opponent
- **Blue opponent default remains undecided** — run
  `scripts/validate_blue_opponent_protocol.py` before any long baseline
  to compare opponent behaviour
- `greedy_fsm` may be designated as a **hard opponent** rather than the
  default baseline opponent

## 4. What is Frozen

- Aircraft models: MAV = A-4, attack_uav = f16
- MAV/UAV missile counts: MAV = 0 missiles, attack_uav = 2 missiles
- Paper-aligned composition: red attack UAV count == blue attack UAV count
- V2 observation mode: `mav_shared_geo`
- Decision frequency: `sim_freq = 60`, `agent_interaction_steps = 12` → `decision_dt = 0.2s`
- Reward default: `brma_legacy`
- Observation dimensions: actor_dim = 96, critic_dim = 480 (V2 mav_shared_geo)

## 5. What Remains Open

- Whether to train with `brma_legacy` or `minimal_v1` reward
- Whether `greedy_fsm` should replace `rule_nearest` for baseline training
- Whether reward/termination need further changes after short smoke
- **No method module yet** — do not enter until protocol is frozen

## 6. Protocol Review Checklist

| Check | Status |
|---|---|
| Configs have explicit `hetero_reward_mode` | ✓ |
| `brma_legacy` is default | ✓ |
| `minimal_v1` is optional overlay | ✓ |
| Paper-aligned composition correct (red attack = blue attack) | ✓ |
| Balanced config has documented attack-UAV asymmetry | ✓ |
| Decision frequency standardized (60 Hz / 12 steps) | ✓ |
| `max_steps` >= 1000 for main configs | ✓ |
| Aircraft models frozen | ✓ |
| Missile counts frozen | ✓ |
| Observation mode frozen (V2 mav_shared_geo) | ✓ |
| Termination unchanged | ✓ |
| greedy_fsm is diagnostic only | ✓ |
| No method module | ✓ |

## 7. Next Environment Task

```
environment_protocol_review_then_optional_training_decision
```

After this review: user decides whether to start short training smoke,
continue improving greedy_fsm, or both. Do NOT automatically enter
a method module.
