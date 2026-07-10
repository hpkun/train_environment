# brma_tam_scripted_composite_v1

`brma_tam_scripted_composite_v1` is a diagnostic composite reward for the current scripted-launch and scripted-evasion JSBSim setting. It is not a full TAM-HAPPO reward reproduction.

## Active Reward

Attack UAVs use a whitelist total:

`BRMA pitch + BRMA roll + BRMA velocity + 10 R_V + 15 R_A + 10 R_D + R_E`

The BRMA altitude, boundary, advantage, terminal, and death terms are logged only and are not added to the active total.

The MAV uses:

`BRMA pitch + BRMA roll + BRMA velocity + safety + support + event`

MAV safety uses distance, actual incoming missile threat, and 3D blue-to-MAV aspect. MAV support uses a dynamic battlefield center built from alive attack UAVs and alive blue aircraft plus current MAV observation awareness.

## Boundaries

- The mode requires `observation_mode: mav_shared_geo`.
- The mode requires `missile_evasion.mode: brma_scripted` and red scripted evasion enabled.
- R_DM/dodge is diagnostic only because the environment already applies scripted evasion.
- The reward target is the closest alive blue aircraft by true 3D state. It is not gated by launch range, AO, TA, lock, cooldown, ammo, engaged-target deconfliction, or observation track.
- The fire-control launch target still uses the environment launch gate. Reward target and launch target may differ.
- No missile dynamics, hit model, PID, blue rule, action space, or observation space behavior is changed by this reward mode.

## Configs

- `uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_brma_tam_scripted_composite_v1.yaml`
- `uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4_f16_mav_surrogate_brma_tam_scripted_composite_v1.yaml`

## Audit

Use:

```bash
python scripts/audit_brma_tam_scripted_composite_v1.py --episodes 2 --max-steps 50
```

The audit performs fixed-action rollout only. It is intended to verify logging and reward scale, not policy performance.
