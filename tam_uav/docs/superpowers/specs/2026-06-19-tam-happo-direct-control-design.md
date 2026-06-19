# TAM-HAPPO 4D Direct-Control Design

## Scope

Implement the TAM-HAPPO direct flight-control path only in `tam_uav`. Preserve the legacy three-dimensional PID target interface for legacy configurations. Do not modify `hetero_uav`, aircraft XML, engine XML, missile dynamics, the core reward, or observation dimensions.

## Environment contract

`action_interface: tam_direct_fcs_4d` selects a four-dimensional action space ordered as throttle, aileron, elevator, and rudder. Each value is clipped to `[-1, 1]`, optionally quantized to `tam_action_levels` evenly spaced levels, and written directly to JSBSim. Throttle is mapped to `[tam_throttle_min, tam_throttle_max]`; the three surface controls remain in `[-1, 1]`. This branch never invokes target parsing or PID control.

The environment records raw, clipped/quantized, and mapped commands for diagnostics. Scripted missile-evasion overrides are gated independently for red and blue and default off in the TAM configurations. Missile-warning observations remain unchanged.

## Composition and configuration

The formal 3v2 configuration uses F22 `red_0` as the missile-free MAV and F16 for every UAV. The 5v4 configuration uses the same roles and control contract and exists for evaluation and zero-shot scaling only. Initial geometry is copied from the closest existing `happo_ref_v0` 3v2 and F22 5v4 configurations.

## Training and opponent path

Training and evaluation obtain `action_dim` from `env.action_space[red_id].shape[0]` and pass it to policy and rollout-buffer construction. Checkpoint metadata records this dimension. Existing continuous Gaussian policies remain unchanged except for receiving dimension four.

`tam_direct_fsm` preserves the existing nearest/BRMA target-selection semantics and translates the selected relative target into throttle, aileron, elevator, and rudder commands. Its output is clipped to `[-1, 1]` and is quantized/mapped by the environment like learned actions.

## Verification

Tests cover configuration loading, action shape, exact quantization and throttle mapping, model/role composition, PID bypass, blue-rule output, dynamic training dimensions, and a minimal checkpoint smoke. The fixed-action audit records command and aircraft-response telemetry for F22 and F16. A 2k training smoke must save `latest/model.pt`; only after that succeeds may the requested 50k run start. No 500k or 1M run is authorized.

## Repository isolation

All tracked changes remain under `tam_uav`. `hetero_uav - 副本` is not deleted because its current file set differs materially from `tam_uav`, so it cannot be safely classified as an accidental duplicate.
