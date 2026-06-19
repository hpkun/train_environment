# A-4 Candidate Fix Diagnostics

## Scope

This diagnostic pass evaluated candidate A-4 stability fixes without applying
any changes to the formal environment. No reward, missile, observation, PID,
termination, aircraft XML, or algorithm code was modified.

## Throttle Path

The runtime throttle sweep used `uav_env.JSBSim.simulator.AircraftSimulator` and
the formal JSBSim data package.

### A-4

Throttle command sweep over 10 seconds:

- cmd `0.0`: final speed `216.446 m/s`
- cmd `0.5`: final speed `226.374 m/s`
- cmd `1.0`: final speed `251.818 m/s`
- speed delta, `1.0 - 0.0`: `35.372 m/s`

Engine indicators responded as expected: final `n1/n2` rose from `30/60` at
cmd `0.0` to `100/100` at cmd `1.0`, and thrust rose to about `6414 lbf`.

Conclusion: A-4 throttle path appears active. It is weaker than f16, but not inactive.

### f16

Throttle command sweep over 10 seconds:

- cmd `0.0`: final speed `221.543 m/s`
- cmd `0.5`: final speed `256.616 m/s`
- cmd `1.0`: final speed `305.431 m/s`
- speed delta, `1.0 - 0.0`: `83.887 m/s`

Conclusion: f16 throttle path appears active, with substantially stronger acceleration than A-4.

## Candidate Single-Aircraft Fixes

All scenarios ran for 60 seconds with the formal BRMA PID and A-4 model.

| Scenario | Final Altitude | Altitude Delta | Final Speed | Mean Vertical Velocity | Crash |
|---|---:|---:|---:|---:|---|
| baseline | 3766.6 m | -2329.4 m | 255.4 m/s | -38.8 m/s | false |
| initial_altitude_7000m | 4526.8 m | -2473.2 m | 255.4 m/s | -41.2 m/s | false |
| initial_altitude_8000m | 5713.3 m | -2286.7 m | 256.4 m/s | -38.1 m/s | false |
| initial_altitude_9000m | 6207.7 m | -2792.3 m | 254.8 m/s | -46.5 m/s | false |
| initial_speed_260mps | 3880.3 m | -2215.7 m | 255.0 m/s | -36.9 m/s | false |
| initial_speed_280mps | 3902.5 m | -2193.5 m | 254.8 m/s | -36.6 m/s | false |
| initial_speed_300mps | 3945.4 m | -2150.6 m | 255.5 m/s | -35.8 m/s | false |
| pitch_bias_0.05 | 4786.2 m | -1309.8 m | 254.5 m/s | -21.8 m/s | false |
| pitch_bias_0.10 | 5760.8 m | -335.2 m | 252.0 m/s | -5.6 m/s | false |
| pitch_bias_0.15 | 6894.7 m | +798.7 m | 216.9 m/s | +13.3 m/s | false |
| bounded_random_0.3 | 3865.1 m | -2230.9 m | 134.4 m/s | -37.2 m/s | false |
| bounded_random_0.5 | 1192.3 m | -4903.7 m | 120.2 m/s | -81.7 m/s | false |

## Heterogeneous Environment Temporary Diagnostics

These diagnostics used a script-local subclass/patch only. The formal
`HeteroUavCombatEnv` was not changed.

Key observations from 200-step rollouts:

- Default zero policy is stable for red_0 A-4, final altitude about `4721 m`.
- Default full random/bounded_random without a bound keeps red_0 alive but drops to about `2898 m`.
- Red_0 initial altitude `+1000 m` improves full-random final/min altitude to about `3034 m`.
- Red_0 initial altitude `+2000 m` improves full-random final/min altitude to about `5892 m`.
- Red_0 initial speed `+30/+50 m/s` helps zero policy but does not prevent random-policy low-altitude crash.
- Red_0 action bound `0.3` gives much better bounded-random altitude margin than bound `0.5`, but some runs still terminate early for non-low-altitude reasons.

## Recommended Minimal Formal Change

The least invasive candidate is to adjust the heterogeneous scenario initial
condition for A-4, especially red_0 initial altitude. It does not alter PID,
aircraft XML, reward, missile, observation, action semantics, or termination.

If a second low-intrusion option is needed, use a model-specific bounded action
diagnostic/training wrapper for A-4/MAV before changing the formal environment.

## Not Recommended First

- Direct MAV GCAS: no repository evidence currently supports MAV/A-4 GCAS as a paper requirement.
- Aircraft XML changes: unnecessary for this diagnosis and too invasive.
- PID changes: would affect both f16 and A-4 unless carefully model-gated.
- Reward/missile/observation changes: unrelated to the diagnosed flight-dynamics mismatch.

## Current Interpretation

A-4 throttle is active, but A-4 has weaker acceleration and different trim/control response than f16. The BRMA PID and default targets are f16-oriented, so nominal A-4 level flight is not naturally altitude-holding. Pitch bias is most effective in single-aircraft diagnostics, while higher initial altitude is the cleanest scenario-level mitigation.

## Applied Minimal Fix

The adopted minimal fix is an agent-type-level initial altitude offset:

- `mav` type receives `init_altitude_offset_m: 2000.0` (configurable in `aircraft_type_params`).
- A-4 mav starts at ~8096 m (26562 ft) instead of the paper baseline 6096 m (20000 ft).
- f16 aircraft and all other types remain at the paper baseline altitude.

**What was NOT changed**:
- No aircraft XML modifications.
- No PID changes.
- No MAV GCAS logic.
- No reward / missile / observation / action / termination changes.
- No UavCombatEnv base class behavior change (`UavCombatEnv` has no `agent_init_offsets`).
- This is a scenario initial-condition configuration, not a controller or safety mechanism.

**Verification results (200 steps each)**:
- Zero policy: A-4 red_0 min altitude = 4688 m, no crash.
- Bounded random (±0.5): A-4 red_0 min altitude = 5732 m, no crash.
- Full random: A-4 red_0 survives (4016 m), F-16 red_1 crashes (2483 m).

**Files modified**:
- `uav_env/JSBSim/envs/hetero_uav_combat_env.py` — added `_make_init_state` override, `_init_offsets_for`, FT/M/fps conversion.
- `uav_env/JSBSim/configs/hetero_2v2_mav_attack.yaml` — added `init_altitude_offset_m` / `init_speed_offset_mps` to `aircraft_type_params`.
- `scripts/diagnose_hetero_a4_init_applied.py` — new diagnostic script.
- `scripts/diagnose_jsbsim_formal_env.py` — added `bounded_random` policy, init offset output.
- `tests/test_jsbsim_hetero_init_offsets.py` — 9 tests, all pass.
- `docs/a4_candidate_fix_diagnostics.md` — this section.
