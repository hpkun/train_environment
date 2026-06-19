# A-4 Stability Audit

## Paper And Repository Evidence

Repository text search covered the current `hetero_uav` project, parent BRMA environment notes, root `docs`, scripts, and available extracted paper text using keywords including `GCAS`, `ground collision`, `terrain avoidance`, `low altitude`, `safety protection`, `MAV`, `A-4`, `heterogeneous`, `HAPPO`, `TAM`, and `UAV missile`.

Findings:

- The repository contains BRMA environment evidence for a Blue-only GCAS engineering option.
- The copied BRMA environment explicitly describes GCAS as a Blue-only safety net.
- Existing paper-alignment notes state that Blue GCAS is an engineering option and should be disabled for paper-aligned training defaults.
- No repository text was found that explicitly states MAV or A-4 should have GCAS, terrain avoidance, or low-altitude protection.
- No original heterogeneous MAV/A-4 paper text was found that confirms MAV GCAS.

Conclusion: there is no current repository evidence that MAV/A-4 GCAS is part of the original heterogeneous setting. Adding MAV GCAS now would be an environment-design change, not a confirmed paper requirement.

## A-4 And f16 Model Structure Differences

The XML-only model audit found:

| Field | A-4 | f16 |
|---|---:|---:|
| wingarea | 260.00 | 300 |
| wingspan | 27.50 | 30 |
| chord | 9.45 | 11.32 |
| htailarea | 52.00 | 63.7 |
| vtailarea | 31.20 | 54.75 |
| emptywt | 10250 | 17400 |
| ixx / iyy / izz | 5778 / 25900 / 20082 | 9496 / 55814 / 63100 |
| engine | J52 | F100-PW-229 |
| thruster | direct | direct |
| fuel tanks | 1 | 4 |
| flight-control channels | 6 | 10 |
| function count | 26 | 42 |
| aerosurface scales | 4 | 7 |
| system references | none | pushback, hook |
| Systems directory | absent | present |
| XML size | 731 lines | 1941 lines |

The A-4 model is structurally simpler than f16 and has fewer aerodynamic/control functions and channels. A-4 does have elevator, aileron, and rudder command paths. The text audit did not find `fcs/throttle-cmd-norm` in A-4 XML, while f16 does expose it.

## A-4 BRMA PID Diagnosis

Formal BRMA PID diagnostics used:

- `uav_env.JSBSim.simulator.AircraftSimulator`
- `uav_env.JSBSim.pid_controller.PIDController`

The old `uav_env/JSBSim/core/aircraft.py` path was not used.

Key 60-second results:

| Scenario | Final Altitude | Final Speed | Mean Vertical Velocity | Mean Throttle | Mean Elevator | Crash |
|---|---:|---:|---:|---:|---:|---|
| A-4 level `[0,0,0]` | 3766.6 m | 255.4 m/s | -38.8 m/s | 0.861 | -0.017 | false |
| A-4 mild climb `[0.1,0,0]` | 5760.8 m | 252.0 m/s | -5.6 m/s | 0.992 | -0.042 | false |
| A-4 higher speed `[0,0,0.5]` | 3845.6 m | 283.3 m/s | -37.5 m/s | 1.000 | -0.019 | false |
| A-4 bounded random `[-0.3,0.3]` | 5216.6 m | 148.7 m/s | -14.7 m/s | 0.643 | -0.038 | false |
| f16 level `[0,0,0]` | 5796.0 m | 252.3 m/s | -5.0 m/s | 0.442 | -0.053 | false |

Interpretation:

- A-4 loses much more altitude than f16 under the same nominal level action.
- A mild positive pitch target recovers most of the altitude loss.
- Higher target speed alone only modestly improves altitude retention.
- A-4 level flight shows sustained negative vertical velocity.
- A-4 needs much higher throttle command than f16 under the same target.

Primary diagnosis: the formal BRMA PID and default high-level targets are effectively tuned around f16 behavior and are not trimmed for level A-4 flight. The A-4 model has different aerodynamic/control response and lower altitude margin under nominal level commands.

## Random Crash Cause

The earlier `jsbsim_hetero` 200-step random rollout used full action-space random commands. In that setting `red_0` A-4 can be driven into low altitude. The diagnosis suggests this is mainly from:

- non-trimmed level behavior already losing altitude;
- random pitch/heading/speed commands reducing energy and altitude margin;
- no red/MAV GCAS in the BRMA environment path;
- A-4 dynamics/control response differing from the f16 baseline.

This does not indicate a JSBSim load failure, missing Systems resource, or a required immediate GCAS hook.

## Why Not Add MAV GCAS Immediately

Adding MAV GCAS would change the environment contract and may alter learning dynamics. The repository currently only supports Blue-only GCAS as a BRMA engineering option; no evidence was found that original MAV/A-4 heterogeneous experiments used MAV GCAS. Therefore, adding MAV GCAS should require an explicit design decision and documentation.

## Candidate Fixes

Ordered from least invasive to most invasive:

1. Increase A-4 initial altitude or initial speed in heterogeneous scenarios.
2. Use a smaller action range for MAV/A-4 while preserving the global action-space shape.
3. Add a model-level target-pitch bias or trim for A-4.
4. Adjust A-4 target velocity mapping or minimum velocity.
5. Use the A-4 identity/role externally while retaining f16 dynamics for controlled baseline comparisons.
6. Add explicit MAV safety protection only after documenting it as an intentional environment change.

Recommended minimal next step: test higher A-4 initial altitude/speed and a bounded MAV action range in diagnostics before changing the formal environment.
