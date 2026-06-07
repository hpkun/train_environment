# F-22 MAV Model Decision

## Why F-22

The MAV model is switched to F-22 because the user judged its appearance and
role to be closer to the MAV in the heterogeneous air-combat paper. This is an
engineering approximation for the experiment environment. It does not claim
that the original paper's MAV physical parameters are exactly F-22 parameters.

## What Changed

Only the paper-aligned mainline configs are changed:

- `uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml`
- `uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml`

In these configs, the MAV aircraft model changes from A-4 to `f22`.

The MAV still has `num_missiles=0`. Attack UAVs still use `f16` and still have
`num_missiles=2`.

## What Did Not Change

- observation
- reward
- termination
- missile
- action
- evasion
- PID
- MAPPO
- MAV role

## Caveats

F-22 performance may be stronger than A-4 and may differ from both A-4 and f16.
Earlier A-4 training logs should not be interpreted as F-22-environment
results. After the F-22 audit passes, the next useful validation is a fresh
100k pilot run, not parameter tuning or a method-module change.
