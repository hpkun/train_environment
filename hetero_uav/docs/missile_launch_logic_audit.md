# Missile Launch Logic Audit

## Purpose

This audit checks whether missile launch accounting matches the environment
setting. It focuses on remaining ammo, launch conditions, launch intervals, and
target roles. It is not training and does not evaluate an algorithm.

## Expected Launch Logic

UAV carries 2 missiles in the current hetero paper-aligned configs.

MAV carries 0 missiles, so the MAV should never launch a missile.

A launch requires the existing geometry gate: range, AO, and TA must satisfy the
environment thresholds. A launch also requires lock delay, cooldown, remaining
ammo, and engaged target deconfliction.

## Current Fix

The environment now checks remaining ammo before target search. If an aircraft
has no missile left, the launch scan is blocked and its lock state is cleared.
This keeps lock delay and cooldown from bypassing the ammo constraint.

Tacview export now writes explicit entity types. Aircraft use
`Type=Air+FixedWing`; missile entities use `Type=Weapon+Missile`.

Launch diagnostics record shooter and target role/model metadata when the
hetero environment exposes it. The diagnostics also record missile count before
and after launch.

## How To Run

```powershell
python scripts/diagnose_missile_launch_logic.py --config uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml --steps 500 --blue-policy rule_nearest
```

The output JSON includes total launches, launches by shooter, launches by
target role, launches against MAV, ammo violations, MAV launch violations, and
minimum launch interval per shooter.

## ACMI Interpretation

Aircraft should appear as fixed-wing entities in Tacview. Missiles should
appear as missile entities when missiles are recorded.

One UAV should not exceed its configured missile count. The MAV should not
launch a missile. The diagnostics show whether blue or red missiles targeted
the MAV.
