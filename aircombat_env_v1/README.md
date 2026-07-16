# aircombat_env_v1

A standalone, repeatable JSBSim F-16 trim and PID experiment. The controller
structure follows BRMA-MAPPO Section 2.4, but its numerical gains, trim biases,
actuator signs, and acceptance thresholds are local engineering values—not
paper parameters.

Dependencies are `numpy`, `PyYAML`, `pymap3d`, `jsbsim`, `scipy`, and `pytest`.
All commands below run from the repository root.

## Required workflow

```powershell
python aircombat_env_v1/scripts/check_actuator_signs.py
python aircombat_env_v1/scripts/find_trim.py
python aircombat_env_v1/scripts/find_trim.py --accept
python aircombat_env_v1/scripts/tune_pid.py
python aircombat_env_v1/scripts/tune_pid.py --accept-candidate
python aircombat_env_v1/scripts/validate_pid.py
python aircombat_env_v1/scripts/validate_pid.py --mark-validated
```

Inspect each generated output before using the corresponding accept flag.
Parameter status progresses from `initial_guess` to `candidate`, then to
`validated`. Validation requires all 72 short 40-second cases and eight
representative 200-second long cases to pass. `--mark-validated` refuses to
update the formal configuration otherwise.

Every closed-loop script uses the same 60 Hz physics/PID loop and 5 Hz command
interface: one high-level command is held for 12 physics frames. In the
combined task the interface is still called at 5 Hz, while deterministic target
values change only every 3, 4, or 5 seconds.

## Tests and smoke runs

```powershell
python -m pytest aircombat_env_v1/tests -q
python -m pytest aircombat_env_v1/tests -q -m integration
python aircombat_env_v1/scripts/check_actuator_signs.py --stabilization-duration 3
python aircombat_env_v1/scripts/find_trim.py --duration 5 --grid-size 5
python aircombat_env_v1/scripts/tune_pid.py --roll-duration 1 --pitch-duration 1 --speed-duration 1 --joint-duration 1 --maxiter 1 --popsize 2
python aircombat_env_v1/scripts/validate_pid.py --short-duration 1 --long-duration 1
```

Smoke outputs prove workflow execution only; they are not validated flight
parameters. Timestamped artifacts are written under `aircombat_env_v1/outputs`
and never overwrite prior runs. See `PID_DESIGN.md` for formulas, status rules,
and metric definitions.
