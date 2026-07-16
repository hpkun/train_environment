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
python aircombat_env_v1/scripts/tune_pid.py --pitch-integral-only --config path/to/candidate_config.yaml --joint-duration 90
python aircombat_env_v1/scripts/validate_pid.py --mode quick
python aircombat_env_v1/scripts/validate_pid.py --mode full
python aircombat_env_v1/scripts/validate_pid.py --mode full --mark-validated
```

Inspect each generated output before using the corresponding accept flag.
Parameter status progresses from `initial_guess` to `candidate`, then to
`validated`. Quick mode runs eight representative 200-second health cases.
Full mode runs the 72-case matrix plus those eight long cases.
`--mark-validated` is rejected in quick mode and refuses to update the formal
configuration unless full validation passes.

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
python aircombat_env_v1/scripts/validate_pid.py --mode quick --quick-duration 1
```

Smoke outputs prove workflow execution only; they are not validated flight
parameters. Timestamped artifacts are written under `aircombat_env_v1/outputs`
and never overwrite prior runs. See `PID_DESIGN.md` for formulas, status rules,
and metric definitions.

## Minimal JSBSim 1v1 environment

`AirCombat1v1Env` is a single-agent Gymnasium environment: external actions
control red and an internal `straight` or pure-`pursuit` rule controls blue.
The action is a three-vector for target pitch, relative heading, and speed;
the observation is a clipped 16-vector. One environment step holds both
commands for 12 interleaved 60 Hz JSBSim/PID frames.

The fixed `tail_chase` scenario uses a geometric attack zone instead of
missiles. This is intentionally only a learnability check, with no radar,
weapons, rewards beyond the small geometric potential, or multi-agent API.

```powershell
python aircombat_env_v1/scripts/check_1v1_env.py
python aircombat_env_v1/scripts/run_rule_1v1.py --episodes 5 --opponent straight
```
