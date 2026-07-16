# aircombat_env_v1

A standalone, minimal JSBSim F-16 flight-control experiment. It contains no
reinforcement-learning, combat, reward, missile, sensor, or multi-agent code.

## Install and test

Python dependencies are `numpy`, `PyYAML`, `pymap3d`, `jsbsim`, `scipy`, and
`pytest`. From the repository root:

```powershell
python -m pytest aircombat_env_v1/tests
python aircombat_env_v1/scripts/check_actuator_signs.py
python aircombat_env_v1/scripts/find_trim.py
python aircombat_env_v1/scripts/tune_pid.py
python aircombat_env_v1/scripts/validate_pid.py
```

Run actuator-sign measurement first, then trim search, sequential PID tuning,
and validation. Scripts write timestamped experimental outputs beneath
`aircombat_env_v1/outputs/`; trim alone updates `configs/f16_pid_v1.yaml` as
required. Tuning never overwrites prior output.

Configuration provenance and the boundary between paper statements and local
engineering choices are documented in `PID_DESIGN.md`.
