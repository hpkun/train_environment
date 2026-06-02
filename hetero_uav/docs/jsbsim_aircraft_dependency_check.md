# JSBSim Aircraft Dependency Check

## Scope

This note documents the dependency check for the formal JSBSim aircraft models under:

- `uav_env/JSBSim/data/aircraft/A-4/A-4.xml`
- `uav_env/JSBSim/data/aircraft/f16/f16.xml`

The check follows the XML declarations, not directory-shape assumptions. A model only needs a `Systems/` directory when its aircraft XML declares `<system file="...">` dependencies.

## A-4 Result

- XML exists: yes
- Aircraft directory: `uav_env/JSBSim/data/aircraft/A-4`
- `Systems/` directory exists: no
- Engine references:
  - `J52` -> `uav_env/JSBSim/data/engine/J52.xml`
- Thruster references:
  - `direct` -> `uav_env/JSBSim/data/engine/direct.xml`
- System references: none
- Include references: none
- Missing dependencies: none
- `load_model("A-4")`: success
- `run_ic()`: success

Conclusion: A-4 does not declare any system XML dependencies, so `A-4/Systems/` is not required.

## f16 Result

- XML exists: yes
- Aircraft directory: `uav_env/JSBSim/data/aircraft/f16`
- `Systems/` directory exists: yes
- Engine references:
  - `F100-PW-229` -> `uav_env/JSBSim/data/engine/F100-PW-229.xml`
- Thruster references:
  - `direct` -> `uav_env/JSBSim/data/engine/direct.xml`
- System references:
  - `pushback` -> `uav_env/JSBSim/data/aircraft/f16/Systems/pushback.xml`
  - `hook` -> `uav_env/JSBSim/data/aircraft/f16/Systems/hook.xml`
- Include references: none
- Missing dependencies: none
- `load_model("f16")`: success
- `run_ic()`: success

Conclusion: f16 declares `<system file="pushback">` and `<system file="hook">`, so the `f16/Systems/` directory is required.

## Judgment Rule

- Directory structures do not need to be identical between aircraft models.
- Required resources are determined by actual XML references.
- A missing directory is not a problem unless the aircraft XML references files inside it.
- `load_model()` and `run_ic()` are the final runtime validation checks.

## Adding New Aircraft

For any new aircraft model:

1. Parse the aircraft XML for `engine`, `thruster`, `system`, and `include` references.
2. Verify referenced files exist under the JSBSim package data root.
3. Do not create placeholder directories or copy unrelated systems for visual consistency.
4. Run `python scripts/check_jsbsim_aircraft_dependencies.py`.
5. Confirm `load_model()` and `run_ic()` both succeed.
