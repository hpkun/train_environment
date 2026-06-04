# hetero_uav

`hetero_uav` is a self-contained debug environment project for heterogeneous
MAV-UAV cooperative air combat. It is intended as the environment foundation for
later heterogeneous zero-shot composition generalization research.

The design is informed by Chen et al. 2026: MAVs and UAVs are represented as
autonomous agents, heterogeneity is expressed through platform type, sensing
range, missile load, maneuver limits, and reward role, and the environment
exposes per-agent observations plus a centralized global state for CTDE methods.

This first version is an environment skeleton and debug implementation with a
minimal ordinary MAPPO baseline for trainability checks. It does not implement
type-aware attention, a mask generator, HAPPO, BRMA-MAPPO changes, or paper
experiments.

## Project Structure

```text
hetero_uav/
  README.md
  setup.py
  requirements.txt
  scripts/
    check_env.py
    run_random_env.py
  tests/
    test_env_smoke.py
  examples/
    README.md
  uav_env/
    __init__.py
    make_env.py
    registry.py
    configs/
      hetero_2v2_debug.yaml
      hetero_3v3_debug.yaml
    JSBSim/
      __init__.py
      envs/
        __init__.py
        hetero_uav_env.py
      core/
        __init__.py
        aircraft.py
        aircraft_types.py
        agent.py
        missile.py
        sensor.py
        scenario.py
        observation.py
        reward.py
        termination.py
        info.py
        utils.py
      tasks/
        __init__.py
        hetero_combat_task.py
      models/
        README.md
      configs/
        README.md
    wrappers/
      __init__.py
      mappo_wrapper.py
      gym_wrapper.py
```

## Install

```bash
cd hetero_uav
pip install -e .
```

## Check The Environment

```bash
python scripts/check_env.py --config uav_env/configs/hetero_2v2_debug.yaml
```

The script loads the config, creates `HeteroUAVEnv`, calls `reset`, prints
observation/state/action shapes, and runs 10 random steps.

## Run A Random Rollout

```bash
python scripts/run_random_env.py --config uav_env/configs/hetero_2v2_debug.yaml
```

The script runs one full random episode and prints alive counts, MAV survival,
rewards, and an episode summary.

## Python API

```python
from uav_env import make_env

env = make_env("uav_env/configs/hetero_2v2_debug.yaml")
obs, info = env.reset()
obs, rewards, terminated, truncated, info = env.step(env.sample_actions())
state = env.get_state()
avail_actions = env.get_avail_actions()
```

The package exposes:

- `uav_env.HeteroUAVEnv`
- `uav_env.make_env`
- `uav_env.registry.make`

`HeteroUAVEnv` is implemented in
`uav_env/JSBSim/envs/hetero_uav_env.py`.

## MAPPO Adapter

The native environment follows the current `train_environment` worker style:

- `reset(seed=None, options=None) -> (obs, info)`
- `step(actions) -> (obs, rewards, terminated, truncated, info)`

By default, debug configs set `controlled_side: "red"` and
`opponent_policy: "rule_nearest"`. In this mode, `agent_ids`,
`num_agents`, `action_space`, `observation_space`, returned observations,
rewards, and dones only expose red agents for training. Blue agents remain
inside the environment and are controlled by a nearest-target rule policy, so
they still affect observations, global state, missile logic, rewards,
termination, and `info` statistics.

Set `controlled_side: "all"` to expose both red and blue agents as controllable
agents, matching the original all-agent debug behavior.

For MAPPO runners that expect stacked arrays and centralized state, use:

```python
from uav_env import make_env
from uav_env.wrappers import MAPPOEnvWrapper

env = MAPPOEnvWrapper(make_env("uav_env/configs/hetero_2v2_debug.yaml"))
obs, state, info = env.reset()
next_obs, next_state, rewards, dones, info = env.step(actions)
```

## Current Environment Interface

`HeteroUAVEnv` provides:

- `reset(seed=None, options=None)`
- `step(actions)`
- `close()`
- `render(mode=None)`
- `get_obs()`
- `get_state()`
- `get_avail_actions()`
- `num_agents`, `n_agents`
- `obs_shape`, `state_shape`, `action_shape`
- `action_space`, `observation_space`

Actions are shared across all agent types and use three continuous values in
`[-1, 1]`: target pitch, target heading, and target velocity. This mirrors the
high-level action style used by the existing BRMA-MAPPO environment, while the
first implementation maps those commands through simplified kinematics.

## Heterogeneity

The debug configs support:

- `mav`: stronger radar, leader survival reward role.
- `attack_uav`: missile-carrying combat UAV.
- `scout_uav`: reserved sensor-focused UAV type.
- `interceptor_uav`: reserved high-speed intercept type.

`hetero_2v2_debug.yaml` uses 1 red MAV, 1 red attack UAV, and 2 blue attack
UAVs. `hetero_3v3_debug.yaml` uses 1 red MAV, 2 red attack UAVs, and 3 blue
attack UAVs.

## Environment Design Choice

The first version of this environment intentionally uses the high-level action
format `[pitch, heading, speed]`, keeping the training-facing action interface
close to the current BRMA-MAPPO style. This keeps the environment useful for
early MAPPO integration while avoiding premature coupling to low-level actuator
commands.

The default flight dynamics model is a simplified kinematic skeleton. An
optional JSBSim backend now exists behind the same public aircraft interface,
but the YAML configs still default to `dynamics_backend: "simple"` so the
environment remains runnable without the Python `jsbsim` package.

Set `dynamics_backend: "jsbsim"` or pass `dynamics_backend="jsbsim"` to
`make_env` to construct `JSBSimAircraftPlatform` instances. In that mode,
high-level `[pitch, heading, speed]` actions are converted to JSBSim control
commands by a small PID controller. The current JSBSim integration is a minimal
single-aircraft and environment backend check; it is not yet a tuned
research-grade flight controller, and the project is not entering training at
this stage.

Local JSBSim assets are stored under `uav_env/JSBSim/models/`:

- `aircraft/A-4/A-4.xml` from the JSBSim upstream A4 model provided for this
  project;
- `aircraft/F-16/F-16.xml` copied from the parent project's existing local
  aircraft data;
- `engine/J52.xml`, `engine/F100-PW-229.xml`, and `engine/direct.xml` for the
  aircraft engines/thruster references.

Use the JSBSim diagnostics without running training:

```bash
pip install -r requirements.txt
# or:
pip install jsbsim==1.1.6

python scripts/check_jsbsim_models.py
python scripts/run_jsbsim_single_aircraft.py --model A-4 --duration 10
python scripts/run_jsbsim_single_aircraft.py --model F-16 --duration 10
python scripts/check_env.py --config uav_env/configs/hetero_2v2_jsbsim_debug.yaml
pytest tests/test_jsbsim_backend.py
```

`scripts/check_jsbsim_models.py` checks the local model tree, aircraft XML,
engine/thruster XML, `load_model`, and `run_ic`. 
`scripts/run_jsbsim_single_aircraft.py` runs one aircraft for a short duration
and prints time, local position, altitude, speed, attitude, and crash status.
`uav_env/configs/hetero_2v2_jsbsim_debug.yaml` is the environment-level JSBSim
smoke-test config.

If `jsbsim` is not installed, model-file checks still run, command-line scripts
print install hints, and backend execution tests are skipped with an explicit
dependency message.

Environment finalization and protocol taxonomy are tracked in:

- `docs/hetero_environment_finalization_plan.md`

Use the readiness audit before entering any method module:

```bash
python scripts/audit_hetero_environment_readiness.py --include-v1 --steps 3
```

## Formal Heterogeneous Composition Configs

Formal JSBSim train/test composition configs are under
`uav_env/JSBSim/configs/`:

- `hetero_train_2v2_mav_attack.yaml`
- `hetero_test_3v3_mav_2attack.yaml`
- `hetero_test_3v3_mav_attack_scout.yaml`
- `hetero_test_3v3_mav_attack_interceptor.yaml`

Use the composition diagnostic before starting MAPPO work:

```bash
python scripts/diagnose_hetero_compositions.py
pytest tests/test_jsbsim_hetero_compositions.py
```

The older `uav_env/JSBSim/configs/hetero_2v2_mav_attack.yaml` is retained as a
debug alias. The formal train config is
`hetero_train_2v2_mav_attack.yaml`.

MAPPO training should not start until composition configs and type/role
observation fields pass these tests.

Even with the simplified FDM, the environment now implements a minimal credible
combat layer:

- `radar_range` affects whether enemy entities are visible in each agent's
  observation;
- observations include entity type, side, alive, visible, and missile-left
  fields, while invisible enemy kinematic fields are masked;
- fire-control checks alive status, missile inventory, cooldown, sensor
  visibility, attack range, launch range, and LOS angle before automatic
  launch;
- `attack_range` is retained for shaping/future extensions, while
  `launch_range` controls whether a missile is actually consumed in the current
  deterministic hit-zone model;
- missile events record launches, hits, misses, blocked launches, cooldown, and
  remaining missiles into `info`;
- reward shaping uses `reward_role` so MAVs emphasize survival, attack UAVs
  emphasize attack windows and kills, scouts receive detection/survival reward,
  and interceptors receive pressure/attack-window reward;
- termination reports `win_flag`, `winner`, and `termination_reason` for blue
  elimination, red elimination, MAV loss, episode-limit alive/kills advantage,
  or draw.

Heterogeneity is represented through `aircraft type`, `radar_range`,
`missile_num`, `max_speed_scale`, `max_g`, and `reward_role`. MAV platforms use
the `leader_survival` reward role, while UAV platforms use attack-oriented or
specialized roles such as `attack`, `scout`, and `intercept`.

By default, blue-side agents are controlled by the `rule_nearest` script policy.
Each blue agent steers toward the nearest alive red target while the red side is
exposed as the controlled training side. This setup is meant to validate the
heterogeneous composition zero-shot generalization problem first: train on a
small MAV + attack-UAV composition, then evaluate on larger or compositionally
different MAV/UAV teams. Higher-fidelity JSBSim/PID/PN missile implementation is
left as the next environment fidelity step.

For serious learning experiments, the next validation step is to run an ordinary
MAPPO trainability baseline against this environment layer, then replace the
kinematic FDM with real JSBSim/PID/PN components once the task semantics are
stable.

## Parent Project Dependency

There is no runtime dependency on the parent `train_environment` project. The
code does not import `my_uav_env`, `envs.JSBSim`, or modify `sys.path` to reach
the parent directory.

The current limitation is fidelity, not independence: the JSBSim backend is
available for local model loading and stepping, while Tacview export and full
proportional-navigation missile dynamics are not migrated yet. They can be
added later inside `uav_env/JSBSim/` by extending the backend classes in
`core/aircraft.py` and `core/missile.py`.

## Tests

```bash
pytest tests/test_env_smoke.py
```

The smoke test checks import, config loading, reset, one step, observation/state
shape, and required `info` fields.

## MAPPO Baseline

The project includes a minimal ordinary MAPPO baseline under
`algorithms/mappo/`. It is intentionally small and only validates that
`HeteroUAVEnv` can be used in a CTDE training loop:

- shared continuous-action actor, `action_dim=3`;
- per-red-agent actor input from local observation;
- centralized critic input from global state;
- GAE;
- PPO clipped objective;
- CPU/GPU device selection;
- model save/load.

Train on the padded 2v2 MAV + attack-UAV setup:

```bash
python scripts/train_mappo.py --config configs/train_mappo_hetero_2v2.yaml
```

Fast smoke run:

```bash
python scripts/train_mappo.py --config configs/train_mappo_hetero_2v2.yaml --debug
```

Evaluate a saved model on a zero-shot composition config:

```bash
python scripts/eval_mappo.py --model outputs/mappo_hetero_2v2/<run>/model.pt --env-config uav_env/configs/hetero_test_3v3_mav_attack_scout.yaml --episodes 100
```

The ordinary MLP MAPPO baseline requires compatible padded observation/state
dimensions between train and eval configs. For that reason,
`hetero_train_2v2_mav_attack.yaml` keeps the actual scenario at 2v2 but pads
`max_red_agents` and `max_blue_agents` to 3.
