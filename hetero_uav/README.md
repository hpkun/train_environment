# hetero_uav

`hetero_uav` is a self-contained debug environment project for heterogeneous
MAV-UAV cooperative air combat. It is intended as the environment foundation for
later heterogeneous zero-shot composition generalization research.

The design is informed by Chen et al. 2026: MAVs and UAVs are represented as
autonomous agents, heterogeneity is expressed through platform type, sensing
range, missile load, maneuver limits, and reward role, and the environment
exposes per-agent observations plus a centralized global state for CTDE methods.

This first version is an environment skeleton and debug implementation only. It
does not implement type-aware attention, a mask generator, HAPPO, MAPPO, BRMA-
MAPPO changes, or paper experiments.

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

The current flight dynamics model is a simplified kinematic skeleton, not a
complete JSBSim six-degree-of-freedom model. The code is organized under
`uav_env/JSBSim/` so that the proxy aircraft, missile, and control components
can later be replaced by real JSBSim aircraft models, PID control, and
proportional-navigation missile dynamics without changing the public environment
API.

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

## Parent Project Dependency

There is no runtime dependency on the parent `train_environment` project. The
code does not import `my_uav_env`, `envs.JSBSim`, or modify `sys.path` to reach
the parent directory.

The current limitation is fidelity, not independence: JSBSim aircraft XML files,
PID control, Tacview export, and full proportional-navigation missile dynamics
are not migrated yet. They can be added later inside `uav_env/JSBSim/` by
replacing the proxy classes in `core/aircraft.py` and `core/missile.py`.

## Tests

```bash
pytest tests/test_env_smoke.py
```

The smoke test checks import, config loading, reset, one step, observation/state
shape, and required `info` fields.
