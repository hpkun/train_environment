<!-- Copilot / AI agent instructions for CloseAirCombat -->

# Repository intent
This repository implements a lightweight, gym-wrapped aerial combat environment built on JSBSim and baseline RL algorithms (PPO, MAPPO). Use this file to help an AI agent become productive quickly: focus on env configs, training scripts, and the hierarchical control pattern used across tasks.

## Big-picture architecture (how components interact)
- Environments: `envs/JSBSim/` contains the JSBSim integration, task configs and env implementations. Tasks are defined in `envs/JSBSim/configs` and loaded by the envs under `envs/JSBSim/envs/`.
- RL algorithms: `algorithms/ppo` and `algorithms/mappo` implement policies, actor/critic, and trainer logic. Self-play and shared-weights variants live under `algorithms/mappo` and `algorithms/ppo` subfolders.
- Runners & orchestration: `runner/` contains training/inference runners (`jsbsim_runner.py`, `selfplay_jsbsim_runner.py`, `share_jsbsim_runner.py`). The bash scripts in `scripts/` call these with common flags.
- Rendering: `renders/` produces TacView `.acmi` files (`renders/render_1v1.py`). Real-time telemetry is supported via TacView Advanced (see README instructions).

Key data flow: training scripts → runner → envs (JSBSim sim) → algorithms (policy/actor) → optional rendering/evaluation. `config.py` centralizes CLI/default parameters.

## Project-specific workflows & commands
- Environment setup (recommended):
  - `conda create -n jsbsim python=3.8`
  - `pip install torch pymap3d jsbsim==1.1.6 geographiclib gym==0.20.0 wandb icecream setproctitle` (see README for full list)
  - Initialize submodule: `git submodule init && git submodule update` (JSBSim submodule required)
- Quick training: run one of the provided scripts in `scripts/`:
  - `bash scripts/train_heading.sh` (SingleControl)
  - `bash scripts/train_selfplay.sh` (SingleCombat self-play)
  - `bash scripts/train_vsbaseline.sh` (vs-baseline)
  - `bash scripts/train_selfplay_shoot.sh` (shoot missile tasks)
  - `bash scripts/train_share_selfplay.sh` (MultipleCombat)
- Example train flags the agent should know to pass or mutate:
  - `--env-name` ∈ ['SingleControl','SingleCombat','MultipleCombat']
  - `--scenario` → YAML files under `envs/JSBSim/configs`
  - `--algorithm` ∈ [ppo,mappo]
  - Self-play: `--use-selfplay`, `--selfplay-algorithm`, `--n-choose-opponents`
  - Eval/render: `--use-eval`, `--n-eval-rollout-threads`, `--eval-interval`, `--render-mode` (e.g. `real_time`)
  - Missile shoot prior: `--use-prior` (used for parameterized shooting tasks)

## Conventions & patterns to follow
- Hierarchical control: many combat tasks use a two-level approach — an upper-level policy outputs high-level commands (heading, altitude, velocity) while the low-level controller/policy (trained under `SingleControl`) executes those commands. See `scripts/train_heading.sh` and `envs/JSBSim` for examples.
- Config-driven scenarios: scenario YAMLs in `envs/JSBSim/configs` determine the task parameters. When proposing changes, prefer adding or modifying YAMLs for new scenarios instead of hardcoding values.
- Algorithm placement: actor/policy/critic/trainer files are colocated under `algorithms/ppo` and `algorithms/mappo`. Mirror existing file patterns when adding new algorithm experiments.
- Logging/experiment tracking: code uses `wandb` by default when `--use-wandb` is supplied. Keep `--wandb-name` meaningful for user identification.

## Important files to inspect when implementing features or fixes
- Project config and CLI defaults: `config.py`
- Scenario definitions: `envs/JSBSim/configs/`
- JSBSim glue and envs: `envs/JSBSim/core/`, `envs/JSBSim/envs/` and `envs/env_wrappers.py`
- Training runners: `runner/jsbsim_runner.py`, `runner/selfplay_jsbsim_runner.py`, `runner/share_jsbsim_runner.py`
- Algorithms: `algorithms/ppo/*`, `algorithms/mappo/*`
- Scripts: `scripts/train_*.sh` (entry points used by most users)
- Rendering helpers: `renders/render_1v1.py`, `renders/render_2v2.py`
- Tests: `tests/test_ppo.py`, `tests/test_jsbsim.py`

## Integration points & external dependencies
- JSBSim binary & Python bindings: repo expects a working JSBSim (submodule + build). If JSBSim is not available the envs will fail to instantiate.
- TacView Advanced: optional, for real-time telemetry. Real-time rendering requires `--render-mode real_time` and TacView configured with IP:port from console.
- Python packages: specific versions of `jsbsim`, `gym==0.20.0` and PyTorch are used; prefer reproducing the environment as described in README.

## Typical agent tasks and quick examples
- Add a new scenario: add a YAML to `envs/JSBSim/configs` and reference it with `--scenario`.
- Run a short local eval (no wandb):
  - `bash scripts/train_selfplay.sh --use-eval --eval-interval 1 --use-wandb false`
- Render an existing saved run to ACMI: run `python renders/render_1v1.py` and open result with TacView.

## When to run tests / debugging tips
- Use `tests/test_jsbsim.py` to validate JSBSim integration. If tests fail, confirm JSBSim submodule/build and PYTHONPATH.
- Common failure: missing JSBSim binary or wrong Python version. Follow README environment setup.

---
If anything here is unclear or you want more detail (examples of CLI flag combinations, common debug errors, or a condensed developer checklist), tell me which section to expand and I will iterate.
