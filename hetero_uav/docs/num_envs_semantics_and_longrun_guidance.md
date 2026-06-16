# num_envs Semantics and Long-Run Guidance

This note audits the current `hetero_uav` HAPPO training entrypoint and contrasts
it with the parent `brmamappo` training script. It is documentation only: no
training logic, environment behavior, reward, missile, PID, blue rule, action
space, or observation dimension is changed.

## Scope

Audited files:

- `hetero_uav/scripts/train_happo_reference.py`
- parent project `train_ppo.py`

The current long-running `full_10m_normal_geometry_max1000_env1` experiment
should not be stopped or overwritten by this audit.

## hetero_uav train_happo_reference.py

### Environment creation

`train_happo_reference.py` creates environments with a Python list:

```python
envs = [
    make_env(args.config, env_type="jsbsim_hetero",
             hetero_reward_mode=args.reward_mode, max_steps=args.max_steps)
    for _ in range(args.num_envs)
]
```

These environments live in the same Python process. There is no
`multiprocessing`, no worker process, no pipe-based remote step, and no
process-level timeout per environment.

### Step execution

Rollout collection iterates through `envs` in a normal Python loop:

```python
while len(buffer) < rollout_transitions and total_steps < args.total_env_steps:
    for env_idx, rollout_env in enumerate(envs):
        ...
        next_obs, rewards, terminated, truncated, next_info = rollout_env.step(action_dict)
        ...
        total_steps += 1
```

This means `--num-envs` in `hetero_uav` is serial rollout batching, not true
parallel environment stepping.

### Step accounting

The configured rollout transition count is:

```python
transitions_per_rollout = rollout_length * num_envs
```

Each individual `rollout_env.step(...)` increments `total_steps` by `1`.
Therefore:

- `rollout_length=256, num_envs=1` collects 256 transitions before one PPO update.
- `rollout_length=256, num_envs=4` collects 1024 transitions before one PPO update.
- Those 1024 transitions are collected serially in one process, not as four
  parallel 256-step workers.

The buffer size for a rollout is `rollout_length * num_envs`, and PPO update is
called after that serial batch has been collected:

```python
buffer = HAPPORolloutBuffer(rollout_transitions, ...)
...
stats = trainer.update(buffer, ...)
```

### Practical meaning

In the current implementation, increasing `--num-envs` changes the rollout batch
size and the number of independent environment instances sampled between PPO
updates. It does not provide subprocess parallelism or near-linear wall-clock
speedup.

Potential benefits:

- More diverse rollout data per PPO update.
- Fewer optimizer updates for the same total environment-step budget.

Costs and risks:

- Wall-clock collection remains serial, so speedup should not be expected.
- A single hung `env.step`, opponent action, reset, or JSBSim call blocks the
  whole training process.
- Larger rollout batches make one iteration longer, so logs and eval triggers
  arrive less frequently in wall-clock time.
- With recurrent policies, larger serial batches also increase the amount of
  hidden-state bookkeeping per update, even though it is still a single process.

## Parent train_ppo.py

The parent `train_ppo.py` uses true subprocess vectorization.

Evidence from the script:

- Imports `multiprocessing as mp`.
- Sets thread caps before and after imports:
  - `OMP_NUM_THREADS=1`
  - `MKL_NUM_THREADS=1`
  - `NUMEXPR_NUM_THREADS=1`
  - `torch.set_num_threads(1)`
- The default `Config.num_envs` is `8`.
- `SubprocVecEnv` creates worker processes with `mp.get_context("spawn")`,
  `ctx.Pipe()`, and `ctx.Process(...)`.
- `reset()` sends reset commands to each worker and waits with a `300s` timeout.
- `step()` sends one action dict per worker and waits with a `60s` timeout.
- The parent loop calls `vec_env.step(actions_list)`, receives batched results,
  and then increments `total_steps += config.num_envs`.

So the parent project has true multiprocessing parallelism and process-level
worker timeout handling. `hetero_uav` currently does not.

## Impact on speed and stability

### Speed

For `hetero_uav`, `--num-envs 4` should not be interpreted as four CPU/GPU
workers running concurrently. It is more accurately:

> collect four environment streams serially before updating.

If JSBSim stepping is the bottleneck, serial `num_envs=4` can be slower per log
line and may not improve total wall-clock throughput. GPU acceleration mainly
helps network forward/update; it does not make serial JSBSim stepping parallel.

### Stability

The parent `SubprocVecEnv` can identify a worker that fails to respond to reset
or step. The current `hetero_uav` serial loop cannot isolate a single env hang:
the entire trainer blocks inside the current call.

Current `hetero_uav` mitigations are heartbeat and stall-watchdog logging. These
help locate the last stage before a stall, but they do not provide process
isolation or automatic worker restart.

### 10M long training

For the current 10M long run path, `--num-envs 1` is the safer default:

- It keeps iteration cadence predictable.
- It minimizes JSBSim instances in one process.
- It reduces the chance that one of several serial envs blocks the whole run.
- It matches the current stable long-run recommendation.

`--num-envs 4` can still be used as a controlled batching experiment, but it
should not be expected to make the current implementation four times faster.

## Recommendation

### For current 10M experiments

Use:

```text
--num-envs 1
```

Keep heartbeat logging enabled for long runs, and prefer periodic eval/checkpoint
saving. The long-run command should also use unbuffered Python (`python -u`) so
console logs remain visible.

### Watchdog and resume

Long runs should use:

- heartbeat logging;
- stall watchdog when available;
- frequent checkpoints or eval checkpoints;
- a resume plan that can continue from a saved checkpoint if a process stalls.

This is more important for current experiments than adding true parallelism.

### True parallel envs

Porting parent-style `SubprocVecEnv` may be useful later, but it should be
treated as future engineering work, not a small parameter tweak.

Required work would include:

- process-isolated `HeteroUavCombatEnv` creation;
- Windows `spawn` compatibility;
- JSBSim startup staggering;
- worker reset and step timeouts;
- robust worker shutdown/restart;
- serialization of observations, info dicts, and diagnostics;
- recurrent hidden-state handling across worker boundaries;
- compatibility with rich logging, heartbeat, checkpoint, and eval paths.

Current recommendation: do not port `SubprocVecEnv` until the current long-run
baseline and paper experiment path are stable enough to justify the engineering
cost.

## Summary

| Item | hetero_uav train_happo_reference.py | parent train_ppo.py |
| --- | --- | --- |
| `num_envs` meaning | serial rollout batching | subprocess vectorized environments |
| Parallel stepping | no | yes |
| Env creation | list of env objects in one process | one worker process per env |
| Step timeout per env | no process-level timeout | `step(timeout=60s)` |
| Reset timeout per env | no process-level timeout | `reset(timeout=300s)` |
| Step accounting | `total_steps += 1` per serial env step | `total_steps += num_envs` per vector step |
| Rollout size | `rollout_length * num_envs` | `num_steps * num_envs` |
| PPO update timing | after serial batch collection | after vectorized rollout collection |
| Long-run recommendation | `num_envs=1` for stability | true parallelism is built in |

