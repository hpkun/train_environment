# num_envs Semantics and Long-Run Guidance

This note audits the current `hetero_uav` HAPPO training entrypoints and
contrasts them with the parent `brmamappo` training script. It documents the
runner split after disabling misleading serial `--num-envs` behavior in the
legacy single-process entrypoint.

## Scope

Audited files:

- `hetero_uav/scripts/train_happo_reference.py`
- parent project `train_ppo.py`

The current long-running `full_10m_normal_geometry_max1000_env1` experiment
should not be stopped or overwritten by this audit.

## hetero_uav train_happo_reference.py

`scripts/train_happo_reference.py` is now the single-process runner. It rejects
`--num-envs > 1` with a clear error. The old serial multi-env rollout batching
path is disabled because it looked like parallel rollout threads while actually
stepping all environments in one Python process.

Use this entrypoint for stable single-env runs:

```text
python -u scripts/train_happo_reference.py --num-envs 1 ...
```

Use `scripts/train_happo_reference_parallel.py` for true multiprocessing rollout
workers.

### Environment creation

`train_happo_reference.py` creates environments with a Python list:

```python
envs = [
    make_env(args.config, env_type="jsbsim_hetero",
             hetero_reward_mode=args.reward_mode, max_steps=args.max_steps)
    for _ in range(args.num_envs)
]
```

The legacy implementation created these environments in the same Python
process. There was no `multiprocessing`, no worker process, no pipe-based remote
step, and no process-level timeout per environment. That behavior is no longer
available through the formal runner.

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

This was serial rollout batching, not true parallel environment stepping. It is
kept here as historical context for why the old option is disabled.

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

In the old implementation, increasing `--num-envs` changed the rollout batch
size and the number of independent environment instances sampled between PPO
updates. It did not provide subprocess parallelism or near-linear wall-clock
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

## New hetero_uav train_happo_reference_parallel.py

`scripts/train_happo_reference_parallel.py` is the new true parallel entrypoint.
It uses multiprocessing workers:

- one child process creates one `HeteroUavCombatEnv`;
- the main process communicates with workers through `Pipe`;
- `--num-envs N` means `N` worker subprocesses;
- reset and step have explicit timeouts;
- a timed-out worker is terminated and restarted, and the current rollout aborts
  instead of silently mixing corrupted samples;
- recurrent hidden state is indexed by `env_idx`;
- per-env rich logs are written under `rich_logs/env_XX/` when rich logging is
  enabled, avoiding multiple workers writing the same CSV.

This runner is the one that corresponds to true rollout threads. It is also the
right place to continue future process-isolation work such as worker restart and
resume.

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

## BRMA-MAPPO rollout thread alignment

The BRMA-MAPPO paper uses multiple rollout threads, reported as
`Rollout_threads=32`. The old `train_happo_reference.py --num-envs 4` did not
match that idea because it was serial batching. The new parallel runner is the
closest implementation path in this project for paper-style rollout threads,
although exact parity with the parent BRMA implementation still requires careful
validation.

## Impact on speed and stability

### Speed

For old `train_happo_reference.py`, `--num-envs 4` should not be interpreted as
four CPU/GPU workers running concurrently. It was more accurately:

> collect four environment streams serially before updating.

If JSBSim stepping is the bottleneck, serial `num_envs=4` can be slower per log
line and may not improve total wall-clock throughput. GPU acceleration mainly
helps network forward/update; it does not make serial JSBSim stepping parallel.

### Stability

The parent `SubprocVecEnv` can identify a worker that fails to respond to reset
or step. The old `hetero_uav` serial loop could not isolate a single env hang:
the entire trainer blocked inside the current call.

Current `hetero_uav` mitigations are heartbeat and stall-watchdog logging. These
help locate the last stage before a stall, but they do not provide process
isolation or automatic worker restart.

### 10M long training

For the legacy single-process runner, `--num-envs 1` is the only supported mode:

- It keeps iteration cadence predictable.
- It minimizes JSBSim instances in one process.
- It reduces the chance that one of several serial envs blocks the whole run.
- It matches the current stable long-run recommendation.

For true rollout workers, use `train_happo_reference_parallel.py`. Start with a
small smoke and 2 workers before scaling higher, because each worker owns a
JSBSim environment process.

## Recommendation

### For current 10M experiments

Use for the legacy runner:

```text
--num-envs 1
```

For parallel long runs, use `scripts/train_happo_reference_parallel.py` with
worker timeouts, heartbeat logging, and frequent checkpoints/eval checkpoints.
The long-run command should also use unbuffered Python (`python -u`) so console
logs remain visible.

### Watchdog and resume

Long runs should use:

- heartbeat logging;
- stall watchdog when available;
- frequent checkpoints or eval checkpoints;
- a resume plan that can continue from a saved checkpoint if a process stalls.

This is more important for current experiments than adding true parallelism.

### True parallel envs

The initial multiprocessing runner now exists, but it should still be treated as
new infrastructure that needs staged validation before replacing stable
single-env long runs.

Required work would include:

- process-isolated `HeteroUavCombatEnv` creation;
- Windows `spawn` compatibility;
- JSBSim startup staggering;
- worker reset and step timeouts;
- robust worker shutdown/restart;
- serialization of observations, info dicts, and diagnostics;
- recurrent hidden-state handling across worker boundaries;
- compatibility with rich logging, heartbeat, checkpoint, and eval paths.

Current recommendation: use the new parallel runner only after small smoke tests
confirm worker stability for the target config and policy architecture.

## Summary

| Item | hetero_uav train_happo_reference.py | parent train_ppo.py |
| --- | --- | --- |
| `num_envs` meaning | fixed to `1`; old serial batching disabled | subprocess vectorized environments |
| Parallel stepping | no | yes |
| Env creation | one env object in one process | one worker process per env |
| Step timeout per env | no process-level timeout | `step(timeout=60s)` |
| Reset timeout per env | no process-level timeout | `reset(timeout=300s)` |
| Step accounting | `total_steps += 1` per serial env step | `total_steps += num_envs` per vector step |
| Rollout size | `rollout_length * num_envs` | `num_steps * num_envs` |
| PPO update timing | after serial batch collection | after vectorized rollout collection |
| Long-run recommendation | `num_envs=1` only | true parallelism is built in |

| Item | new train_happo_reference_parallel.py |
| --- | --- |
| `num_envs` meaning | true worker subprocess count |
| Parallel stepping | yes, one child process per env |
| Env creation | worker-local `HeteroUavCombatEnv` |
| Step timeout per env | yes |
| Reset timeout per env | yes |
| Rich logging | per-env directories plus main aggregate metrics |
| Recommended use | staged smoke, then controlled long runs |
