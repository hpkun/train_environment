# Parallel Env Stall Debug Plan

## Current Symptom

The env1 10M formal run is the stable background experiment and should not be
stopped or overwritten by parallel-env debugging.

The 4-env run with the same HAPPO reference training command progressed beyond
short smoke tests but stalled during a longer run around 487k environment steps.
The process remained alive, CPU time stopped increasing, `train_log.csv` and
`heartbeat.log` stopped updating, and `eval_log.csv` was still empty. This
indicates a rollout-side stall before the first 500k evaluation, not an eval
subprocess or checkpoint issue.

## Excluded Causes

- Not stdout buffering: CSV and heartbeat files also stopped.
- Not the first online evaluation: the run stopped before 500k steps.
- Not NaN: training logs reported `nan_detected=0`.
- Not checkpoint saving: the last completed iteration was already logged.
- Not rich logging alone: heartbeat stopped too.

## Suspected Cause

The likely failure mode is single-process synchronous 4-env sampling: one JSBSim
env blocks in `policy_act`, opponent action, `env.step`, or `reset`, and the
whole rollout stops. The earlier failed command also omitted `--max-steps 1000`,
so it used the training script default `max_steps=64`, causing very frequent
environment resets. High reset frequency is considered high-risk for long
JSBSim multi-instance runs.

## Debug Heartbeat Design

`--debug-rollout-heartbeat` records every transition event instead of sampling
every N steps. Each line records wall time, iteration, rollout local step,
env index, event, total env steps, episode step/id, alive counts, missile count,
sim time, `max_steps`, and `num_envs`.

The stall watchdog can be enabled with:

```powershell
--heartbeat-stall-timeout-sec 300 --exit-on-heartbeat-stall
```

On timeout it writes:

- `heartbeat_stall_report.json`
- `heartbeat_stall_report.md`
- `heartbeat_stall_stack.txt`

## max_steps=64 vs max_steps=1000

`max_steps=64` is a smoke-test setting and should not be used for long
air-combat training. It resets each env about 15.6x more often than
`max_steps=1000`. Long 4-env stability tests must use `--max-steps 1000`.

## Process-Isolated Env

If `num_envs=4, max_steps=1000` still stalls with full heartbeat diagnostics,
the next engineering direction should be process-isolated environments. That is
not implemented in this round because the current goal is diagnosis and minimal
risk reduction, not a new parallel environment backend.
