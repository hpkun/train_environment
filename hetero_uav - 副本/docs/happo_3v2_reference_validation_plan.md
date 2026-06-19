# HAPPO 3v2 Reference Validation Plan

## Purpose

This project is a zero-shot heterogeneous MAV/UAV cooperative air-combat
experiment. It is not a full air-combat engineering system and it is not a
low-level flight-control reinforcement-learning project.

The next validation path is HAPPO-style 3v2 reference validation: check whether
a paper-informed heterogeneous policy setup can produce reasonable 3v2 combat
behavior before running larger 5v4 transfer experiments.

## Why Shared MLP Is Not Enough

The current shared MLP MAPPO baseline is runnable, but the 1M best checkpoint
does not prove combat effectiveness. The 100-episode re-evaluation is dominated
by timeout outcomes, MAV survival remains zero, and blue elimination remains
zero.

Timeout survival should not be reported as successful air combat. A valid
environment-method pair should show MAV support, UAV engagement, missile hits,
blue deaths, and non-timeout combat outcomes.

## Why HAPPO 3v2 Is The Right Validation Step

The heterogeneous MAV/UAV paper is mainly useful here as a method and protocol
reference: MAV and UAV have different roles, observations should support
cooperation, and the method should handle heterogeneous agents. HAPPO-style
separate actors are a cleaner test than a single shared MLP actor because the
MAV support role and UAV attack role should not be forced through one identical
policy head.

This is still only a reference validation step, not full TAM-HAPPO.

## Paper-Grounded Boundary

HAPPO reference v0 is not a TAM-HAPPO reproduction. It is an environment
validation path for checking whether heterogeneous MAV/UAV policies can produce
reasonable 3v2 behavior in the current codebase.

If GRU is not implemented, the result is only a no-temporal HAPPO ablation. It
must not be described as having the paper's temporal state-memory module.

If multi-head attention is not implemented in the centralized value network,
the result cannot claim an attention-enhanced value network. It is a
non-attention HAPPO-style baseline.

Because the environment keeps high-level `[pitch, heading, speed]` with PID, it
does not reproduce the paper action space `[Ct, Ca, Ce, Cr]`. The paper action
uses throttle, aileron, elevator, and rudder controls, with `Ct` in `[0.4, 0.9]`,
`Ca/Ce/Cr` in `[-1, 1]`, and 40 discrete levels per dimension.

Missile launch and evasion remain scripted environment mechanics. This is a
local engineering choice for the current environment, not a paper-consistent
implementation of learned missile-aware maneuvering.

Therefore HAPPO reference v0 is for environment validation and controlled
ablation evidence only. It should not be used as a paper result reproduction.

## Similarities With The Heterogeneous Paper

- 3v2 setup: one MAV with two UAVs against two UAV opponents.
- MAV is a support agent and should not be the main missile shooter.
- UAVs are the attack agents.
- The key experimental question is heterogeneous composition and later
  zero-shot transfer.
- The method direction is role-aware or heterogeneous-policy learning.

## Differences From The Heterogeneous Paper

- Current action interface remains high-level `[pitch, heading, speed]` with
  PID, not low-level throttle/aileron/elevator/rudder.
- Current missile launch and evasion remain BRMA-style scripted mechanics.
- Current first-stage plan does not include temporal attention or full
  TAM-HAPPO.
- Current aircraft are engineering approximations: F-22 MAV and F-16 UAVs.
- Current first-stage reward should stay `brma_legacy` unless a separate
  reference reward is explicitly implemented and audited.

These differences are acceptable for the next validation step because changing
action interface, missile mechanics, and algorithm at the same time would make
the result hard to interpret.

## Current Implementation Status

Earlier work added HAPPO reference v0 as a minimal runnable validation path:

- `happo_ref_v0` reward mode is available by explicit config or CLI choice;
- MAV and UAV use separate actors;
- the critic is centralized;
- actor updates run sequentially by role;
- train, eval, smoke, 200k runner, and 1M runner scripts exist.

The implementation remains a reference validation baseline. It is not full
TAM-HAPPO.

## happo_ref_v0 Reward Components

The reward mode is an additive role overlay on top of the existing BRMA reward.
It is enabled only when `hetero_reward_mode: happo_ref_v0` or
`--reward-mode happo_ref_v0` is used.

MAV components:

- `mav_survival`;
- `mav_support`;
- `mav_attack` fixed at zero;
- `mav_dodge` fixed at zero;
- `event`;
- `safety`;
- `death_penalty`.

UAV components:

- `uav_attack_window`;
- `uav_fire`;
- `uav_hit`;
- `uav_dodge`;
- `event`;
- `safety`;
- `death_penalty`.

Unavailable components are recorded as zero rather than raising errors, so the
reward can be audited from `info["reward_components"]`.

## HAPPO Reference v0 Scope

HAPPO reference v0 should be the smallest coherent method change:

- separate MAV actor;
- separate UAV actor;
- centralized critic;
- sequential HAPPO-style actor update;
- continuous 3D Gaussian action output;
- active-agent mask and team-done logic preserved;
- no attention in the first stage;
- no GRU or temporal module in the first stage;
- no observation-dimension change in the first stage.

The first smoke test should only verify that this setup can collect rollout,
update once, save, load, and evaluate without NaN.

The sequential update is a simplified HAPPO-style v0 role-wise PPO update. It
keeps separate MAV/UAV update phases but does not implement the full strict
HAPPO correction-factor machinery.

## Reward Position

`happo_ref_v0` reward exists from earlier implementation work, but this
paper-grounded review does not validate it as exact paper reward reproduction.

The current recommended first HAPPO validation should keep `brma_legacy` to
avoid mixing method validation with reward redesign. A role reward can be
revisited only after the HAPPO smoke path is stable and after each paper reward
module is mapped to available observations.

## Success Criteria

HAPPO 3v2 validation should not be judged by timeout red-win rate alone. The
minimum useful signals are:

- MAV does not always die early;
- UAVs create attack windows and fire effective missiles;
- blue deaths are not always zero;
- episodes are not all timeout draws;
- combat deaths dominate over crash or control anomalies;
- ACMI shows MAV supporting while UAVs engage.

## Training Protocol

1. Run HAPPO reference v0 smoke only.
2. If smoke passes, run:

   `python scripts/run_happo_3v2_reference_200k.py`

3. If the 200k run shows MAV survival, UAV engagement, and non-timeout combat,
   then consider a 1M run and 5v4 zero-shot transfer.

## 200k Success Criteria

- MAV survival is clearly above the shared MLP baseline.
- `blue_dead_mean > 0`.
- `red_missile_hits_mean > 0`.
- Episodes are not all timeout draws.
- ACMI, if exported later, shows different MAV/UAV roles.

## 200k Result Summary

The 200k HAPPO reference v0 run completed and produced `latest` and `best`
checkpoints under `outputs/happo_3v2_reference_200k`.

The training log stayed in timeout-survival behavior: the latest train row has
`red_win=1.0`, `timeout=1.0`, `mav_survival=1.0`, `red_alive_final=3.0`,
`blue_alive_final=2.0`, and zero missiles fired/hit. This should not be
reported as effective combat.

The 50-episode checkpoint re-evaluation is more diagnostic:

- best 3v2: `red_win_rate=0.0`, `blue_win_rate=0.92`,
  `mav_survival_rate=0.0`, `blue_dead_mean=0.22`,
  `red_missile_hits_mean=0.22`;
- best 5v4: `red_win_rate=0.0`, `blue_win_rate=1.0`,
  `mav_survival_rate=0.0`, `blue_dead_mean=0.56`,
  `red_missile_hits_mean=0.56`;
- latest 3v2: `red_win_rate=0.0`, `blue_win_rate=1.0`,
  `mav_survival_rate=0.0`, `blue_dead_mean=0.0`,
  `red_missile_hits_mean=0.0`;
- latest 5v4: `red_win_rate=0.0`, `blue_win_rate=0.98`,
  `mav_survival_rate=0.0`, `blue_dead_mean=0.14`,
  `red_missile_hits_mean=0.14`.

ACMI export succeeded for best/latest 3v2 episode 0. Both exported episodes end
in `blue_win_elimination`, and the MAV dies in both.

Decision: HAPPO v0 partially works as a runnable validation path, but it does
not validate the environment strongly enough for a 1M HAPPO reference run.
Reward/observation/targeting should be inspected before adding GRU, attention,
or full TAM-HAPPO.

The key unresolved contradiction is that the latest training row reports
timeout red-win survival, while deterministic checkpoint evaluation reports
blue elimination wins and zero MAV survival. This must be treated as a
train/eval consistency issue, not as evidence that the policy has learned a
valid timeout survival strategy.

The next validation gate is:

1. train/eval consistency audit;
2. deterministic versus stochastic checkpoint evaluation;
3. MAV failure-mode audit;
4. explicit 1M readiness decision.

Do not start a 1M HAPPO reference run until this gate is resolved.

The MAV failure gate currently blocks 1M. Death-event logging identifies
`Crash_LowAlt` as the dominant MAV death reason, fixed safe MAV action does
not produce survival, action scaling does not rescue the MAV, and available
blue missile metadata does not show MAV-target preference. The current primary
failure hypothesis is F-22 control or dynamics instability.

## Failure Triage Order

If 200k fails, inspect in this order:

1. reward component magnitudes;
2. blue target preference;
3. missile hit logic;
4. MAV dynamics;
5. observation sharing;
6. only then consider temporal modules or attention.

## Stop Doing

- Do not keep tuning shared MLP MAPPO as the main evidence path.
- Do not run more 500k or 1M jobs that only inspect the latest checkpoint.
- Do not switch now to low-level 4D control.
- Do not implement full TAM-HAPPO before HAPPO reference v0 is validated.
- Do not add more broad engineering audit scripts unless they directly answer
  the HAPPO 3v2 validation question.
