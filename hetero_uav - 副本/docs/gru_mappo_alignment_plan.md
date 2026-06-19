# GRU MAPPO Alignment Plan

## Why Not Keep Adding Steps To Shared MLP

The protocol-aligned shared MLP MAPPO runs show that longer training is not a
reliable fix by itself. In the failed longer runs, red alive counts stayed near
zero, blue win rate stayed near one, MAV survival stayed at zero, and the latest
checkpoint did not retain the short-window improvements seen in smaller pilots.

This points to a method-capacity issue for the current partially observable
heterogeneous 3v2 setting. A feed-forward shared MLP sees only the current
flattened observation and cannot retain target history, missile-warning history,
or role-dependent temporal context.

## Why GRU Before Attention

The parent project training protocol is recurrent before it is attention-heavy:

- `train_vanilla_mappo.py` uses a flattened observation encoder followed by a
  GRU actor.
- `train_ppo.py` uses attention and masking, but it also keeps recurrent actor
  and critic hidden states.

Therefore GRU-MLP is the smaller next step than entity attention. It tests
whether memory alone improves learning while keeping the current observation,
reward, action space, missile logic, and PID stack fixed.

## Parent Vanilla GRU Points

- Actor: flattened observation -> MLP encoder -> `GRUCell` -> MLP action head.
- Critic: centralized feed-forward MLP over global red observation/state.
- Hidden size: 128.
- Layers: effectively one recurrent cell.
- Actor and critic do not share GRU state.
- Rollout stores initial and final actor hidden state.
- PPO update rebuilds per-env, per-agent sequences and unrolls actor GRU from
  the rollout initial hidden state.
- Done resets actor hidden state during collection.

## Parent Attention GRU Points

- Actor uses entity attention/masking plus recurrent hidden state.
- Critic path stores recurrent hidden state in the attention PPO script.
- Rollout stores actor and critic initial/final hidden states.
- PPO update runs sequence unrolls and computes actor, critic, and mask losses.
- This is larger than the minimal next step for `hetero_uav`.

## Current hetero_uav Gaps

- Current `MAPPOActorCritic` is feed-forward.
- Current `RolloutBuffer` stores tensors by `(T, num_red, dim)` but no hidden
  state.
- Current PPO trainer flattens actor observations with `actor_obs.view(-1, dim)`.
- Current update assumes stateless policy evaluation.
- The alive mask and team-done fixes are correct and should be preserved.
- Dead-agent active masks must also reset or mask recurrent hidden state.

## Minimal GRU-MLP Boundary

The minimal implementation should add recurrence only:

- actor encoder: `96 -> 256 -> Tanh`
- actor GRU: hidden size 128, one layer
- actor head: `128 -> action_dim`
- critic encoder: `480 -> 256 -> Tanh`
- critic GRU: hidden size 128, one layer
- critic head: `128 -> value`
- learnable `action_log_std` matching the current MAPPO baseline

It should not change:

- reward;
- termination;
- missile launch or missile dynamics;
- action space;
- PID;
- aircraft XML;
- observation dimension;
- attention;
- HAPPO.

## Required Code Changes

- Add `algorithms/mappo/recurrent_policy.py` for `GRUMAPPOActorCritic`.
- Add `algorithms/mappo/recurrent_buffer.py` with sequence storage and initial
  hidden states.
- Add `algorithms/mappo/recurrent_trainer.py` with full-rollout sequence PPO
  before any minibatch sequence slicing.
- Update train/eval runners to maintain hidden state and reset it on episode
  done or inactive/dead agents.
- Extend model selection with `actor_arch="gru_mlp"` while keeping dimensions
  unchanged.

## Why This Round Does Not Implement GRU

Correct recurrent MAPPO is a coordinated change across policy, storage, trainer,
runner, eval, save/load, and tests. A partial implementation that simply adds a
GRU module while keeping flat-batch PPO would not match the parent recurrent
protocol and could produce misleading results.

This round therefore records the alignment audit and minimal implementation
plan only.

## Smoke And Pilot Recommendation

Before any 200k pilot, run a 64-step GRU smoke path that verifies:

- forward shapes;
- hidden state shapes;
- hidden reset behavior;
- train + eval + save/load;
- metadata reports `actor_arch == "gru_mlp"`;
- no NaN.

Only after that smoke path passes should a GRU-MLP 200k pilot be considered.
