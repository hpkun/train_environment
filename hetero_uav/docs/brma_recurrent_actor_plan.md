# BRMA Recurrent Actor Plan

## Paper module

BRMA-MAPPO uses a GRU-based recurrent actor to capture temporal dependencies across
sequential decision steps (paper Section 3.3-3.4). The Actor applies:

```
EntityObservationEncoder → GRUCell → action_head → Normal(μ, σ)
```

The Critic applies the same pattern with an independent GRU and encoder.

## Parent project implementation

| Component | File | Line |
|---|---|---|
| Actor with GRU | `algorithm/mappo_nets.py` | L19-91 |
| Critic with GRU | `algorithm/mappo_nets.py` | L96-154 |
| Entity encoder | `attention_models.py` | L14-99 |
| SharedReplayBuffer | `algorithms/utils/buffer.py` | (parent repo) |
| Hidden state init | `torch.zeros(B, rnn_hidden_size)` | L200-201 |
| Done reset | `rnn_states[dones==True] = 0` | runner `jsbsim_runner.py:127` |
| Buffer stores | `rnn_states_actor` and `rnn_states_critic` | buffer arrays |

## Current hetero_uav minimal adaptation

### Scope

- Actor-only GRU (critic stays 480-dim MLP, per requirement "hidden state 不进入 critic")
- One-step GRU state replay (smoke version, not full TBPTT)
- New `policy_arch = "brma_recurrent"`

### Architecture

```
flat_obs (96-dim)
  → _flat_to_entities() → entity tensor [B, N, 19]
  → BRMAEntityObservationEncoder → pooled [B, 256]
  → nn.GRUCell(256, 128) → rnn_hidden [B, 128]   ← NEW
  → MAV actor head (128→3) / UAV actor head (128→3)
  → Normal(μ, σ)
```

### Why no mask generator

The mask generator (MaskVectorGenerator) is excluded per P3a scope constraint #5/#6.
It will be added in a later phase with random scale mask and biased random mask.

### Hidden state lifecycle

1. **Init**: `torch.zeros(num_red, rnn_hidden_size)` per env at rollout start
2. **Step**: `rnn_hidden_new = grucell(pooled, rnn_hidden_old)` — old state passed to `act()`, new state returned
3. **Done reset**: When episode done for an env, zero out hidden state for all agents in that env
4. **Buffer**: Store `rnn_hidden` (before step) in buffer for PPO replay
5. **PPO update**: For each transition, pass stored `rnn_hidden_old` to `evaluate_actions()` which runs one-step GRU forward

### Smoke-version limitations (honest disclosure)

This is a **smoke-level GRU integration**, not a full recurrent PPO:

1. One-step GRU replay: during PPO update, each timestep is replayed independently with its
   stored hidden state. There is no TBPTT gradient flow across timesteps.
2. The critic does not use GRU (stays 480-dim MLP), unlike the parent project's full GRU critic.
3. Hidden state is stored per-agent in the buffer, not compressed or truncated.
4. This is sufficient for rollout/update/eval to function correctly, but does not claim
   full recurrent PPO optimization benefits.

## File changes

### Modified files

1. `scripts/train_happo_reference.py`:
   - Add `--policy-arch brma_recurrent` option
   - `_build_policy()` handles brma_recurrent
   - Rollout loop maintains `rnn_hidden` per env per agent
   - Done reset zeros hidden state
   - Buffer stores/retrieves rnn_hidden
   - Trainer passes rnn_hidden to evaluate_actions

2. `algorithms/happo/happo_buffer.py`:
   - Add `rnn_hidden` array storage
   - `store()` accepts rnn_hidden
   - `get()` returns rnn_hidden

3. `algorithms/happo/happo_trainer.py`:
   - `update()` passes rnn_hidden to evaluate_actions when present

### New files

4. `algorithms/happo/brma_recurrent_policy.py`:
   - `BRMARecurrentHAPPOReferencePolicy` class

5. `scripts/run_brma_recurrent_smoke.py`:
   - Smoke runner for brma_recurrent

6. `tests/test_brma_recurrent_policy.py`:
   - Unit tests for recurrent policy

### Unchanged files

- `algorithms/happo/brma_entity_policy.py` — non-recurrent path unchanged
- `algorithms/happo/happo_policy.py` — flat path unchanged
- `uav_env/` — no env changes
