# Role-Conditioned MAPPO — Minimal Implementation Plan

## 1. Method Position

Role-conditioned MAPPO is the **minimal method extension** of the
current shared MLP MAPPO baseline.  It is **not** TAM-HAPPO, **not**
entity-attention, and **not** HAPPO sequential update.  It is a single
architectural change: replace one shared actor head with two
role-specific heads while keeping everything else identical.

| Aspect | Current baseline | Role-conditioned (this plan) | Entity-attention (future) |
|---|---|---|---|
| Actor | 1 shared MLP | shared encoder + 2 role heads | per-entity encoder + attention |
| Critic | flat MLP (480→256→128→1) | unchanged | unchanged |
| Observation | flat (96-dim) | unchanged | entity-structured |
| Training | shared PPO | unchanged | unchanged |
| Reward | brma_legacy | unchanged | unchanged |

## 2. Current Baseline

```
Actor:  96  → Linear(256) → Tanh → Linear(128) → Tanh → Linear(3)
Critic: 480 → Linear(256) → Tanh → Linear(128) → Tanh → Linear(1)

Shared action_log_std (3,) initialized to ln(0.3)

MAPPOActorCritic.forward(actor_obs, critic_state, deterministic):
  → actor_obs   [num_red, 96]
  → critic_state [1, 480]
  → action [num_red, 3], value scalar
```

MAV and UAV agents currently share a single actor MLP. The only
role information comes from the `ego_role` one-hot (4-dim) in the
96-dim flat observation.  The network must learn to extract role from
this flat vector.

## 3. Motivation

MAV and UAV have fundamentally different tasks:
- **MAV**: unarmed, provides situational awareness and shared tracks;
  should survive and position for observation
- **UAV**: armed (2 missiles), engages blue aircraft; should attack and kill

A fully shared MLP actor may struggle to form stable role
specialization, especially with only 50k–100k environment steps.  The
hypothesis is that providing **explicit architectural role separation**
will help each role learn its task faster and more stably.

## 4. Minimal Architecture

### Option A (Recommended): Shared Encoder + Role-Conditioned Heads

```
               ┌─→ MAV Actor Head (256→128→3) ─→ action_mav
actor_obs ─→   │
               └─→ UAV Actor Head (256→128→3) ─→ action_uav
```

1. **Shared encoder** (96→256→Tanh): extracts common features from the
   flat V2 observation
2. **MAV head** (256→128→Tanh→3): produces 3-dim Gaussian mean for MAV
3. **UAV head** (256→128→Tanh→3): produces 3-dim Gaussian mean for UAV
4. **Action log_std**: shared across both heads (same as baseline)
5. **Critic**: unchanged (480→256→128→1)

Forward path:
```python
features = self.shared_encoder(actor_obs)       # [N, 256]
role_mask_mav = (ego_role[:, 0] > 0.5)          # MAV indices
role_mask_uav = (ego_role[:, 0] < 0.5)          # UAV indices
mean = torch.zeros(N, 3)
mean[role_mask_mav]  = self.mav_head(features[role_mask_mav])
mean[role_mask_uav]  = self.uav_head(features[role_mask_uav])
```

Role identification uses `ego_role` (4-dim one-hot from observation
adapter V2), where index 0 = mav, index 1 = attack_uav.

### Why Not Two Separate Encoders?

A shared encoder with role-specific heads is preferred because:
- Lower parameter count (fewer weights to train)
- Shared features (flight dynamics, ego state, common geometry)
- Easier to compare with baseline (only the head changes)
- Less risk of overfitting on 50k steps

### Why Not Attention / HAPPO / GRU First?

- Entity attention is the next logical step, but it requires changing
  the observation pipeline from flat vectors to entity-structured input
- HAPPO sequential update changes the training loop, not just the
  network architecture
- GRU/temporal features require per-agent history buffers
- Each of these multiplies the implementation surface and debugging
  scope
- Role-conditioned actor is the smallest possible architectural change
  that directly tests the role-specialization hypothesis

When role-conditioned MAPPO shows improvement over the baseline:
1. First add entity-attention (structured observation + per-entity
   encoder + attention pooling)
2. Then add temporal features (GRU/LSTM per agent)
3. HAPPO sequential update is a separate training-loop change that can
   be evaluated independently

### Why Not Continue Tuning role_v1?

- role_v1 50k result was clearly weaker than brma_legacy (red_win=0.00
  vs 0.10 in 3v2)
- The reward scale mismatch (kill +8, death -10 vs per-step stability
  reward <1) caused critic explosion and action saturation
- Best checkpoint was iter=1 (128 steps) — training got worse over time
- Even if role_v1 could be tuned to work, it would not isolate the
  method contribution: the paper needs to answer "does role separation
  help?" not "can we tune rewards?"

## 5. Files To Modify (Future Implementation)

| File | Change |
|---|---|
| `algorithms/mappo/policy.py` | Add `RoleConditionedActorCritic` or extend `MAPPOActorCritic` with role heads |
| `scripts/train_mappo_baseline.py` | Instantiate role-conditioned model when a flag is set |
| `scripts/eval_mappo_zero_shot.py` | Load role-conditioned model correctly |
| `scripts/run_main_mappo_experiment.py` | May add `--role-conditioned` flag or new runner |

Optionally add a lightweight runner:
`scripts/run_main_mappo_role_conditioned_experiment.py`

## 6. Files NOT To Modify

- `uav_env/JSBSim/env.py` — reward, termination, missile launch
- `uav_env/JSBSim/envs/hetero_uav_combat_env.py` — reward overlay
- `uav_env/JSBSim/simulator.py` — aircraft dynamics, PID
- `uav_env/JSBSim/pid_controller.py` — PID
- `uav_env/JSBSim/adapters/hetero_obs_adapter_v2.py` — observation dimensions
- `uav_env/JSBSim/configs/` — aircraft XML, action trim
- `algorithms/mappo/trainer.py` — PPO update logic

## 7. Experiment Protocol

**Comparison:**
- **Baseline**: brma_legacy + shared MLP MAPPO (current)
- **Method**: brma_legacy + role-conditioned MAPPO

**Training:**
- Config: `hetero_mav_shared_geo_3v2.yaml`
- Blue opponent: `rule_nearest`
- Steps: 50k pilot first (for direction screening)

**Evaluation:**
- Configs: 3v2 + 5v4
- Episodes: 20

**Metrics:**
- red_win_rate, blue_win_rate, draw_rate, timeout_rate
- mav_survival_rate
- red_alive_final_mean, blue_alive_final_mean
- avg_return, avg_length
- action_saturation_rate

## 8. Success Criteria (Pilot Stage)

The 50k pilot is **not** a final convergence claim.  It is only for
direction screening:

- If role-conditioned MAPPO shows **better** 3v2 red_win, 5v4 transfer,
  or MAV survival than the shared MLP baseline → proceed to implement
  entity attention
- If it is **similar** to baseline → role separation alone may be
  insufficient; entity attention may still be worth trying
- If it is **worse** than baseline → reconsider the architecture or
  verify the implementation

## 9. Next Implementation Step

Implement role-conditioned MAPPO actor:
- Add `RoleConditionedActorCritic` in `algorithms/mappo/policy.py`
- Add flag to training script
- Run 50k pilot
- Do **not** change reward, termination, missile, action space, PID,
  or observation dimensions
