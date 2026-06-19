# Algorithm Path: MAPPO Baseline vs HAPPO

## 1. Why TAM-HAPPO Uses HAPPO

The TAM-HAPPO (Heterogeneous-Agent Proximal Policy Optimisation) paper uses
HAPPO because:

- **Heterogeneous agents**: agents have different observation/action spaces,
  dynamics, and roles.
- **Sequential policy update**: each agent's policy is updated in a fixed
  order, using the multi-agent advantage decomposition.
- **Inactive-agent mask**: agents that are dead or out of the scenario are
  masked from the loss / advantage computation.
- **Heterogeneous action spaces**: different agents can have different
  action dimensions or unavailable actions.
- **Centralized value network**: global state and joint action input.

## 2. Why This Project Starts with MAPPO

We start with a plain MAPPO (shared-policy) baseline before adding HAPPO
complexity because:

1. **Unified action space**: All controlled red agents use the same BRMA
   3-dim high-level action (pitch, heading, velocity).  There is no
   heterogeneous action space to decompose.

2. **Fixed input dimensions**: HeteroObsAdapter provides fixed-dimension
   actor (140-dim) and critic (700-dim) inputs regardless of composition.
   MAPPO's centralized critic and shared actor work directly on these.

3. **Composition zero-shot transfer** benefits from a **shared policy**
   conditioned on role information.  Adding HAPPO's per-agent sequential
   update first would obscure whether transfer benefits come from the
   observation adapter or the algorithm.

4. **MAPPO is simpler and faster to converge**: Before investing in HAPPO's
   sequential advantage decomposition, we need to confirm that the
   adapter-based representation can learn at all.

5. **Incremental validation**: Each HAPPO component (sequential update,
   inactive-agent mask, joint advantage) introduces its own debugging
   surface.  Starting with MAPPO isolates adapter correctness from
   algorithm correctness.

## 3. When to Move Beyond MAPPO

- After plain MAPPO trains stably across at least two compositions.
- If role-conditioned shared policy plateaus and cannot close the gap
  to rule-based Blue.
- If we introduce truly heterogeneous action spaces (e.g., scout has
  different action dimension than attack UAV).
- If sequential update is needed because asymmetric policy coupling
  (e.g., observability differences between MAV and UAV) prevents
  joint optimisation.

## 4. Planned Algorithm Stages

| Stage | Algorithm | Key addition |
|---|---|---|
| 1 | Plain shared-policy MAPPO | HeteroObsAdapter + CentralizedCritic |
| 2 | Role-aware MAPPO | Role embedding in actor / critic |
| 3 | Entity attention encoder | Valid/alive masks, AttentionActor |
| 4 | HAPPO / HAPPO-like | Sequential update, inactive-agent mask |
| 5 | GRU / temporal memory | Recurrent policy across timesteps |

## 5. Paper Positioning

- **TAM-HAPPO** validates MAV/UAV heterogeneous cooperation with HAPPO.
- **BRMA-MAPPO** uses mask/attention for variable-composition generalisation.
- **This project** studies heterogeneous composition zero-shot transfer:
  train on one MAV/UAV composition, evaluate on another.
- The MAPPO baseline is the necessary first step; HAPPO is a
  justified-but-later extension.
