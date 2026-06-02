# MAPPO Baseline Stage 1

## 1. What this is

A minimal plain shared-policy MAPPO baseline for heterogeneous UAV/MAV
composition.  This is **Stage 1** of the algorithm roadmap.

## 2. Why not HAPPO

- All controlled red agents share the same 3-dim high-level action space.
- HeteroObsAdapter already provides fixed-dimension inputs regardless of composition.
- MAPPO is simpler to debug and sufficient to validate the adapter pipeline.
- HAPPO sequential update will be added in a later stage.

## 3. Architecture

- **Actor MLP**: [140, 256, 128] -> Gaussian mean (3,) + learnable log_std (3,)
- **Critic MLP**: [700, 256, 128, 1]
- **Shared policy**: all red agents use the same actor.
- **Centralized critic**: sees the concatenated padded red team observations.

## 4. Current scope

This is a **smoke test / pipeline validation**, not a paper experiment.
It does NOT produce zero-shot transfer claims, win-rate statistics, or
convergence plots.

## 5. Next stages

1. Stage 2: Role-aware MAPPO (role embedding in actor/critic).
2. Stage 3: Entity attention encoder with valid/alive masks.
3. Stage 4: HAPPO-like sequential update.
4. Stage 5: GRU / temporal memory.
