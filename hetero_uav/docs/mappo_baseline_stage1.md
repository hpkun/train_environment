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

## 5. Opponent Policy

The Stage 1 runner now separates red controlled actions from blue opponent
actions:

- Red actions come from the shared-policy MAPPO actor.
- Blue actions come from `algorithms.mappo.opponent_policy.OpponentPolicy`.

Supported opponent modes:

- `zero`: all blue actions are `[0, 0, 0]`. This is only for smoke/debug.
- `random`: blue actions are sampled uniformly from `[-1, 1]`.
- `rule_nearest`: each blue agent chooses the nearest non-zero red entity in
  its own observation and steers toward it with a simple `[pitch, heading,
  speed]` command.

The `rule_nearest` opponent is intentionally minimal. It is not a full tactical
FSM and does not produce formal win-rate conclusions. It is used so Stage 1 can
be evaluated against a non-trivial scripted opponent instead of a stationary
zero-action placeholder.

## 6. Why Still Plain MAPPO

The immediate goal is trainability diagnostics: verify that the observation
adapter, shared actor, centralized critic, save/load path, and multi-composition
evaluation path are stable.

HAPPO, role-aware attention, entity attention, GRU, and other algorithm changes
are deliberately deferred. Plain MAPPO is the smallest runner that can validate
the environment and adapter plumbing before adding algorithmic complexity.

## 7. Next stages

1. Stage 2: Role-aware MAPPO (role embedding in actor/critic).
2. Stage 3: Entity attention encoder with valid/alive masks.
3. Stage 4: HAPPO-like sequential update.
4. Stage 5: GRU / temporal memory.

The next practical step is trainability diagnostics with the `rule_nearest`
opponent, not role-aware algorithm development.
