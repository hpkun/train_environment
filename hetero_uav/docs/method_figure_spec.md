# Method Figure Specification

This file provides a text specification for drawing the method figure.

## Main Actor Path

```text
Observation
  |
  v
Entity set construction
  - self entity
  - ally entities
  - enemy entities
  - alive / valid / observed masks
  |
  v
Mask module
  - dead / padding / observed masks
  - random scale mask disabled for main training
  - biased mask generator [opt-in]
  - self entity always kept
  - dead / padding entities remain masked
  |
  v
BRMA-style entity attention encoder
  - shared entity MLP
  - multi-head attention
  - key padding mask
  |
  v
GRU recurrent actor
  |
  v
Role-wise actor heads
  - MAV head
  - shared UAV head
  |
  v
Action distribution
  - Gaussian mean/std
  - 3D high-level action [pitch, heading, speed]
  |
  v
JSBSim environment
```

## Critic Side Path

```text
Global state [480]
  |
  v
Centralized MLP critic
  |
  v
Value estimation for PPO update
```

## Figure Annotations

Use these labels in the figure:

- Shared module:
  - entity MLP;
  - attention encoder;
  - GRUCell;
  - centralized critic.
- Role-specific module:
  - MAV actor head;
  - shared UAV actor head.
- Main mask source:
  - dead / padding / observed masks.
- Opt-in module:
  - biased mask generator.
- Evaluation path:
  - random scale mask disabled;
  - full currently observable entity set is used.

## Suggested Caption

> BRMA-style recurrent entity-attention actor for heterogeneous MAV/UAV
> air combat. The actor decodes the fixed 96-dimensional observation into an
> entity set, applies dead/padding/observed masks, encodes entities through
> multi-head attention, updates a GRU actor state, and dispatches actions through
> role-wise MAV/UAV policy heads. The critic remains centralized over the
> 480-dimensional global state.
