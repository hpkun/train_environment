# Final Method Architecture

This project keeps several policy paths so experiments can compare method
complexity without breaking earlier checkpoints.

## Current Opt-In Architectures

| Policy Arch | Encoder | Recurrent | Mask | Actor Heads | Critic |
|---|---|---:|---|---|---|
| `flat` | 96-dim MLP | No | No | MAV head + shared UAV head | 480-dim MLP |
| `entity_attention` | Local entity attention | No | No | MAV head + shared UAV head | 480-dim MLP |
| `brma_entity` | BRMA-style EntityObservationEncoder | No | No | MAV head + shared UAV head | 480-dim MLP |
| `brma_recurrent` | BRMA-style EntityObservationEncoder | GRUCell | No | MAV head + shared UAV head | 480-dim MLP |
| `brma_recurrent_masked` | BRMA-style EntityObservationEncoder | GRUCell | Random scale mask and/or biased mask | MAV head + shared UAV head | 480-dim MLP |

## Recommended Paper Method

The recommended main method for the final paper experiment is:

```text
BRMA-style recurrent masked entity-attention actor
```

Use `--policy-arch brma_recurrent_masked` with either:

- `--brma-random-scale-mask` for the primary masked probe;
- `--brma-biased-mask` for the biased-mask probe or ablation.

The recommended baselines are:

- `flat`: weak MLP baseline and compatibility baseline;
- `brma_entity`: non-recurrent entity encoder ablation;
- `brma_recurrent`: recurrent entity encoder ablation without masks.

## Method Diagram

The most complete opt-in path can be drawn as:

```text
actor obs
-> entity set adapter
-> random scale mask or biased mask
-> BRMA EntityObservationEncoder
-> multi-head attention
-> GRUCell
-> role-wise actor heads
-> action distribution
```

The centralized critic remains a flat global-state MLP in the current code.

## Relation To BRMA-MAPPO

Implemented:

- entity-style observation decoding;
- BRMA-style entity encoder;
- multi-head attention;
- GRU actor path;
- random scale mask forward path;
- biased mask generator forward path.

Not fully implemented:

- full BRMA mask KL/objective training;
- strict biased random masked attention training loop;
- arbitrary-size attention generalization beyond configured capacity.
- biased mask objective is simplified to forward mask generation and logging.

## Relation To HAPPO

Implemented:

- heterogeneous role-wise actors;
- centralized critic;
- simplified role-wise PPO update.

Not fully implemented:

- strict HAPPO sequential correction;
- formal multi-agent advantage decomposition.
- GRU replay is a practical one-step hidden-state integration, not full TBPTT
  recurrent PPO.

## Non-Method Diagnostics

Launch-envelope audits, oracle checks, ACMI exporters, and heading diagnostics
are engineering diagnostics. They should not be presented as algorithmic method
modules.

## Safe Wording

Safe: "We implement an opt-in BRMA-style recurrent masked entity-attention
actor for the heterogeneous MAV/UAV setting."

Avoid:

- "This is a full BRMA-MAPPO or full TAM-HAPPO reproduction."
- "The biased mask objective exactly matches the original paper."
- "The GRU uses a full TBPTT recurrent PPO implementation."
