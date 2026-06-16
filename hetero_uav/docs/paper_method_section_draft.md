# Paper Method Section Draft

This draft is intended as source material for a thesis/report method section.
It describes the implemented method path and its implementation boundary. It
does not claim a full reproduction of BRMA-MAPPO or TAM-HAPPO.

## 1. Problem Definition

We study cooperative air combat with heterogeneous MAV/UAV flight vehicles. The
training scenario is a 3v2 engagement: the red team contains one MAV and two
UAVs, while the blue team contains two UAV opponents. The transfer scenario is
a 5v4 engagement: the red team scales to one MAV and four UAVs, and the blue
team scales to four UAV opponents.

The goal is fixed-capacity zero-shot scale transfer from 3v2 training to 5v4
evaluation. The additional red UAVs in 5v4 reuse the same shared UAV actor. No
fine-tuning is performed in the 5v4 evaluation scenario.

The MAV and UAVs have different roles:

- MAV: information/support role and survival-oriented command role;
- UAV: attack role, responsible for entering the launch envelope and scoring
  missile hits.

The environment uses a JSBSim-based high-level control interface. The action is
a 3-dimensional command:

```text
[target_pitch_norm, target_heading_norm, target_speed_norm]
```

This keeps the current experiment focused on multi-agent coordination and
zero-shot transfer, not low-level flight-control reinforcement learning.

## 2. Network Structure

The final implemented method path is opt-in through:

```text
policy_arch = brma_recurrent_masked
```

The actor path is:

```text
actor_obs[96]
-> entity construction
-> mask module
-> BRMA-style entity attention encoder
-> GRU recurrent actor
-> role-wise MAV/UAV policy heads
-> Gaussian action distribution
```

### Entity Construction

The actor observation keeps the fixed 96-dimensional V2 schema. Internally, the
policy decodes this flat vector into a fixed-capacity entity set containing:

- self entity;
- ally entities;
- enemy entities;
- valid/alive/observed masks.

This preserves checkpoint compatibility and the current environment observation
dimension while enabling a BRMA-style entity encoder inside the policy.

### BRMA-Style Entity Attention Encoder

Each entity is embedded by a shared MLP. The embedded entities are then processed
by multi-head self-attention with a key padding mask. Dead, padded, and
unobserved entities are excluded from attention. The self entity is used for
pooled actor features.

This is a BRMA-style entity attention encoder adapted to the current fixed V2
observation, not a verbatim full BRMA-MAPPO reproduction.

### Mask Module

The method supports two opt-in mask mechanisms:

- random scale mask: randomly drops valid non-self entities during training;
- biased mask generator: predicts keep probabilities and masks selected
  low-keep-probability entities.

Self entities are not masked. Dead and padded entities are never reintroduced.
Random scale masking is disabled during evaluation, so evaluation uses the full
available observation.

The biased mask path currently implements forward mask generation and logging.
It does not implement the full BRMA biased mask KL/objective training loop.

### GRU Recurrent Actor

After the entity attention encoder, a GRUCell updates the actor hidden state.
The recurrent hidden state is reset on episode reset. During PPO update, the
stored one-step hidden state is replayed.

This should be described as a GRU recurrent actor. It should not be described as
full TBPTT recurrent PPO.

### Role-Wise Policy Heads

The actor uses role-wise policy heads:

- a MAV actor head for `red_0`;
- a shared UAV actor head for all red UAVs.

This supports the heterogeneous MAV/UAV actor structure while allowing the 5v4
additional red UAVs to reuse the same UAV actor.

### Centralized Critic

The critic uses the centralized 480-dimensional global state and remains an MLP.
The critic is not entity-attention based and is not recurrent in the current
implementation.

## 3. Training Framework

Training uses a HAPPO-style role-wise PPO framework:

- decentralized actor execution;
- centralized critic for value estimation;
- separate MAV and UAV actor heads;
- simplified role-wise actor updates.

This is not a strict HAPPO implementation. In particular, the current version
does not claim full sequential policy correction or formal multi-agent advantage
decomposition.

## 4. Method Boundary

Safe wording:

> We implement an opt-in BRMA-style recurrent masked entity-attention actor for
> a heterogeneous MAV/UAV cooperative air-combat setting, with role-wise MAV/UAV
> policy heads and a centralized critic.

Do not claim:

- complete BRMA-MAPPO reproduction;
- complete TAM-HAPPO reproduction;
- full biased random mask objective;
- full recurrent PPO / TBPTT;
- strict HAPPO sequential correction;
- proof that each module independently improves performance;
- arbitrary-size zero-shot generalization beyond the configured fixed capacity.

