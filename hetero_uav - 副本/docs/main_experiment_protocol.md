# Main Experiment Protocol

## Research Goal

The main experiment studies heterogeneous MAV/UAV cooperative air combat with zero-shot scale transfer. The immediate goal is not to reproduce full TAM-HAPPO, not to learn low-level flight control, and not to keep expanding environment engineering. The protocol should test whether a red team trained in 3v2 can reuse the same observation schema and UAV actor in a larger 5v4 setting without fine-tuning.

## Training Scenario

- Training scenario: 3v2 heterogeneous red-vs-blue combat.
- Red team: `red_0` is the MAV role; `red_1` and `red_2` are attack UAV roles.
- Blue team: two attack UAVs controlled by a rule-based opponent.
- Current operational training configs use `mav_shared_geo` observation and high-level action `[pitch, heading, speed]` through PID/JSBSim control.
- The centralized critic is retained.

For HAPPO reference validation, the intended policy layout is separate role actors:

- `red_0` uses the MAV actor.
- All red attack UAVs use one shared UAV actor.

The F-16 MAV surrogate branch should be treated as an algorithm-validation surrogate, not as the final aircraft-identity claim.

## Zero-Shot Evaluation Scenario

- Seen evaluation: 3v2.
- Zero-shot scale-transfer evaluation: 5v4.
- The 5v4 evaluation must not fine-tune the model.
- Additional 5v4 red UAV slots reuse the same shared UAV actor trained on the 3v2 UAV slots.
- The centralized critic input remains fixed by adapter padding; it is an evaluation-time input contract, not a new policy trained for 5v4.

This is therefore a zero-shot scale-transfer protocol: same actor observation schema, same MAV actor, same shared UAV actor, larger team composition.

## Observation Design

The mainline observation protocol is `mav_shared_geo` with the V2 adapter:

- Actor observation dimension: 96.
- Critic state dimension: 480.
- Adapter maximums: `max_red=5`, `max_blue=4`.
- The 3v2 and 5v4 settings use the same actor observation schema.
- Smaller scenarios are padded and masked rather than represented with a separate incompatible observation format.

The V2 adapter includes ego features, role one-hot information, ally entities, enemy entities, validity masks, alive masks, and enemy observed masks. MAV shared tracks are represented through observation fields such as enemy observed mask and track source.

This is a BRMA-inspired entity/mask observation. It is not the full BRMA attention encoder: the current adapter preserves entity groups and masks but then flattens them to a fixed vector consumed by the current policy networks. No biased random masked attention or permutation-invariant entity encoder is implemented in the current mainline.

The current zero-shot claim is fixed-capacity 3v2-to-5v4 scale transfer. It should not be described as arbitrary-scale generalization beyond the configured `max_red=5`, `max_blue=4` capacity.

## Role Reward Design

There are three reward modes relevant to the current project state:

- `brma_legacy`: baseline reward inherited from the BRMA-style environment. It remains the baseline comparison and must not be silently changed.
- `role_v1`: role-oriented ablation reward. It gives the MAV survival, support, and capped team-contribution terms; attack UAVs receive attack-window, kill/event, and death-penalty terms.
- `happo_ref_v0`: HAPPO reference validation reward. It is also role-oriented, but should be described as a reference-validation reward rather than a final paper reward.

The intended role design is:

- MAV reward emphasizes survival, sensing/support, and capped contribution when UAVs kill.
- UAV reward emphasizes attack geometry, fire/hit/kill events, and survival from death/crash penalties.
- MAV is not encouraged to act as an attacking aircraft; in the main heterogeneous configs it has zero missiles.

Current limitation: role rewards have not produced a stable final combat policy. Do not add more reward versions before solving the basic engagement problem.

## Policy Architecture

The HAPPO reference v0 architecture matches the minimum role-sharing requirement:

- One MAV actor for `red_0`.
- One shared UAV actor for every red non-MAV UAV.
- In 3v2, `red_1` and `red_2` share the UAV actor.
- In 5v4, `red_1` through `red_4` reuse the same UAV actor.
- The critic is centralized.
- The current reference policy has no GRU and no attention, so it must not be called full TAM-HAPPO.

The shared MLP MAPPO baseline remains useful only as a weak baseline because it does not separate MAV and UAV policy roles.

## Evaluation Metrics

Report at minimum:

- average return;
- episode length;
- red win rate;
- blue win rate;
- draw rate;
- timeout rate;
- MAV survival rate;
- final red alive count;
- final blue alive count;
- red missiles fired;
- red missile hits;
- blue deaths;
- NaN detection;
- actor observation dimension and critic state dimension checks.

For the paper narrative, distinguish timeout alive-advantage wins from actual elimination wins. A policy that survives to timeout without firing or hitting is not a completed combat policy.

## Current Limitation

The current environment and training stack satisfy the basic protocol shape, but the learned policies have not yet produced a robust combat solution. The direct chase oracle shows that the red fire chain works, while learned policies usually fail to close, align, and satisfy the launch envelope.

The next experimental bottleneck is tactical engagement behavior, not another broad environment audit.

## What Is Not Claimed

The current project state must not claim:

- full TAM-HAPPO reproduction;
- paper low-level action-space reproduction;
- GRU temporal modeling;
- attention-enhanced value network;
- final F-22 MAV validity;
- solved 5v4 heterogeneous transfer;
- learned missile evasion;
- combat effectiveness based only on timeout survival.
