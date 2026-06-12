# Main Experiment Checklist

Status labels:

- `satisfied`: the current code/protocol supports the requirement.
- `partially satisfied`: pieces exist, but the mainline result or documentation is not yet final.
- `missing`: not implemented or not supported by current evidence.

## 1. Unified Observation

Status: `satisfied`

- [x] Uses `mav_shared_geo` / V2 observation for the mainline protocol.
- [x] 3v2 and 5v4 use the same actor observation schema.
- [x] `HeteroObsAdapterV2` pads and masks to `max_red=5`, `max_blue=4`.
- [x] Actor observation dimension is fixed at 96.
- [x] Critic state dimension is fixed at 480.
- [x] 3v2 and 5v4 do not require two incompatible actor observation formats.

Notes:

- Some older configs and scripts still exist for legacy/debug paths. They should not be treated as the current paper protocol.
- The mainline paper protocol should refer to the V2 adapter and `mav_shared_geo` only.

## 2. MAV/UAV Heterogeneous Reward

Status: `partially satisfied`

- [x] `brma_legacy` remains available and unchanged as a baseline reward.
- [x] `role_v1` exists as a role-oriented reward ablation.
- [x] MAV reward terms in `role_v1` emphasize survival, support, death penalty, and capped team contribution.
- [x] UAV reward terms in `role_v1` emphasize attack window, kill bonus, death penalty, and light missile-warning penalty.
- [x] Main heterogeneous configs keep MAV missiles at 0, so MAV is not treated as the attacking aircraft.
- [ ] A final paper-ready role reward has not been selected.
- [ ] `role_v1` has not yet outperformed the baseline or solved engagement.

Decision:

- Do not add another reward version now.
- Treat `brma_legacy` as the baseline reward and `role_v1` / `happo_ref_v0` as ablation/reference-validation rewards.
- The next bottleneck is learned engagement behavior, not more reward variants.

## 3. MAV Actor and Shared UAV Actor

Status: `satisfied` for HAPPO reference v0; `not applicable` to shared MLP MAPPO baseline.

- [x] `HAPPOReferencePolicy` has a distinct MAV actor.
- [x] `HAPPOReferencePolicy` has one shared UAV actor.
- [x] Role inference maps the MAV role to the MAV actor and all non-MAV red UAVs to the UAV actor.
- [x] In 3v2, `red_1` and `red_2` share the UAV actor.
- [x] In 5v4, `red_1` through `red_4` reuse the same UAV actor.
- [x] UAVs are not trained with one actor per slot.
- [x] The centralized critic is retained.

Notes:

- This is HAPPO reference v0, not full TAM-HAPPO.
- There is no GRU and no attention in the current implementation.

## 4. 3v2 to 5v4 Zero-Shot Test

Status: `satisfied` as a protocol; combat performance remains unsolved.

- [x] Training is defined on 3v2.
- [x] Evaluation includes both 3v2 seen and 5v4 zero-shot settings.
- [x] 5v4 evaluation does not fine-tune.
- [x] New 5v4 UAV slots reuse the shared UAV actor.
- [x] The documentation now explicitly calls this zero-shot scale transfer.

Notes:

- The protocol is paper-aligned enough to report as an experimental setup.
- The current learned results are not yet strong enough to claim solved zero-shot combat transfer.

## Overall Assessment

| core design | status | summary |
|---|---|---|
| Unified observation | satisfied | V2 `mav_shared_geo` provides one padded actor schema for 3v2 and 5v4 |
| MAV/UAV heterogeneous reward | partially satisfied | Role rewards exist, but no final reward has solved engagement |
| MAV actor + shared UAV actor | satisfied | HAPPO reference v0 has one MAV actor and one shared UAV actor |
| 3v2 to 5v4 zero-shot test | satisfied as protocol | 5v4 reuses actors without fine-tuning, but performance is not solved |

## Paper Readiness

The current protocol can be written into the paper as the experimental setup and as a negative/diagnostic baseline story. It should not be written as a completed successful method yet.

The defensible claim is:

> The environment and protocol support unified observation, role-separated actors, shared UAV policy transfer, and 3v2-to-5v4 zero-shot evaluation; current learned policies still fail mainly at engagement geometry, so the next minimal step is to make the policy learn approach-and-fire behavior.

