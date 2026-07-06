# TAM-HAPPO Launch and Heterogeneous Cooperation Alignment Audit

This audit checks whether the current missile launch gate, MAV shared information path, UAV attack behavior, and `tam_brma_paper_aligned_v1` reward mode jointly support the intended TAM-HAPPO heterogeneous MAV/UAV cooperation design.

Scope: audit and design only. No training was started. No formal environment logic, reward formula, missile dynamics, hit model, PID, aircraft XML, blue rule, action space, or observation dimension was changed.

## 1. Current Launch Logic Summary

In the current environment, red attack UAVs can launch only when all of the following are true:

- the shooter is alive, has missiles, is not in launch cooldown, and is not a MAV;
- the target is alive and is not already engaged by another missile;
- the shooter has a valid track to the target, either direct observation or MAV-shared track for red UAVs;
- the target is inside the BRMA-style launch geometry gate:
  - 3D range is between `MISSILE_LAUNCH_MIN_RANGE = 500 m` and `missile_launch_range_m = 10000 m` by default;
  - attack angle/AO is below `45 deg`;
  - target aspect/TA is above `90 deg`;
  - optional boresight gate is disabled by default;
- the target remains continuously valid through the lock delay;
- no same-target deconfliction or kill-cooldown block applies.

The learned policy does not output a fire action. Fire control is an environment-owned scripted mechanism, consistent with the BRMA-style preset missile launch script.

## 2. Current Launch Condition Table

| condition | current implementation | code location | BRMA-MAPPO evidence | TAM-HAPPO evidence | judgment | whether modification recommended |
|---|---|---|---|---|---|---|
| Environment-owned auto launch | Policy outputs high-level flight action; `_check_missile_launch()` decides fire. | `uav_env/JSBSim/env.py::_check_missile_launch` | BRMA text describes missile launch commands as preset scripts rather than learned actions. | TAM narrative describes UAV lock/launch behavior, but does not specify that fire is a learned action. | Strongly BRMA-aligned; compatible with TAM role narrative. | Keep for mainline. |
| MAV cannot launch | Red `mav` role is blocked from launch; configs set `mav.num_missiles: 0`. | `env.py::_has_launch_track`, config agent type blocks | BRMA is homogeneous UAV and does not define MAV. | TAM explicitly describes MAV as battlefield information / mission guidance unit, while UAVs carry missiles. | TAM-aligned. | Keep. |
| UAV missile count | Attack UAVs carry missiles, normally `num_missiles: 2`. | 3v2/5v4 configs | BRMA uses short-range AAM setup. | TAM 3v2/5v4 descriptions state UAVs carry two missiles. | TAM-aligned. | Keep. |
| Track requirement | Red UAV launch requires direct track or MAV-shared track; blue uses own observation/fallback. | `env.py::_has_launch_track`; `envs/hetero_uav_combat_env.py::_build_mav_shared_geo_obs` | BRMA relies on individual electro-optical/IR sensor detection. | TAM states MAV provides battlefield information and mission guidance. | Direct track is BRMA-aligned; MAV shared track is TAM-mechanism-aligned but implementation-specific. | Keep as current mechanism; audit track-source usage. |
| Default launch range | `MISSILE_LAUNCH_RANGE_THRESH = 10000 m`; config may override `missile_launch_range_m`. | `env.py` constants and `__init__` | BRMA table/text gives photoelectric sensor detection range 10 km. | TAM text found in local extract does not provide equally explicit launch-distance contract; it defines UAV attack/lock behavior narratively. | Explicitly BRMA-backed; not enough evidence to call it TAM-backed. | Keep for BRMA-style baseline. Do not change without separate config and evidence. |
| Minimum range | `MISSILE_LAUNCH_MIN_RANGE = 500 m`. | `env.py` | Not clearly found as a BRMA paper constant in the local extract. | Not clearly found in TAM paper extract. | Engineering safety / numerical realism supplement. | Keep unless diagnostics prove it blocks valid launches; do not claim paper-backed. |
| AO gate | `AO < 45 deg`. | `env.py::_build_launch_geometry_3d` | BRMA describes rear-hemisphere / 3-9 line launch; exact AO threshold is an implementation interpretation. | TAM describes lock/launch but does not provide this exact gate in the inspected text. | BRMA-style engineering concretization. | Keep for BRMA-style contract; relax only in diagnostic ablation. |
| TA gate | `TA > 90 deg`. | `env.py::_build_launch_geometry_3d` | BRMA 3-9 line/rear-hemisphere condition supports rear-aspect launch logic. | TAM does not clearly specify this exact TA threshold in inspected text. | BRMA-aligned in concept; threshold is implementation-specific. | Keep for BRMA-style contract; do not remove for TAM without evidence. |
| Continuous lock delay | `missile_lock_delay_frames = round(0.25 * sim_freq)`. | `env.py::__init__`, `_check_missile_launch` | BRMA states continuous detection for 0.25 s before launch. | TAM inspected text does not provide this exact delay. | Explicit BRMA-backed. | Keep. |
| Attack interval / cooldown | Default `missile_attack_interval_sec = 0.5`; cooldown frames are derived from sim frequency. | `env.py::__init__`, `_launch_missile` | BRMA gives launch interval 0.5 s. | TAM local extract contains `Attack Interval (s) 25`. | Current value is BRMA-aligned and conflicts with TAM table if TAM attack interval is used as the launch cooldown. | For TAM-specific ablation, use a separate config with 25 s interval; do not silently change main BRMA-style contract. |
| Same-target deconfliction | Targets already engaged by active missiles are skipped. | `env.py::_select_missile_target`, `_is_target_engaged` | BRMA text says launch is blocked if the opponent has been targeted by a recent missile. | TAM does not clearly specify this exact deconfliction rule in inspected text. | BRMA-aligned. | Keep. |
| Target ranking | Default configs use `red_target_selection_mode: closest`; optional `mav_threat_rank` exists. | `env.py::_select_missile_target`; configs | BRMA does not require MAV-aware ranking. | TAM mission guidance could motivate MAV-informed ranking, but inspected text does not define a scoring formula. | Closest is conservative; MAV-aware ranking is paper-inferred/engineering. | Keep closest in paper-aligned configs; use MAV-aware ranking only in diagnostic or explicitly labeled mechanism-aligned configs. |
| Missile hit model | Scripted missile handles post-launch guidance and probabilistic hit. | `uav_env/JSBSim/simulator.py` | BRMA gives IR close-range AAM with probability depending on missile velocity/LOS alignment. | TAM does not provide the same explicit hit equation in inspected text. | BRMA-aligned. | Keep outside this audit; no change recommended here. |

## 3. TAM-HAPPO Design Purpose Summary

TAM-HAPPO is a heterogeneous MAV/UAV cooperative combat setting, not a homogeneous UAV-only task.

From the local TAM-HAPPO paper text and project paper-grounded notes:

- MAV is a high-value support/command unit.
- MAV does not carry missiles.
- MAV provides battlefield information and mission guidance.
- MAV should prioritize survivability and generally stay behind or retreat to safer rear airspace.
- UAVs are attack units. They carry missiles, lock targets, launch, pursue, surround, evade, and destroy enemies.
- The heterogeneous design is meant to separate information/support/safety responsibilities from direct attack responsibilities.

Therefore, a TAM-HAPPO-aligned implementation should make MAV information useful to UAV attack decisions without turning MAV into an attacker, and should avoid evaluating success only through MAV survival if UAVs never convert information into attack opportunities.

## 4. Conflicts Between Current Implementation and TAM-HAPPO Design Purpose

### 4.1 Whether MAV shared information truly affects UAV attack opportunities

Current implementation does let MAV shared information affect target eligibility:

- `mav_shared_geo` can set `enemy_observed_mask` and `enemy_track_source[:, 1]` for red UAVs when MAV sees a blue target.
- `_has_launch_track()` accepts MAV-shared tracks for red non-MAV shooters.

However, MAV shared information does not relax the geometric launch gate. A UAV still needs 500 m to 10 km range, AO below 45 deg, TA above 90 deg, lock delay, cooldown, and deconfliction. In practice, MAV shared information can help if the UAV is geometrically ready but lacks direct observation. It does not by itself solve approach, heading alignment, rear-aspect positioning, or closing geometry.

Judgment: partially aligned. The support channel exists, but its effect is limited to observation/track qualification.

### 4.2 Whether 10 km range limits MAV remote support

The MAV observation range in current paper-aligned configs can be much larger than the UAV direct observation range, e.g. `mav_observation_range_m: 80000`. This supports long-range battlefield awareness.

The actual missile launch range remains 10 km by default. This is consistent with BRMA-MAPPO's explicit photoelectric sensor / short-range AAM setting, but it means MAV long-range shared tracks mainly provide early information rather than immediate remote fire authorization.

Judgment: not a bug for BRMA-style launch; a limitation if the intended TAM interpretation is that MAV mission guidance should more directly expand UAV attack opportunity.

### 4.3 Whether TA/AO/continuous lock have TAM-HAPPO basis

Continuous lock delay of 0.25 s and 0.5 s interval have explicit BRMA support. TA/AO gates instantiate BRMA rear-hemisphere / 3-9 line launch constraints.

The inspected TAM-HAPPO text supports UAV lock/launch behavior and MAV guidance, but it does not provide the same exact AO/TA/0.25 s lock gate. TAM also lists an attack interval of 25 s, which differs from the current 0.5 s BRMA-style interval.

Judgment: AO/TA/0.25 s lock are BRMA-style engineering/contract items, not clearly TAM-explicit items. The 0.5 s launch interval is BRMA-aligned and not TAM-aligned if TAM's 25 s attack interval is interpreted as launch cooldown.

### 4.4 Whether launch interval matches TAM-HAPPO

Current default: 0.5 s.

BRMA evidence: 0.5 s launch interval appears explicitly in local BRMA paper text.

TAM evidence: local TAM text includes `Attack Interval (s) 25`.

Judgment: current interval is BRMA-aligned. A TAM-specific config should use `missile_attack_interval_sec: 25.0` if the goal is strict TAM launch-interval matching. This should be a separate config, because changing it globally would alter the BRMA-style missile contract and prior baselines.

### 4.5 Whether UAV reward encourages using MAV information to complete attacks

For `tam_brma_paper_aligned_v1`, the intended active reward is a paper-aligned mixture:

- UAV reward follows BRMA-style flight, attack-geometry/situation, and terminal terms.
- MAV reward receives TAM-style safety/support/event terms in a controlled way.
- Shared-track, launch-with-MAV-track, and hit-with-MAV-track fields are logged.

The reward does not currently give a direct active UAV bonus for firing or hitting specifically via MAV-shared tracks. This is conservative and avoids adding a hand-coded "use MAV" reward. But it also means the training signal for using MAV information is indirect: the UAV must discover that shared information helps it reach better BRMA attack geometry and terminal outcomes.

Judgment: conservative and defensible, but possibly weak for short training probes. Adding active UAV attack event rewards would move toward TAM-HAPPO event-reward design, but it should be done in a separate reward mode/config and clearly labeled.

## 5. Three Modification Schemes

### Scheme A: Conservative Paper-Aligned Scheme

Purpose: preserve the explicit BRMA missile contract and TAM heterogeneous roles without adding unproven launch privileges.

Proposed status:

| change | classification | recommendation |
|---|---|---|
| Keep MAV unarmed and UAV armed. | paper-backed by TAM | Keep. |
| Keep environment-owned scripted fire. | paper-backed by BRMA; compatible with TAM narrative | Keep. |
| Keep 10 km launch range. | paper-backed by BRMA | Keep. |
| Keep 0.25 s lock delay. | paper-backed by BRMA | Keep. |
| Keep 0.5 s attack interval. | paper-backed by BRMA; not TAM interval | Keep for BRMA-style baseline, but do not call TAM interval aligned. |
| Keep AO/TA/rear-hemisphere gate. | paper-backed concept from BRMA; threshold engineering | Keep. |
| Keep MAV shared track as track qualification only. | TAM-inferred mechanism | Keep, log carefully. |
| Keep `red_target_selection_mode: closest`. | conservative baseline | Keep for strict baseline. |
| Do not add active shared-track fire/hit reward. | avoids paper overclaim | Keep as log-only unless moving to Scheme B. |

Needed configs: current `tam_brma_paper_aligned_v1` configs are suitable for Scheme A if their documentation states that the launch contract is BRMA-style and TAM alignment is role-level/cooperation-level.

Code changes: none required for launch logic.

Reward changes: none required.

Tests needed:

- config test: MAV missiles are zero and UAV missiles are two;
- observation/track-source test: MAV-shared target can be marked as shared track;
- launch diagnostic test: shared track qualifies target only when geometry gate also passes;
- no dimension/action changes.

2K/50K/500K judging criteria:

- 2K: no crash, finite logs, valid launch diagnostics.
- 50K: nonzero red launch opportunities or clear logged bottleneck reasons.
- 500K: red missiles/hits and MAV survival should be evaluated separately; absence of attack must be diagnosed by range/AO/TA/track-source block reason, not just return.

### Scheme B: TAM-HAPPO Mechanism-Aligned Scheme

Purpose: make the environment closer to TAM-HAPPO's heterogeneous mechanism while still avoiding unsupported launch-gate relaxation.

Proposed status:

| change | classification | recommendation |
|---|---|---|
| Add separate config with `missile_attack_interval_sec: 25.0`. | paper-backed by TAM table | Recommended as a TAM interval ablation, not a silent replacement. |
| Keep MAV unarmed / UAV armed. | paper-backed by TAM | Required. |
| Keep MAV shared track and log direct vs shared source. | TAM-inferred implementation of battlefield information | Required. |
| Consider UAV event reward for lock/launch/kill only if directly mapped to TAM UAV event reward. | paper-backed in broad category, formula may be implementation-specific | Separate reward mode only. |
| Consider target ranking using MAV shared information. | paper-inferred from mission guidance | Diagnostic or mechanism-aligned config only; do not let it bypass geometry gates. |
| Keep range/AO/TA gate unless exact TAM launch-geometry evidence is added. | conservative paper boundary | Recommended. |

Candidate config names:

- `hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_paper_aligned_v1_tam_interval25.yaml`
- `hetero_mav_shared_geo_5v4_f16_mav_surrogate_tam_brma_paper_aligned_v1_tam_interval25.yaml`

Code changes: not required if existing `missile_attack_interval_sec` config hook is used.

Reward changes: only if a separate TAM UAV-event reward mode is explicitly created; do not mutate Scheme A.

Tests needed:

- config-load test for 25 s interval;
- assert cooldown frames equal `25 * sim_freq`;
- assert MAV remains unarmed;
- compare launch rate and win rate separately from reward return.

2K/50K/500K judging criteria:

- 2K: correct cooldown and no launch-spam.
- 50K: determine whether longer interval changes blue/red pressure and episode outcomes.
- 500K: compare against Scheme A; do not attribute improvement solely to algorithm if missile opportunity changed.

### Scheme C: Diagnostic Enhanced Scheme

Purpose: isolate bottlenecks. This scheme is not paper-backed as a final method.

Proposed status:

| change | classification | recommendation |
|---|---|---|
| Increase launch range to 12-15 km. | diagnostic-only unless new paper evidence is cited | Use only to test range bottleneck. |
| Relax AO to 60 deg. | diagnostic-only | Use only to test angle bottleneck. |
| Relax TA or remove TA. | diagnostic-only; BRMA-deviating | Avoid for mainline. |
| Enable `red_target_selection_mode: mav_threat_rank`. | paper-inferred / diagnostic | Useful for support-effect diagnosis; not strict paper. |
| Add active shared-track launch/hit rewards. | diagnostic or TAM-event adaptation, depending formula | Keep in separate reward mode; not pure BRMA-UAV. |
| Let MAV-shared track bypass direct observation only. | already implemented as track qualification | Keep; do not let it bypass range/AO/TA without explicit diagnostic labeling. |

Candidate config names:

- `..._diagnostic_range15.yaml`
- `..._diagnostic_mav_threat_rank.yaml`
- `..._diagnostic_ao60.yaml`

Code changes: only through existing config hooks where possible. Avoid changing default environment behavior.

Reward changes: only in new diagnostic reward modes.

Tests needed:

- assertions that diagnostic configs are clearly labeled;
- no default-config changes;
- launch block reason distributions before/after.

2K/50K/500K judging criteria:

- If red fire appears only after range/AO/TA relaxation, the bottleneck is launch geometry, not necessarily policy architecture.
- If red fire appears only after active shared-track reward, the bottleneck is sparse/indirect cooperation reward.
- Diagnostic success should not be reported as paper-aligned performance.

## 6. Recommended Execution Order

1. **Use Scheme A first.**
   - Do not change launch logic.
   - Run short smoke and launch diagnostics with existing `tam_brma_paper_aligned_v1`.
   - Confirm how often UAVs have direct track, MAV-shared track, range OK, AO OK, TA OK, and lock-ready.

2. **If Scheme A shows zero or near-zero red launch opportunities, run Scheme C diagnostics, not as final results.**
   - First diagnose range bottleneck.
   - Then diagnose AO/TA bottleneck.
   - Then diagnose whether MAV-shared track is ever used for launch candidates.

3. **If TAM-specific fidelity is required, add Scheme B as a separate config.**
   - The first TAM-specific change should be the 25 s attack interval because it has explicit TAM table evidence.
   - Do not mix this with range/AO/TA relaxation in the same experiment.

4. **Only after diagnostic evidence, decide whether to create a new TAM-event reward mode.**
   - Adding UAV event reward for launch/kill can be justified as TAM-HAPPO-inspired only if it is mapped to the TAM UAV event reward category.
   - It should not be mixed into the current BRMA-UAV conservative reward without renaming and documentation.

## 7. Risk Notes

- Changing launch range can make performance improve simply because missiles get more opportunities, not because heterogeneous cooperation improved.
- Canceling or relaxing TA/AO can deviate from the BRMA 3-9 line / rear-hemisphere launch contract.
- Making MAV shared information directly qualify or prioritize launches can become a hand-coded cooperation rule if it bypasses learned approach and geometry.
- Adding UAV attack event rewards changes the method from a pure BRMA-UAV reward trunk to a TAM-HAPPO-UAV event reward adaptation.
- A 25 s TAM attack interval may reduce launch frequency and make early training harder; it should be tested separately from reward or policy changes.
- `mav_threat_rank` target selection is mechanism-motivated but not an explicit TAM formula. It should be reported as a mechanism-aligned or diagnostic variant, not strict paper reproduction.

## 8. Final Recommendation

Use the current launch logic as a **BRMA-style missile launch contract with TAM-HAPPO heterogeneous roles**, not as a full TAM-HAPPO missile model.

For the next controlled experiments:

1. Keep Scheme A as the main paper-conservative baseline.
2. Add more launch diagnostics before changing gates.
3. If TAM-specific launch fidelity is needed, introduce a separate Scheme B config with `missile_attack_interval_sec: 25.0`.
4. Use Scheme C only to diagnose whether learning is blocked by range, AO/TA, or weak MAV-information reward coupling.

Do not change the default launch gate, reward formula, missile dynamics, PID, aircraft XML, blue rule, action space, or observation dimension as part of this audit.

