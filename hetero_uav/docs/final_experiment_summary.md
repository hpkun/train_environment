# Final Experiment Summary

## 1. Experiment Goal

The current experiment goal is to build a heterogeneous MAV/UAV cooperative air-combat framework for 3v2 training and 5v4 zero-shot scale-transfer evaluation.

This is an experimental mainline for thesis/report use. It is not a full BRMA-MAPPO reproduction, not a full TAM-HAPPO reproduction, and not a low-level flight-control reinforcement-learning project.

## 2. Core Design

The mainline is organized around four design choices.

1. BRMA-inspired unified observation

   The project uses `mav_shared_geo` / V2 observation as a fixed-capacity entity/mask observation. It separates ego, ally, and enemy information, includes masks, and keeps the actor input compatible between 3v2 and 5v4.

2. MAV/UAV heterogeneous role reward

   The reward design separates MAV support/survival objectives from UAV attack objectives. `brma_legacy` remains the baseline reward. `role_v1` and `happo_ref_v0` are role-oriented ablation/reference rewards, not final proof that reward design is solved.

3. MAV independent actor + shared UAV actor

   HAPPO reference v0 uses one MAV actor for `red_0` and one shared UAV actor for all red attack UAVs. In 3v2, `red_1` and `red_2` share the UAV actor. In 5v4, `red_1` through `red_4` reuse the same UAV actor.

4. 3v2 train and 5v4 zero-shot eval

   Training is defined on 3v2. Evaluation includes the seen 3v2 setting and the larger 5v4 zero-shot setting. The 5v4 evaluation does not fine-tune the policy.

## 3. Observation Alignment

The current V2 observation supports fixed-capacity 3v2-to-5v4 scale transfer:

- actor observation dimension: `96`;
- critic state dimension: `480`;
- maximum red slots: `5`;
- maximum blue slots: `4`;
- 3v2 is represented by padding unused red slots and using `red_valid_mask`;
- 5v4 adds red UAVs that use the same actor observation schema;
- masks include valid/alive information and enemy observed information;
- unobserved enemies can remain alive but masked/zeroed in actor observation.

This is BRMA-inspired, but it is not the full BRMA observation encoder. The current implementation does not include multi-head entity attention, biased random masked attention, a mask generator, or a permutation-invariant set encoder. Therefore, the safe claim is fixed-capacity 3v2-to-5v4 zero-shot scale transfer, not arbitrary-scale BRMA-MAPPO generalization.

## 4. Completed Experiments

### Shared MLP MAPPO 1M

The shared MLP MAPPO 1M run is the weak baseline. Its best 100-episode checkpoint mostly produced timeout survival/draw behavior:

- 3v2: red win 0.05, blue win 0.00, draw 0.95, timeout 1.00, MAV survival 0.00.
- 5v4: red win 0.15, blue win 0.12, draw 0.73, timeout 1.00, MAV survival 0.00.

Conclusion: shared MLP MAPPO does not establish reliable heterogeneous combat ability.

### HAPPO Reference v0 With F-22 MAV

The F-22 MAV branch exposed instability under the current high-level `[pitch, heading, speed]` action interface and PID/JSBSim backend. It is closer visually to the intended MAV idea, but it is not a stable immediate vehicle for validating the learning method.

Conclusion: F-22 should not remain the current blocker.

### HAPPO Reference v0 With F-16 MAV Surrogate 200k

The F-16 MAV surrogate fixed the immediate MAV survival stability issue enough to validate the algorithm path. The 200k latest checkpoint showed the strongest learned attack signal so far:

- best checkpoint: MAV survival 1.00, red fire 0.02, red hit 0.02, blue death 0.02, mostly timeout alive advantage;
- latest checkpoint: MAV survival 0.86, red fire 1.48, red hit 1.20, blue death 1.20, mixed timeout and red elimination.

Conclusion: F-16 surrogate demonstrates that learned red attack behavior can appear, but the result is not yet stable enough to be the final combat baseline.

### HAPPO Reference v0 With F-16 MAV Surrogate 1M

The 1M training rollout completed without NaN and showed survival in the training log:

- latest train row: return about +12.99, red win 1.00, MAV survival 1.00, red missiles fired 0, missile hits 0.

Independent 100-episode evaluation did not confirm a stable combat policy:

- best 3v2: red win 0.03, blue win 0.57, draw 0.40, MAV survival 0.00, red hit 0.07;
- best 5v4: red win 0.33, blue win 0.02, draw 0.65, MAV survival 0.00, red hit 1.24;
- latest collapsed to blue elimination wins in both 3v2 and 5v4.

### Oracle-Pretrain Fine-Tune 200k

The direct-chase oracle dataset verified that the red attack chain is usable:
red UAVs can close distance, fire, hit, and kill blue aircraft. Behavior
cloning the shared UAV actor from this dataset reduced the supervised action
loss, but the 200k closed-loop fine-tune still did not pass the combat-pilot
gate on the normal 3v2 geometry.

- best 3v2: survival-oriented, MAV survival 0.83, but red fire/hit/blue death
  all 0;
- latest 3v2: some red fire/hit, but MAV survival 0;
- latest 5v4: aggressive attack signal, but still MAV survival 0.

Conclusion: the next step is an easy combat task with shorter initial distance
and better initial heading, not another blind 1M run.

### Easy Combat 100k

The easy combat task shortened the initial red-UAV/blue-UAV distance and
aligned the initial headings while keeping reward, observation, action,
missile dynamics, PID, aircraft XML, and actor/critic dimensions unchanged.
Training used the oracle-pretrained checkpoint, CUDA, and the hardcoded
4-environment rollout path.

The 100k run completed without NaN, but it still did not produce red attack
behavior in fast checkpoint evaluation:

- best 3v2: red fire 0.00, red hit 0.00, blue death 0.00, MAV survival 1.00;
- latest 3v2: red fire 0.00, red hit 0.00, blue death 0.00, MAV survival 1.00;
- both checkpoints won only by timeout alive advantage.

Conclusion: easy combat improved survival but did not make HAPPO reference v0
transfer oracle imitation into closed-loop attack behavior.

### Oracle-Pretrained Closed-Loop Diagnosis and 50k Anchor

The original oracle-pretrained checkpoint was directly evaluated before
fine-tuning and did not fire in easy combat. The checkpoint did load correctly,
but the pretrain loss used plain MSE on circular heading actions. This has been
fixed by using wrapped heading error for the heading dimension.

An easy-combat oracle dataset and checkpoint were then generated:

- dataset samples: `17988`;
- easy oracle red fire mean: `2.16`;
- easy oracle red hit mean: `2.00`;
- easy oracle blue death mean: `2.00`;
- wrapped action-match MSE: `0.010325`.

The easy-combat oracle checkpoint can fire in closed loop with a fixed safe MAV
action. A default-off UAV imitation anchor was added and run for 50k steps on
the easy-combat task:

- output: `outputs/happo_easy_combat_oracle_anchor_50k`;
- latest eval: `red_win_rate=0.95`, `mav_survival_rate=1.00`,
  `red_missiles_fired_mean=1.70`, `red_missile_hits_mean=1.55`,
  `blue_dead_mean=1.50`;
- best eval: `red_win_rate=0.85`, `mav_survival_rate=1.00`,
  `red_missiles_fired_mean=1.45`, `red_missile_hits_mean=1.30`,
  `blue_dead_mean=1.30`.

50-episode confirmation:

- latest eval: `red_win_rate=1.00`, `mav_survival_rate=1.00`,
  `red_missiles_fired_mean=1.50`, `red_missile_hits_mean=1.48`,
  `blue_dead_mean=1.48`;
- best eval: `red_win_rate=0.86`, `mav_survival_rate=0.92`,
  `red_missiles_fired_mean=1.32`, `red_missile_hits_mean=1.16`,
  `blue_dead_mean=1.16`.

Conclusion: the immediate no-fire failure after oracle pretrain was fixed on
the easy-combat task. This remains a curriculum/easy-task result, not the final
normal-geometry or 5v4 zero-shot result.

### Normal-Geometry Oracle Anchor 100k

The wrapped-heading oracle pretrain was also regenerated for the normal 3v2
geometry:

- dataset reused: `outputs/direct_chase_oracle_dataset/direct_chase_oracle_3v2.npz`;
- wrapped action-match MSE: `0.028370`;
- cosine similarity: `0.850247`.

The 100k normal-geometry oracle-anchor run completed without NaN:

- output: `outputs/happo_normal_geometry_oracle_anchor_100k`;
- `total_env_steps_actual=100000`;
- `num_envs=4`;
- `init_checkpoint=outputs/oracle_pretrain/uav_actor_oracle_pretrained_wrapped_normal/model.pt`.

Fast 3v2 eval failed the combat gate:

- `red_missiles_fired_mean=0.05`;
- `red_missile_hits_mean=0.00`;
- `blue_dead_mean=0.00`;
- `mav_survival_rate=1.00`;
- all episodes timed out, with red wins coming from alive advantage.

A 20-episode latest-only 3v2/5v4 check showed:

- 3v2 latest: `red_missiles_fired_mean=0.05`, `red_missile_hits_mean=0.00`,
  `blue_dead_mean=0.00`, `mav_survival_rate=1.00`;
- 5v4 latest: `red_missiles_fired_mean=0.75`, `red_missile_hits_mean=0.50`,
  `blue_dead_mean=0.50`, `mav_survival_rate=0.00`.

Decision: `normal_geometry_combat_success=false`. The easy task is learnable,
but normal geometry has not transferred successfully.

Conclusion: HAPPO reference v0 with F-16 surrogate is mainly a survival baseline, not a reliable combat baseline.

### Red Direct Chase Oracle Sanity Check

The direct chase oracle proves that the environment can produce red firing, hits, and kills when the policy solves engagement geometry:

- vs blue zero: red fire 2.00, red hit 2.00, blue death 2.00, red win 1.00;
- vs blue BRMA rule: red fire 2.25, red hit 2.00, blue death 2.00, red win 1.00.

Conclusion: the red attack pipeline is operational. The learned policy failure is not caused by a broken missile/fire-control chain.

### BRMA Observation Alignment Test

The observation alignment test verifies the fixed-capacity V2 input contract:

- 3v2 actor observation dimension equals 5v4 actor observation dimension;
- 3v2 critic state dimension equals 5v4 critic state dimension;
- 3v2 missing slots are padded and masked;
- 5v4 additional red UAVs use the same actor observation dimension.

Conclusion: the unified observation can support the 3v2-to-5v4 zero-shot protocol.

### Oracle-Pretrain Fine-Tune Path

The direct-chase oracle imitation path has been implemented as the next minimal step after the survival baselines:

- collect red UAV direct-chase oracle samples;
- behavior-clone only the shared UAV actor;
- keep MAV actor and critic frozen during pretraining;
- fine-tune HAPPO reference v0 for 200k steps from the oracle-pretrained checkpoint.

In this Codex run, the real JSBSim collection and 200k fine-tune were not executed because the active Python environment lacks `gymnasium/jsbsim`, and `conda run -n brmamappo` was blocked by sandbox permission review timeout. No oracle-pretrain performance result is claimed yet.

## 5. Results Table

| experiment | train scenario | eval scenario | aircraft | policy | reward | MAV survival | red fire | red hit | blue death | win type | conclusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Shared MLP MAPPO 1M | 3v2 | 3v2 / 5v4 | F-22 MAV branch + F-16 UAVs | shared MLP MAPPO | brma_legacy | 0.00 in 100-episode best eval | not reliable | not reliable | 0.00 blue elimination | mostly timeout/draw | weak baseline only |
| HAPPO reference v0 with F-22 MAV | 3v2 | 3v2 / 5v4 | F-22 MAV + F-16 UAVs | MAV actor + shared UAV actor | happo_ref_v0 | eval 0.00 | unstable | unstable | not robust | latest blue elimination | F-22 unstable under current interface |
| HAPPO reference v0 F-16 surrogate 200k best | 3v2 | 3v2 | F-16 MAV surrogate + F-16 UAVs | MAV actor + shared UAV actor | happo_ref_v0 | 1.00 | 0.02 | 0.02 | 0.02 | timeout red alive advantage | survival, almost no combat |
| HAPPO reference v0 F-16 surrogate 200k latest | 3v2 | 3v2 | F-16 MAV surrogate + F-16 UAVs | MAV actor + shared UAV actor | happo_ref_v0 | 0.86 | 1.48 | 1.20 | 1.20 | mixed timeout and red elimination | strongest learned attack signal, not stable |
| HAPPO reference v0 F-16 surrogate 1M train latest | 3v2 | training rollout | F-16 MAV surrogate + F-16 UAVs | MAV actor + shared UAV actor | happo_ref_v0 | 1.00 | 0 | 0 | 0 | timeout survival | survival baseline, not combat |
| HAPPO reference v0 F-16 surrogate 1M best eval 3v2 | 3v2 | 3v2 | F-16 MAV surrogate + F-16 UAVs | MAV actor + shared UAV actor | happo_ref_v0 | 0.00 | 0.07 hit proxy | 0.07 | 0.07 | mostly blue alive advantage / draw | not usable combat baseline |
| HAPPO reference v0 F-16 surrogate 1M best eval 5v4 | 3v2 | 5v4 zero-shot | F-16 MAV surrogate + F-16 UAVs | MAV actor + shared UAV actor | happo_ref_v0 | 0.00 | 1.72 | 1.24 | 1.23 | timeout alive advantage / draw | some transfer signal, MAV survival fails |
| Red direct chase oracle vs blue zero | none | 3v2 sanity | F-16 MAV surrogate + F-16 UAVs | scripted direct chase | environment fire-control | red team survives in sanity case | 2.00 | 2.00 | 2.00 | red elimination win 1.00 | attack chain works |
| Red direct chase oracle vs blue BRMA rule | none | 3v2 sanity | F-16 MAV surrogate + F-16 UAVs | scripted direct chase | environment fire-control | red team survives in sanity case | 2.25 | 2.00 | 2.00 | red elimination win 1.00 | learned policy lacks engagement behavior |
| BRMA observation alignment test | none | 3v2 / 5v4 contract | not aircraft-dependent | V2 adapter contract | none | not applicable | not applicable | not applicable | not applicable | not a combat test | unified observation contract verified |
| Oracle-pretrain HAPPO 200k | 3v2 | 3v2 / 5v4 | F-16 MAV surrogate + F-16 UAVs | UAV actor BC + HAPPO fine-tune | happo_ref_v0 | pending | pending | pending | pending | pending | implemented, real run pending |
| Easy-combat oracle anchor 50k latest | easy 3v2 | easy 3v2 | F-16 MAV surrogate + F-16 UAVs | UAV actor BC + HAPPO + imitation anchor | happo_ref_v0 | 1.00 | 1.50 | 1.48 | 1.48 | red win 1.00 | first learned easy-task combat result |
| Normal-geometry oracle anchor 100k latest | normal 3v2 | normal 3v2 | F-16 MAV surrogate + F-16 UAVs | UAV actor BC + HAPPO + imitation anchor | happo_ref_v0 | 1.00 | 0.05 | 0.00 | 0.00 | timeout red alive advantage | no normal-geometry combat transfer |
| Normal-geometry oracle anchor 100k latest | normal 3v2 | 5v4 zero-shot | F-16 MAV surrogate + F-16 UAVs | UAV actor BC + HAPPO + imitation anchor | happo_ref_v0 | 0.00 | 0.75 | 0.50 | 0.50 | timeout draw/blue alive advantage | some attack signal but MAV survival fails |
| Geometry curriculum medium 50k best | easy to medium 3v2 | medium 3v2 | F-16 MAV surrogate + F-16 UAVs | UAV actor BC + HAPPO + imitation anchor | happo_ref_v0 | 0.90 | 1.50 | 1.30 | 1.30 | red win 1.00 | medium geometry preserves attack behavior |
| Geometry curriculum normal 50k best | medium to normal 3v2 | normal 3v2 | F-16 MAV surrogate + F-16 UAVs | UAV actor BC + HAPPO + imitation anchor | happo_ref_v0 | 0.94 | 1.82 | 1.56 | 1.52 | red win 0.92 / red elimination 0.52 | first normal-geometry learned combat checkpoint |
| Geometry curriculum normal 50k latest | medium to normal 3v2 | normal 3v2 | F-16 MAV surrogate + F-16 UAVs | UAV actor BC + HAPPO + imitation anchor | happo_ref_v0 | 1.00 | 0.00 | 0.00 | 0.00 | blue alive advantage | latest still collapses; use best checkpoint |

## 6. Main Findings

- Shared MLP MAPPO is a weak baseline.
- F-22 MAV is unstable under the current high-level action + PID interface.
- F-16 MAV surrogate resolves the immediate MAV survival stability issue for method validation.
- HAPPO reference v0 closes the heterogeneous actor and zero-shot evaluation loop: MAV actor, shared UAV actor, 3v2 train, 5v4 zero-shot eval.
- Current learned results are mainly survival baselines, not combat baselines.
- The red attack pipeline is operational; the direct chase oracle can fire, hit, and destroy blue aircraft.
- The learned policy's key failure is tactical engagement: it does not reliably learn approach angle, alignment, and launch-envelope satisfaction.
- The oracle-pretrain path revealed and fixed a heading-wrap loss issue.
- A UAV imitation anchor restores red fire/hit behavior on the easy-combat task.
- The same anchor does not yet transfer robustly to normal 3v2 geometry or 5v4 zero-shot.
- A single medium-geometry curriculum stage restores normal 3v2 attack behavior
  in the best checkpoint: `red_missiles_fired_mean=1.82`,
  `red_missile_hits_mean=1.56`, `blue_dead_mean=1.52`,
  `mav_survival_rate=0.94`.
- The normal-geometry latest checkpoint can still collapse to no red fire, so
  checkpoint selection is part of the current experimental protocol.

## 7. What Can Be Claimed

The current project can claim:

- a heterogeneous MAV/UAV zero-shot experiment framework has been built;
- unified V2 observation supports 3v2 training and 5v4 fixed-capacity zero-shot evaluation;
- MAV actor and shared UAV actor can be used for heterogeneous policy structure;
- current methods show survival-transfer behavior;
- the red attack chain is verified by oracle sanity checks.
- geometry curriculum can recover a learned normal-geometry 3v2 combat
  checkpoint under the F-16 MAV surrogate setup.

## 8. What Cannot Be Claimed

The current project cannot claim:

- full BRMA-MAPPO reproduction;
- full TAM-HAPPO reproduction;
- biased random masked attention;
- GRU or attention implementation;
- solved 5v4 combat zero-shot transfer;
- a latest-checkpoint learned policy with stable kill ability.

## 9. Recommended Next Step

Continue from the geometry-curriculum normal best checkpoint for a normal
geometry 200k run, then evaluate 5v4 zero-shot transfer.
