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

## 6. Main Findings

- Shared MLP MAPPO is a weak baseline.
- F-22 MAV is unstable under the current high-level action + PID interface.
- F-16 MAV surrogate resolves the immediate MAV survival stability issue for method validation.
- HAPPO reference v0 closes the heterogeneous actor and zero-shot evaluation loop: MAV actor, shared UAV actor, 3v2 train, 5v4 zero-shot eval.
- Current learned results are mainly survival baselines, not combat baselines.
- The red attack pipeline is operational; the direct chase oracle can fire, hit, and destroy blue aircraft.
- The learned policy's key failure is tactical engagement: it does not reliably learn approach angle, alignment, and launch-envelope satisfaction.
- The oracle-pretrain path is the minimal next intervention, but its combat result is still pending a real brmamappo run.

## 7. What Can Be Claimed

The current project can claim:

- a heterogeneous MAV/UAV zero-shot experiment framework has been built;
- unified V2 observation supports 3v2 training and 5v4 fixed-capacity zero-shot evaluation;
- MAV actor and shared UAV actor can be used for heterogeneous policy structure;
- current methods show survival-transfer behavior;
- the red attack chain is verified by oracle sanity checks.

## 8. What Cannot Be Claimed

The current project cannot claim:

- full BRMA-MAPPO reproduction;
- full TAM-HAPPO reproduction;
- biased random masked attention;
- GRU or attention implementation;
- solved combat zero-shot transfer;
- a learned policy with stable kill ability.

## 9. Recommended Next Step

Use the red direct chase oracle as a minimal imitation/action-guidance source, or build an easy initial-geometry task, so the learned policy first acquires approach-and-fire behavior.
