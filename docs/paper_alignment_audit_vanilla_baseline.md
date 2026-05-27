# Paper-aligned vanilla baseline consistency audit

## 1. Scope

This pass is audit-only. It does not change environment behavior, reward logic,
training algorithms, Blue rule policy, missile/radar logic, observation spaces,
or any launch window.

Inputs reviewed:

- Paper PDF: `Tan 等 - 2026 - Biased random masked attention MAPPO algorithm for zero-shot scale generalization of multi-UAV air c.pdf`
- `docs/current_environment_alignment_status.md`
- `docs/paper_env_reward_audit.md`
- `docs/reward_formula_alignment_plan.md`
- `my_uav_env/env.py`
- `rule_based_agent.py`
- `train_vanilla_mappo.py`
- `configs/experiment_presets.py`
- `results/vanilla_2v2_main_entropy_diag_100k_results.csv`

The 100K diagnostic run is interpreted only as diagnostic evidence. It is not
used here to justify algorithm, reward, entropy, standard-deviation, or missile
launch-window changes.

## 2. Environment geometry and initial condition

| Item | Paper statement / equation / table | Current implementation | Status | Notes |
|---|---|---|---|---|
| Initial geometry | Table 4: head-on flight, initial distance 10 km | `_make_init_state()` places Blue and Red 5 km from center on opposite longitudes, headings 90 deg / -90 deg, altitude 20,000 ft | MATCH | Initial head-on distance and headings match the paper statement. Formation lateral spacing is 500 m for multi-agent starts; paper does not give formation spacing. |
| Initial altitude | Table 1 includes altitude `h`; Table 4 does not explicitly give initial altitude in extracted text | `_make_init_state()` uses 20,000 ft | NEEDS PAPER TEXT VERIFICATION | Current docstring says strict paper baseline, but extracted Table 4 only exposes head-on distance. Need verify whether 20,000 ft appears in figures/code/supplement. |
| Battlefield horizontal area | Table 4 extracted text: `100 km × 100 km × 10 km`; eq.18: penalty if `|x| > 4e4` or `|y| > 4e4` | `BATTLEFIELD_HALF_SIZE = 40000.0`, boundary reward triggers outside ±40 km | PARTIAL | Current implementation matches eq.18 and the user's requested 80 km × 80 km audit target, but Table 4 text says 100 km × 100 km. Treat as paper internal inconsistency until source text is resolved. |
| Battlefield vertical size | Table 4: 10 km vertical battlefield | `BATTLEFIELD_ALTITUDE_MAX = 10000.0`, `BATTLEFIELD_ALTITUDE_MIN = 2500.0` | PARTIAL | Ceiling matches 10 km. Floor/crash at 2.5 km is an engineering crash threshold; paper reward eq.17 uses relative altitude and does not specify the same hard crash floor in extracted text. |
| Max episode length | Table 3: max episode length 1400 | `Config.max_episode_length = 1400`; presets use 1400 for main 2v2 | MATCH | With `sim_freq=60` and `agent_interaction_steps=12`, this equals 0.2 s per env step and 280 s maximum simulated duration. Paper gives 1400 steps but not the exact control frequency in extracted text. |
| Physics/control stepping | Paper uses JSBSim 6-DoF and hierarchical PID action tracking | `sim_freq=60`, `agent_interaction_steps=12`; one policy action every 12 physics frames | PARTIAL | Reasonable and internally consistent. Paper does not expose the exact policy interval in extracted text. |
| UAV count for current run | Paper main in-domain training is 6v6; zero-shot tests 8v8 and 10v10 | `train_vanilla_mappo.py` default and `vanilla_2v2_main` preset are 2v2 | MISMATCH for paper main; MATCH for current baseline intent | This repository intentionally keeps vanilla baseline at 2v2 for staged diagnosis. It should not be reported as a full paper 6v6 baseline. |
| Maximum speed / overload | Table 4: max speed 600 m/s, max overload 9g | `MAX_SPEED=600.0`, `OVERLOAD_G_LIMIT=9.0` | MATCH | Environment constants match Table 4. |

## 3. Observation and partial observability

| Item | Paper statement / equation / table | Current implementation | Status | Notes |
|---|---|---|---|---|
| Paper self observation | Table 1: 10 fields `[x, y, h, V, roll, pitch, heading, alpha, beta, Vd]` | Default env observation uses 11-dim engineering entity vector, not Table 1 | MISMATCH for vanilla training | `train_vanilla_mappo.py` uses `_flatten_obs()` over the default Dict observation. |
| Paper relative observation | Table 2: 10 fields `[x_body, y_body, z_body, theta_v_body, psi_v_body, V, theta_LOS_body, psi_LOS_body, q_LOS, d]` | Default entity vector is `[dx, dy, dz, AO_signed, TA, R, V_tgt, sin_roll, cos_roll, sin_pitch, cos_pitch]` | MISMATCH for vanilla training | The default 11-dim vector is an engineering representation, not strict Table 2. |
| Strict Table 1/Table 2 prototype | Paper §2.3 / Tables 1-2 | `get_strict_entity_observation()` and `get_strict_team_observations()` expose strict 10-dim prototype | PARTIAL | Available for attention actor modes, but not used by vanilla MAPPO training. Alpha/beta and q_LOS still have documented prototype caveats. |
| Radar FOV | §2.1.2: 120-degree sector, elevation `[-10, 32]` deg | `_is_detected_by_radar()` uses ±60 deg azimuth and `[-10, +32]` elevation | MATCH | Radar cannot detect missiles, consistent with paper. |
| RCS / Rmax | §2.1.2: RCS table interpolation; `Rmax = K * RCS^(1/4)` | Simplified front/side RCS approximation; `_compute_radar_max_range()` uses fourth root | PARTIAL | Formula aligned; RCS table is unavailable, so RCS model remains approximate. |
| AWACS / coarse blind-zone observation | Table 2 note: when radar does not detect enemy, approximate position can be obtained through RWS/AWACS; enemy speed unavailable and set to 0 | `_get_obs()` fills coarse body-frame position/range/AO with `TA=0`, `V_tgt=0`, attitude masked | MATCH / PARTIAL | Concept matches Table 2 note. Exact RWS/AWACS fidelity and noise model are not specified in extracted paper text. |
| Red training observation | Paper BRMA/MAPPO-Attention uses entity encoder; vanilla MLP baselines use larger fixed observation in 6v6 | `train_vanilla_mappo.py` flattens current 11-dim engineering Dict observation | PARTIAL | Valid as repository vanilla baseline, but not the paper's BRMA observation encoder path and not strict Table 1/Table 2. |

## 4. Missile launch logic

| Item | Paper statement / equation / table | Current implementation | Status | Notes |
|---|---|---|---|---|
| Actor fire action | §2.1.3 and §4.1: missile launch commands are generated by preset script; missiles launch automatically once conditions are met | `_check_missile_launch()` auto-scans all alive agents; actor action space has pitch/heading/velocity only | MATCH | No fire action in actor is paper-consistent. |
| Sensor basis for firing | §2.1.3: electro-optical sensor detects target in detection cone for >0.25 s; Table 4: photoelectric sensor detection range 10 km | `_check_missile_launch()` uses geometric cone and range; does not call radar detection | PARTIAL | It approximates photoelectric / seeker launch geometry rather than radar detection. This is plausible but should be called photoelectric launch logic, not radar launch logic. |
| AO gate | Paper says target must remain in sensor detection cone; figures/text emphasize launch behind 3-9 line. Exact AO cone angle not explicit in extracted text | `AO < 45°` | NEEDS PAPER TEXT VERIFICATION | 45° may be a project assumption for seeker cone; verify against paper code or appendix. |
| Range gate | Table 4: photoelectric sensor detection range 10 km | `500 m < R < 10 km` | PARTIAL | 10 km matches. 500 m minimum safe range is engineering guard, not explicitly found in extracted paper text. |
| TA / 3-9 line | §2.1.3: missiles launched only after enemy crosses 3-9 line; rear hemisphere is 180-degree fan behind target | `TA > 90°` | MATCH | This is the correct rear-hemisphere interpretation. |
| Lock delay | §2.1.3: target continuously detected for over 0.25 s | `missile_lock_delay_frames = round(0.25 * sim_freq)` | MATCH | Counted per physics frame. |
| Launch interval | §2.1.3 and Table 4: 0.5 s launch interval | `missile_cooldown_frames = round(0.5 * sim_freq)` | MATCH | Counted per physics frame. |
| Same-target suppression | §2.1.3: no recent missile launch events have targeted the same target | `_engaged_targets` hot-updated from in-flight missiles and per-step target assignments | PARTIAL / engineering enhancement | Concept aligned. Exact definition of "recent" is not specified in extracted text; implementation is stronger because it also deconflicts same-frame/team assignment. |
| Need radar detection to launch | Paper separates radar and electro-optical missile launch sensor | Current launch does not require `_is_detected_by_radar()` | MATCH / PARTIAL | This is acceptable if launch sensor is photoelectric as §2.1.3 says. If original code gated by radar track too, this needs verification. |
| Hit probability | §2.1.3: `0.05 + 0.95 * max(0, Vm·LOS/(|Vm||LOS|))` | `MissileSimulator._roll_hit_probability()` uses missile velocity and missile-to-target LOS dot product | MATCH | Already aligned. |
| Launch diagnostics | Not part of paper | `info["__launch_diag__"]` and vanilla CSV fields record gates/blocks/launches | ENGINEERING DIAGNOSTIC | Does not change behavior. Useful for interpreting launch asymmetry. |

Do not change launch conditions based only on 100K diagnostics. Current evidence
shows Red rarely satisfies the combined geometry; that is a tactics/learning
state, not proof that the launch window is wrong.

## 5. Blue rule-based baseline

| Item | Paper statement / equation / table | Current implementation | Status | Notes |
|---|---|---|---|---|
| Blue rule policy | §4.1: Blue uniformly adopts a rule-based strategy; performs target allocation/pursuit, missile evasion by warning | `rule_based_agent.blue_coordinated_actions()` controls Blue; environment handles scripted missile evasion layer | MATCH in concept | Blue being non-learning is paper-consistent. |
| Target allocation/pursuit | §4.1 states target allocation and pursuit according to battlefield situation | Greedy assignment, target scoring from observation, lead pursuit for radar tracks | PARTIAL | Paper does not provide exact rule equations in extracted text. Current implementation is an engineering baseline. |
| AWACS reacquisition | Table 2 note supports approximate enemy position through RWS/AWACS when radar is blind | Blue distinguishes radar vs AWACS coarse tracks and can pursue coarse bearing with lower confidence | PARTIAL / engineering implementation | Uses observation-space coarse bearing, not Red global position. Concept is paper-consistent, exact baseline unknown. |
| Boundary safety | Paper has boundary reward, not explicit Blue boundary autopilot | Blue has ownship-only no-target boundary patrol and near-boundary safety override | ENGINEERING ENHANCEMENT | Necessary to avoid scripted Blue flying out while no target is selected. Could make Blue stronger/more stable than an unspecified paper rule baseline. |
| GCAS | Paper does not describe Blue-only GCAS in extracted text | Training default `enable_blue_gcas=False`; env supports optional Blue GCAS | MATCH for training default / engineering option | As long as default stays false, Blue-only GCAS is not active in vanilla training. |
| Anti-stall / hard deck / energy protection | Paper has flight status rewards and missile evasion scripts; not exact Blue autopilot details | Blue rule policy includes hard deck, descent-rate safety, stall/anti-stall and trim compensation | ENGINEERING ENHANCEMENT | These protect the rule baseline. They may make Blue stronger or more robust than a minimal paper baseline. |

Risk assessment: current Blue may be stronger than an unspecified paper rule
baseline because of boundary safety, stall protection, lead pursuit, AWACS
reacquisition, and target deconfliction. However, these are Blue-only scripted
baseline details; changing them during the vanilla MAPPO diagnostic would
confound comparison unless paper code precisely defines the Blue policy.

## 6. Reward formula

| Reward item | Paper statement / equation / table | Current implementation | Status | Notes |
|---|---|---|---|---|
| Weighted sum | Eq.23 paragraph: weights `0.01, 0.002, 0.04, 0.04, 0.02, 0.15` | `_compute_rewards()` uses exactly these weights | MATCH | Components are per-agent plus terminal share. |
| Pitch eq.15 | Severe penalty if `|theta| > pi/3`; linear penalty for `pi/4 < |theta| < pi/3` | `_pitch_penalty()` matches extracted formula structure | MATCH / NEEDS VISUAL VERIFICATION | Formula text extraction is garbled but current form matches prior audit interpretation. |
| Roll eq.16 | Penalty if `|roll| > pi/4 & |pitch| > pi/4` | `_roll_penalty()` uses both conditions | MATCH | Aligned. |
| Altitude eq.17 | Pairwise relative altitude, quadratic segments, `0.1` high-altitude tail | `_altitude_reward()` uses `altitude_reward_pairwise_mean_eq17()` over alive enemies | MATCH / PARTIAL | Structure aligned; exact coefficients `h1/h2` and thresholds should remain marked for paper/code verification. |
| Boundary eq.18 | Fixed penalty if `|x| > 4e4` or `|y| > 4e4` | `_boundary_penalty()` returns `-10` once if either axis exceeds ±40 km | MATCH to eq.18 | Tension remains with Table 4 `100 km × 100 km`. |
| Velocity eq.19 | Penalty if below 0.3 Mach; severe if below 0.2 Mach | `_speed_penalty()` uses Mach = speed / 340, same two segments | MATCH / PARTIAL | Mach reference constant is engineering approximation. |
| Ta eq.20 | Extracted text shows first segment may be `10` for `q_LOS < 4°`, then piecewise decline | Current `ta_angle_advantage_fixed()` is continuous, non-negative, normalized `[0,1]` | PARTIAL / intentional scale deviation | Current reward version keeps normalized scale to avoid silent 10x reward change. A `10`-scale Ta should be a separate ablation if verified. |
| Td eq.21 | `1` when distance ≤ 15 km; `exp(1 - D/15)` when larger | `td_distance_advantage()` follows this form with distance in meters converted internally | MATCH | Current situation reward uses 3D Euclidean distance. |
| q_LOS / situation geometry | Eq.20 uses `q_LOS`; Table 2 includes LOS angles and qLOS | Current `_situation_reward()` uses 3D body-x LOS angle from `compute_body_x_q_los()` | PARTIAL | Plausible strict Table 2 interpretation; still needs paper/code verification against exact q_LOS definition. |
| Situation eq.22 | `sum_j(lambda1 * Ta_i^j * Td_i^j - lambda2 * Ta_j^i * Td_j^i)`, lambda1=1, lambda2=0.8 | `_situation_reward()` uses `1.0 * Ta_ij * Td_ij - 0.8 * Ta_ji * Td_ij` | PARTIAL | Since distance is symmetric, `Td_ij == Td_ji`; using one `Td` is equivalent if same distance function applies both ways. |
| Terminal eq.23 | Team-level `rend = 30 * (Nred - Nblue)` unless equal | Current env computes team-level value and shares across team members | PARTIAL / engineering implementation | Sum over team equals paper team reward. Per-agent share avoids multiplying terminal reward by team size. |
| Death / crash penalty | Not found in extracted paper reward equations | Dead/crashed agent gets `r_death = -10` on crash frame | ENGINEERING ADDITION | This improves causal credit for crashes but is not in extracted paper formula. |

Current reward version is `fixed_ta_alt_eq17_3dlos_v1`. It should not be mixed
with earlier reward logs.

## 7. MAPPO algorithm and hyperparameters

| Item | Paper statement / equation / table | Current implementation | Status | Notes |
|---|---|---|---|---|
| Algorithm | Paper main method is BRMA-MAPPO; comparison includes non-attention baselines and MAPPO-Attention | `train_vanilla_mappo.py` is vanilla MLP MAPPO baseline | PARTIAL | This is not BRMA-MAPPO and should be labeled vanilla baseline. |
| Training scale | Paper in-domain training uses 6v6 | Current default/preset main is 2v2 | MISMATCH for paper main | Intentional staged baseline. Do not claim paper main 6v6 result. |
| Rollout threads | Table 3: 32 | Current default `num_envs=8`; preset `vanilla_2v2_main` uses 8 | MISMATCH | Reduced for local stability/resource constraints. |
| Total steps | Table 3 extracted text appears as `1 5 107`, likely `1.5e7` or OCR artifact; earlier project assumed `1e7` | Current default `total_env_steps=10_000_000` | NEEDS PAPER TEXT VERIFICATION | Do not silently claim exact match. Verify original PDF table visually. |
| Max episode length | Table 3: 1400 | Current `max_episode_length=1400` | MATCH | |
| Replay buffer size | Table 3: 2000 | Current `replay_buffer_size=2000` | MATCH | With 8 envs, rollout is 250 env steps per worker. |
| Hidden sizes | Table 3: `[128, 128]` | Actor: input FC 128, GRU 128, MLP head 64; critic hidden 128 via config | PARTIAL | Not a two-layer `[128,128]` actor MLP in the strict paper sense; has GRU and smaller action head. |
| Recurrent policy | Table 3: `Use_recurrent_policy=True`, `Recurrent_n=1`; MAPPO-Attention also uses GRU | `VanillaActor` uses one `GRUCell` | MATCH | |
| Centralized critic | CTDE critic uses global state | Current critic input is flattened concat of Red agents' default observations | PARTIAL | It is centralized over Red observations, but not paper native global state `S={s_i}`. |
| Entropy coefficient | Table 3: 0.05 | `entropy_coef=0.05`; `_current_entropy_coef()` constant | MATCH | 100K std/entropy diagnostics do not indicate entropy/std failure. |
| Actor/Critic LR | Table 3: actor 0.0002, critic 0.0005 | Current defaults match | MATCH | |
| Gamma/lambda/clip | Standard MAPPO settings; extracted Table 3 does not list all in visible text | `gamma=0.99`, `gae_lambda=0.95`, `clip_epsilon=0.2` | NEEDS PAPER TEXT VERIFICATION | Likely standard, but not visible in extracted table. |
| Minibatches / update epochs | Paper extracted Table 3 does not show these | `n_update_epochs=10`, `n_minibatches=8` | NEEDS PAPER TEXT VERIFICATION | Keep stable unless paper/code says otherwise. |
| Action distribution / std | Paper does not expose exact distribution in extracted text | Gaussian policy with learnable `action_log_std`, initialized ln(0.3), clamp sampled actions | PARTIAL | Common PPO implementation detail; not paper-verified. |
| Checkpoint best logic | Paper does not define checkpoint naming | best reward / best recent win-rate logic | ENGINEERING RECORDING | Does not affect PPO updates unless `--resume-from-best` is used. |

## 8. Current 100K diagnostic interpretation

CSV reviewed: `results/vanilla_2v2_main_entropy_diag_100k_results.csv`.

Summary from 50 result rows through 100,000 env steps:

- Cumulative launch opportunity totals:
  - `LaunchDiagRedRangeOk = 3,208,665`
  - `LaunchDiagBlueRangeOk = 2,377,482`
  - `LaunchDiagRedGeometryOk = 15`
  - `LaunchDiagBlueGeometryOk = 14,790`
  - `LaunchDiagRedLaunches = 1`
  - `LaunchDiagBlueLaunches = 854`
- Final row at 100K:
  - `LaunchDiagRedGeometryOk = 0`, `LaunchDiagRedLaunches = 0`
  - `LaunchDiagBlueGeometryOk = 191`, `LaunchDiagBlueLaunches = 8`
  - `LaunchDiagRedRangeOk = 54,838`, `LaunchDiagBlueRangeOk = 46,344`
  - `RedRangeToGeometryRate = 0.0`
  - `BlueRangeToGeometryRate = 0.00412`
  - `ActionStdMean = 0.33947`, with observed 100K range `0.30168 -> 0.33947`

Interpretation:

1. Red enters range often, but almost never satisfies the combined launch geometry
   of range + AO + TA. This supports the hypothesis that Red is failing to
   maneuver into rear-hemisphere firing geometry, not that missiles are disabled.
2. Blue obtains far more combined geometry opportunities and therefore more
   launches. This is consistent with Blue being a scripted pursuit baseline.
3. Action standard deviation does not show collapse or explosion in the first
   100K steps. Entropy/std should not be changed based on this diagnostic.
4. 100K steps is too short to decide that vanilla MAPPO failed as a paper-aligned
   baseline. It only diagnoses early training geometry asymmetry.

## 9. Blocking mismatches before 10M run

Only issues that affect paper fidelity are listed here.

1. **Training scale mismatch**: current formal baseline is 2v2, while paper
   in-domain training is 6v6. A 10M 2v2 run is useful but should not be labeled
   the paper's 6v6 main result.
2. **Rollout threads mismatch**: paper Table 3 uses 32 rollout threads; current
   default uses 8. This is a resource-driven mismatch, not an algorithmic bug.
3. **Observation mismatch for vanilla**: current vanilla Red policy uses the
   11-dim engineering flattened observation, not strict Table 1/Table 2.
4. **Critic global state mismatch**: vanilla critic uses Red flattened
   observations concat, not paper native global state.
5. **Battlefield area ambiguity**: eq.18 implies ±40 km, but Table 4 extracted
   text says `100 km × 100 km × 10 km`. This must be verified before claiming
   exact environment match.
6. **Ta scale mismatch**: current normalized `Ta` is intentionally not the
   possible paper `10`-scale first segment. This should remain a separate reward
   scale ablation, not a silent baseline change.
7. **Blue rule baseline exactness**: paper states rule-based target allocation,
   pursuit, and missile evasion but does not expose exact rule implementation in
   extracted text. Current Blue policy includes engineering protections and may
   be stronger than the paper baseline.
8. **Total training steps ambiguity**: extracted Table 3 OCR for max steps is
   ambiguous. Verify visually whether paper uses `1e7`, `1.5e7`, or another
   value before final reproduction claims.

Not blockers for continuing diagnostics:

- Red low launch count at 100K.
- Blue high launch count at 100K.
- Action std rising from about 0.30 to 0.34.
- Current missile launch window, given it is paper-consistent in concept and
  diagnostically instrumented.

## 10. Recommended next step

Recommended path:

1. Continue with a **500K or 1M paper-aligned diagnostic** under the current
   `fixed_ta_alt_eq17_3dlos_v1` reward version and existing missile launch
   window. This is the correct next step because 100K is too short and std is
   stable.
2. Do **not** change MAPPO entropy, learnable std, reward scales, or missile
   launch gates based on the current 100K data.
3. Use launch diagnostics during the 500K/1M run to track whether Red's
   `RangeOk -> GeometryOk` conversion improves.
4. Only after the 500K/1M diagnostic should a full **10M 2v2 vanilla baseline**
   be considered. Label it as 2v2 vanilla baseline, not paper 6v6 main result.
5. Before a true paper reproduction run, decide explicitly whether to run:
   - 6v6 vanilla MLP baseline with fixed observation dimensions,
   - MAPPO-Attention with strict observation,
   - or BRMA-MAPPO.

Changes that must be deferred to non-paper or separate ablation branches:

- Widening or relaxing missile launch AO/R/TA conditions.
- Changing entropy coefficient or action std behavior.
- Changing Ta from normalized `[0,1]` to the possible paper `10`-scale version.
- Weakening or strengthening Blue rule policy without a precise paper baseline
  specification.
- Replacing observation/global state representations mid-run.

No code behavior was changed by this audit pass.
