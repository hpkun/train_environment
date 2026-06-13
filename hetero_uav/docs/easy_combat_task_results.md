# Easy Combat Task Results

## 1. Why switch to easy combat

Oracle-pretrain plus 200k fine-tuning did not pass the combat-pilot gate on
the normal 3v2 geometry. The best checkpoint was still a survival policy:
`red_missiles_fired_mean=0`, `red_missile_hits_mean=0`, and
`blue_dead_mean=0` on 3v2. The latest checkpoint showed some attack behavior,
but MAV survival collapsed.

The easy combat task is therefore a minimal curriculum step: shorten the
initial approach geometry so the learned policy can first learn
approach-and-fire before returning to the normal geometry.

## 2. What changed

Config:

`uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_easy_combat_f16_mav_surrogate.yaml`

The task keeps the same 3v2 aircraft and role setup but changes only initial
geometry:

- red UAVs start roughly 7.8 km south of the corresponding blue UAVs;
- red UAVs initially face the blue side;
- blue UAVs initially face the red side;
- the MAV starts behind the red UAV line and slightly higher, rather than in
  the center of the merge.

## 3. What did not change

- reward remains `happo_ref_v0`;
- observation remains `mav_shared_geo`;
- actor observation dimension remains `96`;
- critic state dimension remains `480`;
- action remains high-level `[pitch, heading, speed]`;
- missile launch and missile dynamics are unchanged;
- PID and aircraft XML are unchanged;
- red_0 remains MAV role, F-16 surrogate, and unarmed;
- red_1/red_2 remain F-16 UAVs with 2 missiles each;
- blue remains 2 F-16 attack UAVs.

## 4. 100k training result

Command:

```powershell
python scripts/run_happo_easy_combat_100k.py
```

The run completed:

- `total_env_steps_actual=100000`;
- `num_envs=4`;
- `rollout_length_per_env=256`;
- `transitions_per_rollout=1024`;
- `init_checkpoint=outputs/oracle_pretrain/uav_actor_oracle_pretrained/model.pt`;
- `nan_detected=false`.

The final training-log row showed a survival-oriented trend:

- `red_win=0.55`;
- `blue_win=0.09`;
- `draw=0.36`;
- `timeout=1.00`;
- `mav_survival=0.99`;
- `red_alive_final=2.46`;
- `blue_alive_final=2.00`.

## 5. Fast eval result

Fast eval was run with:

```powershell
python scripts/evaluate_happo_3v2_reference_checkpoints.py --output-dir outputs/happo_easy_combat_100k --fast --checkpoint-mode all --configs uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_easy_combat_f16_mav_surrogate.yaml
```

Best checkpoint, 20 episodes:

- `red_win_rate=0.55`;
- `blue_win_rate=0.00`;
- `draw_rate=0.45`;
- `timeout_rate=1.00`;
- `mav_survival_rate=1.00`;
- `red_missiles_fired_mean=0.00`;
- `red_missile_hits_mean=0.00`;
- `blue_dead_mean=0.00`.

Latest checkpoint, 20 episodes:

- `red_win_rate=1.00`;
- `blue_win_rate=0.00`;
- `draw_rate=0.00`;
- `timeout_rate=1.00`;
- `mav_survival_rate=1.00`;
- `red_missiles_fired_mean=0.00`;
- `red_missile_hits_mean=0.00`;
- `blue_dead_mean=0.00`.

The fast eval did not show red attack signal. The wins are timeout alive
advantage, not combat kills.

## 6. 50-episode eval result

Not run. The requested gate for running 50 episodes was attack signal in fast
eval, but both best and latest had zero red fire, zero red hit, and zero blue
death.

## 7. ACMI observation

Best 3v2 ACMI was exported:

`outputs/happo_easy_combat_100k/acmi/best_3v2_episode0.acmi`

Summary:

- `outcome=timeout`;
- `red_alive_final=3`;
- `blue_alive_final=2`;
- `mav_alive=true`;
- `red_missiles_fired=0`;
- `red_missile_hits=0`;
- `missiles_fired=0`;
- `missile_hits=0`.

The ACMI summary does not show red UAV firing or hitting.

## 8. Decision rule

Easy task success requires at least one of best/latest 3v2 checkpoints to meet:

- `red_missiles_fired_mean > 0.5`;
- `red_missile_hits_mean > 0.1` or `blue_dead_mean > 0.1`;
- `mav_survival_rate >= 0.3`;
- `blue_win_rate < 0.9`;
- not purely timeout alive advantage.

If it passes, continue easy combat to 200k and then gradually restore the
normal geometry. If it fails, do not run 1M; record that HAPPO reference v0
does not transfer oracle imitation into closed-loop combat under the current
interface.

Current result: `easy_task_success=false`. Do not run 200k from this checkpoint
without changing the task or policy prior. Do not return to normal geometry yet.

## 9. Oracle-pretrained closed-loop diagnosis

The first 100k easy-combat run used the original oracle checkpoint trained from
the normal-geometry direct-chase dataset. A direct closed-loop evaluation of
that checkpoint in easy combat showed no red fire:

- deterministic + fixed safe MAV: `red_missiles_fired_mean=0.00`;
- stochastic + fixed safe MAV: `red_missiles_fired_mean=0.00`;
- deterministic + MAV policy: `red_missiles_fired_mean=0.00`.

The direct-chase oracle itself can fire and hit in easy combat, so the failure
was in the learned imitation policy, not in the fire-control chain.

The root cause was a pretrain/action-scaling issue: the oracle pretrain used
plain MSE on absolute heading even though the heading action is circular.
The loss now uses wrapped heading error for the heading dimension.

## 10. Easy-combat oracle checkpoint and 50k anchor

A new easy-combat oracle dataset and checkpoint were created:

- dataset: `outputs/direct_chase_oracle_dataset/direct_chase_oracle_3v2_easy_combat.npz`;
- samples: `17988`;
- checkpoint: `outputs/oracle_pretrain/uav_actor_oracle_pretrained_easy_combat/model.pt`;
- wrapped action-match MSE: `0.010325`.

The new checkpoint can fire in closed loop when MAV is held to the fixed safe
action:

- deterministic: `red_missiles_fired_mean=1.10`;
- stochastic: `red_missiles_fired_mean=0.25`.

A default-off UAV imitation anchor was then used for a 50k easy-combat run:

```powershell
python scripts/run_happo_easy_combat_oracle_anchor_50k.py
```

Fast eval, 20 episodes:

- best checkpoint: `red_win_rate=0.85`, `mav_survival_rate=1.00`,
  `red_missiles_fired_mean=1.45`, `red_missile_hits_mean=1.30`,
  `blue_dead_mean=1.30`;
- latest checkpoint: `red_win_rate=0.95`, `mav_survival_rate=1.00`,
  `red_missiles_fired_mean=1.70`, `red_missile_hits_mean=1.55`,
  `blue_dead_mean=1.50`.

This is the first learned-policy easy-combat result with consistent red fire,
hits, blue deaths, and MAV survival. It does not yet prove normal-geometry
combat transfer or 5v4 zero-shot transfer.

50-episode eval was then run because fast eval showed attack signal:

- best checkpoint: `red_win_rate=0.86`, `mav_survival_rate=0.92`,
  `red_missiles_fired_mean=1.32`, `red_missile_hits_mean=1.16`,
  `blue_dead_mean=1.16`;
- latest checkpoint: `red_win_rate=1.00`, `mav_survival_rate=1.00`,
  `red_missiles_fired_mean=1.50`, `red_missile_hits_mean=1.48`,
  `blue_dead_mean=1.48`.

The regenerated `final_decision.json` marks `easy_task_success=true`. The
general combat-pilot gate remains false because this evaluation only covers
the easy 3v2 task and does not include 5v4 zero-shot.

## 11. Return to normal geometry

The next run returned to normal 3v2 geometry using the wrapped-heading normal
oracle checkpoint and the same UAV imitation anchor:

- output: `outputs/happo_normal_geometry_oracle_anchor_100k`;
- total steps: `100000`;
- normal 3v2 fast eval latest: `red_missiles_fired_mean=0.05`,
  `red_missile_hits_mean=0.00`, `blue_dead_mean=0.00`,
  `mav_survival_rate=1.00`;
- 5v4 latest 20-episode check: `red_missiles_fired_mean=0.75`,
  `red_missile_hits_mean=0.50`, `blue_dead_mean=0.50`,
  `mav_survival_rate=0.00`.

Conclusion: easy geometry is learnable, but direct transfer back to normal
geometry is not yet successful. The next allowed step is a normal-geometry
curriculum that gradually restores distance and heading from the easy spawn.
