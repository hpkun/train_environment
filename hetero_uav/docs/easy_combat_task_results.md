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
