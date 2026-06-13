# Oracle-Pretrain Fine-Tune Results

## 1. Why oracle imitation

The previous HAPPO reference v0 runs showed survival behavior but weak combat behavior. The red direct-chase oracle already demonstrated that the environment can support red UAV firing, missile hits, and blue kills when the policy closes the attack geometry. Therefore, the next minimal intervention is to inject approach-and-fire behavior into the shared UAV actor through behavior cloning.

This is not TAM-HAPPO, not GRU, not attention, and not a new reward version.

## 2. Dataset collection

Dataset script:

```powershell
python scripts/collect_direct_chase_oracle_dataset.py --episodes 50
```

Default output:

- `outputs/direct_chase_oracle_dataset/direct_chase_oracle_3v2.npz`
- `outputs/direct_chase_oracle_dataset/direct_chase_oracle_3v2_summary.json`

The dataset stores only red UAV samples, not MAV attack samples:

- `actor_obs`: V2 actor observation, 96 dimensions;
- `oracle_action`: direct-chase action, 3 dimensions;
- launch range / angle / envelope flags;
- missile fired / hit step indicators;
- episode and step metadata.

## 3. UAV shared actor pretraining

Pretrain script:

```powershell
python scripts/pretrain_uav_actor_from_oracle.py --epochs 20
```

Default behavior:

- load `HAPPOReferencePolicy`;
- initialize from `outputs/happo_3v2_reference_f16_mav_surrogate_1m_fast/best/model.pt`;
- train only the shared UAV actor path;
- freeze MAV actor;
- freeze critic;
- optimize `MSE(policy_mean, oracle_action)`;
- save a full policy state dict to `outputs/oracle_pretrain/uav_actor_oracle_pretrained/model.pt`.

## 4. MAV actor is not imitated for attack

The MAV actor is intentionally not behavior-cloned from direct-chase attack samples. The MAV role remains survival/support oriented, and the MAV remains unarmed in the current role setup.

## 5. 200k fine-tune result

Run:

```powershell
python scripts/run_happo_oracle_pretrain_finetune_200k.py
```

The completed run was stable (`nan_detected=false`) and reached
`total_env_steps_actual=200000`, but it did not pass the combat-pilot gate.
The best 3v2 checkpoint remained a survival policy:

- `red_missiles_fired_mean=0`;
- `red_missile_hits_mean=0`;
- `blue_dead_mean=0`;
- `mav_survival_rate=0.83`.

The latest 3v2 checkpoint produced some attack signal, but MAV survival
collapsed:

- `red_missiles_fired_mean=1.07`;
- `red_missile_hits_mean=0.15`;
- `blue_dead_mean=0.15`;
- `mav_survival_rate=0.0`.

Conclusion: oracle imitation alone did not reliably transfer to normal
closed-loop 3v2 combat while preserving MAV survival.

## 6. 3v2 seen result

After the 200k run, evaluate with:

```powershell
python scripts/evaluate_happo_3v2_reference_checkpoints.py --output-dir outputs/happo_oracle_pretrain_finetune_200k --episodes 100 --checkpoint-mode all
```

## 7. 5v4 zero-shot result

The evaluation script uses the checkpoint metadata to keep the trained 3v2 config as the seen config and evaluates 5v4 as the zero-shot scale-transfer config.

## 8. Red fire / hit / blue death

The red fire chain is still valid. The oracle dataset produced
`red_missiles_fired_mean=2.38`, `red_missile_hits_mean=1.94`, and
`blue_dead_mean=1.94`. The learned policy did not consistently reproduce this
under normal 3v2 geometry, so the next step is the easy combat task rather
than 1M training.

The final decision will read:

- `red_missiles_fired_mean`;
- `red_missile_hits_mean`;
- `blue_dead_mean`;
- `mav_survival_rate`;
- `blue_win_rate`.

## 9. Better than survival baseline

Pending real run.

The expected improvement over the survival baseline is not return alone. The key check is whether red UAVs fire and hit more often than the HAPPO reference v0 survival baseline.

## 10. Whether to enter 1M

Decision rule:

- run 1M only if `usable_as_combat_pilot = true`;
- otherwise do not run 1M.

The generated decision files are:

- `outputs/happo_oracle_pretrain_finetune_200k/final_decision.json`
- `outputs/happo_oracle_pretrain_finetune_200k/final_decision.md`

## 11. If it fails

If oracle-pretrain 200k does not pass the combat-pilot gate, the next step should only be an easy combat task: shorten initial distance and adjust initial heading so learned policy first acquires approach-and-fire behavior.

## 12. Closed-loop oracle-pretrain diagnosis

The original oracle-pretrained checkpoint was evaluated directly in the easy
combat closed loop before PPO fine-tuning:

- deterministic UAV actor + fixed safe MAV action: `red_missiles_fired_mean=0.00`;
- stochastic UAV actor + fixed safe MAV action: `red_missiles_fired_mean=0.00`;
- deterministic UAV actor + MAV policy action: `red_missiles_fired_mean=0.00`.

Action-match on the original dataset showed the checkpoint did load and the
UAV actor did learn the supervised samples in aggregate:

- raw `mse_mean_action_vs_oracle=0.075882`;
- `cosine_similarity=0.904774`.

The closed-loop failure was therefore not a missile-chain failure and not a
checkpoint-loading failure. The issue was that the original pretrain used
plain MSE on absolute heading. Since heading action is circular (`-1` and `+1`
both represent headings near ±pi), plain MSE over-penalized wrap-boundary
samples and could pull the learned heading toward the wrong side.

The pretrain loss now uses wrapped heading error for the heading dimension
while leaving pitch and speed as ordinary regression targets.

## 13. Easy-combat oracle checkpoint

An easy-combat-specific oracle dataset was collected:

- dataset: `outputs/direct_chase_oracle_dataset/direct_chase_oracle_3v2_easy_combat.npz`;
- samples: `17988`;
- oracle red fire mean: `2.16`;
- oracle red hit mean: `2.00`;
- oracle blue death mean: `2.00`;
- launch envelope rate: `0.2112`.

A new UAV actor checkpoint was pretrained from this dataset:

- checkpoint: `outputs/oracle_pretrain/uav_actor_oracle_pretrained_easy_combat/model.pt`;
- loss: `wrapped_heading_mse`;
- final validation loss: `0.010088`;
- wrapped action-match MSE: `0.010325`.

Closed-loop evaluation of this new checkpoint showed that the initial
pretrained UAV actor can fire before PPO fine-tuning:

- deterministic UAV actor + fixed safe MAV action:
  `red_missiles_fired_mean=1.10`;
- stochastic UAV actor + fixed safe MAV action:
  `red_missiles_fired_mean=0.25`;
- deterministic UAV actor + MAV policy action:
  `red_missiles_fired_mean=0.00`.

This means the UAV pretrain can produce attack behavior, but the full
policy-driven MAV/UAV closed loop can still disrupt launch geometry.

## 14. UAV imitation anchor 50k

Because the initial UAV pretrain could fire but previous PPO fine-tuning washed
out the attack behavior, a default-off UAV imitation anchor was added to the
HAPPO trainer. It applies only to the shared UAV actor and only when explicitly
enabled:

- `--uav-imitation-dataset`;
- `--uav-imitation-coef`;
- `--uav-imitation-until-steps`;
- `--uav-imitation-batch-size`.

The 50k easy-combat anchor run completed without NaN:

- output: `outputs/happo_easy_combat_oracle_anchor_50k`;
- `total_env_steps_actual=50000`;
- `num_envs=4`;
- `uav_imitation_coef=0.1`;
- `uav_imitation_until_steps=50000`.

Fast evaluation, 20 episodes:

- best checkpoint: `red_win_rate=0.85`, `mav_survival_rate=1.00`,
  `red_missiles_fired_mean=1.45`, `red_missile_hits_mean=1.30`,
  `blue_dead_mean=1.30`;
- latest checkpoint: `red_win_rate=0.95`, `mav_survival_rate=1.00`,
  `red_missiles_fired_mean=1.70`, `red_missile_hits_mean=1.55`,
  `blue_dead_mean=1.50`.

Conclusion: the anchor fixes the immediate “no red fire after fine-tuning”
failure on the easy-combat task. This is still an easy-task result, not a
normal-geometry 3v2/5v4 zero-shot combat result.

Follow-up 50-episode eval:

- best checkpoint: `red_win_rate=0.86`, `mav_survival_rate=0.92`,
  `red_missiles_fired_mean=1.32`, `red_missile_hits_mean=1.16`,
  `blue_dead_mean=1.16`;
- latest checkpoint: `red_win_rate=1.00`, `mav_survival_rate=1.00`,
  `red_missiles_fired_mean=1.50`, `red_missile_hits_mean=1.48`,
  `blue_dead_mean=1.48`.
