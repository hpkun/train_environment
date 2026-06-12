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

The implementation path is ready:

```powershell
python scripts/run_happo_oracle_pretrain_finetune_200k.py
```

In this Codex execution environment, the real JSBSim data collection and 200k fine-tune were not run because the active Python environment lacks `gymnasium/jsbsim`, and `conda run -n brmamappo ...` was blocked by sandbox permission review timeout. No result numbers are claimed here.

## 6. 3v2 seen result

Pending real run.

After the 200k run, evaluate with:

```powershell
python scripts/evaluate_happo_3v2_reference_checkpoints.py --output-dir outputs/happo_oracle_pretrain_finetune_200k --episodes 100 --checkpoint-mode all
```

## 7. 5v4 zero-shot result

Pending real run.

The evaluation script uses the checkpoint metadata to keep the trained 3v2 config as the seen config and evaluates 5v4 as the zero-shot scale-transfer config.

## 8. Red fire / hit / blue death

Pending real run.

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

