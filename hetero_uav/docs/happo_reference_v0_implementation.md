# HAPPO Reference v0 Implementation

## Scope

HAPPO reference v0 is a runnable environment-validation baseline. It is not the
final paper method and not full TAM-HAPPO.

## Policy

- MAV actor: `actor_obs_dim -> 256 -> Tanh -> 128 -> Tanh -> action_dim`.
- UAV actor: `actor_obs_dim -> 256 -> Tanh -> 128 -> Tanh -> action_dim`.
- Centralized critic: `critic_state_dim -> 256 -> Tanh -> 128 -> Tanh -> 1`.
- Gaussian continuous action distribution with separate MAV/UAV log std.
- Action dimension remains 3: `[pitch, heading, speed]`.
- Mean is clamped to `[-0.999, 0.999]`; sampled action is clamped to `[-1, 1]`.

## Role Mapping

Explicit role ids take priority. By convention:

- `red_0` is MAV;
- other red agents are UAVs.

When explicit roles are not provided, the policy can infer MAV/UAV from the
role one-hot segment in actor observation indices `7:11`.

## Buffer

The rollout buffer stores actor observations, centralized critic state,
actions, old log probabilities, rewards, repeated team dones, values,
active-agent masks, and role ids.

Dead red agents are removed from actor loss through `active_masks`. Individual
death does not truncate centralized GAE.

## Trainer

The trainer performs:

- centralized critic update;
- MAV actor update;
- UAV actor update.

This is a simplified HAPPO-style v0 sequential update. It preserves separate
role update phases but does not implement strict full HAPPO correction factors.

## Reward Mode

`happo_ref_v0` is an explicit additive role overlay. It does not replace
`brma_legacy` unless selected by config or CLI.

Recorded components include MAV survival/support/event/safety/death terms and
UAV attack-window/fire/hit/dodge/event/safety/death terms.

## Scripts

- `scripts/train_happo_reference.py`
- `scripts/eval_happo_reference.py`
- `scripts/smoke_happo_3v2_reference.py`
- `scripts/run_happo_3v2_reference_200k.py`
- `scripts/run_happo_3v2_reference_1m_fast.py`

ACMI export is skipped in v0. It can be added after the smoke and 200k
validation path is stable.

## Known Limitations

- No attention.
- No GRU or temporal module.
- No full TAM-HAPPO reproduction.
- No low-level 4D action conversion.
- Missile launch and evasion remain scripted environment mechanics.
