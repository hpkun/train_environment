"""Quick 256-step v2 smoke test."""
import sys, torch, numpy as np, os, shutil
sys.path.insert(0, '.')
from uav_env import make_env
from algorithms.pure_happo import PureHAPPOPolicy
from algorithms.pure_happo.trainer import PureHAPPOTrainer
from algorithms.happo.rollout_safety import sanitize_policy_inputs, zero_inactive_actions
from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
from algorithms.mappo.opponent_policy import OpponentPolicy

device = torch.device('cpu')
adapter = HeteroObsAdapterV2(max_red=5, max_blue=4)
tmpdir = 'outputs/_smoke_v2_final2'
shutil.rmtree(tmpdir, ignore_errors=True)
os.makedirs(tmpdir, exist_ok=True)

env = make_env('uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_brma_tam_scale_aligned_v2.yaml', max_steps=1000)
policy = PureHAPPOPolicy(actor_obs_dim=140, critic_state_dim=700, num_agents=3).to(device)
trainer = PureHAPPOTrainer(policy, actor_lr=1e-4, critic_lr=5e-4, clip_param=0.2, entropy_coef=0.01, value_coef=0.5, max_grad_norm=10.0, ppo_epochs=3, critic_epochs=5, gamma=0.99, gae_lambda=0.95, seed=1)

obs, info = env.reset(seed=1)
opponent = OpponentPolicy(mode='brma_rule', seed=200)
all_actor_obs = []; all_critic = []; all_actions = []; all_logprobs = []
all_rewards = []; all_dones = []; all_values = []; all_active = []
next_values_list = []; env_ids = []

newly_dead_found = False
dead_before_found = False
v2_ident_errors = []

for step in range(256):
    adapted = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
    active = np.zeros(3, dtype=np.float32)
    for i, rid in enumerate(env.red_ids):
        sim = env.red_planes.get(rid)
        active[i] = 1.0 if (sim and sim.is_alive) else 0.0
    actor_obs_np = np.stack([adapted['actor_obs'].get(rid, np.zeros(140, dtype=np.float32)) for rid in env.red_ids])
    san = sanitize_policy_inputs(actor_obs_np, active, critic_state=adapted['critic_state'])
    with torch.no_grad():
        out = policy.act(torch.as_tensor(san['actor_obs'], device=device),
                        critic_state=torch.as_tensor(san.get('critic_state', adapted['critic_state']), device=device),
                        deterministic=False)
    actions_np = zero_inactive_actions(out['action'].cpu().numpy(), active)
    action_dict = {rid: actions_np[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}
    action_dict.update(opponent.act(obs, env.blue_ids, env=env))

    alive_before_ids = {rid for i, rid in enumerate(env.red_ids) if active[i] > 0.5}
    obs, rewards, term, trunc, info = env.step(action_dict)
    team_done = all(term.values()) or all(trunc.values())

    rc = info.get('reward_components', {})
    for rid in env.red_ids:
        sim = env.red_planes.get(rid)
        post_alive = bool(sim.is_alive) if sim else False
        if rid in alive_before_ids and not post_alive:
            newly_dead_found = True
            c = rc.get(rid, {})
            evt = float(c.get('scale_v2_event', 0) or 0)
            total = float(c.get('scale_v2_total', 0) or 0)
            if evt > -0.5:
                print(f'WARN: newly-dead {rid} has event={evt:.4f}')
        if rid not in alive_before_ids and not post_alive:
            dead_before_found = True
            total = float(rc.get(rid,{}).get('scale_v2_total', 0) or 0)
            if abs(total) > 1e-9:
                print(f'WARN: dead-before {rid} has non-zero total={total:.6f}')

    v2_ident_errors.extend([abs(float(rc.get(rid,{}).get('scale_v2_identity_error',0) or 0)) for rid in env.red_ids])

    all_actor_obs.append(actor_obs_np); all_critic.append(adapted['critic_state']); all_actions.append(actions_np)
    all_logprobs.append(out['log_prob'].cpu().numpy())
    all_rewards.append(np.array([float(rewards.get(rid, 0)) for rid in env.red_ids], dtype=np.float32))
    all_dones.append(np.full(3, float(team_done), dtype=np.float32))
    all_values.append(float(out['value'].item()) if out['value'] is not None else 0.0)
    all_active.append(active)
    next_values_list.append(float(policy.value(torch.as_tensor(adapted['critic_state'], device=device)).item()))
    env_ids.append(0)

    if team_done:
        obs, info = env.reset(seed=1 + step)
        opponent.reset_memory()

T = len(all_actor_obs)
buffer = type('Buf', (), {'get': lambda self, dev: {
    'actor_obs': torch.as_tensor(np.stack(all_actor_obs), dtype=torch.float32, device=dev),
    'critic_state': torch.as_tensor(np.stack(all_critic), dtype=torch.float32, device=dev),
    'actions': torch.as_tensor(np.stack(all_actions), dtype=torch.float32, device=dev),
    'old_log_probs': torch.as_tensor(np.stack(all_logprobs), dtype=torch.float32, device=dev),
    'rewards': torch.as_tensor(np.stack(all_rewards), dtype=torch.float32, device=dev),
    'dones': torch.as_tensor(np.stack(all_dones), dtype=torch.float32, device=dev),
    'values': torch.as_tensor(np.array(all_values, dtype=np.float32).reshape(T), device=dev),
    'active_masks': torch.as_tensor(np.stack(all_active), dtype=torch.float32, device=dev),
    'next_values': torch.as_tensor(np.array(next_values_list, dtype=np.float32).reshape(T), device=dev),
    'env_ids': torch.as_tensor(np.array(env_ids), dtype=torch.long, device=dev),
    'role_ids': None, 'actor_entity_tokens': None, 'rnn_hidden': None,
}})()
results = trainer.update(buffer)

max_id = max(v2_ident_errors) if v2_ident_errors else 0
finite_ok = all(np.isfinite(float(v)) for k,v in results.items() if isinstance(v, (int,float)) and k != 'critic_loss_per_epoch')
final_keys = ['final_approx_kl_mav','final_approx_kl_uav','final_approx_kl_abs_mav','final_approx_kl_abs_uav',
              'final_clip_fraction_mav','final_clip_fraction_uav','final_actor_parameter_delta_mav','final_actor_parameter_delta_uav']
final_ok = all(k in results for k in final_keys)
delta_ok = results.get('final_actor_parameter_delta_mav',0) > 0 and results.get('final_actor_parameter_delta_uav',0) > 0
nonfinite = sum(1 for k,v in results.items() if isinstance(v, (int,float)) and not np.isfinite(float(v)))

print(f'newly_dead_found={newly_dead_found} dead_before_found={dead_before_found}')
print(f'v2_identity_max={max_id:.10f} finite={finite_ok} final_keys={final_ok} delta>0={delta_ok}')
print(f'final_kl_mav={results.get("final_approx_kl_mav",0):.6f} final_kl_uav={results.get("final_approx_kl_uav",0):.6f}')
print(f'final_kl_abs_mav={results.get("final_approx_kl_abs_mav",0):.6f} final_kl_abs_uav={results.get("final_approx_kl_abs_uav",0):.6f}')
print(f'final_clip_mav={results.get("final_clip_fraction_mav",0):.4f} final_clip_uav={results.get("final_clip_fraction_uav",0):.4f}')
print(f'final_ratio_std_mav={results.get("final_ratio_std_mav",0):.4f} final_ratio_std_uav={results.get("final_ratio_std_uav",0):.4f}')
print(f'final_ratio_p95_mav={results.get("final_ratio_p95_mav",0):.4f} final_ratio_p95_uav={results.get("final_ratio_p95_uav",0):.4f}')
print(f'final_ratio_p99_mav={results.get("final_ratio_p99_mav",0):.4f} final_ratio_p99_uav={results.get("final_ratio_p99_uav",0):.4f}')
print(f'final_delta_mav={results.get("final_actor_parameter_delta_mav",0):.6f} final_delta_uav={results.get("final_actor_parameter_delta_uav",0):.6f}')
print(f'critic_loss={results.get("critic_loss",0):.6f}')
print(f'entropy_mav={results.get("entropy_mav",0):.4f} entropy_uav={results.get("entropy_uav",0):.4f}')
print(f'value_ev_old={results.get("value_explained_variance_old",0):.4f} value_ev_new={results.get("value_explained_variance_new",0):.4f}')
print(f'nonfinite_count={nonfinite}')

all_ok = (max_id < 1e-6 and finite_ok and final_ok and delta_ok and nonfinite == 0)
env.close()
shutil.rmtree(tmpdir, ignore_errors=True)
print(f'SMOKE_PASS={all_ok}')
print('READY_FOR_100K_PURE_HAPPO_LEARNABILITY_PROBE' if all_ok else 'NOT_READY')
