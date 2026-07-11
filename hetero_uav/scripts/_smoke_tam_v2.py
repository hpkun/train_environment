"""Quick TAM 256-step smoke to verify rich logging fields."""
import sys, torch, numpy as np, csv, math, shutil
from pathlib import Path
sys.path.insert(0, '.')
from uav_env import make_env
from algorithms.pure_happo import PureHAPPOPolicy
from algorithms.pure_happo.trainer import PureHAPPOTrainer
from algorithms.happo.rollout_safety import sanitize_policy_inputs, zero_inactive_actions
from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
from algorithms.mappo.opponent_policy import OpponentPolicy
from scripts.rich_logging import RichExperimentLogger

device = torch.device('cpu'); adapter = HeteroObsAdapterV2(max_red=5, max_blue=4)
smoke_dir = Path('outputs/_smoke_tam_v2')
rich_dir = smoke_dir / 'rich_logs'
shutil.rmtree(str(smoke_dir), ignore_errors=True)
rich_dir.mkdir(parents=True, exist_ok=True)
logger = RichExperimentLogger(rich_dir, run_id='smoke_tam', method_name='pure_happo_v2',
    scenario_name='tam_3v2', device='cpu', num_envs=1, rollout_length_per_env=256,
    transitions_per_rollout=256, mode='summary')

env = make_env('uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_brma_tam_scale_aligned_v2.yaml', max_steps=1000)
policy = PureHAPPOPolicy(actor_obs_dim=140, critic_state_dim=700, num_agents=3).to(device)
trainer = PureHAPPOTrainer(policy, actor_lr=1e-4, critic_lr=5e-4, clip_param=0.2, entropy_coef=0.01,
    value_coef=0.5, max_grad_norm=10.0, ppo_epochs=3, critic_epochs=5, gamma=0.99, gae_lambda=0.95, seed=1)

obs, info = env.reset(seed=1)
opponent = OpponentPolicy(mode='tam_greedy_rule', seed=200)
all_actor_obs=[]; all_critic=[]; all_actions=[]; all_logprobs=[]
all_rewards=[]; all_dones=[]; all_values=[]; all_active=[]; nv_list=[]; eids=[]

for step in range(256):
    adapted = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
    active = np.zeros(3, dtype=np.float32)
    for i, rid in enumerate(env.red_ids):
        sim = env.red_planes.get(rid); active[i] = 1.0 if sim and sim.is_alive else 0.0
    obs_np = np.stack([adapted['actor_obs'].get(rid, np.zeros(140, dtype=np.float32)) for rid in env.red_ids])
    san = sanitize_policy_inputs(obs_np, active, critic_state=adapted['critic_state'])
    with torch.no_grad():
        out = policy.act(torch.as_tensor(san['actor_obs'], device=device),
                        critic_state=torch.as_tensor(san.get('critic_state', adapted['critic_state']), device=device),
                        deterministic=False)
    acts = zero_inactive_actions(out['action'].cpu().numpy(), active)
    ad = {rid: acts[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}
    ad.update(opponent.act(obs, env.blue_ids, env=env))
    obs, rewards, term, trunc, info = env.step(ad)
    td = all(term.values()) or all(trunc.values())
    all_actor_obs.append(obs_np); all_critic.append(adapted['critic_state']); all_actions.append(acts)
    all_logprobs.append(out['log_prob'].cpu().numpy())
    all_rewards.append(np.array([float(rewards.get(rid,0)) for rid in env.red_ids], dtype=np.float32))
    all_dones.append(np.full(3, float(td), dtype=np.float32))
    all_values.append(float(out['value'].item()) if out['value'] is not None else 0.0)
    all_active.append(active)
    nv_list.append(float(policy.value(torch.as_tensor(adapted['critic_state'], device=device)).item()))
    eids.append(0)
    if td: obs, info = env.reset(seed=1+step); opponent.reset_memory()

T = len(all_actor_obs)
buf = type('B',(),{'get':lambda s,d: {
    'actor_obs':torch.as_tensor(np.stack(all_actor_obs),dtype=torch.float32,device=d),
    'critic_state':torch.as_tensor(np.stack(all_critic),dtype=torch.float32,device=d),
    'actions':torch.as_tensor(np.stack(all_actions),dtype=torch.float32,device=d),
    'old_log_probs':torch.as_tensor(np.stack(all_logprobs),dtype=torch.float32,device=d),
    'rewards':torch.as_tensor(np.stack(all_rewards),dtype=torch.float32,device=d),
    'dones':torch.as_tensor(np.stack(all_dones),dtype=torch.float32,device=d),
    'values':torch.as_tensor(np.array(all_values,dtype=np.float32).reshape(T),device=d),
    'active_masks':torch.as_tensor(np.stack(all_active),dtype=torch.float32,device=d),
    'next_values':torch.as_tensor(np.array(nv_list,dtype=np.float32).reshape(T),device=d),
    'env_ids':torch.as_tensor(np.array(eids),dtype=torch.long,device=d),
    'role_ids':None,'actor_entity_tokens':None,'rnn_hidden':None}})()
res = trainer.update(buf)

final_keys = ['final_approx_kl_mav','final_approx_kl_uav','final_approx_kl_abs_mav','final_approx_kl_abs_uav',
    'final_clip_fraction_mav','final_clip_fraction_uav','final_ratio_std_mav','final_ratio_std_uav',
    'final_ratio_p95_mav','final_ratio_p95_uav','final_ratio_p99_mav','final_ratio_p99_uav',
    'final_actor_parameter_delta_mav','final_actor_parameter_delta_uav',
    'effective_scale_v2_total','effective_scale_v2_component_sum','effective_scale_v2_identity_error',
    'effective_scale_v2_uav_progress','effective_scale_v2_uav_event',
    'effective_scale_v2_mav_role','effective_scale_v2_mav_event','effective_scale_v2_terminal']

logger.write_train_metrics({
    'total_env_steps_actual': 256, 'train_steps': 256,
    'avg_episode_return': 0, 'avg_team_reward': 0, 'avg_mav_reward': 0, 'avg_uav_reward': 0,
    'red_win_rate': 0, 'blue_win_rate': 0, 'draw_rate': 0, 'timeout_rate': 0,
    'red_elimination_win_rate': 0, 'red_timeout_alive_advantage_rate': 0,
    'mav_survival_rate': 0, 'red_alive_final_mean': 0, 'blue_alive_final_mean': 0,
    'red_missiles_fired_mean': 0, 'blue_missiles_fired_mean': 0,
    'red_missile_hits_mean': 0, 'blue_missile_hits_mean': 0,
    'red_dead_mean': 0, 'blue_dead_mean': 0, 'kill_death_ratio': 0, 'relative_win_ratio': 0,
    'actor_loss': 0, 'critic_loss': res.get('critic_loss',0), 'entropy': 0,
    'policy_gradient_norm': 0, 'value_gradient_norm': 0, 'action_saturation_rate': 0,
    'mav_action_saturation_rate': 0, 'uav_action_saturation_rate': 0,
    'approx_kl_mav': res.get('approx_kl_mav',0), 'approx_kl_uav': res.get('approx_kl_uav',0),
    'mask_keep_ratio': 1, 'mask_entropy': 0, 'masked_entity_count': 0, 'nan_detected': 0,
    **{k: res.get(k, 0) for k in final_keys},
})
logger.close()
env.close()

# Verify rich CSV
rows = list(csv.DictReader(open(str(rich_dir / 'train_metrics.csv'), encoding='utf-8')))
r = rows[-1]
all_ok = True
id_err = abs(float(r.get('effective_scale_v2_identity_error', 999)))
if id_err > 1e-6:
    print(f'IDENTITY ERROR: {id_err}'); all_ok = False
for k in final_keys:
    v = r.get(k, 'MISSING')
    if v in ('MISSING', '', None):
        print(f'MISSING: {k}'); all_ok = False
    else:
        try:
            fv = float(v)
            if not math.isfinite(fv): print(f'NONFINITE: {k}={v}'); all_ok = False
        except: print(f'NOT_FLOAT: {k}={v}'); all_ok = False
print(f'TAM smoke: {"PASS" if all_ok else "FAIL"} (id_error={id_err:.10f})')
for k in final_keys[:8]:
    print(f'  {k}={r.get(k, "?")}')
shutil.rmtree(str(smoke_dir), ignore_errors=True)
