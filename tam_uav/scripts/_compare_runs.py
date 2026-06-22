"""One-shot comparison of paper-hparams vs per-agent 100k runs."""
import csv, json, os

def get_metrics(run_dir):
    rows = list(csv.DictReader(open(f'{run_dir}/train_log.csv')))
    r = rows[-1]
    eval_rows = list(csv.DictReader(open(f'{run_dir}/eval_log.csv')))
    status = json.load(open(f'{run_dir}/runner_status.json'))
    evals = {}
    for e in eval_rows:
        cfg = e['scenario']
        evals[cfg] = {
            'steps': e['total_steps'], 'red_win': e['red_win_rate'],
            'mav_surv': e['mav_survival_rate'],
            'uav_fired': e.get('red_uav_fired_mean',''),
            'uav_hits': e.get('red_uav_hits_mean',''),
        }
    return {
        'run_dir': run_dir, 'completed': status['runner_completed_normally'],
        'total_steps': status['total_env_steps_actual'], 'nan': status['nan_detected'],
        'final_return': float(r['avg_return']),
        'red_win': float(r['red_win']), 'blue_win': float(r['blue_win']),
        'mav_surv': float(r['mav_survival']),
        'red_alive': float(r['red_alive_final']), 'blue_alive': float(r['blue_alive_final']),
        'mav_death_step_mean': r.get('mav_death_step_mean_recent',''),
        'mav_crash_lowalt_rate': r.get('mav_crash_lowalt_rate_recent',''),
        'red_fired': int(r['red_missiles_fired']), 'hits': int(r['missile_hits']),
        'uav_fired': r.get('red_uav_fired_rollout',''), 'uav_hits': r.get('red_uav_hits_rollout',''),
        'lm': float(r['actor_loss_mav']), 'lu': float(r['actor_loss_uav']),
        'kl_mav': float(r['approx_kl_mav']), 'em': float(r['entropy_mav']),
        'advantage_mode': r.get('advantage_mode','old_csv'),
        'dilution': r.get('dilution_ratio_abs','old_csv'),
        'per_agent_enabled': r.get('per_agent_advantage_enabled','old_csv'),
        'dominant_throttle_mav': r.get('dominant_bin_mav_throttle',''),
        'dominant_aileron_mav': r.get('dominant_bin_mav_aileron',''),
        'dominant_elevator_mav': r.get('dominant_bin_mav_elevator',''),
        'dominant_rudder_mav': r.get('dominant_bin_mav_rudder',''),
        'max_prob_mav': r.get('max_action_prob_mav',''),
        'bin_usage_mav': r.get('action_bin_usage_mav',''),
        'edge_bin': r.get('edge_bin_rate',''),
        'evals': evals,
    }

paper = get_metrics('outputs/tam_papermode_paperhparams_3v2_100k_val')
peragent = get_metrics('outputs/tam_advantage_peragent_3v2_100k_val')

out_dir = 'outputs/tam_advantage_ablation_comparison'
os.makedirs(out_dir, exist_ok=True)

improvements = []
if peragent['red_fired'] > 0 and paper['red_fired'] == 0:
    improvements.append('Red fired: 0 -> >0')
if peragent['mav_death_step_mean'] and paper['mav_death_step_mean']:
    if float(peragent['mav_death_step_mean']) > float(paper['mav_death_step_mean']) + 20:
        improvements.append(f"MAV death step: {paper['mav_death_step_mean']} -> {peragent['mav_death_step_mean']}")
if float(peragent['mav_surv']) > float(paper['mav_surv']) + 0.05:
    improvements.append(f"MAV survival: {paper['mav_surv']:.2f} -> {peragent['mav_surv']:.2f}")
if float(peragent['final_return']) > float(paper['final_return']) + 2:
    improvements.append(f"Return: {paper['final_return']:.1f} -> {peragent['final_return']:.1f}")

if len(improvements) >= 2:
    verdict = 'SUBSTANTIVE: per_agent_reward improves multiple metrics'
elif len(improvements) == 1:
    verdict = 'MARGINAL: per_agent_reward improves one metric only'
else:
    verdict = 'INSUFFICIENT: per_agent_reward fixes signal scale but does not improve outcomes'

comparison = {
    'paper_hparams_team_average': paper,
    'per_agent_reward_ablation': peragent,
    'judgment': {'improvements': improvements, 'verdict': verdict,
                 'next_step': ('ppo_epochs=4 ablation with per_agent_reward' if improvements else
                               'F22 action stability + direct-FCS calibration audit')},
}
json.dump(comparison, open(f'{out_dir}/comparison.json','w'), indent=2)

md = ['# Advantage Mode Ablation Comparison', '',
      '## Paper-hparams team_average 100k',
      f"completed={paper['completed']} steps={paper['total_steps']} nan={paper['nan']}",
      f"return={paper['final_return']:.1f} rw={paper['red_win']:.2f} bw={paper['blue_win']:.2f}",
      f"mav_surv={paper['mav_surv']:.2f} death_step={paper['mav_death_step_mean']}",
      f"crash_lowalt={paper['mav_crash_lowalt_rate']}",
      f"red_fired={paper['red_fired']} hits={paper['hits']}",
      f"uav_fired={paper['uav_fired']} uav_hits={paper['uav_hits']}",
      f"lm={paper['lm']:.4f} kl_mav={paper['kl_mav']:.6f} em={paper['em']:.2f}",
      f"adv_mode={paper['advantage_mode']} dilution={paper['dilution']}",
      f"dominant_mav=[{paper['dominant_throttle_mav']},{paper['dominant_aileron_mav']},{paper['dominant_elevator_mav']},{paper['dominant_rudder_mav']}]",
      f"max_prob={paper['max_prob_mav']} bin_usage={paper['bin_usage_mav']} edge={paper['edge_bin']}",
      'Evals:']
for cfg, e in paper['evals'].items():
    md.append(f"  {cfg}: rw={e['red_win']} mav={e['mav_surv']} uav_f={e['uav_fired']} uav_h={e['uav_hits']}")

md += ['', '## Per-agent reward 100k',
       f"completed={peragent['completed']} steps={peragent['total_steps']} nan={peragent['nan']}",
       f"return={peragent['final_return']:.1f} rw={peragent['red_win']:.2f} bw={peragent['blue_win']:.2f}",
       f"mav_surv={peragent['mav_surv']:.2f} death_step={peragent['mav_death_step_mean']}",
       f"crash_lowalt={peragent['mav_crash_lowalt_rate']}",
       f"red_fired={peragent['red_fired']} hits={peragent['hits']}",
       f"uav_fired={peragent['uav_fired']} uav_hits={peragent['uav_hits']}",
       f"lm={peragent['lm']:.4f} kl_mav={peragent['kl_mav']:.6f} em={peragent['em']:.2f}",
       f"adv_mode={peragent['advantage_mode']} dilution={peragent['dilution']}",
       f"dominant_mav=[{peragent['dominant_throttle_mav']},{peragent['dominant_aileron_mav']},{peragent['dominant_elevator_mav']},{peragent['dominant_rudder_mav']}]",
       f"max_prob={peragent['max_prob_mav']} bin_usage={peragent['bin_usage_mav']} edge={peragent['edge_bin']}",
       'Evals:']
for cfg, e in peragent['evals'].items():
    md.append(f"  {cfg}: rw={e['red_win']} mav={e['mav_surv']} uav_f={e['uav_fired']} uav_h={e['uav_hits']}")

md += ['', '## Judgment', '',
       f'Improvements: {len(improvements)}'] + [f'- {i}' for i in improvements] + [
       '', f'**Verdict: {verdict}**', '',
       f'Next: {comparison["judgment"]["next_step"]}']

with open(f'{out_dir}/comparison.md','w') as f:
    f.write('\n'.join(md))
print('\n'.join(md))
