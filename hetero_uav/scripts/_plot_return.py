"""Plot detailed Return vs Steps for hetero_entity_recurrent F22 PID 500k training."""
import csv, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = 'outputs/hetero_entity_recurrent_f22_pid_500k_main'

# ── Read train_log ──
with open(f'{OUT}/train_log.csv') as f:
    reader = csv.DictReader(f)
    data = list(reader)

steps     = np.array([int(r['total_steps'])          for r in data], dtype=float)
ret       = np.array([float(r['avg_return'])         for r in data])
red_win   = np.array([float(r['red_win'])            for r in data])
blue_win  = np.array([float(r['blue_win'])           for r in data])
timeout   = np.array([float(r['timeout'])            for r in data])
mav_active= np.array([float(r['mav_active_sample_count']) for r in data])
loss_mav  = np.array([float(r['actor_loss_mav'])     for r in data])
loss_uav  = np.array([float(r['actor_loss_uav'])     for r in data])
entropy_mav = np.array([float(r['entropy_mav'])      for r in data])
entropy_uav = np.array([float(r['entropy_uav'])      for r in data])
critic_loss = np.array([float(r['critic_loss'])       for r in data])
missile_hits = np.array([float(r['missile_hits'])     for r in data])
nan_flag   = np.array([int(r['nan_detected'])         for r in data])
mav_surv   = np.array([float(r['mav_survival'])       for r in data])
red_missiles_fired = np.array([float(r['red_missiles_fired']) for r in data])

steps_k = steps / 1000
n = len(steps)
W = 40

def smooth(y, w=W):
    s = np.convolve(y, np.ones(w)/w, mode='valid')
    return s

def x_smooth(steps_arr, w=W):
    """Return x-axis values aligned with smoothed y values."""
    return steps_arr[w-1:]

s = slice(W-1, None)  # for W=40 only

# ── Read eval_log ──
eval_steps, eval_3v2_win, eval_3v2_hits, eval_5v4_win, eval_5v4_hits = [], [], [], [], []
with open(f'{OUT}/eval_log.csv') as f:
    for row in csv.DictReader(f):
        cfg = row['config']
        if '3v2' in cfg:
            eval_steps.append(int(row['total_steps'])/1000)
            eval_3v2_win.append(float(row['red_win_rate']))
            eval_3v2_hits.append(float(row['red_missile_hits_mean']))
        elif '5v4' in cfg:
            eval_5v4_win.append(float(row['red_win_rate']))
            eval_5v4_hits.append(float(row['red_missile_hits_mean']))

# ── Create detailed multi-panel figure ──
fig, axes = plt.subplots(3, 2, figsize=(20, 14))
fig.suptitle('F22 PID 500k -- Hetero Entity Recurrent HAPPO (brma_rule opponent)',
             fontsize=16, fontweight='bold', y=0.98)

# ── Panel 1: Return ──
ax = axes[0, 0]
ax.plot(steps_k, ret, alpha=0.10, color='#1f77b4', linewidth=0.3)
ax.plot(steps_k[s], smooth(ret), color='#1f77b4', linewidth=2.2, label=f'smoothed (w={W})')
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
ax.fill_between(steps_k, 0, ret, where=(ret>0), color='#2ca02c', alpha=0.05)
ax.fill_between(steps_k, 0, ret, where=(ret<=0), color='#d62728', alpha=0.04)
for es, ew3 in zip(eval_steps, eval_3v2_win):
    c = '#2ca02c' if ew3 >= 0.5 else '#d62728'
    ax.axvline(x=es, color=c, linestyle=':', linewidth=0.6, alpha=0.5)
r_min, r_max = np.min(ret), np.max(ret)
i_min, i_max = np.argmin(ret), np.argmax(ret)
ax.annotate(f'min {r_min:+.1f} @{steps_k[i_min]:.0f}k', xy=(steps_k[i_min], r_min),
            xytext=(steps_k[i_min]+30, r_min+20), fontsize=8, color='#d62728',
            arrowprops=dict(arrowstyle='->', color='#d62728', alpha=0.6))
ax.annotate(f'max {r_max:+.1f} @{steps_k[i_max]:.0f}k', xy=(steps_k[i_max], r_max),
            xytext=(steps_k[i_max]-60, r_max-15), fontsize=8, color='#2ca02c',
            arrowprops=dict(arrowstyle='->', color='#2ca02c', alpha=0.6))
# Phase boxes
for x0, x1, label, color in [
    (0, 70, 'early\npeak', '#2ca02c'),
    (70, 220, 'collapse', '#d62728'),
    (220, 350, 'recovery', '#ff7f0e'),
    (350, 510, 'stable plateau\n(timeout-survival)', '#1f77b4'),
]:
    ax.axvspan(x0, x1, alpha=0.04, color=color)
    ax.text((x0+x1)/2, -58, label, ha='center', fontsize=7, color=color, alpha=0.8,
            fontweight='bold')
ax.set_ylabel('Avg Episode Return', fontsize=11)
ax.set_title('Return vs Steps', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, loc='lower right')
ax.grid(True, alpha=0.2)
ax.set_xlim(0, steps_k[-1]*1.01)

# ── Panel 2: red_win + blue_win + timeout rates ──
ax = axes[0, 1]
ax.plot(steps_k[s], smooth(red_win), color='#2ca02c', linewidth=2.0, label='red_win')
ax.plot(steps_k[s], smooth(blue_win), color='#d62728', linewidth=2.0, label='blue_win')
ax.plot(steps_k[s], smooth(timeout), color='#ff7f0e', linewidth=2.0, label='timeout', linestyle='--')
ax.set_ylim(-0.05, 1.10)
ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=0.6, alpha=0.4)
ax.set_ylabel('Rate', fontsize=11)
ax.set_title('Win / Timeout Rates (smoothed)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='center right')
ax.grid(True, alpha=0.2)

# ── Panel 3: Eval red_win + hits ──
ax = axes[1, 0]
ax.plot(eval_steps, eval_3v2_win, 'o-', color='#1f77b4', linewidth=1.8, markersize=8,
        label='3v2 red_win')
ax.plot(eval_steps, eval_5v4_win, 's--', color='#ff7f0e', linewidth=1.8, markersize=8,
        label='5v4 red_win')
ax.plot(eval_steps, eval_3v2_hits, 'o-', color='#2ca02c', linewidth=1.5, markersize=6,
        alpha=0.7, label='3v2 missile_hits_mean')
ax.plot(eval_steps, eval_5v4_hits, 's--', color='#d62728', linewidth=1.5, markersize=6,
        alpha=0.7, label='5v4 missile_hits_mean')
ax.set_ylim(-0.1, max(max(eval_3v2_hits), max(eval_5v4_hits), 1.5) + 0.5)
ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=0.6, alpha=0.4)
ax.set_ylabel('Rate / Hits', fontsize=11)
ax.set_xlabel('Steps (x1000)', fontsize=11)
ax.set_title('Periodic Eval -- red_win & missile_hits', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, loc='best', ncol=2)
ax.grid(True, alpha=0.2)

# ── Panel 4: MAV active sample count ──
ax = axes[1, 1]
ax.plot(x_smooth(steps_k, w=20), smooth(mav_active, w=20), color='#9467bd', linewidth=2.0,
        label='mav_active_sample_count')
ax.fill_between(steps_k, 0, mav_active, alpha=0.12, color='#9467bd')
ax.axhline(y=128, color='gray', linestyle=':', linewidth=0.6, alpha=0.4,
           label='128 (half of rollout=256)')
ax.set_ylabel('Count', fontsize=11)
ax.set_xlabel('Steps (x1000)', fontsize=11)
ax.set_title('MAV Active Sample Count per Iteration', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.2)

# ── Panel 5: Actor losses ──
ax = axes[2, 0]
ax.plot(steps_k[s], smooth(loss_mav), color='#1f77b4', linewidth=1.8, label='actor_loss_mav')
ax.plot(steps_k[s], smooth(loss_uav), color='#ff7f0e', linewidth=1.8, label='actor_loss_uav')
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
ax.set_ylabel('Loss', fontsize=11)
ax.set_xlabel('Steps (x1000)', fontsize=11)
ax.set_title('Actor Loss (MAV / UAV)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.2)

# ── Panel 6: Entropy + Critic Loss ──
ax = axes[2, 1]
ax2 = ax.twinx()
ax.plot(steps_k[s], smooth(entropy_mav), color='#1f77b4', linewidth=1.8, label='entropy_mav')
ax.plot(steps_k[s], smooth(entropy_uav), color='#ff7f0e', linewidth=1.8,
        label='entropy_uav', linestyle='--')
clipped_cl = np.clip(critic_loss, 0, 80)
ax2.plot(steps_k[s], smooth(clipped_cl), color='#d62728', linewidth=1.3, alpha=0.7,
         label='critic_loss (clipped <= 80)')
ax.set_ylabel('Entropy', fontsize=11, color='#1f77b4')
ax2.set_ylabel('Critic Loss', fontsize=11, color='#d62728')
ax.set_xlabel('Steps (x1000)', fontsize=11)
ax.set_title('Entropy & Critic Loss', fontsize=12, fontweight='bold')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, fontsize=7, loc='upper right')
ax.grid(True, alpha=0.2)

# ── Stats box ──
stats_text = (
    f'Iterations: {n}  |  Steps: {int(steps[0])} -> {int(steps[-1])}  |  '
    f'Return: [{r_min:+.1f}, {r_max:+.1f}]  final={ret[-1]:+.1f}\n'
    f'Return>0: {np.sum(ret>0)}/{n} iters ({np.sum(ret>0)/n*100:.1f}%)  |  '
    f'NaN detected: {np.sum(nan_flag)}  |  mav_survival max: {np.max(mav_surv):.2f}  |  '
    f'Red missiles fired (total): {np.sum(red_missiles_fired):.0f}'
)
fig.text(0.02, 0.01, stats_text, fontsize=9, family='monospace', va='bottom')

plt.tight_layout(rect=[0, 0.04, 1, 0.94])
outpath = f'{OUT}/return_vs_steps_detailed.png'
plt.savefig(outpath, dpi=200, bbox_inches='tight')
print(f'Saved: {outpath}')
print(f'{n} rows  steps=[{steps[0]:.0f},{steps[-1]:.0f}]  ret=[{r_min:+.1f},{r_max:+.1f}]  final={ret[-1]:+.1f}')
print(f'Eval points: {len(eval_steps)}')
