#!/usr/bin/env python3
"""Generate a compact diagnostic report for a completed vanilla-HAPPO run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

DEFAULT_RUN_DIR = Path('/root/hpk/train_environment/tam_uav/outputs/happo_2v2_1m_seed2026')


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('run_directory', nargs='?', type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument('--log', type=Path, default=None)
    p.add_argument('--log-tail', type=int, default=30)
    p.add_argument('--report-name', default='happo_training_report.txt')
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f'Missing file: {path}')
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def fnum(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if text.lower() == 'true':
        return 1.0
    if text.lower() == 'false':
        return 0.0
    try:
        x = float(text)
        return x if math.isfinite(x) else None
    except ValueError:
        return None


def bval(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {'true', '1', 'yes'}:
        return True
    if text in {'false', '0', 'no'}:
        return False
    return None


def col(rows: list[dict[str, str]], key: str) -> list[float]:
    result = []
    for row in rows:
        x = fnum(row.get(key))
        if x is not None:
            result.append(x)
    return result


def qtile(values: Iterable[float], q: float) -> float | None:
    data = sorted(float(x) for x in values if math.isfinite(float(x)))
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return data[lo]
    w = pos - lo
    return data[lo] * (1 - w) + data[hi] * w


def avg(values: Iterable[float]) -> float | None:
    data = [float(x) for x in values if math.isfinite(float(x))]
    return statistics.fmean(data) if data else None


def sd(values: Iterable[float]) -> float | None:
    data = [float(x) for x in values if math.isfinite(float(x))]
    return statistics.stdev(data) if len(data) > 1 else (0.0 if data else None)


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return 'N/A'
    if isinstance(value, bool):
        return str(value)
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(x) >= 1_000_000:
        return f'{x:,.0f}'
    if abs(x) >= 1000:
        return f'{x:,.3f}'
    if abs(x) >= 10:
        return f'{x:.4f}'
    return f'{x:.{digits}f}'


def hsize(size: int) -> str:
    x = float(size)
    for unit in ('B', 'KiB', 'MiB', 'GiB'):
        if x < 1024 or unit == 'GiB':
            return f'{x:.2f} {unit}'
        x /= 1024
    return f'{size} B'


def stats(rows: list[dict[str, str]], key: str) -> dict[str, float | None]:
    values = col(rows, key)
    return {
        'mean': avg(values), 'std': sd(values), 'min': min(values) if values else None,
        'p50': qtile(values, .5), 'p95': qtile(values, .95), 'p99': qtile(values, .99),
        'max': max(values) if values else None, 'last': values[-1] if values else None,
    }


def windows(rows: list[dict[str, str]], key: str) -> tuple[float | None, float | None, float | None]:
    values = [fnum(row.get(key)) for row in rows]
    values = [x for x in values if x is not None]
    if not values:
        return None, None, None
    n = len(values)
    w = max(1, n // 10)
    middle = max(0, n // 2 - w // 2)
    return avg(values[:w]), avg(values[middle:middle + w]), avg(values[-w:])


def agents_from(config: dict[str, Any], rows: list[dict[str, str]]) -> list[str]:
    ids = config.get('agent_ids')
    if isinstance(ids, list) and ids:
        return [str(x) for x in ids]
    if not rows:
        return []
    return sorted({k.split('/', 1)[1] for k in rows[0] if k.startswith('active_sample_count/')})


def rollout_lengths(rows: list[dict[str, str]]) -> list[int]:
    lengths = []
    previous = 0
    for row in rows:
        explicit = fnum(row.get('rollout_collected_steps'))
        if explicit is not None:
            lengths.append(int(round(explicit)))
            continue
        step = fnum(row.get('environment_steps'))
        if step is not None:
            current = int(round(step))
            lengths.append(current - previous)
            previous = current
    return lengths


def false_count(rows: list[dict[str, str]], key: str) -> int:
    return sum(bval(row.get(key)) is False for row in rows)


def max_with_step(rows: list[dict[str, str]], key: str) -> tuple[float | None, int | None]:
    values = []
    for row in rows:
        x = fnum(row.get(key))
        step = fnum(row.get('environment_steps'))
        if x is not None and step is not None:
            values.append((x, int(step)))
    return max(values) if values else (None, None)


def make_report(run_dir: Path, config: dict[str, Any], summary: dict[str, Any], rows: list[dict[str, str]], log: Path | None, log_tail: int) -> str:
    lines: list[str] = []
    agents = agents_from(config, rows)
    lengths = rollout_lengths(rows)
    planned = int(config.get('rollout_length', 0) or 0)
    final_steps = int(summary.get('environment_steps') or fnum(rows[-1].get('environment_steps')) or 0)
    final_episodes = int(summary.get('episodes') or fnum(rows[-1].get('episodes')) or 0)

    lines += ['=' * 88, 'VANILLA HAPPO TRAINING REPORT', '=' * 88]
    lines += [f'Run directory: {run_dir}', f'Updates: {len(rows)}', f'Final environment steps: {final_steps}', f'Completed episodes: {final_episodes}', '']

    lines += ['[1] Configuration']
    for label, key in (
        ('scenario', 'scenario'), ('seed', 'seed'), ('requested total steps', 'total_environment_steps'),
        ('rollout length', 'rollout_length'), ('requested device', 'device'), ('actual device', 'actual_device'),
        ('algorithm mode', 'algorithm_mode'), ('actor sharing', 'actor_sharing_label'),
        ('actor observation dim', 'actor_observation_dim'), ('critic state dim', 'critic_state_dim'),
        ('hidden dim', 'hidden_dim'), ('actor lr', 'actor_lr'), ('critic lr', 'critic_lr'),
        ('ppo epochs', 'ppo_epochs'), ('minibatch size', 'minibatch_size'), ('clip', 'clip_param'),
        ('entropy coef', 'entropy_coef'), ('gamma', 'gamma'), ('gae lambda', 'gae_lambda')):
        lines.append(f'{label}: {config.get(key, "N/A")}')
    lines.append(f'uses TAM/recurrence/attention: {config.get("uses_tam", "N/A")}/{config.get("uses_recurrence", "N/A")}/{config.get("uses_attention", "N/A")}')
    lines.append('')

    lines += ['[2] Output integrity']
    for name in ('config_snapshot.json', 'summary.json', 'training.csv', 'checkpoint_final.pt'):
        path = run_dir / name
        lines.append(f'{name}: {hsize(path.stat().st_size) if path.exists() else "MISSING"}')
    for key in ('checkpoint_type', 'resume_semantics', 'OPTIMIZATION_PIPELINE_ACTIVE', 'HAPPO_RUNTIME_INVARIANTS_VALID', 'EARLY_PERFORMANCE_SIGNAL_OBSERVED', 'LEARNING_CONVERGENCE_NOT_VALIDATED'):
        lines.append(f'{key}: {summary.get(key, "N/A")}')
    lines.append('')

    lines += ['[3] Rollout diagnostics']
    if lengths:
        short = sum(x < planned for x in lengths) if planned else 0
        tiny = sum(x < 32 for x in lengths)
        lines.append(f'length min/p50/p95/max: {min(lengths)}/{fmt(qtile(lengths,.5),2)}/{fmt(qtile(lengths,.95),2)}/{max(lengths)}')
        lines.append(f'shorter than configured {planned}: {short}/{len(lengths)}')
        lines.append(f'tiny rollouts <32: {tiny}/{len(lengths)}')
        counts: dict[int, int] = {}
        for x in lengths:
            counts[x] = counts.get(x, 0) + 1
        common = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:12]
        lines.append('common lengths (length x count): ' + ', '.join(f'{x}x{n}' for x, n in common))
    else:
        lines.append('No rollout lengths reconstructed.')
    lines.append('')

    lines += ['[4] Runtime correctness']
    lines.append(f'NaN/Inf count sum: {fmt(sum(col(rows,"nan_inf_count")),0)}')
    lines.append(f'target consistency violation sum: {fmt(sum(col(rows,"target_consistency_violation")),0)}')
    lines.append(f'invalid optimization update rows: {false_count(rows,"optimization_update_contract_valid")}/{len(rows)}')
    lines.append(f'invalid runtime invariant rows: {false_count(rows,"runtime_invariants_valid")}/{len(rows)}')
    lines.append('')

    lines += ['[5] Active samples']
    for aid in agents:
        key = f'active_sample_count/{aid}'
        s = stats(rows, key)
        values = col(rows, key)
        lines.append(f'{aid}: min={fmt(s["min"],0)}, p50={fmt(s["p50"],1)}, p95={fmt(s["p95"],1)}, max={fmt(s["max"],0)}, zero={sum(x==0 for x in values)}, positive<32={sum(0<x<32 for x in values)}')
        contract = f'update_contract_valid/{aid}'
        if rows and contract in rows[0]:
            lines.append(f'  invalid contract rows: {false_count(rows,contract)}/{len(rows)}')
    lines.append('')

    lines += ['[6] PPO actor diagnostics']
    for aid in agents:
        kl_key = f'approx_kl/{aid}'
        kl = stats(rows, kl_key)
        max_kl, max_step = max_with_step(rows, kl_key)
        ent = stats(rows, f'entropy/{aid}')
        clip = stats(rows, f'clip_fraction/{aid}')
        grad = stats(rows, f'gradient_norm/{aid}')
        e0, em, e1 = windows(rows, f'entropy/{aid}')
        lines.append(f'{aid} KL mean/p95/p99/max(last): {fmt(kl["mean"])}/{fmt(kl["p95"])}/{fmt(kl["p99"])}/{fmt(max_kl)} at step {max_step}; last={fmt(kl["last"])}')
        lines.append(f'{aid} entropy early/middle/late: {fmt(e0)}/{fmt(em)}/{fmt(e1)}; min={fmt(ent["min"])} last={fmt(ent["last"])}')
        lines.append(f'{aid} clip mean/p95/max/last: {fmt(clip["mean"])}/{fmt(clip["p95"])}/{fmt(clip["max"])}/{fmt(clip["last"])}')
        lines.append(f'{aid} actor grad mean/p95/max/last: {fmt(grad["mean"])}/{fmt(grad["p95"])}/{fmt(grad["max"])}/{fmt(grad["last"])}')
    fmin, fmax = stats(rows, 'importance_factor_min'), stats(rows, 'importance_factor_max')
    lines.append(f'importance factor global min/max: {fmt(fmin["min"])}/{fmt(fmax["max"])}; final min/max={fmt(fmin["last"])}/{fmt(fmax["last"])}')
    lines.append('')

    lines += ['[7] Critic diagnostics']
    cg = stats(rows, 'critic_gradient_norm')
    lines.append(f'critic grad mean/p95/max/last: {fmt(cg["mean"])}/{fmt(cg["p95"])}/{fmt(cg["max"])}/{fmt(cg["last"])}')
    for aid in agents:
        loss = stats(rows, f'critic_loss/{aid}')
        ev = stats(rows, f'explained_variance/{aid}')
        e0, em, e1 = windows(rows, f'explained_variance/{aid}')
        lines.append(f'{aid} critic loss mean/p95/max/last: {fmt(loss["mean"])}/{fmt(loss["p95"])}/{fmt(loss["max"])}/{fmt(loss["last"])}')
        lines.append(f'{aid} explained variance early/middle/late: {fmt(e0)}/{fmt(em)}/{fmt(e1)}; min={fmt(ev["min"])} max={fmt(ev["max"])} last={fmt(ev["last"])}')
    lines.append('')

    episode_rows = [r for r in rows if (fnum(r.get('episode_length')) or 0) > 0 or (fnum(r.get('rollout_episode_count')) or 0) > 0]
    lines += ['[8] Sampled training-episode performance', f'updates containing completed episodes: {len(episode_rows)}/{len(rows)}']
    for label, key in (
        ('team return','mean_episode_return'), ('agent-0 return','mav_return'), ('other UAV return','uav_return'),
        ('win rate','win_rate'), ('draw rate','draw_rate'), ('episode length','episode_length'),
        ('launch rate','launch_rate'), ('hit rate','hit_rate'), ('survival rate','survival_rate'),
        ('structural failure rate','structural_failure_rate'), ('boundary rate','boundary_rate')):
        a, m, z = windows(episode_rows, key)
        lines.append(f'{label} early/middle/late: {fmt(a)}/{fmt(m)}/{fmt(z)}')
    if episode_rows:
        lines.append('last completed-episode updates:')
        for row in episode_rows[-10:]:
            lines.append('  step={step}, episodes={episodes}, return={ret}, win={win}, hit={hit}, survival={surv}, structural={struct}'.format(
                step=int(fnum(row.get('environment_steps')) or 0), episodes=int(fnum(row.get('episodes')) or 0),
                ret=fmt(fnum(row.get('mean_episode_return'))), win=fmt(fnum(row.get('win_rate'))),
                hit=fmt(fnum(row.get('hit_rate'))), surv=fmt(fnum(row.get('survival_rate'))),
                struct=fmt(fnum(row.get('structural_failure_rate')))))
    lines.append('')

    lines += ['[9] Final CSV row']
    final = rows[-1]
    keys = ['environment_steps','episodes','mean_episode_return','win_rate','hit_rate','survival_rate','structural_failure_rate','boundary_rate','critic_loss','critic_gradient_norm','importance_factor_min','importance_factor_max','optimization_update_contract_valid','runtime_invariants_valid','nan_inf_count','target_consistency_violation']
    for aid in agents:
        keys += [f'active_sample_count/{aid}',f'approx_kl/{aid}',f'entropy/{aid}',f'clip_fraction/{aid}',f'explained_variance/{aid}',f'update_contract_valid/{aid}']
    for key in keys:
        if key in final and str(final[key]).strip():
            lines.append(f'{key}: {final[key]}')
    lines.append('')

    eval_files = sorted(run_dir.glob('evaluation_*.json'))
    lines += ['[10] Evaluation availability', f'evaluation JSON files: {len(eval_files)}', f'latest_evaluation present: {summary.get("latest_evaluation") is not None}', 'This run disabled evaluation. The report can assess optimization stability and sampled training episodes, but cannot establish policy quality against fixed untrained/random/rule baselines.', '']

    lines += ['[11] Automatic flags']
    flags = []
    if planned and lengths and any(x < planned for x in lengths):
        flags.append(f'{sum(x < planned for x in lengths)} updates had rollouts shorter than {planned}.')
    for aid in agents:
        av = col(rows, f'active_sample_count/{aid}')
        kv = col(rows, f'approx_kl/{aid}')
        if any(x == 0 for x in av):
            flags.append(f'{aid} had zero-active-sample updates.')
        if kv and max(kv) > .1:
            flags.append(f'{aid} maximum approximate KL exceeded 0.1: {max(kv):.6f}.')
    invalid = false_count(rows, 'optimization_update_contract_valid')
    if invalid:
        flags.append(f'{invalid} updates failed optimization_update_contract_valid.')
    if sum(col(rows, 'nan_inf_count')):
        flags.append('NaN/Inf events were recorded.')
    if sum(col(rows, 'target_consistency_violation')):
        flags.append('Target-consistency violations were recorded.')
    if not flags:
        flags.append('No selected numerical/runtime warning was triggered.')
    for flag in flags:
        lines.append(f'- {flag}')
    lines.append('- Long-run learning and convergence remain unverified without a fixed evaluation set.')
    lines.append('')

    if log is not None:
        lines += ['[12] Log tail']
        if log.exists():
            lines += log.read_text(encoding='utf-8', errors='replace').splitlines()[-max(0, log_tail):]
        else:
            lines.append(f'Log not found: {log}')
        lines.append('')

    lines += ['=' * 88, 'END OF REPORT', '=' * 88]
    return '\n'.join(lines) + '\n'


def main() -> int:
    args = args_parser()
    run_dir = args.run_directory.expanduser().resolve()
    if not run_dir.is_dir():
        print(f'ERROR: run directory not found: {run_dir}', file=sys.stderr)
        return 2
    try:
        config = read_json(run_dir / 'config_snapshot.json')
        summary = read_json(run_dir / 'summary.json')
        rows = read_rows(run_dir / 'training.csv')
        if not rows:
            raise RuntimeError('training.csv has no data rows')
        report = make_report(run_dir, config, summary, rows, args.log.expanduser().resolve() if args.log else None, args.log_tail)
        report_path = run_dir / args.report_name
        report_path.write_text(report, encoding='utf-8')
        print(report, end='')
        print(f'\nReport saved to: {report_path}', file=sys.stderr)
        return 0
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
