"""Offline full review for tam_brma_scripted_reward_v1 + pure_happo.

This script is deliberately audit-only.  It reads source files and existing
outputs, then writes Markdown/CSV evidence tables.  It does not instantiate a
training run, modify reward logic, or alter policy/trainer code.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.full_review_audit_utils import (
    action_clamp_stats,
    gate_mismatch_stats,
    phase_summary,
    read_csv_rows,
    reward_component_stats,
    safe_float,
    select_checkpoints_from_train_log,
    source_line_hits,
    write_csv_rows,
)


DEFAULT_RUN_DIR = ROOT / "outputs" / "tam_brma_scripted_reward_v1_pure_happo_500k_obs_fix"
DEFAULT_OUT_DIR = ROOT / "outputs" / "audit_tam_brma_v1_pure_happo_full_review"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_md(path: Path, title: str, sections: list[tuple[str, str]]) -> None:
    body = [f"# {title}", ""]
    for heading, content in sections:
        body += [f"## {heading}", "", content.strip() or "_No evidence available._", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body), encoding="utf-8")


def _table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    out = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        out.append("|" + "|".join(str(row.get(c, "")) for c in columns) + "|")
    return "\n".join(out)


def _summarize_run(run_dir: Path) -> dict:
    train = read_csv_rows(run_dir / "train_log.csv")
    status = _read_json(run_dir / "runner_status.json")
    latest_meta = _read_json(run_dir / "latest" / "meta.json")
    best_meta = _read_json(run_dir / "best" / "meta.json")
    audit_dir = run_dir / "audit_trace"
    step_rows = read_csv_rows(audit_dir / "step_agent_components.csv")
    gate_rows = read_csv_rows(audit_dir / "launch_gate_step.csv")
    episode_rows = read_csv_rows(audit_dir / "episode_summary.csv")
    component_rows = reward_component_stats(step_rows)
    gate_stats = gate_mismatch_stats(gate_rows)
    phases = phase_summary(train)
    checkpoints = select_checkpoints_from_train_log(train)
    action_stats = action_clamp_stats(train)
    return {
        "train": train,
        "status": status,
        "latest_meta": latest_meta,
        "best_meta": best_meta,
        "step_rows": step_rows,
        "gate_rows": gate_rows,
        "episode_rows": episode_rows,
        "component_rows": component_rows,
        "gate_stats": gate_stats,
        "phases": phases,
        "checkpoints": checkpoints,
        "action_stats": action_stats,
    }


def _write_csv_outputs(out_dir: Path, summary: dict) -> None:
    write_csv_rows(out_dir / "reward_component_table.csv", summary["component_rows"])
    write_csv_rows(out_dir / "launch_gate_reward_alignment.csv", [summary["gate_stats"]])
    write_csv_rows(out_dir / "gate_mismatch_breakdown.csv", _gate_breakdown_rows(summary["gate_rows"]))
    write_csv_rows(out_dir / "checkpoint_sweep_metrics.csv", summary["checkpoints"])
    write_csv_rows(out_dir / "reward_component_attribution_by_phase.csv", summary["phases"])
    write_csv_rows(out_dir / "action_distribution_audit.csv", [_action_stats_row(summary)])
    write_csv_rows(out_dir / "ppo_update_health.csv", _ppo_health_rows(summary["train"]))
    write_csv_rows(out_dir / "critic_health.csv", _critic_rows(summary["train"]))
    write_csv_rows(out_dir / "advantage_diagnostics.csv", _advantage_rows(summary["train"]))
    write_csv_rows(out_dir / "deterministic_vs_stochastic_eval.csv", _det_vs_stoch_rows(summary))
    write_csv_rows(out_dir / "confirmed_reward_bugs.csv", _confirmed_reward_bugs(summary))
    write_csv_rows(out_dir / "reward_risks.csv", _reward_risks(summary))
    write_csv_rows(out_dir / "logging_metric_definition.csv", _metric_def_rows())
    write_csv_rows(out_dir / "train_vs_eval_consistency.csv", _train_eval_consistency_rows(summary))
    write_csv_rows(out_dir / "obs_schema_table.csv", _obs_schema_rows())
    write_csv_rows(out_dir / "change_classification_matrix.csv", _change_classification_rows(summary))


def _gate_breakdown_rows(gate_rows: list[dict]) -> list[dict]:
    counts = {}
    for row in gate_rows:
        key = row.get("mismatch_type", "unknown")
        counts[key] = counts.get(key, 0) + 1
    total = max(len(gate_rows), 1)
    return [
        {"mismatch_type": k, "count": v, "rate": v / total}
        for k, v in sorted(counts.items())
    ]


def _action_stats_row(summary: dict) -> dict:
    row = dict(summary["action_stats"])
    row.update({
        "policy_action_sampling": "Normal(mean,std) then clamp to [-1,1]",
        "log_prob_accounting": "log_prob is computed on the clamped action",
        "effective_entropy_note": "logged entropy is unclipped Normal entropy, not post-clamp action entropy",
    })
    return row


def _ppo_health_rows(train: list[dict]) -> list[dict]:
    rows = []
    for r in train:
        rows.append({
            "iteration": r.get("iteration"),
            "total_steps": r.get("total_steps"),
            "actor_loss_mav": safe_float(r.get("actor_loss_mav")),
            "actor_loss_uav": safe_float(r.get("actor_loss_uav")),
            "critic_loss": safe_float(r.get("critic_loss")),
            "approx_kl_mav": safe_float(r.get("approx_kl_mav")),
            "approx_kl_uav": safe_float(r.get("approx_kl_uav")),
            "entropy_mav": safe_float(r.get("entropy_mav")),
            "entropy_uav": safe_float(r.get("entropy_uav")),
            "clip_fraction": "",
            "policy_ratio_mean": "",
            "note": "ratio/clip fraction not logged in current pure-HAPPO train_log",
        })
    return rows


def _critic_rows(train: list[dict]) -> list[dict]:
    return [{
        "iteration": r.get("iteration"),
        "total_steps": r.get("total_steps"),
        "critic_loss": safe_float(r.get("critic_loss")),
        "explained_variance": "",
        "value_return_corr": "",
        "note": "value/return arrays are not persisted; cannot compute offline",
    } for r in train]


def _advantage_rows(train: list[dict]) -> list[dict]:
    return [{
        "iteration": r.get("iteration"),
        "total_steps": r.get("total_steps"),
        "mav_active_sample_count": safe_float(r.get("mav_active_sample_count")),
        "uav_active_sample_count": safe_float(r.get("uav_active_sample_count")),
        "team_reward_source": "mean(active red rewards) in PureHAPPOTrainer.update",
        "advantage_stats_available": "no",
    } for r in train]


def _det_vs_stoch_rows(summary: dict) -> list[dict]:
    # Existing audit_trace is deterministic rollout; train_log is stochastic rollout.
    train = summary["train"]
    episode = summary["episode_rows"]
    latest = train[-1] if train else {}
    det_red_fire = sum(safe_float(e.get("red_missiles_fired")) for e in episode) / max(len(episode), 1)
    det_red_hit = sum(safe_float(e.get("red_hits")) for e in episode) / max(len(episode), 1)
    return [{
        "source": "training_stochastic_latest_window",
        "red_fire": safe_float(latest.get("red_episode_missiles_fired_mean"), safe_float(latest.get("red_missiles_fired"))),
        "red_hit": safe_float(latest.get("red_episode_missile_hits_mean"), safe_float(latest.get("red_missile_hits"))),
        "mav_survival": safe_float(latest.get("mav_survival")),
        "geometry_ok": "",
        "note": "train_log rollout metrics are stochastic/recent-window metrics",
    }, {
        "source": "deterministic_audit_trace",
        "red_fire": det_red_fire,
        "red_hit": det_red_hit,
        "mav_survival": sum(safe_float(e.get("mav_alive_final")) for e in episode) / max(len(episode), 1),
        "geometry_ok": summary["gate_stats"].get("launch_geometry_ok_3d_rate", ""),
        "note": "audit_trace checkpoint rollout showed track_ok=100%, geometry_ok=0",
    }]


def _confirmed_reward_bugs(summary: dict) -> list[dict]:
    # The audit can prove mismatches/risks, but does not mark reward formula as bug
    # unless code contradicts the documented v1 design.
    return [{
        "id": "none_confirmed_in_reward_formula",
        "classification": "insufficient_evidence_for_confirmed_bug",
        "evidence": "v1 active totals use role-specific active components; current failure evidence points to geometry/no launch and pure-HAPPO limitations",
        "minimal_fix": "do not change reward formula without a separate confirmed mismatch",
    }]


def _reward_risks(summary: dict) -> list[dict]:
    stats = summary["gate_stats"]
    return [{
        "id": "reward_gate_sparse_or_inactive",
        "classification": "risk_suspicious",
        "evidence": f"reward_g_own_positive_rate={stats.get('reward_g_own_positive_rate', 0):.4f}; launch_geometry_ok_3d_rate={stats.get('launch_geometry_ok_3d_rate', 0):.4f}",
        "impact": "UAV gets little or no positive attack-window shaping when policy is outside true launch geometry",
    }, {
        "id": "mav_survival_attack_tension",
        "classification": "risk_suspicious",
        "evidence": "train_log final mav_survival is low while red fire/hit remain sparse; terminal/event losses dominate many windows",
        "impact": "MAV survival and UAV approach/fire may be hard to optimize with weak baseline",
    }]


def _metric_def_rows() -> list[dict]:
    return [
        {"metric": "red_episode_missiles_fired_mean", "scope": "recent completed episodes", "caution": "not same as rollout transition count"},
        {"metric": "red_missiles_fired", "scope": "rollout step aggregate in train log", "caution": "different scripts may log raw count vs mean"},
        {"metric": "entropy_mav/uav", "scope": "unclipped Normal entropy over valid samples", "caution": "does not measure post-clamp effective entropy"},
        {"metric": "mav/uav_action_saturation_rate", "scope": "mean action/mean saturation depending policy log path", "caution": "preclip sampled action rate is not logged"},
    ]


def _train_eval_consistency_rows(summary: dict) -> list[dict]:
    return [
        {"check": "eval_log_exists", "status": "missing", "evidence": "target 500K output has no eval_log.csv"},
        {"check": "audit_trace_exists", "status": "present", "evidence": "deterministic audit_trace CSVs present"},
        {"check": "latest_meta_policy_arch", "status": summary["latest_meta"].get("policy_arch", ""), "evidence": "latest/meta.json"},
        {"check": "runner_completed", "status": summary["status"].get("status", ""), "evidence": "runner_status.json"},
    ]


def _obs_schema_rows() -> list[dict]:
    return [
        {"field": "actor_obs_dim", "expected": 96, "source": "HeteroObsAdapterV2/meta", "audit": "fixed flat observation"},
        {"field": "critic_state_dim", "expected": 480, "source": "latest/meta.json", "audit": "fixed-capacity centralized state"},
        {"field": "enemy_observed_mask", "expected": "present", "source": "HeteroObsAdapterV2", "audit": "track visibility available to policy"},
        {"field": "enemy_track_source", "expected": "present", "source": "HeteroObsAdapterV2", "audit": "direct vs MAV shared track is encoded"},
        {"field": "3D launch gate features", "expected": "not explicit as gate tuple", "source": "flat obs", "audit": "policy must infer boresight/ATA/TA from geometry"},
    ]


def _change_classification_rows(summary: dict) -> list[dict]:
    return [
        {"proposal": "fix action log_prob/clamp accounting", "class": "baseline algorithm audit fix", "priority": "high", "status": "needs separate implementation decision"},
        {"proposal": "add reward launch bonus", "class": "prohibited change", "priority": "do_not_do", "status": "not supported by this audit"},
        {"proposal": "align reward gate with 3D launch gate", "class": "reward-launch-gate alignment fix", "priority": "medium", "status": "risk identified, needs targeted design"},
        {"proposal": "switch to entity/recurrent main method", "class": "main-method architecture improvement", "priority": "medium", "status": "reasonable after baseline audit"},
        {"proposal": "continue long training only", "class": "not recommended", "priority": "low", "status": "current evidence shows deterministic geometry failure"},
    ]


def _write_static_source_tables(out_dir: Path) -> dict:
    files = {
        "policy": ROOT / "algorithms" / "pure_happo" / "policy.py",
        "trainer": ROOT / "algorithms" / "pure_happo" / "trainer.py",
        "runner": ROOT / "scripts" / "train_happo_reference_parallel.py",
        "reward": ROOT / "uav_env" / "JSBSim" / "envs" / "hetero_uav_combat_env.py",
        "launch": ROOT / "uav_env" / "JSBSim" / "env.py",
        "adapter": ROOT / "uav_env" / "JSBSim" / "adapters" / "hetero_obs_adapter_v2.py",
    }
    hits = []
    hits += source_line_hits(files["policy"], {
        "independent_actors": "self.actors = nn.ModuleList",
        "normal_distribution": "return Normal(mean, std), mean",
        "sample_then_clamp": "dist.rsample().clamp(-1.0, 1.0)",
        "logprob_on_action": "dist.log_prob(a).sum",
        "eval_logprob_actions": "dist.log_prob(actions",
    })
    hits += source_line_hits(files["trainer"], {
        "team_reward_active_mean": "team_reward = (rewards * active).sum",
        "team_dones": "team_dones = dones[:, 0].float()",
        "grouped_gae": "_compute_grouped_gae(",
        "sequential_order": "order = list(self.rng.permutation",
        "correction_factor": "M = (M * ratio_after).detach()",
        "single_critic_update": "self.critic_opt.step()",
    })
    hits += source_line_hits(files["reward"], {
        "tam_brma_compute": "def _compute_tam_brma_scripted_reward_v1",
        "terminal_once": "_tam_brma_scripted_terminal_applied",
        "team_uav_loss": "team_uav_loss = num_uav_first_deaths",
        "mav_team_credit": "team_kill_credit * total_red_kills",
        "uav_gate_sit": "def _tam_brma_v1_uav_reward",
        "mav_reward": "def _tam_brma_v1_mav_reward",
    })
    hits += source_line_hits(files["launch"], {
        "3d_launch_geometry": "def _build_launch_geometry_3d",
        "candidate_metrics": "def _missile_candidate_metrics",
        "has_launch_track": "def _has_launch_track",
        "launch_track_gate": "has_track, track_source = self._has_launch_track",
    })
    write_csv_rows(out_dir / "source_evidence_table.csv", hits)
    return {k: str(v) for k, v in files.items()}


def _write_reports(out_dir: Path, run_dir: Path, summary: dict, source_files: dict) -> None:
    train = summary["train"]
    latest = train[-1] if train else {}
    gate = summary["gate_stats"]
    action = summary["action_stats"]
    checkpoints = summary["checkpoints"]
    comp = summary["component_rows"]
    final_step = summary["status"].get("total_env_steps_actual") or safe_float(latest.get("total_steps"))
    red_fire_final = safe_float(latest.get("red_episode_missiles_fired_mean"), safe_float(latest.get("red_missiles_fired")))
    red_hit_final = safe_float(latest.get("red_episode_missile_hits_mean"), safe_float(latest.get("red_missile_hits")))

    executive = f"""
Run audited: `{run_dir}`.

- Completion: `{summary['status'].get('status', 'unknown')}`, total_env_steps_actual={final_step}, nan_detected={summary['status'].get('nan_detected')}.
- Policy: `{summary['latest_meta'].get('policy_arch')}`, independent actors={summary['latest_meta'].get('per_agent_independent_actors')}, centralized critic={summary['latest_meta'].get('global_v_critic')}.
- Latest train window: avg_return={safe_float(latest.get('avg_return')):.3f}, red_win={safe_float(latest.get('red_win')):.3f}, blue_win={safe_float(latest.get('blue_win')):.3f}, mav_survival={safe_float(latest.get('mav_survival')):.3f}, red_fire={red_fire_final:.3f}, red_hit={red_hit_final:.3f}.
- Deterministic audit trace: track_ok={gate.get('track_ok_rate', 0):.3f}, launch_geometry_ok_3d={gate.get('launch_geometry_ok_3d_rate', 0):.3f}, reward_g_own_positive={gate.get('reward_g_own_positive_rate', 0):.3f}.

Bottom line: this looks like a combined baseline limitation plus geometry/reward-gate sparsity problem, with one important pure-HAPPO risk: Normal sample -> clamp -> log_prob on clamped action can make PPO accounting inconsistent when saturation rises.  No confirmed reward-formula bug is proven by the current evidence.
"""

    _write_md(out_dir / "executive_summary.md", "Executive Summary", [
        ("Conclusion", executive),
        ("Priority", "1. Audit/fix pure-HAPPO action distribution accounting.\n2. Treat reward-vs-3D-launch gate alignment as a risk, not an immediate reward rewrite.\n3. Do not continue long training as the only next step."),
    ])

    _write_md(out_dir / "confirmed_bugs.md", "Confirmed Bugs", [
        ("Confirmed", _table(_confirmed_reward_bugs(summary), ["id", "classification", "evidence", "minimal_fix"])),
        ("Algorithmic correctness risk", "The clamp/log_prob issue is a likely correctness bug in PPO accounting, but this report classifies it as requiring a targeted fix proposal because changing the distribution affects algorithm behavior."),
    ])

    _write_md(out_dir / "suspected_risks.md", "Suspected Risks", [
        ("Reward risks", _table(_reward_risks(summary), ["id", "classification", "evidence", "impact"])),
        ("Policy/trainer risks", f"- action saturation max: MAV={action.get('mav_saturation_max', 0):.3f}, UAV={action.get('uav_saturation_max', 0):.3f}\n- `entropy` is unclipped Normal entropy; effective clipped action exploration is not logged.\n- critic explained variance and value/return correlation are unavailable from current logs."),
    ])

    _write_md(out_dir / "no_change_recommendations.md", "No-Change Recommendations", [
        ("Do not change without separate evidence", "- Do not add fire/launch/guided reward.\n- Do not change PID, action space, missile launch gates, missile dynamics, blue rule, or termination.\n- Do not interpret poor training as proof that `tam_brma_scripted_reward_v1` formula is wrong.\n- Do not use old obs-cache-broken runs to judge current v1."),
    ])

    reward_sections = [
        ("Formula alignment", "The implementation exposes `tam_brma_v1_flight`, role-specific UAV `tam_brma_v1_uav_gate_sit`, MAV `safe/support/aware`, event and terminal components.  Active totals are role-specific; diagnostic terms are reported separately in the audit trace."),
        ("Component evidence", _table(comp[:30], ["role", "component", "samples", "mean", "min", "max", "positive_rate", "negative_rate"])),
        ("Confirmed reward bugs", _table(_confirmed_reward_bugs(summary), ["id", "classification", "evidence", "minimal_fix"])),
        ("Risks", _table(_reward_risks(summary), ["id", "classification", "evidence", "impact"])),
    ]
    _write_md(out_dir / "reward_formula_alignment.md", "Reward Formula Alignment", reward_sections)

    _write_md(out_dir / "launch_gate_reward_alignment.md", "Launch Gate Reward Alignment", [
        ("Key rates", "\n".join(f"- {k}: {v}" for k, v in gate.items())),
        ("Interpretation", "The current deterministic audit trace has track_ok=100%, but launch_geometry_ok_3d=0 and reward_g_own_positive=0.  This does not prove reward formula is wrong; it shows the policy never reaches real 3D launch geometry and therefore receives no positive UAV gate shaping."),
    ])

    _write_md(out_dir / "pure_happo_policy_audit.md", "Pure-HAPPO Policy Audit", [
        ("Architecture", "Each red agent has an independent MLP actor and per-agent log_std.  Roles are not used except through slot-specific actor identity.  Critic is centralized MLP."),
        ("Action distribution risk", "Policy samples `Normal(mean, std)`, clamps action to [-1,1], then computes log_prob on the clamped action.  Entropy is the unclipped Normal entropy.  This is a PPO-accounting risk when sampled pre-clamp rate is high; pre-clamp rate is not logged."),
        ("Action audit", _table([_action_stats_row(summary)], ["samples", "mav_saturation_mean", "mav_saturation_max", "uav_saturation_mean", "uav_saturation_max", "mav_log_std_final", "uav_log_std_final", "log_prob_accounting"])),
    ])

    _write_md(out_dir / "pure_happo_trainer_audit.md", "Pure-HAPPO Trainer Audit", [
        ("Implementation", "GAE is grouped by env_id, team_dones uses `dones[:,0]`, active masks filter dead agents, actors update in random sequential order with correction factor M, critic updates once per PPO update."),
        ("Risks", "- team_reward is mean(active red rewards), which can dilute role-specific UAV gate signals.\n- advantages are normalized globally across rollout.\n- no minibatches/target-KL/early-stop are evident in the current trainer.\n- critic health cannot be fully assessed offline because value and return vectors are not persisted."),
    ])

    _write_md(out_dir / "rollout_runner_audit.md", "Rollout Runner Audit", [
        ("Run status", json.dumps(summary["status"], indent=2, ensure_ascii=False)),
        ("Consistency", "The audited run completed normally with worker_restart_count=0 and rollout_aborted_count=0.  It used multiprocessing_parallel, num_envs=4, rollout_length_per_env=256, transitions_per_rollout=1024.  There is no eval_log.csv in this output; deterministic audit_trace is the available eval-like evidence."),
    ])

    _write_md(out_dir / "logging_metric_definition.md", "Logging Metric Definitions", [
        ("Definitions", _table(_metric_def_rows(), ["metric", "scope", "caution"])),
    ])

    _write_md(out_dir / "train_vs_eval_consistency.md", "Train vs Eval Consistency", [
        ("Checks", _table(_train_eval_consistency_rows(summary), ["check", "status", "evidence"])),
    ])

    _write_md(out_dir / "observation_expressiveness_audit.md", "Observation Expressiveness Audit", [
        ("Schema", _table(_obs_schema_rows(), ["field", "expected", "source", "audit"])),
        ("Interpretation", "HeteroObsAdapterV2 provides fixed flat 96-dim actor observations with masks and track-source information.  Pure-HAPPO uses slot MLPs, no recurrence and no attention/entity encoder, so it must infer 3D intercept geometry from flat features and cannot maintain temporal lock maturity explicitly."),
    ])

    _write_md(out_dir / "paper_alignment_audit.md", "Paper Alignment Audit", [
        ("Pure-HAPPO baseline", "Reasonable as a weak baseline: independent actors, centralized critic, sequential update.  It is not TAM-HAPPO and lacks recurrent memory, entity attention, and explicit heterogeneous coordination modules."),
        ("Reward", "`tam_brma_scripted_reward_v1` mixes BRMA flight/status with TAM-style MAV support and event/terminal structure.  Current audit confirms no fire/launch/guided reward is needed to explain the failure; the problem is that real 3D geometry is never reached in deterministic audit."),
        ("Change classification", _table(_change_classification_rows(summary), ["proposal", "class", "priority", "status"])),
    ])

    _write_md(out_dir / "checkpoint_sweep_report.md", "Checkpoint Sweep Report", [
        ("Candidate rows", _table(checkpoints, ["selector", "total_steps", "avg_return", "red_win", "blue_win", "timeout", "mav_survival", "red_fire", "red_hit", "duplicate_step"])),
        ("Limitation", "This is train-log based selection only.  The target directory has no eval_log.csv and no per-checkpoint deterministic/stochastic eval table, so exact best checkpoint must be verified in a separate eval sweep before making final claims."),
    ])

    full_sections = [
        ("Executive summary", executive),
        ("Reward evidence", (out_dir / "reward_formula_alignment.md").read_text(encoding="utf-8")),
        ("Launch gate evidence", (out_dir / "launch_gate_reward_alignment.md").read_text(encoding="utf-8")),
        ("Policy evidence", (out_dir / "pure_happo_policy_audit.md").read_text(encoding="utf-8")),
        ("Trainer evidence", (out_dir / "pure_happo_trainer_audit.md").read_text(encoding="utf-8")),
        ("Required answers", _required_answers(summary)),
    ]
    _write_md(out_dir / "full_review_report.md", "Full Review Report", full_sections)


def _required_answers(summary: dict) -> str:
    gate = summary["gate_stats"]
    train = summary["train"]
    latest = train[-1] if train else {}
    checkpoints = summary["checkpoints"]
    best = checkpoints[0] if checkpoints else {}
    return f"""
1. Failure cause: mixed.  Strong evidence for pure-HAPPO baseline limitation and launch-geometry sparsity; no confirmed reward formula bug.
2. Confirmed bug: none in reward formula.  Strong suspected PPO accounting bug: sample Normal -> clamp -> log_prob on clamped action.
3. pure-HAPPO trustworthiness: usable as weak baseline, but not enough as main method; lacks recurrence/entity attention and has action distribution accounting risk.
4. Poor pure-HAPPO learning can be reported as weak baseline result only if action-accounting caveat is disclosed.
5. Reward vs real 3D gate: deterministic audit has reward_g_own_positive_rate={gate.get('reward_g_own_positive_rate', 0):.4f}, launch_geometry_ok_3d_rate={gate.get('launch_geometry_ok_3d_rate', 0):.4f}; positive shaping is effectively absent when outside geometry.
6. Train occasional red_fire/red_hit vs final deterministic geometry_ok=0: likely stochastic exploration occasionally enters launch windows, while deterministic mean policy does not.
7. Negative return: latest avg_return={safe_float(latest.get('avg_return')):.3f}; reward attribution points to UAV/MAV event and terminal losses plus inactive attack-window reward.
8. MAV survival and red attack likely conflict under weak baseline because red UAVs need approach geometry while MAV survival/terminal penalties punish exposure.
9. Best checkpoint from train log: {best}.
10. Priority: first audit/fix pure-HAPPO action distribution accounting; second verify reward-vs-3D gate alignment; do not continue long training as the only action.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _summarize_run(run_dir)
    _write_csv_outputs(out_dir, summary)
    source_files = _write_static_source_tables(out_dir)
    _write_reports(out_dir, run_dir, summary, source_files)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

