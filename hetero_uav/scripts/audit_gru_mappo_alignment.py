"""Audit parent-project GRU MAPPO protocol against hetero_uav.

This is a read-only planning script. It inspects the parent training scripts
and the current hetero_uav MAPPO implementation, then writes a concise protocol
alignment report. It does not implement GRU, attention, HAPPO, or run training.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


def _read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.MULTILINE | re.DOTALL) is not None


def _extract_default_int(text: str, name: str, default: int | None = None) -> int | None:
    match = re.search(rf"{re.escape(name)}\s*:\s*int\s*=\s*(\d+)", text)
    if match:
        return int(match.group(1))
    match = re.search(rf"{re.escape(name)}\s*=\s*(\d+)", text)
    if match:
        return int(match.group(1))
    return default


def _audit_parent_vanilla() -> dict[str, Any]:
    path = WORKSPACE / "train_vanilla_mappo.py"
    text = _read(path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "actor_has_gru": "nn.GRUCell" in text and "class VanillaActor" in text,
        "critic_has_gru": False,
        "actor_critic_share_gru": False,
        "actor_gru_type": "GRUCell",
        "actor_hidden_size": _extract_default_int(text, "rnn_hidden_size", 128),
        "actor_gru_layers": 1,
        "critic_type": "centralized feed-forward MLP over concatenated red observations",
        "buffer_stores_rnn_actor_init": "rnn_actor_init" in text,
        "buffer_stores_rnn_actor_final": "rnn_actor_final" in text,
        "buffer_stores_rnn_critic_init": False,
        "buffer_stores_alive_mask": "self.alive" in text,
        "buffer_stores_done": "self.dones" in text,
        "ppo_sequence_update": "Actor GRU unroll" in text or "rnn_a = torch.as_tensor" in text,
        "minibatch_sequence_update": "Trajectory-minibatch" in text,
        "hidden_reset_on_done": "rnn_hidden_actor[env_idx, i] = np.zeros" in text,
        "eval_hidden_protocol_found": "deterministic" in text and "rnn_hidden" in text,
        "notes": [
            "Actor is recurrent, critic is feed-forward centralized.",
            "Rollout stores initial actor hidden state and alive/done masks.",
            "PPO update reconstructs per-env per-agent sequences and unrolls actor GRU.",
            "Done resets actor hidden state during rollout collection.",
        ],
    }


def _audit_parent_attention() -> dict[str, Any]:
    train_path = WORKSPACE / "train_ppo.py"
    attention_path = WORKSPACE / "attention_models.py"
    text = _read(train_path)
    attn = _read(attention_path)
    return {
        "train_path": str(train_path),
        "attention_model_path": str(attention_path),
        "exists": train_path.exists(),
        "actor_has_attention": "AttentionActor" in text or "MultiheadAttention" in attn,
        "actor_has_gru": "rnn_hidden" in text and ("GRUCell" in attn or "self.rnn" in attn),
        "critic_has_gru": "rnn_critic_init" in text,
        "actor_critic_share_gru": False,
        "hidden_size": _extract_default_int(text, "hidden_size", 128),
        "rnn_hidden_size": _extract_default_int(text, "rnn_hidden_size", 128),
        "rnn_layers": 1,
        "buffer_stores_actor_and_critic_hidden": "rnn_actor_init" in text and "rnn_critic_init" in text,
        "buffer_stores_mask_metadata": "mask_entropy" in text and "num_enemy_drop" in text,
        "sequence_update": "rnn_a = rnn_a_init.clone()" in text and "rnn_c" in text,
        "hidden_reset_on_done": "rnn_hidden_actor[env_idx, i] = np.zeros" in text,
        "notes": [
            "Attention path combines entity attention, mask generator, and recurrent actor/critic hidden states.",
            "It is larger than the minimal next step needed for hetero_uav.",
            "The recurrent mechanism is still central even in the attention version.",
        ],
    }


def _audit_current_hetero() -> dict[str, Any]:
    policy = _read(ROOT / "algorithms" / "mappo" / "policy.py")
    storage = _read(ROOT / "algorithms" / "mappo" / "storage.py")
    trainer = _read(ROOT / "algorithms" / "mappo" / "trainer.py")
    train = _read(ROOT / "scripts" / "train_mappo_baseline.py")
    return {
        "policy_path": "algorithms/mappo/policy.py",
        "storage_path": "algorithms/mappo/storage.py",
        "trainer_path": "algorithms/mappo/trainer.py",
        "actor_is_feedforward": "nn.Sequential" in policy and "nn.GRU" not in policy,
        "critic_is_feedforward": "self.critic = nn.Sequential" in policy,
        "buffer_stores_hidden_state": "rnn" in storage.lower() or "hidden" in storage.lower(),
        "trainer_assumes_flat_batch": "actor_obs.view(-1" in trainer,
        "active_mask_present": "active_mask" in trainer,
        "team_done_fix_present": "_team_dones_from_repeated_agent_dones" in trainer,
        "eval_hidden_state_present": "hidden" in train.lower() or "rnn" in train.lower(),
        "current_actor_archs": [
            "mlp",
            "role_conditioned",
        ] if "RoleConditionedMAPPOActorCritic" in policy else ["mlp"],
        "notes": [
            "Current hetero_uav MAPPO is feed-forward and flattens T x agents for PPO actor evaluation.",
            "Alive mask and team done logic are already fixed and should carry into recurrent training.",
            "No recurrent hidden state is stored or reset in the current rollout buffer.",
        ],
    }


def _required_changes() -> list[dict[str, str]]:
    return [
        {
            "target": "algorithms/mappo/recurrent_policy.py",
            "change": "Add GRUMAPPOActorCritic with separate actor and critic GRUs.",
            "reason": "Parent protocol uses recurrent state; current policy is feed-forward.",
        },
        {
            "target": "algorithms/mappo/recurrent_buffer.py",
            "change": "Store full rollout sequences plus actor/critic initial hidden states and active masks.",
            "reason": "PPO update must reconstruct per-agent sequences rather than flat samples.",
        },
        {
            "target": "algorithms/mappo/recurrent_trainer.py",
            "change": "Implement full-rollout sequence PPO first, before minibatch sequence slicing.",
            "reason": "Smallest correct recurrent update is safer than a partial flat-batch GRU update.",
        },
        {
            "target": "training/eval runners",
            "change": "Maintain and reset hidden state on episode done and dead-agent inactive masks.",
            "reason": "Hidden state leakage across episodes or dead agents would invalidate recurrent training.",
        },
        {
            "target": "adapter/model selection",
            "change": "Support actor_arch='gru_mlp' without changing observation dimensions.",
            "reason": "The experiment should isolate recurrence, not observation or reward changes.",
        },
    ]


def _minimal_plan() -> dict[str, Any]:
    return {
        "decision": "plan_only_this_round",
        "why_not_implement_now": [
            "Correct GRU MAPPO requires coordinated changes to policy, buffer, trainer, train runner, eval runner, save/load, and tests.",
            "A half-implemented recurrent policy with flat PPO update would not match the parent recurrent protocol.",
            "The current request is to audit parent GRU and prepare the minimal plan, not to change reward or launch long training.",
        ],
        "minimal_gru_mlp_boundary": {
            "actor_encoder": "96 -> 256 -> Tanh",
            "actor_gru": "GRU or GRUCell hidden=128, layers=1",
            "actor_head": "128 -> action_dim",
            "critic_encoder": "480 -> 256 -> Tanh",
            "critic_gru": "hidden=128, layers=1",
            "critic_head": "128 -> value",
            "action_log_std": "same learnable parameter as current MAPPO",
            "no_changes": [
                "reward",
                "termination",
                "missile",
                "action space",
                "PID",
                "aircraft XML",
                "observation dimension",
                "attention",
                "HAPPO",
            ],
        },
        "smoke_before_pilot": [
            "policy forward shape test",
            "hidden state reset test",
            "64-step train/eval/save/load smoke",
            "no NaN and actor_dim=96 critic_dim=480 metadata checks",
        ],
        "pilot_after_smoke": "Only after smoke passes, decide whether to run a GRU-MLP 200k pilot.",
    }


def _risks() -> list[str]:
    return [
        "Using GRU with the current flat PPO batch would be incorrect because temporal order and initial hidden states are lost.",
        "Dead-agent hidden states must be masked or reset consistently with red_alive_mask.",
        "Team episode done should remain separate from individual death, as in the alive/done fix.",
        "Recurrent eval needs deterministic action with persistent hidden state and reset at episode boundaries.",
        "Adding attention at the same time would confound recurrence with entity aggregation.",
    ]


def build_audit() -> dict[str, Any]:
    return {
        "original_vanilla_gru_protocol": _audit_parent_vanilla(),
        "original_attention_gru_protocol": _audit_parent_attention(),
        "current_hetero_feedforward_protocol": _audit_current_hetero(),
        "required_code_changes": _required_changes(),
        "risks": _risks(),
        "minimal_gru_plan": _minimal_plan(),
    }


def _markdown(data: dict[str, Any]) -> str:
    vanilla = data["original_vanilla_gru_protocol"]
    attention = data["original_attention_gru_protocol"]
    current = data["current_hetero_feedforward_protocol"]
    lines = [
        "# GRU MAPPO Alignment Audit",
        "",
        "## Summary",
        "",
        "The parent project does not rely on a pure feed-forward policy. Its vanilla",
        "MAPPO path already uses a recurrent actor, and the attention PPO path also",
        "uses recurrent state. This makes GRU-MLP the smallest method step before",
        "attention.",
        "",
        "## Original Vanilla GRU Protocol",
        "",
        f"- Actor has GRU: {vanilla['actor_has_gru']}",
        f"- Critic has GRU: {vanilla['critic_has_gru']}",
        f"- Actor/critic share GRU: {vanilla['actor_critic_share_gru']}",
        f"- GRU type: {vanilla['actor_gru_type']}",
        f"- Hidden size: {vanilla['actor_hidden_size']}",
        f"- Layers: {vanilla['actor_gru_layers']}",
        f"- Stores actor hidden init/final: {vanilla['buffer_stores_rnn_actor_init']}/{vanilla['buffer_stores_rnn_actor_final']}",
        f"- Sequence PPO update: {vanilla['ppo_sequence_update']}",
        f"- Hidden reset on done: {vanilla['hidden_reset_on_done']}",
        "",
        "## Original Attention GRU Protocol",
        "",
        f"- Actor attention: {attention['actor_has_attention']}",
        f"- Actor GRU: {attention['actor_has_gru']}",
        f"- Critic GRU state stored: {attention['critic_has_gru']}",
        f"- Hidden size: {attention['hidden_size']}",
        f"- RNN hidden size: {attention['rnn_hidden_size']}",
        f"- Sequence update: {attention['sequence_update']}",
        "",
        "## Current hetero_uav Gap",
        "",
        f"- Actor feed-forward: {current['actor_is_feedforward']}",
        f"- Buffer stores hidden state: {current['buffer_stores_hidden_state']}",
        f"- Trainer assumes flat batch: {current['trainer_assumes_flat_batch']}",
        f"- Active alive mask present: {current['active_mask_present']}",
        f"- Team done fix present: {current['team_done_fix_present']}",
        "",
        "## Required Code Changes",
    ]
    for item in data["required_code_changes"]:
        lines.append(f"- {item['target']}: {item['change']} Reason: {item['reason']}")
    lines.extend([
        "",
        "## Minimal GRU Plan",
        "",
        f"- Decision this round: {data['minimal_gru_plan']['decision']}",
        "- Implement GRU-MLP only after policy, recurrent buffer, recurrent trainer, runner, eval, save/load, and smoke tests are scoped together.",
        "- Do not change reward, termination, missile, action space, PID, aircraft XML, observation dimension, attention, or HAPPO.",
        "",
        "## Risks",
    ])
    for risk in data["risks"]:
        lines.append(f"- {risk}")
    lines.extend([
        "",
        "## Recommendation",
        "",
        "Do not keep adding steps to the shared MLP baseline. The next method step",
        "should be a minimal GRU-MLP plan and smoke implementation, then a 200k pilot",
        "only after the recurrent smoke path passes.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit GRU MAPPO alignment against parent project protocol.")
    parser.add_argument("--output-json", default="outputs/protocol_audit/gru_mappo_alignment.json")
    parser.add_argument("--output-md", default="outputs/protocol_audit/gru_mappo_alignment.md")
    args = parser.parse_args()

    data = build_audit()
    out_json = ROOT / args.output_json
    out_md = ROOT / args.output_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    out_md.write_text(_markdown(data), encoding="utf-8")

    print(f"vanilla_actor_has_gru: {data['original_vanilla_gru_protocol']['actor_has_gru']}", flush=True)
    print(f"attention_actor_has_gru: {data['original_attention_gru_protocol']['actor_has_gru']}", flush=True)
    print(f"current_buffer_stores_hidden: {data['current_hetero_feedforward_protocol']['buffer_stores_hidden_state']}", flush=True)
    print(f"decision: {data['minimal_gru_plan']['decision']}", flush=True)
    print(f"output_json: {out_json}", flush=True)
    print(f"output_md: {out_md}", flush=True)


if __name__ == "__main__":
    main()
