"""Diagnose whether death-causing transitions enter actor loss.

Single-purpose script: find MAV Crash_LowAlt episode, inspect active masks
at the death transition and surrounding timesteps.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _build_red_alive_mask(info: dict, env, red_ids: list[str]) -> np.ndarray:
    mask = np.zeros(len(red_ids), dtype=np.float32)
    for i, rid in enumerate(red_ids):
        agent_info = info.get(rid, {}) if isinstance(info, dict) else {}
        if isinstance(agent_info, dict) and "alive" in agent_info:
            alive = bool(agent_info["alive"])
        else:
            sim = env.red_planes.get(rid)
            alive = bool(sim is not None and sim.is_alive)
        mask[i] = 1.0 if alive else 0.0
    return mask


def run_diagnostic(config_path: str, episodes: int, max_steps: int, device_str: str) -> dict:
    from uav_env import make_env
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
    from algorithms.happo.tam_categorical_recurrent_policy import TAMCategoricalRecurrentHAPPOPolicy

    device = torch.device(device_str)
    adapter = HeteroObsAdapterV2()

    # Try to use old checkpoint for deterministic crash reproduction
    ckpt_candidates = [
        ROOT / "outputs" / "tam_papermode_3v2_2M_probe" / "latest_failure" / "model.pt",
        ROOT / "outputs" / "tam_papermode_3v2_2M_probe" / "latest" / "model.pt",
        ROOT / "outputs" / "tam_papermode_3v2_2M_probe" / "eval_checkpoints" / "step_803072" / "model.pt",
    ]
    checkpoint_path = None
    for cp in ckpt_candidates:
        if cp.exists():
            checkpoint_path = str(cp)
            break

    policy = None
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}", flush=True)
        meta_path = Path(checkpoint_path).parent / "meta.json"
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        policy = TAMCategoricalRecurrentHAPPOPolicy(
            entity_dim=int(meta.get("entity_dim", 19)),
            actor_obs_dim=int(meta.get("actor_obs_dim", 96)),
            critic_state_dim=int(meta.get("critic_state_dim", 480)),
            action_dim=int(meta.get("action_dim", 4)),
            action_levels=int(meta.get("action_levels", 40)),
            rnn_hidden_size=int(meta.get("rnn_hidden_size", 128)),
        ).to(device)
        policy.load(checkpoint_path, map_location=device)
        policy.eval()
    else:
        print("No checkpoint found, using random actions", flush=True)

    findings = []
    for ep in range(episodes):
        env = make_env(
            str(ROOT / config_path) if not Path(config_path).is_absolute() else config_path,
            env_type="jsbsim_hetero",
            hetero_reward_mode="happo_ref_v0",
            max_steps=max_steps,
        )
        obs, info = env.reset(seed=ep * 100 + 42)

        red_ids = env.red_ids
        roles = [0 if env.agent_roles.get(rid) == "mav" else 1 for rid in red_ids]

        transitions = []
        mav_death_episode = False
        mav_death_idx = None

        for step_idx in range(max_steps):
            alive_before = _build_red_alive_mask(info, env, red_ids)
            adapted = adapter.adapt_all(obs, info=info, red_ids=red_ids, blue_ids=env.blue_ids)
            actor_obs_np = np.stack([
                adapted["actor_obs"].get(rid, np.zeros(adapter.flat_actor_obs_dim, dtype=np.float32))
                for rid in red_ids
            ])

            if policy is not None:
                rnn_hidden = np.zeros((len(red_ids), 128), dtype=np.float32) if step_idx == 0 else rnn_hidden
                with torch.no_grad():
                    out = policy.act(
                        torch.as_tensor(actor_obs_np, device=device),
                        roles=roles,
                        deterministic=True,
                        rnn_hidden=torch.as_tensor(rnn_hidden, device=device),
                    )
                actions_np = out["action"].cpu().numpy()
                rnn_hidden = out.get("rnn_hidden", rnn_hidden).cpu().numpy() if "rnn_hidden" in out else rnn_hidden
            else:
                actions_np = np.random.randint(0, 40, (len(red_ids), 4)).astype(np.int64)

            action_dict = {rid: actions_np[i].astype(np.int64) for i, rid in enumerate(red_ids)}
            next_obs, rewards, terminated, truncated, next_info = env.step(action_dict)

            alive_after = _build_red_alive_mask(next_info, env, red_ids)
            death_mask = alive_before * (1.0 - alive_after)

            transitions.append({
                "step": step_idx,
                "mav_alive_before": bool(alive_before[0] > 0.5),
                "mav_alive_after": bool(alive_after[0] > 0.5),
                "mav_death_mask": float(death_mask[0]),
                "mav_reward": float(rewards.get("red_0", 0.0)),
                "mav_action": actions_np[0].tolist(),
                "mav_death_reason": env._death_reasons.get("red_0") if not alive_after[0] else None,
            })

            if death_mask[0] > 0.5 and mav_death_idx is None:
                mav_death_episode = True
                mav_death_idx = step_idx

            if all(terminated.values()) or all(truncated.values()):
                break
            obs, info = next_obs, next_info

        env.close()

        if mav_death_episode and mav_death_idx is not None:
            t = mav_death_idx
            window = []
            for offset in range(-2, 3):
                idx = t + offset
                if 0 <= idx < len(transitions):
                    tr = transitions[idx]
                    window.append({
                        "relative": f"t{offset:+d}",
                        "step": tr["step"],
                        "mav_alive_before": tr["mav_alive_before"],
                        "mav_alive_after": tr["mav_alive_after"],
                        "mav_death_mask": tr["mav_death_mask"],
                        "actor_active_would_be": 1.0 if tr["mav_alive_before"] else 0.0,
                        "reward": tr["mav_reward"],
                        "action": tr["mav_action"],
                        "death_reason": tr.get("mav_death_reason"),
                    })
            findings.append({
                "episode": ep,
                "mav_death_step": t,
                "death_reason": transitions[t].get("mav_death_reason"),
                "window": window,
                "death_transition_uses_active_1": bool(transitions[t]["mav_alive_before"]),
                "death_transition_alive_after_0": not bool(transitions[t]["mav_alive_after"]),
                "post_death_active_0": all(
                    tr["mav_alive_before"] == False
                    for tr in transitions[t+1:t+3] if t+1 < len(transitions)
                ) if t + 1 < len(transitions) else True,
            })
            break  # One episode is enough

    # Also check from buffer perspective if possible
    trainer_active = verify_from_trainer_perspective(
        config_path, device_str, checkpoint_path)

    return {
        "checkpoint_used": checkpoint_path,
        "episodes_tested": len(findings),
        "findings": findings,
        "conclusion": {
            "death_transition_actor_active": all(
                f["death_transition_uses_active_1"] for f in findings
            ) if findings else "no_death_found",
            "death_transition_not_alive_after": all(
                f["death_transition_alive_after_0"] for f in findings
            ) if findings else "no_death_found",
            "post_death_transitions_inactive": all(
                f["post_death_active_0"] for f in findings
            ) if findings else "no_death_found",
        },
        "trainer_simulation": trainer_active,
    }


def verify_from_trainer_perspective(config_path, device_str, checkpoint_path):
    """Simulate what the training loop would store."""
    from uav_env import make_env
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
    from algorithms.happo.happo_buffer import HAPPORolloutBuffer
    from algorithms.happo.tam_categorical_recurrent_policy import TAMCategoricalRecurrentHAPPOPolicy

    device = torch.device(device_str)
    adapter = HeteroObsAdapterV2()

    env = make_env(
        str(ROOT / config_path) if not Path(config_path).is_absolute() else config_path,
        env_type="jsbsim_hetero", hetero_reward_mode="happo_ref_v0", max_steps=1000,
    )
    red_ids = env.red_ids
    roles = [0 if env.agent_roles.get(rid) == "mav" else 1 for rid in red_ids]

    policy = None
    if checkpoint_path:
        meta_path = Path(checkpoint_path).parent / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        policy = TAMCategoricalRecurrentHAPPOPolicy(
            entity_dim=int(meta.get("entity_dim", 19)),
            actor_obs_dim=int(meta.get("actor_obs_dim", 96)),
            critic_state_dim=int(meta.get("critic_state_dim", 480)),
            action_dim=int(meta.get("action_dim", 4)),
            action_levels=int(meta.get("action_levels", 40)),
            rnn_hidden_size=int(meta.get("rnn_hidden_size", 128)),
        ).to(device)
        policy.load(checkpoint_path, map_location=device)
        policy.eval()

    buf = HAPPORolloutBuffer(
        max_len=500, num_red=len(red_ids), actor_dim=adapter.flat_actor_obs_dim,
        critic_dim=adapter.critic_state_dim, action_dim=4, role_ids=roles,
        rnn_hidden_size=128, action_dtype=np.int64, num_envs=1,
    )

    obs, info = env.reset(seed=42)
    for step in range(500):
        alive_before = _build_red_alive_mask(info, env, red_ids)
        adapted = adapter.adapt_all(obs, info=info, red_ids=red_ids, blue_ids=env.blue_ids)
        actor_obs_np = np.stack([
            adapted["actor_obs"].get(rid, np.zeros(96, dtype=np.float32)) for rid in red_ids
        ])

        if policy is not None:
            rnn_h = np.zeros((len(red_ids), 128), dtype=np.float32) if step == 0 else rnn_h
            with torch.no_grad():
                out = policy.act(torch.as_tensor(actor_obs_np, device=device), roles=roles,
                                deterministic=True, rnn_hidden=torch.as_tensor(rnn_h, device=device))
            actions_np = out["action"].cpu().numpy()
            log_probs_np = out["log_prob"].cpu().numpy()
            rnn_h = out.get("rnn_hidden", rnn_h).cpu().numpy() if "rnn_hidden" in out else rnn_h
        else:
            actions_np = np.random.randint(0, 40, (len(red_ids), 4)).astype(np.int64)
            log_probs_np = np.zeros(len(red_ids), dtype=np.float32)

        action_dict = {rid: actions_np[i].astype(np.int64) for i, rid in enumerate(red_ids)}
        next_obs, rewards, terminated, truncated, next_info = env.step(action_dict)

        next_active = _build_red_alive_mask(next_info, env, red_ids)
        death_transition_mask = alive_before * (1.0 - next_active)
        reward_np = np.array([float(rewards.get(rid, 0.0)) for rid in red_ids], dtype=np.float32)
        done_np = np.full(len(red_ids), float(all(terminated.values()) or all(truncated.values())), dtype=np.float32)

        buf.store(
            actor_obs_np, adapted["critic_state"], actions_np, log_probs_np,
            reward_np, done_np, 0.0, alive_before, next_value=0.0, env_id=0,
            env_step_index=step,
            episode_start_masks=np.full(len(red_ids), 0.0, dtype=np.float32),
            death_transition_masks=death_transition_mask,
            rnn_hidden=np.zeros((len(red_ids), 128), dtype=np.float32),
        )

        if death_transition_mask[0] > 0.5:
            break
        if all(terminated.values()) or all(truncated.values()):
            obs, info = env.reset(seed=42 + step)
        else:
            obs, info = next_obs, next_info

    env.close()

    stored = {
        "active_masks_mav": buf.active_masks[:buf.pos, 0].tolist(),
        "death_transition_masks_mav": buf.death_transition_masks[:buf.pos, 0].tolist(),
        "rewards_mav": buf.rewards[:buf.pos, 0].tolist(),
    }

    death_idx = next((i for i, v in enumerate(stored["death_transition_masks_mav"]) if v > 0.5), None)
    return {
        "death_transition_index_in_buffer": death_idx,
        "death_transition_active_mask_at_index": stored["active_masks_mav"][death_idx] if death_idx is not None else None,
        "death_transition_stored_with_active_1": (
            stored["active_masks_mav"][death_idx] > 0.5
        ) if death_idx is not None else None,
        "post_death_active_masks": stored["active_masks_mav"][death_idx+1:death_idx+4] if death_idx is not None else [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml")
    parser.add_argument("--output-dir", default="outputs/tam_death_mask_semantics")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    print("=== Death Mask Semantics Diagnostic ===", flush=True)
    report = run_diagnostic(args.config, args.episodes, args.max_steps, args.device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "death_mask_semantics.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    conc = report["conclusion"]
    lines = [
        "# Death Mask Semantics Diagnostic",
        "",
        f"Checkpoint: {report.get('checkpoint_used') or 'random'}",
        "",
        "## Conclusion",
        "",
        f"1. Death transition actor_active_mask=1: **{conc['death_transition_actor_active']}**",
        f"2. Death transition alive_after=0: **{conc['death_transition_not_alive_after']}**",
        f"3. Post-death transitions inactive: **{conc['post_death_transitions_inactive']}**",
        "",
        "## Trainer Buffer Simulation",
    ]
    ts = report.get("trainer_simulation", {})
    lines.append(f"- death idx in buffer: {ts.get('death_transition_index_in_buffer')}")
    lines.append(f"- active_mask at death idx: {ts.get('death_transition_active_mask_at_index')}")
    lines.append(f"- stored with active=1: {ts.get('death_transition_stored_with_active_1')}")
    lines.append(f"- post-death active: {ts.get('post_death_active_masks')}")

    for f in report.get("findings", [])[:1]:
        lines.extend(["", "## Death Transition Window", ""])
        for w in f["window"]:
            lines.append(
                f"- {w['relative']} step={w['step']} alive_before={w['mav_alive_before']} "
                f"alive_after={w['mav_alive_after']} actor_active={w['actor_active_would_be']} "
                f"reward={w['reward']:.2f} action={w['action']} reason={w.get('death_reason')}"
            )

    (out_dir / "death_mask_semantics.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
