"""Run one episode with a checkpoint and save ACMI file."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--config", default="uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml")
    p.add_argument("--output", default="outputs/tam_acmi_render")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--device", default="cuda")
    p.add_argument("--deterministic", action="store_true", default=True)
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    from uav_env import make_env
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
    from algorithms.happo.tam_categorical_recurrent_policy import TAMCategoricalRecurrentHAPPOPolicy

    # Load model
    model_path = Path(args.model)
    meta_path = model_path.parent / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    policy = TAMCategoricalRecurrentHAPPOPolicy(
        entity_dim=int(meta.get("entity_dim", 19)),
        actor_obs_dim=int(meta.get("actor_obs_dim", 96)),
        critic_state_dim=int(meta.get("critic_state_dim", 480)),
        action_dim=int(meta.get("action_dim", 4)),
        action_levels=int(meta.get("action_levels", 40)),
        rnn_hidden_size=int(meta.get("rnn_hidden_size", 128)),
    ).to(device)
    policy.load(str(model_path), map_location=device)
    policy.eval()

    adapter = HeteroObsAdapterV2()
    env = make_env(str(ROOT / args.config), env_type="jsbsim_hetero",
                   hetero_reward_mode="happo_ref_v0", max_steps=args.max_steps)

    from algorithms.mappo.opponent_policy import OpponentPolicy
    opponent = OpponentPolicy(mode="tam_direct_fsm", seed=args.seed + 100)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    acmi_path = out_dir / f"episode_seed{args.seed}.acmi"

    env.render()
    obs, info = env.reset(seed=args.seed)
    red_ids = env.red_ids
    roles = [0 if env.agent_roles.get(rid) == "mav" else 1 for rid in red_ids]
    rnn_h = np.zeros((len(red_ids), 128), dtype=np.float32)

    total_reward = 0.0
    death_step = None
    for step in range(args.max_steps):
        adapted = adapter.adapt_all(obs, info=info, red_ids=red_ids, blue_ids=env.blue_ids)
        actor_obs = np.stack([adapted["actor_obs"].get(rid, np.zeros(96, dtype=np.float32)) for rid in red_ids])
        with torch.no_grad():
            out = policy.act(torch.as_tensor(actor_obs, device=device), roles=roles,
                            deterministic=args.deterministic,
                            rnn_hidden=torch.as_tensor(rnn_h, device=device))
        actions = out["action"].cpu().numpy()
        rnn_h = out["rnn_hidden"].cpu().numpy()
        action_dict = {rid: actions[i].astype(np.int64) for i, rid in enumerate(red_ids)}
        action_dict.update(opponent.act(obs, env.blue_ids, env=env))
        next_obs, rewards, terminated, truncated, next_info = env.step(action_dict)
        total_reward += float(rewards.get("red_0", 0.0))
        if env.red_planes.get("red_0") and not env.red_planes["red_0"].is_alive and death_step is None:
            death_step = step + 1
            print(f"MAV died at step {death_step}: {env._death_reasons.get('red_0','unknown')}")
        if all(terminated.values()) or all(truncated.values()):
            print(f"Episode ended at step {step+1}: terminated={all(terminated.values())} truncated={all(truncated.values())}")
            break
        obs, info = next_obs, next_info

    n = env.save_acmi(str(acmi_path))
    env.close()
    print(f"ACMI saved: {acmi_path} ({n} frames)")
    print(f"Total reward (red_0): {total_reward:.2f}")
    if death_step:
        print(f"MAV death step: {death_step}")


if __name__ == "__main__":
    main()
