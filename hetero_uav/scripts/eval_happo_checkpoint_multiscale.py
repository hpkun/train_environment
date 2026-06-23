"""Eval-only script: load a checkpoint, evaluate across configs, output JSON+CSV.

No training updates. No PPO. No new checkpoints saved.

Usage:
  python scripts/eval_happo_checkpoint_multiscale.py \\
    --checkpoint outputs/xxx/best_combined \\
    --config uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_paper_reward_v1.yaml \\
    --eval-configs 3v2.yaml 5v4.yaml 7v6.yaml \\
    --episodes 50 --device cuda --output-json eval.json --output-csv eval.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.happo import (
    BRMAEntityHAPPOReferencePolicy,
    BRMARecurrentMaskedHAPPOReferencePolicy,
    BRMARecurrentHAPPOReferencePolicy,
    EntityHAPPOReferencePolicy,
    HAPPOReferencePolicy,
)
from algorithms.happo.hetero_entity_recurrent_policy import (
    HeteroEntityRecurrentPolicy,
    validate_entity_policy_meta,
)
from algorithms.happo.rollout_safety import (
    zero_inactive_actions,
    zero_inactive_hidden,
)
from uav_env.JSBSim.adapters.hetero_entity_set_adapter import HeteroEntitySetAdapter

try:
    from algorithms.mappo.opponent_policy import OpponentPolicy
    from uav_env import make_env
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
except ModuleNotFoundError:
    OpponentPolicy = None
    make_env = None
    HeteroObsAdapterV2 = None

DEFAULT_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_paper_reward_v1.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4_f16_mav_surrogate_paper_reward_v1.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_7v6_f16_mav_surrogate_paper_reward_v1.yaml",
]


def _load_meta(model_path: Path) -> dict:
    meta_path = model_path.parent / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    # Also try model_path directly if it's a directory
    if model_path.is_dir():
        mp = model_path / "model.pt"
        meta_path = model_path / "meta.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def _resolve_checkpoint(checkpoint_arg: str) -> Path:
    """Resolve --checkpoint to model.pt path.

    Accepts:
      - outputs/xxx/best_combined  (directory with model.pt + meta.json)
      - outputs/xxx/best_combined/model.pt
      - outputs/xxx/latest/model.pt
    """
    p = Path(checkpoint_arg)
    if not p.is_absolute():
        p = ROOT / p
    if p.is_dir():
        return p / "model.pt"
    return p


def _build_policy_from_meta(meta: dict, device: torch.device):
    policy_arch = meta.get("policy_arch", "flat")
    if policy_arch == "hetero_entity_recurrent":
        validate_entity_policy_meta(meta)
        return HeteroEntityRecurrentPolicy(
            entity_dim=int(meta.get("entity_dim", 21)),
            action_dim=3,
            hidden_dim=int(meta.get("hidden_dim", 128)),
            rnn_hidden_size=int(meta.get("rnn_hidden_size", 128)),
            num_attention_heads=int(meta.get("num_attention_heads", 4)),
        ).to(device)
    if policy_arch == "flat":
        return HAPPOReferencePolicy(
            int(meta.get("actor_obs_dim", 96)),
            int(meta.get("critic_state_dim", 480)),
        ).to(device)
    if policy_arch == "entity_attention":
        return EntityHAPPOReferencePolicy(
            entity_dim=int(meta.get("entity_dim", 19)),
            critic_state_dim=int(meta.get("critic_state_dim", 480)),
            action_dim=3,
        ).to(device)
    if policy_arch == "brma_entity":
        return BRMAEntityHAPPOReferencePolicy(
            entity_dim=int(meta.get("entity_dim", 19)),
            critic_state_dim=int(meta.get("critic_state_dim", 480)),
            action_dim=3,
        ).to(device)
    if policy_arch == "brma_recurrent":
        return BRMARecurrentHAPPOReferencePolicy(
            entity_dim=int(meta.get("entity_dim", 19)),
            critic_state_dim=int(meta.get("critic_state_dim", 480)),
            action_dim=3,
            rnn_hidden_size=int(meta.get("rnn_hidden_size", 128)),
        ).to(device)
    if policy_arch == "brma_recurrent_masked":
        return BRMARecurrentMaskedHAPPOReferencePolicy(
            entity_dim=int(meta.get("entity_dim", 19)),
            critic_state_dim=int(meta.get("critic_state_dim", 480)),
            action_dim=3,
            rnn_hidden_size=int(meta.get("rnn_hidden_size", 128)),
            random_scale_mask=bool(meta.get("random_scale_mask", False)),
            random_mask_prob=float(meta.get("random_mask_prob", 0.25)),
            biased_mask=bool(meta.get("biased_mask", False)),
        ).to(device)
    raise ValueError(f"unsupported policy_arch: {policy_arch}")


def _team_done(terminated: dict, truncated: dict) -> bool:
    return bool(terminated and all(terminated.values())) or bool(truncated and all(truncated.values()))


def _episode_result(env) -> dict:
    red_alive = sum(1 for rid in env.red_ids
                    if env.red_planes.get(rid) is not None and env.red_planes[rid].is_alive)
    blue_alive = sum(1 for bid in env.blue_ids
                     if env.blue_planes.get(bid) is not None and env.blue_planes[bid].is_alive)
    red_dead = len(env.red_ids) - red_alive
    blue_dead = len(env.blue_ids) - blue_alive
    mav_alive = bool(env.red_planes.get("red_0") and env.red_planes["red_0"].is_alive)
    if blue_alive == 0 and red_alive > 0:
        return {"winner": "red", "end_reason": "blue_eliminated",
                "red_dead": red_dead, "blue_dead": blue_dead, "mav_alive": mav_alive,
                "timeout_alive_advantage": ""}
    elif red_alive == 0 and blue_alive > 0:
        return {"winner": "blue", "end_reason": "red_eliminated",
                "red_dead": red_dead, "blue_dead": blue_dead, "mav_alive": mav_alive,
                "timeout_alive_advantage": ""}
    elif red_alive == 0 and blue_alive == 0:
        return {"winner": "draw", "end_reason": "mutual_elimination",
                "red_dead": red_dead, "blue_dead": blue_dead, "mav_alive": mav_alive,
                "timeout_alive_advantage": ""}
    # Timeout: always draw (asymmetric counts do NOT make a winner)
    if red_alive > blue_alive:
        timeout_adv = "red_timeout_alive_advantage"
    elif blue_alive > red_alive:
        timeout_adv = "blue_timeout_alive_advantage"
    else:
        timeout_adv = "timeout_draw"
    return {"winner": "draw", "end_reason": "timeout",
            "red_dead": red_dead, "blue_dead": blue_dead, "mav_alive": mav_alive,
            "timeout_alive_advantage": timeout_adv}


def _update_missile_stats(mstats: dict, info: dict, prev_hits: dict) -> None:
    mt = info.get("__missile_term__", {})
    if isinstance(mt, dict):
        red_hit = int(mt.get("red", {}).get("hit", 0))
        blue_hit = int(mt.get("blue", {}).get("hit", 0))
        mstats["red_hits"] += max(red_hit - prev_hits.get("red", 0), 0)
        mstats["blue_hits"] += max(blue_hit - prev_hits.get("blue", 0), 0)
        prev_hits["red"] = red_hit
        prev_hits["blue"] = blue_hit
    for aid, agent_info in (info or {}).items():
        if isinstance(agent_info, dict):
            fired = int(agent_info.get("missiles_fired_this_step", 0))
            if aid.startswith("red_"):
                mstats["red_fired"] += fired
            else:
                mstats["blue_fired"] += fired


def evaluate_config(policy, cfg_path: str, args, adapter, device) -> dict:
    env = make_env(cfg_path, env_type="jsbsim_hetero")
    opponent = OpponentPolicy(mode=args.opponent_policy, seed=args.seed + 33)
    entity_mode = isinstance(adapter, HeteroEntitySetAdapter)
    _rnn_hidden_size = getattr(policy, "rnn_hidden_size", 0)

    returns, lengths, red_alive, blue_alive = [], [], [], []
    results, missile_stats_list = [], []
    mav_sat_values, uav_sat_values = [], []
    nan_detected = False

    for ep in range(args.episodes):
        ep_seed = args.seed + ep
        obs, info = env.reset(seed=ep_seed)
        eval_rnn_hidden = None
        if _rnn_hidden_size > 0:
            eval_rnn_hidden = np.zeros((len(env.red_ids), _rnn_hidden_size), dtype=np.float32)
        ep_ret, ep_len = 0.0, 0
        mstats = {"red_fired": 0, "blue_fired": 0, "red_hits": 0, "blue_hits": 0}
        prev_hits = {"red": 0, "blue": 0}

        while True:
            adapted = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
            active = np.ones(len(env.red_ids), dtype=np.float32)
            for i, rid in enumerate(env.red_ids):
                ai = (info or {}).get(rid, {})
                active[i] = 1.0 if ai.get("alive", True) else 0.0
            active_rows = active > 0.5

            if entity_mode:
                actor_tokens = adapted["actor_entity_tokens"].copy()
                actor_keep = adapted["actor_keep_mask"].copy()
                critic_tokens = adapted["critic_entity_tokens"].copy()
                critic_keep = adapted["critic_keep_mask"].copy()
                cc = adapted.get("critic_counts", np.zeros(4, dtype=np.float32))
                actor_tokens[~active_rows] = 0.0
                actor_keep[~active_rows] = 0.0
                actor_keep[~active_rows, 0] = 1.0
                if eval_rnn_hidden is not None:
                    eval_rnn_hidden = zero_inactive_hidden(eval_rnn_hidden, active)
                if active_rows.any() and (
                    not np.isfinite(actor_tokens[active_rows]).all()
                    or not np.isfinite(critic_tokens[critic_keep > 0.5]).all()
                ):
                    nan_detected = True; break
                act_kwargs = {}
                if eval_rnn_hidden is not None:
                    act_kwargs["rnn_hidden"] = torch.as_tensor(eval_rnn_hidden, device=device)
                with torch.no_grad():
                    out = policy.act(
                        torch.as_tensor(actor_tokens, device=device),
                        torch.as_tensor(actor_keep, device=device),
                        torch.as_tensor(adapted["role_ids"], device=device),
                        torch.as_tensor(critic_tokens, device=device),
                        torch.as_tensor(critic_keep, device=device),
                        deterministic=True,
                        critic_counts=torch.as_tensor(cc, device=device),
                        **act_kwargs)
            else:
                actor_obs = np.stack([
                    adapted["actor_obs"].get(rid, np.zeros(adapter.flat_actor_obs_dim, dtype=np.float32))
                    for rid in env.red_ids])
                critic = adapted["critic_state"]
                with torch.no_grad():
                    out = policy.act(
                        torch.as_tensor(actor_obs, device=device),
                        roles=[0 if env.agent_roles.get(rid) == "mav" else 1 for rid in env.red_ids],
                        critic_state=torch.as_tensor(critic, device=device),
                        deterministic=True)

            actions_raw = out["action"].cpu().numpy()
            actions = zero_inactive_actions(actions_raw, active)
            if eval_rnn_hidden is not None and "rnn_hidden" in out:
                eval_rnn_hidden = zero_inactive_hidden(out["rnn_hidden"].cpu().numpy(), active)
            if active_rows.any() and not np.isfinite(actions[active_rows]).all():
                nan_detected = True; break

            action_dict = {rid: actions[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}
            action_dict.update(opponent.act(obs, env.blue_ids, env=env))
            obs, rewards, terminated, truncated, info = env.step(action_dict)
            _update_missile_stats(mstats, info, prev_hits)

            mav_sat_values.append(float(np.mean(np.abs(actions[0:1]) >= 0.999)))
            if len(env.red_ids) > 1:
                uav_sat_values.append(float(np.mean(np.abs(actions[1:]) >= 0.999)))

            ep_ret += sum(float(rewards.get(rid, 0.0)) for rid in env.red_ids)
            ep_len += 1
            if _team_done(terminated, truncated):
                break

        ra = sum(1 for rid in env.red_ids if env.red_planes.get(rid) is not None and env.red_planes[rid].is_alive)
        ba = sum(1 for bid in env.blue_ids if env.blue_planes.get(bid) is not None and env.blue_planes[bid].is_alive)
        returns.append(ep_ret); lengths.append(ep_len)
        red_alive.append(ra); blue_alive.append(ba)
        results.append(_episode_result(env))
        missile_stats_list.append(mstats)

    env.close()

    reason_counts = Counter(r["end_reason"] for r in results)
    winner_counts = Counter(r["winner"] for r in results)
    timeout_adv_counts = Counter(r.get("timeout_alive_advantage", "") for r in results)
    n = max(len(results), 1)
    num_red = len(env.red_ids)
    num_blue = len(env.blue_ids)
    red_dead_mean = float(np.mean([r["red_dead"] for r in results]))
    blue_dead_mean = float(np.mean([r["blue_dead"] for r in results]))
    red_kf = blue_dead_mean / max(num_blue, 1)
    red_lf = red_dead_mean / max(num_red, 1)
    red_hits_mean = float(np.mean([m["red_hits"] for m in missile_stats_list]))
    blue_hits_mean = float(np.mean([m["blue_hits"] for m in missile_stats_list]))
    return {
        "config": cfg_path,
        "num_red": num_red, "num_blue": num_blue,
        "avg_return": float(np.mean(returns)),
        "avg_length": float(np.mean(lengths)),
        "red_win_rate": reason_counts["blue_eliminated"] / n,
        "blue_win_rate": reason_counts["red_eliminated"] / n,
        "draw_rate": winner_counts["draw"] / n,
        "timeout_rate": reason_counts["timeout"] / n,
        "red_elimination_win_rate": reason_counts["blue_eliminated"] / n,
        "blue_elimination_win_rate": reason_counts["red_eliminated"] / n,
        "red_timeout_alive_advantage_rate": timeout_adv_counts.get("red_timeout_alive_advantage", 0) / n,
        "blue_timeout_alive_advantage_rate": timeout_adv_counts.get("blue_timeout_alive_advantage", 0) / n,
        "red_kill_fraction": red_kf,
        "red_loss_fraction": red_lf,
        "net_kill_fraction": red_kf - red_lf,
        "mav_survival_rate": sum(1 for r in results if r["mav_alive"]) / n,
        "red_alive_final_mean": float(np.mean(red_alive)),
        "blue_alive_final_mean": float(np.mean(blue_alive)),
        "red_dead_mean": red_dead_mean,
        "blue_dead_mean": blue_dead_mean,
        "kill_death_ratio": blue_dead_mean / max(red_dead_mean, 1e-6),
        "red_missiles_fired_mean": float(np.mean([m["red_fired"] for m in missile_stats_list])),
        "blue_missiles_fired_mean": float(np.mean([m["blue_fired"] for m in missile_stats_list])),
        "red_missile_hits_mean": red_hits_mean,
        "blue_missile_hits_mean": blue_hits_mean,
        "mav_action_saturation_rate": float(np.mean(mav_sat_values)) if mav_sat_values else 0.0,
        "uav_action_saturation_rate": float(np.mean(uav_sat_values)) if uav_sat_values else 0.0,
        "nan_detected": nan_detected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir or model.pt path")
    parser.add_argument("--config", help="Training config (use with --eval-configs for eval)")
    parser.add_argument("--eval-configs", nargs="*", default=None,
                        help="Config paths to evaluate (default: 3v2/5v4/7v6 F16-MAV paper)")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--opponent-policy", default="brma_rule")
    parser.add_argument("--output-json", default=None, help="Path for JSON summary")
    parser.add_argument("--output-csv", default=None, help="Path for CSV summary")
    args = parser.parse_args()

    # Resolve checkpoint
    ckpt = _resolve_checkpoint(args.checkpoint)
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
        return 2

    meta = _load_meta(ckpt)
    device = torch.device(args.device)
    policy = _build_policy_from_meta(meta, device)
    policy.load(str(ckpt), map_location=device)
    policy.eval()

    entity_mode = meta.get("policy_arch") == "hetero_entity_recurrent"
    adapter = HeteroEntitySetAdapter() if entity_mode else HeteroObsAdapterV2()

    configs = args.eval_configs or DEFAULT_CONFIGS
    records = []

    print(f"algorithm: {meta.get('policy_arch', '?')}")
    print(f"episodes: {args.episodes}")
    print(f"configs: {len(configs)}")
    print()

    for cfg in configs:
        print(f"=== {cfg} ===", flush=True)
        record = evaluate_config(policy, cfg, args, adapter, device)
        records.append(record)
        for key in ["avg_return", "red_win_rate", "blue_win_rate", "timeout_rate",
                     "red_elimination_win_rate", "red_timeout_alive_advantage_rate",
                     "red_kill_fraction", "net_kill_fraction",
                     "mav_survival_rate", "red_missile_hits_mean", "blue_missile_hits_mean"]:
            print(f"  {key}: {record[key]}", flush=True)
        print()

    # JSON output
    if args.output_json:
        out_j = ROOT / args.output_json if not Path(args.output_json).is_absolute() else Path(args.output_json)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        out_j.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"JSON: {out_j}")

    # CSV output
    if args.output_csv:
        out_c = ROOT / args.output_csv if not Path(args.output_csv).is_absolute() else Path(args.output_csv)
        out_c.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "config", "num_red", "num_blue",
            "avg_return", "avg_length",
            "red_win_rate", "blue_win_rate", "draw_rate", "timeout_rate",
            "red_elimination_win_rate", "red_timeout_alive_advantage_rate",
            "red_kill_fraction", "net_kill_fraction",
            "mav_survival_rate",
            "red_missile_hits_mean", "blue_missile_hits_mean",
            "nan_detected",
        ]
        with open(out_c, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in records:
                writer.writerow(r)
        print(f"CSV: {out_c}")


if __name__ == "__main__":
    main()
