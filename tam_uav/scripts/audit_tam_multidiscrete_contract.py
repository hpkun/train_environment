"""Audit the formal TAM 4D MultiDiscrete categorical action contract."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from gymnasium.spaces import Box, MultiDiscrete

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.happo import TAMCategoricalRecurrentHAPPOPolicy
from algorithms.happo.happo_buffer import HAPPORolloutBuffer
from algorithms.mappo.opponent_policy import OpponentPolicy
from uav_env import make_env
from uav_env.JSBSim.env import UavCombatEnv


CONFIGS = [
    "uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml",
    "uav_env/JSBSim/configs/tam_happo_f22_5v4_direct.yaml",
]


def _check(name, condition, detail):
    return {"name": name, "passed": bool(condition), "detail": str(detail)}


def run_audit():
    checks = []
    for cfg in CONFIGS:
        data = yaml.safe_load((ROOT / cfg).read_text(encoding="utf-8"))
        checks.append(_check(
            f"config:{Path(cfg).stem}",
            data.get("tam_action_distribution") == "multidiscrete_categorical",
            data.get("tam_action_distribution"),
        ))
        env = make_env(cfg)
        spaces = [env.action_space[aid] for aid in env.red_ids + env.blue_ids]
        checks.append(_check(
            f"env_space:{Path(cfg).stem}",
            all(isinstance(space, MultiDiscrete) and np.array_equal(space.nvec, [40] * 4)
                for space in spaces),
            type(spaces[0]).__name__,
        ))
        obs, _ = env.reset(seed=0)
        indices = np.array([0, 13, 26, 39], dtype=np.int64)
        env._parse_actions({env.red_ids[0]: indices})
        recorded = env._last_tam_action_commands[env.red_ids[0]]
        checks.append(_check(
            f"same_indices:{Path(cfg).stem}",
            recorded["action_distribution"] == "multidiscrete_categorical"
            and np.array_equal(recorded["action_indices"], indices),
            recorded,
        ))
        blue = OpponentPolicy("tam_direct_fsm").act(obs, env.blue_ids, env=env)
        checks.append(_check(
            f"blue_indices:{Path(cfg).stem}",
            all(action.dtype == np.int64 and action.shape == (4,) for action in blue.values()),
            {key: value.tolist() for key, value in blue.items()},
        ))
        env.close()

    policy = TAMCategoricalRecurrentHAPPOPolicy(hidden_dim=32, rnn_hidden_size=32)
    names = [name for name, _ in policy.named_parameters()]
    checks.append(_check(
        "categorical_policy",
        not any("action_log_std" in name for name in names)
        and "Normal" not in (ROOT / "algorithms/happo/tam_categorical_recurrent_policy.py").read_text(encoding="utf-8")
        and any(isinstance(module, torch.nn.MultiheadAttention) for module in policy.critic.modules()),
        type(policy).__name__,
    ))
    obs = torch.randn(2, 96)
    state = torch.randn(480)
    hidden = policy.init_hidden(2)
    with torch.no_grad():
        out = policy.act(obs, [0, 1], state, rnn_hidden=hidden)
        evaluated, *_ = policy.evaluate_actions(
            obs, [0, 1], state, out["action"], rnn_hidden=hidden
        )
    checks.append(_check(
        "categorical_log_prob_replay",
        torch.allclose(out["log_prob"], evaluated),
        float((out["log_prob"] - evaluated).abs().max()),
    ))
    buffer = HAPPORolloutBuffer(1, 2, 96, 480, 4, [0, 1], action_dtype="int64")
    checks.append(_check("buffer_int64", buffer.actions.dtype == np.int64, buffer.actions.dtype))

    legacy = UavCombatEnv(
        max_num_red=1, max_num_blue=1, action_interface="tam_direct_fcs_4d",
        tam_action_distribution="continuous_quantized",
    )
    checks.append(_check(
        "legacy_box_only", isinstance(legacy.action_space[legacy.red_ids[0]], Box),
        type(legacy.action_space[legacy.red_ids[0]]).__name__,
    ))
    legacy.close()
    return {"passed": all(item["passed"] for item in checks), "checks": checks}


def main():
    result = run_audit()
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    (output / "tam_multidiscrete_contract_audit.json").write_text(
        json.dumps(result, indent=2, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)),
        encoding="utf-8",
    )
    lines = ["# TAM MultiDiscrete Contract Audit", ""]
    lines.extend(
        f"- {'PASS' if item['passed'] else 'FAIL'} `{item['name']}`: {item['detail']}"
        for item in result["checks"]
    )
    (output / "tam_multidiscrete_contract_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
