from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from uav_env.make_env import make_env


CONFIG_DIR = Path(__file__).parents[1] / "uav_env" / "JSBSim" / "configs"


@pytest.mark.parametrize(
    ("name", "red_count", "blue_count"),
    [
        ("tam_happo_f22_3v2_direct.yaml", 3, 2),
        ("tam_happo_f22_5v4_direct.yaml", 5, 4),
    ],
)
def test_tam_formal_config_contract(name: str, red_count: int, blue_count: int):
    path = CONFIG_DIR / name
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["action_interface"] == "tam_direct_fcs_4d"
    assert config["tam_action_levels"] == 40
    assert config["tam_throttle_min"] == 0.4
    assert config["tam_throttle_max"] == 0.9
    assert config["scripted_evasion_red"] is False
    assert config["scripted_evasion_blue"] is False
    assert config["sim_freq"] == 60
    assert config["agent_interaction_steps"] == 12
    assert config["max_steps"] == 1000
    assert config["hetero_reward_mode"] == "happo_ref_v0"
    assert config["observation_mode"] == "mav_shared_geo"
    assert config["red_agent_types"] == ["mav"] + ["attack_uav"] * (red_count - 1)
    assert config["blue_agent_types"] == ["attack_uav"] * blue_count

    env = make_env(str(path))
    assert env.max_num_red == red_count
    assert env.max_num_blue == blue_count
    assert env.agent_models["red_0"] == "f22"
    assert env._num_missiles_for("red_0") == 0
    for agent_id in env.agent_ids:
        assert env.action_space[agent_id].shape == (4,)
        if agent_id != "red_0":
            assert env.agent_models[agent_id] == "f16"
            assert env._num_missiles_for(agent_id) > 0
