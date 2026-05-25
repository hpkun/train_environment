"""Pure smoke test for experiment presets.  No env, no JSBSim."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.experiment_presets import EXPERIMENT_PRESETS, get_preset, list_presets


def main() -> None:
    names = list_presets()
    assert len(names) > 0, "presets should not be empty"
    assert "vanilla_1v1_smoke" in names

    p = get_preset("vanilla_1v1_smoke")
    assert p["num_red"] == 1
    assert "log_file" in p
    assert "checkpoint_dir" in p

    # get_preset returns a copy — mutation should not pollute original
    p["num_red"] = 999
    p2 = get_preset("vanilla_1v1_smoke")
    assert p2["num_red"] == 1, "get_preset should return a copy"

    # attention preset has obs_adapter
    pa = get_preset("attention_1v1_smoke")
    assert pa["obs_adapter"] == "current"
    ps = get_preset("attention_1v1_strict_smoke")
    assert ps["obs_adapter"] == "strict"

    # unknown preset raises KeyError
    try:
        get_preset("no_such_preset")
        assert False, "should have raised KeyError"
    except KeyError:
        pass

    print("Available presets:")
    for name in names:
        p = EXPERIMENT_PRESETS[name]
        obs = p.get("obs_adapter", "")
        via = f"  via --obs-adapter {obs}" if obs else ""
        print(f"  {name} ({p['num_red']}v{p['num_blue']}, "
              f"{p['num_envs']} envs, {p['total_env_steps']} steps){via}")

    print("experiment presets smoke test passed")


if __name__ == "__main__":
    main()
