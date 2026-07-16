import numpy as np

from aircombat_env_v1.scenario import make_scenario


def _values(pair):
    return tuple(tuple(vars(state).values()) for state in pair)


def test_randomized_scenario_is_reproducible_with_seed():
    first = make_scenario(
        "randomized_tail_chase", np.random.default_rng(12))
    second = make_scenario(
        "randomized_tail_chase", np.random.default_rng(12))
    assert _values(first) == _values(second)


def test_randomized_scenario_changes_with_seed():
    first = make_scenario(
        "randomized_tail_chase", np.random.default_rng(12))
    second = make_scenario(
        "randomized_tail_chase", np.random.default_rng(13))
    assert _values(first) != _values(second)
