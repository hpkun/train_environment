from types import SimpleNamespace

from scripts.eval_tam_happo_direct import _eval_deterministic
from scripts.train_tam_happo_direct import _train_rollout_deterministic


def test_eval_defaults_to_deterministic_argmax():
    assert _eval_deterministic(SimpleNamespace()) is True
    assert _eval_deterministic(SimpleNamespace(stochastic_eval=False)) is True


def test_stochastic_eval_is_explicit_opt_in():
    assert _eval_deterministic(SimpleNamespace(stochastic_eval=True)) is False


def test_train_rollout_remains_stochastic():
    assert _train_rollout_deterministic() is False
