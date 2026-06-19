"""Thin compatibility wrapper keeping the native Gymnasium-like return format."""


class GymLikeWrapper:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, actions):
        return self.env.step(actions)
