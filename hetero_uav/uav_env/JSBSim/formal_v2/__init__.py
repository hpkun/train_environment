"""Paper-aligned formal heterogeneous 3v2 Pure HAPPO V2."""
from ..formal_v1.env import Hetero3v2PureHAPPOEnv


def make_formal_env(**config) -> Hetero3v2PureHAPPOEnv:
    return Hetero3v2PureHAPPOEnv(**config)


__all__ = ["Hetero3v2PureHAPPOEnv", "make_formal_env"]
