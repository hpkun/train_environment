"""Spawn-based vector environment for independent JSBSim workers."""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback

import numpy as np


def _worker(remote, env_kwargs, seed):
    for name, value in (
        ("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"), ("KMP_DUPLICATE_LIB_OK", "TRUE"),
    ):
        os.environ[name] = value
    try:
        from .env import AirCombat1v1Env
        env = AirCombat1v1Env(**env_kwargs)
        initial_seed = int(seed)
        remote.send(("ok", None))
        while True:
            command, payload = remote.recv()
            try:
                if command == "reset":
                    reset_seed = initial_seed if payload is None else payload
                    observation, info = env.reset(seed=reset_seed)
                    initial_seed = None
                    remote.send(("ok", (observation, info)))
                elif command == "step":
                    observation, reward, terminated, truncated, info = env.step(
                        payload)
                    if terminated or truncated:
                        terminal_observation = observation.copy()
                        terminal_info = dict(info)
                        observation, reset_info = env.reset()
                        info = dict(info)
                        info["terminal_observation"] = terminal_observation
                        info["terminal_info"] = terminal_info
                        info["reset_info"] = reset_info
                    remote.send(("ok", (
                        observation, reward, terminated, truncated, info)))
                elif command == "set_curriculum":
                    env.set_curriculum_stage(payload)
                    remote.send(("ok", None))
                elif command == "close":
                    env.close()
                    remote.send(("ok", None))
                    break
                else:
                    raise ValueError(f"unknown worker command: {command}")
            except Exception:
                remote.send(("error", traceback.format_exc()))
    except Exception:
        remote.send(("error", traceback.format_exc()))
    finally:
        remote.close()


class SubprocVecEnv:
    def __init__(self, num_envs=8, base_seed=1, env_kwargs=None,
                 timeout=60.0):
        self.num_envs = int(num_envs)
        self.timeout = float(timeout)
        self.closed = False
        context = mp.get_context("spawn")
        self.remotes = []
        self.processes = []
        env_kwargs = dict(env_kwargs or {})
        for rank in range(self.num_envs):
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker,
                args=(child, env_kwargs, int(base_seed) + rank),
                daemon=True)
            process.start()
            child.close()
            self.remotes.append(parent)
            self.processes.append(process)
        for remote in self.remotes:
            self._receive(remote, "worker initialization")

    def _receive(self, remote, operation):
        if not remote.poll(self.timeout):
            raise TimeoutError(f"{operation} timed out after {self.timeout}s")
        try:
            status, payload = remote.recv()
        except EOFError as error:
            raise RuntimeError(f"{operation} worker exited") from error
        if status == "error":
            raise RuntimeError(f"{operation} failed:\n{payload}")
        return payload

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [
            self._receive(remote, "reset") for remote in self.remotes]
        observations, infos = zip(*results)
        return np.stack(observations), list(infos)

    def step(self, actions):
        if len(actions) != self.num_envs:
            raise ValueError("one action is required for each environment")
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [
            self._receive(remote, "step") for remote in self.remotes]
        observations, rewards, terminated, truncated, infos = zip(*results)
        return (
            np.stack(observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminated, dtype=bool),
            np.asarray(truncated, dtype=bool),
            list(infos),
        )

    def set_curriculum_stage(self, stage):
        for remote in self.remotes:
            remote.send(("set_curriculum", int(stage)))
        for remote in self.remotes:
            self._receive(remote, "set curriculum")

    def close(self):
        if self.closed:
            return
        for remote, process in zip(self.remotes, self.processes):
            if process.is_alive():
                try:
                    remote.send(("close", None))
                except (BrokenPipeError, EOFError):
                    pass
        for remote, process in zip(self.remotes, self.processes):
            if process.is_alive():
                try:
                    self._receive(remote, "close")
                except (RuntimeError, TimeoutError):
                    process.terminate()
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join()
            remote.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback_value):
        self.close()
