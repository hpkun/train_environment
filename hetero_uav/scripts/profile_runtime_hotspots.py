"""Lightweight runtime profile for HAPPO/oracle-pretrain components."""
from __future__ import annotations

import argparse
import json
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _record(name: str, start: float, **extra) -> dict:
    return {"name": name, "wall_time_sec": time.perf_counter() - start, **extra}


def _write_outputs(records: list[dict], output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    slow = sorted(
        [r for r in records if r.get("status") == "ok"],
        key=lambda r: r.get("wall_time_sec", 0.0),
        reverse=True,
    )
    payload = {
        "records": records,
        "slowest_components": [r["name"] for r in slow[:3]],
        "recommended_next_speedup": (
            "reuse generated artifacts and use --fast checkpoint screening before formal evaluation"
        ),
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Runtime Hotspot Profile", ""]
    for record in records:
        lines.append(f"## {record['name']}")
        lines.append(f"- status: {record.get('status')}")
        lines.append(f"- wall_time_sec: {record.get('wall_time_sec', 0.0):.4f}")
        if record.get("steps_per_sec") is not None:
            lines.append(f"- steps_per_sec: {record['steps_per_sec']:.2f}")
        if record.get("error"):
            lines.append(f"- error: {record['error']}")
        lines.append("")
    lines.append("## Recommended Next Speedup")
    lines.append(payload["recommended_next_speedup"])
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _profile_adapter(calls: int) -> dict:
    start = time.perf_counter()
    try:
        adapter_path = ROOT / "uav_env" / "JSBSim" / "adapters" / "hetero_obs_adapter_v2.py"
        spec = importlib.util.spec_from_file_location("hetero_obs_adapter_v2", adapter_path)
        module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise RuntimeError(f"cannot load adapter: {adapter_path}")
        spec.loader.exec_module(module)
        HeteroObsAdapterV2 = module.HeteroObsAdapterV2

        adapter = HeteroObsAdapterV2()
        def one_obs(role_index: int) -> dict:
            role = np.zeros(4, dtype=np.float32)
            role[role_index] = 1.0
            return {
                "ego_geo_state": np.zeros(7, dtype=np.float32),
                "ego_role": role,
                "missile_warning": np.zeros(1, dtype=np.float32),
                "ally_geo_states": np.zeros((2, 5), dtype=np.float32),
                "ally_roles": np.zeros((2, 4), dtype=np.float32),
                "ally_alive_mask": np.ones(2, dtype=np.float32),
                "enemy_geo_states": np.zeros((2, 5), dtype=np.float32),
                "enemy_alive_mask": np.ones(2, dtype=np.float32),
                "enemy_observed_mask": np.ones(2, dtype=np.float32),
                "enemy_track_source": np.zeros((2, 2), dtype=np.float32),
            }

        obs = {f"red_{i}": one_obs(0 if i == 0 else 1) for i in range(3)}
        info = {f"red_{i}": {"alive": True} for i in range(3)}
        red_ids = ["red_0", "red_1", "red_2"]
        blue_ids = ["blue_0", "blue_1"]
        for _ in range(calls):
            adapter.adapt_all(obs, info=info, red_ids=red_ids, blue_ids=blue_ids)
        return _record("adapter_adapt_all", start, status="ok", calls=calls,
                       steps_per_sec=calls / max(time.perf_counter() - start, 1e-9))
    except Exception as exc:
        return _record("adapter_adapt_all", start, status="skipped", error=repr(exc))


def _profile_policy(calls: int) -> dict:
    start = time.perf_counter()
    try:
        import torch
        from algorithms.happo import HAPPOReferencePolicy

        policy = HAPPOReferencePolicy(96, 480).to("cpu")
        obs = torch.randn(3, 96)
        roles = torch.tensor([0, 1, 1])
        state = torch.randn(480)
        for _ in range(calls):
            policy.act(obs, roles=roles, critic_state=state, deterministic=True)
        return _record("policy_act", start, status="ok", calls=calls,
                       steps_per_sec=calls / max(time.perf_counter() - start, 1e-9))
    except Exception as exc:
        return _record("policy_act", start, status="skipped", error=repr(exc))


def _profile_fake_update() -> dict:
    start = time.perf_counter()
    try:
        import torch
        from algorithms.happo import HAPPOReferencePolicy, HAPPOReferenceTrainer, HAPPORolloutBuffer

        policy = HAPPOReferencePolicy(96, 480).to("cpu")
        trainer = HAPPOReferenceTrainer(policy, ppo_epochs=1)
        buffer = HAPPORolloutBuffer(8, 3, 96, 480, 3, role_ids=[0, 1, 1])
        for _ in range(8):
            actor_obs = np.random.randn(3, 96).astype(np.float32)
            state = np.random.randn(480).astype(np.float32)
            actions = np.clip(np.random.randn(3, 3), -1, 1).astype(np.float32)
            with torch.no_grad():
                out = policy.act(actor_obs, roles=[0, 1, 1], critic_state=state)
            log_probs = out["log_prob"].detach().cpu().numpy().astype(np.float32)
            value = float(out["value"].detach().cpu().numpy()[0])
            buffer.store(actor_obs, state, actions, log_probs,
                         np.ones(3, dtype=np.float32), np.zeros(3, dtype=np.float32),
                         value, np.ones(3, dtype=np.float32))
        trainer.update(buffer)
        return _record("fake_ppo_update", start, status="ok", updates=1)
    except Exception as exc:
        return _record("fake_ppo_update", start, status="skipped", error=repr(exc))


def _profile_env_steps(steps: int) -> dict:
    start = time.perf_counter()
    try:
        from scripts.red_attack_audit_utils import DEFAULT_CONFIG, make_env, team_done

        env = make_env(DEFAULT_CONFIG, hetero_reward_mode="happo_ref_v0", max_steps=steps)
        try:
            obs, _info = env.reset(seed=123)
            done = False
            executed = 0
            while executed < steps and not done:
                actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.red_ids + env.blue_ids}
                obs, _rewards, terminated, truncated, _info = env.step(actions)
                done = team_done(terminated, truncated)
                executed += 1
        finally:
            env.close()
        return _record("env_step", start, status="ok", steps=executed,
                       steps_per_sec=executed / max(time.perf_counter() - start, 1e-9))
    except Exception as exc:
        return _record("env_step", start, status="skipped", error=repr(exc))


def _run_subprocess(name: str, cmd: list[str], timeout: int = 120) -> dict:
    start = time.perf_counter()
    try:
        result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True,
                                encoding="utf-8", errors="replace", timeout=timeout)
        status = "ok" if result.returncode == 0 else "skipped"
        error = None if result.returncode == 0 else (result.stderr or result.stdout)[-1000:]
        return _record(name, start, status=status, returncode=result.returncode, error=error)
    except Exception as exc:
        return _record(name, start, status="skipped", error=repr(exc))


def _prepare_pretrain_inputs(scratch: Path) -> tuple[Path, Path] | tuple[None, None]:
    try:
        from algorithms.happo import HAPPOReferencePolicy

        dataset = scratch / "fake_dataset.npz"
        checkpoint = scratch / "fake_init.pt"
        rng = np.random.default_rng(0)
        np.savez_compressed(
            dataset,
            actor_obs=rng.normal(size=(256, 96)).astype(np.float32),
            oracle_action=np.clip(rng.normal(size=(256, 3)), -1.0, 1.0).astype(np.float32),
        )
        policy = HAPPOReferencePolicy(96, 480)
        policy.save(checkpoint)
        return dataset, checkpoint
    except Exception:
        return None, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile runtime hotspots without long training")
    parser.add_argument("--output-json", default="outputs/runtime_profile/runtime_profile.json")
    parser.add_argument("--output-md", default="outputs/runtime_profile/runtime_profile.md")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--calls", type=int, default=500)
    args = parser.parse_args()

    output_json = ROOT / args.output_json
    output_md = ROOT / args.output_md
    scratch = output_json.parent / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    pretrain_dataset, pretrain_init = _prepare_pretrain_inputs(scratch)

    records = [
        _profile_env_steps(args.steps),
        _profile_adapter(args.calls),
        _profile_policy(args.calls),
        _profile_fake_update(),
        _run_subprocess("dataset_collect_1_episode", [
            "python", "scripts/collect_direct_chase_oracle_dataset.py",
            "--episodes", "1",
            "--max-steps", "50",
            "--max-samples", "256",
            "--output", str((scratch / "dataset.npz").relative_to(ROOT)),
            "--summary-json", str((scratch / "dataset_summary.json").relative_to(ROOT)),
        ]),
    ]
    if pretrain_dataset is None or pretrain_init is None:
        records.append({"name": "pretrain_1_epoch_small_dataset", "wall_time_sec": 0.0,
                        "status": "skipped", "error": "could not create fake pretrain inputs"})
    else:
        records.append(_run_subprocess("pretrain_1_epoch_small_dataset", [
            "python", "scripts/pretrain_uav_actor_from_oracle.py",
            "--dataset", str(pretrain_dataset.relative_to(ROOT)),
            "--init-checkpoint", str(pretrain_init.relative_to(ROOT)),
            "--epochs", "1",
            "--max-train-samples", "256",
            "--output-checkpoint", str((scratch / "pretrain.pt").relative_to(ROOT)),
            "--output-meta", str((scratch / "pretrain_meta.json").relative_to(ROOT)),
            "--device", "cpu",
        ]))
    _write_outputs(records, output_json, output_md)
    print(f"output_json: {output_json}")
    print(f"output_md: {output_md}")
    for record in records:
        print(f"{record['name']}: {record.get('status')} {record.get('wall_time_sec', 0.0):.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
