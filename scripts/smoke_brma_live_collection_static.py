"""Static smoke test for BRMA live collection scaffold. No env, no JSBSim."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from attention_models import AttentionActor
from brma.collection import collect_brma_dry_run_step
from brma.mask_generator import BRMAMaskGenerator, BRMAMaskGeneratorConfig
from brma.rollout_schema import BRMARolloutSchemaConfig
from train_attention_mappo import AttentionRolloutBuffer


def main() -> None:
    # ---- 1. parse_args_attention default ----
    old_argv = sys.argv
    try:
        sys.argv = ["train_attention_mappo.py"]
        from train_attention_mappo import parse_args_attention
        args = parse_args_attention()
        assert args.brma_mode == "off"
    finally:
        sys.argv = old_argv

    # ---- 2. parse_args_attention dry-run ----
    try:
        sys.argv = ["train_attention_mappo.py", "--brma-mode", "dry-run",
                    "--brma-temperature", "0.1"]
        from train_attention_mappo import parse_args_attention
        args = parse_args_attention()
        assert args.brma_mode == "dry-run"
    finally:
        sys.argv = old_argv

    # ---- 3. invalid brma mode ----
    try:
        sys.argv = ["train_attention_mappo.py", "--brma-mode", "train"]
        from train_attention_mappo import parse_args_attention
        args = parse_args_attention()
        assert False, "should have raised SystemExit"
    except SystemExit:
        pass
    finally:
        sys.argv = old_argv

    # ---- 4. preset list ----
    from configs.experiment_presets import list_presets
    names = list_presets()
    assert "attention_1v1_strict_eq33_attncritic_brma_dryrun_smoke" in names

    # ---- 5. default buffer has no brma_storage ----
    buf_default = AttentionRolloutBuffer(
        num_steps=3, num_envs=2, num_red=1, action_dim=3, rnn_hidden_size=128)
    assert buf_default.brma_storage is None

    # ---- 6. buffer with BRMA storage enabled ----
    sc_cfg = BRMARolloutSchemaConfig(
        num_steps=4, num_envs=2, num_agents=2,
        n_entities=5, entity_dim=10, enabled=True)
    buf_brma = AttentionRolloutBuffer(
        num_steps=4, num_envs=2, num_red=2, action_dim=3, rnn_hidden_size=128,
        brma_storage_config=sc_cfg)
    assert buf_brma.brma_storage is not None
    assert buf_brma.brma_storage.has_storage
    assert buf_brma.brma_storage.summary()["enabled"] == True

    # ---- 7. integrated dry-run collector shape smoke ----
    actor = AttentionActor(entity_dim=10, action_dim=3, encoder_mode="paper_eq33")
    mg_cfg = BRMAMaskGeneratorConfig(entity_feature_dim=10)
    mask_gen = BRMAMaskGenerator(mg_cfg)
    N, D = 5, 10
    collect_brma_dry_run_step(
        actor=actor, mask_generator=mask_gen,
        storage=buf_brma.brma_storage,
        step=0, env_idx=0, agent_idx=0,
        entities=np.random.randn(N, D).astype(np.float32),
        entity_mask=np.zeros(N, dtype=np.int64),
        rnn_hidden=np.zeros(128, dtype=np.float32),
        action=np.zeros(3, dtype=np.float32),
        n_ego=1, n_allies=1, n_enemies=3,
    )
    sm = buf_brma.brma_storage.summary()
    assert sm["valid_count"] == 1

    print("brma live collection static smoke test passed")


if __name__ == "__main__":
    main()
