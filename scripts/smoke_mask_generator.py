from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brma.mask_generator import MaskGeneratorConfig, MaskVectorGenerator


def main():
    gen = MaskVectorGenerator(MaskGeneratorConfig(random_mask_prob=0.0))
    entity_mask = np.array([0, 0, 0, 0], dtype=np.int64)
    keep = gen.generate_random_keep_mask(entity_mask, 1, 1, 2)
    assert keep.shape == (4,)
    assert keep[0]

    invalid_entity_mask = np.array([0, 0, 1, 0], dtype=np.int64)
    keep_invalid = gen.generate_random_keep_mask(invalid_entity_mask, 1, 1, 2)
    assert not keep_invalid[2]

    forced_gen = MaskVectorGenerator(
        MaskGeneratorConfig(random_mask_prob=1.0, seed=7))
    forced_keep = forced_gen.generate_random_keep_mask(entity_mask, 1, 1, 2)
    assert forced_keep[0]
    assert not forced_keep[1]
    assert int(forced_keep[2:].sum()) == 1

    no_enemy_gen = MaskVectorGenerator(
        MaskGeneratorConfig(keep_enemies=False, random_mask_prob=0.0))
    no_enemy_keep = no_enemy_gen.generate_random_keep_mask(entity_mask, 1, 1, 2)
    assert not np.any(no_enemy_keep[2:])

    padding_mask = gen.convert_keep_mask_to_attention_key_padding_mask(
        np.array([True, False], dtype=bool))
    assert padding_mask.tolist() == [False, True]

    seeded_a = MaskVectorGenerator(
        MaskGeneratorConfig(random_mask_prob=0.5, seed=123))
    seeded_b = MaskVectorGenerator(
        MaskGeneratorConfig(random_mask_prob=0.5, seed=123))
    keep_a = seeded_a.generate_random_keep_mask(entity_mask, 1, 1, 2)
    keep_b = seeded_b.generate_random_keep_mask(entity_mask, 1, 1, 2)
    assert np.array_equal(keep_a, keep_b)

    out = gen.generate(entity_mask, 1, 1, 2)
    assert set(out.keys()) == {"keep_mask", "key_padding_mask", "meta"}
    assert np.array_equal(out["key_padding_mask"], ~out["keep_mask"])

    print("mask generator smoke test passed")


if __name__ == "__main__":
    main()
