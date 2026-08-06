from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from model_inspection import inspect_checkpoint
from safetensors.torch import save_file
from trunk_packer import pack_trunk, read_index, validate_index


def _fixture(path: Path) -> None:
    tensors = {
        "embed_tokens.weight": torch.arange(16, dtype=torch.float16).reshape(4, 4),
        "lm_head.weight": torch.arange(16, dtype=torch.float16).reshape(4, 4),
        "norm.weight": torch.ones(4, dtype=torch.float16),
    }
    for layer in range(2):
        prefix = f"layers.{layer}."
        tensors.update(
            {
                prefix
                + "attn.w_in.weight": torch.ones((4, 4), dtype=torch.float16)
                * (layer + 1),
                prefix + "moe.router.weight": torch.ones((2, 4), dtype=torch.float16),
                prefix
                + "moe.shared_w13.weight": torch.ones((8, 4), dtype=torch.float16),
                prefix
                + "moe.shared_w2.weight": torch.ones((4, 4), dtype=torch.float16),
                prefix + "moe.w13": torch.ones((2, 4, 4), dtype=torch.float16),
                prefix + "moe.w2": torch.ones((2, 4, 4), dtype=torch.float16),
            }
        )
    save_file(tensors, str(path))


def test_inspection_classifies_routed_experts(tmp_path: Path) -> None:
    checkpoint = tmp_path / "fixture.safetensors"
    _fixture(checkpoint)
    manifest = inspect_checkpoint(checkpoint)
    assert manifest["tensor_count"] == 15
    assert manifest["architecture"]["num_layers"] == 2
    assert manifest["categories"]["routed_expert"] == 2 * (2 * 4 * 4 + 2 * 4 * 4) * 2
    assert all(
        tensor["category"] != "routed_expert"
        for tensor in manifest["tensors"]
        if tensor["name"] == "embed_tokens.weight"
    )


def test_packer_copies_exact_nonexpert_bytes_and_rejects_expert_in_trunk(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "fixture.safetensors"
    trunk = tmp_path / "trunk.bin"
    index_path = tmp_path / "trunk.json"
    _fixture(checkpoint)
    manifest = inspect_checkpoint(checkpoint)
    index = pack_trunk(checkpoint, manifest, trunk, index_path)
    assert (
        sum(layer["payload_bytes"] for layer in index["layers"])
        == manifest["categories"]["trunk"]
    )
    assert trunk.stat().st_size == index["file_bytes"]
    assert read_index(trunk) == index
    raw = checkpoint.read_bytes()
    packed = trunk.read_bytes()
    for layer in index["layers"]:
        for tensor in layer["tensors"]:
            source_start = tensor["source_absolute_offset"]
            packed_start = layer["payload_offset"] + tensor["local_offset"]
            assert (
                packed[packed_start : packed_start + tensor["byte_count"]]
                == raw[source_start : source_start + tensor["byte_count"]]
            )
    bad = json.loads(json.dumps(index))
    bad["layers"][0]["tensors"][0]["name"] = "layers.0.moe.w13"
    with pytest.raises(ValueError, match="routed expert"):
        validate_index(bad, trunk.stat().st_size)
