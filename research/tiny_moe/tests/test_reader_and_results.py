from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from model_inspection import inspect_checkpoint
from streamed_runner import LayerReader
from test_manifest_and_packer import _fixture
from trunk_packer import pack_trunk


def test_reader_uses_one_exact_read_and_views_current_buffer(tmp_path: Path) -> None:
    checkpoint = tmp_path / "fixture.safetensors"
    trunk = tmp_path / "trunk.bin"
    index_path = tmp_path / "trunk.json"
    _fixture(checkpoint)
    manifest = inspect_checkpoint(checkpoint)
    index = pack_trunk(checkpoint, manifest, trunk, index_path)
    reader = LayerReader(trunk, index, max_buffers=2)
    try:
        layer = reader.read_layer(1, 0, phase="decode")
        views = reader.tensors_from_buffer(layer, 0)
        assert views["layers.1.attn.w_in.weight"].shape == (4, 4)
        assert torch.equal(
            views["layers.1.attn.w_in.weight"],
            torch.ones((4, 4), dtype=torch.float16) * 2,
        )
        accounting = reader.accounting.snapshot()
        assert accounting["read_calls"] == 1
        assert accounting["payload_bytes"] == layer["payload_bytes"]
        assert accounting["requested_bytes"] == layer["payload_bytes"]
        assert accounting["read_events"][0]["layer_id"] == 1
    finally:
        reader.close()


def test_final_real_matrix_has_all_modes_and_repeated_exact_results() -> None:
    path = Path(__file__).parents[1] / "results" / "matrix_final" / "GATE3_MATRIX.json"
    if not path.exists():
        pytest.skip("real Gate 3 matrix has not been generated")
    result = json.loads(path.read_text(encoding="utf-8"))
    assert set(result["modes"]) == {
        "fully_resident",
        "fifty_percent",
        "two_layers",
        "one_layer",
        "zero_layers",
    }
    assert len(result["cases"]) == 15
    for case in result["cases"]:
        assert case["comparison"]["all_identical"]
        assert case["repeat_identical_to_first"]
        assert case["read_gate"]["each_nonresident_layer_once_per_forward"]
        assert case["memory_audit"]["python_mmap_objects"] == 0
        assert not case["memory_audit"]["checkpoint_file_mappings"]
