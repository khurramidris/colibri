"""Pack Tiny-MoE non-routed layer tensors into an aligned random-access trunk."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any, BinaryIO

from model_inspection import ALIGNMENT, sha256_file

MAGIC = b"COLITRK1"
VERSION = 1
HEADER = struct.Struct("<8sIIIIQQQQ")
HEADER_BYTES = 64


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) & -alignment


def _zero_pad(handle: BinaryIO, count: int) -> None:
    if count:
        handle.write(b"\0" * count)


def _copy_range(source: Path, output: BinaryIO, start: int, length: int) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(
                    f"short source read at {start} ({remaining} bytes remain)"
                )
            output.write(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def validate_index(index: dict[str, Any], file_size: int | None = None) -> None:
    if index.get("format") != "colibri-tiny-moe-trunk-v1":
        raise ValueError("unsupported trunk index format")
    if index.get("alignment_bytes") != ALIGNMENT:
        raise ValueError("trunk alignment is not 4096 bytes")
    layers = index.get("layers")
    if not isinstance(layers, list) or [layer["layer_id"] for layer in layers] != list(
        range(len(layers))
    ):
        raise ValueError("layers must be a contiguous numeric sequence")
    ranges: list[tuple[int, int, str]] = []
    for layer in layers:
        start = layer.get("payload_offset")
        length = layer.get("payload_bytes")
        io_length = layer.get("io_bytes")
        if not all(isinstance(value, int) for value in (start, length, io_length)):
            raise ValueError("layer offsets and lengths must be integers")
        if (
            start % ALIGNMENT
            or io_length % ALIGNMENT
            or io_length < length
            or length <= 0
        ):
            raise ValueError(f"malformed layer range: {layer}")
        if start < index["payload_offset"] or start + io_length > index["file_bytes"]:
            raise ValueError(f"layer range outside file: {layer}")
        tensor_cursor = 0
        names: set[str] = set()
        for tensor in layer.get("tensors", []):
            name = tensor.get("name")
            local = tensor.get("local_offset")
            tensor_length = tensor.get("byte_count")
            if not isinstance(name, str) or name in names:
                raise ValueError(
                    f"duplicate/malformed tensor in layer {layer['layer_id']}"
                )
            if ".moe.w13" in name or ".moe.w2" in name:
                raise ValueError(f"routed expert accidentally entered trunk: {name}")
            if (
                local != tensor_cursor
                or not isinstance(tensor_length, int)
                or tensor_length <= 0
            ):
                raise ValueError(f"non-contiguous tensor payload in {name}")
            if local + tensor_length > length:
                raise ValueError(f"tensor exceeds layer payload: {name}")
            names.add(name)
            tensor_cursor += tensor_length
        if tensor_cursor != length:
            raise ValueError(
                f"layer {layer['layer_id']} payload does not match tensors"
            )
        ranges.append((start, start + io_length, f"layer-{layer['layer_id']}"))
    ranges.sort()
    for range_index in range(1, len(ranges)):
        previous = ranges[range_index - 1]
        current = ranges[range_index]
        if previous[1] > current[0]:
            raise ValueError(f"overlapping layer ranges: {previous} and {current}")
    if file_size is not None and file_size != index["file_bytes"]:
        raise ValueError(f"file size {file_size} != declared {index['file_bytes']}")


def pack_trunk(
    checkpoint: str | os.PathLike[str],
    manifest: dict[str, Any],
    output: str | os.PathLike[str],
    index_path: str | os.PathLike[str],
) -> dict[str, Any]:
    source = Path(checkpoint)
    if not source.is_file():
        raise FileNotFoundError(source)
    if manifest.get("source_sha256") and manifest["source_sha256"] != sha256_file(
        source
    ):
        raise ValueError("checkpoint SHA-256 does not match manifest")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    tensors_by_layer: dict[int, list[dict[str, Any]]] = {}
    for tensor in manifest["tensors"]:
        if tensor["category"] == "routed_expert":
            continue
        if tensor["category"] != "trunk":
            continue
        tensors_by_layer.setdefault(tensor["layer_id"], []).append(tensor)

    layer_specs: list[dict[str, Any]] = []
    index_offset = ALIGNMENT
    # The index JSON is written after the layer data is planned. Its padded
    # length is deterministic because the JSON is sorted and compact.
    provisional = {
        "format": "colibri-tiny-moe-trunk-v1",
        "alignment_bytes": ALIGNMENT,
        "layers": [],
    }
    # Estimate the index using the final tensor metadata shape; reserve enough
    # aligned space, then verify the exact index fits before writing.
    provisional["layers"] = [
        {
            "layer_id": layer_id,
            "payload_offset": 0,
            "payload_bytes": sum(item["byte_count"] for item in items),
            "io_bytes": align_up(sum(item["byte_count"] for item in items)),
            "tensors": [
                {
                    "name": item["name"],
                    "dtype": item["dtype"],
                    "shape": item["shape"],
                    "local_offset": 0,
                    "byte_count": item["byte_count"],
                    "source_absolute_offset": item["absolute_offset"],
                }
                for item in sorted(items, key=lambda item: item["absolute_offset"])
            ],
        }
        for layer_id, items in sorted(tensors_by_layer.items())
    ]
    # Reserve a deterministic safety margin for the final per-tensor hashes.
    # The index is metadata, so a bounded margin is preferable to a second
    # pass over the checkpoint merely to discover its final JSON length.
    provisional_length = len(
        json.dumps(provisional, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    reserve_index_bytes = max(64 * 1024, align_up(provisional_length * 2))
    payload_cursor = align_up(index_offset + reserve_index_bytes)

    with output.open("wb") as handle:
        handle.write(b"\0" * HEADER_BYTES)
        _zero_pad(handle, index_offset - HEADER_BYTES)
        _zero_pad(handle, reserve_index_bytes)
        for layer_id, items in sorted(tensors_by_layer.items()):
            ordered = sorted(items, key=lambda item: item["absolute_offset"])
            payload_cursor = align_up(payload_cursor)
            current = handle.tell()
            if current > payload_cursor:
                raise ValueError("internal payload cursor moved backwards")
            _zero_pad(handle, payload_cursor - current)
            local_cursor = 0
            packed_tensors = []
            for item in ordered:
                if item["category"] != "trunk":
                    raise ValueError("non-trunk tensor selected")
                raw_digest = _copy_range(
                    source, handle, item["absolute_offset"], item["byte_count"]
                )
                packed_tensors.append(
                    {
                        "name": item["name"],
                        "dtype": item["dtype"],
                        "shape": list(item["shape"]),
                        "local_offset": local_cursor,
                        "byte_count": item["byte_count"],
                        "source_absolute_offset": item["absolute_offset"],
                        "source_sha256": raw_digest,
                    }
                )
                local_cursor += item["byte_count"]
            layer_bytes = local_cursor
            io_bytes = align_up(layer_bytes)
            _zero_pad(handle, io_bytes - layer_bytes)
            layer_specs.append(
                {
                    "layer_id": layer_id,
                    "payload_offset": payload_cursor,
                    "payload_bytes": layer_bytes,
                    "io_bytes": io_bytes,
                    "tensors": packed_tensors,
                }
            )
            payload_cursor += io_bytes

        file_bytes = align_up(handle.tell())
        _zero_pad(handle, file_bytes - handle.tell())
        index = {
            "format": "colibri-tiny-moe-trunk-v1",
            "version": VERSION,
            "alignment_bytes": ALIGNMENT,
            "source": str(source),
            "source_sha256": manifest.get("source_sha256"),
            "index_offset": index_offset,
            "payload_offset": align_up(index_offset + reserve_index_bytes),
            "layers": layer_specs,
            "file_bytes": file_bytes,
        }
        index_bytes = json.dumps(index, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        if len(index_bytes) > reserve_index_bytes:
            raise ValueError("reserved index space was insufficient")
        handle.seek(index_offset)
        handle.write(index_bytes)
        _zero_pad(handle, reserve_index_bytes - len(index_bytes))
        header = HEADER.pack(
            MAGIC,
            VERSION,
            ALIGNMENT,
            len(layer_specs),
            sum(len(layer["tensors"]) for layer in layer_specs),
            index_offset,
            len(index_bytes),
            index["payload_offset"],
            file_bytes,
        )
        handle.seek(0)
        handle.write(header)
        _zero_pad(handle, HEADER_BYTES - HEADER.size)
        handle.flush()
        os.fsync(handle.fileno())

    validate_index(index, output.stat().st_size)
    index_file = Path(index_path)
    index_file.parent.mkdir(parents=True, exist_ok=True)
    index_file.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return index


def read_index(path: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as handle:
        raw = handle.read(HEADER_BYTES)
        if len(raw) != HEADER_BYTES:
            raise ValueError("short trunk header")
        (
            magic,
            version,
            alignment,
            layers,
            records,
            index_offset,
            index_length,
            payload_offset,
            file_bytes,
        ) = HEADER.unpack(raw[: HEADER.size])
        if magic != MAGIC or version != VERSION or alignment != ALIGNMENT:
            raise ValueError("invalid trunk header")
        handle.seek(index_offset)
        index = json.loads(handle.read(index_length).decode("utf-8"))
    if (
        index.get("payload_offset") != payload_offset
        or index.get("file_bytes") != file_bytes
    ):
        raise ValueError("binary and JSON trunk headers disagree")
    if (
        len(index.get("layers", [])) != layers
        or sum(len(layer.get("tensors", [])) for layer in index["layers"]) != records
    ):
        raise ValueError("binary and JSON trunk counts disagree")
    validate_index(index, path.stat().st_size)
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--index", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    index = pack_trunk(args.checkpoint, manifest, args.output, args.index)
    print(
        json.dumps(
            {
                "output": args.output,
                "index": args.index,
                "file_bytes": index["file_bytes"],
                "layers": len(index["layers"]),
                "payload_bytes": sum(
                    layer["payload_bytes"] for layer in index["layers"]
                ),
                "io_bytes": sum(layer["io_bytes"] for layer in index["layers"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
