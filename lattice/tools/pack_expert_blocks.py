#!/usr/bin/env python3
"""Repack Kimi-style native MXFP4 experts into independently readable blocks.

Each output layer file contains fixed-size 4096-byte-aligned records. One record
holds the W1/W3 output rows and matching W2 input columns for one intermediate-
channel block. A runtime can therefore skip the record entirely: no NVMe read,
RAM copy, PCIe transfer, or matmul for that block.

This tool preserves packed e2m1 nibbles and ue8m0 scale bytes exactly. It does
not dequantize or requantize weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

MAGIC = b"LTBLK1\0\0"
VERSION = 1
HEADER_BYTES = 4096
HEADER_LIMIT = 256 * 1024 * 1024
LAYER_RE = re.compile(r"(?:^|\.)model\.layers\.(\d+)\.block_sparse_moe\.experts\.\d+\.w1\.weight_packed$")


@dataclass(frozen=True)
class TensorRef:
    path: Path
    offset: int
    nbytes: int
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Layout:
    latent: int
    intermediate: int
    group_size: int
    block_channels: int
    experts: int
    blocks: int
    w1p: int
    w1s: int
    w2p: int
    w2s: int
    record_bytes: int
    total_bytes: int


def fnv1a64(data: bytes) -> int:
    value = 1469598103934665603
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return value


def align(value: int, boundary: int) -> int:
    return (value + boundary - 1) & ~(boundary - 1)


def make_layout(latent: int, intermediate: int, group_size: int, block_channels: int, experts: int) -> Layout:
    if (
        latent <= 0
        or intermediate <= 0
        or group_size <= 0
        or block_channels <= 0
        or experts <= 0
        or group_size % 2
        or latent % group_size
        or block_channels % group_size
        or intermediate % block_channels
    ):
        raise ValueError(
            "dimensions must be positive; latent and block width must be group-aligned; "
            "intermediate must divide into fixed blocks"
        )
    blocks = intermediate // block_channels
    w1p = block_channels * (latent // 2)
    w1s = block_channels * (latent // group_size)
    w2p = latent * (block_channels // 2)
    w2s = latent * (block_channels // group_size)
    payload = 2 * (w1p + w1s) + w2p + w2s
    record_bytes = align(payload, HEADER_BYTES)
    total_bytes = HEADER_BYTES + experts * blocks * record_bytes
    if total_bytes >= 1 << 64:
        raise ValueError("block file exceeds uint64 address space")
    return Layout(
        latent=latent,
        intermediate=intermediate,
        group_size=group_size,
        block_channels=block_channels,
        experts=experts,
        blocks=blocks,
        w1p=w1p,
        w1s=w1s,
        w2p=w2p,
        w2s=w2s,
        record_bytes=record_bytes,
        total_bytes=total_bytes,
    )


def encode_header(layout: Layout, layer: int) -> bytes:
    header = bytearray(HEADER_BYTES)
    header[:8] = MAGIC
    struct.pack_into("<IIIIIIIII", header, 8,
                     VERSION, HEADER_BYTES, layer,
                     layout.latent, layout.intermediate, layout.group_size,
                     layout.block_channels, layout.experts, layout.blocks)
    struct.pack_into("<QQ", header, 48, layout.record_bytes, layout.total_bytes)
    struct.pack_into("<Q", header, 64, fnv1a64(header[:64]))
    return bytes(header)


def decode_header(data: bytes) -> tuple[Layout, int]:
    if len(data) != HEADER_BYTES or data[:8] != MAGIC:
        raise ValueError("block header magic mismatch")
    version, header_bytes, layer, latent, intermediate, group_size, block_channels, experts, blocks = struct.unpack_from(
        "<IIIIIIIII", data, 8
    )
    if version != VERSION or header_bytes != HEADER_BYTES:
        raise ValueError("block header version mismatch")
    if struct.unpack_from("<Q", data, 64)[0] != fnv1a64(data[:64]):
        raise ValueError("block header checksum mismatch")
    layout = make_layout(latent, intermediate, group_size, block_channels, experts)
    record_bytes, total_bytes = struct.unpack_from("<QQ", data, 48)
    if blocks != layout.blocks or record_bytes != layout.record_bytes or total_bytes != layout.total_bytes:
        raise ValueError("block header derived geometry mismatch")
    return layout, layer


def nested_find(root: Any, keys: Iterable[str]) -> Any | None:
    wanted = set(keys)
    stack = [root]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            for key, child in value.items():
                if key in wanted:
                    return child
                stack.append(child)
        elif isinstance(value, list):
            stack.extend(value)
    return None


def read_index(model_dir: Path) -> dict[str, TensorRef]:
    refs: dict[str, TensorRef] = {}
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise ValueError(f"no *.safetensors files found in {model_dir}")
    for path in files:
        file_size = path.stat().st_size
        with path.open("rb") as fp:
            raw = fp.read(8)
            if len(raw) != 8:
                raise ValueError(f"{path}: truncated header length")
            (header_size,) = struct.unpack("<Q", raw)
            if header_size <= 0 or header_size > HEADER_LIMIT or 8 + header_size > file_size:
                raise ValueError(f"{path}: invalid header length {header_size}")
            payload = fp.read(header_size)
            if len(payload) != header_size:
                raise ValueError(f"{path}: short header read")
        try:
            header = json.loads(payload.decode("utf-8").rstrip(" \t\r\n\0"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}: invalid header JSON: {exc}") from exc
        if not isinstance(header, dict):
            raise ValueError(f"{path}: header is not an object")
        data_base = 8 + header_size
        data_size = file_size - data_base
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(name, str) or not isinstance(meta, dict) or name in refs:
                raise ValueError(f"{path}: malformed or duplicate tensor {name!r}")
            offsets = meta.get("data_offsets")
            shape = meta.get("shape")
            dtype = meta.get("dtype")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(v, int) for v in offsets)
                or offsets[0] < 0
                or offsets[1] < offsets[0]
                or offsets[1] > data_size
                or not isinstance(shape, list)
                or not all(isinstance(v, int) and v >= 0 for v in shape)
                or not isinstance(dtype, str)
            ):
                raise ValueError(f"{path}: malformed metadata for {name}")
            refs[name] = TensorRef(
                path=path,
                offset=data_base + offsets[0],
                nbytes=offsets[1] - offsets[0],
                dtype=dtype,
                shape=tuple(shape),
            )
    return refs


def infer_prefix(refs: dict[str, TensorRef]) -> str:
    plain = any(name.startswith("model.layers.") for name in refs)
    language = any(name.startswith("language_model.model.layers.") for name in refs)
    if plain and language:
        raise ValueError("both plain and language_model tensor prefixes are present")
    if language:
        return "language_model."
    if plain:
        return ""
    raise ValueError("cannot find model.layers.* tensors")


def infer_layers(refs: dict[str, TensorRef], prefix: str) -> list[int]:
    layers: set[int] = set()
    for name in refs:
        candidate = name[len(prefix):] if prefix and name.startswith(prefix) else name
        match = LAYER_RE.search(candidate)
        if match:
            layers.add(int(match.group(1)))
    if not layers:
        raise ValueError("no Kimi-style routed expert tensors found")
    return sorted(layers)


def parse_layers(text: str, available: list[int]) -> list[int]:
    if text == "all":
        return available
    chosen: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start, end = int(left), int(right)
            if end < start:
                raise ValueError(f"invalid layer range {part}")
            chosen.update(range(start, end + 1))
        else:
            chosen.add(int(part))
    missing = sorted(chosen - set(available))
    if missing:
        raise ValueError(f"requested layers have no expert tensors: {missing}")
    return sorted(chosen)


def read_exact(fd: int, offset: int, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.pread(fd, remaining, offset)
        if not chunk:
            raise OSError(f"short pread at offset {offset}, wanted {remaining} bytes")
        chunks.append(chunk)
        offset += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class TensorReader:
    def __init__(self) -> None:
        self._fds: dict[Path, int] = {}

    def read(self, ref: TensorRef, expected: int, name: str) -> bytes:
        if ref.nbytes != expected:
            raise ValueError(f"{name}: {ref.nbytes} bytes, expected {expected}")
        if ref.dtype not in {"U8", "I8"}:
            raise ValueError(f"{name}: expected byte-packed dtype, found {ref.dtype}")
        fd = self._fds.get(ref.path)
        if fd is None:
            fd = os.open(ref.path, os.O_RDONLY)
            self._fds[ref.path] = fd
        return read_exact(fd, ref.offset, ref.nbytes)

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def __enter__(self) -> "TensorReader":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def tensor_name(prefix: str, layer: int, expert: int, matrix: str, suffix: str) -> str:
    return (
        f"{prefix}model.layers.{layer}.block_sparse_moe.experts."
        f"{expert}.{matrix}.weight_{suffix}"
    )


def load_expert(
    reader: TensorReader,
    refs: dict[str, TensorRef],
    prefix: str,
    layer: int,
    expert: int,
    layout: Layout,
) -> tuple[bytes, bytes, bytes, bytes, bytes, bytes]:
    full_w1p = layout.intermediate * (layout.latent // 2)
    full_w1s = layout.intermediate * (layout.latent // layout.group_size)
    full_w2p = layout.latent * (layout.intermediate // 2)
    full_w2s = layout.latent * (layout.intermediate // layout.group_size)
    result: list[bytes] = []
    for matrix, suffix, expected in (
        ("w1", "packed", full_w1p),
        ("w1", "scale", full_w1s),
        ("w2", "packed", full_w2p),
        ("w2", "scale", full_w2s),
        ("w3", "packed", full_w1p),
        ("w3", "scale", full_w1s),
    ):
        name = tensor_name(prefix, layer, expert, matrix, suffix)
        ref = refs.get(name)
        if ref is None:
            raise ValueError(f"missing tensor: {name}")
        result.append(reader.read(ref, expected, name))
    return result[0], result[1], result[2], result[3], result[4], result[5]


def w2_column_slice(data: bytes, rows: int, full_row_bytes: int, start: int, width: int) -> bytes:
    output = bytearray(rows * width)
    dst = 0
    for row in range(rows):
        src = row * full_row_bytes + start
        output[dst:dst + width] = data[src:src + width]
        dst += width
    return bytes(output)


def make_record(
    layout: Layout,
    block: int,
    tensors: tuple[bytes, bytes, bytes, bytes, bytes, bytes],
) -> bytes:
    w1p, w1s, w2p, w2s, w3p, w3s = tensors
    row_start = block * layout.block_channels
    w1p_row = layout.latent // 2
    w1s_row = layout.latent // layout.group_size
    w2p_row = layout.intermediate // 2
    w2s_row = layout.intermediate // layout.group_size
    w2p_start = row_start // 2
    w2s_start = row_start // layout.group_size
    w2p_width = layout.block_channels // 2
    w2s_width = layout.block_channels // layout.group_size

    parts = [
        w1p[row_start * w1p_row:(row_start + layout.block_channels) * w1p_row],
        w1s[row_start * w1s_row:(row_start + layout.block_channels) * w1s_row],
        w3p[row_start * w1p_row:(row_start + layout.block_channels) * w1p_row],
        w3s[row_start * w1s_row:(row_start + layout.block_channels) * w1s_row],
        w2_column_slice(w2p, layout.latent, w2p_row, w2p_start, w2p_width),
        w2_column_slice(w2s, layout.latent, w2s_row, w2s_start, w2s_width),
    ]
    payload = b"".join(parts)
    expected = 2 * (layout.w1p + layout.w1s) + layout.w2p + layout.w2s
    if len(payload) != expected:
        raise AssertionError(f"record payload {len(payload)} != {expected}")
    return payload + bytes(layout.record_bytes - len(payload))


def existing_valid(path: Path, layout: Layout, layer: int) -> bool:
    try:
        if path.stat().st_size != layout.total_bytes:
            return False
        with path.open("rb") as fp:
            header = fp.read(HEADER_BYTES)
        old_layout, old_layer = decode_header(header)
        return old_layout == layout and old_layer == layer
    except (OSError, ValueError):
        return False


def pack_layer(
    output: Path,
    refs: dict[str, TensorRef],
    prefix: str,
    layer: int,
    layout: Layout,
    force: bool,
) -> dict[str, object]:
    if output.exists() and not force and existing_valid(output, layout, layer):
        return {"layer": layer, "path": str(output), "status": "reused", "bytes": layout.total_bytes}
    tmp = output.with_name(output.name + f".tmp.{os.getpid()}")
    tmp.unlink(missing_ok=True)
    digest = hashlib.sha256()
    header = encode_header(layout, layer)
    try:
        with tmp.open("wb", buffering=8 * 1024 * 1024) as fp, TensorReader() as reader:
            fp.write(header)
            digest.update(header)
            try:
                os.posix_fallocate(fp.fileno(), 0, layout.total_bytes)
                fp.seek(HEADER_BYTES)
            except (AttributeError, OSError):
                pass
            for expert in range(layout.experts):
                tensors = load_expert(reader, refs, prefix, layer, expert, layout)
                for block in range(layout.blocks):
                    record = make_record(layout, block, tensors)
                    fp.write(record)
                    digest.update(record)
                if expert == 0 or (expert + 1) % 32 == 0 or expert + 1 == layout.experts:
                    print(
                        f"layer {layer}: expert {expert + 1}/{layout.experts}",
                        file=sys.stderr,
                        flush=True,
                    )
            fp.flush()
            os.fsync(fp.fileno())
        if tmp.stat().st_size != layout.total_bytes:
            raise ValueError(f"packed file size {tmp.stat().st_size} != expected {layout.total_bytes}")
        os.replace(tmp, output)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    receipt = {
        "schema": "lattice.expert-blocks.v1",
        "layer": layer,
        "path": str(output),
        "status": "packed",
        "bytes": layout.total_bytes,
        "sha256": digest.hexdigest(),
        "layout": layout.__dict__,
        "source_prefix": prefix,
    }
    receipt_path = output.with_suffix(output.suffix + ".json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--layers", default="all", help="all, comma list, or ranges such as 1-4,8")
    parser.add_argument("--latent", type=int)
    parser.add_argument("--intermediate", type=int)
    parser.add_argument("--experts", type=int)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--block-channels", type=int, default=256)
    parser.add_argument("--prefix", choices=("auto", "plain", "language_model"), default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    try:
        config_path = model_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        refs = read_index(model_dir)
        if args.prefix == "auto":
            prefix = infer_prefix(refs)
        elif args.prefix == "language_model":
            prefix = "language_model."
        else:
            prefix = ""
        latent = args.latent or int(nested_find(config, ("routed_expert_hidden_size", "expert_hidden_size")) or 0)
        intermediate = args.intermediate or int(nested_find(config, ("moe_intermediate_size", "expert_intermediate_size")) or 0)
        experts = args.experts or int(nested_find(config, ("num_experts", "n_routed_experts", "num_local_experts")) or 0)
        layout = make_layout(latent, intermediate, args.group_size, args.block_channels, experts)
        available = infer_layers(refs, prefix)
        layers = parse_layers(args.layers, available)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "schema": "lattice.expert-block-pack.v1",
            "model_dir": str(model_dir),
            "output_dir": str(output_dir),
            "prefix": prefix,
            "layers": layers,
            "layout": layout.__dict__,
            "total_output_bytes": layout.total_bytes * len(layers),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        if args.dry_run:
            return 0
        receipts = []
        for layer in layers:
            output = output_dir / f"layer-{layer:03d}.ltblk"
            receipts.append(pack_layer(output, refs, prefix, layer, layout, args.force))
        (output_dir / "manifest.json").write_text(
            json.dumps({**summary, "receipts": receipts}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0
    except (OSError, ValueError, OverflowError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
