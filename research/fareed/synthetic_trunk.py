#!/usr/bin/env python3
"""Synthetic layer-addressable compressed-trunk streaming prototype.

The prototype proves the mechanics required before touching a real checkpoint:

* one packed file with an explicit layer index;
* a declared resident-byte budget that pins a deterministic prefix;
* ``pread`` for non-resident layers;
* one-layer look-ahead using a bounded double buffer;
* exact CRC verification and requested-byte accounting;
* resident and streamed execution over identical compressed bytes.

It is deliberately model-agnostic. The tiny matrix payload used by the tests
is only an oracle for byte-identical execution; it is not a model benchmark.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import tempfile
import zlib
from array import array
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

TRUNK_MAGIC = b"CERNTRK1"
TRUNK_VERSION = 1
HEADER = struct.Struct("<8sIIQQ32s")
ENTRY = struct.Struct("<IIQQII")
MATRIX_MAGIC = b"QTM1"
MATRIX_HEADER = struct.Struct("<4sII")
DEFAULT_ALIGNMENT = 4096


class TrunkFormatError(RuntimeError):
    """Raised when a trunk file is malformed or corrupted."""


@dataclass(frozen=True)
class LayerEntry:
    layer_id: int
    offset: int
    length: int
    crc32: int
    io_length: int


@dataclass(frozen=True)
class ReaderStats:
    resident_budget_bytes: int
    resident_bytes: int
    streamed_payload_bytes_per_pass: int
    streamed_direct_io_bytes_per_pass: int
    startup_payload_read_bytes: int
    startup_read_calls: int
    payload_read_bytes: int
    read_calls: int
    resident_hits: int
    streamed_hits: int
    max_layer_bytes: int
    modeled_peak_working_bytes: int


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) & ~(alignment - 1)


def pack_trunk(
    path: str | os.PathLike[str],
    layers: Sequence[bytes],
    *,
    alignment: int = DEFAULT_ALIGNMENT,
) -> list[LayerEntry]:
    """Pack ordered layer payloads into one indexed file."""
    if not layers:
        raise ValueError("at least one layer is required")
    _align_up(0, alignment)
    for layer in layers:
        if not isinstance(layer, (bytes, bytearray, memoryview)):
            raise TypeError("layers must be byte-like")
        if len(layer) == 0:
            raise ValueError("empty layer payloads are not supported")

    index_offset = HEADER.size
    data_offset = _align_up(index_offset + ENTRY.size * len(layers), alignment)
    entries: list[LayerEntry] = []
    cursor = data_offset
    for layer_id, payload_raw in enumerate(layers):
        payload = bytes(payload_raw)
        cursor = _align_up(cursor, alignment)
        io_length = _align_up(len(payload), alignment)
        entries.append(
            LayerEntry(
                layer_id=layer_id,
                offset=cursor,
                length=len(payload),
                crc32=zlib.crc32(payload) & 0xFFFFFFFF,
                io_length=io_length,
            )
        )
        cursor += io_length

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        handle.write(
            HEADER.pack(
                TRUNK_MAGIC,
                TRUNK_VERSION,
                len(entries),
                index_offset,
                alignment,
                b"\0" * 32,
            )
        )
        for entry in entries:
            handle.write(
                ENTRY.pack(
                    entry.layer_id,
                    0,
                    entry.offset,
                    entry.length,
                    entry.crc32,
                    entry.io_length,
                )
            )
        if handle.tell() > data_offset:
            raise AssertionError("index overlaps payload region")
        handle.write(b"\0" * (data_offset - handle.tell()))
        for entry, payload_raw in zip(entries, layers):
            payload = bytes(payload_raw)
            if handle.tell() > entry.offset:
                raise AssertionError("payload overlap")
            handle.write(b"\0" * (entry.offset - handle.tell()))
            handle.write(payload)
            handle.write(b"\0" * (entry.io_length - entry.length))
        handle.flush()
        os.fsync(handle.fileno())
    return entries


def read_index(path: str | os.PathLike[str]) -> tuple[int, list[LayerEntry]]:
    with Path(path).open("rb") as handle:
        raw = handle.read(HEADER.size)
        if len(raw) != HEADER.size:
            raise TrunkFormatError("truncated trunk header")
        magic, version, n_layers, index_offset, alignment, _ = HEADER.unpack(raw)
        if magic != TRUNK_MAGIC:
            raise TrunkFormatError("bad trunk magic")
        if version != TRUNK_VERSION:
            raise TrunkFormatError(f"unsupported trunk version {version}")
        _align_up(0, alignment)
        if index_offset != HEADER.size:
            raise TrunkFormatError("unexpected index offset")
        entries: list[LayerEntry] = []
        for expected_id in range(n_layers):
            raw_entry = handle.read(ENTRY.size)
            if len(raw_entry) != ENTRY.size:
                raise TrunkFormatError("truncated trunk index")
            layer_id, flags, offset, length, crc32, io_length = ENTRY.unpack(raw_entry)
            if flags != 0:
                raise TrunkFormatError("unsupported layer flags")
            if layer_id != expected_id:
                raise TrunkFormatError("layer ids are not contiguous")
            if offset % alignment:
                raise TrunkFormatError("layer offset is not aligned")
            if length <= 0:
                raise TrunkFormatError("invalid layer length")
            if io_length < length or io_length % alignment:
                raise TrunkFormatError("invalid aligned I/O length")
            entries.append(LayerEntry(layer_id, offset, length, crc32, io_length))
    file_size = Path(path).stat().st_size
    previous_end = 0
    for entry in entries:
        if entry.offset < previous_end:
            raise TrunkFormatError("overlapping layer payloads")
        if entry.offset + entry.io_length > file_size:
            raise TrunkFormatError("layer extends beyond file")
        previous_end = entry.offset + entry.io_length
    return alignment, entries


class TrunkReader:
    """Partially resident deterministic trunk reader."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        resident_budget_bytes: int,
        verify_crc: bool = True,
        prefetch: bool = True,
    ) -> None:
        if resident_budget_bytes < 0:
            raise ValueError("resident_budget_bytes must be non-negative")
        self.path = Path(path)
        self.alignment, self.entries = read_index(self.path)
        self.fd = os.open(self.path, os.O_RDONLY)
        self.verify_crc = verify_crc
        self.prefetch_enabled = prefetch
        self.resident_budget_bytes = int(resident_budget_bytes)
        self._resident: dict[int, bytes] = {}
        self._startup_payload_read_bytes = 0
        self._startup_read_calls = 0
        self._payload_read_bytes = 0
        self._read_calls = 0
        self._resident_hits = 0
        self._streamed_hits = 0
        self._closed = False

        used = 0
        for entry in self.entries:
            if used + entry.length > self.resident_budget_bytes:
                break
            payload = self._pread_exact(entry, phase="startup")
            self._resident[entry.layer_id] = payload
            used += entry.length
        self.resident_bytes = used
        self.streamed_payload_bytes_per_pass = sum(
            entry.length for entry in self.entries if entry.layer_id not in self._resident
        )
        self.streamed_direct_io_bytes_per_pass = sum(
            entry.io_length for entry in self.entries if entry.layer_id not in self._resident
        )
        self.max_layer_bytes = max(entry.length for entry in self.entries)
        max_streamed = max(
            (entry.io_length for entry in self.entries if entry.layer_id not in self._resident),
            default=0,
        )
        self.modeled_peak_working_bytes = self.resident_bytes + 2 * max_streamed

    def close(self) -> None:
        if not self._closed:
            os.close(self.fd)
            self._closed = True

    def __enter__(self) -> "TrunkReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("reader is closed")

    def _pread_exact(self, entry: LayerEntry, *, phase: str) -> bytes:
        self._assert_open()
        chunks: list[bytes] = []
        remaining = entry.length
        offset = entry.offset
        while remaining:
            chunk = os.pread(self.fd, remaining, offset)
            if not chunk:
                raise TrunkFormatError(f"short read for layer {entry.layer_id}")
            chunks.append(chunk)
            if phase == "startup":
                self._startup_payload_read_bytes += len(chunk)
                self._startup_read_calls += 1
            elif phase == "token":
                self._payload_read_bytes += len(chunk)
                self._read_calls += 1
            elif phase != "none":
                raise ValueError(f"unknown read phase {phase!r}")
            remaining -= len(chunk)
            offset += len(chunk)
        payload = b"".join(chunks)
        if self.verify_crc and (zlib.crc32(payload) & 0xFFFFFFFF) != entry.crc32:
            raise TrunkFormatError(f"CRC mismatch for layer {entry.layer_id}")
        return payload

    def read_layer(self, layer_id: int) -> bytes:
        self._assert_open()
        if not 0 <= layer_id < len(self.entries):
            raise IndexError(layer_id)
        resident = self._resident.get(layer_id)
        if resident is not None:
            self._resident_hits += 1
            return resident
        self._streamed_hits += 1
        return self._pread_exact(self.entries[layer_id], phase="token")

    def iter_layers(self) -> Iterator[tuple[int, bytes]]:
        """Yield one ordered pass, prefetching at most one future layer."""
        self._assert_open()
        if not self.prefetch_enabled:
            for entry in self.entries:
                yield entry.layer_id, self.read_layer(entry.layer_id)
            return

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="trunk-prefetch") as pool:
            future: Future[bytes] | None = None
            future_layer: int | None = None

            def submit(layer_id: int) -> tuple[Future[bytes] | None, int | None]:
                resident = self._resident.get(layer_id)
                if resident is not None:
                    return None, None
                self._streamed_hits += 1
                return (
                    pool.submit(
                        self._pread_exact,
                        self.entries[layer_id],
                        phase="token",
                    ),
                    layer_id,
                )

            for position, entry in enumerate(self.entries):
                layer_id = entry.layer_id
                resident = self._resident.get(layer_id)
                if resident is not None:
                    self._resident_hits += 1
                    payload = resident
                elif future is not None and future_layer == layer_id:
                    payload = future.result()
                    future = None
                    future_layer = None
                else:
                    self._streamed_hits += 1
                    payload = self._pread_exact(entry, phase="token")

                next_id = position + 1
                if future is None and next_id < len(self.entries):
                    future, future_layer = submit(next_id)
                yield layer_id, payload

            if future is not None:
                future.result()

    def stats(self) -> ReaderStats:
        return ReaderStats(
            resident_budget_bytes=self.resident_budget_bytes,
            resident_bytes=self.resident_bytes,
            streamed_payload_bytes_per_pass=self.streamed_payload_bytes_per_pass,
            streamed_direct_io_bytes_per_pass=self.streamed_direct_io_bytes_per_pass,
            startup_payload_read_bytes=self._startup_payload_read_bytes,
            startup_read_calls=self._startup_read_calls,
            payload_read_bytes=self._payload_read_bytes,
            read_calls=self._read_calls,
            resident_hits=self._resident_hits,
            streamed_hits=self._streamed_hits,
            max_layer_bytes=self.max_layer_bytes,
            modeled_peak_working_bytes=self.modeled_peak_working_bytes,
        )


def make_quantized_matrix_payload(weights: Sequence[Sequence[float]]) -> bytes:
    """Create a tiny per-row-int8 compressed matrix payload."""
    rows = len(weights)
    if rows == 0:
        raise ValueError("empty matrix")
    cols = len(weights[0])
    if cols == 0 or any(len(row) != cols for row in weights):
        raise ValueError("matrix must be rectangular")

    scales: list[float] = []
    quantized = array("b")
    for row in weights:
        amax = max(abs(float(value)) for value in row)
        scale = max(amax / 127.0, 1.0e-12)
        scales.append(scale)
        for value in row:
            q = int(round(float(value) / scale))
            quantized.append(max(-127, min(127, q)))

    return (
        MATRIX_HEADER.pack(MATRIX_MAGIC, rows, cols)
        + struct.pack(f"<{rows}f", *scales)
        + quantized.tobytes()
    )


def apply_quantized_matrix(payload: bytes, vector: Sequence[float]) -> tuple[float, ...]:
    if len(payload) < MATRIX_HEADER.size:
        raise TrunkFormatError("truncated matrix payload")
    magic, rows, cols = MATRIX_HEADER.unpack_from(payload, 0)
    if magic != MATRIX_MAGIC:
        raise TrunkFormatError("bad matrix payload magic")
    if len(vector) != cols:
        raise ValueError("vector width mismatch")
    scale_offset = MATRIX_HEADER.size
    data_offset = scale_offset + rows * 4
    expected = data_offset + rows * cols
    if len(payload) != expected:
        raise TrunkFormatError("matrix payload length mismatch")
    scales = struct.unpack_from(f"<{rows}f", payload, scale_offset)
    q = memoryview(payload)[data_offset:].cast("b")

    output: list[float] = []
    for row in range(rows):
        acc = 0.0
        base = row * cols
        for col, value in enumerate(vector):
            acc += float(q[base + col]) * float(value)
        output.append(acc * scales[row])
    return tuple(output)


def execute_trunk(reader: TrunkReader, vector: Sequence[float]) -> tuple[float, ...]:
    state = tuple(float(value) for value in vector)
    for _, payload in reader.iter_layers():
        transformed = apply_quantized_matrix(payload, state)
        state = tuple(math.tanh(a + 0.125 * b) for a, b in zip(transformed, state))
    return state


def make_synthetic_layers(
    *,
    n_layers: int,
    width: int,
    seed: int = 0,
) -> list[bytes]:
    if n_layers <= 0 or width <= 0:
        raise ValueError("n_layers and width must be positive")
    state = (seed + 1) & 0xFFFFFFFFFFFFFFFF

    def rand() -> float:
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
        state ^= state >> 7
        state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
        return ((state & 0xFFFF) / 32767.5 - 1.0) * 0.15

    layers: list[bytes] = []
    for layer in range(n_layers):
        matrix = []
        for row in range(width):
            values = []
            for col in range(width):
                value = rand()
                if row == col:
                    value += 0.8 + 0.01 * layer
                values.append(value)
            matrix.append(values)
        layers.append(make_quantized_matrix_payload(matrix))
    return layers


def differential_report(
    *,
    n_layers: int = 12,
    width: int = 32,
    resident_layers: int = 4,
    passes: int = 3,
    seed: int = 0,
) -> dict[str, object]:
    layers = make_synthetic_layers(n_layers=n_layers, width=width, seed=seed)
    with tempfile.TemporaryDirectory(prefix="cerno-trunk-") as tmp:
        path = Path(tmp) / "trunk.bin"
        entries = pack_trunk(path, layers)
        budget = sum(entry.length for entry in entries[:resident_layers])
        vector = tuple((i + 1) / width for i in range(width))

        with TrunkReader(path, resident_budget_bytes=sum(e.length for e in entries)) as full:
            reference = execute_trunk(full, vector)

        outputs = []
        with TrunkReader(path, resident_budget_bytes=budget, prefetch=True) as streamed:
            for _ in range(passes):
                outputs.append(execute_trunk(streamed, vector))
            stats = streamed.stats()

        reference_bytes = struct.pack(f"<{len(reference)}d", *reference)
        matches = [
            struct.pack(f"<{len(output)}d", *output) == reference_bytes
            for output in outputs
        ]
        return {
            "n_layers": n_layers,
            "width": width,
            "resident_layers": resident_layers,
            "passes": passes,
            "bit_identical_each_pass": matches,
            "resident_bytes": stats.resident_bytes,
            "streamed_payload_bytes_per_pass": stats.streamed_payload_bytes_per_pass,
            "streamed_direct_io_bytes_per_pass": stats.streamed_direct_io_bytes_per_pass,
            "startup_payload_read_bytes": stats.startup_payload_read_bytes,
            "startup_read_calls": stats.startup_read_calls,
            "payload_read_bytes": stats.payload_read_bytes,
            "expected_payload_read_bytes": stats.streamed_payload_bytes_per_pass * passes,
            "modeled_direct_io_bytes": stats.streamed_direct_io_bytes_per_pass * passes,
            "read_calls": stats.read_calls,
            "modeled_peak_working_bytes": stats.modeled_peak_working_bytes,
            "file_bytes": path.stat().st_size,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--resident-layers", type=int, default=4)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = differential_report(
        n_layers=args.layers,
        width=args.width,
        resident_layers=args.resident_layers,
        passes=args.passes,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all(report["bit_identical_each_pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
