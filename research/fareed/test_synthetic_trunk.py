from __future__ import annotations

import os
import struct
import tempfile
import unittest
from pathlib import Path

from synthetic_trunk import (
    TrunkFormatError,
    TrunkReader,
    _pread,
    _pwrite,
    differential_report,
    execute_trunk,
    make_synthetic_layers,
    pack_trunk,
    read_index,
)


class SyntheticTrunkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="trunk-test-")
        self.path = Path(self.tmp.name) / "trunk.bin"
        self.layers = make_synthetic_layers(n_layers=7, width=16, seed=42)
        self.entries = pack_trunk(self.path, self.layers)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_index_is_aligned_and_round_trips(self) -> None:
        alignment, entries = read_index(self.path)
        self.assertEqual(4096, alignment)
        self.assertEqual(self.entries, entries)
        for entry in entries:
            self.assertEqual(0, entry.offset % alignment)
            self.assertEqual(0, entry.io_length % alignment)
            self.assertGreaterEqual(entry.io_length, entry.length)
            with self.path.open("rb") as handle:
                handle.seek(entry.offset)
                self.assertEqual(self.layers[entry.layer_id], handle.read(entry.length))

    def test_budget_pins_only_whole_prefix_layers(self) -> None:
        first_two = self.entries[0].length + self.entries[1].length
        budget = first_two + self.entries[2].length - 1
        with TrunkReader(self.path, resident_budget_bytes=budget) as reader:
            self.assertEqual(first_two, reader.resident_bytes)
            self.assertEqual(
                sum(entry.length for entry in self.entries[2:]),
                reader.streamed_payload_bytes_per_pass,
            )
            self.assertLessEqual(reader.resident_bytes, budget)
            stats = reader.stats()
            self.assertEqual(first_two, stats.startup_payload_read_bytes)
            self.assertEqual(2, stats.startup_read_calls)

    def test_resident_and_streamed_execution_are_bit_identical(self) -> None:
        vector = tuple((i + 1) / 16 for i in range(16))
        full_budget = sum(entry.length for entry in self.entries)
        partial_budget = sum(entry.length for entry in self.entries[:2])
        with TrunkReader(self.path, resident_budget_bytes=full_budget) as full:
            expected = execute_trunk(full, vector)
        with TrunkReader(self.path, resident_budget_bytes=partial_budget) as streamed:
            actual = execute_trunk(streamed, vector)
        self.assertEqual(
            struct.pack(f"<{len(expected)}d", *expected),
            struct.pack(f"<{len(actual)}d", *actual),
        )

    def test_prefetch_reads_each_streamed_layer_once_per_pass(self) -> None:
        budget = sum(entry.length for entry in self.entries[:3])
        with TrunkReader(self.path, resident_budget_bytes=budget, prefetch=True) as reader:
            list(reader.iter_layers())
            list(reader.iter_layers())
            stats = reader.stats()
        expected_streamed_layers = len(self.entries) - 3
        self.assertEqual(expected_streamed_layers * 2, stats.read_calls)
        self.assertEqual(
            stats.streamed_payload_bytes_per_pass * 2,
            stats.payload_read_bytes,
        )
        self.assertEqual(3 * 2, stats.resident_hits)
        self.assertEqual(expected_streamed_layers * 2, stats.streamed_hits)

    def test_no_residency_and_full_residency_extremes(self) -> None:
        with TrunkReader(self.path, resident_budget_bytes=0) as none:
            self.assertEqual(0, none.resident_bytes)
            self.assertEqual(
                sum(e.length for e in self.entries),
                none.streamed_payload_bytes_per_pass,
            )
        with TrunkReader(
            self.path,
            resident_budget_bytes=sum(e.length for e in self.entries),
        ) as all_resident:
            list(all_resident.iter_layers())
            stats = all_resident.stats()
            self.assertEqual(0, stats.streamed_payload_bytes_per_pass)
            self.assertEqual(0, stats.streamed_direct_io_bytes_per_pass)
            self.assertEqual(0, stats.payload_read_bytes)
            self.assertEqual(len(self.entries), stats.resident_hits)

    def test_modeled_peak_uses_two_aligned_streaming_buffers(self) -> None:
        budget = sum(entry.length for entry in self.entries[:1])
        with TrunkReader(self.path, resident_budget_bytes=budget) as reader:
            max_streamed = max(entry.io_length for entry in self.entries[1:])
            self.assertEqual(
                reader.resident_bytes + 2 * max_streamed,
                reader.modeled_peak_working_bytes,
            )

    def test_corruption_is_detected(self) -> None:
        target = self.entries[4]
        fd = os.open(self.path, os.O_RDWR)
        try:
            original = _pread(fd, 1, target.offset + 3)
            _pwrite(fd, bytes([original[0] ^ 0xFF]), target.offset + 3)
        finally:
            os.close(fd)
        with TrunkReader(self.path, resident_budget_bytes=0) as reader:
            with self.assertRaisesRegex(TrunkFormatError, "CRC mismatch"):
                reader.read_layer(4)

    def test_bad_header_is_rejected(self) -> None:
        fd = os.open(self.path, os.O_RDWR)
        try:
            _pwrite(fd, b"BADMAGIC", 0)
        finally:
            os.close(fd)
        with self.assertRaisesRegex(TrunkFormatError, "bad trunk magic"):
            read_index(self.path)

    def test_closed_reader_refuses_access(self) -> None:
        reader = TrunkReader(self.path, resident_budget_bytes=0)
        reader.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            reader.read_layer(0)

    def test_differential_report_accounts_exact_payload_requests(self) -> None:
        report = differential_report(
            n_layers=9,
            width=16,
            resident_layers=3,
            passes=4,
            seed=9,
        )
        self.assertTrue(all(report["bit_identical_each_pass"]))
        self.assertEqual(
            report["expected_payload_read_bytes"],
            report["payload_read_bytes"],
        )
        self.assertGreaterEqual(
            report["modeled_direct_io_bytes"],
            report["payload_read_bytes"],
        )


if __name__ == "__main__":
    unittest.main()
