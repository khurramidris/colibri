from __future__ import annotations

import os
import sys
import time
import unittest

from lattice.common import LatticeError
from lattice.process import run_bounded


class ProcessTests(unittest.TestCase):
    def test_output_is_bounded_at_capture_boundary(self):
        result = run_bounded(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.write('x'*10000000); sys.stderr.write('e'*8000000)",
            ],
            env=dict(os.environ),
            timeout=20,
            output_limit=4096,
        )
        self.assertEqual(result.returncode, 0)
        self.assertTrue(result.output_truncated)
        self.assertEqual(result.stdout_bytes, 10_000_000)
        self.assertEqual(result.stderr_bytes, 8_000_000)
        self.assertLess(len(result.stdout.encode()), 5000)
        self.assertLess(len(result.stderr.encode()), 5000)
        self.assertTrue(result.stdout.endswith("x" * 4096))
        self.assertTrue(result.stderr.endswith("e" * 4096))

    def test_timeout_terminates_process(self):
        started = time.monotonic()
        result = run_bounded(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            env=dict(os.environ),
            timeout=1,
        )
        self.assertTrue(result.timed_out)
        self.assertLess(time.monotonic() - started, 12)

    def test_missing_executable_is_user_actionable(self):
        with self.assertRaisesRegex(LatticeError, "cannot start subprocess"):
            run_bounded(
                ["lattice-executable-that-does-not-exist"],
                env=dict(os.environ),
                timeout=1,
            )


if __name__ == "__main__":
    unittest.main()
