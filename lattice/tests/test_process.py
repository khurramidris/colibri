from __future__ import annotations

import os
import sys
import time
import unittest

from lattice.process import run_bounded


class ProcessTests(unittest.TestCase):
    def test_output_is_bounded_at_capture_boundary(self):
        result = run_bounded([sys.executable, "-c", "import sys; print('x'*200000); print('e'*200000, file=sys.stderr)"],
            env=dict(os.environ), timeout=10, output_limit=4096)
        self.assertEqual(result.returncode, 0)
        self.assertTrue(result.output_truncated)
        self.assertLess(len(result.stdout.encode()), 5000)
        self.assertLess(len(result.stderr.encode()), 5000)

    def test_timeout_terminates_process(self):
        started = time.monotonic()
        result = run_bounded([sys.executable, "-c", "import time; time.sleep(30)"], env=dict(os.environ), timeout=1)
        self.assertTrue(result.timed_out)
        self.assertLess(time.monotonic() - started, 8)


if __name__ == "__main__":
    unittest.main()
