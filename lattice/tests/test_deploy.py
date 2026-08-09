from __future__ import annotations

import unittest

from lattice.common import LatticeError
from lattice.deploy import prepare_launch_args, validate_launch_args


class DeployTests(unittest.TestCase):
    def test_profile_overrides_are_rejected_in_both_flag_forms(self):
        for arguments in (
            ["serve", "--policy", "experimental-fast"],
            ["serve", "--ctx=8192"],
            ["chat", "--gpu", "none"],
            ["run", "--topk=8", "hello"],
            ["serve", "--auto-tier"],
            ["chat", "--attach", "http://127.0.0.1:8000"],
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(LatticeError, "overrides the verified profile"):
                    validate_launch_args(arguments)

    def test_serving_and_generation_controls_remain_available(self):
        validate_launch_args([
            "serve", "--host", "127.0.0.1", "--port", "8000",
            "--api-key", "secret", "--max-queue", "4", "--ngen", "512",
        ])

    def test_chat_is_forced_to_a_private_local_engine(self):
        self.assertEqual(prepare_launch_args(["chat"]), ["chat", "--no-attach"])
        self.assertEqual(
            prepare_launch_args(["chat", "--no-attach"]),
            ["chat", "--no-attach"],
        )

    def test_non_deployment_subcommands_are_rejected(self):
        for command in ("tune", "convert", "mirror", "build"):
            with self.subTest(command=command):
                with self.assertRaisesRegex(LatticeError, "launch supports only"):
                    prepare_launch_args([command])

    def test_model_override_is_rejected(self):
        with self.assertRaisesRegex(LatticeError, "--model"):
            validate_launch_args(["chat", "--model=/tmp/other-model"])


if __name__ == "__main__":
    unittest.main()
