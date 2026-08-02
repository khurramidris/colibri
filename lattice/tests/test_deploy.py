from __future__ import annotations

import unittest

from lattice.common import LatticeError
from lattice.deploy import (
    ensure_profile_deployable,
    prepare_launch_args,
    validate_launch_args,
)


class DeployTests(unittest.TestCase):
    def test_unreviewed_or_profile_override_flags_are_rejected(self):
        for arguments in (
            ["serve", "--policy", "experimental-fast"],
            ["serve", "--ctx=8192"],
            ["chat", "--gpu", "none"],
            ["run", "--topk=8", "hello"],
            ["serve", "--auto-tier"],
            ["chat", "--attach", "http://127.0.0.1:8000"],
            ["serve", "--future-flag", "value"],
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(LatticeError, "reviewed allowlist"):
                    validate_launch_args(arguments)

    def test_reviewed_serving_controls_are_available(self):
        validate_launch_args([
            "serve", "--host", "127.0.0.1", "--port=8000",
            "--api-key", "secret", "--max-queue", "4", "--ngen", "512",
            "--cors-origin", "https://example.test", "--allowed-host", "example.test",
            "--queue-timeout", "30", "--kv-slots", "2", "--model-id", "local",
        ])
        validate_launch_args(["web", "--no-browser", "--port", "8000"])
        validate_launch_args(["chat", "--api-key", "secret", "--no-attach"])
        validate_launch_args(["run", "--ngen", "64", "hello", "world"])

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

    def test_missing_values_and_unexpected_positionals_are_rejected(self):
        for arguments, pattern in (
            (["serve", "--port"], "requires a value"),
            (["serve", "--port", "--host"], "requires a non-empty value"),
            (["info", "unexpected"], "does not accept positional"),
            (["run", "--ngen", "32"], "requires a prompt"),
            (["web", "--no-browser=true"], "does not accept a value"),
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(LatticeError, pattern):
                    validate_launch_args(arguments)

    def test_non_deployable_profile_is_blocked(self):
        with self.assertRaisesRegex(LatticeError, "deployment is blocked"):
            ensure_profile_deployable({
                "deployable": False,
                "deployment_blocker": "uncalibrated evidence",
            })
        ensure_profile_deployable({"deployable": True})


if __name__ == "__main__":
    unittest.main()
