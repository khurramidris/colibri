from pathlib import Path

path = Path("lattice/deploy.py")
text = path.read_text(encoding="utf-8")
marker = "from .workspace import Workspace\n"
insert = '''\n\nPROFILE_OVERRIDE_FLAGS = frozenset({\n    "--model", "--ram", "--ctx", "--gpu", "--vram", "--policy",\n    "--repin", "--cap", "--topp", "--topk", "--temp", "--auto-tier",\n    "--attach",\n})\nALLOWED_LAUNCH_COMMANDS = frozenset({"run", "chat", "serve", "web", "info"})\n\n\ndef validate_launch_args(coli_args: list[str]) -> None:\n    """Reject CLI options that would invalidate the promoted profile.\n\n    Serving controls such as host, port, API key, queue depth and generation\n    length remain available. Placement, model, policy and sampling controls are\n    bound to qualification evidence and cannot be replaced after verification.\n    """\n    for argument in coli_args:\n        flag = argument.split("=", 1)[0]\n        if flag in PROFILE_OVERRIDE_FLAGS:\n            raise LatticeError(\n                f"launch forbids {flag} because it overrides the verified profile; "\n                "create and qualify a new workspace instead"\n            )\n\n\ndef prepare_launch_args(coli_args: list[str]) -> list[str]:\n    prepared = list(coli_args or ["serve"])\n    command = prepared[0]\n    if command not in ALLOWED_LAUNCH_COMMANDS:\n        raise LatticeError(\n            f"launch supports only {', '.join(sorted(ALLOWED_LAUNCH_COMMANDS))}; "\n            f"received {command!r}"\n        )\n    validate_launch_args(prepared)\n    if command == "chat" and "--no-attach" not in prepared:\n        # Colibri chat can auto-attach to an already-running server. Force a\n        # private local engine so the verified profile cannot be bypassed.\n        prepared.insert(1, "--no-attach")\n    return prepared\n'''
if "PROFILE_OVERRIDE_FLAGS" in text:
    raise SystemExit("deployment guard already exists")
if marker not in text:
    raise SystemExit("deployment import marker did not match")
text = text.replace(marker, marker + insert)
old_model_guard = '''    if "--model" in coli_args or any(arg.startswith("--model=") for arg in coli_args):\n        raise LatticeError("launch forbids --model overrides because the profile is bound to one model fingerprint")\n'''
if old_model_guard not in text:
    raise SystemExit("old model-only launch guard did not match")
text = text.replace(old_model_guard, "    coli_args = prepare_launch_args(coli_args)\n")
old_default = '''    if not coli_args:\n        coli_args = ["serve"]\n'''
text = text.replace(old_default, "")
old_command = '''        command = [sys.executable, str(coli), *coli_args]\n'''
new_command = '''        # Saved `coli tune` profiles are a second, independent source of\n        # execution overrides. Disable them so the launched process uses only\n        # the environment that Lattice just recomputed and verified.\n        command = [sys.executable, str(coli), "--no-tune-profile", *coli_args]\n'''
if old_command not in text:
    raise SystemExit("launch command did not match")
path.write_text(text.replace(old_command, new_command), encoding="utf-8")

Path("lattice/tests/test_deploy.py").write_text('''from __future__ import annotations

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
''', encoding="utf-8")
