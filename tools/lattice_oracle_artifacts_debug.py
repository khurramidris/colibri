from pathlib import Path

path = Path("lattice/tests/test_integration.py")
text = path.read_text(encoding="utf-8")
old = '            self.assertEqual(sum(run["status"] == "failed" for run in runs), expected_failures)\n'
new = '''            self.assertEqual(
                sum(run["status"] == "failed" for run in runs),
                expected_failures,
                [run.get("error") for run in runs if run["status"] == "failed"],
            )
'''
if text.count(old) != 1:
    raise SystemExit(f"expected one integration failure assertion, found {text.count(old)}")
path.write_text(text.replace(old, new), encoding="utf-8")
