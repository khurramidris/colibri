from pathlib import Path

colibri = Path("lattice/colibri.py")
text = colibri.read_text(encoding="utf-8")
old = 'if key in FORBIDDEN_AMBIENT_KEYS and key not in {"DRAFT", "AUTOPIN", "REPIN"}:'
new = 'if key in FORBIDDEN_AMBIENT_KEYS and key not in {"DRAFT", "PIN_GB", "AUTOPIN", "REPIN"}:'
if text.count(old) != 1:
    raise SystemExit(f"expected one qualification exception set, found {text.count(old)}")
colibri.write_text(text.replace(old, new), encoding="utf-8")

test_path = Path("lattice/tests/test_colibri.py")
tests = test_path.read_text(encoding="utf-8")
old_test = '''        env = clean_environment({}, {"DRAFT": "0", "COLI_CUDA": "1"})
        self.assertEqual(env["DRAFT"], "0")
'''
new_test = '''        env = clean_environment({}, {"DRAFT": "0", "PIN_GB": "all", "COLI_CUDA": "1"})
        snapshot = qualification_environment(env)
        self.assertEqual(snapshot["DRAFT"], "0")
        self.assertEqual(snapshot["PIN_GB"], "all")
'''
if tests.count(old_test) != 1:
    raise SystemExit(f"expected one semantic override test block, found {tests.count(old_test)}")
test_path.write_text(tests.replace(old_test, new_test), encoding="utf-8")
