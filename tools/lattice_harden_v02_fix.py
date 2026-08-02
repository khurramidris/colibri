from pathlib import Path

path = Path("lattice/colibri.py")
text = path.read_text(encoding="utf-8")
old = '''def _qualification_key(key: str) -> bool:
    if key in FORBIDDEN_AMBIENT_KEYS or key in SERVING_ONLY_KEYS:
        return False
'''
new = '''def _qualification_key(key: str) -> bool:
    if key in SERVING_ONLY_KEYS:
        return False
    if key in FORBIDDEN_AMBIENT_KEYS and key not in {"DRAFT", "AUTOPIN", "REPIN"}:
        return False
'''
if text.count(old) != 1:
    raise SystemExit(f"expected one qualification policy block, found {text.count(old)}")
path.write_text(text.replace(old, new), encoding="utf-8")
