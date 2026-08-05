#!/usr/bin/env python3
"""Correct the EOF anchor in the one-time native OLMoE migration."""
from __future__ import annotations

from pathlib import Path

path = Path(__file__).with_name("materialize_olmoe_native_contract.py")
text = path.read_text(encoding="utf-8")
old = '''    "    free(buf); free(arena);\\n    return 0;\\n}\\n",
    "    free(out); free(prompt); free(full); free(buf); free(arena);\\n"
    "    return 0;\\n"
    "}\\n",
    "reference cleanup",
)'''
new = '''    "    free(buf); free(arena);\\n    return 0;\\n}",
    "    free(out); free(prompt); free(full); free(buf); free(arena);\\n"
    "    return 0;\\n"
    "}\\n",
    "reference cleanup",
)'''
if text.count(old) != 1:
    raise SystemExit(f"expected one EOF anchor, found {text.count(old)}")
path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")
print("native migration EOF anchor corrected")
