from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class LatticeError(RuntimeError):
    """Expected, user-actionable Lattice failure."""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def short_id(prefix: str, payload: Any) -> str:
    digest = sha256_bytes(canonical_json(payload))[:16]
    return f"{prefix}-{digest}"


def validate_id(value: str, label: str = "id") -> str:
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        raise LatticeError(f"invalid {label}: expected lowercase letters, digits, '.', '_' or '-' (max 64 chars)")
    return value


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise LatticeError(f"file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise LatticeError(f"invalid JSON in {path}: {error}") from error


def atomic_write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if exclusive and path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == data:
            return
        raise LatticeError(f"refusing to overwrite immutable record: {path}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive and path.exists():
            existing = path.read_text(encoding="utf-8")
            if existing == data:
                os.unlink(tmp_name)
                return
            raise LatticeError(f"refusing to overwrite immutable record: {path}")
        os.replace(tmp_name, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except (AttributeError, OSError):
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def ensure_under(root: Path, child: Path) -> Path:
    root_resolved = root.resolve()
    child_resolved = child.resolve()
    try:
        child_resolved.relative_to(root_resolved)
    except ValueError as error:
        raise LatticeError(f"path escapes workspace: {child}") from error
    return child_resolved


def bounded_text(text: str, limit: int = 65536) -> tuple[str, bool]:
    encoded = text.encode("utf-8", "replace")
    if len(encoded) <= limit:
        return text, False
    suffix = encoded[-limit:].decode("utf-8", "replace")
    return "[output truncated; tail follows]\n" + suffix, True
