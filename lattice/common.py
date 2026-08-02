from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
DEFAULT_JSON_LIMIT = 64 * 1024 * 1024
MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 2_000_000


class LatticeError(RuntimeError):
    """Expected, user-actionable Lattice failure."""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _validate_json_value(value: Any, *, label: str = "JSON value") -> None:
    """Reject values that are not finite, bounded JSON data.

    Python's json module accepts NaN and Infinity by default. Evidence must be
    portable, standards-compliant JSON, so both parsing and serialization fail
    closed on those values. A depth/node cap also prevents pathological files
    from consuming unbounded recursion or validation time.
    """
    stack: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES:
            raise LatticeError(f"{label} exceeds the maximum JSON node count")
        if depth > MAX_JSON_DEPTH:
            raise LatticeError(f"{label} exceeds the maximum JSON nesting depth")
        if current is None or isinstance(current, (str, bool, int)):
            continue
        if isinstance(current, float):
            if not math.isfinite(current):
                raise LatticeError(f"{label} contains a non-finite number")
            continue
        if isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)
            continue
        if isinstance(current, dict):
            for key, item in current.items():
                if not isinstance(key, str):
                    raise LatticeError(f"{label} contains a non-string object key")
                stack.append((item, depth + 1))
            continue
        raise LatticeError(f"{label} contains a non-JSON value of type {type(current).__name__}")


def strict_json_loads(text: str, *, label: str = "JSON") -> Any:
    try:
        value = json.loads(text, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise LatticeError(f"invalid {label}: {error}") from error
    _validate_json_value(value, label=label)
    return value


def canonical_json(value: Any) -> bytes:
    _validate_json_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


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


def load_json(path: Path, *, max_bytes: int = DEFAULT_JSON_LIMIT) -> Any:
    try:
        stat = path.stat()
    except FileNotFoundError as error:
        raise LatticeError(f"file not found: {path}") from error
    except OSError as error:
        raise LatticeError(f"cannot inspect JSON file {path}: {error}") from error
    if not path.is_file() or path.is_symlink():
        raise LatticeError(f"JSON path is not a regular file: {path}")
    if stat.st_size > max_bytes:
        raise LatticeError(f"JSON file exceeds {max_bytes} bytes: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise LatticeError(f"JSON file is not UTF-8: {path}") from error
    except OSError as error:
        raise LatticeError(f"cannot read JSON file {path}: {error}") from error
    return strict_json_loads(text, label=f"JSON in {path}")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        directory_fd = os.open(path, flags)
    except (AttributeError, OSError):
        return
    try:
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        os.close(directory_fd)


def _read_exact_existing(path: Path, expected: bytes) -> bool:
    try:
        stat = path.stat()
        if not path.is_file() or path.is_symlink() or stat.st_size != len(expected):
            return False
        return path.read_bytes() == expected
    except OSError:
        return False


def atomic_write_bytes(path: Path, data: bytes, *, exclusive: bool = False) -> None:
    """Publish bytes durably.

    In exclusive mode a same-directory hard link atomically claims the final
    path. This avoids the check-then-replace race that could overwrite evidence
    created by another process. Repeating the exact same write is idempotent;
    different bytes are refused. This is application-level write-once behavior,
    not filesystem immutability or remote attestation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            try:
                os.link(tmp_path, path)
            except FileExistsError as error:
                if _read_exact_existing(path, data):
                    return
                raise LatticeError(f"refusing to overwrite write-once record: {path}") from error
            except OSError as error:
                # Some filesystems do not support hard links. O_EXCL still
                # provides an atomic final-path claim without overwriting.
                try:
                    target_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                except FileExistsError as exists_error:
                    if _read_exact_existing(path, data):
                        return
                    raise LatticeError(f"refusing to overwrite write-once record: {path}") from exists_error
                except OSError:
                    raise LatticeError(f"cannot publish write-once record {path}: {error}") from error
                try:
                    with os.fdopen(target_fd, "wb") as target:
                        target.write(data)
                        target.flush()
                        os.fsync(target.fileno())
                except BaseException:
                    try:
                        path.unlink()
                    except OSError:
                        pass
                    raise
            _fsync_directory(path.parent)
            return
        os.replace(tmp_path, path)
        _fsync_directory(path.parent)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def atomic_write_text(path: Path, text: str, *, exclusive: bool = False) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), exclusive=exclusive)


def atomic_write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    _validate_json_value(value)
    try:
        data = (
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise LatticeError(f"cannot serialize JSON for {path}: {error}") from error
    atomic_write_bytes(path, data, exclusive=exclusive)


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
