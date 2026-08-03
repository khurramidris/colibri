from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from .model import AtlasError, AtlasManifest, RouteRecord, validate_complete_routes

MAX_LINE_BYTES = 2 * 1024 * 1024
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 100_000


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _validate_json(value: Any) -> None:
    stack = [(value, 0)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES:
            raise AtlasError("JSON exceeds node limit")
        if depth > MAX_JSON_DEPTH:
            raise AtlasError("JSON exceeds nesting limit")
        if current is None or isinstance(current, (str, bool, int)):
            continue
        if isinstance(current, float):
            if not math.isfinite(current):
                raise AtlasError("JSON contains a non-finite number")
            continue
        if isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)
            continue
        if isinstance(current, dict):
            for key, item in current.items():
                if not isinstance(key, str):
                    raise AtlasError("JSON object key must be a string")
                stack.append((item, depth + 1))
            continue
        raise AtlasError(f"unsupported JSON value: {type(current).__name__}")


def canonical_json(value: Any) -> bytes:
    _validate_json(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_loads(text: str, *, label: str) -> Any:
    try:
        value = json.loads(text, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise AtlasError(f"invalid {label}: {error}") from error
    _validate_json(value)
    return value


def load_manifest(path: Path) -> AtlasManifest:
    if path.is_symlink() or not path.is_file():
        raise AtlasError(f"manifest is not a regular file: {path}")
    return AtlasManifest.from_dict(strict_loads(path.read_text("utf-8"), label=str(path)))


def load_routes(path: Path, manifest: AtlasManifest) -> list[RouteRecord]:
    if path.is_symlink() or not path.is_file():
        raise AtlasError(f"trace is not a regular file: {path}")
    records: list[RouteRecord] = []
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            if len(raw) > MAX_LINE_BYTES:
                raise AtlasError(f"{path}:{line_number}: line exceeds {MAX_LINE_BYTES} bytes")
            stripped = raw.strip()
            if not stripped or stripped.startswith(b"#"):
                continue
            try:
                text = stripped.decode("utf-8")
            except UnicodeDecodeError as error:
                raise AtlasError(f"{path}:{line_number}: line is not UTF-8") from error
            value = strict_loads(text, label=f"{path}:{line_number}")
            if not isinstance(value, dict):
                raise AtlasError(f"{path}:{line_number}: route row must be an object")
            try:
                records.append(
                    RouteRecord.from_dict(
                        value,
                        n_layers=manifest.n_layers,
                        n_experts=manifest.n_experts,
                        top_k=manifest.top_k,
                    )
                )
            except AtlasError as error:
                raise AtlasError(f"{path}:{line_number}: {error}") from error
    return validate_complete_routes(records, manifest)


def atomic_write_bytes(path: Path, data: bytes, *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            try:
                os.link(tmp, path)
            except FileExistsError as error:
                if path.is_file() and not path.is_symlink() and path.read_bytes() == data:
                    return
                raise AtlasError(f"refusing to overwrite write-once evidence: {path}") from error
            except OSError as link_error:
                try:
                    target_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                except FileExistsError as error:
                    if path.is_file() and not path.is_symlink() and path.read_bytes() == data:
                        return
                    raise AtlasError(f"refusing to overwrite write-once evidence: {path}") from error
                except OSError as error:
                    raise AtlasError(f"cannot publish write-once evidence {path}: {link_error}") from error
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
        else:
            os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value: Any, *, exclusive: bool = True) -> None:
    data = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    atomic_write_bytes(path, data, exclusive=exclusive)


def atomic_write_text(path: Path, text: str, *, exclusive: bool = True) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), exclusive=exclusive)


def write_jsonl(path: Path, values: Iterable[dict[str, Any]], *, exclusive: bool = True) -> None:
    payload = b"".join(canonical_json(value) + b"\n" for value in values)
    atomic_write_bytes(path, payload, exclusive=exclusive)
