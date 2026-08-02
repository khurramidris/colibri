from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


@dataclass(frozen=True)
class ProcessResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float
    output_truncated: bool


def _terminate_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        else:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        try:
            process.kill()
        except OSError:
            pass


def _tail(stream: BinaryIO, limit: int) -> tuple[str, bool]:
    stream.flush()
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    truncated = size > limit
    stream.seek(max(0, size - limit), os.SEEK_SET)
    data = stream.read(limit)
    text = data.decode("utf-8", "replace")
    if truncated:
        text = "[output truncated; tail follows]\n" + text
    return text, truncated


def run_bounded(
    command: list[str],
    *,
    env: dict[str, str],
    timeout: int,
    output_limit: int = 65536,
    cwd: Path | None = None,
) -> ProcessResult:
    """Run a process while keeping controller memory bounded by output_limit."""
    if timeout < 1:
        raise ValueError("timeout must be positive")
    if output_limit < 1024:
        raise ValueError("output_limit must be at least 1024 bytes")
    start = time.monotonic()
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    with tempfile.TemporaryFile(mode="w+b") as stdout_file, tempfile.TemporaryFile(mode="w+b") as stderr_file:
        process = subprocess.Popen(
            command,
            env=env,
            cwd=str(cwd) if cwd else None,
            stdin=subprocess.DEVNULL,
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=(os.name != "nt"),
            creationflags=creationflags,
        )
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_tree(process)
            process.wait()
        stdout, out_truncated = _tail(stdout_file, output_limit)
        stderr, err_truncated = _tail(stderr_file, output_limit)
    return ProcessResult(
        command=tuple(command),
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        duration_seconds=time.monotonic() - start,
        output_truncated=out_truncated or err_truncated,
    )
