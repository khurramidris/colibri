from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from .common import LatticeError


@dataclass(frozen=True)
class ProcessResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float
    output_truncated: bool
    stdout_bytes: int = 0
    stderr_bytes: int = 0


class _TailCapture:
    """Drain a pipe continuously while retaining only its bounded tail."""

    def __init__(self, stream: BinaryIO, limit: int):
        self.stream = stream
        self.limit = limit
        self.tail = bytearray()
        self.total = 0
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            while True:
                chunk = self.stream.read(65536)
                if not chunk:
                    break
                self.total += len(chunk)
                if len(chunk) >= self.limit:
                    self.tail[:] = chunk[-self.limit:]
                else:
                    overflow = len(self.tail) + len(chunk) - self.limit
                    if overflow > 0:
                        del self.tail[:overflow]
                    self.tail.extend(chunk)
        except (OSError, ValueError) as error:
            # Closing a pipe after process termination is an expected way to
            # unblock a descendant that inherited the handle. Preserve the
            # captured tail and expose only unexpected failures to the caller.
            self.error = error

    def text(self) -> tuple[str, bool]:
        truncated = self.total > self.limit
        text = bytes(self.tail).decode("utf-8", "replace")
        if truncated:
            text = "[output truncated; tail follows]\n" + text
        return text, truncated


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
                check=False,
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


def _finish_reader(
    thread: threading.Thread,
    capture: _TailCapture,
    stream: BinaryIO,
) -> None:
    thread.join(timeout=5)
    if thread.is_alive():
        try:
            stream.close()
        except OSError:
            pass
        thread.join(timeout=5)
    if thread.is_alive():
        raise LatticeError("subprocess output reader did not terminate")


def run_bounded(
    command: list[str],
    *,
    env: dict[str, str],
    timeout: int,
    output_limit: int = 65536,
    cwd: Path | None = None,
) -> ProcessResult:
    """Run a process and retain bounded stdout/stderr tails.

    Pipes are drained concurrently into in-memory ring buffers. Unlike the old
    temporary-file implementation, a noisy process cannot consume unbounded
    temporary-disk space while the controller waits for it to finish.
    """
    if not command or any(not isinstance(value, str) or not value for value in command):
        raise LatticeError("subprocess command must contain non-empty strings")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
        raise LatticeError("timeout must be a positive integer")
    if isinstance(output_limit, bool) or not isinstance(output_limit, int) or output_limit < 1024:
        raise LatticeError("output_limit must be at least 1024 bytes")
    start = time.monotonic()
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    try:
        process = subprocess.Popen(
            command,
            env=env,
            cwd=str(cwd) if cwd else None,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=(os.name != "nt"),
            creationflags=creationflags,
        )
    except OSError as error:
        raise LatticeError(f"cannot start subprocess {command[0]!r}: {error}") from error
    assert process.stdout is not None and process.stderr is not None
    stdout_capture = _TailCapture(process.stdout, output_limit)
    stderr_capture = _TailCapture(process.stderr, output_limit)
    stdout_thread = threading.Thread(target=stdout_capture.run, name="lattice-stdout", daemon=True)
    stderr_thread = threading.Thread(target=stderr_capture.run, name="lattice-stderr", daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_tree(process)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired as error:
            raise LatticeError("subprocess tree did not terminate after timeout") from error
    finally:
        _finish_reader(stdout_thread, stdout_capture, process.stdout)
        _finish_reader(stderr_thread, stderr_capture, process.stderr)
        try:
            process.stdout.close()
            process.stderr.close()
        except OSError:
            pass
    stdout, out_truncated = stdout_capture.text()
    stderr, err_truncated = stderr_capture.text()
    returncode = process.returncode
    if returncode is None:
        raise LatticeError("subprocess ended without a return code")
    return ProcessResult(
        command=tuple(command),
        returncode=int(returncode),
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        duration_seconds=time.monotonic() - start,
        output_truncated=out_truncated or err_truncated,
        stdout_bytes=stdout_capture.total,
        stderr_bytes=stderr_capture.total,
    )
