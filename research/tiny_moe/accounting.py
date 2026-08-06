"""Process and storage accounting helpers for Gate 3 runs."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from threading import Event, Thread
from typing import Any


def rss_bytes(pid: int | None = None) -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    try:
        return int(psutil.Process(pid).memory_info().rss)
    except (psutil.Error, OSError):
        return None


class PeakRSSSampler:
    def __init__(self, pid: int | None = None, interval_seconds: float = 0.01) -> None:
        self.pid = os.getpid() if pid is None else pid
        self.interval_seconds = interval_seconds
        self.samples: list[tuple[float, int]] = []
        self._stop = Event()
        self._thread: Thread | None = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            value = rss_bytes(self.pid)
            if value is not None:
                self.samples.append((time.perf_counter(), value))
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._thread = Thread(target=self._sample, name="rss-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        values = [value for _, value in self.samples]
        return {
            "sample_count": len(values),
            "rss_start_bytes": values[0] if values else None,
            "rss_end_bytes": values[-1] if values else None,
            "rss_peak_bytes": max(values) if values else None,
            "rss_samples": [
                {"t": timestamp, "rss_bytes": value}
                for timestamp, value in self.samples
            ],
        }


def cache_observation(path: str | Path) -> dict[str, Any]:
    """Report whether the process exposes a mapping for the packed file.

    This is deliberately not presented as physical-device traffic. Buffered
    reads can be served by the OS cache, and Windows does not expose an
    equivalent portable per-file page-cache residency counter here.
    """
    result: dict[str, Any] = {
        "path": str(path),
        "platform": os.name,
        "page_cache_bytes": None,
        "source": "not-available",
    }
    try:
        import psutil

        target = str(Path(path).resolve()).lower()
        mappings = []
        for mapping in psutil.Process().memory_maps(grouped=False):
            if mapping.path and str(Path(mapping.path).resolve()).lower() == target:
                mappings.append(
                    {
                        "path": mapping.path,
                        "rss": mapping.rss,
                        "private": getattr(mapping, "private", None),
                    }
                )
        result["mapped_file_entries"] = mappings
        result["source"] = "psutil.memory_maps"
    except (ImportError, OSError):
        result["mapped_file_entries"] = []
    return result


def timed_call(call: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    result = call()
    return result, time.perf_counter() - started


def summarize_read_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_layer: dict[str, dict[str, Any]] = {}
    by_phase: dict[str, dict[str, int]] = {}
    for event in events:
        layer = str(event["layer_id"])
        summary = by_layer.setdefault(
            layer,
            {"reads": 0, "payload_bytes": 0, "requested_bytes": 0, "load_seconds": 0.0},
        )
        summary["reads"] += 1
        summary["payload_bytes"] += event["payload_bytes"]
        summary["requested_bytes"] += event["requested_bytes"]
        summary["load_seconds"] += event["load_seconds"]
        phase = event["phase"]
        phase_summary = by_phase.setdefault(
            phase, {"reads": 0, "payload_bytes": 0, "requested_bytes": 0}
        )
        phase_summary["reads"] += 1
        phase_summary["payload_bytes"] += event["payload_bytes"]
        phase_summary["requested_bytes"] += event["requested_bytes"]
    return {"by_layer": by_layer, "by_phase": by_phase}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()
    print(json.dumps(cache_observation(args.path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
