from __future__ import annotations

from collections import Counter, defaultdict, OrderedDict
from dataclasses import dataclass
from typing import Iterable

from .model import AtlasManifest, RouteRecord


@dataclass(slots=True)
class CacheOutcome:
    policy: str
    capacity: int
    hits: int
    misses: int
    transferred_bytes: int
    nonempty_miss_layers: int
    token_miss_bytes: dict[str, int]
    token_miss_layers: dict[str, int]

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def as_dict(self) -> dict:
        return {
            "policy": self.policy,
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "transferred_bytes": self.transferred_bytes,
            "nonempty_miss_layers": self.nonempty_miss_layers,
            "token_miss_bytes": self.token_miss_bytes,
            "token_miss_layers": self.token_miss_layers,
        }


def _token_key(record: RouteRecord) -> str:
    return f"{record.run_id}/{record.request_id}/{record.phase}/{record.token}"


def _ordered(records: Iterable[RouteRecord]) -> list[RouteRecord]:
    return sorted(records, key=lambda r: (r.run_id, r.request_id, 0 if r.phase == "prefill" else 1, r.token, r.layer))


def _outcome(policy: str, capacity: int, events: list[tuple[RouteRecord, int]], manifest: AtlasManifest) -> CacheOutcome:
    hits = sum(hit for _, hit in events)
    total = len(events)
    token_misses: dict[str, int] = defaultdict(int)
    layer_miss_seen: set[tuple[str, int]] = set()
    token_layers: dict[str, int] = defaultdict(int)
    for record, hit in events:
        key = _token_key(record)
        token_misses.setdefault(key, 0)
        token_layers.setdefault(key, 0)
        if hit:
            continue
        token_misses[key] += manifest.expert_bytes
        marker = (key, record.layer)
        if marker not in layer_miss_seen:
            layer_miss_seen.add(marker)
            token_layers[key] += 1
    misses = total - hits
    return CacheOutcome(
        policy=policy,
        capacity=capacity,
        hits=hits,
        misses=misses,
        transferred_bytes=misses * manifest.expert_bytes,
        nonempty_miss_layers=len(layer_miss_seen),
        token_miss_bytes=dict(sorted(token_misses.items())),
        token_miss_layers=dict(sorted(token_layers.items())),
    )


def simulate_static(
    train: Iterable[RouteRecord], test: Iterable[RouteRecord], capacity: int, manifest: AtlasManifest
) -> CacheOutcome:
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for record in train:
        counts[record.layer].update(record.experts)
    resident = {
        layer: {expert for expert, _ in counts[layer].most_common(capacity)}
        for layer in range(manifest.n_layers)
    }
    events = [(record, int(expert in resident[record.layer])) for record in _ordered(test) if record.phase == "decode" for expert in record.experts]
    return _outcome("static_train_frequency", capacity, events, manifest)


def simulate_lru(test: Iterable[RouteRecord], capacity: int, manifest: AtlasManifest) -> CacheOutcome:
    resident: dict[int, set[int]] = defaultdict(set)
    recency: dict[int, dict[int, int]] = defaultdict(dict)
    tick = 0
    events: list[tuple[RouteRecord, int]] = []
    for record in _ordered(test):
        layer_resident = resident[record.layer]
        required = set(record.experts)
        if record.phase == "decode":
            for expert in record.experts:
                events.append((record, int(expert in layer_resident)))
        for expert in record.experts:
            tick += 1
            recency[record.layer][expert] = tick
        if capacity <= 0:
            layer_resident.clear()
            continue
        layer_resident.update(required)
        while len(layer_resident) > capacity:
            victim = min(layer_resident, key=lambda item: (recency[record.layer].get(item, -1), item))
            layer_resident.remove(victim)
    return _outcome("lru", capacity, events, manifest)


def simulate_lfu(test: Iterable[RouteRecord], capacity: int, manifest: AtlasManifest) -> CacheOutcome:
    resident: dict[int, set[int]] = defaultdict(set)
    frequency: dict[int, Counter[int]] = defaultdict(Counter)
    recency: dict[int, dict[int, int]] = defaultdict(dict)
    tick = 0
    events: list[tuple[RouteRecord, int]] = []
    for record in _ordered(test):
        layer_resident = resident[record.layer]
        required = set(record.experts)
        if record.phase == "decode":
            for expert in record.experts:
                events.append((record, int(expert in layer_resident)))
        for expert in record.experts:
            tick += 1
            frequency[record.layer][expert] += 1
            recency[record.layer][expert] = tick
        if capacity <= 0:
            layer_resident.clear()
            continue
        layer_resident.update(required)
        while len(layer_resident) > capacity:
            victim = min(
                layer_resident,
                key=lambda item: (
                    frequency[record.layer][item],
                    recency[record.layer].get(item, -1),
                    item,
                ),
            )
            layer_resident.remove(victim)
    return _outcome("lfu", capacity, events, manifest)


def simulate_request_prefill_lfu(test: Iterable[RouteRecord], capacity: int, manifest: AtlasManifest) -> CacheOutcome:
    by_request: dict[tuple[str, str], list[RouteRecord]] = defaultdict(list)
    for record in _ordered(test):
        by_request[(record.run_id, record.request_id)].append(record)
    events: list[tuple[RouteRecord, int]] = []
    for values in by_request.values():
        prefill: dict[int, Counter[int]] = defaultdict(Counter)
        for record in values:
            if record.phase == "prefill":
                prefill[record.layer].update(record.experts)
        resident = {
            layer: {expert for expert, _ in prefill[layer].most_common(capacity)}
            for layer in range(manifest.n_layers)
        }
        for record in values:
            if record.phase != "decode":
                continue
            for expert in record.experts:
                events.append((record, int(expert in resident[record.layer])))
    return _outcome("request_prefill_lfu", capacity, events, manifest)


def simulate_request_adaptive_lfu(test: Iterable[RouteRecord], capacity: int, manifest: AtlasManifest) -> CacheOutcome:
    by_request: dict[tuple[str, str], list[RouteRecord]] = defaultdict(list)
    for record in _ordered(test):
        by_request[(record.run_id, record.request_id)].append(record)
    events: list[tuple[RouteRecord, int]] = []
    for values in by_request.values():
        frequency: dict[int, Counter[int]] = defaultdict(Counter)
        recency: dict[int, dict[int, int]] = defaultdict(dict)
        resident: dict[int, set[int]] = defaultdict(set)
        tick = 0
        for record in values:
            if record.phase != "prefill":
                continue
            for expert in record.experts:
                tick += 1
                frequency[record.layer][expert] += 1
                recency[record.layer][expert] = tick
        for layer in range(manifest.n_layers):
            resident[layer] = {expert for expert, _ in frequency[layer].most_common(capacity)}
        for record in values:
            if record.phase != "decode":
                continue
            layer_resident = resident[record.layer]
            for expert in record.experts:
                events.append((record, int(expert in layer_resident)))
            for expert in record.experts:
                tick += 1
                frequency[record.layer][expert] += 1
                recency[record.layer][expert] = tick
            if capacity <= 0:
                layer_resident.clear()
                continue
            layer_resident.update(record.experts)
            while len(layer_resident) > capacity:
                victim = min(
                    layer_resident,
                    key=lambda item: (
                        frequency[record.layer][item],
                        recency[record.layer].get(item, -1),
                        item,
                    ),
                )
                layer_resident.remove(victim)
    return _outcome("request_adaptive_lfu", capacity, events, manifest)


def simulate_belady(test: Iterable[RouteRecord], capacity: int, manifest: AtlasManifest) -> CacheOutcome:
    ordered = _ordered(test)
    by_layer: dict[int, list[RouteRecord]] = defaultdict(list)
    for record in ordered:
        by_layer[record.layer].append(record)

    event_order: dict[tuple[str, str, str, int, int], int] = {
        (record.run_id, record.request_id, record.phase, record.token, record.layer): index
        for index, record in enumerate(ordered)
    }
    all_events: list[tuple[int, RouteRecord, int]] = []
    for layer, routes in by_layer.items():
        future: dict[int, list[int]] = defaultdict(list)
        for route_index, record in enumerate(routes):
            for expert in record.experts:
                future[expert].append(route_index)
        pointers: Counter[int] = Counter()
        resident: set[int] = set()
        for route_index, record in enumerate(routes):
            if record.phase == "decode":
                for expert in record.experts:
                    all_events.append((event_order[(record.run_id, record.request_id, record.phase, record.token, record.layer)], record, int(expert in resident)))
            for expert in record.experts:
                pointers[expert] += 1
            if capacity <= 0:
                resident.clear()
                continue
            resident.update(record.experts)
            def next_use(item: int) -> int:
                positions = future[item]
                pointer = pointers[item]
                return positions[pointer] if pointer < len(positions) else 10**18
            if len(resident) > capacity:
                resident = set(sorted(resident, key=lambda item: (next_use(item), item))[:capacity])
    events = [(record, hit) for _, record, hit in sorted(all_events, key=lambda item: item[0])]
    return _outcome("belady_oracle", capacity, events, manifest)


def evaluate_caches(
    train: Iterable[RouteRecord], test: Iterable[RouteRecord], manifest: AtlasManifest
) -> list[dict]:
    results: list[dict] = []
    for capacity in manifest.cache_capacities:
        policies = [
            simulate_static(train, test, capacity, manifest),
            simulate_lru(test, capacity, manifest),
            simulate_lfu(test, capacity, manifest),
            simulate_request_prefill_lfu(test, capacity, manifest),
            simulate_request_adaptive_lfu(test, capacity, manifest),
            simulate_belady(test, capacity, manifest),
        ]
        results.extend(item.as_dict() for item in policies)
    return results
