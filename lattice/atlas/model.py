from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

SCHEMA_ROUTE = "lattice.route.v1"
SCHEMA_MANIFEST = "lattice.atlas.manifest.v1"
PHASES = {"prefill", "decode"}


class AtlasError(RuntimeError):
    """Expected, user-actionable atlas failure."""


def _as_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AtlasError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise AtlasError(f"{name} must be >= {minimum}")
    return value


def _as_float(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AtlasError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AtlasError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise AtlasError(f"{name} must be >= {minimum}")
    return result


@dataclass(frozen=True, slots=True)
class RouteRecord:
    run_id: str
    request_id: str
    phase: str
    token: int
    layer: int
    experts: tuple[int, ...]
    scores: tuple[float, ...]
    router_ns: int | None = None
    expert_ns: int | None = None

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        *,
        n_layers: int,
        n_experts: int,
        top_k: int,
    ) -> "RouteRecord":
        if value.get("schema") != SCHEMA_ROUTE:
            raise AtlasError(f"route schema must be {SCHEMA_ROUTE!r}")
        run_id = value.get("run_id")
        request_id = value.get("request_id")
        phase = value.get("phase")
        if not isinstance(run_id, str) or not run_id:
            raise AtlasError("run_id must be a non-empty string")
        if not isinstance(request_id, str) or not request_id:
            raise AtlasError("request_id must be a non-empty string")
        if phase not in PHASES:
            raise AtlasError(f"phase must be one of {sorted(PHASES)}")
        token = _as_int(value.get("token"), "token", minimum=0)
        layer = _as_int(value.get("layer"), "layer", minimum=0)
        if layer >= n_layers:
            raise AtlasError(f"layer {layer} is outside [0, {n_layers})")
        raw_experts = value.get("experts")
        raw_scores = value.get("scores")
        if not isinstance(raw_experts, list) or len(raw_experts) != top_k:
            raise AtlasError(f"experts must contain exactly top_k={top_k} entries")
        if not isinstance(raw_scores, list) or len(raw_scores) != top_k:
            raise AtlasError(f"scores must contain exactly top_k={top_k} entries")
        experts = tuple(_as_int(item, "expert", minimum=0) for item in raw_experts)
        if len(set(experts)) != len(experts):
            raise AtlasError("experts must be unique within a route")
        if any(item >= n_experts for item in experts):
            raise AtlasError(f"expert id must be inside [0, {n_experts})")
        scores = tuple(_as_float(item, "score") for item in raw_scores)
        router_ns = value.get("router_ns")
        expert_ns = value.get("expert_ns")
        if router_ns is not None:
            router_ns = _as_int(router_ns, "router_ns", minimum=0)
        if expert_ns is not None:
            expert_ns = _as_int(expert_ns, "expert_ns", minimum=0)
        return cls(
            run_id=run_id,
            request_id=request_id,
            phase=phase,
            token=token,
            layer=layer,
            experts=experts,
            scores=scores,
            router_ns=router_ns,
            expert_ns=expert_ns,
        )

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": SCHEMA_ROUTE,
            "run_id": self.run_id,
            "request_id": self.request_id,
            "phase": self.phase,
            "token": self.token,
            "layer": self.layer,
            "experts": list(self.experts),
            "scores": list(self.scores),
        }
        if self.router_ns is not None:
            result["router_ns"] = self.router_ns
        if self.expert_ns is not None:
            result["expert_ns"] = self.expert_ns
        return result


@dataclass(frozen=True, slots=True)
class TransferProfile:
    name: str
    bandwidth_gbps: float
    fixed_us_per_nonempty_layer: float
    overlap_fraction: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TransferProfile":
        name = value.get("name")
        if not isinstance(name, str) or not name:
            raise AtlasError("transfer profile name must be non-empty")
        bandwidth = _as_float(value.get("bandwidth_gbps"), "bandwidth_gbps", minimum=1e-9)
        fixed = _as_float(
            value.get("fixed_us_per_nonempty_layer", 0.0),
            "fixed_us_per_nonempty_layer",
            minimum=0.0,
        )
        overlap = _as_float(value.get("overlap_fraction", 0.0), "overlap_fraction", minimum=0.0)
        if overlap > 1.0:
            raise AtlasError("overlap_fraction must be <= 1")
        return cls(name, bandwidth, fixed, overlap)


@dataclass(frozen=True, slots=True)
class AtlasManifest:
    model: str
    model_revision: str
    checkpoint_sha256: str
    runtime: str
    runtime_revision: str
    workload_sha256: str
    n_layers: int
    n_experts: int
    top_k: int
    expert_bytes: int
    cache_capacities: tuple[int, ...]
    target_cache_capacity: int
    train_fraction: float
    transfer_profiles: tuple[TransferProfile, ...]
    prefetch_width: int
    gates: dict[str, float]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AtlasManifest":
        if value.get("schema") != SCHEMA_MANIFEST:
            raise AtlasError(f"manifest schema must be {SCHEMA_MANIFEST!r}")
        model = value.get("model")
        revision = value.get("model_revision")
        if not isinstance(model, str) or not model:
            raise AtlasError("model must be non-empty")
        if not isinstance(revision, str) or not revision:
            raise AtlasError("model_revision must be a pinned revision or digest")
        checkpoint_sha256 = value.get("checkpoint_sha256")
        runtime = value.get("runtime")
        runtime_revision = value.get("runtime_revision")
        workload_sha256 = value.get("workload_sha256")
        for field_name, field_value in (
            ("checkpoint_sha256", checkpoint_sha256),
            ("runtime", runtime),
            ("runtime_revision", runtime_revision),
            ("workload_sha256", workload_sha256),
        ):
            if not isinstance(field_value, str) or not field_value:
                raise AtlasError(f"{field_name} must be a non-empty pinned identity")
        n_layers = _as_int(value.get("n_layers"), "n_layers", minimum=1)
        n_experts = _as_int(value.get("n_experts"), "n_experts", minimum=2)
        top_k = _as_int(value.get("top_k"), "top_k", minimum=1)
        if top_k > n_experts:
            raise AtlasError("top_k cannot exceed n_experts")
        expert_bytes = _as_int(value.get("expert_bytes"), "expert_bytes", minimum=1)
        raw_caps = value.get("cache_capacities")
        if not isinstance(raw_caps, list) or not raw_caps:
            raise AtlasError("cache_capacities must be a non-empty list")
        caps = tuple(sorted(set(_as_int(item, "cache capacity", minimum=0) for item in raw_caps)))
        if any(item > n_experts for item in caps):
            raise AtlasError("cache capacity cannot exceed n_experts")
        target_cache_capacity = _as_int(value.get("target_cache_capacity"), "target_cache_capacity", minimum=0)
        if target_cache_capacity not in caps:
            raise AtlasError("target_cache_capacity must be listed in cache_capacities")
        train_fraction = _as_float(value.get("train_fraction", 0.7), "train_fraction", minimum=0.0)
        if not 0.0 < train_fraction < 1.0:
            raise AtlasError("train_fraction must be strictly between 0 and 1")
        raw_profiles = value.get("transfer_profiles")
        if not isinstance(raw_profiles, list) or not raw_profiles:
            raise AtlasError("transfer_profiles must be a non-empty list")
        profiles = tuple(TransferProfile.from_dict(item) for item in raw_profiles)
        if len({item.name for item in profiles}) != len(profiles):
            raise AtlasError("transfer profile names must be unique")
        prefetch_width = _as_int(value.get("prefetch_width", top_k), "prefetch_width", minimum=1)
        if prefetch_width > n_experts:
            raise AtlasError("prefetch_width cannot exceed n_experts")
        raw_gates = value.get("gates", {})
        if not isinstance(raw_gates, dict):
            raise AtlasError("gates must be an object")
        gates = {str(key): _as_float(item, f"gate {key}") for key, item in raw_gates.items()}
        return cls(
            model=model,
            model_revision=revision,
            checkpoint_sha256=checkpoint_sha256,
            runtime=runtime,
            runtime_revision=runtime_revision,
            workload_sha256=workload_sha256,
            n_layers=n_layers,
            n_experts=n_experts,
            top_k=top_k,
            expert_bytes=expert_bytes,
            cache_capacities=caps,
            target_cache_capacity=target_cache_capacity,
            train_fraction=train_fraction,
            transfer_profiles=profiles,
            prefetch_width=prefetch_width,
            gates=gates,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_MANIFEST,
            "model": self.model,
            "model_revision": self.model_revision,
            "checkpoint_sha256": self.checkpoint_sha256,
            "runtime": self.runtime,
            "runtime_revision": self.runtime_revision,
            "workload_sha256": self.workload_sha256,
            "n_layers": self.n_layers,
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "expert_bytes": self.expert_bytes,
            "cache_capacities": list(self.cache_capacities),
            "target_cache_capacity": self.target_cache_capacity,
            "train_fraction": self.train_fraction,
            "transfer_profiles": [
                {
                    "name": item.name,
                    "bandwidth_gbps": item.bandwidth_gbps,
                    "fixed_us_per_nonempty_layer": item.fixed_us_per_nonempty_layer,
                    "overlap_fraction": item.overlap_fraction,
                }
                for item in self.transfer_profiles
            ],
            "prefetch_width": self.prefetch_width,
            "gates": dict(sorted(self.gates.items())),
        }


def validate_complete_routes(records: Iterable[RouteRecord], manifest: AtlasManifest) -> list[RouteRecord]:
    ordered = sorted(records, key=lambda r: (r.run_id, r.request_id, r.phase, r.token, r.layer))
    seen: set[tuple[str, str, str, int, int]] = set()
    per_token: dict[tuple[str, str, str, int], set[int]] = {}
    for record in ordered:
        key = (record.run_id, record.request_id, record.phase, record.token, record.layer)
        if key in seen:
            raise AtlasError(f"duplicate route row: {key}")
        seen.add(key)
        per_token.setdefault(key[:-1], set()).add(record.layer)
    expected = set(range(manifest.n_layers))
    incomplete = [key for key, layers in per_token.items() if layers != expected]
    if incomplete:
        preview = ", ".join(map(str, incomplete[:3]))
        raise AtlasError(f"incomplete token routes; every token must include all layers: {preview}")
    if not ordered:
        raise AtlasError("trace contains no route records")
    return ordered
