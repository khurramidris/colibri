from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from lattice.atlas.cache import evaluate_caches, simulate_belady, simulate_lru
from lattice.atlas.decision import evaluate_decision
from lattice.atlas.io import atomic_write_json, load_routes, write_jsonl
from lattice.atlas.metrics import route_metrics
from lattice.atlas.model import AtlasError, AtlasManifest, RouteRecord
from lattice.atlas.prefetch import evaluate_prefetch
from lattice.atlas.split import stable_request_split
from lattice.atlas.synthetic import generate_routes
from lattice.atlas.transfer import evaluate_transfer


def manifest() -> AtlasManifest:
    return AtlasManifest.from_dict({
        "schema": "lattice.atlas.manifest.v1",
        "model": "synthetic",
        "model_revision": "sha256:test",
        "checkpoint_sha256": "sha256:checkpoint",
        "runtime": "synthetic",
        "runtime_revision": "sha256:runtime",
        "workload_sha256": "sha256:workload",
        "n_layers": 4,
        "n_experts": 16,
        "top_k": 2,
        "expert_bytes": 1024,
        "cache_capacities": [0, 2, 4, 8],
        "target_cache_capacity": 4,
        "train_fraction": 0.5,
        "prefetch_width": 2,
        "transfer_profiles": [{
            "name": "test",
            "bandwidth_gbps": 1.0,
            "fixed_us_per_nonempty_layer": 0.0,
            "overlap_fraction": 0.0
        }],
        "gates": {
            "cache_practical_hit_rate": 0.2,
            "cache_oracle_hit_rate": 0.2,
            "transfer_reduction": 0.1,
            "prefetch_recall": 0.1,
            "prefetch_max_overfetch": 1.0
        }
    })


class AtlasTests(unittest.TestCase):
    def setUp(self):
        self.manifest = manifest()
        self.records = list(generate_routes(
            seed=4, requests=8, tokens=12,
            n_layers=4, n_experts=16, top_k=2,
            hot_experts=4, hot_probability=0.95,
        ))

    def test_trace_round_trip_and_completeness(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            write_jsonl(path, (record.as_dict() for record in self.records))
            loaded = load_routes(path, self.manifest)
            self.assertEqual(len(loaded), len(self.records))

    def test_incomplete_token_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            write_jsonl(path, (record.as_dict() for record in self.records[:-1]))
            with self.assertRaises(AtlasError):
                load_routes(path, self.manifest)

    def test_request_split_has_no_identity_leakage(self):
        train, test, assignment = stable_request_split(self.records, 0.5)
        train_ids = {(r.run_id, r.request_id) for r in train}
        test_ids = {(r.run_id, r.request_id) for r in test}
        self.assertFalse(train_ids & test_ids)
        self.assertEqual(len(assignment), 8)

    def test_belady_is_not_worse_than_lru(self):
        _, test, _ = stable_request_split(self.records, 0.5)
        for capacity in self.manifest.cache_capacities:
            oracle = simulate_belady(test, capacity, self.manifest)
            lru = simulate_lru(test, capacity, self.manifest)
            self.assertGreaterEqual(oracle.hit_rate + 1e-12, lru.hit_rate)

    def test_end_to_end_analysis_objects(self):
        train, test, _ = stable_request_split(self.records, 0.5)
        metrics = route_metrics(self.records, self.manifest)
        caches = evaluate_caches(train, test, self.manifest)
        prefetch = evaluate_prefetch(train, test, self.manifest)
        transfer = evaluate_transfer(caches, self.manifest.transfer_profiles)
        decision = evaluate_decision(
            caches, transfer, prefetch, None, self.manifest.gates,
            target_capacity=self.manifest.target_cache_capacity,
        )
        self.assertEqual(metrics["requests"], 8)
        self.assertTrue(caches)
        self.assertTrue(prefetch)
        self.assertIn(decision["cache_prefetch"]["decision"], {"go", "no-go"})

    def test_manifest_rejects_unpinned_empty_revision(self):
        raw = self.manifest.as_dict()
        raw["model_revision"] = ""
        with self.assertRaises(AtlasError):
            AtlasManifest.from_dict(raw)

    def test_transfer_includes_zero_miss_tokens(self):
        train, test, _ = stable_request_split(self.records, 0.5)
        caches = evaluate_caches(train, test, self.manifest)
        transfer = evaluate_transfer(caches, self.manifest.transfer_profiles)
        full_cache = next(
            row for row in transfer
            if row["capacity"] == 8 and row["policy"] == "belady_oracle"
        )
        decode_tokens = len({
            (r.run_id, r.request_id, r.token)
            for r in test if r.phase == "decode"
        })
        self.assertEqual(full_cache["tokens_evaluated"], decode_tokens)
        self.assertLessEqual(full_cache["tokens_with_misses"], decode_tokens)

    def test_decision_is_bound_to_target_capacity(self):
        train, test, _ = stable_request_split(self.records, 0.5)
        caches = evaluate_caches(train, test, self.manifest)
        transfer = evaluate_transfer(caches, self.manifest.transfer_profiles)
        prefetch = evaluate_prefetch(train, test, self.manifest)
        decision = evaluate_decision(
            caches, transfer, prefetch, None, self.manifest.gates,
            target_capacity=2,
        )
        self.assertEqual(decision["target_cache_capacity"], 2)
        self.assertEqual(decision["cache_prefetch"]["best_practical_cache"]["capacity"], 2)
        self.assertEqual(decision["cache_prefetch"]["best_oracle_cache"]["capacity"], 2)

    def test_write_once_json_rejects_different_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            atomic_write_json(path, {"a": 1})
            atomic_write_json(path, {"a": 1})
            with self.assertRaises(AtlasError):
                atomic_write_json(path, {"a": 2})

    def test_prefetch_evaluation_has_non_oracle_baselines(self):
        train, test, _ = stable_request_split(self.records, 0.5)
        names = {row["predictor"] for row in evaluate_prefetch(train, test, self.manifest)}
        self.assertTrue({
            "cross_layer_signature",
            "cross_layer_expert_vote",
            "target_layer_popularity",
            "last_token_same_layer",
            "oracle",
        }.issubset(names))


if __name__ == "__main__":
    unittest.main()
