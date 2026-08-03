from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from .model import AtlasManifest, RouteRecord


@dataclass(slots=True)
class PrefetchScore:
    predictor: str
    targets: int = 0
    true_positive: int = 0
    predicted: int = 0
    actual: int = 0

    @property
    def recall(self) -> float:
        return self.true_positive / self.actual if self.actual else 0.0

    @property
    def precision(self) -> float:
        return self.true_positive / self.predicted if self.predicted else 0.0

    @property
    def overfetch_fraction(self) -> float:
        return (self.predicted - self.true_positive) / self.predicted if self.predicted else 0.0

    def as_dict(self) -> dict:
        return {
            "predictor": self.predictor,
            "targets": self.targets,
            "true_positive": self.true_positive,
            "predicted": self.predicted,
            "actual": self.actual,
            "recall": self.recall,
            "precision": self.precision,
            "overfetch_fraction": self.overfetch_fraction,
        }


def _group(records: Iterable[RouteRecord]) -> dict[tuple[str, str, str, int], dict[int, RouteRecord]]:
    result: dict[tuple[str, str, str, int], dict[int, RouteRecord]] = defaultdict(dict)
    for record in records:
        result[(record.run_id, record.request_id, record.phase, record.token)][record.layer] = record
    return result


def _score(name: str, pairs: Iterable[tuple[set[int], set[int]]]) -> dict:
    score = PrefetchScore(name)
    for predicted, actual in pairs:
        score.targets += 1
        score.predicted += len(predicted)
        score.actual += len(actual)
        score.true_positive += len(predicted & actual)
    return score.as_dict()


def evaluate_prefetch(train: Iterable[RouteRecord], test: Iterable[RouteRecord], manifest: AtlasManifest) -> list[dict]:
    train_groups = _group(train)
    test_groups = _group(test)
    popularity: dict[int, Counter[int]] = defaultdict(Counter)
    transitions: dict[tuple[int, tuple[int, ...]], Counter[int]] = defaultdict(Counter)
    expert_votes: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    for layers in train_groups.values():
        for layer, record in layers.items():
            popularity[layer].update(record.experts)
            next_record = layers.get(layer + 1)
            if next_record is not None:
                transitions[(layer, tuple(sorted(record.experts)))].update(next_record.experts)
                for current_expert in record.experts:
                    expert_votes[(layer, current_expert)].update(next_record.experts)

    width = manifest.prefetch_width
    cross_layer_pairs: list[tuple[set[int], set[int]]] = []
    expert_vote_pairs: list[tuple[set[int], set[int]]] = []
    popularity_pairs: list[tuple[set[int], set[int]]] = []
    oracle_pairs: list[tuple[set[int], set[int]]] = []
    for group_key, layers in test_groups.items():
        if group_key[2] != "decode":
            continue
        for layer in range(manifest.n_layers - 1):
            current = layers.get(layer)
            target = layers.get(layer + 1)
            if current is None or target is None:
                continue
            fallback = popularity[layer + 1].most_common(width)
            predicted_counts = transitions.get((layer, tuple(sorted(current.experts))))
            choices = predicted_counts.most_common(width) if predicted_counts else fallback
            predicted = {expert for expert, _ in choices}
            vote_counts: Counter[int] = Counter()
            for current_expert in current.experts:
                vote_counts.update(expert_votes.get((layer, current_expert), Counter()))
            vote_choices = vote_counts.most_common(width) if vote_counts else fallback
            actual = set(target.experts)
            cross_layer_pairs.append((predicted, actual))
            expert_vote_pairs.append(({expert for expert, _ in vote_choices}, actual))
            popularity_pairs.append(({expert for expert, _ in fallback}, actual))
            oracle_pairs.append((set(list(actual)[:width]), actual))

    last_token_pairs: list[tuple[set[int], set[int]]] = []
    by_request_layer: dict[tuple[str, str, str, int], list[RouteRecord]] = defaultdict(list)
    for record in test:
        if record.phase == "decode":
            by_request_layer[(record.run_id, record.request_id, record.phase, record.layer)].append(record)
    for values in by_request_layer.values():
        values.sort(key=lambda item: item.token)
        previous: RouteRecord | None = None
        for record in values:
            if previous is not None:
                predicted = set(previous.experts[:width])
                last_token_pairs.append((predicted, set(record.experts)))
            previous = record

    return [
        _score("cross_layer_signature", cross_layer_pairs),
        _score("cross_layer_expert_vote", expert_vote_pairs),
        _score("target_layer_popularity", popularity_pairs),
        _score("last_token_same_layer", last_token_pairs),
        _score("oracle", oracle_pairs),
    ]
