"""Small pytrec_eval compatibility layer for HELMET's fixed re-ranking metrics."""

from __future__ import annotations

import math
import sys
import types
from typing import Any


def _cutoffs(measures: set[str], prefix: str) -> list[int]:
    values: set[int] = set()
    for measure in measures:
        if measure.startswith(prefix + "."):
            values.update(int(value) for value in measure.split(".", 1)[1].split(","))
    return sorted(values)


def _average_precision(ranked_relevance: list[int], relevant_count: int, cutoff: int) -> float:
    if relevant_count <= 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, relevance in enumerate(ranked_relevance[:cutoff], start=1):
        if relevance > 0:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(relevant_count, cutoff)


def _ndcg(ranked_relevance: list[int], ideal_relevance: list[int], cutoff: int) -> float:
    def dcg(values: list[int]) -> float:
        return sum(
            (2**relevance - 1) / math.log2(rank + 1)
            for rank, relevance in enumerate(values[:cutoff], start=1)
        )

    ideal = dcg(ideal_relevance)
    return 0.0 if ideal == 0 else dcg(ranked_relevance) / ideal


class RelevanceEvaluator:
    """Implement the subset of pytrec_eval.RelevanceEvaluator used by HELMET."""

    def __init__(self, qrels: dict[str, dict[str, int]], measures: set[str]) -> None:
        self.qrels = qrels
        self.measures = set(measures)

    def evaluate(self, results: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        evaluated: dict[str, dict[str, float]] = {}
        ndcg_cutoffs = _cutoffs(self.measures, "ndcg_cut")
        map_cutoffs = _cutoffs(self.measures, "map_cut")
        recall_cutoffs = _cutoffs(self.measures, "recall")
        precision_cutoffs = _cutoffs(self.measures, "P")
        for query_id, qrel in self.qrels.items():
            ranked = sorted(
                results.get(query_id, {}).items(), key=lambda item: item[1], reverse=True
            )
            ranked_relevance = [int(qrel.get(document_id, 0)) for document_id, _ in ranked]
            ideal_relevance = sorted((int(value) for value in qrel.values()), reverse=True)
            relevant_count = sum(value > 0 for value in qrel.values())
            metrics: dict[str, float] = {}
            for cutoff in ndcg_cutoffs:
                metrics[f"ndcg_cut_{cutoff}"] = _ndcg(
                    ranked_relevance, ideal_relevance, cutoff
                )
            for cutoff in map_cutoffs:
                metrics[f"map_cut_{cutoff}"] = _average_precision(
                    ranked_relevance, relevant_count, cutoff
                )
            for cutoff in recall_cutoffs:
                hits = sum(value > 0 for value in ranked_relevance[:cutoff])
                metrics[f"recall_{cutoff}"] = (
                    0.0 if relevant_count == 0 else hits / relevant_count
                )
            for cutoff in precision_cutoffs:
                hits = sum(value > 0 for value in ranked_relevance[:cutoff])
                metrics[f"P_{cutoff}"] = hits / cutoff
            metrics["recip_rank"] = next(
                (
                    1.0 / rank
                    for rank, relevance in enumerate(ranked_relevance, start=1)
                    if relevance > 0
                ),
                0.0,
            )
            evaluated[query_id] = metrics
        return evaluated


def install_if_missing() -> None:
    """Install the compatibility module only when real pytrec_eval is unavailable."""

    try:
        __import__("pytrec_eval")
        return
    except ModuleNotFoundError:
        pass
    module = types.ModuleType("pytrec_eval")
    module.RelevanceEvaluator = RelevanceEvaluator  # type: ignore[attr-defined]
    sys.modules["pytrec_eval"] = module


def compatibility_metadata() -> dict[str, Any]:
    return {
        "implementation": "conceptlm_helmet_trec_eval_compat",
        "supported_metrics": ["ndcg_cut", "map_cut", "recall", "P", "recip_rank"],
    }
