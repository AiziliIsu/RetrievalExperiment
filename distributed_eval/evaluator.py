from __future__ import annotations

import statistics
import time
from typing import Dict, List, Optional

import numpy as np

from .normalizer import TextNormalizer
from .result_writer import ResultWriter


def _first_correct_rank(retrieved_source_ids: List[str], expected_id: str) -> Optional[int]:
    for idx, source_id in enumerate(retrieved_source_ids, start=1):
        if str(source_id) == str(expected_id):
            return idx
    return None


def _dcg(relevances: List[int]) -> float:
    return float(sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances)))


def _ndcg_at_k(retrieved_source_ids: List[str], expected_id: str, k: int) -> float:
    rel = [1 if str(x) == str(expected_id) else 0 for x in retrieved_source_ids[:k]]
    dcg = _dcg(rel)
    idcg = _dcg([1] + [0] * (k - 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def _rank_of_first_match(items: List[Dict], expected_id: str) -> Optional[int]:
    for idx, item in enumerate(items, start=1):
        if str(item.get("source_id", "")) == str(expected_id):
            return idx
    return None


class Evaluator:
    def __init__(
        self,
        embedder,
        qdrant,
        collection_name: str,
        top_k: int,
        normalizer: TextNormalizer,
        writer: ResultWriter,
        model_name: str,
        language: str,
        retrieval_pipeline=None,
    ):
        self.embedder = embedder
        self.qdrant = qdrant
        self.collection_name = collection_name
        self.top_k = int(top_k)
        self.normalizer = normalizer
        self.writer = writer
        self.model_name = model_name
        self.language = language
        self.retrieval_pipeline = retrieval_pipeline

    @staticmethod
    def _is_cuda_oom(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "out of memory" in msg and "cuda" in msg

    @staticmethod
    def _clear_cuda_cache() -> None:
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def evaluate(self, questions: List[Dict], resume: bool = True) -> Dict:
        if resume:
            completed, prior = self.writer.load_completed()
        else:
            completed, prior = set(), {
                "evaluated": 0,
                "hit1": 0, "hit3": 0, "hit5": 0, "hit10": 0,
                "sum_reciprocal_rank": 0.0, "sum_ndcg": 0.0, "latencies": [],
            }

        # Pre-populate running totals from checkpoint so resumed runs are correct.
        hit1: int = prior["hit1"]
        hit3: int = prior["hit3"]
        hit5: int = prior["hit5"]
        hit10: int = prior["hit10"]
        sum_reciprocal_rank: float = prior["sum_reciprocal_rank"]
        sum_ndcg: float = prior["sum_ndcg"]
        latencies: List[float] = list(prior["latencies"])  # kept for median
        evaluated: int = prior["evaluated"]
        failed: int = 0

        for q in questions:
            qid = str(q.get("id_question", "")).strip()
            if resume and qid in completed:
                continue

            expected_id = str(q.get("id", "")).strip()
            query_text = self.normalizer.normalize(str(q.get("question", "")))
            if not query_text or not expected_id or not qid:
                failed += 1
                continue

            started = time.perf_counter()
            try:
                retrieval_debug = {}
                for attempt in range(2):
                    try:
                        query_vector = self.embedder.embed_query(query_text)
                        if self.retrieval_pipeline is not None:
                            hits, retrieval_debug = self.retrieval_pipeline.retrieve(query_text, query_vector)
                        else:
                            hits = self.qdrant.search(self.collection_name, query_vector, self.top_k)
                        break
                    except Exception as exc:
                        if self._is_cuda_oom(exc) and attempt == 0:
                            self._clear_cuda_cache()
                            continue
                        raise
            except Exception as exc:
                failed += 1
                if failed <= 5:
                    print(
                        f"[Evaluator] Failed question_id={qid} "
                        f"error={type(exc).__name__}: {exc}"
                    )
                continue
            latency_ms = (time.perf_counter() - started) * 1000.0

            retrieved_source_ids = [str(h["source_id"]) for h in hits]
            rank = _first_correct_rank(retrieved_source_ids, expected_id)
            q_hit1 = int(rank == 1)
            q_hit3 = int(rank is not None and rank <= 3)
            q_hit5 = int(rank is not None and rank <= 5)
            q_hit10 = int(rank is not None and rank <= 10)
            q_rr = 0.0 if rank is None else 1.0 / rank
            q_ndcg = _ndcg_at_k(retrieved_source_ids, expected_id, min(self.top_k, 5))

            initial_hits = list((retrieval_debug or {}).get("initial_hits") or [])
            reranked_hits = list((retrieval_debug or {}).get("reranked_hits") or [])
            dense_rank = _rank_of_first_match(initial_hits, expected_id) if initial_hits else None
            rerank_rank = _rank_of_first_match(reranked_hits, expected_id) if reranked_hits else None

            hit1 += q_hit1
            hit3 += q_hit3
            hit5 += q_hit5
            hit10 += q_hit10
            sum_reciprocal_rank += q_rr
            sum_ndcg += q_ndcg
            latencies.append(latency_ms)
            evaluated += 1

            self.writer.mark_question_completed(qid, {
                "reciprocal_rank": q_rr,
                "ndcg": q_ndcg,
                "hit1": q_hit1,
                "hit3": q_hit3,
                "hit5": q_hit5,
                "hit10": q_hit10,
                "latency_ms": latency_ms,
            })

        summary = {
            "model_name": self.model_name,
            "language": self.language,
            "collection_name": self.collection_name,
            "total_questions": len(questions),
            "evaluated_questions": evaluated,
            "failed_questions": failed,
            "recall@1": hit1 / evaluated if evaluated else 0.0,
            "recall@3": hit3 / evaluated if evaluated else 0.0,
            "recall@5": hit5 / evaluated if evaluated else 0.0,
            "recall@10": hit10 / evaluated if evaluated else 0.0,
            "mrr": sum_reciprocal_rank / evaluated if evaluated else 0.0,
            "ndcg": sum_ndcg / evaluated if evaluated else 0.0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "median_latency_ms": float(statistics.median(latencies)) if latencies else 0.0,
        }
        self.writer.save_summary(summary)
        return summary
