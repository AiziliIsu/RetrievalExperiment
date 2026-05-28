from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _tokenize_for_bm25(text: str) -> List[str]:
    # Simple multilingual tokenizer with optional stemming for RU/EN.
    tokens = re.findall(r"[\w']+", (text or "").lower(), flags=re.UNICODE)
    if not tokens:
        return []
    try:
        from nltk.stem.snowball import SnowballStemmer

        stemmer_ru = SnowballStemmer("russian")
        stemmer_en = SnowballStemmer("english")
        stemmed = []
        for tok in tokens:
            if re.search(r"[а-яё]", tok):
                stemmed.append(stemmer_ru.stem(tok))
            elif re.search(r"[a-z]", tok):
                stemmed.append(stemmer_en.stem(tok))
            else:
                stemmed.append(tok)
        return stemmed
    except Exception:
        return tokens


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for ranked_ids in rankings:
        for rank, item_id in enumerate(ranked_ids, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores


def weighted_sum_fusion(
    dense_scores: Dict[str, float],
    sparse_scores: Dict[str, float],
    dense_weight: float,
    sparse_weight: float,
) -> Dict[str, float]:
    def _minmax(values: Dict[str, float]) -> Dict[str, float]:
        if not values:
            return {}
        mn = min(values.values())
        mx = max(values.values())
        if mx <= mn:
            return {k: 1.0 for k in values.keys()}
        return {k: (v - mn) / (mx - mn) for k, v in values.items()}

    dn = _minmax(dense_scores)
    sn = _minmax(sparse_scores)
    keys = set(dn.keys()) | set(sn.keys())
    return {k: dense_weight * dn.get(k, 0.0) + sparse_weight * sn.get(k, 0.0) for k in keys}


@dataclass
class RetrievalOptions:
    strategy: str = "dense"  # dense | hybrid
    top_k: int = 10
    candidate_k: int = 50
    sparse_strategy: str = "bm25"  # bm25 | bge_m3_sparse
    fusion: str = "rrf"  # rrf | weighted
    dense_weight: float = 0.6
    sparse_weight: float = 0.4


@dataclass
class RerankOptions:
    enabled: bool = False
    model: str = "bge-reranker-v2-m3"
    top_k: int = 10


@dataclass
class WindowOptions:
    enabled: bool = False
    backward: int = 1
    forward: int = 1


class BM25SparseRetriever:
    def __init__(self, cache_rows: Iterable[Dict]):
        try:
            from rank_bm25 import BM25Okapi
        except Exception as exc:
            raise ImportError("BM25 sparse retrieval requires rank-bm25 package.") from exc

        self._rows: List[Dict] = []
        corpus = []
        for row in cache_rows:
            payload = dict(row.get("payload") or {})
            text = str(row.get("text", ""))
            chunk_id = str(payload.get("chunk_id", ""))
            source_id = str(payload.get("id", ""))
            if not chunk_id:
                continue
            self._rows.append(
                {
                    "chunk_id": chunk_id,
                    "source_id": source_id,
                    "text": text,
                    "payload": payload,
                }
            )
            corpus.append(_tokenize_for_bm25(text))

        self._bm25 = BM25Okapi(corpus)

    def search(self, query_text: str, limit: int) -> List[Dict]:
        if not self._rows:
            return []
        q_tokens = _tokenize_for_bm25(query_text)
        if not q_tokens:
            return []
        scores = np.asarray(self._bm25.get_scores(q_tokens), dtype=np.float32)
        top_idx = np.argsort(-scores)[:limit]
        out = []
        for idx in top_idx:
            row = self._rows[int(idx)]
            out.append(
                {
                    "chunk_id": row["chunk_id"],
                    "source_id": row["source_id"],
                    "payload": row["payload"],
                    "score": float(scores[int(idx)]),
                }
            )
        return out


class BgeM3SparseCandidateScorer:
    """
    Candidate-level neural sparse scorer using BGE-M3 lexical weights.

    This scorer runs on candidates returned by dense retrieval to avoid a full
    sparse re-indexing step.
    """

    def __init__(self):
        try:
            from FlagEmbedding import BGEM3FlagModel
        except Exception as exc:
            raise ImportError(
                "BGE-M3 sparse strategy requires FlagEmbedding package. "
                "Install with: pip install FlagEmbedding"
            ) from exc
        # Device policy:
        # - auto (default): use CUDA when available
        # - cpu: force sparse scorer on CPU (recommended for 4GB VRAM laptops)
        # - cuda: force sparse scorer on CUDA
        device_pref = os.getenv("BGE_M3_SPARSE_DEVICE", "auto").strip().lower()
        self._device_pref = device_pref
        use_fp16 = False
        devices = None
        try:
            import torch

            has_cuda = bool(torch.cuda.is_available())
            if device_pref == "cpu":
                devices = "cpu"
                use_fp16 = False
            elif device_pref == "cuda":
                devices = "cuda"
                use_fp16 = has_cuda
            else:
                if has_cuda:
                    devices = "cuda"
                    use_fp16 = True
                else:
                    devices = "cpu"
                    use_fp16 = False
        except Exception:
            devices = "cpu"
            use_fp16 = False
        self._model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=use_fp16, devices=devices)

    def _encode_sparse(self, text: str) -> Dict[int, float]:
        result = self._model.encode([text], return_sparse=True)
        sparse_vecs = result.get("lexical_weights") or []
        if not sparse_vecs:
            return {}
        vec = sparse_vecs[0]
        return {int(k): float(v) for k, v in vec.items()}

    @staticmethod
    def _dot(a: Dict[int, float], b: Dict[int, float]) -> float:
        if len(a) > len(b):
            a, b = b, a
        return float(sum(v * b.get(k, 0.0) for k, v in a.items()))

    def score_candidates(self, query_text: str, candidates: List[Dict]) -> Dict[str, float]:
        q = self._encode_sparse(query_text)
        out: Dict[str, float] = {}
        for hit in candidates:
            chunk_id = str(hit.get("chunk_id", ""))
            text = str((hit.get("payload") or {}).get("text", ""))
            if not chunk_id or not text:
                continue
            out[chunk_id] = self._dot(q, self._encode_sparse(text))
        return out


class SentenceTransformersReranker:
    MODEL_MAP = {
        "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
        "bge-reranker-v2-minicpm-28": "BAAI/bge-reranker-v2-minicpm-28",
        "qwen3-reranker-0.6b": "Qwen/Qwen3-Reranker-0.6B",
        "jina-reranker-v2": "jinaai/jina-reranker-v2-base-multilingual",
    }

    def __init__(self, name: str):
        from sentence_transformers import CrossEncoder

        model_name = self.MODEL_MAP.get(name.lower(), name)
        device_pref = os.getenv("BGE_RERANKER_DEVICE", "auto").strip().lower()
        device = None
        model_kwargs = {}
        trust_remote_code = False
        try:
            import torch

            if device_pref == "cpu":
                device = "cpu"
            elif device_pref == "cuda":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            # Fit reranker on low-VRAM GPUs when using CUDA.
            if device == "cuda":
                model_kwargs["torch_dtype"] = torch.float16
        except Exception:
            device = "cpu"
        
        # Jina reranker requires trust_remote_code
        if "jina" in name.lower():
            trust_remote_code = True
        
        self._model = CrossEncoder(model_name, device=device, model_kwargs=model_kwargs, trust_remote_code=trust_remote_code)

    def rerank(self, query_text: str, candidates: List[Dict]) -> List[Dict]:
        if not candidates:
            return []
        pairs = [(query_text, str(c.get("rerank_text", ""))) for c in candidates]
        scores = self._model.predict(pairs)
        out = []
        for c, s in zip(candidates, scores):
            row = dict(c)
            row["rerank_score"] = float(s)
            out.append(row)
        out.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return out


def create_reranker(model_name: str):
    low = model_name.strip().lower()
    if low in {"bge-reranker-v2-m3", "bge-reranker-v2-minicpm-28", "qwen3-reranker-0.6b", "jina-reranker-v2"}:
        return SentenceTransformersReranker(low)
    raise ValueError(
        "Unsupported reranker model. Use one of: "
        "bge-reranker-v2-m3, bge-reranker-v2-miniCPM-28, qwen3-reranker-0.6b, jina-reranker-v2"
    )


class RetrievalPipeline:
    def __init__(
        self,
        qdrant,
        collection_name: str,
        embedder,
        retrieval: RetrievalOptions,
        rerank: RerankOptions,
        window: WindowOptions,
        sparse_cache_rows: Optional[Iterable[Dict]] = None,
    ):
        self.qdrant = qdrant
        self.collection_name = collection_name
        self.embedder = embedder
        self.retrieval = retrieval
        self.rerank = rerank
        self.window = window
        self._bm25 = None
        self._bge_sparse = None
        self._reranker = create_reranker(rerank.model) if rerank.enabled else None
        self._sparse_fallback = False
        self._sparse_fallback_reason = None

        if self.retrieval.strategy == "hybrid" and self.retrieval.sparse_strategy == "bm25":
            if sparse_cache_rows is None:
                raise ValueError("BM25 hybrid retrieval requires sparse_cache_rows from processed cache.")
            self._bm25 = BM25SparseRetriever(sparse_cache_rows)
        if self.retrieval.strategy == "hybrid" and self.retrieval.sparse_strategy == "bge_m3_sparse":
            self._bge_sparse = BgeM3SparseCandidateScorer()

    def _dense_hits(self, query_vector: np.ndarray, limit: int) -> List[Dict]:
        return self.qdrant.search(self.collection_name, query_vector, limit)

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

    def _hybrid_hits(self, query_text: str, query_vector: np.ndarray) -> List[Dict]:
        self._sparse_fallback = False
        self._sparse_fallback_reason = None
        dense_hits = self._dense_hits(query_vector, self.retrieval.candidate_k)
        dense_by_id: Dict[str, Dict] = {str(h.get("chunk_id", "")): h for h in dense_hits if h.get("chunk_id")}

        if self.retrieval.sparse_strategy == "bm25":
            sparse_hits = self._bm25.search(query_text, self.retrieval.candidate_k) if self._bm25 else []
            sparse_scores = {str(h["chunk_id"]): float(h.get("score", 0.0)) for h in sparse_hits}
        else:
            if self._bge_sparse:
                sparse_scores = {}
                for attempt in range(2):
                    try:
                        sparse_scores = self._bge_sparse.score_candidates(query_text, dense_hits)
                        break
                    except Exception as exc:
                        if self._is_cuda_oom(exc) and attempt == 0:
                            # Retry once after clearing CUDA cache.
                            self._clear_cuda_cache()
                            continue
                        if self._is_cuda_oom(exc):
                            # Keep evaluation running with dense-only for this query.
                            self._clear_cuda_cache()
                            sparse_scores = {}
                            self._sparse_fallback = True
                            self._sparse_fallback_reason = f"{type(exc).__name__}: {exc}"
                            break
                        raise
            else:
                sparse_scores = {}

        dense_scores = {str(h["chunk_id"]): float(h.get("score", 0.0)) for h in dense_hits}

        if self.retrieval.fusion == "rrf":
            dense_ranked = [str(h["chunk_id"]) for h in dense_hits if h.get("chunk_id")]
            sparse_ranked = [cid for cid, _ in sorted(sparse_scores.items(), key=lambda x: x[1], reverse=True)]
            fused = reciprocal_rank_fusion([dense_ranked, sparse_ranked])
        else:
            fused = weighted_sum_fusion(
                dense_scores=dense_scores,
                sparse_scores=sparse_scores,
                dense_weight=self.retrieval.dense_weight,
                sparse_weight=self.retrieval.sparse_weight,
            )

        merged: Dict[str, Dict] = {}
        for cid, score in fused.items():
            row = dict(dense_by_id.get(cid, {}))
            if not row:
                continue
            row["score"] = float(score)
            row["dense_score"] = float(dense_scores.get(cid, 0.0))
            row["sparse_score"] = float(sparse_scores.get(cid, 0.0))
            merged[cid] = row

        return sorted(merged.values(), key=lambda x: x.get("score", 0.0), reverse=True)[: self.retrieval.top_k]

    def _windowed_text(self, hit: Dict) -> str:
        payload = hit.get("payload") or {}
        base_text = str(payload.get("text", ""))
        chunk_id = str(hit.get("chunk_id", ""))
        if not self.window.enabled or not chunk_id:
            return base_text
        neighbors = self.qdrant.fetch_chunk_neighbors(
            collection_name=self.collection_name,
            chunk_id=chunk_id,
            backward=int(self.window.backward),
            forward=int(self.window.forward),
        )
        if not neighbors:
            return base_text
        texts = [str(row.get("payload", {}).get("text", "")) for row in neighbors]
        texts = [t for t in texts if t]
        return "\n\n".join(texts) if texts else base_text

    def retrieve(self, query_text: str, query_vector: np.ndarray) -> Tuple[List[Dict], Dict]:
        if self.retrieval.strategy == "dense":
            hits = self._dense_hits(query_vector, self.retrieval.top_k)
        elif self.retrieval.strategy == "hybrid":
            hits = self._hybrid_hits(query_text, query_vector)
        else:
            raise ValueError("retrieval.strategy must be one of: dense, hybrid")

        debug = {
            "retrieval_strategy": self.retrieval.strategy,
            "sparse_strategy": self.retrieval.sparse_strategy if self.retrieval.strategy == "hybrid" else None,
            "fusion": self.retrieval.fusion if self.retrieval.strategy == "hybrid" else None,
            "rerank_enabled": self.rerank.enabled,
            "window_enabled": self.window.enabled,
            "sparse_fallback": self._sparse_fallback,
            "sparse_fallback_reason": self._sparse_fallback_reason,
        }

        if self._reranker:
            candidates = []
            for h in hits:
                row = dict(h)
                row["rerank_text"] = self._windowed_text(h)
                candidates.append(row)
            reranked_hits = self._reranker.rerank(query_text, candidates)
            debug["initial_hits"] = [
                {
                    "chunk_id": str(h.get("chunk_id", "")),
                    "source_id": str(h.get("source_id", "")),
                    "score": float(h.get("score", 0.0)),
                    "dense_score": float(h.get("dense_score", 0.0)) if h.get("dense_score") is not None else None,
                    "sparse_score": float(h.get("sparse_score", 0.0)) if h.get("sparse_score") is not None else None,
                }
                for h in hits
            ]
            debug["reranked_hits"] = [
                {
                    "chunk_id": str(h.get("chunk_id", "")),
                    "source_id": str(h.get("source_id", "")),
                    "score": float(h.get("score", 0.0)),
                    "rerank_score": float(h.get("rerank_score", 0.0)),
                    "dense_score": float(h.get("dense_score", 0.0)) if h.get("dense_score") is not None else None,
                    "sparse_score": float(h.get("sparse_score", 0.0)) if h.get("sparse_score") is not None else None,
                }
                for h in reranked_hits
            ]
            hits = reranked_hits[: self.rerank.top_k]

        return hits, debug
