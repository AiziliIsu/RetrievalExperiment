"""
Sentence Transformers Implementation
Supports: LaBSE, multilingual-e5-*, paraphrase-multilingual-mpnet, etc.
"""
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from .base_embedder import BaseEmbedder


class SentenceTransformersEmbedder(BaseEmbedder):
    """Sentence Transformers embeddings wrapper."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Determine device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading {self.name} on {self.device}...")

        # Load model  <-- add trust_remote_code here
        self.trust_remote_code = config.get("trust_remote_code", False)
        self.model_kwargs = dict(config.get("model_kwargs", {}))

        # Do not force safetensors globally: some models ship only PyTorch weights.
        self.model_kwargs.setdefault("use_safetensors", False)

        self.model = SentenceTransformer(
            self.model_id,
            device=self.device,
            trust_remote_code=self.trust_remote_code,
            model_kwargs=self.model_kwargs,
        )

        # Prefixes for E5 models
        self.query_prefix = config.get('query_prefix', '')
        self.document_prefix = config.get('document_prefix', '')

        # Normalization
        self.normalize = config.get('normalize', True)

    @staticmethod
    def _is_cuda_oom(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "out of memory" in msg and "cuda" in msg

    @staticmethod
    def _clear_cuda_cache() -> None:
        try:
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        batch_size = max(1, min(int(self.batch_size), len(texts)))
        while True:
            try:
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=self.normalize,
                )
                return embeddings.astype(np.float32)
            except Exception as exc:
                if not self._is_cuda_oom(exc) or batch_size <= 1:
                    raise
                self._clear_cuda_cache()
                batch_size = max(1, batch_size // 2)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed documents."""
        # Add document prefix if configured (for E5 models)
        if self.document_prefix:
            texts = [self.document_prefix + text for text in texts]
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        outputs = []
        step = max(1, min(int(self.batch_size), len(texts)))
        for start in range(0, len(texts), step):
            outputs.append(self._encode_batch(texts[start : start + step]))
        return np.vstack(outputs).astype(np.float32)
    
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query."""
        # Add query prefix if configured (for E5 models)
        if self.query_prefix:
            text = self.query_prefix + text

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )
        
        return embedding.astype(np.float32)
