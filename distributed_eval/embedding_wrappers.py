from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
from embedders.sentence_transformers_embedder import SentenceTransformersEmbedder


class GeminiEmbedder:
    """Gemini embedding wrapper (requires `google-genai`)."""

    def __init__(self, config: Dict):
        try:
            from google import genai
        except Exception as exc:
            raise ImportError(
                "Gemini support requires `google-genai` package."
            ) from exc

        api_key_env = config.get("api_key_env", "GEMINI_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"{api_key_env} not found in environment")

        self._genai = genai
        self.client = genai.Client(api_key=api_key)
        self.name = config["name"]
        self.model_id = config["model_id"]
        self.dimension = int(config["dimension"])
        self.batch_size = int(config.get("batch_size", 64))

    def _embed(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            res = self.client.models.embed_content(
                model=self.model_id,
                contents=text,
            )
            vectors.append(res.embeddings[0].values)
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]

    def get_info(self) -> Dict:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "dimension": self.dimension,
            "batch_size": self.batch_size,
        }


def create_embedder(model_cfg: Dict):
    provider = model_cfg.get("provider", "").lower()
    if provider == "sentence_transformers":
        return SentenceTransformersEmbedder(model_cfg)
    if provider == "gemini":
        return GeminiEmbedder(model_cfg)
    raise ValueError(f"Unsupported provider: {provider}")

