from __future__ import annotations

import re
import uuid
from typing import Dict, Iterable, List

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, HnswConfigDiff, PointStruct, SearchParams, VectorParams


class DistributedQdrantManager:
    def __init__(self, config: Dict):
        self.client = QdrantClient(url=config["url"], api_key=config.get("api_key"))
        self.search_params = SearchParams(
            hnsw_ef=int(config["search"]["ef"]),
            exact=bool(config["search"]["exact"]),
        )
        self.hnsw = HnswConfigDiff(
            m=int(config["hnsw"]["m"]),
            ef_construct=int(config["hnsw"]["ef_construct"]),
        )
        distance_map = {"Cosine": Distance.COSINE, "Euclidean": Distance.EUCLID, "Dot": Distance.DOT}
        self.distance = distance_map[config["distance"]]

    def collection_exists(self, name: str) -> bool:
        try:
            self.client.get_collection(name)
            return True
        except Exception:
            return False

    def points_count(self, name: str) -> int:
        info = self.client.get_collection(name)
        return int(info.points_count or 0)

    def ensure_collection(self, name: str, dimension: int) -> None:
        if self.collection_exists(name):
            return
        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=int(dimension), distance=self.distance),
            hnsw_config=self.hnsw,
        )

    def recreate_collection(self, name: str, dimension: int) -> None:
        if self.collection_exists(name):
            self.client.delete_collection(name)
        self.ensure_collection(name, dimension)

    @staticmethod
    def point_id_from_chunk_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, str(chunk_id)))

    def existing_point_ids(self, collection_name: str, chunk_ids: List[str], batch_size: int = 512) -> set[str]:
        point_ids = [self.point_id_from_chunk_id(chunk_id) for chunk_id in chunk_ids]
        found: set[str] = set()
        for i in range(0, len(point_ids), batch_size):
            records = self.client.retrieve(
                collection_name=collection_name,
                ids=point_ids[i : i + batch_size],
                with_payload=False,
                with_vectors=False,
            )
            for record in records:
                found.add(str(record.id))
        return found

    def upload_embeddings(
        self,
        collection_name: str,
        vectors: np.ndarray,
        payloads: List[Dict],
        texts: List[str],
        batch_size: int,
    ) -> None:
        points = []
        for vector, payload, text in zip(vectors, payloads, texts):
            # Qdrant point id must be uint or UUID.
            # Use stable UUID derived from chunk_id for repeatable inserts.
            raw_id = str(payload["chunk_id"])
            point_id = self.point_id_from_chunk_id(raw_id)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload={"text": text, **payload},
                )
            )
        for i in range(0, len(points), batch_size):
            self.client.upsert(collection_name=collection_name, points=points[i : i + batch_size])

    def search(self, collection_name: str, query_vector: np.ndarray, top_k: int) -> List[Dict]:
        hits = self.client.query_points(
            collection_name=collection_name,
            query=query_vector.tolist(),
            limit=int(top_k),
            search_params=self.search_params,
        ).points
        return [
            {
                "score": h.score,
                "payload": h.payload,
                "chunk_id": h.payload.get("chunk_id"),
                "source_id": h.payload.get("id"),
            }
            for h in hits
        ]

    def retrieve_by_chunk_ids(self, collection_name: str, chunk_ids: List[str]) -> List[Dict]:
        if not chunk_ids:
            return []
        point_ids = [self.point_id_from_chunk_id(str(cid)) for cid in chunk_ids]
        records = self.client.retrieve(
            collection_name=collection_name,
            ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )
        by_chunk = {}
        for r in records:
            payload = r.payload or {}
            chunk_id = str(payload.get("chunk_id", ""))
            if not chunk_id:
                continue
            by_chunk[chunk_id] = {
                "score": 0.0,
                "payload": payload,
                "chunk_id": chunk_id,
                "source_id": payload.get("id"),
            }
        return [by_chunk[cid] for cid in chunk_ids if cid in by_chunk]

    def fetch_chunk_neighbors(self, collection_name: str, chunk_id: str, backward: int, forward: int) -> List[Dict]:
        m = re.match(r"^(.*)::chunk::(\d+)$", str(chunk_id))
        if not m:
            # Product-like records are single chunk.
            return self.retrieve_by_chunk_ids(collection_name, [chunk_id])

        base = m.group(1)
        idx = int(m.group(2))
        start = max(0, idx - max(0, int(backward)))
        end = idx + max(0, int(forward))
        requested = [f"{base}::chunk::{i}" for i in range(start, end + 1)]
        return self.retrieve_by_chunk_ids(collection_name, requested)
