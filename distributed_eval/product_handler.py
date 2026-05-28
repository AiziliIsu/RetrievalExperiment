from __future__ import annotations

import hashlib
from typing import Dict, List

from .chunker import FixedChunking
from .normalizer import TextNormalizer


def _base_payload(record: Dict) -> Dict:
    return {
        key: value
        for key, value in record.items()
        if key not in {"содержание", "content", "source_name"}
    }


def _record_id(record: Dict) -> str:
    rec_id = str(record.get("id") or "").strip()
    if not rec_id:
        # Stable fallback: hash of the full record content so duplicates stay deterministic
        raw = str(sorted(record.items())).encode("utf-8")
        rec_id = "auto_" + hashlib.sha1(raw).hexdigest()[:16]
    return rec_id


def is_product_record(record: Dict, cfg: Dict) -> bool:
    type_field = cfg.get("type_field")
    product_values = {str(v).lower() for v in cfg.get("product_type_values", [])}
    if not type_field:
        return bool(record.get("is_product", False))
    value = str(record.get(type_field, "")).lower()
    return value in product_values


def process_record(
    record: Dict,
    chunking: FixedChunking,
    normalizer: TextNormalizer,
    product_cfg: Dict,
) -> List[Dict]:
    """
    Returns normalized chunks with payload.
    Payload always contains the original source `id`.
    """
    source_id = _record_id(record)
    content = str(record.get("содержание") or record.get("content", "")).strip()
    if not content:
        return []

    text = normalizer.normalize(content)
    payload = _base_payload(record)
    payload["id"] = source_id

    if is_product_record(record, product_cfg):
        return [
            {
                "text": text,
                "payload": {
                    **payload,
                    "chunk_id": f"{source_id}::product",
                    "chunk_index": 0,
                    "chunk_total": 1,
                    "record_type": "product",
                },
            }
        ]

    chunks = chunking.split(text)
    total = len(chunks)
    rows = []
    for idx, chunk in enumerate(chunks):
        rows.append(
            {
                "text": chunk,
                "payload": {
                    **payload,
                    "chunk_id": f"{source_id}::chunk::{idx}",
                    "chunk_index": idx,
                    "chunk_total": total,
                    "record_type": "document",
                },
            }
        )
    return rows

