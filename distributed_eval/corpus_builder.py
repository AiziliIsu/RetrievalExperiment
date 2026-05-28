from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Generator

from .chunker import FixedChunking
from .loader import iter_corpus_records_with_source, list_corpus_files
from .normalizer import TextNormalizer
from .product_handler import process_record


def _cache_key(corpus_path: str, chunk_size: int, overlap: int, file_filter_cfg: Dict) -> str:
    p = Path(corpus_path)
    stamp = (
        f"{p.resolve()}::{p.stat().st_mtime_ns}::{chunk_size}::{overlap}"
        f"::{json.dumps(file_filter_cfg, sort_keys=True, ensure_ascii=False)}"
    )
    return hashlib.sha1(stamp.encode("utf-8")).hexdigest()[:16]


def _cache_jsonl_path(cache_dir: str, key: str) -> Path:
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    return cdir / f"processed_corpus_{key}.jsonl"


def _cache_meta_path(cache_dir: str, key: str) -> Path:
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    return cdir / f"processed_corpus_{key}.meta.json"


def _cache_report_path(cache_dir: str, key: str) -> Path:
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    return cdir / f"processed_corpus_{key}.report.json"


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def ensure_processed_cache(
    corpus_path: str,
    source_name: str,
    normalizer: TextNormalizer,
    chunking_cfg: Dict,
    product_cfg: Dict,
    file_filter_cfg: Dict,
    cache_dir: str,
) -> Dict:
    """
    Builds a complete cache atomically (tmp -> rename) and writes metadata.
    If cache+meta are complete, reuses them.
    """
    chunk_size = int(chunking_cfg["chunk_size"])
    overlap = int(chunking_cfg["overlap"])
    key = _cache_key(corpus_path, chunk_size, overlap, file_filter_cfg)

    cache_path = _cache_jsonl_path(cache_dir, key)
    meta_path = _cache_meta_path(cache_dir, key)
    report_path = _cache_report_path(cache_dir, key)

    if cache_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if bool(meta.get("complete")):
                expected = int(meta.get("row_count", -1))
                actual = _count_lines(cache_path)
                if expected == actual and expected >= 0:
                    return {
                        "cache_path": str(cache_path),
                        "row_count": expected,
                        "rebuilt": False,
                        "key": key,
                        "report_path": str(report_path),
                    }
        except Exception:
            pass

    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    chunker = FixedChunking(chunk_size=chunk_size, overlap=overlap)
    row_count = 0
    listed_files = list_corpus_files(
        corpus_path,
        recursive=bool(file_filter_cfg.get("recursive", False)),
        exclude_contains=list(file_filter_cfg.get("exclude_contains", [])),
        exclude_endswith_tokens=list(file_filter_cfg.get("exclude_endswith_tokens", [])),
    )
    file_stats: Dict[str, Dict] = {}

    def _file_bucket(path: str) -> Dict:
        bucket = file_stats.get(path)
        if bucket is None:
            bucket = {
                "records_seen": 0,
                "rows_written": 0,
                "used_ids": set(),
                "skipped_ids": [],
            }
            file_stats[path] = bucket
        return bucket

    with tmp_path.open("w", encoding="utf-8") as out:
        for record, source_path in iter_corpus_records_with_source(
            corpus_path,
            recursive=bool(file_filter_cfg.get("recursive", False)),
            exclude_contains=list(file_filter_cfg.get("exclude_contains", [])),
            exclude_endswith_tokens=list(file_filter_cfg.get("exclude_endswith_tokens", [])),
        ):
            bucket = _file_bucket(source_path)
            bucket["records_seen"] += 1
            source_id = str(record.get("id", "")).strip()

            try:
                processed = process_record(
                    record=record,
                    chunking=chunker,
                    normalizer=normalizer,
                    product_cfg=product_cfg,
                )
            except Exception as exc:
                bucket["skipped_ids"].append(
                    {
                        "id": source_id,
                        "reason": f"processing_error: {exc}",
                    }
                )
                continue

            if not processed:
                bucket["skipped_ids"].append(
                    {
                        "id": source_id,
                        "reason": "empty_or_unusable_content",
                    }
                )
                continue

            if source_id:
                bucket["used_ids"].add(source_id)

            for row in processed:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                row_count += 1
                bucket["rows_written"] += 1

    os.replace(tmp_path, cache_path)
    meta = {
        "complete": True,
        "row_count": row_count,
        "cache_path": str(cache_path),
        "key": key,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    report_files = {}
    for fpath, stat in file_stats.items():
        report_files[fpath] = {
            "records_seen": int(stat["records_seen"]),
            "rows_written": int(stat["rows_written"]),
            "used_ids": sorted(stat["used_ids"]),
            "skipped_ids": stat["skipped_ids"],
        }

    report = {
        "cache_key": key,
        "corpus_path": corpus_path,
        "included_files": listed_files["included"],
        "skipped_files": listed_files["skipped"],
        "used_files": [p for p, s in report_files.items() if int(s["rows_written"]) > 0],
        "files": report_files,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "cache_path": str(cache_path),
        "row_count": row_count,
        "rebuilt": True,
        "key": key,
        "report_path": str(report_path),
    }


def iter_cached_rows(cache_path: str) -> Generator[Dict, None, None]:
    cpath = Path(cache_path)
    if not cpath.exists():
        return
    with cpath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

