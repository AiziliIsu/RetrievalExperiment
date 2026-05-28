"""
Distributed experiment entrypoint.

Only two runtime switches should change between machines:
1) model
2) language

Examples:
  python run_experiment.py --model bge-m3 --lang ru
  python run_experiment.py --config distributed_config.yaml
"""
from __future__ import annotations

import argparse
import datetime
import fnmatch
import json
import os
import re
import sys
from pathlib import Path

import certifi
from dotenv import load_dotenv

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
_shared_cache_root = Path.home() / ".cache" / "rq1_draft"
_hf_cache_root = _shared_cache_root / "huggingface"
os.environ.setdefault("HF_HOME", str(_hf_cache_root.resolve()))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str((_hf_cache_root / "sentence_transformers").resolve()))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str((_hf_cache_root / "hub").resolve()))
os.environ.setdefault("TRANSFORMERS_CACHE", str((_hf_cache_root / "transformers").resolve()))


def _resolve_cache_dir(value: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value.strip()))
    if not expanded or expanded == ".cache_distributed":
        return _shared_cache_root / "runtime"
    return Path(expanded).resolve()


def _safe_print(message: str) -> None:
    try:
        print(message)
    except UnicodeEncodeError:
        enc = (getattr(sys.stdout, "encoding", None) or "utf-8")
        print(message.encode(enc, errors="replace").decode(enc, errors="replace"))


def _matches_allowlist(path: str, patterns: list[str]) -> bool:
    if not patterns:
        return True

    normalized_path = Path(path).as_posix().lower()
    filename = Path(path).name.lower()
    for pattern in patterns:
        normalized_pattern = pattern.replace("\\", "/").lower()
        if fnmatch.fnmatch(normalized_path, normalized_pattern) or fnmatch.fnmatch(filename, normalized_pattern):
            return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="distributed_config.yaml")
    parser.add_argument("--model", default=None, help="Override model name (e.g. bge-m3)")
    parser.add_argument("--lang", default=None, help="Override language: ru|kg|en")
    parser.add_argument("--force-reindex", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--index-only", action="store_true", help="Build/verify collection only, then exit")
    parser.add_argument("--eval-only", action="store_true", help="Skip indexing and only run question evaluation")
    parser.add_argument("--retrieval-strategy", default=None, help="dense|hybrid")
    parser.add_argument("--sparse-strategy", default=None, help="bm25|bge_m3_sparse")
    parser.add_argument("--fusion", default=None, help="rrf|weighted")
    parser.add_argument("--retrieval-k", type=int, default=None, help="Final retrieved top_k before metrics")
    parser.add_argument("--candidate-k", type=int, default=None, help="Candidate pool size for hybrid retrieval")
    parser.add_argument("--enable-rerank", action="store_true", help="Enable reranking stage")
    parser.add_argument("--reranker-model", default=None, help="Reranker model selector")
    parser.add_argument("--rerank-k", type=int, default=None, help="Top-k after reranking")
    parser.add_argument("--window-backward", type=int, default=None, help="Neighbor chunks before N")
    parser.add_argument("--window-forward", type=int, default=None, help="Neighbor chunks after N")
    parser.add_argument("--enable-window", action="store_true", help="Enable neighboring chunk window expansion")
    parser.add_argument(
        "--skip-completed-from-dir",
        default=None,
        help=(
            "Directory containing existing '*_summary.json' files. "
            "Matching run_ids will be skipped."
        ),
    )
    return parser.parse_args()


def main() -> None:
    from distributed_eval.corpus_builder import ensure_processed_cache, iter_cached_rows
    from distributed_eval.distributed_config import DistributedConfig
    from distributed_eval.embedding_wrappers import create_embedder
    from distributed_eval.evaluator import Evaluator
    from distributed_eval.experiment_pipeline import (
        RetrievalOptions,
        RetrievalPipeline,
        RerankOptions,
        WindowOptions,
    )
    from distributed_eval.loader import discover_question_files, load_questions
    from distributed_eval.normalizer import TextNormalizer
    from distributed_eval.qdrant_manager import DistributedQdrantManager
    from distributed_eval.result_writer import ResultWriter
    from distributed_eval.summary_aggregation import build_aggregated_summary

    args = parse_args()
    load_dotenv()

    dcfg = DistributedConfig(args.config)
    selection = dcfg.resolve_selection(args.model, args.lang)
    model_cfg = dcfg.model_config(selection.model)

    frozen = dcfg.cfg["frozen"]
    experiments_cfg = dcfg.cfg.get("experiments", {})
    retrieval_cfg = dict(experiments_cfg.get("retrieval", {}))
    rerank_cfg = dict(experiments_cfg.get("reranking", {}))
    window_cfg = dict(experiments_cfg.get("chunk_window", {}))
    question_file_allowlist = list(dcfg.cfg.get("dataset", {}).get("question_file_allowlist", []) or [])

    retrieval_opts = RetrievalOptions(
        strategy=str(args.retrieval_strategy or retrieval_cfg.get("strategy", "dense")).strip().lower(),
        top_k=int(args.retrieval_k or retrieval_cfg.get("top_k", frozen["retrieval"]["top_k"])),
        candidate_k=int(args.candidate_k or retrieval_cfg.get("candidate_k", 50)),
        sparse_strategy=str(args.sparse_strategy or retrieval_cfg.get("sparse_strategy", "bm25")).strip().lower(),
        fusion=str(args.fusion or retrieval_cfg.get("fusion", "rrf")).strip().lower(),
        dense_weight=float(retrieval_cfg.get("dense_weight", 0.6)),
        sparse_weight=float(retrieval_cfg.get("sparse_weight", 0.4)),
    )
    rerank_opts = RerankOptions(
        enabled=bool(args.enable_rerank or rerank_cfg.get("enabled", False)),
        model=str(args.reranker_model or rerank_cfg.get("model", "bge-reranker-v2-m3")),
        top_k=int(args.rerank_k or rerank_cfg.get("top_k", retrieval_opts.top_k)),
    )
    window_opts = WindowOptions(
        enabled=bool(args.enable_window or window_cfg.get("enabled", False)),
        backward=int(args.window_backward if args.window_backward is not None else window_cfg.get("backward", 1)),
        forward=int(args.window_forward if args.window_forward is not None else window_cfg.get("forward", 1)),
    )

    if retrieval_opts.strategy not in {"dense", "hybrid"}:
        raise ValueError("retrieval.strategy must be one of: dense, hybrid")
    if retrieval_opts.sparse_strategy not in {"bm25", "bge_m3_sparse"}:
        raise ValueError("sparse_strategy must be one of: bm25, bge_m3_sparse")
    if retrieval_opts.fusion not in {"rrf", "weighted"}:
        raise ValueError("fusion must be one of: rrf, weighted")

    dataset_name = dcfg.cfg["dataset"]["name"]
    corpus_path = dcfg.cfg["dataset"].get("corpus_root") or dcfg.cfg["dataset"]["corpus_path"]
    file_filter_cfg = dcfg.cfg["dataset"].get("file_filters", {})
    collection_name = dcfg.collection_name(dataset_name, selection.model)
    qdrant = DistributedQdrantManager(dcfg.cfg["qdrant"])

    _safe_print(f"MODEL_NAME = {selection.model}")
    _safe_print(f"LANGUAGE (default) = {selection.language}")
    _safe_print(f"COLLECTION = {collection_name}")
    _safe_print(
        "Experiment strategy: "
        f"retrieval={retrieval_opts.strategy}, "
        f"sparse={retrieval_opts.sparse_strategy}, fusion={retrieval_opts.fusion}, "
        f"rerank={rerank_opts.enabled}({rerank_opts.model}), "
        f"window={window_opts.enabled}(backward={window_opts.backward}, forward={window_opts.forward})"
    )

    normalizer = TextNormalizer(frozen["preprocessing"])
    embedder = create_embedder(model_cfg)
    cache_info = ensure_processed_cache(
        corpus_path=corpus_path,
        source_name=dataset_name,
        normalizer=normalizer,
        chunking_cfg=frozen["chunking"],
        product_cfg=dcfg.cfg.get("product_rules", {}),
        file_filter_cfg=file_filter_cfg,
        cache_dir=str(_resolve_cache_dir(dcfg.cfg["runtime"]["cache_dir"])),
    )
    expected_rows = int(cache_info["row_count"])
    _safe_print(f"Prepared processed cache rows: {expected_rows} (rebuilt={cache_info['rebuilt']})")
    if cache_info.get("report_path"):
        _safe_print(f"Processing trace report: {cache_info['report_path']}")

    qdrant.ensure_collection(collection_name, model_cfg["dimension"])
    current_points = qdrant.points_count(collection_name) if qdrant.collection_exists(collection_name) else 0

    if args.eval_only and current_points < expected_rows:
        raise RuntimeError(
            f"Collection '{collection_name}' is incomplete: points={current_points}, expected={expected_rows}. "
            "Run without --eval-only to build index first."
        )

    need_index = (current_points != expected_rows) or args.force_reindex
    if args.eval_only:
        need_index = False

    if need_index:
        if args.force_reindex:
            qdrant.recreate_collection(collection_name, model_cfg["dimension"])
            _safe_print("Indexing mode: force reindex (collection recreated).")
        else:
            _safe_print(f"Indexing mode: resume/repair from existing collection ({current_points}/{expected_rows}).")

        batch_size = int(frozen["embedding"]["batch_size"])
        cache_dir = _resolve_cache_dir(dcfg.cfg["runtime"]["cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        progress_path = cache_dir / f"index_progress_{collection_name}.json"

        def _write_progress(status: str, processed_rows: int, uploaded_rows: int, skipped_existing: int) -> None:
            snapshot = {
                "collection": collection_name,
                "status": status,
                "expected_rows": expected_rows,
                "initial_points": current_points,
                "processed_rows": processed_rows,
                "uploaded_rows": uploaded_rows,
                "skipped_existing": skipped_existing,
                "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
            }
            progress_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

        text_batch = []
        payload_batch = []
        processed = 0
        uploaded = 0
        skipped_existing = 0

        _write_progress(
            status="running",
            processed_rows=processed,
            uploaded_rows=uploaded,
            skipped_existing=skipped_existing,
        )

        for row in iter_cached_rows(cache_info["cache_path"]):
            text_batch.append(row["text"])
            payload_batch.append(row["payload"])
            if len(text_batch) < batch_size:
                continue

            chunk_ids = [str(p["chunk_id"]) for p in payload_batch]
            existing_ids = qdrant.existing_point_ids(collection_name, chunk_ids)
            missing_payloads = []
            missing_texts = []
            for text, payload, chunk_id in zip(text_batch, payload_batch, chunk_ids):
                point_id = qdrant.point_id_from_chunk_id(chunk_id)
                if point_id in existing_ids:
                    skipped_existing += 1
                    continue
                missing_payloads.append(payload)
                missing_texts.append(text)

            if missing_texts:
                vectors = embedder.embed_documents(missing_texts)
                qdrant.upload_embeddings(
                    collection_name=collection_name,
                    vectors=vectors,
                    payloads=missing_payloads,
                    texts=missing_texts,
                    batch_size=batch_size,
                )
                uploaded += len(missing_texts)

            processed += len(text_batch)
            text_batch = []
            payload_batch = []
            _write_progress(
                status="running",
                processed_rows=processed,
                uploaded_rows=uploaded,
                skipped_existing=skipped_existing,
            )

        if text_batch:
            chunk_ids = [str(p["chunk_id"]) for p in payload_batch]
            existing_ids = qdrant.existing_point_ids(collection_name, chunk_ids)
            missing_payloads = []
            missing_texts = []
            for text, payload, chunk_id in zip(text_batch, payload_batch, chunk_ids):
                point_id = qdrant.point_id_from_chunk_id(chunk_id)
                if point_id in existing_ids:
                    skipped_existing += 1
                    continue
                missing_payloads.append(payload)
                missing_texts.append(text)

            if missing_texts:
                vectors = embedder.embed_documents(missing_texts)
                qdrant.upload_embeddings(
                    collection_name=collection_name,
                    vectors=vectors,
                    payloads=missing_payloads,
                    texts=missing_texts,
                    batch_size=batch_size,
                )
                uploaded += len(missing_texts)

            processed += len(text_batch)
            _write_progress(
                status="running",
                processed_rows=processed,
                uploaded_rows=uploaded,
                skipped_existing=skipped_existing,
            )

        final_points = qdrant.points_count(collection_name)
        _write_progress(
            status="complete",
            processed_rows=processed,
            uploaded_rows=uploaded,
            skipped_existing=skipped_existing,
        )
        _safe_print(f"Index progress snapshot: {progress_path}")
        _safe_print(
            f"Indexed rows seen: {processed}, newly uploaded: {uploaded}, "
            f"already present: {skipped_existing}, collection points: {final_points}"
        )
        if final_points != expected_rows:
            raise RuntimeError(
                f"Collection verification failed: points={final_points}, expected={expected_rows}."
            )
    else:
        _safe_print(f"Indexing skipped: collection already complete ({current_points}/{expected_rows}).")

    if args.index_only:
        _safe_print("Index-only mode complete. Exiting before question evaluation.")
        return

    sparse_rows = None
    if retrieval_opts.strategy == "hybrid" and retrieval_opts.sparse_strategy == "bm25":
        _safe_print("Loading processed cache rows for BM25 sparse retrieval...")
        sparse_rows = list(iter_cached_rows(cache_info["cache_path"]))

    retrieval_pipeline = RetrievalPipeline(
        qdrant=qdrant,
        collection_name=collection_name,
        embedder=embedder,
        retrieval=retrieval_opts,
        rerank=rerank_opts,
        window=window_opts,
        sparse_cache_rows=sparse_rows,
    )

    questions_path = dcfg.cfg["dataset"]["questions_path"]
    discovered = discover_question_files(questions_path)
    questions_path_obj = Path(questions_path)

    if question_file_allowlist:
        discovered = [entry for entry in discovered if _matches_allowlist(entry["path"], question_file_allowlist)]

    eval_jobs = []
    if discovered:
        for entry in discovered:
            eval_jobs.append(
                {
                    "path": entry["path"],
                    "language": entry["language"],
                    "source_name": Path(entry["path"]).name,
                }
            )
        _safe_print(f"Discovered question files: {len(eval_jobs)}")
    else:
        if questions_path_obj.is_dir():
            raise RuntimeError(
                "No question files found in questions_path directory. "
                "Expected filenames containing questions_en, questions_ru, or questions_kg."
            )
        # Backward-compatible mode for direct question file path.
        eval_jobs.append(
            {
                "path": questions_path,
                "language": selection.language,
                "source_name": Path(questions_path).name,
            }
        )

    def _safe_filename_component(value: str) -> str:
        # Keep Unicode and extension; only replace Windows-forbidden filename chars.
        cleaned = value.strip()
        cleaned = re.sub(r"[<>:\"/\\|?*]", "_", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.rstrip(" .") or "questions"

    skip_existing_run_outputs = set()
    if args.skip_completed_from_dir:
        skip_dir = Path(args.skip_completed_from_dir)
        if not skip_dir.exists() or not skip_dir.is_dir():
            raise ValueError(f"--skip-completed-from-dir is not a directory: {skip_dir}")
        for p in skip_dir.glob("*_summary.json"):
            if p.name.startswith("final_"):
                continue
            skip_existing_run_outputs.add(p.name)
        _safe_print(
            "Skip list loaded from directory: "
            f"{skip_dir} ({len(skip_existing_run_outputs)} summary files)"
        )

    summaries = []
    for job in eval_jobs:
        q_path = job["path"]
        q_lang = job["language"]
        q_source_name = job["source_name"]

        questions = load_questions(q_path, q_lang)
        _safe_print(f"Filtered questions for '{q_lang}' from '{q_source_name}': {len(questions)}")
        if not questions:
            _safe_print(f"Skipping empty question file: {q_source_name}")
            continue

        question_file_name = _safe_filename_component(Path(q_source_name).name)
        run_id = f"{dataset_name}_{selection.model}_{q_lang}_{question_file_name}"
        summary_name = f"{run_id}_summary.json"
        if summary_name in skip_existing_run_outputs:
            _safe_print(f"Skipping (already completed externally): {summary_name}")
            continue

        state_dir = str(_resolve_cache_dir(dcfg.cfg["runtime"]["cache_dir"]) / "evaluation_state")
        writer = ResultWriter(output_dir=dcfg.cfg["runtime"]["results_dir"], run_id=run_id, state_dir=state_dir)
        evaluator = Evaluator(
            embedder=embedder,
            qdrant=qdrant,
            collection_name=collection_name,
            top_k=retrieval_opts.top_k,
            normalizer=normalizer,
            writer=writer,
            model_name=selection.model,
            language=q_lang,
            retrieval_pipeline=retrieval_pipeline,
        )
        summary = evaluator.evaluate(questions, resume=not args.no_resume)

        summary["run_id"] = run_id
        summary["question_file"] = q_source_name
        summary["question_file_path"] = q_path
        summary["frozen"] = frozen
        summary["machine"] = os.getenv("COMPUTERNAME") or os.getenv("HOSTNAME", "")
        writer.save_summary(summary)
        summaries.append(summary)

    if summaries:
        aggregated = build_aggregated_summary(
            summaries,
            dataset_name=dataset_name,
            model_name=selection.model,
            collection_name=collection_name,
        )
        final_summary_path = Path(dcfg.cfg["runtime"]["results_dir"]) / "final_aggregated_summary.json"
        final_summary_path.parent.mkdir(parents=True, exist_ok=True)
        final_summary_path.write_text(json.dumps(aggregated, indent=2, ensure_ascii=False), encoding="utf-8")
        _safe_print(f"Final aggregated summary: {final_summary_path}")

    if summaries:
        _safe_print("\nRun summary:")
        _safe_print(json.dumps(summaries[-1], indent=2, ensure_ascii=False))
    else:
        _safe_print("\nRun summary: no evaluation jobs were executed.")


if __name__ == "__main__":
    main()
