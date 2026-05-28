# Knowledge Transfer Document

**Project:** RetrievalExperiment (RQ1 draft)
**Prepared for:** Future engineers taking over or extending this project

---

## 1. Project status

The framework is fully functional and was used to run distributed retrieval evaluations across three multilingual embedding models and three languages (Russian, Kyrgyz, English). All core pipeline stages — corpus ingestion, indexing, evaluation, and aggregation — are implemented and tested on real data.

The codebase is at an **experimental/research stage**: it produces correct results and is resumable, but it is not hardened as a production service. No automated test suite exists.

---

## 2. What was accomplished

- Implemented a complete retrieval evaluation pipeline supporting dense, hybrid (BM25 and BGE-M3 sparse), and reranking modes.
- Evaluated three HuggingFace multilingual embedding models: `multilingual-e5-large`, `BAAI/bge-m3`, `Alibaba-NLP/gte-multilingual-base`.
- Evaluated retrieval quality separately for Russian, Kyrgyz, and English questions against a shared multilingual corpus.
- Established reproducible frozen experiment settings (chunking, preprocessing) so results across machines are comparable.
- Implemented resumable evaluation: interrupted runs continue from the last completed question.
- Implemented resumable indexing: interrupted indexing resumes by skipping already-uploaded points.
- Built an aggregation script (`aggregate_runs.py`) to merge results from distributed machines.

---

## 3. Design decisions and rationale

### SHA1 corpus cache
The preprocessed corpus (chunked, normalized) is cached to disk keyed by SHA1 of the source files + config. This avoids re-chunking on every run. If the corpus or chunking settings change, the cache key changes and a fresh cache is built automatically.

### Collection naming
Qdrant collection names encode the dataset name, model, chunk size, and overlap: `<dataset>_<model>_c<chunksize>_o<overlap>`. Changing chunking settings therefore automatically creates a new collection rather than silently comparing results with a different chunking scheme.

### Product records skip chunking
Records with `record_type: product` in the corpus are embedded whole (no chunking). This is because product entries are short and structured, and splitting them would break the semantic unit. The behavior is controlled by `product_rules` in the config and can be extended.

### Two-phase run (index-only / eval-only)
The recommended workflow separates indexing from evaluation. This lets you verify the index is complete (correct point count) before running the expensive evaluation loop. It also allows re-running evaluation with different retrieval strategies without rebuilding the index.

### BGE-M3 sparse on candidates only
The `bge_m3_sparse` strategy does not build a full sparse index. Instead, it scores only the top candidates returned by the dense retriever using BGE-M3's lexical head. This trades recall (a full sparse index would surface different candidates) for implementation simplicity and no additional storage. For a fairer sparse comparison, consider switching to `bm25` or implementing a true sparse index.

### Chunk window expansion
When reranking is enabled, the `chunk_window` option expands each retrieved chunk to include neighboring chunks before the reranker sees it. This improves context at the cost of longer reranker inputs. It is off by default.

---

## 4. Data (not included in this repository)

The corpus and question files are not stored in this repository. They must be obtained separately and placed locally. Update `distributed_config.yaml` to point to them:

```yaml
dataset:
  corpus_root: "/path/to/corpus"
  questions_path: "/path/to/questions"
```

**Corpus structure expected:** A folder (optionally nested) of `.json` or `.jsonl` files. Each record must have an `id` field and a `содержание` field (the text to embed). Records may also have a `record_type` field; the value `product` triggers special handling.

**Question file structure expected:** A `.jsonl` file where each line has `id_question`, `id` (the expected source document), `question`, and `language` fields. Multiple question files in a folder are supported, auto-detected by filename convention (`questions_ru`, `questions_kg`, `questions_en`).

---

## 5. Environment and secrets

API keys are never stored in this repository. Copy `.env.example` to `.env` on each machine and fill in only the keys you need. The models used in this project (`bge-m3`, `multilingual-e5`, `gte-multilingual-base`) use `sentence_transformers` and require no API key — they download from HuggingFace automatically.

The `.env` file is listed in `.gitignore` and must never be committed. Verify this with `git status` before pushing.

---

## 6. Known issues and limitations

| Issue | Impact | Suggested fix |
|-------|--------|---------------|
| No automated tests | Changes to core logic may silently break metrics | Add pytest unit tests for `evaluator.py`, `chunker.py`, and `corpus_builder.py` |
| BGE-M3 sparse scorer is slow | Using `bge_m3_sparse` on CPU is very slow at query time | Use `bm25` for sparse, or ensure a GPU is available |
| No dataset versioning | Corpus changes require manual `--force-reindex` | Add a corpus hash to the collection name or implement version tagging |
| Kyrgyz BM25 is unstemmed | BM25 tokenizer does not stem Kyrgyz tokens | Integrate a Kyrgyz stemmer or morphological analyzer |
| No structured logging | Print statements only; no log files | Replace `_safe_print` with Python `logging` module |
| `fix_summaries.py` is a one-off utility | Not integrated into the main pipeline | Either promote to a proper subcommand or document clearly as a rescue tool |

---

## 7. Recommended next steps

**Short term (before next experiment round):**

1. Add at least smoke tests for `chunker.py` and `evaluator.py` to catch regressions.
2. Pin exact dependency versions in `requirements.txt` (use `pip freeze` after a working install).
3. Document the exact corpus and question file versions used in any published results — include file counts, date obtained, and any preprocessing applied outside this framework.

**Medium term (if extending the project):**

4. Implement a true sparse index (e.g., Qdrant sparse vectors) to make `bge_m3_sparse` a fairer comparison.
5. Add support for additional evaluation metrics (Precision@k, MAP).
6. Add a `--dry-run` flag that reports what would be indexed/evaluated without executing.
7. Consolidate `RUN_ON_OTHER_MACHINES_GUIDE.md` into the README to reduce documentation fragmentation.

**Long term (if productionizing):**

8. Replace the file-based checkpoint/resume system with a proper job queue (e.g., Celery, RQ).
9. Add a configuration validation step at startup that catches path errors before the run begins.
10. Consider containerizing the experiment runner (Dockerfile) for full environment reproducibility.

---

## 8. How to add a new embedding model

1. Open `distributed_config.yaml` and add an entry under `models_registry`:

```yaml
models_registry:
  my-new-model:
    provider: "sentence_transformers"
    model_id: "org/model-name-on-huggingface"
    dimension: 768
    batch_size: 16
    normalize: true
    query_prefix: ""
    document_prefix: ""
```

2. Run with `--model my-new-model`. No code changes are needed for `sentence_transformers` models.

3. For a new provider type, implement `BaseEmbedder` in `embedders/`, then add a branch in `embedding_wrappers.create_embedder()`.

---

## 9. How to add a new reranker

Add an entry to `SentenceTransformersReranker.MODEL_MAP` in `distributed_eval/experiment_pipeline.py`:

```python
MODEL_MAP = {
    ...
    "my-reranker": "org/reranker-model-on-huggingface",
}
```

Then set `experiments.reranking.model: "my-reranker"` in the config.

---

## 10. Contact and project history

This project was developed as part of an NLP/IR research study evaluating multilingual retrieval for Central Asian languages. The initial implementation targets Russian and Kyrgyz alongside English as a baseline.

For questions about the corpus or evaluation protocol design decisions not documented here, refer to the associated research notes or contact the original author.
