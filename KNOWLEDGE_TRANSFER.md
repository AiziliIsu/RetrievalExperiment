# Knowledge Transfer

---

## 1. Project status

The framework is fully functional and was used to run distributed retrieval evaluations across three multilingual embedding models and three languages (Russian, Kyrgyz, English). All core pipeline stages — corpus ingestion, indexing, evaluation, and aggregation — are implemented and tested on real data.

This project was developed as part of an NLP/IR research study evaluating multilingual retrieval accuracy for mixed language in one session (KG/EN/RU). The initial implementation targets Russian and Kyrgyz due to time constraints.

---

## 2. What was accomplished

- Implemented a complete retrieval evaluation pipeline supporting dense, hybrid, and reranking modes.
- Evaluated three HuggingFace multilingual embedding models.
- Evaluated retrieval quality separately for Russian, Kyrgyz, and English questions against a shared multilingual corpus seperately first with source dataset in kyrgyz and russian.
- Established reproducible frozen experiment settings (chunking, preprocessing) so results across machines are comparable.
- Implemented resumable evaluation: interrupted runs continue from the last completed question.
- Implemented resumable indexing: interrupted indexing resumes by skipping already-uploaded points.

---

## 3. Design decisions and rationale

### SHA1 corpus cache
The preprocessed corpus (chunked, normalized) is cached to disk keyed by SHA1 of the source files + config. This avoids re-chunking on every run. If the corpus or chunking settings change, the cache key changes and a fresh cache is built automatically.

### Collection naming
Qdrant collection names encode the dataset name, model, chunk size, and overlap: `<dataset>_<model>_c<chunksize>_o<overlap>`. Changing chunking settings therefore automatically creates a new collection.

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

**Corpus structure expected:** A folder (optionally nested) of `.json` or `.jsonl` files. Each record must have an `id` field and a `содержание` field (the text to embed). Records may also have a `record_type` field; the value `product` triggers special handling if the source_dataset is in Kyrgyz.

**Question file structure expected:** A `.jsonl` file where each line has `id_question`, `id` (the expected source document), `question`, and `language` fields. Multiple question files in a folder are supported, auto-detected by filename convention (`questions_ru`, `questions_kg`, `questions_en`).


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



