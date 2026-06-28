# RetrievalExperiment

A distributed evaluation framework for multilingual information retrieval. It benchmarks dense embedding models (and optional hybrid/reranking pipelines) on Russian, Kyrgyz, and English corpora using a Qdrant vector database.

## What this project does

Given a document corpus and a set of evaluation questions (each with a known answer document), the framework:

1. Chunks and embeds the corpus, uploading vectors to a Qdrant collection.
2. For each question, retrieves the top-k most similar chunks.
3. Computes retrieval metrics: MRR, NDCG@5, Hit@1/3/5/10, and query latency.
4. Writes per-question logs and a JSON summary per run.
5. Supports aggregating results from multiple machines into one final report.

Designed for distributed team runs where each machine evaluates a different model or language.

---

## Repository structure

```
RetrievalExperiment/
├── run_experiment.py          # Main entrypoint — CLI + orchestration
├── fix_summaries.py           # Recalculate metrics from existing per-question logs
├── distributed_config.yaml    # All experiment settings (edit before running)
├── requirements.txt
├── .env                       # Fill in API keys
├── distributed_eval/          # Core library
│   ├── distributed_config.py  # Config parser, model registry
│   ├── chunker.py             # Fixed-size overlapping chunking
│   ├── corpus_builder.py      # Corpus loading, dedup, SHA1 cache
│   ├── loader.py              # JSON/JSONL parser, question file discovery
│   ├── normalizer.py          # Text preprocessing
│   ├── product_handler.py     # Special handling for product-type records
│   ├── embedding_wrappers.py  # Provider factory (sentence-transformers, google-genai, cohere)
│   ├── qdrant_manager.py      # Qdrant collection CRUD, search, neighbor fetch
│   ├── evaluator.py           # MRR / NDCG / Hit metric computation
│   ├── experiment_pipeline.py # Dense, hybrid-BM25, hybrid-BGE-M3, reranking logic
│   ├── result_writer.py       # Streaming JSONL writer + checkpoint/resume
│   └── summary_aggregation.py # Aggregate per-question stats into run summary
└── embedders/
    ├── base_embedder.py            # Abstract base class
    └── sentence_transformers_embedder.py  # HuggingFace model wrapper
```

---

## Quick start

### 1. Prerequisites

- Python 3.10+
- [Qdrant](https://qdrant.tech/documentation/quickstart/) running locally or remotely

Start Qdrant with Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 2. Install dependencies

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure paths

Edit `distributed_config.yaml`:

```yaml
run:
  model: "bge-m3"     # choose from models_registry below
  language: "ru"       # ru | kg | en

dataset:
  corpus_root: "/path/to/your/corpus"     # folder with .json/.jsonl corpus files
  questions_path: "/path/to/questions"    # .jsonl file or folder with questions_ru/kg/en files
```

Only these two sections need to change per machine. All other settings (chunking, retrieval strategy, Qdrant HNSW params) are frozen for reproducibility.

### 4. API keys (optional)

The default models (`bge-m3`, `multilingual-e5`, `gte-multilingual-base`) use HuggingFace and need no API key. If you add an API-based provider:

```bash
cp .env.example .env
# edit .env and fill in the relevant key
```

### 5. Run

```bash
# Single combined run (index + evaluate)
python run_experiment.py --model bge-m3 --lang ru

# Recommended: two-phase (safer, allows inspection between phases)
python run_experiment.py --model bge-m3 --lang ru --index-only
python run_experiment.py --model bge-m3 --lang ru --eval-only

# Force full reindex (use when corpus or chunking settings changed)
python run_experiment.py --model bge-m3 --lang ru --force-reindex
```

---

## Supported models

| Key | HuggingFace model | Dimension |
|-----|-------------------|-----------|
| `multilingual-e5` | `intfloat/multilingual-e5-large` | 1024 |
| `bge-m3` | `BAAI/bge-m3` | 1024 |
| `gte-multilingual-base` | `Alibaba-NLP/gte-multilingual-base` | 768 |

Models are downloaded automatically to `~/.cache/rq1_draft/huggingface` on first run.

---

## Dataset format

### Corpus files

The corpus can be a folder of `.json` or `.jsonl` files (searched recursively). Each record must have at minimum:

```json
{
  "id": "unique-document-id",
  "содержание": "Full text of the document..."
}
```

Records with `"record_type": "product"` are treated specially: no chunking, the full `содержание` field is embedded as one vector.

### Question files

Each question file is a `.jsonl` where each line is:

```json
{
  "id_question": "q001",
  "id": "expected-source-document-id",
  "question": "What is the capital of Kyrgyzstan?",
  "language": "en"
}
```

Questions are filtered by `language` at runtime. A folder of question files is supported — files are auto-detected by naming convention (`questions_ru`, `questions_kg`, `questions_en`).

---

## Configuration reference

All settings live in `distributed_config.yaml`. Key sections:

| Section | What it controls |
|---------|-----------------|
| `run` | Active model and language (change per machine) |
| `dataset` | Corpus path, questions path (change per machine) |
| `frozen.chunking` | `chunk_size` and `overlap` — **keep identical across all machines** |
| `frozen.retrieval` | Default top_k |
| `experiments.retrieval` | Strategy (`dense` or `hybrid`), fusion method (`rrf` or `weighted`) |
| `experiments.reranking` | Enable/disable reranker, model choice, post-rerank top_k |
| `experiments.chunk_window` | Context window expansion around retrieved chunks |
| `qdrant` | DB URL, HNSW index parameters |
| `models_registry` | All supported embedding models and their config |

**Important:** Changing `chunk_size` or `overlap` changes the collection name, making results incomparable with prior runs.

---

## Output files

| File | Location | Description |
|------|----------|-------------|
| `<run_id>_summary.json` | `results_distributed/` | Per-run metrics (MRR, NDCG, Hit@k, latency) |
| `final_aggregated_summary.json` | `results_distributed/` | Combined summary across all evaluated question files |
| `index_progress_<collection>.json` | `~/.cache/rq1_draft/runtime/` | Live indexing progress; safe to inspect during a run |
| `<run_id>_completed_questions.jsonl` | `~/.cache/rq1_draft/runtime/evaluation_state/` | Per-question checkpoint; enables resuming interrupted evaluations |
| `processed_corpus_<hash>.pkl` | `~/.cache/rq1_draft/runtime/` | Cached preprocessed corpus; invalidated automatically on config change |



## Retrieval pipeline options

The pipeline supports three modes, all configured in `experiments.retrieval`:

- **Dense only** (`strategy: dense`): Standard vector search via Qdrant HNSW.
- **Hybrid BM25** (`strategy: hybrid`, `sparse_strategy: bm25`): Fuses dense vectors with BM25 keyword scores using RRF or weighted sum. BM25 index is built in memory from the preprocessed corpus cache.
- **Hybrid BGE-M3 sparse** (`strategy: hybrid`, `sparse_strategy: bge_m3_sparse`): Uses the lexical head of BGE-M3 to score dense candidates without a separate sparse index.

Optional post-retrieval stages:

- **Reranking** (`reranking.enabled: true`): Runs a cross-encoder reranker on the top candidates. Supported rerankers: `bge-reranker-v2-m3`, `qwen3-reranker-0.6b`, `jina-reranker-v2`.
- **Chunk window expansion** (`chunk_window.enabled: true`): Expands each retrieved chunk with its neighboring chunks before presenting to the reranker.

---

## Reproducing results

To reproduce an experiment exactly:

1. Use the same `frozen.*` settings (chunk size, overlap, preprocessing flags).
2. Use the same corpus version (same files, same `dataset.name`).
3. Use the same question file.
4. Choose the model and language, then run index-only followed by eval-only.

The `frozen` section of each summary file records the exact settings used during that run.


## Development notes

- The `distributed_eval/` package is the reusable core; `run_experiment.py` is a thin orchestration shell.
- Adding a new embedding provider: implement `BaseEmbedder` in `embedders/`, then register it in `embedding_wrappers.create_embedder()`.
- Adding a new reranker model: add an entry to `SentenceTransformersReranker.MODEL_MAP` in `experiment_pipeline.py`.
- Results in `results_distributed/` and the model/corpus cache in `~/.cache/rq1_draft/` are gitignored.
