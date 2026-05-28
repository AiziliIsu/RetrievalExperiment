from __future__ import annotations

import argparse
import json
from pathlib import Path

from distributed_eval.summary_aggregation import build_aggregated_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate run summaries into one JSON report.")
    parser.add_argument("--results-dir", default="results_distributed")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--collection-name", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    summaries = []
    for path in sorted(results_dir.glob("*_summary.json")):
        if path.name.startswith("final_"):
            continue
        with path.open("r", encoding="utf-8") as f:
            summaries.append(json.load(f))

    if not summaries:
        raise RuntimeError(f"No summary files found in {results_dir}")

    dataset_name = args.dataset_name or str(summaries[0].get("run_id", "")).split("_")[0]
    model_name = args.model_name or str(summaries[0].get("model_name", ""))
    collection_name = args.collection_name or str(summaries[0].get("collection_name", ""))

    payload = build_aggregated_summary(
        summaries,
        dataset_name=dataset_name,
        model_name=model_name,
        collection_name=collection_name,
    )
    output_path = Path(args.output) if args.output else results_dir / "final_aggregated_summary.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()