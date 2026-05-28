from __future__ import annotations

import datetime
from collections import defaultdict
from typing import Dict, Iterable, List


METRIC_KEYS = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg", "avg_latency_ms", "median_latency_ms"]


def _weighted_average(items: List[Dict], key: str, weight_key: str = "evaluated_questions") -> float:
    total_weight = 0.0
    total_value = 0.0
    for item in items:
        weight = float(item.get(weight_key, 0) or 0)
        value = item.get(key)
        if value is None or weight <= 0:
            continue
        total_weight += weight
        total_value += float(value) * weight
    return total_value / total_weight if total_weight else 0.0


def _aggregate_group(items: List[Dict]) -> Dict:
    total_questions = int(sum(int(item.get("total_questions", 0) or 0) for item in items))
    evaluated_questions = int(sum(int(item.get("evaluated_questions", 0) or 0) for item in items))
    failed_questions = int(sum(int(item.get("failed_questions", 0) or 0) for item in items))

    payload = {
        "total_question_files": len(items),
        "total_questions": total_questions,
        "evaluated_questions": evaluated_questions,
        "failed_questions": failed_questions,
        "question_files": [item.get("question_file") for item in items if item.get("question_file")],
        "run_ids": [item.get("run_id") for item in items if item.get("run_id")],
        "aggregation_method": "weighted_by_evaluated_questions",
    }

    for key in METRIC_KEYS:
        payload[key] = _weighted_average(items, key)
    return payload


def build_aggregated_summary(
    summaries: Iterable[Dict],
    *,
    dataset_name: str,
    model_name: str,
    collection_name: str,
) -> Dict:
    summary_list = list(summaries)
    by_language: Dict[str, List[Dict]] = defaultdict(list)
    for summary in summary_list:
        language = str(summary.get("language", "")).strip().lower() or "unknown"
        by_language[language].append(summary)

    return {
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "dataset_name": dataset_name,
        "model_name": model_name,
        "collection_name": collection_name,
        "summary_count": len(summary_list),
        "overall": _aggregate_group(summary_list) if summary_list else {},
        "by_language": {language: _aggregate_group(items) for language, items in sorted(by_language.items())},
        "runs": summary_list,
    }