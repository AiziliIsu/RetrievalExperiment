from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Set


class ResultWriter:
    def __init__(self, output_dir: str, run_id: str, state_dir: str | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir = Path(state_dir) if state_dir else self.output_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.completed_path = self.state_dir / f"{run_id}_completed_questions.jsonl"
        self.summary_path = self.output_dir / f"{run_id}_summary.json"

    def load_completed(self) -> tuple:
        """Returns (completed_ids, prior_stats) accumulated from the checkpoint file.

        prior_stats keys: evaluated, hit1, hit3, hit5, hit10,
                          sum_reciprocal_rank, sum_ndcg, latencies.
        Rows that were saved without per-question scores (old format) are counted
        in completed_ids but contribute 0 to the accumulated stats — the caller
        should treat such a checkpoint as unrecoverable for metrics and re-run.
        """
        completed: Set[str] = set()
        stats: Dict = {
            "evaluated": 0,
            "hit1": 0,
            "hit3": 0,
            "hit5": 0,
            "hit10": 0,
            "sum_reciprocal_rank": 0.0,
            "sum_ndcg": 0.0,
            "latencies": [],
        }
        if not self.completed_path.exists():
            return completed, stats
        with self.completed_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid = str(row.get("question_id", "")).strip()
                if not qid:
                    continue
                completed.add(qid)
                if "reciprocal_rank" in row:
                    stats["evaluated"] += 1
                    stats["hit1"] += int(row.get("hit1", 0))
                    stats["hit3"] += int(row.get("hit3", 0))
                    stats["hit5"] += int(row.get("hit5", 0))
                    stats["hit10"] += int(row.get("hit10", 0))
                    stats["sum_reciprocal_rank"] += float(row.get("reciprocal_rank", 0.0))
                    stats["sum_ndcg"] += float(row.get("ndcg", 0.0))
                    stats["latencies"].append(float(row.get("latency_ms", 0.0)))
        return completed, stats

    def mark_question_completed(self, question_id: str, per_question: Dict) -> None:
        qid = str(question_id).strip()
        if not qid:
            return
        record = {"question_id": qid}
        record.update(per_question)
        with self.completed_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_summary(self, summary: Dict) -> None:
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

