from __future__ import annotations

from pathlib import Path

import pandas as pd


def _load_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def generate_charts(df: pd.DataFrame, out_dir: str) -> None:
    plt = _load_matplotlib()
    if plt is None or df.empty:
        return

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Recall@k comparison
    recall_cols = ["recall@1", "recall@3", "recall@5"]
    if all(c in df.columns for c in recall_cols):
        plot_df = df[["model_name", "language"] + recall_cols].copy()
        plot_df["label"] = plot_df["model_name"] + "_" + plot_df["language"]
        ax = plot_df.set_index("label")[recall_cols].plot(kind="bar", figsize=(10, 5))
        ax.set_title("Recall@k comparison")
        ax.set_ylabel("Score")
        plt.tight_layout()
        plt.savefig(out / "recall_at_k_comparison.png", dpi=150)
        plt.close()

    # MRR comparison
    if {"model_name", "language", "mrr"} <= set(df.columns):
        plot_df = df[["model_name", "language", "mrr"]].copy()
        plot_df["label"] = plot_df["model_name"] + "_" + plot_df["language"]
        ax = plot_df.set_index("label")["mrr"].plot(kind="bar", figsize=(10, 5))
        ax.set_title("MRR comparison")
        ax.set_ylabel("MRR")
        plt.tight_layout()
        plt.savefig(out / "mrr_comparison.png", dpi=150)
        plt.close()

    # Latency distribution
    if "avg_latency_ms" in df.columns:
        ax = df["avg_latency_ms"].plot(kind="hist", bins=20, figsize=(8, 5))
        ax.set_title("Latency distribution (avg per run)")
        ax.set_xlabel("Latency (ms)")
        plt.tight_layout()
        plt.savefig(out / "latency_distribution.png", dpi=150)
        plt.close()

    # Per-language performance (Recall@5)
    if {"language", "recall@5"} <= set(df.columns):
        ax = df.groupby("language")["recall@5"].mean().plot(kind="bar", figsize=(6, 4))
        ax.set_title("Per-language Recall@5")
        ax.set_ylabel("Recall@5")
        plt.tight_layout()
        plt.savefig(out / "per_language_performance.png", dpi=150)
        plt.close()

    # Failure analysis
    if {"model_name", "language", "failed_questions", "total_questions"} <= set(df.columns):
        tmp = df.copy()
        tmp["failure_rate"] = tmp["failed_questions"] / tmp["total_questions"].clip(lower=1)
        tmp["label"] = tmp["model_name"] + "_" + tmp["language"]
        ax = tmp.set_index("label")["failure_rate"].plot(kind="bar", figsize=(10, 5))
        ax.set_title("Failure analysis (failed/total)")
        ax.set_ylabel("Failure rate")
        plt.tight_layout()
        plt.savefig(out / "failure_analysis.png", dpi=150)
        plt.close()

