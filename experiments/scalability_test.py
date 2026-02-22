# experiments/scalability_test.py
"""Scalability: how search latency changes with corpus size N.

Search-only latency (embedding excluded) for:
  - Float32 exact (baseline)
  - HNSW (baseline)
  - BinaryFlat (ours candidate)
"""

from __future__ import annotations

import time
from statistics import median
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import faiss

import config


def _packbits_from_vectors(vectors: np.ndarray, thresholds: np.ndarray | None):
    if thresholds is None:
        return np.packbits(vectors > 0, axis=1).astype(np.uint8)
    return np.packbits(vectors > thresholds, axis=1).astype(np.uint8)


def _median_search_ms(fn, n_queries: int) -> float:
    runs = []
    for _ in range(max(1, config.TIMING_REPEATS)):
        t0 = time.perf_counter()
        fn()
        runs.append((time.perf_counter() - t0) / n_queries * 1000.0)
    return float(median(runs))


def run_scalability_test(domain: str = "books", steps: int = 5, n_queries: int = 200, topk: int = 10):
    print(f"--- 🧪 Scalability (search-only) ({domain.upper()}) ---")

    float_path = config.ARTIFACTS_DIR / f"{domain}_float.npy"
    if not float_path.exists():
        raise FileNotFoundError("Missing artifacts. Run: python -m src.indexer")

    full_vectors = np.load(float_path)
    d = full_vectors.shape[1]

    thresholds_path = config.ARTIFACTS_DIR / f"{domain}_thresholds.npy"
    thresholds = np.load(thresholds_path, mmap_mode="r") if thresholds_path.exists() else None
    full_binary = _packbits_from_vectors(full_vectors, thresholds)

    max_n = len(full_vectors)
    subsets = [max(1, int(max_n * (i + 1) / steps)) for i in range(steps)]

    q_vecs = full_vectors[: min(n_queries, max_n)].astype(np.float32)
    q_bins = full_binary[: min(n_queries, max_n)]

    rows = []
    for n in subsets:
        print(f"Testing N={n} ...")

        # Float32 exact
        idx_f = faiss.IndexFlatL2(d)
        idx_f.add(full_vectors[:n])

        # HNSW
        idx_h = faiss.IndexHNSWFlat(d, config.HNSW_M)
        idx_h.hnsw.efConstruction = config.HNSW_EF_CONSTRUCTION
        idx_h.hnsw.efSearch = int(max(config.HNSW_EFSEARCH_LIST) if config.HNSW_EFSEARCH_LIST else 64)
        idx_h.add(full_vectors[:n])

        # Binary
        idx_b = faiss.IndexBinaryFlat(d)
        idx_b.add(full_binary[:n])

        lat_f = _median_search_ms(lambda: idx_f.search(q_vecs, topk), len(q_vecs))
        lat_h = _median_search_ms(lambda: idx_h.search(q_vecs, topk), len(q_vecs))
        lat_b = _median_search_ms(lambda: idx_b.search(q_bins, topk), len(q_bins))

        rows.append({
            "Domain": domain,
            "N": int(n),
            "Float32_ms": lat_f,
            "HNSW_ms": lat_h,
            "Binary_ms": lat_b,
            "TopK": int(topk),
            "N_queries": int(len(q_vecs)),
        })

    df = pd.DataFrame(rows)
    out = config.LOGS_DIR / "scalability.csv"
    df.to_csv(out, index=False)
    print(f"✅ Saved: {out.name}")
    print(df.to_markdown(index=False))

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df["N"], df["Float32_ms"], marker="o", label="Float32 (baseline)")
    plt.plot(df["N"], df["HNSW_ms"], marker="^", label="HNSW (baseline)")
    plt.plot(df["N"], df["Binary_ms"], marker="s", label="BinaryFlat (ours candidate)")
    plt.xlabel("Database size (N)")
    plt.ylabel("Search latency per query (ms)")
    plt.title(f"Scalability (search-only) — {domain.upper()}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig = config.FIGURES_DIR / f"scalability_{domain}.png"
    plt.savefig(fig, dpi=300)
    plt.close()
    print(f"✅ Saved: {fig.name}")


if __name__ == "__main__":
    for d in ("books", "arxiv"):
        run_scalability_test(d)
