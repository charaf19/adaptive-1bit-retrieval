# experiments/latency_profiling.py
import time
import numpy as np
import pandas as pd

import config
from src.engine import RetrievalEngine


def profile_latency(domain="books", k_recall=None, k_rerank=10, n_trials=30):
    print(f"--- ⏱️ Latency Profiling ({domain}) ---")
    engine = RetrievalEngine(domain)
    k_recall = int(k_recall or config.RERANK_CANDIDATES)
    query = "test query for profiling"

    rows = []
    for _ in range(n_trials):
        # Embed
        t0 = time.perf_counter()
        q_float = engine.embed_query(query)
        t1 = time.perf_counter()

        # Binary search
        q_binary = engine.binarize(q_float)
        _, I = engine.index.search(q_binary, k_recall)
        t2 = time.perf_counter()

        # Float rerank
        candidates = I[0]
        candidates = candidates[candidates >= 0]
        if len(candidates) == 0:
            continue
        cand_vecs = engine.float_vectors[candidates]
        _ = (q_float @ cand_vecs.T)[0]
        t3 = time.perf_counter()

        rows.append({
            "embed_ms": (t1 - t0) * 1000,
            "binary_ms": (t2 - t1) * 1000,
            "rerank_ms": (t3 - t2) * 1000,
            "total_ms": (t3 - t0) * 1000,
            "k_recall": k_recall,
            "k_rerank": k_rerank,
            "domain": domain,
        })

    df = pd.DataFrame(rows)
    out = config.LOGS_DIR / f"latency_profile_{domain}.csv"
    df.to_csv(out, index=False)

    mean = df[["embed_ms", "binary_ms", "rerank_ms", "total_ms"]].mean()
    print(mean.to_string())
    print(f"✅ Saved: {out.name}")


if __name__ == "__main__":
    for d in ("books", "arxiv"):
        profile_latency(d)
