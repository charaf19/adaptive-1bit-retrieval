# experiments/sensitivity_analysis.py
import time
from statistics import median
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from src.engine import RetrievalEngine


def _make_query_set(engine: RetrievalEngine, n=300):
    qs = []
    for i, text in enumerate(engine.texts):
        if len(qs) >= n:
            break
        if "\n" in text[:150]:
            q = text.split("\n", 1)[0].replace("Title: ", "").strip()
        else:
            q = " ".join(text.split()[:10]).strip()
        if len(q) >= 6:
            qs.append((engine.ids[i], q))
    return qs


def run_sensitivity(domain: str):
    print(f"\n--- 🧪 Sensitivity Analysis ({domain.upper()}) ---")
    engine = RetrievalEngine(domain)

    test_set = _make_query_set(engine, n=config.EVAL_N_QUERIES)
    if not test_set:
        print("⚠️ No valid queries.")
        return

    k_vals = config.KRECALL_LIST
    rows = []

    true_ids = [tid for tid, _ in test_set]
    queries = [q for _, q in test_set]

    # Embed once (timed)
    t0 = time.perf_counter()
    q_vecs = engine.embed_queries(queries, show_progress=True)
    embed_ms = (time.perf_counter() - t0) / len(q_vecs) * 1000.0
    q_bins = engine.binarize(q_vecs)

    def rerank_ids(cand_I, topk=10):
        out = []
        for qi in range(len(q_vecs)):
            cand = cand_I[qi]
            cand = cand[cand >= 0]
            if len(cand) == 0:
                out.append([])
                continue
            cand_vecs = engine.float_vectors[cand]
            scores = (q_vecs[qi:qi+1] @ cand_vecs.T)[0]
            order = np.argsort(scores)[-min(topk, len(scores)):][::-1]
            out.append([engine.ids[int(cand[j])] for j in order])
        return out

    for k in k_vals:
        # time binary search
        search_runs = []
        for _ in range(max(1, config.TIMING_REPEATS)):
            t0 = time.perf_counter()
            I = engine.index.search(q_bins, int(k))[1]
            search_runs.append((time.perf_counter() - t0) / len(q_bins) * 1000.0)
        search_ms = float(median(search_runs))

        # time rerank
        rerank_runs = []
        reranked = None
        for _ in range(max(1, config.TIMING_REPEATS)):
            t0 = time.perf_counter()
            reranked = rerank_ids(I, topk=10)
            rerank_runs.append((time.perf_counter() - t0) / len(q_vecs) * 1000.0)
        rerank_ms = float(median(rerank_runs))

        # recall
        hits = 0
        for t, ids in zip(true_ids, reranked):
            if t in ids[:10]:
                hits += 1
        recall = hits / len(true_ids) * 100.0

        total_ms = embed_ms + search_ms + rerank_ms
        rows.append({
            "domain": domain,
            "method": "Binary+Rerank (ours)",
            "k_recall": int(k),
            "k_rerank": 10,
            "recall_at_10": recall,
            "embed_ms": embed_ms,
            "binary_search_ms": search_ms,
            "rerank_ms": rerank_ms,
            "total_ms": total_ms,
            "n_queries": len(true_ids),
        })
        print(f"k={k:>3} -> Recall@10={recall:5.1f}% | total={total_ms:6.2f} ms (embed={embed_ms:5.2f}, bin={search_ms:5.2f}, rerank={rerank_ms:5.2f})")

    df = pd.DataFrame(rows)
    out_csv = config.LOGS_DIR / f"sensitivity_{domain}.csv"
    df.to_csv(out_csv, index=False)
    print(f"✅ Saved: {out_csv.name}")

    # Figure
    fig_path = config.FIGURES_DIR / f"sensitivity_{domain}.png"
    plt.figure(figsize=(6, 4))
    plt.plot(df["k_recall"], df["recall_at_10"], marker="o")
    plt.xlabel("k (binary recall depth)")
    plt.ylabel("Recall@10 (%)")
    plt.title(f"Sensitivity ({domain.upper()}) — Binary+Rerank (ours)")
    plt.grid(True, alpha=0.3)
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"✅ Saved: {fig_path.name}")


if __name__ == "__main__":
    for d in ("books", "arxiv"):
        run_sensitivity(d)
