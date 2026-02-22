# experiments/benchmark_baselines.py
"""Baseline comparisons with apples-to-apples latency.

Produces TWO tables per domain:
  1) search-only: index search time only (embedding excluded)
  2) end-to-end: embed + search + (optional) rerank

Includes control baselines such as Float32+Rerank.
Any method that is part of our proposal is tagged with "(ours)".
"""

from __future__ import annotations

import time
from statistics import median
import numpy as np
import pandas as pd
import faiss

import config
from src.engine import RetrievalEngine


def _make_query_set(engine: RetrievalEngine, n: int):
    """Create pseudo-queries derived from documents (self-hit evaluation)."""
    qs = []
    for i, text in enumerate(engine.texts):
        if len(qs) >= n:
            break
        # try to use title-like prefix if present
        if "\n" in text[:200]:
            q = text.split("\n", 1)[0].replace("Title:", "").strip()
        else:
            q = " ".join(text.split()[:12]).strip()
        if len(q) >= 6:
            qs.append((engine.ids[i], q))
    return qs


def _recall_at_k(true_ids, retrieved_ids_list, k: int) -> float:
    hits = 0
    for t, ids in zip(true_ids, retrieved_ids_list):
        if t in ids[:k]:
            hits += 1
    return hits / max(len(true_ids), 1) * 100.0


def _time_search(fn, n_queries: int) -> float:
    """Return median per-query latency (ms) over TIMING_REPEATS."""
    per_run = []
    for _ in range(max(1, config.TIMING_REPEATS)):
        t0 = time.perf_counter()
        fn()
        per_run.append((time.perf_counter() - t0) / n_queries * 1000.0)
    return float(median(per_run))


def _rerank_ids(engine: RetrievalEngine, q_vecs: np.ndarray, cand_I: np.ndarray, topk: int):
    """Rerank candidate indices by cosine similarity and return doc ids."""
    ids = []
    rerank_ms_runs = []

    for _ in range(max(1, config.TIMING_REPEATS)):
        t0 = time.perf_counter()
        out = []
        for qi in range(len(q_vecs)):
            cand = cand_I[qi]
            cand = cand[cand >= 0]
            if len(cand) == 0:
                out.append([])
                continue
            cand_vecs = engine.float_vectors[cand]
            scores = (q_vecs[qi : qi + 1] @ cand_vecs.T)[0]
            order = np.argsort(scores)[-min(topk, len(scores)) :][::-1]
            out.append([engine.ids[int(cand[j])] for j in order])
        rerank_ms_runs.append((time.perf_counter() - t0) / len(q_vecs) * 1000.0)
        ids = out

    return ids, float(median(rerank_ms_runs))


def run_baselines(domain: str, n_queries: int | None = None, topk: int | None = None):
    n_queries = int(n_queries or config.EVAL_N_QUERIES)
    topk = int(topk or config.EVAL_TOPK)

    print(f"\n--- 🧪 Baselines Benchmark ({domain.upper()}) ---")
    engine = RetrievalEngine(domain)

    # Load pre-built indices from artifacts (built by src.indexer)
    flat = faiss.read_index(str(config.ARTIFACTS_DIR / f"{domain}_flat.index"))
    pq = faiss.read_index(str(config.ARTIFACTS_DIR / f"{domain}_pq_m{config.PQ_M}.index"))
    hnsw = faiss.read_index(str(config.ARTIFACTS_DIR / f"{domain}_hnsw_m{config.HNSW_M}.index"))
    ivf_flat = faiss.read_index(str(config.ARTIFACTS_DIR / f"{domain}_ivf_flat_nlist{config.IVF_NLIST}.index"))
    ivf_pq = faiss.read_index(str(config.ARTIFACTS_DIR / f"{domain}_ivf_pq_nlist{config.IVF_NLIST}_m{config.PQ_M}.index"))

    # Set search parameters
    if hasattr(hnsw, "hnsw"):
        hnsw.hnsw.efSearch = int(max(config.HNSW_EFSEARCH_LIST) if config.HNSW_EFSEARCH_LIST else 64)
    if hasattr(ivf_flat, "nprobe"):
        ivf_flat.nprobe = int(config.IVF_NPROBE)
    if hasattr(ivf_pq, "nprobe"):
        ivf_pq.nprobe = int(config.IVF_NPROBE)

    queries = _make_query_set(engine, n=n_queries)
    true_ids = [tid for tid, _ in queries]
    qs = [q for _, q in queries]

    # Embed once (timed) => used for ALL methods
    t0 = time.perf_counter()
    q_vecs = engine.embed_queries(qs, show_progress=True)
    embed_ms = (time.perf_counter() - t0) / len(q_vecs) * 1000.0

    # Quantize once
    q_bins = engine.binarize(q_vecs)

    # Helper: map FAISS indices to doc ids
    def to_ids(I: np.ndarray):
        return [[engine.ids[int(i)] for i in row if i >= 0] for row in I]

    # ------------------------
    # SEARCH-ONLY table
    # ------------------------
    rows_search = []

    # Float32 exact
    lat = _time_search(lambda: flat.search(q_vecs, topk), len(q_vecs))
    I = flat.search(q_vecs, topk)[1]
    rows_search.append({
        "Domain": domain,
        "Method": "Float32 (baseline)",
        "Recall@10": _recall_at_k(true_ids, to_ids(I), 10),
        "Search_ms": lat,
        "TopK": topk,
        "N_queries": len(q_vecs),
    })

    # PQ
    lat = _time_search(lambda: pq.search(q_vecs, topk), len(q_vecs))
    I = pq.search(q_vecs, topk)[1]
    rows_search.append({
        "Domain": domain,
        "Method": f"PQ (baseline, m={config.PQ_M})",
        "Recall@10": _recall_at_k(true_ids, to_ids(I), 10),
        "Search_ms": lat,
        "TopK": topk,
        "N_queries": len(q_vecs),
    })

    # HNSW
    lat = _time_search(lambda: hnsw.search(q_vecs, topk), len(q_vecs))
    I = hnsw.search(q_vecs, topk)[1]
    rows_search.append({
        "Domain": domain,
        "Method": f"HNSW (baseline, m={config.HNSW_M}, ef={getattr(hnsw.hnsw, 'efSearch', 'NA')})",
        "Recall@10": _recall_at_k(true_ids, to_ids(I), 10),
        "Search_ms": lat,
        "TopK": topk,
        "N_queries": len(q_vecs),
    })

    # IVF-Flat
    lat = _time_search(lambda: ivf_flat.search(q_vecs, topk), len(q_vecs))
    I = ivf_flat.search(q_vecs, topk)[1]
    rows_search.append({
        "Domain": domain,
        "Method": f"IVF-Flat (baseline, nlist={config.IVF_NLIST}, nprobe={getattr(ivf_flat,'nprobe','NA')})",
        "Recall@10": _recall_at_k(true_ids, to_ids(I), 10),
        "Search_ms": lat,
        "TopK": topk,
        "N_queries": len(q_vecs),
    })

    # IVF-PQ
    lat = _time_search(lambda: ivf_pq.search(q_vecs, topk), len(q_vecs))
    I = ivf_pq.search(q_vecs, topk)[1]
    rows_search.append({
        "Domain": domain,
        "Method": f"IVF-PQ (baseline, nlist={config.IVF_NLIST}, nprobe={getattr(ivf_pq,'nprobe','NA')}, m={config.PQ_M})",
        "Recall@10": _recall_at_k(true_ids, to_ids(I), 10),
        "Search_ms": lat,
        "TopK": topk,
        "N_queries": len(q_vecs),
    })

    # BinaryFlat (candidate generator)
    lat = _time_search(lambda: engine.index.search(q_bins, topk), len(q_bins))
    I = engine.index.search(q_bins, topk)[1]
    rows_search.append({
        "Domain": domain,
        "Method": "BinaryFlat (ours candidate)",
        "Recall@10": _recall_at_k(true_ids, to_ids(I), 10),
        "Search_ms": lat,
        "TopK": topk,
        "N_queries": len(q_vecs),
    })

    df_search = pd.DataFrame(rows_search)
    out_search = config.LOGS_DIR / f"baseline_search_{domain}.csv"
    df_search.to_csv(out_search, index=False)
    print("\n[Search-only] (embedding excluded)")
    print(df_search.to_markdown(index=False))
    print(f"✅ Saved: {out_search.name}")

    # ------------------------
    # END-TO-END table (embed + search + optional rerank)
    # ------------------------
    rows_e2e = []

    def add_e2e(method: str, search_ms: float, rerank_ms: float, recall10: float):
        rows_e2e.append({
            "Domain": domain,
            "Method": method,
            "Recall@10": recall10,
            "Embed_ms": embed_ms,
            "Search_ms": search_ms,
            "Rerank_ms": rerank_ms,
            "Total_ms": embed_ms + search_ms + rerank_ms,
            "TopK": topk,
            "Candidates": int(config.RERANK_CANDIDATES),
            "N_queries": len(q_vecs),
        })

    # Helpers to build candidate sets and rerank
    def candidates_from(index, k: int):
        return index.search(q_vecs, k)[1]

    # Float32 (no rerank)
    search_ms = float(df_search[df_search["Method"] == "Float32 (baseline)"]["Search_ms"].iloc[0])
    I = flat.search(q_vecs, topk)[1]
    add_e2e("Float32 (baseline)", search_ms, 0.0, _recall_at_k(true_ids, to_ids(I), 10))

    # Float32 + rerank (control baseline)
    search_ms = _time_search(lambda: flat.search(q_vecs, config.RERANK_CANDIDATES), len(q_vecs))
    I_cand = candidates_from(flat, config.RERANK_CANDIDATES)
    ids_rerank, rerank_ms = _rerank_ids(engine, q_vecs, I_cand, topk)
    add_e2e("Float32+Rerank (control)", search_ms, rerank_ms, _recall_at_k(true_ids, ids_rerank, 10))

    # PQ + rerank
    search_ms = _time_search(lambda: pq.search(q_vecs, config.RERANK_CANDIDATES), len(q_vecs))
    I_cand = candidates_from(pq, config.RERANK_CANDIDATES)
    ids_rerank, rerank_ms = _rerank_ids(engine, q_vecs, I_cand, topk)
    add_e2e(f"PQ+Rerank (baseline, m={config.PQ_M})", search_ms, rerank_ms, _recall_at_k(true_ids, ids_rerank, 10))

    # HNSW + rerank
    search_ms = _time_search(lambda: hnsw.search(q_vecs, config.RERANK_CANDIDATES), len(q_vecs))
    I_cand = candidates_from(hnsw, config.RERANK_CANDIDATES)
    ids_rerank, rerank_ms = _rerank_ids(engine, q_vecs, I_cand, topk)
    add_e2e(f"HNSW+Rerank (baseline, m={config.HNSW_M})", search_ms, rerank_ms, _recall_at_k(true_ids, ids_rerank, 10))

    # IVF-PQ + rerank
    search_ms = _time_search(lambda: ivf_pq.search(q_vecs, config.RERANK_CANDIDATES), len(q_vecs))
    I_cand = candidates_from(ivf_pq, config.RERANK_CANDIDATES)
    ids_rerank, rerank_ms = _rerank_ids(engine, q_vecs, I_cand, topk)
    add_e2e(
        f"IVF-PQ+Rerank (baseline, nlist={config.IVF_NLIST}, nprobe={getattr(ivf_pq,'nprobe','NA')})",
        search_ms,
        rerank_ms,
        _recall_at_k(true_ids, ids_rerank, 10),
    )

    # Binary + rerank (ours)
    search_ms = _time_search(lambda: engine.index.search(q_bins, config.RERANK_CANDIDATES), len(q_bins))
    I_cand = engine.index.search(q_bins, config.RERANK_CANDIDATES)[1]
    # rerank expects candidate indices; already indices in I_cand
    ids_rerank, rerank_ms = _rerank_ids(engine, q_vecs, I_cand, topk)
    add_e2e(f"Binary+Rerank (ours, k={config.RERANK_CANDIDATES})", search_ms, rerank_ms, _recall_at_k(true_ids, ids_rerank, 10))

    df_e2e = pd.DataFrame(rows_e2e)
    out_e2e = config.LOGS_DIR / f"baseline_end2end_{domain}.csv"
    df_e2e.to_csv(out_e2e, index=False)
    print("\n[End-to-end] (embed + search + rerank)")
    print(df_e2e.to_markdown(index=False))
    print(f"✅ Saved: {out_e2e.name}")


if __name__ == "__main__":
    for d in ("books", "arxiv"):
        run_baselines(d)
