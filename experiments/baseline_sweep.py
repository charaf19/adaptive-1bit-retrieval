# experiments/baseline_sweep.py
"""Generate a *tuned* ANN baseline sweep and Pareto-ready logs.

Why this exists
--------------
Single-point baselines (e.g., IVF nprobe=10) are often under-tuned and can look unfair.
This script sweeps the key ANN hyperparameters and logs multiple operating points so
you can plot Recall@10 vs Search_ms and report the best baseline settings.

Outputs (per domain)
--------------------
results/logs/baseline_sweep_search_<domain>.csv

Notes
-----
- Search-only: embedding time is excluded (embedded once and reused for all methods).
- Any method that is part of our proposal is tagged with "(ours)".
"""

from __future__ import annotations

import time
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
import faiss

import config
from src.engine import RetrievalEngine


def _make_query_set(engine: RetrievalEngine, n: int):
    qs = []
    for i, text in enumerate(engine.texts):
        if len(qs) >= n:
            break
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
    per_run = []
    for _ in range(max(1, config.TIMING_REPEATS)):
        t0 = time.perf_counter()
        fn()
        per_run.append((time.perf_counter() - t0) / n_queries * 1000.0)
    return float(median(per_run))


def _to_ids(engine: RetrievalEngine, I: np.ndarray):
    return [[engine.ids[int(i)] for i in row if i >= 0] for row in I]


def _load_all(pattern: str) -> list[tuple[str, faiss.Index]]:
    out = []
    for p in sorted(config.ARTIFACTS_DIR.glob(pattern)):
        try:
            out.append((p.name, faiss.read_index(str(p))))
        except Exception:
            # skip incompatible / corrupted
            continue
    return out


def run_sweep(domain: str):
    topk = int(config.BASELINE_SWEEP_TOPK)
    n_queries = int(config.BASELINE_SWEEP_N_QUERIES)

    print(f"\n--- 📈 Baseline Sweep (Search-only) — {domain.upper()} ---")
    engine = RetrievalEngine(domain)

    # Load float32 exact (always present)
    flat = faiss.read_index(str(config.ARTIFACTS_DIR / f"{domain}_flat.index"))

    # Load sweep-capable indices
    pq_list = _load_all(f"{domain}_pq_m*.index")
    hnsw = faiss.read_index(str(config.ARTIFACTS_DIR / f"{domain}_hnsw_m{config.HNSW_M}.index"))
    ivf_flat_list = _load_all(f"{domain}_ivf_flat_nlist*.index")
    ivf_pq_list = _load_all(f"{domain}_ivf_pq_nlist*_m*.index")

    queries = _make_query_set(engine, n=n_queries)
    true_ids = [tid for tid, _ in queries]
    qs = [q for _, q in queries]

    # Embed once (excluded from search-only timing)
    q_vecs = engine.embed_queries(qs, show_progress=True)
    q_bins = engine.binarize(q_vecs)

    rows = []

    def add_row(method: str, params: str, I: np.ndarray, search_ms: float, k_used: int):
        rows.append({
            "Domain": domain,
            "Method": method,
            "Params": params,
            "Recall@10": _recall_at_k(true_ids, _to_ids(engine, I), 10),
            "Search_ms": float(search_ms),
            "TopK": int(topk),
            "K_used": int(k_used),
            "N_queries": len(q_vecs),
            "Timing_repeats": int(config.TIMING_REPEATS),
        })

    # ---- Float32 exact (baseline)
    search_ms = _time_search(lambda: flat.search(q_vecs, topk), len(q_vecs))
    I = flat.search(q_vecs, topk)[1]
    add_row("Float32 (baseline)", "IndexFlatL2", I, search_ms, topk)

    # ---- PQ sweep (baseline)
    for fname, pq in pq_list:
        # parse m from filename
        m = "?"
        if "_pq_m" in fname:
            try:
                m = int(fname.split("_pq_m", 1)[1].split(".index", 1)[0])
            except Exception:
                m = "?"
        search_ms = _time_search(lambda pq=pq: pq.search(q_vecs, topk), len(q_vecs))
        I = pq.search(q_vecs, topk)[1]
        add_row(f"PQ (baseline)", f"m={m},nbits={config.PQ_NBITS}", I, search_ms, topk)

    # ---- HNSW efSearch sweep (baseline)
    if hasattr(hnsw, "hnsw"):
        for ef in config.HNSW_EFSEARCH_LIST:
            hnsw.hnsw.efSearch = int(ef)
            search_ms = _time_search(lambda: hnsw.search(q_vecs, topk), len(q_vecs))
            I = hnsw.search(q_vecs, topk)[1]
            add_row("HNSW (baseline)", f"m={config.HNSW_M},efSearch={ef}", I, search_ms, topk)

    # ---- IVF-Flat sweep (baseline)
    for fname, ivf in ivf_flat_list:
        # parse nlist
        nlist = "?"
        if "_nlist" in fname:
            try:
                nlist = int(fname.split("_nlist", 1)[1].split(".index", 1)[0])
            except Exception:
                nlist = "?"
        for nprobe in config.IVF_NPROBE_LIST:
            if hasattr(ivf, "nprobe"):
                ivf.nprobe = int(nprobe)
            search_ms = _time_search(lambda ivf=ivf: ivf.search(q_vecs, topk), len(q_vecs))
            I = ivf.search(q_vecs, topk)[1]
            add_row("IVF-Flat (baseline)", f"nlist={nlist},nprobe={nprobe}", I, search_ms, topk)

    # ---- IVF-PQ sweep (baseline)
    for fname, ivf in ivf_pq_list:
        nlist = "?"; m = "?"
        try:
            nlist = int(fname.split("_nlist", 1)[1].split("_m", 1)[0])
        except Exception:
            pass
        try:
            m = int(fname.split("_m", 1)[1].split(".index", 1)[0])
        except Exception:
            pass
        for nprobe in config.IVF_NPROBE_LIST:
            if hasattr(ivf, "nprobe"):
                ivf.nprobe = int(nprobe)
            search_ms = _time_search(lambda ivf=ivf: ivf.search(q_vecs, topk), len(q_vecs))
            I = ivf.search(q_vecs, topk)[1]
            add_row("IVF-PQ (baseline)", f"nlist={nlist},nprobe={nprobe},m={m}", I, search_ms, topk)

    # ---- BinaryFlat candidate sweep (ours)
    for k in config.KRECALL_LIST:
        k = int(k)
        search_ms = _time_search(lambda k=k: engine.index.search(q_bins, k), len(q_bins))
        I = engine.index.search(q_bins, k)[1]
        # Recall@10 computed on top10 of returned list; search time corresponds to k
        add_row("BinaryFlat (ours)", f"k={k}", I, search_ms, k)

    df = pd.DataFrame(rows)
    out = config.LOGS_DIR / f"baseline_sweep_search_{domain}.csv"
    df.to_csv(out, index=False)
    print(df.sort_values(["Recall@10", "Search_ms"], ascending=[False, True]).head(20).to_markdown(index=False))
    print(f"✅ Saved: {out.name}")


if __name__ == "__main__":
    for d in ("books", "arxiv"):
        run_sweep(d)
