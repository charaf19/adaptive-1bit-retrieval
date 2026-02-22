# experiments/benchmark_efficiency.py
import time
from statistics import median
import numpy as np
import faiss
import pandas as pd

import config
from src.engine import RetrievalEngine


def _make_query_set(engine: RetrievalEngine, n=200):
    rng = np.random.default_rng(config.RANDOM_SEED)
    indices = rng.choice(len(engine.texts), size=min(n, len(engine.texts)), replace=False)

    queries = []
    for idx in indices:
        text = engine.texts[idx]
        if "\n" in text[:150]:
            q = text.split("\n", 1)[0].replace("Title: ", "").strip()
        else:
            q = " ".join(text.split()[:10]).strip()
        if len(q) >= 6:
            queries.append((idx, q))
    return queries


def run_benchmark(domain: str):
    print(f"\n--- 🧪 Efficiency Benchmark ({domain.upper()}) ---")

    float_path = config.ARTIFACTS_DIR / f"{domain}_float.npy"
    if not float_path.exists():
        print(f"❌ Missing artifacts for {domain}. Run: python -m src.indexer")
        return

    vectors = np.load(float_path)

    # Load indices built by src.indexer
    idx_float = faiss.read_index(str(config.ARTIFACTS_DIR / f"{domain}_flat.index"))
    idx_pq = faiss.read_index(str(config.ARTIFACTS_DIR / f"{domain}_pq_m{config.PQ_M}.index"))
    idx_hnsw = faiss.read_index(str(config.ARTIFACTS_DIR / f"{domain}_hnsw_m{config.HNSW_M}.index"))
    idx_ivf_flat = faiss.read_index(str(config.ARTIFACTS_DIR / f"{domain}_ivf_flat_nlist{config.IVF_NLIST}.index"))
    idx_ivf_pq = faiss.read_index(str(config.ARTIFACTS_DIR / f"{domain}_ivf_pq_nlist{config.IVF_NLIST}_m{config.PQ_M}.index"))
    idx_bin = faiss.read_index_binary(str(config.ARTIFACTS_DIR / f"{domain}.index"))

    if hasattr(idx_hnsw, "hnsw"):
        idx_hnsw.hnsw.efSearch = int(max(config.HNSW_EFSEARCH_LIST) if config.HNSW_EFSEARCH_LIST else 64)
    if hasattr(idx_ivf_flat, "nprobe"):
        idx_ivf_flat.nprobe = int(config.IVF_NPROBE)
    if hasattr(idx_ivf_pq, "nprobe"):
        idx_ivf_pq.nprobe = int(config.IVF_NPROBE)

    engine = RetrievalEngine(domain)
    queries = _make_query_set(engine, n=200)
    if not queries:
        print("⚠️ No valid queries.")
        return

    def time_index(search_fn):
        lat = []
        hits = 0
        for true_idx, _ in queries:
            qv = vectors[true_idx].reshape(1, -1)
            # small repeats for stability
            runs = []
            I0 = None
            for _ in range(max(1, config.TIMING_REPEATS)):
                start = time.perf_counter()
                I0 = search_fn(qv)
                runs.append((time.perf_counter() - start) * 1000.0)
            lat.append(median(runs))
            if true_idx in I0:
                hits += 1
        return float(np.mean(lat)), hits / len(queries) * 100

    lat_f, rec_f = time_index(lambda qv: idx_float.search(qv, 100)[1][0])
    lat_pq, rec_pq = time_index(lambda qv: idx_pq.search(qv, 100)[1][0])
    lat_h, rec_h = time_index(lambda qv: idx_hnsw.search(qv, 100)[1][0])
    lat_if, rec_if = time_index(lambda qv: idx_ivf_flat.search(qv, 100)[1][0])
    lat_ipq, rec_ipq = time_index(lambda qv: idx_ivf_pq.search(qv, 100)[1][0])

    # binary: packbits the float query
    def bin_search(qv):
            qb = engine.binarize(qv)
            return idx_bin.search(qb, 100)[1][0]
    lat_b, rec_b = time_index(bin_search)

    df = pd.DataFrame([
        {"Domain": domain, "Method": "Float32 (baseline)", "Recall@100(%)": rec_f, "Search_ms": lat_f},
        {"Domain": domain, "Method": f"PQ (baseline, m={config.PQ_M})", "Recall@100(%)": rec_pq, "Search_ms": lat_pq},
        {"Domain": domain, "Method": f"HNSW (baseline, m={config.HNSW_M})", "Recall@100(%)": rec_h, "Search_ms": lat_h},
        {"Domain": domain, "Method": f"IVF-Flat (baseline, nlist={config.IVF_NLIST})", "Recall@100(%)": rec_if, "Search_ms": lat_if},
        {"Domain": domain, "Method": f"IVF-PQ (baseline, nlist={config.IVF_NLIST}, m={config.PQ_M})", "Recall@100(%)": rec_ipq, "Search_ms": lat_ipq},
        {"Domain": domain, "Method": "BinaryFlat (ours candidate)", "Recall@100(%)": rec_b, "Search_ms": lat_b},
    ])

    out = config.LOGS_DIR / f"efficiency_{domain}.csv"
    df.to_csv(out, index=False)
    print(df.to_markdown(index=False))
    print(f"✅ Saved: {out.name}")


if __name__ == "__main__":
    for d in ("books", "arxiv"):
        run_benchmark(d)
