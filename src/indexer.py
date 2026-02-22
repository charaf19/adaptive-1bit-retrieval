# src/indexer.py
import json
import pickle
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config
from src.utils import maybe_prefix, quantize_standard, compute_adaptive_thresholds, quantize_adaptive


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def run_indexing(domains=("books", "arxiv")):
    print(f"Loading embedding model: {config.EMBEDDING_MODEL} ...")
    model = SentenceTransformer(config.EMBEDDING_MODEL, trust_remote_code=True)

    # Summary metrics (domain-level)
    metrics = {}
    # Detailed per-index metrics (one row per index file)
    index_rows = []

    for domain in domains:
        print(f"\n--- 🧱 Indexing Domain: {domain.upper()} ---")
        input_path = config.PROCESSED_DIR / f"{domain}.jsonl"
        if not input_path.exists():
            print(f"⚠️  Missing {input_path}. Run: python -m src.data_loader")
            continue

        data = load_jsonl(input_path)
        ids = [d["id"] for d in data]
        clean_texts = [d["text"] for d in data]
        embed_texts = [maybe_prefix(t, "document") for t in clean_texts]

        print(f"Embedding {len(embed_texts)} items ...")
        t_embed0 = time.perf_counter()
        embeddings = model.encode(
            embed_texts,
            convert_to_numpy=True,
            batch_size=config.BATCH_SIZE,
            show_progress_bar=True
        ).astype(np.float32)
        faiss.normalize_L2(embeddings)
        embed_s = time.perf_counter() - t_embed0

        # Quantize
        thresholds_path = config.ARTIFACTS_DIR / f"{domain}_thresholds.npy"
        if config.QUANTIZATION_MODE == "adaptive":
            print("Quantizing to 1-bit (adaptive per-dimension median) ...")
            thresholds = compute_adaptive_thresholds(embeddings)
            np.save(thresholds_path, thresholds.astype(np.float32))
            embeddings_binary = quantize_adaptive(embeddings, thresholds)
        else:
            print("Quantizing to 1-bit (standard > 0) ...")
            if thresholds_path.exists():
                thresholds_path.unlink(missing_ok=True)
            embeddings_binary = quantize_standard(embeddings)

        # Save artifacts
        print("Saving artifacts ...")
        with open(config.ARTIFACTS_DIR / f"{domain}_texts.pkl", "wb") as f:
            pickle.dump({"ids": ids, "texts": clean_texts}, f)

        # Float vectors (memmap friendly)
        float_npy = config.ARTIFACTS_DIR / f"{domain}_float.npy"
        np.save(float_npy, embeddings)

        # ------------------------
        # Build & save FAISS indices for baselines
        # (search-only comparisons must be apples-to-apples)
        # ------------------------
        d = embeddings.shape[1]

        # 1) Binary (ours candidate generator)
        d_bits = d
        idx_bin = faiss.IndexBinaryFlat(d_bits)
        t0 = time.perf_counter(); idx_bin.add(embeddings_binary); add_s = time.perf_counter() - t0
        bin_path = config.ARTIFACTS_DIR / f"{domain}.index"
        faiss.write_index_binary(idx_bin, str(bin_path))
        index_rows.append({
            "domain": domain,
            "index": "BinaryFlat (ours)",
            "params": "d_bits=%d" % d_bits,
            "train_s": 0.0,
            "add_s": add_s,
            "build_s": add_s,
            "size_mb": bin_path.stat().st_size / 1024**2,
        })

        # 2) Float32 exact (baseline)
        idx_flat = faiss.IndexFlatL2(d)
        t0 = time.perf_counter(); idx_flat.add(embeddings); add_s = time.perf_counter() - t0
        flat_path = config.ARTIFACTS_DIR / f"{domain}_flat.index"
        faiss.write_index(idx_flat, str(flat_path))
        index_rows.append({
            "domain": domain,
            "index": "Float32 Flat (baseline)",
            "params": "IndexFlatL2",
            "train_s": 0.0,
            "add_s": add_s,
            "build_s": add_s,
            "size_mb": flat_path.stat().st_size / 1024**2,
        })

        # 3) PQ (baseline) — build either the default PQ_M or a sweep of m values
        pq_m_values = (config.PQ_M_LIST if config.BUILD_PQ_SWEEP else [config.PQ_M])
        pq_m_values = sorted(set([int(x) for x in pq_m_values] + [int(config.PQ_M)]))
        pq_paths = []
        for m in pq_m_values:
            pq_path = config.ARTIFACTS_DIR / f"{domain}_pq_m{m}.index"
            idx_pq = faiss.IndexPQ(d, m, config.PQ_NBITS)
            t0 = time.perf_counter(); idx_pq.train(embeddings); train_s = time.perf_counter() - t0
            t0 = time.perf_counter(); idx_pq.add(embeddings); add_s = time.perf_counter() - t0
            faiss.write_index(idx_pq, str(pq_path))
            pq_paths.append(pq_path)
            index_rows.append({
                "domain": domain,
                "index": "PQ (baseline)",
                "params": f"m={m},nbits={config.PQ_NBITS}",
                "train_s": train_s,
                "add_s": add_s,
                "build_s": train_s + add_s,
                "size_mb": pq_path.stat().st_size / 1024**2,
            })

        # 4) HNSW (baseline)
        hnsw_path = config.ARTIFACTS_DIR / f"{domain}_hnsw_m{config.HNSW_M}.index"
        idx_hnsw = faiss.IndexHNSWFlat(d, config.HNSW_M)
        idx_hnsw.hnsw.efConstruction = config.HNSW_EF_CONSTRUCTION
        t0 = time.perf_counter(); idx_hnsw.add(embeddings); add_s = time.perf_counter() - t0
        faiss.write_index(idx_hnsw, str(hnsw_path))
        index_rows.append({
            "domain": domain,
            "index": "HNSW (baseline)",
            "params": f"m={config.HNSW_M},efC={config.HNSW_EF_CONSTRUCTION}",
            "train_s": 0.0,
            "add_s": add_s,
            "build_s": add_s,
            "size_mb": hnsw_path.stat().st_size / 1024**2,
        })

        # 5) IVF-Flat (baseline) — optionally build an nlist sweep
        nlist_values = (config.IVF_NLIST_LIST if config.BUILD_IVF_NLIST_SWEEP else [config.IVF_NLIST])
        nlist_values = sorted(set([int(x) for x in nlist_values] + [int(config.IVF_NLIST)]))
        ivf_flat_paths = []
        for nlist in nlist_values:
            ivf_flat_path = config.ARTIFACTS_DIR / f"{domain}_ivf_flat_nlist{nlist}.index"
            quantizer = faiss.IndexFlatL2(d)
            idx_ivf_flat = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
            t0 = time.perf_counter(); idx_ivf_flat.train(embeddings); train_s = time.perf_counter() - t0
            t0 = time.perf_counter(); idx_ivf_flat.add(embeddings); add_s = time.perf_counter() - t0
            faiss.write_index(idx_ivf_flat, str(ivf_flat_path))
            ivf_flat_paths.append(ivf_flat_path)
            index_rows.append({
                "domain": domain,
                "index": "IVF-Flat (baseline)",
                "params": f"nlist={nlist}",
                "train_s": train_s,
                "add_s": add_s,
                "build_s": train_s + add_s,
                "size_mb": ivf_flat_path.stat().st_size / 1024**2,
            })

        # 6) IVF-PQ (baseline) — keep PQ_M fixed (paper default) but optionally sweep nlist
        ivf_pq_paths = []
        for nlist in nlist_values:
            ivf_pq_path = config.ARTIFACTS_DIR / f"{domain}_ivf_pq_nlist{nlist}_m{config.PQ_M}.index"
            quantizer2 = faiss.IndexFlatL2(d)
            idx_ivf_pq = faiss.IndexIVFPQ(quantizer2, d, nlist, config.PQ_M, config.PQ_NBITS)
            t0 = time.perf_counter(); idx_ivf_pq.train(embeddings); train_s = time.perf_counter() - t0
            t0 = time.perf_counter(); idx_ivf_pq.add(embeddings); add_s = time.perf_counter() - t0
            faiss.write_index(idx_ivf_pq, str(ivf_pq_path))
            ivf_pq_paths.append(ivf_pq_path)
            index_rows.append({
                "domain": domain,
                "index": "IVF-PQ (baseline)",
                "params": f"nlist={nlist},m={config.PQ_M},nbits={config.PQ_NBITS}",
                "train_s": train_s,
                "add_s": add_s,
                "build_s": train_s + add_s,
                "size_mb": ivf_pq_path.stat().st_size / 1024**2,
            })

        # ------------------------
        # Metrics (use on-disk index sizes for fair reporting)
        # ------------------------
        float_mb = float_npy.stat().st_size / 1024**2
        binary_mb = bin_path.stat().st_size / 1024**2
        # For sweeps, report the default PQ_M size as the primary PQ size
        pq_primary_path = config.ARTIFACTS_DIR / f"{domain}_pq_m{config.PQ_M}.index"
        pq_mb = pq_primary_path.stat().st_size / 1024**2 if pq_primary_path.exists() else (pq_paths[0].stat().st_size / 1024**2)
        hnsw_mb = hnsw_path.stat().st_size / 1024**2
        ivf_flat_primary = config.ARTIFACTS_DIR / f"{domain}_ivf_flat_nlist{config.IVF_NLIST}.index"
        ivf_pq_primary = config.ARTIFACTS_DIR / f"{domain}_ivf_pq_nlist{config.IVF_NLIST}_m{config.PQ_M}.index"
        ivf_flat_mb = ivf_flat_primary.stat().st_size / 1024**2 if ivf_flat_primary.exists() else ivf_flat_paths[0].stat().st_size / 1024**2
        ivf_pq_mb = ivf_pq_primary.stat().st_size / 1024**2 if ivf_pq_primary.exists() else ivf_pq_paths[0].stat().st_size / 1024**2

        metrics[domain] = {
            "float_mb": float_mb,
            "binary_mb": binary_mb,
            "pq_mb": pq_mb,
            "hnsw_mb": hnsw_mb,
            "ivf_flat_mb": ivf_flat_mb,
            "ivf_pq_mb": ivf_pq_mb,
            "compression_x": float_mb / max(binary_mb, 1e-9),
            "quantization_mode": config.QUANTIZATION_MODE,
            "pq_m": config.PQ_M,
            "pq_nbits": config.PQ_NBITS,
            "hnsw_m": config.HNSW_M,
            "ivf_nlist": config.IVF_NLIST,
            "embed_s": embed_s,
        }
        print(
            f"✅ Index sizes (MB): float={float_mb:.2f} | binary={binary_mb:.2f} | pq={pq_mb:.2f} | "
            f"hnsw={hnsw_mb:.2f} | ivf_flat={ivf_flat_mb:.2f} | ivf_pq={ivf_pq_mb:.2f}"
        )
        print(f"✅ Compression (float→binary): {metrics[domain]['compression_x']:.1f}×")

    # Save indexing metrics
    out_path = config.RESULTS_DIR / "indexing_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"📄 Saved: {out_path}")

    # Save detailed build/size metrics (per index)
    out_csv = config.LOGS_DIR / "index_build_metrics.csv"
    try:
        import pandas as pd

        pd.DataFrame(index_rows).to_csv(out_csv, index=False)
        print(f"📄 Saved: {out_csv}")
    except Exception as e:
        print(f"⚠️ Could not write index_build_metrics.csv: {e}")


if __name__ == "__main__":
    run_indexing()
