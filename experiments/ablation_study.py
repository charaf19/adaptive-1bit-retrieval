# experiments/ablation_study.py
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

import config
from src.utils import quantize_standard, compute_adaptive_thresholds, quantize_adaptive, maybe_prefix

MODELS_ALL = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "Nomic": "nomic-ai/nomic-embed-text-v1.5",
    # heavy (optional)
    "Mixedbread": "mixedbread-ai/mxbai-embed-large-v1",
}

def _selected_models():
    """Select models via env var BINARY_RAG_ABLATION_MODELS.

    Default: MiniLM,Nomic (fast). Add Mixedbread only if you really want it.
    Example: set BINARY_RAG_ABLATION_MODELS=MiniLM,Nomic,Mixedbread
    """
    sel = os.getenv("BINARY_RAG_ABLATION_MODELS", "MiniLM,Nomic")
    wanted = [x.strip() for x in sel.split(",") if x.strip()]
    out = {}
    for name in wanted:
        if name in MODELS_ALL:
            out[name] = MODELS_ALL[name]
    return out

import os


def run_ablation(n_docs: int | None = None, n_queries: int | None = None):
    print("--- 🧪 Ablation: Standard vs Adaptive Binary Quantization ---")

    n_docs = int(n_docs or os.getenv("BINARY_RAG_ABLATION_DOCS", "1500"))
    n_queries = int(n_queries or os.getenv("BINARY_RAG_ABLATION_QUERIES", "200"))

    input_path = config.PROCESSED_DIR / "books.jsonl"
    if not input_path.exists():
        raise FileNotFoundError("Missing data/books.jsonl. Run: python -m src.data_loader")

    texts, titles = [], []
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_docs:
                break
            item = json.loads(line)
            texts.append(item["text"])
            titles.append(item.get("title", ""))

    results = []
    models = _selected_models()
    if not models:
        raise ValueError("No ablation models selected. Check BINARY_RAG_ABLATION_MODELS.")

    for model_short, model_id in models.items():
        print(f"\nModel: {model_short}")
        model = SentenceTransformer(model_id, trust_remote_code=True)

        doc_inputs = []
        for t in texts:
            if "nomic" in model_id.lower():
                doc_inputs.append("search_document: " + t)
            else:
                doc_inputs.append(t)

        doc_vecs = model.encode(
            doc_inputs,
            convert_to_numpy=True,
            batch_size=config.BATCH_SIZE,
            show_progress_bar=True,
        ).astype(np.float32)
        faiss.normalize_L2(doc_vecs)

        query_idx = [i for i, t in enumerate(titles) if t and len(t) > 10][:n_queries]
        q_inputs = []
        for i in query_idx:
            if "nomic" in model_id.lower():
                q_inputs.append("search_query: " + titles[i])
            else:
                q_inputs.append(titles[i])

        q_vecs = model.encode(
            q_inputs,
            convert_to_numpy=True,
            batch_size=config.BATCH_SIZE,
            show_progress_bar=False,
        ).astype(np.float32)
        faiss.normalize_L2(q_vecs)

        # Float baseline
        idx_f = faiss.IndexFlatIP(doc_vecs.shape[1]); idx_f.add(doc_vecs)
        _, I_f = idx_f.search(q_vecs, 10)
        hits_f = sum(1 for j, idxs in enumerate(I_f) if query_idx[j] in idxs)
        recall_f = hits_f / len(query_idx) * 100

        # Standard binary
        bin_doc_std = quantize_standard(doc_vecs)
        bin_q_std = quantize_standard(q_vecs)
        idx_b1 = faiss.IndexBinaryFlat(doc_vecs.shape[1]); idx_b1.add(bin_doc_std)
        _, I_b1 = idx_b1.search(bin_q_std, 10)
        hits_b1 = sum(1 for j, idxs in enumerate(I_b1) if query_idx[j] in idxs)
        recall_b1 = hits_b1 / len(query_idx) * 100

        # Adaptive binary
        thr = compute_adaptive_thresholds(doc_vecs)
        bin_doc_ad = quantize_adaptive(doc_vecs, thr)
        bin_q_ad = quantize_adaptive(q_vecs, thr)
        idx_b2 = faiss.IndexBinaryFlat(doc_vecs.shape[1]); idx_b2.add(bin_doc_ad)
        _, I_b2 = idx_b2.search(bin_q_ad, 10)
        hits_b2 = sum(1 for j, idxs in enumerate(I_b2) if query_idx[j] in idxs)
        recall_b2 = hits_b2 / len(query_idx) * 100

        results.append({
            "Model": model_short,
            "Float32 Recall@10(%)": round(recall_f, 2),
            "Standard Binary (ours) Recall@10(%)": round(recall_b1, 2),
            "Adaptive Binary (ours) Recall@10(%)": round(recall_b2, 2),
            "Adaptive Gain(%)": round(recall_b2 - recall_b1, 2),
            "Docs": int(n_docs),
            "Queries": int(len(query_idx)),
        })

    df = pd.DataFrame(results)
    out = config.LOGS_DIR / "ablation_adaptive_quantization.csv"
    df.to_csv(out, index=False)

    print("\n" + df.to_markdown(index=False))
    print(f"✅ Saved: {out.name}")


if __name__ == "__main__":
    run_ablation()
