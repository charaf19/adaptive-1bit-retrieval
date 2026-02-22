# experiments/model_robustness.py
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

import os
import config

MODELS_ALL = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "Nomic": "nomic-ai/nomic-embed-text-v1.5",
    # heavy (optional)
    "Mixedbread": "mixedbread-ai/mxbai-embed-large-v1",
}

def _selected_models():
    sel = os.getenv("BINARY_RAG_ROBUSTNESS_MODELS", "MiniLM,Nomic")
    wanted = [x.strip() for x in sel.split(",") if x.strip()]
    out = {}
    for name in wanted:
        if name in MODELS_ALL:
            out[name] = MODELS_ALL[name]
    return out

def run_comparison(domain: str, n_docs: int | None = None, n_trials: int | None = None):
    print(f"\n--- 🧪 Model Robustness ({domain.upper()}) ---")
    input_path = config.PROCESSED_DIR / f"{domain}.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing data for {domain}. Run: python -m src.data_loader")

    n_docs = int(n_docs or os.getenv("BINARY_RAG_ROBUSTNESS_DOCS", "1000"))
    n_trials = int(n_trials or os.getenv("BINARY_RAG_ROBUSTNESS_TRIALS", "200"))

    texts = []
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_docs:
                break
            texts.append(json.loads(line)["text"])

    rows = []
    models = _selected_models()
    if not models:
        raise ValueError("No robustness models selected. Check BINARY_RAG_ROBUSTNESS_MODELS.")

    for name, model_id in models.items():
        print(f"Testing {name} ...")
        model = SentenceTransformer(model_id, trust_remote_code=True)
        inputs = []
        for t in texts:
            if "nomic" in model_id.lower():
                inputs.append("search_document: " + t)
            else:
                inputs.append(t)

        vecs = model.encode(
            inputs,
            convert_to_numpy=True,
            batch_size=config.BATCH_SIZE,
            show_progress_bar=True,
        ).astype(np.float32)
        faiss.normalize_L2(vecs)

        idx_f = faiss.IndexFlatIP(vecs.shape[1]); idx_f.add(vecs)
        idx_b = faiss.IndexBinaryFlat(vecs.shape[1]); idx_b.add(np.packbits(vecs > 0, axis=1).astype(np.uint8))

        hits_f, hits_b = 0, 0
        for i in range(min(n_trials, len(vecs))):
            q = vecs[i:i+1]
            _, I_f = idx_f.search(q, 1)
            _, I_b = idx_b.search(np.packbits(q > 0, axis=1).astype(np.uint8), 10)
            if int(I_f[0][0]) == i:
                hits_f += 1
            if i in I_b[0]:
                hits_b += 1

        rows.append({
            "Domain": domain,
            "Model": name,
            "Float32_Top1_Hits": hits_f,
            "BinaryFlat_Top10_Hits (ours candidate)": hits_b,
            "Drop_Hits": hits_f - hits_b,
            "Docs": int(len(vecs)),
            "Trials": int(min(n_trials, len(vecs))),
        })

    df = pd.DataFrame(rows)
    out = config.LOGS_DIR / f"robustness_{domain}.csv"
    df.to_csv(out, index=False)
    print(df.to_markdown(index=False))
    print(f"✅ Saved: {out.name}")


if __name__ == "__main__":
    for d in ("books", "arxiv"):
        run_comparison(d)
