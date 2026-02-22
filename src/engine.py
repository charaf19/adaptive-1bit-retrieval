# src/engine.py
from __future__ import annotations
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import config
from src.utils import maybe_prefix, quantize_standard, quantize_adaptive


class RetrievalEngine:
    def __init__(self, domain: str):
        if domain not in {"books", "arxiv"}:
            raise ValueError(f"Invalid domain '{domain}'. Must be 'books' or 'arxiv'.")
        self.domain = domain

        print(f"   🔎 Loading Retrieval Engine for: {domain.upper()} ...")

        # Embedding model
        self.model = SentenceTransformer(config.EMBEDDING_MODEL, trust_remote_code=True)

        # Artifacts
        index_path = config.ARTIFACTS_DIR / f"{domain}.index"
        float_path = config.ARTIFACTS_DIR / f"{domain}_float.npy"
        text_path = config.ARTIFACTS_DIR / f"{domain}_texts.pkl"
        thresholds_path = config.ARTIFACTS_DIR / f"{domain}_thresholds.npy"

        if not index_path.exists() or not float_path.exists() or not text_path.exists():
            raise FileNotFoundError(
                f"Artifacts missing for {domain}. Run: python -m src.indexer"
            )

        self.index = faiss.read_index_binary(str(index_path))
        self.float_vectors = np.load(float_path, mmap_mode="r")

        with open(text_path, "rb") as f:
            data = pickle.load(f)
            self.texts = data["texts"]
            self.ids = data["ids"]

        self.thresholds = None
        if thresholds_path.exists():
            self.thresholds = np.load(thresholds_path, mmap_mode="r")

    def embed_query(self, query: str) -> np.ndarray:
        q = maybe_prefix(query, "query")
        q_vec = self.model.encode([q], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q_vec)
        return q_vec

    def embed_queries(self, queries: list[str], batch_size: int | None = None, show_progress: bool = True) -> np.ndarray:
        """Embed a batch of queries (normalized float32)."""
        bs = int(batch_size or config.BATCH_SIZE)
        qs = [maybe_prefix(q, "query") for q in queries]
        q_vecs = self.model.encode(
            qs,
            convert_to_numpy=True,
            batch_size=bs,
            show_progress_bar=show_progress,
        ).astype(np.float32)
        faiss.normalize_L2(q_vecs)
        return q_vecs

    def binarize(self, vecs: np.ndarray) -> np.ndarray:
        if self.thresholds is not None:
            return quantize_adaptive(vecs, self.thresholds)
        return quantize_standard(vecs)

    def search(self, query: str, k_recall: int = None, k_rerank: int = None):
        k_recall = int(k_recall or config.DEFAULT_K_RECALL)
        k_rerank = int(k_rerank or config.DEFAULT_K_RERANK)

        q_float = self.embed_query(query)
        return self.search_from_vec(q_float, k_recall=k_recall, k_rerank=k_rerank)

    def search_from_vec(self, q_float: np.ndarray, k_recall: int = None, k_rerank: int = None):
        """Run (ours) retrieval using a pre-computed query embedding."""
        k_recall = int(k_recall or config.DEFAULT_K_RECALL)
        k_rerank = int(k_rerank or config.DEFAULT_K_RERANK)

        q_binary = self.binarize(q_float)
        candidates = self.search_candidates_binary(q_binary, k_recall)
        if len(candidates) == 0:
            return []

        return self.rerank_candidates(q_float, candidates, k_rerank)

    def search_candidates_binary(self, q_binary: np.ndarray, k_recall: int) -> np.ndarray:
        """Return candidate indices from the binary index (stage 1)."""
        _, I = self.index.search(q_binary, int(k_recall))
        cand = I[0]
        cand = cand[cand >= 0]
        return cand

    def rerank_candidates(self, q_float: np.ndarray, candidates: np.ndarray, k_rerank: int) -> list[dict]:
        """Rerank candidates by cosine similarity (inner product on normalized vectors)."""
        cand_vecs = self.float_vectors[candidates]
        scores = (q_float @ cand_vecs.T)[0]
        top_local = np.argsort(scores)[-min(int(k_rerank), len(scores)):][::-1]

        results = []
        for idx in top_local:
            global_id = int(candidates[idx])
            results.append({
                "id": self.ids[global_id],
                "score": float(scores[idx]),
                "text": self.texts[global_id],
            })
        return results
