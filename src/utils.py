# src/utils.py
from __future__ import annotations
import numpy as np
import config

def maybe_prefix(text: str, kind: str) -> str:
    """Apply model-specific prefixes (Nomic uses search_query/search_document)."""
    model_id = config.EMBEDDING_MODEL.lower()
    if "nomic" in model_id:
        if kind == "query":
            return f"{config.QUERY_PREFIX}{text}"
        if kind == "document":
            return f"{config.DOC_PREFIX}{text}"
    return text

def quantize_standard(vectors: np.ndarray) -> np.ndarray:
    return np.packbits(vectors > 0, axis=1).astype(np.uint8)

def compute_adaptive_thresholds(vectors: np.ndarray) -> np.ndarray:
    # Median per dimension maximizes entropy (50/50 split).
    return np.median(vectors, axis=0)

def quantize_adaptive(vectors: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.packbits(vectors > thresholds, axis=1).astype(np.uint8)
