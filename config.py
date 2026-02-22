# config.py
# Central configuration for the Binary-RAG project.
#
# You can override any path via environment variables.
#
# Example:
#   export BINARY_RAG_GENERATION_MODEL_PATH="models/tinyllama.gguf"
#   export BINARY_RAG_EMBEDDING_MODEL="nomic-ai/nomic-embed-text-v1.5"
#
import os
from pathlib import Path

# ----------------------------
# Project directories
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = Path(os.getenv("BINARY_RAG_DATA_DIR", PROJECT_ROOT / "data"))
PROCESSED_DIR = Path(os.getenv("BINARY_RAG_PROCESSED_DIR", DATA_DIR / "processed"))

ARTIFACTS_DIR = Path(os.getenv("BINARY_RAG_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
RESULTS_DIR = Path(os.getenv("BINARY_RAG_RESULTS_DIR", PROJECT_ROOT / "results"))
LOGS_DIR = RESULTS_DIR / "logs"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

MODELS_DIR = Path(os.getenv("BINARY_RAG_MODELS_DIR", PROJECT_ROOT / "models"))

for d in [DATA_DIR, PROCESSED_DIR, ARTIFACTS_DIR, RESULTS_DIR, LOGS_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Data ingestion
# ----------------------------
# Set to None to ingest until dataset ends (not recommended).
INGEST_LIMIT = int(os.getenv("BINARY_RAG_INGEST_LIMIT", "10000"))

# ----------------------------
# Embeddings / Retrieval
# ----------------------------
# Default embedding model (downloaded automatically by sentence-transformers on first run).
EMBEDDING_MODEL = os.getenv("BINARY_RAG_EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")

# For some embedding models (e.g., Nomic), recommended prefixes for query/document.
# These are used only if the model id includes the substring "nomic".
QUERY_PREFIX = os.getenv("BINARY_RAG_QUERY_PREFIX", "search_query: ")
DOC_PREFIX = os.getenv("BINARY_RAG_DOC_PREFIX", "search_document: ")

# Quantization mode:
#   - "standard": threshold at 0.0 (sign bit)
#   - "adaptive": per-dimension median threshold computed on the corpus (stored on disk)
QUANTIZATION_MODE = os.getenv("BINARY_RAG_QUANTIZATION_MODE", "standard").lower()
assert QUANTIZATION_MODE in {"standard", "adaptive"}

# Search defaults
DEFAULT_K_RECALL = int(os.getenv("BINARY_RAG_DEFAULT_K_RECALL", "200"))
DEFAULT_K_RERANK = int(os.getenv("BINARY_RAG_DEFAULT_K_RERANK", "10"))

# ----------------------------
# Evaluation / experiments
# ----------------------------
EVAL_N_QUERIES = int(os.getenv("BINARY_RAG_EVAL_N_QUERIES", "300"))
EVAL_TOPK = int(os.getenv("BINARY_RAG_EVAL_TOPK", "10"))

# Timing: repeat search-only measurements and take median (reduces noise)
TIMING_REPEATS = int(os.getenv("BINARY_RAG_TIMING_REPEATS", "7"))

# Rerank candidates (used in baseline comparisons)
RERANK_CANDIDATES = int(os.getenv("BINARY_RAG_RERANK_CANDIDATES", "200"))

# ----------------------------
# Baseline index parameters (FAISS)
# ----------------------------
PQ_M = int(os.getenv("BINARY_RAG_PQ_M", "96"))
PQ_NBITS = int(os.getenv("BINARY_RAG_PQ_NBITS", "8"))

HNSW_M = int(os.getenv("BINARY_RAG_HNSW_M", "32"))
HNSW_EF_CONSTRUCTION = int(os.getenv("BINARY_RAG_HNSW_EF_CONSTRUCTION", "200"))

IVF_NLIST = int(os.getenv("BINARY_RAG_IVF_NLIST", "100"))
IVF_NPROBE = int(os.getenv("BINARY_RAG_IVF_NPROBE", "10"))

# Sweep lists (comma-separated)
PQ_M_SWEEP = os.getenv("BINARY_RAG_PQ_M_SWEEP", "64,96,128")
IVF_NLIST_SWEEP = os.getenv("BINARY_RAG_IVF_NLIST_SWEEP", "100")
IVF_NPROBE_SWEEP = os.getenv("BINARY_RAG_IVF_NPROBE_SWEEP", "1,4,8,16,32")
HNSW_EFSEARCH_SWEEP = os.getenv("BINARY_RAG_HNSW_EFSEARCH_SWEEP", "16,32,64,128")
KRECALL_SWEEP = os.getenv("BINARY_RAG_KRECALL_SWEEP", "10,50,100,200,400")

def _parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]

PQ_M_LIST = _parse_int_list(PQ_M_SWEEP)
IVF_NLIST_LIST = _parse_int_list(IVF_NLIST_SWEEP)
IVF_NPROBE_LIST = _parse_int_list(IVF_NPROBE_SWEEP)
HNSW_EFSEARCH_LIST = _parse_int_list(HNSW_EFSEARCH_SWEEP)
KRECALL_LIST = _parse_int_list(KRECALL_SWEEP)

# Whether to build additional sweep indices during indexing (PQ m sweep / IVF nlist sweep).
# Default ON for paper-grade comparisons; set to 0 if you want faster indexing.
BUILD_PQ_SWEEP = os.getenv("BINARY_RAG_BUILD_PQ_SWEEP", "1") not in {"0", "false", "False"}
BUILD_IVF_NLIST_SWEEP = os.getenv("BINARY_RAG_BUILD_IVF_NLIST_SWEEP", "1") not in {"0", "false", "False"}

# Pareto / baseline sweep evaluation settings
BASELINE_SWEEP_TOPK = int(os.getenv("BINARY_RAG_BASELINE_SWEEP_TOPK", str(EVAL_TOPK)))
BASELINE_SWEEP_N_QUERIES = int(os.getenv("BINARY_RAG_BASELINE_SWEEP_N_QUERIES", str(EVAL_N_QUERIES)))

# Embedding compute
BATCH_SIZE = int(os.getenv("BINARY_RAG_BATCH_SIZE", "64"))
RANDOM_SEED = int(os.getenv("BINARY_RAG_RANDOM_SEED", "42"))

# ----------------------------
# Generation (optional)
# ----------------------------
# Path to a local GGUF model file. Download via README commands.
GENERATION_MODEL_PATH = str(Path(os.getenv("BINARY_RAG_GENERATION_MODEL_PATH", MODELS_DIR / "tinyllama.gguf")))
GENERATION_N_CTX = int(os.getenv("BINARY_RAG_GENERATION_N_CTX", "2048"))
GENERATION_N_THREADS = int(os.getenv("BINARY_RAG_GENERATION_N_THREADS", "4"))
