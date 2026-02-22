# Adaptive-1Bit: 1-bit FAISS + Asymmetric Memory-Mapped Reranking

This repository implements a **two-stage Retrieval-Augmented Generation (RAG)** pipeline:

1. **Stage 1 (Recall):** FAISS **binary** search using **1-bit quantized** embeddings (very fast + memory-efficient)
2. **Stage 2 (Precision):** Re-rank candidates using the original **float32** embeddings
3. **(Optional) Generation:** Explain the recommendation with a local **GGUF** LLM (llama-cpp-python)

The code is designed so that **every experiment produces reproducible CSV tables and PNG figures** for direct insertion into a paper.

## Contributions & State-of-the-Art Novelty

While traditional vector databases rely heavily on Approximate Nearest Neighbor (ANN) graphs (like HNSW) or Product Quantization (PQ) which consume vast memory or suffer from complex graph maintenance, **Binary-RAG** presents a highly efficient alternative designed to maximize both speed and recall without requiring specialized model re-training.

1. **Adaptive 1-bit Quantization (Entropy Maximization):** Standard 1-bit approaches use naive zero-thresholding (`x > 0`). Binary-RAG alternatively introduces a data-driven approach by computing the median value per embedding dimension across the targeted corpus. This ensures optimally balanced bits (entropy maximization), capturing significantly more informational variance from off-the-shelf continuous models (e.g., Nomic, MiniLM).
2. **Zero-Training Drop-In Replacement:** Unlike recent methods that require specific contrastive training or Matryoshka Representation Learning to produce binary-friendly embeddings, this pipeline effectively adapts *any standard continuous dense model* into an ultra-compressed 1-bit index instantly without any tuning. 
3. **Asymmetric Two-Stage Pipeline:** It cleverly bridges the hardware-accelerated advantages of POPCNT operations in a brute-force FAISS `IndexBinaryFlat` (stage 1) with an asymmetric memory-mapped `float32` re-ranking step (stage 2). This yields near-exact L2 recall while drastically minimizing the search index's main memory footprint compared to large HNSW graphs.
4. **Comprehensive Benchmarking & Reproducibility:** Provides a fully automated, paper-ready framework to rigorously benchmark the proposed approach against well-established ANN algorithms (HNSW, IVF-PQ, conventional PQ, IVF-Flat). It automatically sweeps hyperparameters and plots Pareto-optimal frontiers (Latency vs. Recall).

## Repository structure

```
.
├── config.py
├── requirements.txt
├── src/
│   ├── data_loader.py
│   ├── indexer.py
│   ├── engine.py
│   ├── generator.py
│   └── analytics.py
├── experiments/
│   ├── sensitivity_analysis.py
│   ├── benchmark_baselines.py
│   ├── benchmark_efficiency.py
│   ├── ablation_study.py
│   ├── model_robustness.py
│   ├── latency_profiling.py
│   ├── evaluate_recommender.py
│   ├── failure_analysis.py
│   ├── rag_qa_eval.py
│   └── scalability_test.py
└── scripts/
    ├── download_models.sh
    ├── download_models.ps1
    ├── run_all.sh
    └── run_all.ps1
```

## 0) System prerequisites

- Python **3.10+**
- Recommended: Linux/macOS (Windows works but FAISS install can be trickier)
- For best speed: a CPU with AVX2

## 1) Install dependencies

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### Optional: enable local GGUF generation (llama-cpp-python)

Generation is **not required** for the retrieval experiments or paper figures.
Only scripts that import `src.generator.Generator` need it.

**Recommended (pre-built wheels):**

```bash
# CPU wheel (works without a compiler)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# NVIDIA GPU wheel (pick your CUDA version: cu121/cu122/cu123/cu124/cu125)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

**Windows tip:** to *force* a wheel (avoid building from source), use:

```powershell
pip install --only-binary=:all: llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

If you prefer installing from source, you must have a working C/C++ toolchain on your OS.

#### Windows troubleshooting: "Can't find 'nmake'" / "CMAKE_C_COMPILER not set"

If `pip install llama-cpp-python` tries to build from source on Windows, it may fail with errors like:
`Can't find 'nmake'` or `CMAKE_C_COMPILER not set`.

Fix options:

1) **Use the pre-built wheel commands above** (recommended).
2) Install **Visual Studio Build Tools** (Desktop development with C++) and run pip inside the
   "x64 Native Tools Command Prompt".
3) Use MinGW via **w64devkit** and force the generator/compiler (example):

```powershell
$env:CMAKE_GENERATOR = "MinGW Makefiles"
$env:CMAKE_ARGS = "-DGGML_OPENBLAS=on -DCMAKE_C_COMPILER=C:/w64devkit/bin/gcc.exe -DCMAKE_CXX_COMPILER=C:/w64devkit/bin/g++.exe"
pip install llama-cpp-python
```

## 2) Download the (optional) GGUF generation model

If you want **generation** (LLM explanations), download a small GGUF model:

```bash
bash scripts/download_models.sh
```

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_models.ps1
```

By default the script downloads a small instruct GGUF into `models/` and also copies it to:
`models/tinyllama.gguf` (the default expected by `config.py`).

If you already have a GGUF file, just set:

```bash
export BINARY_RAG_GENERATION_MODEL_PATH="path/to/your_model.gguf"
```

> Note: Retrieval experiments do **not** require the generation model. Only scripts that import `src.generator.Generator` do.

## 3) Ingest datasets (Books + ArXiv)

Data is streamed from Hugging Face and saved into `data/processed/*.jsonl`.

```bash
python -m src.data_loader
```

Control dataset size:

```bash
export BINARY_RAG_INGEST_LIMIT=10000   # default is 10000
```

## 4) Build indexes (binary + float artifacts)

```bash
python -m src.indexer
```

This produces:

- `artifacts/{domain}.index` (FAISS binary index)
- `artifacts/{domain}_float.npy` (float vectors)
- `artifacts/{domain}_flat.index` (FAISS Float32 exact index)
- `artifacts/{domain}_pq_m*.index` (FAISS PQ baseline)
- `artifacts/{domain}_hnsw_m*.index` (FAISS HNSW baseline)
- `artifacts/{domain}_ivf_flat_nlist*.index` (FAISS IVF-Flat baseline)
- `artifacts/{domain}_ivf_pq_nlist*_m*.index` (FAISS IVF-PQ baseline)
- `artifacts/{domain}_texts.pkl` (ids + raw texts)
- `results/indexing_metrics.json` (for Figure 1)

### Quantization mode (standard vs adaptive)

- Standard (default): threshold at 0 → `bit = (x > 0)`
- Adaptive: threshold at per-dimension median (stored in `artifacts/{domain}_thresholds.npy`)

Enable adaptive:

```bash
export BINARY_RAG_QUANTIZATION_MODE=adaptive
python -m src.indexer
```

## 5) Run experiments (each one produces paper assets)

All outputs are written into:

- Tables (CSV/Markdown/LaTeX): `results/tables/`
- Logs (CSV): `results/logs/`
- Figures (PNG): `results/figures/`

### 5.1 Sensitivity analysis (k_recall vs recall & latency)

```bash
python -m experiments.sensitivity_analysis
```

Outputs:
- `results/logs/sensitivity_books.csv`
- `results/logs/sensitivity_arxiv.csv`
- `results/figures/sensitivity_books.png`
- `results/figures/sensitivity_arxiv.png`

### 5.2 Baseline benchmark (Float32 vs PQ vs Binary vs Ours)

```bash
python -m experiments.benchmark_baselines
```

Outputs:
- `results/logs/baseline_search_books.csv` (search-only latency, embedding excluded)
- `results/logs/baseline_search_arxiv.csv`
- `results/logs/baseline_end2end_books.csv` (embed + search + rerank breakdown)
- `results/logs/baseline_end2end_arxiv.csv`

### 5.3 Baseline tuning sweep + Pareto frontier (recommended for papers)

Many ANN baselines (especially IVF) are sensitive to hyperparameters (e.g., `nprobe`).
This sweep generates multiple operating points so you can report **best-tuned baselines**
and produce a **Recall@10 vs Search_ms** Pareto plot.

```bash
python -m experiments.baseline_sweep
```

Outputs:
- `results/logs/baseline_sweep_search_books.csv`
- `results/logs/baseline_sweep_search_arxiv.csv`

Key knobs (environment variables):
- `BINARY_RAG_IVF_NPROBE_SWEEP` (e.g., `1,4,8,16,32,64`)
- `BINARY_RAG_HNSW_EFSEARCH_SWEEP` (e.g., `16,32,64,128,256`)
- `BINARY_RAG_KRECALL_SWEEP` (binary candidate size sweep)
- `BINARY_RAG_IVF_NLIST_SWEEP` (requires rebuilding indices via `python -m src.indexer`)

### 5.4 Index efficiency (Recall@100 vs latency for raw indexes)

```bash
python -m experiments.benchmark_efficiency
```

Outputs:
- `results/logs/efficiency_books.csv`
- `results/logs/efficiency_arxiv.csv`

### 5.5 Ablation (adaptive median thresholds)

```bash
python -m experiments.ablation_study
```

By default this uses fast models (MiniLM + Nomic). To include heavier models:

```bash
export BINARY_RAG_ABLATION_MODELS=MiniLM,Nomic,Mixedbread
python -m experiments.ablation_study
```

Output:
- `results/logs/ablation_adaptive_quantization.csv`

### 5.6 Model robustness (different embedding models)

```bash
python -m experiments.model_robustness
```

Control models similarly:

```bash
export BINARY_RAG_ROBUSTNESS_MODELS=MiniLM,Nomic
python -m experiments.model_robustness
```

Outputs:
- `results/logs/robustness_books.csv`
- `results/logs/robustness_arxiv.csv`

### 5.6 Latency profiling (component breakdown)

```bash
python -m experiments.latency_profiling
```

Output:
- `results/logs/latency_profile_books.csv`

### 5.7 Scalability (latency vs N)

```bash
python -m experiments.scalability_test
```

Outputs:
- `results/logs/scalability.csv`
- `results/figures/scalability_books.png`
- `results/figures/scalability_arxiv.png`

### 5.8 (Optional) Recommender evaluation + failure analysis + QA check

Requires the GGUF generation model.

```bash
python -m experiments.evaluate_recommender
python -m experiments.failure_analysis
python -m experiments.rag_qa_eval
```

## Windows troubleshooting (llama-cpp-python)

If you see errors like **"Can't find 'nmake'"** or **"CMAKE_C_COMPILER not set"**, it means pip tried to build
`llama-cpp-python` from source but your build tools aren't available.

Fix options:

1) **Use the pre-built wheels** (recommended): see the install commands above.

2) **Install build tools (Visual Studio Build Tools)** and re-run `pip install llama-cpp-python`.

3) **Use MinGW via w64devkit** (set generator + compilers), then install:

```powershell
$env:CMAKE_GENERATOR = "MinGW Makefiles"
$env:CMAKE_ARGS = "-DGGML_OPENBLAS=on -DCMAKE_C_COMPILER=C:/w64devkit/bin/gcc.exe -DCMAKE_CXX_COMPILER=C:/w64devkit/bin/g++.exe"
pip install llama-cpp-python
```

## 6) Generate all final paper figures + tables

```bash
python -m src.analytics
```

This writes Markdown tables (paper-ready) under `results/tables/`.

Outputs (paper-ready):
- `results/figures/baselines_e2e_books.png`
- `results/figures/baselines_e2e_arxiv.png`
- `results/figures/baselines_search_books.png`
- `results/figures/baselines_search_arxiv.png`
- `results/figures/sensitivity_books.png`
- `results/figures/sensitivity_arxiv.png`
- `results/figures/scalability_books.png`
- `results/figures/scalability_arxiv.png`
- `results/tables/table_baselines_books.(csv|md|tex)`
- `results/tables/table_baselines_arxiv.(csv|md|tex)`
- `results/tables/table_baselines_books_search.(csv|md|tex)`
- `results/tables/table_baselines_arxiv_search.(csv|md|tex)`
- `results/tables/table_indexing_metrics.(csv|md|tex)`

## 7) One-command full pipeline

```bash
bash scripts/run_all.sh
```

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
```

## Notes to avoid debugging issues

- Always run commands **from the repo root**.
- If FAISS install fails on your system, try:
  - Linux: `pip install faiss-cpu`
  - Conda: `conda install -c conda-forge faiss-cpu`
- Hugging Face datasets/models are cached automatically; set cache dirs if needed:
  ```bash
  export HF_HOME="$PWD/.hf"
  export HF_DATASETS_CACHE="$PWD/.hf/datasets"
  export TRANSFORMERS_CACHE="$PWD/.hf/transformers"
  ```

## Citation / paper integration

- All tables are saved as **CSV + Markdown + LaTeX** (see `results/tables/`).
- All figures are saved as **PNG @ 300 dpi** (see `results/figures/`).

