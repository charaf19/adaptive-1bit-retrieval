$ErrorActionPreference = "Stop"

Write-Host "1) Ingest data" -ForegroundColor Cyan
python -m src.data_loader

Write-Host "2) Build indexes (binary + baseline FAISS indices)" -ForegroundColor Cyan
python -m src.indexer

Write-Host "3) Run experiments" -ForegroundColor Cyan
python -m experiments.sensitivity_analysis
python -m experiments.benchmark_baselines
python -m experiments.baseline_sweep
python -m experiments.benchmark_efficiency
python -m experiments.latency_profiling
python -m experiments.model_robustness
python -m experiments.ablation_study
python -m experiments.scalability_test

Write-Host "4) Generate figures + tables for the paper" -ForegroundColor Cyan
python -m src.analytics

Write-Host "✅ Done. Check results\figures and results\tables" -ForegroundColor Green
