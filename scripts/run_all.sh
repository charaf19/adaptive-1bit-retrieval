#!/usr/bin/env bash
set -euo pipefail

echo "1) Ingest data"
python -m src.data_loader

echo "2) Build indexes"
python -m src.indexer

echo "3) Run experiments"
python -m experiments.sensitivity_analysis
python -m experiments.benchmark_baselines
python -m experiments.baseline_sweep
python -m experiments.benchmark_efficiency
python -m experiments.latency_profiling
python -m experiments.model_robustness
python -m experiments.ablation_study
python -m experiments.scalability_test

echo "4) Generate figures + tables for paper"
python -m src.analytics

echo "✅ Done. Check results/figures and results/tables"
