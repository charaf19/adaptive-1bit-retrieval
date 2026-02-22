#!/usr/bin/env bash
set -euo pipefail

mkdir -p models

echo "Downloading a small GGUF instruct model into ./models ..."
echo "You can change the repo/file below to any GGUF you prefer."

# Option A (recommended): Qwen2.5 0.5B Instruct GGUF (small + fast)
# If this repo/file changes over time, pick any GGUF from Hugging Face and update.
# Correct upstream repo (the old ggml-org path can 404)
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen2.5-0.5B-Instruct-GGUF}"
MODEL_FILE="${MODEL_FILE:-qwen2.5-0.5b-instruct-q4_k_m.gguf}"

if ! command -v hf >/dev/null 2>&1; then
  echo "hf (huggingface_hub) CLI not found. Install: pip install -U huggingface_hub"
  exit 1
fi

hf download "$MODEL_REPO" "$MODEL_FILE" --local-dir models

# Create a stable filename that config.py expects by default
cp -f "models/$MODEL_FILE" "models/tinyllama.gguf"

echo "✅ Model downloaded:"
ls -lh models/tinyllama.gguf
