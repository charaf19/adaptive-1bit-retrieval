$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path "models" | Out-Null

Write-Host "Downloading GGUF model into .\models ..." -ForegroundColor Cyan

# Correct upstream repo (the old ggml-org path can 404)
$MODEL_REPO = $env:MODEL_REPO
if ([string]::IsNullOrWhiteSpace($MODEL_REPO)) { $MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF" }

$MODEL_FILE = $env:MODEL_FILE
if ([string]::IsNullOrWhiteSpace($MODEL_FILE)) { $MODEL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf" }

Write-Host "Repo: $MODEL_REPO" -ForegroundColor Yellow
Write-Host "File: $MODEL_FILE" -ForegroundColor Yellow

try {
  $null = & hf --version
} catch {
  Write-Host "hf CLI not found. Install: pip install -U huggingface_hub" -ForegroundColor Red
  exit 1
}

& hf download $MODEL_REPO $MODEL_FILE --local-dir "models" | Out-Null

# Create stable filename expected by default config
Copy-Item -Force "models\$MODEL_FILE" "models\tinyllama.gguf"

Write-Host "✅ Model downloaded: models\tinyllama.gguf" -ForegroundColor Green
Get-Item "models\tinyllama.gguf" | Format-List Name,Length,LastWriteTime
