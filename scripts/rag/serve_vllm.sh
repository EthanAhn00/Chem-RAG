#!/usr/bin/env bash
# Serve Qwen3-8B (bf16) on a single GPU with vLLM.
# Usage from the repository root: bash scripts/rag/serve_vllm.sh
#        GPU=1 PORT=8001 bash scripts/rag/serve_vllm.sh
# Check the serving environment and shared-GPU authorization before launch.
set -euo pipefail

GPU="${GPU:-0}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
MAXLEN="${MAXLEN:-32768}"
UTIL="${UTIL:-0.90}"

echo "[vllm] GPU=$GPU model=$MODEL port=$PORT max_model_len=$MAXLEN"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | sed -n "$((GPU+1))p"

export CUDA_VISIBLE_DEVICES="$GPU"
exec vllm serve "$MODEL" \
  --served-model-name Qwen3-8B \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$UTIL" \
  --max-num-seqs 32 \
  --dtype bfloat16 \
  --port "$PORT"
