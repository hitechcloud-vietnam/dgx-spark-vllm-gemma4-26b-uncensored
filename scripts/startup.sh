#!/bin/bash
set -e

# Container startup script — runs INSIDE the Docker container
# Upgrades transformers to support Gemma-4, then starts vLLM

# Auto-detect where vllm's gemma4.py lives (handles Python version changes)
GEMMA4_PY="$(python3 -c "import glob; paths=glob.glob('/usr/local/lib/python*/site-packages/vllm/model_executor/models/gemma4.py')+glob.glob('/usr/local/lib/python*/dist-packages/vllm/model_executor/models/gemma4.py'); print(paths[0] if paths else '')")"
if [ -z "$GEMMA4_PY" ] || [ ! -f "$GEMMA4_PY" ]; then
    echo "❌ Could not locate vllm model_executor/models/gemma4.py"
    exit 1
fi

echo "Updating transformers to support gemma4..."
# Pin to a known-good version that supports gemma4 (~30s on first container start)
pip install "transformers==5.5.4" -q

echo "Starting vLLM server with AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4..."
exec python3 -m vllm.entrypoints.openai.api_server \
  --model /root/.cache/huggingface/gemma-4-26B-it-uncensored-nvfp4 \
  --served-model-name gemma-4-26b-uncensored-vllm \
  --quantization compressed-tensors \
  --load-format safetensors \
  --max-model-len 262000 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 131072 \
  --gpu-memory-utilization 0.60 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
