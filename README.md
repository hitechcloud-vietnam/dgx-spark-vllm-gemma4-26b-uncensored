# DGX Spark vLLM Gemma-4-26B Uncensored Guide

> High-performance LLM inference on NVIDIA DGX Spark using vLLM with uncensored Gemma-4 models.

[![Docker](https://img.shields.io/badge/Docker-vllm/vllm--openai:cu130--nightly-blue)](https://hub.docker.com/r/vllm/vllm-openai)
[![Hardware](https://img.shields.io/badge/Hardware-DGX%20Spark%20(GB10)-green)](https://www.nvidia.com/en-us/data-center/dgx-spark/)
[![Throughput](https://img.shields.io/badge/Throughput-45%2B%20tok/s-orange)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository documents how to run **fast, uncensored large language models** on the **NVIDIA DGX Spark** (GB10 Blackwell GPU) using vLLM. We achieved **45+ tokens/second** with the [AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4](https://huggingface.co/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4) model — a significant performance win over both Ollama and slower 31B quantized variants.

### What Makes This Setup Special

- **CUDA 13.0 + Blackwell optimization**: Uses the `cu130-nightly` vLLM image with SM12.1 support
- **NVFP4 quantization**: Leverages Blackwell's native FP4 tensor cores for 2-5× speedup
- **FP8 KV cache**: Halves memory usage without accuracy loss
- **CUDA graphs + chunked prefill**: Additional 20-40% throughput gains
- **Auto-start on boot**: Includes systemd user service for persistence after reboot

## Performance

| Setup | Model | Avg Speed | Memory | Notes |
|-------|-------|-----------|---------|-------|
| **This Setup** ✅ | Gemma-4-26B Uncensored NVFP4 | **45.26 tok/s** | ~16.3 GB | Fastest, uncensored |
| vLLM (LilaRest) | Gemma-4-31B NVFP4 | 9.16 tok/s | ~18.5 GB | Too slow on DGX Spark |
| Ollama | gemma4:31b | 8.05 tok/s | ~19 GB | Baseline |

**The 26B uncensored model is 5× faster than the 31B alternatives.**

### Detailed Benchmark Results

Tested on DGX Spark (GB10) with 128GB unified memory, `max_tokens=200`, 5 diverse prompts, warmup excluded:

```
Test 1:  200 tokens in 4.42s → 45.21 tok/s
Test 2:  200 tokens in 4.41s → 45.38 tok/s
Test 3:  200 tokens in 4.43s → 45.17 tok/s
Test 4:  200 tokens in 4.41s → 45.32 tok/s
Test 5:  200 tokens in 4.42s → 45.23 tok/s
─────────────────────────────────────
Average: 45.26 tok/s (σ = 0.07)
```

## Quick Start

### 1. Prerequisites

- NVIDIA DGX Spark or any **Blackwell SM12.1+** GPU (GB10, RTX 5090, etc.)
- Docker with NVIDIA Container Toolkit
- At least 20GB free disk space for the model
- `python3` and `curl` available on the host

### 2. Clone this repo

```bash
git clone https://github.com/hitechcloud-vietnam/dgx-spark-vllm-gemma4-26b-uncensored.git
cd dgx-spark-vllm-gemma4-26b-uncensored
```

### 3. Download the model (recommended)

```bash
bash scripts/download-model.sh
```

This downloads the ~15GB model to `~/.cache/huggingface/gemma-4-26B-it-uncensored-nvfp4`.

### 4. One-Command Start

```bash
bash scripts/start.sh
```

**Note**: First startup takes ~5-10 minutes while the container:
1. Upgrades `transformers` to support Gemma-4
2. Downloads the model if not pre-downloaded (~15GB)
3. Loads weights (~100s)
4. Compiles CUDA graphs (~55s with caching)

To stop:

```bash
bash scripts/stop.sh
```

### 5. Test It

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-26b-uncensored-vllm",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 200
  }'
```

## Manual Docker Run

If you prefer to run Docker manually instead of `scripts/start.sh`:

```bash
mkdir -p ~/.cache/huggingface

# Detect the gemma4.py path inside the container (avoids hardcoding Python version)
GEMMA4_PY="$(docker run --rm --entrypoint python3 \
  vllm/vllm-openai@sha256:a6cb8f72c66a419f2a7bf62e975ca0ba33dd4097b6b26858d166647c4cf4ba1f \
  -c \"import glob; paths=glob.glob('/usr/local/lib/python*/site-packages/vllm/model_executor/models/gemma4.py')+glob.glob('/usr/local/lib/python*/dist-packages/vllm/model_executor/models/gemma4.py'); print(paths[0] if paths else '')\")"

# Pinned digest — the rolling cu130-nightly tag can ship with breakages.
docker run -d --name vllm-gemma4-26b \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  --entrypoint /bin/bash \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)/scripts/startup.sh:/startup.sh" \
  -v "$(pwd)/patches/gemma4_patched.py:${GEMMA4_PY}" \
  vllm/vllm-openai@sha256:a6cb8f72c66a419f2a7bf62e975ca0ba33dd4097b6b26858d166647c4cf4ba1f \
  /startup.sh
```

The `startup.sh` script upgrades `transformers` inside the container before launching vLLM, and the `gemma4_patched.py` mount is **required** for the AEON-7 model to load correctly with `compressed-tensors` NVFP4.

## Auto-Start on Boot (Systemd)

To make vLLM automatically start after reboots:

```bash
bash scripts/install-service.sh
systemctl --user start vllm-gemma4-26b.service
```

To also start at **boot time** (before anyone logs in), enable lingering:

```bash
sudo loginctl enable-linger $USER
```

To check status:

```bash
systemctl --user status vllm-gemma4-26b.service
```

## Open WebUI Integration

You can connect [Open WebUI](https://github.com/open-webui/open-webui) to this local vLLM endpoint for a chat-GPT-like interface.

### Install Open WebUI

```bash
# Using pip (recommended for local single-user setups)
pip install open-webui
```

Or via pipx:
```bash
pipx install open-webui
```

### Start Open WebUI (Host install)

If you installed Open WebUI with **pip/pipx on the host**, it shares the same network as vLLM:

```bash
# Set the OpenAI-compatible API base to your local vLLM
export OPENAI_API_BASE_URL="http://localhost:8000/v1"

# Start Open WebUI
open-webui serve
```

Then open `http://localhost:8080` in your browser.

In the Open WebUI settings:
1. Go to **Admin Panel → Settings → Connections**
2. Under **OpenAI API**, set:
   - **API URL**: `http://localhost:8000/v1`
   - **API Key**: `sk-1234567890` (any dummy key works; vLLM doesn't validate it)
3. Click **Save**
4. Go to **Admin Panel → Settings → Models**
5. Verify `gemma-4-26b-uncensored-vllm` appears in the model list
6. Select it from the model dropdown in the chat page

### Open WebUI in Docker

If Open WebUI runs inside a Docker container, **`localhost` / `127.0.0.1` will not work** because inside a container those refer to the container itself, not the host machine where vLLM is running.

Use the host-guest DNS name instead:

```
http://host.docker.internal:8000/v1
```

> **Note**: `host.docker.internal` works on Docker Desktop and Docker Engine 20.10+ on Linux. If it doesn't resolve on your system, start the Open WebUI container with `--network=host` (Linux only) or use the host's LAN IP (e.g., `http://192.168.1.42:8000/v1`).

## Why This Works on DGX Spark

The DGX Spark's **GB10 GPU** (Blackwell architecture, SM12.1) has several unique characteristics that make model selection critical:

### What Works

| Technique | Why It Helps | Gain |
|-----------|--------------|------|
| **CUDA 13.0** | Blackwell FP4 requires CUDA 13+ | Baseline |
| **NVFP4 quantization** | Native FP4 tensor core support | ~2-5× |
| **compressed-tensors** | Model's native format, no conversion overhead | ~10-20% |
| **FP8 KV cache** | Halves KV cache memory, more batching | ~15% |
| **CUDA graphs** | Avoids Python overhead per token | ~20-40% |
| **Chunked prefill** | Better interleaving of prefill/decode | ~10% |

### What Doesn't Work Well

| Setup | Problem |
|-------|---------|
| **LilaRest/gemma-4-31B-it-NVFP4-turbo** | Only 9 tok/s — modelopt quantization path underperforms on DGX Spark compared to compressed-tensors |
| **Ollama gemma4:31b** | 8 tok/s — no CUDA graph optimization, no FP4 tensor core path |
| **Standard vLLM cu124** | Missing Blackwell SM12.1 support entirely |

## Systematic Setup Journey

Here is exactly how we arrived at this 45 tok/s configuration:

### Phase 1: Baseline (Ollama)
- Started with `gemma4:31b` on Ollama
- **Result**: 8.05 tok/s
- **Issue**: Ollama's runtime lacks CUDA graph capture and FP4-optimized kernels

### Phase 2: vLLM with LilaRest 31B
- Tried `LilaRest/gemma-4-31B-it-NVFP4-turbo` on vLLM `cu130-nightly`
- Required `--quantization modelopt` and `transformers>=5.5.0` for gemma4 support
- **Result**: 9.16 tok/s
- **Issue**: Surprisingly slow. The `modelopt` quantization backend has higher kernel latency on DGX Spark compared to `compressed-tensors` format models.

### Phase 3: The Winner (AEON-7 26B Uncensored)
- Switched to `AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4`
- Uses `--quantization compressed-tensors` — the model's native format
- Added `--enable-chunked-prefill` and `--enable-prefix-caching`
- Set `--gpu-memory-utilization 0.60` (262K context fits comfortably)
- **Result**: **45.26 tok/s** — a **5× improvement** over both alternatives

### Key Breakthrough

The **compressed-tensors NVFP4 format** directly maps to vLLM's `FlashInferCutlassNvFp4LinearKernel`, which is specifically optimized for Blackwell. The `modelopt` backend used by LilaRest triggers a slower fallback path on SM12.1.

> **Lesson**: On DGX Spark, prefer models natively quantized with `compressed-tensors` + NVFP4 over `modelopt` wrapped models.

## Repository Structure

```
dgx-spark-vllm-gemma4-26b-uncensored/
├── README.md                          # This file
├── patches/
│   └── gemma4_patched.py              # Required patch for AEON-7 NVFP4 loading
├── scripts/
│   ├── start.sh                       # One-command container startup
│   ├── stop.sh                        # One-command container stop
│   ├── startup.sh                     # In-container startup (upgrades transformers)
│   ├── benchmark.sh                   # Reproduce our 45 tok/s benchmark
│   ├── download-model.sh              # Pre-download the model
│   └── install-service.sh             # Install systemd auto-start service
├── systemd/
│   └── vllm-gemma4-26b.service        # Systemd user service file
├── benchmarks/
│   └── results-gemma4-26b.csv         # Raw benchmark data
├── docs/
│   ├── ARCHITECTURE.md                # Deep dive into DGX Spark + vLLM
│   ├── TROUBLESHOOTING.md             # Common issues and fixes
│   └── MODEL_COMPARISON.md            # Full comparison matrix
└── LICENSE
```

## Benchmarking

Run the official benchmark script to reproduce our results:

```bash
bash scripts/benchmark.sh
```

This will:
1. Send a warmup request (excluded from results)
2. Run 5 diverse prompts with `max_tokens=200`
3. Print average throughput and consistency metrics

## Environment Variables

These are baked into the `cu130-nightly` image and should not need changing:

```bash
TORCH_CUDA_ARCH_LIST="8.7 8.9 9.0 10.0+PTX 12.0 12.1"
CUDA_VERSION=13.0.1
```

## Troubleshooting

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for:
- "model type gemma4 not recognized" fixes
- Transformer version conflicts
- Memory errors and how to tune `--gpu-memory-utilization`

## Credits

- **Model**: [AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4](https://huggingface.co/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4)
- **Base Model**: Google Gemma 4
- **vLLM**: [vllm-project/vllm](https://github.com/vllm-project/vllm)
- **Hardware**: NVIDIA DGX Spark

## License

MIT License — see [LICENSE](LICENSE).
