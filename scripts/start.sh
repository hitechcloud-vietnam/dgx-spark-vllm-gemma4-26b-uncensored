#!/bin/bash
set -e

# DGX Spark vLLM Startup Script
# Starts the AEON-7 Gemma-4-26B Uncensored NVFP4 model with optimal settings

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CONTAINER_NAME="vllm-gemma4-26b"
# Pinned to a known-good cu130-nightly digest because the rolling tag can introduce
# breakages (e.g. newer nightlies have shipped with incompatible nixl_ep libraries).
IMAGE="vllm/vllm-openai@sha256:a6cb8f72c66a419f2a7bf62e975ca0ba33dd4097b6b26858d166647c4cf4ba1f"
MODEL_PATH="/root/.cache/huggingface/gemma-4-26B-it-uncensored-nvfp4"
STARTUP_SCRIPT="$SCRIPT_DIR/startup.sh"
PATCH_FILE="$REPO_DIR/patches/gemma4_patched.py"

echo "=================================="
echo "DGX Spark vLLM Starter"
echo "=================================="

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Check NVIDIA runtime
if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo "⚠️  Warning: NVIDIA Container Runtime may not be configured."
    echo "   Make sure 'docker run --gpus all' works on your system."
fi

# Check patch file exists
if [ ! -f "$PATCH_FILE" ]; then
    echo "❌ Patch file not found: $PATCH_FILE"
    echo "   This patch is required for the AEON-7 model to load correctly."
    exit 1
fi

# Check startup script exists
if [ ! -f "$STARTUP_SCRIPT" ]; then
    echo "❌ Startup script not found: $STARTUP_SCRIPT"
    exit 1
fi

# Stop existing container
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping existing container: $CONTAINER_NAME"
    docker stop "$CONTAINER_NAME" > /dev/null 2>&1 || true
    docker rm "$CONTAINER_NAME" > /dev/null 2>&1 || true
fi

# Ensure HuggingFace cache directory exists
mkdir -p ~/.cache/huggingface

echo ""
echo "Pulling image: $IMAGE"
docker pull "$IMAGE"

# Auto-detect where vLLM's gemma4.py lives inside the container
# (avoids breaking when the image switches Python versions)
echo "Detecting vLLM gemma4.py path inside container..."
GEMMA4_PY="$(docker run --rm --entrypoint python3 "$IMAGE" -c "import glob; paths=glob.glob('/usr/local/lib/python*/site-packages/vllm/model_executor/models/gemma4.py')+glob.glob('/usr/local/lib/python*/dist-packages/vllm/model_executor/models/gemma4.py'); print(paths[0] if paths else '')")"
if [ -z "$GEMMA4_PY" ] || [ ! -n "$GEMMA4_PY" ]; then
    echo "⚠️  Could not auto-detect gemma4.py path. Falling back to hardcoded path."
    GEMMA4_PY="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py"
fi
echo "  → $GEMMA4_PY"

echo ""
echo "Starting vLLM container..."
echo "Model: $MODEL_PATH"
echo "This may take 5-10 minutes on first run for model download + CUDA graph compilation."
echo ""

docker run -d --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  --entrypoint /bin/bash \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$STARTUP_SCRIPT:/startup.sh" \
  -v "$PATCH_FILE:$GEMMA4_PY" \
  "$IMAGE" \
  /startup.sh

echo ""
echo "Container started: $CONTAINER_NAME"
echo ""
echo "⏳ Waiting for server to be ready (this can take 5-10 min on first run)..."

for i in {1..60}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo ""
        echo "✅ Server is ready!"
        echo ""
        echo "Test it:"
        echo "  curl http://localhost:8000/v1/chat/completions \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"model\":\"gemma-4-26b-uncensored-vllm\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":100}'"
        echo ""
        echo "View logs: docker logs -f $CONTAINER_NAME"
        echo "Stop: docker stop $CONTAINER_NAME"
        exit 0
    fi
    echo -n "."
    if [ $((i % 10)) -eq 0 ]; then
        echo " ($((i*10))s)"
    fi
    sleep 10
done

echo ""
echo "⚠️  Server did not become ready within 10 minutes."
echo "Check logs: docker logs $CONTAINER_NAME"
exit 1
