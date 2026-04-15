#!/bin/bash
set -e

# Convenience script to stop the vLLM container

CONTAINER_NAME="vllm-gemma4-26b"

if docker ps -q -f name="^/${CONTAINER_NAME}$" | grep -q .; then
    echo "Stopping vLLM container: $CONTAINER_NAME..."
    docker stop "$CONTAINER_NAME" > /dev/null
    docker rm "$CONTAINER_NAME" > /dev/null 2>&1 || true
    echo "✅ Stopped."
else
    echo "vLLM container is not running."
fi
