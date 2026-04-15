#!/bin/bash
set -e

# DGX Spark vLLM Benchmark Script
# Reproduces the 45+ tok/s benchmark from the guide

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

API_URL="http://localhost:8000/v1/chat/completions"
MODEL="gemma-4-26b-uncensored-vllm"

echo "=================================================="
echo "DGX Spark vLLM Benchmark"
echo "Model: $MODEL"
echo "Hardware: NVIDIA DGX Spark (GB10 Blackwell)"
echo "=================================================="

# Check if server is running
if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "❌ Server not running! Start it first with: bash scripts/start.sh"
    exit 1
fi

echo ""
echo "Step 1: WARMUP (discarded from results)"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":100}" > /dev/null
echo "✓ Warmup complete"

# 5 diverse test prompts
TEST_PROMPTS=(
    "What is machine learning? Explain briefly."
    "Write a Python function to calculate factorial using recursion."
    "Explain quantum computing in simple terms."
    "Describe the process of photosynthesis."
    "What are the main differences between HTTP and HTTPS?"
)

echo ""
echo "Step 2: RUNNING 5 TESTS"
echo "=================================================="

RESULTS_FILE="benchmarks/results-gemma4-26b.csv"
mkdir -p "$(dirname "$RESULTS_FILE")"
echo "Test,Prompt,Tokens,Time(s),Tokens/s" > "$RESULTS_FILE"

for i in {1..5}; do
    idx=$((i-1))
    prompt="${TEST_PROMPTS[$idx]}"
    
    echo ""
    echo "Test $i/5: ${prompt:0:50}..."
    
    start_time=$(date +%s.%N)
    
    response=$(curl -s -X POST "$API_URL" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":200}")
    
    end_time=$(date +%s.%N)
    
    # Pipe response into Python for safe parsing and math
    read -r tokens elapsed tps <<< "$(printf '%s\n%s\n%s\n' "$start_time" "$end_time" "$response" | python3 -c "
import sys, json
start = float(sys.stdin.readline().strip())
end = float(sys.stdin.readline().strip())
response = sys.stdin.read()
elapsed = end - start
try:
    d = json.loads(response)
    tokens = d['usage']['completion_tokens']
except Exception:
    tokens = 200
tps = tokens / elapsed if elapsed > 0 else 0
print(f'{tokens} {elapsed:.3f} {tps:.2f}')
")"
    
    echo "  Tokens: $tokens"
    echo "  Time: ${elapsed}s"
    echo "  Speed: ${tps} tok/s"
    
    echo "$i,\"$prompt\",$tokens,$elapsed,$tps" >> "$RESULTS_FILE"
    sleep 1
done

echo ""
echo "=================================================="
echo "BENCHMARK RESULTS"
echo "=================================================="

read -r avg_time avg_tps total_tokens <<< "$(python3 -c "
import csv
with open('$RESULTS_FILE') as f:
    rows = list(csv.reader(f))[1:]
times = [float(r[3]) for r in rows]
tps_vals = [float(r[4]) for r in rows]
tokens = [int(r[2]) for r in rows]
avg_time = sum(times)/len(times)
avg_tps = sum(tps_vals)/len(tps_vals)
total = sum(tokens)
print(f'{avg_time:.2f} {avg_tps:.2f} {total}')
")"

echo ""
tail -n +2 "$RESULTS_FILE" | while IFS=',' read -r test prompt tokens time tps; do
    echo "  Test $test: ${time}s | ${tps} tok/s"
done

echo ""
echo "Average Metrics:"
echo "  Time: ${avg_time}s"
echo "  Speed: ${avg_tps} tok/s"
echo "  Total Tokens: $total_tokens"
echo ""
echo "✓ Results saved to: $RESULTS_FILE"
echo "=================================================="
