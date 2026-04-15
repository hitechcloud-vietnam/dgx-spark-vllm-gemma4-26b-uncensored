# Model Comparison Matrix

Comprehensive comparison of Gemma-4 variants tested on the NVIDIA DGX Spark.

## Summary Table

| Model | Size | Quantization | Backend | Speed | Memory | Quality | Recommendation |
|-------|------|--------------|---------|-------|--------|---------|----------------|
| **AEON-7 Uncensored 26B** ✅ | 26B | compressed-tensors NVFP4 | vLLM cu130 | **45.11 tok/s** | ~16.3 GB | High | **Best Choice** |
| LilaRest 31B Turbo | 31B | modelopt NVFP4 | vLLM cu130 | 9.16 tok/s | ~18.5 GB | Very High | Too slow on DGX |
| Ollama gemma4:31b | 31B | GGUF | Ollama | 8.05 tok/s | ~19 GB | High | Baseline only |
| Google Base (BF16) | 31B | None | — | — | ~58.9 GB | Highest | Won't fit |

## Why 26B Beats 31B on DGX Spark

This surprises many people. Here's the technical explanation:

### The Bottleneck is Kernel Choice, Not Model Size

On Blackwell SM12.1, the quantization backend determines which GPU kernels are used:

| Backend | Kernel Path | SM12.1 Optimized? | DGX Spark Speed |
|---------|-------------|-------------------|-----------------|
| **compressed-tensors** | FlashInferCutlassNvFp4LinearKernel | ✅ Yes | **45 tok/s** |
| modelopt | Generic fallback kernels | ⚠️ Partial | ~9 tok/s |
| GGUF (llama.cpp) | No FP4 tensor cores | ❌ No | ~8 tok/s |

The 26B model uses `compressed-tensors` natively, so vLLM can route every layer through the fastest Blackwell kernels. The 31B `modelopt` model falls back to less optimized paths.

### Raw Numbers

**AEON-7 26B Uncensored**:
- Load time: ~100s
- CUDA graph compile: ~55s
- Memory: 16.3 GB
- Context: 262K tokens
- Speed: 45.11 tok/s

**LilaRest 31B Turbo**:
- Load time: ~120s
- CUDA graph compile: ~55s
- Memory: 18.5 GB
- Context: 16K tokens (recommended)
- Speed: 9.16 tok/s

**Ollama 31B**:
- Load time: Instant (if cached)
- No CUDA graph compile
- Memory: 19 GB
- Speed: 8.05 tok/s

## Quality Comparison

While we didn't run formal evals, the models rank as follows based on community benchmarks:

| Model | MMLU Pro (est.) | GPQA (est.) | Censorship |
|-------|-----------------|-------------|------------|
| Google Base | 85.3% | 75.7% | High |
| LilaRest 31B | 83.9% | 72.7% | Medium |
| Ollama 31B | ~84% | ~73% | High |
| **AEON-7 26B** | ~83% | ~71% | **None** |

The 26B uncensored trades ~1-2% accuracy for **5× speed** and **no censorship**.

## When to Use Which

### Use AEON-7 26B Uncensored if:
- ✅ You want maximum speed on DGX Spark
- ✅ You need uncensored outputs
- ✅ You want 262K context support
- ✅ You prioritize throughput over absolute benchmark scores

### Use LilaRest 31B if:
- You need the absolute highest benchmark scores
- You don't mind 5× slower inference
- You have an RTX 5090 or PRO 6000 (may perform better there)

### Use Ollama if:
- You want simplicity and instant startup
- You don't need API access
- 8 tok/s is acceptable for your use case

## Benchmark Methodology

All tests conducted on the same DGX Spark machine with identical conditions:
- **Hardware**: NVIDIA DGX Spark (GB10, 128GB unified memory)
- **Driver**: 580.142
- **CUDA**: 13.0
- **Test**: 5 diverse prompts, `max_tokens=200`, warmup excluded
- **Metric**: Average tokens per second over all 5 tests

### Prompts Used
1. "What is machine learning? Explain briefly."
2. "Write a Python function to calculate factorial using recursion."
3. "Explain quantum computing in simple terms."
4. "Describe the process of photosynthesis."
5. "What are the main differences between HTTP and HTTPS?"

This ensures the benchmark covers explanation, code, science, and technical topics.

## Recommendation

For DGX Spark users, the evidence is clear: **AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4** with vLLM `cu130-nightly` and `--quantization compressed-tensors` is the optimal configuration, delivering **5× the speed** of alternatives while maintaining excellent quality and offering uncensored outputs.
