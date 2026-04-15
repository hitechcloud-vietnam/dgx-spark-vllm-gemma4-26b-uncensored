# Architecture Deep Dive: DGX Spark + vLLM + Blackwell

This document explains why this specific combination of hardware, software, and model format achieves exceptional performance on the DGX Spark.

## The Hardware: NVIDIA DGX Spark (GB10)

The DGX Spark features the **GB10 GPU**, a Blackwell-architecture chip with:

- **SM 12.1** (Streaming Multiprocessor 12.1)
- **Blackwell Tensor Cores** with native FP4 support
- **128GB unified memory** (LPDDR5x, shared with CPU)
- **CUDA Compute Capability 10.1**

### Why This Matters

Blackwell introduces **NVFP4** (NVIDIA FP4) — a 4-bit floating-point format that:
1. Reduces model size by 4× compared to FP16
2. Uses dedicated FP4 tensor cores for inference
3. Maintains quality comparable to INT8 quantization

However, FP4 only works with **CUDA 13.0+** and requires kernels specifically compiled for SM12.1.

## The Software Stack

```
┌─────────────────────────────────────┐
│  vLLM v0.19+ (cu130-nightly)       │  ← Blackwell FP4 kernels
│  ├── FlashInfer Cutlass NVFP4      │  ← Optimized GEMM
│  ├── CUDA Graphs                   │  ← Avoid Python overhead
│  ├── Chunked Prefill               │  ← Better scheduling
│  └── FP8 KV Cache                  │  ← Memory efficiency
├─────────────────────────────────────┤
│  PyTorch 2.3+ (CUDA 13)            │  ← Blackwell support
├─────────────────────────────────────┤
│  CUDA 13.0 Toolkit                 │  ← SM12.1 compilation
├─────────────────────────────────────┤
│  NVIDIA Driver 580+                │  ← Blackwell firmware
└─────────────────────────────────────┘
```

## The Model Format: Compressed Tensors + NVFP4

### Why This Format Wins

| Format | Quantization Backend | DGX Spark Performance |
|--------|---------------------|----------------------|
| **compressed-tensors NVFP4** ✅ | `FlashInferCutlassNvFp4LinearKernel` | **45+ tok/s** |
| modelopt NVFP4 | Slower fallback path | ~9 tok/s |
| GGUF (Ollama) | CPU-like kernels | ~8 tok/s |
| FP16/BF16 | No quantization | Too large, won't fit |

The **AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4** model is natively stored in `compressed-tensors` format with NVFP4 quantization. This allows vLLM to use its fastest kernel path.

### Technical Details

```python
# From vLLM logs
(EngineCore pid=144) Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM
(EngineCore pid=144) Using 'VLLM_CUTLASS' NvFp4 MoE backend
```

This kernel:
- Uses Blackwell's native FP4 tensor cores
- Implements fused dequantization + matmul
- Minimizes memory bandwidth bottlenecks

## Performance Optimizations Explained

### 1. CUDA Graphs

**Problem**: Python overhead per token generation is ~10-20ms.

**Solution**: vLLM captures CUDA graphs for common batch sizes (1, 2, 4, 8, ..., 256).

**Result**: Once compiled, token generation uses pure GPU kernels with no Python intervention.

```python
# vLLM config
cudagraph_mode='full_and_piecewise'
cudagraph_capture_sizes=[1, 2, 4, 8, ..., 256]
```

### 2. Chunked Prefill

**Problem**: Long prompts block short generation requests.

**Solution**: Split prefill (prompt processing) into chunks that interleave with decode steps.

**Flag**: `--enable-chunked-prefill`

### 3. FP8 KV Cache

**Problem**: KV cache for 262K context with BF16 = ~60GB (too large).

**Solution**: Store KV cache in FP8 (1 byte per value vs 2 bytes).

**Result**: Halves KV cache memory, allows larger batches.

**Flag**: `--kv-cache-dtype fp8`

### 4. Prefix Caching

**Problem**: Common system prompts get recomputed for every request.

**Solution**: Cache the KV values for repeated prefix tokens.

**Flag**: `--enable-prefix-caching`

## Memory Layout on DGX Spark

With `--gpu-memory-utilization 0.60` and 128GB unified memory:

```
Total GPU Memory: ~128GB
├─ Model Weights (NVFP4): ~16.3 GB
├─ KV Cache (FP8): ~40 GB (for 262K context)
├─ CUDA Graphs: ~5 GB
├─ Activations/Scratch: ~10 GB
└─ Free for Batching: ~56 GB
```

This allows:
- 262K max context per request
- Up to 128 concurrent sequences
- 131K batched tokens

## Why 26B Beats 31B

Counter-intuitively, the 26B model is **5× faster** than 31B variants:

| Model | Size | Quantization | Kernel Path | Speed |
|-------|------|--------------|-------------|-------|
| 26B Uncensored | 26B | compressed-tensors NVFP4 | Fast | **45 tok/s** |
| 31B LilaRest | 31B | modelopt NVFP4 | Slow fallback | 9 tok/s |

The 31B model uses the `modelopt` quantization backend, which doesn't have optimized kernels for Blackwell SM12.1, causing it to fall back to slower generic paths.

## Inference Flow

```
1. Request received
   ↓
2. Tokenize (CPU, ~1ms)
   ↓
3. Prefill Phase (process prompt)
   ├─ Use cached prefix tokens (if available)
   └─ Chunk processing to avoid blocking
   ↓
4. Decode Phase (generate tokens)
   ├─ Execute CUDA graph for current batch size
   ├─ NVFP4 GEMM on tensor cores
   └─ Sample next token
   ↓
5. Stream response
```

## Compilation Caching

vLLM uses `torch.compile` with persistent caching:

```
Cache Location: ~/.cache/vllm/torch_compile_cache/
Purpose: Avoid recompiling CUDA graphs on restart
Size: ~1-2 GB
Benefit: First request is fast instead of 55s compile
```

## Monitoring

Watch these metrics during inference:

```bash
# GPU utilization
nvidia-smi dmon -s u

# vLLM metrics (if enabled)
curl http://localhost:8000/metrics
```

Expected:
- GPU Utilization: 90-100% during generation
- Power: 60-70W (DGX Spark GB10)
- Memory: ~60GB used

## References

- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/blackwell/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [CUDA 13.0 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
