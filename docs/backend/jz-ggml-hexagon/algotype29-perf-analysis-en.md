# algotype=29 Performance Analysis: JZ vs Qualcomm ggml-hexagon Backend

## Background

When `mulmat_algotype=29`, both the JZ and Qualcomm versions of the ggml-hexagon backend route through Qualcomm's `execute_op` path (the `execute_op` implementation lives in `htp/` and is shared by both versions). However, the DSP entry points differ: JZ uses `kernels/entry.c`, while Qualcomm uses `htp/main.c`. The AP-side implementations differ significantly, leading to substantial performance differences.

This document provides a comprehensive comparison of the AP-side differences when algotype=29, ranked by performance impact (largest to smallest).

## Relevant Files

| File | Description |
|------|-------------|
| `ggml/src/ggml-hexagon/ggml-hexagon.cpp` | JZ version AP code (5491 lines) |
| `ggml/src/ggml-hexagon/ggml-hexagon-qcom.cpp` | Qualcomm version AP code (4392 lines) |
| `ggml/src/ggml-hexagon/kernels/entry.c` | JZ version DSP entry point |
| `ggml/src/ggml-hexagon/htp/main.c` | Qualcomm version DSP entry point |
| `ggml/src/ggml-hexagon/htp/matmul-ops.c` | Qualcomm DSP matmul kernel (shared) |

> Note: Ignore all files under the `refs/` directory in the project root.

---

## 1. Weight Repack Timing (Largest Difference)

### Problem

Qualcomm HMX kernels (`hvx_mm_2d_repacked_*`) expect Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4 weights in tile-based layout. Both versions perform tiled repack, but at completely different times.

### JZ Version

- **Timing**: Every `graph_compute_batch` call
- **Implementation**: Phase 4.5 clears `g_tiled_ion_offsets` and repacks all weights
- **Location**: `ggml-hexagon.cpp` Phase 4.5 (L4478-L4612)
- **Cost**: Full tiled repack + memcpy of all quantized weights on every inference
- **Lifecycle**: Repacked to temporary ION region, freed after Phase 8

```cpp
// JZ: clear + full repack on every call
g_tiled_ion_offsets.clear();
for (uint32_t i = 0; i < n_tensors; i++) {
    // ... repack Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4 ...
}
```

### Qualcomm Version

- **Timing**: `set_tensor` (during model load, one-time)
- **Implementation**: `ggml_backend_hexagon_repack_buffer_type`
- **Location**: `ggml-hexagon-qcom.cpp` L1005, L860-906, L1058-1064
- **Cost**: Zero (repacked data used directly during inference)
- **Lifecycle**: Repacked data permanently stored in shared buffer

```cpp
// QCom: repack once at set_tensor
if (ggml_backend_buffer_is_hexagon_repack(buf)) {
    // repack to tiled layout, stored permanently in shared buffer
}
```

### Performance Impact

For a typical LLM model (hundreds of MB of quantized weights), the JZ version repacks all weights on every inference, consuming significant CPU time and memory bandwidth. This is the largest source of performance difference under algotype=29.

---

## 2. FastRPC Call Pattern

### JZ Version

- **Call method**: Single synchronous `ggmlop_dsp_execute_batch`
- **Parameters**: 2 scalars (`batch_offset`, `total_desc_size`)
- **Data transfer**: Single ION mempool + offset addressing
- **Pipelining**: None (AP blocks waiting for DSP completion)

```cpp
// JZ: synchronous call, AP waits for DSP
int hexagon_error = ggmlop_dsp_execute_batch(ctx->ggmlop_handle,
                                              batch_offset,
                                              total_desc_size);
```

### Qualcomm Version

- **Call method**: dspqueue message queue
- **Parameters**: `dspqueue_write` + `dspqueue_buffer` (containing `fd + offset + size`)
- **Data transfer**: fd + offset two-level addressing, supporting multiple independent shared buffers
- **Pipelining**: Up to 16 batches in-flight (`opt_opqueue=16`), AP/DSP parallel execution

```cpp
// QCom: dspqueue pipeline
int err = dspqueue_write(this->queue, 0, 1, &dbuf, sizeof(req),
                          (const uint8_t*) &req, DSPQUEUE_TIMEOUT);
```

### Performance Impact

The Qualcomm version's pipeline allows the AP to build batch N+1 while the DSP executes batch N, hiding AP-side batch construction overhead. The JZ version is strictly sequential: build batch -> submit -> wait for DSP -> collect results -> next round.

---

## 3. Op Fusion Scope

### JZ Version (Phase 2.5, L4310-L4384)

| Fusion Type | Supported | Notes |
|-------------|-----------|-------|
| RMS_NORM + MUL -> HTP_OP_RMS_NORM_MUL | Yes | Simple linear scan of adjacent ops |
| MUL_MAT + ADD -> HTP_OP_MUL_MAT_ADD | Yes | Bias add / residual add |
| MUL_MAT QKV merge | **No** | - |
| MUL_MAT FFN merge | **No** | - |
| Graph reorder | **No** | - |

Safety check: Only checks `src_use_count == 1` (intermediate dst is single-use), no VTCM budget check.

### Qualcomm Version (`try_fuse_node`, L3473-L3549)

| Fusion Type | Supported | Notes |
|-------------|-----------|-------|
| RMS_NORM + MUL -> HTP_OP_RMS_NORM_MUL | Yes | Uses `ggml_can_fuse` |
| MUL_MAT + ADD -> HTP_OP_MUL_MAT_ADD | Yes | Uses `ggml_can_fuse` |
| MUL_MAT QKV merge -> HTP_OP_MUL_MAT_QKV | **Yes** | 3 mul_mat merged into 1, reordered to KVQ |
| MUL_MAT FFN merge -> HTP_OP_MUL_MAT_FFN | **Yes** | gate + up merged into 1 |
| Graph reorder | **Yes** | Stacks MUL_MATs with same src1 for VTCM reuse |

Safety check: `ggml_can_fuse` + VTCM budget check (`kparams.vtcm_size <= sess->vtcm_size`) + `is_mergeable_mul_mat`.

### Performance Impact

QKV fusion reduces 2 MUL_MAT dispatches per layer, FFN fusion reduces 1 per layer. Graph reorder optimizes src1 VTCM reuse. These have significant impact during PP (batch processing) phase.

---

## 4. Graph Cache

### JZ Version

- **Graph cache**: None
- **Every inference**: Rebuilds hex_op_desc array, re-runs op fusion, re-runs weight repack

### Qualcomm Version

- **Graph cache**: Caches htp_nodes by `graph->uid` (L3559-L3600)
- **On cache hit**: Skips fusion + precompute, reuses cached op descriptors
- **Op reorder**: `graph_optimize_reorder` (L3624-L3669), stacks MUL_MATs with same src1

### Performance Impact

The Qualcomm version can skip AP-side graph construction overhead during repeated inference (e.g., same graph in TG phase). The JZ version starts from scratch every time.

---

## 5. Cache Coherency Management

### JZ Version

- **Method**: Manual DC CVAC/CIVAC management
- **Flush**: Phase 6.5, manually flush with range merging
- **Invalidate**: Phase 7.5, manually invalidate with range merging

### Qualcomm Version

- **Method**: dspqueue driver automatic management
- **Flags**: `DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT`
- **Advantage**: Driver-level optimization, reduced userspace overhead

### Performance Impact

Manual cache management has userspace overhead in scenarios with many tensors. dspqueue's driver-level management is more efficient.

---

## 6. Batch Auto-Splitting

### JZ Version

- **Strategy**: Entire graph as one batch, no splitting
- **Limitation**: May be constrained by ION pool size

### Qualcomm Version

- **Strategy**: Auto-split into multiple batches based on vmem/buffer/tensor limits
- **Implementation**: `enqueue_op` with `if (!op_batch->fit_op(node)) flush_batch()`
- **Advantage**: Adapts to different graph sizes, avoids memory overflow

---

## 7. Tensor Descriptor Data Structure

### JZ Version `hex_tensor_desc`

```c
typedef struct hex_tensor_desc {
    int32_t  type;
    int32_t  ne[4];
    int32_t  nb[4];
    int32_t  op_params[16];  // op-specific params (includes FP16 cache request)
    uint32_t flags;          // 0=ION, 1=mirrored, 2=weight(skip flush)
    uint32_t data_offset;   // offset relative to ION mempool base
    uint32_t data_len;
} hex_tensor_desc;
```

- Single ION offset addressing
- flags encodes cache strategy
- op_params embedded in tensor descriptor

### Qualcomm Version `htp_tensor`

```c
struct htp_tensor {
    uint32_t data;       // offset within buffer
    uint32_t size;
    uint32_t flags;      // HTP_TENSOR_COMPUTE / HTP_TENSOR_FLUSHED
    uint16_t type;
    uint16_t bi;         // buffer index (points to htp_buf_desc array)
    uint32_t ne[4];
    uint32_t nb[4];
};
```

- Two-level addressing: `bi` (buffer index) + `data` (offset)
- Supports multiple independent shared buffers
- fd can be directly mmapped by DSP

---

## Summary: Performance Difference Ranking

| Rank | Difference | JZ Version | Qualcomm Version | Impact |
|------|-----------|------------|-----------------|--------|
| 1 | Weight repack timing | Every graph_compute_batch | Once at set_tensor | **Largest** |
| 2 | FastRPC call pattern | Synchronous blocking | dspqueue pipeline | Large |
| 3 | Op fusion scope | 2 fusions | 4 fusions + reorder | Medium |
| 4 | Graph cache | None | Cached by uid | Medium |
| 5 | Cache coherency | Manual management | Driver automatic | Small |
| 6 | Batch auto-splitting | Whole graph one batch | Auto-split | Small |

---

## Optimization Recommendations

To improve JZ version algotype=29 performance, ranked by cost-effectiveness:

### Priority 1: Move tiled repack to set_tensor (highest ROI)

Change Phase 4.5's per-call repack to one-time repack at set_tensor, similar to Qualcomm's repack buffer type design. This completely eliminates per-inference weight repack overhead.

### Priority 2: Introduce graph cache

Cache hex_op_desc array and fusion results by `graph->uid` to avoid redundant construction.

### Priority 3: Extend op fusion

Add QKV fusion and FFN fusion to reduce MUL_MAT dispatch count.

### Priority 4: Introduce dspqueue pipeline (largest change)

Migrate from synchronous FastRPC calls to dspqueue message queue for AP/DSP pipelining. This requires significant architectural changes.
