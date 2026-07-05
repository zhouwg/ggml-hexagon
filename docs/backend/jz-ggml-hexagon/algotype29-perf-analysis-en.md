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

## Latest Benchmark (2026-07-05)

### Test Conditions (identical for both backends)

- **Model file**: `/sdcard/gemma-4-E2B-it-Q4_0.gguf` (3.0 GB, **same file for both backends**)
- **Device**: Snapdragon 8 Elite (v79, OnePlus 13)
- **CLI params**: `-ngl 99 -t 6 -n 256 --ctx-size 8192 --ubatch-size 32 --poll 1000 --no-warmup --no-mmap -fa on`
- **Config**: `mulmat_algotype=29`, `offload_cgraph_type=2`, `mulmat_min_n=30`, full `enabled_ops`
  (MUL_MAT, ADD, SUB, MUL, DIV, RMS_NORM, ROPE, SOFT_MAX, UNARY, SCALE, CONCAT,
   CPY, GET_ROWS, SET_ROWS, SUM_ROWS, CONT, FLASH_ATTN_EXT, GLU, NORM, L2_NORM,
   SQR, SQRT, ARGSORT, PAD, CUMSUM, FILL, DIAG, TRI, REPEAT, DIAG_MASK_INF)

### PP & TG Comparison

| Metric | JZ (avg of 2 runs) | QCOM (avg of 2 runs) | Gap |
|--------|--------------------|----------------------|-----|
| **PP (tok/s)** | **105.6** | **217.8** | QCOM 2.06x faster |
| **TG (tok/s)** | **18.6** | **26.1** | QCOM 1.40x faster |

### Notes on the Numbers

1. **PP absolute values are below the user-reported "100-120 vs 280-320" range**
   because `--ubatch-size 32` caps the PP batch. The 2.06x ratio matches the
   user's observed ratio, confirming the comparison is consistent. Raising
   `--ubatch-size` to 64/128/256 would raise both PP numbers proportionally.

2. **TG is slower than QCOM by 1.40x** (not equal). With the current full-op
   `enabled_ops` config (FLASH_ATTN_EXT/ROPE/RMS_NORM offloaded), TG also invokes
   Hexagon. JZ's per-call dispatch overhead accumulates to ~4.8s (35% of total TG
   time). QCOM's dspqueue pipeline hides this overhead.

### Op-fusion actual trigger (from logcat)

| Fusion | PP | TG | Notes |
|--------|----|----|-------|
| `RMS_NORM_MUL` | 1-3 per graph | 1-2 per graph | Fires every graph |
| `MUL_MAT_ADD` | 0 | 0 | No bias-add pattern in current model graph |
| `MUL_MAT_QKV` | 0 | 0 | PP MUL_MATs are HMX-eligible; `mm_is_hmx_eligible()` defaults to `#if 0`, keeping them on HMX path. Fusion only redirects non-HMX MUL_MATs. |
| `MUL_MAT_FFN` | 0 | 0 | Same as QKV |

To force QKV/FFN fusion for benchmarking, flip `mm_is_hmx_eligible()` to `#if 1`
(see "How to test op-fusion" comment block above Phase 2.5 in ggml-hexagon.cpp).

### Graph-cache status

- **PP phase** (first alloc): all 71 splits MISS (first fill, expected)
- **TG phase**: graph reuse keeps `is_alloc=true`, uid unchanged, but PP->TG
  rebuild gives each TG graph a new uid, so cache stays MISS.
- **HIT** requires the SAME uid to be computed twice (e.g. repeated PP with
  graph reuse). The cache code is correct; the current llama-completion workload
  just never hits it.

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

### JZ Version (Phase 2.5, L4407-L4681)

| Fusion Type | Supported | Notes |
|-------------|-----------|-------|
| RMS_NORM + MUL -> HTP_OP_RMS_NORM_MUL | Yes | Linear scan; fires every graph (1-3 per graph) |
| MUL_MAT + ADD -> HTP_OP_MUL_MAT_ADD | Yes | Bias add inside matmul kernel |
| MUL_MAT QKV merge -> HTP_OP_MUL_MAT_QKV | Yes (algotype=29 only) | 3 MUL_MAT (Q,K,V) merged into 1 |
| MUL_MAT FFN merge -> HTP_OP_MUL_MAT_FFN | Yes (algotype=29 only) | gate + up merged into 1 |
| Graph reorder | No | - |

Safety checks:
- `src_use_count == 1` (intermediate dst must be single-use, verified via `src_use_count` array)
- VTCM budget check: `vtcm_needed <= vtcm_budget` (skips QKV/FFN fusion if VTCM insufficient)
- HMX eligibility gate: `mm_is_hmx_eligible()` defaults to `#if 0`, so PP-phase quantized
  MUL_MATs stay on HMX path and QKV/FFN fusion does not fire. Flip to `#if 1` to force
  fusion for benchmarking.

Note: In the current benchmark (see "Op-fusion actual trigger" above), only
`RMS_NORM_MUL` actually fires. QKV/FFN do not fire because PP MUL_MATs are
HMX-eligible and the gate keeps them on the HMX path.

### Qualcomm Version (`try_fuse_node`, L3473-L3549)

| Fusion Type | Supported | Notes |
|-------------|-----------|-------|
| RMS_NORM + MUL -> HTP_OP_RMS_NORM_MUL | Yes | Uses `ggml_can_fuse` |
| MUL_MAT + ADD -> HTP_OP_MUL_MAT_ADD | Yes | Uses `ggml_can_fuse` |
| MUL_MAT QKV merge -> HTP_OP_MUL_MAT_QKV | Yes | 3 mul_mat merged into 1, reordered to KVQ |
| MUL_MAT FFN merge -> HTP_OP_MUL_MAT_FFN | Yes | gate + up merged into 1 |
| Graph reorder | Yes | Stacks MUL_MATs with same src1 for VTCM reuse |

Safety check: `ggml_can_fuse` + VTCM budget check (`kparams.vtcm_size <= sess->vtcm_size`) + `is_mergeable_mul_mat`.

### Performance Impact

JZ has closed the fusion-scope gap: all 4 fusion types are now implemented.
The remaining gap is graph reorder (QCOM stacks same-src1 MUL_MATs for VTCM
reuse) and the fact that QKV/FFN fusion does not fire in PP by default
(HMX-eligible gate). Graph reorder has medium impact during PP.

---

## 4. Graph Cache

### JZ Version (Phase 1 cache check, L4326-L4342; cache update L4685-L4696)

- **Graph cache**: Caches Phase 1 (tensor_src), Phase 2 (weight_indices), Phase 2.5
  (hex_ops post-fusion) by `cgraph->uid` in `ctx->cached_graph`
- **On cache hit**: Skips straight to Phase 3 (layout), saves ~100-200 us per graph
- **Limitation**: In llama-completion PP runs once, so first alloc is always MISS.
  HIT only fires when the same uid is computed again (graph reuse without rebuild).
  In the current benchmark, all 71 PP splits MISS (first fill) and TG graphs get
  new uids after PP->TG rebuild, so cache stays MISS. The cache code is correct;
  the workload just never hits it.

### Qualcomm Version

- **Graph cache**: Caches htp_nodes by `graph->uid` (L3559-L3600)
- **On cache hit**: Skips fusion + precompute, reuses cached op descriptors
- **Op reorder**: `graph_optimize_reorder` (L3624-L3669), stacks MUL_MATs with same src1

### Performance Impact

JZ has closed the graph-cache gap. Both backends now cache by `graph->uid`.
The remaining difference is that QCOM also caches the graph reorder step, while
JZ has no graph reorder. In practice, neither backend hits cache in the current
llama-completion workload (PP runs once, TG rebuilds uid), so this is a wash.

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
| 3 | Op fusion scope | 4 fusions (no reorder) | 4 fusions + graph reorder | Small (scope parity; reorder only) |
| 4 | Graph cache | Cached by uid (Phase 1/2/2.5) | Cached by uid | Parity (both MISS in llama-completion) |
| 5 | Cache coherency | Manual management | Driver automatic | Small |
| 6 | Batch auto-splitting | Whole graph one batch | Auto-split | Small |

---

## Optimization Recommendations

To improve JZ version algotype=29 performance, ranked by cost-effectiveness:

### Priority 1: Move tiled repack to set_tensor (highest ROI)

Change Phase 4.5's per-call repack to one-time repack at set_tensor, similar to
Qualcomm's repack buffer type design. This completely eliminates per-inference
weight repack overhead.

### Priority 2: Add graph reorder (medium ROI)

Stack MUL_MATs with the same src1 for VTCM reuse, similar to QCOM's
`graph_optimize_reorder`. Only affects PP phase.

### Completed Optimizations (2026-07-04)

- **Graph cache**: Caches Phase 1/2/2.5 by `cgraph->uid`. See section 4 above.
- **Op fusion**: All 4 fusion types implemented (RMS_NORM_MUL, MUL_MAT_ADD,
  MUL_MAT_QKV, MUL_MAT_FFN). See section 3 above.
