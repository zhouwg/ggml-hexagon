# HMX Pipeline Design Draft

Design draft for migrating Qualcomm's HMX pipeline framework into our
ggml-hexagon backend. This document captures the architectural reasoning
and design decisions made before implementation; see HMX_PIPELINE_MIGRATION.md
for task tracking and verification results.

## 1. Background

### 1.1 Problem

llama.cpp on Hexagon cDSP (algotype=30, HMX sync mode) shows a large PP
(prompt processing) performance gap vs Qualcomm's HTP reference:

| Path      | PP (tok/s) | TG (tok/s) | Notes                     |
|-----------|------------|------------|---------------------------|
| a30 sync  | ~9         | ~17        | Our baseline              |
| HTP ref   | ~300       | ~17        | Qualcomm HTP reference    |

TG (token generation) is competitive; the gap is concentrated in PP.

### 1.2 Root Cause Analysis

HMX compute kernel is shared (same htp-ops-lib origin). The gap is in the
scheduling framework:

- Our HMX path (algotype=30) runs everything in the calling thread
  synchronously: lock HMX -> dequant weight -> compute -> unlock. No overlap
  between DMA, HVX dequant, and HMX compute.
- Qualcomm's reference uses hmx_queue (async HMX) + worker_pool (multi-thread
  HVX) + DMA prefetch + double buffer, achieving DMA/HVX/HMX pipeline
  overlap. While HMX is computing tile N, HVX is dequanting tile N+1, and
  DMA is prefetching tile N+2.

### 1.3 Goal

Close the PP gap by porting Qualcomm's pipeline framework. Target: PP >= 15
tok/s (1.5x+ over sync baseline) while keeping TG at parity with sync.

## 2. Architecture

### 2.1 Three-Stage Pipeline

```
 Stage 1: DMA            Stage 2: HVX             Stage 3: HMX
 (weight prefetch)       (dequant + writeback)    (async compute)
 +-----------------+     +------------------+     +-----------------+
 | dma_queue        |    | worker_pool       |    | hmx_queue        |
 | weight_raw[0/1]  | -> | weight_fp16[0/1] | -> | async submit     |
 | double buffered  |     | output[0/1]      |     | double buffered  |
 +-----------------+     +------------------+     +-----------------+
        |                         |                         |
   DMA engine              HVX threads (6)            HMX hardware
                                                    (single lock)
```

- Stage 1 (DMA): Prefetch weight tiles from DDR to VTCM via dma_queue.
  Double buffered (weight_raw[2]) so DMA of tile N+1 overlaps with compute
  of tile N.
- Stage 2 (HVX): Dequantize weight (Q4_0/Q4_1/BF16 -> FP16) and write back
  output, using worker_pool (6 HVX threads). Double buffered
  (weight_fp16[2], output[2]).
- Stage 3 (HMX): Submit matmul tiles to hmx_queue. The hmx_queue worker
  thread holds the HMX hardware lock for the batch, so the calling thread
  can return to prepare the next tile (Stage 1+2) while HMX computes.

### 2.2 Activation Handling

Activation (src1) is single-buffered (not double). Rationale:
- Activation is small (K x N, N typically 32-39 for PP).
- Activation needs FP32 -> FP16 conversion (or FP32 direct), done once per
  N-chunk at pipeline entry.
- Weight dominates VTCM budget, so double-buffering weight is the priority.

### 2.3 Tile Chunking

Following Qualcomm's hmx_mm_2d_precomputed:
- M_chunk_n_cols: weight columns per chunk (bounded by VTCM budget)
- N_chunk_n_rows: activation rows per chunk (must be multiple of
  HMX_FP16_TILE_N_ROWS = 32)
- K must be multiple of HMX_FP16_TILE_N_COLS = 32

VTCM budget search: start from max M_chunk, halve until fits in 8MB VTCM.
Double buffer requires: weight_raw x2 + weight_fp16 x2 + act + output x2 +
act_fp32 + scales <= vtcm_size.

### 2.4 Pipeline Condition

Pipeline is entered only when N > HMX_FP16_TILE_N_ROWS (32). This matches
Qualcomm's htp_mm_hmx_pipeline heuristic:
- Large N (PP phase, batch decoding): pipeline overlap pays off
- Small N (N=1 GEMV decode): sync mode avoids pipeline overhead

Note: gemma-4-E2B has N=39 in both PP and decode phases (model-specific
attention structure), so decode also enters pipeline. This is fine as long
as per-mulmat pipeline overhead is low (see section 4.2 for the sync-log
trap that initially regressed TG).

## 3. Component Design

### 3.1 hmx_queue (Step 1-2)

Ported from htp/hmx-queue.{h,c} with adaptations:
- Removed Qualcomm's htp_thread_trace instrumentation (we have no such infra)
- Fixed depth() bug: was (idx_read - idx_read), now (idx_write - idx_read)
- Fixed memory leak in error path of hmx_queue_create
- Fixed atomicity bugs discovered during integration (see 3.1.1)

Lifecycle:
- Created in ggmlop_dsp_open (after power_on_hvx_hmx + VTCM alloc)
- Persisted across mulmat calls (not per-mulmat create/destroy)
- Destroyed in ggmlop_dsp_close
- Per-mulmat: suspend (release HMX lock) at end, resume at next call's begin

#### 3.1.1 Atomicity Fixes (discovered during integration)

Original Qualcomm code had non-atomic access bugs that caused intermittent
pipeline hang under multi-run stress:
1. hmx_queue_process: `!d->done` (non-atomic read) -> `!atomic_load(&d->done)`
2. hmx_queue_push: `q->desc[iw] = d` (struct assignment bypasses atomic
   semantics for done field) -> explicit field assignment +
   `atomic_store(&q->desc[iw].done, 0)`

### 3.2 DMA Queue

Reuses existing dma_queue API in mulmat.c (originally for F32 activation
prefetch). Per-mulmat create(16) at pipeline begin, flush+delete at end.
Capacity 16 matches Qualcomm default.

### 3.3 worker_pool

Reuses existing worker_pool (shared default pool, 6 HVX threads). Not
per-path pool init. Used for:
- Dequant weight rows (Q4_0/Q4_1/BF16 -> FP16) in parallel across threads
- Output writeback (FP16 -> FP32) in parallel

### 3.4 FP16 Weight Cache

Preserved from algotype=32 sync path (independent optimization). On cache
hit, skip DMA + dequant entirely (weight_fp16 already in VTCM). Cache is
keyed by src0->data pointer + size, so repeated mulmat with same weight
(e.g. same layer across tokens) hits cache after first call.

### 3.5 Fallback to Sync

ggmlop_dsp_mulmat_hmx falls back to ggmlop_dsp_mulmat_hmx_sync when:
- hmx_queue unavailable (not created, or create failed)
- src0 type not supported by HMX pipeline (F32 weight needs fp32 intermediate
  buffer which pipeline VTCM layout does not allocate)
- src1 type not F32/F16/BF16
- N <= HMX_FP16_TILE_N_ROWS (small N, sync is faster)
- K not multiple of HMX_FP16_TILE_N_COLS
- Batched weights (src0->ne[2] > 1 or ne[3] > 1, rare)
- VTCM unavailable or unaligned
- hmx_queue_push fails (queue full or worker killed)

This ensures correctness for all tensor shapes; pipeline is a pure
performance optimization.

## 4. Key Decisions

### 4.1 Single hmx_queue + Single worker_pool

Match Qualcomm's htp_context design. One global hmx_queue, one global
worker_pool. Avoids resource fragmentation and simplifies lifecycle
management.

### 4.2 The FARF(ALWAYS) Sync-Log Trap

Critical lesson: `GGMLHEXAGON_LOG_INFO` -> `ggml_log_always` -> `FARF(ALWAYS)`
is SYNCHRONOUS on DSP. The DSP thread blocks until logcat delivers the
message. Pipeline emits 6+ such logs per mulmat (begin/state/cleanup/end).
At decode phase (N=39 enters pipeline), 9 mulmats/token * 6 logs = 54 sync
FARF calls/token, adding ~770ms/token overhead -> TG 1.14 tok/s (15x
regression vs sync 17.12).

Fix: convert pipeline's 8 normal-path log calls to `PIPE_DBG` macro (disabled
by `#if 0`, re-enable for debugging). Keep fallback/error logs (rare, low
volume).

This trap is non-obvious because the logs look harmless and the regression
appears as "fastrpc overhead" or "CPU-side cost" in profiling. The root
cause is that FARF(ALWAYS) on cDSP is a blocking syscall, not async printf.

### 4.3 HMX Lock Management

HMX hardware lock (HAP_compute_res_hmx_lock) is managed inside hmx_queue:
- hmx_queue worker thread acquires lock on first real task, holds it for
  the batch
- On SUSPEND signal (end of mulmat), worker releases lock
- Sync path (ggmlop_dsp_mulmat_hmx_sync) acquires lock directly

This means sync and pipeline paths do not contend for HMX lock as long as
pipeline properly suspends at end of each mulmat.

### 4.4 Double Buffer Priority

When VTCM is too small for full double buffer, priority order:
1. Keep weight_raw double buffered (DMA overlap is the biggest win)
2. Keep weight_fp16 double buffered (HVX dequant overlap)
3. Fall back to single output buffer
4. If still doesn't fit, reduce M_chunk
5. If still doesn't fit, fall back to sync entirely

### 4.5 Per-Mulmat DMA Queue Lifecycle

DMA queue is created per-mulmat (not persistent). Rationale:
- DMA queue is lightweight (no thread, just a descriptor ring)
- Per-mulmat create/destroy ensures clean state, no cross-mulmat leakage
- hmx_queue is persistent (heavy: has worker thread + HMX lock state)

## 5. Performance Model

### 5.1 Expected Overlap

For a weight tile of size M_chunk x K:
- DMA time: T_dma (DDR -> VTCM, weight_raw)
- HVX time: T_hvx (dequant weight_raw -> weight_fp16, 6 threads)
- HMX time: T_hmx (matmul weight_fp16 x activation -> output)

Without pipeline (sync): T_total = T_dma + T_hvx + T_hmx (sequential)
With pipeline (3-stage, steady state): T_total = max(T_dma, T_hvx, T_hmx)

Since HMX is the bottleneck (hardware matmul), and DMA/HVX can overlap,
pipeline approaches: T_total ~= T_hmx + pipeline fill/drain overhead.

### 5.2 Observed Results

| Algotype | Mode     | PP (tok/s) | TG (tok/s) | TG (ms/tok) |
|----------|----------|------------|------------|-------------|
| 30       | sync     | 8.89       | 17.12      | 58.43       |
| 32       | pipeline | 15.72      | 1.14       | 878.85      |
| 32       | pipeline | 15-16      | 16.78      | 59.60       |

Row 3 is post-fix (PIPE_DBG conversion). PP 1.7-1.8x over sync confirms
pipeline overlap is effective. TG matches sync (pipeline overhead in
decode is negligible once sync logs are removed).

DSP-side per-mulmat elapse time confirms pipeline is 1.5-1.8x faster than
sync for the same MUL_MAT shape, validating the overlap model.

## 6. Risks and Mitigations

### 6.1 VTCM Budget

8MB VTCM must hold: weight_raw x2 + weight_fp16 x2 + act + output x2 +
act_fp32 + scales. For large M (e.g. 12288), weight dominates. M_chunk
search reduces M_chunk until fit. Worst case M_chunk=576 (observed in
gemma-4-E2B), still faster than sync.

### 6.2 Threading Correctness

hmx_queue worker thread + worker_pool threads + calling thread all access
shared VTCM buffers. Correctness relies on:
- Double buffer index isolation (producer/consumer never touch same slot)
- hmx_queue_push/pop atomicity (fixed)
- dma_queue completion ordering (FIFO)
- worker_pool barrier between dequant and writeback phases

### 6.3 Thermal Throttling

Consecutive benchmark runs cause device thermal throttling, reducing PP
observed numbers. Always cool down between runs, or compare within same
thermal state.

### 6.4 Non-Obvious: Sync Log Cost

Any FARF(ALWAYS) in hot path is a hidden performance trap. Pipeline paths
that run per-token (decode) must avoid FARF(ALWAYS) in normal execution.
Use compile-time gated macros (PIPE_DBG) for debug logs.

## 7. File Map

| File | Role |
|------|------|
| kernels/hmx-queue.h    | hmx_queue API (push/pop/flush/suspend) |
| kernels/hmx-queue.c    | hmx_queue worker thread implementation |
| kernels/mulmat.c       | ggmlop_dsp_mulmat_hmx (pipeline) + sync fallback |
| kernels/worker-pool.{h,c} | HVX thread pool (dequant/writeback) |
| kernels/ggml-dsp.c     | ggml_log_always (FARF ALWAYS, sync) |
| kernels/ggml-dsp.h     | GGMLHEXAGON_LOG_INFO macro definition |
| entry.c                | g_hmx_queue lifecycle (create in open, destroy in close) |
| htp/hmx-queue.{h,c}    | Qualcomm reference (do not modify) |
| htp/matmul-ops.c       | Qualcomm pipeline reference (hmx_mm_2d_precomputed) |

## 8. Future Work

- AP-side cgraph cache: avoid re-uploading graph per token (Phase 2)
- Op fusion (QKV / FFN): reduce fastrpc round trips (Phase 2)
- Larger N in decode: investigate why gemma-4-E2B has N=39 in decode
  (model architecture, not pipeline concern)
- Profiler integration: enable enable_profiler=1 for cDSP NPU visualization
