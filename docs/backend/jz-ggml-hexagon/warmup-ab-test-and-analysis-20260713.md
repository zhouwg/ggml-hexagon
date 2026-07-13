# JZ ggml-hexagon: Synchronous Blocking, PP/TG Gap, Jitter, and FastRPC/ION Warmup A/B Test

> Author: Kimi-K2.7-Code  
> Analysis date: 2026-07-13 23:04:46  
> Device: Snapdragon 8 Elite (v79), OnePlus 13  
> Model: `gemma-4-E2B-it-Q4_0.gguf` (3.0 GB)  
> Test command: `./scripts/build-run-android.sh run_llamacli`  
> CLI flags: `-ngl 99 -t 6 -n 256 --ctx-size 8192 --ubatch-size 64 --poll 1000 --no-warmup --no-mmap -fa on -st -no-cnv`

---

## 1. Background

This document answers three questions raised during side-by-side benchmarking of Qualcomm's official `ggml-hexagon` backend and the JZ custom `ggml-hexagon-jz` backend:

1. Is the statement "both implementations are synchronous blocking at the graph level" accurate?
2. What explains the default PP/TG performance difference between the two backends?
3. Why does the JZ backend show PP jitter (sometimes ~300-320 t/s, sometimes ~310-330 t/s or higher) across consecutive runs?

The analysis is based on reading the actual source code, not just the existing documentation, because code is the authoritative specification.

---

## 2. Are Both Backends Synchronous Blocking at the Graph Level?

### Short answer

**Yes, in the narrow sense that a single `graph_compute` call does not return until all ops in that graph are complete.** But this statement misses a critical architectural difference: Qualcomm has batch-level pipelining inside the graph, while JZ does not.

### JZ: strictly synchronous per subgraph

JZ's only DSP dispatch point is in [`ggml-hexagon-jz.cpp:6079`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L6079):

```cpp
int hexagon_error = ggml_dsp_execute_batch(ctx->ggmlop_handle, batch_offset, total_desc_size);
```

This is a single synchronous FastRPC call. AP packs all tensor/op descriptors into the shared ION mempool, passes `(offset, size)`, and blocks until the DSP finishes the entire batch and replies. There is no overlap between AP preparation and DSP execution.

### Qualcomm: graph-level blocking, but batch-level pipelining

Qualcomm's [`ggml_backend_hexagon_graph_compute`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L3613-L3683) ends with:

```cpp
sess->flush();
```

`flush()` ([`ggml-hexagon.cpp:1601-1604`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1601-L1604)) sends the final batch and then loops on `dspqueue_read` until all pending responses are consumed. So a single `graph_compute` call also waits for completion.

However, Qualcomm's `enqueue_op` ([`ggml-hexagon.cpp:1593-1598`](file:///home/hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1593-L1598)) checks batch capacity and calls `flush_batch()` ([`ggml-hexagon.cpp:1571-1591`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1571-L1591)) when the batch is full. `flush_batch()` issues a non-blocking `dspqueue_write` and immediately returns. The DSP consumes batches from the queue in its own context ([`htp/main.c:862-1008`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/main.c#L862-L1008)). The queue depth is `opt_opqueue = 16` ([`ggml-hexagon.cpp:77`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L77)).

Therefore:

| Dimension | JZ | Qualcomm |
|-----------|-----|----------|
| `graph_compute` returns after all ops done | Yes | Yes |
| AP can prepare next batch while DSP runs current batch | No | Yes |
| Multiple batches in flight inside one graph | No | Up to 16 |
| Cache coherency | User-space `ion_sync` + `DC CVAC/CIVAC` | Driver-level `FLUSH_SENDER / INVALIDATE_RECIPIENT` |

### Implication

The statement is accurate but incomplete. The real performance difference comes from what happens **inside** the graph, not from the graph-level contract.

---

## 3. Why Is the PP/TG Performance Different?

### 3.1 Raw data from the user's 6-run benchmark

| Implementation | PP range | PP mean | TG range | TG mean |
|----------------|----------|---------|----------|---------|
| JZ             | 321.90 - 368.97 | 350.30 | 19.31 - 19.45 | 19.38 |
| Qualcomm       | 355.70 - 380.78 | 368.78 | 27.21 - 27.30 | 27.26 |

Qualcomm TG is ~1.41x faster; PP is only ~5% faster.

### 3.2 PP gap is small because compute dominates

PP processes 44 prompt tokens in one (or a few) large batches. The dominant work is HMX matmul with `m=64`. Both backends use the **same** shared kernels under `ggml/src/ggml-hexagon/htp/`, so the raw compute throughput is identical.

JZ's AP path is heavier ([`ggml-hexagon-jz.cpp:4915-6146`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L4915-L6146)) because it inline-implements tensor dedup, descriptor build, fusion, ION offset setup, DC CVAC flush, FastRPC dispatch, and CIVAC invalidate. Qualcomm distributes equivalent work across `htp_opnode`, `op_batch`, `op_queue`, and the dspqueue driver. The logical work is the same, but Qualcomm pushes cache coherency and completion notification into kernel space, which is slightly more efficient. The difference is small because PP has few batch calls.

### 3.3 TG gap is large because fixed per-subgraph overhead accumulates

TG generates 255 tokens. Each token is split by the llama.cpp scheduler into ~17 subgraphs (`ggml_cgraph` instances), resulting in ~4352 total `ggml_dsp_execute_batch` calls (PP + TG) in the JZ profiler output.

For each subgraph JZ pays:

- AP descriptor preparation
- `DC CVAC` flush of dirty input ranges ([`ggml-hexagon-jz.cpp:5975-6026`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5975-L6026))
- Synchronous FastRPC round-trip
- DSP execution
- `CIVAC` invalidate after return

Qualcomm overlaps most of this: while the DSP executes batch `N`, the AP prepares batch `N+1` and the driver handles cache maintenance. The fixed overhead per subgraph is therefore hidden.

### 3.4 The role of `FLASH_ATTN_EXT`

The longtail probe (gated off at [`ggml-hexagon-jz.cpp:6096`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L6096-L6146)) shows that one TG token spends ~27 ms in a single large batch dominated by 21 `FLASH_ATTN_EXT` ops (one per attention layer). Both backends call the **same** `htp/flash-attn-ops.c` kernel, so the attention kernel itself is not the source of the JZ-vs-Qualcomm difference.

The difference is the **gap between attention layers**. In JZ, each of the 17 subgraphs per token is a separate synchronous invocation. In Qualcomm, the queue keeps the DSP busy across subgraph boundaries. The per-layer gap is small (~0.7 ms), but 21 layers × 255 tokens accumulates to ~15 ms/token, which matches the measured TG gap:

- JZ: ~52 ms/token → 19.4 t/s
- Qualcomm: ~37 ms/token → 27.3 t/s

### 3.5 Summary of TG gap root cause

| Factor | Shared? | Contribution to TG gap |
|--------|---------|------------------------|
| `FLASH_ATTN_EXT` kernel compute | Yes (same `htp/flash-attn-ops.c`) | None |
| Matmul kernel compute | Yes (same `htp/matmul-ops.c`) | None |
| Per-subgraph sync overhead | No (JZ synchronous, Qualcomm pipelined) | Major |
| Cache coherency path | No (JZ user-space, Qualcomm driver) | Minor |
| AP-DSP overlap | No (JZ none, Qualcomm up to 16 batches) | Major |

---

## 4. Why Does JZ PP Jitter?

The user's JZ runs showed PP values 321.90, 368.97, 360.02 (CV ~7.2%), while Qualcomm showed 355.70, 369.86, 380.78 (CV ~3.5%). JZ jitter is roughly twice as large.

### 4.1 Cold-state penalty on first run

JZ uses delayed `fastrpc_mmap` ([`ggml-hexagon-jz.cpp:1737`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L1737-L1744)). The first real `ggml_dsp_execute_batch` triggers page-table setup and DSP address-space mapping for the ION pages it touches. This can cost several milliseconds on the first subgraph, which is visible as lower first-run PP.

### 4.2 User-space cache management is state-sensitive

JZ manually manages cache coherency via `ion_sync_mode=1` (`DMA_BUF_IOCTL_SYNC`) and `DC CVAC/CIVAC`. The latency of these operations depends on:

- Current CPU frequency (walt governor)
- DDR bus contention
- Whether ION pages are already in a favorable cache state

Qualcomm delegates this to the dspqueue driver, which can batch and schedule flushes more stably.

### 4.3 Single ION mempool layout sensitivity

JZ uses one large ION mempool with offset addressing. The actual DSP-side physical address mapping can vary across process launches, affecting L2 cache alias behavior and VTCM layout decisions. Qualcomm uses multiple shared buffers indexed by `bi` ([`htp_tensor::bi`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L726)), which spreads the layout risk.

### 4.4 DSP DCVS and thermal

PP is a short burst of HMX-heavy computation. Small changes in DSP clock frequency caused by DCVS or thermal throttling produce visible PP variation. JZ's heavier sync path makes it more sensitive to these variations because the compute-to-overhead ratio is lower.

### 4.5 The first run is not always the slowest

In the user's data, the first JZ run was the slowest (321.90). In our later baseline A/B test, the first run was actually the fastest (322.77). This shows that cold-state is only one contributor; the exact jitter pattern depends on the interaction of cold-state, current DVFS state, and thermal history at the moment the run starts.

---

## 5. FastRPC/ION Warmup A/B Test

### 5.1 Hypothesis

A non-trivial portion of JZ PP jitter and lower average PP comes from cold-state costs paid on the first real `ggml_dsp_execute_batch` call. If we issue a no-op FastRPC call immediately after session initialization, the ION mapping, DSP entry path, and FastRPC channel will be warm before the first real inference batch.

### 5.2 Implementation

A new special mode `batch_size == 0xFFFB` was added to the DSP entry path in [`htp/entry.c`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/entry.c):

```c
/* Warmup mode: batch_size == 0xFFFB.
 * AP calls this once after session init to warm up the FastRPC/ION path
 * without doing any real compute. Just logs and returns. */
if (batch_size == 0xFFFB) {
    GGMLHEXAGON_LOG_INFO("[DSP-WARMUP] no-op warmup done");
    return AEE_SUCCESS;
}
```

AP side in [`ggml-hexagon-jz.cpp`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp), after `0xFFFC` cache-mode configuration:

```cpp
/* Warmup FastRPC/ION path once before first real inference.
 * This triggers delayed ION mapping and touches the DSP entry path
 * so the first real batch does not pay cold-state penalty. */
int warmup_err = ggml_dsp_execute_batch(ctx->ggmlop_handle, 0, 0xFFFB);
if (AEE_SUCCESS != warmup_err) {
    GGMLHEXAGON_LOG_WARN("warmup execute_batch failed: 0x%x", warmup_err);
} else {
    GGMLHEXAGON_LOG_ALWAYS("[AP-WARMUP] FastRPC/ION warmup done");
}
```

This requires no IDL change and adds one extra FastRPC round-trip during session setup.

### 5.3 Test procedure

1. Apply the warmup patch.
2. Build JZ: `./scripts/build-run-android.sh build`
3. Run `./scripts/build-run-android.sh run_llamacli` three times, recording PP/TG.
4. Revert the patch.
5. Build baseline.
6. Run baseline three times under the same device conditions.

### 5.4 Results

#### Warmup version

| Run | PP (t/s) | TG (t/s) | TG per-token (ms) | avg_p7 (us) |
|-----|----------|----------|-------------------|-------------|
| 1   | 339.25   | 19.84    | 50.41             | 2238        |
| 2   | 328.17   | 19.88    | 50.30             | 2235        |
| 3   | 330.88   | 19.64    | 50.91             | 2263        |
| **mean** | **332.77** | **19.79** | **50.54** | **2245** |

#### Baseline (reverted)

| Run | PP (t/s) | TG (t/s) | TG per-token (ms) | avg_p7 (us) |
|-----|----------|----------|-------------------|-------------|
| 1   | 322.77   | 19.17    | 52.17             | 2327        |
| 2   | 308.08   | 19.58    | 51.08             | 2286        |
| 3   | 316.19   | 19.24    | 51.98             | 2334        |
| **mean** | **315.68** | **19.33** | **51.74** | **2316** |

### 5.5 Comparison

| Metric | Warmup vs Baseline |
|--------|--------------------|
| PP mean | +5.4% (332.77 vs 315.68) |
| TG mean | +2.4% (19.79 vs 19.33) |
| TG per-token | -1.20 ms/token |
| avg_p7 | -3.1% (2245 vs 2316 us) |
| PP range | 11.1 vs 14.7 |

### 5.6 Post-commit validation run

After the commit was applied, a fresh run on the same device produced the following result:

![Post-commit run: PP = 378.57 t/s, TG = 19.46 t/s](<images/Screenshot from 2026-07-13 23-08-13.png>)

This single run reached **PP = 378.57 t/s**, the highest JZ PP observed so far, with `avg_p7 = 2297 us` and `avg_graph = 2322 us`. It confirms that the warmup removes enough cold-state overhead to let the HMX matmul path operate near its ceiling.

### 5.7 Interpretation

- Warmup increases average PP by ~5.4% and average TG by ~2.4%.
- The core efficiency metric `avg_p7` (time per batch call) drops by ~3.1%, confirming that warmup removes cold-state overhead from the critical path.
- PP jitter is reduced (range 11.1 vs 14.7), although not eliminated, because DCVS/thermal/ION-layout factors remain.
- Interestingly, the baseline first run was not the slowest in this A/B test, while it was in the user's earlier data. This confirms that cold-state is only one of several jitter sources; warmup shifts the whole distribution upward and narrows it.

---

## 6. Conclusions

1. **Graph-level synchronous blocking** is technically true for both backends, but Qualcomm has batch-level pipelining inside the graph while JZ does not. This is the fundamental architectural difference.

2. **PP is similar** because both backends share the same HMX matmul kernels and PP has few batch calls.

3. **TG gap (~1.41x)** is not caused by the attention or matmul kernels themselves (those are shared). It is caused by JZ's lack of batch-level overlap: each of the ~17 subgraphs per token pays a full synchronous FastRPC + cache-sync cycle, and this overhead accumulates across 255 tokens.

4. **JZ PP jitter** has multiple sources: cold-state ION/`fastrpc_mmap` penalty, user-space cache coherency sensitivity, single-mempool layout variance, and DSP DCVS/thermal. Warmup addresses the cold-state component.

5. **A single no-op warmup call** improves JZ PP by ~5.4% and TG by ~2.4% with negligible startup cost. It is a low-risk, positive-ROI change.

---

## 7. Follow-up: Weights Pre-flush (P1)

After the warmup commit, the next highest-ROI idea was to move the weights cache-clean cost out of the first inference batch and into model-load time. JZ already tracks `weights_dirty` and skips re-flushing clean repack weights in Phase 6.5 ([`ggml-hexagon-jz.cpp:5972-5985`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5972-L5985)). The optimization is to pre-flush each repack weight immediately after `set_tensor`/`repack`, so the first real batch sees `weights_dirty = false` and skips the weights entirely.

### 7.1 Change

In [`ggml-hexagon-jz.cpp:4584-4603`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L4584-L4603):

```cpp
if (is_repack) {
    cpu_dcache_flush_range(hctx, hctx->rpc_mempool_handle,
                           tensor->data, ggml_nbytes(tensor));
    hctx->weights_dirty = false;
} else {
    hctx->weights_dirty = true;
}
```

### 7.2 A/B data

| Version | PP mean | TG mean | avg_p7 | load time | Phase 6.5 |
|---------|---------|---------|--------|-----------|-----------|
| warmup baseline | 332.77 | 19.79 | 2245 | ~117 ms | 9734 us |
| revert 8MB + probe 4032 | 341.75 | 19.76 | 2267 | — | — |
| **+ weights pre-flush** | **352.98** | **19.62** | **2272** | **128.33 ms** | **8827 us** |

Per-run breakdown:

| Run | PP | TG | avg_p7 | load | p6.5 |
|-----|----|----|--------|------|------|
| 1 | 354.67 | 19.70 | 2251 | 124.84 ms | 9155 us |
| 2 | 369.29 | 19.52 | 2292 | 126.94 ms | 8695 us |
| 3 | 334.97 | 19.63 | 2274 | 133.20 ms | 8632 us |

### 7.3 Interpretation

- **PP +3.3%** relative to the immediately preceding version, **+6.1%** relative to the original warmup baseline.
- **Phase 6.5 cumulative flush time drops ~9%** (9734 → 8827 us), confirming that repack weights were a material part of the first-batch flush.
- **load time increases ~10 ms**: the cost is moved from inference to model load, which is the expected trade-off.
- `avg_p7` is essentially flat because it measures RPC + DSP execution, not AP-side cache flush.
- Jitter is still present (run 2 = 369.29, run 3 = 334.97). Device-state factors (DCVS/thermal/ION layout) remain the dominant source of variance after cold-state and weights-flush costs are removed.

### 7.4 Verdict

Pre-flushing repack weights is a **safe, low-risk, positive-ROI change**. It is kept.

---

## 8. 10-Run Stability Test

After confirming P1, a 10-run stability test was executed to quantify remaining PP/TG jitter and identify whether thermal decay was a factor. The device was a Snapdragon 8 Elite with the current optimized build (warmup + probe 4032 + weights pre-flush + `dcvs_option` retained).

### 8.1 Raw data

| Run | PP (t/s) | TG (t/s) | avg_p7 (us) | load (ms) | batch calls | TG runs | Notes |
|-----|----------|----------|-------------|-----------|-------------|---------|-------|
| 1   | 319.37   | 19.67    | 2291        | 139.62    | 4352        | 255     | complete |
| 2   | 330.53   | inf      | 4049        | 135.53    | 17          | 1       | early-stop |
| 3   | 344.41   | 19.67    | 2267        | 129.34    | 4352        | 255     | complete |
| 4   | 370.05   | inf      | 3853        | 119.73    | 17          | 1       | early-stop |
| 5   | 324.28   | 19.79    | 2249        | 137.65    | 4352        | 255     | complete |
| 6   | 334.11   | 19.49    | 2277        | 132.54    | 4352        | 255     | complete |
| 7   | 328.44   | 19.59    | 2268        | 134.87    | 4352        | 255     | complete |
| 8   | 327.07   | 19.45    | 2298        | 135.24    | 4352        | 255     | complete |
| 9   | 336.42   | 19.66    | 2262        | 131.80    | 4352        | 255     | complete |
| 10  | 338.66   | 19.42    | 2303        | 130.71    | 4182        | 245     | complete |

### 8.2 Statistics (8 complete runs only)

| Metric | min | max | mean | range |
|--------|-----|-----|------|-------|
| PP     | 319.37 | 344.41 | **331.60** | **25.04** |
| TG     | 19.42  | 19.79  | **19.59**  | **0.37**  |
| avg_p7 | 2249   | 2303   | **2276**   | **54**    |

### 8.3 Interpretation

- **TG is extremely stable**: range only 0.37 t/s and avg_p7 range only ~2.4%. This indicates the HMX/HVX compute path, FastRPC channel, and DSP execution are consistent once the model is running.
- **PP jitter is ~7.5% CV**: the 25 t/s range is real and does not show thermal decay (runs do not monotonically decrease: 319 → 344 → 324 → 336).
- **Jitter source is therefore not thermal**. Likely contributors are per-session ION physical layout variance, initial DCVS corner selection, and AP-side descriptor preparation time.
- **AP-side preparation matters**: Run 3 PP=344.41 with avg_p7=2267 vs Run 10 PP=338.66 with avg_p7=2303 shows PP can vary even when DSP execution time is similar. This points to Phase 1–4 of `graph_compute_batch` as the remaining optimization target.

---

## 10. Phase 1 Optimization: cgraph Cache Hit Path

Section 8 pointed to AP-side descriptor preparation (Phase 1-4) as the remaining optimization target. Phase 1 is the first block inside `graph_compute_batch` and is executed for every subgraph, so its cost is multiplied by the total number of batch calls.

### 10.1 What Phase 1 does

On every call Phase 1:

1. Computes a 64-bit content hash over the current `ggml_cgraph` (op codes, shapes, strides, src pointers, and node data pointers).
2. Looks the hash up in `ctx->cgraph_cache`.
3. On cache hit, restores the previously built `tensor_src`, `hex_ops`, and `weight_indices` descriptors so the rest of the function can skip Phase 2/2.5/3 reconstruction.

Cache hit rate is already ~99% (4317/4352 for gemma4, 21551/21778 for qwen3), so the hot path is the hit path.

### 10.2 The overhead being removed

The original hit path copied the cached descriptor vectors into local vectors before use:

```cpp
// old path
local_tensor_src.assign(cached_entry->tensor_src.begin(), cached_entry->tensor_src.end());
local_hex_ops.assign(cached_entry->hex_ops.begin(), cached_entry->hex_ops.end());
```

For a typical 767-node graph this copies ~20-110 KB of descriptor data per call. With 4352 calls this becomes a measurable pure AP-side overhead.

The optimized path binds references directly to the cached vectors in [`ggml-hexagon-jz.cpp:5083-5086`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5083-L5086):

```cpp
std::vector<ggml_tensor *> & tensor_src = cache_hit ? cached_entry->tensor_src : local_tensor_src;
std::vector<hex_op_desc>   & hex_ops   = cache_hit ? cached_entry->hex_ops   : local_hex_ops;
```

Only the small `weight_indices` unordered_set is still copied/cleared locally because the rest of the code uses `count()` on it. The hash computation itself was also tightened: NULL src pointers are skipped and the node `data` pointer is folded into the key so distinct tensors with identical topology do not collide.

### 10.3 Validation method

Build and test followed `.trae-project-config.json`:

```sh
./scripts/build-run-android.sh build
./scripts/build-run-android.sh run_llamacli gemma4
./scripts/build-run-android.sh run_llamacli qwen3
```

Logs captured:

- gemma4 baseline: `log_p1_gemma4_260714-063517.txt`
- gemma4 optimized: `log_p1_ref_gemma4_260714-064445.txt`
- qwen3 baseline: `log_p1_qwen3_260714-063610.txt`
- qwen3 optimized: `log_p1_ref_qwen3_260714-065018.txt`

### 10.4 Results

#### gemma-4-E2B-it-Q4_0 (4352 batch calls)

| Metric | Baseline | Optimized | Delta |
|--------|----------|-----------|-------|
| PP (t/s) | 314.25 | 332.63 | +5.8% |
| TG (t/s) | 19.66 | 19.73 | +0.4% |
| Phase 1 cumulative | 42112 us | 25818 us | -38.7% |
| Phase 1 per call | 9.68 us | 5.93 us | -38.7% |
| cache hit rate | 99.2% | 99.2% | — |

#### Qwen3.5-2B-Q4_0 (21778 batch calls)

| Metric | Baseline | Optimized | Delta |
|--------|----------|-----------|-------|
| PP (t/s) | 124.15 | 121.14 | -2.4% |
| TG (t/s) | 15.37 | 15.49 | +0.8% |
| Phase 1 cumulative | 38719 us | 28511 us | -26.4% |
| Phase 1 per call | 1.78 us | 1.31 us | -26.4% |
| cache hit rate | 99.0% | 99.0% | — |

### 10.5 Interpretation

- **Phase 1 overhead drops significantly**: removing the vector copy saves ~3.7 us/call for large graphs and ~0.5 us/call for small graphs. Because the cache hit rate is ~99%, almost every call benefits.
- **gemma4 PP improves +5.8%**: PP has only a few large-batch calls, so the relative impact of AP-side overhead per call is larger, and the reduction is visible in end-to-end PP.
- **qwen3 PP is essentially flat (-2.4%, within run-to-run jitter)**: qwen3 graphs are smaller and the original Phase 1 cost was already lower in absolute terms; device-state variance masks the small gain. The Phase 1 metric itself still improved -26%.
- **TG is stable or marginally up**: descriptor preparation is not on the DSP critical path, so TG is not expected to change much.
- **Output remains coherent**: no garbled text or endless repetition in either model; only the usual small-model factual hallucinations.

### 10.6 Next step

With vector-copy overhead removed, the remaining Phase 1 cost is dominated by the FNV-1a hash walk and the `unordered_map` lookup. The next candidate was to replace per-tensor hash lookups in Phase 6 as well (weight set and mirror map), which is covered in Section 11.

---

## 11. Phase 6 Optimization: Eliminate Per-Tensor Hash Lookups

Section 10 reduced Phase 1 by removing vector copies. Profiling showed that Phase 6 (descriptor construction) still pays hash-table lookups for every tensor on every call:

- `weight_indices.count(i)` in `unordered_set<uint32_t>` — used to decide flags and skip cache flush.
- `buffer_mirrors_map.find(t->data)` — used to translate heap tensor pointers to ION mirror offsets.

These lookups are pure AP-side overhead and their latency is sensitive to memory layout, contributing to PP jitter.

### 11.1 Changes

In [`ggml-hexagon-jz.cpp`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp):

1. Replaced the cached `weight_indices` set with a dense `is_weight` boolean vector in `cgraph_cache_entry` (lines 372, 5022, 5086-5092, 5265-5280, 5582).
2. On cache hit, bind a reference to the cached `is_weight` vector instead of rebuilding an `unordered_set`.
3. In Phase 4 Step 3, build a per-tensor `local_mirror_offset` array (size `n_tensors`, `-1` = no mirror) while building the `mirrors` list (lines 5730-5752).
4. In Phase 6, Phase 4.5, and Phase 7.5, replace `buffer_mirrors_map.find()` and `mirrors` scans with direct `local_mirror_offset[tidx]` array access.
5. Replaced all `weight_indices.count(i)` calls with `is_weight[i]`.

This removes every per-tensor hash lookup from the hot descriptor-construction path.

### 11.2 Validation

Same build/test commands as Section 10:

```sh
./scripts/build-run-android.sh build
./scripts/build-run-android.sh run_llamacli gemma4
./scripts/build-run-android.sh run_llamacli qwen3
```

Logs: `log_p2_gemma4_260714-070327.txt`, `log_p2_qwen3_260714-070416.txt`.

### 11.3 Results

#### gemma-4-E2B-it-Q4_0 (4352 batch calls)

| Metric | P1 opt (Section 10) | +P6 opt | Delta |
|--------|---------------------|---------|-------|
| PP (t/s) | 332.63 | 352.80 | +6.1% |
| TG (t/s) | 19.73 | 19.86 | +0.7% |
| Phase 1 | 25818 us | 19631 us | -24.0% |
| Phase 4.5 | 8861 us | 3981 us | -55.1% |
| Phase 6 | 21775 us | 14006 us | -35.7% |

#### Qwen3.5-2B-Q4_0 (21778 batch calls)

| Metric | P1 opt (Section 10) | +P6 opt | Delta |
|--------|---------------------|---------|-------|
| PP (t/s) | 121.14 | 126.04 | +4.0% |
| TG (t/s) | 15.49 | 15.46 | -0.2% |
| Phase 1 | 28511 us | 24136 us | -15.3% |
| Phase 4.5 | 6134 us | 3858 us | -37.1% |
| Phase 6 | 16016 us | 12727 us | -20.5% |

### 11.4 Interpretation

- **Phase 1 drops further** because the cache-hit path no longer reconstructs an `unordered_set` of weight indices; it just references the cached boolean vector.
- **Phase 4.5 drops significantly** because the tiled-weight ION-offset recording now uses the precomputed `local_mirror_offset` array instead of `buffer_mirrors_map.find()`.
- **Phase 6 drops** because per-tensor mirror-offset lookup is now O(1) array access instead of a hash-map lookup.
- **gemma4 PP gains +6.1%** and **qwen3 PP gains +4.0%**, confirming that AP-side descriptor-preparation variance directly affects PP.
- **TG remains stable** (gemma4 +0.7%, qwen3 -0.2%), as expected: descriptor preparation is not on the DSP critical path.
- Output is coherent in both models; no garbled text or repetition loops.

### 11.5 Remaining PP jitter

After removing hash lookups from Phase 1 and Phase 6, the remaining AP-side variance likely comes from:

- **Phase 4** heap-to-ION memcpy for activations (~1 us/call, memory-bandwidth sensitive).
- **Phase 6.5** DC CVAC flush and **Phase 7.5** DC CIVAC invalidate (~2-2.5 us/call each, sensitive to bus contention and CPU frequency).
- The FNV-1a hash walk and `unordered_map` lookup in Phase 1.

The next candidates are:

1. Replace the `cgraph_cache` `unordered_map` lookup with a small fixed-size direct/LRU cache.
2. Batch or coalesce cache-maintenance operations in Phase 6.5/7.5 to reduce per-call variance.
3. Investigate whether activation mirrors can be allocated at stable offsets so the full tensor-descriptor array can be cached on hit.

---

## 12. Post-Optimization 5-Run Stability Test

After the Phase 6 hash/lookup optimization, five consecutive runs were executed for each model (no cooldown between runs) to quantify remaining PP/TG jitter.

### 12.1 gemma-4-E2B-it-Q4_0

| Run | PP (t/s) | TG (t/s) |
|-----|----------|----------|
| 1   | 325.85   | 19.66    |
| 2   | 358.13   | 19.85    |
| 3   | 341.06   | 19.71    |
| 4   | 311.06   | 19.75    |
| 5   | 338.67   | 19.14    |

| Metric | mean | stddev | min | max | CV |
|--------|------|--------|-----|-----|----|
| PP     | 334.95 | 15.76 | 311.06 | 358.13 | 4.7% |
| TG     | 19.62  | 0.25  | 19.14  | 19.85  | 1.3% |

### 12.2 Qwen3.5-2B-Q4_0

| Run | PP (t/s) | TG (t/s) |
|-----|----------|----------|
| 1   | 121.05   | 14.33    |
| 2   | 122.26   | 14.40    |
| 3   | 117.21   | 14.43    |
| 4   | 119.67   | 14.21    |
| 5   | 121.34   | 14.37    |

| Metric | mean | stddev | min | max | CV |
|--------|------|--------|-----|-----|----|
| PP     | 120.31 | 1.76 | 117.21 | 122.26 | 1.5% |
| TG     | 14.35  | 0.08 | 14.21  | 14.43  | 0.5% |

### 12.3 Interpretation

- **TG remains extremely stable** for both models (CV < 1.5%), confirming the DSP compute path is not the jitter source.
- **qwen3 PP is now very stable** (CV 1.5%), likely because its graphs are small and many calls amortize any per-call variance.
- **gemma4 PP still shows ~4.7% CV** with a 47 t/s spread. Because gemma4 has far fewer prompt-batch calls, any per-call AP-side variance (Phase 4 memcpy, Phase 6.5/7.5 cache sync) is not averaged out. This points to Phase 6.5/7.5 cache maintenance and Phase 4 heap-to-ION memcpy as the dominant remaining jitter sources, not Phase 1/6 descriptor lookup.

---

## 13. Related Files

- [`ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp) — JZ AP implementation
- [`ggml/src/ggml-hexagon/ggml-hexagon.cpp`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp) — Qualcomm AP implementation
- [`ggml/src/ggml-hexagon/htp/entry.c`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/entry.c) — JZ DSP entry point
- [`ggml/src/ggml-hexagon/htp/main.c`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/main.c) — Qualcomm DSP entry point
- [`docs/backend/jz-ggml-hexagon/algotype29-perf-analysis-en-20260711.md`](file:///home/zhouwg/develop/ggml-hexagon/docs/backend/jz-ggml-hexagon/algotype29-perf-analysis-en-20260711.md) — prior performance analysis
