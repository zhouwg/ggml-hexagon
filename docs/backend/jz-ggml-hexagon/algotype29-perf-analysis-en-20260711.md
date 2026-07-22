# JZ ggml-hexagon: Structure and Optimization Analysis (2026-07-22)

## Author

- Primary author: GLM-5.2, authored based on a full review of the JZ (`ggml-hexagon-jz.cpp`, `htp/entry.c`) and Qualcomm (`ggml-hexagon.cpp`, `htp/main.c`) codebases after syncing with upstream master.
- Fact-check by MiniMax-M3 (image insertion, review, fact-check, GitHub doc-rendering fix).
- TG gap root-cause correction by Kimi-K2.7-Code.
- TG optimization and doc revision by Kimi-K3 (bf16 support, lm-head DSP offload, dsp_cache_mode bit0 re-enable, DSP debug log removal, v75 thread clamp, doc updates).
- Comprehensive rewrite by GLM-5.2 (removed historical correction chains and outdated data, consolidated to current state for first-time readers).
- Project author: Jeff Zhou (zhouwg) (2024-03 to present, see timeline at https://github.com/zhouwg/ggml-hexagon/discussions/18).
- Project co-authors: GLM-5.2, MiniMax-M3 (AP&DSP optimization, build system unification, CI improvements, multi-document drafting), DeepSeek-V4-Pro (AP-side cache coherency optimization with `ion_sync_mode` knob family, CI improvements, code refine in removed algotype !=29 kernels), Kimi-K2.7-Code (fix issues in dsp_cache_mode=5/7, PP optimization, two important docs), Kimi-K3 (TG optimization, DSP debug log removal, v75 thread clamp, doc updates).

## 1. Background

Both the JZ and Qualcomm versions of the ggml-hexagon backend route through Qualcomm's `execute_op` path (the `execute_op` implementation lives in `htp/` and is shared by both versions). The DSP entry points differ: JZ uses `htp/entry.c`, while Qualcomm uses `htp/main.c`. The AP-side implementations differ significantly, leading to performance differences. (The `mulmat_algotype` config knob that previously selected between the self-built and Qualcomm paths was removed in the dual path cleanup; the "algotype=29" label survives only as a historical comment in `ggml-hexagon.cfg` and in the document filename.)

JZ ggml-hexagon is built on two fundamental architectural choices that originated from the upstream PR [#12326](https://github.com/ggml-org/llama.cpp/pull/12326) (March 2025):

1. **Native FastRPC** - Direct synchronous FastRPC calls via Hexagon SDK, not dspqueue's async wrapper. PR-12326 already implemented this as a FastRPC-based per-op path (alongside a QNN SDK path); JZ ggml-hexagon evolved it into the ion-based op-batch design
2. **ION shared memory pool** - Single shared memory pool with offset addressing, inspired by the "shared buffer or memory pool" idea proposed in PR-12326

Based on these two choices, JZ ggml-hexagon sidesteps both the `dspqueue` async wrapper and per-buffer ION allocations used by the Qualcomm backend: synchronous native FastRPC needs no async scheduling layer, and a single shared ION pool with offset addressing needs no per-buffer handle table. The result is a more architecturally concise AP/DSP integration with fewer abstractions and moving parts than the Qualcomm backend.

These are deliberate design choices, not limitations. The theoretical basis is that LLM inference is inherently serial (autoregressive TG + serially dependent subgraphs), which limits the benefit of async pipelining.

## 2. Current Benchmark (2026-07-22)

### 2.1 Test Conditions

- **Model file**: `/sdcard/gemma-4-E2B-it-Q4_0.gguf` (3.0 GB, 35 layers; 21 with own KV, 14 share with earlier layers)
- **Device**: Snapdragon 8 Elite (v79, OnePlus 13)
- **CLI params**: `-ngl 99 -t 6 -n 256 --ctx-size 8192 --ubatch-size 64 --poll 1000 --no-warmup --no-mmap -fa on --jinja -st`
- **Prompt**: `"Hello, good morning, you are a powerful domain expert and know many things, now pls help to introduce the movie Once Upon a Time in America briefly, pls pay attention short then 1000 words\n"` (58 tokens with jinja system/user/model tags)
- **JZ config**: `dsp_cache_mode=5`, `ion_sync_mode=1`, `enable_graph_optimize=1`, `thread_counts=6`, build v0.99.3.9-dev
- **Offloaded MUL_MAT types**: F32, F16, BF16, Q4_0, Q8_0, Q4_1, IQ4_NL, MXFP4

### 2.2 PP and TG Comparison (5 runs each, same device, same model, same day)

**Table 1: JZ vs Qualcomm PP/TG (5 runs, 2026-07-22)**

| Run | JZ PP (tok/s) | JZ TG (tok/s) | QCOM PP (tok/s) | QCOM TG (tok/s) |
| --- | ------------- | ------------- | --------------- | --------------- |
| 1   | 684.15        | 27.22         | 459.35          | 25.00           |
| 2   | 691.23        | 26.99         | 444.47          | 24.96           |
| 3   | 682.22        | 26.67         | 443.36          | 24.96           |
| 4   | 692.41        | 27.32         | 392.14          | 24.84           |
| 5   | 682.30        | 26.37         | 436.41          | 24.80           |
| **mean** | **686.46** | **26.91**     | **435.14**      | **24.91**       |

JZ exceeds Qualcomm on both PP and TG:
- **PP**: JZ 686.46 vs QCOM 435.14 (JZ 1.58x faster)
- **TG**: JZ 26.91 vs QCOM 24.91 (JZ 1.08x faster)

Both backends run the same HMX kernels from `ggml/src/ggml-hexagon/htp`; the difference is architectural. The single ION mempool enables a session-resident repacked lm-head that the per-buffer ION design cannot express economically, and the user-space cache management enables role-aware invalidation policies that the closed driver's uniform per-batch flush cannot express. See [ion-mempool-vs-perbuffer-analysis-20260713.md](ion-mempool-vs-perbuffer-analysis-20260713.md) for the architectural analysis.

### 2.3 JZ-side RPC stats (run 1)

```
batch_calls=256, avg_p7=35989 us, avg_graph=36533 us
cgraph cache: hits=253 misses=3 (hit_rate=98.8%) entries=3
per-call overhead: n=256 min=182 max=3311 avg=543 us (graph_dur - p7)
p7 3-way: rpc_setup=51 dsp_exec=9213373 civac=2523 us
rpc overhead (warmup): n=6 min=78 max=132 avg=89 us (pure FastRPC/ION transport)
```

## Relevant Files

**Table 2: Relevant files**

| File                                          | Description                                   |
| --------------------------------------------- | --------------------------------------------- |
| `ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp`   | JZ version AP code                            |
| `ggml/src/ggml-hexagon/ggml-hexagon.cpp`      | Qualcomm version AP code (upstream)           |
| `ggml/src/ggml-hexagon/CMakeLists.txt`        | Unified build (QCOM base + `GGML_HEXAGON_JZ` option) |
| `ggml/src/ggml-hexagon/htp/Makefile`          | JZ DSP skel build (entry.c + shared kernels)  |
| `ggml/src/ggml-hexagon/htp/CMakeLists.txt`    | QCOM DSP skel build (main.c + shared kernels) |
| `ggml/src/ggml-hexagon/htp/entry.c`           | JZ version DSP entry point                    |
| `ggml/src/ggml-hexagon/htp/dsp-ctx.h`         | JZ DSP session context + descriptors          |
| `ggml/src/ggml-hexagon/htp/main.c`            | Qualcomm version DSP entry point              |
| `ggml/src/ggml-hexagon/htp/*.c`               | Shared DSP kernels (both backends)            |

## 3. FastRPC Call Pattern

### 3.1 JZ Version

- **Call method**: Single synchronous `ggmlop_dsp_execute_batch`
- **Parameters**: 2 scalars (`batch_offset`, `total_desc_size`)
- **Data transfer**: Single ION mempool + offset addressing
- **Pipelining**: None (AP blocks waiting for DSP completion)

### 3.2 Qualcomm Version

- **Call method**: dspqueue message queue
- **Parameters**: `dspqueue_write` + `dspqueue_buffer` (containing `fd + offset + size`)
- **Data transfer**: fd + offset two-level addressing, supporting multiple independent shared buffers
- **Pipelining**: Up to 16 batches in-flight (`opt_opqueue=16`), AP/DSP parallel execution

### 3.3 Performance Impact

JZ's synchronous FastRPC architecture is a deliberate design choice. The pure FastRPC/ION transport overhead is ~89 us/call (warmup probe), which is negligible against the ~36 ms/token DSP execution time. With lm-head offloaded to DSP and `dsp_cache_mode=5`, JZ TG exceeds QCOM TG without any batch-level pipelining - the synchronous architecture is not the TG bottleneck on this workload.

## 4. Op Fusion

### 4.1 JZ Version (Phase 2.5)

**Table 3: JZ op fusion types (Phase 2.5)**

| Fusion Type                                  | Supported | Notes                                          |
| -------------------------------------------- | --------- | ---------------------------------------------- |
| RMS_NORM + MUL -> HTP_OP_RMS_NORM_MUL        | Yes       | Linear scan; fires every graph (1-3 per graph) |
| MUL_MAT + ADD -> HTP_OP_MUL_MAT_ADD          | Yes       | Bias add inside matmul kernel; VTCM budget checked |
| MUL_MAT QKV merge -> HTP_OP_MUL_MAT_QKV      | Yes       | 3 MUL_MAT (Q,K,V) merged into 1                |
| MUL_MAT FFN merge -> HTP_OP_MUL_MAT_FFN      | Yes       | gate + up merged into 1                        |
| Graph reorder                                | Yes       | Forward 16-group window; runtime-configurable  |

### 4.2 VTCM budget check for MUL_MAT_ADD fusion

The MUL_MAT + ADD fusion checks the VTCM budget before firing, mirroring Qualcomm's guard ([ggml-hexagon.cpp:3595](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L3595)):

```cpp
const size_t vtcm_budget = (size_t)ctx->socinfo.vtcm_size_in_mb * 1024 * 1024;
if ((size_t)kparams->vtcm_size > vtcm_budget) {
    return false;  // skip fusion, let MUL_MAT and ADD run separately
}
```

### 4.3 Qualcomm Version (`try_fuse_node`)

**Table 4: Qualcomm op fusion types (try_fuse_node)**

| Fusion Type                                  | Supported | Notes                                          |
| -------------------------------------------- | --------- | ---------------------------------------------- |
| RMS_NORM + MUL -> HTP_OP_RMS_NORM_MUL        | Yes       | Uses `ggml_can_fuse`                           |
| MUL_MAT + ADD -> HTP_OP_MUL_MAT_ADD          | Yes       | Uses `ggml_can_fuse`                           |
| MUL_MAT QKV merge -> HTP_OP_MUL_MAT_QKV      | Yes       | 3 mul_mat merged into 1, reordered to KVQ      |
| MUL_MAT FFN merge -> HTP_OP_MUL_MAT_FFN      | Yes       | gate + up merged into 1                        |
| Graph reorder                                | Yes       | Stacks MUL_MATs with same src1 for VTCM reuse  |

Fusion scope is at parity (all 5 fusion types). Graph reorder is implemented in JZ and runtime-configurable; for gemma4 PP=58 tokens on 8 Elite v79 it is a no-op (within noise), but kept on by default for future-proofing.

## 5. Graph Cache

The cache uses a **content hash** (FNV-1a over each node's `{op, ne[4], nb[4], src[0..2] ptr, data ptr}`) instead of the dead `cgraph->uid` key:

```cpp
const uint64_t content_hash = compute_content_hash();
auto it = ctx->cgraph_cache.find(content_hash);
if (it != ctx->cgraph_cache.end() &&
    it->second.n_nodes == cgraph->n_nodes &&
    it->second.hex_ops.size() > 0) {
    cache_hit = true;
    ctx->cgraph_cache_hits++;
}
```

On a hit, the cache skips Phase 1 (tensor dedup), Phase 2 (op descriptor build), and Phase 2.5 (op fusion) entirely. Current hit rate is 98.8% (253 hits / 256 calls). The 3 misses correspond to the first fill of 3 unique graph structures. After warmup, every subsequent token hits cache.

## 6. mm_params_cache (JZ only, enabled)

JZ caches precomputed `htp_mm_kernel_params` by a composite key (src0 tensor pointer XOR weight data pointer XOR ne11):

```cpp
const uintptr_t cache_key = (uintptr_t) src0 ^ (uintptr_t) src0->data ^ ((uintptr_t) ne11 << 32);
auto it = ctx->mm_params_cache.find(cache_key);
if (it != ctx->mm_params_cache.end()) {
    *kparams = it->second;
    return;
}
```

This skips the multi-hundred-microsecond thread/chunk search in `htp_mm_hvx_vtcm_layout_build` / `htp_mm_hmx_vtcm_layout_build` for repeated MUL_MAT calls with the same weight tensor. For TG (where ne11=1 for every token), the cache hits after the first token. The cached kparams are valid for the session lifetime because weights are static (never modified after model load).

Qualcomm's `ggml_hexagon_precompute_matmul_params` performs the same VTCM layout computation but does not cache the result across calls.

## 7. Weight Repack

JZ implements a **repack buffer type** with `is_host=false`:

```cpp
static bool ggml_backend_hexagon_repack_buffer_is_host(ggml_backend_buffer_type_t buft) {
    return false;  // forces GGML core to call set_tensor
}
```

When the model loader encounters quantized weights (Q4_0, Q4_1, Q8_0, IQ4_NL, MXFP4), the `supports_op` gate in MUL_MAT ensures they are allocated in the repack buffer type. Because `is_host=false`, GGML core routes data through `set_tensor`, which performs the in-place tile repack at model load time (one-time cost).

### lm-head DSP offload

The lm-head (262144x1536, Q4_K) is offloaded to DSP. The `ne[1] > 32768` rejection for large quantized weights was removed, and a new `repack_q4k_as_q4_0_tiled_to_buf` converter ([ggml-hexagon-jz.cpp:4162](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L4162)) stores the lm-head as Q4_0 tiled layout in the single ION pool (~214 MB, session-resident, streamed from DDR once per token). This eliminates the ~30 ms/token CPU lm-head matvec that was the single largest TG cost. Qualcomm's per-buffer ION design makes a 214 MB per-buffer map/unmap per session prohibitive, so their lm-head stays on CPU.

## 8. Cache Coherency Management

### 8.1 JZ Version

- **AP side**: Configurable via `ion_sync_mode`
  - `0` = both (DC CVAC + ion_sync, default)
  - `1` = ion_sync only (DMA_BUF_IOCTL_SYNC, driver-level) - **optimal**
  - `2` = DC CVAC only (manual cache line management)
- **DSP side**: Configurable via `dsp_cache_mode` bitmask (default = 5)
  - bit 0: first-touch weight invalidation (exact sorted pointer array `g_weight_inval_ptrs`, `WEIGHT_INVAL_MAX_PTRS=4096` in entry.c; repack weights written once at model load, single first-touch dcinva covers the whole session)
  - bit 1: skip dcinva for prior dst (off - deferred-flush pattern is unsafe to combine with bit2 for anything above a single cacheline)
  - bit 2: bulk dst flush at batch end (collect/sort/merge dst ranges)
  - bit 3: selective bulk flush (skip batch-end flush for intermediates still consumed by later ops in the same batch; `g_tensor_last_use_op` in entry.c)

### 8.2 dsp_cache_mode=5 default

Mode 5 (bit0 + bit2) is the default. The first-touch weight invalidation tracker uses an exact sorted pointer array instead of a hash bitmap (whose address collisions caused garble in earlier iterations). A two-pass defense (DSP-side unmark on dst write + AP-side ever-dst set) closes the cross-graph stale-read window. This is what made the session-resident repacked lm-head practical: per-token weight traffic is ~1.9 GB, and re-invalidating it every token would cost ~9.2 ms/token of DSP-side dcinva sweeping; with first-touch, that cost is paid once at session start.

### 8.3 ion_sync_mode=1 is optimal

Mode 1 uses the kernel's `DMA_BUF_IOCTL_SYNC` which is faster than userspace DC CVAC/CIVAC for large ranges. Mode 0 (double cache maintenance) drops PP significantly; mode 2 (manual only) is similar to mode 0.

### 8.4 Qualcomm Version

dspqueue driver automatic management via `DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT`. The driver applies uniform per-batch flush/invalidate flags to every buffer - there is no per-role differentiation. JZ's user-space management distinguishes weight tensors (flags=2, written once at load) from activations, so bit 0 can eliminate their per-token re-invalidation. This policy flexibility is an advantage the closed driver cannot replicate without per-role buffer semantics.

## 9. Session Consistency Gate

JZ mirrors Qualcomm's `ggml_hexagon_supported_buffer` check to prevent the scheduler from mixing tensors across different Hexagon sessions or non-Hexagon buffers:

```cpp
static bool ggmlhexagon_tensor_buffer_is_owned_by(ggml_backend_dev_t dev, const struct ggml_tensor * t) {
    if (!t || !t->buffer) return true;  // neutral
    // Accept if buffer is hexagon (main or repack) on this device
    // Reject if hexagon on different device or non-hexagon
}
```

## 10. VTCM Session-Lifetime

VTCM is acquired once per session, not per batch. `ggml_dsp_open` does one `HAP_compute_res_acquire_cached` and `ggml_dsp_close` does one `HAP_compute_res_release_cached`. VTCM is held continuously for the session. This matches the Qualcomm HTP pattern (`vtcm_acquire` / `vtcm_release` only fire when transitioning between "active processing" and "forced release").

```c
// ggml_dsp_open (after HAP_compute_res_acquire succeeds)
dsp_vtcm_acquire();   // once per session, sets vtcm_valid=1

// ggml_dsp_close (before HAP_compute_res_release)
dsp_vtcm_release();   // once per session, sets vtcm_valid=0

// execute_batch: no per-batch acquire/release calls
```

Trade-off: lose the ability to respond to a forced-release callback from another session. For single-session use (the current deployment) this is a non-issue.

## 11. Dual Path Removal

JZ routes **all** ops through the shared `htp/` execute_op path, exactly like Qualcomm. The entire algotype=32 path (JZ's own self-built kernel dispatch in `kernels/mulmat.c`, `kernels/flash_attn.c`, etc.) and the `ggml-dsp` port (`kernels/ggml-dsp.c` 9946 lines, `kernels/ggml-dsp.h` 2256 lines) are deleted - 24298 lines removed. JZ's four files are now merged into `htp/` alongside Qualcomm's kernels:

- `entry.c` - FastRPC entry point, cache management, `dsptensor` <-> `htp_tensor` bridge, `execute_op` dispatch
- `dsp-ctx.h` - `struct dsp_context`, `dsptensor`, `hex_tensor_desc`, `hex_op_desc`, `hex_batch_hdr`
- `ggml_dsp.idl` - FastRPC interface
- `Makefile` - builds `libggmldsp-skel.so` from `entry.c` + all `htp/*.c` sources

The `mulmat_algotype` config knob is removed entirely. The "algotype=29" label survives only as a historical comment in `ggml-hexagon.cfg` and in this document filename.

`dsptensor` is retained as JZ's AP-side tensor descriptor format (single ION offset addressing), but it is now only a thin wrapper. The `dsptensor_to_htp_tensor` bridge in [entry.c](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/entry.c) converts it to the shared `htp_tensor` (with `bi=0`) before calling `execute_op`.

## 12. Tensor Descriptor Data Structure

### 12.1 JZ Version

JZ uses three descriptor types at different stages:

**AP-side**: `hex_tensor_desc` (single ION offset addressing).

```c
typedef struct hex_tensor_desc {
    int32_t  type;
    int32_t  ne[4];
    int32_t  nb[4];
    int32_t  op_params[16];
    uint32_t flags;          // 0=ION, 1=mirrored, 2=weight(skip flush)
    uint32_t data_offset;
    uint32_t data_len;
} hex_tensor_desc;
```

**DSP-side**: `dsptensor` (defined in [dsp-ctx.h](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/dsp-ctx.h))

```c
struct dsptensor {
   int32_t type;
   int32_t ne[4];
   int32_t nb[4];
   int32_t op;            // op code embedded in tensor descriptor
   int32_t op_params[16]; // op-specific params embedded in tensor descriptor
   int32_t flags;
   void *  data;          // direct pointer (DSP address space is 32-bit)
   int     data_len;
};
```

### 12.2 Qualcomm Version `htp_tensor`

```c
struct htp_tensor {
    uint32_t data;
    uint32_t size;
    uint32_t flags;
    uint16_t type;
    uint16_t bi;         // buffer index
    uint32_t ne[4];
    uint32_t nb[4];
};
```

### 12.3 Difference

**Table 5: Tensor descriptor comparison**

| Aspect               | `dsptensor` (JZ)                  | `htp_tensor` (Qualcomm)                       |
| -------------------- | --------------------------------- | --------------------------------------------- |
| Op metadata          | Embedded (`op` + `op_params[16]`) | Separated into `htp_op_desc`                  |
| Data addressing      | Direct `void *` pointer           | `bi` (buffer index) + `uint32_t` offset       |
| Multi-buffer support | No (single ION pool)              | Yes (via `bi` indexing into `htp_buf_desc[]`) |
| Type width           | `int32_t`                         | `uint16_t`/`uint32_t` (more compact)          |

The `bi` field is the key differentiator for multi-buffer support. JZ always passes `bi=0` (single ION pool); Qualcomm uses it to index into `htp_buf_desc[]`.

## 13. Upstream Merge Adaptations

### 13.1 Unary precompute port

Upstream commit `fb30ba9a6` introduced a new `op_unary` pipeline that requires host-precomputed `htp_unary_kernel_params` (n_threads, col_tile, vtcm_size, etc.). JZ ported `ggml_hexagon_precompute_unary_params` from `ggml-hexagon.cpp`, adapted to use `ctx->n_threads` and `ctx->socinfo.vtcm_size`. `HTP_OP_TRI` is routed to `op_unary()` in entry.c to match upstream `htp/main.c`.

### 13.2 VTCM layout API

Upstream commit `81ff7abe5` brought in Qualcomm's new VTCM layout API:
- `htp_mm_hvx_get_vtcm_sizes` -> `htp_mm_hvx_vtcm_layout_build` + `struct htp_mm_hvx_vtcm_layout`
- `htp_mm_hvx_id_get_vtcm_sizes` -> `htp_mm_hmx_vtcm_layout_build` + `struct htp_mm_hmx_vtcm_layout`
- `broadcast_rk2/rk3/rv2/rv3` fields moved from `u.hvx` union member to `htp_fa_kernel_params` struct top level

JZ adapted via adapter functions in `ggml-hexagon-jz.cpp` and `entry.c` that translate old API calls to the new layout-build API.

### 13.3 Build system unification

The unified `CMakeLists.txt` is based on QCOM's version (minimal diff to upstream) with a single addition:

```cmake
option(GGML_HEXAGON_JZ "Use JZ's AP implementation" OFF)
```

- **`GGML_HEXAGON_JZ=OFF` (default)**: exactly QCOM's upstream behavior. Uses `ggml-hexagon.cpp`, builds DSP skels via `ExternalProject_Add` for v73/v75/v79/v81, links `htp_iface` stub.
- **`GGML_HEXAGON_JZ=ON`**: uses `ggml-hexagon-jz.cpp`, builds a single DSP skel via `make -C htp/` (JZ's Makefile), links `cdsprpc`, sets `HEXAGON_DEFAULT_LIB_SEARCH_PATH`, copies `ggml-hexagon.cfg`.

## 14. Compiler Optimization

### 14.1 AP-side (PARITY)

Both backends compile AP-side code with the same ARMv8.7-A + dotprod + fp16 + i8mm flags:

```cmake
set(OPT_FLAG " -O3 -march=armv8.7-a+dotprod+fp16+i8mm -mcpu=cortex-x1 -mtune=cortex-x1 -ffp-model=fast -fno-finite-math-only")
```

### 14.2 DSP-side (JZ-only)

JZ's [htp/Makefile](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/Makefile) uses `-O3 -ffast-math -fno-vectorize` (no LTO) with `-DNDEBUG` for the DSP skel. Qualcomm's `htp/cmake-toolchain.cmake` uses `-O2 -flto -fvectorize`. A 3-row sweep confirmed LTO is a net regression for JZ's codebase (-9% to -14% PP); `-O3 no-LTO` is empirically the best choice.

### 14.3 flash-attn-ops.c -O2 workaround

`flash-attn-ops.c` is forced to `-O2` because at `-O3` Hexagon LLVM 19.0.07 emits a `PromoteFloatResult` fatal error on `f16 = freeze`. A 10-flag sweep confirmed no combination of `-fno-X` / `-mllvm -disable-X` flags can break the workaround. The bug is in the Hexagon backend (`HexagonDAGToDAGISel` -> `PromoteFloatResult`), not in any front-end pass. Needs an LLVM backend patch.

## 15. Profiler Infrastructure

### 15.1 AP-side profiler

Tracks cumulative time for phases p1, p2, p2.5, p3, p4, p4.5, p5, p6, p6.5, p7, p7.5, p8 per `graph_compute_batch` call. Breaks down Phase 7 into `rpc_setup` + `dsp_exec` + `civac`. Computes min/p50/p95/max histograms for the last 1024 calls. Reports cgraph cache hit/miss counts.

### 15.2 longtail profiler

A `LOG_ALWAYS` longtail probe inside the FastRPC dispatch path logs the op composition of any batch call whose `dsp_exec` exceeds 5 ms. Throttled to one log per 100 ms wall-clock. Probe code is preserved in source inside `#if 0 ... #endif` for future re-activation; runtime cost is zero.

### 15.3 mul_mat coverage tracer

Per-batch counters in `hexagon_op_exec_stats_t`: `n_mul_mat_total`, `n_hmx_used`, `n_fused_qkv`, `n_fused_ffn`, `n_fused_mul_mat_add`.

### 15.4 DSP-side profiler (gated off by default)

`entry.c` includes a per-op timing profiler that records min/max/avg execution time per op type, dumped via `dump_op_prof`. Wrapped in `#if HEX_OP_PROF ... #endif` with `HEX_OP_PROF` default to 0 (off). Restore by passing `-DHEX_OP_PROF=1` to `make`.

## 16. v75/8Gen3 Thread Clamp

On Snapdragon 8 Gen 3 (v75), the code deadlocks at the default `thread_counts=6` due to cDSP hardware-thread oversubscription. An op needs `thread_counts+1` co-resident QuRT threads (FastRPC main + N-1 work-queue workers + 1 hmx_queue thread); v75 has 6 hardware threads (v79 has 8), so N=6 requires 7 > 6 and the unscheduled worker never decrements the task barrier.

`ggml_dsp_setclocks` clamps to `max_hw_threads - 2` and reports the effective value through a `rout int32 real_thread_counts` IDL out-param; AP mirrors it into `ctx->n_threads` so the precomputed `kparams->n_threads` matches the DSP work-queue. On v75 this clamps to 4; on v79 it stays at 6.

## 17. Summary

**Table 6: Performance summary (2026-07-22, 5-run mean)**

| Backend | PP (tok/s) | TG (tok/s) |
| ------- | ---------- | ---------- |
| JZ      | 686.46     | 26.91      |
| QCOM    | 435.14     | 24.91      |
| **JZ advantage** | **1.58x** | **1.08x** |

JZ exceeds Qualcomm on both PP and TG. The three enabling changes:

1. **lm-head offloaded to DSP** - Q4_K stored as Q4_0 tiled repack (~214 MB, session-resident in the single ION pool). Eliminated the ~30 ms/token CPU lm-head matvec. The single ION pool turns this into a one-time repack; Qualcomm's per-buffer ION design makes a 214 MB per-buffer map/unmap per session prohibitive.
2. **dsp_cache_mode=5** (bit0 + bit2) - First-touch weight invalidation via exact sorted pointer array + two-pass defense. Eliminates ~9.2 ms/token of redundant weight re-invalidation. bit3 (selective bulk flush) also added.
3. **DSP-side debug logging removed** - Skel built with `-DNDEBUG`; FARF debug paths compiled out.

Also: BF16 added to offloaded MUL_MAT types (stored as F16 in repack buffer, reusing F16 DSP kernels).

**Table 7: Optimization status**

| Item                    | Status                                          |
| ----------------------- | ----------------------------------------------- |
| Weight repack timing    | At `set_tensor` (one-time at model load)        |
| lm-head on DSP          | Q4_K -> Q4_0 tiled repack, ~214 MB session-resident |
| Op fusion scope         | PARITY - all 5 fusion types; VTCM guard for MUL_MAT_ADD |
| Graph cache             | Content-hash based (98.8% hit rate)             |
| Graph reorder           | Implemented, no measurable PP benefit for gemma4 PP=58 |
| Dual path               | REMOVED - single shared `htp/` kernel path      |
| VTCM lifetime           | Session-lifetime (matches Qualcomm pattern)     |
| dsp_cache_mode          | Mode 5 (bit0 + bit2); bit3 selective bulk flush |
| ion_sync_mode           | Mode 1 (DMA_BUF_IOCTL_SYNC, optimal)            |
| mm_params_cache         | Enabled (caches VTCM layout by weight ptr + ne11) |
| DSP debug logging       | Removed (`-DNDEBUG` skel build)                 |
| BF16 offload            | Supported (stored as F16 in repack buffer)      |
| v75/8Gen3 thread clamp  | Clamps to `max_hw_threads - 2` via IDL out-param |
| LTO                     | Rejected (net regression -9% to -14% PP)        |
| flash-attn-ops.c -O3    | Blocked (LLVM 19.0.07 PromoteFloatResult bug)   |

## 18. Related Documents

1. [ion-mempool-vs-perbuffer-analysis-20260713.md](ion-mempool-vs-perbuffer-analysis-20260713.md) - JZ ggml-hexagon vs Qualcomm ggml-hexagon: Architecture Analysis; explains the decisive single-pool advantage (session-resident repacked weights)
2. [warmup-ab-test-and-analysis-20260713.md](warmup-ab-test-and-analysis-20260713.md) - FastRPC/ION warmup A/B test; batch-level pipelining analysis of QCOM's dspqueue
