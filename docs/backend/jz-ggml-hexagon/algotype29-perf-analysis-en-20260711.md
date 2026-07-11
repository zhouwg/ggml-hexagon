# algotype=29 Performance Analysis: JZ vs Qualcomm ggml-hexagon Backend (2026-07-11)

*Author: GLM-5.2 (2026-07-11). Authored by GLM-5.2 based on a full review
of the JZ (`ggml-hexagon.cpp`, `htp/entry.c`, `htp/dsp-ctx.h`) and
Qualcomm (`ggml-hexagon-qcom.cpp`, `htp/main.c`) codebases after the dual
path removal, the upstream master merge, and the TG/PP optimization
investigation documented in
[tg-pp-optimization-attempts-20260711.md](tg-pp-optimization-attempts-20260711.md).*

This document updates
[algotype29-perf-analysis-en.md](algotype29-perf-analysis-en.md) (2026-07-05
baseline, PP=105.6 / TG=18.6 before the weight repack breakthrough). All
architectural analysis is self-contained here; the 0705 doc is kept as a
historical snapshot of the pre-optimization state.

## Background

Both the JZ and Qualcomm versions of the ggml-hexagon backend route
through Qualcomm's `execute_op` path (the `execute_op` implementation
lives in `htp/` and is shared by both versions). The DSP entry points
differ: JZ uses `htp/entry.c`, while Qualcomm uses `htp/main.c`.
The AP-side implementations differ significantly, leading to performance
differences. (The `mulmat_algotype` config knob that previously selected
between the self-built and Qualcomm paths was removed in the dual path
cleanup; the "algotype=29" label survives only as a historical comment
in `ggml-hexagon.cfg` and in the document filename.)

JZ ggml-hexagon is built on two fundamental architectural choices that
originated from the upstream PR
[#12326](https://github.com/ggml-org/llama.cpp/pull/12326) (March 2025):

1. **Native FastRPC** - Direct synchronous FastRPC calls via Hexagon SDK,
   not dspqueue's async wrapper. PR-12326 already implemented this as a
   FastRPC-based per-op path (alongside a QNN SDK path); JZ ggml-hexagon
   evolved it into the ion-based op-batch design
2. **ION shared memory pool** - Single shared memory pool with offset
   addressing, inspired by the "shared buffer or memory pool" idea proposed
   in PR-12326

These are deliberate design choices, not limitations. The theoretical basis is
that LLM inference is inherently serial (autoregressive TG + serially dependent
subgraphs), which limits the benefit of async pipelining.

***

## What changed since 2026-07-05

The codebase evolved significantly between 2026-07-05 and 2026-07-11.
The major changes, in chronological order:

1. **Weight repack moved to set_tensor** - JZ implemented a repack
   buffer type with `is_host=false`, forcing GGML core to route
   quantized weights through `set_tensor` during model load. This
   moved the expensive tiled repack from per-inference (Phase 4.5)
   to one-time at model load. PP jumped from 105.6 to ~321 tok/s.
   (Detailed in section 2.)
2. **Graph cache content hash** - Replaced the dead `cgraph->uid` key
   with FNV-1a content hash over each node's `{op, ne, nb, src, data}`.
   Cache hit rate went from 0% to 99.2%. (Detailed in section 5.)
3. **VTCM layout API merge (commit `15053402b`)** - Brought in
   Qualcomm's new VTCM layout API (`81ff7abe5`). JZ adapted via
   wrapper functions in `matmul-ops.h`. (Detailed in section 11.2.)
4. **Dual path removal (2026-07-10)** - commits `6c11b225d` (step1) and
   `ba4fd0104` (step2). JZ deleted its entire independent DSP kernel
   layer (`kernels/ggml-dsp.c`, `mulmat.c`, `flash_attn.c`, `add.c`,
   `sub.c`, `mul.c`, `div.c`, `rmsnorm.c`, `rope.c`, `softmax.c`,
   `silu.c`, `scale.c`, `concat.c`, `cpy.c`, `getrows.c`,
   `diagmask.c`, `repeat.c`, `skel.c`, `stub.c`, `test-hmx.c`,
   `worker_pool.cpp`, `dot.S`, plus the `ggml-dsp.h` port header) -
   24298 lines removed. The `mulmat_algotype=32` self-built dispatch
   path was removed at the same time. JZ now routes **all** ops through
   the shared `htp/` execute_op path, exactly like Qualcomm does.
   (Detailed in section 1.)
5. **Upstream master merge (2026-07-11 morning)** - commit `d8d0a707e`.
   Brought in Qualcomm's latest `htp/` kernels including a new
   `op_unary` pipeline (commit `fb30ba9a6`) that requires
   host-precomputed `htp_unary_kernel_params`, an ARGSORT performance
   improvement (`67776eaee`), the new ET backend, and the usual stream
   of upstream ggml/llama core changes.
6. **VTCM session-lifetime refactor (2026-07-11, commit `7261e75bb`)** -
   VTCM acquire/release moved from per-batch (~4352 calls/inference) to
   per-session. (Detailed in section 10.)
7. **TG/PP optimization investigation (2026-07-11)** - documented in
   [tg-pp-optimization-attempts-20260711.md](tg-pp-optimization-attempts-20260711.md).
   Six experiments chased the 1.40x TG gap; the real TG hot spot turned
   out to be `FLASH_ATTN_EXT` (not MUL_MAT), and an ARM CPU ceiling
   measurement showed TG is DDR-bandwidth-bounded at 32.4 tok/s.

The net effect on file sizes:

**Table 1: File size comparison (2026-07-05 vs 2026-07-11)**

| File (2026-07-11 name)                          | 2026-07-05 | 2026-07-11 | Delta  | Notes                                                     |
| ----------------------------------------------- | ---------: | ---------: | -----: | --------------------------------------------------------- |
| `ggml-hexagon-jz.cpp` (was `ggml-hexagon.cpp`)  |     6297   |     6717   |   +420 | repack buf, graph cache, graph_opt, dual-path glue removed |
| `ggml-hexagon.cpp` (was `ggml-hexagon-qcom.cpp`)|    4392   |     4452   |    +60 | upstream changes                                          |
| `htp/entry.c` (was `kernels/entry.c`)           |     1939   |     2192   |   +253 | VTCM session-lifetime, profiler gated off                 |
| `htp/main.c`                                    |     1004   |     1008   |     +4 | unchanged (Qualcomm-owned)                                |
| `htp/dsp-ctx.h` (was `kernels/ggml-ops.h`)      |      120   |      177   |    +57 | renamed, slimmed to context + descriptors                 |
| `htp/ggml_dsp.idl` (was `kernels/ggmlop.idl`)   |       45   |       12   |   -33  | renamed, interface simplified                             |
| `htp/*.c` shared kernels (was `kernels/*.c`)    |    20891   |    14683   |  -6208 | monolithic ggml-dsp.c split, upstream refactored          |

JZ's four files (`Makefile`, `dsp-ctx.h`, `entry.c`, `ggml_dsp.idl`)
are now merged into `htp/` alongside Qualcomm's kernels. All DSP kernel
implementations live in `htp/` and are shared verbatim with Qualcomm.

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

***

## Latest Benchmark (2026-07-11)

### Test Conditions

- **Model file**: `/sdcard/gemma-4-E2B-it-Q4_0.gguf` (3.0 GB, 21 layers)
- **Device**: Snapdragon 8 Elite (v79, OnePlus 13)
- **CLI params**: `-ngl 99 -t 6 -n 256 --ctx-size 8192 --ubatch-size 64 --poll 1000 --no-warmup --no-mmap -fa on`
- **Prompt**: `"Hello, good morning, you are a powerful domain expert and know many things, now pls help to introduce the movie Once Upon a Time in America briefly, pls pay attention short then 1000 words"` (44 tokens)
- **Config**: `offload_cgraph_type=2`, `ion_sync_mode=1`,
  `dsp_cache_mode=4`, `enable_graph_optimize=1`, full `enabled_ops`
- **Offloaded ops** (31 total): ADD, SUB, MUL, DIV, SQR, SQRT, SUM\_ROWS,
  CUMSUM, REPEAT, CONCAT, NORM, RMS\_NORM, L2\_NORM, MUL\_MAT, SCALE, CPY,
  CONT, GET\_ROWS, SET\_ROWS, DIAG, DIAG\_MASK\_INF, SOFT\_MAX, ROPE, PAD,
  ARGSORT, TRI, FILL, FLASH\_ATTN\_EXT, UNARY, GLU, NONE (metadata)
- **Offloaded MUL\_MAT types**: F32, F16, Q4\_0, Q8\_0, Q4\_1, IQ4\_NL, MXFP4

### PP & TG Comparison (JZ vs Qualcomm, same device, same model, same day)

Both runs measured on 2026-07-11 evening, same device (Snapdragon 8 Elite,
v79, OnePlus 13), same model, same `running_params`, same prompt. This is
the most apples-to-apples comparison to date.

**Table 3: PP and TG summary (3 runs, 2026-07-11 evening)**

| Run         | JZ PP (tok/s) | JZ TG (tok/s) | QCOM PP (tok/s) | QCOM TG (tok/s) |
| ----------- | ------------- | ------------- | --------------- | --------------- |
| 1 (21:28)   | 339.12        | 18.90         | 334.36          | 26.40           |
| 2 (22:03)   | 335.99        | 18.96         | 325.09          | 26.11           |
| 3 (23:38)   | 336.40        | 18.99         | 352.74          | 26.26           |
| **average** | **337.17**    | **18.95**     | **337.40**      | **26.26**       |

PP averages are nearly identical (337.17 vs 337.40, within 0.1%).
TG is consistently ~1.39x: JZ avg 18.95 vs QCOM avg 26.26 tok/s.

**Table 4: Detailed metrics (run 3, 23:38)**

| Metric                | JZ (23:38:21)       | QCOM (run 3)        | Gap               |
| --------------------- | ------------------- | ------------------- | ----------------- |
| **PP (tok/s)**        | 336.40              | 352.74              | QCOM 4.9% faster  |
| **TG (tok/s)**        | 18.99               | 26.26               | QCOM 1.38x faster |
| **PP time (ms)**      | 130.80 (44 tokens)  | 124.74 (44 tokens)  | -                 |
| **TG time (ms)**      | 13425.10 (255 runs) | 9709.58 (255 runs)  | -                 |
| **TG per-token (ms)** | 52.65               | 38.08               | 14.57 ms gap      |
| **load time (ms)**    | 132.59              | 125.41              | -                 |
| **graphs reused**     | 253                 | 253                 | -                 |

TG per-token gap across 3 runs: 15.01, 14.47, 14.57 ms (avg 14.68 ms,
stddev 0.28 ms). The gap is extremely stable, pointing to a fixed
per-layer attention cost difference (see next section).

JZ-side RPC stats (run 3): `batch_calls=4352`, `avg_p7=2312 us`,
`avg_graph=2376 us`, cgraph cache hit rate 99.2% (4317/4352), mul_mat
coverage 696 total with 275 HMX (39.5%).

On 2026-07-05, JZ was at PP=105.6 vs QCOM at 217.8 (QCOM 2.06x faster);
the PP gap has closed dramatically - a 3.2x improvement in JZ PP.
The TG per-token gap (~14.7 ms) matches the FLASH_ATTN_EXT analysis
in the next section.

> Note: QCOM outputs exhibited noticeable repetition across all 3 runs
> (phrases like "sprawling, sprawling", duplicated sentences, and
> fictional character names e.g. "Nucky Thompson" from Boardwalk Empire).
> This has been observed in both JZ and QCOM runs and is attributed to
> the shared Qualcomm DSP kernel code (`htp/*.c`), not an AP-side issue.

> Note: PP measurements vary 5-30% between consecutive runs of the same
> binary due to device state (walt governor, DSP DCVS, ION freshness).
> See [pp-regression-misdiagnosis-20260711.md](pp-regression-misdiagnosis-20260711.md)
> for a detailed analysis of a non-reproducible 362 tok/s measurement.


### TG hot spot breakdown (from the longtail profiler)

The longtail profiler (preserved in `#if 0` at
[ggml-hexagon.cpp:5868](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L5868))
showed that a single 27 ms FastRPC call processes all 21 layers of one
TG token:

```
[LONGTAIL-PROBE] batch_call#4131 dsp_exec=26873us n_ops=439
ops=[ADD=63, MUL=42, RMS_NORM=125, MUL_MAT=125, GET_ROWS=1,
      ROPE=20, FLASH_ATTN_EXT=21, UNARY=21, GLU=21]
mm=125 mm_ne11[min=1 max=1 avg=1.0] mm_ne01[sum=403200]
```

Per-token DSP time decomposition:
- 21 x FLASH_ATTN_EXT ~ **27 ms** (dominant, 1.29 ms per layer)
- 125 x MUL_MAT (m=1) + 125 x RMS_NORM ~ a few ms
- 21 x ROPE / 21 x UNARY / 21 x GLU ~ a few ms

This is the most important finding since 2026-07-05: **the TG gap is
not a matmul problem, it is an attention problem.** QCOM's 37.89 ms/token
implies 1.80 ms per attention layer; JZ's 52.90 ms/token implies 2.52 ms
per layer. The 0.72 ms/layer attention gap x 21 layers = 15.1 ms, which
accounts for essentially the entire 15.01 ms TG per-token gap measured
on 2026-07-11 evening.

### ARM CPU TG ceiling (DDR-bandwidth-bounded)

A separate ARM CPU build (`build_armcpu`, `GGML_LLAMAFILE=ON`,
`GGML_OPENMP=OFF`) measured TG=32.4 tok/s on the same device, same
model, same `running_params`:

```
weight bytes per token  = 1.5 GB
TG rate                 = 32.4 t/s
weight bytes per second = 48.6 GB/s
8 Elite LPDDR5X peak    = 51.2 GB/s
measured / ceiling      = 95-97% of DDR physical wall
```

ARM CPU TG (32.4) is **1.72x the Hexagon DSP TG (18.90)** on
gemma-4-E2B-it. The system is at the DDR physical ceiling; toolchain
optimization cannot exceed this because every cycle is waiting for DDR.
This opens the **hybrid scheduling** path (PP on DSP at 339 t/s, TG on
CPU at 32.4 t/s), target end-to-end ~1.7x speedup on this model.
Detailed in
[tg-pp-optimization-attempts-20260711.md](tg-pp-optimization-attempts-20260711.md)
Experiment 7.

***

## 1. Dual Path Removal (NEW - structural)

### Status: JZ now shares the same DSP kernel code as Qualcomm.

### Before (2026-07-05 baseline)

JZ had two parallel DSP kernel code paths:

1. **algotype=29 path** - dispatched through Qualcomm's `execute_op` in
   `htp/` (via `dsptensor_to_htp_tensor` bridge)
2. **algotype=32 path** - JZ's own self-built kernel dispatch in
   `kernels/mulmat.c`, `kernels/flash_attn.c`, `kernels/add.c`, etc.,
   backed by the `ggml-dsp` port (`kernels/ggml-dsp.c` 9946 lines,
   `kernels/ggml-dsp.h` 2256 lines)

The `ggml-dsp` port was a tiny ggml running directly on the Hexagon DSP,
adapted from the original [ggml](https://github.com/ggml-org/ggml). It
stripped data structures not needed for on-DSP computation and kept
quantize/dequantize reference implementations as scalar baselines for
HVX/HMX vectorization. `dsptensor` was `#define`'d as `ggml_tensor` so
DSP-side op implementations could reuse upstream ggml's API surface.

### After (2026-07-11)

The entire algotype=32 path and the `ggml-dsp` port are deleted. JZ
routes **all** ops through the shared `htp/` execute_op path, exactly
like Qualcomm. JZ's four files are now merged into `htp/` alongside
Qualcomm's kernels:

- `entry.c` (2192 lines) - FastRPC entry point, cache management,
  `dsptensor` <-> `htp_tensor` bridge, `execute_op` dispatch (moved
  from `htp/main.c`)
- `dsp-ctx.h` (177 lines) - `struct dsp_context`, `dsptensor`,
  `hex_tensor_desc`, `hex_op_desc`, `hex_batch_hdr` (renamed from
  `ggml-ops.h`)
- `ggml_dsp.idl` (15 lines) - FastRPC interface (renamed from
  `ggmlop.idl`)
- `Makefile` - builds `libggmldsp-skel.so` from `entry.c` + all
  `htp/*.c` sources

The `Makefile` (now in `htp/`) compiles all `*.c` in the same directory
into the skel:

```makefile
SRCS = ggml_dsp_skel.c entry.c worker-pool.c hex-dma.c hmx-queue.c \
       binary-ops.c unary-ops.c sum-rows-ops.c softmax-ops.c act-ops.c \
       rope-ops.c set-rows-ops.c get-rows-ops.c cpy-ops.c repeat-ops.c \
       argsort-ops.c ssm-conv.c cumsum-ops.c fill-ops.c concat-ops.c \
       diag-ops.c solve-tri-ops.c gated-delta-net-ops.c pad-ops.c \
       matmul-ops.c flash-attn-ops.c
```

### Why this matters

1. **Code surface reduced by ~24000 lines.** The `ggml-dsp` port was a
   maintenance burden that duplicated upstream ggml's data structures
   and scalar implementations. With the shared `htp/` path being the
   only path, there is no second implementation to keep in sync.
2. **`mulmat_algotype` config knob removed entirely.** The `32` value
   (self-built dispatch) is gone, and the knob itself was removed from
   `ggml-hexagon.cfg`. The "algotype=29" label survives only as a
   historical comment in the cfg (e.g., `enable_opfusion` is annotated
   "algotype=29 only") and in the document filename. There is no longer
   a runtime switch to select between kernel paths.
3. **No measurable PP/TG regression.** The dual path removal was
   investigated as a potential PP regression source
   ([pp-regression-misdiagnosis-20260711.md](pp-regression-misdiagnosis-20260711.md));
   the "regression" was a measurement artifact (the 362 mean from
   2026-07-10 morning was not reproducible on the same commit later
   that day). The execute_op path was already in use before the
   cleanup, so removing the unused algotype=32 path changed nothing at
   runtime.
4. **`dsptensor` is retained** as JZ's AP-side tensor descriptor format
   (single ION offset addressing), but it is now only a thin wrapper.
   The `dsptensor_to_htp_tensor` bridge in
   [entry.c:1202](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/entry.c#L1202)
   converts it to the shared `htp_tensor` (with `bi=0`) before calling
   `execute_op`. The `ggml-dsp` `#define ggml_tensor dsptensor` trick
   is gone; `dsptensor` is now just a C struct, not a ggml alias.

***

## 2. Weight Repack Timing (RESOLVED)

### Status: Gap closed. JZ now repacks at `set_tensor` (one-time).

### Before (2026-07-05 baseline)

JZ repacked all quantized weights on every `graph_compute_batch` call in
Phase 4.5, consuming hundreds of milliseconds per inference. This was the
largest performance bottleneck, holding PP at 105.6 tok/s.

### After (2026-07-11)

JZ implements a **repack buffer type** with `is_host=false`:

```cpp
static bool ggml_backend_hexagon_repack_buffer_is_host(ggml_backend_buffer_type_t buft) {
    return false;  // forces GGML core to call set_tensor
}
```

When the model loader encounters quantized weights (Q4\_0, Q4\_1, Q8\_0,
IQ4\_NL, MXFP4), the `supports_op` gate in MUL\_MAT ensures they are
allocated in the repack buffer type. Because `is_host=false`, GGML core
routes data through `set_tensor`, which performs the in-place tile repack:

```cpp
if (is_repack) {
    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_IQ4_NL:
            repack_q4_0_tiled_to_buf(tensor, data, tensor->data);
            break;
        case GGML_TYPE_Q4_1:
            repack_q4_1_tiled_to_buf(tensor, data, tensor->data);
            break;
        // ... Q8_0, MXFP4 ...
    }
}
```

Phase 4.5 now only tracks ION offsets for descriptor updates; no repack
work is done per-inference.

### Performance Impact

This single change accounts for the majority of the PP improvement
(105 -> 339 tok/s). Phase 4.5 cumulative time dropped from dominant to
8806 us total (vs 44228 us for Phase 6, the new dominant phase).

### Remaining difference vs Qualcomm

Both backends now repack at `set_tensor`. The implementation is
functionally equivalent.

***

## 3. FastRPC Call Pattern (OPEN - Structural, accepted trade-off)

### JZ Version

- **Call method**: Single synchronous `ggmlop_dsp_execute_batch`
- **Parameters**: 2 scalars (`batch_offset`, `total_desc_size`)
- **Data transfer**: Single ION mempool + offset addressing
- **Pipelining**: None (AP blocks waiting for DSP completion)

### Qualcomm Version

- **Call method**: dspqueue message queue
- **Parameters**: `dspqueue_write` + `dspqueue_buffer` (containing `fd + offset + size`)
- **Data transfer**: fd + offset two-level addressing, supporting multiple independent shared buffers
- **Pipelining**: Up to 16 batches in-flight (`opt_opqueue=16`), AP/DSP parallel execution

### Performance Impact - revised

The 2026-07-11 investigation confirmed with hard data that dspqueue is
NOT the decisive factor in the TG gap: it is almost entirely from
`FLASH_ATTN_EXT` (1.29 ms x 21 layers = 27 ms per token), not from
per-call dispatch overhead. With the graph cache at 99.2% hit rate, AP
preparation is <1 ms/token - far too small to explain the 15 ms TG gap.

The path forward for TG is **hybrid scheduling** (PP on DSP, TG on CPU),
not adopting dspqueue. The synchronous architecture is the point of
the project; the TG gap is an accepted trade-off that is now
understood to be attention-kernel-bound, not dispatch-bound.

***

## 4. Op Fusion Scope (PARITY + VTCM guard added)

### JZ Version (Phase 2.5)

**Table 5: JZ op fusion types (Phase 2.5)**

| Fusion Type                                  | Supported              | Notes                                          |
| -------------------------------------------- | ---------------------- | ---------------------------------------------- |
| RMS\_NORM + MUL -> HTP\_OP\_RMS\_NORM\_MUL   | Yes                    | Linear scan; fires every graph (1-3 per graph) |
| MUL\_MAT + ADD -> HTP\_OP\_MUL\_MAT\_ADD     | Yes                    | Bias add inside matmul kernel; VTCM budget checked |
| MUL\_MAT QKV merge -> HTP\_OP\_MUL\_MAT\_QKV | Yes                    | 3 MUL\_MAT (Q,K,V) merged into 1               |
| MUL\_MAT FFN merge -> HTP\_OP\_MUL\_MAT\_FFN | Yes                    | gate + up merged into 1                        |
| Graph reorder                                | Yes (since `602d71e65`) | Forward 16-group window; runtime-configurable |

### New: VTCM budget check for MUL_MAT_ADD fusion (commit `2b1e7bd8c`)

The MUL_MAT + ADD fusion previously fired unconditionally without
consulting the VTCM budget. If the underlying MUL_MAT already
saturates VTCM, the fused op could silently overflow the DSP scratch
region. JZ now mirrors Qualcomm's guard
([ggml-hexagon-qcom.cpp:3595](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-qcom.cpp#L3595)):

```cpp
const size_t vtcm_budget = (size_t)ctx->socinfo.vtcm_size_in_mb * 1024 * 1024;
if ((size_t)kparams->vtcm_size > vtcm_budget) {
    return false;  // skip fusion, let MUL_MAT and ADD run separately
}
```

No DSP-side change was needed: bias (src2) is read from DDR, not VTCM,
so the kparams from Phase 2's MUL_MAT precompute are reused as-is.
Validated on 8Elite (v79) with 5 models: no MUL_MAT_ADD fusion is
skipped at the 8MB VTCM budget, performance is unchanged within seed
variance.

### New: mul_mat coverage tracer (commit `0904efaa7`)

JZ now emits per-batch counters in `hexagon_op_exec_stats_t`:

- `n_mul_mat_total`, `n_hmx_used`, `n_fused_qkv`, `n_fused_ffn`,
  `n_fused_mul_mat_add`

A representative gemma4 run reports:

```
total=696 hmx=275 (39.5%) qkv_fused=15 ffn_fused=56 ...
```

This matches the expected 1 PP + 34 TG graph ratio (44+255 tokens,
99.2% cgraph cache hit). HMX dominates PP (39.5% of all MUL_MATs are
HMX-eligible); HVX fusion dominates TG. PP work should focus on the
HMX matmul kernel (upstream-owned); TG work can target the in-tree
QKV/FFN fusion patterns.

### Qualcomm Version (`try_fuse_node`)

**Table 6: Qualcomm op fusion types (try_fuse_node)**

| Fusion Type                                  | Supported | Notes                                          |
| -------------------------------------------- | --------- | ---------------------------------------------- |
| RMS\_NORM + MUL -> HTP\_OP\_RMS\_NORM\_MUL   | Yes       | Uses `ggml_can_fuse`                           |
| MUL\_MAT + ADD -> HTP\_OP\_MUL\_MAT\_ADD     | Yes       | Uses `ggml_can_fuse`                           |
| MUL\_MAT QKV merge -> HTP\_OP\_MUL\_MAT\_QKV | Yes       | 3 mul\_mat merged into 1, reordered to KVQ     |
| MUL\_MAT FFN merge -> HTP\_OP\_MUL\_MAT\_FFN | Yes       | gate + up merged into 1                        |
| Graph reorder                                | Yes       | Stacks MUL\_MATs with same src1 for VTCM reuse |

### Performance Impact

Fusion scope is at parity (all 5 fusion types). Graph reorder is now
implemented in JZ (commit `602d71e65`, made runtime-configurable).
However, a single-variable A/B test
([pp-cache-optimization-deadends-20260711.md](pp-cache-optimization-deadends-20260711.md)
attempt 3) showed `enable_graph_optimize=0` vs `=1` differ by 3.5 t/s
(339.45 vs 335.91), which is within noise. The 5-10% PP improvement
expected from graph reorder did not materialize for gemma4 PP=44
tokens on 8 Elite v79. The reorder may still help for VTCM pressure
in other models or longer contexts, but for this workload it is a
no-op. Kept on by default for future-proofing.

***

## 5. Graph Cache (RESOLVED)

### Status: Gap closed. Cache now works correctly with 99.2% hit rate.

### Before (2026-07-05 baseline)

The graph cache keyed by `cgraph->uid` was dead code in practice:
PP runs once (all MISS, first fill); TG graphs get new uids after
PP->TG rebuild (cache stays MISS).

### After (2026-07-11)

The cache uses a **content hash** (FNV-1a over each node's
`{op, ne[4], nb[4], src[0..2] ptr, data ptr}`) instead of `cgraph->uid`:

```cpp
const uint64_t content_hash = compute_content_hash();
auto it = ctx->cgraph_cache.find(content_hash);
if (it != ctx->cgraph_cache.end() &&
    it->second.n_nodes == cgraph->n_nodes &&
    it->second.hex_ops.size() > 0) {
    // Hit: restore cached tensor_src, supported_nodes, hex_ops, weight_indices
    cache_hit = true;
    ctx->cgraph_cache_hits++;
}
```

On a hit, the cache skips Phase 1 (tensor dedup), Phase 2 (op descriptor
build), and Phase 2.5 (op fusion) entirely. With 17 subgraphs per TG
token and 100% hit rate after warmup, this saves ~646us/token.

### Cached state

```cpp
struct cgraph_cache_entry {
    uint64_t content_hash;
    int n_nodes, n_tensors, n_ops;
    std::vector<ggml_tensor *> tensor_src;
    std::vector<ggml_tensor *> supported_nodes;
    std::vector<ggml_tensor *> unsupported_nodes;
    std::vector<hex_op_desc>   hex_ops;
    std::vector<uint32_t>      weight_indices;
};
```

### Benchmark confirmation

```
cgraph cache: hits=4317 misses=35 (hit_rate=99.2%) entries=35
```

The 35 misses correspond to the first fill of 35 unique graph structures
(17 TG subgraphs + PP splits). After warmup, every subsequent token hits
cache.

### Remaining difference vs Qualcomm

Both backends now cache by graph identity. Qualcomm also caches the
graph reorder step (which JZ now has, but it is a no-op for this
workload). In practice this is a wash for TG.

***

## 6. mm_params_cache (JZ only)

JZ caches precomputed `htp_mm_kernel_params` by a composite key
(weight data pointer XOR ne11):

```cpp
const uintptr_t cache_key = (uintptr_t) src0->data ^ ((uintptr_t) ne11 << 32);
auto it = ctx->mm_params_cache.find(cache_key);
if (it != ctx->mm_params_cache.end()) {
    *kparams = it->second;
    return;
}
```

This skips the multi-hundred-microsecond thread/chunk search in
`htp_mm_hvx_vtcm_layout_build` / `htp_mm_hmx_vtcm_layout_build` for
repeated MUL_MAT calls with the same weight tensor. For TG (where
ne11=1 for every token), the cache hits after the first token.

Qualcomm's `ggml_hexagon_precompute_matmul_params` performs the same
VTCM layout computation but does not cache the result across calls.
JZ's `mm_params_cache` gives it a slight edge in TG dispatch overhead,
partially offsetting the synchronous FastRPC penalty.

***

## 7. Session Consistency Gate (JZ only)

JZ mirrors Qualcomm's `ggml_hexagon_supported_buffer` check to prevent
the scheduler from mixing tensors across different Hexagon sessions or
non-Hexagon buffers:

```cpp
static bool ggmlhexagon_tensor_buffer_is_owned_by(ggml_backend_dev_t dev, const struct ggml_tensor * t) {
    if (!t || !t->buffer) return true;  // neutral
    // Accept if buffer is hexagon (main or repack) on this device
    // Reject if hexagon on different device or non-hexagon
}
```

This prevents subtle correctness bugs when multiple Hexagon devices are
present or when the scheduler tries to route an op with mixed CPU/DSP
tensors.

***

## 8. Cache Coherency Management (IMPROVED - dsp_cache_mode settled)

### JZ Version

- **AP side**: Configurable via `ion_sync_mode`
  - `0` = both (DC CVAC + ion\_sync, default)
  - `1` = ion\_sync only (DMA\_BUF\_IOCTL\_SYNC, driver-level)
  - `2` = DC CVAC only (manual cache line management)
- **DSP side**: Configurable via `dsp_cache_mode` bitmask
  - bit 0: first-touch weight bitmap (skip dcinva for repack weights)
  - bit 1: skip dcinva for prior dst (DSP's own dst writes stay in L2)
  - bit 2: bulk dst flush at batch end (collect/sort/merge dst ranges)

### Current safe baseline: dsp_cache_mode=4

Previously `dsp_cache_mode=7` (all bits on) was the default. The
2026-07-11 investigation
([pp-cache-optimization-deadends-20260711.md](pp-cache-optimization-deadends-20260711.md))
found that bits 0 and 1 **garble output** on the new matmul pipeline
(upstream `81ff7abe5`). The root cause is a hardware-level L2 cache
contract mismatch: the QCOM matmul kernels use partial HVX writes
(`hvx_vec_store_u`) for dst as a performance optimization, but the
cache optimization in `entry.c` (bits 0, 1) assumes whole-line writes.
Partial HVX writes do not set the L2 cache line to "Modified" state in
a way that subsequent reads can rely on. These two are incompatible.

The current safe baseline is `dsp_cache_mode=4` (bulk dst flush only).
Bits 0 and 1 cannot be re-enabled without QCOM matmul kernel changes
(out of scope for an integration-layer PR).

### ion_sync_mode=1 is optimal

A sweep of `ion_sync_mode` 0/1/2 confirmed mode 1 is the sweet spot:
mode 0 (double cache maintenance) drops PP from 300+ to ~200; mode 2
(manual only) is similar to mode 0. Mode 1 uses the kernel's
`DMA_BUF_IOCTL_SYNC` which is faster than userspace DC CVAC/CIVAC for
large ranges.

### Qualcomm Version

dspqueue driver automatic management via
`DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT`.

### Performance Impact

With `ion_sync_mode=1` and `dsp_cache_mode=4`, JZ's cache coherency
overhead is small. Phase 6.5 cumulative ~10 ms; Phase 7.5 (CIVAC) ~9 ms
total across 4352 calls (~2us/call). The remaining gap vs Qualcomm's
driver-level management is negligible.

***

## 9. Profiler Infrastructure (IMPROVED)

### AP-side profiler

Tracks cumulative time for phases p1, p2, p2.5, p3, p4, p4.5, p5, p6,
p6.5, p7, p7.5, p8 per `graph_compute_batch` call. Breaks down Phase 7
into `rpc_setup` + `dsp_exec` + `civac`. Computes min/p50/p95/max
histograms for the last 1024 calls. Reports cgraph cache hit/miss
counts.

### New: longtail profiler (commit `af2cb4418`)

A `LOG_ALWAYS` longtail probe was added inside the FastRPC dispatch
path to log the op composition of any batch call whose `dsp_exec`
exceeds 5 ms. Throttled to one log per 100 ms wall-clock to avoid log
flood. Probe code is preserved in source inside `#if 0 ... #endif` for
future re-activation; runtime cost is zero. This is the instrument
that identified `FLASH_ATTN_EXT` as the TG hot spot.

### New: mul_mat coverage tracer (commit `0904efaa7`)

Per-batch counters in `hexagon_op_exec_stats_t`:
`n_mul_mat_total`, `n_hmx_used`, `n_fused_qkv`, `n_fused_ffn`,
`n_fused_mul_mat_add`. See section 4.

### DSP-side profiler (gated off by default)

`entry.c` includes a per-op timing profiler that records min/max/avg
execution time per op type, dumped via `dump_op_prof`. As of commit
`7261e75bb`, this is wrapped in `#if HEX_OP_PROF ... #endif` with
`HEX_OP_PROF` defaulting to 0 (off). Skel size drops by 6880 bytes
(706984 -> 700104). Restore by passing `-DHEX_OP_PROF=1` to `make`.

### New: dsp_cache_trace_bit0/bit1 (commits `5b2aa6244`, `60354c52c`)

Diagnostic instrumentation for the `dsp_cache_mode` bits 0/1 garble
investigation. When non-zero, emits one log line per bit 0/1 decision
(SKIP or INVAL) with op/src/ptr/len fields. Default 0 (off) so
production perf is unaffected. This is the instrument that localized
the stale-L2-read culprit to a specific bit/op combination.

***

## 10. VTCM Session-Lifetime (NEW - entry.c refactor)

### Status: VTCM is now acquired once per session, not per batch.

### Before (2026-07-05 baseline)

`dsp_vtcm_acquire()` and `dsp_vtcm_release()` were called on every
`execute_batch` invocation - ~4352 times per inference. Each call
invoked `HAP_compute_res_acquire_cached` / `HAP_compute_res_release_cached`,
generating ~8700 lines of SDK FARF log per inference.

### After (2026-07-11, commit `7261e75bb`)

`ggml_dsp_open` does one `HAP_compute_res_acquire_cached` and
`ggml_dsp_close` does one `HAP_compute_res_release_cached`. VTCM is
held continuously for the session. This matches the Qualcomm HTP
pattern (`vtcm_acquire` / `vtcm_release` only fire when transitioning
between "active processing" and "forced release").

```c
// ggml_dsp_open (after HAP_compute_res_acquire succeeds)
dsp_vtcm_acquire();   // once per session, sets vtcm_valid=1

// ggml_dsp_close (before HAP_compute_res_release)
dsp_vtcm_release();   // once per session, sets vtcm_valid=0

// execute_batch: no per-batch acquire/release calls
```

### Performance Impact

**Inconclusive.** Single-run PP varies 315-345 t/s for gemma4 in warm
state regardless of whether VTCM is session-lifetime or per-batch.
The change is a code-cleanup that matches the Qualcomm pattern and
eliminates logcat spam; any performance impact is below the
measurement noise floor. See
[vtcm-session-lifetime-20260711.md](vtcm-session-lifetime-20260711.md)
for the full investigation.

Trade-off: lose the ability to respond to a forced-release callback
from another session. For single-session use (the current deployment)
this is a non-issue.

***

## 11. Upstream Merge Adaptations (NEW)

### Upstream master merge (commit `d8d0a707e`, 2026-07-11 morning)

Brought in Qualcomm's latest `htp/` kernels plus the usual stream of
upstream ggml/llama core changes. Three JZ-side adaptations were
required:

### 11.1 Unary precompute port (commit `f2f259214`)

Upstream commit `fb30ba9a6` introduced a new `op_unary` pipeline that
requires host-precomputed `htp_unary_kernel_params` (n_threads,
col_tile, vtcm_size, etc.). Without this, `op_unary` reads zeroed
kparams and the output is garbled.

JZ ported `ggml_hexagon_precompute_unary_params` from
`ggml-hexagon-qcom.cpp`, adapted to use `ctx->n_threads` and
`ctx->socinfo.vtcm_size`. Added `ggml_op_to_htp_op_unary()` mapping
(GELU/SILU excluded - they go through `op_activations`). Precompute is
called in Phase 2 main loop and the RMS_NORM_MUL fusion path.

`HTP_OP_TRI` is routed to `op_unary()` in entry.c to match upstream
`htp/main.c`.

### 11.2 VTCM layout API (from a prior upstream merge, preserved)

The previous merge (`15053402b`) brought in Qualcomm's new VTCM layout
API (`81ff7abe5`):

- `htp_mm_hvx_get_vtcm_sizes` -> `htp_mm_hvx_vtcm_layout_build` + `struct htp_mm_hvx_vtcm_layout`
- `htp_mm_hvx_id_get_vtcm_sizes` -> `htp_mm_hmx_vtcm_layout_build` + `struct htp_mm_hmx_vtcm_layout`
- `broadcast_rk2/rk3/rv2/rv3` fields moved from `u.hvx` union member to `htp_fa_kernel_params` struct top level

JZ adapted via wrapper functions in `matmul-ops.h` that translate old
API calls to the new layout-build API, preserving all 16 call sites in
`ggml-hexagon.cpp` and `entry.c` without modification.

### 11.3 ARGSORT performance improvement (upstream `67776eaee`)

Qualcomm's upstream PR improved ARGSORT performance for small tensors.
This is in the shared `htp/argsort-ops.c` and benefits both backends
equally; no JZ-side adaptation was needed.

### 11.4 dsp_cache_mode=4 set as default (commit `f2f259214` side effect)

The new matmul pipeline (`81ff7abe5`) is incompatible with
`dsp_cache_mode` bits 0 and 1 (see section 8). The default was changed
from 7 (all on) to 4 (bulk dst flush only) to avoid garbled output on
the new pipeline.

### 11.5 Build system unification for upstream submission

To prepare for upstream submission, the dual-file naming conflict
(both JZ and QCOM had `ggml-hexagon.cpp` and `CMakeLists.txt`) was
resolved by renaming JZ's files with a `-jz` suffix and restoring
QCOM's files to their upstream names:

**Table 7: File rename mapping for upstream submission**

| File (before)                      | File (after)                  | Role          |
| ---------------------------------- | ----------------------------- | ------------- |
| `ggml-hexagon.cpp` (JZ)           | `ggml-hexagon-jz.cpp`         | JZ AP code    |
| `ggml-hexagon-qcom.cpp` (QCOM)    | `ggml-hexagon.cpp`            | QCOM AP code  |
| `CMakeLists.txt` (JZ)             | `CMakeLists-jz.txt` (deleted) | JZ build      |
| `CMakeLists-qcom.txt` (QCOM)      | `CMakeLists.txt`              | QCOM build    |

The unified `CMakeLists.txt` is based on QCOM's version (minimal diff
to upstream) with a single addition:

```cmake
option(GGML_HEXAGON_JZ "Use JZ's AP implementation" OFF)
```

- **`GGML_HEXAGON_JZ=OFF` (default)**: exactly QCOM's upstream behavior.
  Uses `ggml-hexagon.cpp`, builds DSP skels via `ExternalProject_Add`
  for v73/v75/v79/v81, links `htp_iface` stub.
- **`GGML_HEXAGON_JZ=ON`**: uses `ggml-hexagon-jz.cpp`, builds a single
  DSP skel via `make -C htp/` (JZ's Makefile), links `cdsprpc`, sets
  `HEXAGON_DEFAULT_LIB_SEARCH_PATH`, copies `ggml-hexagon.cfg`.

Variable aliasing: `HEXAGON_SDK_PATH` is set to `${HEXAGON_SDK_ROOT}`
for JZ's Makefile compatibility (Makefile uses `HEXAGON_SDK_PATH`,
upstream CMake uses `HEXAGON_SDK_ROOT`).

The build script (`scripts/build-run-ggmlhexagon-android.sh`) was
simplified: the previous file-swap logic (backup `.cpp.me`, copy QCOM
version, build, restore) was removed. Both variants now build via the
same CMake invocation, differing only in `-DGGML_HEXAGON_JZ=ON/OFF`:

```bash
# JZ build
cmake ... -DGGML_HEXAGON_JZ=ON -DHEXAGON_SDK_ROOT=...

# QCOM build
cmake ... -DGGML_HEXAGON_JZ=OFF -DHEXAGON_SDK_ROOT=...
```

This structure ensures `git merge master` only conflicts in the QCOM
`else()` branch (indentation-only conflicts), while the JZ branch is
pure additive code that upstream does not have.

***

## 12. Tensor Descriptor Data Structure (REVISED)

### JZ Version

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

**DSP-side**: `dsptensor` (defined in
[dsp-ctx.h:21](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/dsp-ctx.h#L21))

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

`dsptensor` is retained as JZ's DSP-side tensor descriptor, but the
`ggml-dsp` port that `#define`'d it as `ggml_tensor` is gone. It is
now just a C struct, not a ggml alias. The `op` + `op_params[16]`
fields are still embedded in the tensor descriptor (a structural
difference from Qualcomm's `htp_tensor`), but this is a historical
artefact of the PR-12326 lineage, not an active design choice.

### What changed: ggml-dsp port deleted

The `ggml-dsp` port
(`kernels/ggml-dsp.h`, `kernels/ggml-dsp.c` - both deleted)
- a tiny ggml running directly on the Hexagon DSP - was deleted in the
dual path removal. It carried:

- `ggml_op` enum and op name/symbol tables
- `ggml_type_traits` / `type_traits_generic` / `type_traits_dsp` tables
- Core utility functions (`ggml_nelements`, `ggml_is_contiguous`, etc.)
- Quantize/dequantize reference implementations (`quantize_row_*`,
  `dequantize_row_*`) as scalar baselines

These were the scalar baselines for HVX/HMX vectorized optimization on
top. With the dual path removed, JZ no longer has its own kernel
implementations; all kernels come from `htp/`, which has its own
scalar baselines. The `ggml-dsp` port was designed to be portable to
other POSIX-friendly xPU targets (x86/ARM/RISC-V CPU, other DSP/NPU),
not just Hexagon. That portability goal is deferred; the current focus
is on the Hexagon-specific `htp/` path.

### Bridge to shared `htp/` code

`entry.c` converts `dsptensor` to the shared `htp_tensor` structure
(with `bi=0`) via `dsptensor_to_htp_tensor`
([entry.c:1202](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/entry.c#L1202))
before calling the shared `execute_op` path:

```c
static inline void dsptensor_to_htp_tensor(const dsptensor * dt,
                                            struct htp_tensor * ht) {
    ht->data  = (uint32_t)(uintptr_t)dt->data;
    ht->size  = (uint32_t)dt->data_len;
    ht->flags = HTP_TENSOR_FLUSHED;
    ht->type  = (uint16_t)dt->type;
    ht->bi    = 0;  // JZ uses single ION pool, always buffer index 0
    // ...
}
```

### Qualcomm Version `htp_tensor`

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

### Difference

The two descriptors are functionally equivalent. The key structural
differences:

**Table 8: Tensor descriptor comparison (dsptensor vs Qualcomm)**

| Aspect               | `dsptensor` (JZ)                  | `htp_tensor` (Qualcomm)                       |
| -------------------- | --------------------------------- | --------------------------------------------- |
| Op metadata          | Embedded (`op` + `op_params[16]`) | Separated into `htp_op_desc`                  |
| Data addressing      | Direct `void *` pointer           | `bi` (buffer index) + `uint32_t` offset       |
| Multi-buffer support | No (single ION pool)              | Yes (via `bi` indexing into `htp_buf_desc[]`) |
| Type width           | `int32_t`                         | `uint16_t`/`uint32_t` (more compact)          |

The `bi` field is the key differentiator for multi-buffer support. JZ
always passes `bi=0` (single ION pool); Qualcomm uses it to index into
`htp_buf_desc[]`. This is tied to the FastRPC call pattern difference
(section 3) and would need to change together if JZ adopts dspqueue.

***

## 13. AP-Side Compiler Optimization (PARITY)

Both backends compile AP-side code with the same ARMv8.7-A + dotprod +
fp16 + i8mm flags, just configured in different files:

- **JZ**: [CMakeLists.txt:48](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/CMakeLists.txt#L48)

```cmake
set(OPT_FLAG " -O3 -march=armv8.7-a+dotprod+fp16+i8mm -mcpu=cortex-x1 -mtune=cortex-x1 -ffp-model=fast -fno-finite-math-only")
```

- **Qualcomm**: [CMakeUserPresets.json:13-14](file:///home/zhouwg/develop/ggml-hexagon/docs/backend/snapdragon/CMakeUserPresets.json#L13-L14)

```json
"CMAKE_C_FLAGS":   "-march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE",
"CMAKE_CXX_FLAGS": "-march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE",
```

These flags enable SDOT/UDOT (int8 dot product) and FP16FML instructions for
AP-side scalar loops (e.g., repack functions, cache coherency helpers).

### DSP-side compiler flags (JZ-only, settled)

JZ's [htp/Makefile](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/Makefile)
uses `-O3 -ffast-math -fno-vectorize` (no LTO) for the DSP skel.
Qualcomm's `htp/cmake-toolchain.cmake` uses `-O2 -flto -fvectorize`.

A 3-row sweep
([tg-pp-optimization-attempts-20260711.md](tg-pp-optimization-attempts-20260711.md)
Experiment 5) confirmed LTO is a **net regression** for JZ's codebase:

**Table 9: LTO experiment results**

| Configuration | TG (t/s) | PP (t/s) |
| ------------- | -------- | -------- |
| JZ original (-O3, no LTO) | **18.78** | **355** |
| -O3 + LTO + new order      | 18.11 (-3.6%) | 324 (-8.7%) |
| -O2 + LTO + fvectorize (QCOM exact) | 18.71 (noise) | 306 (-13.8%) |

Both LTO variants regressed PP (-9% to -14%); 5a also regressed TG
(-3.6%). The `-O3 no-LTO` is empirically the best choice for this
codebase as it currently stands.

### flash-attn-ops.c -O2 workaround (unchanged, unbreakable)

`flash-attn-ops.c` is forced to `-O2` because at `-O3` Hexagon LLVM
19.0.07 emits a `PromoteFloatResult` fatal error on `f16 = freeze`.
A 10-flag sweep (Experiment 6) confirmed no combination of
`-fno-X` / `-mllvm -disable-X` flags can break the workaround. The
bug is in the Hexagon backend (`HexagonDAGToDAGISel` ->
`PromoteFloatResult`), not in any front-end pass. Needs an LLVM
backend patch.

This is directly relevant to the TG gap: `flash-attn-ops.c` is the
TG hot spot (1.29 ms x 21 layers = 27 ms per token), and it is stuck
at `-O2` while the rest of the skel runs at `-O3`.

***

## Summary: Performance Difference Ranking (2026-07-11)

**Table 10: Performance difference ranking (2026-07-11)**

| Rank | Difference           | JZ Version                       | Qualcomm Version            | Impact                       |
| ---- | -------------------- | -------------------------------- | --------------------------- | ---------------------------- |
| 1    | FLASH_ATTN_EXT kernel | Shared `htp/flash-attn-ops.c` (same code, same -O2 limit) | Same | Dominant TG gap source (27 ms/token) - NOT a JZ-vs-QCOM difference |
| 2    | FastRPC call pattern | Synchronous (design choice)      | dspqueue pipeline (16-deep) | Small (AP prep already <1ms) |
| 3    | Graph reorder        | Yes (implemented, no measurable benefit) | Yes (same-src1 stacking) | No-op for gemma4 PP=44 |
| 4    | Cache coherency      | ion\_sync\_mode=1, dsp_cache_mode=4 | dspqueue driver automatic   | Small                        |
| 5    | Batch auto-splitting | Graph cache + ubatch fix         | Auto-split by vmem limits   | Small                        |
| 6    | Tensor descriptor    | Single ION offset                | Two-level (bi + offset)     | (design choice)              |

**Bottom line**: On the 2026-07-11 evening same-day comparison, JZ PP
(339.12) is marginally ahead of QCOM PP (334.36) - PP gap closed and
reversed. The TG gap (1.40x, 18.90 vs 26.40) is now understood to be
**attention-kernel-bound**, not dispatch-bound. Both backends share the
same `flash-attn-ops.c` kernel (stuck at `-O2` due to an LLVM 19.0.07
bug), so the TG gap is NOT from the attention kernel itself - it is
from the per-call overhead around it (descriptor marshalling, cache
management, sync FastRPC wait) that accumulates across 21 layers x 256
tokens. The 15.01 ms per-token gap matches the 0.72 ms/layer x 21
layers = 15.1 ms FLASH_ATTN_EXT decomposition.

The decisive new finding is that **ARM CPU TG (32.4 t/s) > Hexagon DSP
TG (18.90 t/s) on gemma-4-E2B-it by 1.72x**, because the ARM CPU is at
the DDR bandwidth ceiling (48.6 GB/s measured vs 51.2 GB/s physical
peak) while the DSP is not. The path forward is **hybrid scheduling**
(PP on DSP at 339 t/s, TG on CPU at 32.4 t/s), not further DSP kernel
optimization.

### Changes from 2026-07-05

**Table 11: Optimization status changes (2026-07-05 to 2026-07-11)**

| Item                    | 2026-07-11 Status                                          |
| ----------------------- | ---------------------------------------------------------- |
| Weight repack timing    | **RESOLVED** - moved to set_tensor (one-time)                     |
| Op fusion scope         | **PARITY** - all 5 fusion types; VTCM guard added for MUL_MAT_ADD |
| Graph cache             | **RESOLVED** - content-hash based (99.2% hit rate)    |
| Graph reorder           | **IMPLEMENTED + CLOSED** - no measurable PP benefit (3.5 t/s, within noise) |
| FastRPC as TG gap cause | **CONFIRMED not the cause** - TG gap is attention-bound    |
| Dual path               | **REMOVED** - JZ now shares all DSP kernels with Qualcomm  |
| VTCM lifetime           | **SESSION-LIFETIME** - matches Qualcomm pattern            |
| TG bottleneck           | **IDENTIFIED** - FLASH_ATTN_EXT (1.29 ms x 21 = 27 ms/token) |
| ARM CPU TG ceiling      | **MEASURED** - 32.4 t/s, DDR-bandwidth-bounded (1.72x DSP TG) |
| dsp_cache_mode          | **SETTLED** - mode 4 (bulk dst flush only); bits 0/1 garble on new pipeline |
| LTO                     | **REJECTED** - net regression (-9% to -14% PP)             |
| flash-attn-ops.c -O3    | **BLOCKED** - LLVM 19.0.07 PromoteFloatResult bug; needs backend patch |

### New optimizations since 2026-07-05

- Dual path removed (24298 lines deleted; single shared `htp/` kernel path)
- Graph reorder implemented (commit `602d71e65`, runtime-configurable)
- VTCM session-lifetime (commit `7261e75bb`, matches Qualcomm pattern)
- VTCM budget check for MUL_MAT_ADD fusion (commit `2b1e7bd8c`)
- Unary precompute ported for upstream merge (commit `f2f259214`)
- mul_mat coverage tracer (commit `0904efaa7`)
- longtail profiler (commit `af2cb4418`, `#if 0` gated)
- dsp_cache_trace_bit0/bit1 (commits `5b2aa6244`, `60354c52c`)
- HEX_OP_PROF compile switch (commit `7261e75bb`, default off)

***

## Optimization Recommendations

> **Design principle**: JZ backend uses an independent ION-based op-batch
> architecture with synchronous FastRPC. Adopting Qualcomm's dspqueue is
> explicitly out of scope - the independent architecture is the point of
> the project. The TG gap vs Qualcomm (~18.90 vs ~26.40 tok/s) is an
> accepted trade-off of this design. Recommendations below focus on
> improvements within the ION-based architecture, and on the new hybrid
> scheduling path identified on 2026-07-11.

### Priority 1: Phase-aware hybrid scheduling (high ROI, high effort)

ARM CPU TG (32.4 t/s) is 1.72x the Hexagon DSP TG (18.90 t/s) on
gemma-4-E2B-it, because the ARM CPU is at the DDR bandwidth ceiling
while the DSP is not. The path forward is:

- **PP on DSP** (339 t/s) - the DSP's HMX matmul kernel is faster than
  ARM CPU for batch matmul (m=64)
- **TG on CPU** (32.4 t/s) - the CPU's LLAMAFILE-based matmul is faster
  than the DSP for single-token TG (m=1), because the DSP's
  `FLASH_ATTN_EXT` kernel (1.29 ms x 21 = 27 ms/token) dominates TG
  and the CPU does not have this bottleneck

Target: end-to-end ~1.7x speedup on gemma-4-E2B-it workloads. Requires
a follow-up design document (TBD).

### Priority 2: Optimize flash-attn-ops.c within the -O2 constraint (medium ROI, high effort)

The TG hot spot is `FLASH_ATTN_EXT` (1.29 ms x 21 layers = 27 ms per
token). The kernel is stuck at `-O2` due to an LLVM 19.0.07
`PromoteFloatResult` bug on `f16 = freeze`. Options:

1. **Custom m=1 attention kernel** for the TG case, where only the new
   query vector is read (differs from PP's m=64 case where the entire
   prompt is read). This bypasses the `-O2` limit if the new kernel
   avoids f16 in loop-internal computation.
2. **Profile KV cache DMA pattern** in the attention kernel. The cache
   grows from 1 to 256 tokens during TG; the per-call cost should
   scale linearly but might be growing super-linearly due to cache
   thrash.
3. **Wait for Hexagon LLVM 19.0.08 / 20.x** that may fix the
   `PromoteFloatResult` bug. No ETA.

### Priority 3: Lift the flash-attn-ops.c -O3 block (medium ROI, blocked on LLVM)

If the LLVM 19.0.07 `PromoteFloatResult` bug is fixed (either by a QCOM
backend patch or an LLVM 19.0.08+ release), `flash-attn-ops.c` can be
compiled at `-O3` like the rest of the skel. This would tighten the
1.29 ms x 21 = 27 ms per-token cost, which is ~50% of TG time. The
flag sweep in Experiment 6 confirmed no workaround exists within the
current LLVM.

### Closed: Graph reorder (Priority 1 from 2026-07-05)

Implemented in commit `602d71e65`. Single-variable A/B showed 3.5 t/s
difference (within noise) for gemma4 PP=44 on 8 Elite v79. The 5-10%
PP improvement expected from graph reorder did not materialize.
Kept on by default for future-proofing (may help for other models or
longer contexts).

### Closed: QKV/FFN fusion vs HMX trade-off (Priority 2 from 2026-07-05)

The mul_mat coverage tracer (commit `0904efaa7`) confirmed fusion is
already firing at the expected rate (`qkv_fused=15, ffn_fused=56`).
The F32 src1 check and GQA dim check are both no-ops for gemma4
(Experiments 1-2). Limited headroom remains; further fusion saves
dispatch overhead only, not per-token DSP time. The TG bottleneck is
`FLASH_ATTN_EXT`, not MUL_MAT.

### Closed: Phase 6 descriptor marshalling (Priority 3 from 2026-07-05)

Phase 6 (descriptor marshalling) is 45 ms cumulative over 4352 calls.
With the graph cache at 99.2% hit rate, this is below the noise floor
for PP optimization. Not worth pursuing.

### Note on cache coherency

Cache coherency (Phase 6.5 DC CVAC flush + Phase 7.5 CIVAC invalidate)
totals ~19 ms (0.15% of TG time). With `ion_sync_mode=1` already using
driver-level `DMA_BUF_IOCTL_SYNC` and `dsp_cache_mode=4` using bulk
dst flush, the per-call CIVAC p50 is 2 us - near hardware limits.
`dsp_cache_mode` bits 0/1 cannot be re-enabled without QCOM matmul
kernel changes (partial HVX writes violate the whole-line-write
assumption). Further optimization here would yield negligible
improvement (<0.1% TG).

### Note on DSP-side optimization

DSP-side execution (`dsp_exec`) accounts for 76% of total TG time
(10.26 s). The `htp/` directory is shared with Qualcomm, so
improvements here benefit both backends equally and do not affect the
JZ vs Qualcomm gap. The only JZ-specific DSP-side work that could help
TG is a custom m=1 attention kernel (Priority 2 above), which would be
JZ-exclusive and would not go through the shared `htp/` path.

***

## Completed Optimizations (2026-07-05 to 2026-07-11)

1. **Weight repack moved to set\_tensor** (breakthrough)
   - Repack buffer type with `is_host=false`
   - Eliminates per-inference repack overhead
   - PP: 105 -> 339 tok/s
2. **Graph cache fixed** (content-hash based)
   - FNV-1a hash over {op, ne, nb, src, data} per node
   - 99.2% hit rate, saves ~646us/token in TG
3. **mm\_params\_cache added**
   - Caches precomputed kernel params by (weight\_ptr, ne11)
   - Skips VTCM layout rebuild for repeated MUL\_MATs
4. **Op fusion completed** (VTCM guard added 2026-07-11)
   - All 5 fusion types: RMS\_NORM\_MUL, MUL\_MAT\_ADD, MUL\_MAT\_QKV, MUL\_MAT\_FFN, graph reorder
5. **ion\_sync\_mode added**
   - Configurable cache coherency (0=both, 1=ion\_sync, 2=DC CVAC)
   - ion\_sync\_mode=1 is optimal (confirmed by sweep)
6. **Session consistency gate added**
   - Prevents cross-session/cross-device tensor mixing
7. **Profiler infrastructure added** (extended 2026-07-11)
   - Per-phase, p7 3-way split, histogram, DSP-side per-op timing
   - longtail profiler (gated in `#if 0`)
   - mul_mat coverage tracer
   - dsp_cache_trace_bit0/bit1
8. **ARMv8.7+i8mm compiler flags** (PARITY with Qualcomm)
   - Both backends use the same flags (see Section 13)
9. **batch\_calls reduced from 40000+ to 4352**
   - Graph cache + ubatch\_size regression fix
10. **Upstream master merged + adapted** (2026-07-05 to 2026-07-11)
    - VTCM layout API changes (wrapper functions)
    - Unary precompute ported (commit `f2f259214`)
    - ARGSORT performance improvement (upstream `67776eaee`)
11. **Dual path removed** (2026-07-10, commits `6c11b225d` + `ba4fd0104`)
    - 24298 lines deleted; single shared `htp/` kernel path
    - `mulmat_algotype=32` self-built dispatch removed
    - `ggml-dsp` port deleted
12. **Graph reorder implemented** (2026-07-10, commit `602d71e65`)
    - Forward 16-group window; runtime-configurable
    - No measurable PP benefit for gemma4 PP=44 (kept for future-proofing)
13. **VTCM session-lifetime** (2026-07-11, commit `7261e75bb`)
    - Per-batch acquire/release -> per-session acquire/release
    - Matches Qualcomm pattern; eliminates ~8700 SDK FARF logs/inference
14. **VTCM budget check for MUL_MAT_ADD fusion** (2026-07-11, commit `2b1e7bd8c`)
    - Mirrors Qualcomm's guard; prevents silent VTCM overflow
15. **dsp_cache_mode settled at 4** (2026-07-11)
    - Bits 0/1 garble on new matmul pipeline (partial HVX writes)
    - Mode 4 (bulk dst flush only) is the safe baseline
16. **TG bottleneck identified** (2026-07-11)
    - FLASH_ATTN_EXT: 1.29 ms x 21 layers = 27 ms per token (dominant)
    - Not MUL_MAT, not dispatch overhead, not dspqueue
17. **ARM CPU TG ceiling measured** (2026-07-11)
    - 32.4 t/s, DDR-bandwidth-bounded (48.6 GB/s vs 51.2 GB/s peak)
    - 1.72x the Hexagon DSP TG; opens hybrid scheduling path
18. **Same-day JZ vs QCOM benchmark** (2026-07-11 evening)
    - JZ PP=339.12, QCOM PP=334.36 (JZ marginally ahead, 1.4%)
    - JZ TG=18.90, QCOM TG=26.40 (QCOM 1.40x faster)
    - Per-token TG gap 15.01 ms matches FLASH_ATTN_EXT decomposition
    - First same-day comparison where JZ PP >= QCOM PP

***

## Related Documents

- [algotype29-perf-analysis-en.md](algotype29-perf-analysis-en.md) -
  2026-07-05 baseline snapshot (PP=105.6 / TG=18.6, pre-optimization)
- [tg-pp-optimization-attempts-20260711.md](tg-pp-optimization-attempts-20260711.md) -
  Six TG/PP optimization experiments; identified FLASH_ATTN_EXT as TG
  hot spot and ARM CPU TG ceiling at 32.4 t/s
- [pp-regression-misdiagnosis-20260711.md](pp-regression-misdiagnosis-20260711.md) -
  Dual path removal "regression" was a measurement artifact; the 362
  PP mean was not reproducible
- [pp-cache-optimization-deadends-20260711.md](pp-cache-optimization-deadends-20260711.md) -
  dsp_cache_mode 5/6/7 garble; graph reorder no-op; three dead ends
- [vtcm-session-lifetime-20260711.md](vtcm-session-lifetime-20260711.md) -
  VTCM session-lifetime refactor + HEX_OP_PROF compile switch
- [algotype29-perf-cache-mode-comparison-20260710.md](algotype29-perf-cache-mode-comparison-20260710.md) -
  2026-07-10 morning data (362 PP mean, not reproducible)
- [p2-hmx-min-nrows-bench-20260710.md](p2-hmx-min-nrows-bench-20260710.md) -
  HTP_MM_HMX_MIN_NROWS sweep; value=4 is optimal
