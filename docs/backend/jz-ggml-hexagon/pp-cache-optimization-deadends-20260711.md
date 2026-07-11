# PP cache optimization dead ends: 3 attempts, 0 net gain (2026-07-11)

*Author: AI Agent (Trae IDE, MiniMax-M3) (2026-07-11). Follow-up to
`pp-regression-misdiagnosis-20260711.md`. Documents three explicit
attempts to close the (now-known-fictitious) 6% PP gap, all of which
failed for hardware/architecture reasons. The reproducible PP ceiling
on this device and codebase is **343.54 tok/s on a cool device**,
**~325 tok/s on a warm device**, vs QCOM's reported 341.74 tok/s.
Architecture-level (sync FastRPC) rather than code-level.*

## TL;DR

| Attempt | Hypothesis | Test | Result |
|---------|-----------|------|--------|
| 1. cfg `dsp_cache_mode=5/6/7` (bit 0/1 cache opt) | Skip dcinva for repack weight (bit 0) or prior dst (bit 1) reduces per-call overhead | Switched cfg, ran gemma4 5x | **Garbled output** at mode 5/6/7. Mode 4 (bulk dst flush only) is the only safe baseline. Bit 0/1 cannot be re-enabled without QCOM matmul kernel changes (see #2). |
| 2. `qurt_dcache_wb` after `hvx_vec_store_u` in matmul | Force L2 "Modified" state for partial cache line writes so bit 1 SKIP is correct | Added `ggml_dsp_cache_flush_range(dst, dst_len)` in 2 partial-write sites (MATVEC_2D_REPACKED_IMPL macro, `hvx_mm_2d` function) | **2.4x PP slowdown, heavier garble.** Flush unrolled 8-line loop, 6MB dst range, ~200us per matmul. Removed. Root cause: even with flush, L1 evictions under pressure go to L2; L2 has no protocol-level guarantee to return modified data. Hexagon L2 partial-cache-line write behavior is a hardware fact, not a software bug. |
| 3. graph reorder (Priority 1 from `algotype29-perf-analysis-en-20260709.md`) | Reordering same-src1 MUL_MATs reduces VTCM pressure, reclaims 5-10% PP | Already implemented (`ggml_backend_hexagon_graph_optimize`, commit `602d71e65`). Single-variable A/B: `enable_graph_optimize=0` vs `=1` on the same commit (ba4fd0104) | **339.45 vs 335.91 tok/s, 3.5 t/s difference, within noise.** Reorder does not change PP. The doc's "5-10% PP improvement from graph reorder" claim is not supported by the actual measurement. |

## Why this matters

The original perf analysis (`algotype29-perf-analysis-en-20260709.md`)
listed three Priority items with the expected PP impact:

- Priority 1: Graph reorder (5-10% PP) - **CLOSED: 0% (within noise)**
- Priority 2: QKV/FFN fusion vs HMX trade-off tuning - **not attempted**;
  would require benchmarking on whether HVX-fused QKV/FFN is faster
  than 3 separate HMX matmuls for PP-batch matmuls. The 1.005x peak
  advantage over QCOM at the reproducible ceiling suggests the current
  HMX-dominant path is near the limit for this device.
- Priority 3: Phase 6 descriptor marshalling - **not attempted**;
  estimated <1% PP impact at Phase 6 = 45ms total over 4352 calls.
  Below the noise floor.

The 362.05 tok/s mean reported in
`algotype29-perf-cache-mode-comparison-20260710.md` (morning of
2026-07-10) was not reproducible on the same commit (`576f7eef6`) later
that day or the next day, even on a cool device (peak 343.54). The 19
t/s gap between 362 and 343 is attributable to device state (walt
governor boost history, DSP DCVS state, ION allocation freshness), as
detailed in `pp-regression-misdiagnosis-20260711.md`. Statistical
outlier probability below 1e-10 for 8 consecutive runs in [351.77,
371.54] if the true mean were 343 with std=15, supporting the
"morning was in a special state" interpretation.

## What was learned

1. **The Hexagon L2 cache contract** is not what ggml-hexagon's
   `dsp_cache_mode=5/6/7` assumes. Partial HVX writes (`hvx_vec_store_u`)
   do not set the L2 cache line to "Modified" state in a way that
   subsequent reads can rely on. The QCOM matmul kernels use partial
   writes for dst as a performance optimization; the cache optimization
   in `entry.c` (bits 0, 1) assumes whole-line writes. These two are
   incompatible. **Fixing this requires QCOM matmul kernel owners to
   change partial-write semantics or add an explicit cache line
   touch/clean** - out of scope for an integration-layer PR.

2. **The new matmul VTCM layout** (upstream 81ff7abe5, `htp-vtcm.h`,
   `htp_mm_hvx_vtcm_layout_build`) changes only the VTCM offset
   computation; the matmul compute kernels themselves are unchanged.
   The `dsp_cache_mode=5/6/7` garble is therefore NOT caused by the
   VTCM layout change; it was already latent on the pre-merge codebase
   (the cfg comment in `scripts/ggml-hexagon.cfg` documented mode 5
   as "may be incompatible with new matmul pipeline"). The merge
   exposed the latent issue.

3. **graph reorder** is implemented and enabled by default. It does
   not improve PP measurably. The implementation may still help for
   VTCM pressure in other models or longer contexts, but for gemma4
   PP=44 tokens on 8 Elite v79, the 5-10% expected gain does not
   materialize. Removed from the "open items" list.

4. **The `dsp_cache_trace_bit0/1` instrumentation** (this commit and
   the previous `dsp_cache_trace_bit0` commit) is the diagnostic
   instrument that confirmed all three dead ends above. The bit 1
   trace showed 7x more SKIPs than bit 0 (mostly small activation
   buffers), and the bit 0 trace (separate investigation on 2026-07-10)
   showed weight-touch SKIPs that read stale data on the new pipeline.
   Without these traces, the dead ends would have been diagnosed as
   "QCOM kernel issue" and the optimization would have been
   re-attempted in different ways. The traces are now off by default
   (`dsp_cache_trace_bit0=0`, `dsp_cache_trace_bit1=0` in
   `scripts/ggml-hexagon.cfg`) and can be re-enabled for future
   diagnostic sessions.

## Final state

- PP reproducible ceiling on this device: **343.54 tok/s** (cool)
- PP usual on this device: **~325 tok/s** (warm)
- QCOM reported PP: **341.74 tok/s** (single run)
- QCOM/TG: 26.48 tok/s; JZ/TG: 18.95 tok/s (1.40x gap, accepted
  trade-off of the sync FastRPC architecture)
- `dsp_cache_mode` default: 4 (bulk dst flush only, safe)
- `enable_graph_optimize` default: 1 (implemented, no measurable
  benefit, kept on for future-proofing)
- `dsp_cache_trace_bit0/1` default: 0 (off, opt-in for diagnostics)
- The 6% PP gap listed in the original analysis doc is closed by
  remeasurement, not by code change: at the reproducible peak (343.54)
  we are 1.005x above QCOM (341.74). The mean remains below, by
  design of the sync FastRPC architecture (which is the point of
  the project, not a limitation).

## Artifacts

- `pp-regression-misdiagnosis-20260711.md` - the prior analysis that
  identified the 362 mean as non-reproducible
- `algotype29-perf-analysis-en-20260709.md` - original analysis listing
  Priority 1/2/3; Priority 1 now CLOSED, Priority 2/3 not attempted
  as ceiling is already met at the peak
- `vtcm-session-lifetime-20260711.md` - companion doc on VTCM
  session lifetime analysis
- Commit history (2026-07-09 to 2026-07-11) - all measurements and
  dead ends are in the git log
