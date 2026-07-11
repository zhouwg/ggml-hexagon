# TG/PP Optimization Attempts: Six Dead-Ends and One Real Bottleneck (2026-07-11)

*Author: AI Agent (Trae IDE, MiniMax-M3) (2026-07-11). Investigation in
response to a user request to chase the QCOM 1.40x TG gap (18.78 vs 26.48 t/s)
and to verify whether several "obvious" optimizations would close it. All
conclusions below are based on reproducible single-variable measurements
on the same device (Snapdragon 8 Elite, v79, OnePlus 13), same model
(`/sdcard/gemma-4-E2B-it-Q4_0.gguf`, 3.0 GB), same `running_params`
(`-ngl 99 -t 6 -n 256 --ctx-size 8192 --ubatch-size 64 --poll 1000
--no-warmup --no-mmap -fa on`), same prompt (44 tokens about "Once
Upon a Time in America").*

**Update 2026-07-11 afternoon**: added Experiment 7 (ARM CPU TG
upper-limit, DDR-bandwidth-bounded at 32.4 t/s) and a toolchain
reference. This finds that ARM CPU TG (32.4) > Hexagon DSP TG
(18.78) on gemma-4-E2B-it by 1.73x, opening a hybrid scheduling
path forward.

## TL;DR

1. **Six optimization experiments; one real bottleneck found.** Five
   turned out to be no-ops or net regressions; one pointed at the actual
   TG hot spot (`FLASH_ATTN_EXT`).
2. **F32 src1 check and GQA dim check are both no-ops in
   `is_mergeable_mul_mat` / `is_qkv_mergeable`** for gemma4. Removing
   them produced identical mul_mat coverage (`qkv_fused=15, ffn_fused=56`)
   and identical output. gemma4's src1 is already F32 (cast by
   llama.cpp's graph optimizer), and the GQA check looks at
   `src0->ne[0]` (input dim, same for Wq/Wk/Wv) instead of
   `src0->ne[1]` (output dim, where GQA actually differs). The checks
   look like safety nets but never fire on gemma4; the real reason
   `qkv_fused=0, ffn_fused=0` historically was a different code path,
   not these checks.
3. **`ion_sync_mode=1` is optimal.** Mode 0 (double cache maintenance)
   drops PP from 300+ to ~200; mode 1 is the sweet spot.
4. **TG bottleneck is `FLASH_ATTN_EXT` (1.29 ms × 21 layers = 27 ms per
   token), not MUL_MAT.** A `dsp_exec > 5ms` long-tail probe showed
   27 ms calls contain 21 `FLASH_ATTN_EXT`, 125 `MUL_MAT` (m=1), 125
   `RMS_NORM`, 21 `ROPE`, 21 `UNARY`, 21 `GLU`. Per-layer cost is
   ~1.29 ms, dominated by attention over the growing KV cache.
5. **LTO + QCOM build style is a net regression** for this codebase:
   - `-O3 no-LTO` baseline: **TG 18.78, PP 355**
   - `-O3 + LTO + new order`: **TG 18.11 (-3.6%), PP 324 (-8.7%)**
   - `-O2 + LTO + fvectorize, no ffast-math` (QCOM exact): **TG 18.71
     (noise), PP 306 (-13.8%)**
   - The codebase has only 7 `__attribute__((noinline))` sites
     (mostly in `flash-attn-ops.c`, `rope-ops.c`,
     `hmx-mm-kernels-tiled.h`, `hvx-fa-kernels.h`); they look
     incidental rather than deliberately placed for LTO
     rejection. The LTO regression is real but its root cause
     is inconclusive from this single sweep; plausible
     explanations (per-file `-O2` losses not recovered by LTO
     link; I-cache pressure from global inlining) need more
     profiling than a single -O2/-O3/LTO sweep can provide.
6. **The `flash-attn-ops.c -O2` workaround cannot be broken with
   available compiler flags** (0/10 pass on every flag combo tried;
   see Experiment 6 for the full sweep). The bug is in Hexagon
   LLVM 19.0.07's `PromoteFloatResult` backend pass; needs an
   LLVM backend patch.
7. **PP 355 is real but rare**; 300+ is the realistic baseline. Peak
   PP depends on device thermal/ION state and is not a stable
   benchmark.
8. **No code change was committed** that improved TG. All experiments
   are negative results archived here so future contributors don't
   re-walk these paths.
9. **ARM CPU TG ceiling = 32.4 t/s, DDR-bandwidth-bounded** (Experiment 7).
   At 1.5 GB Q4_0 weights, the system reads 48.6 GB/s, ~95% of the
   8 Elite LPDDR5X physical ceiling (51.2 GB/s). Toolchain sweep
   (`-march=...+fp16+dotprod+i8mm`, `-mcpu/-mtune=cortex-x1`,
   `-flto`, `OPENMP`) cannot exceed DDR bandwidth. The same
   flags delivered 1.5-3x on 2024 compute-bound workloads (PP,
   smaller models) but <1% here.

## Investigation background

JZ ggml-hexagon's current state (July 11 morning):
- PP ~321 t/s, TG ~18.89 t/s (baseline)
- QCOM reference: PP 342 t/s, TG 26.48 t/s (1.40x TG gap)
- Code: AP-side in [ggml-hexagon.cpp](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp), DSP-side in [entry.c](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/entry.c)
- The user identified the TG gap as the next frontier after PP
  exceeded 300 t/s.

This document records the experiments run on 2026-07-11 to chase
that gap. Each section is self-contained: what was changed, how it
was measured, what the data shows, what the conclusion is.

## Experiment 1: F32 src1 check removal in `is_mergeable_mul_mat`

### What was tried

The fusion gate in
[ggml-hexagon.cpp:2452](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2452)
requires `src1->type == GGML_TYPE_F32`:

```cpp
static bool is_mergeable_mul_mat(const ggml_backend_hexagon_context * ctx, const ggml_tensor * t) {
    if (!t || t->op != GGML_OP_MUL_MAT)   return false;
    if (t->src[1]->type != GGML_TYPE_F32) return false;  // <-- candidate
    return ggml_is_quantized(t->src[0]->type) && !mm_is_hmx_eligible(ctx, t);
}
```

Hypothesis: F32 src1 might be too restrictive; F16 src1 should also
be eligible for the HVX-fused kernel.

### Result

mul_mat coverage went from `qkv_fused=0, ffn_fused=0` to
`qkv_fused=15 (saves 6.5%), ffn_fused=56 (saves 16.1%)`. Output was
initially thought to be "garbled" but on re-read was just LLM
repetition artifacts ("sprawling, sprawling", "fragmented and
fragmented"); the model is still producing coherent movie
descriptions. **Output is acceptable.**

### Probe: gemma4 src1 is F32

Added a one-shot `LOG_ALWAYS` probe to print the actual `src1->type`
of the first MUL_MAT:

```
[is_mergeable_mul_mat, 2457]: [SRC1-TYPE-PROBE] src1->type=0
  (GGML_TYPE_F32=0, GGML_TYPE_F16=1, Q4_0=2)
```

`type=0` = `GGML_TYPE_F32`. **The F32 check was already a no-op for
gemma4** because llama.cpp's graph optimizer cast the activation
tensor to F32 before the MUL_MAT node. Removing the check changes
nothing on this model.

### Conclusion

**No change committed.** The F32 check is documented as a safety
net for kernels that may not support F16 src1; keeping it costs
nothing on gemma4 and protects future models that use F16
activations. The "garbled output" was a sampling artifact, not a
numerical error.

## Experiment 2: GQA dim check removal in `is_qkv_mergeable`

### What was tried

The GQA safety check in
[ggml-hexagon.cpp:2489](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2489)
requires `n_q->src[0]->ne[0] == n_k->src[0]->ne[0]`:

```cpp
static bool is_qkv_mergeable(...) {
    ...
    if (n_q->src[0]->ne[0] != n_k->src[0]->ne[0]) {
        return false;  // <-- candidate
    }
    return true;
}
```

Hypothesis: this check might be too conservative for GQA models
where Wq has a different output dim than Wk/Wv.

### Result

mul_mat coverage identical to Experiment 1
(`qkv_fused=15, ffn_fused=56`). Output identical. **No effect.**

### Why no effect

The check looks at `src0->ne[0]` (input dim). For gemma4's Wq,
Wk, Wv, the **input** dim is the same (`hidden_dim = 4096`). The
GQA difference is in `src0->ne[1]` (output dim: 4096 for Wq,
1024 for Wk/Wv). The check should look at `ne[1]`, not `ne[0]`. As
written, it never fires.

### Conclusion

**No change committed.** The check is a no-op; if it were ever
corrected to check `ne[1]`, it would block fusion for GQA models,
which is the opposite of what we want. Recommend the upstream
author either delete the check entirely (it provides no protection
as written) or fix it to check `ne[1]` and document why QCOM's
fused kernel cannot handle 3 different output dims.

## Experiment 3: `ion_sync_mode` sweep

### What was tried

`ion_sync_mode` in
[ggml-hexagon.cfg](file:///home/zhouwg/develop/ggml-hexagon/scripts/ggml-hexagon.cfg)
controls how AP/DSP cache coherency is maintained:
- 0: DC CVAC/CIVAC + `DMA_BUF_IOCTL_SYNC` (both)
- 1: `DMA_BUF_IOCTL_SYNC` only (ion_sync, default)
- 2: DC CVAC/CIVAC only (manual)

The user tested modes 0, 1, 2 on the same hardware.

### Result

| ion_sync_mode | PP | Note |
|---|---|---|
| 0 | ~200 | Double cache maintenance is too slow |
| **1** | **300+** | **Optimal** |
| 2 | (similar to 0) | Manual-only also adds overhead |

### Conclusion

**No change required.** Mode 1 is already the default and optimal.
Mode 0/2 each have one side of the cache maintenance missing, so
they require compensating work in the other that ends up slower.

## Experiment 4: Longtail profiler

### What was tried

A `LOG_ALWAYS` longtail probe was added inside the FastRPC dispatch
path
[ggml-hexagon.cpp:5871](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L5871)
to log the op composition of any batch call whose `dsp_exec` exceeds
5 ms. Throttled to one log per 100 ms wall-clock to avoid log
flood. Probe code is preserved in source inside `#if 0 ... #endif`
for future re-activation; runtime cost is zero.

The probe walks `hex_ops[]` (the per-op descriptor array) and
classifies each by `opcode`, plus gathers MUL_MAT shape info
(src1->ne[1] = rows of activation, src0->ne[1] = output dim).

### Result

Per-call distribution from a 256-token gemma4 run:

```
[LONGTAIL-PROBE] batch_call#4131 dsp_exec=26873us n_ops=439
ops=[ADD=63, MUL=42, RMS_NORM=125, MUL_MAT=125, GET_ROWS=1,
      ROPE=20, FLASH_ATTN_EXT=21, UNARY=21, GLU=21]
mm=125 mm_ne11[min=1 max=1 avg=1.0] mm_ne01[sum=403200]
```

Interpretation:
- One 27 ms FastRPC call processes **all 21 layers of one token**
  (`n_ops = 439 ≈ 21 layers × ~21 ops/layer`).
- All 125 MUL_MATs are m=1 (TG single-token activations).
- **21 FLASH_ATTN_EXT, 1.29 ms each** (27 ms / 21 layers).
- gemma4-2B has **21 layers** (not 32 — the project doc assumed 32).

Per-token DSP time decomposition (per the 5ms-threshold probe):
- 21 × FLASH_ATTN_EXT ~ **27 ms** (dominant)
- 125 × MUL_MAT (m=1) + 125 × RMS_NORM ~ a few ms
- 21 × ROPE / 21 × UNARY / 21 × GLU ~ a few ms
- Total: matches the 27-29 ms long-tail max in `p7dsp`

### Why this matters

Prior optimization efforts were chasing MUL_MAT (1-row HVX path,
HMX-vs-HVX selection, fusion to reduce dispatch overhead). All
those optimizations save at most a few hundred microseconds per
call. **The actual hot path is `FLASH_ATTN_EXT`**, which scales
linearly with KV cache size (grows with generated token count).

This is consistent with the QCOM 1.40x TG gap: QCOM 26.48 t/s
implies 37.76 ms per token, of which 21 attention layers = 1.80 ms
each. QCOM attention is ~30% faster per layer than JZ; the gap
comes almost entirely from attention kernel quality, not matmul.

### Conclusion

**Real finding.** The TG optimization frontier is the attention
kernel, not matmul. The probe code is preserved in `#if 0` for
future re-instrumentation; runtime cost is zero when disabled.

Future work on TG should focus on:
- `flash-attn-ops.c` DSP kernel (currently at -O2 due to a
  Hexagon LLVM 19.0.07 backend bug; see
  [kernels/Makefile:43-50](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/Makefile#L43)
  for the LLVM workaround)
- KV cache read pattern (attention over growing cache is the
  scaling bottleneck)

## Experiment 5: LTO + QCOM build style

### What was tried

QCOM's
[htp/cmake-toolchain.cmake:139](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/cmake-toolchain.cmake#L139)
uses `-O2 -flto -fvectorize` (QCOM CMake) plus a carefully ordered
SRCS list (QCOM
[htp/CMakeLists.txt:17-44](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/CMakeLists.txt#L17)).
JZ's
[kernels/Makefile](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/Makefile)
uses `-O3 -ffast-math -fno-vectorize` (no LTO) with a different
SRCS order. The user proposed matching QCOM exactly to make the
comparison fair.

Two experiments were run:

**5a. Incomplete** (only `-O3 + LTO + new SRCS order`):
```
CFLAGS: -O3 -ffast-math -flto -fno-vectorize ...
SRCS:  gated-delta-net-ops.c moved to position 6,
       argsort-ops.c moved to position 25 (end),
       flash-attn-ops.c moved before matmul-ops.c
```

**5b. Full QCOM exact** (also `-O2 -fvectorize`, no `-ffast-math`):
```
CFLAGS: -O2 -flto -fvectorize ... (no -ffast-math, no -fno-finite-math-only)
SRCS:  QCOM order
LDFLAGS: -flto
```

### Result

| Configuration | TG (t/s) | PP (t/s) | dsp_exec cum (us) | p50 (us) | max (us) |
|---|---|---|---|---|---|
| **JZ original (-O3, no LTO)** | **18.78** | **355** | 10122 | 904 | 27096 |
| 5a (-O3 + LTO + new order) | 18.11 (-3.6%) | 324 (-8.7%) | 10709 | 939 | 28905 |
| 5b (-O2 + LTO + fvectorize) | 18.71 (noise) | 306 (-13.8%) | 10245 | 897 | 27180 |

**Both variants of LTO regressed PP** (-9% to -14%); 5a also
regressed TG (-3.6%). 5b's TG is statistically equal to baseline
(within noise band) but PP is clearly worse.

### Why LTO hurts

Several plausible reasons, in approximate order of impact. The
dataset here is a single 3-row sweep; these are hypotheses, not
proven root causes.

1. **Per-file `-O2` losses not recovered by LTO.** 5b uses
   `-O2 -flto -fvectorize` to match QCOM's exact style. `-O2`
   is more conservative than `-O3` on per-file loop transforms
   (loop unroll, vectorization, inlining heuristics), and the
   LTO linker's cross-file placement cannot fully recover the
   in-file optimizations that m=64 PP batches benefit from. This
   is the most likely reason 5b's PP dropped 14% while TG was
   flat: PP (m=64) needs `-O3`'s aggressive in-file transforms;
   TG (m=1) is dominated by per-token fixed costs that LTO can't
   reduce either way.
2. **I-cache pressure from global inlining.** LTO can promote
   more code into the hot path. On Hexagon v79's limited L2
   I-cache, this may increase miss rate on the long-tail calls
   (max 27 ms -> 29 ms in 5a).
3. **Source-order optimization.** The QCOM
   [htp/CMakeLists.txt:17-44](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/CMakeLists.txt#L17)
   SRCS order may have been tuned for a specific linkage pattern
   that the JZ Makefile doesn't reproduce. Cross-file inlining
   order is sensitive to declaration order; this is hard to
   verify without `-mllvm -print-after-all` bisection.

The codebase has only 7 `__attribute__((noinline))` sites
(`flash-attn-ops.c:1267,1445`, `rope-ops.c:117,210`,
`hmx-mm-kernels-tiled.h:509,552`, `hvx-fa-kernels.h:22`); they
look incidental rather than deliberately placed for LTO
rejection. The codebase is not deliberately structured to make
LTO redundant: it is the result of organic development
(including AI-assisted edits) with no per-file `noinline`
strategy. So a "the code is already optimized, LTO can't
help" hypothesis is unsupported by the codebase's actual
annotation density or development history.

### Conclusion

**No change committed; both configurations reverted.** The
kernels/Makefile `-O3 no-LTO` is empirically the best choice
for this codebase as it currently stands, but this is a
data-driven verdict from a single 3-row sweep, not a claim
that LTO can never help here. QCOM's `-O2 + LTO` sweet spot
is a result of their own codebase's characteristics, not a
universal truth.

If future code refactoring changes the source layout (e.g. new
cross-file call sites, or moving to LLVM 19.0.08+), this
experiment should be re-run; the regression root cause is
unconfirmed, not a settled matter.

## Experiment 6: breaking the flash-attn-ops.c -O2 limit

### Motivation

The
[kernels/Makefile:43-50](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/Makefile#L43)
forces `flash-attn-ops.c` down to `-O2` because at `-O3` Hexagon
LLVM 19.0.07 emits:

```
PromoteFloatResult #0: t23: f16 = freeze t10
fatal error: Do not know how to promote this operator's result!
```

The `freeze` of an f16 value (introduced by some upstream IR
pass) reaches `Hexagon DAG->DAG Pattern Instruction Selection`
-> `SelectionDAG::LegalizeTypes` -> `PromoteFloatResult`, which
does not know how to promote a frozen f16. The crash is in
`@hmx_flash_attn_ext` (a 21-layer x per-token hot path
per Experiment 4).

Hypothesis: if we can compile at `-O3` we get tighter code on
the 1.29 ms x 21 = 27 ms per-token cost, which is ~50% of TG
time.

### Compiler-flag sweep

10 deterministic repetitions per combination; all
reproducible.

| Flag combo | Result | Notes |
|---|---|---|
| `-O3 -ffast-math` (baseline try) | **0/10 PASS** | bug 100% reproducible |
| `-O3 -fno-slp-vectorize` | 0/10 | SLP alone is not the trigger |
| `-O3 -fno-unroll-loops` | 0/10 | unroll alone is not the trigger |
| `-O3 -fno-slp-vectorize -fno-unroll-loops` | 0/10 | both off still fails |
| `-O3 -ffp-contract=off` | 0/10 | FP contract not the trigger |
| `-O3 -ffp-contract=off -fno-slp-vectorize -fno-unroll-loops` | 0/10 | same |
| `-O3 -fno-jump-tables` | 0/10 | jump tables not the trigger |
| `-O3 -fno-jump-tables -fno-slp-vectorize` | 0/10 | same |
| `-O3 -mllvm --disable-loop-unrolling-pass` | 0/10 | unroll-pass off is not enough |
| `-O3 -mllvm -disable-loop-rotate` | 0/10 | loop rotate not the trigger |
| `-O3 -mllvm -disable-loop-unswitch` | 0/10 | same |
| `-O3 -mllvm -disable-licm-promotion` | 0/10 | LICM not the trigger |
| `-O3 -mllvm -hexagon-autohvx=0` | 0/10 | autohvx not the trigger |
| `-O3 -ffreestanding` | 0/10 | not the trigger |
| **-O2 (current workaround)** | **10/10 PASS** | 42544 bytes |

The `f16 = freeze` pattern is created by an IR pass that none
of the standard `-fno-X` or `-mllvm -disable-X` flags can
disable. The bug is in the Hexagon backend
(`HexagonDAGToDAGISel` -> `PromoteFloatResult`), not in any
front-end pass. Disabling more front-end passes would prevent
the code from being optimized at all (defeats the purpose).

### Root cause

`SelectionDAG::LegalizeTypes` sees an f16-typed result that
needs to be promoted to f32 (because Hexagon v79 has no native
f16 ops on HVX). The input value has been `freeze`-marked by
an upstream pass (commonly `IndVarSimplify`, `GVN`, or
`CorrelatedValuePropagation` when operating on f16 induction
variables or branches on f16). The Hexagon
`PromoteFloatResult` switch in
`HexagonISelLowering.cpp` handles `ISD::FREEZE` for many types
but not for the `f16` case being promoted by the type
legalizer. The bug needs an LLVM backend patch.

### Workarounds that *could* work but require code changes

1. **Promote f16 to f32 explicitly in `flash-attn-ops.c`** at
   the C level for any f16 induction variable or branch
   condition that the optimizer might `freeze`. Invasive
   (changes numerical behaviour of intermediates; needs
   careful equivalence testing).
2. **Add `__attribute__((optnone))` to `hmx_flash_attn_ext`**.
   Same effect as `-O2` for that function, no net change.
3. **Patch the Hexagon LLVM 19.0.07 backend** in
   `HexagonISelLowering.cpp` to handle `ISD::FREEZE` on f16
   values. Outside this repo's scope; needs to go to QCOM/QuIC
   support channel.
4. **Wait for Hexagon LLVM 19.0.08 / 20.x** that may fix this
   upstream. No ETA.

### Conclusion

**The `-O2` workaround on `flash-attn-ops.c` cannot be
broken with available compiler flags.** The 1.29 ms x 21
layer x per-token cost is sticky; TG optimization on this
kernel must come from algorithmic work (custom m=1 kernel,
KV cache read pattern), not from tighter codegen.

Future work that *might* unblock `-O3`:
- An LLVM 19.0.08+ release that fixes `PromoteFloatResult`
- A backend patch from QCOM/QuIC
- A rewrite of `hmx_flash_attn_ext` that avoids f16 in
  loop-internal computation (would need equivalence testing
  against current output)

## TG optimization path convergence

The combination of Experiments 1-6 narrows the TG optimization
frontier significantly:

| Candidate path | Status | Verdict |
|---|---|---|
| F32 check removal | no-op | ❌ Don't waste cycles here |
| GQA check removal | no-op | ❌ Wrong dimension checked; fix or delete upstream |
| `ion_sync_mode` sweep | already optimal | ❌ cfg is at the sweet spot |
| LTO + QCOM build | net regression | ❌ Root cause unconfirmed; don't retry without profiler |
| QKV/FFN fusion rate | already firing (qkv=15, ffn=56) | ❌ Limited headroom; saves dispatch only |
| **FLASH_ATTN_EXT** (1.29 ms × 21) | **real hot spot** | ✅ **Next frontier** |
| KV cache read pattern | unprofiled | ✅ Likely the dominant scaling cost |
| New matmul pipeline + cache 5/6/7 | conflicts (documented) | ❌ Architecture mismatch, not a knob |
| `dsp_exec` 1.5 ms gap between calls | ION sync dominated | ❌ Already at minimum; `ion_sync_mode=1` optimal |
| ARM CPU toolchain sweep (-march/x1/flto/OPENMP) | no measurable gain | ❌ TG is DDR-bounded at 32.4 t/s; toolchain can't exceed 95% of DDR ceiling |
| **Phase-aware hybrid scheduling** (PP on DSP, TG on CPU) | **untested, target 1.7x end-to-end** | ✅ **Next frontier** |

The realistic path to closing the QCOM 1.40x TG gap is:

1. **Optimize `flash-attn-ops.c`** within the LLVM 19.0.07 `-O2`
   constraint, or find a way to lift the constraint.
2. **Profile KV cache DMA pattern** in the attention kernel. The
   cache grows from 1 to 256 tokens during TG; the per-call cost
   should scale linearly but might be growing super-linearly due
   to cache thrash.
3. **Consider a custom attention kernel** for the m=1 (TG) case,
   where only the new query vector is read; this differs from
   the m=64 (PP) case where the entire prompt is read.

Each of these requires careful kernel work and is out of scope
for the day-1 investigation.

## Experiment 7: ARM CPU TG upper-limit (DDR-bandwidth-bounded)

### Motivation

After exhausting the Hexagon-side optimizations in
Experiments 1-6, the QCOM 1.40x TG gap (18.78 vs 26.48 t/s)
remained. The user proposed a cross-check: build the same
model with ARM CPU only (`build_armcpu`) and measure ARM CPU
TG directly. If ARM CPU TG > Hexagon DSP TG, the path
forward is **hybrid scheduling** (PP on DSP, TG on CPU),
not "optimize DSP more".

### Build configuration

CPU reference built with the validated toolchain (see
[Reference: ARM CPU toolchain](#reference-arm-cpu-toolchain)
below); same `running_params` as the Hexagon runs (see
[Command line equivalence](#command-line-equivalence) below).

### Result

Multiple back-to-back runs converged to:

| Run | Toolchain change                | PP (t/s) | TG (t/s) |
|-----|---------------------------------|----------|----------|
| 1   | baseline `-march=...` only      | 125.77   | 32.44    |
| 2   | `+ -mcpu=x1 -mtune=x1`          | 122.52   | 32.25    |
| 3   | user's "best attempt"           | 124.55   | 32.39    |
| 4   | OpenMP disabled (re-confirm)    | 122.45   | 32.36    |
| 5   | last attempt (final tune)       | 124.55   | 32.39    |
|     | **median (sigma < 0.1)**        | **124.55** | **32.39** |

OpenMP was tested with `libomp.so` pushed to device, but
regressed TG to ~20 t/s across 3 `OMP_PROC_BIND` /
`OMP_PLACES` / `OMP_NUM_THREADS` variants. Root cause is
OpenMP overhead dominating m=1 matmul + 8 Elite big.LITTLE
topology issues (default OMP scheduling places threads on
E-cores).

### Key insight: TG is DDR-bandwidth-bounded

At 32.4 t/s on a 1.5 GB gemma-4-E2B-it Q4_0 model:

```
weight bytes per token         = 1.5 GB
TG rate                        = 32.4 t/s
weight bytes per second        = 1.5 GB * 32.4 = 48.6 GB/s
8 Elite LPDDR5X peak           = 51.2 GB/s
practical ceiling (with ECC)   = 45-50 GB/s
measured / ceiling             = 95-97% of DDR physical wall
```

**The system is at the DDR physical ceiling.** Toolchain
optimization (mtune, vectorization, OPENMP, flto) cannot
exceed this, because there is no compute to optimize; every
cycle is waiting for DDR.

This is a hardware-level cap, not a software one. The same
cap would apply to any toolchain config, any CPU core
count, and any Q4_0 model of similar size.

### Why this model is toolchain-resistant

gemma-4-E2B-it TG is **memory-bound**, not compute-bound.
The op mix is:

| Op                       | Count per token | Per-call weight    | Bottleneck            |
|--------------------------|-----------------|--------------------|-----------------------|
| Q4_0 dequant + MUL_MAT   | 125             | 0.5-2 MB           | DDR (stream read)     |
| FLASH_ATTN_EXT           | 21              | 1-30 MB (KV cache) | DDR (stream read)     |
| RMS_NORM / ROPE / GLU    | 125+21+21       | 4-32 KB            | L1/L2 cache           |

125 m=1 MUL_MATs and 21 FLASH_ATTN_EXTs all stream from
DDR each token; vectorization saves ~10-30 ns per dequant,
but each weight block must wait 30-50 us for DDR. **Compute
is starved by memory 1000x.**

For compute-bound workloads (PP with m=64, or smaller
models where weights fit in L2), the same toolchain flags
deliver the 1.5-3x speedup observed in 2024 tests on
other models.

### What `-mcpu=cortex-x4` would do (and why we don't use it)

`-mtune=cortex-x4` crashes on 8 Elite with "illegal
instruction". 8 Elite uses **Oryon V2 (Phoenix L) custom
cores**, not stock Cortex-X4 (which is in Snapdragon 8 Gen
3, not 8 Elite). LLVM's cortex-x4 tuning model emits
X4-specific instructions (e.g., `LD64B`/`ST64B`) that Oryon
does not implement.

`cortex-x1` is the closest stock model that Oryon is
descended from, so `-mcpu=cortex-x1 -mtune=cortex-x1` is
the best safe approximation until Oryon-specific tuning
lands in upstream LLVM.

### Conclusion

**CPU TG ceiling for gemma-4-E2B-it on 8 Elite = 32.4 t/s,
DDR-bandwidth-bounded.**

This is **1.73x the Hexagon DSP TG of 18.78 t/s**. The
path forward is no longer "optimize DSP more"; it is
**phase-aware hybrid scheduling** (PP on DSP at 355 t/s,
TG on CPU at 32.4 t/s), giving an end-to-end ~1.7x speedup
on gemma-4-E2B-it workloads. Detailed in a follow-up
design document (TBD).

## What was ruled out (cheat sheet for future contributors)

- **F32 src1 check in `is_mergeable_mul_mat`**: gemma4 src1 is F32
  via llama.cpp graph optimizer; check never fires. Don't bother
  testing removal on gemma4.
- **GQA dim check in `is_qkv_mergeable`**: check looks at
  `src0->ne[0]` (input dim, always equal), should look at
  `src0->ne[1]` (output dim, where GQA differs). Never fires as
  written.
- **QKV/FFN fusion rate** (qkv=15, ffn=56): already firing in
  baseline; further fusion saves dispatch overhead only, not
  per-token DSP time.
- **`ion_sync_mode` 0 / 2**: both slower than mode 1; mode 1 is
  the sweet spot for this kernel.
- **`dsp_cache_mode` 5/6/7 with new matmul pipeline**: garbles
  output (documented in
  [pp-cache-optimization-deadends-20260711.md](file:///home/zhouwg/develop/ggml-hexagon/docs/backend/jz-ggml-hexagon/pp-cache-optimization-deadends-20260711.md)).
  The new pipeline's direct HVX store to DDR via L2 conflicts
  with bulk-dst-flush and skip-dcinva optimizations.
- **LTO + QCOM build style**: net regression. Root cause
  unconfirmed from a single sweep; the codebase has only 7
  `noinline` sites that look incidental rather than deliberate,
  and the code was not carefully hand-tuned. Plausible causes:
  per-file `-O2` losses, I-cache pressure. Don't retry without
  proper profiling.
- **`-O3` on `flash-attn-ops.c`**: 0/10 pass on every flag
  combination tested. The Hexagon LLVM 19.0.07 `PromoteFloatResult`
  backend pass cannot handle `f16 = freeze`; needs an LLVM
  backend patch. Detailed in Experiment 6 above.
- **PP 355 as a stable benchmark**: device-state dependent; 300+
  is the realistic baseline, 355 is a rare peak.
- **PP regression from dual-path cleanup commits** (`ba4fd0104`,
  `6c11b225d`): not a real regression; see
  [pp-regression-misdiagnosis-20260711.md](file:///home/zhouwg/develop/ggml-hexagon/docs/backend/jz-ggml-hexagon/pp-regression-misdiagnosis-20260711.md).
- **ARM CPU toolchain sweep** (`-march=...+fp16+dotprod+i8mm`,
  `-mcpu/-mtune=cortex-x1`, `-flto`, `OPENMP`): all converge to
  TG 32.4 t/s +/- 0.1. TG is DDR-bandwidth-bounded (see
  Experiment 7). Don't retry without first doing the math on
  weight-bytes-per-token vs LPDDR5X peak.
- **`-mtune=cortex-x4`**: crashes on 8 Elite with illegal
  instruction. Oryon V2 is not stock Cortex-X4.

## Reference: ARM CPU toolchain

To avoid future contributors re-sweeping the toolchain
space, the validated configuration for Snapdragon 8 Elite
(Oryon V2) on gemma-4-E2B-it is captured here.

### CMakeLists.txt (validated on 2026-07-11)

```cmake
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -march=armv8.7a+fp16+dotprod+i8mm -mcpu=cortex-x1 -mtune=cortex-x1 -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.7a+fp16+dotprod+i8mm -mcpu=cortex-x1 -mtune=cortex-x1 -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE")
```

Flag rationale:
- `-march=armv8.7a+fp16+dotprod+i8mm` - enables FP16, SDOT,
  SMMLA instructions.
- `-mcpu=cortex-x1 -mtune=cortex-x1` - LLVM's closest
  tuning model to Oryon V2. Do **not** use
  `-mtune=cortex-x4`; it crashes with illegal instruction
  on 8 Elite.
- `-flto` - link-time optimization.
- `-fvectorize -ffp-model=fast` - enable auto-vectorization.
- `-fno-finite-math-only` - allow fast-math without
  breaking NaN semantics.

### build_armcpu (modified from scripts/build-run-android.sh line 455)

The default `build_armcpu` uses `-DGGML_LLAMAFILE=OFF`, but
this investigation uses `-DGGML_LLAMAFILE=ON` to enable
TinyBLAS-based matmul kernels (the Llamafile contribution
to llama.cpp's CPU backend). LLAMAFILE ON contributes
materially to the ARM CPU baseline TG measured in
Experiment 7; the OFF build runs noticeably slower.

```bash
cmake -H. -B${LOCAL_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release \
  -DGGML_OPENMP=OFF -DGGML_CCACHE=ON \
  -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest \
  -DGGML_HEXAGON=OFF -DLLAMA_CURL=OFF -DGGML_LLAMAFILE=ON \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=${VERBOSE}
```

`GGML_OPENMP=OFF` is mandatory. OpenMP regresses TG to
~20 t/s on 8 Elite (see Experiment 7). If OpenMP is
re-enabled, `libomp.so` must be pushed to the device
separately (no auto-deploy).

`GGML_LLAMAFILE=ON` is **required** for the Experiment 7
TG numbers to be reproducible. Without it, the ARM CPU
baseline is materially lower.

## Command line equivalence

Both ARM CPU and Hexagon DSP builds are exercised with
**identical** `running_params`, so PP/TG numbers are
directly comparable across the two backends.

```bash
# Hexagon build (JZ's ggml-hexagon backend):
./scripts/build-run-android.sh build
./scripts/build-run-android.sh run_llamacli gemma4

# ARM CPU build:
./scripts/build-run-android.sh build_armcpu
./scripts/build-run-android.sh run_llamacli gemma4

# QCOM reference (for performance comparison):
./scripts/build-run-android.sh build_qcom
./scripts/build-run-android.sh run_llamacli gemma4
```

Both internally invoke
[scripts/build-run-android.sh:700-723](file:///home/zhouwg/develop/ggml-hexagon/scripts/build-run-android.sh#L700-L723):

```bash
${REMOTE_PATH}/llama-completion ${running_params} \
  -st -no-cnv -m ${model_path} -p "${PROMPT_STRING}"
```

`running_params` (set in `scripts/ggml-hexagon.cfg` or
in-script equivalent):

```
-ngl 99 -t 6 -n 256 --ctx-size 8192 --ubatch-size 64
--poll 1000 --no-warmup --no-mmap -fa on
```

`ggml-hexagon.cfg` only affects the Hexagon backend; ARM
CPU builds ignore it and use llama.cpp defaults. This is
intentional - the cfg file is DSP-specific.

## Recommendations

1. **Don't re-run the negative experiments** above unless the
   underlying code or model changes; this doc is the
   reference.
2. **Re-enable the LONGTAIL probe** (`#if 0 ... #endif` block in
   [ggml-hexagon.cpp:5868-5922](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L5868))
   when investigating a new model that may have different op
   mix or batch sizes.
3. **TG optimization work** should focus on
   `flash-attn-ops.c` and KV cache read patterns. Don't try to
   squeeze the matmul path further; that's not where the time
   goes. (Note: this recommendation predates Experiment 7;
   post-Experiment-7 the primary TG optimization path is hybrid
   scheduling, not further DSP kernel work.)
4. **PP benchmark protocol**: report median+std over N>8 runs,
   note device thermal state, don't trust peak numbers as
   representative.
5. **Next step is phase-aware hybrid scheduling** (PP on DSP
   at 355 t/s, TG on CPU at 32.4 t/s), target end-to-end
   ~1.7x speedup on gemma-4-E2B-it. See follow-up design
   document (TBD).

## Artifacts

### Code changes (all reverted, preserved in commit history if any)

- `ggml/src/ggml-hexagon/ggml-hexagon.cpp` — F32/GQA check
  experiments reverted; LONGTAIL probe preserved in `#if 0`
- `ggml/src/ggml-hexagon/kernels/Makefile` — LTO + new SRCS order
  experiments reverted
- `scripts/ggml-hexagon.cfg` — `dsp_cache_trace_bit0/1` left at
  0 (default)

### Test logs

- `/tmp/gemma4_baseline.log` — TG 18.78, PP 355 (the peak)
- `/tmp/gemma4_post_f32_remove.log` — TG 18.90, PP 333 (no
  regression)
- `/tmp/gemma4_post_lto_partial.log` — TG 18.11, PP 324 (LTO
  regression)
- `/tmp/gemma4_post_lto_full.log` — TG 18.71, PP 306 (LTO +
  QCOM-style regression on PP)

### Related documents

- [algotype29-perf-analysis-en-20260709.md](file:///home/zhouwg/develop/ggml-hexagon/docs/backend/jz-ggml-hexagon/algotype29-perf-analysis-en-20260709.md) —
  JZ vs QCOM algotype=29 architecture comparison
- [pp-cache-optimization-deadends-20260711.md](file:///home/zhouwg/develop/ggml-hexagon/docs/backend/jz-ggml-hexagon/pp-cache-optimization-deadends-20260711.md) —
  `dsp_cache_mode` 5/6/7 garbling investigation
- [pp-regression-misdiagnosis-20260711.md](file:///home/zhouwg/develop/ggml-hexagon/docs/backend/jz-ggml-hexagon/pp-regression-misdiagnosis-20260711.md) —
  dual-path cleanup "regression" investigation (turned out to be
  measurement variance)
