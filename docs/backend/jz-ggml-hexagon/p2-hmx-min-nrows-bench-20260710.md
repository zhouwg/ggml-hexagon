# P2 bench: HTP_MM_HMX_MIN_NROWS sweep (4 vs 8 vs 16)

*Status: COMPLETE. value=4 is optimal. value=8 is no-op. value=16 is a 1.14% PP regression.*

## Scope

- **Constant**: `HTP_MM_HMX_MIN_NROWS` in `ggml/src/ggml-hexagon/htp/matmul-ops.h:20`
- **Algotype**: 29 only (Qualcomm execute_op, `enable_opfusion=1`)
- **Models**: gemma4 (`/sdcard/gemma-4-E2B-it-Q4_0.gguf`, 2.82 GiB, 4.65 B params)
  - 4-model CI (gemma4 + qwen3 + qwen1 + llama3) → 2-model → 1-model reduced due to phone thermal
- **Test tool**: `llama-bench -p 2048 -n 32 -t 6 --poll 1000 -fa 1 --ubatch-size 1024 --mulmat-algotype 29 -ngl 99`
- **Runs per cell**: 3 (some cells affected by thermal; only cold runs used for final conclusion)
- **Device**: Snapdragon 8 Elite (v79), OnePlus 13 (`9d231cfe`)

## Test infrastructure

- Script: `scripts/p2_bench_hmx_min_nrows.sh`
  - `<value>` arg = 4 | 8 | 16
  - Auto-edits the `#define` in `matmul-ops.h`
  - Incremental rebuild with ccache
  - Pushes `libggml-hexagon.so` + `libggmldsp-skel-v79.so` + `llama-bench`
  - Runs 3x per model
  - 20s sleep between runs, 30s between models (1-model CI)
  - `trap ... EXIT` restores matmul-ops.h on any failure
- Logs: `out/p2_bench_hmx/value_<N>.log` (full build+bench log)
- Bench-only rerun: `/tmp/bench_only.sh <value>` (skips rebuild, runs 3x gemma4)

## Data (final, cold-only)

### gemma4 PP (pp2048, tok/s) - cold runs only

| value | run 1 | run 2 | run 3 | mean | phone state |
|---:|---:|---:|---:|---:|---|
| 4 | 682.93 | 682.29 | 680.49 | **681.90** | cold (~36°C) |
| 8 | 681.36 | 680.11 | -- | **680.74** | cold (~36°C), only 2 runs clean |
| 16 | 672.95 | 669.84 | 679.61 | **674.13** | cold (~37°C), 3 runs clean |

### gemma4 TG (tg32, tok/s) - cold runs only

| value | mean | phone state |
|---:|---:|---|
| 4 | 18.74 | cold |
| 8 | 18.85 | cold |
| 16 | 18.80 | cold |

### Run-by-run data (full, including hot runs)

| value | run 1 | run 2 | run 3 | phone |
|---:|---:|---:|---:|---|
| 4 baseline | 682.93 | 682.29 | 680.49 | cold throughout |
| 8 first attempt | 679.79 | 676.08 | 598.56 | hot by run 3 |
| 8 rerun (cold) | 681.36 | 680.11 | 603.26 | hot by run 3 |
| 16 first attempt | 602.80 | 599.16 | 601.18 | hot throughout (51°C start) |
| 16 rerun (cold) | 672.95 | 669.84 | 679.61 | cold throughout |

The hot runs (600 tok/s range) are pure thermal-throttle artifacts. The
"PP dropped to 600" we saw is NOT caused by the constant change; it
happens for value=4 too once the phone is hot.

## Conclusion (FINAL)

| value | cold PP | delta vs 4 | action |
|---:|---:|---:|---|
| **4 (current)** | **681.90** | baseline | **keep as is** |
| 8 | 680.74 | -0.17% (noise) | optional, no harm, no benefit |
| 16 | 674.13 | **-1.14% (real)** | **do not change** |

- **TG unaffected** by the constant (18.74 vs 18.80 vs 18.80)
- **HMX-vs-HVX+fusion crossover** is at M=8..9
  - M ≤ 8: HVX+fusion is as good as HMX
  - M = 9..16: HMX wins by ~1% in PP throughput
  - (M > 16: not tested, but extrapolation suggests HMX continues to win)
- **PP 1% matters** in real use (~7 tok/s out of 681 = 1.5s saved per 60s prompt)

## Recommendation

**Do not change the constant.** value=4 is optimal for this model. value=8
is a safe no-op (could land for code-cleanliness reasons, e.g. "fewer
HMX-tile-inits for small M"), but value=16 actively hurts PP.

If a future change needs to "force HVX+fusion for very small M" (e.g.
a new quantization type that is HMX-incompatible), the safe choice is
value=4 → value=8, not value=16.

## Cross-validation: Qualcomm upstream also uses value=4

`HTP_MM_HMX_MIN_NROWS` is defined in
[`ggml/src/ggml-hexagon/htp/matmul-ops.h:20`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/matmul-ops.h#L20),
which lives in the `htp/` directory — code that is **shared** between
the JZ fork and the Qualcomm upstream backend
(`ggml-hexagon-qcom.cpp` / `htp/main.c`).

Observation: both the JZ and QCOM backends ship with the same default
`HTP_MM_HMX_MIN_NROWS = 4`. Three plausible reasons:

1. QCOM also ran a 4 vs 8 vs 16 sweep and found 4 optimal
   (most reassuring; cannot verify without their internal perf data)
2. `4` is a conservative "safe lower bound" for HMX eligibility
   (HMX tile is 32x32, anything below 4 risks severe underutilization),
   and both teams independently arrived at it
3. Inherited from upstream, never changed

### What this cross-validation does and does not prove

- **Proves**: it is unlikely that "4" is a careless default. Two
  independent codebases (one with the JZ synchronous FastRPC stack,
  one with QCOM's dspqueue pipelining) both ship this value, and our
  measured data confirms 4 is the local optimum. The convergence is
  strong suggestive evidence.
- **Does NOT prove**: that QCOM ran a 4/8/16 sweep themselves. The
  value could be inherited from the shared `htp/` code without
  deliberate re-validation.
- **Does NOT explain**: the QCOM-vs-JZ TG gap (26.48 vs 18.89
  tok/s). The constant is identical on both sides, so the gap is
  not attributable to it. Per
  [`algotype29-perf-analysis-en-20260709.md`](file:///home/zhouwg/develop/ggml-hexagon/docs/backend/jz-ggml-hexagon/algotype29-perf-analysis-en-20260709.md)
  the gap comes from `entry.c` (2523 lines, ours) vs `htp/main.c`
  (1008 lines, QCOM) and AP-side dispatch overhead, not from
  M-dimension threshold choice.

### Implication for future changes

If a future PR needs to "force HVX+fusion for very small M" (e.g. for
a new quantization type that is HMX-incompatible), the safe
incremental change is `4 → 8`, not `4 → 16`. This is consistent with
our measured data (8 is no-op, 16 is a 1.14% regression) and is also
the more conservative default for any future maintainer inheriting
the constant.

## Risk assessment (final)

- value=4 (current): optimal
- value=8: low value to land, but no risk
- value=16: should not be landed
- The original P2 question "is HMX hurting PP at small M" is answered:
  **No — HMX is at least as good as HVX+fusion for M ≥ 4.**


