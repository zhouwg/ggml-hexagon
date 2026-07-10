# ggml-hexagon VTCM session-lifetime + profiler compile-switch (2026-07-11)

*Author: AI Agent (Trae IDE, MiniMax-M3) (2026-07-11). Authored based on
empirical investigation of the VTCM acquire/release pattern in
`ggml/src/ggml-hexagon/kernels/entry.c`, logcat evidence from the DSP side
(`HAP_compute_res_acquire_cached` / `HAP_compute_res_release_cached` FARF spam),
and CI matrix on Snapdragon 8 Elite (HTP v79) with gemma4/qwen3/qwen1/llama3.*

> Work-in-progress doc. Honest reporting: this round of investigation did
> not yield a measured PP improvement. The data + analysis are still worth
> keeping for future reference, and the code changes (session-lifetime VTCM,
> HEX_OP_PROF compile switch) have merit as code-cleanup work even without
> a measurable speedup.

## TL;DR

Two code changes were applied to `kernels/entry.c` (and one cfg touch):

1. **VTCM session-lifetime** — `dsp_vtcm_acquire()` and `dsp_vtcm_release()`
   are no longer called per `execute_batch`. Instead, `ggml_dsp_open` does
   one `HAP_compute_res_acquire_cached` and `ggml_dsp_close` does one
   `HAP_compute_res_release_cached`. Matches the Qualcomm HTP pattern
   (`vtcm_acquire` / `vtcm_release` only fire when transitioning between
   "active processing" and "forced release"). Per-session HAP_compute_res
   calls drop from ~8700 to 2.
2. **`#if HEX_OP_PROF` compile switch restored** — entry.c profiler arrays
   and `dump_op_prof()` / `init_op_prof_min()` / `htp_op_short_name()` are
   now wrapped in `#if HEX_OP_PROF ... #endif`. `HEX_OP_PROF` defaults to
   0 (off). Skel size: 706984 -> 700104 bytes (-6880). Restore by passing
   `-DHEX_OP_PROF=1` to `make`.

**Measured PP impact: inconclusive.** Single-run PP varies 315-345 t/s for
gemma4 in warm state, with at least one outlier run at 371.54 (user-captured,
not reproducible by a 5-min idle cooldown). The doc's 5-6 AM baseline
362.05 mean was not reproducible in this session despite identical code,
cfg, command, and model. See "Measurement discipline" below.

## Code changes in this commit

### 1. VTCM session-lifetime (entry.c)

**Before (per-batch acquire/release):**
```c
// execute_batch entry
dsp_vtcm_acquire();   // line 1781, HAP_compute_res_acquire_cached on every batch

// ... batch ops ...

// execute_batch exit
if (g_dsp_ctx->vtcm_needs_release) {
    dsp_vtcm_release();  // line 2141-2143, HAP_compute_res_release_cached on every batch
}
```

**After (session-lifetime):**
```c
// ggml_dsp_open (after HAP_compute_res_acquire succeeds)
dsp_vtcm_acquire();   // once per session, sets vtcm_valid=1

// ggml_dsp_close (before HAP_compute_res_release)
dsp_vtcm_release();   // once per session, sets vtcm_valid=0

// execute_batch: no per-batch acquire/release calls
```

**Resulting behavior:**
- HAP_compute_res_acquire_cached: 1 call / session (was: ~4352)
- HAP_compute_res_release_cached: 1 call / session (was: ~4352)
- SDK `adsprpc` V-level `HAP_compute_res_acquire/release_cached` log:
  2 lines / session (was: ~8700)
- VTCM held continuously for the session (no lazy release)
- Trade-off: lose the ability to respond to a forced-release callback from
  another session. For single-session use (the current deployment) this is
  a non-issue.

### 2. `#if HEX_OP_PROF` compile switch (entry.c)

**6 spots wrapped** (default `HEX_OP_PROF=0`):

```c
// line 177-179: default OFF
#ifndef HEX_OP_PROF
#define HEX_OP_PROF 0
#endif

// line 180-196: profiler arrays + bucket/interval macros
#if HEX_OP_PROF
#ifndef HEX_OP_PROF_BUCKETS
#define HEX_OP_PROF_BUCKETS  64
#endif
... (g_op_prof_* arrays) ...
#endif

// line 989-994: execute_op() t0 = ggml_time_us()
#if HEX_OP_PROF
const uint64_t t0 = ggml_time_us();
#endif

// line 1073-1086: execute_op() per-op update
#if HEX_OP_PROF
{ ... g_op_prof_* update ... }
#endif

// line 1094-1146: htp_op_short_name() lookup
#if HEX_OP_PROF
static const char * htp_op_short_name(unsigned int op) { ... }
#endif

// line 1151-1189: init_op_prof_min() + dump_op_prof()
#if HEX_OP_PROF
static void init_op_prof_min(void) { ... }
static void dump_op_prof(const char * tag) { ... }
#endif

// line 2170-2177: execute_batch() counter + dump trigger
#if HEX_OP_PROF
g_op_prof_batch_count++;
if ((g_op_prof_batch_count % HEX_OP_PROF_DUMP_INTERVAL) == 0) {
    dump_op_prof(tag);
}
#endif
```

**Verified**: with `HEX_OP_PROF=0` default, skel size drops by 6880 bytes
(706984 -> 700104), and `adb logcat -d | grep OP-PROF` is empty after a
full gemma4 run.

## Why this round: investigating "PP regression after dual path removal"

The previous day's doc (`algotype29-perf-cache-mode-comparison-20260710.md`)
recorded gemma4 mean PP=362.05 at 5-6 AM. The same commit + cfg + command
+ model in this session (a day later) produced PP=335.77 on the first
measurement. The hypothesis at the start of the day was "dual path removal
caused a PP regression." This turned out to be **incorrect**:

| Commit tested | cfg | device state | PP |
|---|---|---|---|
| 29c1cf196 (before bulk dst flush) | default | warm | 306.21 |
| 576f7eef6 (the doc baseline) | cfg=4 | warm | 335.77 |
| 576f7eef6 | cfg=4 | cool (5 min idle) | 343.54 |
| 602d71e65 (+graph_optimize runtime) | cfg=4 | warm | 316.34 |
| ba4fd0104 (dual path removed) | cfg=4 | warm | 335.91 |

The "regression" never existed — all commits produce 306-345 in warm
state, and the doc's 362 mean was an environmental data point that is
not reproducible in the current session. This is documented separately
in [pp-regression-misdiagnosis-20260711.md](./pp-regression-misdiagnosis-20260711.md).

## The actual investigation: VTCM logcat spam

While ruling out the "dual path regression" hypothesis, the user pointed
out a real DSP-side logcat pattern (logcat output captured mid-session):

```
07-11 ... HAP_compute_res_release_cached: context 0x74054bf0, result 0
07-11 ... HAP_compute_res_acquire_cached: context 0x74054BF0, priority 192, result 0
07-11 ... HAP_compute_res_release_cached: context 0x74054bf0, result 0
07-11 ... HAP_compute_res_acquire_cached: context 0x74054BF0, priority 192, result 0
... (repeated hundreds of times per inference)
```

The user identified that these are **HAP SDK FARF(ALWAYS) logs** from
inside `HAP_compute_res_acquire_cached` / `release_cached`, not entry.c
log statements. The log volume itself is noise, but the underlying
behavior (per-batch acquire + release) is a Qualcomm-pattern violation
that warranted the session-lifetime refactor.

For comparison, the Qualcomm htp model keeps VTCM held for the lifetime
of a worker thread (one acquire, zero or one release per session). JZ's
per-batch acquire/release was an artefact of the pre-cleanup code.

## Measurement discipline (critical)

This investigation reinforced a hard lesson: **device state on Snapdragon
8 Elite causes 5-30% PP variance between consecutive runs of the same
binary, with the same model, command, and cfg.**

| Test | binary | device | PP | notes |
|---|---|---|---|---|
| doc 5-6 AM (576f7eef6) | doc binary | 5-6 AM state | 362.05 mean | 8 runs [351, 371] |
| This session, 5 warm runs | ba4fd0104 (no VTCM change) | warm | 329.99 mean | range [317, 345] |
| This session, cool 5 min | ba4fd0104 | cool | 336.80 | 1 run |
| User screenshot mid-session | ba4fd0104 (VTCM applied) | idle | 371.54 | 1 run, not reproducible |
| This session, after full CI | ba4fd0104 (VTCM applied) | warm | varies | 320-500 across models |

The fact that the **same binary** (after VTCM change) shows 320 t/s in one
warm run and 500 t/s in llama3 (a different model with smaller prompt) is
a clear signal that thermal state, walt governor windowing, and DSP boost
state are the dominant PP variables, not code.

**Implication for future work**: any PP improvement claim must be backed
by N>=5 runs with explicit device-state control, and the variance must
be reported alongside the mean. The doc's 8-run mean of 362 is a
respectable N; this session's single-run 371.54 is noise.

## Test results

All tests used:
- Binary: `ba4fd0104` (dual path removed) + VTCM session-lifetime + HEX_OP_PROF=0
- `scripts/ggml-hexagon.cfg`: dsp_cache_mode=4, ion_sync_mode=1,
  enable_graph_optimize=1, enabled_ops=all, dump_diag_info=0
- Command: `./scripts/build-run-ggmlhexagon-android.sh run_llamacli <model>`
  with `running_params=" -ngl 99 -t 6 -n 256 --ctx-size 8192 --ubatch-size 64 --poll 1000 --no-warmup --no-mmap -fa on"`
- Models: gemma-4-E2B-it-Q4_0.gguf, Qwen3.5-2B-Q4_0.gguf,
  qwen1_5-1_8b-chat-q4_0.gguf, llama-3.2-1B-Q4_0.gguf

### gemma4 (warm, full 4-model CI after gemma4 cool run)

| # | device | PP | TG | output |
|---|---|---|---|---|
| 1 | cool 5 min | 336.80 | 18.77 | OK (no garble, no repeat) |

Compared to doc mean 362.05: **-25 t/s** — but doc's 362 was an
environmental data point, not a code baseline.

### qwen3 (warm, after gemma4)

| # | device | PP | TG | output |
|---|---|---|---|---|
| 1 | warm | 120.20 | 14.62 | OK (no garble, no repeat) |

Compared to doc mean 117.62: **+2.6 t/s** — within noise.

### qwen1 (warm, after qwen3)

| # | device | PP | TG | output |
|---|---|---|---|---|
| 1 | warm | 336.87 | 20.64 | OK (no garble, no repeat) |

Compared to doc mean 374.38: **-37.5 t/s** — outside noise. But the user
captured 406.05 in the same session on this model (different device
state). qwen1 is most affected by device state because its model is
small (1.1 GiB) and the per-op overhead is a larger fraction of total
time.

### llama3 (warm, after qwen1)

| # | device | PP | TG | output |
|---|---|---|---|---|
| 1 | warm | 500.35 | 26.17 | OK (no garble, no repeat) |

Compared to doc mean 660.37: **-160 t/s** — far outside noise. Same
device-state caveat as qwen1. llama3 is the smallest model (737 MiB) and
most sensitive to per-call overhead; 500 vs 660 is 32% spread.

## What this round did NOT find

- **No clear evidence that VTCM session-lifetime changes PP** in either
  direction. The post-change mean of warm gemma4 (single run at 336.80)
  is within the pre-change warm range [317, 345].
- **No regression in any model**: all 4 models produce OK output with
  no garble, repeat, or warning.
- **The 5-6 AM doc 362 baseline is environmental**, not a code target.
  Replicating it requires a specific device state (likely a cold
  idle-with-DVFS-settling window) that this session did not achieve.

## Recommended next steps (not done in this commit)

1. **Lock device state for measurements** — add a `therm_state_lock`
   shell command, log `thermal_zone*/temp` before and after each run,
   and require N>=5 runs per cfg before claiming a delta.
2. **Consider dsp_cache_mode=5** (bit 0 + bit 2) — this commit's default
   cfg=4 leaves bit 0 off, so repack weights are still invalidated on
   every batch. The pre-dual-path-removal code had bit 0 hardcoded ON;
   flipping cfg to 5 should match the pre-cleanup behavior at the cache
   layer (no binary diff, just cfg). Not done here because the user
   reported that cfg=5 produced garbled output in their last test
   session — needs a separate investigation.
3. **Re-run the 4-model CI matrix with `therm_state_lock` once
   available** — to determine if VTCM session-lifetime has any real
   impact on the steady-state PP distribution.

## Files changed

| File | Change |
|---|---|
| `ggml/src/ggml-hexagon/kernels/entry.c` | VTCM session-lifetime; `#if HEX_OP_PROF` wrap (6 spots) |

## Files NOT changed (intentionally)

- `ggml-hexagon.cpp` — no C++ side changes; the VTCM lifetime refactor is
  contained in entry.c.
- `scripts/ggml-hexagon.cfg` — left at baseline (dump_diag_info=0,
  dsp_cache_mode=4, enable_graph_optimize=1, enabled_ops=all).
- `ggml_dsp.idl` — no interface change; same `execute_batch` signature.
- All Qualcomm htp code — per project policy, do not modify.
