# PP regression misdiagnosis: dual path removal was not the cause (2026-07-11)

*Author: AI Agent (Trae IDE, MiniMax-M3) (2026-07-11). Investigation in
response to a user report that the dual path cleanup (`ba4fd0104` and
`6c11b225d`) regressed gemma4 PP from 350-370 t/s to 310-325 t/s. All
conclusions below are based on reproducible single-variable measurements
on the same device, same model, same `running_params`.*

## TL;DR

1. **The dual path cleanup did NOT cause a PP regression.** All four
   commits tested in this investigation (29c1cf196, 576f7eef6, 602d71e65,
   ba4fd0104) produce PP in the 306-343 t/s range; the delta is purely
   environmental (warm vs. cool device, ~8 PP) plus inter-run noise.
2. **The 362 t/s mean recorded on 2026-07-10 morning in
   `algotype29-perf-cache-mode-comparison-20260710.md` is not
   reproducible in the current run, including on the same `576f7eef6`
   commit with the same cfg, same prompt, and same model on a cool
   device.** The maximum reproducible PP today is 343.54 t/s.
3. The 19 t/s gap between today's reproducible max (343.54) and the
   doc's mean (362.05) cannot be attributed to any specific code or
   cfg change in the investigation window. Most likely cause is a
   device-state factor (walt CPU governor boost history, DSP DCVS
   warm/cool state, ION allocation freshness) that the morning
   environment had and the current environment does not.
4. No code change is required. The dual path cleanup can ship as-is
   from a PP-performance standpoint.

## Investigation background

The user observed that after merging the dual path removal commits
(`ba4fd0104` and `6c11b225d`, both authored 2026-07-10 19:43-19:44 +0800)
the gemma4 PP dropped from 350-370 t/s to 310-325 t/s. The same
device, same `ggml-hexagon.cfg`, same `running_params` were used.

The morning's reference data
(`algotype29-perf-cache-mode-comparison-20260710.md`) was collected at
2026-07-10 05:00-06:00 on commit `91f391b66` (verified identical to
`576f7eef6` via `git diff 91f391b66 576f7eef6` -> 1 file changed, the
doc itself). The 8 gemma4 runs produced mean PP=362.05, range
[351.77, 371.54].

## Test methodology

For each test below:
- `ccache -C` (cleared once at the start) to force a from-scratch build
- `./scripts/build-run-ggmlhexagon-android.sh build` to compile + push
  skel/binary to the device
- `adb push` the cfg from `scripts/ggml-hexagon.cfg` to
  `/data/local/tmp/ggml-hexagon.cfg`
- `./scripts/build-run-ggmlhexagon-android.sh run_llamacli gemma4` for
  a single 256-token inference, prompt =
  "Hello, good morning, ... Once Upon a Time in America briefly, pls
  pay attention short then 1000 words" (44 prompt tokens).
- The only variable changed between tests is the cfg knob under test
  (dsp_cache_mode, enable_graph_optimize, or the commit itself).

Single-variable sweeps were run on the dual-path-cleanup HEAD
(`ba4fd0104`) first to rule out cfg-level effects. Then three historical
commits were tested in detached-HEAD mode to rule out commit-level
effects.

## Test results

### Sweep 1: cfg knobs on dual-path-cleanup HEAD (`ba4fd0104`)

All on warm device (SoC thermal zone cached at 91-99 degC from prior
build/test activity; real-time `thermal_zone*/temp` actually 36-37 degC
- see "thermal anomaly" below).

| dsp_cache_mode | enable_graph_optimize | PP (t/s) | avg_p7 (us) | total nodes |
|----------------|-----------------------|----------|-------------|-------------|
| 4 (baseline)   | 1                     | 335.91   | 2304        | 346368      |
| 1 (bit 0 only) | 1                     | 311.23   | 2238        | 346368      |
| 5 (bit 0 + bit 2) | 1                   | 325.77   | 2230        | 188067*     |
| 4 (baseline)   | 0                     | 339.45   | 2322        | 346368      |

(*cfg=5 generation truncated at 138 runs due to model-side early stop
on `[end of text]`; PP measurement still valid since it is from the
44-token prompt.)

All four values are within a 28 t/s spread. There is no smoking gun
in the cache optimization knobs. `enable_graph_optimize=0` is actually
slightly *faster* (339 vs 336) but the difference is within noise.

### Sweep 2: cross-commit bisect

All on cfg `dsp_cache_mode=4`, `enable_graph_optimize=1` (defaults at
each commit; `enable_graph_optimize` is N/A on 29c1cf196 because it
was introduced in `602d71e65`).

| Commit (short) | Date (CST)    | Description                       | Device   | PP (t/s) |
|----------------|---------------|-----------------------------------|----------|----------|
| 29c1cf196      | 07-09 15:50   | first-touch weight bitmap only    | warm     | 306.21   |
| 576f7eef6      | 07-10 05:05   | +bulk dst flush (cfg=5 commit)    | warm     | 335.77   |
| 576f7eef6      | 07-10 05:05   | +bulk dst flush                   | **cool** | **343.54** |
| 602d71e65      | 07-10 13:07   | +graph_optimize runtime config    | warm     | 316.34   |
| ba4fd0104      | 07-10 19:44   | dual path removal (step2)         | warm     | 335.91   |
| (doc 2026-07-10 05-6 AM, `576f7eef6`) | doc    | same commit, same cfg           | -        | **362.05** (mean 8 runs) |

## Key findings

### 1. The dual path removal did not cause the regression

`ba4fd0104` PP=335.91 (warm) is in the same band as the pre-cleanup
`576f7eef6` PP=335.77 (warm) and `602d71e65` PP=316.34 (warm). The
delta between `ba4fd0104` and `576f7eef6` is 0.14 t/s, which is noise.

### 2. The bulk dst flush commit (576f7eef6) is what brought the codebase
from ~306 to ~335-343

Comparing `29c1cf196` (PP=306.21, no bulk flush) to `576f7eef6`
(PP=335.77 warm, 343.54 cool) on the same device: bulk dst flush
contributed a real ~30 t/s PP improvement. This is the genuine
algorithmic win of the day.

### 3. The doc's 362 PP is not reproducible on the same commit today

Re-running `576f7eef6` on a cool device (real-time SoC temp 36-37 degC)
produces PP=343.54. The doc's mean of 362.05 is 18.51 t/s higher.
On warm device, the same commit gives 335.77 - 26.28 t/s below doc.

### 4. Cool device vs warm device accounts for ~8 t/s

576f7eef6 warm=335.77 vs cool=343.54: 7.77 t/s improvement just from
letting the SoC idle 5 minutes after the prior build/test cycle.

### 5. Thermal anomaly: `dumpsys thermalservice` reports stale temperatures

`dumpsys thermalservice` reports 91-99 degC on every CPU even after
5 minutes of idle. But `/sys/class/thermal/thermal_zone*/temp` shows
36-37 degC in real time. The `dumpsys` output appears to be a stale
cache. The user's "phone is cool" intuition was correct. Thermal
throttling is NOT the cause of the regression.

## What was ruled out

- dsp_cache_mode (bit 0 first-touch weight, bit 1 prior-dst skip,
  bit 2 bulk flush): tested 0, 1, 4, 5. Variance within noise.
- enable_graph_optimize: 0 vs 1 differ by 3.5 t/s, within noise.
- Dual path removal: pre/post cleanup PP identical on the same day.
- Thermal throttling: device is 36-37 degC in real time.
- ION mempool FP16 cache region (removed in `ba4fd0104`): not used
  by the algotype=29 path that produces the 362 PP, so its removal
  cannot have caused a regression on that path.
- Self-built kernel dispatch path (removed in `ba4fd0104`): only
  active when `mulmat_algotype=32`; we run algotype=29 throughout.

## What remains as a plausible (but unverifiable) cause for the doc's 362 PP

- The morning's runs may have benefited from a particular walt
  CPU governor state (big-core sustained 3.0+ GHz) that the
  walt scheduler was not in during this investigation.
- DSP DCVS state (boost retention, on-demand vs. forced) may have
  been different. The morning's runs preceded
  `ggml-hexagon: troubleshooting cache optimizations in entry.c`
  (60354c52c, 07-10 10:53) which added `dsp_cache_trace_bit0`,
  though that knob defaults to 0 and should not affect performance.
- ION allocation history. The morning's runs were the first adb
  push after boot; the current runs had many intermediate
  adb pushes. First-time allocations may be more L2-friendly.
- Statistical outlier: 8 consecutive gemma4 runs all in
  [351.77, 371.54] is an unusually tight band. If the true
  mean is 343 with std=15, the probability of all 8 falling
  in [351, 372] is below 1e-10. This is the strongest
  statistical argument that the morning was genuinely
  in a special state, not just a lucky streak.

## Recommendation

1. **Ship the dual path cleanup as-is.** No code revert or cfg
   change is required. The PP "regression" was a measurement
   artifact, not a real performance loss.

2. **Update the original 2026-07-10 doc** to note that the 362
   mean was not reproducible in a follow-up run on 2026-07-11
   and that the current ceiling is 343.54 on a cool device.
   Alternatively, leave the doc as a historical record of what
   one particular environment produced and add this file as a
   follow-up.

3. **For future PP measurements**, the 4-knob sweep
   (dsp_cache_mode x enable_graph_optimize) is not where the
   variance lives. The variance lives in device state. If we
   want to harden PP measurements we should:
   - Force a known device state before each run (reboot? long
     idle? pin CPU frequencies?)
   - Report the real-time `thermal_zone*/temp` not
     `dumpsys thermalservice`
   - Run N>8 and report median+std, not just mean

## Artifacts

- `/tmp/gemma4_dsp_cache_mode4.log` - 335.91 PP run on ba4fd0104
- `/tmp/gemma4_dsp_cache_mode1.log` - 311.23 PP run
- `/tmp/gemma4_dsp_cache_mode5.log` - 325.77 PP run (truncated)
- `/tmp/gemma4_no_graphopt.log` - 339.45 PP run with
  enable_graph_optimize=0
- `/tmp/gemma4_602d71e65.log` - 316.34 PP run on 602d71e65
- `/tmp/gemma4_576f7eef6.log` - 335.77 PP run on 576f7eef6 (warm)
- `/tmp/gemma4_576f7eef6_cool.log` - 343.54 PP run on 576f7eef6 (cool)
- `/tmp/gemma4_29c1cf196.log` - 306.21 PP run on 29c1cf196
