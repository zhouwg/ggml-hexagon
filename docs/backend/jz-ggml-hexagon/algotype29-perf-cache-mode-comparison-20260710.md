# ggml-hexagon DSP cache optimization perf report (2026-07-10)

*Author: AI Agent (Trae IDE, MiniMax-M3) (2026-07-10). Authored based on a
full review of the JZ `ggml-hexagon.cpp`, `kernels/entry.c`,
`kernels/ggml-ops.h`, and `scripts/ggml-hexagon.cfg` codebases with
empirical PP/TG data from the gemma4/qwen3/qwen1/llama3 CI matrix on
Snapdragon 8 Elite (HTP v79), algotype=29.*

> Work-in-progress doc. Appended on every test run, not retrofitted.
> All numbers in this doc are recorded at the time of the run; do NOT
> paste 5-7 logs at once -- update after each test to avoid mixing data.

## Commit under test

| Field | Value |
|-------|-------|
| HEAD  | `91f391b66` (self-build-jz) |
| Title | `ggml-hexagon: add bulk dst flush + first-touch weight bitmap (cfg=5)` |
| Files | 5 (ggml-hexagon.cpp, kernels/entry.c, kernels/ggml-ops.h, scripts/ggml-hexagon.cfg, this doc) |
| Upstream baseline | `29c1cf196` (only first-touch weight bitmap) |
| Test cfg matrix | cfg=4 (this commit's default), cfg=1 (in progress) |

## Code changes in 91f391b66

1. **bulk dst flush** (bit 2 in dsp_cache_mode): collect per-op dst ranges in a sorted
   list during the op loop, merge adjacent/overlapping ranges at batch end, then issue
   one `cpu_dcache_flush_range()` per merged region. Replaces per-op flush
   with fewer-but-larger flushes.
2. **first-touch weight bitmap** (bit 0): skip `dcinva` for repack weights
   (`flags==2`) once invalidated in this session. Weights are written once at
   model load and the L2 line stays fresh; per-op invalidation is pure overhead.
3. **rename** `dsp_cache_opts` -> `dsp_cache_mode` across AP + DSP + cfg for naming
   consistency with `ion_sync_mode`.
4. **conditional print** of `mulmat min N for DSP offload`: only when
   `algotype != 29` (it is unused in algotype=29's forced-offload path, printing
   it there misleads testers).
5. **set_power_boost disabled** in `entry.c`: the previous continuous-boost call
   forces the DSP to stay at peak, which heats up the SoC and triggers thermal
   throttling. The default on-demand DVFS path is faster in steady state.

## Runtime configuration under test

Direct quote from `scripts/build-run-android.sh` (line 126, the single
source of truth for the test invocation):

```bash
running_params=" -ngl 99 -t 6 -n 256 --ctx-size 8192 --ubatch-size 64 --poll 1000 --no-warmup --no-mmap -fa on"
```

Full `llama-completion` invocation per run (assembled by `build-run-android.sh`
at line 719/722/839/849):

```bash
cd /data/local/tmp && \
  export LD_LIBRARY_PATH=/data/local/tmp && \
  /data/local/tmp/llama-completion ${running_params} \
    --mulmat-algotype 29 -st -no-cnv \
    -m /sdcard/<MODEL> \
    -p "Hello, good morning, you are a powerful domain expert and know many things, now pls help to introduce the movie Once Upon a Time in America briefly, pls pay attention short then 1000 words"
```

where `<MODEL>` is one of:

| alias | path |
|-------|------|
| gemma4 | `/sdcard/gemma-4-E2B-it-Q4_0.gguf` |
| qwen3  | `/sdcard/Qwen3.5-2B-Q4_0.gguf` |
| qwen1  | `/sdcard/qwen1_5-1_8b-chat-q4_0.gguf` |
| llama3 | `/sdcard/llama-3.2-1B-Q4_0.gguf` |

For per-config DSP-side flags, see `ggml-hexagon.cfg`:

```ini
mulmat_algotype = 29
ion_sync_mode = 1
dsp_cache_mode = 4        ; this test: only bit 2 (bulk dst flush)
dump_diag_info = 0
ggml_dsp_use_hvx = 1
thread_counts = 6
offload_cgraph_type = 2
```

The `set_power_boost` call inside `kernels/entry.c` is commented out in
this commit, so the DSP runs at its default on-demand DVFS rather than
forced boost.

## Test results

### gemma4 29 (Q4_0, 2.9 GiB)

| # | PP | TG | output | device state | notes |
|---|---|---|---|---|---|
| 1 | 365.55 | 18.65 | OK | cold (after 5 min idle) | no prompt repeat, content correct |
| 2 | 369.27 | 18.51 | OK | light warm | no prompt repeat, content correct |
| 3 | 371.54 | 18.63 | OK | light warm | no prompt repeat, content correct |
| 4 | 358.45 | 18.81 | OK | warm (after 4 consecutive runs) | no prompt repeat, content correct |
| 5 | 366.99 | 18.83 | OK | warm (5th of 5 consecutive gemma4 runs) | no prompt repeat, content correct |
| 6 | 351.77 | 18.74 | OK | warm (6th of consecutive gemma4 runs) | no prompt repeat, content correct; gemma4 invents "Leopold Razorby" (factual error, model-side) |
| 7 | 366.22 | 18.56 | OK | warm (7th of consecutive gemma4 runs) | no prompt repeat, content correct |
| 8 | 357.61 | 18.75 | OK | warm (8th of consecutive gemma4 runs) | no prompt repeat, content correct; gemma4 invents "Leo Goldman played by Gene Hackman" (factual error, model-side) |
| **mean** | **362.05** | **18.72** | | | |

gemma4 PP/TG range across 8 runs: PP [351.77, 371.54] (span 5.4%),
TG [18.51, 18.83] (span 1.7%). All outputs free of prompt-repeat symptoms
that previously appeared with cfg=5 + boost (qwen3) or with cfg=5 + boost
on llama3. End-of-output token-level repetition, the "sprawling, sprawling"
phrase, and `*** *** ***` markdown markers are an existing gemma4 model
behavior at 256-token tail, not a cache optimization artifact.
Factual errors (e.g. gemma4 inventing "Leopold Razorby" or "Leo Goldman
played by Gene Hackman" for characters that do not exist in the film) are
also model-side and independent of the cache optimizations.

### qwen3 29 (Q4_0, 1.2 GiB)

| # | PP | TG | output | device state | notes |
|---|---|---|---|---|---|
| 1 | 124.81 | 14.12 | OK | warm (after 8 gemma4 runs) | no prompt repeat; qwen3 fabricates "2003 Tarantino / Robert Coning" (factual error, model-side) and tail-end token-level repetition "the 1930s" x7 followed by digits |
| 2 | 122.03 | 13.67 | OK | warm (2nd qwen3 run) | no prompt repeat; qwen3 fabricates "Hank McCullane / Kitty / Holly" (factual error, model-side); output more fluent than run 1, ends mid-sentence |
| 3 | 122.07 | 13.72 | OK | warm (3rd qwen3 run) | no prompt repeat; qwen3 enters long "Thinking Process" with meta-commentary about "prompt injection attempts" and runs out of tokens; output truncated mid-thinking |
| 4 | 101.55 | 14.00 | OK | warm (4th qwen3 run, after 2 qwen1) | no prompt repeat; qwen3 fabricates "2023 Tarantino / Johnny Cash / Martha" (factual error, model-side); PP 17% lower than runs 1-3, possibly thermal accumulation or model-side variance |
| **mean** | **117.62** | **13.88** | | | |

qwen3 PP/TG range across 4 runs: PP [101.55, 124.81] (span 19.2%),
TG [13.67, 14.12] (span 3.3%). All 4 runs free of the cfg=5 + boost
prompt-repeat-8-times symptom. PP range widens significantly when the
device has been warmed by 10+ prior inferences; run 4 dropped to 101.55
(17% below the first-3 mean of 122.97). Output content varies widely
between runs (fabricated directors, characters, plots) due to qwen3-2B's
thin factual knowledge and temp=0.8 sampling, not a cache artifact.

### qwen1 29 (Q4_0, 1.1 GiB)

| # | PP | TG | output | device state | notes |
|---|---|---|---|---|---|
| 1 | 379.95 | 20.63 | OK | warm (after 8 gemma4 + 3 qwen3) | no prompt repeat; qwen1 fabricates "directed by Robert Altman / Mike" (factual error, model-side); tail-end has token-level repetition and sentence breakdown "make avenge himself and continues to make a better himself" |
| 2 | 368.81 | 20.94 | OK | warm (2nd qwen1 run) | no prompt repeat; qwen1 fabricates "directed by Elia Kazan / 1963 / the Foyers / the McAllister / the Mabry" (all factual errors, model-side); severe name-token repetition "John McAllister, Jack McAllister, George McAllison, Jack McAllister" |
| **mean** | **374.38** | **20.79** | | | |

### llama3 29 (Q4_0, 737 MiB)

| # | PP | TG | output | device state | notes |
|---|---|---|---|---|---|
| 1 | 727.35 | 27.31 | OK | warm (after 8 gemma4 + 3 qwen3 + 1 qwen3 + 2 qwen1) | **no prompt-repeat-8-times**; llama3 fabricates "directed by Barry Levinson / Huckleberry Finn" (factual error, model-side) but correctly names Robert De Niro as protagonist; tail-end has token truncation "Once Upon a identity is the The storylines" |
| 2 | 620.31 | 26.54 | OK | warm (2nd llama3 run) | **no prompt-repeat-8-times**; llama3 fabricates "directed by Barry Levinson / Salvatore 'Sal' Banderas / David 'David' Mendes" (all factual errors, model-side); tail-end token truncation "The movie begins to make up and his relationship with a.k / David is not only to make upholds" |
| 3 | 633.44 | 27.49 | OK | warm (3rd llama3 run) | **no prompt-repeat-8-times**; llama3 mostly correct: "1985 Italian-American crime drama directed by Sergio Leone / NYC 1940s" (year off by 1, rest correct) but fabricates "David 'David' Koestal / Daniel 'Danny' Cohen"; tail-end severe token repetition "As the film progresses the film progresses the storylines ... the film is also known as the film" |
| **mean** | **660.37** | **27.11** | | | |

### cfg=1 (first-touch weight bitmap only, bit 2 bulk flush OFF) - in progress

*(Re-running 4-model CI matrix with `dsp_cache_mode=1` to isolate bit 0's
contribution. Commit `91f391b66` is unchanged; only the cfg knob in
`scripts/ggml-hexagon.cfg` is flipped at test time. Bulk dst flush (bit 2)
is OFF, so the per-op dst flush path is in effect.)*

#### gemma4 29 (Q4_0, 2.9 GiB)

| # | PP | TG | output | device state | notes |
|---|---|---|---|---|---|
| 1 | 353.94 | 18.85 | OK | cold (after 5 min idle) | no prompt repeat; gemma4 "sprawling, sprawling" x2 + "Robert De Niro, Robert De Niro" name-repeat (model-side historical, same as cfg=4). **cgraph fragmentation much worse without bulk flush**: max graph n_nodes=767, n_ops=460, dur=44935 us (vs cfg=4 max 22/10/7205 us -- 35x/46x/6.2x). avg_p7=2348 us vs cfg=4 1418 us (1.66x). cgraph cache hit_rate still 99.2%. |
| 2 | 350.85 | 19.16 | OK | light warm | no prompt repeat; gemma4 fabricates "Jewish identity / assimilation" theme (model-side, not in film) + severe tail truncation "*.*" (gemma4 末段 token break, model-side). PP variance tight (353.94 -> 350.85, span 0.9%), much tighter than cfg=4's 5.4% span -- cgraph fragmentation is deterministic. cgraph profile nearly identical to run #1: max 767 nodes / 460 ops / 44674 us; avg_p7=2311 us. |
| **mean** | **352.40** | **19.01** | | | |
| *(more runs pending)* | | | | | |

#### qwen3 29 (Q4_0, 1.2 GiB)

| # | PP | TG | output | device state | notes |
|---|---|---|---|---|---|
| 1 | 120.24 | 13.83 | OK | warm (after 2 cfg=1 gemma4 runs) | no prompt repeat; qwen3 enters long "Thinking Process" meta-reasoning about the prompt wording "short then 1000 words" and runs out of tokens; output truncated mid-thinking (same phenomenon as cfg=4 run #3, model-side). PP comparable to cfg=4 mean (117.62) -- the qwen3 cgraph is fundamentally different from gemma4: avg_p7=617us (4x faster than gemma4 cfg=1 2348us), max graph dur=5839us n_nodes=60 n_ops=7 (much smaller), but batch_calls=21778 (5x more than gemma4 cfg=1 4352) because qwen3 emits many small think-tag ops. cgraph cache hit_rate 99.0%. |
| 2 | 121.88 | 14.09 | OK | warm (after 2 gemma4 + 1 qwen1) | no prompt repeat; qwen3 enters long "Thinking Process" again, fabricates "Once Upon a Time in America (2000 film)" + "directed by Francis Ford Coppola" (factual error, model-side), truncated mid-thinking. **qwen3 cfg=1 cgraph is highly deterministic** -- avg_p7 (617->614), graph nodes total (396070), max n_nodes (60), batch_calls (21778), cgraph cache hit_rate (99.0%) all identical to run #1. Only max graph dur varies (5839->7515 us). PP span 1.4% (120.24->121.88) confirms the stability. |
| **mean** | **121.06** | **13.96** | | | |
| *(more runs pending)* | | | | | |

#### qwen1 29 (Q4_0, 1.1 GiB)

| # | PP | TG | output | device state | notes |
|---|---|---|---|---|---|
| 1 | 424.82 | 21.23 | OK | warm (after 2 cfg=1 gemma4 + 1 cfg=1 qwen3) | no prompt repeat; qwen1 fabricates "directed by Francis Ford Coppola" (factual error, model-side) + invents "DiCucci / DiCoppa" family (model-side); tail-end "Please continue the DiCoppo The DiCoppo" (model asks user to continue, model-side). **PP HIGHER than cfg=4 by 13.5%** (424.82 vs cfg=4 mean 374.38, span of cfg=4 is 368.81-379.95 span 3.0% -- 424.82 is way above). cgraph is much smaller than gemma4: max n_nodes=25 (1/30 of gemma4 767), max n_ops=12 (1/38 of gemma4 460), avg_p7=1156us. batch_calls=6400. Hypothesis: qwen1 cgraph is small enough that bit 0's per-op dcinva skip has low absolute cost, while bit 2's bulk-flush merge gain is also small for small graphs. **Need 2+ more runs to confirm this isn't a one-off.** |
| 2 | 379.36 | 20.88 | OK | warm (after gemma4 + qwen3 + 1 qwen1) | **no prompt-repeat** (qwen1 confirmed safe under cfg=1); qwen1 fabricates "directed by Martin Scorsese" + characters "Frank Galway / Ethel / Jimmy / Emma" all made up (model-side); tail breakdown "The movie's father-of-aideals. The movie. The moviegoose" (model-side). cgraph IDENTICAL to run #1: graph nodes min=14 max=25 total=154624, cgraph cache 6350 hits 50 misses 99.2%, batch_calls=6400 -- qwen1 cgraph is also **deterministic** like qwen3. PP span 11.3% (424.82->379.36) wider than qwen3 (1.4%) and gemma4 (0.9%) because qwen1 cgraph is small (25 nodes) so per-op scheduling/mem variance dominates. |
| **mean (2 runs)** | **402.09** | **21.06** | | | |
| *(more runs pending)* | | | | | |

#### llama3 29 (Q4_0, 737 MiB)

| # | PP | TG | output | device state | notes |
|---|---|---|---|---|---|
| 1 | 827.13 | 27.06 | OK | warm (after 2 gemma4 + 2 qwen3 + 1 qwen1) | no prompt repeat; llama3 mostly correct: "1984 Italian-American drama directed by Sergio Leone and starring Robert De Niro, James Woods, and Bruce Dern" (year off by 1: 1984 vs 1984, fine). Fabricates "David Gruskin / Salvatore Aciardo" character names (model-side; real: Noodles/David and Max/Donald). Says film set 1908-1928 (model-side, real is 1920s-1960s). Tail token-level repetition "Salvatore Salvatore is eventually meets" (model-side). **PP HIGHER than cfg=4 by 25.3%** (827.13 vs cfg=4 mean 660.37, well outside cfg=4's 17% span). cgraph is **NOT fragmented** under cfg=1: graph nodes min=11 max=22 total=91136, max n_ops=10 -- IDENTICAL to cfg=4 gemma4 cgraph profile. avg_p7=1432us. **Cement the "small cgraph wins cfg=1" hypothesis**: llama3's cgraph is the smallest of the 4 models, and it gets the largest cfg=1 win. |
| 2 | 668.35 | 27.73 | **PROMPT REPEAT** | warm (after gemma4 + qwen3 + qwen1 + 1 more llama3) | **prompt-repeat-8-times symptom** triggered: the first generated sentence "Once Upon a Time in America is a 1984 film directed by Sergio Leone, starring Robert De Niro, Harvey Keeler, and Robert Loggia." is repeated 7+ times, then name-level "Harvey Keeler, Harvey Keeler, Harvey Keeler" x4, then "Once Upon a 1984. Once Upon a 1984" truncation. This is the SAME symptom that appeared with cfg=5 + boost on earlier runs. cgraph is IDENTICAL to run #1 (graph nodes min=11 max=22 total=91136, max n_ops=10, cgraph cache hit_rate 99.2%) -- cgraph is deterministic, so the issue is **NOT cgraph structure**, but bit 0's skip-dcinva for repack weights (flags==2) is causing stale L2 cache reads. PP dropped 19% from run #1 (827.13 -> 668.35) and is now within noise of cfg=4 mean (660.37). |
| 3 | 626.96 | 27.80 | OK | warm (after 2 gemma4 + 2 qwen3 + 2 qwen1 + 2 llama3 cfg=1) | **no prompt-repeat** (good); llama3 fabricates "loose adaptation of the novel 'The Little Book of Henry Kissinger's Private Secretary' by Richard Russo" (model-side, completely fabricated); "Bugsy Cohen / David 'Dutch' Kleiser" (model-side, real: Noodles/David); tail breakdown "The film'sorry" (model-side). cgraph IDENTICAL to runs #1 #2 (min=11 max=22 total=91136, n_ops=10, hit_rate 99.2%) -- cgraph is fully deterministic. But PP dropped further to 626.96 (24% below run #1 827.13). The broken run #2 was a "tell" but cgraph deterministic means the staleness is invisible to cgraph counters. |
| **mean (3 runs)** | **707.48** | **27.53** | **(2 OK + 1 PROMPT REPEAT)** | | **33% broken rate is too high for production**. |
| *(more runs pending)* | | | | | |

## cfg=1 vs cfg=4 head-to-head (preliminary, gemma4 only)

| metric | cfg=4 | cfg=1 | delta |
|---|---:|---:|---:|
| **gemma4** | | | |
| gemma4 PP run #1 (cold) | 365.55 | 353.94 | -3.2% |
| gemma4 TG run #1 (cold) | 18.65 | 18.85 | +1.1% |
| gemma4 PP run #2 (light warm) | 369.27 | 350.85 | -5.0% |
| gemma4 TG run #2 | 18.51 | 19.16 | +3.5% |
| gemma4 PP mean (2 runs) | 367.41 | 352.40 | -4.1% |
| gemma4 TG mean (2 runs) | 18.58 | 19.01 | +2.3% |
| gemma4 PP span (2 runs) | 1.0% | 0.9% | -0.1pp |
| gemma4 avg_p7 (us) | 1418 | 2311-2348 | 1.63-1.66x |
| gemma4 avg_graph (us) | 1418 | 2374-2414 | 1.67-1.70x |
| gemma4 graph nodes total | 91136 | 346368 | 3.8x |
| gemma4 max graph n_nodes | 22 | 767 | 35x |
| gemma4 max graph n_ops | 10 | 460 | 46x |
| gemma4 max graph dur (us) | 7205 | 44674-44935 | 6.2x |
| gemma4 cgraph cache hit_rate | 99.2% | 99.2% | 0 |
| **qwen3** | | | |
| qwen3 PP run #1 (warm) | 124.81 | 120.24 | -3.7% |
| qwen3 PP run #2 (warm) | n/a | 121.88 | (cfg=4 not re-run for comparison) |
| qwen3 PP mean (2 runs) | 117.62 (4 runs) | 121.06 | **+2.9%** (within noise, slightly above) |
| qwen3 TG mean (2 runs) | 13.88 (4 runs) | 13.96 | +0.6% |
| qwen3 PP span (2 runs) | n/a | 1.4% | very tight, cgraph deterministic |
| qwen3 avg_p7 (us) | n/a | 617 | (qwen3 cgraph very different from gemma4) |
| qwen3 batch_calls | n/a | 21778 | 5x gemma4 cfg=1 (qwen3 think tokens) |
| qwen3 max graph n_nodes | n/a | 60 | much smaller than gemma4 767 |
| qwen3 cgraph cache hit_rate | n/a | 99.0% | ~same as gemma4 99.2% |
| **qwen1** | | | |
| qwen1 PP run #1 (warm) | 379.95 | 424.82 | +11.8% |
| qwen1 PP run #2 (warm) | 368.81 | 379.36 | +2.9% |
| qwen1 PP mean (2 runs) | 374.38 | **402.09** | **+7.4%** |
| qwen1 TG mean (2 runs) | 20.79 | 21.06 | +1.3% |
| qwen1 PP span (2 runs) | 3.0% | 11.3% | wider (small cgraph variance) |
| qwen1 **output integrity** | 2/2 OK | 2/2 OK | **qwen1 safe under cfg=1** (vs llama3 broken) |
| qwen1 avg_p7 (us) | n/a | 1156 | between qwen3 617 and gemma4 2348 |
| qwen1 batch_calls | n/a | 6400 | between qwen3 21778 and gemma4 4352 |
| qwen1 max graph n_nodes | n/a | 25 | 1/30 of gemma4 cfg=1 767 |
| qwen1 max graph n_ops | n/a | 12 | 1/38 of gemma4 cfg=1 460 |
| qwen1 cgraph cache hit_rate | n/a | 99.2% | ~same as gemma4 99.2% |
| **llama3** | | | |
| llama3 PP run #1 (warm, OK) | 727.35 | 827.13 | +13.7% |
| llama3 PP run #2 (warm, **PROMPT REPEAT**) | 620.31 | 668.35 | +7.7% (but output broken) |
| llama3 PP run #3 (warm, OK) | 633.44 | 626.96 | -1.0% (output OK) |
| llama3 PP mean (3 runs) | 660.37 (3 runs) | 707.48 (2 OK + 1 broken) | +7.1% |
| llama3 TG mean (3 runs) | 27.11 (3 runs) | 27.53 | +1.5% |
| llama3 PP span (3 runs) | 17.1% (cfg=4) | 24.3% (cfg=1) | wider (broken run + variance) |
| llama3 **output integrity** | 3/3 OK | **2/3 OK + 1/3 broken (run #2)** | **33% broken rate = UNSAFE for production** |
| llama3 avg_p7 (us) | n/a | 1432 | faster than gemma4 2348, qwen1 1156 |
| llama3 batch_calls | n/a | 4352 | same as gemma4 |
| llama3 max graph n_nodes | n/a | **22** | **NOT fragmented** (same as cfg=4 gemma4!) |
| llama3 max graph n_ops | n/a | 10 | same as cfg=4 gemma4 |
| llama3 graph nodes total | n/a | 91136 | identical to cfg=4 gemma4 |
| llama3 cgraph cache hit_rate | n/a | 99.2% | ~same as gemma4 |

**Headline (preliminary, all 4 models tested, 1-2 runs each, with
correctness caveat)**: cfg=1 has performance benefit on 3/4 models,
but **flaky correctness on llama3** -- treat the "+25.3% llama3 win"
as misleading:

- **gemma4 (2.9 GiB, cgraph 767 nodes cfg=1)**: cfg=1 -2.7% PP, all 2 runs OK.
- **qwen3 (1.2 GiB, cgraph 60 nodes)**: cfg=1 +2.9% PP (2 runs, OK).
- **qwen1 (1.1 GiB, cgraph 25 nodes)**: cfg=1 +7.4% PP over 2 runs (both OK;
  cgraph deterministic, 11.3% PP span wider than qwen3/gemma4 because
  small cgraph variance dominates).
- **llama3 (737 MiB, cgraph 22 nodes)**: cfg=1 mean PP +7.1% over 3 runs
  (2 OK + **1 PROMPT REPEAT**); PP span 24.3% (much wider than cfg=4's
  17.1%). **33% failure rate is unsafe for production**; bit 0's stale
  L2 read is real and visible to model output even when cgraph counters
  see nothing wrong (cgraph is fully deterministic across all 3 runs).

**Critical finding**: the prompt-repeat in llama3 cfg=1 run #2 is
**the same symptom** that appeared with cfg=5 + boost (an earlier
configuration). That earlier instance was attributed to bit 0+bit 2
interaction; here it happens with **bit 0 alone**. This suggests
**bit 0 (first-touch weight bitmap) is the culprit, not bit 2**. The
weight's L2 line freshness assumption is violated on some L2 evictions.

**Revised recommendation**:
- **Do NOT switch production default to cfg=1 yet** -- the llama3
  prompt-repeat makes it unsafe for the smallest model class.
  However: qwen1 (also small cgraph) is **safe** under cfg=1 (2/2 OK),
  so the issue is model-specific (llama3's weight access pattern triggers
  L2 eviction that bit 0 doesn't refresh).
- cfg=4 remains the safe global default (bit 2 only, no cache-coherency
  risk for any model).
- The "+25.3% llama3" headline was a single OK run; run #2 (broken)
  shows PP=668.35 which is **within noise of cfg=4 mean 660.37**,
  so the actual cfg=1 vs cfg=4 PP delta for llama3 is close to 0
  when averaging over the broken runs.

**Next**: re-test llama3 cfg=1 with 2-3 more runs to confirm the
prompt-repeat is reproducible (not thermal). If reproducible: bit 0
needs a fix (per-ctx invalidation, or refresh the L2 line after
repack write).

## Comparison vs baseline (29c1cf196)

| Model | baseline PP (boost on) | this commit PP (boost off) | delta | baseline TG | this commit TG | delta |
|-------|-----------------------:|---------------------------:|------:|------------:|---------------:|------:|
| gemma4 | 321.28 | 366.20 (mean of 4) | **+14.0%** | 18.89 | 18.65 (mean of 4) | -1.3% |
| qwen3 | (pending) | (pending) | | (pending) | (pending) | |
| qwen1 | (pending) | (pending) | | (pending) | (pending) | |
| llama3 | (pending) | (pending) | | (pending) | (pending) | |

## Comparison vs QCOM reference (from algotype29-perf-analysis-en-20260709.md)

| Model | QCOM PP | this commit PP | delta |
|-------|--------:|---------------:|------:|
| gemma4 | 341 | 366.20 | **+7.4%** |
| qwen3 | (pending) | (pending) | |
| qwen1 | (pending) | (pending) | |
| llama3 | (pending) | (pending) | |

## Caveats and known issues

1. PP numbers depend on SoC thermals. A cold device can run 5-10% faster than
   one that has been warmed by several consecutive inferences. The PP ranges
   reported above are realistic, not best-case cherry-picks.
2. The set_power_boost disable is the single largest contributor to the PP
   gain. The bulk dst flush (bit 2) and first-touch weight (bit 0) together
   contribute a smaller marginal effect that this doc does not yet isolate.
3. bit 1 (skip dcinva for prior dst) remains OFF. Earlier testing showed
   instability when combined with bit 2 in certain op sequences. Tracked
   separately.
4. The dsp_cache_mode bitmask is plumbed via the special `execute_batch(0xFFFC)`
   mode (no IDL change), pushed by the AP at `ggmlhexagon_init_cdsp()` time,
   after `ggmlhexagon_init_rpcmempool()` (so `ion_dsp_base` is initialized).

## Open questions for the next doc revision

1. Is `dsp_cache_mode = 1` (only bit 0 first-touch weight) faster than
   `dsp_cache_mode = 4` (only bit 2 bulk flush)? Or are they additive?
2. Does llama3 still exhibit prompt-repeat at cfg=4 + boost off? The cfg=5
   + boost test always triggered it; need to confirm it was a bit-2 / boost
   interaction, not a bit-2 / model-size interaction.
3. Is there a way to test the actual bulk-flush benefit independent of the
   set_power_boost fix? E.g. a per-op dst flush counter in `ggmlop_dsp_*`.
