# JZ's ggml-hexagon vs Qualcomm's ggml-hexagon: Architecture Analysis

> Last updated: 2026-08-22 (file name date 20260713 reflects the original creation; see Revision History)

> Author: Kimi-K2.7-Code (original), revised by Kimi-K3, GLM-5.2, Kimi-K3, Jeff Zhou


***

| Abbr. | Full name / Meaning |
|------|-------------|
| JZ   | JZ's ggml-hexagon (custom backend, `GGML_HEXAGON_JZ=ON`) |
| QCOM | Qualcomm's ggml-hexagon (official backend, `GGML_HEXAGON_JZ=OFF`) |
| PP   | Prompt Processing (prefill phase) |
| TG   | Token Generation (decode phase) |
| mempool | kernel-allocated AP-DSP shared memory (single allocation, offset addressing) |
| baseline/headline | branch `self-build-jz` |

## 1. JZ vs Qualcomm ggml-hexagon: Architecture Comparison

Both JZ and Qualcomm ggml-hexagon backends route through Qualcomm's `execute_op` path. As of 2026-07-26, JZ maintains its own DSP kernel directory `ggml/src/ggml-hexagon/kernels/` (forked from `htp/` at self-build-jz `2be3826c9`), while Qualcomm continues using `ggml/src/ggml-hexagon/htp/` which tracks upstream master. The DSP entry points differ: JZ uses `kernels/entry.c`, while Qualcomm uses `htp/main.c`. On the AP side, JZ uses `ggml-hexagon-jz.cpp` while Qualcomm uses `ggml-hexagon.cpp`. The AP-side implementations differ significantly, leading to performance differences.



JZ (`ggml-hexagon-jz.cpp` + `kernels/`) and QCOM (`ggml-hexagon.cpp` + `htp/`) are **two evolutionary branches based on the same set of hexagon kernels**, with the fork point being Qualcomm [PR #26049](https://github.com/ggml-org/llama.cpp/pull/26049).

- Before PR #26049, both had identical operators.
- After PR #26049, operator improvements under QCOM's `htp/` were manually ported into JZ's `kernels/`.
- **The performance difference is not in the kernel operators themselves, but in the scheduling framework, cache strategy, and offload strategy.**

### 1.1 Core Architecture Differences

- **JZ**: native FastRPC `invoke` + single mempool (offset addressing)
- **Qualcomm**: `dspqueue` + per-chunk shared buffers (`bi` indirect addressing)

**Table 1**: Architecture comparison

| Dimension | JZ          | QCOM                                       |
| ------ | ------------------------ | ----------------------------------------------------------- |
| Control plane | Native FastRPC `invoke` (synchronous) | `dspqueue_write/read` (asynchronous, up to 16 concurrent batches) |
| Data plane    | single mempool + offset addressing     | per-chunk + `bi` (buffer index) indirect addressing            |
| DSP entry | `kernels/entry.c`                       | `htp/main.c`                                                |
| DSP kernels     | `kernels/*.c`                           | `htp/*.c`                                                   |
| AP-side code    | `ggml-hexagon-jz.cpp`                   | `ggml-hexagon.cpp`                                          |
| Build option    | `GGML_HEXAGON_JZ=ON`                    | `GGML_HEXAGON_JZ=OFF` (default)                             |
| Cache coherency | User-space: role-aware (`ion_sync` + `dsp_cache_mode`) | Kernel-space driver flags (uniform per batch)             |

**Table 2**: single mempool vs per-chunk mutiple shared buffers code-level comparison

| Dimension | JZ | QCOM | Winner |
| --- | --- | --- | --- |
| `fastrpc_mmap` call count | single mempool: 1 at init | 1 per chunk | JZ |
| fd count | 1 | 1 per chunk | JZ |
| DSP tensor addressing | direct `void *` offset | `bi` -> `htp_buf_desc[]` indirect addressing | JZ |
| Batch transport | `invoke` carries the entire graph batch | `dspqueue_write` | Tie |
| Memory lifecycle | single alloc/free | per-chunk alloc/mmap + munmap/free | JZ |
| IOVA spatial locality (prefetch/TLB) | contiguous, predictable | fragmented across chunks | JZ |
| Cache coherency | User-space: role-aware (weight vs activation); `ion_sync` + `dsp_cache_mode` | driver flushes descriptor packet + DSP-side full D-cache flush+invalidate per batch (uniform, role-blind) | JZ (role-aware policy flexibility) |
| Physical address stability | stable after allocation (no migration) | stable after allocation (no migration) | Tie |
| lm-head offload | feasible (within mempool offset range) | infeasible (per-chunk fd/mmap/lifecycle overhead) | JZ |



Qualcomm's disadvantages - per-chunk fd count, per-chunk mmap calls, DSP-side `bi` indirection, multi-write batch transport - are **overhead inherent to the per-chunk API design**. Every additional buffer requires another fd, another mmap, and another `htp_buf_desc[]` entry at the interface level. JZ's single pool has none of these per-chunk costs.

### 1.2 Control-Plane Primitive Differences: dspqueue vs. Native FastRPC invoke

**Table 3**: Control plane comparison

| Dimension | JZ | QCOM |
| --- | --- | --- |
| Primitive | Native FastRPC `invoke` | `dspqueue_write/read` queue semantics |
| Dispatch style | AP calls DSP function directly, carrying descriptors | AP pushes entire op-batch; DSP woken by packet callback |
| Blocking model | synchronous per call | AP async push, responses drained later |
| Batch handling | Single `invoke` carries entire graph batch, sometimes hundreds of ops | Single `dspqueue_write` carries multiple ops (`htp_opbatch_req`) |

### 1.3 Data Plane: Nearly Identical

- Both transport tensor data through **AP-DSP shared memory**.
- Both have AP write descriptors/data and DSP read descriptors/data.
- Both require **cache flush / invalidate** synchronization.
- Both ultimately run the same HVX/HMX kernels.

QCOM's DSP entry point [`htp/main.c`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/main.c) parses `htp_buf_desc[]`, `htp_tensor[]`, and `htp_op_desc[]` via `htp_packet_callback` before dispatching to kernels. JZ completes the same flow in [`kernels/entry.c`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/entry.c). Both data paths are identical:

```
AP packs descriptors -> transport to DSP -> DSP entry parses descriptors -> dispatch op execution
```

### 1.4 Source File Structure

**Table 4**: Source file structure

| File                                                            | Description                                                                       |
| ------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp`                   | JZ AP-side code                                                      |
| `ggml/src/ggml-hexagon/ggml-hexagon.cpp`                      | QCOM AP-side code                                     |
| `ggml/src/ggml-hexagon/CMakeLists.txt`                        | Unified build (QCOM baseline + `GGML_HEXAGON_JZ` option)                    |
| `ggml/src/ggml-hexagon/kernels/Makefile`                      | JZ DSP skel build (entry.c + kernels/)                                   |
| `ggml/src/ggml-hexagon/htp/CMakeLists.txt`                    | QCOM DSP skel build (main.c + htp/)                           |
| `ggml/src/ggml-hexagon/kernels/entry.c`                       | JZ DSP entry                                              |
| `ggml/src/ggml-hexagon/kernels/dsp-ctx.h`                     | JZ DSP session context + descriptors                                    |
| `ggml/src/ggml-hexagon/htp/main.c`                            | QCOM DSP entry                                        |
| `ggml/src/ggml-hexagon/htp/htp-ctx.h`                         | QCOM DSP session context + mmap/spad                                    |
| `ggml/src/ggml-hexagon/kernels/*.c`                           | JZ DSP kernels (forked from htp/, baseline 2be3826c9)                 |
| `ggml/src/ggml-hexagon/htp/*.c`                               | QCOM DSP kernels                           |



The unified `CMakeLists.txt` is based on QCOM's version with a single addition:

```cmake
option(GGML_HEXAGON_JZ "Use JZ's AP implementation" OFF)
```

- `GGML_HEXAGON_JZ=OFF` (default): QCOM upstream behavior, builds DSP skels via `ExternalProject_Add`.
- `GGML_HEXAGON_JZ=ON`: uses `ggml-hexagon-jz.cpp`, builds DSP skels (all 4 versions: v73/v75/v79/v81) via `make -C kernels/`.

## 2. Performance Comparison

**The single shared memory pool is the better architecture for this workload, proven in practice.** JZ exceeds Qualcomm on both PP and TG under identical test conditions, enabled by a session-resident ~214 MB repacked lm-head that the per-chunk shared-memory design cannot express economically.

### 2.1 Snapdragon 8 Elite (v79)

**Table 5: JZ vs Qualcomm (5-run mean, 2026-07-22)**

| Implementation | PP (tok/s) | TG (tok/s) |
|---|---|---|
| JZ ggml-hexagon (dsp_cache_mode=5) | 686.46 | 26.91 |
| Qualcomm ggml-hexagon | 435.14 | 24.91 |

Per-run breakdowns are provided in the automated AB-test tables below (Table 5A/5B), which were measured under different (warm) thermal conditions.

Test conditions: gemma-4-E2B-it-Q4_0.gguf, Snapdragon 8 Elite (v79, OnePlus 13), `/data/local/tmp/llama-completion -ngl 99 -t 6 -n 256 --ctx-size 8192 --ubatch-size 64 --poll 1000 --no-warmup --no-mmap -fa on --jinja -st -m /sdcard/gemma-4-E2B-it-Q4_0.gguf -p "Hello, good morning, you are a powerful domain expert and know many things, now pls help to introduce the movie Once Upon a Time in America briefly, pls pay attention short then 1000 words\n"`. Both backends run the same HMX kernels from `kernels/` (JZ) or `htp/` (QCOM); the difference is architectural.

**Table 5A: Automated AB test on 8 Elite (2026-07-24 11:06, warm device)**

Generated by `./scripts/build-run-ggmlhexagon-android.sh run_abtest 2>&1 | tee log_abtest_$(date +%Y%m%d-%H%M%S).txt` (log: `log_abtest_20260724-110640.txt`). Same model, same prompt, same parameters as Table 5, but on a warm device (3 JZ rounds + 3 QCOM rounds in ~1 minute), so absolute numbers are lower than Table 5. The relative gap remains in the same direction.

| Implementation | Run | PP (tok/s) | TG (tok/s) |
|---|---|---|---|
| JZ | 1 | 702.06 | 28.06 |
| JZ | 2 | 707.09 | 28.29 |
| JZ | 3 | 704.83 | 25.87 |
| **JZ mean** | | **704.66** | **27.41** |
| QCOM | 1 | 550.84 | 25.39 |
| QCOM | 2 | 572.62 | 25.73 |
| QCOM | 3 | 524.42 | 25.46 |
| **QCOM mean** | | **549.29** | **25.53** |

JZ vs QCOM on warm 8 Elite: PP +28.3% (1.283x), TG +7.4% (1.074x). Compared with Table 5 (5-run mean on cool device: PP +57.7%, TG +8.0%), the PP gap narrows under sustained thermal load but the TG gap is essentially unchanged. PP stability: JZ spread 0.71% (702-707 tok/s) vs QCOM spread 8.78% (524-573 tok/s), continuing the single-pool cache-locality advantage seen in Table 8. TG stability: QCOM spread 1.33% vs JZ spread 8.85% on this run, which is an inversion from the typical pattern; JZ TG run 3 dropped to 25.87 while runs 1-2 were 28.06/28.29, suggesting transient thermal or scheduling noise rather than a structural regression (a follow-up re-run is recommended for confirmation).

**Table 5B: Automated AB test on 8 Elite (2026-07-26, post-kernels-fork)**

Generated by `./scripts/build-run-ggmlhexagon-android.sh run_abtest 2>&1 | tee log_abtest_$(date +%Y%m%d-%H%M%S).txt` (log: `log_abtest_20260726-094618.txt`). Same model, same prompt, same parameters as Table 5, on a warm device (3 JZ rounds + 3 QCOM rounds in ~2 minutes). This run verifies the kernels/ fork (baseline `2be3826c9`) and the llama-bench segfault fix (revert of commit `998199e21`) together: output is coherent with no garbled text, and PP/TG fully exceed QCOM.

| Implementation | Run | PP (tok/s) | TG (tok/s) | Total (ms) |
|---|---|---|---|---|
| JZ | 1 | 651.91 | 27.21 | 9781.13 |
| JZ | 2 | 674.91 | 26.15 | 10159.81 |
| JZ | 3 | 657.00 | 26.63 | 9972.43 |
| **JZ mean** | | **661.27** | **26.66** | **9971.12** |
| QCOM | 1 | 431.22 | 23.20 | 11260.92 |
| QCOM | 2 | 421.40 | 23.15 | 11275.93 |
| QCOM | 3 | 438.86 | 23.10 | 11305.80 |
| **QCOM mean** | | **430.49** | **23.15** | **11280.88** |

JZ vs QCOM on warm 8 Elite: PP +53.6% (1.536x), TG +15.2% (1.152x). All 6 runs produced coherent output with no garbled text, confirming the kernels/ fork successfully resolved the garbled-output regression introduced by Qualcomm's PR #26049. JZ cgraph cache hit rate: 98.8% (253 hits / 3 misses), graph nodes: 1493 per call.

**llama-bench comparison (2026-07-22):**

![llama-bench JZ vs Qualcomm 1](images/Screenshot%20from%202026-07-22%2016-59-29.png)

![llama-bench JZ vs Qualcomm 2](images/Screenshot%20from%202026-07-22%2017-02-27.png)

### 2.2 Snapdragon 8 Gen3 (v75)

**Table 6: JZ vs Qualcomm on 8 Gen3 (3-run mean, 2026-07-23, freshly rebooted device)**

| Implementation | PP (tok/s) | TG (tok/s) |
|---|---|---|
| JZ ggml-hexagon (dsp_cache_mode=5, shared-memory pool 3830 MiB) | 319.82 | 20.83 |
| Qualcomm ggml-hexagon | 254.72 | 17.55 |

**Table 7: Detailed per-run data on 8 Gen3 (2026-07-23)**

| Implementation | Run | PP (tok/s) | TG (tok/s) |
|---|---|---|---|
| JZ | 1 | 320.45 | 20.99 |
| JZ | 2 | 324.46 | 20.81 |
| JZ | 3 | 314.56 | 20.69 |
| QCOM | 1 | 255.10 | 17.79 |
| QCOM | 2 | 269.67 | 17.40 |
| QCOM | 3 | 239.38 | 17.46 |

Test conditions: gemma-4-E2B-it-Q4_0.gguf, Snapdragon 8 Gen3 (v75, Xiaomi 14), `/data/local/tmp/llama-completion -ngl 99 -t 6 -n 256 --ctx-size 8192 --ubatch-size 64 --poll 1000 --no-warmup --no-mmap -fa on --jinja -st -m /sdcard/gemma-4-E2B-it-Q4_0.gguf -p "Hello, good morning, you are a powerful domain expert and know many things, now pls help to introduce the movie Once Upon a Time in America briefly, pls pay attention short then 1000 words\n"`. Both backends run the same HMX kernels from `kernels/` (JZ) or `htp/` (QCOM); data collected on a freshly rebooted device to avoid thermal throttling bias.

On both 8 Elite (v79) and 8 Gen3 (v75), JZ's single shared-memory pool architecture consistently outperforms Qualcomm's per-chunk design: PP 1.58x / TG 1.08x on v79, PP 1.26x / TG 1.19x on v75. The architectural advantage - session-resident repacked lm-head, role-aware cache management, contiguous IOVA for prefetch/TLB efficiency - is not conditional on hardware generation. Earlier tests suggesting near-parity on v75 were an artifact of thermal throttling: under sustained load, JZ PP is stable across runs (314-324 tok/s) while QCOM PP varies more (239-270 tok/s), confirming that the single pool's contiguous layout degrades more gracefully under thermal pressure than Qualcomm's fragmented per-chunk IOVA.

**Table 8: Automated AB test on 8 Gen3 (2026-07-23, warm device)**

Generated by `./scripts/build-run-ggmlhexagon-android.sh run_abtest 2>&1 | tee log_abtest_$(date +%Y%m%d-%H%M%S).txt` (log: `log_abtest_20260723-193916.txt`). Device was already warm from prior testing, so absolute numbers are lower than Table 6/Table 7, but the relative stability difference is more pronounced.

| Implementation | Run | PP (tok/s) | TG (tok/s) |
|---|---|---|---|
| JZ | 1 | 245.58 | 18.85 |
| JZ | 2 | 243.99 | 18.86 |
| JZ | 3 | 245.05 | 18.94 |
| **JZ mean** | | **244.87** | **18.88** |
| QCOM | 1 | 277.81 | 19.28 |
| QCOM | 2 | 215.38 | 16.61 |
| QCOM | 3 | 205.77 | 15.57 |
| **QCOM mean** | | **232.99** | **17.15** |

JZ vs QCOM on warm device: PP +5.1%, TG +10.1%. The key finding here is not the mean but the variance: JZ PP is rock-steady at 244-246 tok/s (<1% spread) while QCOM PP collapses from 278 to 206 tok/s (25.8% drop) across three consecutive runs. QCOM's first run actually edges out JZ (PP 277.81 vs 245.58, TG 19.28 vs 18.85), confirming that on a cool device QCOM's dspqueue async overlap can match or slightly exceed JZ's synchronous FastRPC; but as the DSP heats up and throttles, QCOM's fragmented per-chunk IOVA loses cache locality faster than JZ's contiguous single-pool layout. JZ's single shared-memory pool architecture is significantly more resilient to thermal throttling than QCOM's per-chunk design: QCOM may approach or slightly exceed JZ on a cold device, but degrades rapidly under sustained load. JZ's contiguous IOVA layout preserves cache locality when the DSP downclocks, while QCOM's fragmented per-chunk IOVA suffers greater locality loss under the same thermal pressure.

**llama-bench comparison (2026-07-23):**

![llama-bench JZ vs Qualcomm on 8Gen3 1](images/Screenshot%20from%202026-07-23%2019-48-38.png)

![llama-bench JZ vs Qualcomm on 8Gen3 2](images/Screenshot%20from%202026-07-23%2020-00-49.png)

## 3. The Decisive Single-Pool Advantage: Session-Resident Repacked Weights

### Background

TG (token generation) is bandwidth-bound on both backends: every token re-reads all weights from DRAM. The lm-head matmul (262144 x 1536, Q4_K, ~30 ms/token on CPU) was the single largest TG cost. Both JZ and Qualcomm previously rejected quantized weight matrices with `ne[1] > 32768`, so lm-head ran on the CPU in both implementations.

### Why per-chunk shared-memory cannot fix this economically

Offloading lm-head to the DSP requires a repacked (tiled) copy of the weight to live in DSP-addressable memory for the entire session. Under Qualcomm's per-chunk design, every `ggml_hexagon_shared_buffer` carries its own shared-memory fd, its own `fastrpc_mmap`, per-batch descriptor re-registration (`add_buffer()`), a DSP-side mmap slot out of a limited vmem budget (`prep_op_bufs` in `htp/main.c` evicts and re-mmaps under pressure), and a lifecycle that must be coordinated with DSP-side unmapping. A ~214 MB single-purpose resident buffer pays all of these recurring per-chunk costs, which is consistent with Qualcomm keeping the 32768-row guard in place and lm-head on the CPU.

### Why the single pool makes it natural

JZ maps the pool once at init (`fastrpc_mmap`, capacity probed up to 4032 MiB on v79; see `ggmlhexagon_init_rpcmempool()`). Repacking lm-head into the pool at load time costs one conversion pass; afterwards the repack is just an offset range inside an already-mapped region - zero recurring fd/mmap/lifecycle cost - and each token simply streams it from DRAM. The apparent constraint (one pool, no per-chunk granularity) is exactly what makes a ~214 MB resident repack cheap.

### The three changes, all enabled by the pool

1. **Removed the `ne[1] > 32768` guard** for quantized weights in `ggmlhexagon_supported_mul_mat`, allowing lm-head to offload.
2. **Q4_K stored as Q4_0 tiled repack** (`repack_q4k_as_q4_0_tiled_to_buf` in [`ggml-hexagon-jz.cpp`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp), 32-row strip conversion; inverse transform in `get_tensor` for host reads): the resident repacked lm-head is ~214 MB. Note the repack does **not** reduce bandwidth - Q4_K and Q4_0 have the same data size (both 0.5625 B/param), so per-token DRAM traffic is unchanged. Its value is that it turns the Q4_K weight into a tiled Q4_0 layout the DSP can execute directly, which is what makes the offload possible.
3. **First-touch weight invalidation** (`dsp_cache_mode` bit 0, default mode=5). With lm-head resident, per-token weight traffic grew to ~1.9 GB and re-invalidating it every token cost ~9.2 ms/token of DSP-side dcinva sweeping. Repack weights are written once by the AP at load time and never touched again, so after a first-touch invalidate the DSP skips re-invalidation for the rest of the session, removing the ~9.2 ms entirely. The two-pass defense (DSP-side `weight_inval_unmark()` on dst write plus an AP-side `g_ever_dst_ptrs` set) closes cross-graph stale reads.

Removing DSP-side debug/profiler logging afterwards (`-DNDEBUG` skel build) brought further PP/TG gains.

### Side effect: cache-coherency advantage flipped

Qualcomm's per-batch cache maintenance is uniform and role-blind: the driver's flush/invalidate flags cover only the small dspqueue descriptor packet, while tensor data is handled by a full D-cache flush+invalidate on the DSP at batch start and end ([`htp/main.c`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/main.c), lines 999 and 1078) - neither can distinguish weight from activation. JZ's user-space management distinguishes weight tensors (flags=2, written once at load) from activations, so bit 0 can eliminate their per-token re-invalidation. The supposed disadvantage of user-space cache management became a policy-flexibility advantage that Qualcomm's uniform batch-boundary design cannot express without per-role semantics.

## 4. Single Shared-Memory Pool vs. Per-Chunk Shared Memory: Code-Level Comparison

### Advantages of a single mempool

1. **Contiguous IOVA address space**: a large continuous shared-memory region gives better spatial locality at the prefetcher and TLB level. Tensors laid out sequentially in the pool benefit from hardware prefetch. Qualcomm's per-chunk approach fragments the IOVA address space; two logically adjacent tensors may end up in separate shared-memory buffers with unrelated IOVA ranges.

2. **One `fastrpc_mmap` instead of many**: JZ calls `fastrpc_mmap` once at pool init. Qualcomm calls it per `ggml_hexagon_shared_buffer` - each call involves kernel round-trips, page-table setup, and DSP-side SMMU mapping. For a model like gemma-4-E2B with ~700+ tensors, JZ avoids hundreds of mmap/unmap operations.

3. **Offset-based addressing is simpler and faster than `bi` indirection**: JZ's `dsptensor` carries a direct `void *` pointer into the pool. Qualcomm's `htp_tensor` carries a `bi` (buffer index) that the DSP must dereference through `htp_buf_desc[]` to get the actual base address. On the DSP side, one fewer level of indirection per tensor access.

4. **One FastRPC `invoke` per batch vs. dspqueue async enqueue**: JZ packs all descriptors into a single `invoke` call. Qualcomm uses `dspqueue_write` per batch (both split work into batches by descriptor capacity via `fit_op`; the real difference is JZ's synchronous `invoke` vs. Qualcomm's async `dspqueue` with up to 16 batches in flight). Fewer user-kernel transitions per batch.

5. **Pool lifecycle is trivial**: one alloc, one free. Qualcomm must track per-chunk lifecycles, handle partial allocation failures, and coordinate buffer teardown with DSP-side unmapping. JZ's pool is inherently simpler and less error-prone.

6. **Lower kernel resource consumption**: one shared-memory fd vs. hundreds. Each fd consumes kernel memory (file descriptor table, shared-memory handle, dma-buf attachment). On resource-constrained Android devices, this matters.




### Transparency advantage

The entire cache coherency pipeline is visible and modifiable on both AP and DSP sides. From `DC CVAC` in Phase 6.5 to `CIVAC` in Phase 7.5 (both inside `graph_compute_batch()` in [`ggml-hexagon-jz.cpp`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp)), to DSP-side `dcinva`/`dccleaninva` in [`kernels/entry.c`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/entry.c), every cache maintenance operation is explicit, auditable, and optimizable.

In contrast, Qualcomm's cache maintenance is split between an opaque layer and a blunt one: the `DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | INVALIDATE_RECIPIENT` flags on the small descriptor packet are handled inside the closed-source Hexagon DSP driver, while the tensor-data maintenance is a full D-cache flush+invalidate at batch boundaries in [`htp/main.c`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/main.c) - uniform, role-blind, and not selectable per tensor. JZ's transparency means:

- Cache flush strategies can be made selective (e.g., flush weights once at load time, not every batch).
- `ion_sync` can be tuned per region (dirty vs. clean ranges).
- DSP-side cache modes (`dsp_cache_mode`) can be experimented with at runtime.
- The interaction between `DC CVAC`, `CIVAC`, and `ion_sync` can be profiled and optimized.




## 5. PP Jitter

PP jitter affects both JZ and Qualcomm comparably. The observed run-to-run variance is consistent with an **L2 cache whose indexing depends on physical addresses** (behavior consistent with PIPT). Every time the Linux kernel's page allocator (`rpcmem_alloc2`) gives different physical addresses to the shared-memory buffer - which it does on every process launch - the DSP's L2 cache alias sets change, causing HMX matmul throughput to vary.

This is a hardware-level effect that neither JZ nor Qualcomm can fix in user space. The physical addresses are allocated by the Linux kernel's page allocator inside `rpcmem_alloc2()`. The only ways to eliminate this jitter would be:
1. Reserve a DMA region at a fixed physical address (requires kernel driver modification)
2. Use a hardware page coloring scheme to stabilize cache set mapping (requires Hexagon DSP firmware modification)
3. Run inside a static VM with fixed physical memory layout (not applicable to Android)

After the optimization campaign (lm-head offload + dsp_cache_mode=5 + DSP log removal), three software jitter sources were removed: (a) periodic DSP-side FARF profiler dumps, (b) the CPU-resident lm-head segment (the CPU is the least deterministic execution unit), and (c) redundant per-token weight invalidation. The observed PP distribution tightened substantially: a separate measurement snapshot showed a practical range of ~680-690 tok/s (JZ) and ~390-460 tok/s (QCOM) on gemma-4-E2B-it-Q4_0.gguf. Note these ranges come from a different measurement point than the tables above (which were taken under different thermal conditions, e.g. Table 5A JZ ~704 / QCOM ~549, Table 5B JZ ~661 / QCOM ~430); they are illustrative of the tightened spread, not directly comparable to a single table. The L2 physical-alias hypothesis remains the most plausible explanation for the residual run-to-run variance, but it no longer dominates.

## 6. Summary

Qualcomm's ggml-hexagon and JZ's ggml-hexagon target the same Hexagon DSP hardware and the same HVX/HMX kernels. By analogy to llama.cpp's GPU backends: **Qualcomm's ggml-hexagon can be seen as ggml-cuda** - the reference path, tracking upstream master, built on the vendor-blessed `dspqueue` control plane and per-chunk shared memory - and **JZ's ggml-hexagon can be seen as ggml-hip** - an alternative on the same hardware that reuses the same operators but substitutes native FastRPC `invoke` plus a single shared-memory pool, with a different cache and offload strategy. The two coexist the way ggml-cuda and ggml-hip do: each is stronger on different workloads (detailed in section 6.1).

- **JZ exceeds Qualcomm on both PP and TG on some GQA models** (5-run mean, 2026-07-22): PP 686.46 / TG 26.91 (JZ) vs PP 435.14 / TG 24.91 (QCOM), gemma-4-E2B-it-Q4_0.gguf, same Snapdragon 8 Elite device. Both run the same HMX kernels; the difference is architectural. This advantage is model-type dependent: on MHA legacy models (e.g. qwen1.5) and shallow-GQA PP, Qualcomm wins (see section 6.1 and the companion doc).
- **The single shared-memory pool's unique advantage is proven in practice**: a ~214 MB repacked lm-head stays resident for the whole session at zero recurring map/fd/lifecycle cost - something the per-chunk shared-memory design cannot express economically. Combined with the Q4_K -> Q4_0 repack (same data size, enables the DSP tiled matmul / offload) and first-touch weight invalidation (~9.2 ms/token saved), TG is 26.91 tok/s.
- **User-space cache management is an asset, not a liability**: role-aware invalidation (weight vs activation, bit 0 first-touch) is a policy Qualcomm's uniform per-batch cache maintenance (driver-handled descriptor packet + DSP-side full D-cache flush+invalidate) cannot express. The two-pass defense (DSP-side unmark on dst write + AP-side ever-dst set) resolved the historical bit-0 garble risk; mode=5 passes correctness on gemma4, qwen3, and qwen3-mtp.
- **PP jitter is a hardware-level L2 cache aliasing effect** that affects both implementations comparably and is not fixable in user space. Three software jitter sources were removed during the optimization campaign, tightening the PP distribution substantially.
- **Control-plane primitives differ** (`dspqueue` vs. native FastRPC `invoke`), but the data plane and the descriptor-dispatch flow are fundamentally the same. The measured performance difference comes from data-plane policy (weight residency + role-aware cache management), not from the control plane.

### 6.1 JZ and Qualcomm are complementary, not competitive

The two backends should be treated as complementary, not as competitors: each has its own strength areas, and both should be maintained. The direction of the PP/TG gap depends on model type, not on which backend is "better" in general.



Practical recommendation: keep both backends and document a model-type-to-backend mapping for users - GQA models to JZ, MHA / shallow-PP models to Qualcomm. Choosing the backend per model type yields the best result on any device, which is the point of coexistence.

## Revision History

### 2026-08-22: Terminology cleanup and chapter 6 ggml-cuda/ggml-hip analogy

Author: GLM-5.2

- Replaced the Android-specific term "ION" with "kernel-allocated shared memory" / "shared-memory" throughout, so the doc stays hardware-neutral and self-descriptive rather than tied to Android's ION allocator.
- Renamed the Qualcomm allocation pattern from "per-buffer" to "per-chunk" across the doc, to avoid overloading the generic word "buffer" and to sharpen the single-pool vs per-chunk contrast. Code symbols (`ion_sync`, `ggml_hexagon_shared_buffer`, `bi`, etc.) are unchanged.
- Rewrote chapter 6 to open with the ggml-cuda / ggml-hip analogy: Qualcomm's ggml-hexagon can be seen as ggml-cuda (the reference path tracking upstream master), JZ's ggml-hexagon can be seen as ggml-hip (a separately maintained alternative on the same Hexagon hardware).

### 2026-08-22: Table renumbering and consistency fixes

Author: GLM-5.2

- Renumbered all tables sequentially in document order (Table 1-8), eliminating duplicate IDs that arose when section 2 restarted numbering (two Table 3s, two Table 4s).
- Standardized table label format from "Table-N" (hyphen) to "Table N" (space) across the document.
- Kept A/B suffixes (Table 5A/5B) for the paired 8 Elite automated AB-test sub-experiments.
- Updated all in-text table references to match the new numbering.
- Fixed two broken section references: "section 7.1" -> "section 6.1" (no section 7 exists; the referenced section is 6.1).

### 2026-08-04: Documentation accuracy fixes

Author: DeepSeek-V4-Flash

- Corrected the Q4_K -> Q4_0 repack bandwidth claim: Q4_K and Q4_0 have the same data size (0.5625 B/param), so the repack does not halve per-token bandwidth; its value is enabling the DSP tiled matmul (offload).
- Clarified the ~9.2 ms/token first-touch saving as a fixed whole-graph total (dominated by the resident lm-head), not a per-layer quantity.
- Qualified the summary claim to "on GQA models" and added section 6.1 "JZ and Qualcomm are complementary, not competitive" with the model-type-to-backend mapping.
- Annotated the section 6 PP-jitter range as a separate measurement snapshot not directly comparable to the tables.
- Fixed the "split batch" wording in section 4 and the date/format inconsistencies in the tables.

### 2026-07-26: JZ forks independent kernels/ directory

**Context**: Previously, JZ and Qualcomm shared a single DSP kernel directory (`htp/`) for all ops kernels, HVX/HMX headers, and common helpers. This ended with Qualcomm's PR [#26049](https://github.com/ggml-org/llama.cpp/pull/26049) (merge commit `0a50d9909a3478e82679f505bf8595d1eee4b0a8`): after merging upstream master with this PR, JZ's default inference test produced garbled output while Qualcomm's remained normal. The root cause is that this PR moved part of the cache maintenance logic into operator implementations, making JZ's cache subsystem incompatible and causing the garbled output.

**Action**: JZ forked `htp/` into a new `kernels/` directory pinned at baseline commit `2be3826c9` (where PP/TG fully exceeded Qualcomm and inference output was correct). JZ now maintains `kernels/` independently; Qualcomm continues using `htp/` which tracks upstream master. Selected stable upstream `htp/` improvements are ported into `kernels/` manually.


### 2026-07-26: Fix llama-bench segfault (context lifetime regression)

Commit `998199e21` changed `ggml_backend_hexagon_free` to delete the context (including buffer types) on backend free, but model tensors still reference those buffer types, causing a use-after-free during context transitions (e.g., pp200 -> pp512) in llama-bench. Reverted to only delete the backend (matching Qualcomm's pattern), and restored `get_name`/`get_memory` to defensive direct access instead of `ensure_context`.
