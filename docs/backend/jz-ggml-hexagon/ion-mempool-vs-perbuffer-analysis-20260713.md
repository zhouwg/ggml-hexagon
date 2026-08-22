# The mempool/FastRPC and dspqueue ggml-hexagon variants: Architecture Analysis

> Last updated: 2026-08-22 (file name date 20260713 reflects the original creation; see Revision History)

> Author: Kimi-K2.7-Code (original), revised by Kimi-K3, GLM-5.2, Kimi-K3, GLM-5.3, Jeff Zhou


***

| Abbr. | Full name / Meaning |
|------|-------------|
| fastrpc variant | the mempool/FastRPC-invoke ggml-hexagon variant (`GGML_HEXAGON_USE_MEMPOOL=ON`; `ggml-hexagon-fastrpc.cpp` + `htp/entry.c`), formerly "JZ's ggml-hexagon" |
| dspqueue variant | the dspqueue/per-chunk ggml-hexagon variant (`GGML_HEXAGON_USE_MEMPOOL=OFF`, default; `ggml-hexagon.cpp` + `htp/main.c`), Qualcomm's official backend, formerly "QCOM" |
| PP   | Prompt Processing (prefill phase) |
| TG   | Token Generation (decode phase) |
| mempool | kernel-allocated AP-DSP shared memory (single allocation, offset addressing) |
| baseline/headline | branch `self-build-jz` |

## 1. fastrpc variant vs dspqueue variant: Architecture Comparison

Both variants route through the shared `execute_op` dispatch path. As of 2026-08-22 there is ONE DSP ops source tree, `ggml/src/ggml-hexagon/htp/`: the `kernels/` directory forked on 2026-07-26 (pinned at self-build-jz `2be3826c9`) was merged back and deleted, and the mempool-specific code paths inside the ops are guarded by `GGML_HEXAGON_USE_MEMPOOL`. The variants differ in DSP entry point (`htp/entry.c` vs `htp/main.c`) and AP-side code (`ggml-hexagon-fastrpc.cpp` vs `ggml-hexagon.cpp`), selected at build time by `GGML_HEXAGON_USE_MEMPOOL`. Both variants produce the same pair of output artifacts: the AP-side backend library `libggml-hexagon.so` and the DSP-side (NPU) skel `libggml-htp-vXX.so` - the build option changes what runs inside them, not their names. The AP-side implementations differ significantly, leading to performance differences.



The fastrpc variant (`ggml-hexagon-fastrpc.cpp` + `htp/entry.c`) and the dspqueue variant (`ggml-hexagon.cpp` + `htp/main.c`) are **two build-time variants of one shared hexagon ops tree**. The tree was temporarily forked (2026-07-26, `kernels/`) after Qualcomm [PR #26049](https://github.com/ggml-org/llama.cpp/pull/26049) broke the fastrpc variant's inference with garbled output; the fork was merged back on 2026-08-22.

- Before PR #26049, both had identical operators.
- While the fork existed, operator improvements under `htp/` were manually ported into `kernels/`.
- Since the 2026-08-22 merge, both variants compile the same `htp/*.c` op sources; only the entry point and the mempool-guarded code paths differ.
- This eliminates the cost of maintaining two kernel trees: an upstream operator improvement lands once in `htp/` and benefits both variants - no manual porting, no porting lag, no risk of the fork silently drifting behind.
- **The performance difference is not in the kernel operators themselves, but in the scheduling framework, cache strategy, and offload strategy.**

### 1.1 Core Architecture Differences

- **fastrpc variant**: native FastRPC `invoke` + single mempool (offset addressing)
- **dspqueue variant**: `dspqueue` + per-chunk shared buffers (`bi` indirect addressing)

**Table 1**: Architecture comparison

| Dimension | fastrpc variant | dspqueue variant |
| ------ | ------------------------ | ----------------------------------------------------------- |
| Control plane | Native FastRPC `invoke` (synchronous) | `dspqueue_write/read` (asynchronous, up to 16 concurrent batches) |
| Data plane    | single mempool + offset addressing     | per-chunk + `bi` (buffer index) indirect addressing            |
| DSP entry | `htp/entry.c`                          | `htp/main.c`                                                |
| DSP ops   | `htp/*.c` (shared single source tree)  | `htp/*.c` (shared single source tree)                        |
| AP-side code    | `ggml-hexagon-fastrpc.cpp`            | `ggml-hexagon.cpp`                                          |
| Build option    | `GGML_HEXAGON_USE_MEMPOOL=ON`         | `GGML_HEXAGON_USE_MEMPOOL=OFF` (default)                    |
| DSP skel output | `libggml-htp-vXX.so`                  | `libggml-htp-vXX.so` (same name)                            |
| Cache coherency | User-space: role-aware (`ion_sync` + `dsp_cache_mode`) | Kernel-space driver flags (uniform per batch)             |

**Table 2**: single mempool vs per-chunk multiple shared buffers code-level comparison

| Dimension | fastrpc variant | dspqueue variant | Winner |
| --- | --- | --- | --- |
| `fastrpc_mmap` call count | single mempool: 1 at init | 1 per chunk | fastrpc |
| fd count | 1 | 1 per chunk | fastrpc |
| DSP tensor addressing | direct `void *` offset | `bi` -> `htp_buf_desc[]` indirect addressing | fastrpc |
| Batch transport | `invoke` carries the entire graph batch | `dspqueue_write` | Tie |
| Memory lifecycle | single alloc/free | per-chunk alloc/mmap + munmap/free | fastrpc |
| IOVA spatial locality (prefetch/TLB) | contiguous, predictable | fragmented across chunks | fastrpc |
| Cache coherency | User-space: role-aware (weight vs activation); `ion_sync` + `dsp_cache_mode` | driver flushes descriptor packet + DSP-side full D-cache flush+invalidate per batch (uniform, role-blind) | fastrpc (role-aware policy flexibility) |
| Physical address stability | stable after allocation (no migration) | stable after allocation (no migration) | Tie |
| lm-head offload | feasible (within mempool offset range) | infeasible (per-chunk fd/mmap/lifecycle overhead) | fastrpc |



The dspqueue variant's disadvantages - per-chunk fd count, per-chunk mmap calls, DSP-side `bi` indirection, multi-write batch transport - are **overhead inherent to the per-chunk API design**. Every additional buffer requires another fd, another mmap, and another `htp_buf_desc[]` entry at the interface level. The fastrpc variant's single pool has none of these per-chunk costs.

### 1.2 Control-Plane Primitive Differences: dspqueue vs. Native FastRPC invoke

**Table 3**: Control plane comparison

| Dimension | fastrpc variant | dspqueue variant |
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

The dspqueue variant's DSP entry point [`htp/main.c`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/main.c) parses `htp_buf_desc[]`, `htp_tensor[]`, and `htp_op_desc[]` via `htp_packet_callback` before dispatching to kernels. The fastrpc variant completes the same flow in [`htp/entry.c`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/entry.c). Both data paths are identical:

```
AP packs descriptors -> transport to DSP -> DSP entry parses descriptors -> dispatch op execution
```

### 1.4 Source File Structure

**Table 4**: Source file structure (single ops tree, two build variants)

| File                                                            | Description                                                                       |
| ------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `ggml/src/ggml-hexagon/ggml-hexagon-fastrpc.cpp`              | fastrpc variant AP-side code                                                      |
| `ggml/src/ggml-hexagon/ggml-hexagon.cpp`                      | dspqueue variant AP-side code                                     |
| `ggml/src/ggml-hexagon/CMakeLists.txt`                        | Unified AP-side build (variant selected by `GGML_HEXAGON_USE_MEMPOOL`)                    |
| `ggml/src/ggml-hexagon/htp/CMakeLists.txt`                    | Unified DSP skel build for BOTH variants (ExternalProject per arch)                           |
| `ggml/src/ggml-hexagon/htp/entry.c`                           | fastrpc variant DSP entry                                              |
| `ggml/src/ggml-hexagon/htp/dsp-ctx.h`                         | fastrpc variant DSP session context + descriptors                                    |
| `ggml/src/ggml-hexagon/htp/ggml_dsp.idl`                      | fastrpc variant FastRPC IDL                                    |
| `ggml/src/ggml-hexagon/htp/main.c`                            | dspqueue variant DSP entry                                        |
| `ggml/src/ggml-hexagon/htp/htp-ctx.h`                         | dspqueue variant DSP session context + mmap/spad                                    |
| `ggml/src/ggml-hexagon/htp/htp_iface.idl`                     | dspqueue variant IDL                                    |
| `ggml/src/ggml-hexagon/htp/*.c`                               | shared DSP kernels (single source tree; mempool-specific bits guarded by `GGML_HEXAGON_USE_MEMPOOL`)                           |



Both the AP-side and DSP-side builds are selected by one CMake option:

```cmake
option(GGML_HEXAGON_USE_MEMPOOL "ggml-hexagon: mempool/FastRPC-invoke AP implementation (single ION mempool, synchronous invoke); OFF = dspqueue/per-buffer implementation" OFF)
```

- `GGML_HEXAGON_USE_MEMPOOL=OFF` (default): dspqueue variant (`ggml-hexagon.cpp`, `htp/main.c` + `htp_iface.idl`), upstream behavior.
- `GGML_HEXAGON_USE_MEMPOOL=ON`: fastrpc variant (`ggml-hexagon-fastrpc.cpp`, `htp/entry.c` + `ggml_dsp.idl`, qaic-generated `ggml_dsp_stub.c` on the AP side).
- Both variants build their DSP skels via the same `ExternalProject` mechanism in `htp/CMakeLists.txt` and produce identically named `libggml-htp-vXX.so` skels (the fastrpc variant's skel was formerly `libggmldsp-skel-vXX.so`, renamed 2026-08-22).
- fastrpc-variant skels land in `${CMAKE_BINARY_DIR}/bin/`; dspqueue-variant skels land in the `ggml/src/ggml-hexagon/` build subdir (the CI script's `detect_build_type` distinguishes the two builds by this location).
- The old `kernels/Makefile` and `scripts/build-kernels.sh` standalone skel build were removed; the normal CMake build covers both variants.

## 2. Performance Comparison

Pls refer to section-6 Benchmark Results in [https://github.com/ggml-org/llama.cpp/discussions/26227](https://github.com/ggml-org/llama.cpp/discussions/26227).

## 3. The Decisive Single-Pool Advantage: Session-Resident Repacked Weights

### Background

TG (token generation) is bandwidth-bound on both backends: every token re-reads all weights from DRAM. The lm-head matmul (262144 x 1536, Q4_K, ~30 ms/token on CPU) was the single largest TG cost. Both variants previously rejected quantized weight matrices with `ne[1] > 32768`, so lm-head ran on the CPU in both variants.

### Why per-chunk shared-memory cannot fix this economically

Offloading lm-head to the DSP requires a repacked (tiled) copy of the weight to live in DSP-addressable memory for the entire session. Under the dspqueue variant's per-chunk design, every `ggml_hexagon_shared_buffer` carries its own shared-memory fd, its own `fastrpc_mmap`, per-batch descriptor re-registration (`add_buffer()`), a DSP-side mmap slot out of a limited vmem budget (`prep_op_bufs` in `htp/main.c` evicts and re-maps under pressure), and a lifecycle that must be coordinated with DSP-side unmapping. A ~214 MB single-purpose resident buffer pays all of these recurring per-chunk costs, which is consistent with the dspqueue variant keeping the 32768-row guard in place and lm-head on the CPU.

### Why the single pool makes it natural

The fastrpc variant maps the pool once at init (`fastrpc_mmap`, capacity probed up to 4032 MiB on v79; see `ggmlhexagon_init_rpcmempool()`). Repacking lm-head into the pool at load time costs one conversion pass; afterwards the repack is just an offset range inside an already-mapped region - zero recurring fd/mmap/lifecycle cost - and each token simply streams it from DRAM. The apparent constraint (one pool, no per-chunk granularity) is exactly what makes a ~214 MB resident repack cheap.

### The three changes, all enabled by the pool

1. **Removed the `ne[1] > 32768` guard** for quantized weights in `ggmlhexagon_supported_mul_mat`, allowing lm-head to offload.
2. **Q4_K stored as Q4_0 tiled repack** (`repack_q4k_as_q4_0_tiled_to_buf` in [`ggml-hexagon-fastrpc.cpp`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-fastrpc.cpp), 32-row strip conversion; inverse transform in `get_tensor` for host reads): the resident repacked lm-head is ~214 MB. Note the repack does **not** reduce bandwidth - Q4_K and Q4_0 have the same data size (both 0.5625 B/param), so per-token DRAM traffic is unchanged. Its value is that it turns the Q4_K weight into a tiled Q4_0 layout the DSP can execute directly, which is what makes the offload possible.
3. **First-touch weight invalidation** (`dsp_cache_mode` bit 0, default mode=5). With lm-head resident, per-token weight traffic grew to ~1.9 GB and re-invalidating it every token cost ~9.2 ms/token of DSP-side dcinva sweeping. Repack weights are written once by the AP at load time and never touched again, so after a first-touch invalidate the DSP skips re-invalidation for the rest of the session, removing the ~9.2 ms entirely. The two-pass defense (DSP-side `weight_inval_unmark()` on dst write plus an AP-side `g_ever_dst_ptrs` set) closes cross-graph stale reads.

Removing DSP-side debug/profiler logging afterwards (`-DNDEBUG` skel build) brought further PP/TG gains.

### Side effect: cache-coherency advantage flipped

The dspqueue variant's per-batch cache maintenance is uniform and role-blind: the driver's flush/invalidate flags cover only the small dspqueue descriptor packet, while tensor data is handled by a full D-cache flush+invalidate on the DSP at batch start and end ([`htp/main.c`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/main.c), lines 999 and 1078) - neither can distinguish weight from activation. The fastrpc variant's user-space management distinguishes weight tensors (flags=2, written once at load) from activations, so bit 0 can eliminate their per-token re-invalidation. The supposed disadvantage of user-space cache management became a policy-flexibility advantage that the dspqueue variant's uniform batch-boundary design cannot express without per-role semantics.

## 4. Single Shared-Memory Pool vs. Per-Chunk Shared Memory: Code-Level Comparison

### Advantages of a single mempool

1. **Contiguous IOVA address space**: a large continuous shared-memory region gives better spatial locality at the prefetcher and TLB level. Tensors laid out sequentially in the pool benefit from hardware prefetch. The dspqueue variant's per-chunk approach fragments the IOVA address space; two logically adjacent tensors may end up in separate shared-memory buffers with unrelated IOVA ranges.

2. **One `fastrpc_mmap` instead of many**: the fastrpc variant calls `fastrpc_mmap` once at pool init. The dspqueue variant calls it per `ggml_hexagon_shared_buffer` - each call involves kernel round-trips, page-table setup, and DSP-side SMMU mapping. For a model like gemma-4-E2B with ~700+ tensors, the fastrpc variant avoids hundreds of mmap/unmap operations.

3. **Offset-based addressing is simpler and faster than `bi` indirection**: the fastrpc variant's `dsptensor` carries a direct `void *` pointer into the pool. The dspqueue variant's `htp_tensor` carries a `bi` (buffer index) that the DSP must dereference through `htp_buf_desc[]` to get the actual base address. On the DSP side, one fewer level of indirection per tensor access.

4. **One FastRPC `invoke` per batch vs. dspqueue async enqueue**: the fastrpc variant packs all descriptors into a single `invoke` call. The dspqueue variant uses `dspqueue_write` per batch (both split work into batches by descriptor capacity via `fit_op`; the real difference is the fastrpc variant's synchronous `invoke` vs. the dspqueue variant's async `dspqueue` with up to 16 batches in flight). Fewer user-kernel transitions per batch.

5. **Pool lifecycle is trivial**: one alloc, one free. The dspqueue variant must track per-chunk lifecycles, handle partial allocation failures, and coordinate buffer teardown with DSP-side unmapping. The fastrpc variant's pool is inherently simpler and less error-prone.

6. **Lower kernel resource consumption**: one shared-memory fd vs. hundreds. Each fd consumes kernel memory (file descriptor table, shared-memory handle, dma-buf attachment). On resource-constrained Android devices, this matters.




### Transparency advantage

The entire cache coherency pipeline is visible and modifiable on both AP and DSP sides. From `DC CVAC` in Phase 6.5 to `CIVAC` in Phase 7.5 (both inside `graph_compute_batch()` in [`ggml-hexagon-fastrpc.cpp`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-fastrpc.cpp)), to DSP-side `dcinva`/`dccleaninva` in [`htp/entry.c`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/entry.c), every cache maintenance operation is explicit, auditable, and optimizable.

In contrast, the dspqueue variant's cache maintenance is split between an opaque layer and a blunt one: the `DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | INVALIDATE_RECIPIENT` flags on the small descriptor packet are handled inside the closed-source Hexagon DSP driver, while the tensor-data maintenance is a full D-cache flush+invalidate at batch boundaries in [`htp/main.c`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/main.c) - uniform, role-blind, and not selectable per tensor. The fastrpc variant's transparency means:

- Cache flush strategies can be made selective (e.g., flush weights once at load time, not every batch).
- `ion_sync` can be tuned per region (dirty vs. clean ranges).
- DSP-side cache modes (`dsp_cache_mode`) can be experimented with at runtime.
- The interaction between `DC CVAC`, `CIVAC`, and `ion_sync` can be profiled and optimized.




## 5. PP Jitter

PP jitter affects both variants comparably. The observed run-to-run variance is consistent with an **L2 cache whose indexing depends on physical addresses** (behavior consistent with PIPT). Every time the Linux kernel's page allocator (`rpcmem_alloc2`) gives different physical addresses to the shared-memory buffer - which it does on every process launch - the DSP's L2 cache alias sets change, causing HMX matmul throughput to vary.

This is a hardware-level effect that neither variant can fix in user space. The physical addresses are allocated by the Linux kernel's page allocator inside `rpcmem_alloc2()`. The only ways to eliminate this jitter would be:
1. Reserve a DMA region at a fixed physical address (requires kernel driver modification)
2. Use a hardware page coloring scheme to stabilize cache set mapping (requires Hexagon DSP firmware modification)
3. Run inside a static VM with fixed physical memory layout (not applicable to Android)

Not all jitter is hardware. The optimization campaign (section 3) removed three software jitter sources: (a) periodic DSP-side FARF profiler dumps, (b) the CPU-resident lm-head segment (the CPU is the least deterministic execution unit), and (c) redundant per-token weight invalidation. After their removal the PP distribution tightened substantially; the residual run-to-run variance is consistent with the L2 physical-alias effect above, but on a much smaller spread.

## 6. Summary

The dspqueue variant and the fastrpc variant target the same Hexagon DSP hardware and the same HVX/HMX kernels (since the 2026-08-22 merge they even share one ops source tree). By analogy to llama.cpp's GPU backends: **the dspqueue variant can be seen as ggml-cuda** - the reference path, tracking upstream master, built on the vendor-blessed `dspqueue` control plane and per-chunk shared memory - and **the fastrpc variant can be seen as ggml-hip** - an alternative on the same hardware that reuses the same operators but substitutes native FastRPC `invoke` plus a single shared-memory pool, with a different cache and offload strategy. The two coexist the way ggml-cuda and ggml-hip do: each is stronger on different workloads.

- **The fastrpc variant exceeds the dspqueue variant on both PP and TG on some GQA models** (5-run mean, 2026-07-22): PP 686.46 / TG 26.91 (fastrpc) vs PP 435.14 / TG 24.91 (dspqueue), gemma-4-E2B-it-Q4_0.gguf, same Snapdragon 8 Elite device. Both run the same HMX kernels; the difference is architectural. This advantage is model-type dependent: on MHA legacy models (e.g. qwen1.5) and shallow-GQA PP, the dspqueue variant wins (see the companion doc).
- **The single shared-memory pool's unique advantage is proven in practice**: a ~214 MB repacked lm-head stays resident for the whole session at zero recurring map/fd/lifecycle cost - something the per-chunk shared-memory design cannot express economically.
- **User-space cache management is an asset, not a liability**: role-aware invalidation (weight vs activation, bit 0 first-touch) is a policy the dspqueue variant's uniform per-batch cache maintenance (driver-handled descriptor packet + DSP-side full D-cache flush+invalidate) cannot express.
- **PP jitter is a hardware-level L2 cache aliasing effect** that affects both implementations comparably and is not fixable in user space.
- **Control-plane primitives differ** (`dspqueue` vs. native FastRPC `invoke`), but the data plane and the descriptor-dispatch flow are fundamentally the same. The measured performance difference comes from data-plane policy (weight residency + role-aware cache management), not from the control plane.

Taken together, these findings say the two variants are complementary, not competitive: the direction of the PP/TG gap depends on model type, not on which variant is "better" in general, so both should be maintained. Practical recommendation: document a model-type-to-variant mapping for users - GQA models to the fastrpc variant (`GGML_HEXAGON_USE_MEMPOOL=ON`, CI `build`), MHA / shallow-PP models to the dspqueue variant (default, CI `build_dspqueue`). Choosing the variant per model type yields the best result on any device, which is the point of coexistence.

## Revision History

### 2026-08-22: kernels/ merged back into htp/; single ops tree with two build variants

Author: GLM-5.3

- Merged the forked `kernels/` directory back into `htp/` and deleted it: `htp/` is now the single shared DSP ops source tree for both variants; mempool-specific code paths inside the ops are guarded by `GGML_HEXAGON_USE_MEMPOOL` at compile time.
- Renamed the build option `GGML_HEXAGON_JZ` to `GGML_HEXAGON_USE_MEMPOOL` (default OFF) and the AP-side source `ggml-hexagon-jz.cpp` to `ggml-hexagon-fastrpc.cpp`.
- Unified the DSP skel build: `htp/CMakeLists.txt` builds both variants via the same `ExternalProject` mechanism (`entry.c` + `ggml_dsp.idl` for mempool, `main.c` + `htp_iface.idl` for dspqueue); both produce identically named `libggml-htp-vXX.so` skels (the mempool variant's skel was formerly `libggmldsp-skel-vXX.so`). Removed the standalone `kernels/Makefile` / `scripts/build-kernels.sh` path.
- Renamed CI commands and labels to technical names: `build_dspqueue` (formerly `build_qcom`), `update_fastrpc_libs` / `update_dspqueue_libs` (formerly `update_jz_libs` / `update_qcom_libs`); AB-test backups are now `*-fastrpc.so` / `*-dspqueue.so`.
- Updated this doc's terminology to match: "JZ"/"QCOM" in the technical body replaced by "fastrpc variant"/"dspqueue variant"; JZ/Qualcomm names are kept only for lineage and history (PR attribution, older revision entries). Also fixed stale references: `kernels/` paths -> `htp/`, "Table 5A/5B" -> "the since-removed inline perf tables".
- Folded the former section 6.1 (complementarity + model-type-to-variant recommendation) into chapter 6 as its closing paragraph, removing the lone subsection; removed the now-dangling "section 6.1" cross-references.

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
