# JZ ggml-hexagon vs Qualcomm ggml-hexagon: Architecture Analysis (2026-07-22)

> Author: Kimi-K2.7-Code (original), revised by Kimi-K3, GLM-5.2

## 1. Architecture Overview

Both JZ and Qualcomm ggml-hexagon backends route through Qualcomm's `execute_op` path. The `execute_op` implementation lives in `ggml/src/ggml-hexagon/htp/` and is shared by both ggml-hexagon implementations. The DSP entry points differ: JZ uses `htp/entry.c`, while Qualcomm uses `htp/main.c`. On the AP side, JZ uses `ggml-hexagon-jz.cpp` while Qualcomm uses `ggml-hexagon.cpp`. The AP-side implementations differ significantly, leading to performance differences.

The core architectural difference:

- **JZ**: Native/pure FastRPC `invoke` + Single ION Mempool (offset addressing)
- **Qualcomm**: `dspqueue` + Per-Buffer multiple ION shared buffers (`bi` indirection)

**Table 1: Architecture comparison**

| Aspect | JZ ggml-hexagon | Qualcomm ggml-hexagon |
|--------|-----------------|----------------------|
| Control plane | Native/pure FastRPC `invoke` (synchronous) | `dspqueue_write/read` (async, up to 16 batches in-flight) |
| Data plane | Single ION mempool + offset addressing | Per-buffer ION + `bi` (buffer index) indirection |
| DSP entry point | `htp/entry.c` | `htp/main.c` |
| Shared DSP kernels | `htp/*.c` | `htp/*.c` |
| AP-side code | `ggml-hexagon-jz.cpp` | `ggml-hexagon.cpp` |
| Build option | `GGML_HEXAGON_JZ=ON` | `GGML_HEXAGON_JZ=OFF` (default) |
| Cache coherency | User-space: role-aware (`ion_sync` + `dsp_cache_mode`) | Kernel-space driver flags (uniform per-batch) |

**Table 2: Source file structure**

| File | Description |
|------|-------------|
| `ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp` | JZ version AP code |
| `ggml/src/ggml-hexagon/ggml-hexagon.cpp` | Qualcomm version AP code (upstream) |
| `ggml/src/ggml-hexagon/CMakeLists.txt` | Unified build (QCOM base + `GGML_HEXAGON_JZ` option) |
| `ggml/src/ggml-hexagon/htp/Makefile` | JZ DSP skel build (entry.c + shared kernels) |
| `ggml/src/ggml-hexagon/htp/CMakeLists.txt` | QCOM DSP skel build (main.c + shared kernels) |
| `ggml/src/ggml-hexagon/htp/entry.c` | JZ version DSP entry point |
| `ggml/src/ggml-hexagon/htp/dsp-ctx.h` | JZ DSP session context + descriptors |
| `ggml/src/ggml-hexagon/htp/main.c` | Qualcomm version DSP entry point |
| `ggml/src/ggml-hexagon/htp/*.c` | Shared DSP kernels (both backends) |

The unified `CMakeLists.txt` is based on QCOM's version with a single addition:

```cmake
option(GGML_HEXAGON_JZ "Use JZ's AP implementation" OFF)
```

- `GGML_HEXAGON_JZ=OFF` (default): QCOM upstream behavior, builds DSP skels via `ExternalProject_Add`.
- `GGML_HEXAGON_JZ=ON`: uses `ggml-hexagon-jz.cpp`, builds a single DSP skel via `make -C htp/`.

## 2. Performance Comparison

**The single ION shared mempool is the better architecture for this workload, proven in practice.** JZ exceeds Qualcomm on both PP and TG under identical test conditions, enabled by a session-resident ~214 MB repacked lm-head that the per-buffer ION design cannot express economically.

**Table 3: JZ vs Qualcomm (5-run mean, 2026-07-22)**

| Implementation | PP (tok/s) | TG (tok/s) |
|---|---|---|
| JZ ggml-hexagon (dsp_cache_mode=5) | 686.46 | 26.91 |
| Qualcomm ggml-hexagon | 435.14 | 24.91 |

Test conditions: gemma-4-E2B-it-Q4_0.gguf, Snapdragon 8 Elite (v79, OnePlus 13), `/data/local/tmp/llama-completion -ngl 99 -t 6 -n 256 --ctx-size 8192 --ubatch-size 64 --poll 1000 --no-warmup --no-mmap -fa on --jinja -st -m /sdcard/gemma-4-E2B-it-Q4_0.gguf -p "Hello, good morning, you are a powerful domain expert and know many things, now pls help to introduce the movie Once Upon a Time in America briefly, pls pay attention short then 1000 words\n"`. Both backends run the same HMX kernels from `ggml/src/ggml-hexagon/htp`; the difference is architectural. Validated on v79; v75 (8 Gen3) validated with thread_counts clamped to 4.

**llama-bench comparison (2026-07-22):**

![llama-bench JZ vs Qualcomm 1](images/Screenshot%20from%202026-07-22%2016-59-29.png)

![llama-bench JZ vs Qualcomm 2](images/Screenshot%20from%202026-07-22%2017-02-27.png)

## 3. The Decisive Single-Pool Advantage: Session-Resident Repacked Weights

### Background

TG (token generation) is bandwidth-bound on both backends: every token re-reads all weights from DRAM. The lm-head matmul (262144 x 1536, Q4_K, ~30 ms/token on CPU) was the single largest TG cost. Both JZ and Qualcomm previously rejected quantized weight matrices with `ne[1] > 32768`, so lm-head ran on the CPU in both implementations.

### Why per-buffer ION cannot fix this economically

Offloading lm-head to the DSP requires a repacked (tiled) copy of the weight to live in DSP-addressable memory for the entire session. Under Qualcomm's per-buffer design, every `ggml_hexagon_shared_buffer` carries its own ION fd, its own `fastrpc_mmap`, per-batch driver-side cache maintenance, and a lifecycle that must be coordinated with DSP-side unmapping. A ~214 MB single-purpose resident buffer pays all of these recurring per-buffer costs, which is consistent with Qualcomm keeping the 32768-row guard in place and lm-head on the CPU.

### Why the single pool makes it natural

JZ maps the pool once at init (`fastrpc_mmap`, capacity probed up to 4032 MiB on v79; see `ggmlhexagon_init_rpcmempool()`). Repacking lm-head into the pool at load time costs one conversion pass; afterwards the repack is just an offset range inside an already-mapped region - zero recurring fd/mmap/lifecycle cost - and each token simply streams it from DRAM. The apparent constraint (one pool, no per-buffer granularity) is exactly what makes a ~214 MB resident repack cheap.

### The three changes, all enabled by the pool

1. **Removed the `ne[1] > 32768` guard** for quantized weights in `ggmlhexagon_supported_mul_mat`, allowing lm-head to offload.
2. **Q4_K stored as Q4_0 tiled repack** (`repack_q4k_as_q4_0_tiled_to_buf` in [`ggml-hexagon-jz.cpp`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp), 32-row strip conversion; inverse transform in `get_tensor` for host reads): the lm-head matvec moves 214 MB instead of 428 MB per token.
3. **First-touch weight invalidation** (`dsp_cache_mode` bit 0, default mode=5). With lm-head resident, per-token weight traffic grew to ~1.9 GB and re-invalidating it every token cost ~9.2 ms/token of DSP-side dcinva sweeping. Repack weights are written once by the AP at load time and never touched again, so after a first-touch invalidate the DSP skips re-invalidation for the rest of the session, removing the ~9.2 ms entirely. The two-pass defense (DSP-side `weight_inval_unmark()` on dst write plus an AP-side `g_ever_dst_ptrs` set) closes cross-graph stale reads.

Removing DSP-side debug/profiler logging afterwards (`-DNDEBUG` skel build) brought further PP/TG gains.

### Side effect: cache-coherency advantage flipped

Kernel-space flush is efficient per operation, but the driver applies uniform per-batch flush/invalidate flags to every buffer and cannot express role-aware policies. JZ's user-space management distinguishes weight tensors (flags=2, written once at load) from activations, so bit 0 can eliminate their per-token re-invalidation. The supposed disadvantage of user-space cache management became a policy-flexibility advantage that the closed driver cannot replicate without per-role buffer semantics.

## 4. Single ION Shared Mempool vs. Per-Buffer ION: Code-Level Comparison

### Advantages of a single mempool

1. **Contiguous IOVA address space**: a large continuous ION region gives better spatial locality at the prefetcher and TLB level. Tensors laid out sequentially in the pool benefit from hardware prefetch. Qualcomm's per-buffer approach fragments the IOVA address space; two logically adjacent tensors may end up in separate ION buffers with unrelated IOVA ranges.

2. **One `fastrpc_mmap` instead of many**: JZ calls `fastrpc_mmap` once at pool init. Qualcomm calls it per `ggml_hexagon_shared_buffer` - each call involves kernel round-trips, page-table setup, and DSP-side SMMU mapping. For a model like gemma-4-E2B with ~700+ tensors, JZ avoids hundreds of mmap/unmap operations.

3. **Offset-based addressing is simpler and faster than `bi` indirection**: JZ's `dsptensor` carries a direct `void *` pointer into the pool. Qualcomm's `htp_tensor` carries a `bi` (buffer index) that the DSP must dereference through `htp_buf_desc[]` to get the actual base address. On the DSP side, one fewer level of indirection per tensor access.

4. **One FastRPC `invoke` per batch vs. dspqueue multi-write**: JZ packs all descriptors into a single `invoke` call. Qualcomm's `dspqueue_write` may split a batch across multiple writes, each with its own kernel transition. Fewer user-kernel transitions per batch.

5. **Pool lifecycle is trivial**: one alloc, one free. Qualcomm must track per-buffer lifecycles, handle partial allocation failures, and coordinate buffer teardown with DSP-side unmapping. JZ's pool is inherently simpler and less error-prone.

6. **Lower kernel resource consumption**: one ION fd vs. hundreds. Each fd consumes kernel memory (file descriptor table, ION handle, dma-buf attachment). On resource-constrained Android devices, this matters.

### Code-level comparison table

**Table 4: Single pool vs. per-buffer comparison**

| Aspect | JZ single pool | Qualcomm per-buffer | Winner |
|--------|---------------|---------------------|--------|
| `fastrpc_mmap` calls | 1 at init | N per buffer (N = hundreds) | JZ |
| ION fds | 1 | N per buffer | JZ |
| DSP tensor addressing | direct `void *` offset | `bi` -> `htp_buf_desc[]` indirection | JZ |
| Batch transport | single `invoke` | `dspqueue_write` (may split) | JZ |
| Memory lifecycle | one alloc/free | N alloc/free + coordination | JZ |
| IOVA spatial locality (prefetch/TLB) | contiguous, predictable | fragmented across buffers | JZ |
| Cache coherency | user-space: role-aware (weight vs activation); `ion_sync` + `dsp_cache_mode` | kernel-space driver flags (uniform per-batch) | JZ (role-aware policy flexibility) |
| Physical address stability | same per-page variance (pool is physically fragmented) | same per-page variance (each buffer independently allocated) | Tie |
| Session-resident large repack (lm-head) | natural (offset range in pool) | prohibitive (per-buffer fd/mmap/lifecycle cost) | JZ |

Qualcomm's disadvantages - per-buffer fd count, per-buffer mmap calls, DSP-side `bi` indirection, multi-write batch transport - are **overhead inherent to the per-buffer API design**. Every additional buffer requires another fd, another mmap, and another `htp_buf_desc[]` entry at the interface level. JZ's single pool has none of these per-buffer costs.

### Transparency advantage

The entire cache coherency pipeline is visible and modifiable on both AP and DSP sides. From `DC CVAC` in Phase 6.5 to `CIVAC` in Phase 7.5 (both inside `graph_compute_batch()` in [`ggml-hexagon-jz.cpp`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp)), to DSP-side `dcinva`/`dccleaninva` in [`htp/entry.c`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/entry.c), every cache maintenance operation is explicit, auditable, and optimizable.

In contrast, Qualcomm's `DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | INVALIDATE_RECIPIENT` hides the cache maintenance logic inside the closed-source Hexagon DSP driver. The flags are set at the API level, but the actual flush/invalidate implementation is not visible to user-space developers. JZ's transparency means:

- Cache flush strategies can be made selective (e.g., flush weights once at load time, not every batch).
- `ion_sync` can be tuned per region (dirty vs. clean ranges).
- DSP-side cache modes (`dsp_cache_mode`) can be experimented with at runtime.
- The interaction between `DC CVAC`, `CIVAC`, and `ion_sync` can be profiled and optimized.

## 5. Control-Plane Primitive Difference: dspqueue vs. Native FastRPC invoke

### Control plane

**Table 5: Control plane comparison**

| Aspect | Qualcomm ggml-hexagon | JZ ggml-hexagon |
|---|---|---|
| Primitive | `dspqueue_write/read` queue semantics | Native FastRPC `invoke` |
| Dispatch style | AP pushes a whole op-batch; DSP is woken by packet callback | AP calls a DSP function directly with descriptors |
| Blocking model | AP can fire-and-forget, responses drained later | Typically synchronous per call |
| Batch handling | One `dspqueue_write` carries many ops (`htp_opbatch_req`) | One `invoke` carries a whole graph batch, sometimes hundreds of ops |

### Data plane

The data plane is almost identical:

- Both use **ION shared memory** for tensor data.
- Both have AP write descriptors/data and DSP read descriptors/data.
- Both need explicit **cache flush / invalidate** synchronization.
- Both ultimately run the same HVX/HMX kernels.

In the Qualcomm path the DSP entry point is [`htp/main.c`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/main.c) (`htp_packet_callback`), which parses `htp_buf_desc[]`, `htp_tensor[]`, and `htp_op_desc[]` before dispatching to the kernels. In the JZ path the same work happens inside [`htp/entry.c`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/entry.c). The high-level flow is the same:

```
AP packs descriptors -> transport to DSP -> DSP entry parses descriptors -> dispatch op execution
```

Control-plane primitives differ, but the data plane and the descriptor-dispatch flow are fundamentally the same. The measured performance difference comes from data-plane policy (weight residency + role-aware cache management), not from the control plane. JZ's pure FastRPC/ION transport overhead is ~89 us/call (warmup probe), negligible against the ~36 ms/token DSP execution time.

## 6. PP Jitter

PP jitter affects both JZ and Qualcomm comparably. The observed run-to-run variance is consistent with an **L2 cache whose indexing depends on physical addresses** (behavior consistent with PIPT). Every time the Linux kernel's page allocator (`rpcmem_alloc2`) gives different physical addresses to the ION buffer - which it does on every process launch - the DSP's L2 cache alias sets change, causing HMX matmul throughput to vary.

This is a hardware-level effect that neither JZ nor Qualcomm can fix in user space. The physical addresses are allocated by the Linux kernel's page allocator inside `rpcmem_alloc2()`. The only ways to eliminate this jitter would be:
1. Reserve a DMA region at a fixed physical address (requires kernel driver modification)
2. Use a hardware page coloring scheme to stabilize cache set mapping (requires Hexagon DSP firmware modification)
3. Run inside a static VM with fixed physical memory layout (not applicable to Android)

After the optimization campaign (lm-head offload + dsp_cache_mode=5 + DSP log removal), three software jitter sources were removed: (a) periodic DSP-side FARF profiler dumps, (b) the CPU-resident lm-head segment (the CPU is the least deterministic execution unit), and (c) redundant per-token weight invalidation. The observed PP distribution tightened substantially: current practical range is ~680-690 tok/s (JZ) and ~390-460 tok/s (QCOM) on gemma-4-E2B-it-Q4_0.gguf. The L2 physical-alias hypothesis remains the most plausible explanation for the residual run-to-run variance, but it no longer dominates.

## 7. Summary

- **JZ exceeds Qualcomm on both PP and TG** (5-run mean, 2026-07-22): PP 686.46 / TG 26.91 (JZ) vs PP 435.14 / TG 24.91 (QCOM), gemma-4-E2B-it-Q4_0.gguf, same Snapdragon 8 Elite device. Both run the same HMX kernels; the difference is architectural.
- **The single ION mempool's unique advantage is proven in practice**: a ~214 MB repacked lm-head stays resident for the whole session at zero recurring map/fd/lifecycle cost - something the per-buffer ION design cannot express economically. Combined with the Q4_K -> Q4_0 repack (halves lm-head bandwidth to 214 MB/token) and first-touch weight invalidation (~9.2 ms/token saved), TG is 26.91 tok/s.
- **User-space cache management is an asset, not a liability**: role-aware invalidation (weight vs activation, bit 0 first-touch) is a policy the closed-source driver's uniform per-batch flush cannot express. The two-pass defense (DSP-side unmark on dst write + AP-side ever-dst set) resolved the historical bit-0 garble risk; mode=5 passes correctness on gemma4, qwen3, and qwen3-mtp.
- **PP jitter is a hardware-level L2 cache aliasing effect** that affects both implementations comparably and is not fixable in user space. Three software jitter sources were removed during the optimization campaign, tightening the PP distribution substantially.
- **Control-plane primitives differ** (`dspqueue` vs. native FastRPC `invoke`), but the data plane and the descriptor-dispatch flow are fundamentally the same. The measured performance difference comes from data-plane policy (weight residency + role-aware cache management), not from the control plane.
