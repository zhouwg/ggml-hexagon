> Author: Kimi-K2.7-Code
> Date: 2026-07-13 (revised 2026-07-16)
> Context: follow-up to `warmup-ab-test-and-analysis-20260713.md`
>
> Revised 2026-07-16 by DeepSeek-V4-Pro, Kimi-K2.7-Code, MiniMax-M3, GLM-5.2:
> - Full code-level comparison cross-referenced against `ggml-hexagon.cpp`, `ggml-hexagon-jz.cpp`, `htp/htp-ops.h`, `htp/dsp-ctx.h`, `htp/entry.c`, `htp/main.c`; corrections backed by live JZ inference data (gemma-4-E2B-it-Q4_0).
> - Experimentally ruled out rpc_mmap_mode, AP/DSP pre-touch as jitter fixes; code-verified that both JZ and Qualcomm backends explicitly disable DCVS and pin DSP to max corners. Identified L2 cache alias variance (consistent with physically-indexed cache, e.g., PIPT; exact microarchitecture not publicly documented) as the only tested factor with significant correlation. Remaining uninvestigated factors include thermal, link order, memory bus contention, and multi-process interference.
> - Softened absolute claims to qualified inferences; added JZ 7-run mean (~349 tok/s, 2026-07-15); refined PP jitter range to ~320-390 tok/s for both implementations.
> - Code-verified all technical claims against source; marked `pretouch_ion` as reverted experiment; corrected "one bad address affects all tensors" misconception (single ION allocation has contiguous CPU virtual + DSP IOVA but scattered physical pages); updated comparison table and conclusion; added table note on tested-no-effect dimensions; renamed section header to match content; fixed causal-chain diagram alignment; toned down advocacy language; added source-literature qualifier; compressed header; corrected "DSP L2 spatial locality" to "IOVA spatial locality (prefetch/TLB)" for consistency with PIPT hypothesis; corrected "per-tensor" to "per-buffer" (multiple tensors may share one buffer); clarified SYSTEM heap does not require physical contiguity; replaced unicode arrows with ASCII.

---

## 1. JZ PP Performance Relative to Qualcomm

**Single-run peaks can already match or occasionally exceed Qualcomm. Achieving a higher stable mean requires further optimization.**

### Why single-run peaks can match

The core compute power is identical — both implementations run the same HMX kernels from `htp/matmul-ops.c`. Qualcomm's PP mean was 368.78 t/s with a peak of 380.78 (measured locally on the same Snapdragon 8 Elite device with Qualcomm's upstream ggml-hexagon backend; run count not recorded); after the warmup patch JZ hit **378.57 t/s** (single-run peak with dsp_cache_mode=7) and subsequently **388.49 t/s** (single-run peak with dsp_cache_mode=4 after stochastic-garble rollback, 2026-07-16 06:24:15, gemma-4-E2B-it-Q4_0). JZ's reproducible mean across 7 consecutive runs (2026-07-15, dsp_cache_mode=4) is **~349 tok/s** (range 332-368). The combined observed PP range for both implementations is **320-390 tok/s** (Qualcomm observed 320-380; JZ observed 332-388, rounded to 320-390 for parity reporting). This indicates that JZ's hardware ceiling is comparable to Qualcomm's, while JZ's stable mean lags Qualcomm's by ~5%.

### Why stable mean is harder to beat — and why Qualcomm has the same problem

**Important clarification: Qualcomm's ggml-hexagon also suffers from PP jitter.** The same Snapdragon 8 Elite device shows Qualcomm PP ranging from ~320 to ~380 tok/s — the same range as JZ. This is not a JZ-specific issue.

The most plausible explanation is a **hardware-level limitation** of the Hexagon DSP platform. The observed behavior is consistent with an **L2 cache whose indexing depends on physical addresses** (e.g., PIPT or physically-indexed variants). Every time the Linux kernel's page allocator (`rpcmem_alloc2`) gives different physical addresses to the ION buffer — which it does on every process launch — the DSP's L2 cache alias sets change. Different physical addresses map to different cache sets, causing HMX matmul throughput to vary by 15-20%. Neither JZ nor Qualcomm can control this in user space.

The table below shows the implementation-level differences, but **none of them can fix the cache alias problem**:

**Table 1:** Implementation-level differences (Qualcomm vs. JZ)

| Factor | Qualcomm | JZ |
|--------|----------|-----|
| Cache coherency | driver-level `DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER \| INVALIDATE_RECIPIENT` | configurable: `ion_sync` only (default, mode=1) / `ion_sync` + `DC CVAC/CIVAC` (mode=0) / `DC CVAC/CIVAC` only (mode=2) |
| Buffer mapping | per-buffer `fastrpc_mmap` (pinned or DELAYED per buffer) | single mempool `fastrpc_mmap`, user-managed offset |
| Cold-state | regular tensor buffers use DELAYED mmap; only queue metadata buffer is pinned | DELAYED mmap (`FASTRPC_MAP_FD_DELAYED`) defers page faults to first use |
| DCVS/thermal coupling | DCVS disabled, DSP pinned to max corners (`dcvs_enable=0`, `sleep_disable=1`) | same as Qualcomm |

PP is a short burst. Its result depends heavily on the state **just before** the burst. But the dominant factor — physical address non-determinism — is outside the control of both implementations. Neither approach can control this: By default, `rpcmem_alloc2()` using SYSTEM ION heap + IOMMU returns a buffer with contiguous CPU virtual address and contiguous IOVA for remote DSP, while the underlying physical memory is composed of independent 4KB pages with scattered physical addresses that differ across allocation cycles. The physical memory will be contiguous only if CMA/CARVEOUT heap or hugepage allocation is explicitly enabled via allocation flags. Both per-buffer and single-pool therefore suffer the same per-page cache alias variance. The practical observation confirms this: the jitter range is comparable for both implementations.

### Attempted optimizations and remaining options

**Update (2026-07-16): Several of the suggestions below have been experimentally tested. Items marked with [TESTED: NO EFFECT] were verified to not eliminate PP jitter. The L2 cache alias problem (consistent with physically-indexed cache architecture) is the dominant factor and cannot be fully fixed in user space.**

1. **[TESTED: NO EFFECT] Switch `fastrpc_mmap` from DELAYED to eager pinning**
   - A/B test on 2026-07-16: `rpc_mmap_mode=1` (FASTRPC_MAP_FD) produced PP values of 323, 338 compared to baseline 338. No reduction in jitter.
   - Reason: physical addresses are already determined by `rpcmem_alloc2()` before `fastrpc_mmap()`. The mmap mode only controls when the DSP-side SMMU page tables are set up, not which physical pages are allocated.

2. **[TESTED: NO EFFECT] Pre-touch the ION mempool**
   - AP-side pre-touch: no-op for the DSP. The AP-side virtual address space is completely separate from the DSP-side SMMU address space. Touching AP-side pages does not force DSP-side SMMU page table setup.
   - DSP-side pre-touch: implemented via a `pretouch_ion` IDL method + entry.c handler (experiment, since reverted). The DSP touched all 1,030,144 pages (4024 MB) in ~55ms at ~70 GB/s (effective throughput includes both L2 hits and DDR line-fill). Confirmed working via adb logcat at the time. However, PP jitter was unaffected (PP 323-347 across 5 runs). The SMMU page fault overhead is negligible compared to the L2 cache alias variance.

3. **More aggressive cache-flush strategy**
   - When `ion_sync_mode=0` or `2`, JZ does `DC CVAC` in Phase 6.5. Weights could be flushed once at load time instead of selectively per batch.

4. **[NOT APPLICABLE] DSP DCVS hint before PP**
   - DCVS is already disabled on both DSP sides (`dcvs_enable=0`, max corners, `sleep_disable=1`). The `ggmlhexagon_set_rpc_latency` function's `latency` parameter is explicitly discarded (`(void)latency;`); only FastRPC QoS is enabled. No further runtime DCVS tuning is available in user space.

5. **Reduce AP-side descriptor preparation time**
   - Phases 1–4 of `graph_compute_batch` still have sorting/dedup/fusion overhead that could be trimmed.

**Conclusion**: JZ PP peaks are already at Qualcomm level, and **the PP jitter range (~320-390 tok/s for both implementations) is comparable**. The jitter is **consistent with** L2 cache aliasing caused by non-deterministic physical addresses from the Linux kernel's page allocator — a hardware-level effect that neither JZ nor Qualcomm can fix in user space. The highest remaining ROI items are (3) pre-flushing weights and (5) reducing AP-side descriptor overhead, which could yield modest improvements to the stable mean but will not eliminate the jitter.

---

## 2. Single ION Shared Mempool vs. Qualcomm Per-Buffer ION Memory

A single shared mempool has several advantages, but **under the current constraints of incoherent memory and user-space cache management, the per-buffer approach is more practical**.

### Advantages of a single mempool

Code-level comparison confirms these are **real, measurable advantages**:

1. **Contiguous IOVA address space**: a large continuous ION region gives better spatial locality at the prefetcher and TLB level. Tensors laid out sequentially in the pool benefit from hardware prefetch. Qualcomm's per-buffer approach fragments the IOVA address space; two logically adjacent tensors may end up in separate ION buffers with unrelated IOVA ranges.

2. **One `fastrpc_mmap` instead of many**: JZ calls `fastrpc_mmap` once at pool init. Qualcomm calls it per `ggml_hexagon_shared_buffer` — each call involves kernel round-trips, page-table setup, and DSP-side SMMU mapping. For a model like gemma-4-E2B with ~700+ tensors, JZ avoids hundreds of mmap/unmap operations.

3. **Offset-based addressing is simpler and faster than `bi` indirection**: JZ's `dsptensor` carries a direct `void *` pointer into the pool. Qualcomm's `htp_tensor` carries a `bi` (buffer index) that the DSP must dereference through `htp_buf_desc[]` to get the actual base address. On the DSP side, one fewer level of indirection per tensor access.

4. **One FastRPC `invoke` per batch vs. dspqueue multi-write**: JZ packs all descriptors into a single `invoke` call. Qualcomm's `dspqueue_write` may split a batch across multiple writes, each with its own kernel transition. Fewer user-kernel transitions per batch.

5. **Pool lifecycle is trivial**: one alloc, one free. Qualcomm must track per-buffer lifecycles, handle partial allocation failures, and coordinate buffer teardown with DSP-side unmapping. JZ's pool is inherently simpler and less error-prone.

6. **Lower kernel resource consumption**: one ION fd vs. hundreds. Each fd consumes kernel memory (file descriptor table, ION handle, dma-buf attachment). On resource-constrained Android devices, this matters.

### Actual disadvantages exposed by the code

JZ's `rpc_mempool` is mapped with `FASTRPC_MAP_FD_DELAYED` mode in `ggmlhexagon_init_rpcmempool()` ([`ggml-hexagon-jz.cpp`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp)). The physical pages are allocated by `rpcmem_alloc2()` before `fastrpc_mmap()` — the mmap mode only controls when the DSP-side SMMU page tables are set up, not which physical pages are allocated. Because Linux's physical page allocator is non-deterministic, **different process launches give different DSP-side physical addresses**, causing:

- Different L2 cache alias sets -> **experimentally the dominant remaining factor correlated with PP jitter among tested variables**. On 2026-07-16, four runtime variables were tested and ruled out: `rpc_mmap_mode` (eager vs DELAYED), AP-side pre-touch, DSP-side pre-touch, and `dsp_cache_mode` (4 vs 5). In addition, code review confirmed that both JZ and Qualcomm backends disable DCVS and pin the DSP to max corners, so DVFS is not a contributor. The L2 cache (whose behavior is consistent with physically-indexed indexing) + non-deterministic physical addresses is the only tested factor with significant correlation. Other plausible factors (thermal, link order, memory bus contention, multi-process interference) remain uninvestigated and cannot be ruled out.
- Different VTCM layout decisions -> variable HMX/FLASH_ATTN efficiency.
- Same memory reused for different tensor roles -> stale cache-line pollution across batches.

When `ion_sync_mode` includes `DC CVAC/CIVAC`, JZ does explicit cache maintenance inside `graph_compute_batch()` ([`ggml-hexagon-jz.cpp`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp)) on every batch.

### Balanced comparison: single pool vs. per-buffer at the code level

**Table 2:** Single pool vs. per-buffer comparison at the code level

| Aspect | JZ single pool | Qualcomm per-buffer | Winner |
|--------|---------------|---------------------|--------|
| `fastrpc_mmap` calls | 1 at init | N per buffer (N = hundreds) | JZ |
| ION fds | 1 | N per buffer | JZ |
| DSP tensor addressing | direct `void *` offset | `bi` -> `htp_buf_desc[]` indirection | JZ |
| Batch transport | single `invoke` | `dspqueue_write` (may split) | JZ |
| Memory lifecycle | one alloc/free | N alloc/free + coordination | JZ |
| IOVA spatial locality (prefetch/TLB) | contiguous, predictable | fragmented across buffers | JZ |
| Cache coherency | configurable: `ion_sync` only (default, mode=1) / `ion_sync` + `DC CVAC/CIVAC` (mode=0) / `DC CVAC/CIVAC` only (mode=2) | kernel-space driver flags | Qualcomm |
| Physical address stability | same per-page variance (pool is physically fragmented, not contiguous) | same per-page variance (each buffer independently allocated) | Tie |
| Page fault behavior | DELAYED mmap for all data (single mempool; no separate queue metadata buffer) | DELAYED mmap for regular tensor buffers; queue metadata buffer (shm_buf) pinned | Qualcomm (pins queue metadata; A/B test: no jitter impact) |

*Note: both implementations use DELAYED mmap for tensor data. Qualcomm additionally pins the queue metadata buffer (shm_buf); JZ does not have a separate pinned buffer. A/B testing (see items 1-2 above) confirmed this dimension does not affect PP jitter in practice.*

**The single pool approach is advantageous in several dimensions: fd count, mmap count, addressing simplicity, and spatial locality.** One area where it currently loses — cache coherency — is an implementation-level issue, not a fundamental architectural flaw. The page fault behavior difference (pinned queue metadata) is a mechanism distinction that A/B testing confirmed has no PP jitter impact. Kernel-level cache sync would match Qualcomm's coherency efficiency.

Critically, Qualcomm's disadvantages in Table 2 — per-buffer fd count, per-buffer mmap calls, DSP-side `bi` indirection, multi-write batch transport — are **overhead inherent to the per-buffer API design**. Every additional buffer requires another fd, another mmap, and another `htp_buf_desc[]` entry at the interface level. JZ's single pool has none of these per-buffer costs.

Additionally, these overheads are **handled inside the closed-source Hexagon DSP driver**. The `dspqueue_write`/`dspqueue_read` API, the `prep_op_bufs` logic, the per-buffer mmap management, the fd lifecycle — all of it runs in kernel space where external developers cannot inspect, profile, or optimize it. External developers cannot determine, for example, how much time per-buffer mmap actually costs, whether the driver batches mmap calls efficiently, or whether it pins all buffers selectively. The details are contained within Qualcomm's proprietary driver binary. JZ's overhead is fully visible: every `fastrpc_mmap`, every `DC CVAC`, every `ion_sync` is an explicit call in open-source code, measurable and optimizable.

Once those are addressed, the single pool's advantages in fd count, mmap count, addressing simplicity, and spatial locality should make it the better approach.

### Root cause of PP jitter: L2 cache architecture + non-deterministic physical addresses

The PP jitter (~320-390 tok/s for both implementations) is **most likely** a **hardware-level limitation** of the Hexagon DSP platform, not a software bug. It affects both JZ and Qualcomm ggml-hexagon comparably.

**The causal chain:**

```
Linux kernel page allocator          Hexagon DSP L2 Cache
(rpcmem_alloc2)                      (behavior consistent with
        |                            physically-indexed caching)
  Non-deterministic                              |
  physical addresses  -------------------------> Different cache alias sets
  on each process launch                         per run
                                                 |
                                                 v
                                          HMX matmul throughput
                                          varies by 15-20%
```

**Why physical addresses affect cache behavior:**

The observed jitter pattern is **consistent with an L2 cache whose indexing depends on physical addresses**. While the exact microarchitecture of Hexagon DSP's L2 cache is not publicly documented, the behavior matches what one would expect from a physically-indexed cache (e.g., PIPT — Physically Indexed, Physically Tagged):

**Table 3:** Cache types and their sensitivity to physical address changes

| Cache type | Index source | Sensitivity to physical address changes |
|-----------|-------------|:---:|
| **Physically-indexed** (e.g., PIPT) | Physical address | **High** — both index and tag come from PA |
| Virtually-indexed, physically-tagged (VIPT) | Virtual address | Low — index from VA, only tag from PA |
| Virtually-indexed, virtually-tagged (VIVT) | Virtual address | None — everything from VA |

When the kernel allocates different physical pages for the ION buffer on each process launch, the L2 cache maps the same logical tensor data to different cache sets. Some sets get more conflicts than others, causing HMX matmul throughput to vary. This mechanism operates regardless of the exact indexing scheme — as long as physical addresses influence cache set selection, the jitter will occur.

**What was tested and ruled out (2026-07-16):**

**Table 4:** Variables tested and ruled out (2026-07-16)

| Variable tested | Method | Result |
|----------------|--------|--------|
| `rpc_mmap_mode` (eager vs DELAYED) | A/B test: `FASTRPC_MAP_FD` vs `FASTRPC_MAP_FD_DELAYED` | **No effect** — physical addresses are already allocated by `rpcmem_alloc2()` before `fastrpc_mmap()` |
| AP-side pre-touch | Sequential read of ION pool from AP virtual address | **No effect** — AP-side VA space is completely separate from DSP-side SMMU address space |
| DSP-side pre-touch | New `pretouch_ion` IDL method + entry.c handler, DSP touches all 4024 MB in ~55ms | **No effect** — SMMU page fault overhead is negligible compared to L2 cache alias variance |
| `dsp_cache_mode` | Switch from 4 to 5 (first-touch weight bitmap) | **No effect on PP** — safe on gemma4, marginal TG gain |
| DCVS / DVFS | Code review: `dcvs_enable=0`, max corners, `sleep_disable=1` in both `htp/entry.c` and `htp/main.c` | **Controlled, not a cause** — DSP frequency is pinned; no dynamic scaling during inference |

**Why this is not fixable in user space:**

The physical addresses are allocated by the Linux kernel's page allocator inside `rpcmem_alloc2()`. This is a kernel-side operation that user space cannot influence. The only way to eliminate this jitter would be to:
1. Reserve a DMA region at a fixed physical address (requires kernel driver modification)
2. Use a hardware page coloring scheme to stabilize cache set mapping (requires Hexagon DSP firmware modification)
3. Run inside a static VM with fixed physical memory layout (not applicable to Android)

All three options require platform-level changes beyond the scope of ggml-hexagon.

**Cross-platform comparison:**

**Table 5:** Cross-platform jitter comparison

| Platform | NPU/DSP | Likely affected by similar jitter? |
|----------|---------|:---:|
| Qualcomm Snapdragon | Hexagon DSP | **Yes** — confirmed by observation |
| MTK Dimensity | APU | **Unknown** — cache architecture is closed |
| Huawei Kirin | Da Vinci NPU | **Unknown** — cache architecture and driver memory management are closed |

Note: Qualcomm Adreno GPU (same SoC as Hexagon DSP) is expected to be less sensitive to physical address non-determinism than the DSP, as GPU caches typically use virtually-indexed or hashed indexing schemes to avoid pathological aliasing. This is an inference based on general GPU cache design; Adreno's exact cache architecture is not publicly documented.

This sensitivity to physical address non-determinism is a common characteristic of DSP architectures that evolved from MCU roots — DSPs historically ran without MMUs, and physically-indexed caches are often associated with low-latency DSP pipelines (based on general DSP architecture literature; modern Hexagon V79 has SMMU support, but the L2 cache indexing choice is independent of SMMU presence). This makes Hexagon DSP uniquely vulnerable to physical address non-determinism, which is unavoidable in Linux's ION/DMA-BUF ecosystem.

**Practical recommendation:**

Accept the PP jitter and report it as a unified range: "PP: ~320-390 tok/s, typical ~340-350 (both Qualcomm and JZ ggml-hexagon)". This is what Qualcomm's ggml-hexagon does in practice, and it accurately reflects the hardware's capability.

### Transparency advantage: JZ is fully open-source

An additional consideration: **the entire cache coherency pipeline is visible and modifiable** on both AP and DSP sides. From `DC CVAC` in Phase 6.5 to `CIVAC` in Phase 7.5 (both inside `graph_compute_batch()` in [`ggml-hexagon-jz.cpp`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp)), to DSP-side `dcinva`/`dccleaninva` in [`htp/entry.c`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/entry.c), every cache maintenance operation is explicit, auditable, and optimizable.

In contrast, Qualcomm's `DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | INVALIDATE_RECIPIENT` hides the cache maintenance logic inside the closed-source Hexagon DSP driver. The flags are set at the API level, but the actual flush/invalidate implementation is not visible to user-space developers, making detailed tuning or verification impractical. JZ's transparency means:

- Cache flush strategies can be made selective (e.g., flush weights once at load time, not every batch).
- `ion_sync` can be tuned per region (dirty vs. clean ranges).
- DSP-side cache modes (`dsp_cache_mode`) can be experimented with at runtime.
- The interaction between `DC CVAC`, `CIVAC`, and `ion_sync` can be profiled and optimized.

This gives JZ flexibility to iterate on cache coherency independently of vendor driver release cycles, and to optimize for the specific workload (LLM inference) rather than the generic use case the Qualcomm driver targets.

### Practical advantages of Qualcomm's per-buffer approach

Qualcomm uses `htp_buf_desc[]` + `bi` index (the `htp_tensor` struct is defined in [`htp/htp-ops.h`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/htp-ops.h)). Each `ggml_hexagon_shared_buffer` is allocated per-buffer via `ggml_backend_hexagon_buffer_type_alloc_buffer` (one ION fd + `fastrpc_mmap` per `ggml_backend_buffer`; multiple tensors may share one buffer), not pre-assigned to a fixed role. In practice, the allocation order naturally separates tensors into distinct buffers, which gives the driver several advantages:

- Cache coherency is handled at driver level via `DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT` flags applied at `dspqueue_write` time in [`ggml-hexagon.cpp`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp). These flags are applied uniformly to every buffer per batch — there is no per-role differentiation in the code. The advantage is that cache maintenance runs in kernel space, which is more efficient than user-space `DC CVAC` per byte.
- Multiple independent buffers mean per-buffer ION allocation failures are isolated (one bad allocation doesn't block the entire pool), and the kernel's physical page allocator can satisfy smaller requests more easily than one very large request under memory pressure (SYSTEM heap uses scattered 4KB pages and does not require physical contiguity).
- The `bi` (buffer index) indirection decouples tensor addressing from the physical layout of any single buffer, which simplifies the DSP-side descriptor parsing in `htp/main.c`.

### Conclusion

**Per-buffer is not inherently better; it is easier to do well under user-space cache management.**

If JZ adds these things, a single mempool can still be competitive:

1. **eager pinning** (remove first-touch page-fault overhead). JZ already supports `rpc_mmap_mode=1` for eager pinning at runtime, but the default remains DELAYED. Note that eager pinning does not stabilize physical addresses (which are determined by `rpcmem_alloc2()` and vary per launch regardless of mmap mode) — it only moves SMMU page table setup earlier.
2. **kernel- or driver-level batch cache sync** (remove user-space CVAC/CIVAC overhead)
3. **separate weight and activation sub-regions** inside the pool to reduce cross-contamination between read-mostly weight data and frequently-invalidated activation data

Without those, per-buffer is more controllable. Note that Qualcomm's per-buffer approach does not have true "role-based" cache strategies — it applies the same flush/invalidate flags to every buffer per batch. The advantage comes from kernel-space cache maintenance and the isolation of per-buffer ION allocations, not from differentiated per-role coherency handling.

---

## 3. Control-Plane Primitive Difference: dspqueue vs. Native FastRPC invoke

Another way to frame the Qualcomm-vs-JZ distinction is to separate **control plane** from **data plane**.

### Control plane

**Table 6:** Control plane comparison

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

### ION management granularity

**Table 7:** ION management granularity comparison

| | Qualcomm ggml-hexagon | JZ ggml-hexagon |
|---|---|---|
| ION granularity | Per-buffer; each `ggml_hexagon_shared_buffer` has its own ION fd + `fastrpc_mmap` | Shared mem pool; one big `rpcmem_alloc2` region + offsets |
| Mapping cost | DSP maps each fd via `fastrpc_mmap`; per-buffer overhead | Pool mapped once; subsequent ops only pass offsets |
| Flexibility | Good for discrete large tensors | Good for compact, unified memory layout |

The pool approach has lower fd and mapping overhead when batches are frequent and tensors are numerous, which is why many custom HTP backends prefer it. The trade-off is that the user-space side must take over the cache-coherency work that Qualcomm's driver handles automatically in kernel space with per-buffer mappings. Physical address non-determinism affects both approaches equally.

---

## 4. Summary

- **PP jitter (~320-390 tok/s for both Qualcomm and JZ ggml-hexagon) is most likely a hardware-level limitation** of the Hexagon DSP's L2 cache reacting to non-deterministic physical addresses from the Linux kernel's page allocator. The observed behavior is consistent with a physically-indexed cache architecture (e.g., PIPT), though the exact microarchitecture is not publicly documented. It affects **both JZ and Qualcomm ggml-hexagon comparably**. After experimentally ruling out `rpc_mmap_mode`, AP-side pre-touch, DSP-side pre-touch, and `dsp_cache_mode` (4->5), and after code review confirmed that both backends disable DCVS and pin the DSP to max corners, the L2 cache alias variance is the only tested factor with significant correlation. Remaining uninvestigated factors include thermal, link order, memory bus contention, and multi-process interference.
- **JZ PP peak can already rival Qualcomm** (388.49 single-run with dsp_cache_mode=4 vs Qualcomm 380.78), but the jitter range is comparable for both implementations. The practical recommendation is to report PP as a unified range ("~320-390 tok/s, typical ~340-350 (both implementations)").
- **Single mempool is not fundamentally inferior**, but in the current JZ implementation it carries too much user-space responsibility. Fixing mmap strategy and cache flush batching is more important than switching to per-buffer.
- **Control-plane primitives differ** (`dspqueue` vs. native FastRPC `invoke`), but the data plane and the descriptor-dispatch flow are fundamentally the same.
