> Author: Kimi-K2.7-Code  
> Analysis date: 2026-07-13 23:36:05  
> Context: follow-up discussion after `warmup-ab-test-and-analysis-20260713.md`
>
> Reviewed & corrected by: DeepSeek-V4-Pro  
> Review date: 2026-07-16  
> Review context: full code-level comparison of Qualcomm vs JZ `ggml-hexagon` implementations, cross-referenced against `ggml-hexagon.cpp`, `ggml-hexagon-jz.cpp`, `htp/htp-ops.h`, `htp/dsp-ctx.h`, `htp/entry.c`, `htp/main.c`. Corrections backed by live JZ 5-run inference data (PP 308-343, TG 18.5-19.6 on gemma-4-E2B-it-Q4_0).

---

## 1. Can JZ PP Stable Mean Exceed Qualcomm's?

**Single-run peaks can already match or occasionally exceed Qualcomm. The stable mean is harder to beat, but not impossible with more optimization.**

### Why single-run peaks can match

The core compute power is identical — both implementations run the same HMX kernels from `htp/matmul-ops.c`. Qualcomm's PP mean was 368.78 t/s with a peak of 380.78; after the warmup patch JZ immediately hit **378.57 t/s**. This proves JZ's hardware ceiling is not lower than Qualcomm's.

### Why stable mean is harder to beat

Qualcomm's advantage is not in the matmul itself but in **system-state stability**:

| Factor | Qualcomm | JZ |
|--------|----------|-----|
| Cache coherency | driver-level `DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER \| INVALIDATE_RECIPIENT` | user-space `ion_sync` + `DC CVAC/CIVAC` |
| Buffer mapping | per-buffer `fastrpc_mmap` (pinned or DELAYED per buffer) | single mempool `fastrpc_mmap`, user-managed offset |
| Cold-state | pinned buffers (`FASTRPC_MAP_FD`) avoid first-touch page faults | DELAYED mmap (`FASTRPC_MAP_FD_DELAYED`) defers page faults to first use |
| DCVS/thermal coupling | both backends rely on OS/DSP autonomous DCVS; neither has explicit scheduling hooks | same as Qualcomm |

PP is a short burst. Its result depends heavily on the state **just before** the burst. Qualcomm pushes a lot of "state warming" and "memory stabilization" into the driver, so its jitter window is narrow. JZ's user-space implementation is more transparent but also more exposed to external state.

### What JZ can still do to push stable PP above Qualcomm

1. **Switch `fastrpc_mmap` from DELAYED to eager pinning**
   - Cost: several hundred ms to 1s extra startup, memory pinned earlier.
   - Gain: removes per-run first page-fault variance.

2. **Pre-touch the ION mempool and try to keep physical addresses stable**
   - If the DSP side sees similar physical addresses across runs, L2 alias behavior becomes more predictable.

3. **More aggressive cache-flush strategy**
   - Currently JZ does `DC CVAC` in Phase 6.5. Weights could be flushed once at load time instead of selectively per batch.

4. **DSP DCVS hint before PP**
   - Send a high-load hint to CDSP0 so the DSP stays at high frequency during prompt eval.

5. **Reduce AP-side descriptor preparation time**
   - Phases 1–4 of `graph_compute_batch` still have sorting/dedup/fusion overhead that could be trimmed.

**Conclusion**: JZ PP peaks are already at Qualcomm level, but **beating Qualcomm's stable mean requires re-implementing in user space the work Qualcomm's driver does for free**. The highest ROI items are (1) non-DELAYED mmap and (3) pre-flushing weights, which could add another 3–5% to the stable mean.

---

## 2. Single ION Shared Mempool vs. Qualcomm Per-Buffer ION Memory

Your intuition that a single mempool is better is not completely wrong, but **under the current constraints of incoherent memory + user-space cache management, per-buffer is more practical**.

### Theoretical advantages of a single mempool (your intuition)

The original section listed these as "theoretical" — but code-level comparison confirms they are **real, measurable advantages**:

1. **Contiguous address space**: a large continuous ION region gives better DSP L2 spatial locality. Tensors laid out sequentially in the pool benefit from hardware prefetch and cache-line sharing across adjacent tensors. Qualcomm's per-buffer approach fragments the physical address space; two logically adjacent tensors may end up in pages far apart.

2. **One `fastrpc_mmap` instead of many**: JZ calls `fastrpc_mmap` once at pool init. Qualcomm calls it per `ggml_hexagon_shared_buffer` — each call involves kernel round-trips, page-table setup, and DSP-side SMMU mapping. For a model like gemma-4-E2B with ~700+ tensors, JZ avoids hundreds of mmap/unmap operations.

3. **Offset-based addressing is simpler and faster than `bi` indirection**: JZ's `dsptensor` carries a direct `void *` pointer into the pool. Qualcomm's `htp_tensor` carries a `bi` (buffer index) that the DSP must dereference through `htp_buf_desc[]` to get the actual base address. On the DSP side, one fewer level of indirection per tensor access.

4. **One FastRPC `invoke` per batch vs. dspqueue multi-write**: JZ packs all descriptors into a single `invoke` call. Qualcomm's `dspqueue_write` may split a batch across multiple writes, each with its own kernel transition. Fewer user-kernel transitions per batch.

5. **Pool lifecycle is trivial**: one alloc, one free. Qualcomm must track per-buffer lifecycles, handle partial allocation failures, and coordinate buffer teardown with DSP-side unmapping. JZ's pool is inherently simpler and less error-prone.

6. **Lower kernel resource consumption**: one ION fd vs. hundreds. Each fd consumes kernel memory (file descriptor table, ION handle, dma-buf attachment). On resource-constrained Android devices, this matters.

### Actual disadvantages exposed by the code

JZ's `rpc_mempool` is mapped with `FASTRPC_MAP_FD_DELAYED` mode ([`ggml-hexagon-jz.cpp:1778`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L1778-L1780)). The first batch of each run triggers the actual mapping. Because Linux/Hexagon physical allocation is non-deterministic, **different process launches may give different DSP-side physical addresses**, causing:

- Different L2 cache alias sets → higher miss rates on some runs (hypothesis, not directly measured).
- Different VTCM layout decisions → variable HMX/FLASH_ATTN efficiency.
- Same memory reused for different tensor roles → more frequent stale cache-line pollution.

To compensate, JZ has to do a lot of `DC CVAC/CIVAC` work in [`ggml-hexagon-jz.cpp:5935-5958`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5935-L5958) on every batch.

### Balanced comparison: single pool vs. per-buffer at the code level

| Aspect | JZ single pool | Qualcomm per-buffer | Winner |
|--------|---------------|---------------------|--------|
| `fastrpc_mmap` calls | 1 at init | N per buffer (N = hundreds) | JZ |
| ION fds | 1 | N per buffer | JZ |
| DSP tensor addressing | direct `void *` offset | `bi` → `htp_buf_desc[]` indirection | JZ |
| Batch transport | single `invoke` | `dspqueue_write` (may split) | JZ |
| Memory lifecycle | one alloc/free | N alloc/free + coordination | JZ |
| DSP L2 spatial locality | contiguous, predictable | fragmented, unpredictable | JZ |
| Cache coherency | user-space `DC CVAC/CIVAC` | kernel-space driver flags | Qualcomm |
| Physical address stability | shared across all tensors, one bad address hurts all | isolated per buffer, failure is scoped | Qualcomm |
| Page fault behavior | DELAYED mmap: first touch triggers faults | pinned buffers: no page faults | Qualcomm |

**The single pool approach is architecturally superior in most dimensions.** The two areas where it currently loses — cache coherency and physical address stability — are implementation-level issues, not fundamental architectural flaws. Both are fixable: eager pinning (`rpc_mmap_mode=1`) addresses the page fault issue, and kernel-level cache sync would match Qualcomm's coherency efficiency.

Critically, Qualcomm's disadvantages in the comparison table — per-buffer fd count, per-buffer mmap calls, DSP-side `bi` indirection, multi-write batch transport — are **architectural overhead that cannot be eliminated**. They are inherent to the per-buffer design. Every additional tensor adds another fd, another mmap, another `htp_buf_desc[]` entry. JZ's single pool has none of these per-tensor costs.

Worse, these overheads are **hidden inside the closed-source Hexagon DSP driver**. The `dspqueue_write`/`dspqueue_read` API, the `prep_op_bufs` logic, the per-buffer mmap management, the fd lifecycle — all of it runs in kernel space where no external developer can inspect, profile, or optimize it. You cannot answer basic questions like: how much time does per-buffer mmap actually cost? Is the driver batching mmap calls efficiently? Does it pin all buffers or selectively? The answers are locked inside Qualcomm's proprietary driver binary. JZ's overhead is fully visible: every `fastrpc_mmap`, every `DC CVAC`, every `ion_sync` is an explicit call in open-source code, measurable and optimizable.

Once those are addressed, the single pool's advantages in fd count, mmap count, addressing simplicity, and spatial locality should make it the better approach.

### Transparency advantage: JZ is fully open-source

A critical but often overlooked advantage of JZ ggml-hexagon: **the entire cache coherency pipeline is visible and modifiable** on both AP and DSP sides. From `DC CVAC` in Phase 6.5 ([`ggml-hexagon-jz.cpp:5935-5958`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5935-L5958)) to `CIVAC` in Phase 7.5 ([`ggml-hexagon-jz.cpp:6022-6070`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L6022-L6070)), to DSP-side `dcinva`/`dcflush` in [`htp/entry.c`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/entry.c#L340-L382), every cache maintenance operation is explicit, auditable, and optimizable.

In contrast, Qualcomm's `DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | INVALIDATE_RECIPIENT` is a black box — the flags are set, but the actual cache maintenance logic lives inside the closed-source Hexagon DSP driver. You cannot inspect what it does, you cannot tune it, and you cannot fix it if it's suboptimal. JZ's transparency means:

- Cache flush strategies can be made selective (e.g., flush weights once at load time, not every batch).
- `ion_sync` can be tuned per region (dirty vs. clean ranges).
- DSP-side cache modes (`dsp_cache_mode`) can be experimented with at runtime.
- The interaction between `DC CVAC`, `CIVAC`, and `ion_sync` can be profiled and optimized.

This is a long-term strategic advantage: JZ can iterate and improve its cache coherency faster than Qualcomm can ship driver updates, and JZ can optimize for the specific workload (LLM inference) rather than the generic use case the Qualcomm driver targets.

### Why Qualcomm's per-buffer approach wins in practice

Qualcomm uses `htp_buf_desc[]` + `bi` index (the `htp_tensor` struct is defined in [`htp/htp-ops.h`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/htp-ops.h#L120-L132)). Each `ggml_hexagon_shared_buffer` is allocated per-tensor via `ggml_backend_hexagon_buffer_type_alloc_buffer`, not pre-assigned to a fixed role. In practice, the allocation order naturally separates tensors into distinct buffers, which gives the driver several advantages:

- Cache coherency is handled at driver level via `DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT` flags applied at `dspqueue_write` time ([`ggml-hexagon.cpp:1397`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1397)). These flags are applied uniformly to every buffer per batch — there is no per-role differentiation in the code. The advantage is that cache maintenance runs in kernel space, which is more efficient than user-space `DC CVAC` per byte.
- Multiple independent buffers mean per-buffer ION allocation failures are isolated (one bad allocation doesn't block the entire pool), and the kernel's physical page allocator can satisfy smaller requests more easily than a single large contiguous allocation.
- The `bi` (buffer index) indirection decouples tensor addressing from the physical layout of any single buffer, which simplifies the DSP-side descriptor parsing in `htp/main.c`.

### A more accurate conclusion

**Per-buffer is not inherently better; it is easier to do well under user-space cache management.**

If JZ adds these things, a single mempool can still be competitive:

1. **eager pinning + stable physical addresses** (remove layout variance). JZ already supports `rpc_mmap_mode=1` for eager pinning at runtime, but the default remains DELAYED.
2. **kernel- or driver-level batch cache sync** (remove user-space CVAC/CIVAC overhead)
3. **separate weight and activation sub-regions** inside the pool to reduce cross-contamination between read-mostly weight data and frequently-invalidated activation data

Without those, per-buffer is more controllable, which is why Qualcomm is more stable. Note that Qualcomm's per-buffer approach does not have true "role-based" cache strategies — it applies the same flush/invalidate flags to every buffer per batch. The advantage comes from kernel-space cache maintenance and the isolation of per-buffer ION allocations, not from differentiated per-role coherency handling.

---

## 3. Control-Plane Primitive Difference: dspqueue vs. Native FastRPC invoke

Another way to frame the Qualcomm-vs-JZ distinction is to separate **control plane** from **data plane**.

### Control plane

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

| | Qualcomm ggml-hexagon | JZ ggml-hexagon |
|---|---|---|
| ION granularity | Per-buffer; each `ggml_hexagon_shared_buffer` has its own ION fd + `fastrpc_mmap` | Shared mem pool; one big `rpcmem_alloc2` region + offsets |
| Mapping cost | DSP maps each fd via `fastrpc_mmap`; per-buffer overhead | Pool mapped once; subsequent ops only pass offsets |
| Flexibility | Good for discrete large tensors | Good for compact, unified memory layout |

The pool approach has lower fd and mapping overhead when batches are frequent and tensors are numerous, which is why many custom HTP backends prefer it. The trade-off is that the user-space side must take over the cache-coherency and physical-address-stability work that Qualcomm's driver does automatically with per-buffer mappings.

---

## 4. Summary

- **JZ PP peak can already rival Qualcomm**, but stable mean leadership needs state-management work that is currently done by Qualcomm's driver.
- **Single mempool is not fundamentally inferior**, but in the current JZ implementation it carries too much user-space responsibility. Fixing mmap strategy and cache flush batching is more important than switching to per-buffer.
- **Control-plane primitives differ** (`dspqueue` vs. native FastRPC `invoke`), but the data plane and the descriptor-dispatch flow are fundamentally the same.
