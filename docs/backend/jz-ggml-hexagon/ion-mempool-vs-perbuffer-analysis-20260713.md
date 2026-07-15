> Author: Kimi-K2.7-Code  
> Analysis date: 2026-07-13 23:36:05  
> Context: follow-up discussion after `warmup-ab-test-and-analysis-20260713.md`

---

## 1. Can JZ PP Stable Mean Exceed Qualcomm's?

**Single-run peaks can already match or occasionally exceed Qualcomm. The stable mean is harder to beat, but not impossible with more optimization.**

### Why single-run peaks can match

The core compute power is identical — both implementations run the same HMX kernels from `htp/matmul-ops.c`. Qualcomm's PP mean was 368.78 t/s with a peak of 380.78; after the warmup patch JZ immediately hit **378.57 t/s**. This proves JZ's hardware ceiling is not lower than Qualcomm's.

### Why stable mean is harder to beat

Qualcomm's advantage is not in the matmul itself but in **system-state stability**:

| Factor | Qualcomm | JZ |
|--------|----------|-----|
| Cache coherency | driver-level batch flush/invalidate | user-space `ion_sync` + `DC CVAC/CIVAC` |
| Buffer mapping | per-buffer, driver independently managed | single mempool, user-managed offset |
| Cold-state | dspqueue has internal prefetch/pinning | relies on delayed mmap first touch |
| DCVS/thermal coupling | driver may participate in scheduling hints | fully OS/DSP autonomous |

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

1. **Contiguous address space**: a large continuous ION region may give better DSP L2 spatial locality.
2. **Fewer mmap calls**: one `fastrpc_mmap` instead of many.
3. **Simpler management**: all tensors addressed by offset, AP allocator is straightforward.

### Actual disadvantages exposed by the code

JZ's `rpc_mempool` is mapped with `FASTRPC_MAP_FD_DELAYED` mode ([`ggml-hexagon-jz.cpp:1778`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L1778-L1780)). The first batch of each run triggers the actual mapping. Because Linux/Hexagon physical allocation is non-deterministic, **different process launches may give different DSP-side physical addresses**, causing:

- Different L2 cache alias sets → higher miss rates on some runs.
- Different VTCM layout decisions → variable HMX/FLASH_ATTN efficiency.
- Same memory reused for different tensor roles → more frequent stale cache-line pollution.

To compensate, JZ has to do a lot of `DC CVAC/CIVAC` work in [`ggml-hexagon-jz.cpp:5935-5958`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5935-L5958) on every batch.

### Why Qualcomm's per-buffer approach wins in practice

Qualcomm uses `htp_buf_desc[]` + `bi` index (around [`ggml-hexagon.cpp:726`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L726) in `htp_tensor`). Each buffer has a relatively fixed role:

- input/output buffers
- weight buffers
- scratch/intermediate buffers

This lets the driver:

- choose flush/invalidate strategies per buffer role;
- spread physical-address risk across multiple buffers instead of one big pool;
- batch coherency operations in kernel space, which is more efficient than user-space `DC CVAC` per byte.

### A more accurate conclusion

**Per-buffer is not inherently better; it is easier to do well under user-space cache management.**

If JZ adds these three things, a single mempool can still be competitive:

1. **eager pinning + stable physical addresses** (remove layout variance)
2. **kernel- or driver-level batch cache sync** (remove user-space CVAC/CIVAC)
3. **role-based zoning inside the pool** instead of mixing all tensors in one flat offset space

Without those, per-buffer is more controllable, which is why Qualcomm is more stable.

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
| ION granularity | Per-buffer; each buffer/tensor has its own fd | Shared mem pool; one big region + offsets |
| Mapping cost | DSP maps each fd via `HAP_mmap` / `prep_op_bufs` | Pool mapped once; subsequent ops only pass offsets |
| Flexibility | Good for discrete large tensors | Good for compact, unified memory layout |

The pool approach has lower fd and mapping overhead when batches are frequent and tensors are numerous, which is why many custom HTP backends prefer it. The trade-off is that the user-space side must take over the cache-coherency and physical-address-stability work that Qualcomm's driver does automatically with per-buffer mappings.

---

## 4. Summary

- **JZ PP peak can already rival Qualcomm**, but stable mean leadership needs state-management work that is currently done by Qualcomm's driver.
- **Single mempool is not fundamentally inferior**, but in the current JZ implementation it carries too much user-space responsibility. Fixing mmap strategy and cache flush batching is more important than switching to per-buffer.
- **Control-plane primitives differ** (`dspqueue` vs. native FastRPC `invoke`), but the data plane and the descriptor-dispatch flow are fundamentally the same.
