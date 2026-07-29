# Why Qualcomm ggml-hexagon Cannot Offload lm-head to the DSP (2026-07-24)

> Chinese version: [why-perbuffer-cannot-offload-lmhead-20260724-zh.md](why-perbuffer-cannot-offload-lmhead-20260724-zh.md) (authors: MiniMax-M3, Kimi-K3, GLM-5.2)

## 1. Background: lm-head is the biggest TG bottleneck

TG (token generation) is DRAM-bandwidth-bound in both implementations: every token re-reads all weights from DRAM. The lm-head matrix (262144 x 1536, Q4_K), which maps hidden states to vocabulary space, takes about 30 ms/token on the CPU and is the single largest TG cost.

JZ ggml-hexagon and Qualcomm ggml-hexagon both previously had a `ne[1] > 32768` guard that kept large quantized weight matrices off the DSP, so lm-head ran on the CPU in both. JZ ggml-hexagon removed the guard and, together with Q4_K -> Q4_0 repack and the first-touch weight invalidation mechanism, offloaded lm-head to the DSP; Qualcomm ggml-hexagon still keeps the guard. This is the direct reason JZ ggml-hexagon's TG (26.91 tok/s) overtakes Qualcomm ggml-hexagon's TG (24.91 tok/s) (multi-run mean, see [ion-mempool-vs-perbuffer-analysis-20260713.md](file:///home/zhouwg/develop/ggml-hexagon/docs/backend/jz-ggml-hexagon/ion-mempool-vs-perbuffer-analysis-20260713.md)).

## 2. Per-buffer ION: the fixed per-buffer cost

Qualcomm ggml-hexagon's `ggml_hexagon_shared_buffer` (see [`ggml-hexagon.cpp`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp)) holds, independently for each buffer:

- an ION fd (allocated by `rpcmem_alloc2`, see `alloc()`)
- a `fastrpc_mmap` (kernel SMMU mapping, see `alloc()`)
- a DSP-side `htp_buf_desc[]` entry + `bi` indirection (see `ggml_hexagon_opqueue::add_buffer()`)
- AP-DSP lifecycle coordination (alloc / munmap / destroy, see `free()`)

A 214MB lm-head as a single buffer holds all of these resources for the entire session: kernel fd slot, ION handle, SMMU mapping. This is the "symmetry" requirement of the per-buffer API design: there is no special case for a huge buffer.

JZ ggml-hexagon's single mempool (see [`ggmlhexagon_init_rpcmempool()`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp)): one mmap at init; lm-head is just an offset range inside the pool, with zero recurring cost.

## 3. dspqueue's per-batch recurring cost with large buffers

Qualcomm ggml-hexagon's dspqueue depth is 16 (`opt_opqueue = 16`, see [`ggml-hexagon.cpp`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp) for `opt_opqueue`), i.e. up to 16 batches in flight. Every batch re-registers its required `htp_buf_desc` (fd / base / size) via `ggml_hexagon_opqueue::add_buffer()`.

Once the 214MB lm-head enters the dspqueue path:

- every token's op-batch calls `add_buffer()` to re-register lm-head; its `htp_buf_desc[]` entry is refilled every batch
- `ggml_hexagon_opqueue::push()` carries `dbuf` every batch, so the 214MB buffer's fd/size info is transmitted repeatedly
- the DSP-side [`prep_op_bufs()`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/main.c) reuses existing mappings by fd; when the total mapping exceeds the `max_vmem` budget, it evicts unused mappings and re-`HAP_mmap`s. A 214MB buffer alone takes a large chunk of the vmem budget, producing real repeated mmap/munmap cost under budget pressure
- buffer lifetime follows the tensor (`ggml_hexagon_shared_buffer` hangs off `buffer->context`), not the dspqueue, but every batch must still re-declare the reference

In theory Qualcomm ggml-hexagon could keep lm-head outside dspqueue (raw ION only), but that breaks the unified path: "a buffer must be registered in the batch descriptor to be accessible to DSP ops".

JZ ggml-hexagon's single mempool is all-or-nothing: mmap the pool once, all tensors are naturally accessible, and lm-head needs no special handling.

## 4. Uniform, role-blind cache maintenance

Qualcomm ggml-hexagon's per-batch cache maintenance has two layers, and neither can see tensor role (weight vs activation):

### 4.1 Descriptor packet flags (a few KB, not tensor data)

In `ggml_hexagon_opqueue::push()`, the dspqueue packet carries fixed flags:

```cpp
dbuf.flags = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
```

Note: `dbuf` is only the batch descriptor block (buf/tensor/op descriptors, a few KB inside a dedicated shared block, see [`ggml_hexagon_opqueue::push()`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp)); tensor data buffers never carry dspqueue flags. They are mapped once via `fastrpc_mmap` at alloc time, and the DSP skel mmaps their fds on first use (`prep_op_bufs`). The actual flush/invalidate logic behind these flags is hidden inside the closed-source Hexagon DSP driver.

### 4.2 Tensor data: the DSP-side full cache sweep (the root of the blunt policy)

The real cache maintenance for tensor data happens on the DSP side in [`process_opbatch()`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/main.c): a full D-cache flush+invalidate at batch start and again at batch end:

```c
qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
```

This full-cache sweep is thoroughly one-size-fits-all:

- it cannot distinguish weights from activations: the entire D-cache is flushed+invalidated twice per batch
- no mechanism can express "weights are written once at load and never invalidated after first touch": first-touch weight optimization has nowhere to live in this design

### 4.3 Contrast: JZ ggml-hexagon's role-aware cache optimizations

JZ ggml-hexagon's user-space cache optimizations consist of three mutually orthogonal mechanisms, named by scope: a global mechanism, a per-tensor mechanism, and a per-batch mechanism. All three are configured on the AP side, have no priority order, and operate on independent scopes.

| Mechanism | Field | Type | Controls | Purpose | Function |
|---|---|---|---|---|---|
| **Global** | `dsp_cache_mode` | 4-bit switch bitmask | DSP-side cache flush behavior | first-touch / dcinva skip / bulk dst flush | `ggmlhexagon_init_dsp` |
| **Per-tensor** | `td->flags` | per-tensor role tag | tensor role (weight/mirrored/normal) | distinguish weights so the first-touch path applies | `ggmlhexagon_backend_graph_compute_batch` |
| **Per-batch** | `ion_sync_mode` | 3-value mode selector | AP-side cache coherency mechanism (CVAC vs ion_sync) | skip manual DC CVAC, whole-pool kernel sync | `ggmlhexagon_backend_graph_compute_batch` (Phase 6.5/7.5) |

Details:

**Global mechanism: AP-side `dsp_cache_mode`** (see `struct hexagon_appcfg_t` in [`ggml-hexagon-jz.cpp`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp), [lines 402-405](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L402-L405)) is a 4-bit switch bitmask, pushed to the DSP as a whole in `ggmlhexagon_init_dsp`, controlling **DSP-side** cache flush behavior. **Default `dsp_cache_mode = 5` (0b0101)**: bit 0 (first-touch) and bit 2 (bulk dst flush) are both enabled by default; the other bits are off unless explicitly enabled.

- bit 0 (0x1): first-touch path **enabler**. When on, weight tensors with `td->flags=2` take the first-touch path: written once at load, and the DSP permanently skips invalidate after the first touch. JZ-side measurement: eliminates ~9.2 ms/token of redundant weight re-invalidation (with lm-head resident, per-token weight traffic is ~1.9GB; the number is measured bit0 off vs on, across all weights)
- bit 1 (0x2): skip dcinva for prior dst
- bit 2 (0x4): bulk dst flush at batch end
- bit 3 (0x8): selective bulk flush: skip dsts still consumed by later ops in the same batch

**Per-tensor mechanism: AP-side `td->flags`** (see `ggml-hexagon-jz.cpp` at [line 5793](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5793), [line 5799](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5799), [line 5802](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5802), [line 5827](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5827)) is a per-tensor role tag, pushed to the DSP via kernel_params:

- flags = 2: weight role (skip cache flush after first touch)
- flags = 1: mirrored (may share dst with later ops in the same batch)
- flags = 0: normal (regular per-batch cache maintenance)

**Per-batch mechanism: AP-side `ion_sync_mode`** (defined at `ggml-hexagon-jz.cpp` [line 394](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L394), controlled at [lines 5868-5873](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5868-L5873)) selects the **AP-side cache coherency mechanism**, deciding how Phase 6.5 (flush) and Phase 7.5 (invalidate) synchronize:

- mode = 0: both DC CVAC/CIVAC and DMA_BUF_IOCTL_SYNC (most conservative, not default)
- mode = 1: ion_sync only (code default) - skips manual DC CVAC/CIVAC and issues a single `ioctl(DMA_BUF_IOCTL_SYNC_IOCTL)`, a **whole-pool kernel sync**; also skips the per-tensor/cgraph range scans in Phase 6.5/7.5 ([lines 5897-5910](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5897-L5910), [lines 6045-6050](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L6045-L6050)), which are pure overhead under mode=1
- mode = 2: DC CVAC/CIVAC only - manual cache maintenance, no kernel sync

The global and per-tensor mechanisms govern **DSP-side** cache behavior (role-aware first-touch); the per-batch mechanism governs the **AP-side** cache coherency path (whole-pool ioctl vs per-tensor scans). The three mechanisms differ in scope, are orthogonal, and can be configured independently.

## 5. The 32768 guard is direct evidence of the design tradeoff

Qualcomm ggml-hexagon keeps an explicit lm-head rejection in [`ggml_hexagon_supported_mul_mat()`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp):

```cpp
// hardcoded limit to refuse the lm-head for now
if (src0->ne[1] > 32768) {
    return false;
}
```

The comment says "refuse the lm-head" outright: this is not a performance optimization, but an admission that the per-buffer design cannot host a 214MB resident weight at an acceptable cost.

- 32K rows is the "practical ceiling" of the per-buffer API (a single buffer of tens of MB, where fd/mmap cost is still acceptable)
- the 214MB lm-head far exceeds it; forcing it onto the DSP would be dragged down by per-buffer lifecycle cost

JZ ggml-hexagon has no such guard; lm-head offloads naturally.

## 6. One-line summary

Qualcomm ggml-hexagon's per-buffer design embeds an assumption: "every buffer is an independent, short-lived small object under a uniform cache policy". lm-head is the exact opposite: huge, resident for the whole session, and in need of a dedicated cache policy. Supporting lm-head would require symmetry-breaking changes across three layers - dspqueue, driver, and buffer descriptor - which is essentially converging toward the single mempool design.

JZ ggml-hexagon's single mempool is not an "aggressive design": it makes offloading lm-head to the DSP fall into place naturally - lm-head is just a range inside the pool, with no independent lifecycle, no per-buffer cost, and no need for a dedicated cache policy.

## 7. The "disadvantage becomes advantage" inversion

Qualcomm ggml-hexagon hands cache maintenance to a blunt batch-boundary operation (driver-handled descriptor packet + DSP-side full cache sweep); JZ ggml-hexagon appears "forced to manage the cache itself". But precisely because user space can see tensor role, distinguish weights from activations, and decide first-touch behavior, the 214MB lm-head becomes optimizable.

Contrast with Qualcomm ggml-hexagon:

- Apparent advantage: cache maintenance handled uniformly by the framework, less code
- Actual cost: DSP-side full D-cache flush+invalidate twice per batch, blind to tensor role; first-touch weight optimization cannot be expressed
- Apparent disadvantage: JZ ggml-hexagon must implement `ion_sync` + `dsp_cache_mode` itself
- Actual benefit: role-aware policy eliminates ~9.2 ms/token of redundant weight re-invalidation (JZ-side measurement, across all weights)

This is a classic layered-design inversion: the higher the abstraction layer, the smaller the optimization space; the lower the abstraction layer, the greater the policy flexibility and the larger the optimization space.

## 8. Reading the PP/TG gap

Multi-run means on the same Snapdragon 8 Elite phone (see [ion-mempool-vs-perbuffer-analysis-20260713.md](file:///home/zhouwg/develop/ggml-hexagon/docs/backend/jz-ggml-hexagon/ion-mempool-vs-perbuffer-analysis-20260713.md)):

- PP (686.46 vs 435.14 tok/s): cumulative effect of lifecycle overhead + IOVA locality
- TG (26.91 vs 24.91 tok/s): directly decided by whether lm-head is offloaded to the DSP
- the small TG gap (+8%) is not mainly explained by dspqueue pipelining: dspqueue's in-flight batches primarily help in PP, where multiple ops per token can be scheduled in parallel. TG is strictly sequential (each token depends on the previous one), so dspqueue's pipelining in TG is limited to descriptor-prep / DSP-compute overlap within a single token, and its contribution is naturally capped. The TG gap more directly reflects whether lm-head is offloaded to the DSP and the role-aware cache management difference

## 9. Layer count determines PP/TG crossover point

The PP/TG gap between JZ and Qualcomm is not fixed; it scales with model layer count:

```
JZ net advantage = per_layer_dsp_saving x n_layers - dspqueue_overlap_advantage
                    └── scales linearly with layers ──┘    └── fixed, does not scale ──┘
```

### 9.1 Two limits on Qualcomm's dspqueue overlap advantage

1. **LLM property limit**: TG is strictly sequential (each token depends on the previous one). dspqueue's 16 in-flight batches in TG can only overlap descriptor-prep with DSP compute within a single token; its contribution is naturally capped.

2. **Cannot offload lm-head limit**: Qualcomm keeps lm-head on CPU (32768 guard). In PP, the last op (lm-head) runs on CPU while the DSP idles, breaking the dspqueue pipeline. This is worse in PP than TG because PP's lm-head has m=batch_size, so CPU computation takes longer and DSP idle time is larger.

### 9.2 JZ advantage scales with layer count

JZ's per-layer DSP advantage (first-touch cache saving ~9.2ms/token + mempool zero overhead + HMX pipeline) accumulates linearly with layer count. Qualcomm's dspqueue overlap is a fixed advantage that does not scale with layers.

When `n_layers x per_layer_saving > dspqueue_overlap`, JZ also wins PP. The crossover depends on model structure.

### 9.3 Three-model comparison (Snapdragon 8 Elite, 2026-07-29)

Model structure facts verified from GGUF headers:

| Model | Layers | Attention | hidden | lm_head type | lm_head source | vocab | PP JZ vs QCOM | TG JZ vs QCOM |
|---|---|---|---|---|---|---|---|---|
| gemma4-E2B | 35 | GQA 8:1 | 1536 | Q4_K | tied (token_embd) | 262K | JZ wins | JZ wins |
| qwen3.5-2B | 25 | GQA 4:1 | 2048 | Q6_K | tied (token_embd) | 248K | QCOM wins | JZ wins (1.8x) |
| qwen1.5-1.8b | 24 | MHA 1:1 | 2048 | Q6_K | standalone | 152K | QCOM wins | QCOM wins |

TG numbers (JZ vs QCOM, tok/s):
- gemma4-E2B: 27.2 vs ~24.9 (JZ +9%)
- qwen3.5-2B: 27.39 vs 13.65 (JZ 1.8x, after Q6_K -> Q4_0 repack offload)
- qwen1.5-1.8b: ~19 vs 24.12 (JZ -21%, MHA corner case)

Key observations:
- gemma4 (35 layers, GQA 8:1): per-layer accumulation crosses both PP and TG thresholds; JZ wins PP & TG
- qwen3.5 (25 layers, GQA 4:1): crosses TG threshold but not PP threshold; dspqueue overlap still wins PP
- qwen1.5 (24 layers, MHA 1:1): MHA's large K/V matrices [2048,2048] cause VTCM pressure, reducing DSP per-layer efficiency; does not cross either threshold. Legacy MHA models are corner cases; modern models use GQA where JZ shines

### 9.4 Conclusion

JZ's architectural advantage grows with model layer count. The more layers a model has, the more JZ's per-layer DSP advantage accumulates, eventually overtaking Qualcomm's fixed dspqueue overlap advantage in both PP and TG. Modern GQA models with 30+ layers (e.g., gemma4) benefit the most from JZ's architecture.

## 10. Implications for Qualcomm ggml-hexagon

If Qualcomm ggml-hexagon improves in the future:

- split dspqueue buffers into two classes: "resident shared buffers" + "per-batch transient buffers"
- add a "weight role" concept to the driver/DSP skel
- allow buffers to exist independently of the dspqueue lifetime

But this is essentially redesigning per-buffer into "a few buffers + a pool" - in other words, the single ION mempool approach.
