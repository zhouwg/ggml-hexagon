# Buffer 用途分析报告

> [!WARNING]
> **本文档基于 pre-remove-dual-path 状态编写,部分描述已过时。**
>
> 最新且与代码一致的权威分析是
> [algotype29-perf-analysis-en-20260711.md](algotype29-perf-analysis-en-20260711.md)。
>
> 已知过时内容(待修正):
> - 第一章 "ION Pool 布局" 中的 "FP16 weight cache: 3452 ~ 3964 MiB (512 MiB, 用于 weight repack)" — **不存在**。FP16 weight cache 是 algotype=32 (self-built) 路径的遗留概念,见 [HMX_PIPELINE_MIGRATION.md:76,123](HMX_PIPELINE_MIGRATION.md)。post-remove-dual-path 后 ION pool 是单一连续区,无此 region。
> - 第一章 "配置来源: ggml-hexagon.cpp:2751 `ION layout: total=3964MB, cache_offset=3452MB, cache_size=512MB, data_region=3452MB`" — **line 2751 实际是 `is_repack` 标志计算**,不是 ION layout 配置。ION pool 大小来源于 `ggml-hexagon-jz.cpp:1764` 的日志输出。
> - 第七章 "Phase 4.5 x4x2 repack(部分 F16 权重)" — 应为 32x32 tiled quantized repack (Q4_0/Q4_1/Q8_0/MXFP4),保持量化类型,**不是 F16 格式**。详见 [algotype29-perf-analysis-en-20260711.md:266-308](algotype29-perf-analysis-en-20260711.md#L266-L308)。
>
> 第四-七章关于 ION mirror、cache coherency、CC-1~6 的分析仍然有效,描述了当前 JZ 实现的核心机制。

基于 `log_ap.txt` 和源码分析，以下是 4 次 `alloc_buffer` 调用的完整归属分析。

---

## 一、ION Pool 布局

```
ION Pool (total=3964 MiB, fd=17)
|-- data_region: 0 ~ 3452 MiB       (用于 tensor 分配)
|   |-- #1 模型权重:    0 ~ 1128 MiB       (1183088640 bytes)
|   |-- #2 ISWA kv_base: 1128 ~ 1896 MiB   (805306368 bytes)
|   |-- #3 ISWA kv_swa:  1896 ~ 1908 MiB   (12582912 bytes)
|   `-- 剩余可用:        1908 ~ 3452 MiB   (1544 MiB) <- 不够 #4 的 2449 MiB
|
`-- FP16 weight cache: 3452 ~ 3964 MiB   (512 MiB, 用于 weight repack)
```

配置来源：[ggml-hexagon.cpp:2751](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2751) `ION layout: total=3964MB, cache_offset=3452MB, cache_size=512MB, data_region=3452MB`

---

## 二、4 次 alloc_buffer 归属

### 时间线（log_ap.txt line 133-149）

| # | 时间 | 大小 | 位置 | pool_used | 用途 |
|---|------|------|------|-----------|------|
| 1 | 09:53:52.001 | 1128.28 MiB | ION-pool offset=0 | 28.46% | **模型权重** |
| 2 | 09:53:54.035 | 768.00 MiB | ION-pool offset=1128MiB | 47.84% | **ISWA kv_base** |
| 3 | 09:53:54.101 | 12.00 MiB | ION-pool offset=1896MiB | 48.14% | **ISWA kv_swa** |
| 4 | 09:53:54.106 | 2449.00 MiB | **heap** | 48.14% | **1st PP compute buffer** |

### 归属判定依据

**#1 (1128 MiB) - 模型权重**
- 时间：09:53:52.001，在第二次 `ggml_backend_hexagon_init`（09:53:54.033）**之前 2 秒**
- 来源：[llama-model.cpp:1497-1551](file:///home/zhouwg/develop/ggml-hexagon/src/llama-model.cpp#L1497) 按 buft 分组调用 `ggml_backend_alloc_ctx_tensors_from_buft`
- 大小合理：Gemma 4 E2B Q4_0 模型权重约 1.1-1.2 GiB

**#2 (768 MiB) - ISWA kv_base（非 SWA 层 KV cache）**
- 时间：09:53:54.035，在 backend init 后 2ms
- 来源：[llama-kv-cache-iswa.h:80](file:///home/zhouwg/develop/ggml-hexagon/src/llama-kv-cache-iswa.h#L80) `std::unique_ptr<llama_kv_cache> kv_base` 独立调用 `ggml_backend_alloc_ctx_tensors_from_buft`
- 大小合理：假设 12 层非 SWA，n_ctx~=8192，n_embd_head=256，n_head_kv=8，F16 -> 每层 K+V = 2x8192x256x8x2 = 64 MiB -> 12x64 = 768 MiB

**#3 (12 MiB) - ISWA kv_swa（SWA 层 KV cache）**
- 时间：09:53:54.101，#2 后 66ms
- 来源：[llama-kv-cache-iswa.h:81](file:///home/zhouwg/develop/ggml-hexagon/src/llama-kv-cache-iswa.h#L81) `std::unique_ptr<llama_kv_cache> kv_swa` 独立调用 `ggml_backend_alloc_ctx_tensors_from_buft`
- 大小合理：SWA window=1024，受 `n_kv_shared_layers` 机制影响，实际有 KV cache 的层很少 -> 12 MiB

**#4 (2449 MiB heap) - 1st PP compute buffer**
- 时间：09:53:54.106，#3 后 5ms
- 来源：[llama-context.cpp:612](file:///home/zhouwg/develop/ggml-hexagon/src/llama-context.cpp#L612) `sched_reserve` 中 1st PP `graph_reserve` 触发 `ggml_gallocr_reserve_n` -> `ggml_vbuffer_alloc` -> `alloc_buffer`
- **fallback 原因**：ION pool data_region 剩余 1543 MiB < 需要 2449 MiB
- 大小合理：35 层 PP compute（n_tokens=2048, n_embd=2048, n_ff=8192），考虑 gallocr 的 reuse 后约 2.4 GiB

---

## 三、为什么 TG 和 2nd PP 不触发 alloc_buffer

[llama-context.cpp:610-651](file:///home/zhouwg/develop/ggml-hexagon/src/llama-context.cpp#L610) 中 `sched_reserve` 调用 3 次 `graph_reserve`：

```
1st PP (line 612): graph_reserve(n_tokens, n_seqs, n_outputs_pp, ...)  // 触发 #4
TG    (line 632): graph_reserve(n_seqs,   n_seqs, n_seqs,      ...)    // reuse #4
2nd PP (line 647): graph_reserve(n_tokens, n_seqs, n_outputs_pp, ...)  // reuse #4
```

[ggml-alloc.c:904-944](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-alloc.c#L904) 的 `ggml_gallocr_reserve_n_impl` 重用逻辑：

```c
// 仅当 new_chunk_size > cur_chunk_size 时才 realloc
if (new_chunk_size > cur_chunk_size) {
    realloc = true;
}
```

- TG (12 MiB) < 1st PP (2449 MiB) -> 不触发 realloc
- 2nd PP 参数与 1st PP 相同 -> 不触发 realloc

所以 4 次 alloc_buffer 全部发生在 sched_reserve 之前（#1 模型权重 + #2/#3 KV cache）和 sched_reserve 1st PP（#4）。

---

## 四、#4 heap buffer 的 cache 一致性问题

### 问题根源

[ggml-hexagon.cpp:4196](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L4196) 当 ION pool 不足时 fallback 到 heap：

```cpp
GGMLHEXAGON_LOG_WARN("ion pool exhausted: needed %zu MiB, remaining %zu MiB -- falling back to system memory", ...);
buffer_ctx->buffer = ggml_aligned_malloc(size_aligned);
```

这导致 PP compute buffer 分配在 heap 上，而 DSP 无法直接访问 heap 内存。

### ION mirror 机制（补救方案）

[ggml-hexagon.cpp:4452-4622](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L4452) 和 [line 4810+](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L4810) 实现了 ION mirror 机制：

```
Phase 1 (mirror-in):  heap tensor -> ION mirror (memcpy + swap tensor->data)
Phase 6.5:            AP->DSP cache flush (DC CVAC + DSB + DMA_BUF_IOCTL_SYNC WRITE)
Phase 7:              DSP execute (读写 ION mirror)
Phase 7.5:            DSP->AP cache invalidate (DC CIVAC + DSB + DMA_BUF_IOCTL_SYNC READ)
Phase 3 (mirror-out): ION mirror -> heap (memcpy copy-back)
```

### 日志中的问题证据

log_ap.txt line 155（第一次 PP graph_compute）：

```
[AP-POST] batch last-op[0] dst[tensor2]:
  ION_f32=[-4.2930, 22.5000, -0.8926, 4.6211]   <- ION mirror 中有正确数据
  PTR_f32=[0.0000, 0.0000, 0.0000, 0.0000]      <- heap 中是旧数据(全0)
  ion_off=0x77448000  ptr=0xb400007cfc43a800
```

`ION_f32` 有值但 `PTR_f32=[0,0,0,0]`，说明：
- DSP 正确写入了 ION mirror
- 但 heap 中的 `dst_tensor->data` 还没有更新（copy-back 未执行或 cache 未失效）

这正是 [ggml-hexagon.cpp:5474-5489](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L5474) `supports_buft` 注释中警告的问题：

> `ATTENTION: in ION mempool mode, only support hexagon buffer type (ION memory).`
> `Do NOT accept host buffer type, otherwise the scheduler will allocate tensors on the heap, requiring ION mirror + copy-back which has unsolvable cache coherency issues on ARM64 (no DC IVAC in user-space).`

---

## 五、关键结论

1. **#4 (2449 MiB heap) 是 cache 一致性问题的直接根源**
   - ION pool data_region (3452 MiB) 被模型权重 + KV cache 占用 1908 MiB 后，剩余 1543 MiB 不够 PP compute buffer (2449 MiB)
   - 触发 heap fallback，进入 ION mirror + copy-back 路径
   - ARM64 用户空间无法 invalidate D-cache（无 DC IVAC 指令），导致 cache 一致性问题

2. **可能的修复方向**：
   - **增大 ION pool**：让 2449 MiB PP compute buffer 能放进 ION pool（需要 data_region >= 1908 + 2449 = 4357 MiB，但当前 total 只有 3964 MiB）
   - **减小 PP compute buffer**：限制 n_tokens（如 `--ubatch-size 1024`）或启用 flash attention（`-fa`）减少中间结果
   - **减小 KV cache**：使用 `--no-kv-offload` 让 KV cache 留在 CPU，腾出 ION pool 给 compute buffer
   - **修复 ION mirror 的 cache 一致性**：在 mirror-out 后对 heap 地址做 cache invalidate（需要内核驱动支持）

3. **与高通实现的对比**：高通基于 dspqueue 的实现通过 `DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT` 让 dspqueue 在 ioctl 层自动处理 cache 一致性，不需要手动 mirror + copy-back，因此不存在此问题。

---

## 六、对照实验验证（--no-kv-offload）

为验证第五节结论 1（"#4 heap fallback 是 cache 一致性问题的直接根源"），使用 `--no-kv-offload` 让 KV cache 留在 CPU，腾出 ION pool 给 compute buffer，重新运行测试。

- 原始日志：[log_ap.txt](file:///home/zhouwg/develop/ggml-hexagon/log_ap.txt)
- 对照日志：[log_ap_no-kv-offload.txt](file:///home/zhouwg/develop/ggml-hexagon/log_ap_no-kv-offload.txt)
- 测试模型：gemma-4-E2B-it-Q4_0.gguf
- running_params：`-ngl 99 -t 6 -n 256 --no-mmap --poll 1000`（原始）vs 追加 `--no-kv-offload`（对照）

### 6.1 alloc_buffer 对比

| 项 | 原始（log_ap.txt） | 对照（--no-kv-offload） |
|----|--------------------|--------------------------|
| alloc_buffer 总次数 | 4 | 2 |
| #1 模型权重 | 1128.28 MiB, ION-pool, offset=0 | 1128.28 MiB, ION-pool, offset=0 |
| #2 ISWA kv_base | 768.00 MiB, ION-pool, offset=1128MiB | （无，KV cache 在 CPU） |
| #3 ISWA kv_swa | 12.00 MiB, ION-pool, offset=1896MiB | （无，KV cache 在 CPU） |
| #4 PP compute buffer | 2449.00 MiB, **heap fallback** | 62.00 MiB, ION-pool, offset=1128MiB |
| ION pool 占用峰值 | 1908 MiB (48.14%) | 1190 MiB (30.03%) |
| heap fallback | **有**（needed 2449 MiB, remaining 1543 MiB） | **无** |

对照实验中 PP compute buffer 从 2449 MiB 缩小到 62 MiB，原因：KV cache 留在 CPU 后，attention 层的 K/V 相关中间张量不再进入 DSP compute graph，graph 变小（多数 graph 仅 1~2 个 node，且都是 MUL_MAT），compute buffer 需求大幅下降。

### 6.2 [AP-POST] 日志对比（注意：此日志在 copy-back 之前）

第一次 PP graph_compute 的 `[AP-POST]` 输出（dst tensor 的前 4 个 float）：

| 项 | 原始（log_ap.txt:155） | 对照（log_ap_no-kv-offload.txt:148） |
|----|------------------------|----------------------------------------|
| ION_f32 | [-4.2930, 22.5000, -0.8926, 4.6211] | [-4.2930, 22.5000, -0.8926, 4.6211] |
| PTR_f32 | **[0.0000, 0.0000, 0.0000, 0.0000]** | **[-4.2930, 22.5000, -0.8926, 4.6211]** |
| ion_off | 0x77448000 | 0x46882800 |
| ptr | 0xb400007cfc43a800 | 0x74a587e800 |
| 一致性 | **不一致**（ION 有值，heap 全 0） | **一致** |

**重要更正**：后续代码审查发现，`[AP-POST]` 日志位于 [ggml-hexagon.cpp:5177](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L5177) 的 Phase 7.3，**在 Phase 7.5 (cache invalidate) 和 Phase 8 (copy-back) 之前**。因此：

- `ION_f32` 读取 ION mirror（AP cache 可能未被 invalidate，显示的是自然驱逐后的 DRAM 数据）
- `PTR_f32` 读取 heap（copy-back 尚未执行，显示的是旧值）

`PTR_f32=[0,0,0,0]` 在 Phase 7.3 是 **expected behavior**，不表示 cache 一致性 bug。对照实验中 `PTR_f32 == ION_f32` 只是因为 dst tensor 在 ION pool 中（两个指针指向同一地址），并非 cache 一致性更好。

要验证 copy-back 是否真正生效，需启用 `dump_diag_info=1` 查看 [ggml-hexagon.cpp:5269](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L5269) 的 `[POST-COPY]` 日志（在 Phase 8 之后）。详见第七节分析。

### 6.3 性能对比

| 项 | 原始 | 对照（--no-kv-offload） | 变化 |
|----|------|--------------------------|------|
| PP (prompt eval) | 8.65 t/s | 2.84 t/s | -67% |
| TG (text gen) | 17.08 t/s | 17.14 t/s | +0.4% (持平) |

- **TG 持平**：TG 单 token 计算，compute buffer 本就很小，原始配置下也不触发 heap fallback，所以无差异。
- **PP 大幅下降**：KV cache 移到 CPU 后，attention 相关计算（K/V projection、softmax、score\*V 等）回退到 CPU 执行，PP 阶段的 DSP offload 比例显著降低，每秒处理 token 数从 8.65 跌到 2.84。

这印证了"PP 阶段是 DSP offload 的主要受益方"，也说明 `--no-kv-offload` 只能作为验证手段，不是可接受的最终修复方案。

### 6.4 验证结论（已更正）

1. **#4 heap fallback 触发 ION mirror 路径** - 原始配置下 PP compute buffer (2449 MiB) 超出 ION pool 剩余空间 (1543 MiB)，fallback 到 heap，进入 ION mirror + copy-back 路径。对照实验消除 heap fallback 后，mirror 路径不再被触发。
2. **[AP-POST] 日志的 PTR_f32=[0,0,0,0] 是 expected behavior**（更正此前误判）- 该日志在 Phase 7.3（copy-back 之前），heap 此时还是旧值。Phase 8 的 copy-back（DC CIVAC + memcpy）才是真正更新 heap 的步骤。此前据此判断 "ION mirror + copy-back 不可靠" 是误判。
3. **--no-kv-offload 不是最终方案** - 虽然消除了 heap fallback，但 PP 性能从 8.65 跌到 2.84 t/s（-67%），代价过大。
4. **真正的 corner case 在 ION mirror 的内存管理** - 详见第七节分析，核心问题是 `rpc_mempool_usage` 内存泄漏导致多次 graph_compute 后 ION pool 耗尽。

---

## 七、ION mirror corner case 分析与修复方案

### 7.1 ION mirror 完整流程

`ggmlhexagon_backend_graph_compute_special_ion`（[ggml-hexagon.cpp:4656](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L4656)）核心流程：

```
Phase 1   分析 cgraph，划分 op batch
Phase 2   收集 weight_indices（来自 ION pool 的权重）
Phase 3   对 dst/intermediate tensor 决定输出位置
Phase 4   为 heap tensor 分配 ION mirror (bump from rpc_mempool_usage)
            [line 4850: aligned_offset = (usage + 127) & ~127]
            [line 4859: ctx->rpc_mempool_usage = aligned_offset + mirror_size]  <- 只增不减!
Phase 4.5 x4x2 repack（部分 F16 权重）
            [line 4936: ctx->rpc_mempool_usage = aligned_offset + repack_size]
Phase 5   构建 batch descriptor
            [line 4981: ctx->rpc_mempool_usage = batch_offset_aligned + total_desc_size]
Phase 6   构建 tensor descriptors（含 data_offset）
            [line 5020-5036: 若 mirror 失败，data_offset=0 指向权重!]
Phase 6.5 AP->DSP cache flush (DC CVAC + DMA_BUF_IOCTL_SYNC WRITE)
            [line 5115: if (weight_indices.count(i)) continue; <- 跳过权重]
Phase 7   DSP execute
Phase 7.3 [AP-POST] 日志（copy-back 之前!）
            [line 5177: 此时 heap 仍是旧值]
Phase 7.5 DSP->AP cache invalidate (DC CIVAC + DMA_BUF_IOCTL_SYNC READ)
Phase 8   copy-back (memcpy ION->heap) + [POST-COPY] 日志
            [line 5269: [POST-COPY] 在 copy-back 之后，可用于验证]
Phase 8   释放 ion_regions
            [line 5283: ctx->ion_regions[ri].in_use = false; <- 只标记，不回退 bump pointer!]
```

cache 一致性机制本身是正确的（DC CVAC -> DSP execute -> DC CIVAC -> memcpy copy-back）。此前据 [AP-POST] 判断"不可靠"是误判（见 CC-2）。真正的 corner case 集中在内存管理。

### 7.2 六个 corner case

#### CC-1: rpc_mempool_usage 内存泄漏（核心问题）

**位置**：5 处赋值全部递增，0 处递减
- [line 4502](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L4502): legacy mirror
- [line 4859](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L4859): ION mirror
- [line 4936](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L4936): x4x2 repack
- [line 4981](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L4981): batch desc
- [line 5787](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L5787): another alloc

**释放**：[line 5283](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L5283) 仅 `in_use = false`，**不回退 rpc_mempool_usage 指针**。

**影响**：每次 graph_compute 后 ION pool 占用累计增长。多次调用后（testops 跑数千 case，或长 prompt 多个 PP batch），rpc_mempool_usage 触顶，[line 4850 的容量检查](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L4850) 失败 -> mirror 分配失败 -> data_offset=0 -> DSP 读写模型权重 -> 数据损坏 -> 结果不一致。

**对照实验解释**：单次 PP/TG 推理不会触发（初始 rpc_mempool_usage 有足够空间），testops 连续跑多个 case 累积到上限后才暴露。

#### CC-2: [AP-POST] 日志位置误导（已在 6.2 更正）

**位置**：[line 5177](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L5177) Phase 7.3，在 Phase 7.5 (cache invalidate) 和 Phase 8 (copy-back) **之前**。

**误判**：此前据 `PTR_f32=[0,0,0,0]` 判断 "ION mirror + copy-back 不可靠"。

**真相**：copy-back 尚未执行，heap 显示旧值是 expected behavior。验证 copy-back 应用 [POST-COPY] 日志（[line 5269](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L5269)，Phase 8 之后）。

#### CC-3: Phase 6.5 跳过权重

**位置**：[line 5115](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L5115) `if (weight_indices.count(i)) continue;`

**设计意图**：权重在加载时已 flush 过一次（[alloc_buffer](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L4196) 路径），运行时 read-only，无需重复 flush。

**潜在风险**：DMA_BUF_IOCTL_SYNC 是全 buffer 粒度（[ion_sync_for_direction line 746](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L746)），SYNC_START|WRITE 实际覆盖全 buffer，理论上不应有 stale cache；但 DC CVAC 的范围跳过权重，若 AP cache 中权重 cache line 在两次 graph_compute 之间被其他操作污染（不应该，但 best-fit reuse 可能命中权重区域？），跳过可能漏 flush。

**测试方案**：临时删除 line 5115 的 continue，对比 testops 结果。若一致则跳过逻辑正确，若不一致则存在 cache 污染。

#### CC-4: DC CIVAC vs DC IVAC

**位置**：[cpu_dcache_inval_range line 811](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L811)。

**ARM64 限制**：EL0 (user-space) 无 `dc ivac`（EL1 only），只能用 `dc civac` (Clean + Invalidate)。

**实际等价性**：Phase 7.5 时 cache line 应为 clean（Phase 6.5 已 clean，DSP 写后 AP cache 自然驱逐到 DRAM），所以 civac 等价于 ivac。若 CC-3 跳过权重且权重在 AP cache 中是 dirty 的，civac 会先写回再 invalidate - 但因权重 read-only，dirty 不应发生。

#### CC-5: DMA_BUF_IOCTL_SYNC 全 buffer 粒度

**位置**：[ion_sync_for_direction line 746](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L746)。

**问题**：`struct { uint64_t flags; }` 不含 range 参数，对整个 3964 MiB ION buffer 做 SYNC_START|END。

**影响**：性能（全 buffer sync 远慢于 range sync）；正确性（因 Phase 6.5 DC CVAC 已对需 flush 的区域做范围 flush，DMA_BUF_IOCTL_SYNC 作为额外全量 sync 不会引入错误，只是冗余）。

**结论**：性能问题，非性命攸关。可考虑改用 `DMA_BUF_IOCTL_SYNC_PARTIAL`（需内核支持）。

#### CC-6: data_offset=0 指向模型权重

**位置**：[line 5020-5036](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L5020)。

**触发条件**：mirror 分配失败（CC-1 累积后）或 buffer_mirrors_map 中无记录。

**影响**：DSP 写入 data_offset=0 会破坏模型权重，导致后续所有计算错误。

**修复方案**：P0 修 CC-1 后 mirror 分配总能成功；P1 mirror 失败时 fallback 该 op 到 CPU（而非 data_offset=0）。

### 7.3 修复方案表

| 优先级 | 方案 | 修复 corner case | 实现位置 | 风险 |
|--------|------|------------------|----------|------|
| **P0** | graph_compute 开头保存 rpc_mempool_usage，结尾恢复 | CC-1, CC-6 | 函数入口/出口 (line 4656/5284) | 低，纯内存管理 |
| **P1** | mirror 失败时 fallback 该 op 到 CPU | CC-6 | line 5032 附近 | 中，scheduler 可能不正确处理混合执行（用户已确认此前测试有问题） |
| **P2** | 启用 dump_diag_info=1 验证 [POST-COPY] 一致性 | CC-2 | 运行时参数 | 低，仅观测 |
| **P3** | 临时删除 line 5115 的 continue，对比 testops | CC-3, CC-4 | line 5115 | 低，仅性能影响 |

### 7.4 高通 `--ubatch-size 1024` 分析

高通测试脚本 `--ubatch-size 1024`（vs 默认 2048）的作用：

1. **compute buffer 体积减半**：PP compute buffer 与 ubatch 成正比，从约 2449 MiB 降到约 1225 MiB。
2. **降低 ION pool 瞬时压力**：即便高通用 dspqueue 自动处理 cache 一致性（无 mirror + copy-back），DSP 端 VTCM (8 MB) 和 HMX scratch 仍吃不下过大 batch 的中间张量。
3. **dspqueue 也有限制**：dspqueue 虽然避免了 mirror + copy-back，但 buffer 仍需在 ION/DMA-heap 中分配，过大 batch 仍会耗尽。

**结论**：`--ubatch-size 1024` 是高通针对 8Elite DSP 容量限制的 workaround，与我们的 CC-1 是不同层面的问题。但若我们暂时不修 CC-1，也可借鉴 `--ubatch-size 1024` 作为临时缓解（减小 mirror 累积速率）。

### 7.5 关键结论

1. **CC-1 (rpc_mempool_usage 泄漏) 是 testops 不稳定和算子干扰的根因**。
2. **cache 一致性机制本身是正确的**（DC CVAC -> DSP -> DC CIVAC -> memcpy），此前"不可靠"判断是误判。
3. **[AP-POST] 日志位置误导**，验证 copy-back 应用 [POST-COPY]。
4. **修复优先级**：P0 (CC-1) > P2 (验证) > P3 (CC-3 测试) > P1 (CC-6 fallback)。
5. **fallback CPU 是兜底**，正常情况下不应被触发，否则说明 P0 未完全修复。
