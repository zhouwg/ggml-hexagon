# Performance Optimization Analysis: JZ's ggml-hexagon backend vs Qualcomm's ggml-hexagon backend

## English Summary

### Problem Statement

When only `MUL_MAT` with `F32` is offloaded to DSP, PP performance reaches 20 t/s.
When more ops (ADD, SUB, DIV, etc.) are also offloaded, PP drops to 8.8 t/s.
This means elementwise ops on DSP are SLOWER than ARM CPU in our custom backend.

### Root Cause: 4 Per-Tensor Overheads

Our custom backend (`ggml-hexagon.cpp`, `offload_cgraph_type==2`) pays 4 expensive
costs per elementwise op that the QCOM backend does not:

| Overhead | Our Backend | QCOM Backend |
|----------|-------------|--------------|
| Phase 4: heap->ION memcpy mirror | Every activation tensor memcpy'd | Zero-copy (tensor in shared buffer) |
| Phase 6.5: DC CVAC cache clean | Manual cache line loop | dspqueue flag auto-handled |
| Phase 7.5: DC CIVAC cache inval | Manual cache line loop | dspqueue flag auto-handled |
| Phase 8: ION->heap copy-back | Every dst tensor memcpy'd | Zero-copy (DSP writes shared buffer) |

For a `[N, 4096]` F32 ADD op:
- ARM NEON: ~1-3 us (data in cache, no cross-domain transfer)
- Our DSP path: 4x memcpy/cache + FastRPC latency = ~50-200 us

### Key Architecture Difference

| Aspect | QCOM Backend | Our Backend (special_ion) |
|--------|--------------|---------------------------|
| Tensor storage | `fastrpc_mmap` shared buffers (zero-copy) | ION mempool + heap mirrors (memcpy) |
| Cache sync | `dspqueue` flags (coherent infra) | Manual DC CVAC + DC CIVAC per call |
| Per-op data copy | None (descriptors only) | Heap->ION mirror + ION->heap copy-back |
| Size threshold for elementwise | None (zero-copy, no overhead) | None (this was the bug) |
| MUL_MAT N-threshold | None | `mulmat_min_n=30` |
| Graph cache | Yes (`graph->uid`) | No |
| Op fusion | Yes (RMS_NORM+MUL, QKV, FFN) | No |
| Pipelining | Yes (queue depth 16) | No (synchronous) |

### Why QCOM Backend Does Not Need MUL_MAT N-Threshold

QCOM backend's per-offload overhead is near-zero:
- No mirror/copy-back (tensors in shared buffers)
- No manual cache maintenance (dspqueue flags)
- FastRPC latency hidden by pipelining (queue depth=16)
- Graph plan cached by uid

For N=1 small MUL_MAT:
- QCOM: descriptor write + dspqueue_write = few us
- Ours: mirror(2MB) + DC CVAC(2MB) + FastRPC(50-200us) + DC CIVAC + copy-back(2MB) = 500us+

### Optimization Directions (Ranked)

1. **Add elementwise size threshold** (HIGH impact, LOW cost)
   - ~10 lines in `ggmlhexagon_can_handle_op_through_cdsp_special`
   - Small ops stay on ARM, large ops go to DSP
   - Config: `elementwise_min_elems` in ggml-hexagon.cfg

2. **Allocate activations in ION** (HIGH impact, MEDIUM cost)
   - Eliminates Phase 4 mirror and Phase 8 copy-back
   - Architectural fix matching QCOM's zero-copy design

3. **Skip unnecessary cache maintenance** (HIGH impact, MEDIUM cost)
   - Phase 6.5: only flush CPU-written tensors (not previous DSP op dst)
   - Phase 7.5: only inval tensors CPU will read (not next DSP op src)

4. **Graph cache** (MEDIUM impact, MEDIUM cost)
   - Cache tensor_index_map, hex_ops plan by `cgraph->uid`

5. **Op fusion** (MEDIUM impact, HIGH cost)
   - RMS_NORM+MUL, QKV triple, FFN pair fusion

6. **Move weight conversion to set_tensor** (MEDIUM impact, MEDIUM cost)
   - Q4_0->x4x2 repack currently done in graph_compute Phase 4.5
   - Should be done once at tensor load time, not per graph_compute

---

## Chinese Summary (中文总结)

### 问题描述

当仅 offload `MUL_MAT + F32` 到 DSP 时，PP 性能达到 20 t/s。
当 offload 更多算子（ADD/SUB/DIV 等）时，PP 降到 8.8 t/s。
说明在我们的后端中，elementwise 算子在 DSP 上比 ARM CPU 更慢。

### 根因：4 项 Per-Tensor 开销

我们的后端每个 elementwise 算子有 4 项 QCOM 后端没有的开销：

| 开销 | 我们的后端 | QCOM 后端 |
|------|-----------|-----------|
| Phase 4: heap->ION memcpy 镜像 | 每个 activation tensor 都要 memcpy | 零拷贝（tensor 在共享 buffer 中） |
| Phase 6.5: DC CVAC cache clean | 手动遍历 cache line | dspqueue flag 自动处理 |
| Phase 7.5: DC CIVAC cache inval | 手动遍历 cache line | dspqueue flag 自动处理 |
| Phase 8: ION->heap copy-back | 每个 dst tensor 都要 memcpy | 零拷贝（DSP 直接写共享 buffer） |

对于 `[N, 4096]` F32 的 ADD 算子：
- ARM NEON: ~1-3 us（数据在 cache 中，无需跨域传输）
- 我们的 DSP 路径: 4 次 memcpy/cache + FastRPC 延迟 = ~50-200 us

### 为什么 QCOM 后端不需要 MUL_MAT N-阈值

QCOM 后端的每次 offload 开销接近零：
- 无 mirror/copy-back（tensor 在共享 buffer 中）
- 无手动 cache 维护（dspqueue flag）
- FastRPC 延迟被 pipelining 隐藏（queue depth=16）
- Graph 计划按 uid 缓存

对于 N=1 的小 MUL_MAT：
- QCOM: 描述符写入 + dspqueue_write = 几 us
- 我们: mirror(2MB) + DC CVAC(2MB) + FastRPC(50-200us) + DC CIVAC + copy-back(2MB) = 500us+

### 关键架构差异

| 方面 | QCOM 后端 | 我们的后端 |
|------|----------|-----------|
| Tensor 存储 | fastrpc_mmap 共享 buffer（零拷贝） | ION mempool + heap 镜像（memcpy） |
| Cache 同步 | dspqueue flag（一致性基础设施） | 手动 DC CVAC + DC CIVAC |
| 每算子数据拷贝 | 无（仅描述符） | heap->ION 镜像 + ION->heap 回拷 |
| Graph 缓存 | 有（按 graph->uid） | 无 |
| Op fusion | 有（RMS_NORM+MUL, QKV, FFN） | 无 |
| 流水线 | 有（queue depth=16） | 无（同步） |

### 优化方向（按性价比排序）

1. **添加 elementwise size 阈值**（高收益，低成本）
   - 在 `ggmlhexagon_can_handle_op_through_cdsp_special` 中约 10 行改动
   - 小算子留在 ARM，大算子送 DSP
   - 配置项：`elementwise_min_elems`

2. **activation tensor 直接分配在 ION 中**（高收益，中等成本）
   - 消除 Phase 4 镜像和 Phase 8 回拷
   - 架构级修复，接近 QCOM 的零拷贝设计

3. **跳过不必要的 cache 维护**（高收益，中等成本）
   - Phase 6.5：只 flush CPU 写过的 tensor
   - Phase 7.5：只 inval CPU 要读的 tensor

4. **Graph 缓存**（中等收益，中等成本）
   - 按 `cgraph->uid` 缓存 tensor_index_map、hex_ops 计划

5. **Op fusion**（中等收益，高成本）
   - RMS_NORM+MUL、QKV 三合一、FFN pair fusion

6. **权重转换移到 set_tensor**（中等收益，中等成本）
   - Q4_0->x4x2 repack 当前在 graph_compute Phase 4.5 中做
   - 应该在 tensor 加载时做一次，而非每次 graph_compute 都做

### 关键文件和行号

- `ggml/src/ggml-hexagon/ggml-hexagon.cpp`:
  - 配置结构体: `hexagon_appcfg_t` (line ~280)
  - supports_op (relaxed): `ggmlhexagon_can_handle_op_through_cdsp_special` (line ~3710)
  - graph_compute: `ggmlhexagon_backend_graph_compute_special_ion` (line ~4683)
    - Phase 4 mirror: line ~4854
    - Phase 6.5 DC CVAC: line ~5132
    - Phase 7 FastRPC: line ~5176
    - Phase 7.5 DC CIVAC: line ~5218
    - Phase 8 copy-back: line ~5264

- `ggml/src/ggml-hexagon/ggml-hexagon-qcom.cpp`:
  - 共享 buffer: `ggml_hexagon_shared_buffer` (line ~255)
  - opbatch: `ggml_hexagon_opbatch` (line ~1086)
  - opqueue (pipelining): `ggml_hexagon_opqueue` (line ~1301)
  - graph_compute: `ggml_backend_hexagon_graph_compute` (line ~3363)
  - op fusion: `try_fuse_node` (line ~3305)
