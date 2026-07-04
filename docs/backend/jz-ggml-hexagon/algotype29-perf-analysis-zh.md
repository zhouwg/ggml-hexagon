# algotype=29 性能差异分析：JZ 版本 vs 高通版本

## 背景

当 `mulmat_algotype=29` 时，JZ 版本和高通版本的 ggml-hexagon backend 都走 Qualcomm 的 `execute_op` 路径（`execute_op` 实现位于 `htp/` 目录下，被两版本共享调用）。但 DSP 端入口不同：JZ 为 `kernels/entry.c`，高通为 `htp/main.c`。两者在 AP 端的实现差异巨大，导致性能显著不同。

本文档全面对比两个版本在 algotype=29 时的 AP 端差异，按性能影响程度从大到小排序。

## 相关文件

| 文件 | 说明 |
|------|------|
| `ggml/src/ggml-hexagon/ggml-hexagon.cpp` | JZ 版本 AP 代码（5491 行） |
| `ggml/src/ggml-hexagon/ggml-hexagon-qcom.cpp` | 高通版本 AP 代码（4392 行） |
| `ggml/src/ggml-hexagon/kernels/entry.c` | DSP 端入口（共享） |
| `ggml/src/ggml-hexagon/htp/matmul-ops.c` | 高通 DSP matmul kernel（共享） |

> 注意：忽略工程根目录下 `refs/` 目录下的所有文件。

---

## 1. 权重 repack 时机（最大差异）

### 问题描述

Qualcomm HMX kernel（`hvx_mm_2d_repacked_*`）期望 Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4 权重以 tile-based 布局存放。两个版本都做了 tiled repack，但时机完全不同。

### JZ 版本

- **时机**：每次 `graph_compute_batch` 调用
- **实现**：Phase 4.5 中 `g_tiled_ion_offsets.clear()` + 全量 repack
- **位置**：`ggml-hexagon.cpp` Phase 4.5（L4478-L4612）
- **开销**：每次推理都对所有量化权重重新做 tiled repack + memcpy
- **生命周期**：repack 到临时 ION region，用完在 Phase 8 后释放

```cpp
// JZ: 每次调用都 clear + 全量 repack
g_tiled_ion_offsets.clear();
for (uint32_t i = 0; i < n_tensors; i++) {
    // ... repack Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4 ...
}
```

### 高通版本

- **时机**：`set_tensor`（模型加载时，一次性）
- **实现**：`ggml_backend_hexagon_repack_buffer_type`
- **位置**：`ggml-hexagon-qcom.cpp` L1005, L860-906, L1058-1064
- **开销**：零（推理时直接使用 repacked 数据）
- **生命周期**：repacked 数据永久存放在 shared buffer 中

```cpp
// QCom: set_tensor 时 repack 一次
if (ggml_backend_buffer_is_hexagon_repack(buf)) {
    // repack to tiled layout, stored permanently in shared buffer
}
```

### 性能影响

对于一个典型 LLM 模型（几百 MB 量化权重），JZ 版本每次推理都做全量 repack，消耗大量 CPU 时间和内存带宽。这是 algotype=29 下最大的性能差异来源。

---

## 2. FastRPC 调用模式

### JZ 版本

- **调用方式**：单次同步 `ggmlop_dsp_execute_batch`
- **参数传递**：2 个 scalar（`batch_offset`, `total_desc_size`）
- **数据传递**：单个 ION mempool + offset 寻址
- **流水线**：无（AP 阻塞等待 DSP 完成）

```cpp
// JZ: 同步调用，AP 等 DSP
int hexagon_error = ggmlop_dsp_execute_batch(ctx->ggmlop_handle,
                                              batch_offset,
                                              total_desc_size);
```

### 高通版本

- **调用方式**：dspqueue 消息队列
- **参数传递**：`dspqueue_write` + `dspqueue_buffer`（含 `fd + offset + size`）
- **数据传递**：fd + offset 二级寻址，支持多个独立 shared buffer
- **流水线**：最多 16 batch 在途（`opt_opqueue=16`），AP/DSP 并行

```cpp
// QCom: dspqueue 流水线
int err = dspqueue_write(this->queue, 0, 1, &dbuf, sizeof(req),
                          (const uint8_t*) &req, DSPQUEUE_TIMEOUT);
```

### 性能影响

高通版本的流水线允许 AP 在 DSP 执行 batch N 时同时构建 batch N+1，隐藏了 AP 端的 batch 构建开销。JZ 版本是严格串行的：构建 batch -> 提交 -> 等 DSP -> 收回结果 -> 下一轮。

---

## 3. op fusion 范围

### JZ 版本（Phase 2.5，L4310-L4384）

| fusion 类型 | 支持 | 说明 |
|-------------|------|------|
| RMS_NORM + MUL -> HTP_OP_RMS_NORM_MUL | 是 | 简单线性扫描相邻 op |
| MUL_MAT + ADD -> HTP_OP_MUL_MAT_ADD | 是 | bias add / residual add |
| MUL_MAT QKV 合并 | **否** | - |
| MUL_MAT FFN 合并 | **否** | - |
| graph reorder | **否** | - |

安全检查：仅检查 `src_use_count == 1`（中间 dst 单次使用），无 VTCM 预算检查。

### 高通版本（`try_fuse_node`，L3473-L3549）

| fusion 类型 | 支持 | 说明 |
|-------------|------|------|
| RMS_NORM + MUL -> HTP_OP_RMS_NORM_MUL | 是 | 使用 `ggml_can_fuse` |
| MUL_MAT + ADD -> HTP_OP_MUL_MAT_ADD | 是 | 使用 `ggml_can_fuse` |
| MUL_MAT QKV 合并 -> HTP_OP_MUL_MAT_QKV | **是** | 3 个 mul_mat 合 1，重排为 KVQ 顺序 |
| MUL_MAT FFN 合并 -> HTP_OP_MUL_MAT_FFN | **是** | gate + up 合 1 |
| graph reorder | **是** | 相同 src1 的 MUL_MAT 堆叠，便于 VTCM 复用 |

安全检查：`ggml_can_fuse` + VTCM 预算检查（`kparams.vtcm_size <= sess->vtcm_size`）+ `is_mergeable_mul_mat`。

### 性能影响

QKV fusion 减少每层 2 次 MUL_MAT dispatch，FFN fusion 减少每层 1 次。graph reorder 优化 src1 的 VTCM 复用。这些在 PP 阶段（batch processing）影响显著。

---

## 4. graph cache

### JZ 版本

- **graph 缓存**：无
- **每次推理**：重新构建 hex_op_desc 数组、重新做 op fusion、重新做 weight repack

### 高通版本

- **graph 缓存**：按 `graph->uid` 缓存 htp_nodes（L3559-L3600）
- **命中时**：跳过 fusion + precompute，直接复用缓存的 op 描述符
- **op reorder**：`graph_optimize_reorder`（L3624-L3669），将相同 src1 的 MUL_MAT 堆叠

### 性能影响

高通版本在重复推理（如 TG 阶段的相同 graph）时可以跳过 AP 端的图构建开销。JZ 版本每次都从头开始。

---

## 5. cache coherency 管理

### JZ 版本

- **管理方式**：手动管理 DC CVAC/CIVAC
- **flush**：Phase 6.5 中按 range 合并手动 flush
- **invalidate**：Phase 7.5 中按 range 合并手动 inval

### 高通版本

- **管理方式**：dspqueue 驱动自动管理
- **flag**：`DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT`
- **优势**：驱动层优化，减少用户态开销

### 性能影响

手动 cache 管理在大量 tensor 场景下有用户态开销。dspqueue 的驱动层管理更高效。

---

## 6. batch 自动切分

### JZ 版本

- **策略**：全图一个 batch，不切分
- **限制**：可能受 ION pool 大小限制

### 高通版本

- **策略**：按 vmem/buffer/tensor 上限自动切分多 batch
- **实现**：`enqueue_op` 中 `if (!op_batch->fit_op(node)) flush_batch()`
- **优势**：适应不同大小的 graph，避免内存溢出

---

## 7. tensor descriptor 数据结构

### JZ 版本 `hex_tensor_desc`

```c
typedef struct hex_tensor_desc {
    int32_t  type;
    int32_t  ne[4];
    int32_t  nb[4];
    int32_t  op_params[16];  // op-specific params（含 FP16 cache 请求）
    uint32_t flags;          // 0=ION, 1=mirrored, 2=weight(skip flush)
    uint32_t data_offset;   // 相对于 ION mempool base 的偏移
    uint32_t data_len;
} hex_tensor_desc;
```

- 单一 ION offset 寻址
- flags 编码 cache 策略
- op_params 嵌在 tensor 描述符里

### 高通版本 `htp_tensor`

```c
struct htp_tensor {
    uint32_t data;       // buffer 内偏移
    uint32_t size;
    uint32_t flags;      // HTP_TENSOR_COMPUTE / HTP_TENSOR_FLUSHED
    uint16_t type;
    uint16_t bi;         // buffer index（指向 htp_buf_desc 数组）
    uint32_t ne[4];
    uint32_t nb[4];
};
```

- 二级寻址：`bi`（buffer index）+ `data`（offset）
- 支持多个独立 shared buffer
- fd 可被 DSP 直接 mmap

---

## 总结：性能差异原因排序

| 排名 | 差异 | JZ 版本 | 高通版本 | 影响程度 |
|------|------|---------|---------|---------|
| 1 | 权重 repack 时机 | 每次 graph_compute_batch | set_tensor 一次 | **最大** |
| 2 | FastRPC 调用模式 | 同步阻塞 | dspqueue 流水线 | 大 |
| 3 | op fusion 范围 | 2 种 fusion | 4 种 fusion + reorder | 中 |
| 4 | graph cache | 无 | 有（按 uid 缓存） | 中 |
| 5 | cache coherency | 手动管理 | 驱动自动管理 | 小 |
| 6 | batch 自动切分 | 全图一个 batch | 自动切分 | 小 |

---

## 优化建议

如果要提升 JZ 版本 algotype=29 的性能，按性价比排序：

### 优先级 1：把 tiled repack 移到 set_tensor（性价比最高）

将 Phase 4.5 的 per-call repack 改为 set_tensor 时的 one-time repack，类似高通版本的 repack buffer type 设计。这可以完全消除每次推理的权重 repack 开销。

### 优先级 2：引入 graph cache

按 `graph->uid` 缓存 hex_op_desc 数组和 fusion 结果，避免重复构建。

### 优先级 3：扩展 op fusion

添加 QKV fusion 和 FFN fusion，减少 MUL_MAT dispatch 次数。

### 优先级 4：引入 dspqueue 流水线（改动最大）

从同步 FastRPC 调用改为 dspqueue 消息队列，实现 AP/DSP 流水线。这需要较大的架构改动。
