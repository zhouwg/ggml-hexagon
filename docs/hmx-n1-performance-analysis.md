# DSP HMX路径N=1性能瓶颈分析

## 1. 问题描述

在ggml-hexagon后端中，使用`mulmat_algotype=30`（HMX + x4x2 repack）进行Q4_0模型推理时，
decode阶段（N=1）性能极差（~0.3-0.4 tok/s），而`enabled_types=F32`基线可达19.69 tok/s，
差距约50倍。

## 2. 已排除的假设

| 假设 | 排除原因 |
|------|---------|
| CPU-DSP并行度差异 | LLM推理的token依赖性决定了无法真正并行，高通的`synchronize`也是阻塞等待 |
| Heap->ION镜像开销 | F32-only模式也有此执行路径，但性能好很多。开销本身不大 |
| x4x2反量化是瓶颈 | x4x2比Q4_0反量化快27%，但整体性能仍无显著提升 |
| FastRPC调用次数差异 | 两种模式的FastRPC调用次数相似（每个subgraph 1-2个node） |

## 3. 根因分析

### 3.1 F32-only为什么快

当`enabled_types=F32`时，`ggmlhexagon_type_is_enabled(Q4_0)`返回false，
**Q4_0 MUL_MAT根本不走DSP**，全部留在CPU上执行。

- CPU用ARM NEON的SDOT指令处理Q4_0 MUL_MAT非常高效（llama.cpp高度优化）
- DSP只处理少量F32 op，执行时间短
- **总时间由CPU的快速Q4_0 MUL_MAT主导**

### 3.2 algotype=30为什么慢

当`enabled_types=Q4_0`时，**所有Q4_0 MUL_MAT走DSP的HMX路径**。
但HMX对N=1（decode）效率极低：

| 因素 | N=1 (decode) | N>32 (PP) |
|------|-------------|-----------|
| HMX tile利用率 | 1/32 = 3%（32x32 tile只用1列） | ~100% |
| Pipeline模式 | 不启用（m <= 32） | 4级流水线（DMA->dequant->HMX->store） |
| 多线程dequant | 不启用 | 启用 |
| HMX异步队列 | 不启用 | 启用 |
| 双缓冲 | 不启用 | 启用 |

### 3.3 高通实现的关键差异

高通`hmx_matmul_2d_f32`（[hmx-matmul-ops.c:1198](../ggml/src/ggml-hexagon/htp/hmx-matmul-ops.c#L1198)）：

```c
const bool use_pipeline = (m > 32);
const int  num_threads  = (m <= 32) ? 1 : ctx->n_threads;
```

- **m > 32时**：启用4级流水线，DMA预取+多线程dequant+HMX异步队列+双缓冲输出，
  这些重叠执行让HMX算力优势充分发挥。这是PP阶段快于CPU的原因。
- **m <= 32时（即N=1 decode）**：顺序执行的HMX路径，**没有流水线**，和我们一样慢。

此外，高通在batched模式下对非F16权重会回退到HVX路径
（[matmul-ops.c:4762](../ggml/src/ggml-hexagon/htp/matmul-ops.c#L4762)）：

```c
if (is_batched) {
    if (src0->type == HTP_TYPE_F16) {
        ret = hmx_matmul_f16_f32_batched(...);
    } else {
        return op_matmul_hvx(octx);  // HVX fallback for quantized weights
    }
}
```

### 3.4 两种实现的详细对比

| 维度 | 高通(QCOM) | JZ |
|------|-----------|-----|
| 通信机制 | dspqueue共享内存队列 | FastRPC直接调用 |
| 每图调用次数 | 多次dspqueue_write | 1次FastRPC调用 |
| 数据搬运 | HAP_mmap零拷贝 | Heap->ION memcpy |
| HMX lock/unlock | 每MUL_MAT op一次 | 每MUL_MAT op一次 |
| VTCM管理 | batch级acquire/release | 每op检查 |
| DMA/HMX流水线 | 支持（4级，双缓冲） | 不支持 |
| HMX异步队列 | 支持（hmx_queue） | 不支持 |
| 算子融合 | 支持RMS_NORM+MUL | 不支持 |
| 算子重排 | 支持MUL_MAT重排复用src1 | 不支持 |
| 缓存一致性 | 由dspqueue框架自动处理 | 手动DC CVAC/DC IVAC |

### 3.5 性能量化估算

以1536x12288矩阵、N=1为例：

| 路径 | 每op耗时 | 210个MUL_MAT总耗时 | 估算tok/s |
|------|---------|-------------------|-----------|
| DSP HVX (algotype=33) | ~64 ms | ~13440 ms | ~0.07 |
| DSP HMX (algotype=30) | ~22 ms | ~4600 ms | ~0.22 |
| DSP HMX (algotype=32) | ~30 ms | ~6300 ms | ~0.16 |
| CPU ARM NEON (F32-only) | ~0.24 ms | ~50 ms | ~20 |

CPU比DSP HMX快约90倍，比DSP HVX快约260倍（对于大矩阵的N=1推理）。
**在DSP内部，HMX是N=1的最快算法**。

## 4. x4x2的作用与局限

x4x2 repack将反量化时间减少约27%：

| 矩阵大小 | algotype=30 (x4x2) | algotype=32 (Q4_0) | 加速比 |
|----------|-------------------|-------------------|--------|
| 2048x1536 | ~3000 us | ~4050 us | ~26% |
| 1536x12288 | ~21800 us | ~30000 us | ~27% |
| 1536x2048 | ~3800 us | ~5050 us | ~25% |

但x4x2无法解决HMX对N=1的根本不适应：
- tile浪费（3%利用率）不受x4x2影响
- Pipeline不启用不受x4x2影响
- Per-op开销（HMX lock/unlock、VTCM分配）不受x4x2影响

## 5. algotype=0/33测试结果

测试配置：`mulmat_algotype=0`（HVX多线程）和`mulmat_algotype=33`（HVX+VTCM+多线程），
`enabled_types=Q4_0`，模型gemma-4-E2B-it-Q4_0.gguf。

### 5.1 algotype=0（HVX多线程）

**结论：HVX在DSP上对N=1同样很慢**，与HMX路径一样远慢于CPU。

### 5.2 algotype=33（HVX+VTCM+多线程）- N值自动切换测试

实现了N值自动切换逻辑：当algotype=30/32且N<=32时自动切换到algotype=33。
DSP端log确认切换生效：`N=1 <= 32, switching HMX(algotype=32) -> HVX(algotype=33) for decode`

**但测试结果出乎意料：HVX比HMX慢3倍！**

| 矩阵大小 | HVX (algotype=33) | HMX (algotype=30) | HMX (algotype=32) | CPU ARM NEON |
|----------|-------------------|-------------------|-------------------|-------------|
| 1536x512 | ~2805 us | - | - | ~0.05 ms |
| 1536x2048 | ~10724 us | ~3800 us | ~5050 us | ~0.12 ms |
| 4096x1536 | ~21254 us | - | - | ~0.24 ms |
| 1536x6144 | ~32050 us | - | - | ~0.70 ms |
| 1536x12288 | **~63900 us** | ~21800 us | ~30000 us | ~1.40 ms |
| 12288x1536 | **~63500 us** | ~17700 us | ~24200 us | ~1.40 ms |

**HVX逐行dot product对大矩阵N=1效率极低**。HMX虽然tile利用率只有3%，
但单次HMX矩阵操作仍然比逐行HVX dot product快得多。

### 5.3 结论

1. **DSP（无论HVX还是HMX）对N=1 Q4_0 MUL_MAT都不如ARM NEON CPU**
2. **在DSP内部，HMX是N=1的最快算法**（比HVX快3倍）
3. **N值自动切换到HVX是错误策略**，已回退
4. 正确的优化方向：**AP端N值感知调度** - N=1时Q4_0 MUL_MAT不offload到DSP

## 6. 优化方案

### 6.1 优先级最高：AP端N值感知调度

在AP端`ggmlhexagon_backend_graph_compute_special_ion`中，根据当前是decode还是PP阶段，
动态调整`enabled_types`：
- **Decode阶段**（N=1）：`enabled_types=F32`，Q4_0 MUL_MAT留在CPU
- **PP阶段**（N>32）：`enabled_types=Q4_0`，Q4_0 MUL_MAT offload到DSP

这需要AP端感知当前batch的N值，并可能需要修改ggml调度逻辑。

### 6.2 优先级次高：ION batch内op融合

当前DSP端逐op独立lock/unlock HMX（[entry.c:926](../ggml/src/ggml-hexagon/kernels/entry.c#L926)）。
可修改为：
1. 预扫描batch中的op，识别连续的MUL_MAT组
2. 对每组只lock HMX一次，处理完所有MUL_MAT后unlock
3. 非MUL_MAT op不需要HMX，不持锁

### 6.4 长期优化

| 优化项 | 说明 | 难度 |
|--------|------|------|
| DMA/HMX流水线 | 4级流水线重叠DMA/dequant/HMX/store | 高 |
| HMX异步队列 | hmx_queue异步提交HMX操作 | 高 |
| 算子融合 | RMS_NORM+MUL融合 | 中 |
| 算子重排 | 相同src1的MUL_MAT堆叠复用VTCM | 中 |
| 零拷贝数据路径 | 替代Heap->ION memcpy | 高 |

## 7. 关键代码索引

| 内容 | 文件 | 关键行号 |
|------|------|---------|
| JZ graph_compute (ION batch) | ggml-hexagon.cpp | 7796-8363 |
| JZ DSP batch执行 | kernels/entry.c | 869-1067 |
| JZ MUL_MAT分发 | kernels/mulmat.c | 5246-5276 |
| JZ HMX matmul | kernels/mulmat.c | 4525-4974 |
| JZ VTCM管理 | kernels/entry.c | 569-626 |
| QCOM graph_compute | ggml-hexagon-qcom.cpp | 3200-3248 |
| QCOM HMX matmul | htp/hmx-matmul-ops.c | 1167-1397 |
| QCOM pipeline模式 | htp/hmx-matmul-ops.c | 1198-1260 |
| QCOM batched fallback to HVX | htp/matmul-ops.c | 4762 |
| QCOM synchronize | ggml-hexagon-qcom.cpp | 3251-3258 |
| QCOM VTCM管理 | htp/htp-ctx.h | 74-78 |
| QCOM packet callback | htp/main.c | 801-913 |

## 8. 测试计划

1. ~~测试algotype=0（HVX多线程）在N=1时的推理性能~~ — 已完成，HVX在DSP上也慢
2. 实现N值自动切换逻辑：N<=32用HVX，N>32用HMX
3. 对比N值切换后的decode性能（vs 纯HMX）
4. 验证PP阶段（N>32）HMX路径性能不受影响
5. 评估AP端N值感知调度的可行性
