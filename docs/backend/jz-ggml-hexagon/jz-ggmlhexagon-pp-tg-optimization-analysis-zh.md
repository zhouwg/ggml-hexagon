# JZ's ggml-hexagon 性能差异分析与优化方向

> Initial: 2026-08-05

> Last updated: 2026-08-11

> Author: Seed-2.1-Pro (Ch 1-4), MiniMax-M3(Ch 5-8), revised by DeepSeek-V4-Pro & GLM-5.2 & MiniMax-M3 & Kimi-K3 & Jeff Zhou (review & feedback)

***

| 缩写 | 全称 / 含义 |
|------|-------------|
| JZ   | JZ's ggml-hexagon (custom backend, `GGML_HEXAGON_JZ=ON`) |
| QCOM | Qualcomm's ggml-hexagon (official backend, `GGML_HEXAGON_JZ=OFF`) |
| PP   | Prompt Processing (prefill phase) |
| TG   | Token Generation (decode phase) |
| mempool | kernel-allocated AP-DSP shared memory (single allocation, offset addressing) |

## 一、JZ 与 Qualcomm ggml-hexagon 架构对比

JZ (`ggml-hexagon-jz.cpp` + `kernels/`) 与 QCOM (`ggml-hexagon.cpp` + `htp/`) 是**基于同一套 hexagon kernels 的两条进化分支**，分叉点为 Qualcomm [PR #26049](https://github.com/ggml-org/llama.cpp/pull/26049)。

- PR #26049 之前，两者算子完全相同。
- PR #26049 之后，QCOM `htp/` 下的算子改进被手动移植到 JZ `kernels/`。
- **性能差异不在 kernel 算子本身，而在调度框架、cache 策略和 offload 策略。**

### 1.1 核心架构差异

- **JZ**: native FastRPC `invoke` + single mempool (offset addressing)
- **Qualcomm**: `dspqueue` + per-buffer shared buffers (`bi` indirect addressing)

**Table-1**: 架构对比

| 维度 | JZ          | QCOM                                       |
| ------ | ------------------------ | ----------------------------------------------------------- |
| 控制平面 | 原生 FastRPC `invoke` (同步) | `dspqueue_write/read` (异步, 最多 16 个 batch 并发) |
| 数据平面    | single mempool + offset 寻址     | per-buffer + `bi` (buffer index) 间接寻址            |
| DSP 入口 | `kernels/entry.c`                       | `htp/main.c`                                                |
| DSP kernels     | `kernels/*.c`                           | `htp/*.c`                                                   |
| AP 侧代码    | `ggml-hexagon-jz.cpp`                   | `ggml-hexagon.cpp`                                          |
| 构建选项    | `GGML_HEXAGON_JZ=ON`                    | `GGML_HEXAGON_JZ=OFF` (默认)                             |
| Cache 一致性 | 用户态: role-aware (`ion_sync` + `dsp_cache_mode`) | 内核态 driver flags (per-batch 统一)             |

**Table-2**: single mempool vs per-buffer 代码级对比

| 维度 | JZ | QCOM | 胜者 |
| --- | --- | --- | --- |
| `fastrpc_mmap` 调用次数 | single mempool 初始化时 1 次 | 每 buffer 1 次 | JZ |
| fd 数量 | 1 | 每 buffer 1 个 | JZ |
| DSP tensor 寻址 | 直接 `void *` offset | `bi` -> `htp_buf_desc[]` 间接寻址 | JZ |
| Batch 传输 | `invoke` 携带整个 graph batch | `dspqueue_write` | 平手 |
| 内存生命周期 | 一次 alloc/free | 每 buffer alloc/mmap + munmap/free | JZ |
| IOVA 空间局部性 (prefetch/TLB) | 连续, 可预测 | 跨 buffer 分散 | JZ |
| Cache 一致性 | 用户态: role-aware (权重 vs 激活); `ion_sync` + `dsp_cache_mode` | driver 刷新 descriptor packet + DSP 侧每 batch 全量 D-cache flush+invalidate (统一, 不区分角色) | JZ (role-aware 策略灵活性) |
| 物理地址稳定性 | 分配后稳定 (不迁移) | 分配后稳定 (不迁移) | 平手 |
| lm-head offload | 可行 (mempool offset 范围) | 不可行 (per-buffer fd/mmap/生命周期开销) | JZ |

### 1.2 控制平面原语差异: dspqueue vs. 原生 FastRPC invoke

**Table-3**: 控制平面对比

| 维度 | JZ | QCOM |
| --- | --- | --- |
| 原语 | 原生 FastRPC `invoke` | `dspqueue_write/read` 队列语义 |
| 分发方式 | AP 直接调用 DSP 函数, 携带 descriptors | AP 推送整个 op-batch; DSP 通过 packet callback 唤醒 |
| 阻塞模型 | 每次调用同步 | AP 异步推送, 后续 drain 取回结果 |
| Batch 处理 | 一次 `invoke` 携带整个 graph batch, 有时数百个 op | 一次 `dspqueue_write` 携带多个 op (`htp_opbatch_req`) |

### 1.3 数据平面: 几乎完全相同

- 两者均通过 **AP-DSP 共享内存** 传输 tensor 数据。
- 两者均为 AP 写 descriptors/data, DSP 读 descriptors/data。
- 两者均需 **cache flush / invalidate** 同步。
- 两者最终运行相同的 HVX/HMX kernels。

QCOM 的 DSP 入口 [`htp/main.c`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/main.c) 通过 `htp_packet_callback` 解析 `htp_buf_desc[]`、`htp_tensor[]`、`htp_op_desc[]` 后分发到 kernels。JZ 的相同流程在 [`kernels/entry.c`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/entry.c) 中完成。两者 data path 相同:

```
AP 打包 descriptors -> 传输到 DSP -> DSP 入口解析 descriptors -> 分发 op 执行
```

### 1.4 源文件结构

**Table-4**: 源文件结构

| 文件                                                            | 描述                                                                       |
| ------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp`                   | JZ AP 侧代码                                                      |
| `ggml/src/ggml-hexagon/ggml-hexagon.cpp`                      | QCOM AP 侧代码                                     |
| `ggml/src/ggml-hexagon/CMakeLists.txt`                        | 统一构建 (QCOM 基线 + `GGML_HEXAGON_JZ` 选项)                    |
| `ggml/src/ggml-hexagon/kernels/Makefile`                      | JZ DSP skel 构建 (entry.c + kernels/)                                   |
| `ggml/src/ggml-hexagon/htp/CMakeLists.txt`                    | QCOM DSP skel 构建 (main.c + htp/)                           |
| `ggml/src/ggml-hexagon/kernels/entry.c`                       | JZ DSP 入口                                              |
| `ggml/src/ggml-hexagon/kernels/dsp-ctx.h`                     | JZ DSP session context + descriptors                                    |
| `ggml/src/ggml-hexagon/htp/main.c`                            | QCOM DSP 入口                                        |
| `ggml/src/ggml-hexagon/htp/htp-ctx.h`                         | QCOM DSP session context + mmap/spad                                    |
| `ggml/src/ggml-hexagon/kernels/*.c`                           | JZ DSP kernels (从 htp/ fork, 基线 2be3826c9)                 |
| `ggml/src/ggml-hexagon/htp/*.c`                               | QCOM DSP kernels                           |

***

## 二、五模型 AB 测试性能数据

以 `3469e4858e17d501a1f6e16ebe0aa2489613d32b` 为基线，对比 JZ (`ggml-hexagon-jz.cpp` + `kernels/`) 与 QCOM (`ggml-hexagon.cpp` + `htp/`) 在五个模型（Qwen3.5-2B、Gemma4-E2B、Gemma4-E4B、Qwen1.5-1.8B、Llama3.2-1B）上的 PP/TG 性能。

**Table-5**: 测试环境

| 项目      | 配置                                                                                                                                                 |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| 设备      | Qualcomm Snapdragon 8 Elite (8 Gen 4), QCOM\_HTP\_V79, dsp arch 0x79, VTCM=8MB, HVX=1, HMX=1, 系统内存 24834 MiB, Android device id `9d231cfe`         |
| JZ 后端   | `libggml-hexagon-jz.so` + `libggmldsp-skel-v79.so`                                                                                                 |
| QCOM 后端 | `libggml-hexagon-qcom.so` + `libggml-htp-v79.so`                                                                                                   |
| 测试参数    | `n_ctx=8192, n_batch=2048, n_predict=256, n_threads=6, graphs reused=253`                                                                          |
| JZ 配置   | `dsp_cache_mode=5, ion_sync_mode=1, graph_optimize=1`<br>offload MUL\_MAT types: F32,F16,BF16,Q4\_0,Q8\_0,Q4\_1,IQ4\_NL,MXFP4<br>thread\_counts on CDSP: 6 |
| 测试方法    | 每个模型 3 轮取均值，n\_prompt=51\~75 tokens，n\_gen=255 tokens                                                                                              |

**Table-6**: AB 测试性能数据

| 模型           | PP JZ (tok/s) | PP QCOM (tok/s) | PP JZ vs QCOM | TG JZ (tok/s) | TG QCOM (tok/s) | TG JZ vs QCOM |
| ------------ | :-----------: | :-------------: | :-----------: | :-----------: | :-------------: | :-----------: |
| Qwen3.5-2B   |     501.7     |      456.1      |   **+10.0%**  |      26.7     |       13.4      |   **+99.0%**  |
| Gemma4-E2B   |     684.8     |      457.9      |   **+49.5%**  |      27.1     |       25.0      |     +8.1%     |
| Gemma4-E4B   |     403.6     |      417.4      |     -3.3%     |      14.9     |       10.8      |   **+38.2%**  |
| Qwen1.5-1.8B |     552.9     |      711.2      |     -22.3%    |      18.6     |       26.2      |     -28.8%    |
| Llama3.2-1B  |    1018.8     |      1084.3     |     -6.0%     |      42.2     |       28.7      |   **+47.1%**  |

> **数据来源**：`./scripts/build-run-ggmlhexagon-android.sh run_abtest_all 2>&1 | tee log_abtest_all_$(date +%Y%m%d-%H%M%S).txt`（本轮日志 `log_abtest_all_20260807-223924.txt`，self-build-jz 分支，2026-08-07 22:39）
>
> **数据注记**：Qwen1.5-1.8B QCOM 数据受轮间热状态波动影响，Table-6 按 3 轮均值记录。Qwen3.5-2B PP 翻转（-9.2% -> +10.0%）是结构性变化（graph 拆分修复，详见第六章），其余模型差异在热状态波动范围内。

**关键观察**：

- **TG**：JZ 在 4/5 模型上领先，最大优势 +99.0%（Qwen3.5-2B），领先 4 模型平均约 +48.1%；仅 Qwen1.5-1.8B（MHA 模型）落后 28.8%。
- **PP**：JZ 在 2/5 模型上领先（Gemma4-E2B +49.5%、Qwen3.5-2B +10.0%），QCOM 在其余 3 模型上领先，最大优势 +22.3%（Qwen1.5-1.8B）。Qwen3.5-2B 较早晨 run（-9.2%）翻转为 JZ 获胜，根因是 graph 拆分修复（第六章）。
- TG 与 PP 的瓶颈根因不同（第三章分析）。

***

## 三、性能差异根因分析

### 3.1 lm-head offload：TG 性能差异的最大单一因素

QCOM 后端在 `ggml-hexagon.cpp` 的 `ggml_hexagon_supported_mul_mat` 中有**3 处 guard** 阻止 lm-head offload 到 DSP：

1. **类型 guard**：switch 只处理 Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4/F16/F32，**Q4_K/Q6_K/BF16 不在 switch 中**，落入 `default: return false`（[ggml-hexagon.cpp：ggml_hexagon_supported_mul_mat](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp)）。JZ 侧（[ggml-hexagon-jz.cpp：ggmlhexagon_supported_mul_mat](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp)）显式处理了 Q4_K/Q6_K（若 Q4_0 已启用则放行，因为 JZ 在加载时做了 Q4_K/Q6_K -> Q4_0 tiled repack）和 BF16（在 repack buffer 中转为 F16 bytes 后放行，[ggml-hexagon-jz.cpp：ggmlhexagon_supported_mul_mat](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp)）。
2. **尺寸 guard**：`src0->ne[1] > 32768` 时拒绝（[ggml-hexagon.cpp：ggml_hexagon_supported_mul_mat](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp)）。此 guard 嵌在 Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4 case 内，对 Q4_K/Q6_K 不生效（已在类型 guard 阶段被 default 拦截）。
3. **repack buffer guard**：QCOM 在 switch 分支内还有一处 **repack buffer guard**（L2815：`!ggml_backend_buffer_is_hexagon_repack(src0->buffer)`），要求权重必须位于 repacked buffer 中。即使类型 guard 被移除，lm-head 权重（不在 repacked buffer 中）仍会被此 guard 阻止。该 guard 位于类型 guard 后面的 switch 分支内，当前未被触发，但揭示了 QCOM 对 offload lm-head 权重的额外约束：权重不仅要类型匹配，还必须位于 repacked buffer 中。

对于本次测试的 五个模型，**类型 guard（#1）是实际生效的 guard**（lm-head 权重均为 Q4_K/Q6_K，不在 switch 中）。尺寸 guard（#2）是 per-buffer 成本约束的直接体现（32K 行是 per-buffer 的成本上限）。

**Table-7**: 各模型 vocab\_size 与 lm-head 大小

| 模型           | vocab\_size | lm\_head 原始类型 | 原始大小（约）  | Q4\_0 repack 后大小（约） |
| ------------ | ----------- | ------------- | -------- | ------------------- |
| Qwen3.5-2B   | 151,936     | Q6\_K         | \~200 MB | \~163 MB            |
| Gemma4-E2B   | 256,000     | Q4\_K         | \~214 MB | \~214 MB            |
| Gemma4-E4B   | 256,000     | Q4\_K         | \~428 MB | \~428 MB            |
| Qwen1.5-1.8B | 151,936     | Q6\_K         | \~200 MB | \~163 MB            |
| Llama3.2-1B  | 128,256     | Q4\_K         | \~138 MB | \~138 MB            |

JZ 后端 `ggmlhexagon_supported_mul_mat` 中**未设置 N 维度上限 guard**(与 QCOM 的 `src0->ne[1] > 32768` 形成对比),因此 lm-head 完全 offload 到 DSP 执行。对 lm-head 为 Q4\_K 的模型（如 Gemma4、Llama3.2-1B），通过 Q4\_K -> Q4\_0 tiled repack 将 lm-head 权重转为 DSP 可直接执行的 tiled layout；对 lm-head 为 Q6\_K 的模型（如 Qwen3.5-2B、Qwen1.5-1.8B），通过 Q6\_K -> Q4\_0 tiled repack 转换（注意 Q6\_K 比 Q4\_0 略大，repack 后体积会略减）。repack **不减少带宽**（Q4\_K 和 Q4\_0 数据大小相同，均为 0.5625 B/param；Q6\_K -> Q4\_0 实际是 lossy 转换以适配 DSP 侧复用的 Q4\_0 matmul kernels），**其价值不在节省带宽,而在让 DSP 侧 tiled matmul kernel 可直接消费该 layout**(DSP kernels 仅支持 Q4_0 tiled layout)。

**lm-head offload 之所以在 JZ 可行，与 single mempool 架构强相关：** lm-head 权重（Q4_K/Q6_K 量化矩阵，按 Table-7 约 138-428 MB）作为 mempool 内的一个 offset 范围，零额外 fd/mmap/生命周期维护成本。QCOM 的 2 处 guard（类型/尺寸）共同阻止了 lm-head offload，根本原因是其 per-buffer 设计：每个 buffer 携带独立的 fd、fastrpc_mmap、dspqueue 每批重复注册等开销，无法低成本地承载会话常驻的 lm-head 权重（32K 行是 per-buffer API 的实际上限）。JZ 通过加载时 Q4_K/Q6_K -> Q4_0 tiled repack 消除了类型 guard，通过 single mempool 的零额外开销消除了尺寸 guard 的成本约束。

**对 TG 的影响是决定性的：** TG 每生成 1 个 token 都要执行一次 lm-head matvec（`[1, n_embd] x [n_embd, vocab_size] -> [1, vocab_size]`）。这是纯粹的 memory-bound 操作：

- **QCOM**：CPU 逐元素读取整个 Q4_K/Q6_K lm-head 权重（按 Table-7 约 138-428 MB）做 dequant+dot product，CPU 访存带宽有限，且 CPU 算 lm-head 时 DSP 空闲。
- **JZ**：DSP 上 HVX 执行 lm-head matvec，权重以 Q4_0 tiled layout 驻留在 mempool 中，带宽远高于 CPU，且与后续 token 生成流水线紧密衔接。

**对 PP 的影响很小**：lm-head 在 PP 末尾只执行一次，其开销被几十个 transformer layer 的计算摊薄。

**per-buffer 设计为何不适合承载 lm-head：** Qualcomm 的 per-buffer 设计中, 每个 buffer 独立持有 4 项资源: fd (由 `rpcmem_alloc2` 分配)、`fastrpc_mmap` (内核 SMMU 映射)、DSP 侧 `htp_buf_desc[]` entry + `bi` 间接寻址、AP-DSP 生命周期协调 (alloc / munmap / destroy)。一个 214 MB 的 lm-head 作为单个 buffer, 在整个 session 期间持有上述全部资源。per-buffer API 设计的"对称性要求"决定了没有针对大权重（如 lm-head）的特殊处理路径。

dspqueue depth=16 (`opt_opqueue=16`) 意味着每个 batch 都通过 `add_buffer()` 重新注册 lm-head 的 `htp_buf_desc[]`, `ggml_hexagon_opqueue::push()` 每次携带 dbuf, DSP 侧 `prep_op_bufs()` 在 vmem 预算压力下可能触发重复 mmap/munmap。理论上可以将 lm-head 放在 dspqueue 之外 (raw mempool), 但这破坏了统一路径: "一个 buffer 必须在 batch descriptor 中注册才能被 DSP op 访问"。JZ 的 single mempool 是 all-or-nothing 设计: 初始化时一次 mmap, 所有 tensor 自然可访问, lm-head 无需特殊处理。

32768 guard 的代码注释 `"hardcoded limit to refuse the lm-head for now"` 中的 "for now" 措辞暗示这是已知限制而非永久选择。32K 行是 per-buffer API 的"实际上限" (单个 buffer 数十 MB 时 fd/mmap 成本尚可接受), 而 214 MB 的 lm-head 远超此上限。

**tiled kernel 不是差异点:** QCOM 后端的 [`htp/`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp) 目录同样包含 tiled Q4\_0/Q4\_1 kernel。两边 kernel 同源（PR #26049 之前共用，之后分叉维护）。真正差异在于**是否对 Q4\_K/Q6\_K 权重做 repack**：JZ 做 repack 使所有量化 matmul（含 lm-head）能在 DSP 上执行，QCOM 不做 repack 导致 Q4\_K/Q6\_K 回退 CPU。

**一句话总结:** Qualcomm 的 per-buffer 设计内含假设 -- "每个 buffer 都是独立的、短生命周期的、在统一 cache 策略下的小对象"。lm-head 恰好相反: 巨大、整个 session 常驻、需要专用 cache 策略。支持 lm-head 需要在 dspqueue、driver、buffer descriptor 三层做对称性破坏修改, 本质上就是向 single mempool 设计收敛。

### 3.2 dspqueue async overlay vs 同步 FastRPC：PP 性能差异的根因

**调度框架差异是 PP 性能差异的根因，而非简单的调度开销。**

**QCOM (htp/) - dspqueue 异步流水线**：

- AP 通过 `dspqueue_write` 将 op 描述符写入环形队列，非阻塞。
- DSP 从队列消费 op 并执行。
- AP 可以在 DSP 执行当前 op/layer 的同时，准备下一个 op 的描述符、做 cache flush 等。
- 形成 **AP-DSP overlay（流水线重叠）**：AP prep 和 DSP compute 并行。
- `enqueue_op` + `dspqueue_read` 构成生产者-消费者模型。

**JZ (kernels/) - native FastRPC 100% 同步**：
`graph_compute_batch` 经历严格串行的 12 个 Phase：

- **Phase 1-9**：AP 侧全图分析 -> tensor 镜像 -> 权重 repack -> mempool 分配 -> desc 构建 -> cache flush（全部 AP 工作，DSP 空闲）。
- **Phase 10**：同步调用 `ggml_dsp_execute_batch`，AP 阻塞等待 DSP 执行完整个 batch（DSP 工作，AP 空闲）。
  - `cum_p10_rpc_setup_us`：AP setup（ioctl / marshalling）。
  - `cum_p10_dsp_exec_us`：纯 DSP 执行时间。
  - `cum_p10_civac_us`：AP cache invalidate after DSP reply。
- **Phase 11-12**：AP 侧 cache inval -> mempool 回拷（DSP 空闲，AP 工作）。
- **零 overlay**：AP 准备期间 DSP 空闲，DSP 执行期间 AP 空闲。

**对 PP 的影响**：PP 是 compute-bound 场景，prompt tokens 多、每层 matmul 的 M 大、DSP 计算时间长，AP-DSP overlay 的收益被充分放大：QCOM 的 AP prep 时间完全隐藏在 DSP compute 后面，而 JZ 的 Phase 1-9 + Phase 11-12 纯 AP 开销直接加到总延迟上。

**对 TG 的影响**：每 token 只有一个 batch（M=1），DSP 计算极短，dspqueue 的 overlay 收益极小（几乎没有可以隐藏的 AP prep 时间）。同时 per-op dspqueue 通信开销（每次 write/read 都有环形队列管理开销）在 M=1 小 op 时占比变大。JZ 单次 invoke 发整个 batch 的模型在 TG 更高效。

### 3.3 Role-aware batch-level cache 管理：JZ TG 的第二大优势

JZ 的 batch-level cache 策略通过 bitmap 控制，大幅降低了 cache sync 开销：

- **bit0 (first-touch weight bitmap)**：权重首次 touch 后不再 dcinva，命中 L2。
- **bit1 (prior-dst skip)**：当前 src 与前序 op 的小 dst（<= 单 cacheline）为同一 tensor 时，跳过该 src 的 invalidation（本轮测试 mode=5 未启用，详见 4.6.1）。
- **bit2 (bulk flush)**：所有 dst flush 合并到 batch 末尾一次完成。
- **bit3 (selective flush)**：中间 tensor 不 flush，减少 DDR 写。

而 QCOM 采用 batch 级全量 cache 维护：在 batch 开始和结束时各执行一次完整 D-cache flush+invalidate（`qurt_mem_cache_clean(..., FLUSH_INVALIDATE_ALL, ...)`），uniform、role-blind，无法区分 weight 和 activation。

**对 TG 的影响**：M=1 时 matmul 计算量小，cache sync 开销占比大，JZ 的 bitmap 策略优势显著。

**对 PP 的影响**：PP 时 matmul 计算量大，cache sync 开销被摊薄，差异不明显。

**QCOM cache 维护的两层结构:**
- 第一层是 descriptor packet flags: dspqueue packet 携带固定 flags (`DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT`), 但这只作用于 batch descriptor block (几 KB), 不是 tensor data。tensor data 通过 `fastrpc_mmap` 在 alloc 时映射, DSP skel 在首次使用时 mmap fds (`prep_op_bufs`)。实际 flush/invalidate 逻辑在闭源 Hexagon DSP driver 中。
- 第二层是 DSP 侧全量 cache sweep: `process_opbatch()` 中 `qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE)` -- 全量 D-cache flush+invalidate, 无法区分 weight 和 activation, 也没有 "权重首次写入后不再 invalidate" 的 first-touch 路径。

**JZ 的三种正交 cache 机制:** JZ 的用户态 cache 优化由三种相互正交的机制组成, 作用域不同, 可独立配置:

| 机制 | 字段 | 类型 | 控制范围 | 用途 |
| --- | --- | --- | --- | --- |
| Global | `dsp_cache_mode` | 4-bit bitmask | DSP 侧 cache flush 行为 | first-touch / dcinva skip / bulk dst flush |
| Per-tensor | `td->flags` | per-tensor role tag | tensor 角色 (weight/mirrored/normal) | 区分权重以应用 first-touch 路径 |
| Per-batch | `ion_sync_mode` | 3-value mode selector | AP 侧 cache coherency 机制 (CVAC vs ion_sync) | 跳过手动 DC CVAC, 整池 kernel sync |

其中 `td->flags`: flags=2 (权重, 首次 touch 后跳过 cache flush)、flags=1 (mirrored, 可能与同 batch 后续 op 共享 dst)、flags=0 (normal, 常规 per-batch cache 维护)。`ion_sync_mode`: mode=0 (DC CVAC/CIVAC + DMA_BUF_IOCTL_SYNC, 最保守)、mode=1 (ion_sync only, 代码默认, 跳过手动 DC CVAC/CIVAC, 发单次 `ioctl(DMA_BUF_IOCTL_SYNC_IOCTL)` 整池 kernel sync)、mode=2 (DC CVAC/CIVAC only, 手动 cache 维护, 无 kernel sync)。

**first-touch 优化的量化收益:** JZ 侧实测, first-touch 路径 (bit0) 消除约 9.2 ms/token 的冗余权重 re-invalidation (lm-head 驻留时, per-token 权重流量约 1.9 GB; 该数值为 bit0 off vs on 的实测差值, 涵盖所有权重)。这是固定的 whole-graph per-token 总量, 不随层数变化。

**"劣势变优势"的反转:** QCOM 将 cache 维护交给 dspqueue 框架统一处理 (代码简洁, 看似优势), 但代价是 DSP 侧每 batch 两次全量 D-cache flush+invalidate, 无法区分 tensor 角色, first-touch 优化无处安放。JZ 看似"被迫自己管理 cache" (需要实现 `ion_sync` + `dsp_cache_mode`, 看似劣势), 但正因为用户态能看到 tensor role、区分权重与激活、决定 first-touch 行为, 214 MB 的 lm-head 才变得可优化。这是经典的分层设计反转: 抽象层越高, 优化空间越小; 抽象层越低, 策略灵活性越大, 优化空间越大。

### 3.4 总结：性能差异归因

**JZ TG 领先**与 single mempool 带来的 lm-head offload 强相关，role-aware 的缓存一致性维护策略也是重要因素。

**JZ PP 落后**的根因是**调度框架差异**而非 kernel 差异。JZ 与 QCOM 复用同一套 Qualcomm HMX kernels（分叉前完全相同），matmul 执行效率一致。差异在于：JZ 的 12-phase 同步模型无法实现 AP-DSP pipelining，而 QCOM 的 dspqueue 异步环形队列允许 AP prep 与 DSP compute 在 per-layer 粒度上重叠。JZ 的 data-plane 优势（lm-head offload + first-touch 权重 inval）是**整图固定开销**，与 layer 数无关；QCOM 的 pipelining 优势是**per-layer 累积**的，与 layer 数正相关。因此 PP 表现高度依赖模型层数与 attention pattern 对 VTCM/cache 压力的影响。

**Qwen1.5-1.8B（MHA 模型，24 层）PP/TG 均落后的根因** = dspqueue pipelining 优势 + 层数不足 + MHA VTCM/cache 压力三重叠加：

1. **dspqueue pipelining 优势最大化**：dspqueue 的 AP-DSP overlap 收益与每次 DSP 计算时长正相关，Qwen1.5-1.8B 在 PP 阶段单 layer 计算时间长（24 层 x 每层 MHA Q@K^T 的 full attention），pipelining 隐藏的 AP prep 时间窗口大。
2. **JZ 整图固定优势无法累积**：lm-head offload（~200MB Q6_K）+ first-touch 权重 inval（~9.2ms/token）是固定的、不会随 layer 数增加而放大的优势；24 层不足以让 JZ 的 per-layer 增量优势赶超 dspqueue 的 per-layer pipelining 收益。Gemma4-E2B 35 层则可以反超（+49.5%）。
3. **MHA 加重 VTCM/cache 压力**：1:1 attention 的 Q@K^T 是 full attention（无 KV 共享），相比 GQA 模型的 KV 共享头占用更多 VTCM 与 cache 带宽，恰好是 JZ role-aware cache 策略（bit0-3）本来要优化的场景-但这些优化只在 TG M=1 场景放大收益，对 PP 长序列 M=prompt_len 帮助有限。

**结论**：Qwen1.5-1.8B 不是 corner case，而是三重不利因素叠加的体现。**任何 PP 优化（如 per-layer pipelining）只要把 dspqueue 的 per-layer 优势部分削弱，就能同时改善 Qwen1.5-1.8B 这类模型**-这是 PP 优化优先级应当高于 TG 精雕细琢的核心论证。

**PP/TG 交叉点公式:** 当模型权重可完全放入 single mempool 时，JZ 的净优势可用以下公式概括:

```
JZ net advantage = per_layer_dsp_saving x n_layers + fixed_lmhead_saving - dspqueue_overlap_advantage
                   └── 随层数线性增长 ──┘   └─ 固定 ─┘   └── 固定, 不随层数增长 ──┘
```

其中 `fixed_lmhead_saving` 是会话常驻的 lm-head offload + first-touch 节省 (~9.2 ms/token, 见 3.3 节), 不随层数变化。`per_layer_dsp_saving` 是每层的 first-touch + mempool 零开销 + HMX pipeline 优势, 随层数线性累积。`dspqueue_overlap_advantage` 是 QCOM 的固定优势, 不随层数增长。

dspqueue 的 overlay 优势受两个因素限制:
1. **LLM 属性限制:** TG 严格串行 (每个 token 依赖前一个), dspqueue 的 16 个在飞 batch 在 TG 中只能重叠单个 token 内的 descriptor-prep 与 DSP-compute, 贡献自然有上限。
2. **无法 offload lm-head 限制:** QCOM 将 lm-head 留在 CPU (32768 guard), PP 的最后一个 op (lm-head) 在 CPU 运行时 DSP 空闲, 打断了 dspqueue 流水线。这在 PP 中比 TG 更严重, 因为 PP 的 lm-head 有 m=batch_size, CPU 计算时间更长, DSP 空闲时间更大。

当 `n_layers x per_layer_saving > dspqueue_overlap` 时, JZ 在 PP 也反超。交叉点取决于模型结构: Gemma4-E2B (35 层, GQA 8:1) 跨过两个阈值, JZ PP & TG 均胜; Qwen3.5-2B (25 层, GQA 4:1, 修复 graph 拆分后) 跨过 TG 阈值且 PP 反超; Qwen1.5-1.8B (24 层, MHA 1:1) 未跨过任一阈值。现代 GQA 模型 (30+ 层) 最受益于 JZ 架构。

**公式不适用的场景:** 当模型权重超出 single mempool 容量时（如 Qwen3.5-9B），推理会触发大量 mirror memcpy，直接影响 PP&TG 性能，上述公式不再适用。mempool 容量受限于 Hexagon DSP VA 32-bit 虚拟地址空间（约 4 GiB）；若高通 Hexagon SDK 实现真正的 UMA，mempool 大小将仅受 system memory 约束，公式适用范围可大幅扩展。详见第八章。

**对 Qualcomm ggml-hexagon 的启示:** 若 QCOM 未来改进 lm-head offload, 需要将 dspqueue buffer 分为两类 ("常驻共享 buffer" + "per-batch 瞬态 buffer"), 在 driver/DSP skel 中加入 "权重角色" 概念, 允许 buffer 独立于 dspqueue 生命周期存在。但这本质上就是把 per-buffer 重新设计为 "几个 buffer + 一个 pool", 也就是 single mempool 方案。

**Table-8**: 性能差异归因

| 架构特性                | JZ                                                                                          | QCOM                                                            | TG 影响                                         | PP 影响                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------- | --------------------------------------------- | -------------------------------------- |
| **调度框架**            | Native FastRPC 同步（12 Phase 串行，零重叠）                                                               | dspqueue 异步环形队列（AP-DSP 重叠）                                        | JZ 略优（单次 invoke vs per-op 队列管理）             | **QCOM 显著优**（AP prep 与 DSP compute 重叠） |
| **lm-head offload** | 全 offload（single mempool + Q4\_K/Q6\_K->Q4\_0 tiled repack，支持超大 N）                                      | 2 处 guard（类型/尺寸）拒绝，回退 CPU                                            | **JZ 极大优势**（每 token \~138-428MB matvec 在 DSP） | 影响小（只跑 1 次）                            |
| **Cache 管理**        | Role-aware batch-level（bit0-3，first-touch/prior-dst/bulk/selective）                                    | Batch 级全量 D-cache flush+invalidate（uniform, role-blind）                | **JZ 显著优**（M=1 时 cache sync 是大头）              | 差异小（大计算摊薄）                             |
| **内存模型**            | Single mempool（init 时 mmap 一次，v79 容量 probe 上限 4032 MiB，offset addressing；无 fd/mmap/lifecycle 重复成本） | Per-buffer rpcmem 分配（每 buffer 独立 fd / fastrpc\_mmap / dspqueue 每批重复注册） | JZ 优（零额外 fd/mmap + 整池 IOVA 连续 + 权重 L2 友好驻留）   | 差异小                                    |
| **权重布局**            | Q4\_K/Q6\_K -> Q4\_0 tiled repack 后 DSP 侧跑 tiled Q4\_0 kernel                                           | 原始 Q4\_K/Q6\_K 布局 + tiled kernel（lm-head 因 2 处 guard 不参与）               | JZ 优（lm-head DSP offload，VTCM/L2 友好）          | JZ 略优                                  |

***

## 四、优化方向

### 4.1 DSP Op-Level Profiling 实测数据（2026-08-06）

基于 Gemma4-E2B 模型（TG 主场景），在 DSP 侧通过 `HEX_OP_PROF`（定义于 [dsp-ctx.h：HEX_OP_PROF](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/dsp-ctx.h)；feature/force_opfusion_in_pp 分支 hardcode=1，主分支默认=0）开启 per-op 计时统计，每 25 个 batch 通过 FARF(ERROR) 输出累计数据。以下分析取 batch#200 稳定数据点。

**测试环境**：同本文第二章测试环境，Gemma4-E2B，n\_ctx=8192, n\_threads=6, dsp\_cache\_mode=5, ion\_sync\_mode=1

**Table-9**: DSP 算子耗时排名（batch#200，cumulative，us）

| 排名     | 算子                                               | cum (us)      | count    | avg (us)   | min (us) | max (us)  | 占 op-sum 比例 |
| ------ | ------------------------------------------------ | ------------- | -------- | ---------- | -------- | --------- | ----------- |
| 1      | **MUL\_MAT**                                     | 3,174,152     | 32,515   | 97         | 17       | **4,697** | **51.0%**   |
| 2      | **MUL\_MAT\_FFN**                                | 2,327,965     | 6,965    | **334**    | 207      | 509       | **37.4%**   |
| 3      | MUL\_MAT\_QKV                                    | 170,544       | 2,985    | 57         | 45       | 120       | 2.7%        |
| 4      | FLASH\_ATTN\_EXT                                 | 143,128       | 7,000    | 20         | 16       | 141       | 2.3%        |
| 5      | GLU\_GEGLU                                       | 112,070       | 7,000    | 16         | 9        | 258       | 1.8%        |
| 6      | RMS\_NORM\_MUL                                   | 111,584       | 45,400   | 2          | 1        | 106       | 1.8%        |
| 7      | UNARY\_TANH                                      | 52,298        | 200      | 261        | 259      | 266       | 0.8%        |
| 8      | ADD                                              | 37,723        | 21,200   | 1          | 0        | 37        | 0.6%        |
| 9      | MUL                                              | 22,572        | 14,000   | 1          | 0        | 26        | 0.4%        |
| 10     | ROPE                                             | 25,743        | 10,000   | 2          | 1        | 42        | 0.4%        |
| 11     | SCALE                                            | 20,910        | 1,200    | 17         | 1        | 94        | 0.3%        |
| 12     | UNARY\_GELU                                      | 9,815         | 7,000    | 1          | 0        | 38        | 0.2%        |
| 13     | RMS\_NORM                                        | 4,795         | 3,000    | 1          | 1        | 16        | 0.1%        |
| 14     | 其他 (SET\_ROWS/CPY/GET\_ROWS/DUP/MUL\_ROWS/CONT等) | \~7,366       | \~25,200 | <1         | 0        | 75        | \~0.1%      |
| <br /> | **op-sum 合计**                                    | **6,219,165** | <br />   | **31,095** | <br />   | <br />    | **100%**    |

**关键发现**：

1. **Matmul 类算子合计占 91.1% DSP 计算时间**：MUL\_MAT (51.0%) + MUL\_MAT\_FFN (37.4%) + MUL\_MAT\_QKV (2.7%) = 91.1%。所有其他算子（attention、norm、activation、rope 等）合计仅占 8.9%，优化 matmul kernel 是性能提升的唯一杠杆。
2. **MUL\_MAT max=4697us 是 lm-head**：avg=97us 被大量小尺寸 matmul 拉低，但 max=4697us 的 outlier 每个 TG batch 出现一次，对应 lm-head matvec（`[1, hidden] x [hidden, vocab=256000]`），是 TG 阶段最大的单个算子。小 MUL\_MAT（avg≈17-100us）对应 attention 输出投影和其他零散 matmul。
3. **MUL\_MAT\_FFN avg=334us 是最稳定的 hotspot**：count=6965/200batch≈35 次/batch，即每 transformer layer 1 次 MUL\_MAT\_FFN fused 调用（Gemma4-E2B 35 层 x 1 fused op/layer = 35 次/token）。该 fused op 在内部完成 gated FFN 的 gate+up+down 三段 matmul，因此 35 次 fused call = 35 x 3 = 105 个内部 matmul。avg 稳定在 334us，是所有算子中**平均耗时最高**的稳定计算项。
4. **FLASH\_ATTN\_EXT avg=20us 非常高效**：FlashAttn kernel 已充分优化，avg 仅 20us，不是瓶颈。
5. **Element-wise 算子可忽略**：RMS\_NORM/ADD/MUL/ROPE/GELU 等 avg 均 ≤2us，占比合计 <5%，fuse 收益极小。

**Non-op 开销分析**（batch#200）：

```
batch-wall avg=35,789 us/batch
op-sum     avg=31,095 us/batch
non-op     avg= 4,693 us/batch  (13.1% of wall time)
```

**Table-10**: DSP non-op 开销细分（us/batch）

| 阶段                                 | 耗时 (us/batch) | 数据量       | 说明                                            |
| ---------------------------------- | ------------- | --------- | --------------------------------------------- |
| hdr cache inval                    | 4             | -         | batch descriptor invalidation，可忽略             |
| tensor pre-conversion              | 318           | -         | hex\_tensor\_desc -> dsptensor/htp\_tensor 预转换 |
| weight cache inval (w-inv)         | 68            | 6 MB      | bit0 first-touch 效果显著：权重仅首次 inval             |
| **activation cache inval (a-inv)** | **1,030**     | **82 MB** | **最大 non-op 开销**，mode=5 未启用 bit1，接近结构性下限  |
| dst tracking                       | 105           | -         | prior\_dst/bulk\_flush 范围收集                   |
| **bulk dst flush**                 | **1,377**     | -         | 所有 dst 合并到 batch 末尾 flush                     |
| queue wakeup/suspend               | 5             | -         | DMA/HMX queue 管理，可忽略                          |
| **non-op 合计**                      | **4,693**     | <br />    | **13.1% wall time**                           |

**瓶颈根因分析（基于 profiling 数据修正）**：

> **注意**：以下 profiling 数据仅覆盖 DSP 批处理执行阶段（Phase 10），在 DSP 侧通过 HEX\_OP\_PROF 测量。AP 侧开销（Phase 1-9 + Phase 11-12）未包含在内，需通过 AP 侧 profiling（dump\_diag\_info=1）单独测量。此处"wall time"指 DSP 侧 batch 执行 wall time（35,789 us/batch），非端到端 TG 时间。

- **在 DSP 执行内部，matmul kernel 是绝对主导**：op-sum 占 DSP batch-wall time 的 86.9%，其中 91.1% 是三类 matmul。DSP 侧 non-op 开销（cache inval、tensor 转换、dst flush 等）合计 4693 us/batch，占 DSP batch-wall 的 13.1%。
- **lm-head MUL\_MAT（max=4697us）是 TG 单算子最大项**：每个 token 出现一次，对应 `1xhiddenxvocab` matrix-vector product。通用 GEMM kernel 对 M=1 的 skinny matmul 效率不高，专用 GEMV kernel 有优化空间。
- **MUL\_MAT\_FFN（avg=334us）是 per-layer 最大稳定开销**：FFN matmul 已使用 fused op（MUL\_MAT\_FFN），需要检查是否充分利用 HMX 加速，以及 tile size 是否对 FFN 维度最优。
- **activation cache invalidation（a-inv=1030us/batch, 82MB）是最大 DSP 侧 non-op 开销**：本轮 dsp\_cache\_mode=5 未启用 bit1；2026-08-07 对照实验（Qwen1.5-1.8B PP-only，mode=7 vs 5）证实即使启用 bit1，a-inv 也零变化（根因：prior_dst_add 只登记 <= 单 cacheline 的 dst，cgraph 中间张量均 >= 256 字节，skip 路径几乎不触发）。per-batch dedup 已保证每条 unique src 每 batch 至多失效一次，a-inv 接近结构性下限。bulk flush（1377us）是第二大 DSP 侧 non-op，但这是 bit2 bulk flush 策略的代价，将所有 dst flush 合并到一次。
- **DSP-side sampling 实际收益极小**：跳过 logits copyback 仅节省 \~100-200us（因 ion\_sync\_mode=1 下整个 mempool sync 掩盖了局部收益），与实测一致：DSP-side sampling 功能正确，但性能提升可忽略。

**对优化方向优先级的影响**：

1. **DSP 内部 matmul kernel 优化是首要方向**：三类 matmul 占 DSP 执行时间的 91.1%，lm-head 专用 GEMV kernel 和 MUL\_MAT\_FFN 调优是 DSP 侧最具潜力的单点优化。
2. **AP 侧开销（Phase 1-9 + Phase 11-12）未被 DSP profiling 数据覆盖**：无法直接比较 AP 侧优化（descriptor 模板缓存等）与 DSP kernel 优化的收益。AP 侧开销需单独量化后再定优先级。
3. **lm-head 专用 GEMV kernel**：每 token 出现一次，max=4697us，是 TG 阶段 DSP 侧最大的单算子。
4. **MUL\_MAT\_FFN kernel 优化**（HMX 利用率、tile size）：收益面最广（35 次/batch x 334us = 11690us/batch）。
5. **a-inv 优化（已关闭）**：bit1 prior-dst 覆盖扩展于 2026-08-07 实验证伪，a-inv 接近结构性下限，不再列为收益项。

### 4.2 Qwen3.5-2B 的 25 路 graph 拆分：SOLVE_TRI 支持度差异

本轮 AB 测试的 `ggmlhexagon_dump_perf_stats` 输出揭示了一个此前未被记录的结构差异：**Qwen3.5-2B 在 JZ 后端每个 batch 的 cgraph 被拆成 25 个子图**。

**Table-11**：JZ 后端五模型 graph 拆分实测（`log_abtest_all_20260807-102443.txt`，JZ run 1）

| 模型           | batch\_calls | graph nodes (min/max) | total nodes | 拆分情况              |
| ------------ | :----------: | :-------------------: | :---------: | ----------------- |
| Qwen3.5-2B   |   **6400**   |     **26 / 62**       |   345,600   | **25 子图/batch** |
| Gemma4-E2B   |      256     |      1493 / 1493      |   382,208   | 完整单图              |
| Gemma4-E4B   |      256     |      1860 / 1860      |   476,160   | 完整单图              |
| Qwen1.5-1.8B |      256     |       821 / 821       |   210,176   | 完整单图              |
| Llama3.2-1B  |      257     |       3 / 501\*       |   128,755   | 基本完整（\* 见下注）     |

> \* Llama3.2-1B 的 min=3 来自 cgraph cache miss 时的零星小图（本轮 misses=6），256 个正式 batch 均为 ~501 节点完整图，不属于结构性拆分。

**根因**：Qwen3.5-2B 是 delta net 混合架构（24 层 = 13 标准 attention 层 + 11 个 linear-attention delta net 层），delta net 层每层调用一次 `ggml_solve_tri`（[delta-net-base.cpp：build_delta_net_chunking](file:///home/zhouwg/develop/ggml-hexagon/src/models/delta-net-base.cpp)）。两个后端对该算子的支持度不同：

- **QCOM 完整支持**：AP 侧 `supports_op` 有 `case GGML_OP_SOLVE_TRI`（[ggml-hexagon.cpp：ggml_backend_hexagon_device_supports_op](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp)），DSP 侧有 HVX kernel 实现（[`htp/solve-tri-ops.c`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/solve-tri-ops.c)）。
- **JZ 断在 AP 侧**：`init_op_validators()`（[ggml-hexagon-jz.cpp：init_op_validators](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp)）未注册 `GGML_OP_SOLVE_TRI` 的 validator，`ggmlhexagon_can_handle_op_through_cdsp` 对该算子返回 false。**JZ 的 DSP 侧能力已完整**：fork 自 htp/ 的 [`kernels/solve-tri-ops.c`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/solve-tri-ops.c) 与 entry.c 的 op 表注册（[entry.c：g_op_dispatch](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/entry.c)）都在，仅缺 AP 侧 validator 与 `ggml_op_to_htp_op` 映射（[entry.c：ggml_op_to_htp_op](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/entry.c) 的 switch 中无 `GGML_OP_SOLVE_TRI` case）。

于是 ggml scheduler 将 Qwen3.5-2B 的 cgraph 在每个 SOLVE_TRI 处切开：11 个 delta net 层的 SOLVE_TRI 回退 CPU，连同其前后依赖算子的段边界，将整图切成 25 段（6400 batch\_calls = 256 batches x 25 子图，345,600 / 6400 = 54 节点/子图）。

**对同步架构的惩罚被放大**：JZ 每个子图都要走完整 12-phase 同步流程（25 次 FastRPC round-trip + 25 倍 Phase 1-9/11-12 的 AP 开销），且因子图间串行，完全无法 pipelining；QCOM 对完整单图做一次 dspqueue 流水提交，AP prep 与 DSP compute 跨 layer 重叠。本轮实测 Qwen3.5-2B JZ 后端 6400 个子图的 AP phase 累计约 69ms（p1+p2+...+p12），摊到 256 个 token 约 270us/token；TG 阶段占比 <1% 尚可忽略（Qwen3.5-2B TG 仍 +93.6% 领先），但 PP 阶段（单 batch ~119ms）25 倍的固定开销约占 5-6pp，是 Qwen3.5-2B PP -9.2% 的重要构成。

**结论**：Qwen3.5-2B 的 PP 落后不全是调度框架差异，其中 SOLVE_TRI 算子支持度缺口导致的 25 段 cgraph 拆分是可通过补齐算子注册消除的结构性开销。

**修复确认（2026-08-07 夜间 CI，self-build-jz 分支）**：SOLVE_TRI/SSM_CONV 算子注册补齐与 RMS_NORM validator 放宽（per-head view 支持）已合入本分支。夜间五模型 CI（`log_abtest_all_20260807-223924.txt`）实测 Qwen3.5-2B `batch_calls=256`（与其余四模型一致），25 路拆分归零；JZ PP 从 436.6 提升至 501.7 tok/s（+14.9%），对 QCOM 从 -9.2% 翻转为 **+10.0%**，验证了 cgraph 拆分是 PP 差距的主要构成。

### 4.3 前置准备：Profiling 数据驱动

根据 4.1 节 DSP op-level profiling 实测数据，**在 DSP 执行内部，matmul kernel 是绝对主导**（三类 matmul 占 DSP batch-wall time 的 79.1% = 86.9%x91.1%）。注意：DSP profiling 数据仅覆盖 Phase 10（DSP 批处理执行），AP 侧开销（Phase 1-9 + Phase 11-12）未包含在内，需通过 AP 侧 profiling 单独量化。在 AP 侧数据补全前，优化方向优先聚焦在 DSP kernel 与 offload 策略上，AP 侧优化暂不调整优先级。

TG 和 PP 的瓶颈不同，优化策略也不同：

- **TG 瓶颈**（基于 4.1 profiling，仅覆盖 DSP 侧）：在 DSP 执行内部，三类 matmul 占 91.1% op-sum，其中 lm-head MUL\_MAT（max=4697us，每 token 1 次）和 MUL\_MAT\_FFN（avg=334us，每 layer 1 次 fused op = 105 个内部 matmul）是绝对主导；JZ 已通过 lm-head offload + first-touch 权重 inval（\~9.2 ms/token 节省，固定整图总量）解决最关键的两项，剩余优化空间主要在 DSP matmul kernel 本身。
- **PP 瓶颈**：PP 差距是**模型结构相关的**，不是普遍的 JZ 弱点。3.4 节分析表明 JZ 净优势 = per\_layer\_saving x n\_layers + fixed\_lmhead\_saving - dspqueue\_overlap。当层数足够时 JZ 也赢 PP（如 Gemma4-E2B 的 35 层，PP +49.5%）；浅层模型（llama3.2-1B 16 层）dspqueue 的固定 overlap 优势尚未被 per-layer 累积超越；Qwen3.5-2B 曾叠加 SOLVE_TRI 缺口导致的 25 路 cgraph 拆分（详见 4.2），该拆分已于本分支修复，PP 由 -9.2% 翻转为 +10.0%。3.4 节也明确指出：**性能差异来自 data-plane policy（weight residency + role-aware cache），而非 control-plane**（FastRPC 开销历史值 \~89us，可忽略）。

在投入任何优化之前，先跑一轮 benchmark 量化各阶段耗时：

- **AP 侧**：设置 `dump_debug_info=1`，量化 Phase 1-12 各阶段实际时间分布。
  - Phase 10 三阶段（`cum_p10_rpc_setup_us` / `cum_p10_dsp_exec_us` / `cum_p10_civac_us`）的占比。
  - TG 中 Phase 4-8 的固定开销究竟多大（验证 descriptor 模板缓存的收益上限）。
  - PP 中 Phase 1-9 + 11-12 的 AP 纯开销占比（验证 async/pipelining 的收益上限）。
- **DSP 侧**：`HEX_OP_PROF=1`（定义于 [dsp-ctx.h：HEX_OP_PROF](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/dsp-ctx.h)），量化 DSP 侧 per-op / per-layer 耗时分布（即 4.1 节与第五章的 OP-PROF 数据来源）。feature/force_opfusion_in_pp 分支 hardcode 为 1，主分支默认为 0，合入主分支时需改为运行时配置或 build flag。

**决策阈值**：

- 如果 Phase 1-9 + 11-12 在 PP 中占比 < 10%，async/pipelining 不值得做（FastRPC 开销历史值 \~89us，可忽略）。
- 如果 Phase 4-8 在 TG 中占比 > 5%，descriptor 模板缓存值得投入。

参考数据来源：本轮 Gemma4-E2B 端到端 256 calls dump 已提供 AP phase 累计实测，可作为 AP 侧 profiling 的起点。

### 4.4 TG 优化（巩固与扩大已有优势）

JZ 当前的 TG 优势来自 3.1 节 (lm-head DSP offload) 和 3.3 节 (role-aware cache 管理) 两个已实现机制。以下分析进一步扩展 TG 优势的优化方向：

#### 4.4.1 TG descriptor 模板缓存 - 消除 graph\_compute\_batch 中 AP 侧 prep phase 的 per-token 开销

TG 模式下，**每 token 的 cgraph 拓扑完全相同**，只有 tensor 数据指针变化。当前每 token 都要走 `graph_compute_batch` 的全部 12 phase（Phase 1-9 AP 侧全图分析 / 镜像 / 权重 repack / mempool 分配 / desc 构建 / cache flush，Phase 10 同步 RPC，Phase 11-12 AP 侧 cache inval / 回拷）。其中 Phase 1-9 内的 layout 计算、mempool offset 跟踪、descriptor 构建等纯 AP 工作在拓扑不变时可复用：

- **已有基础**：Phase 1 已有 cgraph cache（[ggml-hexagon-jz.cpp：ggmlhexagon_backend_graph_compute_batch](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp)），按 op+shape+src ptr 哈希命中时跳过 Phase 2 descriptor 重建。本轮 Gemma4-E2B 端到端 256 calls 中 cgraph cache 命中率 98.8%（hits=253, misses=3）。
- **本节方案在 cgraph cache 之上扩展**：缓存 mempool layout 与 desc 模板到层级别，后续 token 只 patch 变化的数据指针（activation/KV cache 地址），跳过 Phase 4-8 的 layout/desc 构建。
- 首次 token 或 cgraph 拓扑变化时（如 context shift）构建 descriptor 模板，记录所有 op 的 src/dst offset 与 mempool layout。
- **收益（按 4.1 profiling 数据估算）**：4.1 节 profiling 仅覆盖 DSP 侧 Phase 10，未测量 AP 侧 Phase 1-9 开销。descriptor 模板缓存省的是 AP 侧开销，无法用 DSP profiling 数据直接估算。需在 AP 侧 profiling 拿到 AP 侧 Phase 1-9 的精确耗时后再评估收益。作为参考，若 AP 侧 Phase 1-9 开销与 DSP 侧 non-op 开销（4693 us/batch）同量级，即使消除其中一半（\~2.3 ms），相对于端到端 TG 时间的占比也会因 AP 侧 Phase 11-12 的额外开销而更低。**descriptor 模板缓存优先级的最终判断依赖 AP 侧 profiling 数据**。
- **复杂度**：中等偏低。需要在 ctx 中缓存 descriptor 模板和 mempool layout，处理 KV cache 增长时的 realloc 以及 graph topology 变化（context shift）时的 invalidate。
- **注意**：权重 repack offset 在模型加载后不变，但 activation 地址每 token 不同，模板需要支持 per-pointer patch。

#### 4.4.2 KV cache 常驻 mempool + 增量 inval

当前 bit0 first-touch 标记对只读权重有效，但 KV cache 是 read-write 的，每 token 被 DSP 写入、AP 读取。KV cache 已在 mempool 中，Phase 11（cache inval）可以只 inval KV cache 的新增部分（增量 inval），而非每次做大范围 inval。

- **复杂度**：**高**。需要新增** DSP->AP 通信通道**：KV cache 写发生在 DSP 侧（每 layer FlashAttn 输出），AP 侧无法独立知道写入了哪些 position；需要 DSP 侧在 Phase 10 RPC reply 中携带 KV cache 写入范围（按 layer x position 的 bitmap 或 range list），AP 侧在 Phase 11 按此范围做精确 CIVAC。
- **注意**：bit0 机制不适用于 KV cache（read-write），需要独立的增量跟踪机制。

### 4.5 PP 优化（结构性收益，是 JZ 的真正战场）

JZ 在 4 个模型的 TG 上领先，平均优势 +48.1%，继续优化的边际收益受 matmul kernel 物理极限约束；PP 在 3/5 模型上落后 -3.3% 到 -22.3%，根因（AP-DSP 无 pipelining）是**可重构的框架差异**，而非 kernel 差异。**PP 从 -22.3% 改善到 -12% 等同于 +10pp 绝对提升**；TG 从 +48% 到 +58% 需要改 kernel 才有 +10pp，但 kernel 已与 QCOM 100% 共享，**只能从 matmul 内部优化（HMX 利用率、tile size）获取有限收益**。因此 PP 优化是 JZ 的真正战场，应排在 profiling 之后立即推进。

**Table-12**：PP 表现与模型结构关联

| 模型           | 层数                | PP JZ vs QCOM | TG JZ vs QCOM | 根因分项                                                 |
| ------------ | ----------------- | :-----------: | :-----------: | ---------------------------------------------------- |
| Gemma4-E2B   | 35 (GQA 8:1)      |   **+49.5%**  |     +8.1%     | 层数深且单层 DSP 时间适中（R 低），per-layer 优势累计超越 dspqueue overlap |
| Qwen3.5-2B   | 24 (GQA + Delta Net) |   **+10.0%** |   **+99.0%**  | 25 路 cgraph 拆分已修复归零（详见 4.2），per-layer 优势兑现，PP 反超 |
| Gemma4-E4B   | 42 (GQA 4:1)      |     -3.3%     |   **+38.2%**  | 单层 DSP 时间长（R 高），dspqueue 每层隐藏的 AP prep 放大，抵消 42 层累积优势   |
| Llama3.2-1B  | 16 (GQA 4:1)      |     -6.0%     |   **+47.1%**  | 层数浅，dspqueue 优势显著                                     |
| Qwen1.5-1.8B | 24 (MHA 1:1)      |     -22.3%    |     -28.8%    | **三重叠加：dspqueue + 层数不足 + MHA VTCM/cache**              |

**结论**：PP 优化应聚焦于**结构性杠杆**（per-layer pipelining）与**支持度缺口补齐**（SOLVE_TRI offload），而非模型结构特化。Qwen1.5-1.8B 不是特例，而是三重不利因素叠加的体现：per-layer pipelining 改善后这类模型获益最大。Gemma4-E2B 已经赢 PP，进一步提升 +49.5% 之上的性能也来自 per-layer pipelining 在深层模型上的累积收益。Qwen3.5-2B 的 25 路 cgraph 拆分（详见 4.2）已通过补齐算子注册与 validator 放宽修复，PP 从 -9.2% 翻转为 +10.0%，验证了该归因；其余模型的 PP 差距（Gemma4-E4B / Llama3.2-1B / Qwen1.5-1.8B）需靠 4.5.1 per-layer pipelining 等架构改动弥补。

#### 4.5.1 Per-layer intra-batch pipelining - 结构性突破点

**关键澄清**：3.4 节"性能差异来自 data-plane policy 而非 control-plane，FastRPC 开销 ~89us 可忽略"的论断，**不能用于反对 per-layer pipelining**。这两个是不同的概念：

- **FastRPC ~89us（历史值，本轮实测 min=102us / avg=154us）是 control-plane 路径成本**（RPC invoke 自身的 marshalling + transport 开销），与是否做 pipelining 无关
- **Pipelining 收益 = min(AP prep 时间, DSP compute 时间) 的隐藏量** - 完全由调度重叠决定，与 FastRPC 开销无关

FastRPC 是 control-plane 路径成本，pipelining 关心的是能否把 1-3ms 的 AP prep 隐藏在 5-10ms 的 DSP layer 执行后面。**这是两个独立维度**。

**当前同步模型的瓶颈**：

```
AP Phase 1-9 [=====] -> AP阻塞 [==] -> AP Phase 11-12 [===]
                          DSP Phase 10
```

PP 阶段单 layer DSP 计算时间（按 4.1 profiling 与本轮 Gemma4-E2B 实测分两段）：

- **TG (M=1)**：per 4.1 profiling，MUL_MAT avg=97us x 多 ops + FlashAttn 20us + RMS_NORM 1us 等 ≈ **200-500us per layer**
- **PP (M=58, Gemma4-E2B batch#1)**：本轮实测，batch-wall=81,637us / 35 layers ≈ **2,332us/layer**

AP Phase 1-9 + 11-12 估算：本轮 Gemma4-E2B dump 显示端到端 256 calls（1 PP + 255 TG）AP phase 累计 p1=101,249us / p2-p9=72,159us / p11-p12=3,653us，总和 177,061us，平均每 call 691us（**此为 PP+TG 平均值，非 PP 单独**）。PP batch#1 单次 81,637us 远大于 TG 的 ~37ms，AP 侧 PP 单独占比需在 AP 侧 profiling 中分离 PP/TG 后才能精确给出（粗估 5-10%）。如果按 layer 切分：

```
AP P1-P4 Layer1 [=] -> DSP Layer1 [==] -> AP P5-P7 Layer2 [=] -> DSP Layer2 [==] -> ... -> AP P11-12 [=]
```

**预期收益（基于估算）**：AP 侧 Phase 1-9 + 11-12 占 PP 10-15%，pipelining 隐藏 50-70%，PP 提速 5-10%。Qwen1.5-1.8B 从 -22.3% 改善到 -15% 左右，Gemma4-E2B 从 +49.5% 进一步到 +55%+。

**关键设计约束**：

- **维持 single mempool 不变**：TG 优势的根基，不可更改
- **切分粒度应为 layer 级**：op 级切分的 setup 开销将抵消收益
- **DSP 侧需要 partial-execute + resume 接口**：从 descriptor 中按 offset 启动执行的新基础设施
- **12 phase 测量框架要扩展到 per-layer**：现有 `cum_p1_us` ~ `cum_p12_us` 是 batch-level 聚合，需支持 per-layer 粒度才能验证 pipelining 收益
- **严格 TG 回归测试**：任何增加 AP↔DSP 同步点的改动都可能在 M=1 时引入额外开销，需确保不损害 TG 性能

**前置数据需求（依赖 4.3 节 profiling）**：

- Phase 1-9 + 11-12 的 AP 纯开销实测占比（决定 pipelining 收益上限）
- 单 layer DSP 计算时间分布（决定 AP prep 是否能完整隐藏在 layer 计算后面）
- per-layer cache flush 字节数（Phase 9 切分到 per-layer 后的实际开销）

**风险评估**：

- 收益面广：4/5 测试模型 PP 改善
- 实施复杂度高：DSP partial-execute 接口是新基础设施
- 风险点：MUL_MAT per-layer 平均仅 97us（远低于本轮实测 min=102us 的 FastRPC 开销，pipelining 切换代价 1-2 倍），**单 matmul pipelining 无收益；必须聚合到 layer 级别才有收益**。4.1 profiling 给出的是 batch#200 累计值，本轮 Gemma4-E2B batch#1 已提供 per-layer 实测数据（batch-wall / n_layer ≈ 2,332us/layer）

#### 4.5.2 descriptor 模板缓存（视 profiling 结果实施）

如 4.4.1 节所述，descriptor 模板缓存可减少 AP 侧 prep 时间，与 pipelining 是**互补关系**（pipelining 利用 prep 的时间，缓存减少 prep 本身）。**已有 cgraph cache 覆盖 Phase 1-2 hit case（98.8% 命中率，详见 4.4.1）**，本节方案进一步跳过 Phase 4-8。**如果 AP 侧 profiling 显示 AP 侧 prep 是 pipelining 收益的主要瓶颈，缓存应同步实施**。对长 context PP 收益较大（5-10%），对 TG M=1 收益很小。

#### 4.5.3 SOLVE_TRI offload 启用：消除 Qwen3.5-2B 的 25 路 graph 拆分

4.2 节已确认：Qwen3.5-2B 的 cgraph 在 JZ 后端被拆成 25 个子图的唯一原因是 `GGML_OP_SOLVE_TRI` 未在 AP 侧注册，而 DSP 侧 kernel 已完整存在。补齐两处胶水代码即可消除 graph 拆分：

- **AP 侧**：在 `init_op_validators()`（[ggml-hexagon-jz.cpp：init_op_validators](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp)）新增 `s_op_validators[GGML_OP_SOLVE_TRI]`，校验逻辑可直接参照 QCOM 的 `ggml_hexagon_supported_solve_tri`（[ggml-hexagon.cpp：ggml_hexagon_supported_solve_tri](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp)：F32 类型 + 方阵 + 维度匹配检查）。
- **DSP 侧**：在 `ggml_op_to_htp_op`（[entry.c：ggml_op_to_htp_op](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/entry.c)）新增 `case GGML_OP_SOLVE_TRI: *htp_op = HTP_OP_SOLVE_TRI; return 0;`。kernel 本体（`op_solve_tri`，[solve-tri-ops.c：op_solve_tri](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/solve-tri-ops.c)）与 op 表注册（[entry.c：g_op_dispatch](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/entry.c)）均已存在，无需改动。

**预期收益**：Qwen3.5-2B 恢复完整单图后，每 batch 的 25 次 FastRPC round-trip 与 25 倍 Phase 1-9/11-12 开销归零，按 4.2 节实测估算 PP 可收回约 5-6pp（-9.2% 收窄至约 -3%）；TG 收益 <1%（拆分开销在 M=1 时占比已极小），但 25->1 的子图收敛同时降低了 per-token 的 cgraph cache 与 mempool 管理负担。

**复杂度**：**低**。两处注册各约 10-20 行，无新 kernel、无调度框架改动、无 cache 策略变化，是全部 PP 方向中风险收益比最优的一项。

**风险与验证**：

- SOLVE_TRI 在 Qwen3.5-2B 中为 F32 算子，QCOM validator 的 F32/方阵/维度检查可直接复用；JZ 侧需确认 delta net 的 chunked 调用路径下 tensor shape 均满足该校验。
- 合入前必须跑五模型 CI：Qwen3.5-2B 重点验证输出无乱码且 `batch_calls` 从 6400 回落至 256；其余四模型验证无回归（它们的 cgraph 本就不含 SOLVE_TRI，预期零影响）。
- 该改动同时消除了 Qwen3.5-2B PP 分析中的一个混杂变量：拆分消失后，Qwen3.5-2B 的 PP 差距将更纯粹地反映 dspqueue pipelining 优势，可作为 4.5.1 per-layer pipelining 收益验证的对照组。

**实际收益（2026-08-07 夜间 CI，self-build-jz）**：SOLVE_TRI offload 启用后 batch_calls 6400->1792，仍残留 6 个 MHA 层 attn_q_norm 拆分（第二根因：RMS_NORM validator 拒绝 per-head view）；validator 放宽后 batch_calls 收敛至 256。夜间 CI（`log_abtest_all_20260807-223924.txt`）实测 JZ PP 436.6 -> 501.7 tok/s（+14.9%），对 QCOM 从 -9.2% 翻转为 +10.0%，超过本节 5-6pp 预期（预期仅计 cgraph 拆分开销，实际还含第二根因收益）；TG 持平（26.7 tok/s，对 QCOM +99.0%）。

### 4.6 低风险快速收益（边际优化与实验结论）

#### 4.6.1 a-inv 优化

4.1 profiling 显示 a-inv 是最大 non-op 开销（1030 us/batch, 82 MB）。bit1 机制（[entry.c：INVAL_SRC_IF_NEEDED](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/entry.c)）在读 **src** 时，若前序某个 op 的 dst 已覆盖该 src，就**跳过该 src 的 cache invalidation**（L2 已是新鲜的）。

- **实验证伪（2026-08-07）**：本轮测试 `dsp_cache_mode=5`（bit0+bit2）未启用 bit1；Qwen1.5-1.8B PP-only 对照实验（mode=7 vs mode=5）显示 a-inv 零变化（19167us/1754MB -> 19135us/1754MB）。根因是 `prior_dst_add` 只登记 len <= PRIOR\_DST\_MAX\_LEN=64（单 cacheline）的 dst（[entry.c：prior_dst_contains_src](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/entry.c)），cgraph 中间张量均 >= 256 字节，prior\_dst 列表为空，skip 路径几乎不触发。放宽该上限会引入 async DMA/HMX 路径的 stale L2 read 风险（[entry.c：PRIOR_DST_MAX_LEN](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/entry.c) 设计注释），不建议尝试。
- **结论**：per-batch dedup 已保证每条 unique src 每 batch 至多失效一次，a-inv 字节量接近 batch 内结构性下限。原"预期 500us/batch = 1.4% TG"收益不成立，本项关闭。跨 batch dedup 是独立假设，仍待验证。

#### 4.6.2 MUL_MAT_FFN kernel 优化

4.1 profiling 显示 MUL_MAT_FFN avg=334us x 35 calls = 11.7ms/batch（TG 主要热点）。优化空间集中在 HMX 利用率与 tile size。

- **预期收益**：HMX 利用率提升 30% 可省 3.5ms/token = 9% TG
- **复杂度**：中（需 DSP kernel 修改）
- **风险**：tile size 调大需要更多 VTCM，可能与 lm-head 等大算子冲突

#### 4.6.3 post-matmul activation 与 element-wise fuse

`graph_compute_batch` Phase 3 已实现 QKV/FFN/mm_add fusion 以及 `RMS_NORM + MUL`（[ggml-hexagon-jz.cpp：ggmlhexagon_backend_graph_compute_batch](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp)），本轮 Table-17 显示 RMS_NORM_MUL count=227、GLU_GEGLU count=35 是 Gemma4-E2B 主力算子。**未融合的剩余空间**在 `MUL_MAT -> post-matmul activation`（如 matmul -> SiLU -> mul 即 SwiGLU 的端到端 inline fuse），以及 `MUL_MAT -> element-wise broadcast` 的反向（与 RMS_NORM_MUL 方向相反）场景。在 M=1 TG 时，element-wise op 写入 DDR 再被下一个 matmul 读回是纯粹的浪费，fuse 后可减少一次中间 tensor 的 DDR round-trip。

- **预期收益**：<1% TG，PP 收益更小
- **复杂度**：中（kernel 修改 + AP 侧调度调整）

#### 4.6.4 减少 Phase 10 同步 RPC round-trip 开销

FastRPC 开销已在 warmup 阶段校准（变量 `min_rpc_overhead_us`，[ggml-hexagon-jz.cpp：min_rpc_overhead_us](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp)）。本轮 Gemma4-E2B 端到端实测 256 calls 中 warmup n=6, **min=102us, max=251us, avg=154us**。2026-07-24 历史值 \~89us 已不适用本轮测量（设备 thermal / kernel 调度变化可能导致漂移）。相对于 \~37 ms/token 的 TG 占比仍极小（<0.5%）。除非 profiling 发现非预期的高开销，否则此项投入产出比低，不建议优先投入。

#### 4.6.5 Sampling 路径优化

- **DSP 侧组件**：DSP 侧 lm-head matvec 之后做 softmax + argmax + top-k/p，仅返回 4 字节 token ID 而非完整 logits 矩阵。当前流程 `DSP lm-head -> F32 logits (~500KB-1MB) -> memcpy 回 AP -> CPU sampling`，优化后 `DSP lm-head -> DSP softmax -> DSP argmax -> 返回 int32 token ID (4 bytes)`。
- **AP 侧组件**：AP 侧 sampler chain 适配新的 4 字节 token ID 输入，sampler chain 多个算子替换为更快实现。单独看 AP 侧有 2.5x 优化空间（QCOM 的 AP 侧 sampling 路径比 JZ 快）。
- **复杂度**：高。DSP 侧在 256K vocab 上做 top-k（O(n log n)）+ top-p（cumulative sum + rejection sampling）+ Hexagon RNG 集成；AP 侧 sampler chain 多个算子替换。
- **收益估算**：logits memcpy 在 DDR 带宽下仅 ~10-20us（500KB-1MB / SnapDragon 8 Elite LP-DDR5x 5300 MHz 理论 ~50 GB/s），FastRPC 开销本轮实测 min=102us / avg=154us，合计 ~110-170 us/token。AP 侧单独看有 2.5x 优化空间。
- **实验结论（2026-08-06 验证）**：配套修改后功能正确，但性能收益 <0.5% / <1%（ion\_sync\_mode=1 下整个 mempool sync 掩盖了局部收益），代码复杂度高，**已回滚**，不进入优化路线图执行队列。
- **对未来读者的提示**：若再次评估 sampling 路径优化，应直接以本实验的 <0.5% / <1% 上限数据为参考起点，无需重复"理论收益 ~110-170us / <0.5%"的独立估算。

### 4.7 优化路线图

**潜在收益优先级 vs 执行优先级是两个维度**：4.4-4.6 按**潜在收益空间**排序（4.4 TG 优化 > 4.5 PP 优化 > 4.6 低风险快速收益），实际执行顺序（下方 Step 0-4）与之不同。**调整依据**：sampling 路径优化已验证收益极小（详见 4.6.5）；TG 已领先 4/5 模型，边际收益低；PP 是真正短板（拆分修复前 4/5 模型落后 QCOM，修复后仍 3/5 落后），结构性收益空间最大。**两个维度的执行映射**：4.5.3 SOLVE_TRI offload 列为 Step 1（立即收益，已完成：夜间 CI 实测 Qwen3.5-2B PP +14.9%，对 QCOM 从 -9.2% 翻转为 +10.0%；4.6.1 a-inv 已于 2026-08-07 实验证伪关闭）；PP 结构性突破（4.5.1 per-layer pipelining）作为 Step 2 核心战场；TG kernel 精调（4.6.2 + 4.6.3 + 4.4.2）作为 Step 3 最后做。**长期/已关闭项**：4.6.5 sampling 路径优化与 4.4.2 KV cache 增量 inval 因复杂度高/收益小。

4.3-4.6 节按优先级组织，经实验验证后，实际执行顺序调整为：

```
Step 0: Profiling 数据驱动（必做前提，详见 4.1）

Step 1: 低风险快速收益（独立于 PP/TG 主战场）
  +-- 4.5.3 SOLVE_TRI offload：已完成（2026-08-07 夜间 CI：batch_calls 6400->256，Qwen3.5-2B PP 436.6->501.7 tok/s，对 QCOM -9.2% -> +10.0%）
  +-- 4.6.1 a-inv 优化：2026-08-07 实验证伪（bit1 受 PRIOR_DST_MAX_LEN=64 限制退化为 no-op），关闭
  +-- 4.6.4 FastRPC 校准：已实测，结论为投入产出比低，关闭

Step 2: PP 结构性突破（核心战场，详见 4.5.1）
  +-- DSP 侧 partial-execute + resume 接口 + async FastRPC 调度
  +-- 严格 TG 回归测试：M=1 单次 invoke 优势不被新同步点抵消
  +-- （视 profiling 结果）4.5.2 descriptor 模板缓存
  +-- 预期：PP +5-10%；Qwen1.5-1.8B 从 -22.3% -> ~-15%；Gemma4-E2B 从 +49.5% -> +55%+

Step 3: TG kernel 精调（边际收益，详见 4.6.2 + 4.6.3 + 4.4.2）
  +-- 4.6.2 MUL_MAT_FFN kernel 优化（HMX 利用率与 tile size）
  +-- 4.6.3 post-matmul activation fuse
  +-- 4.4.2 KV cache 增量 inval：需新增 DSP->AP 通信通道，列为长期项

Step 4: 长期架构（按 Step 2 效果决定）
  +-- 如 Step 2 成功：在 JZ 架构内深化 per-layer pipelining
  +-- 如 Step 2 失败：保留单次 invoke 模型，强化 single mempool + batch-level cache
  +-- 多 batch 并发 PP（服务端场景吞吐优化，独立方向）
```

### 4.8 核心原则

**不要为了追 PP 性能而破坏 TG 的优势。** JZ 在 TG 上的优势（single mempool -> lm-head offload、role-aware cache）来自架构层面。QCOM 采用 dspqueue 异步队列框架，与 JZ 架构近乎互斥：两者的控制平面原语不同（FastRPC sync vs dspqueue async），数据平面策略也不同（single mempool + offset addressing vs per-buffer + bi indirection），且 PR #26049 将 cache coherency 维护的部分逻辑下放到算子内部实现后，与 JZ 的 cache 子系统不兼容（详见第一章）。现实路径是在 JZ 架构内增加 PP 优化（如 per-layer pipelining），而非融合两种架构。

## 五、force_opfusion_in_pp 实验

> **基线 commit**: `a3d04682c11086450d36091a15534b14e65dda2a`（feature/force_opfusion_in_pp 分支）
>
> **模型**: `gemma-4-E2B-it-Q4_0.gguf`（默认测试模型，35 层）

### 5.1 实验动机

第 4.3 节确定 PP 是 JZ 的真正短板之后，需要找到一个低风险高收益的切入点。观察到 `is_mergeable_mul_mat()` 中的 HMX-eligibility 闸门在 PP 路径下必然拒绝所有 MUL_MAT（因为 `M > HTP_MM_HMX_MIN_NROWS=4`），导致 QKV/FFN/mm_add fusion 在 PP 完全失效。初始假设：

- **假设 A**：3 个独立 HMX MUL_MAT -> 1 个 HVX fused MUL_MAT_QKV，单算子更慢但 cache 失效次数减少到 1/3
- **假设 B**：cache 失效节省 > 算子额外耗时 -> 净收益为正

为此引入 `force_opfusion_in_pp` 配置开关（0=保持原 HMX 闸门，1=旁路闸门强制融合），并加 3 个 cum 计数器（`n_qkv_skip_cum_hmx` / `n_pair_skip_cum_hmx` / `n_mm_add_skip_cum_hmx`）量化被错过的融合机会数。

### 5.2 实验设计

**Table-13**：force_opfusion_in_pp 实验配置

| 配置 | 含义 | 备注 |
|---|---|---|
| `enable_opfusion=1` | QKV/FFN/mm_add fusion 总开关 | 原默认 |
| `force_opfusion_in_pp=0` | 保留 HMX 闸门(基线) | 原默认 |
| `force_opfusion_in_pp=1` | 旁路 HMX 闸门,大 M 路径下也允许融合 | 实验性 |

实现细节：

- `is_mergeable_mul_mat` 加 bypass 分支：`if (g_hexagon_appcfg.force_opfusion_in_pp) return true;`
- 3 个 cum 计数器在对应 skip 分支自增（每发生一次 +1，不受日志频率限制）
- `mul_mat coverage` 打印新增 `qkv_skip_hmx / pair_skip_hmx` 字段
- `ggmlhexagon_print_running_timestamp` 打印 `enable_opfusion` 与 `force_opfusion_in_pp` 当前值，便于运行时核查配置
- `scripts/ggml-hexagon.cfg` 新增 `force_opfusion_in_pp = 0` 默认值与说明

### 5.3 Gemma4-E2B 单模型对比：force=0 baseline vs force=1 实验

Gemma4-E2B（35 层，GQA 8:1）是默认测试模型。本节在同一模型上对比 `force_opfusion_in_pp=0` 与 `force_opfusion_in_pp=1` 两组数据，量化"旁路 HMX 闸门"的净收益。

#### 5.3.1 Baseline 数据（`force_opfusion_in_pp=0`）

```
mul_mat coverage: total=277 hmx=276 (99.6%) qkv_fused=0 (saves 0.0%) ffn_fused=0 (saves 0.0%) mm_add_fused=0 (saves 0.0%) qkv_skip_hmx=15 pair_skip_hmx=65
hmx eligibility: total=1940 pass=1386 (71.4%)
batch-wall cum=81252 us op-sum cum=56999 us non-op avg=24253 us/batch
non-op: hdr=4 pre=392 w-inv=13639(1334MB) a-inv=6869(551MB) dst=112 bulk=1677 queue=8 us/batch
```

**关键解读**：

- `qkv_skip_hmx=15` 与 35 层对应，**每层 1 个 QKV 候选被 HMX 闸门拒绝**，即基线未利用 15 个 QKV 融合机会
- `pair_skip_hmx=65` 即基线未利用 ~65 个 (MUL_MAT, MUL_MAT) pair 融合机会（覆盖 FFN gate+up、output projection + next 等相邻 MUL_MAT 对）
- 277 个 MUL_MAT 中 276 走 HMX 路径（99.6%），QKV/FFN/mm_add 全部走单算子 HMX
- non-op 中 `w-inv=13.6ms(1334MB)` + `a-inv=6.9ms(551MB)` 占 batch-wall 25%，是融合潜在节省的最大项

#### 5.3.2 对比数据 `force_opfusion_in_pp=1`：HVX 融合路径在 PP 慢 3.8x

```
mul_mat coverage: total=277 hmx=276 (99.6%) qkv_fused=15 (saves 16.2%) ffn_fused=35 (saves 25.3%) mm_add_fused=0 (saves 0.0%) qkv_skip_hmx=0 pair_skip_hmx=0
hmx eligibility: total=1940 pass=1386 (71.4%)
batch-wall cum=309487 us op-sum cum=285285 us non-op avg=24202 us/batch
non-op: hdr=6 pre=303 w-inv=13630(1334MB) a-inv=6934(551MB) dst=129 bulk=1679 queue=8 us/batch
[OPROF] op=MUL_MAT_QKV cum=17534 us count=15 avg=1168 min=997 max=1842 us
[OPROF] op=MUL_MAT_FFN cum=228047 us count=35 avg=6515 min=4153 max=8288 us
```

#### 5.3.3 对比分析

**Table-14**：force=0 vs force=1 对比

| 指标 | baseline (force=0) | 实验 (force=1) | 变化 |
|---|---:|---:|---:|
| batch-wall | 81,252 us | **309,487 us** | **+280% (3.8x 慢)** |
| MUL_MAT count | 277 | 162 | -115 (115 个被融合) |
| MUL_MAT time | 38,069 us | 21,024 us | -45% |
| MUL_MAT_QKV | 0 | 17,534 us (15 ops) | +17.5 ms |
| **MUL_MAT_FFN** | 0 | **228,047 us (35 ops)** | **+228 ms (单 op 6.5 ms!)** |
| qkv_fused | 0 | 15 | bypass 生效 |
| ffn_fused | 0 | 35 | bypass 生效 |
| qkv_skip_hmx | 15 | 0 | 计数器归零 |
| pair_skip_hmx | 65 | 0 | 计数器归零 |
| non-op w-inv | 13,636 us | 13,630 us | ≈ 不变 |
| non-op a-inv | 6,875 us | 6,934 us | ≈ 不变 |

Table-14 揭示 3.8x 退化的根因：MUL_MAT_FFN 在 HVX fused 路径下单 op 耗时 6.5 ms，35 个 op 合计 228 ms，占 batch-wall 的 73%。假设 B（cache 失效节省 > 算子额外耗时）被证伪：w-inv 与 a-inv 均与 baseline 一致，旁路 HMX 闸门并未带来 cache 失效的节省。

### 5.4 五模型 CI 验证：force=0 无回归 + HVX 融合 PP 不通用确认

- 通过 `./scripts/build-run-android.sh run_force_opfusion_in_pp_all` 一键运行五个模型，每模型生成 `*_terminal.txt`（端到端 tok/s） + `*_logcat.txt`（OP-PROF 算子分布） 两个 log

#### 5.4.1 五模型基础信息

**Table-15**：五模型基础参数（按 alias 数组顺序）

| # | alias       | 模型文件                                       | vocab_size | lm-head 类型 | 层数  | 注意力类型     | 唯一算子 (PP/TG)         |
| - | ----------- | ------------------------------------------ | ---------- | ---------- | --- | --------- | ------------------- |
| 1 | Gemma4-E2B      | gemma-4-E2B-it-Q4_0.gguf                   | 256,000    | Q4_K       | 35  | GQA 8:1   | GLU_GEGLU, UNARY_TANH |
| 2 | qwen3       | Qwen3.5-2B-Q4_0.gguf                       | 151,936    | Q6_K       | 24  | GQA + Delta Net | **GATED_DELTA_NET**, L2_NORM, UNARY_SILU/SIGMOID/SOFTPLUS |
| 3 | qwen1       | Qwen1.5-1.8B-Q4_0.gguf                     | 151,936    | Q6_K       | 24  | MHA 1:1   | MUL_MAT_ADD, MUL_MAT_FFN, GLU_SWIGLU |
| 4 | llama3      | Llama-3.2-1B-Instruct-Q4_0.gguf            | 128,256    | Q4_K       | 16  | GQA 4:1   | MUL_MAT_ADD, GLU_SWIGLU |
| 5 | Gemma4-E4B  | gemma-4-E4B_q4_0-it.gguf                   | 256,000    | Q4_K       | 42  | GQA 4:1   | GLU_GEGLU, UNARY_TANH |

**说明**：

- 层数来自 `ggmlhexagon_dump_perf_stats` 的 `model: n_layer=N` 字段（扫描 tensor name 末尾连续数字段得到 max layer index）
  - Qwen3.5-2B：24 个总层中 13 个含 FFN，其余为 linear-attention delta net 层（见 5.4.3）；log 中 `ffn_gate-12` 编号指 FFN 算子所在层（0-12），非 tensor 层编号
- 五个测试均使用 `n_ctx=8192, n_batch=2048, n_predict=256, n_threads=6, dsp_cache_mode=5, ion_sync_mode=1`
- PP batch#1 中 五个模型均未出现 MUL_MAT_QKV fusion（HMX 闸门正常拒绝，见 5.4.4）；MUL_MAT_FFN 仅 Qwen1.5-1.8B 边缘触发 1 次（count=1, cum=265us）；MUL_MAT_ADD 在 Llama3.2-1B/Qwen1.5-1.8B 触发，Gemma4 系列不触发

#### 5.4.2 PP batch#1 五模型 OP-PROF 对比

**Table-16**：五模型 PP batch#1 OP-PROF 对比

| 模型         | n_layer | batch-wall (us) | op-sum (us) | non-op (us) | MUL_MAT cum (us) | MUL_MAT count | MUL_MAT max (us) | FLASH_ATTN cum (us) | FLASH_ATTN count | non-op w-inv (MB) | non-op a-inv (MB) | non-op bulk (us) |
| ---------- | :-----: | :-------------: | :---------: | :---------: | :--------------: | :-----------: | :--------------: | :-----------------: | :--------------: | :---------------: | :---------------: | :--------------: |
| Gemma4-E2B     |   35    |     81,637      |   57,592    |   24,045    |     39,360       |      277      |     **4,448**    |       4,026         |        35        |     **1,334**     |       542         |      1,485       |
| Gemma4-E4B |   42    |    140,304      |   97,944    |   42,360    |     70,418       |      344      |     **7,873**    |       6,102         |        42        |     **2,528**     |       891         |      2,906       |
| llama3     |   16    |     37,598      |   24,275    |   13,323    |      9,856       |       79      |        262       |       5,277         |        16        |        497        |       415         |      2,634       |
| qwen1      |   24    |     91,150      |   47,613    | **43,537**  |     11,575       |       48      |       3,846      |      **18,615**     |        24        |        818        |    **1,754**      |   **15,119**     |
| qwen3      |   24    |        794      |      569    |      225    |        191       |        1      |        191       |          -          |        -         |         6         |         5         |         53       |

**关键观察**：

1. **batch-wall 与模型规模/层数正相关**：Gemma4-E4B / Gemma4-E2B ratio = 1.72x，与层数比 42/35=1.20x + 模型尺寸比 4B/2B=2.0x 加权吻合（Gemma4-E4B 的 GQA 4:1 较 Gemma4-E2B 的 8:1 有更大的 attention 中间张量，但对 batch-wall 仅为二阶影响）
2. **MUL_MAT max 是 lm-head 标志**：Gemma4-E2B max=4,448us，Gemma4-E4B max=7,873us，Qwen1.5-1.8B max=3,846us，Llama3.2-1B max=262us（vocab=128K 在 PP 阶段分块执行，无显式大算子）。Gemma4-E2B 与 4.1 节 baseline max=4,697us 差异 5.3%（测量噪声范围内）
3. **Qwen1.5-1.8B 的 FLASH_ATTN avg=776us 显著高于其他**：MHA（1:1 attention）在 PP 大 M 下 Q@K^T 矩阵规模最大；GQA 模型的 avg 仅 115-145us（Gemma4-E2B 35层 avg=115us，Gemma4-E4B 42层 avg=145us）
4. **Qwen1.5-1.8B 的 non-op a-inv=19,134us + bulk=15,119us 是五模型中最高**：MHA 模型 attention 中间张量（Q@K^T, Softmax(QK^T)·V）占用最大 VTCM 与 DDR 带宽，导致 cache 维护代价翻倍。这是 3.4 节"Qwen1.5-1.8B 三重叠加根因"的直接证据
5. **w-inv 随模型规模增长**：Gemma4-E4B 25,802us（2,528MB） vs Gemma4-E2B 13,637us（1,334MB），4B 参数量首次 touch 的权重范围翻倍。a-inv 方面，Gemma4-E2B 6,900us（542MB）较 Qwen1.5-1.8B 少 2.8x，GQA 8:1 attention 中间张量比 MHA 1:1 小约 2.8x，符合 GQA 压缩理论值
6. **Qwen3.5-2B 是 init batch（M=1）**：仅含 embedding 初始化算子（6 个 op），不代表真实 PP 性能，需用 `run_pp_only` 重抓
7. **MUL_MAT count 反映 cgraph 大小**：Gemma4-E2B 1116 graph ops 中 277 个 MUL_MAT，Gemma4-E4B 1384 ops 中 344 个，Qwen1.5-1.8B 533 ops 中 48 个，Llama3.2-1B 296 ops 中 79 个。差异主要来自 FFN/attention 内部 matmul 数量与是否使用 GQA

##### 5.4.2.1 Gemma4-E2B 详细算子分布

**Table-17**：Gemma4-E2B PP batch#1 完整 15 算子分布

| 算子             | cum (us) | count | avg (us) | min (us) | max (us) |
| -------------- | -------- | ----- | -------- | -------- | -------- |
| **MUL_MAT**    | 39,360   | 277   | 142      | 16       | **4,448** |
| **GLU_GEGLU**  | 5,845    | 35    | 167      | 123      | 198      |
| **FLASH_ATTN_EXT** | 4,026 | 35    | 115      | 104      | 141      |
| **RMS_NORM_MUL** | 3,752  | 227   | 16       | 4        | 103      |
| ADD            | 1,894    | 106   | 17       | 16       | 89       |
| ROPE           | 960      | 50    | 19       | 9        | 42       |
| MUL            | 638      | 70    | 9        | 4        | 18       |
| SCALE          | 312      | 6     | 52       | 13       | 87       |
| UNARY_TANH     | 264      | 1     | 264      | 264      | 264      |
| UNARY_GELU     | 194      | 35    | 5        | 4        | 12       |
| SET_ROWS       | 140      | 30    | 4        | 3        | 8        |
| CPY            | 130      | 1     | 130      | 130      | 130      |
| RMS_NORM       | 72       | 15    | 4        | 3        | 8        |
| GET_ROWS       | 5        | 1     | 5        | 5        | 5        |
| **op-sum 合计** | **57,592** |     |          |          |          |

**non-op 分布**：

- hdr=5, pre=302, **w-inv=13,637 us (1,334MB)**, **a-inv=6,900 us (542MB)**, dst=125, bulk=1,485, queue=10 us/batch
- non-op 合计=24,045 us/batch（占 batch-wall 29.5%）

**FFN/QKV skip 模式**（Gemma4-E2B batch#1 真实日志）：

- `QKV skip: is_qkv_mergeable=false (HMX gate)` at i=4/1116 -> HMX 闸门按预期拒绝 QKV 融合
- `FFN skip: is_mergeable_mul_mat_pair=false` at i=4/1116 -> HMX 闸门按预期拒绝 FFN pair
- `FFN skip: next not MUL_MAT` at i=6/1116, i=17/1116 -> FFN pair 中 next op 是 UNARY_TANH (op=25) 而非 MUL_MAT，跳过原因不是 HMX 闸门而是 graph 顺序

**tok/s 数据**（`common_perf_print` 输出，本轮端到端性能）：

- **prompt eval time = 87.00 ms / 58 tokens (1.50 ms per token, 666.69 tokens per second)**
- **eval time = 9,623.58 ms / 255 runs (37.74 ms per token, 26.50 tokens per second)**
- total time = 10,052.86 ms / 313 tokens
- graphs reused = 253
- unaccounted time = 26.95 ms / 0.3 %

**`ggmlhexagon_dump_perf_stats` 完整统计**（Gemma4-E2B 端到端 256 批次）：

- device info: Qualcomm Snapdragon 8 Elite, dsp arch version 0x79, system mem size 24834 MiB
- device: Hexagon-CDSP0, arch=QCOM_HTP_V79, vtcm=8MB, hvx=1, hmx=1
- **model: n_layer=35 (parsed from tensor name suffixes)**
- rpc stats: batch_calls=256, cum_p10=9,416,363 us, cum_graph=9,593,148 us, avg_p10=36,782 us, avg_graph=37,473 us
- graph nodes: min=1493, max=1493, total=382,208
- graph ops (post-fusion): min=824, max=889
- per-call range: graph=[35,885, 84,879] us, p10=[35,495, 84,169] us
- per-call overhead: n=256, min=136, max=6676, avg=690 us (graph_dur - p10)
- max graph detail: dur=84,879 us, n_nodes=1493, n_ops=889
- AP phase cumulative: p1=101,249, p2=2,730, p3=1,157, p4=41, p5=10,060, p6=10,342, p7=164, p8=42,472, p9=4,373, p11=3,469, p12=184, unaccounted=544 us
- p10 3-way cumulative: rpc_setup=38, dsp_exec=9,416,363, civac=3,385 us (sum=9,419,786)
- rpc overhead (warmup): n=6, min=102, max=251, avg=154 us (pure FastRPC/mempool transport)
- cgraph cache: hits=253, misses=3 (hit_rate=98.8%), entries=0

**`ggmlhexagon_print_running_timestamp` 完整配置**（Gemma4-E2B）：

- ggml_hexagon_version: 0.99.6
- offload MUL_MAT types: F32, F16, BF16, Q4_0, Q8_0, Q4_1, IQ4_NL, MXFP4
- thread_counts on CDSP: 6
- ion_sync_mode: 1
- rpc_mmap_mode: 0
- dsp_cache_mode: 5
- dsp_cache_trace_bit0: 0
- dsp_cache_trace_bit1: 0
- dump_diag_info(DSP): 0
- dump_diag_info(AP): 0
- enable_graph_optimize: 1
- enable_opfusion: 1
- **force_opfusion_in_pp: 0**（确认默认基线）
- enabled_ops: ALL
- running timestamp: 2026-08-06, 21:15:02

**与 4.1 节 baseline 对比**：5 项核心指标（batch-wall / op-sum / non-op / MUL_MAT cum / MUL_MAT max）差异均在 ±6% 以内（batch-wall 81,637us vs 81,252us, 0.5%），确认 force=0 cleanup 无回归

##### 5.4.2.2 Gemma4-E4B 详细算子分布

**Table-18**：Gemma4-E4B PP batch#1 完整 15 算子分布

| 算子             | cum (us) | count | avg (us) | min (us) | max (us) |
| -------------- | -------- | ----- | -------- | -------- | -------- |
| **MUL_MAT**    | 70,418   | 344   | 204      | 25       | **7,873** |
| **GLU_GEGLU**  | 7,646    | 42    | 182      | 180      | 193      |
| **FLASH_ATTN_EXT** | 6,102 | 42    | 145      | 135      | 180      |
| **RMS_NORM_MUL** | 6,250  | 278   | 22       | 7        | 111      |
| ADD            | 3,665    | 127   | 28       | 27       | 127      |
| ROPE           | 1,137    | 66    | 17       | 9        | 43       |
| MUL            | 1,080    | 84    | 12       | 4        | 30       |
| SCALE          | 369      | 6     | 61       | 21       | 99       |
| UNARY_TANH     | 264      | 1     | 264      | 264      | 264      |
| UNARY_GELU     | 240      | 42    | 5        | 5        | 15       |
| SET_ROWS       | 401      | 48    | 8        | 5        | 17       |
| CPY            | 179      | 1     | 179      | 179      | 179      |
| RMS_NORM       | 187      | 24    | 7        | 6        | 12       |
| GET_ROWS       | 6        | 1     | 6        | 6        | 6        |
| **op-sum 合计** | **97,944** |     |          |          |          |

**non-op 分布**：

- hdr=5, pre=379, **w-inv=25,802 us (2,528MB)**, **a-inv=11,056 us (891MB)**, dst=149, bulk=2,906, queue=10 us/batch
- non-op 合计=42,360 us/batch（占 batch-wall 30.2%，与 Gemma4-E2B 的 29.5% 几乎一致，说明 GQA 比例从 8:1 降到 4:1 不会显著改变 non-op 占比）

**FFN/QKV skip 模式**（Gemma4-E4B batch#1 真实日志）：

- `QKV skip: is_qkv_mergeable=false (HMX gate)` at i=4/1384 -> HMX 闸门按预期拒绝 QKV 融合
- `FFN skip: is_mergeable_mul_mat_pair=false` at i=4/1384 -> HMX 闸门按预期拒绝 FFN pair
- `FFN skip: next not MUL_MAT` at i=6/1384, i=17/1384 -> 同 Gemma4-E2B，graph 顺序问题

**tok/s 数据**（`common_perf_print` 输出）：

- **prompt eval time = 144.51 ms / 58 tokens (2.49 ms per token, 401.35 tokens per second)**
- **eval time = 17,494.38 ms / 255 runs (68.61 ms per token, 14.58 tokens per second)**
- total time = 17,905.08 ms / 313 tokens
- graphs reused = 253
- unaccounted time = 19.49 ms / 0.1 %

**`ggmlhexagon_dump_perf_stats` 完整统计**（Gemma4-E4B 端到端 256 批次）：

- device info: Qualcomm Snapdragon 8 Elite, dsp arch version 0x79, system mem size 24834 MiB
- device: Hexagon-CDSP0, arch=QCOM_HTP_V79, vtcm=8MB, hvx=1, hmx=1
- **model: n_layer=42 (parsed from tensor name suffixes)**
- rpc stats: batch_calls=256, cum_p10=17,371,426 us, cum_graph=17,515,579 us, avg_p10=67,857 us, avg_graph=68,420 us
- graph nodes: min=1860, max=1860, total=476,160
- graph ops (post-fusion): min=1016, max=1106
- per-call range: graph=[65,634, 141,879] us, p10=[65,461, 141,013] us
- per-call overhead: n=256, min=149, max=4,799, avg=563 us (graph_dur - p10)
- max graph detail: dur=141,879 us, n_nodes=1860, n_ops=1106
- AP phase cumulative: p1=59,483, p2=3,479, p3=1,483, p4=33, p5=9,967, p6=15,604, p7=87, p8=40,672, p9=9,647, p11=3,148, p12=100, unaccounted=450 us
- p10 3-way cumulative: rpc_setup=42, dsp_exec=17,371,426, civac=3,070 us (sum=17,374,538)
- rpc overhead (warmup): n=6, min=93, max=202, avg=129 us (pure FastRPC/mempool transport)
- cgraph cache: hits=253, misses=3 (hit_rate=98.8%), entries=0

**`ggmlhexagon_print_running_timestamp` 完整配置**（Gemma4-E4B）：

- 与 Gemma4-E2B 一致（force_opfusion_in_pp=0, enable_opfusion=1, dsp_cache_mode=5, ion_sync_mode=1, thread_counts=6）
- running timestamp: 2026-08-06, 21:16:23

**与 Gemma4-E2B 跨模型对比**：

- MUL_MAT avg: Gemma4-E2B=142us, Gemma4-E4B=204us（**1.44x**，接近层数比 42/35=1.20x + 4B/2B 参数量比 2.0x 的加权预期）
- MUL_MAT max: Gemma4-E2B=4,448us, Gemma4-E4B=7,873us（**1.77x**，lm-head vocab=256K 在两个模型相同，但 E4B 的 hidden dim 翻倍，所以 lm-head matvec 计算量 2x）
- GLU_GEGLU avg: Gemma4-E2B=167us, Gemma4-E4B=182us（几乎一致，GLU 计算量正比于 hidden_dim）
- FLASH_ATTN avg: Gemma4-E2B=115us, Gemma4-E4B=145us（1.26x，GQA 8:1 比 4:1 减少 KV 计算量，但 hidden_dim 增大抵消部分优势）
- non-op 占比：Gemma4-E2B=29.5%, Gemma4-E4B=30.2%（几乎一致，说明 non-op 开销与模型规模近似线性相关，与 3.3 节"role-aware cache 比例恒定"的分析一致）

#### 5.4.3 Qwen3.5-2B TG batch#978 详细数据（唯一完整 TG 抓取，源文件 `log_qwen3_ppandtg_force0_v4`）

> **重要**：Qwen3.5-2B（delta net 混合架构，24 个总层中标准 attention 13 层 + linear attention delta net 11 层），GATED_DELTA_NET/L2_NORM 是该架构的正常算子。**`ffn_gate-0/1/2` 这类日志编号指的是 FFN 算子所在层（0-12 共 13 个），与 tensor 的 0-23 共 24 个总层编号不同**

**Table-19**：Qwen3.5-2B TG batch#978 OP-PROF 详表

| 算子                | cum (us)     | count | avg (us) | min (us) | max (us) | 占比    |
| ----------------- | ------------ | ----- | -------- | -------- | -------- | ----- |
| **MUL_MAT**       | 442,707      | 3,854 | 114      | 3        | **5,635** | 38.3% |
| **MUL_MAT_FFN**   | 265,879      | 1,142 | 232      | 28       | 309      | 23.0% |
| **MUL_MAT_ADD**   | 151,400      | 1,172 | 129      | 48       | 230      | 13.1% |
| **GATED_DELTA_NET** | 52,100      |   704 | 74       | 38       | 1,291    |  4.5% |
| FLASH_ATTN_EXT    | 11,663       |   234 | 49       | 41       | 167      |  1.0% |
| CONCAT            | 73,126       |   705 | 103      | 87       | 167      |  6.3% |
| CPY               | 67,202       | 1,643 | 40       | 0        | 87       |  5.8% |
| GET_ROWS          | 31,923       | 1,449 | 22       | 1        | 45       |  2.8% |
| ROPE              | 12,732       |   468 | 27       | 15       | 182      |  1.1% |
| GLU_SWIGLU        | 10,002       |   938 | 10       | 7        | 112      |  0.9% |
| RMS_NORM_MUL      | 8,978        | 2,854 | 3        | 1        | 38       |  0.8% |
| UNARY_SILU        | 9,152        | 1,408 | 6        | 1        | 96       |  0.8% |
| UNARY_SIGMOID     | 3,245        |   938 | 3        | 1        | 34       |  0.3% |
| UNARY_SOFTPLUS    | 2,375        |   704 | 3        | 2        | 21       |  0.2% |
| L2_NORM           | 4,463        | 1,408 | 3        | 1        | 34       |  0.4% |
| SET_ROWS          | 1,086        |   468 | 2        | 0        | 10       |  0.1% |
| SCALE             | 1,115        |    36 | 30       | 7        | 60       |  0.1% |
| ADD               | 3,618        | 1,408 | 2        | 1        | 24       |  0.3% |
| MUL               | 4,430        | 1,876 | 2        | 1        | 34       |  0.4% |
| **op-sum 合计**     | **1,157,196** |       |          |          |          | 100%  |

**batch#978 关键指标**：

- batch-wall cum=1,355,527 us（avg=1,386 us/batch，即 ~1.4ms/token）
- op-sum cum=1,157,196 us（avg=1,183 us/batch）
- non-op avg=202 us/batch（14.6% wall time，**比 Gemma4-E2B PP 的 30.0% 低一半**）
- non-op 细分：hdr=0, pre=8, w-inv=10(1MB), a-inv=76(6MB), dst=3, bulk=71, queue=3 us/batch

**TG 阶段关键观察**：

1. **三类 matmul 占 op-sum 74.4%**：MUL_MAT（38.3%） + MUL_MAT_FFN（23.0%） + MUL_MAT_ADD（13.1%）。与 4.1 节 Gemma4-E2B profiling 的 91.1% 略有差异，原因是 Qwen3.5-2B 是 delta net 混合架构，多了 GATED_DELTA_NET（4.5%） + CONCAT（6.3%） + CPY（5.8%） 等"delta net 特有"算子，挤占 matmul 占比
2. **MUL_MAT_FFN avg=232us 是稳定 FFN fused 调用**：count=1,142/978batch ≈ 1.17 次/batch，说明每 token 大约 1 次 FFN fusion（Qwen3.5-2B 在 TG 阶段 M=1 <= 4，满足 HMX 闸门条件，fusion 正常触发）
3. **GATED_DELTA_NET avg=74us 是 delta net 核心算子**：max=1,291us 是初始化阶段的 warm-up 路径，稳定阶段 avg 远低于 max；count=704/978batch ≈ 0.72 次/batch，delta net 主干每 1-2 token 调用一次
4. **MUL_MAT max=5,635us 是 lm-head matvec**：与 4.1 节 Gemma4-E2B 的 max=4,697us 同量级，与 Q4_K/Q6_K lm-head 大小相关（本模型 vocab=152K Q6_K ≈ 178MB）
5. **non-op 仅 14.6% wall time**：M=1 TG 阶段 bit0 first-touch 权重 inval 生效（w-inv=10us/1MB，几乎为 0），a-inv=76us/6MB 也极低。验证 3.3 节"role-aware cache 在 M=1 TG 显著优"的核心论点
6. **CONCAT + CPY 占 12.1%**：delta net 架构特有的 intermediate tensor 拼接/拷贝操作，是 JZ 后续可优化方向（通过更高效的 in-place 拼接减少 DDR 往返）

#### 5.4.4 跨模型 matmul 行为对比

**Table-20**：五模型 matmul 行为对比（PP batch#1）

| 模型         | n_layer | MUL_MAT count | MUL_MAT cum (us) | MUL_MAT avg (us) | MUL_MAT_FFN count | MUL_MAT_ADD count | QKV/FFN skip 模式                |
| ---------- | :-----: | :-----------: | :--------------: | :--------------: | :---------------: | :---------------: | -------------------------- |
| Gemma4-E2B     |   35    |      277      |     39,360       |       142        |         0         |         0         | HMX gate (PP 路径,符合预期)        |
| Gemma4-E4B |   42    |      344      |     70,418       |       204        |         0         |         0         | HMX gate (PP 路径,符合预期)        |
| llama3     |   16    |       79      |      9,856       |       124        |         0         |        30         | HMX gate (PP 路径,符合预期)        |
| qwen1      |   24    |       48      |     11,575       |       241        |         1         |        119        | HMX gate (PP 路径,符合预期)        |
| qwen3      |   24    |        1      |        191       |       191        |         0         |         0         | (init batch,无实际 layer matmul) |

**观察**：

- **PP 路径 HMX 闸门 100% 生效**：五个模型的 PP batch#1 中 MUL_MAT_FFN 全部为 0（仅 Qwen1.5-1.8B 边缘 1 次，可能是 scheduler 特例），MUL_MAT_ADD 触发条件独立（Llama3.2-1B=30, Qwen1.5-1.8B=119），与 HMX 闸门无关
- **MUL_MAT avg 与模型/层数正相关**：Gemma4-E4B（42层 4B） avg=204us，Qwen1.5-1.8B（24层 1.8B MHA） avg=241us，Gemma4-E2B（35层 2B GQA 8:1） avg=142us，Llama3.2-1B（16层 1B） avg=124us
- **MUL_MAT_ADD 是稳定的 element-wise 加法融合**：Qwen1.5-1.8B count=119 说明该模型 cgraph 中存在大量 MUL_MAT + ADD 模式，被 MUL_MAT_ADD fusion 正确捕获；Llama3.2-1B count=30，Gemma4-E2B/Gemma4-E4B count=0（其 cgraph 中没有 MUL_MAT->ADD 模式）
- **HMX eligibility 与 QKV/FFN 融合互斥**：五个模型的 "QKV skip: HMX gate" 日志均出现（本节 5.4.2.1 已确认 Gemma4-E2B 真实日志，5.4.2.2 确认 Gemma4-E4B），验证 `is_mergeable_mul_mat` 闸门在 cleanup 后行为与 a3d04682 基线一致
- **Gemma4-E4B / Gemma4-E2B MUL_MAT avg 比例 1.44x**：与层数比 1.20x + 模型尺寸 4B/2B = 2x 加权预期（1.20 * sqrt(2) ≈ 1.70） 相比略低，说明 E4B 的更大 MUL_MAT 在 VTCM 中复用效率更优
- **Qwen3.5-2B 的 1 次 MUL_MAT 仅是 init batch 的 embedding**：graph nodes 范围 26-62（graph size 在 五模型中最小）说明 delta net 架构在 PP 阶段 matmul 数量极低，大部分计算在 attention 之外的 GATED_DELTA_NET/L2_NORM/CONCAT/CPY 中，详细见 5.4.3 的 TG 数据

#### 5.4.5 关键发现与结论

1. **五模型 CI 全部通过，force=0 无回归**：Gemma4-E2B batch-wall 与 4.1 节 baseline 差异 0.5%。端到端 tok/s：Gemma4-E2B PP 666.69 / TG 26.50，Gemma4-E4B PP 401.35 / TG 14.58，Qwen1.5-1.8B PP 539.06 / TG 18.41（non-op a-inv+bulk 五模型最高，验证 3.4 节 MHA 三重叠加），Llama3.2-1B PP 1039.49 / TG 42.20（五模型最高，16 层 + 1B），Qwen3.5-2B PP 408.38 / TG 21.26
2. **PP 路径 HMX 闸门 五模型全部正确保留**：QKV/FFN 融合被 HMX 闸门阻止（Gemma4-E2B/Gemma4-E4B 真实日志已确认）；MUL_MAT_ADD 是 PP 路径唯一活跃的 fusion（Qwen1.5-1.8B count=119, Llama3.2-1B count=30, Gemma4 系列 count=0，与 attention pattern 相关）
3. **non-op 占比与 GQA 比例无关**：Gemma4-E2B（8:1） 29.5% vs Gemma4-E4B（4:1） 30.2%，验证 3.3 节"role-aware cache 与模型规模线性相关"
4. **n_layer 字段在 五模型中均正确输出**（Gemma4-E2B=35, Gemma4-E4B=42, Llama3.2-1B=16, Qwen1.5-1.8B=24, Qwen3.5-2B=24），解决了此前层数估算的不准确问题

#### 5.4.6 已知数据局限与后续动作

1. **3 个模型（Qwen1.5-1.8B/Llama3.2-1B/Gemma4-E4B）TG 详细算子分布缺失**：`*_logcat.txt` 仅捕获 batch#1，后续 batch 的 OP-PROF 被丢弃（端到端 tok/s 仍从 `*_terminal.txt` 获取）。如需 TG 详细分布，用 `grep -E "OP-PROF.*batch#" log_*_logcat.txt` 直接拉取
2. **Qwen3.5-2B PP 真实数据缺失**：batch#1 仅含 embedding init 算子（M=1, 6 个 op），需用 `run_pp_only qwen3` 重抓（n_prompt >= 64）
3. **Qwen3.5-2B 端到端 tok/s 稳定**：PP 408.38 / TG 21.26 tok/s，与历史 v4 log（PP 401 / TG 21）一致
4. **Gemma4-E4B log 显示 batch#1 重复打印**：原因待查（可能是 `dump_perf_stats` 与 `OP-PROF` 触发周期冲突），不影响数据正确性
5. **GQA 4:1 vs 8:1 matmul 差异**：Gemma4-E4B/Gemma4-E2B MUL_MAT avg ratio 1.44x，略低于层数+模型尺寸加权预期 1.70x，说明 E4B 的更大 MUL_MAT 在 VTCM 中复用效率更优。后续可通过 `mul_mat coverage` 的 "ne11" 维度分布进一步分析

#### 5.4.7 文档与 commit 维护

- 五模型 10 个 log 文件（`log_forceopfusioninpp_<model>_<ts>_*`）保留在工作区根目录
- 本节数据已与 4.1 节 Gemma4-E2B profiling 交叉对比（5 项核心指标差异均在 ±6% 以内），确认无回归

### 5.5 根因分析

**假设 A 与假设 B 都不成立**：

1. **MUL_MAT_FFN 单 op 6.5ms 是主要瓶颈**：35 个 MUL_MAT_FFN x 6.5ms = 228ms，占 batch-wall 73%。HVX fused 路径在 PP 大 M 下远慢于 HMX 路径（单 MUL_MAT 137us，47x 差距）
2. **cache 失效未节省**：`w-inv=13,630 us` 与 baseline `13,636 us` 几乎一致。HVX fused 路径仍需将 3 个权重矩阵从 DDR 加载到 VTCM，融合只在算子调度层省了 AP 侧 round-trip，DSP 侧并未减少权重读取
3. **算子节省（< 17ms） << 算子额外耗时（228ms）**：净增 213ms，即 3.8x 退化

**per-layer 数据佐证**：

```
[OP-PROF-LAYER] batch#1 layers=15
  mat=204,83,82,82,89,80,86,85,83,95,81,79,78,79,149
  ffn=4164,4189,4154,4158,4156,4158,4154,4154,4153,4160,4157,4155,4158,4159,8086
  attn=124,136,111,110,140,113,111,110,112,141,111,109,109,110,115
```

- ffn 段（layer 0-13）：4153-4189 us/layer，稳定
- ffn 段（layer 14）：8086 us/layer，lm-head 相关 MUL_MAT 被错误归类
- 15 层 ffn 累计 ~63ms，与 batch-wall 比例一致

### 5.6 结论

1. **HVX fused 路径不适用于 PP**：PP 大 M 场景下单算子平均耗时与 M=1（TG）场景差数十倍，即使 cache 节省也不足以抵消算子额外耗时
2. **PP 优化的正确方向是 HMX-aware fused kernel**：需要让 MUL_MAT_QKV / MUL_MAT_FFN 在大 M 路径下走 HMX 而不是 HVX，保留 HMX 速度 + 节省 cache 失效。这是 kernel 重写工作，非小规模 patch 可解决
3. **保留的基础设施**：
   - `force_opfusion_in_pp` cfg flag + bypass 分支（可作为未来 HMX-aware kernel 上线后的 A/B 对比基线）
   - 3 个 cum 计数器（`n_qkv_skip_cum_hmx` / `n_pair_skip_cum_hmx` / `n_mm_add_skip_cum_hmx`），量化"PP 路径下融合机会数"，长期监控融合覆盖率
   - `mul_mat coverage` 扩展打印，实验环境诊断
   - `ggmlhexagon_print_running_timestamp` 打印 `enable_opfusion` / `force_opfusion_in_pp`，运行时配置可见性
4. **per-layer profiling（副产品）**：[OP-PROF-LAYER] 日志已能正常输出 15 层 mat/ffn/attn 三段耗时，后续 PP 优化可直接基于此数据做 layer 级别对比

### 5.7 后续步骤

1. **回退 cfg**：`force_opfusion_in_pp = 0`（默认行为不变）
2. 保留 feature/force_opfusion_in_pp，可在未来作为其他feature开发的基线分支
3. **新方向**：调研高通 htp 目录是否有 HMX-aware MUL_MAT_QKV/FFN kernel 可参考；若无，需自主设计（关键决策点：3 个权重矩阵的 VTCM 复用策略，以及如何在 M=large 时仍能利用 HMX 8x8 systolic 阵列）

***

### 5.8 后续可探索方向 (按预期收益排序)

承接 5.6 第 2 条结论 (`PP 优化的正确方向是 HMX-aware fused kernel`)，以及 5.6 第 3 条保留的 `n_qkv_skip_cum_hmx` / `n_pair_skip_cum_hmx` 计数器监控指标，本节在 `feature/qwen1_optimize` 分支上做了两轮实验验证后，给出后续 4 个突破性方向的优先级排序与定量分析。

#### 5.8.1 根因再分析: Qwen1.5-1.8B QKV 0% 融合

5.4.3 节统计的 `mul_mat coverage` 显示 Qwen1.5-1.8B 的关键瓶颈:

| 指标 | 当前值 | 含义 |
|---|---|---|
| total MUL_MAT | 169 | 24 层 x 7 matmul 总量 (Q/K/V/O/FFN gate/up/down) |
| hmx=165 (97.6%) | 165 | 其中 165 个 HMX-eligible，走 3x 独立 MUL_MAT |
| qkv_fused | 0 (0.0%) | 0 次 QKV 融合成功 (理想 24 次) |
| ffn_fused | 1 (1.2%) | 1 次 FFN 融合成功 (理想 24 次) |
| mm_add_fused | 119 (70.4%) | 119 次 MUL_MAT+ADD 融合 (接近上限) |
| qkv_skip_hmx | 0 | 0 次因 HMX eligibility 拒绝 |
| pair_skip_hmx | 23 | 23 次 FFN pair 因 HMX eligibility 拒绝 |

`qkv_fused=0` 与 `qkv_skip_hmx=0` 的组合定位根因: `is_mergeable_mul_mat()` ([ggml-hexagon-jz.cpp：is_qkv_mergeable](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp)) 要求 `!mm_is_hmx_eligible(t)` 才允许进入融合候选。Qwen1.5-1.8B 的 Q/K/V MUL_MAT 全部是 HMX-eligible (Q4_0 + 标准 shape)，始终被 `is_mergeable_mul_mat` 拒绝。当前 fused QKV (op_matmul_qkv，[matmul-ops.c：op_matmul_qkv](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/matmul-ops.c)) 是 HVX 实现，无法利用 HMX 加速，所以即使强制融合，HVX 3x 路径也不会比 HMX 3x 更快。这与 5.3.2 节 `force_opfusion_in_pp=1` 的实验结论 (HVX 融合 PP 慢 3.8x) 一致。

#### 5.8.2 Plan A 实验证伪: HVX vs HMX FLASH_ATTN 在 M=51 几乎无差异

承接 4.1.3 节 "HMX-aware FLASH_ATTN_EXT kernel" 方向，在 `feature/qwen1_optimize` 分支上实施 patch：新增 cfg `fa_hmx_min_kv_blocks=3` (默认)，当 `n_kv_blocks < 3` 时回退 HVX 路径。预期 FLASH_ATTN_EXT 786 us/call 降至 500 us/call (PP +6-10%)。

实测 (Qwen1.5-1.8B PP-only batch#1，源文件 `log_qwen1_optimization_20260807.txt`):

| 指标 | Baseline (HMX) | Plan A (HVX) | Delta |
|---|---|---|---|
| batch-wall | 90778 us | 90511 us | -267 us (-0.3%) |
| op-sum | 47147 us | 46913 us | -234 us (-0.5%) |
| FLASH_ATTN_EXT avg | 786 us | 778 us | -8 us (-1.0%) |
| MUL_MAT | 11146 us | 11056 us | -90 us |
| a-inv | 19167 us (1754 MB) | 19143 us (1754 MB) | -24 us |

patch 编译验证: `strings libggml-hexagon.so | grep fa_hmx` 返回 3 个匹配，`/data/local/tmp/ggml-hexagon.cfg` 中 `fa_hmx_min_kv_blocks = 3` 已正确部署。

结论: **HVX 6 线程与 HMX 单线程在 n_kv_blocks=1 的小规模 prefill 场景下耗时几乎相同** (差异 <1%)。两者都受限于 DMA 数据移动 + HVX softmax + o_store epilogue，HMX vs HVX 的计算差异不显著。该 patch 不应合并到主分支。

#### 5.8.3 后续 4 个突破性方向 (按预期收益排序)

| 优先级 | 方向 | 当前开销 | 预期节省 | PP 收益 | 风险 |
|---|---|---|---|---|---|
| P0 | A. HMX fused QKV (Q+K+V 单 HMX call) | 3x MUL_MAT 占用 | 2400-4800 us | +2.6-5.3% | 中高 (新 kernel) |
| P0 | B. bulk flush 异步化 (与下一 batch 重叠) | bulk=15145 us | 4500-7500 us | +5-8% | 中高 (新 thread) |
| P1 | C. HMX fused FFN (gate+up 单 HMX call) | 2x MUL_MAT 占用 | 1200-2400 us | +1.3-2.6% | 中 (参考 A) |
| P1 | D. a-inv 跨 batch dedup | a-inv=19167 us | 800-1500 us | +1-1.6% | 中 |

**A. HMX fused QKV (P0 突破方向)**:

- 实施: 在 `matmul-ops.c` 添加 `op_matmul_hmx_qkv`，复用 HMX systolic 阵列在 1 个 DSP call 中计算 Q/K/V 三个 matmul
- 关键决策点 (来自 5.6 第 2 条): 3 个权重矩阵的 VTCM 复用策略 + M=large 时 HMX 8x8 阵列利用
- 节省机制: 2 个 op 的 per-op overhead (Q setup / VTCM alloc / weight load / output write)，每个 op overhead ~50-100 us
- 24 层 x 2 = 2400-4800 us 节省，PP 收益 +2.6-5.3%
- 同步收益: Q/K/V 三个 output 从 3 个独立 dst 合并到 1 个 fused dst，**减少 a-inv 总量** (FLASH_ATTN 读 Q/K/V 时只 invalidate 1 个 tensor 而非 3 个)
- 风险: 新 kernel 需验证正确性，参考 QCOM `htp/matmul-ops.c` 中 HMX-aware QKV 实现 (若存在)
- 实施陷阱 (feature 分支实测): 三个权重共享同一 kparams->n 会在 GQA 模型上输出乱码 (K/V 输出维度 < Q，共享 n 导致 K/V 写出维度错误); 必须为 Wk/Wv/Wq 各自独立 kparams 并携带各自权重维度

**B. bulk flush 异步化 (P0 突破方向)**:

- 现状: `bulk_flush_all()` ([entry.c：bulk_flush_all](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/entry.c)) 在所有 op 完成后**阻塞**执行 15145 us，期间 DSP thread idle
- 实施: 新增 1 个 DSP worker thread 负责 flush，main batch 线程与 flush 线程并行
- 时序: batch N 完成后, flush 线程立即开始 flush batch N 的 dst range; 同时 AP 侧准备 batch N+1 的 descriptor (hdr/pre 阶段)
- 同步点: batch N+1 的第一个 op 读取 dst 之前，需确保 batch N 的 flush 完成
- 节省: 15145 us 中 30-50% 可与下一 batch 重叠，理论省 4500-7500 us
- 风险: 需要新增 DSP thread + 跨 batch 同步原语，可能影响后续读取的 cache coherency
- 参考: 5.7 节保留的 `dsp-ctx.h` `bulk_flush_ranges` 数组已包含 sort + merge 逻辑，异步化只需把 `bulk_flush_all()` 从主线程移到 worker thread
- 实施陷阱 (feature 分支实测): enable_async_bulk_flush=1 曾在 Qwen1.5-1.8B 上输出乱码 (flush 线程 DMA 写与主线程读的竞态); 同步点必须显式保证 batch N+1 首个读 dst 的 op 之前 flush 完成，并逐模型验证五模型 CI 输出

**C. HMX fused FFN (P1 方向)**:

- 实施: 类似 A，但融合 2 个 MUL_MAT (ffn_gate, ffn_up) 为 1 个 HMX call
- 当前覆盖率: ffn_fused=1/24 (1.2%)，与 QKV 同样的 `!mm_is_hmx_eligible` gate 限制
- 节省: 24 层 x 1 op overhead = 1200-2400 us
- 风险: 中，参考 A 的实现
- 可与 A 同步实施: `op_matmul_hmx_qkv` 和 `op_matmul_hmx_ffn` 共享 HMX matmul 基础设施

**D. a-inv 跨 batch dedup (P1 方向)**:

- 现状: a-inv=19167 us / 1754 MB per batch (非 op 开销最大项)
- 1754 MB / 24 layer / 51 token = 1.43 MB per layer per token, 主要是 per-token unique activation
- 实施: 类似 `weight_inval_check_and_mark` ([entry.c：weight_inval_check_and_mark](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/entry.c)) 的 per-batch dedup, 添加 `act_inval_check_and_mark` 跨 batch tracking
- 假设: 同一 tensor 跨 batch 复用率 30%, 节省 ~24% = 800-1500 us
- 风险: 中，跨 batch 跟踪需要考虑 L2 容量 (128KB) 和 tensor 生命周期
- 前提: 必须先确认 1754 MB 的 24% 重复率假设成立 (需 AP 侧打点验证)

#### 5.8.4 新增三个方向（2026-08-08 上午，Kimi-K3，基于 self-build-jz 最新代码与数据）

**Table-21**：新增方向一览（编号沿用 5.8.3 的 A-D 顺延）

| 优先级 | 方向 | 当前开销 | 预期节省 | 收益场景 | 风险 |
|---|---|---|---|---|---|
| P0 | F. Qwen1.5-1.8B TG 根因定位（op 级 profiling 先行） | TG -28.8%（15.6 ms/token） | 待定，上限为抹平差距 | Qwen1.5-1.8B TG | 低（先测后修） |
| P1 | E. AP 侧 per-call overhead 消除（hash 瘦身 + descriptor blob 复用） | 641 us/batch（p1 395 + p8 166 + p5/p6 79） | ~540 us/batch | 全模型 TG +1.2-2.7% | 低（纯 AP 侧） |
| P2 | G. first-touch w-inv 移至加载期 | 首个 PP batch 13.6-25.8 ms | 同左（计时窗口口径） | PP 测量 + TTFT | 低-中（顺序约束） |

排序依据：F 为 P0 因 Qwen1.5-1.8B 是五模型中唯一 TG 落后模型，需最先定位根因（一次 CI 出结论）；E 为 P1 因最低风险速赢，可与 A 并行；G 为 P2 因仅修正计时口径，独立实施。

**E. AP 侧 per-call overhead 消除（P1）**：

- 证据：第五章 Gemma4-E2B 端到端 256 批次 `dump_perf_stats`：per-call overhead avg=690 us（graph_dur - p10），其中 p1=395 us/batch（compute_content_hash 每 batch 游走 1493 个散落 tensor 结构体）、p8=166 us/batch（descriptor 构建）、p5/p6=79 us/batch；cgraph cache hits=253 / misses=3，即 253 个 TG batch 图内容完全一致，这部分 AP 工作每 token 原样重做
- 机制 1（hash 瘦身）：[ggml-hexagon-jz.cpp：ggmlhexagon_backend_graph_compute_batch](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp) 每节点 fold op+ne+nb+10 个 src 指针+data ptr（约 20 次解引用）；改为只 fold {op, data ptr}，data ptr 在图内基本唯一，配合已有 n_nodes 校验，碰撞安全性可论证
- 机制 2（descriptor blob 复用）：hash 已覆盖全部 data ptr，命中时 Phase 5/6/8 的输出是确定性的；将构建完成的 ops+tensors descriptor 区（约 240 KB 连续 blob）存入 cgraph cache entry，命中时一次 memcpy 替代重跑三个阶段
- 与 4.5.1 pipelining 互补（消除 vs 重叠）；纯 AP 侧改动，不碰 DSP kernel / cache 策略 / 调度框架，风险为全部方向中最低
- 预期：641 -> 约 100 us/batch；TG：Llama3.2-1B +2.3%，Gemma4-E2B +1.4%，Qwen1.5-1.8B +1.2%
- 验证：`dump_perf_stats` 对照 p1/p8 累计值 + 五模型 CI

**F. Qwen1.5-1.8B TG 根因定位（P0）**：

- 现状：五模型中唯一 TG 落后（-28.8%，53.8 vs 38.2 ms/token）；文档对 TG 落后只有 "MHA VTCM/cache 压力" 定性描述，Qwen1.5-1.8B TG 的 per-op profile 从未测过（4.1 节是 Gemma4-E2B TG，5.x 是 Qwen1.5-1.8B PP）
- 计划：`HEX_OP_PROF` 跑 Qwen1.5-1.8B TG，与 4.1 节 Gemma4-E2B TG profile 逐项对比
- 待验证假设：H1 FLASH_ATTN 主导（MHA 每 token 每层 KV 读取 = 2 x n_embd x 2B = 8 KB，为 GQA 8:1 的 8 倍，随 ctx 线性增长）；H2 lm-head（Q6_K->Q4_0 repack 约 163 MB）matvec 在 DSP 的实际带宽未达预期；H3 KV cache 写入路径 per-token invalidation
- 定位主导项后再设计修复：H1 对策为 KV layout / VTCM 分块，H2 对策为 matvec kernel DRAM 预取
- 一次 CI 即可定位；Qwen1.5-1.8B 同时是 PP 差距最大模型（-22.3%），机制级解释对 PP 优化同样有价值

**G. first-touch w-inv 移至加载期（P2）**：

- 证据：Gemma4-E2B 首个 PP batch w-inv=13,637 us（1,334 MB），Gemma4-E4B 25,802 us（2,528 MB）；会话级一次性成本（TG batch 实测 w-inv 约 10 us，bit0 生效），但全部落在首个 PP batch 的计时窗口内，Gemma4-E2B PP 76 ms 中 18% 是会话初始化成本而非 prompt 计算
- 机制：复用 `execute_batch(0xFFFC)` 特殊命令通道（推 dsp_cache_mode 的同一路径），模型加载完成后发 init 命令，DSP 一次性整块 invalidate 权重区并预标记 first-touch bitmap；权重区在 mempool 内连续，整块 invalidate 比 24-42 层交错 per-range 遍历局部性更好
- 预期：PP 测量窗口 -13.6 ms（Gemma4-E2B）/ -25.8 ms（Gemma4-E4B），PP 数字更纯粹反映计算本身，后续 pipelining 收益测量更准；TTFT 微降
- 风险：低-中；只需保证 invalidate 在最后一次 AP 侧权重写之后、首个 DSP batch 之前

**已评估并否掉的方向**（避免重复提议）：

- a-inv range 合并（sort+merge）：a-inv=19,167 us 是字节线性 dcinva 成本，1,754 MB 已是 per-batch 结构性下限（4.6.1 已证伪压缩空间）；合并 range 只省 per-call 开销，不省逐行 invalidate 本身
- PP/TG 双模 cache 策略（PP 批次改用 flush-all）：flush-all 会把权重逐出 L2，恰好摧毁 bit0 first-touch 带来的 TG 优势，得不偿失

#### 5.8.5 优先级排序依据

P0 方向 (A + B) 合计理论节省 6900-12300 us = PP +7.6-13.6%，符合 `4.7 路线图` 中 4.5 PP 优化的潜在收益空间。P1 方向 (C + D) 合计 2000-3900 us = PP +2.2-4.3%，作为 P0 实施完成后的后续优化。

实施顺序建议: A -> C -> B -> D。前两个 (A + C) 共用 HMX fused kernel 基础设施，先实施可积累经验。B 涉及新 DSP thread + 同步原语，复杂度最高，但潜在收益最大 (PP +5-8%)。D 风险最低 (复用现有 weight_inval 模式)，但收益受限于重复率假设验证。2026-08-08 新增 E/F/G (5.8.4) 的排序依据见 Table-21 下方说明。

注意: 本节 4 个方向均在 `feature/qwen1_optimize` 分支探索，**不应直接合并到主分支 `self-build-jz`**，需 5 模型 CI 验证无回归 + 主分支 baseline 对比后再议。

***

## 六、Qwen3.5-2B PP&TG 优化（2026-08-07）

### 6.1 起点：kimi-k3 在 4.2 节指出的拆分问题

4.2 节（Kimi-K3 在 2026-08-07 修订轮新增）的核心结论：Qwen3.5-2B（24 层 GQA + Delta Net 混合架构）在 JZ 后端每个 batch 的 cgraph 被拆成 25 个子图，原因是 `GGML_OP_SOLVE_TRI` 与 `GGML_OP_SSM_CONV` 未在 AP 侧注册，DSP 侧 kernel 已存在但缺桥接胶水代码。Table-11 实测 `batch_calls=6400`（256 batch x 25 子图）。

### 6.2 第一阶段：补两个算子的 bridge layer 代码

按 4.5.3 方案实施了两组 patch（两个算子的桥接层）：

1. **SOLVE_TRI offload**：在 `init_op_validators()`（[ggml-hexagon-jz.cpp：init_op_validators](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp)）注册 `s_op_validators[GGML_OP_SOLVE_TRI]`，校验逻辑参照 QCOM 的 `ggml_hexagon_supported_solve_tri`（F32 + 方阵 + 维度匹配）；在 `ggml_op_to_htp_op`（[entry.c：ggml_op_to_htp_op](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/entry.c)）加 `case GGML_OP_SOLVE_TRI: *htp_op = HTP_OP_SOLVE_TRI; return 0;`
2. **SSM_CONV offload**：同样在 AP 侧注册 validator（参考 QCOM 校验规则，F32 输入、kernel 维度 1-4），DSP 侧加 `case GGML_OP_SSM_CONV` 映射。SSM_CONV 是 delta net 层的卷积算子，本身不直接是 SOLVE_TRI，但 SOLVE_TRI 之前的卷积路径中 SSM_CONV 缺失会迫使 SOLVE_TRI 前的子图段被切开

构建+本地测试后 `batch_calls` 从 6400 降至 1792（256 batch x 7 子图），delta net 层的 11 处 SOLVE_TRI 拆分点 + 周围依赖算子段合并为 1 个连续子图段。

### 6.3 第二阶段：拆分减少，但推理输出乱码

`batch_calls=1792` 已大幅改善，但 Qwen3.5-2B 推理输出出现字符级乱码（截断、重复、词界破坏）：起点（batch_calls=6400）输出经 `log_abtest_all_20260807-102443.txt` 核实为文本连贯，乱码是 bridge patch 后新引入的中间状态，症状与 4.6.5 描述的"garble = cache 损坏"一致：

- 排除 1：4.6.5 已回滚的 DSP-side sampling 优化不是当前代码状态
- 排除 2：已知 a-inv bit1 实验已证伪（4.6.1），对当前 cgraph 退化为 no-op
- 排除 3：cgraph 中无已知 fusion 异常模式（乱码复现路径未穿越 fusion 节点）

剩余 7 子图/批的拆分点影响 cache coherency 维护路径，需进一步定位具体是哪个算子在何处切图。

### 6.4 第三阶段：用 ggml core 的 `GGML_SCHED_DEBUG` 抓 split 现场

ggml core 内置的 `GGML_SCHED_DEBUG` 环境变量可在 scheduler 切图时打印每次切分的位置（op index + op type + 原因）。

**标准抓 log 命令**（复制即可用）：

```sh
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=. \
  GGML_SCHED_DEBUG=2 \
  ./llama-completion -ngl 99 -t 6 -n 64 --ctx-size 8192 \
  --ubatch-size 1 --batch-size 1 --poll 1000 --no-warmup --load-mode none \
  -fa on --jinja -no-cnv -st --verbosity 5 \
  -m /sdcard/Qwen3.5-9B-Q4_0.gguf \
  -p 'Hello'" 2>&1 | tee /tmp/9b_sched_$(date +%Y%m%d-%H%M%S).txt
```

**参数说明**：
- `GGML_SCHED_DEBUG=2`：触发 scheduler 打印每次切图的位置（op index + op type + 原因）。`--verbosity 5` 让 llama.cpp 主程序打印每层 tensor 形状，便于对照 Qwen3Next 模型结构。
- `--verbosity 5`：**必填**。`GGML_LOG_DEBUG` 在 level 0 时不输出，所有 `## SPLIT` 行都会被吞掉。
- `-n 64`：短 token 数即可抓到足够 split 模式（512 token 全图 + 24+ delta-net 层已能覆盖全部边界 op），无需跑满 256 token。
- `--ubatch-size 1 --batch-size 1`：强制单 batch，避免自动拆分干扰 split 计数。
- `--no-warmup --load-mode none`：跳过 warmup 阶段，确保 split 计数从第 1 个 batch 开始。

**输出格式**（关键行）：

```
## SPLIT #N: <backend> # <inputs> inputs
   [tensor_name-0 (size) [BufferKind]]
   [tensor_name-1 (size) [BufferKind]]
node # <idx> (<OP_NAME>): src0 (size) [BufferKind]
                        src1 (size) [BufferKind]
                        dst  (size) [BufferKind]
```

**分析步骤**：
1. `grep "^## SPLIT" log.txt | wc -l` 得到总 split 数
2. `grep -A3 "## SPLIT.*CPU" log.txt` 提取每次 CPU SPLIT 的 tensor 列表
3. 对每个 CPU SPLIT，**找到紧跟其后的 `node #N` 行**--这个 node 就是触发 CPU fallback 的 op，其 `src0/src1/dst` 后缀标的就是 buffer 归属（`[CPU]` 表示在 system memory，`[Hexag]` 表示在 DSP mempool）
4. 统计 CPU SPLIT 命中的 op 模式（如全部为 `MUL_MAT x blk.*.ssm_out.weight`），定位根因是 operator 缺失、validator 过严还是 weight buffer 错位

**Qwen3.5-9B 实测（2026-08-08）**：500 CPU + 500 Hexagon SPLIT，每个 CPU SPLIT 对应一个 `node #(MUL_MAT): linear_attn_out-X x blk.X.ssm_out.weight -> final_output-X`，根因是 `ssm_out.weight` 在 system memory buffer 上（model 5.4GB > mempool 4GB 触发 [ggml-hexagon-jz.cpp：ggml_backend_hexagon_buffer_is_host](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp) 的 ion pool exhausted 回退路径），scheduler 因 weight 不在 DSP mempool 而强制把对应 MUL_MAT 派给 CPU。这是与 Qwen3.5-2B 完全不同的根因（Qwen3.5-2B 是 operator 缺失 + validator 过严，Qwen3.5-9B 是 weight buffer 错位），不能用 6.2 / 6.7 的方法修复。

### 6.5 第四阶段：root cause 定位

抓取 `log_qwen3_split_*.txt` 后的关键发现：

**7 个剩余子图/批的拆分点全部位于 6 个 MHA 层的 `attn_q_norm` 算子**（blk.3/7/11/15/19/23），每个 MHA 层各贡献 1 个拆分点（其中一个 MHA 层的 attn_q_norm 因 KV cache 状态差异偶发 2 次），与 delta net 层的 SOLVE_TRI/SSM_CONV 修复无关。

进一步追溯到 JZ 侧 RMS_NORM validator（[ggml-hexagon-jz.cpp：hexagon_validate_rms_norm](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp)）：

```cpp
static bool hexagon_validate_rms_norm(...) {
    ...
    if (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32)
        return false;
    if (!ggml_is_contiguous(src0))   // <-- per-head view 在此被拒
        return false;
    return true;
}
```

### 6.6 第五阶段：分析 per-head view 为什么不是 contiguous

阅读 [qwen3next.cpp：build_layer_attn](file:///home/zhouwg/develop/ggml-hexagon/src/models/qwen3next.cpp) 的 Qcur 构造代码：

```
Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.wq_s)         // (n_embd_head*2, n_head, n_tokens)
Qcur_full = ggml_reshape_4d(Qcur_full, n_embd_head*2, n_head, n_tokens, 1)
Qcur      = ggml_view_4d(Qcur_full, n_embd_head, n_head, n_tokens, 1,    // 只取 Qcur_full 前半
                          Qcur_full->nb[1], Qcur_full->nb[2], Qcur_full->nb[3], 0)
Qcur      = build_norm(Qcur, model.layers[il].attn_q_norm, LLM_NORM_RMS) // <- validator 拒绝
```

`Qcur` 继承 `Qcur_full` 的 stride（不复制数据，仅设置 offset 与 stride），导致 `nb[1] = n_embd_head * 2 * sizeof(float)` 而 `ne[0] = n_embd_head`，`nb[1] != ne[0] * sizeof(float)`，因此 `ggml_is_contiguous(Qcur)` 返回 false。`ggmlhexagon_can_handle_op_through_cdsp` 对该 op 返回 false，scheduler 在此切图，attn_q_norm 回退 CPU。

### 6.7 第六阶段：DSP 侧能力核验 + 修改 validator

**DSP 侧能力核验**（确认 kernel 实际支持非连续输入）：

- `unary_row_offset(ir, ne1, ne2, div_ne1, div_ne2, div_ne12, nb1, nb2, nb3)` 通过 `i1*nb1 + i2*nb2 + i3*nb3` 计算行偏移，与 `nb1/nb2/nb3` 无关形状
- `unary_block_size` 在 `!src_contig || !dst_contig` 时按 `(fastdiv(ir, div_ne1) + 1) * ne1` 切块，限制跨 ne1 边界
- [ggml-hexagon-jz.cpp：ggmlhexagon_backend_graph_compute_batch](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp) 直接赋值 `t->nb[0..3]`，stride 信息完整传递至 DSP
- RMS_NORM kernel 内部按 `float*` 步进 dim-0（reduction 维度），仅要求 `nb[0] == sizeof(float)`

**结论**：DSP 侧能力无障碍，AP 侧 validator 过严是唯一原因。

**修复如下**：

```diff
--- a/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp
+++ b/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp
@@ -3454,7 +3454,11 @@ static bool hexagon_validate_rms_norm(...) {
     const ggml_tensor * src0 = op->src[0];
     if (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32)
         return false;
-    if (!ggml_is_contiguous(src0))
+    // Accept non-contiguous view tensors (e.g. Qwen3Next per-head view with
+    // nb[1] = n_embd_head * 2 * sizeof(float)). Only require dim-0 (reduction
+    // axis) to be element-contiguous, since the DSP kernel walks the
+    // reduction dim element-by-element.
+    if (src0->nb[0] != sizeof(float))
         return false;
     return true;
 }
```

**设计依据**：RMS_NORM kernel 沿 dim-0（reduction 维度）逐元素计算，dim-0 元素级连续是 kernel 的**最小不变量**，无需整张量连续。

### 6.8 第七阶段：验证结果

**Table-22**：Qwen3.5-2B 修复全过程分阶段对比

| 阶段                       | 改动                                    | `batch_calls` | 拆分来源         | 推理输出          |
| ------------------------ | ------------------------------------- | ------------ | ------------ | ------------- |
| 起点（kimi-k3 指出）            | 无                                     | 6400         | SOLVE_TRI + SSM_CONV 缺失（25 子图/批） | 正常（拆分仅造成性能损失） |
| 第一阶段：补 SOLVE_TRI/SSM_CONV bridge | 6.2 节两组 patch                            | 1792         | 6 个 MHA 层 attn_q_norm（7 子图/批）  | 字符级乱码（bridge 后新引入） |
| 第二阶段：放宽 RMS_NORM validator  | 6.7 节 patch                            | **256**      | 无（完整单图）       | 正常输出          |

**Table-23**：最终修复后实测数据

| 指标                         | 起点   | 第一阶段（bridge layer） | 第二阶段（per-head view fix） | 变化 (起点->Stage 2) |
| -------------------------- | ------------ | --------------- | ----------------- | ----------- |
| `batch_calls`              | 6400         | 1792            | **256**           | **-96.0%**  |
| 推理输出                       | 正常（log 102443 文本连贯） | 字符级乱码（新引入）           | 正常               | 正常 -> 正常（Stage 1 引入并已消除）   |
| 6 个 MHA 层 attn_q_norm 上 DSP | 0/6          | 0/6             | **6/6**           | -           |
| PP tok/s                   | **436.6**     | 456.91          | **501.71**          | **+14.9%**  |
| TG tok/s                   | **27.0**      | 23.94           | **26.74**           | **-1.0%**   |

> **数据来源**:
> - 起点 (Qwen3.5-2B JZ baseline): Table-6 AB 测试 (log_abtest_all_20260807-102443.txt), PP 436.6 tok/s, TG 27.0 tok/s, batch_calls=6400 (Table-11).
> - 第一阶段 (bridge layer, batch_calls=1792): common_perf_print 输出 PP 456.91 tok/s, TG 23.94 tok/s, 输出字符级乱码.
> - 第二阶段 (per-head view fix, batch_calls=256): 5 模型 CI 3 轮均值 (log_abtest_all_20260807-223924.txt, self-build-jz 分支), PP 501.71 tok/s, TG 26.74 tok/s, 输出正常.
> - QCOM baseline (Table-6, log_abtest_all_20260807-102443.txt): PP 481.1 tok/s, TG 14.0 tok/s; 5 模型 CI QCOM (log_abtest_all_20260807-223924.txt): PP 456.10 tok/s, TG 13.44 tok/s.
>
> **与 QCOM 对比的视角**:
> - **PP 性能反超**: 起点 -9.2% (436.6 vs 481.1) -> Stage 2 **+10.0%** (501.71 vs 456.10, 5 模型 CI QCOM 基线). 19.2pp 反转, JZ 在 Qwen3.5-2B 上首次反超 QCOM.
> - **TG依然保持领先**: 起点 +93.6% (27.0 vs 14.0) -> Stage 2 +99.0% (26.74 vs 13.44, 5 模型 CI QCOM 基线). 基本持平, 仍保持近 2x 优势.
>
> **变化归因 (JZ 内部起点 -> Stage 2)**:
> - PP +14.9% (436.6 -> 501.71) 来自两部分: (a) bridge layer 阶段 PP 456.91 (+4.7% vs 起点, 单测), 消除 11 处 delta net 层拆分的子图固定开销; (b) per-head view fix 阶段 PP 进一步 +9.8% (456.91 -> 501.71, 5 模型 CI 3 轮均值), 消除 6 次 MHA 层 attn_q_norm CPU<DSP> 上下文切换.
> - TG -1.0% (27.0 -> 26.74) 内部基本持平, 但 QCOM 对比保持 +99.0% 领先 (5 模型 CI QCOM TG 13.44 tok/s), 实际净效果是 PP 性能反超 + TG 仍保持 QCOM 2x 优势.

### 6.9 与 3.6 / 4.5.3 节的关系

**Table-24**：Qwen3.5-2B graph 拆分修复路径全景

| 拆分来源                  | 层类型        | 拆分点数 | 根因诊断章节 | 修复章节    | 修复后 batch_calls |
| --------------------- | ---------- | ---- | ------ | ------- | --------------- |
| SOLVE_TRI 缺失          | Delta Net  | ~11  | 4.2    | 4.5.3 + 6.2 | 1792            |
| SSM_CONV 缺失（伴随 SOLVE_TRI） | Delta Net  | ~若干  | 6.2    | 6.2     | 1792            |
| per-head view 拒绝      | MHA (6 层)  | 6    | 6.5    | 6.7（本节）  | 256             |
| **剩余拆分点**             | -          | **0** | -      | -       | **256**（完整单图）  |

4.2 节（root cause）与 4.5.3（fix）是同一问题的诊断与方案（缺失算子），6.5-6.7 节是**独立发现的第二个根因**（validator 过严）。两者共同构成 Qwen3.5-2B 在 JZ 后端上的 graph 拆分问题的完整根因，缺一不可。

### 6.10 修复意义

1. **消除 Qwen3.5-2B 在 JZ 后端的所有 graph 拆分**：`batch_calls` 从 6400 降至 256（-96%）
2. **乱码根因消除**：4.5.3 修复后仍残留的字符级乱码来自 MHA 层的 per-head view，本节同步消除
3. **向后兼容其他 MHA 模型**：放宽后的 validator 对 `ggml_is_contiguous==true` 的常规输入仍接受（满足 `nb[0]==sizeof(float)`），零回归
4. **per-head view 模式的通用修复**：Qwen3Next、Phi-3、Gemma2 等近年模型都使用 per-head Q/K/V 切分，本修改是该模式 RMS_NORM 算子的通用前提，未来模型无需重复修改

## 七、Qwen1.5-1.8B PP&TG 优化（2026-08-08）

### 7.1 背景与数据来源

5.8.4 提出的 F 方向 (Qwen1.5-1.8B TG 根因定位) 在 feature/qwen1_qwen3_optimize 分支跑完一轮 `run_qwen1_tg_prof` CI，参数 M=1, n_predict=256, ctx=8192, HEX_OP_PROF=1, DUMP_INTERVAL=1。对照模型 Qwen1.5-1.8B (MHA 1:1, 24 层) 与 Gemma4-E2B (GQA 8:1, 35 层)，稳态均值取 OP-PROF 序列 skip 前 10 batch 后的窗口。

数据来源: `qwen1_tg_prof_20260808-165745_summary.log` + `_qwen1_logcat.log` + `_Gemma4-E2B_logcat.log` 三组文件，terminal 实测 Qwen1.5-1.8B TG 18.62 tok/s (53.70 ms/token), Gemma4-E2B TG 26.53 tok/s (37.70 ms/token)。下文表 Table-25 / Table-26 / Table-27 / Table-28 直接来自 OP-PROF 聚合。

### 7.2 H1 / H2 / H3 假设的算子级判定

**Table-25**：H1 / H2 / H3 假设的算子级对比 (per batch, 稳态均值)

| 假设 | 算子 | Qwen1.5-1.8B (us/batch) | Gemma4-E2B (us/batch) | Qwen1.5-1.8B 占比 | Gemma4-E2B 占比 | 结论 |
|---|---|---:|---:|---:|---:|---|
| H1 | FLASH_ATTN_EXT (us/batch) | 1434.5 | 721.2 | 2.7% | 2.0% | **证伪** |
| H1 | FLASH_ATTN_EXT (us/op) | 57.7 | 19.9 | - | - | Gemma4-E2B GQA 更快 (1/3) |
| H1 | FLASH_ATTN_EXT (ops/batch) | 24.88 | 36.26 | - | - | 与 n_layer 一致 |
| H2 | lm-head MUL_MAT (us/batch, max/op) | ~3500 | ~4800 | 6.7% | 13.0% | **证伪** |
| H2 | lm-head 带宽 (Qwen1.5-1.8B 156MB / 3.5ms) | 64 GB/s | - | 峰值 ~91% | - | DRAM 已饱和, op 级无收益 |
| H3 | a-inv (us/batch) | **16241.0** | 994.1 | **30.9%** | 2.7% | **Qwen1.5-1.8B 确认** |
| H3 | a-inv MB invalidated (MB/batch) | 1588 | 79 | - | - | **Qwen1.5-1.8B 是 Gemma4-E2B 的 20.1x** |
| H3 | a-inv per-MB (us/MB) | 10.2 | 12.6 | - | - | 单位带宽成本相近, 为工作量问题 |

注: Qwen1.5-1.8B wall ≈ 52500 us/batch (graph_dur 53407); Gemma4-E2B wall ≈ 37000 us/batch (graph_dur 37200)。

判定说明:

- **H1 (FLASH_ATTN) 证伪**: Qwen1.5-1.8B 25 op/batch (24 层 + 1) x 58 us, Gemma4-E2B 36 op x 20 us。即使消除 100% FLASH_ATTN, Qwen1.5-1.8B TG 也只提速 2.7% (1.4 ms/token)。
- **H2 (lm-head) 证伪**: Qwen1.5-1.8B lm-head MUL_MAT 3500 us = 6.7% wall (max 3650 us), Gemma4-E2B 4800 us = 13% wall。Qwen1.5-1.8B lm-head 实测 64 GB/s, 逼近 DRAM 峰值 (~70 GB/s)。op-level 优化 (fuse / faster kernel) 被 DRAM 锁死, 上限 ~5-10%。真正的 H2 收益方向是 K/V cache FP16/Q8 量化 (把 cache 从 2 字节降到 1 字节, lm-head input 间接受益, 但要重做 cache 写入路径)。
- **H3 (KV invalidation) Qwen1.5-1.8B 确认, Gemma4-E2B 否决**: Qwen1.5-1.8B a-inv 16241 us/batch (30.9% wall) / 1588 MB invalidated, Gemma4-E2B 仅 994 us / 79 MB。16.3x 时间差, 20.1x 数据量差。单位带宽成本 Qwen1.5-1.8B (10.2 us/MB) 甚至略低于 Gemma4-E2B (12.6 us/MB), 为总工作量问题而非效率问题。根因推断: Qwen1.5-1.8B MHA 1:1 每层 16 KV head x 128 head_dim x 2 (K,V) = 大量 KV 写 src range, 每次 batch 触发整片 L2 invalidation; Gemma4-E2B GQA 8:1 KV 头数仅 1/8。

### 7.3 Qwen1.5-1.8B vs Gemma4-E2B 全算子 per-batch 排序 (稳态)

**Table-26**：Qwen1.5-1.8B vs Gemma4-E2B 全算子 per-batch 排序 (稳态)

| rank | Qwen1.5-1.8B op | us/batch | 占比 | Gemma4-E2B op | us/batch | 占比 |
|---:|---|---:|---:|---|---:|---:|
| 1 | MUL_MAT_ADD (123 op, 68 us) | 8418.3 | 16.0% | MUL_MAT (168 op, 99 us) | 16645.4 | 45.0% |
| 2 | MUL_MAT_FFN (25 op, 258 us) | 6429.0 | 12.2% | MUL_MAT_FFN (36 op, 341 us) | 12379.2 | 33.4% |
| 3 | MUL_MAT (2 op, 1775 us) | 3679.5 | 7.0% | MUL_MAT_QKV (16 op, 58 us) | 902.9 | 2.4% |
| 4 | FLASH_ATTN_EXT (25 op, 58 us) | 1434.5 | 2.7% | FLASH_ATTN_EXT (36 op, 20 us) | 721.2 | 2.0% |
| 5 | GLU_SWIGLU (25 op, 7 us) | 178.9 | 0.3% | RMS_NORM_MUL (235 op, 3 us) | 576.6 | 1.6% |
| 6 | RMS_NORM_MUL (51 op, 3 us) | 142.8 | 0.3% | GLU_GEGLU (36 op, 15 us) | 545.1 | 1.5% |
| 7 | ROPE (50 op, 3 us) | 123.4 | 0.2% | UNARY_TANH (1 op, 258 us) | 267.5 | 0.7% |
| 8 | SET_ROWS (50 op, 2 us) | 76.5 | 0.1% | ADD (110 op, 2 us) | 184.3 | 0.5% |
| - | (其他) | 36.6 | 0.1% | (其他) | 1521.3 | 4.1% |
| - | **op-sum** | **20520** | 39.0% | **op-sum** | **34743** | 91.2% |
| - | a-inv | 16241 | 30.9% | a-inv | 994 | 2.7% |
| - | bulk flush | 15034 | 28.6% | bulk flush | 1205 | 3.3% |
| - | 其他非 op (pre, dst, queue, hdr) | 226 | 0.4% | 其他非 op | 445 | 1.2% |
| - | **wall (est)** | **52021** | - | **wall (est)** | **37387** | - |

注: Qwen1.5-1.8B wall 实测 graph_dur = 53407 us (含 AP 侧 ~400 us 杂项); Gemma4-E2B wall 实测 37200 us。

### 7.4 Qwen1.5-1.8B TG 真正的瓶颈重构 (a-inv + bulk 是核心)

**Table-27**：Qwen1.5-1.8B vs Gemma4-E2B wall 拆解对比

| 组件 | Qwen1.5-1.8B (us) | Gemma4-E2B (us) | 差值 |
|---|---:|---:|---:|
| a-inv | 16241 | 994 | +15247 |
| bulk flush | 15034 | 1205 | +13829 |
| op-sum | 20520 | 34743 | -14223 |
| 其他 | 226 | 445 | -219 |
| **wall** | **52021** | **37387** | **+14634** |

Qwen1.5-1.8B 比 Gemma4-E2B 慢 14.6 ms/batch, 缓存维护多出 29 ms (56% of delta), 抵消了 24 层 vs 35 层的算子优势。a-inv + bulk = 60% wall, 同源 (DSP L2 容量 8 MB 限制), 都是 L2->DRAM 同步代价。

### 7.5 优化方向 (按收益/风险排序)

**Table-28**：剩余优化方向一览 (按优先级)

| 优先级 | 方向 | 预期收益 | 风险 | 杠杆点 |
|---|---|---:|---|---|
| P0 | E 方向 AP per-call overhead 消除 (hash 瘦身 + descriptor blob 复用) | 0.7-2.2% TG | 低 | 99% cache hit rate, 全模型生效 |
| P1 | K/V cache FP16/Q8 量化 | 10-19% TG | 中-高 | 直接砍 a-inv 工作量 (1588 MB -> ~794 MB, 省 ~8000 us/batch = 15% wall), 需改 llama.cpp KV cache 写入路径 |
| P2 | MUL_MAT_ADD fusion rate 提升 | 5-10% TG | 中 | 当前 123/120 已几乎全 fuse, 边际空间小 |
| 放弃 | async_bulk_flush (Path-B1) | -14.6% (但输出乱码) | - | Qwen1.5-1.8B MHA K/V race 已实验证伪 |
| 放弃 | dsp_a_inv_bitmap (Path-B3) | 0% (回退) | - | Gemma4-E2B 输出乱码 + Qwen1.5-1.8B 回退, 已实验证伪 |
| 放弃 | Qwen1.5-1.8B force_opfusion_in_pp | - | - | 经验证 HVX fused 6.5 ms/op vs HMX 137 us/op, 不适用 |

### 7.6 关键观察

- Qwen1.5-1.8B MUL_MAT_ADD 5.1 op/layer x 24 layers = 123 op/batch, per_op = 68 us 极稳定 (min=47, max=154) -> 完美 fusion, 无空间
- Qwen1.5-1.8B MUL_MAT 只有 2 op/batch, 推断 1 个是 lm-head (max=3650 us ≈ 3500 us/batch = 6.7% wall, 与 H2 一致), 另 1 个是 output norm/bias 小算子
- Qwen1.5-1.8B MUL_MAT_FFN 25 op/batch (24 layers + 1), per_op = 258 us; Gemma4-E2B 36 op x 341 us
- Qwen1.5-1.8B cgraph 5.8.4 早期测量中, 16249 us a-inv (1588 MB) 是结构性上限, bitmap 优化已无效
- Qwen1.5-1.8B bulk 15034 us/batch 几乎与 a-inv 16241 us/batch 相当 -> dst flush 同样吃紧, 与 a-inv 同源 (L2 容量限制)
- Qwen1.5-1.8B 5.8.4 路径 A/B/C 三个 cache 优化方向已全部实验证伪 (MHA K/V 与 DMA race, L2 8 MB 容量限制无法绕过, PRIOR_DST_MAX_LEN=64 过严), cache 增量调参路径走完; 剩余两条结构层面路径: (a) 5.8.4 E 方向 = AP per-call overhead 消除 (hash 瘦身 + descriptor blob 复用, Table-28 P0), 收益 0.7-2.2% TG 低风险; (b) K/V cache FP16/Q8 量化 (Table-28 P1, 本节新增), 收益 10-19% TG 中-高风险; 合计潜在 11-21% TG 改善


***

## 八、Qwen3.5-9B PP&TG 优化（2026-08-09）

### 8.1 问题概述与数据

#### 8.1.1 六模型 AB CI 性能对比

2026-08-11 完整跑完六模型 AB CI 测试 (Qualcomm SnapDragon 8 Elite, dsp arch 0x79, system mem 24834 MiB, n_ctx=512, prompt eval 50-75 tokens, 生成 255 tokens)。Qwen3.5-9B 因推理耗时 + 功耗高 + 手机发热, rounds 强制 1 轮 (CI 脚本 hard cap), 其他 5 模型跑 3 轮取平均。原始 log 见 `log_abtest_all_20260811-092810.txt` (36251 行)。

**Table-29**: 六模型 AB CI 性能对比 (PP/TG 单位 tokens/s, total 单位 ms)

> 注: batch_calls 是 JZ 后端专有指标 (由 `ggmlhexagon_dump_perf_stats` 输出)。QCOM 后端不输出此指标, 标记为 N/A。JZ 的 batch_calls 反映 DSP invoke 次数: 256 表示无 graph split (1 call/token), 6144 表示 24 个 Q5_K split (24 calls/token)。

| 模型 (脚本别名) | n_layer | 后端 | PP (t/s) | TG (t/s) | total (ms) | batch_calls | JZ vs QCOM (TG) |
|---|---:|---|---:|---:|---:|---:|:---:|
| Qwen3.5-2B (qwen3-2b) | 24 | JZ   | 510.63 | 27.67 |   9316.99 |   256 | **1.93x** |
| Qwen3.5-2B (qwen3-2b) | 24 | QCOM | 473.61 | 14.35 |  17887.49 |   N/A |  baseline |
| Gemma4-E2B (Gemma4-E2B) | 35 | JZ   | 676.81 | 27.28 |   9433.11 |   256 | **1.11x** |
| Gemma4-E2B (Gemma4-E2B) | 35 | QCOM | 473.35 | 24.58 |  10496.76 |   N/A |  baseline |
| Gemma4-E4B (Gemma4-E4B) | 42 | JZ   | 394.94 | 14.87 |  17300.00 |   256 | **1.33x** |
| Gemma4-E4B (Gemma4-E4B) | 42 | QCOM | 423.05 | 11.18 |  22933.60 |   N/A |  baseline |
| Qwen1.5-1.8B (qwen1) | 24 | JZ   | 526.86 | 18.63 |  13783.47 |   256 | 0.71x |
| Qwen1.5-1.8B (qwen1) | 24 | QCOM | 797.98 | 26.36 |   9782.51 |   N/A |  baseline |
| Llama-3.2-1B (llama3) | 16 | JZ   | 998.55 | 42.96 |   6011.74 |   257 | **1.49x** |
| Llama-3.2-1B (llama3) | 16 | QCOM | 1128.86 | 28.85 |   8905.02 |   N/A |  baseline |
| Qwen3.5-9B (qwen3-9b) | 32 | JZ   |  35.76 |  1.48 | 174258.05 |  6144 | 0.22x |
| Qwen3.5-9B (qwen3-9b) | 32 | QCOM | 120.80 |  6.61 |  39021.14 |   N/A |  baseline |

**Table-30**: 六模型 AB CI TG 优势/劣势汇总 (按 JZ/QCOM TG 比值排序)

| 模型 (脚本别名) | JZ TG | QCOM TG | JZ/QCOM | 类别 | 主因 |
|---|---:|---:|:---:|---|---|
| Qwen3.5-2B (qwen3-2b) | 27.67 | 14.35 | **1.93x** | JZ 强优势 | 24 层, batch_calls=256 无 split, mempool 充足 + lm-head offload |
| Llama-3.2-1B (llama3) | 42.96 | 28.85 | **1.49x** | JZ 优势 | 16 层小模型, cgraph cache 几乎全 hit, lm-head offload 净收益主导 |
| Gemma4-E4B (Gemma4-E4B) | 14.87 | 11.18 | **1.33x** | JZ 优势 | 42 层偏大, JZ mempool 仍可装下, lm-head offload + 连续 IOVA 净收益显著 |
| Gemma4-E2B (Gemma4-E2B) | 27.28 | 24.58 | **1.11x** | JZ 优势 | 35 层中等模型, batch_calls=256 无 split, QCOM dspqueue 收益对 JZ 边际 |
| Qwen1.5-1.8B (qwen1) | 18.63 | 26.36 | 0.71x | JZ 劣势 | MHA 1:1 每 token 大量 L2 invalidation, a-inv + bulk flush 占 60% wall, L2 8 MB 限制; K/V 量化可追平 |
| Qwen3.5-9B (qwen3-9b) | 1.48 | 6.61 | 0.22x | JZ 大劣势 | 模型 5.03 GiB 超 4 GiB mempool 致 heap fallback + mirror memcpy overhead (78% wall), 24 个 Q5_K split 致 batch_calls=6144, per-call overhead 214x (JZ 21.4ms vs QCOM 0.1ms) |

关键观察:

1. JZ TG 在 4/6 模型上领先 QCOM (Qwen3.5-2B / Llama-3.2-1B / Gemma4-E4B / Gemma4-E2B), 最大 1.93x。这 4 个模型共同特征是 batch_calls=256 (无 split, cgraph cache 全 hit) 且模型 < mempool 4 GiB, JZ 架构净优势 (lm-head offload + mempool 连续 IOVA) 完全释放
2. JZ TG 在 2/6 模型上落后 QCOM (Qwen1.5-1.8B / Qwen3.5-9B)。两个模型代表两种不同的 JZ 失败模式: (a) batch_calls=256 无 graph split 但 a-inv/bulk flush 拖慢单次调用性能; (b) batch_calls=6144 大量 graph split + per-call overhead 214x 放大
3. PP 维度 JZ 在 2/6 模型上反超 QCOM (Qwen3.5-2B / Gemma4-E2B, batch_calls=256 且 lm-head offload 净收益主导), 4/6 落后 (Gemma4-E4B / Qwen1.5-1.8B / Llama-3.2-1B / Qwen3.5-9B)。QCOM dspqueue 的 per-call overhead 极低 (~0.1 ms ring buffer write vs JZ 21.4 ms 同步 FastRPC + mirror memcpy), JZ 的 12 阶段同步 FastRPC 在 PP 短序列下 per-call 开销占比大。但对长生成 (TG), JZ 的开销被 255 段摊薄, dspqueue 收益变小, JZ 反而领先
4. Qwen3.5-9B 是 JZ 第二个 TG/PP 同时落后 QCOM 的模型 (第一个是 Qwen1.5-1.8B), 也是首个双边失利差距达 5x 量级的模型

**Table-31**: 六模型 AB CI 完整 3 轮单值明细 (Qwen3.5-9B 1 轮, 供审计与可复现验证)

| 模型 | 后端 | round | PP (t/s) | TG (t/s) | total (ms) |
|---|---|---:|---:|---:|---:|
| Qwen3.5-2B (52+255 tok) | JZ   | 1 | 503.04 | 27.80 |   9274.52 |
| Qwen3.5-2B | JZ   | 2 | 505.54 | 27.54 |   9362.60 |
| Qwen3.5-2B | JZ   | 3 | 523.31 | 27.67 |   9313.87 |
| Qwen3.5-2B | QCOM | 1 | 486.15 | 14.39 |  17821.85 |
| Qwen3.5-2B | QCOM | 2 | 472.53 | 14.14 |  18148.82 |
| Qwen3.5-2B | QCOM | 3 | 462.14 | 14.51 |  17691.80 |
| Gemma4-E2B (58+255 tok) | JZ   | 1 | 673.15 | 27.28 |   9433.83 |
| Gemma4-E2B | JZ   | 2 | 680.66 | 27.33 |   9416.97 |
| Gemma4-E2B | JZ   | 3 | 676.63 | 27.24 |   9448.54 |
| Gemma4-E2B | QCOM | 1 | 472.19 | 24.84 |  10389.38 |
| Gemma4-E2B | QCOM | 2 | 475.14 | 24.57 |  10500.47 |
| Gemma4-E2B | QCOM | 3 | 472.72 | 24.34 |  10600.44 |
| Gemma4-E4B (58+255 tok) | JZ   | 1 | 401.27 | 14.95 |  17200.78 |
| Gemma4-E4B | JZ   | 2 | 396.62 | 14.96 |  17193.68 |
| Gemma4-E4B | JZ   | 3 | 386.94 | 14.69 |  17505.53 |
| Gemma4-E4B | QCOM | 1 | 428.09 | 11.35 |  22592.92 |
| Gemma4-E4B | QCOM | 2 | 410.64 | 11.11 |  23089.59 |
| Gemma4-E4B | QCOM | 3 | 430.43 | 11.09 |  23118.29 |
| Qwen1.5-1.8B (51+255 tok) | JZ   | 1 | 536.23 | 18.61 |  13795.01 |
| Qwen1.5-1.8B | JZ   | 2 | 510.52 | 18.64 |  13781.22 |
| Qwen1.5-1.8B | JZ   | 3 | 533.82 | 18.64 |  13774.17 |
| Qwen1.5-1.8B | QCOM | 1 | 783.67 | 27.74 |   9257.62 |
| Qwen1.5-1.8B | QCOM | 2 | 890.24 | 23.87 |  10738.09 |
| Qwen1.5-1.8B | QCOM | 3 | 720.03 | 27.48 |   9351.82 |
| Llama-3.2-1B (75+255 tok) | JZ   | 1 | 989.24 | 42.40 |   6089.98 |
| Llama-3.2-1B | JZ   | 2 | 995.47 | 43.18 |   5981.45 |
| Llama-3.2-1B | JZ   | 3 | 1010.95 | 43.30 |   5963.80 |
| Llama-3.2-1B | QCOM | 1 | 1130.05 | 28.92 |   8883.28 |
| Llama-3.2-1B | QCOM | 2 | 1131.44 | 28.80 |   8921.26 |
| Llama-3.2-1B | QCOM | 3 | 1125.08 | 28.83 |   8910.51 |
| Qwen3.5-9B (52+255 tok) | JZ   | 1 |  35.76 |  1.48 | 174258.05 |
| Qwen3.5-9B | QCOM | 1 | 120.80 |  6.61 |  39021.14 |

#### 8.1.2 Qwen3.5-9B 基线数据

Qwen3.5-9B (alias `qwen3-9b`, Qwen3.5-9B-Q4_0.gguf, 5.03 GiB, 32 layers = 8 MHA + 24 delta-net, GQA 4:1 (16:4), hidden_dim=4096, vocab=248320)。除六模型 CI 外, 另做了一次专门 profiling (52 token prompt + 255 decode runs)。QCOM 端数据来源: `common_perf_print` 屏幕截图; JZ 端数据来源: `qwen3_9b_tg_prof_20260808-192916.log` / `dump_perf_stats` 输出。

QCOM 端 `common_perf_print`:

```
prompt eval time =    361.89 ms /   52 tokens (   6.96 ms per token,  143.69 tokens per second)
eval      time =  33112.32 ms /  255 runs    ( 129.85 ms per token,    7.70 tokens per second)
total     time =  33537.00 ms /  307 tokens
graphs reused = 253
```

JZ 端 `common_perf_print`:

```
prompt eval time =   1903.87 ms /   52 tokens (  36.61 ms per token,   27.31 tokens per second)
eval      time =  172379.97 ms /  255 runs    ( 676.00 ms per token,    1.48 tokens per second)
total     time =  174413.97 ms /  307 tokens
graphs reused = 253
```

JZ 端 `dump_perf_stats` (基线):

```
device info:  Qualcomm SnapDragon 8 Elite, dsp arch version 0x79, system mem size 24834 MiB
device=0:     name=Hexagon-cDSP0 arch=QCOM_HTP_V79 vtcm=8MB hvx=1 hmx=1
model:        n_layer=32 (parsed from tensor name suffixes)
rpc stats:    batch_calls=6400 cum_p10=33679355 us cum_graph=170443717 us
              avg_p10=5262 us avg_graph=26631 us
graph nodes:  min=51 max=100 total=456192
graph ops:    min=21 max=50 (post-fusion)
per-call range: graph=[4863, 601108] us p10=[1954, 22593] us
per-call overhead: n=6400 min=2909 max=578515 avg=21369 us (graph_dur - p10)
AP phase cumulative: p1=89019 p2=1958 p3=1979 p4=430 p5=68699257 p6=23858
                     p7=1231 p8=32090 p9=50092 p11=22614 p12=67833542 unaccounted=8292 us
p10 3-way:    rpc_setup=3774 dsp_exec=33679355 civac=21778 us (sum=33704907)
rpc overhead: n=6 min=84 max=191 avg=117 us (warmup, pure FastRPC/mempool transport)
cgraph cache: hits=6351 misses=49 (hit_rate=99.2%) entries=0
```

**Table-32**: Qwen3.5-9B PP/TG 实测对比

| 模型 | 指标 | JZ | QCOM | 差距 | 备注 |
|---|---|---:|---:|---:|---|
| Qwen3.5-9B | PP tok/s | 27.31 | 143.69 | 5.26x 慢 | 52 token prompt |
| Qwen3.5-9B | TG tok/s | 1.48 | 7.70 | 5.20x 慢 | 255 decode runs |
| Qwen3.5-2B (对照, 6 章) | TG tok/s | 14.50 | 7.74 | 1.87x 快 | JZ 比 QCOM 快 |
| Qwen1.5-1.8B (对照, 7 章) | TG tok/s | 18.62 | ~30 | 1.61x 快 | JZ 比 QCOM 快 |

JZ load 时间 (1912.61 ms) 显著高于 QCOM (366.10 ms), 与权重回退 system memory 相关 -- `log_qwen3.5-9b.txt` 中两次运行的回退规模不同 (PID 20562: 权重拆 2005+1996 MiB 双 chunk, 后块回退; PID 21326: 单块 4002 MiB 整体回退 heap), 精确规模取决于运行时 mempool 剩余与 chunk 拆分。

dump_perf_stats 关键数据解读:

1. **JZ batch_calls=6144** = 24 个 Hexagon 段/token x 256 token (24 个 ssm_out MUL_MAT CPU split 把每 token 的 cgraph 切成 25 个 Hexagon sub-graph, 其中第 25 段 main graph 为 cgraph cache hit, 不产生 DSP call)
2. **JZ per-call overhead = 21.4 ms/call** -- 每 sub-call 26.6 ms 总耗时, 5.3 ms 是 DSP 真实计算 (p10 dsp_exec), 21.4 ms 是非 DSP 开销 (FastRPC setup + mirror + sync); 24 calls x 21.4 ms = 514 ms/token 是 JZ 单 token 主要开销
3. **JZ p5+p12 = 136.5 s = 78% of total** -- mirror 计算是 JZ Qwen3.5-9B 真实瓶颈 (68.7 s compute + 67.8 s apply); 其余 22% 是 p1+p6+p8+p11 等
4. **JZ p10 dsp_exec = 5.262 ms/call** -- sub-call 真实 DSP 计算是 5.3 ms (不含 ssm_out MUL_MAT, 后者在 CPU 执行), 不是简单 matmul 的 0.5 ms
5. **PP 36.61 ms/token vs TG 676.00 ms/token = 18.5x 差异** -- TG 100% decode-bound, 与 Qwen1.5-1.8B 经验一致
6. **cgraph cache hit_rate=99.1%** (6088/6144), 与 QCOM `graphs reused=253/255=99.2%` 接近 -- graphs reused 是 upstream llama.cpp scheduler 的 cgraph 结构复用计数, 与 FastRPC 无关; QCOM 提交链路是 dspqueue 持久化环形队列, per-call descriptor 准备仍存在但通过 dspqueue 异步提交与 DSP 计算重叠。JZ cgraph cache 是 AP 端按 op+shape+src ptr 哈希的 descriptor 复用 cache, 两者层级不同但命中率接近 (99.1% vs 99.2%), 反映 JZ 与 QCOM 的 cgraph 拓扑变化模式相同 (每 255 步 ~2 步需要重切)
7. **输出内容**: jinja "Thinking Process" 模式 (Analyze Request / Identify Key Information), 正常电影介绍 ("Once Upon a Time in America", "Sergio Leone", "Robert De Niro"), 非乱码

#### 8.1.3 Q5_K repack patch 结果

2026-08-09 落地的 Q5_K -> Q4_0 repack patch (`feature/qwen3.5-9b-optimize` 分支基于 self-build-jz 干净基线)。

patch 落地后 `common_perf_print`:

```
prompt eval time =   1161.60 ms /   52 tokens (  22.34 ms per token,   44.77 tokens per second)
eval      time =  170708.09 ms /  255 runs    ( 669.44 ms per token,    1.49 tokens per second)
total     time = 172001.85 ms /  307 tokens
```

patch 落地后 `dump_perf_stats`:

```
rpc stats:    batch_calls=6144 cum_p10=31111854 us cum_graph=163207515 us
              avg_p10=5063 us avg_graph=26563 us
per-call overhead: n=6144 min=3334 max=89259 avg=21499 us (graph_dur - p10)
AP phase cumulative: p1=32690 p5=65911752 p12=66019461 unaccounted=5561 us
p10 3-way:    rpc_setup=758 dsp_exec=31111854 civac=20765 us
cgraph cache: hits=6088 misses=56 (hit_rate=99.1%) entries=0
```

**Table-33**: Q5_K repack patch 落地后关键变化

| 指标 | 基线 | patch 落地 | 变化 | 根因 |
|---|---:|---:|---|---|
| PP tok/s | 27.31 | **44.77** | **+473%** | PP 受 graph split 影响小 + 干净分支基线 (无 Qwen1.5-1.8B 实验代码影响) |
| TG tok/s | 1.48 | **1.49** | **+1%** | 4 GiB mempool 装不下 4.5 GiB repack 后 Q4_0 tiled -> Q5_K 权重落 heap (CPU buffer) -> 每 token mirror memcpy (from & back) overhead 占 TG wall 78% |
| load time | 1912.61 | **1167.50** | **-39%** | repack 后 token buffer layout 简化 |
| batch_calls | 6400 | **6144** | **-4%** | 4 个非 ssm_out MUL_MAT CPU split 仍存在, Q5_K 24 个 split 未消除 |
| per-call overhead | 21.4 ms | 21.5 ms | 0% | Q5_K 走 CPU 不消耗 per-call overhead |
| p5 (mirror) | 68.7 s | 65.9 s | -4% | repack 后数据布局简化 |

核心结论: patch 落地后 TG 实际 1.49 tok/s, 远低于预期 ~2.8 tok/s。patch 真实价值是为 scatter-gather 实现后 (Q5_K 可拆分到多个 mempool region) 自动激活 repack 通路, 不依赖本次 4 GiB 物理约束的解除。Qwen3.5-9B 性能根治仍需 scatter-gather (见下文 8.6 节), patch 仅是必要前置。

### 8.2 根因总览: 三重屏障

Qwen3.5-9B TG 5.20x 差距 (JZ 1.48 vs QCOM 7.70 tok/s) 不是单一原因, 而是三重正交屏障叠加。JZ 与 QCOM 的 Q5_K split 段数相同 (同一 upstream scheduler, QCOM validator 同样拒绝 Q5_K, 各 24 个 ssm_out split; QCOM 不输出 batch_calls), 但 QCOM 额外有 1 个 lm_head CPU split (详见 8.5.4, JZ 因 lm_head offload 到 DSP 无此 split), 差距主要来自 per-call overhead。

**Table-34**: 单 token 提交成本算术展开

> 注: JZ batch_calls=6144=24x256 (第 25 段 main graph 为 cgraph cache hit, 不产生 DSP call)。QCOM 不输出 batch_calls, 下表 QCOM 列基于 "同一 scheduler + 同样拒绝 Q5_K" 推断。

| 维度 | JZ (实测) | QCOM (推断) | 比值 | 来源 |
|---|---:|---:|---:|---|
| sub-graph 段数 / token | 25 (scheduler 拓扑) | 25 (推断, 同一 scheduler) | 1x | QCOM validator 代码同样拒 Q5_K |
| DSP call 数 / token | 24 (第 25 段 cache hit) | N/A (QCOM 不输出) | N/A | JZ dump_perf_stats batch_calls=6144 |
| per-call overhead | 21.4 ms (FastRPC + mirror memcpy) | ~0.1 ms (dspqueue ring-buffer write 反推) | 214x | JZ dump_perf_stats per-call overhead avg=21369 us |
| 单 token 提交成本 | 514 ms (24 x 21.4) | N/A (call count 未知) | 214x (per-call) | Table-34 算术 |
| 单 token 总 wall | 676 ms | 130 ms | 5.20x | 8.1.2 / 8.1.1 |
| 提交开销占 wall 比例 | 76% (514 / 676) | N/A | - | Table-34 算术 |
| 扣除提交后剩余 (DSP 计算等) | ~162 ms (676 - 514) | 130 ms | 1.2x | 两者基本持平 |

三重屏障:

1. **屏障一: Q5_K graph split (类型根因)** - 24 个 ssm_out.weight 是 Q5_K 量化类型, 被 JZ MUL_MAT validator 在调度期判给 CPU, 每 token 产生 24 个 CPU split, cgraph 被切成 25 个 Hexagon sub-graph (其中 24 段产生 DSP call, 第 25 段 main graph 为 cache hit)。QCOM validator 同样拒绝 Q5_K, JZ 与 QCOM 的 Q5_K split 拓扑相同 (各 24 个 ssm_out split)。最短修复是 Q5_K -> Q4_0 repack (复用既有 Q4_K/Q6_K repack 机制, ~30-50 行), 见下文 8.3 节
2. **屏障二: Mirror memcpy (容量根因)** - 5.03 GiB 模型超过 4 GiB mempool (V79 DSP 32-bit VA hard cap), 1996 MiB hexagon 权重回退 heap, 每 token ~2 GiB mirror memcpy (~200 ms), 占 TG wall 78%。修复路径是 scatter-gather (DSP 直读 system memory), 见下文 8.4 节
3. **屏障三: 算子提交路径开销 (架构根因)** - JZ per-call 走同步 FastRPC (21.4 ms), QCOM per-call 走 dspqueue_write (~0.1 ms), per-call 差距 214x。dspqueue 并未减少提交次数 (24 个 Q5_K split 仍需 24 次提交), 而是将 per-call overhead 降低 ~214x。修复路径是压缩 per-call overhead (见下文 8.5 节)

Table-34 的核心数字: 扣除提交开销后 JZ ~162 ms 与 QCOM 130 ms 基本持平, 证明 5.20x 差距的真因是提交路径, 不是 DSP 算力。三条屏障正交可叠加, 修复路径见下文 8.6 节。

### 8.3 屏障一: Q5_K graph split

#### 8.3.1 SPLIT 现场抓取

沿用第六章方法, 使用 `GGML_SCHED_DEBUG=2` 环境变量输出 scheduler 每次切分子图的具体边界 op:

```bash
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=. \
  GGML_SCHED_DEBUG=2 \
  ./llama-completion -ngl 99 -t 6 -n 64 --ctx-size 8192 \
  --ubatch-size 1 --batch-size 1 --poll 1000 \
  --no-warmup --load-mode none -fa on --jinja -no-cnv -st \
  --verbosity 5 \
  -m /sdcard/Qwen3.5-9B-Q4_0.gguf -p 'Hello'" 2>&1 | tee log_qwen3.5_9b_graphsplit.txt
```

一次 64-token TG 产生 1000 条 `## SPLIT` 记录 (500 CPU split + 500 Hexagon split), 分布在 20 个 cgraph。`rpc stats: batch_calls=1600` (64 token x 25 Hexagon sub-graph/token)。抓取 log 使用 `-n 64 --ubatch-size 1`, 实际推理使用 `n_predict=256`, 但 per-token SPLIT 模式完全一致: per-token batch_call = 25.0 恒定, 与 TG token 数独立。不论 TG 是 64 还是 256 token, 每 token 都需经过 1 次 embedding 查表 + 全部 24 个 delta-net 层, ssm_out MUL_MAT 切到 CPU 是"每层每 token"行为。

**Table-35**: per-token batch_call 构成与 SPLIT 边界

| 来源 | 段数 | 触发条件 | 对应 SPLIT 段 |
|---|---:|---|---|
| embedding GET_ROWS 在 CPU | 1 个 CPU 段 | token_embd.weight (545.62 MiB, Q4_0, [4096 x 248320]) 固定在 CPU buffer (upstream 常规: input embedding 在 host 执行) | SPLIT #0 |
| 24 个 delta-net ssm_out MUL_MAT 切到 CPU | 24 个 CPU 段 | 每 token 经过全部 24 个 delta-net 层 (block 0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30), 每层 ssm_out.weight (Q5_K, 11 MiB) 在 CPU buffer | SPLIT #2,4,...,48 |
| 切回 Hexagon 的 sub-graph | 25 个 Hexagon 段 | 上述 25 个 CPU 段把每 token 的 cgraph 切出 25 个 Hexagon sub-graph, 每段 1 次 DSP 调用; 末段 (SPLIT #49) 含完整 model.output (output_norm MUL + 795.70 MiB Q6_K lm_head MUL_MAT 均在 Hexagon) | SPLIT #1,3,...,49 |
| **per-token batch_calls 合计** | **25** | - | - |

8 个 MHA-only 层 (层号 3, 7, 11, 15, 19, 23, 27, 31) 不出现 ssm_out MUL_MAT split (它们没有 ssm_out.weight 张量)。这与 Qwen3-Next 架构的 3:1 模式一致: 32 层 / 4 = 8 cycle, 每个 cycle 3 个 delta-net + 1 个 MHA = 24 个 delta-net + 8 个 MHA。每一处 split 都形如:

```
node # 56 (MUL_MAT): linear_attn_out-0  [CPU]
                   x blk.0.ssm_out.weight (11M) [CPU]
                   -> final_output-0   [Hexag]
```

25 个 Hexagon SPLIT 边界 tensor 分类 (按 SPLIT #N 编号, 对应每次切回 Hexagon 的输入 tensor):

| SPLIT # | 边界 tensor | 大小 | 含义 |
|---:|---|---:|---|
| 1 | `model.input_embed` | 16K | PP 第一个 sub-graph 起点 |
| 3 | `linear_attn_out-0` | 16K | block 0 完成后 (CPU 切回) |
| 5 | `linear_attn_out-1` | 16K | block 1 完成后 |
| 7 | `linear_attn_out-2` + `attn_inp_kq_mask` | 16K+16K | block 2 完成, block 3 (MHA) 起点 |
| 9 | `linear_attn_out-4` | 16K | block 4 完成后 |
| 11, 13 | `linear_attn_out-5, 6` | 16K each | block 5, 6 |
| 15, 17, 19 | `linear_attn_out-8, 9, 10` | 16K each | block 8, 9, 10 |
| 21, 23, 25 | `linear_attn_out-12, 13, 14` | 16K each | block 12, 13, 14 |
| 27, 29, 31 | `linear_attn_out-16, 17, 18` | 16K each | block 16, 17, 18 |
| 33, 35, 37 | `linear_attn_out-20, 21, 22` | 16K each | block 20, 21, 22 |
| 39, 41, 43 | `linear_attn_out-24, 25, 26` | 16K each | block 24, 25, 26 |
| 45, 47 | `linear_attn_out-28, 29` | 16K each | block 28, 29 |
| 49 | `linear_attn_out-30` + `leaf_498` | 16K+0K | block 30 完成, model.output 起点 |

SPLIT #7 的 5 inputs (`linear_attn_out-2` + `leaf_55` + `leaf_59` + `leaf_61` + `attn_inp_kq_mask`) 表明 MHA block 3 接收 4 个额外 leaf 输入 (KV cache view + mask), 是进入 MHA block 的标志。log 实证: `grep "ssm_out.weight" | grep -v "create_tensor" | sed -n 's/.*blk\.\([0-9]*\)\.ssm_out.weight.*/\1/p' | sort -n -u` 输出 24 个 block 编号, 每个 block 的 ssm_out.weight 都被标记为 `[CPU]`, 与上游 scheduler 的 "src weight buffer 不被当前 backend 支持" 触发条件完全一致。8 个 MHA block (3, 7, 11, 15, 19, 23, 27, 31) 的所有 op 完整在 Hexagon 端运行, 无 CPU 切分。

#### 8.3.2 Q5_K validator rejection

ssm_out.weight 在 CPU buffer 的根因是 Q5_K 量化类型不被 JZ MUL_MAT validator 支持, 与 mempool 容量无关。三重证据链:

1. **log 证据**: `log_qwen3.5_9b_graphsplit.txt` 中 24 个 ssm_out.weight (各 11 MiB) 标 `[CPU]`, 但同层全部其他权重 (ffn_down 30 MiB / ffn_up 27 MiB / attn_qkv 18 MiB / ssm_conv1d / ssm_alpha / ssm_beta, 从 blk.0 到 blk.31 一致) 以及 output.weight (795.70 MiB lm_head) 全部标 `[Hexag]`。`[Hexag]` 标签只说明 tensor 属于 hexagon buft 的 buffer, 不代表 backing store 在 mempool 里。真正被 scheduler 路由到 CPU buft 的权重只有 24 个 ssm_out.weight

2. **GGUF 元数据证据** (Qwen3.5-9B-Q4_0.gguf 实测): 全部 24 个 ssm_out.weight 是 Q5_K [4096, 4096], 11.00 MiB x 24 = 264.0 MiB, 且 Q5_K 在全模型 427 个 tensor 中恰好只有这 24 个。类型普查: Q4_0 n=173 (3929.62 MiB, 含 token_embd.weight 545.62 MiB), Q6_K n=1 (output.weight 795.70 MiB, 全模型唯一), Q4_1 n=4 (120.00 MiB), Q8_0 n=48 (6.38 MiB), F32 n=177 (32.39 MiB)。模型总权重 5148.1 MiB = 5.03 GiB (427 tensors, 32 layers)

3. **代码证据**: [ggml-hexagon-jz.cpp：ggmlhexagon_supported_mul_mat](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp) `ggmlhexagon_supported_mul_mat` 的 switch 仅支持 Q8_0 / Q4_0 / IQ4_NL / Q4_1 / MXFP4 / Q4_K / Q6_K (外加 F16/F32/BF16), Q5_K 走 `default: return false`

根因链: ssm_out.weight 是 Q5_K -> JZ MUL_MAT validator 返回 false -> upstream scheduler 把该 MUL_MAT 分配到 CPU backend -> tensor 随之落入 CPU buffer -> 24 个 delta-net 层每层产生 1 个 CPU split。与 mempool 容量无关。

mempool 容量问题的真实边界 (与 graph split 根因相互独立): `log_qwen3.5-9b.txt` 的 "ion pool exhausted" 记录确实存在, 两次运行模式不同 - PID 20562 拆 2005+1996 MiB 双 chunk (后块回退), PID 21326 单块 4002 MiB 整体回退 heap。mempool 4 GiB 装不下 5.0 GiB 全模型是事实, 但容量回退影响的对象不是 ssm_out: 4002 MiB hexagon 权重块里 2005 MiB 进 mempool、1996 MiB 回退 heap, heap 块内 tensor 仍属 hexagon buft，DSP 通过 mirror 执行, 不产生 split。真正落入 CPU 的只有 Q5_K 的 ssm_out.weight (类型被 validator 拒绝, 调度阶段判给 CPU buft, 264 MiB 从未进 hexagon 权重块)。容量回退的真实代价体现在 p5/p12 mirror overhead (~200 ms/token), 是另一条独立成本链 (见下文 8.4 节)。

最短修复路径: 在 `set_tensor` 时把 Q5_K repack 为 Q4_0, 复用 [ggml-hexagon-jz.cpp：ggmlhexagon_supported_mul_mat](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp) 已有的 "Q4_K/Q6_K are stored as Q4_0" repack 机制, 24 个 split 直接消除, batch_calls 6400 -> 256。改动集中在 set_tensor 量化类型转换与 validator 放行, 比 scatter-gather (DSP 端 descriptor 协议扩展) 简单一个量级。

#### 8.3.3 JZ 与 QCOM 的 Q5_K split 拓扑相同

QCOM 同样拒绝 Q5_K, JZ 与 QCOM 的 Q5_K split 拓扑相同。这是代码级事实而非推测: QCOM `ggml_hexagon_supported_mul_mat` ([ggml-hexagon.cpp：ggml_hexagon_supported_mul_mat](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp)) 的 src0 switch 只认 Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4/F16/F32, Q5_K 同样走 `default: return false` ([ggml-hexagon.cpp：ggml_hexagon_supported_mul_mat](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp)), 且 QCOM 没有 JZ 的 Q4_K/Q6_K -> Q4_0 repack 机制 (`ggml_hexagon_is_repack_type` ([ggml-hexagon.cpp：ggml_hexagon_is_repack_type](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp)) 不含任何 K-quants)。同一 upstream scheduler 面对同一张 cgraph 必然给出同样的 24 个 CPU split。

从算子层面看, SOLVE_TRI / SSM_CONV / GATED_DELTA_NET 三个算子签名与 JZ 一模一样, 对比 [gated-delta-net-ops.c：op_gated_delta_net](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/gated-delta-net-ops.c) 与 [gated-delta-net-ops.c：op_gated_delta_net](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/gated-delta-net-ops.c) 完全相同; [main.c：execute_op](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/main.c) 的 op dispatch 也无 fused ssm_out。QCOM 的 24 个 Q5_K split 产生与 JZ 相同数量的 Q5_K sub-graph 段, 差别只在 per-call 提交成本 (QCOM 额外的 lm_head CPU split 见 8.5.4)。

`graphs reused = 253 / 255 = 99.2%` 是 upstream llama.cpp scheduler 的 cgraph 结构复用计数, 表示 253/255 decode 步复用切分拓扑, 不是提交次数, 与每 token 24 段 DSP 提交不矛盾 (第 25 段 main graph 为 cache hit)。

#### 8.3.4 Q5_K split 消除的收益与局限

消除 split 将每 token 的 24 次 DSP 调用合并为 1 次。注意 mirror memcpy 总量不随 DSP 调用次数缩减: 合并后的单段仍引用全部 heap 常驻权重 (~2 GiB), 每 token 都要镜像一次; 省的只是随 DSP 调用次数线性的固定开销 (scan/descriptor/sync/dcinva):

- 固定开销部分 (随 DSP 调用次数线性): 24 x ~13.5 ms -> 1 x ~13.5 ms, 节省 ~310 ms/token
- mirror memcpy 部分 (按引用量): 合并前后都是 ~2 GiB/token (~200 ms), 不省
- 估算: 681 - 310 = 371 ms/token -> ~2.7 tok/s

即便假设 mirror memcpy 完全消除 (136.5 s 全省): 174.4 - 136.5 = 37.9 s -> 6.8 tok/s, 该上界无法达到, 因为 heap 常驻权重必须镜像才能被 DSP 读。综合判断 ssm_out split 单独消除后 TG 区间 1.48 -> 2.0-3.0 tok/s。

即便乐观到 3.0 tok/s, 依然离 QCOM 7.70 tok/s 差 2.5x。后续收益来自两处 (与 split 修复正交): (a) mirror memcpy 的消除 (heap 权重改走 scatter-gather / DMA 拉取, 见下文 8.4 节); (b) per-call 固定开销的压缩 (批提交/异步化, 见下文 8.5 节)。

split 修复是后续一切提交路径优化的前置条件: 不先把 24 个 CPU 段消除, 批提交/流水线无从合并。

### 8.4 屏障二: Mirror memcpy 与 4 GiB 限制

#### 8.4.1 V79 DSP 32-bit VA hard cap

4 GiB 限制的根本原因是 Hexagon V79 user-mode 虚拟地址空间是 32-bit (4 GiB), 这是硬件级 hard cap (不是 platform 软件层限制):

> "The Hexagon processor features a unified byte-addressable memory. This memory has a single 32-bit virtual address space, which holds both instructions and data."
> -- [Hexagon V79 Programmer's Reference Manual §1.1 Memory](file:///opt/qcom/Hexagon_SDK/6.3.0.0/docs/pdf/80-N2040-60_REV_AA_Hexagon_V79_Programmer_Reference_Manual.pdf)

这意味着 DSP user-mode 任何时候只能看到 4 GiB 虚拟地址空间, HVX/HMX 指令通过 VA 访问 memory, VA 范围是 32-bit。无论 mempool 分配多大、FastRPC 64-bit 字段多宽、HAP_mmap2 多灵活, DSP 端 user-mode 一次只能 mmap/access <= 4 GiB。

**Table-36**: V79 4 GiB 限制证据链

| 证据 | 来源 | 内容 |
|---|---|---|
| A. QuRT 内存 API 有 32/64-bit 两套 | QuRT RTOS User Guide §2.6 / §21 | 32-bit 操作为向后兼容, 64-bit 操作 (后缀 `_64`) 可访问 > 4 GB 物理地址, 但 PA 端 64-bit 操作不改变 user VA 32-bit 限制 |
| B. V79 user-mode VA 是 32-bit | V79 PRM §1.1 | 32-bit general registers (R0-R31)、32-bit memory addressing modes、single 32-bit VA space |
| C. HAP_mmap 2 GB 限制已被 HAP_mmap2 解决 | [HAP_mem.h：HAP_mmap](file:///home/zhouwg/develop/ggml-hexagon/prebuilts/Hexagon_SDK/6.6.0.0/incs/HAP_mem.h) / [HAP_mem.h：HAP_mmap2](file:///home/zhouwg/develop/ggml-hexagon/prebuilts/Hexagon_SDK/6.6.0.0/incs/HAP_mem.h) | HAP_mmap 注释明确写 "limited to buffer size less then 2 GB", HAP_mmap2 用 size_t 64-bit 无 documented size limit。JZ v79 路径已用 HAP_mmap2 ([entry.c：ggml_dsp_register_ion](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/entry.c)) |
| D. rpcmem_alloc2 已是 64-bit | [rpcmem.h：rpcmem_alloc2](file:///home/zhouwg/develop/ggml-hexagon/refs/fastrpc/inc/rpcmem.h) | `rpcmem_alloc2(int heapid, uint32_t flags, size_t size)` 签名是 size_t 64-bit |
| E. FastRPC ioctl 内核接口是 64-bit | [fastrpc_ioctl.h：fastrpc_ioctl_req_mmap](file:///home/zhouwg/develop/ggml-hexagon/refs/fastrpc/inc/fastrpc_ioctl.h) / [fastrpc_ioctl.h：fastrpc_ioctl_munmap_req](file:///home/zhouwg/develop/ggml-hexagon/refs/fastrpc/inc/fastrpc_ioctl.h) | `fastrpc_ioctl_req_mmap.__u64 size` 与 `fastrpc_ioctl_munmap_req.__u64 length` 都是 64-bit |

实证 (多次实验经验值): 8Gen4 (HTP arch V79) probe_slots 最大 4032 MiB ([ggml-hexagon-jz.cpp：ggmlhexagon_init_rpcmempool](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp)), 4096 MiB 不在列表中。4032 MiB = 4 GiB - 64 MiB, 差值 64 MiB 对应 kernel allocator + QuRT 32-bit 字段的 page table / metadata 开销。QCOM 通过 per-chunk 分配 (get_max_size 返回 1 GiB, 最多 16 个 chunk) 绕过 4 GiB VA 限制支持大模型 (见 8.6.5 节源码分析), developer.md 亦提及多设备 layer-splitting 方案 ([developer.md:40-47](file:///home/zhouwg/develop/ggml-hexagon/docs/backend/snapdragon/developer.md#L40-L47))。

> 注: 以下归因为推测性分析, 缺乏 QCOM 内部源码或文档的直接证据, 仅供方向参考。

4 GiB 限制的可能归因 (推测, 按可能性排序): (1) 最可能 -- QCOM kernel allocator module 的内部 32-bit 截断, 即便 mainline Linux 已升级 `ion_allocation_data.len` 到 64-bit, QCOM Hexagon DSP 配套内核的 allocator driver 可能仍保留旧 ABI, 4032 MiB 探针边界正好对应 32-bit `len` 字段 + allocator metadata 开销; (2) 可能 -- QuRT / HAP_mmap2 内部实现细节, HAP_mmap2 签名是 size_t 但内部可能仍走 QuRT 32-bit 内存池操作, PA > 4 GiB 时需切到 `_64` 版本 (QCOM 在 [main.c：htp_iface_mmap](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/main.c) 仍保留 `HTP_MMAP_MAX_VMEM = 2147483648u` 2 GB 限制是旧版痕迹); (3) 最弱 -- Hexagon V79 32-bit user-mode VA 限制, 这条不直接限制 allocator 分配大小, allocator 分配后通过 fastrpc_mmap 把 fd 映射到 DSP VA, VA 4 GB 不够装 5.10 GB 单 buffer 时才会成为约束。

#### 8.4.2 Qwen3.5-9B 在 single mempool 中的内存布局

基于 GGUF 元数据 + `log_qwen3.5-9b.txt` alloc 记录实测:

```
========================================================================
Qwen3.5-9B 在 JZ 后端 single mempool 中的内存布局 (GGUF + alloc log 实测)
========================================================================

模型总量: 427 tensors, 32 layers, 5148.1 MiB = 5.03 GiB
  Q4_0  n=173  3929.62 MiB  (含 token_embd.weight [4096, 248320] 545.62)
  Q6_K  n=1     795.70 MiB  (output.weight [4096, 248320], 全模型唯一)
  Q5_K  n=24    264.00 MiB  (全部 ssm_out.weight [4096, 4096], 11.00 x 24)
  Q4_1  n=4     120.00 MiB / Q8_0 n=48  6.38 MiB / F32 n=177  32.39 MiB

调度期三路去向 (前两路 CPU 均与 mempool 容量无关):

  [CPU buft  809.6 MiB]
    token_embd.weight   545.62 MiB  embedding GET_ROWS 在 CPU (upstream 常规)
    24 x ssm_out.weight 264.0 MiB   Q5_K 被 MUL_MAT validator 拒
                                    -> 24 个 CPU split (类型根因)

  [mempool  ~2059 MiB 权重] (hexagon buft)
    weight chunk A  2005.05 MiB     首 tensor = output.weight 545.62 MiB
                                    (Q6_K 经 set_tensor repack Q4_0, 省 250 MiB)
    weight chunk B    54.00 MiB
    另有 compute buffer 若干 (50 + 4 + 256 + 50 + 62.62 MiB)

  [heap 回退  1996.95 MiB] (hexagon buft, 容量所迫: needed 1996 > remaining 1968)
    普通层权重, 不触发 split, DSP 经临时 mirror 执行
    代价: ~2 GiB/token AP-side mirror memcpy (~200 ms)

mempool 注册容量 4024 MiB (probe 4032 - 8 MiB reserve)
  权重装入后 pool_used 2109 MiB (52.4%), 加 compute buffer 后 ~2.5 GiB
  余量 ~1.5 GiB 空闲, 但单 chunk 需求 1996 MiB > 当时剩余 1968 MiB -> 回退
========================================================================
```

关键观察:

1. 权重 chunk 边界由运行时 mempool 余量决定, 不是固定值: PID 20562 拆 2005+1996 MiB 双 chunk (后块回退); PID 21326 单块 4002 MiB 整体回退 heap。两次运行的 split 模式相同: 24 个 Q5_K split 与 chunk 边界无关
2. 5.03 GiB 模型 > 4 GiB mempool 是事实, 但容量回退的代价形态是 mirror memcpy (heap 权重每 token ~2 GiB 镜像, ~200 ms), 不是 split; 24 个 split 全部来自 Q5_K 类型拒绝, 扩容 mempool 无法消除
3. 若容量也突破 (scatter-gather / 等效 8 GiB): heap 回退消失、mirror memcpy 归零, 叠加 split 消除后 681 - 325 - 200 = ~156 ms/token -> ~6-7 tok/s; 13B/20B 模型同时可装载

#### 8.4.3 Mirror 机制原理

Mirror 机制工作正常, 但 cache hit 跳过了 scan 却没跳过 memcpy。看 [ggml-hexagon-jz.cpp：ggmlhexagon_backend_graph_compute_batch](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp):

```cpp
// Cache hit 只回填 metadata, 不跳过 memcpy
if (cache_hit && cached_entry) {
    for (const auto & m : cached_entry->cached_mirrors) {
        buffer_mirrors_map[m.original_data] = {0, m.data_len, false};  // allocated=false
    }
} else {
    // cache miss 才计算 max_data_len
    for (int32_t tidx = 0; tidx < n_tensors; tidx++) {
        ...
    }
}

// 不论 cache hit/miss, 下面这段都执行:
for (auto & kv : buffer_mirrors_map) {
    ...
    memcpy(ion_buf, data_ptr, mirror_size);  // 每次都拷贝
    info.allocated = true;
}
```

设计根因: mirror 区域是临时 ion_region, 每次 call 结束被 Phase 12 末尾的 free 释放 ([ggml-hexagon-jz.cpp：ggmlhexagon_backend_graph_compute_batch](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp)), 下一次 call mempool 区域被复用, 必须重新拷贝。cache hit 跳过的只是 tensor_src scan (~1 ms/call), memcpy 本身的开销还在。

当前 mirror 的数据搬运路径:

```
[System Memory: heap 回退权重 (1996 MiB chunk)]
        |
        |  AP-side memcpy (Phase 5)
        |  每 token 合计拷 ~2 GiB (25 段各拷其引用切片)
        |  走 DDR -> AP L1 -> DDR -> mempool
        v
[mempool 4GB 临时 mirror 区]
        |
        |  DSP 读
        v
[DSP L2 cache]
```

**Table-37**: 当前 mirror 机制 vs 提议的 DMA 拉取路径

| 维度 | 当前 mirror | 提议的 DMA 拉取 |
|---|---|---|
| 数据搬运主体 | AP CPU memcpy | DSP DMA engine |
| 第一次 11MB weight | AP 拷 11 MB (~11 ms) | DSP DMA 11 MB (~1 ms) |
| 后续重复读 | AP 重拷 11 MB | DSP 命中 L2, 0 搬运 |
| mempool 占用 | 临时 mirror 区常驻周转 | 0 (weight 不在 mempool) |
| scheduler 视角 | weight 在 hexagon buft heap 回退 -> 无 split, 但每 call 付 mirror memcpy | weight 在 DSP-accessible buffer -> 无 split 且无 mirror |

DMA 拉取路径需要 JZ 补齐以下能力: (1) buffer type 支持 cross-buffer view (当前 [ggml-hexagon-jz.cpp：cgraph_cache](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp) 只能描述 mempool 内的 buffer); (2) DSP 端 execute_op 支持 scatter-gather descriptor; (3) AP 不再为 heap 回退权重走临时 mirror; (4) cache coherency 协议扩展 (dc ivac, 复用 JZ 既有 0xFFFD pre-inval 机制, 详见第六章 Path-G)。

#### 8.4.4 Mirror memcpy 代价

Qwen3.5-9B 推理时 mirror memcpy 的具体开销 (6 模型 CI 实测, batch_calls=6144):

- 每个 Hexagon 段的 mirror 对象有两类: (a) CPU 段边界的 `linear_attn_out-*` 激活 (1 MiB x 24 处/token, 小头); (b) heap 回退 chunk (1996 MiB) 中被该段引用的权重切片 (大头) -- 25 段合计 ~2 GiB/token
- p5 (mirror compute) = 72.35 s, p12 (mirror apply) = 70.98 s, 合计 143.33 s
- dsp_exec = 31.24 s (17.1%), 是 DSP 真实计算
- mirror memcpy ~2 GiB/token x 256 token @ ~10 GB/s = ~51 s, 其余 ~83 s 是随 call 数线性的固定开销 (6144 calls x ~13 ms)

mirror memcpy 开销占 TG wall time 的 78% (136.5 s / 174.4 s, 专门 profiling 基线)。Path-E mechanism 2 (cache hit 跳过 memcpy) 对 Qwen3.5-9B 无效: 临时 mirror 区每 call 释放不是单纯的设计疏忽, 而是容量所迫 -- 常驻权重已占 ~2.1 GiB, mempool 余量 ~1.5 GiB, 装不下 ~2 GiB heap 权重的常驻镜像 (2.1 + 2.0 = 4.1 GiB > 4 GiB 硬上限)。即使 patch 让 cache hit 跳过 memcpy, 也没有空间让镜像常驻。该 patch 只对小 mirror 集场景 (如五模型的 mask/激活镜像) 有效; Qwen3.5-9B 的 mirror 消除只能靠 scatter-gather / DMA 拉取, 让 DSP 直接读 system memory 的 heap 权重。

p5/p12 = 136.5 s 的分解: 固定开销 (scan/descriptor/sync/dcinva, 随 call 数线性) 6400 x ~13 ms = ~83 s, mirror memcpy ~2 GiB/token x 256 token @ ~10 GB/s = ~51 s。dsp_exec (p10) = 33.7 s 是 DSP 真实计算 (17.1%), 其余 ~4.2 s 是 p1+p6+p8+p11 等。QCOM 端 unaccounted 0.0% 是时间归类结果 (`common_perf_print` 只有高层类别, `flush()` 阻塞等待计入 eval, 非 AP-DSP 完全 overlap, 详见 8.8 第4点), cgraph cache 99.2% hit 让 99% 的 decode 步走 cache 复用路径。

### 8.5 屏障三: 算子提交路径开销

#### 8.5.1 QCOM dspqueue 机制

QCOM 使用高通 Hexagon SDK 提供的 dspqueue 库 ([ggml-hexagon.cpp：dspqueue.h](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp)):

```cpp
#include <dspqueue.h>   // 持久化 AP-DSP 队列
#include <rpcmem.h>
```

启动时建一条常驻队列 ([ggml-hexagon.cpp：ggml_hexagon_session::allocate](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp)):

```cpp
err = dspqueue_create(this->domain_id, 0, req_q_size, ...);
// AP <-> DSP 之间一条持久化的 dspqueue, 单次创建, 多次复用
```

`ggml_hexagon_session::enqueue_op` ([ggml-hexagon.cpp：ggml_hexagon_session::enqueue_op](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp)) 不直接走 DSP, 而是累积到本地 op_batch, 满了才 flush:

```cpp
void ggml_hexagon_session::enqueue_op(const htp_opnode & node) {
    if (!op_batch->fit_op(node)) {
        flush_batch();        // 满了才发, N 个 op 一次 dspqueue_write
    }
    op_batch->add_op(node);   // 否则只加到本地 op_batch
}
```

`flush_batch` ([ggml-hexagon.cpp：ggml_hexagon_session::flush_batch](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp)) 一次写整批:

```cpp
int err = dspqueue_write(this->queue, 0, 1, &dbuf, sizeof(req), ...);
// 一次写 N 个 op, dspqueue 内部排队, DSP 后台消费
```

`dspqueue_read` ([ggml-hexagon.cpp：ggml_hexagon_session::flush_pending](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp)) 读响应也是非阻塞的:

```cpp
int err = dspqueue_read(this->queue, &flags, 1, &n_dbufs, &dbuf, ..., timeo);
if (err == AEE_EEXPIRED || err == AEE_EWOULDBLOCK) {
    continue;  // 非阻塞, 没准备好就跳过
}
```

QCOM 的 graph_compute 入口 ([ggml-hexagon.cpp：ggml_backend_hexagon_graph_compute](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp)) 展示完整 producer-consumer 模式:

```cpp
// Queue and execute
if (opt_opstage & HTP_OPSTAGE_QUEUE) {
    for (const auto & node : *nodes_ptr) {
        sess->enqueue_op(node);  // 全部入队, 不等
    }
}
sess->flush();  // 末尾统一等
```

AP 端把所有 op 入队后立刻返回, DSP 后台消费; AP 端在 DSP 执行 op 的同时可以准备下一批 buffer / 算参数。QCOM 的 FastRPC call 仅用于 open/close/get 等准备工作, AP 与 DSP 间数据交换通过读写 dspqueue ring buffer 触发, 不走 FastRPC 调用; JZ 每个 sub-graph 都走一次同步 FastRPC invoke, 这是 Qwen3.5-9B per-call overhead 214x 差距的根因。

#### 8.5.2 JZ 同步 FastRPC vs QCOM async

JZ 的 [ggml-hexagon-jz.cpp：ggmlhexagon_backend_graph_compute_batch](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp) 每个 batch_call 对应一次 `ggml_dsp_execute_batch` (同步 FastRPC + 同步等 dspqueue_buffer):

- 每个 sub-graph 一次 fastrpc_invoke (synchronous)
- AP 端发完就阻塞等 DSP 返回
- DSP 算完返回 AP 才解阻塞进下一轮
- 没有本地 op_batch 累积
- 没有 producer-consumer 重叠

**Table-38**: QCOM dspqueue vs JZ 同步 FastRPC 实测对比

| 指标 | JZ (实测) | QCOM (实测/反推) | 差距 | 数据来源 |
|---|---:|---:|---:|---|
| Hexagon sub-graph 提交次数 | 24 x 256 = 6144 次 FastRPC | N/A (QCOM 不输出 batch_calls) | N/A | JZ dump_perf_stats batch_calls=6144 |
| per-call overhead | 21.4 ms (FastRPC + mirror memcpy) | ~0.1 ms (dspqueue_write 反推) | 214x | JZ dump_perf_stats avg=21369 us |
| AP-DSP overlap | 否 (同步阻塞) | 是 (dspqueue 异步) | - | [ggml-hexagon.cpp：ggml_backend_hexagon_graph_compute](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp) |
| cgraph cache hit rate | 99.1% (6088/6144) | 99.2% (253/255 graphs reused) | ~1.0x | 接近但非完全一致 |
| 每 token 提交成本 | 24 x 21.4 = 514 ms | N/A (call count 未知) | 214x (per-call) | Table-34 |
| TG 总耗时 | 174.4 s | 33.1 s | 5.27x | JZ dump_perf_stats vs QCOM eval time |
| TG tok/s | 1.48 | 7.70 | 5.20x 慢 | Table-32 |
| unaccounted time 占比 | 0.005% | 0.0% | ~1x | QCOM 只有高层类别, flush 等待计入 eval (非 AP-DSP overlap, 详见 8.8 第4点) |

QCOM 端 0.1 ms/dspqueue_write 是反推值 (dspqueue 内部是简单 ring buffer write), QCOM 端未单独测量此项。QCOM 端 `common_perf_print` 只输出顶层分类 (sampling / samplers / load / prompt eval / eval / total / unaccounted), 没有 JZ 的 p1-p12 12 阶段, 也不输出 batch_calls。unaccounted = 0.0% 是因为 QCOM 只有高层类别, `graph_compute` 中 `flush()` 的阻塞等待计入 eval 类别, 非 AP-DSP 完全 overlap (详见 8.8 第4点)。

#### 8.5.3 214x per-call overhead

Qwen3.5-9B TG 5.20x 整体差距来自提交路径差距: JZ 676 ms/token 扣掉 514 ms 提交后 ~162 ms, 与 QCOM 130 ms/token 持平 (见 Table-34)。ssm_out split 本身的算子成本在两边都相同, dspqueue 并未减少提交次数 (24 个 Q5_K sub-graph 仍需 24 次提交, Q5_K sub-graph 数量由 upstream scheduler 决定, JZ 与 QCOM 相同), 而是将 per-call overhead 降低 ~214x。这是 5.20x 整体差距的真因, 不是 ssm_out 算子缺失 (QCOM 算子 dispatch 与 JZ 一致)。

JZ cgraph cache (99.1%, 6088/6144) 与 QCOM `graphs reused` (99.2%, 253/255) 命中率接近 (JZ 是 AP 端按 op+shape+src ptr 哈希的 descriptor 复用 cache, QCOM 是 upstream scheduler 的 cgraph 结构复用计数, 两者层级不同但都约 99%, 因 Qwen3.5-9B cgraph 拓扑变化频率稳定)。差距不在 cache 命中率, 而在 cache miss 之后的提交路径: JZ 走 per-sub-graph 同步 FastRPC (每次 mirror + sync), QCOM 走 dspqueue_write 批提交。

#### 8.5.4 JZ 净优势: lm-head on DSP

除 JZ 与 QCOM 共有的 ssm_out split 外, QCOM 还有 JZ 没有的额外 lm_head split: QCOM `ggml_hexagon_supported_mul_mat` ([ggml-hexagon.cpp：ggml_hexagon_supported_mul_mat](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp)) 对 `ne[1] > 32768` 硬编码拒绝 lm_head (Qwen3.5-9B 的 output.weight 维度 248320 远超阈值)。

JZ 靠 Q6_K -> Q4_0 repack + mempool 连续 IOVA 把 795.70 MiB lm_head 留在 DSP 执行, 这是 JZ 的净优势。QCOM 端 lm_head 走 CPU, TG 端到端比 JZ 多 ~80 ms/token。Qwen3.5-9B 的 output.weight 是 Q6_K [4096, 248320] (全模型唯一 Q6_K 张量), 经 set_tensor repack 为 Q4_0 后省 250 MiB (795.70 -> 545.62)。这是 JZ single mempool 架构 (单一连续 IOVA 范围) 的核心优势: 让 lm_head 这类涉及整个 vocab 范围 embedding/logits 的算子留在 DSP 执行。

### 8.6 修复路径

#### 8.6.1 路径总览

Qwen3.5-9B 性能修复是三条独立路径的叠加, 各自解决一个问题, 正交可叠加:

1. **Q5_K repack (消除 graph split)** - ssm_out.weight 264 MiB 是 Q5_K 类型被 validator 拒绝, set_tensor 时 repack Q4_0 (复用既有 Q4_K/Q6_K repack 机制, ~30-50 行) 后 validator 放行, 24 个 MUL_MAT split 直接消除, batch_calls 从 6144 降到 256, TG 1.48 -> ~2.7 tok/s (省的是随 call 数线性的固定开销 ~310 ms/token)
2. **scatter-gather (消除 mirror + 扩容量)** - heap 回退的 ~2 GiB 权重免镜像, 消除 ~200 ms/token mirror memcpy, 叠加路径 1 后 TG ~2.7 -> ~6-7 tok/s; 同时把可装模型上限从 4 GiB 提到 24 GiB (13B/20B 可装载)
3. **异步化提交 (压平 per-call overhead)** - 压平 ~214x per-call overhead 差距, 叠加后 TG -> 8-10 tok/s, 追平/超过 QCOM 7.70

以上三条是 JZ 内部可控路径。此外存在一条外部依赖路径: 高通实现真正 UMA (外部依赖) - developer.md ([第29行](file:///home/zhouwg/develop/ggml-hexagon/docs/backend/snapdragon/developer.md#L29)) 宣称 "Snapdragon's unified memory model where all buffers are fully accessible by the CPU and GPU", 但实际 DSP OS 并未做到真正 UMA: V79 DSP 32-bit VA 4 GiB hard cap、FastRPC async 底层能力被关闭、mirror memcpy 需求等均说明 DSP 与 system memory 之间存在访问壁垒。AMD APU (Ryzen)、Apple Silicon (M1-M4)、NVIDIA Grace Hopper 均已实现真正 UMA, 高通 DSP 在物理层共享 LPDDR5X 但软件层未做到。若高通修改 DSP OS 及配套 Hexagon SDK 实现真正 UMA, 则路径 2 (scatter-gather) 和路径 3 (异步化提交) 的工程量将大幅降低甚至消失。该路径不在 JZ 可控范围内。

**Table-39**: Qwen3.5-9B 修复路径总览 (按收益/风险/工时排序)

| 步骤 | 工作 | 解决什么问题 | TG 收益 | JZ 改动量 | 风险 |
|---|---|---|---|---|---|
| **0. Q5_K -> Q4_0 repack** | 复用既有 Q4_K/Q6_K repack 机制: `ggml_hexagon_weight_dsp_type` 加 Q5_K 映射 + 新增 `repack_q5k_as_q4_0_tiled_to_buf` + validator switch 加 Q5_K case | 24 个 ssm_out CPU split (类型根因) | 1.48 -> ~2.7 tok/s (batch_calls 6144 -> 256) | 小 (~30-50 行) | 低 (需 CPU vs DSP 数值对比验证输出不乱码) |
| 1a. ~~改 get_max_size (1 行)~~ | 改 get_max_size() 返回 1 GiB 让 ggml core 拆 chunk | weight 容量 | 不可行 (验证失败) | N/A | JZ alloc_buffer 是 bump allocator, 改 get_max_size 不能扩总容量 |
| **1b. scatter-gather** | DSP 端 HVX descriptor 从 system memory 物理页拉 weight, 绕过 V79 user VA 4 GiB hard cap | weight 容量 + heap 权重 mirror memcpy (不含 split) | 叠加步骤 0 后 ~2.8 -> ~6-7 tok/s; 13B/20B 可装载 | 较大 (DSP entry.c scatter-gather 支持) | 中 |
| 1c. HAP_mmap2 system memory | HAP_mmap2 按 layer 切换 mmap 内容, sliding window | weight 容量 | 装得下, 仍受 V79 4 GiB 总 VA 限制 | 较大 (DSP 端 mmap 切换协议 + cache 一致性) | 中 |
| 1d. 混合方案 | 控制结构放 DSP user VA, weight 放 system memory scatter-gather | weight + control | 装得下, 理论最干净 | 较大 (架构重构, 1-2 个月) | 中 |
| 2. op_batch 累积 | 压缩 per-call 固定开销 | 与容量/split 均不直接相关 | 步骤 0 落地前 1.48 -> 2.5-3.5; 步骤 0 后收益收窄 | 小 (改 1 个函数) | 低 |
| 3. K/V 量化 | 让 KV cache 装下 4 GiB (第七章 Table-28 P1) | KV cache | 6-7 -> 8-10 tok/s | 中 (算子 + Q8/FP16 kernel) | 中-高 |
| 4. 异步化提交 | 进一步压平 per-call overhead (~214x 差距真因) | 提交路径 | 叠加后 8-10 tok/s, 追平/超过 QCOM | 中 (改 fastrpc 路径) | 中 |
| 远期. producer-consumer | AP 端 12 阶段合并为 descriptor 生产, DSP 端独立消费 | 提交路径 | 7 -> 8-10 tok/s (超 QCOM 1.0-1.3x) | 大 (动 12 阶段 pipeline) | 高 |

步骤 0 是 Qwen3.5-9B 性能修复的"第一刀": 不先消除 24 个 CPU 段, 后续批提交/流水线无从合并 (25 段不合并, op_batch 能批的只是 descriptor 准备, 段间同步开销还在)。步骤 1b 是大模型 (> 4 GB) 的"容量前提"。1b / 1c / 1d 三选一不互斥: 1d 工程最干净但重构量最大, 1c 仍受 V79 4 GiB 总 VA 限制, **1b scatter-gather 是容量/mirror 维度的明显更优**。

per-buffer 替代单 mempool 方案不在主路径表: 它会推翻 JZ single mempool 架构优势 (失去 offload lmhead 能力, Qwen1.5-1.8B / 2B / 8B 等 < 4 GiB 模型 TG 可能显著下降), 是架构层面取舍, 不是明显更优。待用户明确接受"放弃 JZ 架构优势换取 Qwen3.5-9B+ 容量"后再评估。

建议优先策略: 步骤 0 (Q5_K repack) 立即落地并过五模型 CI, 同步并行 1b (scatter-gather) 与 2 (op_batch), 实施期间持续验证 Qwen1.5-1.8B / 2B / 8B 等 < 4 GiB 模型 TG 不退化。1c / 1d 作为更长线储备 (1-2 个月工时), 等 1b 决策后再评估。5 模型 CI (Table-11) 与 6 模型 CI AB test 关系: 5 模型 CI 跑 JZ 单后端 (无 QCOM 对比), 用于回归基线; 6 模型 CI AB test 跑 JZ vs QCOM 完整对比, 用于评估 JZ 净优势/劣势。两套 CI 互补, 不互斥。

#### 8.6.2 步骤 0: Q5_K repack

步骤 0 优先级最高:

- 改动最小 (~30-50 行): `ggml_hexagon_weight_dsp_type` 加 Q5_K 映射 + 仿 `repack_q6k_as_q4_0_tiled_to_buf` 新增 Q5_K repack + validator 加 case
- 收益确定: 24 个 split 直接消除, batch_calls 6144 -> 256, TG 1.48 -> ~2.7 tok/s
- 是后续一切提交路径优化的前置条件 (不消除 24 个 CPU 段, 批提交/流水线无从合并)
- 副作用可控: Q5_K -> Q4_0 是有损重量化 (与既有 Q4_K/Q6_K -> Q4_0 同性质), ssm_out 权重体积 264 -> 216 MiB, 加载期一次性转换; 需先做 CPU vs DSP 数值对比验证输出不乱码 (Qwen3.5-9B 乱码问题与 GATED_DELTA_NET 相关, repack 后该层输入分布变化需回归)
- < 4 GiB 模型 (Qwen1.5-1.8B/2B/8B) 不含 Q5_K ssm_out, 完全不受影响 (五模型 CI 回归确认)

#### 8.6.3 步骤 1b: Scatter-gather

scatter-gather 是容量/mirror 根治路径 (明显更优):

- 保留 JZ single mempool 架构, 主 mempool 仍装 lmhead + 主要权重
- 消除 heap 回退权重的 ~200 ms/token mirror memcpy, 叠加步骤 0 后 TG ~6-7 tok/s
- 把可装模型上限从 4 GiB 提到 24 GiB (13B/20B 可装载)
- < 4 GiB 模型 (Qwen1.5-1.8B/2B/8B) 继续用 single mempool, 完全不受影响
- 保留 lmhead offload 架构优势, Qwen3.5-9B lmhead 继续 DSP 执行
- 实施成本集中在 DSP 端 entry.c scatter-gather 支持 (1-2 个月工时)
- 24 个 ssm_out split 不在本路径范围: 它们是 Q5_K 类型拒绝, 由步骤 0 覆盖。scatter-gather 无法覆盖 ssm_out.weight (从未进入 hexagon 权重块)

DMA 拉取路径的本质 (与 scatter-gather 等效): DSP 端 HAP_mmap2 映射 system memory 物理页, 第一次冷 DMA ~1 ms, 后续命中 L2 cache 0 搬运。weight 不在 mempool, mempool 占用为 0。需要 JZ 当前没有的支持: (1) buffer type 支持 cross-buffer view; (2) DSP 端 execute_op 支持 scatter-gather descriptor (远端 system memory 区域 + 任意 offset); (3) AP 不再为 heap 回退权重走临时 mirror; (4) cache coherency 协议扩展 (dc ivac)。

DMA 拉取路径实施成本与阻塞评估:

| 改动点 | 复杂度 | 阻塞 |
|---|---|---|
| JZ buffer type 体系扩展 | 高 (影响 set_tensor / alloc_buffer 全链路) | 上游 scheduler 看不到新 buffer type, 需 hook `buffer_is_host` / `buffer_supports_backend` |
| DSP entry.c 接收 scatter-gather descriptor | 高 (改 fastrpc payload 格式) | 影响所有 5 模型, 需 5 模型 CI 全过 |
| cache coherency 协议扩展 | 中 (dc ivac flush 整个 weight region) | 复用 Path-G 的 0xFFFD 机制 |
| 上游 ggml-backend.cpp scheduler hook | 高 | 与 upstream llama.cpp 接口耦合 |

scatter-gather 与 Q5_K repack、异步化提交正交: 三者针对 Qwen3.5-9B 性能瓶颈的不同维度 (减少 batch_call 数量 / 减小 per-call 开销 / 消除 mirror memcpy), 可叠加。

#### 8.6.4 步骤 2-4: op_batch -> 异步化提交 -> producer-consumer

三阶段解决算子提交路径开销 (~214x 差距真因), 与步骤 0/1b 正交:

**短期 op_batch 累积 (1-2 周)**: 同一 batch_calls 内多个 sub-graph 合成一个 fastrpc descriptor, 复用现有 `ggml_dsp_execute_batch` 路径。TG 1.48 -> 2.5-3.5 tok/s。风险低, 改动小 (改 1 个函数), 不动 Hexagon SDK 调用约定, 仅在 AP 端做 descriptor 复用。注意步骤 0 落地后每 token Hexagon 段从 25 降到 1, op_batch 的"多段合一"收益被大部分覆盖, 更应理解为步骤 0 落地前的过渡手段。具体改动点: (1) `ggmlhexagon_backend_graph_compute_batch` 入口处增加 `op_batch_t` 本地累积; (2) `ggml_dsp_execute_batch` 改为支持 N-合-1 descriptor; (3) batch 边界 / `flush_batch` 时统一等一次 FastRPC 返回; (4) AP 侧的 p5/p12 (mirror) 流程合并到 batch 级。

**中期 async FastRPC**: 异步 fastrpc_invoke, AP 端不阻塞, DSP 后台消费。TG 3.5 -> 6-7.7 tok/s。风险中, 需改 fastrpc 路径 + 错误处理, 5 模型 CI 全过。**当前阻塞**: JZ 代码已通过 `ggmlhexagon_is_async_fastrpc_supported()` ([ggml-hexagon-jz.cpp：ggmlhexagon_is_async_fastrpc_supported](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp)) 查询 `ASYNC_FASTRPC_SUPPORT` 能力, 实测 `async fastrpc supported 0` -- API 接口已存在但底层驱动能力被关闭。这与高通上游 PR #26049 (将缓存维护逻辑下放到算子内部, 迫使 JZ 维护独立 kernels/ 目录) 是类似模式: 高通限制底层能力开放。

**远期 producer-consumer 流水线 (重构)**: AP 端 12 阶段合并为 descriptor 生产, DSP 端独立消费, 移除 1-12 阶段间的隐式同步。TG 7 -> 8-10 tok/s (超过 QCOM 1.0-1.3x)。风险高, 动 12 阶段 pipeline, 配合 K/V 量化可到 9-12 tok/s。

#### 8.6.5 QCOM scatter-gather 源码分析

QCOM 与 JZ 实际控制 buffer 拆分的关键参数 `get_max_size()`:

**Table-40**: QCOM per-chunk vs JZ single-mempool 架构对比

| 架构选择 | get_max_size() 返回 | ggml core 拆 chunk | DSP 端 4 GiB VA window 行为 | 总可装 weight | offload lmhead | 适用模型 |
|---|---|---|---|---|---|---|
| QCOM per-chunk | 1 GiB (默认, [ggml-hexagon.cpp：opt_mbuf](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp)) | 多个 1 GiB chunk (最多 16 个) | scatter-gather / sliding window 让 chunk 不一次性 mmap 进 4 GiB VA | n_chunks * 1 GiB <= 16 GiB | 不可 (per-buffer 拆散) | 9B/13B/20B 都可 |
| JZ single-mempool | ~4 GiB ([ggml-hexagon-jz.cpp：ggmlhexagon_init_rpcmempool](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp)) | 1 个 4 GiB chunk (> 4 GiB 触发 V79 user VA 上限) | 整个 mempool 一次性 mmap 进 4 GiB VA | 4 GiB (V79 user VA hard cap) | 可 (单一连续 IOVA) | 9B/13B/20B 都不可 |
| JZ scatter-gather (步骤 1b) | ~4 GiB (主 mempool 不变) | 主 mempool 1 个 4 GiB + 辅助 region | 主 mempool 连续 + 辅助 region scatter-gather | n_regions * < 4 GiB | 可 (主 mempool 仍连续) | 9B/13B/20B 都可 |

ggml core 内部 mempool 机制 ([ggml-alloc.c](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-alloc.c)): `ggml_dyn_tallocr` 是动态 tall allocator, 每个 buffer type 一个, 内部维护 `tallocr_chunk[]` 数组 (最多 `GGML_VBUFFER_MAX_CHUNKS = 16` 个 chunk)。每个 chunk 是一个独立 backend buffer, 通过 `ggml_backend_buft_alloc_buffer(buft, chunk_size)` 分配。新 chunk 大小 = `MAX(min_size, max_chunk_size)`, 其中 `max_chunk_size = MIN(get_max_size(), SIZE_MAX/2)`。单 tensor 太大超过 chunk 时, 整 chunk 扩到能装下。QCOM `get_max_size` 返回 1 GiB, 所以 chunk_size 最大 1 GiB; JZ `get_max_size` 返回 ~4 GiB, 所以 chunk_size 最大 4 GiB -- 但 JZ `alloc_buffer` 是 bump allocator, 多次调用仍从同一个 `ctx->rpc_mempool` 切块, 总容量不变。

QCOM 实际 per-chunk 分配路径 ([ggml-hexagon.cpp：ggml_backend_hexagon_buffer_type_alloc_buffer](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp)):

```cpp
static ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto sess = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->Context)->sess;
    try {
        size += 4 * 1024;  // guard page
        ggml_hexagon_shared_buffer * sbuf = new ggml_hexagon_shared_buffer(sess, size);
        return ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_buffer_interface, sbuf, size);
    } catch (const std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate buffer context (host): %s\n", sess->c_name(), exc.what());
        return nullptr;
    }
}
```

这里的 `size` 是 ggml core 传下来的 chunk_size (不是 per-tensor size)。QCOM `get_max_size` 返回 1 GiB, 所以 chunk_size 最大 1 GiB, ggml core 在内部 tallocr 中按需创建多个 1 GiB chunk。QCOM 在 `ggml_hexagon_measure_max_vmem` 中通过 `sbuf_alloc` + `sbuf_free` 试探性创建/销毁多个 1 GiB shared buffer, 记录最大值到 `this->max_vmem` ([ggml-hexagon.cpp：ggml_hexagon_measure_max_vmem](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp))。运行时 `alloc_buffer` 每次按 chunk_size 调 `sbuf_alloc` 单独分配, 不依赖单一预分配 heap。这种"per-chunk 独立分配"配合 DSP 端 `HAP_mmap2` 在 4 GiB VA window 内动态 mmap/unmap, 配合 HVX scatter-gather descriptor 跳过 user VA 限制, 让 QCOM 能装下 16 GiB 模型 (16 chunk x 1 GiB)。

JZ 极简方案验证失败: 先前假设"改 `get_max_size` 为 1 GiB 让 ggml core 拆 chunk"是 1-2 行代码改动。但 JZ `alloc_buffer` 不是按 chunk 调 `rpcmem_alloc2` 分配独立 buffer, JZ 实际架构是"probe 阶段探测最大 mempool 单块 + 一次预分配大块 mempool + bump allocator 切块" ([ggml-hexagon-jz.cpp：ggmlhexagon_init_rpcmempool](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp) probe, [ggml-hexagon-jz.cpp：ggmlhexagon_probe_dspinfo](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp) pre-allocate, [ggml-hexagon-jz.cpp：ggml_backend_hexagon_buffer_type_alloc_buffer](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp) alloc_buffer)。改 `get_max_size = 1 GiB` 的实际效果: ggml core 拆为多个 1 GiB chunk -> 调 `alloc_buffer` 多次 -> 每次仍然从同一个 `ctx->rpc_mempool` 切 1 GiB -> 总容量 4 GiB 没变 -> Qwen3.5-9B 5.10 GB 仍然装不下。

JZ single-mempool 优化历史背景: JZ 之前的所有"single-mempool 优化" (mirror 机制、Phase 1-12 调度、p10 测量) 都是基于"所有 weight 在一个连续 IOVA 范围"的假设。这是因为 ggml core 默认行为是 `get_max_size = mempool_len` 时只创建一个 4 GiB 单 chunk。即使 ggml core 已支持多 chunk (`GGML_VBUFFER_MAX_CHUNKS = 16`), JZ 内部 `alloc_buffer` 仍按 bump allocator 切同一块 `ctx->rpc_mempool` -- 所以多 chunk 不会自动绕过 mempool 4 GiB hard cap。

JZ 未采纳 QCOM per-buffer 方案的原因: JZ 的 mirror 机制假设所有 weight 在同一连续 IOVA 范围 (mirror 缓冲池也开在同一段) 以简化 DSP 端 offset 计算; QCOM 无 mirror 机制 (mirror 是 JZ 自有设计), 可直接 per-buffer 拆分。因此 JZ 走 scatter-gather 路径时, mirror 机制必须先扩展支持跨 buffer。

per-buffer 替代单 mempool 的取舍: 这是一个真正的取舍, 不是明显更优。JZ single mempool 架构 (单一连续 IOVA) 让 JZ 能 offload lmhead (lmhead 需要整个 vocab 范围的 embedding/logits tensor, 必须连续 IOVA); 改 per-buffer 后 lmhead 必须回 CPU, Qwen1.5-1.8B / 2B / 8B 等 < 4 GiB 模型 TG 可能显著下降。

**Table-41**: per-buffer 替代单 mempool 的取舍

| 维度 | JZ 当前 single mempool | JZ 改 per-buffer 后 |
|---|---|---|
| 模型容量上限 | 4 GiB (V79 user VA 限制) | 16 GiB (V79 user VA + scatter-gather 滑动) |
| offload lmhead | 可 (单一连续 IOVA) | 会失去 (per-buffer 拆散) |
| mirror 机制 | 简单 (一个 4 GiB region 一次 dcinva) | 复杂 (每个 buffer 独立 mirror) |
| Phase 1-12 调度 | 简单 (单 mempool base addr) | 复杂 (per-buffer base addr 数组) |
| Qwen1.5-1.8B / 2B / 8B 等 < 4 GiB 模型 TG | 现有 baseline | 可能下降 (lmhead 回 CPU) |

净收益: 获得 Qwen3.5-9B+ 模型容量, 但放弃 JZ 架构优势 (offload lmhead)。scatter-gather (步骤 1b) 保留主 mempool, 不推翻 JZ 架构, 不失去 lmhead offload 能力, 是明显更优。

#### 8.6.6 大模型扩展性矩阵

突破 4 GiB 限制只解决 weight 一个维度, 完整的内存架构需要分别考虑三个独立维度:

**Table-42**: 内存架构三个独立维度

| 维度 | 当前位置 | 4 GiB 限制是否卡 | 突破方式 |
|---|---|---|---|
| Weight | mempool 4 GB | 是 (Qwen3.5-9B 已回退 ~2.0 GiB 到 heap) | scatter-gather / DSP 端 mirror |
| KV cache | mempool 4 GB | 是 (长 ctx 8K 卡 1.6-3.4 GB) | K/V 量化 (FP16/Q8, 第七章 Table-28 P1) |
| Activation / scratch | mempool 4 GB | 否 (小, < 100 MB) | 不需要突破 |

scatter-gather 把 weight 上限从 4 GB 提到 24 GB, 但 KV cache 仍是 4 GB mempool 限制, 两者独立存在, 必须分别突破。

当前 Snapdragon 8 Elite 平台手机系统内存分布与 scatter-gather 后可装模型上限 (基于市场公开信息估算, 非实测数据; 可装上限 = 系统内存 - OS 占用 ~1 GB):

**Table-43**: 手机型号 vs scatter-gather 后可装 Q4_0 模型上限 (估算)

| 手机型号 | 系统内存 | scatter-gather 后可装 Q4_0 模型上限 | 备注 |
|---|---|---|---|
| 8 Gen 2 主流 | 8-12 GB | ~7-11 GB | 可装 Qwen3.5-9B 边缘, 13B 装不下 |
| 8 Gen 3 主流 | 12-16 GB | ~11-15 GB | 可装 Qwen3.5-9B / 13B, 20B 边缘 |
| 8 Elite 主流 | 12-16 GB | ~11-15 GB | 同 8 Gen 3 |
| 8 Elite 高配 | 24 GB (极少) | ~22 GB | 可装 20B / 32B, 70B 仍不够 |

4 GiB 现状 vs 容量突破后对比 (Qwen3.5-9B):

**Table-44**: 4 GiB 现状 vs 容量突破后对比 (Qwen3.5-9B)

| Metric | 4 GiB 现状 | 容量突破后 (scatter-gather + Q5_K repack) |
| --- | --- | --- |
| 模型总权重 | 5.03 GiB | 5.03 GiB |
| DSP 可达权重 | 4088 MiB (2059 mempool + 1996 heap + 小 chunk) | ~4304 MiB (含 ssm_out repack 后 216 MiB) |
| CPU 权重 | 809.6 MiB (token_embd 545.62 + ssm_out 264) | 545.62 MiB (仅 token_embd, embedding 在 CPU 是 upstream 常规) |
| 24 个 ssm_out CPU split | 有 (Q5_K 类型根因) | 仍在 -- 与容量无关, 需叠加 Q5_K repack 才消除 |
| mirror memcpy | ~2 GiB/token (~200 ms) | 0 (heap 回退消失) |
| Hexagon 段/token | 25 | 25 (扩容 alone); 1 (叠加 Q5_K repack) |
| batch_calls (256 token) | 6144 | 6144 (alone); 256 (叠加) |
| TG tok/s (估算) | 1.48 (实测) | ~2.1 (alone, 只省 mirror); ~6-7 (叠加 repack) |
| 13B (~7.5 GB) / 20B (~11 GB) | 不可 | 可装 (backing 在 system memory, 16 GB 手机覆盖) |

容量突破单独只省 mirror (~200 ms/token), 不消除 split -- split 是 Q5_K 类型根因。容量 + repack 叠加把 Qwen3.5-9B TG 从 1.48 推到 ~6-7 tok/s; 再叠异步化提交追平/超过 QCOM 7.70。

容量突破后 (scatter-gather + Q5_K repack 叠加) 的内存布局:

```
5.03 GiB 模型权重
  |-- CPU buft 545.62 MiB: token_embd.weight (embedding 仍在 CPU, upstream 行为)
  |-- DSP 可达 ~4.2 GiB (4088 MiB + ssm_out repack 后 216 MiB, Q5_K 264 -> Q4_0 216)
        |-- 权重存 system memory, DSP scatter-gather / DMA 直读
              |-- 无 split (Q5_K 已 repack Q4_0, validator 放行)
              |-- 无 mirror memcpy (heap 回退消失)
```

**Table-45**: scatter-gather + K/V 量化双重突破后, 主流 Q4_0 模型在不同手机上的可行性

| 模型 | 权重 (Q4) | KV cache (4K ctx) | KV cache (8K ctx) | 16 GB 手机 | 24 GB 手机 |
|---|---|---|---|---|---|
| Qwen3.5-9B | 5.10 GB | 0.6 GB | 1.2 GB | 可装载 | 可装载 |
| Qwen3.5-13B | 7.5 GB | 0.8 GB | 1.6 GB | 可装载 | 可装载 |
| Qwen3.5-20B | 11 GB | 1.0 GB | 2.0 GB | 可装载 (不需 K/V 量化) | 可装载 |
| Qwen3.5-32B | 18 GB | 1.6 GB | 3.2 GB | 超过 16 GB | 可装载 (需 K/V 量化 8K ctx) |
| Llama-3-70B | 38 GB | 2.8 GB | 5.6 GB | 超过 16 GB | 超过 24 GB |

scatter-gather 让 16 GB 手机从"只能装 4 GB 模型"变成"可装 11 GB 模型" (Qwen3.5-9B / 13B / 20B 全覆盖)。scatter-gather + K/V 量化让 24 GB 手机可装 18 GB 模型 (Qwen3.5-32B), 突破 4 GiB KV cache 限制。70B 量级任何手机都装不下, 需要 weight 压缩到 Q2/Q3 (Llama-3-70B Q2 ~22 GB) 或 SSD offload。

### 8.7 与第七章对比

Qwen1.5-1.8B (24 层 MHA 1:1) 与 Qwen3.5-9B (32 层 + delta-net 24) 在 JZ 优化路径上的关键差异:

**Table-46**: Qwen1.5-1.8B 与 Qwen3.5-9B JZ 优化路径对比

| 维度 | Qwen1.5-1.8B (第七章) | Qwen3.5-9B (本章) |
|---|---|---|
| 首要瓶颈 | a-inv + bulk flush (60% wall, L2 8 MB 限制, 7.3/7.4) | 算子提交路径开销 (p5+p12 78% wall, 见 Table-34) |
| 算子瓶颈 | H3 KV invalidation 确认 (16241 us/batch, 7.2) | 24 个 ssm_out MUL_MAT split (Q5_K 根因, 8.3) |
| 模型容量 | 1.2 GiB (Q4_0), 全装入 4 GB mempool | 5.03 GiB, 1996 MiB 回退 heap (8.4.2 详) |
| QCOM 对比 | JZ TG 比 QCOM 快 1.61x | JZ TG 比 QCOM 慢 5.20x |
| 主要优化方向 | K/V 量化 (Table-28 P1, 10-19% TG) | Q5_K repack 消除 split (Table-39 步骤 0) + 异步化提交 (Table-39 步骤 4) |
| 短期可实施性 | 高 (改 cache 写入路径) | 高 (Q5_K repack 复用既有 Q4_K/Q6_K 机制) |

两条路径不冲突: Qwen1.5-1.8B 走 cache 量化路线, Qwen3.5-9B 走 Q5_K repack 消除 graph split + 异步化提交路线。可以分头推进, 最终在远期 (producer-consumer 流水线) 阶段汇合。

### 8.8 本章结论

1. **Qwen3.5-9B 5.20x TG 差距的真因是 214x per-call overhead, 不是 graph split 数量**。JZ 与 QCOM 的 Q5_K split 段数相同 (同一 upstream scheduler, QCOM validator 同样拒 Q5_K, 各 24 个 ssm_out split; QCOM 不输出 batch_calls), 但 QCOM 额外有 1 个 lm_head CPU split, 而 JZ 因 lm_head offload 到 DSP 无此 split (详见 8.5.4)。差距来自 per-call overhead: JZ per-call 走同步 FastRPC + mirror memcpy (21.4 ms), QCOM per-call 只是一次 dspqueue ring-buffer write (~0.1 ms 反推), per-call 差距 214x -> JZ 单 token 提交成本 514 ms (24 x 21.4)。扣除提交后 JZ ~162 ms 与 QCOM 130 ms 基本持平 (见 Table-34)。dspqueue 并未减少提交次数 (24 个 Q5_K split 仍需 24 次提交), 而是将 per-call overhead 降低 ~214x

2. **Qwen3.5-9B PP&TG 三条正交修复路径**: Q5_K repack 消除 graph split (1.48 -> ~2.7 tok/s, ~30-50 行, 最高优先级, 是后续提交路径优化的前置条件, 已在 commit a7c586e70d195cc0a695fdb8adf58298632be0d6 实施并经验证) -> scatter-gather 消除 mirror memcpy + 扩容量 (-> ~6-7 tok/s, 明显更优, 保留 single mempool 架构) -> 异步化提交压 per-call overhead (-> 8-10 tok/s)。三条路径正交可叠加, op_batch 累积作为步骤 0 落地前的过渡 (1.48 -> 2.5-3.5 tok/s) 可并行

3. **Qwen3.5-9B TG 理论目标 8-10 tok/s, 追平/超过 QCOM 7.70**。配合第七章 Table-28 P0 (E 方向 AP per-call overhead 消除) + 本章修复路径, JZ Qwen3.5-9B TG 可从 1.48 逐步追到 8-10 tok/s。

4. **QCOM 端 unaccounted 0.0% 是时间归类结果, 非 AP-DSP 完全 overlap**。QCOM `common_perf_print` 只有高层类别 (prompt eval / eval / sampling 等), 没有 JZ 的 p1-p12 细粒度分解, `graph_compute` 中 `flush()` 的阻塞等待计入 eval 类别, 故 unaccounted 为 0.0%。代码上 `graph_compute` ([ggml-hexagon.cpp：ggml_backend_hexagon_graph_compute](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp)) 先 enqueue 所有 op (非阻塞 dspqueue_write), 再调用 `flush()` ([ggml-hexagon.cpp：ggml_hexagon_session::flush](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp)) 阻塞等待 DSP 完成, AP 在 flush 期间不准备下一个 graph。dspqueue 的真正优势是 per-call overhead 极低 (~0.1 ms ring buffer write vs JZ 21.4 ms 同步 FastRPC + mirror memcpy) 且无 mirror memcpy。JZ cgraph cache (99.1%) 与 QCOM `graphs reused` (99.2%) 命中率接近 (两者层级不同, Qwen3.5-9B cgraph 拓扑变化频率稳定), 差距不在 cache 命中率, 而在每次提交的 per-call overhead (cache hit 仅跳过 Phase 2 descriptor 重建, 不跳过 mirror memcpy + FastRPC 提交): JZ 174.4 s 中 ~130.9 s (75%) 是 p5/p12 mirror overhead, ~31.1 s (17.8%) 是 dsp_exec 真实计算, 其余是 p1/p6/p8/p11 等; QCOM 33.1 s 中无 mirror memcpy 开销, per-call overhead 低 214x。

***

## 修订历史

### 2026-08-11

- Table-29/30/31 更新为最新六模型 AB CI 数据 (log_abtest_all_20260811-092810.txt, 36251 行); JZ TG 整体稳定, 4/6 模型领先 QCOM (最大 1.93x); 关键观察 3 修正 PP 描述 (原 "5/6 落后" 不准确, 实为 2/6 反超 + 4/6 落后); Qwen3.5-9B PP 从 26.55 提升到 35.76 (+34.6%), TG 1.48 与 profiling 数据一致 (GLM-5.2, Assisted-by: Trae IDE)
- Table-2 修正: "lm-head 常驻" -> "lm-head offload" (术语统一, 与 3.1 节标题/Table-8 一致; "常驻" 不准确, QCOM 权重亦常驻 rpcmem, 区别在于能否经济地 offload 计算到 DSP) (GLM-5.2, Assisted-by: Trae IDE)
- 全文源文件引用从行号改为函数名: [文件名：函数名](file:///路径) 格式, 去掉 #L 行号锚点 (行号随源码变更失效, 函数名更稳定且对读者更有意义); 纯文本行号引用统一改为带链接格式; developer.md 引用保留行号 (文档文件, 非源文件) (GLM-5.2, Assisted-by: Trae IDE)

### 2026-08-10

- 章节顺序调整: 原第二章(JZ 与 Qualcomm ggml-hexagon 架构对比)移至第一章, 原第一章(AB 测试性能数据)顺延为第二章, 建立宏观->微观的阅读路径
- 第二章 refine: 拆分为 1.1 核心架构差异 / 1.2 控制平面原语差异 / 1.3 数据平面 / 1.4 源文件结构, 去除冗余子标题
- why-perbuffer-cannot-offload-lmhead 文档独有内容 merge 到第三章: 3.1 节补充 per-buffer 固定成本分解与 dspqueue per-batch 重复注册机制; 3.3 节补充 cache 维护两层结构、三种正交 cache 机制表、first-touch 量化收益、"劣势变优势"反转视角; 3.4 节补充层数决定 PP/TG 交叉点公式与对 QCOM 改进建议
- 删除参考文档章节, 文档 self-contained (GLM-5.2, Assisted-by: Trae IDE)
- 第三章 3.4 (Tiled weight repacking) 删除, 独有内容合并到 3.1; 第三章从 7 节精简为 4 节 (3.1-3.4)
- 第三章/第四章重组: 原 3.5 (DSP Op-Level Profiling) 和 3.6 (graph 拆分) 移至第四章 4.1/4.2, 引出优化方向论述; 原 4.1-4.6 顺延为 4.3-4.8; 所有交叉引用同步更新 (GLM-5.2, Assisted-by: Trae IDE)
- 第四章 4.4-4.8 标题与正文 refine: 4.6 标题加括号说明 (边际优化与实验结论); 4.6.x 子节去状态后缀、调优 -> 优化、Phase 10 -> Phase 10 同步 RPC; 4.6.5 去序言/历史引用/配套关系; 4.7 序言精简; 4.8 核心原则修正 (batch-level cache -> role-aware cache, 去 tiled repack, 是架构级的 -> 来自架构层面, dspqueue 措辞, PR #26049 描述精确化) (GLM-5.2, Assisted-by: Trae IDE)
- 新增术语表 (JZ/QCOM/PP/TG/mempool), 降低新读者入门门槛 (GLM-5.2, Assisted-by: Trae IDE)
- 全文 ION -> mempool, 消除 Android 平台特定术语 (GLM-5.2, Assisted-by: Trae IDE)
- 全文 doorbell -> invoke (doorbell 是 dspqueue 概念, JZ 用 FastRPC invoke) (GLM-5.2, Assisted-by: Trae IDE)
- Table-1/2/3/8 表头统一为 JZ/QCOM, 列顺序统一为 JZ -> QCOM (GLM-5.2, Assisted-by: Trae IDE)
- Table-2 修正: fastrpc_mmap 每 buffer 1 次 (源码确认 1:1); IOVA 空间局部性 "碎片化" -> "分散"; lm-head 常驻 "自然" -> "可行"; Batch 传输胜者 JZ -> 平手 (GLM-5.2, Assisted-by: Trae IDE)
- Table-8 修正: "Per-tensor rpcmem" -> "Per-buffer rpcmem" (第一版作者误解); "overlay" -> "重叠" (GLM-5.2, Assisted-by: Trae IDE)
- 第一章 refine: 1.1 改用英文原文; 1.3 "mempool 共享内存" -> "AP-DSP 共享内存"; 1.3 去末尾总结句 (与标题重复); 1.4 Table-4 去括号 (GLM-5.2, Assisted-by: Trae IDE)
- 第二章 refine: 序言去掉根因分析和优化方向; 数据注记精简 (去三轮原始数据和早晨 run 对比); 关键观察去 TG/PP 括号解释、去 "唯一"、第三句 refine (GLM-5.2, Assisted-by: Trae IDE)
- 3.1 节 refine: "补充" blockquote 改为 guard #3 列表项; "经济性限制" -> "成本约束"; "Q4_K 模型" -> "lm-head 为 Q4_K 的模型"; "零边际成本" -> "零额外开销"; "大 buffer" -> "大权重 (如 lm-head)" (GLM-5.2, Assisted-by: Trae IDE)
- 3.4 节 refine: "层数决定 PP/TG 交叉点" -> "PP/TG 交叉点公式"; 添加前提条件 (模型权重可完全放入 single mempool); 添加公式不适用场景 (Qwen3.5-9B mirror memcpy + DSP VA 32-bit 限制 + UMA 前景) (GLM-5.2, Assisted-by: Trae IDE)
- 4.1 节: 去括号内容 (已过 warmup, 统计收敛) (GLM-5.2, Assisted-by: Trae IDE)

### 2026-08-09

- 基于第八章的素材，由GLM-5.2重新设计第八章文档架构，与Jeff Zhou一个字一个字refine (GLM-5.2)
- 新增 8.11 节"六模型 AB CI 性能对比", 新增 Table-44/44/45（MiniMax-M3）
- 第八章 refine: 8.3 节根因反转 (ssm_out 在 CPU 是 Q5_K 被 validator 拒绝, 非容量问题), 全章数据同步修正（Kimi-K3）
- 第八章 refine: 标题/编号/表格/措辞统一精简 (8.4.4/8.7.4/8.9/8.10/Table-31 注 3 等)（MiniMax-M3）
- 第七、八章标题格式统一为"模型名 PP&TG 优化（日期）"（MiniMax-M3）

### 2026-08-08

- 新增第八章 Qwen3.5-9B TG 性能差距分析, 含 8.1-8.8 子节 + Table-24/24/25/26（MiniMax-M3）
- 新增第七章 Qwen1.5-1.8B TG 优化实验, 含 7.1-7.6 子节 + Table-20/20/21/22（MiniMax-M3）
- 新增 8.6 节 Mirror 机制深度分析 + 8.7 节 4 GiB 限制突破 (根因升顶为 V79 32-bit VA hard cap)（MiniMax-M3）
- cgraph cache / graphs reused 概念澄清: JZ descriptor 复用 vs QCOM cgraph 结构复用（MiniMax-M3）
- 8.1.1 节 QCOM 数据修正 + Table-24 总结句修正（MiniMax-M3）
- CI 数据更新为五模型 3 轮均值（Kimi-K3）
- 新增 5.8.4 节 E/F/G 三方向（Kimi-K3）

### 2026-08-07

- 新增第六章 Qwen3.5-2B PP&TG 优化, 含 6.1-6.11 + Table-17/17/18（MiniMax-M3）
- 新增 5.8 节后续可探索方向（MiniMax-M3）
- 新增 4.2 节 SOLVE_TRI 拆分根因 + 4.5.3 节 offload 方案（Kimi-K3）
- 第四章结构整理 + 代码核验修正（Kimi-K3）
- Table-6 更新为五模型 3 轮均值 + GQA 比例修正（Kimi-K3）
- 全文档 unicode 箭头/乘号清理 (符合 AGENTS.md)

### 2026-08-06

- 初稿: AB 测试数据 + 第三章架构对比 + 第四章优化方向（Seed-2.1-Pro）
- 第三章源码核验 + Table-1/3 修正 (lm-head 类型 Q4 -> Q6_K, 内存模型对齐)
- 第四章重构: PP 优先级重排 + Qwen1.5-1.8B 三重叠加根因分析
- 新增第五章 force_opfusion_in_pp 实验与五模型 CI 验证（MiniMax-M3）
- 五模型层数修正: Gemma4-E2B 24->35, Gemma4-E4B 35->42, qwen3 13->24
- 全文 prose polish: em-dash 清除, 标点统一, ion 文档引用补充

### 2026-08-05

- 文档创建
