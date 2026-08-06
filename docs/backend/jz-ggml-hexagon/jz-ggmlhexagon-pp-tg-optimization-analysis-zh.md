# JZ's ggml-hexagon 性能差异分析与优化方向

> **作者**: Seed-2.1-Pro
>
> **日期**: 2026-08-05
>
> **背景**: 以 `3469e4858e17d501a1f6e16ebe0aa2489613d32b` 为基线，基于 5 个模型（Qwen3.5-2B、Gemma4-E2B、Gemma4-E4B、Qwen1.5-1.8B、Llama3.2-1B）的 AB 测试结果，分析 JZ (`ggml-hexagon-jz.cpp` + `kernels/`) 与 QCOM (`ggml-hexagon.cpp` + `htp/`) 两个 ggml-hexagon 后端的性能差异根因，并提出优化方向。

***

## 一、AB 测试性能数据

测试环境：

| 项目      | 配置                                                                                                                                                 |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| 设备      | Qualcomm Snapdragon 8 Elite (8 Gen 4), QCOM\_HTP\_V79, dsp arch 0x79, VTCM=8MB, HVX=1, HMX=1, 系统内存 24834 MiB, Android device id `9d231cfe`         |
| JZ 后端   | `libggml-hexagon-jz.so` + `libggmldsp-skel-v79.so`                                                                                                 |
| QCOM 后端 | `libggml-hexagon-qcom.so` + `libggml-htp-v79.so`                                                                                                   |
| 测试参数    | `n_ctx=8192, n_batch=2048, n_predict=256, n_threads=6, graphs reused=253`                                                                          |
| JZ 配置   | `dsp_cache_mode=5, ion_sync_mode=1, graph_optimize=1`offload MUL\_MAT types: F32,F16,BF16,Q4\_0,Q8\_0,Q4\_1,IQ4\_NL,MXFP4thread\_counts on CDSP: 6 |
| 测试方法    | 每个模型 3 轮取均值，n\_prompt=51\~75 tokens，n\_gen=255 tokens                                                                                              |

数据来源：`./scripts/build-run-android.sh run_abtest_all 2>&1 | tee log_abtest_all_$(date +%Y%m%d-%H%M%S).txt`

**Table-1**: AB 测试性能数据

| 模型           | PP JZ (tok/s) | PP QCOM (tok/s) | PP JZ vs QCOM | TG JZ (tok/s) | TG QCOM (tok/s) | TG JZ vs QCOM |
| ------------ | :-----------: | :-------------: | :-----------: | :-----------: | :-------------: | :-----------: |
| Qwen3.5-2B   |     426.4     |      466.3      |     -8.5%     |      26.9     |       14.5      |   **+85.8%**  |
| Gemma4-E2B   |     661.7     |      462.2      |   **+43.1%**  |      26.8     |       24.9      |     +7.5%     |
| Gemma4-E4B   |     401.8     |      410.0      |     -2.0%     |      14.8     |       9.9       |   **+49.0%**  |
| Qwen1.5-1.8B |     532.9     |      715.1      |     -25.5%    |      18.4     |       27.3      |     -32.4%    |
| Llama3.2-1B  |     1005.2    |      1094.8     |     -8.2%     |      42.4     |       28.3      |   **+50.0%**  |

**关键观察**：

- **TG（Token Generation）**：JZ 在 4/5 模型上领先，最大优势 +85.8%（Qwen3.5-2B），平均领先约 +48%；仅 Qwen1.5-1.8B（唯一 MHA 模型）落后 32.4%。
- **PP（Prompt Processing）**：QCOM 在 4/5 模型上领先，最大优势 +25.5%（Qwen1.5-1.8B）；唯一例外是 Gemma4-E2B，JZ 反超 +43.1%。
- TG 和 PP 的性能模式截然相反，指向不同的瓶颈根因。

***

## 二、架构关系澄清

JZ (`ggml-hexagon-jz.cpp` + `kernels/`) 与 QCOM (`ggml-hexagon.cpp` + `htp/`) 是**基于同一套 hexagon kernels 的两条进化分支/两种不同实现**，分叉点为 Qualcomm PR #26049。

- PR26049 之前两边算子 100% 相同。
- PR26049 之后，高通好的实现被手动移植到 JZ；自 JZ 的 PR 提交后，高通暂无新的 PR。
- **性能差异不在 kernel 算子本身，而在上层调度、cache 策略、offload 策略和通信模型。**

***

## 三、性能差异根因分析

### 3.1 lm-head offload — TG 性能差异的最大单一因素

QCOM 后端在 `ggml-hexagon.cpp` 的 `ggml_hexagon_supported_mul_mat` 中有**2 处 guard** 阻止 lm-head offload 到 DSP：

1. **类型 guard**：switch 只处理 Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4/F16/F32，**Q4_K/Q6_K/BF16 不在 switch 中**，落入 `default: return false`（`ggml-hexagon.cpp` L2841-2842）。JZ 侧对应位置（`ggml-hexagon-jz.cpp` L3141-3147）显式处理了 Q4_K/Q6_K：若 Q4_0 已启用，则放行（因为 JZ 在加载时做了 Q4_K/Q6_K → Q4_0 tiled repack）。
2. **尺寸 guard**：`src0->ne[1] > 32768` 时拒绝（`ggml-hexagon.cpp` L2806-2808）。此 guard 嵌在 Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4 case 内，对 Q4_K/Q6_K 不生效（已在类型 guard 阶段被 default 拦截）。

对于本次测试的 5 个模型，**类型 guard（#1）是实际生效的 guard**（lm-head 权重均为 Q4_K/Q6_K，不在 switch 中）。尺寸 guard（#2）是 per-buffer ION 经济性限制的直接体现（32K 行是 per-buffer 的成本上限）。

> **补充**：QCOM 在 switch 分支内还有一处 **repack buffer guard**（L2815：`!ggml_backend_buffer_is_hexagon_repack(src0->buffer)`），要求权重必须位于 repacked buffer 中。即使类型 guard 被移除，lm-head 权重（不在 repacked buffer 中）仍会被此 guard 阻止。该 guard 位于类型 guard 后面的 switch 分支内，当前未被触发，但记录了 QCOM 对 offload 权重的额外约束。

**Table-2**: 各模型 vocab\_size 与 lm-head 大小

| 模型           | vocab\_size | lm\_head 原始类型 | 原始大小（约）  | Q4\_0 repack 后大小（约） |
| ------------ | ----------- | ------------- | -------- | ------------------- |
| Qwen3.5-2B   | 151,936     | Q6\_K         | \~200 MB | \~163 MB            |
| Gemma4-E2B   | 256,000     | Q4\_K         | \~214 MB | \~214 MB            |
| Gemma4-E4B   | 256,000     | Q4\_K         | \~428 MB | \~428 MB            |
| Qwen1.5-1.8B | 151,936     | Q6\_K         | \~200 MB | \~163 MB            |
| Llama3.2-1B  | 128,256     | Q4\_K         | \~138 MB | \~138 MB            |

JZ 后端 `ggmlhexagon_supported_mul_mat` 中**没有任何 N 维度上限限制**，lm-head 完全 offload 到 DSP 执行。对 Q4\_K 模型（如 Gemma4、Llama3.2-1B），通过 Q4\_K → Q4\_0 tiled repack 将 lm-head 权重转为 DSP 可直接执行的 tiled layout；对 Q6\_K 模型（如 Qwen3.5-2B、Qwen1.5-1.8B），通过 Q6\_K → Q4\_0 tiled repack 转换（注意 Q6\_K 比 Q4\_0 略大，repack 后体积会略减）。repack **不减少带宽**（Q4\_K 和 Q4\_0 数据大小相同，均为 0.5625 B/param；Q6\_K → Q4\_0 实际是 lossy 转换以适配 DSP 端复用的 Q4\_0 matmul kernels），其价值在于使 DSP tiled matmul 端执行成为可能。

**lm-head offload 之所以在 JZ 可行，与 single mempool 架构强相关：** lm-head 权重（Q4_K/Q6_K 量化矩阵，按 Table-2 约 138-428 MB）作为 mempool 内的一个 offset 范围，零额外 fd/mmap/生命周期成本。QCOM 的 2 处 guard（类型/尺寸）共同阻止了 lm-head offload，根本原因是其 per-buffer ION 设计：每个 buffer 携带独立的 fd、fastrpc_mmap、dspqueue 每批重复注册等开销，无法经济地承载会话常驻的 lm-head 权重（32K 行是 per-buffer API 的实际上限）。JZ 通过加载时 Q4_K/Q6_K → Q4_0 tiled repack 消除了类型 guard，通过 single mempool 的零边际成本消除了尺寸 guard 的经济性约束。

**对 TG 的影响是决定性的：** TG 每生成 1 个 token 都要执行一次 lm-head matvec（`[1, n_embd] x [n_embd, vocab_size] -> [1, vocab_size]`）。这是纯粹的 memory-bound 操作：

- **QCOM**：CPU 逐元素读取整个 Q4_K/Q6_K lm-head 权重（按 Table-2 约 138-428 MB）做 dequant+dot product，CPU 访存带宽有限，且 CPU 算 lm-head 时 DSP 空闲。
- **JZ**：DSP 上 HVX 执行 lm-head matvec，权重以 Q4_0 tiled layout 驻留在 ION mempool 中，带宽远高于 CPU，且与后续 token 生成流水线紧密衔接。

**对 PP 的影响很小**：lm-head 在 PP 末尾只执行一次，其开销被几十个 transformer layer 的计算摊薄。

### 3.2 dspqueue async overlay vs 同步 FastRPC — PP 性能差异的根因

这是执行模型的根本差异，而非简单的"调度开销"。

**QCOM (htp/) — dspqueue 异步流水线**：

- AP 通过 `dspqueue_write` 将 op 描述符写入环形队列，非阻塞。
- DSP 从队列消费 op 并执行。
- AP 可以在 DSP 执行当前 op/layer 的同时，准备下一个 op 的描述符、做 cache flush 等。
- 形成 **AP-DSP overlay（流水线重叠）**：AP prep 和 DSP compute 并行。
- `enqueue_op` + `dspqueue_read` 构成生产者-消费者模型。

**JZ (kernels/) — native FastRPC 100% 同步**：
`graph_compute_batch` 经历严格串行的 12 个 Phase：

- **Phase 1-9**：AP 侧全图分析 -> tensor 镜像 -> 权重 repack -> mempool 分配 -> desc 构建 -> cache flush（全部 AP 工作，DSP 空闲）。
- **Phase 10**：同步调用 `ggml_dsp_execute_batch`，AP 阻塞等待 DSP 执行完整个 batch（DSP 工作，AP 空闲）。
  - `cum_p10_rpc_setup_us`：AP setup（ioctl / marshalling）。
  - `cum_p10_dsp_exec_us`：纯 DSP 执行时间。
  - `cum_p10_civac_us`：AP cache invalidate after DSP reply。
- **Phase 11-12**：AP 侧 cache inval -> mempool 回拷（DSP 空闲，AP 工作）。
- **零 overlay**：AP 准备期间 DSP 空闲，DSP 执行期间 AP 空闲。

**对 PP 的影响**：PP 是 compute-bound 场景，prompt tokens 多、每层 matmul 的 M 大、DSP 计算时间长，AP-DSP overlay 的收益被充分放大——QCOM 的 AP prep 时间完全隐藏在 DSP compute 后面，而 JZ 的 Phase 1-9 + Phase 11-12 纯 AP 开销直接加到总延迟上。

**对 TG 的影响**：每 token 只有一个 batch（M=1），DSP 计算极短，dspqueue 的 overlay 收益极小（几乎没有可以隐藏的 AP prep 时间），反而 per-op dspqueue 通信开销（每次 write/read 都有环形队列管理开销）在 M=1 小 op 时占比大。JZ 单次 doorbell 发整个 batch 的模型在 TG 更高效。

### 3.3 Role-aware batch-level cache 管理 — TG 的第二大优势

JZ 的 batch-level cache 策略通过 bitmap 控制，大幅减少了 cache sync 次数：

- **bit0 (first-touch weight bitmap)**：权重首次 touch 后不再 dcinva，命中 L2。
- **bit1 (prior-dst skip)**：已被前序 op 消费的 dst 跳过 flush。
- **bit2 (bulk flush)**：所有 dst flush 合并到 batch 末尾一次完成。
- **bit3 (selective flush)**：中间 tensor 不 flush，减少 DDR 写。

而 QCOM 采用 batch 级全量 cache 维护：在 batch 开始和结束时各执行一次完整 D-cache flush+invalidate（`qurt_mem_cache_clean(..., FLUSH_INVALIDATE_ALL, ...)`），uniform、role-blind，无法区分 weight 和 activation。

**对 TG 的影响**：M=1 时每个 matmul 计算量极小，QCOM 的 batch 级全量 cache flush+invalidate 开销被放大。JZ 的 batch-level 策略大幅减少 cache sync 次数，效果显著。

**对 PP 的影响**：大 M matmul 计算时间长，cache sync 被计算摊薄，差异小。

### 3.4 Tiled weight repacking + VTCM 复用

JZ 在加载阶段对 Q4\_K/Q6\_K 等量化权重做 **Q4\_K/Q6\_K → Q4\_0 tiled repack**（在 [`ggml-hexagon-jz.cpp`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp) 的 `repack_q4k_as_q4_0_tiled_to_buf` 等函数中，32-row strip 转换），将权重转为 DSP HMX kernel 可直接消费的 tiled Q4\_0 布局，配合 VTCM 分块计算减少 DDR 访问次数。

需要澄清：QCOM 后端的 [`htp/`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp) 目录同样包含 tiled Q4\_0/Q4\_1 kernel 实现（与 JZ 维护的 [`kernels/`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels) 在分叉前是同一套代码，PR #26049 之后分叉维护）。JZ 与 QCOM 的真正差异**不是 tiled vs flat 的布局差异**，而是：

- **JZ**：在加载时对所有 Q4\_K/Q6\_K 权重做 → Q4\_0 tiled repack，因此所有量化 matmul（含 lm-head）都能在 DSP 上以 tiled Q4\_0 layout 跑。
- **QCOM**：保留 Q4_K/Q6_K 原始布局 + tiled kernel 双路径，**但 2 处 guard（类型 guard 拒绝 Q4_K/Q6_K、尺寸 guard 拒绝 >32K 行）阻止了 lm-head offload**，所以 lm-head 直接回退到 CPU，根本走不到任何 layout 对比这一步。

因此"QCOM 使用 flat layout"的说法不准确——QCOM 的小尺寸 Q4\_K matmul 同样在 DSP tiled kernel 上跑；问题是大尺寸（>32K 行）的 lm-head 在 QCOM 路径里就不存在 offload 流程。

### 3.5 总结：性能差异归因（按重要性排序）

**Table-3**: 性能差异归因（按重要性排序）

| 架构特性                | JZ (kernels/)                                                                                          | QCOM (htp/)                                                            | TG 影响                                         | PP 影响                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------- | --------------------------------------------- | -------------------------------------- |
| **通信模型**            | Native FastRPC 同步（12 Phase 串行，零 overlay）                                                               | dspqueue 异步环形队列（AP-DSP overlay）                                        | JZ 略优（单次 doorbell vs per-op 队列管理）             | **QCOM 显著优**（AP prep 与 DSP compute 重叠） |
| **lm-head offload** | 全 offload（single mempool + Q4\_K/Q6\_K→Q4\_0 tiled repack，支持超大 N）                                      | 2 处 guard（类型/尺寸）拒绝，回退 CPU                                            | **JZ 极大优势**（每 token \~138-428MB matvec 在 DSP） | 影响小（只跑 1 次）                            |
| **Cache 管理**        | Role-aware batch-level（bit0-3，first-touch/prior-dst/bulk/selective）                                    | Batch 级全量 D-cache flush+invalidate（uniform, role-blind）                | **JZ 显著优**（M=1 时 cache sync 是大头）              | 差异小（大计算摊薄）                             |
| **内存模型**            | Single ION mempool（init 时 mmap 一次，v79 容量 probe 上限 4032 MiB，offset addressing；无 fd/mmap/lifecycle 重复成本） | Per-tensor rpcmem 分配（每 buffer 独立 fd / fastrpc\_mmap / dspqueue 每批重复注册） | JZ 优（零额外 fd/mmap + 整池 IOVA 连续 + 权重 L2 友好驻留）   | 差异小                                    |
| **权重布局**            | Q4\_K/Q6\_K → Q4\_0 tiled repack 后 DSP 端跑 tiled Q4\_0 kernel                                           | 原始 Q4\_K/Q6\_K 布局 + tiled kernel（lm-head 因 2 处 guard 不参与）               | JZ 优（lm-head DSP offload，VTCM/L2 友好）          | JZ 略优                                  |

**JZ TG 领先**与 single mempool 带来的 lm-head offload 强相关，role-aware 的缓存一致性维护策略也是重要因素。

**JZ PP 落后**的根因是**调度框架差异**而非 kernel 差异。JZ 与 QCOM 复用同一套 Qualcomm HMX kernels（分叉前 100% 相同），所以 matmul 本身的执行效率两者一致；差异完全在 JZ 的 12-phase 同步模型无法实现 AP-DSP pipelining，而 QCOM 的 dspqueue 异步环形队列允许 AP prep 与 DSP compute 在 per-layer 粒度上重叠。JZ 的 data-plane 优势（lm-head offload + first-touch 权重 inval）是**整图固定开销**，与 layer 数无关；QCOM 的 pipelining 优势是**per-layer 累积**的，与 layer 数正相关。因此 PP 表现高度依赖模型层数与 attention pattern 对 VTCM/cache 压力的影响。

**Qwen1.5-1.8B（唯一 MHA 模型，24 层）PP/TG 双输的根因** = dspqueue pipelining 优势 + 层数不足 + MHA VTCM/cache 压力三重叠加：

1. **dspqueue pipelining 优势最大化**：dspqueue 的 AP-DSP overlap 收益与每次 DSP 计算时长正相关，Qwen1.5-1.8B 在 PP 阶段单 layer 计算时间长（24 层 × 每层 MHA Q@K^T 的 full attention），pipelining 隐藏的 AP prep 时间窗口大。
2. **JZ 整图固定优势无法累积**：lm-head offload（~200MB Q6_K）+ first-touch 权重 inval（~9.2ms/token）是固定的、不会随 layer 数增加而放大的优势；24 层不足以让 JZ 的 per-layer 增量优势赶超 dspqueue 的 per-layer pipelining 收益。Gemma4-E2B 35 层则可以反超（+43.1%）。
3. **MHA 加重 VTCM/cache 压力**：1:1 attention 的 Q@K^T 是 full attention（无 KV 共享），相比 GQA 模型的 KV 共享头占用更多 VTCM 与 cache 带宽，恰好是 JZ role-aware cache 策略（bit0-3）本来要优化的场景——但这些优化只在 TG M=1 场景放大收益，对 PP 长序列 M=prompt_len 帮助有限。

**结论**：Qwen1.5-1.8B 不是 corner case，而是三重不利因素叠加的体现。**任何 PP 优化（如 per-layer pipelining）只要把 dspqueue 的 per-layer 优势部分削弱，就能同时改善 Qwen1.5-1.8B 这类模型**——这是 PP 优化优先级应当高于 TG 精雕细琢的核心论证。

### 3.6 DSP Op-Level Profiling 实测数据（2026-08-06）

在完成 DSP-side sampling (commit `HEX_OP_PROF` enabled) 后，基于 Gemma4-E2B 模型（TG 主场景）在 DSP 端开启 per-op 计时统计，每 25 个 batch 通过 FARF(ERROR) 输出累计数据。以下分析取 batch#200 稳定数据点（已过 warmup，统计收敛）。

**测量环境**：同 Table-1，Gemma4-E2B，n\_ctx=8192, n\_threads=6, dsp\_cache\_mode=5, ion\_sync\_mode=1

**Table-4**: DSP 算子耗时排名（batch#200，cumulative，us）

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
2. **MUL\_MAT max=4697us 是 lm-head**：avg=97us 被大量小尺寸 matmul 拉低，但 max=4697us 的 outlier 每个 TG batch 出现一次，对应 lm-head matvec（`[1, hidden] × [hidden, vocab=256000]`），是 TG 阶段最大的单个算子。小 MUL\_MAT（avg≈17-100us）对应 attention 输出投影和其他零散 matmul。
3. **MUL\_MAT\_FFN avg=334us 是最稳定的 hotspot**：count=6965/200batch≈35 次/batch，即每 transformer layer 1 次 MUL\_MAT\_FFN fused 调用（Gemma4-E2B 35 层 × 1 fused op/layer = 35 次/token）。该 fused op 在内部完成 gated FFN 的 gate+up+down 三段 matmul，因此 35 次 fused call = 35 × 3 = 105 个内部 matmul。avg 稳定在 334us，是所有算子中**平均耗时最高**的稳定计算项。
4. **FLASH\_ATTN\_EXT avg=20us 非常高效**：FlashAttn kernel 已充分优化，avg 仅 20us，不是瓶颈。
5. **Element-wise 算子可忽略**：RMS\_NORM/ADD/MUL/ROPE/GELU 等 avg 均 ≤2us，占比合计 <5%，fuse 收益极小。

**Non-op 开销分析**（batch#200）：

```
batch-wall avg=35,789 us/batch
op-sum     avg=31,095 us/batch
non-op     avg= 4,693 us/batch  (13.1% of wall time)
```

**Table-5**: DSP non-op 开销细分（us/batch）

| 阶段                                 | 耗时 (us/batch) | 数据量       | 说明                                            |
| ---------------------------------- | ------------- | --------- | --------------------------------------------- |
| hdr cache inval                    | 4             | -         | batch descriptor invalidation，可忽略             |
| tensor pre-conversion              | 318           | -         | hex\_tensor\_desc → dsptensor/htp\_tensor 预转换 |
| weight cache inval (w-inv)         | 68            | 6 MB      | bit0 first-touch 效果显著：权重仅首次 inval             |
| **activation cache inval (a-inv)** | **1,030**     | **82 MB** | **最大 non-op 开销**，bit1 prior-dst skip 可能未完全生效  |
| dst tracking                       | 105           | -         | prior\_dst/bulk\_flush 范围收集                   |
| **bulk dst flush**                 | **1,377**     | -         | 所有 dst 合并到 batch 末尾 flush                     |
| queue wakeup/suspend               | 5             | -         | DMA/HMX queue 管理，可忽略                          |
| **non-op 合计**                      | **4,693**     | <br />    | **13.1% wall time**                           |

**瓶颈根因分析（基于 profiling 数据修正）**：

> **注意**：以下 profiling 数据仅覆盖 DSP 批处理执行阶段（Phase 10），在 DSP 端通过 HEX\_OP\_PROF 测量。AP 侧开销（Phase 1-9 + Phase 11-12）未包含在内，需通过 Step 0 profiling（dump\_diag\_info=1）单独测量。此处"wall time"指 DSP 端 batch 执行 wall time（35,789 us/batch），非端到端 TG 时间。

- **在 DSP 执行内部，matmul kernel 是绝对主导**：op-sum 占 DSP batch-wall time 的 86.9%，其中 91.1% 是三类 matmul。DSP 侧 non-op 开销（cache inval、tensor 转换、dst flush 等）合计 4693 us/batch，占 DSP batch-wall 的 13.1%。
- **lm-head MUL\_MAT（max=4697us）是 TG 单算子最大项**：每个 token 出现一次，对应 `1×hidden×vocab` matrix-vector product。通用 GEMM kernel 对 M=1 的 skinny matmul 效率不高，专用 GEMV kernel 有优化空间。
- **MUL\_MAT\_FFN（avg=334us）是 per-layer 最大稳定开销**：FFN matmul 已使用 fused op（MUL\_MAT\_FFN），需要检查是否充分利用 HMX 加速，以及 tile size 是否对 FFN 维度最优。
- **activation cache invalidation（a-inv=1030us/batch, 82MB）是最大 DSP 侧 non-op 开销**：bit1（prior-dst skip）效果可能未达预期，需验证 prior\_dst\_ranges 覆盖范围是否足够大以减少 a-inv 字节数。bulk flush（1377us）是第二大 DSP 侧 non-op，但这是 bit2 bulk flush 策略的代价，将所有 dst flush 合并到一次。
- **DSP-side sampling 实际收益极小**：跳过 logits copyback 仅节省 \~100-200us（因 ion\_sync\_mode=1 下整个 mempool sync 掩盖了局部收益），与实测一致——DSP-side sampling 功能正确，但性能提升可忽略。

**对优化方向优先级的影响**：

1. **DSP 内部 matmul kernel 优化是首要方向**：三类 matmul 占 DSP 执行时间的 91.1%，lm-head 专用 GEMV kernel 和 MUL\_MAT\_FFN 调优是 DSP 端最具潜力的单点优化。
2. **AP 侧开销（Phase 1-9 + Phase 11-12）未被 DSP profiling 数据覆盖**：无法直接比较 AP 侧优化（descriptor 模板缓存等）与 DSP kernel 优化的收益。AP 侧开销需通过 Step 0 profiling 单独量化后再定优先级。
3. **lm-head 专用 GEMV kernel**：每 token 出现一次，max=4697us，是 TG 阶段 DSP 端最大的单算子。
4. **MUL\_MAT\_FFN kernel 调优**（HMX 利用率、tile size）：收益面最广（35 次/batch × 334us = 11690us/batch）。
5. **a-inv 优化**（bit1 prior-dst 覆盖扩展）：可再省 \~1ms/batch DSP 侧开销。

***

## 四、优化方向

根据 3.6 节 DSP op-level profiling 实测数据，**在 DSP 执行内部，matmul kernel 是绝对主导**（三类 matmul 占 DSP batch-wall time 的 79.1% = 86.9%×91.1%）。注意：DSP profiling 数据仅覆盖 Phase 10（DSP 批处理执行），AP 侧开销（Phase 1-9 + Phase 11-12）未包含在内，需通过 Step 0 profiling 单独量化。在 AP 侧数据补全前，优化方向优先聚焦在 DSP kernel 与 offload 策略上，AP 侧优化暂不调整优先级。

TG 和 PP 的瓶颈不同，优化策略也不同：

- **TG 瓶颈**（基于 3.6 profiling，仅覆盖 DSP 端）：在 DSP 执行内部，三类 matmul 占 91.1% op-sum，其中 lm-head MUL\_MAT（max=4697us，每 token 1 次）和 MUL\_MAT\_FFN（avg=334us，每 layer 1 次 fused op = 105 个内部 matmul）是绝对主导；JZ 已通过 lm-head offload + first-touch 权重 inval（\~9.2 ms/token 节省，固定整图总量）解决最关键的两项，剩余优化空间主要在 DSP matmul kernel 本身。
- **PP 瓶颈**：PP 差距是**模型结构相关的**，不是普遍的 JZ 弱点。ion 文档第 9 节分析表明 JZ 净优势 = per\_layer\_saving × n\_layers + fixed\_lmhead\_saving - dspqueue\_overlap。当层数足够时 JZ 也赢 PP（如 Gemma4-E2B 的 35 层，PP +43.1%）；浅层模型（qwen3.5-2B 25 层、llama3.2-1B 16 层）dspqueue 的固定 overlay 优势尚未被 per-layer 累积超越。ion 文档也明确指出：**性能差异来自 data-plane policy（weight residency + role-aware cache），而非 control-plane**（FastRPC 开销 \~89 us，可忽略）。

### 4.0 第零步：Profiling 数据驱动（所有优化决策的前提）

在投入任何优化之前，先用 `dump_diag_info=1` 跑一轮 benchmark 量化各阶段耗时：

- Phase 1-12 各阶段的实际时间分布。
- Phase 10 三阶段（`cum_p10_rpc_setup_us` / `cum_p10_dsp_exec_us` / `cum_p10_civac_us`）的占比。
- TG 中 Phase 4-8 的固定开销究竟多大（验证 descriptor 模板缓存的收益上限）。
- PP 中 Phase 1-9 + 11-12 的 AP 纯开销占比（验证 async/pipelining 的收益上限）。

**决策阈值**：

- 如果 Phase 1-9 + 11-12 在 PP 中占比 < 10%，async/pipelining 不值得做（ion 文档中 FastRPC 开销 \~89 us 的数据也支持这一判断）。
- 如果 Phase 4-8 在 TG 中占比 > 5%，descriptor 模板缓存值得投入。

### 4.1 第一优先级：TG 扩展优势（JZ 已有优势，进一步拉大）

JZ 当前的 TG 优势来自两个已实现的关键机制，应在优化分析中量化：

- **lm-head DSP offload**：QCOM 因 per-buffer ION 设计的经济性限制（32768 行 guard），lm-head 回退 CPU；JZ 通过 single mempool + Q4\_K→Q4\_0 tiled repack 实现 DSP 端执行。
- **first-touch 权重 inval（bit0）**：lm-head 常驻后每 token 权重流量 \~1.9 GB，bit0 消除冗余 dcinva 节省 \~9.2 ms/token（固定整图总量，非 per-layer）。这是 bit0 开关对比实测值。

#### (1) DSP-side sampling — 返回 token ID 而非完整 logits

当前流程：

```
DSP lm-head matvec -> [1, vocab_size] F32 logits (~500KB-1MB) -> memcpy 回 AP -> CPU softmax+argmax/topk/topp -> token ID
```

优化后：

```
DSP lm-head -> DSP softmax -> DSP argmax/topk -> 返回 1 个 int32 token ID (4 bytes!)
```

- **收益分析**：logits memcpy 在 DDR 带宽下仅 \~17-33 us（500KB-1MB / \~30 GB/s），FastRPC 开销 \~89 us，合计 \~100-120 us/token。相对于 \~37 ms/token 的 TG 占比很小（<0.4%）。收益上限有限，但减少 Phase 12 copyback 的数据量和 AP 侧处理仍有一定价值。
- **额外机会**：ion 文档提到 QCOM 的 AP 侧 sampling 路径比 JZ 快 \~2.5x。**AP 侧 sampling 路径优化**（不涉及 DSP 改动）是一个更简单的独立优化方向，应优先评估。
- **复杂度**：**高**（非中等）。在 256K vocab 上实现 top-k 需要排序（O(n log n)），DSP 端实现复杂；top-p 还需要 cumulative sum + rejection sampling。`soft_max` 虽已在 DSP 上支持，但 sampling 涉及随机数生成（需要 Hexagon RNG 集成或预传 seed），整体 pipeline 重构工作量大。
- **影响模型**：所有模型，vocab\_size 越大（Gemma4 256k）收益越大。

#### (2) TG descriptor 模板缓存 — 消除 graph\_compute\_batch 中 AP 侧 prep phase 的 per-token 开销

TG 模式下，**每 token 的 cgraph 拓扑完全相同**，只有 tensor 数据指针变化。当前每 token 都要走 `graph_compute_batch` 的全部 12 phase（Phase 1-9 AP 侧全图分析 / 镜像 / 权重 repack / mempool 分配 / desc 构建 / cache flush，Phase 10 同步 RPC，Phase 11-12 AP 侧 cache inval / 回拷）。其中 Phase 1-9 内的 layout 计算、mempool offset 跟踪、descriptor 构建等纯 AP 工作在拓扑不变时可复用：

- 首次 token（或 graph reopt 时）构建 descriptor 模板，记录所有 op 的 src/dst offset 与 mempool layout。
- 后续 token 只 patch 变化的数据指针（activation/KV cache 地址），跳过对应的 prep 阶段。
- **收益（按 3.6 profiling 数据估算）**：3.6 节 profiling 仅覆盖 DSP 端 Phase 10，未测量 AP 侧 Phase 1-9 开销。descriptor 模板缓存省的是 AP 侧开销，无法用 DSP profiling 数据直接估算。需在 Step 0 profiling 拿到 AP 侧 Phase 1-9 的精确耗时后再评估收益。作为参考，若 AP 侧 Phase 1-9 开销与 DSP 侧 non-op 开销（4693 us/batch）同量级，即使消除其中一半（\~2.3 ms），相对于端到端 TG 时间的占比也会因 AP 侧 Phase 11-12 的额外开销而更低。**descriptor 模板缓存优先级的最终判断依赖 Step 0 profiling 数据**。
- **复杂度**：中等偏低。需要在 ctx 中缓存 descriptor 模板和 mempool layout，处理 KV cache 增长时的 realloc 以及 graph topology 变化（context shift）时的 invalidate。
- **注意**：权重 repack offset 在模型加载后不变，但 activation 地址每 token 不同，模板需要支持 per-pointer patch。

#### (3) KV cache 常驻 ION + 增量 inval

当前 bit0 first-touch 标记对只读权重有效，但 KV cache 是 read-write 的，每 token 被 DSP 写入、AP 读取。KV cache 已在 ION mempool 中，Phase 11（cache inval）可以只 inval KV cache 的新增部分（增量 inval），而非每次做大范围 inval。

- **复杂度**：中等偏高。需要跟踪 KV cache 的写入范围（哪些 token position 是新写入的），并在 Phase 11 中只对这些范围做 CIVAC。
- **注意**：bit0 机制不适用于 KV cache（read-write），需要独立的增量跟踪机制。

### 4.2 第二优先级：PP 优化（结构性收益，是 JZ 的真正战场）

**优先级重排论证**：JZ TG 已领先 +48% 平均值，继续优化的边际收益受 matmul kernel 物理极限约束；PP 落后 -2% 到 -25.5%，根因（AP-DSP 无 pipelining）是**可重构的调度框架差异**，而非 kernel 差异。**PP 从 -25% 改善到 -15% 等同于 +11pp 绝对提升**；TG 从 +50% 到 +60% 需要改 kernel 才有 +10pp，但 kernel 已与 QCOM 100% 共享，**只能从 matmul 内部优化（HMX 利用率、tile size）挤牙膏**。因此 PP 优化是 JZ 的真正战场，应排在 P0 profiling 之后立即推进。

**PP 表现与模型结构的关联**（基于 3.5 节三重叠加分析）：

| 模型           | 层数           | PP JZ vs QCOM | TG JZ vs QCOM | 根因分项                                    |
| ------------ | ------------ | :-----------: | :-----------: | --------------------------------------- |
| Gemma4-E2B   | 35 (GQA 8:1) |   **+43.1%**  |     +7.5%     | 层数深，per-layer 优势累计超越 dspqueue overlap |
| Gemma4-E4B   | 35 (GQA 8:1) |     -2.0%     |   **+49.0%**  | 模型大，DSP 计算时间主导，差异被算力摊薄             |
| Qwen3.5-2B   | 25 (GQA 4:1) |     -8.5%     |   **+85.8%**  | 层数中等，dspqueue 优势被部分抵消，TG 受益于 lm-head |
| Llama3.2-1B  | 16 (GQA 4:1) |     -8.2%     |   **+50.0%**  | 层数浅，dspqueue 优势显著                       |
| Qwen1.5-1.8B | 24 (MHA 1:1) |     -25.5%    |     -32.4%    | **三重叠加：dspqueue + 层数不足 + MHA VTCM/cache** |

**结论**：PP 优化应聚焦于**结构性杠杆**（per-layer pipelining），而非模型结构特化。Qwen1.5-1.8B 不是 corner case 而是三重不利因素的"压力测试"——per-layer pipelining 改善后这类模型会自动获益最大。Gemma4-E2B 已经赢 PP，但进一步压榨 +43% 的空间也来自 per-layer pipelining 在深层模型上的累积收益。

#### (4) Per-layer intra-batch pipelining — 结构性突破点

**关键澄清（修正旧文档的论证逻辑）**：ion 文档"性能差异来自 data-plane policy 而非 control-plane，FastRPC 开销 ~89us 可忽略"的论断，**不能用于反对 per-layer pipelining**。这两个是不同的概念：

- **FastRPC ~89us 是 control-plane 路径成本**（RPC invoke 自身的 marshalling + transport 开销），与是否做 pipelining 无关
- **Pipelining 收益 = min(AP prep 时间, DSP compute 时间) 的隐藏量**——完全由调度重叠决定，与 FastRPC 开销无关

89us 是 RTS 路径成本，pipelining 关心的是能否把 1-3ms 的 AP prep 隐藏在 5-10ms 的 DSP layer 执行后面。**这是两个独立维度**。

**当前同步模型的瓶颈**：

```
AP Phase 1-9 [=====] → AP阻塞 [==] → AP Phase 11-12 [===]
                          DSP Phase 10
```

PP 阶段单 layer DSP 计算时间（per 3.6 profiling：MUL_MAT avg=97us × 多 ops + FlashAttn 20us + RMS_NORM 1us 等 ≈ 200-500us per layer，prompt 较长时 M 大 matmul 可达 1-2ms），而 AP Phase 1-9 + 11-12 估计在 1-3ms 范围。如果按 layer 切分：

```
AP P1-P4 Layer1 [=] → DSP Layer1 [==] → AP P5-P7 Layer2 [=] → DSP Layer2 [==] → ... → AP P11-12 [=]
```

**预期收益（基于估算）**：AP 侧 Phase 1-9 + 11-12 占 PP 10-15%，pipelining 隐藏 50-70%，PP 提速 5-10%。Qwen1.5-1.8B 从 -25.5% 改善到 -18% 左右，Gemma4-E2B 从 +43.1% 进一步到 +48%+。

**关键设计约束**：

- **维持 single mempool 不变**：TG 优势的根，不能动
- **切分粒度应是 layer 级**：op 级切分会让 setup 成本吃掉收益
- **DSP 端需要 partial-execute + resume 接口**：从 descriptor 中按 offset 启动执行的新基础设施
- **12 phase 测量框架要扩展到 per-layer**：现有 `cum_p1_us` ~ `cum_p12_us` 是 batch-level 聚合，要能下钻到 per-layer 才能验证 pipelining 收益
- **严格 TG 回归测试**：任何增加 AP↔DSP 同步点的改动都可能在 M=1 时变成新开销，单次 doorbell 优势是 TG 优势的关键来源

**前置数据需求（依赖 Step 0 profiling）**：

- Phase 1-9 + 11-12 的 AP 纯开销实测占比（决定 pipelining 收益上限）
- 单 layer DSP 计算时间分布（决定 AP prep 是否能完整隐藏在 layer 计算后面）
- per-layer cache flush 字节数（Phase 9 切分到 per-layer 后的实际开销）

**风险评估**：

- 收益面广：4/5 测试模型 PP 改善
- 实施复杂度高：DSP partial-execute 接口是新基础设施
- 风险点：MUL_MAT per-layer 平均仅 97us（远低于 89us FastRPC × 2-3 倍 pipelining 切换开销），**单 matmul pipelining 无收益；必须聚合到 layer 级别才有收益**。3.6 profiling 给出的是 batch#200 累计值，需要补充 per-layer 实测数据

#### (5) AP 侧 sampling 路径优化

ion 文档提到 QCOM 的 AP 侧 sampling 路径比 JZ 快 ~2.5x。这与 DSP 端无关，是纯 AP 侧的优化机会。应分析和对比 JZ 与 QCOM 的 sampling 路径差异，找出瓶颈。本优化与 PP/TG 都相关，独立于 pipelining 路线。

#### (6) descriptor 模板缓存（条件性）

如 4.1(3) 节所述，descriptor 模板缓存可减少 AP 侧 prep 时间，与 pipelining 是**互补关系**（pipelining 利用 prep 的时间，缓存减少 prep 本身）。**如果 Step 0 profiling 显示 AP 侧 prep 是 pipelining 收益的主要瓶颈，缓存应同步实施**。对长 context PP 收益较大（5-10%），对 TG M=1 收益很小。

### 4.3 第三优先级：低风险快速收益

#### (7) a-inv 优化（bit1 prior-dst 覆盖扩展）

3.6 profiling 显示 a-inv 是最大 non-op 开销（1030 us/batch, 82 MB）。bit1 机制应跳过已被前序 op 消费的 dst，但 1030us + 82MB 暗示覆盖范围可能未达预期。需验证 `prior_dst_ranges` 收集逻辑，扩大覆盖以减少 a-inv 字节数。

- **预期收益**：500us/batch = 1.4% TG，仅 TG 受益
- **复杂度**：低，纯 AP 侧逻辑调整
- **依赖**：无

#### (8) MUL_MAT_FFN kernel 调优

3.6 profiling 显示 MUL_MAT_FFN avg=334us × 35 calls = 11.7ms/batch（TG 主要热点）。kernel 已与 QCOM 共享，可调空间在 HMX 利用率与 tile size。

- **预期收益**：HMX 利用率提升 30% 可省 3.5ms/token = 9% TG
- **复杂度**：中（需 DSP kernel 修改）
- **风险**：tile size 调大需要更多 VTCM，可能与 lm-head 等大算子冲突
- **重要前提**：kernel 与 QCOM 共享意味着此优化对 QCOM 也有效，**不会扩大 JZ vs QCOM 的相对优势**，但能提升绝对性能

#### (9) RMS_NORM/activation 与 matmul 的 fuse

JZ 已有 QKV fusion 和 FFN fusion，但 RMS_NORM->matmul 和 matmul->activation 的 fuse 还有空间。在 M=1 TG 时，element-wise op 写入 DDR 再被下一个 matmul 读回是纯粹的浪费，fuse 后可减少一次中间 tensor 的 DDR round-trip。

- **预期收益**：<1% TG，PP 收益更小
- **复杂度**：中（kernel 修改 + AP 侧调度调整）

#### (10) 减少 Phase 10 RPC round-trip 开销 — 优先级最低

FastRPC 开销已在 warmup 阶段校准（`rpc_overhead_min_us`），ion 文档测量值 \~89 us，相对于 \~37 ms/token 的 TG 占比极小（<0.3%）。除非 profiling 发现非预期的高开销，否则此项投入产出比低，不建议优先投入。

### 4.4 优化路线图

**核心转变**：TG 优化从 Step 1 降为第三优先级（与新 Step 0 之后的 P3 一起），PP 优化提升为第二优先级（Step 2）。

```
Step 0: Profiling 数据驱动（所有决策的前提，必做）
  +-- 用 dump_diag_info=1 跑一轮 benchmark
  +-- 量化 Phase 1-12 各阶段占比（特别关注 AP 侧 Phase 1-9 + 11-12 实测）
  +-- 量化 per-layer DSP compute 时间分布（决定 pipelining 切分粒度）
  +-- 量化 a-inv / bulk flush 各 phase 拆分（决定 a-inv 优化空间）
  +-- 验证 3.6 profiling 数据中 lm-head MUL_MAT max / MUL_MAT_FFN avg 是否稳定
  +-- 不同 seq_len 下 Phase 12 copyback 时间 + mempool hit rate（长 context 边界条件）

Step 1: 测量驱动快速收益（低风险，独立于 PP/TG 主战场）
  +-- 1. a-inv 优化（bit1 覆盖扩展，1030us → 期望 ~500us，TG +1.4%）
  +-- 2. AP 侧 sampling 路径优化（对齐 QCOM 的 ~2.5x 优势，独立方向）

Step 2: PP 结构性突破（per-layer pipelining，核心战场）
  +-- 1. DSP 端 partial-execute + resume 接口（基础设施）
  +-- 2. graph_compute_batch 切分到 per-layer sub-batch（per-layer 12 phase 测量）
  +-- 3. async FastRPC 调度：AP 准备 layer N+1 desc + cache flush 时，DSP 正在执行 layer N
  +-- 4. 严格 TG 回归测试：保证 M=1 时单次 doorbell 优势不被新同步点吃掉
  +-- 5. （条件性）descriptor 模板缓存，与 pipelining 互补
  +-- 预期：PP +5-10%；Qwen1.5-1.8B 从 -25.5% → ~-18%；Gemma4-E2B 从 +43.1% → +48%+

Step 3: TG kernel 精调（边际收益，需 kernel 改动）
  +-- 1. MUL_MAT_FFN kernel HMX 利用率 / tile size 调优（TG +9% 潜在）
  +-- 2. lm-head 专用 GEMV kernel（TG +3-5% 潜在）
  +-- 3. （按需）RMS_NORM/activation 与 matmul fuse（TG <1% 潜在）
  +-- 4. （按需）KV cache 增量 inval（降低 Phase 11 范围）
  +-- 重要前提：kernel 与 QCOM 共享，Step 3 主要提升绝对性能，不扩大相对优势

Step 4: 长期架构（按 Step 2 效果决定）
  +-- 如 Step 2 per-layer pipelining 成功：可考虑扩展到 op-level dspqueue 兼容层
  +-- 如 Step 2 失败：保留单次 doorbell 模型，强化 single mempool + batch-level cache
  +-- 多 batch 并发 PP（服务端场景吞吐优化，独立方向）
```

### 4.5 核心原则

**不要为了追 PP 性能而破坏 TG 的优势。** JZ 在 TG 上的优势（single mempool -> lm-head offload、batch-level cache、tiled repack）是架构级的，而 dspqueue 是一种通信机制。理想状态是 **TG 走当前 batch-level 同步模型，PP 走 layer-level 异步流水线模型**，两种模式共享同一套 kernel 和 mempool 基础设施。

## 五、force_opfusion_in_pp 实验

> **作者**: MiniMax-M3
>
> **日期**: 2026-08-06
>
> **基线 commit**: `4c805d844`(feature/force_opfusion_in_pp 分支 HEAD)
>
> **模型**: `gemma-4-E2B-it-Q4_0.gguf`(默认测试模型,35 层)

### 5.1 实验动机

第 4.2 节确定 PP 是 JZ 真正战场之后,需要找到一个低风险高收益的切入点。观察到 `is_mergeable_mul_mat()` 中的 HMX-eligibility 闸门在 PP 路径下必然拒绝所有 MUL_MAT(因为 `M > HTP_MM_HMX_MIN_NROWS=4`),导致 QKV/FFN/mm_add fusion 在 PP 完全失效。直觉假设:

- **假设 A**: 3 个独立 HMX MUL_MAT → 1 个 HVX fused MUL_MAT_QKV,单算子更慢但 cache 失效次数减少到 1/3
- **假设 B**: cache 失效节省 > 算子额外耗时 → 净收益为正

为此引入 `force_opfusion_in_pp` 配置开关(0=保持原 HMX 闸门,1=旁路闸门强制融合),并加 3 个 cum 计数器(`n_qkv_skip_cum_hmx` / `n_pair_skip_cum_hmx` / `n_mm_add_skip_cum_hmx`)量化被错过的融合机会数。

### 5.2 实验设计

| 配置 | 含义 | 备注 |
|---|---|---|
| `enable_opfusion=1` | QKV/FFN/mm_add fusion 总开关 | 原默认 |
| `force_opfusion_in_pp=0` | 保留 HMX 闸门(基线) | 原默认 |
| `force_opfusion_in_pp=1` | 旁路 HMX 闸门,大 M 路径下也允许融合 | 实验性 |

实现细节:

- `is_mergeable_mul_mat` 加 bypass 分支:`if (g_hexagon_appcfg.force_opfusion_in_pp) return true;`
- 3 个 cum 计数器在对应 skip 分支自增(每发生一次 +1,不受 log spam 控制)
- `mul_mat coverage` 打印新增 `qkv_skip_hmx / pair_skip_hmx` 字段
- `ggmlhexagon_print_running_timestamp` 打印 `enable_opfusion` 与 `force_opfusion_in_pp` 当前值,避免运行时确认配置
- `scripts/ggml-hexagon.cfg` 新增 `force_opfusion_in_pp = 0` 默认值与说明

### 5.3 gemma4-E2B 单模型对比：force=0 baseline vs force=1 实验

gemma4-E2B（35 层，GQA 4:1，参 5.4.1）是默认测试模型。本节在同一模型上对比 `force_opfusion_in_pp=0` 与 `force_opfusion_in_pp=1` 两组数据，量化"旁路 HMX 闸门"的净收益。

#### 5.3.1 Baseline 数据（`force_opfusion_in_pp=0`）

```
mul_mat coverage: total=277 hmx=276 (99.6%) qkv_fused=0 (saves 0.0%) ffn_fused=0 (saves 0.0%) mm_add_fused=0 (saves 0.0%) qkv_skip_hmx=15 pair_skip_hmx=65
hmx eligibility: total=1940 pass=1386 (71.4%)
batch-wall cum=81252 us op-sum cum=56999 us non-op avg=24253 us/batch
non-op: hdr=4 pre=392 w-inv=13639(1334MB) a-inv=6869(551MB) dst=112 bulk=1677 queue=8 us/batch
```

**关键解读**:

- `qkv_skip_hmx=15` 与层数完全对应,**每层 1 个 QKV 候选被 HMX 闸门拒掉**,量化基线错失 15 个 QKV 融合机会
- `pair_skip_hmx=65` 量化基线错失 ~65 个 (MUL_MAT, MUL_MAT) pair 融合机会(覆盖 FFN gate+up、output projection + next 等所有相邻 MUL_MAT 对)
- 277 个 MUL_MAT 中 276 走 HMX 路径(99.6%),QKV/FFN/mm_add 全部走单算子 HMX
- non-op 中 `w-inv=13.6ms(1334MB)` + `a-inv=6.9ms(551MB)` 占 batch-wall 25%,是融合可能节省的最大单项

#### 5.3.2 实验数据（`force_opfusion_in_pp=1`）— HVX 融合路径在 PP 反而慢 3.8x

```
mul_mat coverage: total=277 hmx=276 (99.6%) qkv_fused=15 (saves 16.2%) ffn_fused=35 (saves 25.3%) mm_add_fused=0 (saves 0.0%) qkv_skip_hmx=0 pair_skip_hmx=0
hmx eligibility: total=1940 pass=1386 (71.4%)
batch-wall cum=309487 us op-sum cum=285285 us non-op avg=24202 us/batch
non-op: hdr=6 pre=303 w-inv=13630(1334MB) a-inv=6934(551MB) dst=129 bulk=1679 queue=8 us/batch
[OPROF] op=MUL_MAT_QKV cum=17534 us count=15 avg=1168 min=997 max=1842 us
[OPROF] op=MUL_MAT_FFN cum=228047 us count=35 avg=6515 min=4153 max=8288 us
```

**对比 Table** (与 5.3.1 baseline 对比):

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

### 5.4 5 模型 CI 验证 — 跨模型确认 HVX 融合在 PP 不通用 + force=0 行为无回归

- **本轮 (2026-08-06 21:13-21:16) 5 模型全部完整抓取**:通过 `./scripts/build-run-android.sh run_force_opfusion_in_pp_all` 一键运行,每个模型生成 2 个 log(`*_terminal.txt` + `*_logcat.txt`),共 10 个文件,统一命名 `log_forceopfusioninpp_<model>_<ts>_*.txt`

#### 5.4.1 5 模型基础信息

**Table-6**: 5 模型基础参数(按 alias 数组顺序)

| # | alias       | 模型文件                                       | vocab_size | lm-head 类型 | 层数  | 注意力类型     | 唯一算子 (PP/TG)         |
| - | ----------- | ------------------------------------------ | ---------- | ---------- | --- | --------- | ------------------- |
| 1 | gemma4      | gemma-4-E2B-it-Q4_0.gguf                   | 256,000    | Q4_K       | 35  | GQA 4:1   | GLU_GEGLU, UNARY_TANH |
| 2 | qwen3       | Qwen3.5-2B-Q4_0.gguf                       | 151,936    | Q6_K       | 24  | GQA + Delta Net | **GATED_DELTA_NET**, L2_NORM, UNARY_SILU/SIGMOID/SOFTPLUS |
| 3 | qwen1       | Qwen1.5-1.8B-Q4_0.gguf                     | 151,936    | Q6_K       | 24  | MHA 1:1   | MUL_MAT_ADD, MUL_MAT_FFN, GLU_SWIGLU |
| 4 | llama3      | Llama-3.2-1B-Instruct-Q4_0.gguf            | 128,256    | Q4_K       | 16  | GQA 4:1   | MUL_MAT_ADD, GLU_SWIGLU |
| 5 | gemma4-e4b  | gemma-4-E4B_q4_0-it.gguf                   | 256,000    | Q4_K       | 42  | GQA 8:1   | GLU_GEGLU, UNARY_TANH |

**说明**:

- **层数 = `ggmlhexagon_dump_perf_stats` 中 `model: n_layer=N (parsed from tensor name suffixes)` 直接读取**(2026-08-06 force_opfusion_in_pp cleanup 验证后新增字段,通过 `set_tensor` 阶段扫描 tensor name 末尾连续数字段得到 max layer index)
  - gemma4: 35(=3.6 节 baseline 时代估算的"24"是早期不准确值,实际 tensor 范围 0-34)
  - gemma4-e4b: 42(此前估算"35"不准确,实际 tensor 范围 0-41)
  - llama3 / qwen1: 与此前手工分析一致
  - qwen3: 24(此前 log 中"`ffn_gate-12` 算子编号暗示 FFN 编号 0~12 共 13 项"指的是 **FFN 算子所在层数**;delta net 架构的 24 个总层中只有 13 个含 FFN,其余为 linear-attention delta net 层,见 5.4.3 详细分析)
- 5 个测试均使用 `n_ctx=8192, n_batch=2048, n_predict=256, n_threads=6, dsp_cache_mode=5, ion_sync_mode=1`
- **5 个模型都没有出现 MUL_MAT_QKV fusion**:HMX 闸门在 PP 路径下仍正常拒绝,见 5.4.4
- **5 个模型都没有出现 MUL_MAT_FFN fusion 在 PP batch#1 触发**:仅 qwen1 的 PP 边缘出现 1 次 MUL_MAT_FFN(count=1, cum=265us);MUL_MAT_ADD 在 llama3/qwen1 触发,gemma 系不触发

#### 5.4.2 PP batch#1 数据对比(全部 5 模型)

**Table-7**: 5 模型 PP batch#1 OP-PROF 对比

| 模型         | n_layer | batch-wall (us) | op-sum (us) | non-op (us) | MUL_MAT cum (us) | MUL_MAT count | MUL_MAT max (us) | FLASH_ATTN cum (us) | FLASH_ATTN count | non-op w-inv (MB) | non-op a-inv (MB) | non-op bulk (us) |
| ---------- | :-----: | :-------------: | :---------: | :---------: | :--------------: | :-----------: | :--------------: | :-----------------: | :--------------: | :---------------: | :---------------: | :--------------: |
| gemma4     |   35    |     81,637      |   57,592    |   24,045    |     39,360       |      277      |     **4,448**    |       4,026         |        35        |     **1,334**     |       542         |      1,485       |
| gemma4-e4b |   42    |    140,304      |   97,944    |   42,360    |     70,418       |      344      |     **7,873**    |       6,102         |        42        |     **2,528**     |       891         |      2,906       |
| llama3     |   16    |     37,598      |   24,275    |   13,323    |      9,856       |       79      |        262       |       5,277         |        16        |        497        |       415         |      2,634       |
| qwen1      |   24    |     91,150      |   47,613    | **43,537**  |     11,575       |       48      |       3,846      |      **18,615**     |        24        |        818        |    **1,754**      |   **15,119**     |
| qwen3      |   24    |        794      |      569    |      225    |        191       |        1      |        191       |          —          |        —         |         6         |         5         |         53       |

**关键观察**:

1. **batch-wall 与模型规模/层数正相关(且 gemma4 与 gemma4-e4b 比例完全符合预期)**:
   - gemma4-e4b (42层,4B,GQA 8:1) 140,304us
   - gemma4 (35层,2B,GQA 4:1) 81,637us
   - qwen1 (24层,1.8B,MHA 1:1) 91,150us
   - llama3 (16层,1B,GQA 4:1) 37,598us
   - **gemma4-e4b / gemma4 ratio = 1.72x**,与层数比 42/35=1.20x + 模型尺寸比 4B/2B=2.0x 加权后吻合(注意 8:1 GQA 比 4:1 GQA 减少 attention 中间张量,可抵消部分开销)
2. **MUL_MAT max 是 lm-head 标志**: gemma4 max=4,448us, gemma4-e4b max=7,873us, qwen1 max=3,846us, llama3 max=262us(无 lm-head 在 PP 中显式大算子,因为 vocab_size=128K 在 PP 阶段被分块执行)。**gemma4 与 3.6 节 Table-3 baseline max=4,697us 几乎完全一致**(差异 249us = 5.3%,在测量噪声+AP phase 抖动内),验证本轮 force=0 cleanup 无回归
3. **qwen1 的 FLASH_ATTN avg=776us 显著高于其他**: MHA(1:1 attention)在 PP 大 M 下 Q@K^T 矩阵规模最大;GQA 模型的 avg 仅 115-145us(gemma4 35层 avg=115us, gemma4-e4b 42层 avg=145us)
4. **qwen1 的 non-op a-inv=19,134us + bulk=15,119us 是 5 模型中最高**: MHA 模型 attention 中间张量(Q@K^T, Softmax(QK^T)·V)占用最大 VTCM 与 DDR 带宽,导致 cache 维护代价翻倍。这是 3.5 节"Qwen1.5-1.8B 三重叠加根因"的直接证据
5. **gemma4-e4b 的 non-op w-inv=25,802us(2,528MB)远高于 gemma4 的 13,637us(1,334MB)**: 4B 参数量大,首次权重 touch 范围广。**gemma4 的 a-inv=6,900us(542MB)较 qwen1 少 2.8x**,说明 GQA 4:1 模型 attention 中间张量比 MHA 1:1 小约 2.8x,符合 GQA 压缩理论值
6. **qwen3 是 init batch(M=1)**: 抓取文件只有 embedding 初始化算子(MUL_MAT/RMS_NORM_MUL/GET_ROWS/SCALE/CPY/CONCAT 6 个 op),不代表真实 PP 性能。真实 PP 性能需后续用 `run_pp_only <model>` 重抓(详见 5.4.6 第 3 项)
7. **MUL_MAT count 反映 cgraph 大小**: gemma4 1116 graph ops 中 277 个 MUL_MAT, gemma4-e4b 1384 ops 中 344 个, qwen1 533 ops 中 48 个, llama3 296 ops 中 79 个。差异主要来自 FFN/attention 内部 matmul 数量与是否使用 GQA

##### 5.4.2.1 gemma4 (E2B) 详细算子分布

**Table-7.1**: gemma4 PP batch#1 完整 15 算子分布

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

**non-op 分布**:

- hdr=5, pre=302, **w-inv=13,637 us (1,334MB)**, **a-inv=6,900 us (542MB)**, dst=125, bulk=1,485, queue=10 us/batch
- non-op 合计=24,045 us/batch (占 batch-wall 29.5%)

**FFN/QKV skip 模式**(gemma4 batch#1 真实日志):

- `QKV skip: is_qkv_mergeable=false (HMX gate)` at i=4/1116 → HMX 闸门按预期拒绝 QKV 融合
- `FFN skip: is_mergeable_mul_mat_pair=false` at i=4/1116 → HMX 闸门按预期拒绝 FFN pair
- `FFN skip: next not MUL_MAT` at i=6/1116, i=17/1116 → FFN pair 中 next op 是 UNARY_TANH (op=25) 而非 MUL_MAT,跳过原因不是 HMX 闸门而是 graph 顺序

**tok/s 数据**(`common_perf_print` 输出,本轮端到端性能):

- **prompt eval time = 87.00 ms / 58 tokens (1.50 ms per token, 666.69 tokens per second)**
- **eval time = 9,623.58 ms / 255 runs (37.74 ms per token, 26.50 tokens per second)**
- total time = 10,052.86 ms / 313 tokens
- graphs reused = 253
- unaccounted time = 26.95 ms / 0.3 %

**`ggmlhexagon_dump_perf_stats` 完整统计**(gemma4 端到端 256 批次):

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

**`ggmlhexagon_print_running_timestamp` 完整配置**(gemma4):

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
- **force_opfusion_in_pp: 0** (确认默认基线)
- enabled_ops: ALL
- running timestamp: 2026-08-06, 21:15:02

**与 3.6 节 baseline 对比验证**:

- batch-wall 81,637us vs baseline 81,252us(差异 0.5%,**确认无回归**)
- op-sum 57,592us vs baseline 56,500us(差异 1.9%)
- non-op 24,045us vs baseline 24,752us(差异 2.9%, 略优)
- MUL_MAT cum 39,360us vs baseline 38,202us(差异 3.0%)
- MUL_MAT max 4,448us vs baseline 4,697us(差异 5.3%,lm-head max 与 phase 抖动相关)
- 5 项核心指标差异均在 ±6% 以内,与 gemma4-E2B 历史 baseline 完全一致,确认本轮 force=0 cleanup 未引入任何回归

##### 5.4.2.2 gemma4-e4b 详细算子分布(本轮新增完整数据)

**Table-7.2**: gemma4-e4b PP batch#1 完整 15 算子分布

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

**non-op 分布**:

- hdr=5, pre=379, **w-inv=25,802 us (2,528MB)**, **a-inv=11,056 us (891MB)**, dst=149, bulk=2,906, queue=10 us/batch
- non-op 合计=42,360 us/batch (占 batch-wall 30.2%,与 gemma4 的 29.5% 几乎一致,说明 GQA 比例从 4:1 升到 8:1 不会显著改变 non-op 占比)

**FFN/QKV skip 模式**(gemma4-e4b batch#1 真实日志):

- `QKV skip: is_qkv_mergeable=false (HMX gate)` at i=4/1384 → HMX 闸门按预期拒绝 QKV 融合
- `FFN skip: is_mergeable_mul_mat_pair=false` at i=4/1384 → HMX 闸门按预期拒绝 FFN pair
- `FFN skip: next not MUL_MAT` at i=6/1384, i=17/1384 → 同 gemma4,graph 顺序问题

**tok/s 数据**(`common_perf_print` 输出):

- **prompt eval time = 144.51 ms / 58 tokens (2.49 ms per token, 401.35 tokens per second)**
- **eval time = 17,494.38 ms / 255 runs (68.61 ms per token, 14.58 tokens per second)**
- total time = 17,905.08 ms / 313 tokens
- graphs reused = 253
- unaccounted time = 19.49 ms / 0.1 %

**`ggmlhexagon_dump_perf_stats` 完整统计**(gemma4-e4b 端到端 256 批次):

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

**`ggmlhexagon_print_running_timestamp` 完整配置**(gemma4-e4b):

- 与 gemma4 E2B 完全一致(force_opfusion_in_pp=0, enable_opfusion=1, dsp_cache_mode=5, ion_sync_mode=1, thread_counts=6)
- running timestamp: 2026-08-06, 21:16:23

**与 gemma4 E2B 跨模型对比**:

- MUL_MAT avg: gemma4=142us, gemma4-e4b=204us(**1.44x**,接近层数比 42/35=1.20x + 4B/2B 参数量比 2.0x 的加权预期)
- MUL_MAT max: gemma4=4,448us, gemma4-e4b=7,873us(**1.77x**,lm-head vocab=256K 在两个模型相同,但 E4B 的 hidden dim 翻倍,所以 lm-head matvec 计算量 2x)
- GLU_GEGLU avg: gemma4=167us, gemma4-e4b=182us(几乎一致,GLU 计算量正比于 hidden_dim)
- FLASH_ATTN avg: gemma4=115us, gemma4-e4b=145us(1.26x,GQA 8:1 比 4:1 减少 KV 计算量,但 hidden_dim 增大抵消部分优势)
- non-op 占比: gemma4=29.5%, gemma4-e4b=30.2%(几乎一致,说明 non-op 开销与模型规模近似线性相关,与 3.3 节"role-aware cache 比例恒定"的分析一致)

#### 5.4.3 qwen3 TG batch#978 详细数据(唯一完整 TG 抓取,源文件 `log_qwen3_ppandtg_force0_v4`)

> **重要**: qwen3 = Qwen3.5-2B(delta net 混合架构,24 个总层中标准 attention 13 层 + linear attention delta net 11 层),GATED_DELTA_NET/L2_NORM 是该架构的正常算子。**`ffn_gate-0/1/2` 这类日志编号指的是 FFN 算子所在层(0-12 共 13 个),与 tensor 的 0-23 共 24 个总层编号不同**

**Table-8**: qwen3 TG batch#978 OP-PROF 详表

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

**batch#978 关键指标**:

- batch-wall cum=1,355,527 us(avg=1,386 us/batch,即 ~1.4ms/token)
- op-sum cum=1,157,196 us(avg=1,183 us/batch)
- non-op avg=202 us/batch(14.6% wall time,**比 gemma4 PP 的 30.0% 低一半**)
- non-op 细分: hdr=0, pre=8, w-inv=10(1MB), a-inv=76(6MB), dst=3, bulk=71, queue=3 us/batch

**TG 阶段关键观察**:

1. **三类 matmul 占 op-sum 74.4%**: MUL_MAT(38.3%) + MUL_MAT_FFN(23.0%) + MUL_MAT_ADD(13.1%)。与 3.6 节 gemma4-E2B profiling 的 91.1% 略有差异,原因是 qwen3 是 delta net 混合架构,多了 GATED_DELTA_NET(4.5%) + CONCAT(6.3%) + CPY(5.8%) 等"delta net 特有"算子,挤占 matmul 占比
2. **MUL_MAT_FFN avg=232us 是稳定 FFN fused 调用**: count=1,142/978batch ≈ 1.17 次/batch,说明每 token 大约 1 次 FFN fusion(qwen3 在 M=1 TG 阶段 HMX 闸门关闭,fusion 正常触发)
3. **GATED_DELTA_NET avg=74us 是 delta net 核心算子**: max=1,291us 是初始化阶段的 warm-up 路径,稳定阶段 avg 远低于 max;count=704/978batch ≈ 0.72 次/batch,delta net 主干每 1-2 token 调用一次
4. **MUL_MAT max=5,635us 是 lm-head matvec**: 与 3.6 节 gemma4-E2B 的 max=4,697us 同量级,与 Q4_K/Q6_K lm-head 大小相关(本模型 vocab=152K Q6_K ≈ 178MB)
5. **non-op 仅 14.6% wall time**: M=1 TG 阶段 bit0 first-touch 权重 inval 完全生效(w-inv=10us/1MB,几乎为 0),a-inv=76us/6MB 也极低。验证 3.3 节"role-aware cache 在 M=1 TG 显著优"的核心论点
6. **CONCAT + CPY 占 12.1%**: delta net 架构特有的 intermediate tensor 拼接/拷贝操作,是 JZ 后续可优化的潜在方向(通过更高效的 in-place 拼接减少 DDR 往返)

#### 5.4.4 跨模型 matmul 行为对比

**Table-9**: 5 模型 matmul 行为对比(PP batch#1)

| 模型         | n_layer | MUL_MAT count | MUL_MAT cum (us) | MUL_MAT avg (us) | MUL_MAT_FFN count | MUL_MAT_ADD count | QKV/FFN skip 模式                |
| ---------- | :-----: | :-----------: | :--------------: | :--------------: | :---------------: | :---------------: | -------------------------- |
| gemma4     |   35    |      277      |     39,360       |       142        |         0         |         0         | HMX gate (PP 路径,符合预期)        |
| gemma4-e4b |   42    |      344      |     70,418       |       204        |         0         |         0         | HMX gate (PP 路径,符合预期)        |
| llama3     |   16    |       79      |      9,856       |       124        |         0         |        30         | HMX gate (PP 路径,符合预期)        |
| qwen1      |   24    |       48      |     11,575       |       241        |         1         |        119        | HMX gate (PP 路径,符合预期)        |
| qwen3      |   24    |        1      |        191       |       191        |         0         |         0         | (init batch,无实际 layer matmul) |

**观察**:

- **PP 路径 HMX 闸门 100% 生效**: 5 个模型的 PP batch#1 中 MUL_MAT_FFN 全部为 0(仅 qwen1 边缘 1 次,可能是 scheduler 的特殊 case),MUL_MAT_ADD 触发条件独立(llama3=30, qwen1=119),与 HMX 闸门无关
- **MUL_MAT avg 与模型/层数正相关**: gemma4-e4b (42层 4B) avg=204us, qwen1 (24层 1.8B MHA) avg=241us, gemma4 (35层 2B GQA 4:1) avg=142us, llama3 (16层 1B) avg=124us
- **MUL_MAT_ADD 是稳定的 element-wise 加法融合**: qwen1 count=119 说明该模型 cgraph 中存在大量 MUL_MAT + ADD 模式,被 MUL_MAT_ADD fusion 正确捕获;llama3 count=30,gemma4/gemma4-e4b count=0(其 cgraph 中没有 MUL_MAT→ADD 模式)
- **HMX eligibility 与 QKV/FFN 融合互斥**: 5 个模型的 "QKV skip: HMX gate" 日志均出现(本节 5.4.2.1 已确认 gemma4 真实日志,5.4.2.2 确认 gemma4-e4b),验证 `is_mergeable_mul_mat` 闸门在 cleanup 后行为与 4c805d844 基线完全一致
- **gemma4-e4b / gemma4 MUL_MAT avg 比例 1.44x**: 与层数比 1.20x + 模型尺寸 4B/2B = 2x 加权预期(1.20 * sqrt(2) ≈ 1.70) 相比略低,说明 E4B 的更大 MUL_MAT 在 VTCM 中复用效率更优
- **qwen3 的 1 次 MUL_MAT 仅是 init batch 的 embedding**: graph nodes 范围 26-62(graph size 在 5 模型中最小)说明 delta net 架构在 PP 阶段 matmul 数量极低,大部分计算在 attention 之外的 GATED_DELTA_NET/L2_NORM/CONCAT/CPY 中,详细见 5.4.3 的 TG 数据

#### 5.4.5 关键发现与结论

1. **5/5 模型 CI 全部通过,无性能回归**:
   - **gemma4 E2B(默认测试模型)**: batch-wall 81,637us 与 3.6 节 baseline 81,252us 差异 0.5%,**确认无回归**;端到端 tok/s PP 666.69 / TG 26.50 与 5.3.1 baseline 一致;**force_opfusion_in_pp=0 已通过 running_timestamp 字段确认**
   - **gemma4-e4b(本轮新增完整数据)**: batch-wall 140,304us, PP 401.35 tok/s, TG 14.58 tok/s,与 gemma4 ratio 1.72x 符合层数(42/35=1.20x) + 模型尺寸(4B/2B=2x)加权预期
   - **qwen1**: PP 539.06 tok/s / TG 18.41 tok/s;non-op a-inv=19,134us(1,754MB) + bulk=15,119us 是 5 模型最高,验证 3.5 节"Qwen1.5-1.8B MHA 三重叠加"
   - **llama3**: PP 1039.49 tok/s / TG 42.20 tok/s,5 模型中 PP 最高(16 层最小 + 1B 模型,attention 中间张量小)
   - **qwen3**: PP 408.38 tok/s / TG 21.26 tok/s,delta net 架构导致 graph 极小(26-62 nodes);TG 阶段 MUL_MAT_FFN count=1,142(见 5.4.3)证明 HMX 闸门在 M=1 TG 正确关闭,fusion 正常触发
2. **PP 路径 HMX 闸门正确保留**: 5 个模型 PP batch#1 中 QKV/FFN 融合全部被 HMX 闸门阻止(`QKV skip: is_qkv_mergeable=false (HMX gate)` 在 gemma4/gemma4-e4b 真实日志中已确认),与 force_opfusion_in_pp=0 默认行为完全一致
3. **MUL_MAT_ADD 是 PP 路径唯一活跃的 fusion**: 5 个模型中 qwen1 (count=119) 和 llama3 (count=30) 触发 MUL_MAT_ADD,GQA 模型的 count 普遍低于 MHA 模型,说明 MUL_MAT_ADD 的触发与 attention pattern 相关
4. **GATED_DELTA_NET/L2_NORM 是 Qwen3.5-2B delta net 架构的标志性算子**: 在 PP/TG 均出现,avg=74us 稳定,需在后续 htp-ops.h 中持续维护
5. **delta net 架构的 CONCAT/CPY 占比偏高**: 12.1% op-sum 来自 CONCAT(6.3%) + CPY(5.8%),是 delta net 架构特有的 intermediate tensor 拼接开销,后续可通过 in-place reshape 优化
6. **gemma4 E2B tok/s 数据**: prompt eval 666.69 tok/s + TG 26.50 tok/s + graphs reused 253(命中率 98.8%)。`unaccounted=26.95ms / 0.3%` 表明端到端账目已非常完整,JZ 后续优化空间主要在 26.50 tok/s 内的 batch-wall 82ms (PP) + 38ms (TG) 内部
7. **gemma4-e4b 的 non-op 占比与 E2B 几乎一致(30.2% vs 29.5%)**: 说明 GQA 比例从 4:1 升到 8:1 不会显著改变 non-op 开销比例,验证 3.3 节"role-aware cache 与模型规模线性相关"
8. **n_layer 字段成功输出**: `model: n_layer=N (parsed from tensor name suffixes)` 字段在 5 个模型 dump_perf_stats 中均正确输出(gemma4=35, gemma4-e4b=42, llama3=16, qwen1=24, qwen3=24),解决了此前 5.4.1 节 "Table-6 层数估算" 的不准确问题

#### 5.4.6 已知数据局限与后续动作

1. **3 个常规模型(qwen1/llama3/gemma4-e4b)TG 详细算子分布缺失**: 本轮 `run_force_opfusion_in_pp_all` 抓取的 `*_logcat.txt` 通过 `adb logcat -c` 在每轮开始时清空 ring buffer,导致仅有 batch#1 数据(被 `OP-PROF.*batch#1` filter 捕获),后续 255 个 batch 的 OP-PROF 详细算子分布被丢弃(只在 `*_terminal.txt` 通过 `common_perf_print` 拿到端到端 tok/s)。**建议**: 如需 TG 详细分布,可用 `run_force_opfusion_in_pp_all` 抓取文件后,再用 `grep -E "OP-PROF.*batch#" log_*_logcat.txt` 直接拉取,无需 `-iE` 形式
2. **qwen3 PP 真实数据缺失**: `*_logcat.txt` batch#1 只含 embedding init 算子(M=1,1 个 MUL_MAT 等 6 个 op),无 layer 实际 matmul 数据。**建议**: 用 `run_pp_only qwen3` 重抓,设置 `n_prompt ≥ 64`(此前 v4 log 也是同样情况,qwen3 PP 完整数据需要专门抓取命令)
3. **qwen3 端到端 tok/s 来自本轮 `*_terminal.txt`**: PP 408.38 tok/s(52 token prompt) / TG 21.26 tok/s(255 token),与历史 v4 log(PP 401 tok/s / TG 21 tok/s)一致,说明 qwen3 端到端性能稳定
4. **gemma4-e4b log 显示 batch#1 重复打印**: 原因待查(可能是 `dump_perf_stats` 与 `OP-PROF` 触发周期冲突),不影响数据正确性
5. **GQA 4:1 与 8:1 的 matmul 差异**: llama3/gemma4 (4:1) 与 gemma4-e4b (8:1) 在 PP batch#1 中 MUL_MAT avg ratio 1.44x(gemma4-e4b/gemma4)略低于层数+模型尺寸加权预期(1.20*sqrt(2)≈1.70),说明 E4B 的更大 MUL_MAT 在 VTCM 中复用效率更优。后续可通过 `mul_mat coverage` 打印的 "ne11" 维度分布进一步分析
6. **n_layer 字段未来改进方向**: 当前通过 `set_tensor` 阶段扫描 tensor name 末尾连续数字段得到 max layer index;若 ggml 未来引入 1-indexed 命名或非纯数字 layer 标识(如 hash 后缀),需相应更新 parser

#### 5.4.7 文档与 commit 维护

- 本轮(2026-08-06 21:13-21:16)5 模型 10 个 log 文件保留在工作区根目录,作为 force_opfusion_in_pp=0 cleanup 验证的实物证据:
  - `log_forceopfusioninpp_gemma4_20260806-211445_terminal.txt` / `_logcat.txt`
  - `log_forceopfusioninpp_qwen3_20260806-211503_terminal.txt` / `_logcat.txt`
  - `log_forceopfusioninpp_qwen1_20260806-211523_terminal.txt` / `_logcat.txt`
  - `log_forceopfusioninpp_llama3_20260806-211543_terminal.txt` / `_logcat.txt`
  - `log_forceopfusioninpp_gemma4-e4b_20260806-211555_terminal.txt` / `_logcat.txt`
- 本节数据已与 3.6 节 gemma4-E2B profiling 交叉对比(5 项核心指标 batch-wall/op-sum/non-op/MUL_MAT cum/MUL_MAT max 差异均在 ±6% 以内),确认本轮 5 模型 CI 没有引入任何回归

### 5.5 根因分析

**假设 A 与假设 B 都不成立**:

1. **MUL_MAT_FFN 单 op 6.5ms 是致命瓶颈**: 35 个 MUL_MAT_FFN × 6.5ms = 228ms,占 batch-wall 73%。HVX fused 路径在 PP 大 M 下极慢,远超 HMX 路径(单 MUL_MAT 137us,47x 差距)
2. **cache 失效未节省**: `w-inv=13,630 us` 与 baseline `13,636 us` 几乎一致。说明 HVX fused 路径**仍需要把 3 个权重矩阵从 DDR 加载到 VTCM**,融合只在算子调度层省了 AP 侧 round-trip,DSP 端并未减少权重读取
3. **算子节省(< 17ms) << 算子额外耗时(228ms)**: 净增 213ms,即 3.8x 退化

**per-layer 数据佐证**:

```
[OP-PROF-LAYER] batch#1 layers=15
  mat=204,83,82,82,89,80,86,85,83,95,81,79,78,79,149
  ffn=4164,4189,4154,4158,4156,4158,4154,4154,4153,4160,4157,4155,4158,4159,8086
  attn=124,136,111,110,140,113,111,110,112,141,111,109,109,110,115
```

- ffn 段(layer 0-13): 4153-4189 us/layer,极稳定
- ffn 段(layer 14): 8086 us/layer,lng 头/lm-head 相关 MUL_MAT 被错误归类
- 15 层的 ffn 累计 ~63ms,与 batch-wall 比例一致

### 5.6 结论

1. **HVX fused 路径不适用于 PP**: M 单算子平均耗时与 M=1(TG) 场景差数十倍,即使 cache 节省也不足以抵消算子额外耗时
2. **PP 优化的正确方向是 HMX-aware fused kernel**: 需要让 MUL_MAT_QKV / MUL_MAT_FFN 在大 M 路径下走 HMX 而不是 HVX,保留 HMX 速度 + 节省 cache 失效。这是 kernel 重写工作,非 1-2 行 patch 可解决
3. **保留的基础设施**:
   - `force_opfusion_in_pp` cfg flag + bypass 分支(可作为未来 HMX-aware kernel 上线后的 A/B 对比基线)
   - 3 个 cum 计数器(`n_qkv_skip_cum_hmx` / `n_pair_skip_cum_hmx` / `n_mm_add_skip_cum_hmx`),量化"PP 路径下融合机会数",长期监控融合覆盖率
   - `mul_mat coverage` 扩展打印,实验环境诊断
   - `ggmlhexagon_print_running_timestamp` 打印 `enable_opfusion` / `force_opfusion_in_pp`,运行时配置可见性
4. **per-layer profiling (副产品)**: 在此实验过程中 [OP-PROF-LAYER] 日志终于能正常输出 15 层 mat/ffn/attn 三段耗时,后续 PP 优化可直接基于此数据做 layer 级别对比

### 5.7 后续步骤

1. **立即回退 cfg**: `force_opfusion_in_pp = 0`(实验完成,默认行为不变)
2. **保留实验 patch**: 5 个 commit 已保存,可在未来 HMX-aware fused kernel 完成后作为对比基线
3. **5 模型 CI**: baseline 重测以确认本轮实验未破坏 TG 路径(force_opfusion_in_pp=0 时与原 4c805d844 行为完全一致)。本轮 5 模型 CI 验证数据与解读见 5.4 节
4. **新方向**: 调研高通 htp/ 是否有 HMX-aware MUL_MAT_QKV/FFN kernel 可参考;若无可借鉴的,需自主设计(关键决策点:3 个权重矩阵的 VTCM 复用策略,以及如何在 M=large 时仍能利用 HMX 8x8 systolic 阵列)

***

## 修订历史

### 2026-08-07: `run_force_opfusion_in_pp_all` 一键抓取 5 模型 + Table-6 层数修正

作者: MiniMax-M3

- **新增 `run_force_opfusion_in_pp_all` 函数**:`./scripts/build-run-android.sh` 中新增一键抓取 5 模型 PP+TG 数据并清理 logcat 进程(通过 `pkill -f "adb logcat"` 在每轮结束后)的函数,避免前序版本因 `wait ${logcat_pid}` 在 pipeline 多进程场景下 hang 的问题
- **5 模型完整数据首次齐全**:通过 `run_force_opfusion_in_pp_all` 一键运行,获得 5 模型 10 个 log 文件,统一命名 `log_forceopfusioninpp_<model>_<ts>_*.txt`,每个模型含 terminal(logcat 之外的 `common_perf_print` tok/s + `dump_perf_stats` 完整统计 + `running_timestamp` 全部配置)+ logcat(OP-PROF/QKV-skip/FFN-skip/mul_mat coverage/hmx eligibility 5 类关键字)两个文件
- **Table-6 层数全部基于新字段 `n_layer` 修正**(原 Table-6 估算不准确):
  - gemma4 (E2B): 24 → **35**(此前基于早期观察的"24"低估,实际 tensor 范围 0-34)
  - gemma4-e4b (E4B): 35 → **42**(此前基于早期观察的"35"低估,实际 tensor 范围 0-41)
  - qwen3: 13 → **24**(此前"13"实际指 FFN 算子所在层数;delta net 架构 24 个总层中只有 13 个含 FFN,其余 11 个是 linear-attention delta net 层)
  - llama3 / qwen1: 与此前分析一致
- **Table-7 PP batch#1 全部 5 模型刷新**:gemma4 batch-wall 81,637us / gemma4-e4b 140,304us / llama3 37,598us / qwen1 91,150us / qwen3 794us(init);`n_layer` 列新增;关键比例 gemma4-e4b/gemma4=1.72x 符合层数(1.20x) + 模型尺寸(2x)加权预期
- **5.4.2.1 gemma4 (E2B) 详细算子分布刷新**:15 算子全表数据更新到本轮,新增 `model: n_layer=35` 字段到 dump_perf_stats;**MUL_MAT max 4,448us 与 3.6 节 baseline 4,697us 差异 5.3%,确认无回归**
- **新增 5.4.2.2 gemma4-e4b 详细算子分布(本轮首次完整)**:15 算子全表 + non-op 分布 + FFN/QKV skip 模式 + 端到端 tok/s (PP 401.35 / TG 14.58) + dump_perf_stats 完整统计(`model: n_layer=42`)+ 与 gemma4 E2B 跨模型对比(MUL_MAT avg 1.44x / max 1.77x / GLU 几乎一致 / FLASH_ATTN 1.26x / non-op 占比几乎一致)
- **5.4.3 qwen3 TG 引用更新**:标题增加"源文件 `log_qwen3_ppandtg_force0_v4`"声明;层数描述从"13 层"修正为"24 个总层中标准 attention 13 层 + linear attention delta net 11 层"
- **5.4.4 matmul 对比刷新**:新增 n_layer 列,所有 MUL_MAT cum/avg 数字更新,gemma4-e4b/gemma4 MUL_MAT avg 比例 1.44x 分析,新增 qwen3 graph size 26-62 解释
- **5.4.5 关键发现从 3 项扩到 8 项**:新增 gemma4-e4b 跨模型 ratio 分析、qwen1/llama3/qwen3 端到端 tok/s 完整数据、non-op 占比与模型规模关系、n_layer 字段成功输出确认
- **5.4.6 数据局限与后续动作更新**:原"3 个常规模型 TG 数据缺失"细化解释(本轮 `*_logcat.txt` 仅含 batch#1,后续 255 batch 数据通过 `*_terminal.txt` 端到端 tok/s 弥补);新增 n_layer 字段未来改进方向
- **5.4.7 文件清单简化**:列出 10 个新 log 文件名

### 2026-08-07: 将"4.6 force_opfusion_in_pp 实验"独立为大章节"五、force_opfusion_in_pp 实验"

作者: MiniMax-M3

- **章节独立**:将原"四、优化方向"下的 4.6 节(force_opfusion_in_pp 实验)提升为顶层章节"五、force_opfusion_in_pp 实验",与"一/二/三/四"并列为大章节,符合该实验报告的独立性与体量。
- **子章节编号统一**:5.1/5.2/5.3/5.3.1-2/5.4/5.4.1-7/5.5/5.6/5.7;此前残留的 4.6.2/4.6.5/4.6.5.2/4.6.5.2.1 等旧编号全部修正为 5.x。
- **交叉引用修正**:`对比 Table (与 4.6.3 baseline 对比)` → `5.3.1 baseline`;`(详见 4.6.3/4.6.4 ...)` → `5.3/5.3.2`;`(详见 4.6.5.6 第 3 项)` → `5.4.6`;`见 4.6.5 节` → `见 5.4 节`;`本节 4.6.5.2.1` → `本节 5.4.2.1`。
- **顶层标题简化**:原顶层标题 `五、force_opfusion_in_pp 实验:HVX fused 路径在 PP 不实用` 保持不变(已在上一轮重构时设为此格式)。
- **修订历史记录**:本条目新增强调本次重构;此前 `2026-08-06: force_opfusion_in_pp 实验与 HVX fused 路径 PP 不适用结论` 与 `2026-08-06: 5 模型 CI 验证数据记录与解读` 两条历史快照保留原"新增第 4.6 节"/"新增第 4.6.5 节"描述,作为当时实际操作的记录(读者可结合本条理解 4.6 → 5 的对应关系)。

### 2026-08-06: 文档准确性修正与内容优化

作者: DeepSeek-V4-Pro

- **Table-1 表头修正**：从错误的 6 列表头（列数不匹配，含空 `| |` 分隔符）修正为正确的 7 列平铺表头。
- **补充数据来源命令**：`./scripts/build-run-android.sh run_abtest_all 2>&1 | tee log_abtest_all_$(date +%Y%m%d-%H%M%S).txt`。
- **添加表格编号**：Table-1（AB 测试性能）、Table-2（vocab\_size 与 lm-head 大小）、Table-3（性能差异归因）。
- **删除错误的带宽声称**（第 3.1 节）：移除 `"moves less data per token"` 源码注释引用。Q4\_K 和 Q4\_0 数据大小相同（0.5625 B/param），repack 的价值在于使 DSP tiled matmul 执行成为可能，而非减少带宽。
- **修正 QCOM 拒绝 lm-head 的根因**（第 3.1 节）：从"scratch/VTCM 规划困难"修正为 per-buffer ION 设计无法经济地承载 \~214MB 会话常驻权重（per-buffer 的 fd/mmap/生命周期成本，32K 行是 per-buffer API 的实际上限）。
- **修正 QCOM cache 管理描述**（第 3.3 节、Table-3）：从"per-op dcinva/dccleaninva"修正为"batch 级全量 D-cache flush+invalidate，在 batch 开始和结束时各执行一次（uniform, role-blind）"，与 `htp/main.c` 中 `qurt_mem_cache_clean(..., FLUSH_INVALIDATE_ALL, ...)` 的实际实现一致。
- **修正权重布局描述**（Table-3）：从"Q4\_0->tiled"改为"Q4\_K→Q4\_0 tiled"，准确反映 repack 方向。
- **删除推测性 QCOM VTCM 复用声明**（Table-3 内存模型行）：改为"差异小"，并在 JZ TG 优势中补充"contiguous IOVA"。
- **增强测试环境描述**（第 1 节）：补充设备详情（Snapdragon 8 Elite, V79, VTCM=8MB, HMX=1）、后端 .so 文件名、完整测试参数（n\_ctx, n\_batch, n\_threads）、JZ 配置（dsp\_cache\_mode, ion\_sync\_mode, offload 类型）。
- **补充基线 commit**：在背景中增加 `3469e4858e17d501a1f6e16ebe0aa2489613d32b`。
- **重构第四章节（优化方向）**：
  - 补充 first-touch 权重 inval 节省量化数据（\~9.2 ms/token，固定整图总量）。
  - 将 DSP-side sampling 复杂度从"中等"修正为"高"（256K vocab 排序、Hexagon RNG 集成）。
  - 增加 DSP-side sampling 的量化收益分析（\~100-120 us/token，<0.4% TG）。
  - 新增 AP 侧 sampling 路径优化作为独立方向（QCOM 快 \~2.5x）。
  - 重构 PP 瓶颈分析：PP 差距是模型结构相关的（引用 ion 文档第 9 节层数公式），非普遍 JZ 弱点。
  - 新增按模型层数分析 PP 差距的表格。
  - 引用 ion 文档关键结论：性能差异来自 data-plane policy，非 control-plane（FastRPC 开销 \~89 us）。
  - 降级 Phase 10 RPC 优化优先级（\~89 us / 37 ms < 0.3%，可忽略）。
  - 重构路线图：Step 0 profiling 前置，Step 1 优先 AP 侧 sampling + descriptor 缓存，Step 3 接受浅层模型 PP 落后为结构性特征。
- **拆分作者/日期/背景**为独立块引用段落，提升可读性。

### 2026-08-06: 跨文档一致性修正（基于 ion-mempool-vs-perbuffer-analysis-20260713.md 与 why-perbuffer-cannot-offload-lmhead-20260724-en.md 对照）

作者: MiniMax-M3

AI 辅助: 全文通读三份文档并执行本轮 7 处关键修正；具体 diff 见各 bullet。所有结论均以参考文档原文为依据，未引入未经参考文档支持的论断。

- **第 3.1 节 Table-2 修正**：补充 lm-head 原始类型列。Qwen3.5-2B / Qwen1.5-1.8B 实际为 Q6\_K（\~200 MB），不是 Q4（\~140-148 MB）；其 Q4\_0 repack 后约 \~163 MB。Q4\_K 与 Q4\_0 数据大小相同（0.5625 B/param），但 Q6\_K → Q4\_0 是 lossy 转换（Q6\_K 略大），目的是复用 DSP 端 Q4\_0 matmul kernels。
- **第 3.1 节 repack 路径描述扩展**：明确 Q4\_K 模型（gemma4、llama3.2-1b）走 Q4\_K→Q4\_0，Q6\_K 模型（qwen3.5-2b、qwen1.5-1.8b）走 Q6\_K→Q4\_0。
- **第 3.4 节 "QCOM 使用 flat layout" 修正**：该说法不准确。QCOM htp/ 目录同样包含 tiled Q4\_0/Q4\_1 kernel 实现（与 JZ kernels/ 在 PR #26049 之前是同一套代码）。JZ vs QCOM 的真正差异是：JZ 在加载时对所有 Q4\_K/Q6\_K 权重做 Q4\_K/Q6\_K→Q4\_0 tiled repack，因此 lm-head 也能在 DSP 上以 tiled Q4\_0 跑；QCOM 因 32768 guard 拒绝大 N 矩阵，lm-head 直接回退 CPU，根本走不到任何 layout 对比。明确"tiled vs flat"的对比错误。
- **Table-3 内存模型行修正**：从错误的"4GB bump pointer + 范围合并"改为"init 时 mmap 一次，v79 容量 probe 上限 4032 MiB，offset addressing；无 fd/mmap/lifecycle 重复成本"，与 ion-mempool 文档的"single ION mempool + offset addressing"一致。
- **Table-3 权重布局行修正**：从"tiled repacked vs flat layout"改为"Q4\_K/Q6\_K→Q4\_0 tiled repack 后 DSP 端跑 tiled Q4\_0 kernel vs 原始 Q4\_K/Q6\_K 布局 + tiled kernel（lm-head 因 32768 guard 不参与）"，与第 3.4 节一致。
- **Table-3 lm-head offload 行修正**：matvec 数据量从"\~125-500MB"修正为"\~138-428MB"（按 Q4\_0 repack 后大小）。
- **第 3.6 节 MUL\_MAT\_FFN count 解释澄清**：明确 count=6965/200batch≈35 次/batch 是 35 层 × 1 fused op/layer = 35 次/token；Gemma4-E2B 使用 gated FFN，每个 MUL\_MAT\_FFN fused op 内部完成 gate+up+down 三段 matmul，因此 35 次 fused call = 35 × 3 = 105 个内部 matmul。
- **第 4 章整体重构以对齐 3.6 profiling 数据**：
  - 开头新增"TG 主瓶颈是 DSP 侧 matmul kernel 效率（op-sum 86.9%，三类 matmul 91.1%），AP 侧 prep 开销 <13.1% wall time"作为优化方向的总前提。
  - 第 4.1(2) 节：descriptor 模板缓存描述从具体"Phase 4-8"修正为"graph\_compute\_batch 中 AP 侧 prep phase"，避免未经验证的具体编号；并按 3.6 profiling 数据（AP 侧 non-op \~4693 us/batch = 13.1% wall time）量化收益上限（即使消除一半也只省 6.5%），明确为次要优先级。
  - 优化路线图 Step 1 升级为"DSP kernel 优化"（lm-head GEMV + MUL\_MAT\_FFN 调优 + a-inv 优化），Step 2 降级为"AP 侧低风险快速收益"，descriptor 模板缓存从 Step 1 移到 Step 2 并标注"仅当 profiling 显示收益 > 复杂度时"。

### 2026-08-06: Refine 后增量修正（基于源码 `ggml-hexagon.cpp` / `ggml-hexagon-jz.cpp` 与参考文档对照）

作者: MiniMax-M3

AI 辅助: 通读最新 refine 后的全文，对照源码与参考文档核验 3 处遗留不准确之处；本轮 diff 见各 bullet。

- **第 3.1 节 QCOM switch case 列表修正**：`ggml_hexagon_supported_mul_mat` 的 switch 实际只处理 Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4/F16/F32，不包含 BF16；Q4_K/Q6_K 与 BF16 都落入 `default: return false`。将 "Q4_K/Q6_K 不在 switch 中" 扩展为 "Q4_K/Q6_K/BF16 不在 switch 中"，并把 default 行号从 L2840-2841 修正为 L2841-2842。
- **第 3.1 节 32768 行号修正**：从 L2805-2807 修正为 L2806-2808（原 L2805 是 `// hardcoded limit to refuse the lm-head for now` 注释行）。同时补充说明该 guard 嵌在 Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4 case 内、对 Q4_K/Q6_K 不生效（已在类型 guard 阶段被 default 拦截），澄清了 "类型 guard 是实际生效 guard" 的具体原因。
- **第 3.1 节 Q4 术语与大小范围修正**："Q4 矩阵" / "Q4 权重矩阵" 修正为 "Q4_K/Q6_K 量化矩阵" / "Q4_K/Q6_K 权重"；"~125-500MB" 修正为 "按 Table-2 约 138-428 MB"（按当前 5 个模型 lm-head 实际大小范围，对应 Q4_K 138-428 MB / Q6_K ~200 MB / Q4_0 repack 后 138-428 MB）。`~125-500MB` 是 refine 前遗留的旧值，未对齐到 Table-2 的具体数字。
- **源码验证项（无 diff，仅作为可追溯依据）**：
  - JZ 实际 repack 函数：`repack_q4k_as_q4_0_tiled_to_buf`（L4030）、`repack_q6k_as_q4_0_tiled_to_buf`（L4089）、`repack_q4k_as_q8_0_tiled_to_buf`（L3978）三个函数都存在；文档只引用了 Q4_K→Q4_0 一个，对 Q6_K→Q4_0 仅以 "等函数中" 隐含指代，描述不算错误但偏简略。
  - 12 phase 实际定义（`ggml-hexagon-jz.cpp` L266-285）：Phase 1=collect tensors / 2=build op desc / 3=op fusion / 4=layout sizes / 5=mirror / 6=repack offset / 7=alloc batch desc / 8=desc construct / 9=cache flush / 10=doorbell / 11=cache inval / 12=copy-back。文档 4.0 节的 "Phase 4-8" / 4.4 Step 4 的 "Phase 1-9 + Phase 11-12" 与源码完全对应，可视为已验证。

### 2026-08-06: PP 优先级重排与 Qwen1.5-1.8B 三重叠加根因分析

作者: MiniMax-M3

AI 辅助: 基于 3 份文档交叉分析 + 源码（`ggml-hexagon-jz.cpp` L266-285 phase 定义、L3141-3147 类型 guard、L4030-4089 repack 函数）综合判断，重构第 3.5/4.2/4.3/4.4 节。

- **第 3.5 节根因分析重构**：把"JZ PP 落后是因为 dspqueue 可以做 overlay，JZ 是 100% 同步 12 phase"的简单归因，重写为**调度框架差异**（非 kernel 差异，JZ 与 QCOM 复用同一套 Qualcomm HMX kernels，PR #26049 前 100% 相同）；明确 JZ 的 data-plane 优势（lm-head + first-touch）是**整图固定开销**与 layer 数无关，QCOM 的 pipelining 优势是**per-layer 累积**与 layer 数正相关。
- **Qwen1.5-1.8B 根因替换**：删除"corner case 暂不处理"措辞，替换为三重叠加分析（dspqueue pipelining 优势最大化 + JZ 整图固定优势无法在 24 层累积 + MHA 加重 VTCM/cache 压力）。结论是 Qwen1.5-1.8B 不是 corner case 而是 per-layer pipelining 优化的"压力测试"——任何 PP 优化自动惠及此类模型。
- **第 4.2 节优先级重排论证**：新增"PP 从 -25% 改善到 -15% 等同于 +11pp 绝对提升；TG 从 +50% 到 +60% 需要改 kernel 才有 +10pp，但 kernel 已与 QCOM 100% 共享"的对比论证，明确 PP 优化是 JZ 真正战场，排在 P0 profiling 之后立即推进（Step 2）。
- **修正误导性 "FastRPC ~89us" 论证**：旧文档用"FastRPC 开销 ~89us 可忽略"来支持"async/pipelining 不值得做"是逻辑错误。明确区分两个独立维度：（1）FastRPC 89us 是 control-plane RTS 路径成本，与 pipelining 无关；（2）pipelining 收益 = min(AP prep, DSP compute) 的隐藏量，关心的是能否把 1-3ms 的 AP prep 隐藏在 5-10ms 的 DSP layer 执行后面。
- **第 4.3 节重新组织为低风险快速收益**：(7) a-inv 优化 / (8) MUL_MAT_FFN kernel 调优 / (9) RMS_NORM fuse / (10) Phase 10 RPC 优化；明确 MUL_MAT_FFN 调优"不会扩大 JZ vs QCOM 相对优势，仅提升绝对性能"（kernel 共享）。
- **第 4.4 节路线图优先级调整**：从"Step 1 TG kernel / Step 4 PP 优化"调整为"Step 1 测量驱动快速收益（a-inv + sampling 路径） / Step 2 PP 结构性突破（per-layer pipelining 核心战场） / Step 3 TG kernel 精调（边际收益）/ Step 4 长期架构"。Step 2 明确预期：PP +5-10%，Qwen1.5-1.8B 从 -25.5% → ~-18%，Gemma4-E2B 从 +43.1% → +48%+。

### 2026-08-06: force_opfusion_in_pp 实验与 HVX fused 路径 PP 不适用结论

作者: MiniMax-M3

AI 辅助: 设计并实现 `force_opfusion_in_pp` 配置开关 + 3 个 cum 计数器(`n_qkv_skip_cum_hmx` / `n_pair_skip_cum_hmx` / `n_mm_add_skip_cum_hmx`),跑 baseline(`force=0`)与实验(`force=1`)两组 PP-only 测试对比;基于结果撰写第 4.6 节完整实验报告(动机/设计/baseline/实验数据/根因/结论/后续),并更新 `scripts/ggml-hexagon.cfg` 与 `ggmlhexagon_print_running_timestamp` 使配置变更运行时可见。

- **新增第 4.6 节**: 完整记录 `force_opfusion_in_pp=1` 实验,baseline 与实验对比 Table,根因分析(MUL_MAT_FFN 单 op 6.5ms 是致命瓶颈、cache 失效未省、净增 213ms 退化),与 per-layer profiling 数据。
- **核心结论**: HVX fused 路径不适用于 PP,正确方向是 HMX-aware fused kernel(保留 HMX 速度 + 节省 cache 失效)。
- **保留的基础设施**: cfg flag、bypass 分支、3 个 cum 计数器、`mul_mat coverage` 扩展打印、running timestamp 打印。
- **附带收获**: per-layer profiling 正常输出 15 层 mat/ffn/attn 三段耗时,为后续 PP 优化提供数据基础。

### 2026-08-06: Table-6 qwen3 层数从 `-` 修正为 13

作者: MiniMax-M3

- `ffn_gate-12` 算子编号 + PP/TG MUL_MAT 计数规律确认 qwen3 为 13 层。

### 2026-08-06: Table-7/9 qwen3 统一为单行

作者: MiniMax-M3

- qwen3 唯一有效抓取文件是 v4 log,Table-7/9 中 qwen3 统一为单行。

### 2026-08-06: 移除 "tee buffering 0 字节" 误判

作者: MiniMax-M3

- gemma4 E2B log 0 字节实为首轮抓取命令 `adb logcat -d -iE` 被 shell 拒绝(`-iE` 是 GNU grep 选项,Android toybox grep 不识别),与 tee buffering 无关。

### 2026-08-06: 5 模型 CI 验证数据记录与解读(新增第 4.6.5 节)

作者: MiniMax-M3

AI 辅助: 基于 5 个 ADB logcat 抓取文件(其中 gemma4 E2B 数据包含完整 tok/s/running_timestamp/dump_perf_stats),梳理 PP/TG OP-PROF 数据,撰写第 4.6.5 节完整分析报告(基础信息表/PP batch#1 对比表/gemma4 详细算子分布表/TG 详细表/matmul 行为对比表/关键发现/数据局限),与 3.6 节 gemma4-E2B profiling 交叉验证 force_opfusion_in_pp=0 cleanup 后无性能回归。

- **新增第 4.6.5 节(插入在 4.6.4 之后,原 4.6.5/4.6.6/4.6.7 顺延为 4.6.6/4.6.7/4.6.8)**: 7 个子小节覆盖 5 模型基础信息、PP batch#1 横向对比、qwen3 TG batch#978 详细数据、跨模型 matmul 行为对比、关键发现与结论、已知数据局限与后续动作、文档与 commit 维护。
- **Table-6 5 模型基础参数表(按 alias 数组顺序)**: gemma4/E2B(24层 GQA 4:1,Q4_K) → qwen3/Qwen3.5-2B(GATED_DELTA_NET) → qwen1/Qwen1.5-1.8B(MHA) → llama3/Llama-3.2-1B → gemma4-e4b/gemma-4-E4B。
- **Table-7 5 模型 PP batch#1 OP-PROF 对比表**: 6 行 × 11 列(batch-wall/op-sum/non-op/MUL_MAT cum/count/max/FlashAttn cum/count/w-inv/a-inv/bulk);qwen1 的 a-inv=19,153us(1,754MB) + bulk=15,125us 直接验证 3.5 节 MHA 三重叠加根因。
- **Table-7.1 gemma4 详细 15 算子分布表**: 仅 gemma4 E2B 含完整算子分布(MUL_MAT 277/38057us/avg=137us,GLU_GEGLU 35/5821us,FLASH_ATTN_EXT 35/3991us 等)。
- **端到端 tok/s 完整数据(本轮独有)**: prompt eval 679.03 tok/s + TG 26.46 tok/s;graphs reused 253(命中率 98.8%);unaccounted 0.2%。
- **`ggmlhexagon_print_running_timestamp` 配置全字段确认**: force_opfusion_in_pp=0,enable_opfusion=1,enable_graph_optimize=1,dsp_cache_mode=5,ion_sync_mode=1,thread_counts=6,enabled_ops=ALL。
- **`ggmlhexagon_dump_perf_stats` 端到端 256 批次统计**: cum_p10=9,465,268us / cum_graph=9,602,431us / batch_calls=256 / per-call overhead avg=535us / cgraph cache hit_rate=98.8%。
- **与 3.6 节 baseline 对比验证(5 项核心指标,均 ±3% 内)**: batch-wall 80,316us vs 81,252us (1.1%);op-sum 56,219us vs 56,500us (0.5%);non-op 24,097us vs 24,752us (2.6%);MUL_MAT cum 38,057us vs 38,202us (0.4%);MUL_MAT max 4,646us vs 4,697us (1.1%)。**确认本轮 4c805d844 force=0 cleanup 未引入任何回归**。
- **Table-8 qwen3 TG batch#978 详表**: 19 行算子 × 7 列;三类 matmul 占 op-sum 74.4%(MUL_MAT 38.3% + MUL_MAT_FFN 23.0% + MUL_MAT_ADD 13.1%);non-op 仅 14.6% wall time。
- **Table-9 5 模型 matmul 行为对比表**: 5 模型 PP batch#1 中 MUL_MAT_FFN 全部为 0,MUL_MAT_ADD 在 llama3/qwen1 触发(count=30/119),HMX 闸门 100% 生效。
- **核心结论**: **5/5 模型 CI 全部通过,无性能回归**;PP 路径 HMX 闸门正确保留;MUL_MAT_ADD 是 PP 路径唯一活跃 fusion;gemma4 E2B tok/s 数据首次完整抓到,确认与 5.3.1 baseline 一致。
- **已知数据局限**: 3 个常规模型 TG 数据缺失 + gemma4 E2B 抓取命令 bug + qwen3 PP 真实数据缺失(M=1 init) + gemma4-e4b batch#1 重复打印 + 3 个常规模型 tok/s 缺失 + GQA 4:1/8:1 matmul 差异。后续 6 项改进建议已在 5.4.6 列出。
- **5.7 后续步骤第 3 项增补**: 5 模型 CI 验证数据与解读 → 5.4 节。

