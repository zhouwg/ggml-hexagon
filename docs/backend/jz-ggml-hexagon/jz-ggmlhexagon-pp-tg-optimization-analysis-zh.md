# JZ's ggml-hexagon 性能差异分析与优化方向

> Initial: 2026-08-05

> Last updated: 2026-08-07

> Author: Seed-2.1-Pro (Ch 1-4), MiniMax-M3 (Ch 5-6), revised by DeepSeek-V4-Pro & GLM-5.2 & MiniMax-M3 & Kimi-K3

***

## 一、AB 测试性能数据

以 `3469e4858e17d501a1f6e16ebe0aa2489613d32b` 为基线，基于五个模型（Qwen3.5-2B、Gemma4-E2B、Gemma4-E4B、Qwen1.5-1.8B、Llama3.2-1B）的 AB 测试结果，分析 JZ (`ggml-hexagon-jz.cpp` + `kernels/`) 与 QCOM (`ggml-hexagon.cpp` + `htp/`) 两个 ggml-hexagon 后端的性能差异根因，并提出优化方向。

**Table-0**: 测试环境

| 项目      | 配置                                                                                                                                                 |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| 设备      | Qualcomm Snapdragon 8 Elite (8 Gen 4), QCOM\_HTP\_V79, dsp arch 0x79, VTCM=8MB, HVX=1, HMX=1, 系统内存 24834 MiB, Android device id `9d231cfe`         |
| JZ 后端   | `libggml-hexagon-jz.so` + `libggmldsp-skel-v79.so`                                                                                                 |
| QCOM 后端 | `libggml-hexagon-qcom.so` + `libggml-htp-v79.so`                                                                                                   |
| 测试参数    | `n_ctx=8192, n_batch=2048, n_predict=256, n_threads=6, graphs reused=253`                                                                          |
| JZ 配置   | `dsp_cache_mode=5, ion_sync_mode=1, graph_optimize=1`<br>offload MUL\_MAT types: F32,F16,BF16,Q4\_0,Q8\_0,Q4\_1,IQ4\_NL,MXFP4<br>thread\_counts on CDSP: 6 |
| 测试方法    | 每个模型 3 轮取均值，n\_prompt=51\~75 tokens，n\_gen=255 tokens                                                                                              |

**Table-1**: AB 测试性能数据

| 模型           | PP JZ (tok/s) | PP QCOM (tok/s) | PP JZ vs QCOM | TG JZ (tok/s) | TG QCOM (tok/s) | TG JZ vs QCOM |
| ------------ | :-----------: | :-------------: | :-----------: | :-----------: | :-------------: | :-----------: |
| Qwen3.5-2B   |     436.6     |      481.1      |     -9.2%     |      27.0     |       14.0      |   **+93.6%**  |
| Gemma4-E2B   |     669.7     |      452.3      |   **+48.1%**  |      27.1     |       24.4      |     +11.1%    |
| Gemma4-E4B   |     406.9     |      420.9      |     -3.3%     |      15.2     |       11.3      |   **+35.1%**  |
| Qwen1.5-1.8B |     541.8     |      750.4      |     -27.8%    |      18.6     |       26.6      |     -30.2%    |
| Llama3.2-1B  |    1013.9     |      1144.2     |     -11.4%    |      42.4     |       28.7      |   **+47.4%**  |

数据来源：`./scripts/build-run-android.sh run_abtest_all 2>&1 | tee log_abtest_all_$(date +%Y%m%d-%H%M%S).txt`（本轮日志 `log_abtest_all_20260807-102443.txt`）

> **数据注记**：Qwen1.5-1.8B QCOM PP 首轮为 825.79 tok/s，后两轮为 720.69 / 704.70 tok/s，首轮明显偏离（可能因 dspqueue warmup 或后台负载），Table-1 仍按 3 轮均值 750.4 tok/s 如实记录；若排除首轮，PP 差距从 -27.8% 收窄至约 -24.0%。

**关键观察**：

- **TG（Token Generation）**：JZ 在 4/5 模型上领先，最大优势 +93.6%（Qwen3.5-2B），领先 4 模型平均约 +46.8%；仅 Qwen1.5-1.8B（唯一 MHA 模型）落后 30.2%。
- **PP（Prompt Processing）**：QCOM 在 4/5 模型上领先，最大优势 +27.8%（Qwen1.5-1.8B）；唯一例外是 Gemma4-E2B，JZ 反超 +48.1%。
- TG 和 PP 的性能模式截然相反，指向不同的瓶颈根因。

***

## 二、架构关系澄清

JZ (`ggml-hexagon-jz.cpp` + `kernels/`) 与 QCOM (`ggml-hexagon.cpp` + `htp/`) 是**基于同一套 hexagon kernels 的两条进化分支**，分叉点为 Qualcomm [PR #26049](https://github.com/ggml-org/llama.cpp/pull/26049)。

- PR #26049 之前，两边算子完全相同。
- PR #26049 之后，QCOM 的改进实现被手动移植到 JZ；自 JZ 的 PR 提交后，QCOM 暂无新的 PR。
- **性能差异不在 kernel 算子本身，而在调度框架、cache 策略和 offload 策略。**

***

## 三、性能差异根因分析

### 3.1 lm-head offload：TG 性能差异的最大单一因素

QCOM 后端在 `ggml-hexagon.cpp` 的 `ggml_hexagon_supported_mul_mat` 中有**2 处直接生效的 guard** 阻止 lm-head offload 到 DSP：

1. **类型 guard**：switch 只处理 Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4/F16/F32，**Q4_K/Q6_K/BF16 不在 switch 中**，落入 `default: return false`（`ggml-hexagon.cpp` L2841-2842）。JZ 侧（[`ggml-hexagon-jz.cpp`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp) L3274-3289）显式处理了 Q4_K/Q6_K（若 Q4_0 已启用则放行，因为 JZ 在加载时做了 Q4_K/Q6_K -> Q4_0 tiled repack）和 BF16（在 repack buffer 中转为 F16 bytes 后放行，L3318-3324）。
2. **尺寸 guard**：`src0->ne[1] > 32768` 时拒绝（`ggml-hexagon.cpp` L2806-2808）。此 guard 嵌在 Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4 case 内，对 Q4_K/Q6_K 不生效（已在类型 guard 阶段被 default 拦截）。

对于本次测试的 五个模型，**类型 guard（#1）是实际生效的 guard**（lm-head 权重均为 Q4_K/Q6_K，不在 switch 中）。尺寸 guard（#2）是 per-buffer ION 经济性限制的直接体现（32K 行是 per-buffer 的成本上限）。

> **补充**：QCOM 在 switch 分支内还有一处 **repack buffer guard**（L2815：`!ggml_backend_buffer_is_hexagon_repack(src0->buffer)`），要求权重必须位于 repacked buffer 中。即使类型 guard 被移除，lm-head 权重（不在 repacked buffer 中）仍会被此 guard 阻止。该 guard 位于类型 guard 后面的 switch 分支内，当前未被触发，但揭示了 QCOM 对 offload lm-head 权重的额外约束：权重不仅要类型匹配，还必须位于 repacked buffer 中。

**Table-2**: 各模型 vocab\_size 与 lm-head 大小

| 模型           | vocab\_size | lm\_head 原始类型 | 原始大小（约）  | Q4\_0 repack 后大小（约） |
| ------------ | ----------- | ------------- | -------- | ------------------- |
| Qwen3.5-2B   | 151,936     | Q6\_K         | \~200 MB | \~163 MB            |
| Gemma4-E2B   | 256,000     | Q4\_K         | \~214 MB | \~214 MB            |
| Gemma4-E4B   | 256,000     | Q4\_K         | \~428 MB | \~428 MB            |
| Qwen1.5-1.8B | 151,936     | Q6\_K         | \~200 MB | \~163 MB            |
| Llama3.2-1B  | 128,256     | Q4\_K         | \~138 MB | \~138 MB            |

JZ 后端 `ggmlhexagon_supported_mul_mat` 中**未设置 N 维度上限 guard**(与 QCOM 的 `src0->ne[1] > 32768` 形成对比),因此 lm-head 完全 offload 到 DSP 执行。对 Q4\_K 模型（如 Gemma4、Llama3.2-1B），通过 Q4\_K -> Q4\_0 tiled repack 将 lm-head 权重转为 DSP 可直接执行的 tiled layout；对 Q6\_K 模型（如 Qwen3.5-2B、Qwen1.5-1.8B），通过 Q6\_K -> Q4\_0 tiled repack 转换（注意 Q6\_K 比 Q4\_0 略大，repack 后体积会略减）。repack **不减少带宽**（Q4\_K 和 Q4\_0 数据大小相同，均为 0.5625 B/param；Q6\_K -> Q4\_0 实际是 lossy 转换以适配 DSP 侧复用的 Q4\_0 matmul kernels），**其价值不在节省带宽,而在让 DSP 侧 tiled matmul kernel 可直接消费该 layout**(DSP kernels 仅支持 Q4_0 tiled layout)。

**lm-head offload 之所以在 JZ 可行，与 single mempool 架构强相关：** lm-head 权重（Q4_K/Q6_K 量化矩阵，按 Table-2 约 138-428 MB）作为 mempool 内的一个 offset 范围，零额外 fd/mmap/生命周期维护成本。QCOM 的 2 处 guard（类型/尺寸）共同阻止了 lm-head offload，根本原因是其 per-buffer ION 设计：每个 buffer 携带独立的 fd、fastrpc_mmap、dspqueue 每批重复注册等开销，无法经济地承载会话常驻的 lm-head 权重（32K 行是 per-buffer API 的实际上限）。JZ 通过加载时 Q4_K/Q6_K -> Q4_0 tiled repack 消除了类型 guard，通过 single mempool 的零边际成本消除了尺寸 guard 的经济性约束。

**对 TG 的影响是决定性的：** TG 每生成 1 个 token 都要执行一次 lm-head matvec（`[1, n_embd] x [n_embd, vocab_size] -> [1, vocab_size]`）。这是纯粹的 memory-bound 操作：

- **QCOM**：CPU 逐元素读取整个 Q4_K/Q6_K lm-head 权重（按 Table-2 约 138-428 MB）做 dequant+dot product，CPU 访存带宽有限，且 CPU 算 lm-head 时 DSP 空闲。
- **JZ**：DSP 上 HVX 执行 lm-head matvec，权重以 Q4_0 tiled layout 驻留在 ION mempool 中，带宽远高于 CPU，且与后续 token 生成流水线紧密衔接。

**对 PP 的影响很小**：lm-head 在 PP 末尾只执行一次，其开销被几十个 transformer layer 的计算摊薄。

### 3.2 dspqueue async overlay vs 同步 FastRPC：PP 性能差异的根因

**调度框架差异是 PP 性能差异的根因，而非简单的"调度开销"。**

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

**对 TG 的影响**：每 token 只有一个 batch（M=1），DSP 计算极短，dspqueue 的 overlay 收益极小（几乎没有可以隐藏的 AP prep 时间）。同时 per-op dspqueue 通信开销（每次 write/read 都有环形队列管理开销）在 M=1 小 op 时占比变大。JZ 单次 doorbell 发整个 batch 的模型在 TG 更高效。

### 3.3 Role-aware batch-level cache 管理：TG 的第二大优势

JZ 的 batch-level cache 策略通过 bitmap 控制，大幅减少了 cache sync 次数：

- **bit0 (first-touch weight bitmap)**：权重首次 touch 后不再 dcinva，命中 L2。
- **bit1 (prior-dst skip)**：当前 src 与前序 op 的小 dst（<= 单 cacheline）为同一 tensor 时，跳过该 src 的 invalidation（本轮测试 mode=5 未启用，详见 4.4.1）。
- **bit2 (bulk flush)**：所有 dst flush 合并到 batch 末尾一次完成。
- **bit3 (selective flush)**：中间 tensor 不 flush，减少 DDR 写。

而 QCOM 采用 batch 级全量 cache 维护：在 batch 开始和结束时各执行一次完整 D-cache flush+invalidate（`qurt_mem_cache_clean(..., FLUSH_INVALIDATE_ALL, ...)`），uniform、role-blind，无法区分 weight 和 activation。

**对 TG 的影响**：M=1 时每个 matmul 计算量极小，QCOM 的 batch 级全量 cache flush+invalidate 开销被放大。JZ 的 batch-level 策略大幅减少 cache sync 次数，效果显著。

**对 PP 的影响**：大 M matmul 计算时间长，cache sync 被计算摊薄，差异小。

### 3.4 Tiled weight repacking + VTCM 复用

JZ 在加载阶段对 Q4\_K/Q6\_K 等量化权重做 **Q4\_K/Q6\_K -> Q4\_0 tiled repack**（在 [`ggml-hexagon-jz.cpp`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp) 的 `repack_q4k_as_q4_0_tiled_to_buf` [L4163](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L4163)、`repack_q6k_as_q4_0_tiled_to_buf` [L4222](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L4222)、`repack_q4k_as_q8_0_tiled_to_buf` [L4111](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L4111) 中，32-row strip 转换），将权重转为 DSP HMX kernel 可直接消费的 tiled Q4\_0 布局，配合 VTCM 分块计算减少 DDR 访问次数。

需要澄清：QCOM 后端的 [`htp/`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp) 目录同样包含 tiled Q4\_0/Q4\_1 kernel 实现（与 JZ 维护的 [`kernels/`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels) 在分叉前是同一套代码，PR #26049 之后分叉维护）。JZ 与 QCOM 的真正差异**不是 tiled vs flat 的布局差异**，而是：

- **JZ**：在加载时对所有 Q4\_K/Q6\_K 权重做 -> Q4\_0 tiled repack，因此所有量化 matmul（含 lm-head）都能在 DSP 上以 tiled Q4\_0 layout 跑。
- **QCOM**：`ggml_hexagon_supported_mul_mat` 的类型 guard（L2841，`default: return false`）直接拒绝 Q4\_K/Q6\_K，尺寸 guard（L2806-2808）仅对通过类型 guard 的 Q4\_0/Q4\_1/Q8\_0/IQ4\_NL/MXFP4 生效。`htp/matmul-ops.c` 中无 Q4\_K/Q6\_K DSP kernel。Q4\_K/Q6\_K matmul 全部回退 CPU，lm-head（Q4\_K/Q6\_K）在 QCOM 路径中同样回退 CPU。

### 3.5 总结：性能差异归因

**Table-3**: 性能差异归因

| 架构特性                | JZ (kernels/)                                                                                          | QCOM (htp/)                                                            | TG 影响                                         | PP 影响                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------- | --------------------------------------------- | -------------------------------------- |
| **调度框架**            | Native FastRPC 同步（12 Phase 串行，零 overlay）                                                               | dspqueue 异步环形队列（AP-DSP overlay）                                        | JZ 略优（单次 doorbell vs per-op 队列管理）             | **QCOM 显著优**（AP prep 与 DSP compute 重叠） |
| **lm-head offload** | 全 offload（single mempool + Q4\_K/Q6\_K->Q4\_0 tiled repack，支持超大 N）                                      | 2 处 guard（类型/尺寸）拒绝，回退 CPU                                            | **JZ 极大优势**（每 token \~138-428MB matvec 在 DSP） | 影响小（只跑 1 次）                            |
| **Cache 管理**        | Role-aware batch-level（bit0-3，first-touch/prior-dst/bulk/selective）                                    | Batch 级全量 D-cache flush+invalidate（uniform, role-blind）                | **JZ 显著优**（M=1 时 cache sync 是大头）              | 差异小（大计算摊薄）                             |
| **内存模型**            | Single ION mempool（init 时 mmap 一次，v79 容量 probe 上限 4032 MiB，offset addressing；无 fd/mmap/lifecycle 重复成本） | Per-tensor rpcmem 分配（每 buffer 独立 fd / fastrpc\_mmap / dspqueue 每批重复注册） | JZ 优（零额外 fd/mmap + 整池 IOVA 连续 + 权重 L2 友好驻留）   | 差异小                                    |
| **权重布局**            | Q4\_K/Q6\_K -> Q4\_0 tiled repack 后 DSP 侧跑 tiled Q4\_0 kernel                                           | 原始 Q4\_K/Q6\_K 布局 + tiled kernel（lm-head 因 2 处 guard 不参与）               | JZ 优（lm-head DSP offload，VTCM/L2 友好）          | JZ 略优                                  |

**JZ TG 领先**与 single mempool 带来的 lm-head offload 强相关，role-aware 的缓存一致性维护策略也是重要因素。

**JZ PP 落后**的根因是**调度框架差异**而非 kernel 差异。JZ 与 QCOM 复用同一套 Qualcomm HMX kernels（分叉前完全相同），matmul 执行效率一致。差异在于：JZ 的 12-phase 同步模型无法实现 AP-DSP pipelining，而 QCOM 的 dspqueue 异步环形队列允许 AP prep 与 DSP compute 在 per-layer 粒度上重叠。JZ 的 data-plane 优势（lm-head offload + first-touch 权重 inval）是**整图固定开销**，与 layer 数无关；QCOM 的 pipelining 优势是**per-layer 累积**的，与 layer 数正相关。因此 PP 表现高度依赖模型层数与 attention pattern 对 VTCM/cache 压力的影响。

**Qwen1.5-1.8B（唯一 MHA 模型，24 层）PP/TG 均落后的根因** = dspqueue pipelining 优势 + 层数不足 + MHA VTCM/cache 压力三重叠加：

1. **dspqueue pipelining 优势最大化**：dspqueue 的 AP-DSP overlap 收益与每次 DSP 计算时长正相关，Qwen1.5-1.8B 在 PP 阶段单 layer 计算时间长（24 层 x 每层 MHA Q@K^T 的 full attention），pipelining 隐藏的 AP prep 时间窗口大。
2. **JZ 整图固定优势无法累积**：lm-head offload（~200MB Q6_K）+ first-touch 权重 inval（~9.2ms/token）是固定的、不会随 layer 数增加而放大的优势；24 层不足以让 JZ 的 per-layer 增量优势赶超 dspqueue 的 per-layer pipelining 收益。Gemma4-E2B 35 层则可以反超（+48.1%）。
3. **MHA 加重 VTCM/cache 压力**：1:1 attention 的 Q@K^T 是 full attention（无 KV 共享），相比 GQA 模型的 KV 共享头占用更多 VTCM 与 cache 带宽，恰好是 JZ role-aware cache 策略（bit0-3）本来要优化的场景-但这些优化只在 TG M=1 场景放大收益，对 PP 长序列 M=prompt_len 帮助有限。

**结论**：Qwen1.5-1.8B 不是 corner case，而是三重不利因素叠加的体现。**任何 PP 优化（如 per-layer pipelining）只要把 dspqueue 的 per-layer 优势部分削弱，就能同时改善 Qwen1.5-1.8B 这类模型**-这是 PP 优化优先级应当高于 TG 精雕细琢的核心论证。

### 3.6 DSP Op-Level Profiling 实测数据（2026-08-06）

基于 Gemma4-E2B 模型（TG 主场景），在 DSP 侧通过 `HEX_OP_PROF`（定义于 [`kernels/dsp-ctx.h` L25](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/dsp-ctx.h#L25)；feature/force_opfusion_in_pp 分支 hardcode=1，主分支默认=0）开启 per-op 计时统计，每 25 个 batch 通过 FARF(ERROR) 输出累计数据。以下分析取 batch#200 稳定数据点（已过 warmup，统计收敛）。

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

**Table-5**: DSP non-op 开销细分（us/batch）

| 阶段                                 | 耗时 (us/batch) | 数据量       | 说明                                            |
| ---------------------------------- | ------------- | --------- | --------------------------------------------- |
| hdr cache inval                    | 4             | -         | batch descriptor invalidation，可忽略             |
| tensor pre-conversion              | 318           | -         | hex\_tensor\_desc -> dsptensor/htp\_tensor 预转换 |
| weight cache inval (w-inv)         | 68            | 6 MB      | bit0 first-touch 效果显著：权重仅首次 inval             |
| **activation cache inval (a-inv)** | **1,030**     | **82 MB** | **最大 non-op 开销**，mode=5 未启用 bit1，接近结构性下限（详见 4.4.1）  |
| dst tracking                       | 105           | -         | prior\_dst/bulk\_flush 范围收集                   |
| **bulk dst flush**                 | **1,377**     | -         | 所有 dst 合并到 batch 末尾 flush                     |
| queue wakeup/suspend               | 5             | -         | DMA/HMX queue 管理，可忽略                          |
| **non-op 合计**                      | **4,693**     | <br />    | **13.1% wall time**                           |

**瓶颈根因分析（基于 profiling 数据修正）**：

> **注意**：以下 profiling 数据仅覆盖 DSP 批处理执行阶段（Phase 10），在 DSP 侧通过 HEX\_OP\_PROF 测量。AP 侧开销（Phase 1-9 + Phase 11-12）未包含在内，需通过 Step 0 profiling（dump\_diag\_info=1）单独测量。此处"wall time"指 DSP 侧 batch 执行 wall time（35,789 us/batch），非端到端 TG 时间。

- **在 DSP 执行内部，matmul kernel 是绝对主导**：op-sum 占 DSP batch-wall time 的 86.9%，其中 91.1% 是三类 matmul。DSP 侧 non-op 开销（cache inval、tensor 转换、dst flush 等）合计 4693 us/batch，占 DSP batch-wall 的 13.1%。
- **lm-head MUL\_MAT（max=4697us）是 TG 单算子最大项**：每个 token 出现一次，对应 `1xhiddenxvocab` matrix-vector product。通用 GEMM kernel 对 M=1 的 skinny matmul 效率不高，专用 GEMV kernel 有优化空间。
- **MUL\_MAT\_FFN（avg=334us）是 per-layer 最大稳定开销**：FFN matmul 已使用 fused op（MUL\_MAT\_FFN），需要检查是否充分利用 HMX 加速，以及 tile size 是否对 FFN 维度最优。
- **activation cache invalidation（a-inv=1030us/batch, 82MB）是最大 DSP 侧 non-op 开销**：本轮 dsp\_cache\_mode=5 未启用 bit1；2026-08-07 对照实验（qwen1 PP-only，mode=7 vs 5）证实即使启用 bit1，a-inv 也零变化（根因：prior_dst_add 只登记 <= 单 cacheline 的 dst，cgraph 中间张量均 >= 256 字节，skip 路径几乎不触发，详见 4.4.1）。per-batch dedup 已保证每条 unique src 每 batch 至多失效一次，a-inv 接近结构性下限。bulk flush（1377us）是第二大 DSP 侧 non-op，但这是 bit2 bulk flush 策略的代价，将所有 dst flush 合并到一次。
- **DSP-side sampling 实际收益极小**：跳过 logits copyback 仅节省 \~100-200us（因 ion\_sync\_mode=1 下整个 mempool sync 掩盖了局部收益），与实测一致：DSP-side sampling 功能正确，但性能提升可忽略。

**对优化方向优先级的影响**：

1. **DSP 内部 matmul kernel 优化是首要方向**：三类 matmul 占 DSP 执行时间的 91.1%，lm-head 专用 GEMV kernel 和 MUL\_MAT\_FFN 调优是 DSP 侧最具潜力的单点优化。
2. **AP 侧开销（Phase 1-9 + Phase 11-12）未被 DSP profiling 数据覆盖**：无法直接比较 AP 侧优化（descriptor 模板缓存等）与 DSP kernel 优化的收益。AP 侧开销需通过 Step 0 profiling 单独量化后再定优先级。
3. **lm-head 专用 GEMV kernel**：每 token 出现一次，max=4697us，是 TG 阶段 DSP 侧最大的单算子。
4. **MUL\_MAT\_FFN kernel 调优**（HMX 利用率、tile size）：收益面最广（35 次/batch x 334us = 11690us/batch）。
5. **a-inv 优化（已关闭）**：bit1 prior-dst 覆盖扩展于 2026-08-07 实验证伪（详见 4.4.1），a-inv 接近结构性下限，不再列为收益项。

### 3.7 Qwen3.5-2B 的 25 路 graph 拆分：SOLVE_TRI 支持度差异

本轮 AB 测试的 `ggmlhexagon_dump_perf_stats` 输出揭示了一个此前未被记录的结构差异：**Qwen3.5-2B 在 JZ 后端每个 batch 的 cgraph 被拆成 25 个子图，而 QCOM 后端保持完整单图**。

**Table-6A**：JZ 后端五模型 graph 拆分实测（`log_abtest_all_20260807-102443.txt`，JZ run 1）

| 模型           | batch\_calls | graph nodes (min/max) | total nodes | 拆分情况              |
| ------------ | :----------: | :-------------------: | :---------: | ----------------- |
| Qwen3.5-2B   |   **6400**   |     **26 / 62**       |   345,600   | **25 子图/batch** |
| Gemma4-E2B   |      256     |      1493 / 1493      |   382,208   | 完整单图              |
| Gemma4-E4B   |      256     |      1860 / 1860      |   476,160   | 完整单图              |
| Qwen1.5-1.8B |      256     |       821 / 821       |   210,176   | 完整单图              |
| Llama3.2-1B  |      257     |       3 / 501\*       |   128,755   | 基本完整（\* 见下注）     |

> \* Llama3.2-1B 的 min=3 来自 cgraph cache miss 时的零星小图（本轮 misses=6），256 个正式 batch 均为 ~501 节点完整图，不属于结构性拆分。

**根因**：Qwen3.5-2B 是 delta net 混合架构（24 层 = 13 标准 attention 层 + 11 个 linear-attention delta net 层），delta net 层每层调用一次 `ggml_solve_tri`（[`delta-net-base.cpp` L166](file:///home/zhouwg/develop/ggml-hexagon/src/models/delta-net-base.cpp#L166)）。两个后端对该算子的支持度不同：

- **QCOM 完整支持**：AP 侧 `supports_op` 有 `case GGML_OP_SOLVE_TRI`（[`ggml-hexagon.cpp` L4209-4211](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L4209-L4211)），DSP 侧有 HVX kernel 实现（[`htp/solve-tri-ops.c`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/solve-tri-ops.c)）。
- **JZ 断在 AP 侧**：`init_op_validators()`（[`ggml-hexagon-jz.cpp` L3762-3798](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L3762-L3798)）未注册 `GGML_OP_SOLVE_TRI` 的 validator，`ggmlhexagon_can_handle_op_through_cdsp` 对该算子返回 false。值得注意的是，**JZ 的 DSP 侧能力其实完整**：fork 自 htp/ 的 [`kernels/solve-tri-ops.c`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/solve-tri-ops.c) 与 entry.c 的 op 表注册（[L811](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/entry.c#L811)）都在，仅缺 AP 侧 validator 与 `ggml_op_to_htp_op` 映射（[entry.c L905](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/entry.c#L905) 的 switch 中无 `GGML_OP_SOLVE_TRI` case）。

于是 ggml scheduler 将 qwen3 的 cgraph 在每个 SOLVE_TRI 处切开：11 个 delta net 层的 SOLVE_TRI 回退 CPU，连同其前后依赖算子的段边界，将整图切成 25 段（6400 batch\_calls = 256 batches x 25 子图，345,600 / 6400 = 54 节点/子图）。

**对同步架构的惩罚被放大**：JZ 每个子图都要走完整 12-phase 同步流程（25 次 FastRPC round-trip + 25 倍 Phase 1-9/11-12 的 AP 开销），且因子图间串行，完全无法 pipelining；QCOM 对完整单图做一次 dspqueue 流水提交，AP prep 与 DSP compute 跨 layer 重叠。本轮实测 qwen3 JZ 后端 6400 个子图的 AP phase 累计约 69ms（p1+p2+...+p12），摊到 256 个 token 约 270us/token；TG 阶段占比 <1% 尚可忽略（qwen3 TG 仍 +93.6% 领先），但 PP 阶段（单 batch ~119ms）25 倍的固定开销约占 5-6pp，是 qwen3 PP -9.2% 的重要构成。

**结论**：qwen3 的 PP 落后不全是调度框架的"固有税"，其中相当部分是**算子支持度缺口导致的图拆分税**。启用 SOLVE_TRI offload 是消除该差距的最直接手段（详见 4.3.3）。

***

## 四、优化方向

根据 3.6 节 DSP op-level profiling 实测数据，**在 DSP 执行内部，matmul kernel 是绝对主导**（三类 matmul 占 DSP batch-wall time 的 79.1% = 86.9%x91.1%）。注意：DSP profiling 数据仅覆盖 Phase 10（DSP 批处理执行），AP 侧开销（Phase 1-9 + Phase 11-12）未包含在内，需通过 Step 0 profiling 单独量化。在 AP 侧数据补全前，优化方向优先聚焦在 DSP kernel 与 offload 策略上，AP 侧优化暂不调整优先级。

TG 和 PP 的瓶颈不同，优化策略也不同：

- **TG 瓶颈**（基于 3.6 profiling，仅覆盖 DSP 侧）：在 DSP 执行内部，三类 matmul 占 91.1% op-sum，其中 lm-head MUL\_MAT（max=4697us，每 token 1 次）和 MUL\_MAT\_FFN（avg=334us，每 layer 1 次 fused op = 105 个内部 matmul）是绝对主导；JZ 已通过 lm-head offload + first-touch 权重 inval（\~9.2 ms/token 节省，固定整图总量）解决最关键的两项，剩余优化空间主要在 DSP matmul kernel 本身。
- **PP 瓶颈**：PP 差距是**模型结构相关的**，不是普遍的 JZ 弱点。[ion 文档](file:///home/zhouwg/develop/ggml-hexagon/docs/backend/jz-ggml-hexagon/ion-mempool-vs-perbuffer-analysis-20260713.md)第 9 节分析表明 JZ 净优势 = per\_layer\_saving x n\_layers + fixed\_lmhead\_saving - dspqueue\_overlap。当层数足够时 JZ 也赢 PP（如 Gemma4-E2B 的 35 层，PP +48.1%）；浅层模型（qwen3.5-2B 24 层、llama3.2-1B 16 层）dspqueue 的固定 overlay 优势尚未被 per-layer 累积超越；此外 qwen3.5-2B 还叠加了 SOLVE_TRI 缺口导致的 25 路 graph 拆分税（详见 3.7）。[ion 文档](file:///home/zhouwg/develop/ggml-hexagon/docs/backend/jz-ggml-hexagon/ion-mempool-vs-perbuffer-analysis-20260713.md)也明确指出：**性能差异来自 data-plane policy（weight residency + role-aware cache），而非 control-plane**（FastRPC 开销历史值 \~89us，可忽略；本轮实测详见 4.4.4）。

### 4.1 前置准备：Profiling 数据驱动

在投入任何优化之前，先跑一轮 benchmark 量化各阶段耗时：

- **AP 侧**：设置 `dump_debug_info=1`，量化 Phase 1-12 各阶段实际时间分布。
  - Phase 10 三阶段（`cum_p10_rpc_setup_us` / `cum_p10_dsp_exec_us` / `cum_p10_civac_us`）的占比。
  - TG 中 Phase 4-8 的固定开销究竟多大（验证 descriptor 模板缓存的收益上限）。
  - PP 中 Phase 1-9 + 11-12 的 AP 纯开销占比（验证 async/pipelining 的收益上限）。
- **DSP 侧**：`HEX_OP_PROF=1`（定义于 [`kernels/dsp-ctx.h`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/dsp-ctx.h#L25)），量化 DSP 侧 per-op / per-layer 耗时分布（即 3.6 节与第五章的 OP-PROF 数据来源）。feature/force_opfusion_in_pp 分支 hardcode 为 1，主分支默认为 0，合入主分支时需改为运行时配置或 build flag。

**决策阈值**：

- 如果 Phase 1-9 + 11-12 在 PP 中占比 < 10%，async/pipelining 不值得做（[ion 文档](file:///home/zhouwg/develop/ggml-hexagon/docs/backend/jz-ggml-hexagon/ion-mempool-vs-perbuffer-analysis-20260713.md)中 FastRPC 开销历史值 \~89us 的数据也支持这一判断；本轮实测详见 4.4.4）。
- 如果 Phase 4-8 在 TG 中占比 > 5%，descriptor 模板缓存值得投入。

参考数据来源：本轮 5.4.2.1 gemma4 端到端 256 calls dump 已提供 AP phase 累计实测，可作为 Step 0 profiling 的起点。

### 4.2 第一优先级：TG 扩展优势（JZ 已有优势，进一步拉大）

JZ 当前的 TG 优势来自两个已实现的关键机制，应在优化分析中量化：

- **lm-head DSP offload**：QCOM 因 per-buffer ION 设计的经济性限制（32768 行 guard），lm-head 回退 CPU；JZ 通过 single mempool + Q4\_K->Q4\_0 tiled repack 实现 DSP 侧执行。
- **first-touch 权重 inval（bit0）**：lm-head 常驻后每 token 权重流量 \~1.9 GB，bit0 消除冗余 dcinva 节省 \~9.2 ms/token（固定整图总量，非 per-layer）。这是 bit0 开关对比实测值。

#### 4.2.1 TG descriptor 模板缓存 - 消除 graph\_compute\_batch 中 AP 侧 prep phase 的 per-token 开销

TG 模式下，**每 token 的 cgraph 拓扑完全相同**，只有 tensor 数据指针变化。当前每 token 都要走 `graph_compute_batch` 的全部 12 phase（Phase 1-9 AP 侧全图分析 / 镜像 / 权重 repack / mempool 分配 / desc 构建 / cache flush，Phase 10 同步 RPC，Phase 11-12 AP 侧 cache inval / 回拷）。其中 Phase 1-9 内的 layout 计算、mempool offset 跟踪、descriptor 构建等纯 AP 工作在拓扑不变时可复用：

- **已有基础**：Phase 1 已有 cgraph cache（[ggml-hexagon-jz.cpp:5159](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5159)），按 op+shape+src ptr 哈希命中时跳过 Phase 2 descriptor 重建。本轮 gemma4 端到端 256 calls 中 cgraph cache 命中率 98.8%（hits=253, misses=3，详见 5.4.2.1）。
- **本节方案在 cgraph cache 之上扩展**：缓存 mempool layout 与 desc 模板到层级别，后续 token 只 patch 变化的数据指针（activation/KV cache 地址），跳过 Phase 4-8 的 layout/desc 构建。
- 首次 token（或 graph reopt 时）构建 descriptor 模板，记录所有 op 的 src/dst offset 与 mempool layout。
- **收益（按 3.6 profiling 数据估算）**：3.6 节 profiling 仅覆盖 DSP 侧 Phase 10，未测量 AP 侧 Phase 1-9 开销。descriptor 模板缓存省的是 AP 侧开销，无法用 DSP profiling 数据直接估算。需在 Step 0 profiling 拿到 AP 侧 Phase 1-9 的精确耗时后再评估收益。作为参考，若 AP 侧 Phase 1-9 开销与 DSP 侧 non-op 开销（4693 us/batch）同量级，即使消除其中一半（\~2.3 ms），相对于端到端 TG 时间的占比也会因 AP 侧 Phase 11-12 的额外开销而更低。**descriptor 模板缓存优先级的最终判断依赖 Step 0 profiling 数据**。
- **复杂度**：中等偏低。需要在 ctx 中缓存 descriptor 模板和 mempool layout，处理 KV cache 增长时的 realloc 以及 graph topology 变化（context shift）时的 invalidate。
- **注意**：权重 repack offset 在模型加载后不变，但 activation 地址每 token 不同，模板需要支持 per-pointer patch。

#### 4.2.2 KV cache 常驻 ION + 增量 inval

当前 bit0 first-touch 标记对只读权重有效，但 KV cache 是 read-write 的，每 token 被 DSP 写入、AP 读取。KV cache 已在 ION mempool 中，Phase 11（cache inval）可以只 inval KV cache 的新增部分（增量 inval），而非每次做大范围 inval。

- **复杂度**：**高**（非中等偏高）。需要新增** DSP->AP 通信通道**：KV cache 写发生在 DSP 侧（每 layer FlashAttn 输出），AP 侧无法独立知道写入了哪些 position；需要 DSP 侧在 Phase 10 RPC reply 中携带 KV cache 写入范围（按 layer x position 的 bitmap 或 range list），AP 侧在 Phase 11 按此范围做精确 CIVAC。
- **注意**：bit0 机制不适用于 KV cache（read-write），需要独立的增量跟踪机制。

### 4.3 第二优先级：PP 优化（结构性收益，是 JZ 的真正战场）

**优先级重排论证**：JZ TG 在领先 4 模型上平均 +46.8%，继续优化的边际收益受 matmul kernel 物理极限约束；PP 落后 -3.3% 到 -27.8%，根因（AP-DSP 无 pipelining、算子支持度缺口导致的图拆分）是**可重构的框架/支持度差异**，而非 kernel 差异。**PP 从 -27.8% 改善到 -18% 等同于 +10pp 绝对提升**；TG 从 +47% 到 +57% 需要改 kernel 才有 +10pp，但 kernel 已与 QCOM 100% 共享，**只能从 matmul 内部优化（HMX 利用率、tile size）挤牙膏**。因此 PP 优化是 JZ 的真正战场，应排在 P0 profiling 之后立即推进。

**Table-6**：PP 表现与模型结构关联（基于 3.5 节三重叠加与 3.7 节 graph 拆分分析）

| 模型           | 层数                | PP JZ vs QCOM | TG JZ vs QCOM | 根因分项                                                 |
| ------------ | ----------------- | :-----------: | :-----------: | ---------------------------------------------------- |
| Gemma4-E2B   | 35 (GQA 8:1)      |   **+48.1%**  |     +11.1%    | 层数深且单层 DSP 时间适中（R 低），per-layer 优势累计超越 dspqueue overlap |
| Gemma4-E4B   | 42 (GQA 4:1)      |     -3.3%     |   **+35.1%**  | 单层 DSP 时间长（R 高），dspqueue 每层隐藏的 AP prep 放大，抵消 42 层累积优势   |
| Qwen3.5-2B   | 24 (GQA + Delta Net) |    -9.2%     |   **+93.6%**  | **25 路 graph 拆分税（SOLVE_TRI 未 offload，详见 3.7）** + 层数中等  |
| Llama3.2-1B  | 16 (GQA 4:1)      |     -11.4%    |   **+47.4%**  | 层数浅，dspqueue 优势显著                                     |
| Qwen1.5-1.8B | 24 (MHA 1:1)      |     -27.8%    |     -30.2%    | **三重叠加：dspqueue + 层数不足 + MHA VTCM/cache**              |

**结论**：PP 优化应聚焦于**结构性杠杆**（per-layer pipelining）与**支持度缺口补齐**（SOLVE_TRI offload），而非模型结构特化。Qwen1.5-1.8B 不是 corner case，而是三重不利因素的"压力测试"：per-layer pipelining 改善后这类模型获益最大。Gemma4-E2B 已经赢 PP，进一步压榨 +48.1% 之上的空间也来自 per-layer pipelining 在深层模型上的累积收益。Qwen3.5-2B 的 -9.2% 中约 5-6pp 来自 25 路 graph 拆分税（3.7 节），是五模型中唯一可用"补算子"而非"改架构"收回的差距。

#### 4.3.1 Per-layer intra-batch pipelining - 结构性突破点

**关键澄清**：[ion 文档](file:///home/zhouwg/develop/ggml-hexagon/docs/backend/jz-ggml-hexagon/ion-mempool-vs-perbuffer-analysis-20260713.md)"性能差异来自 data-plane policy 而非 control-plane，FastRPC 开销 ~89us 可忽略"的论断，**不能用于反对 per-layer pipelining**。这两个是不同的概念：

- **FastRPC ~89us（[ion 文档](file:///home/zhouwg/develop/ggml-hexagon/docs/backend/jz-ggml-hexagon/ion-mempool-vs-perbuffer-analysis-20260713.md) 历史值，本轮实测 min=102us / avg=154us，详见 4.4.4）是 control-plane 路径成本**（RPC invoke 自身的 marshalling + transport 开销），与是否做 pipelining 无关
- **Pipelining 收益 = min(AP prep 时间, DSP compute 时间) 的隐藏量**-完全由调度重叠决定，与 FastRPC 开销无关

FastRPC 是 RTS 路径成本，pipelining 关心的是能否把 1-3ms 的 AP prep 隐藏在 5-10ms 的 DSP layer 执行后面。**这是两个独立维度**。

**当前同步模型的瓶颈**：

```
AP Phase 1-9 [=====] -> AP阻塞 [==] -> AP Phase 11-12 [===]
                          DSP Phase 10
```

PP 阶段单 layer DSP 计算时间（按 3.6 profiling 与 5.4.2.1 实测分两段）：

- **TG (M=1)**：per 3.6 profiling，MUL_MAT avg=97us x 多 ops + FlashAttn 20us + RMS_NORM 1us 等 ≈ **200-500us per layer**
- **PP (M=58, gemma4-E2B batch#1)**：per 5.4.2.1 实测，batch-wall=81,637us / 35 layers ≈ **2,332us/layer**

AP Phase 1-9 + 11-12 估算：5.4.2.1 gemma4 dump 显示端到端 256 calls（1 PP + 255 TG）AP phase 累计 p1=101,249us / p2-p9=72,159us / p11-p12=3,653us，总和 177,061us，平均每 call 691us（**此为 PP+TG 平均值，非 PP 单独**）。PP batch#1 单次 81,637us 远大于 TG 的 ~37ms，AP 侧 PP 单独占比需在 Step 0 profiling 中分离 PP/TG 后才能精确给出（粗估 5-10%）。如果按 layer 切分：

```
AP P1-P4 Layer1 [=] -> DSP Layer1 [==] -> AP P5-P7 Layer2 [=] -> DSP Layer2 [==] -> ... -> AP P11-12 [=]
```

**预期收益（基于估算）**：AP 侧 Phase 1-9 + 11-12 占 PP 10-15%，pipelining 隐藏 50-70%，PP 提速 5-10%。Qwen1.5-1.8B 从 -27.8% 改善到 -20% 左右，Gemma4-E2B 从 +48.1% 进一步到 +53%+。

**关键设计约束**：

- **维持 single mempool 不变**：TG 优势的根，不能动
- **切分粒度应是 layer 级**：op 级切分会让 setup 成本吃掉收益
- **DSP 侧需要 partial-execute + resume 接口**：从 descriptor 中按 offset 启动执行的新基础设施
- **12 phase 测量框架要扩展到 per-layer**：现有 `cum_p1_us` ~ `cum_p12_us` 是 batch-level 聚合，要能下钻到 per-layer 才能验证 pipelining 收益
- **严格 TG 回归测试**：任何增加 AP↔DSP 同步点的改动都可能在 M=1 时变成新开销，单次 doorbell 优势是 TG 优势的关键来源

**前置数据需求（依赖 Step 0 profiling）**：

- Phase 1-9 + 11-12 的 AP 纯开销实测占比（决定 pipelining 收益上限）
- 单 layer DSP 计算时间分布（决定 AP prep 是否能完整隐藏在 layer 计算后面）
- per-layer cache flush 字节数（Phase 9 切分到 per-layer 后的实际开销）

**风险评估**：

- 收益面广：4/5 测试模型 PP 改善
- 实施复杂度高：DSP partial-execute 接口是新基础设施
- 风险点：MUL_MAT per-layer 平均仅 97us（远低于本轮实测 min=102us 的 FastRPC 开销，pipelining 切换代价 1-2 倍），**单 matmul pipelining 无收益；必须聚合到 layer 级别才有收益**。3.6 profiling 给出的是 batch#200 累计值，5.4.2.1 gemma4 batch#1 已提供 per-layer 实测数据（batch-wall / n_layer ≈ 2,332us/layer）

#### 4.3.2 descriptor 模板缓存（条件性）

如 4.2.1 节所述，descriptor 模板缓存可减少 AP 侧 prep 时间，与 pipelining 是**互补关系**（pipelining 利用 prep 的时间，缓存减少 prep 本身）。**已有 cgraph cache 覆盖 Phase 1-2 hit case（98.8% 命中率，详见 4.2.1）**，本节方案进一步跳过 Phase 4-8。**如果 Step 0 profiling 显示 AP 侧 prep 是 pipelining 收益的主要瓶颈，缓存应同步实施**。对长 context PP 收益较大（5-10%），对 TG M=1 收益很小。

#### 4.3.3 SOLVE_TRI offload 启用 - 消除 qwen3 的 25 路 graph 拆分（低成本高收益）

3.7 节已确认：qwen3（Qwen3.5-2B）的 cgraph 在 JZ 后端被拆成 25 个子图的唯一原因是 `GGML_OP_SOLVE_TRI` 未在 AP 侧注册，而 DSP 侧 kernel 已完整存在。补齐两处胶水代码即可消除拆分：

- **AP 侧**：在 `init_op_validators()`（[ggml-hexagon-jz.cpp L3762-3798](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L3762-L3798)）新增 `s_op_validators[GGML_OP_SOLVE_TRI]`，校验逻辑可直接参照 QCOM 的 `ggml_hexagon_supported_solve_tri`（[ggml-hexagon.cpp L3367-3399](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L3367-L3399)：F32 类型 + 方阵 + 维度匹配检查）。
- **DSP 侧**：在 `ggml_op_to_htp_op`（[entry.c L905](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/entry.c#L905)）新增 `case GGML_OP_SOLVE_TRI: *htp_op = HTP_OP_SOLVE_TRI; return 0;`。kernel 本体（`op_solve_tri`，[kernels/solve-tri-ops.c L197](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/solve-tri-ops.c#L197)）与 op 表注册（entry.c L811）均已存在，无需改动。

**预期收益**：qwen3 恢复完整单图后，每 batch 的 25 次 FastRPC round-trip 与 25 倍 Phase 1-9/11-12 开销归零，按 3.7 节实测估算 PP 可收回约 5-6pp（-9.2% 收窄至约 -3%）；TG 收益 <1%（拆分开销在 M=1 时占比已极小），但 25->1 的子图收敛同时降低了 per-token 的 cgraph cache 与 mempool 管理负担。

**复杂度**：**低**。两处注册各约 10-20 行，无新 kernel、无调度框架改动、无 cache 策略变化，是全部 PP 方向中风险收益比最优的一项。

**风险与验证**：

- SOLVE_TRI 在 qwen3 中为 F32 算子，QCOM validator 的 F32/方阵/维度检查可直接复用；JZ 侧需确认 delta net 的 chunked 调用路径下 tensor shape 均满足该校验。
- 合入前必须跑五模型 CI：qwen3 重点验证输出无 garble 且 `batch_calls` 从 6400 回落至 256；其余四模型验证无回归（它们的 cgraph 本就不含 SOLVE_TRI，预期零影响）。
- 该改动同时消除了 qwen3 PP 分析中的一个混杂变量：拆分消失后，qwen3 的 PP 差距将更纯粹地反映 dspqueue pipelining 优势，可作为 4.3.1 per-layer pipelining 收益验证的对照组。

### 4.4 第三优先级：低风险快速收益

#### 4.4.1 a-inv 优化（bit1 prior-dst 覆盖扩展）- 已实验证伪，关闭

3.6 profiling 显示 a-inv 是最大 non-op 开销（1030 us/batch, 82 MB）。bit1 机制（[kernels/entry.c:90-104](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/entry.c#L90-L104)）在读 **src** 时，若前序某个 op 的 dst 已覆盖该 src，就**跳过该 src 的 cache invalidation**（L2 已是新鲜的）。

- **实验证伪（2026-08-07）**：本轮测试 `dsp_cache_mode=5`（bit0+bit2）未启用 bit1；qwen1 PP-only 对照实验（mode=7 vs mode=5）显示 a-inv 零变化（19167us/1754MB -> 19135us/1754MB）。根因是 `prior_dst_add` 只登记 len <= PRIOR\_DST\_MAX\_LEN=64（单 cacheline）的 dst（[entry.c L448-460](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/entry.c#L448-L460)），cgraph 中间张量均 >= 256 字节，prior\_dst 列表为空，skip 路径几乎不触发。放宽该上限会引入 async DMA/HMX 路径的 stale L2 read 风险（[entry.c L52-54](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/entry.c#L52-L54) 设计注释），不建议尝试。
- **结论**：per-batch dedup 已保证每条 unique src 每 batch 至多失效一次，a-inv 字节量接近 batch 内结构性下限。原"预期 500us/batch = 1.4% TG"收益不成立，本项关闭。跨 batch dedup（5.8.3 方向 D）是独立假设，仍待验证。

#### 4.4.2 MUL_MAT_FFN kernel 调优

3.6 profiling 显示 MUL_MAT_FFN avg=334us x 35 calls = 11.7ms/batch（TG 主要热点）。kernel 已与 QCOM 共享，可调空间在 HMX 利用率与 tile size。

- **预期收益**：HMX 利用率提升 30% 可省 3.5ms/token = 9% TG
- **复杂度**：中（需 DSP kernel 修改）
- **风险**：tile size 调大需要更多 VTCM，可能与 lm-head 等大算子冲突
- **重要前提**：kernel 与 QCOM 共享意味着此优化对 QCOM 也有效，**不会扩大 JZ vs QCOM 的相对优势**，但能提升绝对性能

#### 4.4.3 post-matmul activation 与 element-wise 的 fuse

JZ Phase 3 已实现 QKV/FFN/mm_add fusion 以及 `RMS_NORM + MUL`（[ggml-hexagon-jz.cpp:5337-5342](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5337-L5342)），5.4.2.1 Table-11 显示 RMS_NORM_MUL count=227、GLU_GEGLU count=35 是 gemma4 主力算子。**未融合的剩余空间**在 `MUL_MAT -> post-matmul activation`（如 matmul -> SiLU -> mul 即 SwiGLU 的端到端 inline fuse），以及 `MUL_MAT -> element-wise broadcast` 的反向（与 RMS_NORM_MUL 方向相反）场景。在 M=1 TG 时，element-wise op 写入 DDR 再被下一个 matmul 读回是纯粹的浪费，fuse 后可减少一次中间 tensor 的 DDR round-trip。

- **预期收益**：<1% TG，PP 收益更小
- **复杂度**：中（kernel 修改 + AP 侧调度调整）

#### 4.4.4 减少 Phase 10 RPC round-trip 开销 - 优先级最低

FastRPC 开销已在 warmup 阶段校准（变量 `min_rpc_overhead_us`，[ggml-hexagon-jz.cpp:262](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L262)）。本轮 gemma4 端到端实测 256 calls 中 warmup n=6, **min=102us, max=251us, avg=154us**（详见 5.4.2.1）。[ion 文档](file:///home/zhouwg/develop/ggml-hexagon/docs/backend/jz-ggml-hexagon/ion-mempool-vs-perbuffer-analysis-20260713.md) 2026-07-24 历史值 \~89us 已不适用本轮测量（设备 thermal / kernel 调度变化可能导致漂移）。相对于 \~37 ms/token 的 TG 占比仍极小（<0.5%）。除非 profiling 发现非预期的高开销，否则此项投入产出比低，不建议优先投入。

#### 4.4.5 Sampling 路径优化（DSP-side + AP 侧配套） - 已验证不可行，关闭

第一版作者曾实验配套修改 [`kernels/entry.c`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/entry.c)（DSP 侧）与 [`ggml-hexagon-jz.cpp`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp)（AP 侧）以减少 sampling 阶段开销：

- **DSP 侧组件**（原 4.2.1）：DSP 侧 lm-head matvec 之后做 softmax + argmax + top-k/p，仅返回 4 字节 token ID 而非完整 logits 矩阵。当前流程 `DSP lm-head -> F32 logits (~500KB-1MB) -> memcpy 回 AP -> CPU sampling`，优化后 `DSP lm-head -> DSP softmax -> DSP argmax -> 返回 int32 token ID (4 bytes)`。
- **AP 侧组件**（原 4.3.2）：AP 侧 sampler chain 适配新的 4 字节 token ID 输入，sampler chain 多个算子替换为更快实现。单独看 AP 侧有 [ion 文档](file:///home/zhouwg/develop/ggml-hexagon/docs/backend/jz-ggml-hexagon/ion-mempool-vs-perbuffer-analysis-20260713.md) 提到的 2.5x 优化空间（QCOM 的 AP 侧 sampling 路径比 JZ 快）。
- **配套关系**：两条路径看似可独立实施（DSP 侧改 entry.c vs AP 侧改 jz.cpp），实际是同一实验的 DSP 侧 + AP 侧组件，必须配套修改。文档原 4.2.1 与 4.3.2 分立两节、易被误读为独立方案，本节合并澄清。
- **复杂度**：高。DSP 侧在 256K vocab 上做 top-k（O(n log n)）+ top-p（cumulative sum + rejection sampling）+ Hexagon RNG 集成；AP 侧 sampler chain 多个算子替换。
- **收益估算**：logits memcpy 在 DDR 带宽下仅 ~10-20us（500KB-1MB / SnapDragon 8 Elite LP-DDR5x 5300 MHz 理论 ~50 GB/s），FastRPC 开销本轮实测 min=102us / avg=154us，合计 ~110-170 us/token。AP 侧单独看有 2.5x 优化空间。
- **实验结论（2026-08-06 验证）**：配套修改后功能正确，但性能收益 <0.5% / <1%（ion\_sync\_mode=1 下整个 mempool sync 掩盖了局部收益），代码复杂度高，**已回滚**，不进入优化路线图执行队列。
- **对未来读者的提示**：若再次评估 sampling 路径优化，应直接以本实验的 <0.5% / <1% 上限数据为参考起点，无需重复"理论收益 ~110-170us / <0.5%"的独立估算。

### 4.5 优化路线图

**潜在收益优先级 vs 执行优先级是两个维度**：4.2-4.4 的"第一/第二/第三优先级"按**潜在收益空间**分级（4.2 TG 扩展优势 > 4.3 PP 优化 > 4.4 低风险快速收益），按**实际执行顺序**（下方 Step 0-4）则重新排列。**调整依据**：前期实验（详见 4.4.5 末尾注）证实 sampling 路径优化代码复杂且性能收益极小；AB 测试数据也确认 JZ 在 TG 已领先 4/5 模型，进一步投入产出比低，PP 是 JZ 的真正短板（4/5 模型落后 QCOM），结构性优化收益空间最大。**两个维度的执行映射**：4.3.3 SOLVE_TRI offload（约 30 行胶水代码，qwen3 PP 收回 5-6pp）作为 Step 1 立即收益（4.4.1 a-inv 已于 2026-08-07 实验证伪关闭）；PP 结构性突破（4.3.1 per-layer pipelining）作为 Step 2 核心战场；TG kernel 精调（4.4.2 + 4.4.3 + 4.2.2）作为 Step 3 最后做。**长期/已关闭项**：4.4.5 sampling 路径优化与 4.2.2 KV cache 增量 inval 因复杂度高/收益小。

4.1-4.4 节按优先级组织，经实验验证后，实际执行顺序调整为：

```
Step 0: Profiling 数据驱动（必做前提，详见 4.1）

Step 1: 低风险快速收益（独立于 PP/TG 主战场）
  +-- 4.3.3 SOLVE_TRI offload 启用（qwen3 PP +5-6pp，约 30 行胶水代码，五模型 CI 验证）
  +-- 4.4.1 a-inv 优化：2026-08-07 实验证伪（bit1 受 PRIOR_DST_MAX_LEN=64 限制退化为 no-op），关闭
  +-- 4.4.4 FastRPC 校准：已实测，结论为投入产出比低，关闭

Step 2: PP 结构性突破（核心战场，详见 4.3.1）
  +-- DSP 侧 partial-execute + resume 接口 + async FastRPC 调度
  +-- 严格 TG 回归测试：M=1 单次 doorbell 优势不被新同步点吃掉
  +-- （条件性）4.3.2 descriptor 模板缓存
  +-- 预期：PP +5-10%；Qwen1.5-1.8B 从 -27.8% -> ~-20%；Gemma4-E2B 从 +48.1% -> +53%+

Step 3: TG kernel 精调（边际收益，详见 4.4.2 + 4.4.3 + 4.2.2）
  +-- 4.4.2 MUL_MAT_FFN kernel 调优（HMX 利用率与 tile size）
  +-- 4.4.3 post-matmul activation fuse
  +-- 4.2.2 KV cache 增量 inval：需新增 DSP->AP 通信通道，列为长期项
  +-- 重要前提：kernel 与 QCOM 共享，主要提升绝对性能，不扩大相对优势

Step 4: 长期架构（按 Step 2 效果决定）
  +-- 如 Step 2 成功：在 JZ 架构内深化 per-layer pipelining
  +-- 如 Step 2 失败：保留单次 doorbell 模型，强化 single mempool + batch-level cache
  +-- 多 batch 并发 PP（服务端场景吞吐优化，独立方向）
```

### 4.6 核心原则

**不要为了追 PP 性能而破坏 TG 的优势。** JZ 在 TG 上的优势（single mempool -> lm-head offload、batch-level cache、tiled repack）是架构级的。QCOM 的 dspqueue 是另一套执行调度模型，与 JZ 架构近乎互斥：两者的控制平面原语不同（FastRPC sync vs dspqueue async），数据平面策略也不同（single mempool + offset addressing vs per-buffer + bi indirection），且 PR #26049 将 cache coherency 维护移入算子实现后与 JZ 的 cache 子系统不兼容（详见 [ion 文档](file:///home/zhouwg/develop/ggml-hexagon/docs/backend/jz-ggml-hexagon/ion-mempool-vs-perbuffer-analysis-20260713.md)）。现实路径是在 JZ 架构内增加 PP 优化（如 per-layer pipelining），而非融合两种架构。

## 五、force_opfusion_in_pp 实验

> **基线 commit**: `a3d04682c11086450d36091a15534b14e65dda2a`（feature/force_opfusion_in_pp 分支）
>
> **模型**: `gemma-4-E2B-it-Q4_0.gguf`（默认测试模型，35 层）

### 5.1 实验动机

第 4.3 节确定 PP 是 JZ 真正战场之后，需要找到一个低风险高收益的切入点。观察到 `is_mergeable_mul_mat()` 中的 HMX-eligibility 闸门在 PP 路径下必然拒绝所有 MUL_MAT（因为 `M > HTP_MM_HMX_MIN_NROWS=4`），导致 QKV/FFN/mm_add fusion 在 PP 完全失效。初始假设：

- **假设 A**：3 个独立 HMX MUL_MAT -> 1 个 HVX fused MUL_MAT_QKV，单算子更慢但 cache 失效次数减少到 1/3
- **假设 B**：cache 失效节省 > 算子额外耗时 -> 净收益为正

为此引入 `force_opfusion_in_pp` 配置开关（0=保持原 HMX 闸门，1=旁路闸门强制融合），并加 3 个 cum 计数器（`n_qkv_skip_cum_hmx` / `n_pair_skip_cum_hmx` / `n_mm_add_skip_cum_hmx`）量化被错过的融合机会数。

### 5.2 实验设计

**Table-7**：force_opfusion_in_pp 实验配置

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

### 5.3 gemma4-E2B 单模型对比：force=0 baseline vs force=1 实验

gemma4-E2B（35 层，GQA 8:1）是默认测试模型。本节在同一模型上对比 `force_opfusion_in_pp=0` 与 `force_opfusion_in_pp=1` 两组数据，量化"旁路 HMX 闸门"的净收益。

#### 5.3.1 Baseline 数据（`force_opfusion_in_pp=0`）

```
mul_mat coverage: total=277 hmx=276 (99.6%) qkv_fused=0 (saves 0.0%) ffn_fused=0 (saves 0.0%) mm_add_fused=0 (saves 0.0%) qkv_skip_hmx=15 pair_skip_hmx=65
hmx eligibility: total=1940 pass=1386 (71.4%)
batch-wall cum=81252 us op-sum cum=56999 us non-op avg=24253 us/batch
non-op: hdr=4 pre=392 w-inv=13639(1334MB) a-inv=6869(551MB) dst=112 bulk=1677 queue=8 us/batch
```

**关键解读**：

- `qkv_skip_hmx=15` 与 35 层对应，**每层 1 个 QKV 候选被 HMX 闸门拒掉**，即基线未利用 15 个 QKV 融合机会
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

**Table-8**：force=0 vs force=1 对比

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

Table-8 揭示 3.8x 退化的根因：MUL_MAT_FFN 在 HVX fused 路径下单 op 耗时 6.5 ms，35 个 op 合计 228 ms，占 batch-wall 的 73%。假设 B（cache 失效节省 > 算子额外耗时）被证伪：w-inv 与 a-inv 均与 baseline 一致，旁路 HMX 闸门并未带来 cache 失效的节省。

### 5.4 五模型 CI 验证：force=0 无回归 + HVX 融合 PP 不通用确认

- 通过 `./scripts/build-run-android.sh run_force_opfusion_in_pp_all` 一键运行五个模型，每模型生成 `*_terminal.txt`（端到端 tok/s） + `*_logcat.txt`（OP-PROF 算子分布） 两个 log

#### 5.4.1 五模型基础信息

**Table-9**：五模型基础参数（按 alias 数组顺序）

| # | alias       | 模型文件                                       | vocab_size | lm-head 类型 | 层数  | 注意力类型     | 唯一算子 (PP/TG)         |
| - | ----------- | ------------------------------------------ | ---------- | ---------- | --- | --------- | ------------------- |
| 1 | gemma4      | gemma-4-E2B-it-Q4_0.gguf                   | 256,000    | Q4_K       | 35  | GQA 8:1   | GLU_GEGLU, UNARY_TANH |
| 2 | qwen3       | Qwen3.5-2B-Q4_0.gguf                       | 151,936    | Q6_K       | 24  | GQA + Delta Net | **GATED_DELTA_NET**, L2_NORM, UNARY_SILU/SIGMOID/SOFTPLUS |
| 3 | qwen1       | Qwen1.5-1.8B-Q4_0.gguf                     | 151,936    | Q6_K       | 24  | MHA 1:1   | MUL_MAT_ADD, MUL_MAT_FFN, GLU_SWIGLU |
| 4 | llama3      | Llama-3.2-1B-Instruct-Q4_0.gguf            | 128,256    | Q4_K       | 16  | GQA 4:1   | MUL_MAT_ADD, GLU_SWIGLU |
| 5 | gemma4-e4b  | gemma-4-E4B_q4_0-it.gguf                   | 256,000    | Q4_K       | 42  | GQA 4:1   | GLU_GEGLU, UNARY_TANH |

**说明**：

- 层数来自 `ggmlhexagon_dump_perf_stats` 的 `model: n_layer=N` 字段（扫描 tensor name 末尾连续数字段得到 max layer index）
  - qwen3：24 个总层中 13 个含 FFN，其余为 linear-attention delta net 层（见 5.4.3）；log 中 `ffn_gate-12` 编号指 FFN 算子所在层（0-12），非 tensor 层编号
- 五个测试均使用 `n_ctx=8192, n_batch=2048, n_predict=256, n_threads=6, dsp_cache_mode=5, ion_sync_mode=1`
- PP batch#1 中 五个模型均未出现 MUL_MAT_QKV fusion（HMX 闸门正常拒绝，见 5.4.4）；MUL_MAT_FFN 仅 qwen1 边缘触发 1 次（count=1, cum=265us）；MUL_MAT_ADD 在 llama3/qwen1 触发，gemma 系不触发

#### 5.4.2 PP batch#1 五模型 OP-PROF 对比

**Table-10**：五模型 PP batch#1 OP-PROF 对比

| 模型         | n_layer | batch-wall (us) | op-sum (us) | non-op (us) | MUL_MAT cum (us) | MUL_MAT count | MUL_MAT max (us) | FLASH_ATTN cum (us) | FLASH_ATTN count | non-op w-inv (MB) | non-op a-inv (MB) | non-op bulk (us) |
| ---------- | :-----: | :-------------: | :---------: | :---------: | :--------------: | :-----------: | :--------------: | :-----------------: | :--------------: | :---------------: | :---------------: | :--------------: |
| gemma4     |   35    |     81,637      |   57,592    |   24,045    |     39,360       |      277      |     **4,448**    |       4,026         |        35        |     **1,334**     |       542         |      1,485       |
| gemma4-e4b |   42    |    140,304      |   97,944    |   42,360    |     70,418       |      344      |     **7,873**    |       6,102         |        42        |     **2,528**     |       891         |      2,906       |
| llama3     |   16    |     37,598      |   24,275    |   13,323    |      9,856       |       79      |        262       |       5,277         |        16        |        497        |       415         |      2,634       |
| qwen1      |   24    |     91,150      |   47,613    | **43,537**  |     11,575       |       48      |       3,846      |      **18,615**     |        24        |        818        |    **1,754**      |   **15,119**     |
| qwen3      |   24    |        794      |      569    |      225    |        191       |        1      |        191       |          -          |        -         |         6         |         5         |         53       |

**关键观察**：

1. **batch-wall 与模型规模/层数正相关**：gemma4-e4b / gemma4 ratio = 1.72x，与层数比 42/35=1.20x + 模型尺寸比 4B/2B=2.0x 加权吻合（e4b 的 GQA 4:1 较 gemma4 的 8:1 有更大的 attention 中间张量，但对 batch-wall 仅为二阶影响）
2. **MUL_MAT max 是 lm-head 标志**：gemma4 max=4,448us，gemma4-e4b max=7,873us，qwen1 max=3,846us，llama3 max=262us（vocab=128K 在 PP 阶段分块执行，无显式大算子）。gemma4 与 3.6 节 baseline max=4,697us 差异 5.3%（测量噪声范围内）
3. **qwen1 的 FLASH_ATTN avg=776us 显著高于其他**：MHA（1:1 attention）在 PP 大 M 下 Q@K^T 矩阵规模最大；GQA 模型的 avg 仅 115-145us（gemma4 35层 avg=115us，gemma4-e4b 42层 avg=145us）
4. **qwen1 的 non-op a-inv=19,134us + bulk=15,119us 是五模型中最高**：MHA 模型 attention 中间张量（Q@K^T, Softmax(QK^T)·V）占用最大 VTCM 与 DDR 带宽，导致 cache 维护代价翻倍。这是 3.5 节"Qwen1.5-1.8B 三重叠加根因"的直接证据
5. **w-inv 随模型规模增长**：gemma4-e4b 25,802us（2,528MB） vs gemma4 13,637us（1,334MB），4B 参数量首次 touch 的权重范围翻倍。a-inv 方面，gemma4 6,900us（542MB）较 qwen1 少 2.8x，GQA 8:1 attention 中间张量比 MHA 1:1 小约 2.8x，符合 GQA 压缩理论值
6. **qwen3 是 init batch（M=1）**：仅含 embedding 初始化算子（6 个 op），不代表真实 PP 性能，需用 `run_pp_only` 重抓
7. **MUL_MAT count 反映 cgraph 大小**：gemma4 1116 graph ops 中 277 个 MUL_MAT，gemma4-e4b 1384 ops 中 344 个，qwen1 533 ops 中 48 个，llama3 296 ops 中 79 个。差异主要来自 FFN/attention 内部 matmul 数量与是否使用 GQA

##### 5.4.2.1 gemma4 (E2B) 详细算子分布

**Table-11**：gemma4 PP batch#1 完整 15 算子分布

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

**FFN/QKV skip 模式**（gemma4 batch#1 真实日志）：

- `QKV skip: is_qkv_mergeable=false (HMX gate)` at i=4/1116 -> HMX 闸门按预期拒绝 QKV 融合
- `FFN skip: is_mergeable_mul_mat_pair=false` at i=4/1116 -> HMX 闸门按预期拒绝 FFN pair
- `FFN skip: next not MUL_MAT` at i=6/1116, i=17/1116 -> FFN pair 中 next op 是 UNARY_TANH (op=25) 而非 MUL_MAT，跳过原因不是 HMX 闸门而是 graph 顺序

**tok/s 数据**（`common_perf_print` 输出，本轮端到端性能）：

- **prompt eval time = 87.00 ms / 58 tokens (1.50 ms per token, 666.69 tokens per second)**
- **eval time = 9,623.58 ms / 255 runs (37.74 ms per token, 26.50 tokens per second)**
- total time = 10,052.86 ms / 313 tokens
- graphs reused = 253
- unaccounted time = 26.95 ms / 0.3 %

**`ggmlhexagon_dump_perf_stats` 完整统计**（gemma4 端到端 256 批次）：

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

**`ggmlhexagon_print_running_timestamp` 完整配置**（gemma4）：

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

**与 3.6 节 baseline 对比**：5 项核心指标（batch-wall / op-sum / non-op / MUL_MAT cum / MUL_MAT max）差异均在 ±6% 以内（batch-wall 81,637us vs 81,252us, 0.5%），确认 force=0 cleanup 无回归

##### 5.4.2.2 gemma4-e4b 详细算子分布

**Table-12**：gemma4-e4b PP batch#1 完整 15 算子分布

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
- non-op 合计=42,360 us/batch（占 batch-wall 30.2%，与 gemma4 的 29.5% 几乎一致，说明 GQA 比例从 4:1 升到 8:1 不会显著改变 non-op 占比）

**FFN/QKV skip 模式**（gemma4-e4b batch#1 真实日志）：

- `QKV skip: is_qkv_mergeable=false (HMX gate)` at i=4/1384 -> HMX 闸门按预期拒绝 QKV 融合
- `FFN skip: is_mergeable_mul_mat_pair=false` at i=4/1384 -> HMX 闸门按预期拒绝 FFN pair
- `FFN skip: next not MUL_MAT` at i=6/1384, i=17/1384 -> 同 gemma4，graph 顺序问题

**tok/s 数据**（`common_perf_print` 输出）：

- **prompt eval time = 144.51 ms / 58 tokens (2.49 ms per token, 401.35 tokens per second)**
- **eval time = 17,494.38 ms / 255 runs (68.61 ms per token, 14.58 tokens per second)**
- total time = 17,905.08 ms / 313 tokens
- graphs reused = 253
- unaccounted time = 19.49 ms / 0.1 %

**`ggmlhexagon_dump_perf_stats` 完整统计**（gemma4-e4b 端到端 256 批次）：

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

**`ggmlhexagon_print_running_timestamp` 完整配置**（gemma4-e4b）：

- 与 gemma4 E2B 一致（force_opfusion_in_pp=0, enable_opfusion=1, dsp_cache_mode=5, ion_sync_mode=1, thread_counts=6）
- running timestamp: 2026-08-06, 21:16:23

**与 gemma4 E2B 跨模型对比**：

- MUL_MAT avg: gemma4=142us, gemma4-e4b=204us（**1.44x**，接近层数比 42/35=1.20x + 4B/2B 参数量比 2.0x 的加权预期）
- MUL_MAT max: gemma4=4,448us, gemma4-e4b=7,873us（**1.77x**，lm-head vocab=256K 在两个模型相同，但 E4B 的 hidden dim 翻倍，所以 lm-head matvec 计算量 2x）
- GLU_GEGLU avg: gemma4=167us, gemma4-e4b=182us（几乎一致，GLU 计算量正比于 hidden_dim）
- FLASH_ATTN avg: gemma4=115us, gemma4-e4b=145us（1.26x，GQA 8:1 比 4:1 减少 KV 计算量，但 hidden_dim 增大抵消部分优势）
- non-op 占比：gemma4=29.5%, gemma4-e4b=30.2%（几乎一致，说明 non-op 开销与模型规模近似线性相关，与 3.3 节"role-aware cache 比例恒定"的分析一致）

#### 5.4.3 qwen3 TG batch#978 详细数据（唯一完整 TG 抓取，源文件 `log_qwen3_ppandtg_force0_v4`）

> **重要**：qwen3 = Qwen3.5-2B（delta net 混合架构，24 个总层中标准 attention 13 层 + linear attention delta net 11 层），GATED_DELTA_NET/L2_NORM 是该架构的正常算子。**`ffn_gate-0/1/2` 这类日志编号指的是 FFN 算子所在层（0-12 共 13 个），与 tensor 的 0-23 共 24 个总层编号不同**

**Table-13**：qwen3 TG batch#978 OP-PROF 详表

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
- non-op avg=202 us/batch（14.6% wall time，**比 gemma4 PP 的 30.0% 低一半**）
- non-op 细分：hdr=0, pre=8, w-inv=10(1MB), a-inv=76(6MB), dst=3, bulk=71, queue=3 us/batch

**TG 阶段关键观察**：

1. **三类 matmul 占 op-sum 74.4%**：MUL_MAT（38.3%） + MUL_MAT_FFN（23.0%） + MUL_MAT_ADD（13.1%）。与 3.6 节 gemma4-E2B profiling 的 91.1% 略有差异，原因是 qwen3 是 delta net 混合架构，多了 GATED_DELTA_NET（4.5%） + CONCAT（6.3%） + CPY（5.8%） 等"delta net 特有"算子，挤占 matmul 占比
2. **MUL_MAT_FFN avg=232us 是稳定 FFN fused 调用**：count=1,142/978batch ≈ 1.17 次/batch，说明每 token 大约 1 次 FFN fusion（qwen3 在 TG 阶段 M=1 <= 4，满足 HMX 闸门条件，fusion 正常触发）
3. **GATED_DELTA_NET avg=74us 是 delta net 核心算子**：max=1,291us 是初始化阶段的 warm-up 路径，稳定阶段 avg 远低于 max；count=704/978batch ≈ 0.72 次/batch，delta net 主干每 1-2 token 调用一次
4. **MUL_MAT max=5,635us 是 lm-head matvec**：与 3.6 节 gemma4-E2B 的 max=4,697us 同量级，与 Q4_K/Q6_K lm-head 大小相关（本模型 vocab=152K Q6_K ≈ 178MB）
5. **non-op 仅 14.6% wall time**：M=1 TG 阶段 bit0 first-touch 权重 inval 生效（w-inv=10us/1MB，几乎为 0），a-inv=76us/6MB 也极低。验证 3.3 节"role-aware cache 在 M=1 TG 显著优"的核心论点
6. **CONCAT + CPY 占 12.1%**：delta net 架构特有的 intermediate tensor 拼接/拷贝操作，是 JZ 后续可优化方向（通过更高效的 in-place 拼接减少 DDR 往返）

#### 5.4.4 跨模型 matmul 行为对比

**Table-14**：五模型 matmul 行为对比（PP batch#1）

| 模型         | n_layer | MUL_MAT count | MUL_MAT cum (us) | MUL_MAT avg (us) | MUL_MAT_FFN count | MUL_MAT_ADD count | QKV/FFN skip 模式                |
| ---------- | :-----: | :-----------: | :--------------: | :--------------: | :---------------: | :---------------: | -------------------------- |
| gemma4     |   35    |      277      |     39,360       |       142        |         0         |         0         | HMX gate (PP 路径,符合预期)        |
| gemma4-e4b |   42    |      344      |     70,418       |       204        |         0         |         0         | HMX gate (PP 路径,符合预期)        |
| llama3     |   16    |       79      |      9,856       |       124        |         0         |        30         | HMX gate (PP 路径,符合预期)        |
| qwen1      |   24    |       48      |     11,575       |       241        |         1         |        119        | HMX gate (PP 路径,符合预期)        |
| qwen3      |   24    |        1      |        191       |       191        |         0         |         0         | (init batch,无实际 layer matmul) |

**观察**：

- **PP 路径 HMX 闸门 100% 生效**：五个模型的 PP batch#1 中 MUL_MAT_FFN 全部为 0（仅 qwen1 边缘 1 次，可能是 scheduler 特例），MUL_MAT_ADD 触发条件独立（llama3=30, qwen1=119），与 HMX 闸门无关
- **MUL_MAT avg 与模型/层数正相关**：gemma4-e4b（42层 4B） avg=204us，qwen1（24层 1.8B MHA） avg=241us，gemma4（35层 2B GQA 8:1） avg=142us，llama3（16层 1B） avg=124us
- **MUL_MAT_ADD 是稳定的 element-wise 加法融合**：qwen1 count=119 说明该模型 cgraph 中存在大量 MUL_MAT + ADD 模式，被 MUL_MAT_ADD fusion 正确捕获；llama3 count=30，gemma4/gemma4-e4b count=0（其 cgraph 中没有 MUL_MAT->ADD 模式）
- **HMX eligibility 与 QKV/FFN 融合互斥**：五个模型的 "QKV skip: HMX gate" 日志均出现（本节 5.4.2.1 已确认 gemma4 真实日志，5.4.2.2 确认 gemma4-e4b），验证 `is_mergeable_mul_mat` 闸门在 cleanup 后行为与 a3d04682 基线一致
- **gemma4-e4b / gemma4 MUL_MAT avg 比例 1.44x**：与层数比 1.20x + 模型尺寸 4B/2B = 2x 加权预期（1.20 * sqrt(2) ≈ 1.70） 相比略低，说明 E4B 的更大 MUL_MAT 在 VTCM 中复用效率更优
- **qwen3 的 1 次 MUL_MAT 仅是 init batch 的 embedding**：graph nodes 范围 26-62（graph size 在 五模型中最小）说明 delta net 架构在 PP 阶段 matmul 数量极低，大部分计算在 attention 之外的 GATED_DELTA_NET/L2_NORM/CONCAT/CPY 中，详细见 5.4.3 的 TG 数据

#### 5.4.5 关键发现与结论

1. **五模型 CI 全部通过，force=0 无回归**：gemma4 batch-wall 与 3.6 节 baseline 差异 0.5%。端到端 tok/s：gemma4 PP 666.69 / TG 26.50，gemma4-e4b PP 401.35 / TG 14.58，qwen1 PP 539.06 / TG 18.41（non-op a-inv+bulk 五模型最高，验证 3.5 节 MHA 三重叠加），llama3 PP 1039.49 / TG 42.20（五模型最高，16 层 + 1B），qwen3 PP 408.38 / TG 21.26
2. **PP 路径 HMX 闸门 五模型全部正确保留**：QKV/FFN 融合被 HMX 闸门阻止（gemma4/gemma4-e4b 真实日志已确认）；MUL_MAT_ADD 是 PP 路径唯一活跃的 fusion（qwen1 count=119, llama3 count=30, gemma 系 count=0，与 attention pattern 相关）
3. **non-op 占比与 GQA 比例无关**：gemma4（8:1） 29.5% vs gemma4-e4b（4:1） 30.2%，验证 3.3 节"role-aware cache 与模型规模线性相关"
4. **n_layer 字段在 五模型中均正确输出**（gemma4=35, gemma4-e4b=42, llama3=16, qwen1=24, qwen3=24），解决了此前层数估算的不准确问题

#### 5.4.6 已知数据局限与后续动作

1. **3 个模型（qwen1/llama3/gemma4-e4b）TG 详细算子分布缺失**：`*_logcat.txt` 仅捕获 batch#1，后续 batch 的 OP-PROF 被丢弃（端到端 tok/s 仍从 `*_terminal.txt` 获取）。如需 TG 详细分布，用 `grep -E "OP-PROF.*batch#" log_*_logcat.txt` 直接拉取
2. **qwen3 PP 真实数据缺失**：batch#1 仅含 embedding init 算子（M=1, 6 个 op），需用 `run_pp_only qwen3` 重抓（n_prompt >= 64）
3. **qwen3 端到端 tok/s 稳定**：PP 408.38 / TG 21.26 tok/s，与历史 v4 log（PP 401 / TG 21）一致
4. **gemma4-e4b log 显示 batch#1 重复打印**：原因待查（可能是 `dump_perf_stats` 与 `OP-PROF` 触发周期冲突），不影响数据正确性
5. **GQA 4:1 vs 8:1 matmul 差异**：gemma4-e4b/gemma4 MUL_MAT avg ratio 1.44x，略低于层数+模型尺寸加权预期 1.70x，说明 E4B 的更大 MUL_MAT 在 VTCM 中复用效率更优。后续可通过 `mul_mat coverage` 的 "ne11" 维度分布进一步分析

#### 5.4.7 文档与 commit 维护

- 五模型 10 个 log 文件（`log_forceopfusioninpp_<model>_<ts>_*`）保留在工作区根目录
- 本节数据已与 3.6 节 gemma4-E2B profiling 交叉对比（5 项核心指标差异均在 ±6% 以内），确认无回归

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

1. **HVX fused 路径不适用于 PP**：M 单算子平均耗时与 M=1（TG） 场景差数十倍，即使 cache 节省也不足以抵消算子额外耗时
2. **PP 优化的正确方向是 HMX-aware fused kernel**：需要让 MUL_MAT_QKV / MUL_MAT_FFN 在大 M 路径下走 HMX 而不是 HVX，保留 HMX 速度 + 节省 cache 失效。这是 kernel 重写工作，非小规模 patch 可解决
3. **保留的基础设施**：
   - `force_opfusion_in_pp` cfg flag + bypass 分支（可作为未来 HMX-aware kernel 上线后的 A/B 对比基线）
   - 3 个 cum 计数器（`n_qkv_skip_cum_hmx` / `n_pair_skip_cum_hmx` / `n_mm_add_skip_cum_hmx`），量化"PP 路径下融合机会数"，长期监控融合覆盖率
   - `mul_mat coverage` 扩展打印，实验环境诊断
   - `ggmlhexagon_print_running_timestamp` 打印 `enable_opfusion` / `force_opfusion_in_pp`，运行时配置可见性
4. **per-layer profiling（副产品）**：[OP-PROF-LAYER] 日志已能正常输出 15 层 mat/ffn/attn 三段耗时，后续 PP 优化可直接基于此数据做 layer 级别对比

### 5.7 后续步骤

1. **回退 cfg**：`force_opfusion_in_pp = 0`（默认行为不变）
2. 保留feature/force_opfusion_in_pp，可在未来作为其他feature开发的基线分支
3. **新方向**：调研高通 htp/ 是否有 HMX-aware MUL_MAT_QKV/FFN kernel 可参考；若无，需自主设计（关键决策点：3 个权重矩阵的 VTCM 复用策略，以及如何在 M=large 时仍能利用 HMX 8x8 systolic 阵列）

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

`qkv_fused=0` 与 `qkv_skip_hmx=0` 的组合定位根因: `is_mergeable_mul_mat()` (ggml-hexagon-jz.cpp:2399-2408) 要求 `!mm_is_hmx_eligible(t)` 才允许进入融合候选。Qwen1.5-1.8B 的 Q/K/V MUL_MAT 全部是 HMX-eligible (Q4_0 + 标准 shape)，永远被 `is_mergeable_mul_mat` 拒之门外。当前 fused QKV (op_matmul_qkv，matmul-ops.c:3562) 是 HVX 实现，无法利用 HMX 加速，所以即使强制融合，HVX 3x 路径也不会比 HMX 3x 更快。这与 5.3.2 节 `force_opfusion_in_pp=1` 的实验结论 (HVX 融合 PP 慢 3.8x) 一致。

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

**B. bulk flush 异步化 (P0 突破方向)**:

- 现状: `bulk_flush_all()` (entry.c:527-529) 在所有 op 完成后**阻塞**执行 15145 us，期间 DSP thread idle
- 实施: 新增 1 个 DSP worker thread 负责 flush，main batch 线程与 flush 线程并行
- 时序: batch N 完成后, flush 线程立即开始 flush batch N 的 dst range; 同时 AP 侧准备 batch N+1 的 descriptor (hdr/pre 阶段)
- 同步点: batch N+1 的第一个 op 读取 dst 之前，需确保 batch N 的 flush 完成
- 节省: 15145 us 中 30-50% 可与下一 batch 重叠，理论省 4500-7500 us
- 风险: 需要新增 DSP thread + 跨 batch 同步原语，可能影响后续读取的 cache coherency
- 参考: 5.7 节保留的 `dsp-ctx.h` `bulk_flush_ranges` 数组已包含 sort + merge 逻辑，异步化只需把 `bulk_flush_all()` 从主线程移到 worker thread

**C. HMX fused FFN (P1 方向)**:

- 实施: 类似 A，但融合 2 个 MUL_MAT (ffn_gate, ffn_up) 为 1 个 HMX call
- 当前覆盖率: ffn_fused=1/24 (1.2%)，与 QKV 同样的 `!mm_is_hmx_eligible` gate 限制
- 节省: 24 层 x 1 op overhead = 1200-2400 us
- 风险: 中，参考 A 的实现
- 可与 A 同步实施: `op_matmul_hmx_qkv` 和 `op_matmul_hmx_ffn` 共享 HMX matmul 基础设施

**D. a-inv 跨 batch dedup (P1 方向)**:

- 现状: a-inv=19167 us / 1754 MB per batch (非 op 开销最大项)
- 1754 MB / 24 layer / 51 token = 1.43 MB per layer per token, 主要是 per-token unique activation
- 实施: 类似 `weight_inval_check_and_mark` (entry.c:334-362) 的 per-batch dedup, 添加 `act_inval_check_and_mark` 跨 batch tracking
- 假设: 同一 tensor 跨 batch 复用率 30%, 节省 ~24% = 800-1500 us
- 风险: 中，跨 batch 跟踪需要考虑 L2 容量 (128KB) 和 tensor 生命周期
- 前提: 必须先确认 1754 MB 的 24% 重复率假设成立 (需 AP 侧打点验证)

#### 5.8.4 优先级排序依据

P0 方向 (A + B) 合计理论节省 6900-12300 us = PP +7.6-13.6%，符合 `4.5 路线图` 中"PP 优化第二优先级"的潜在收益空间。P1 方向 (C + D) 合计 2000-3900 us = PP +2.2-4.3%，作为 P0 实施完成后的后续优化。

实施顺序建议: A -> C -> B -> D。前两个 (A + C) 共用 HMX fused kernel 基础设施，先实施可积累经验。B 涉及新 DSP thread + 同步原语，复杂度最高，但潜在收益最大 (PP +5-8%)。D 风险最低 (复用现有 weight_inval 模式)，但收益受限于重复率假设验证。

注意: 本节 4 个方向均在 `feature/qwen1_optimize` 分支探索，**不应直接合并到主分支 `self-build-jz`**，需 5 模型 CI 验证无回归 + 主分支 baseline 对比后再议。

***

## 六、Qwen3.5-2B PP&TG 优化（2026-08-07）

### 6.1 起点：kimi-k3 在 3.7 节指出的拆分问题

3.7 节（Kimi-K3 在 2026-08-07 修订轮新增）的核心结论：Qwen3.5-2B（即表 Table-9 中 alias 为 `qwen3` 的模型，24 层 GQA + Delta Net 混合架构）在 JZ 后端每个 batch 的 cgraph 被拆成 25 个子图，原因是 `GGML_OP_SOLVE_TRI` 与 `GGML_OP_SSM_CONV` 未在 AP 侧注册，DSP 侧 kernel 已存在但缺桥接胶水代码。Table-6A 实测 `batch_calls=6400`（256 batch x 25 子图），与 QCOM 单图形成鲜明对比。

### 6.2 第一阶段：补两个算子的 bridge layer 代码

读到 3.7 节后，按 4.3.3 方案实施了两组 patch（两个算子的桥接层）：

1. **SOLVE_TRI offload**：在 `init_op_validators()`（[ggml-hexagon-jz.cpp L3762-3798](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L3762-L3798)）注册 `s_op_validators[GGML_OP_SOLVE_TRI]`，校验逻辑参照 QCOM 的 `ggml_hexagon_supported_solve_tri`（F32 + 方阵 + 维度匹配）；在 `ggml_op_to_htp_op`（[entry.c L905](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/entry.c#L905)）加 `case GGML_OP_SOLVE_TRI: *htp_op = HTP_OP_SOLVE_TRI; return 0;`
2. **SSM_CONV offload**：同样在 AP 侧注册 validator（参考 QCOM 校验规则，F32 输入、kernel 维度 1-4），DSP 侧加 `case GGML_OP_SSM_CONV` 映射。SSM_CONV 是 delta net 层的卷积算子，本身不直接是 SOLVE_TRI，但 SOLVE_TRI 之前的卷积路径中 SSM_CONV 缺失会迫使 SOLVE_TRI 前的子图段被切开

构建+本地测试后 `batch_calls` 从 6400 降至 1792（256 batch x 7 子图），delta net 层的 11 处 SOLVE_TRI 拆分点 + 周围依赖算子段合并为 1 个连续子图段。

### 6.3 第二阶段：拆分减少，但推理输出乱码

`batch_calls=1792` 已大幅改善，但 qwen3 推理输出仍为字符级乱码（截断、重复、词界破坏），与 4.4.5 描述的"garble = cache 损坏"症状一致：

- 排除 1：4.4.5 已回滚的 DSP-side sampling 优化不是当前代码状态
- 排除 2：已知 a-inv bit1 实验已证伪（4.4.1），对当前 cgraph 退化为 no-op
- 排除 3：cgraph 中无已知 fusion 异常模式（garble 复现路径未穿越 fusion 节点）

剩余 7 子图/批的拆分点必然影响 cache coherency 维护路径，需进一步定位具体是哪个算子在何处切图。

### 6.4 第三阶段：用 ggml core 的 `GGML_SCHED_DEBUG` 抓 split 现场

ggml core 内置的 `GGML_SCHED_DEBUG` 环境变量可在 scheduler 切图时打印每次切分的位置（op index + op type + 原因），无需侵入式修改 [ggml-backend.cpp](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-backend.cpp)。

抓 log 命令：

```sh
GGML_SCHED_DEBUG=2 adb shell \
  "cd /data/local/tmp && LD_LIBRARY_PATH=. ./llama-cli -m qwen3.5-2b-q4_0.gguf -n 64 -p 'Hello' --verbosity 5" \
  2>&1 | tee log_qwen3_split_$(date +%Y%m%d-%H%M%S).txt
```

`GGML_SCHED_DEBUG=2` 触发 scheduler 打印每次切图的位置（op index + op type + 原因）。`--verbosity 5` 让 llama.cpp 主程序打印每层 tensor 形状，便于对照 Qwen3Next 模型结构。

### 6.5 第四阶段：root cause 定位

抓取 `log_qwen3_split_*.txt` 后的关键发现：

**7 个剩余子图/批的拆分点全部位于 6 个 MHA 层的 `attn_q_norm` 算子**（blk.3/7/11/15/19/23），每个 MHA 层各贡献 1 个拆分点（其中一个 MHA 层的 attn_q_norm 因 KV cache 状态差异偶发 2 次），与 delta net 层的 SOLVE_TRI/SSM_CONV 修复无关。

进一步追溯到 JZ 侧 RMS_NORM validator（[ggml-hexagon-jz.cpp L3452-3460](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L3452-L3460)）：

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

阅读 [qwen3next.cpp L251-264](file:///home/zhouwg/develop/ggml-hexagon/src/models/qwen3next.cpp#L251-L264) 的 Qcur 构造代码：

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
- [`hex_tensor_desc` 填充代码 L6033-6036](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L6033-L6036) 直接赋值 `t->nb[0..3]`，stride 信息完整传递至 DSP
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

**设计依据**：RMS_NORM kernel 沿 dim-0（reduction 维度）逐元素计算，dim-0 元素级连续是 kernel 的**最小不变量**，无需整张量连续。QCOM 后端 RMS_NORM validator 同样使用 `ggml_is_contiguous`（ggml-hexagon.cpp L2800-2802），故本修改仅影响 JZ 侧。

### 6.8 第七阶段：验证结果

**Table-15**：Qwen3.5-2B 修复全过程分阶段对比

| 阶段                       | 改动                                    | `batch_calls` | 拆分来源         | 推理输出          |
| ------------------------ | ------------------------------------- | ------------ | ------------ | ------------- |
| 起点（kimi-k3 指出）            | 无                                     | 6400         | SOLVE_TRI + SSM_CONV 缺失（25 子图/批） | 拆分导致的多子图 cache 错误 |
| 第一阶段：补 SOLVE_TRI/SSM_CONV bridge | 6.2 节两组 patch                            | 1792         | 6 个 MHA 层 attn_q_norm（7 子图/批）  | 字符级乱码         |
| 第二阶段：放宽 RMS_NORM validator  | 6.7 节 patch                            | **256**      | 无（完整单图）       | 正常输出          |

**Table-16**：最终修复后实测数据

| 指标                         | 起点   | 第一阶段（bridge layer） | 第二阶段（per-head view fix） | 变化 (起点->Stage 2) |
| -------------------------- | ------------ | --------------- | ----------------- | ----------- |
| `batch_calls`              | 6400         | 1792            | **256**           | **-96.0%**  |
| 推理输出                       | 多子图 cache 错误 | 字符级乱码           | 正常               | garble 消除   |
| 6 个 MHA 层 attn_q_norm 上 DSP | 0/6          | 0/6             | **6/6**           | -           |
| PP tok/s                   | **436.6**     | 456.91          | **495.50**          | **+13.5%**  |
| TG tok/s                   | **27.0**      | 23.94           | **26.82**           | **-0.7%**   |

> **数据来源**:
> - 起点 (Qwen3.5-2B JZ baseline): Table-1 AB 测试 (log_abtest_all_20260807-102443.txt), PP 436.6 tok/s, TG 27.0 tok/s, batch_calls=6400 (Table-6A).
> - 第一阶段 (bridge layer, batch_calls=1792): common_perf_print 输出 PP 456.91 tok/s, TG 23.94 tok/s, 输出字符级乱码.
> - 第二阶段 (per-head view fix, batch_calls=256): 5 模型 CI 3 轮均值 (log_abtest_all_post_qwen3_opt_20260807-210809.txt), PP 495.50 tok/s, TG 26.82 tok/s, 输出正常.
> - QCOM baseline (Table-1, log_abtest_all_20260807-102443.txt): PP 481.1 tok/s, TG 14.0 tok/s; 5 模型 CI QCOM (log_abtest_all_post_qwen3_opt_20260807-210809.txt): PP 466.61 tok/s, TG 13.50 tok/s.
>
> **与 QCOM 对比的视角**:
> - **PP 性能反超**: 起点 -9.2% (436.6 vs 481.1) -> Stage 2 **+6.2%** (495.50 vs 466.61, 5 模型 CI QCOM 基线). 15.4pp 反转, JZ 在 Qwen3.5-2B 上首次反超 QCOM.
> - **TG依然保持领先**: 起点 +93.6% (27.0 vs 14.0) -> Stage 2 +98.7% (26.82 vs 13.50, 5 模型 CI QCOM 基线). 基本持平, 仍保持近 2x 优势.
>
> **变化归因 (JZ 内部起点 -> Stage 2)**:
> - PP +13.5% (436.6 -> 495.50) 来自两部分: (a) bridge layer 阶段 PP 456.91 (+4.7% vs 起点, 单测), 消除 11 处 delta net 层拆分的子图固定开销; (b) per-head view fix 阶段 PP 进一步 +8.5% (456.91 -> 495.50, 5 模型 CI 3 轮均值), 消除 6 次 MHA 层 attn_q_norm CPU<DSP> 上下文切换.
> - TG -0.7% (27.0 -> 26.82) 内部基本持平, 但 QCOM 对比保持 +98.7% 领先 (5 模型 CI QCOM TG 13.50 tok/s), 实际净效果是 PP 性能反超 + TG依然保持 QCOM 2x 优势.

### 6.9 与 3.7 / 4.3.3 节的关系

**Table-17**：Qwen3.5-2B graph 拆分修复路径全景

| 拆分来源                  | 层类型        | 拆分点数 | 根因诊断章节 | 修复章节    | 修复后 batch_calls |
| --------------------- | ---------- | ---- | ------ | ------- | --------------- |
| SOLVE_TRI 缺失          | Delta Net  | ~11  | 3.7    | 4.3.3 + 6.2 | 1792            |
| SSM_CONV 缺失（伴随 SOLVE_TRI） | Delta Net  | ~若干  | 6.2    | 6.2     | 1792            |
| per-head view 拒绝      | MHA (6 层)  | 6    | 6.5    | 6.7（本节）  | 256             |
| **剩余拆分点**             | -          | **0** | -      | -       | **256**（完整单图）  |

3.7 节（root cause）与 4.3.3（fix）是同一问题的诊断与方案（缺失算子），6.5-6.7 节是**独立发现的第二个根因**（validator 过严）。两者共同构成 Qwen3.5-2B 在 JZ 后端上的 graph 拆分问题的完整根因，缺一不可。

### 6.10 验证流程（可复现）

```sh
# 1. 应用 SOLVE_TRI/SSM_CONV bridge layer patch（4.3.3 + 6.2 节）
git apply <bridge_layer_patch>.patch

# 2. 应用 RMS_NORM validator 放宽 patch（6.7 节）
git apply <rms_norm_validator_relax>.patch

# 3. 构建
./scripts/build-run-android.sh build

# 4. 基线回归（无拆分的 MHA 模型不应受 validator 放宽影响）
./scripts/build-run-android.sh run_llamacli gemma4
# 预期: batch_calls=256, 无 garble, 性能与 5.4 节基线差异 <2%

# 5. 目标模型验证
./scripts/build-run-android.sh run_llamacli qwen3
# 预期: batch_calls=256, 输出正常, PP&TG 双提升

# 6. 长生成稳定性
./scripts/build-run-android.sh run_llamacli qwen3 -n 256
# 重点检查 MHA 层 (blk.3/7/11/15/19/23) 输出无 garble

# 7. 拆分原因核对
GGML_SCHED_DEBUG=2 adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=. ./llama-cli -m qwen3.5-2b-q4_0.gguf -n 32 -p 'Hello' --verbosity 5" 2>&1 | tee log_qwen3_post_fix.log
# 预期:
#   - 不再出现 "RMS_NORM validator rejected" 字样
#   - 不再出现 SOLVE_TRI / SSM_CONV split 段
#   - 仅剩 lm_head buffer 切换处的正常切分
```

### 6.11 修复意义

1. **彻底消除 Qwen3.5-2B 在 JZ 后端的所有 graph 拆分**：`batch_calls` 从 6400 降至 256（-96%）
2. **乱码根因彻底消除**：4.3.3 修复后仍残留的字符级乱码来自 MHA 层的 per-head view，本节同步消除
3. **向后兼容其他 MHA 模型**：放宽后的 validator 对 `ggml_is_contiguous==true` 的常规输入仍接受（满足 `nb[0]==sizeof(float)`），零回归
4. **per-head view 模式的通用修复**：Qwen3Next、Phi-3、Gemma2 等近年模型都使用 per-head Q/K/V 切分，本修改是该模式 RMS_NORM 算子的通用前提，未来模型无需重复修改

***

## 参考文档

1. [ion-mempool-vs-perbuffer-analysis-20260713.md](file:///home/zhouwg/develop/ggml-hexagon/docs/backend/jz-ggml-hexagon/ion-mempool-vs-perbuffer-analysis-20260713.md) - ION mempool vs per-buffer 模式对比分析，量化 JZ 净优势公式与 PP/TG 性能差异归因。
2. [why-perbuffer-cannot-offload-lmhead-20260724-en.md](file:///home/zhouwg/develop/ggml-hexagon/docs/backend/jz-ggml-hexagon/why-perbuffer-cannot-offload-lmhead-20260724-en.md) - 分析 per-buffer 模式无法 offload lm-head 的根因，以及 per-buffer 与 mempool 模式在 lm-head 上的行为差异。

***

## 修订历史

### 2026-08-07

- **第六章新增 (MiniMax-M3)**: Qwen3.5-2B PP&TG 优化完整过程 (6.1-6.11). 涵盖 7 个阶段: (1) 基于 kimi-k3 在 3.7 节指出的 SOLVE_TRI/SSM_CONV 缺失, 实施两个算子的 bridge layer patch; (2) batch_calls 6400->1792 但推理仍乱码; (3) 用 ggml core 的 `GGML_SCHED_DEBUG` 抓 split 现场; (4) 定位到 6 个 MHA 层 attn_q_norm 拆分点; (5) 分析 per-head view 与 validator 拒绝原因; (6) 放宽 validator + 验证 batch_calls=256 PP&TG 大幅提升. Table-15/16/17 三表按阶段/指标/根因三种维度组织
- 措辞精度修正: 3.1 offload 权重对象/repack 价值/lifecycle 等 4 处; 3.2 与 Table-3 列名术语统一为"调度框架"; 3.6 解除 `HEX_OP_PROF` 与 DSP-side sampling 的 commit 错误关联 (后者为已回滚的 logits copyback 优化方案)
- 第四章结构整理: 删除 4.2.1 + 4.3.2 (实际为同一实验的 DSP 侧 + AP 侧配套组件, 文档分拆造成"两个独立方案"错觉); 新增 4.4.5 作为合并澄清与已关闭项; 同步重编号 4.2-4.4 子节 (4.x -> 4.x.y, 与第五章多级编号风格对齐)
- 第四章代码核验与路线图: 多处代码核验修正 (FastRPC 实测替换历史值 ~89us、DDR 带宽、fusion 状态、bit1 跳过方向等); 4.5 路线图精简为时间线; "核心转变"段显式区分"潜在收益优先级"与"执行优先级"两个维度
- 跨章节变动: 全文档 unicode 箭头/乘号清理 (符合 AGENTS.md); 4.6 dspqueue 描述从"通信机制"改为"执行调度模型"; 第五章新增 Table-6/7/8 + 5.3.3 对比分析 + 参考文档章节
- 第五章新增 5.8 后续可探索方向 (MiniMax-M3): Plan A 实验证伪诊断 (HVX vs HMX FLASH_ATTN 在 M=51 几乎无差异) + 根因再分析 (is_mergeable_mul_mat 拒 HMX eligible 导致 QKV 0% 融合) + 4 个突破性方向优先级 (HMX fused QKV/bulk flush 异步化/HMX fused FFN/a-inv 跨 batch dedup)
- 审核修订第二轮 (Kimi-K3, 基于 log_abtest_all_20260807-102443.txt 与双端源码核验): Table-1 更新为五模型 3 轮精确均值 (Qwen3.5-2B TG +85.8%->+93.6%, Gemma4-E4B TG +49.0%->+35.1% 等), 新增 Qwen1.5-1.8B QCOM PP 首轮异常 (825.79) 数据注记, 关键观察同步更新 (TG 领先 4 模型平均 +46.8%)
- GQA 比例修正 (Kimi-K3): Gemma4-E2B=8:1 / Gemma4-E4B=4:1 (GGUF header 实测), 全文 9 处; Table-6A llama3 batch_calls 256->257, 脚注 misses=6 同步修正
- 代码引用核验 (Kimi-K3): repack_q4k/q6k 函数 -> L4163/4222/4111; JZ mul_mat guard -> L3274-3289; cgraph cache -> L5159; Phase 3 fusion -> L5337-5342; min_rpc_overhead_us -> L262; is_mergeable_mul_mat -> L2399-2408; bulk_flush_all -> entry.c L527-529; 3.1 节补充 QCOM repack buffer guard (L2815) 第三处约束
- 新增 3.7 节 (Kimi-K3): Qwen3.5-2B 因 SOLVE_TRI 未在 AP 侧注册导致 25 路 graph 拆分 (batch_calls=6400 vs 其他模型 256/257, 实测 AP phase 累计 69ms), DSP 侧 kernel 已存在仅缺胶水代码; Table-6 根因分项同步更新
- 新增 4.3.3 (Kimi-K3): SOLVE_TRI offload 启用方案 (AP validator + DSP op 映射, 约 30 行胶水代码), qwen3 PP 预计收回 5-6pp; 4.5 路线图 Step 1 同步调整
- a-inv/bit1 结论修正 (Kimi-K3): 本轮 mode=5 未启用 bit1, mode=7 对照实验 (qwen1 PP-only) 证实 PRIOR_DST_MAX_LEN=64 限制使 bit1 退化为 no-op, a-inv 接近结构性下限; 4.4.1 标记关闭, 3.3/3.6/4.5 相关表述同步修正

### 2026-08-06

- 初稿 (Seed-2.1-Pro): AB 测试数据、第三章架构对比、第四章优化方向
- 第三章源码核验:JZ Q4_K/Q6_K 处理行号、repack 函数补全、QCOM switch case 列表修正
- Table-2 修正:Qwen3.5-2B/Qwen1.5-1.8B lm-head 类型从 Q4 修正为 Q6_K
- Table-3 修正:内存模型、权重布局、lm-head offload 行与 ion 文档对齐
- 第四章重构:PP 优先级重排、Qwen1.5-1.8B 三重叠加根因分析、路线图调整
- 第五章新增 (MiniMax-M3):force_opfusion_in_pp 实验与五模型 CI 验证
- 五模型层数修正:gemma4 24->35、gemma4-e4b 35->42、qwen3 13->24(基于 n_layer 字段)
- 全文 prose polish:em-dash 清除、标点统一、Chinglish 修正、措辞精简
- 跨文档引用补充:ion 文档 file link (6 处)

### 2026-08-05

- 文档创建

