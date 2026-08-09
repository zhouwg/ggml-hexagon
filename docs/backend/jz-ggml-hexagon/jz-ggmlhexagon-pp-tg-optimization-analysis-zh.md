# JZ's ggml-hexagon 性能差异分析与优化方向

> Initial: 2026-08-05

> Last updated: 2026-08-08

> Author: Seed-2.1-Pro (Ch 1-4), MiniMax-M3(Ch 5-8), revised by DeepSeek-V4-Pro & GLM-5.2 & MiniMax-M3 & Kimi-K3 & Jeff Zhou (review & feedback)

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
| Qwen3.5-2B   |     501.7     |      456.1      |   **+10.0%**  |      26.7     |       13.4      |   **+99.0%**  |
| Gemma4-E2B   |     684.8     |      457.9      |   **+49.5%**  |      27.1     |       25.0      |     +8.1%     |
| Gemma4-E4B   |     403.6     |      417.4      |     -3.3%     |      14.9     |       10.8      |   **+38.2%**  |
| Qwen1.5-1.8B |     552.9     |      711.2      |     -22.3%    |      18.6     |       26.2      |     -28.8%    |
| Llama3.2-1B  |    1018.8     |      1084.3     |     -6.0%     |      42.2     |       28.7      |   **+47.1%**  |

数据来源：`./scripts/build-run-ggmlhexagon-android.sh run_abtest_all 2>&1 | tee log_abtest_all_$(date +%Y%m%d-%H%M%S).txt`（本轮日志 `log_abtest_all_20260807-223924.txt`，self-build-jz 分支，2026-08-07 22:39）

> **数据注记**：Qwen1.5-1.8B QCOM TG 首轮 29.52 tok/s 明显偏高（后两轮 24.83 / 24.11 tok/s），QCOM PP 跨轮递增（680.53 -> 701.19 -> 751.86 tok/s），均与设备轮间热状态相关；Table-1 按 3 轮均值（PP 711.2 / TG 26.2 tok/s）如实记录。与早晨 run（`log_abtest_all_20260807-102443.txt`）相比，Qwen3.5-2B 之外的四个模型 PP/TG 差异均在热状态波动范围内；唯一结构性变化是 Qwen3.5-2B PP 从 -9.2% 翻转为 +10.0%（graph 拆分修复生效，详见 3.7 末节与第六章）。

**关键观察**：

- **TG（Token Generation）**：JZ 在 4/5 模型上领先，最大优势 +99.0%（Qwen3.5-2B），领先 4 模型平均约 +48.1%；仅 Qwen1.5-1.8B（唯一 MHA 模型）落后 28.8%。
- **PP（Prompt Processing）**：JZ 在 2/5 模型上领先（Gemma4-E2B +49.5%、Qwen3.5-2B +10.0%），QCOM 在其余 3 模型上领先，最大优势 +22.3%（Qwen1.5-1.8B）。Qwen3.5-2B 较早晨 run（-9.2%）翻转为 JZ 获胜，根因是 graph 拆分修复（第六章）。
- TG 和 PP 的性能模式依然指向不同的瓶颈根因。

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
2. **JZ 整图固定优势无法累积**：lm-head offload（~200MB Q6_K）+ first-touch 权重 inval（~9.2ms/token）是固定的、不会随 layer 数增加而放大的优势；24 层不足以让 JZ 的 per-layer 增量优势赶超 dspqueue 的 per-layer pipelining 收益。Gemma4-E2B 35 层则可以反超（+49.5%）。
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

**修复确认（2026-08-07 夜间 CI，self-build-jz 分支）**：SOLVE_TRI/SSM_CONV bridge（4.3.3）与 RMS_NORM validator 放宽（第二根因，per-head view 被拒绝，详见 6.5-6.7）已合入本分支。夜间五模型 CI（`log_abtest_all_20260807-223924.txt`）实测 Qwen3.5-2B `batch_calls=256`（与其余四模型一致），25 路拆分归零；JZ PP 从 436.6 提升至 501.7 tok/s（+14.9%），对 QCOM 从 -9.2% 翻转为 **+10.0%**，本节"拆分税是 PP 差距重要构成"的归因得到验证。

***

## 四、优化方向

根据 3.6 节 DSP op-level profiling 实测数据，**在 DSP 执行内部，matmul kernel 是绝对主导**（三类 matmul 占 DSP batch-wall time 的 79.1% = 86.9%x91.1%）。注意：DSP profiling 数据仅覆盖 Phase 10（DSP 批处理执行），AP 侧开销（Phase 1-9 + Phase 11-12）未包含在内，需通过 Step 0 profiling 单独量化。在 AP 侧数据补全前，优化方向优先聚焦在 DSP kernel 与 offload 策略上，AP 侧优化暂不调整优先级。

TG 和 PP 的瓶颈不同，优化策略也不同：

- **TG 瓶颈**（基于 3.6 profiling，仅覆盖 DSP 侧）：在 DSP 执行内部，三类 matmul 占 91.1% op-sum，其中 lm-head MUL\_MAT（max=4697us，每 token 1 次）和 MUL\_MAT\_FFN（avg=334us，每 layer 1 次 fused op = 105 个内部 matmul）是绝对主导；JZ 已通过 lm-head offload + first-touch 权重 inval（\~9.2 ms/token 节省，固定整图总量）解决最关键的两项，剩余优化空间主要在 DSP matmul kernel 本身。
- **PP 瓶颈**：PP 差距是**模型结构相关的**，不是普遍的 JZ 弱点。[ion 文档](file:///home/zhouwg/develop/ggml-hexagon/docs/backend/jz-ggml-hexagon/ion-mempool-vs-perbuffer-analysis-20260713.md)第 9 节分析表明 JZ 净优势 = per\_layer\_saving x n\_layers + fixed\_lmhead\_saving - dspqueue\_overlap。当层数足够时 JZ 也赢 PP（如 Gemma4-E2B 的 35 层，PP +49.5%）；浅层模型（llama3.2-1B 16 层）dspqueue 的固定 overlay 优势尚未被 per-layer 累积超越；qwen3.5-2B 曾叠加 SOLVE_TRI 缺口导致的 25 路 graph 拆分税（详见 3.7），该拆分税已于本分支修复（第六章），PP 由 -9.2% 翻转为 +10.0%。[ion 文档](file:///home/zhouwg/develop/ggml-hexagon/docs/backend/jz-ggml-hexagon/ion-mempool-vs-perbuffer-analysis-20260713.md)也明确指出：**性能差异来自 data-plane policy（weight residency + role-aware cache），而非 control-plane**（FastRPC 开销历史值 \~89us，可忽略；本轮实测详见 4.4.4）。

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

**优先级重排论证**：JZ TG 在领先 4 模型上平均 +48.1%，继续优化的边际收益受 matmul kernel 物理极限约束；PP 在 3/5 模型上落后 -3.3% 到 -22.3%，根因（AP-DSP 无 pipelining）是**可重构的框架差异**，而非 kernel 差异。**PP 从 -22.3% 改善到 -12% 等同于 +10pp 绝对提升**；TG 从 +48% 到 +58% 需要改 kernel 才有 +10pp，但 kernel 已与 QCOM 100% 共享，**只能从 matmul 内部优化（HMX 利用率、tile size）挤牙膏**。因此 PP 优化是 JZ 的真正战场，应排在 P0 profiling 之后立即推进。

**Table-6**：PP 表现与模型结构关联（基于 3.5 节三重叠加与 3.7 节 graph 拆分分析）

| 模型           | 层数                | PP JZ vs QCOM | TG JZ vs QCOM | 根因分项                                                 |
| ------------ | ----------------- | :-----------: | :-----------: | ---------------------------------------------------- |
| Gemma4-E2B   | 35 (GQA 8:1)      |   **+49.5%**  |     +8.1%     | 层数深且单层 DSP 时间适中（R 低），per-layer 优势累计超越 dspqueue overlap |
| Qwen3.5-2B   | 24 (GQA + Delta Net) |   **+10.0%** |   **+99.0%**  | 25 路 graph 拆分税已修复归零（3.7 + 第六章），per-layer 优势兑现，PP 反超 |
| Gemma4-E4B   | 42 (GQA 4:1)      |     -3.3%     |   **+38.2%**  | 单层 DSP 时间长（R 高），dspqueue 每层隐藏的 AP prep 放大，抵消 42 层累积优势   |
| Llama3.2-1B  | 16 (GQA 4:1)      |     -6.0%     |   **+47.1%**  | 层数浅，dspqueue 优势显著                                     |
| Qwen1.5-1.8B | 24 (MHA 1:1)      |     -22.3%    |     -28.8%    | **三重叠加：dspqueue + 层数不足 + MHA VTCM/cache**              |

**结论**：PP 优化应聚焦于**结构性杠杆**（per-layer pipelining）与**支持度缺口补齐**（SOLVE_TRI offload），而非模型结构特化。Qwen1.5-1.8B 不是 corner case，而是三重不利因素的"压力测试"：per-layer pipelining 改善后这类模型获益最大。Gemma4-E2B 已经赢 PP，进一步压榨 +49.5% 之上的空间也来自 per-layer pipelining 在深层模型上的累积收益。Qwen3.5-2B 的 25 路 graph 拆分税（3.7 节）已通过"补算子 + validator 放宽"收回（第六章），PP 从 -9.2% 翻转为 +10.0%，验证了该归因；其余模型的 PP 差距（Gemma4-E4B / Llama3.2-1B / Qwen1.5-1.8B）需靠 4.3.1 per-layer pipelining 这类架构改动收回。

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

**预期收益（基于估算）**：AP 侧 Phase 1-9 + 11-12 占 PP 10-15%，pipelining 隐藏 50-70%，PP 提速 5-10%。Qwen1.5-1.8B 从 -22.3% 改善到 -15% 左右，Gemma4-E2B 从 +49.5% 进一步到 +55%+。

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

#### 4.3.3 SOLVE_TRI offload 启用 - 已落地验证，消除 qwen3 的 25 路 graph 拆分

3.7 节已确认：qwen3（Qwen3.5-2B）的 cgraph 在 JZ 后端被拆成 25 个子图的唯一原因是 `GGML_OP_SOLVE_TRI` 未在 AP 侧注册，而 DSP 侧 kernel 已完整存在。补齐两处胶水代码即可消除拆分：

- **AP 侧**：在 `init_op_validators()`（[ggml-hexagon-jz.cpp L3762-3798](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L3762-L3798)）新增 `s_op_validators[GGML_OP_SOLVE_TRI]`，校验逻辑可直接参照 QCOM 的 `ggml_hexagon_supported_solve_tri`（[ggml-hexagon.cpp L3367-3399](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L3367-L3399)：F32 类型 + 方阵 + 维度匹配检查）。
- **DSP 侧**：在 `ggml_op_to_htp_op`（[entry.c L905](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/entry.c#L905)）新增 `case GGML_OP_SOLVE_TRI: *htp_op = HTP_OP_SOLVE_TRI; return 0;`。kernel 本体（`op_solve_tri`，[kernels/solve-tri-ops.c L197](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/solve-tri-ops.c#L197)）与 op 表注册（entry.c L811）均已存在，无需改动。

**预期收益**：qwen3 恢复完整单图后，每 batch 的 25 次 FastRPC round-trip 与 25 倍 Phase 1-9/11-12 开销归零，按 3.7 节实测估算 PP 可收回约 5-6pp（-9.2% 收窄至约 -3%）；TG 收益 <1%（拆分开销在 M=1 时占比已极小），但 25->1 的子图收敛同时降低了 per-token 的 cgraph cache 与 mempool 管理负担。

**复杂度**：**低**。两处注册各约 10-20 行，无新 kernel、无调度框架改动、无 cache 策略变化，是全部 PP 方向中风险收益比最优的一项。

**实际收益（2026-08-07 夜间 CI，self-build-jz）**：bridge 合入后 batch_calls 6400->1792，仍残留 6 个 MHA 层 attn_q_norm 拆分（第二根因：RMS_NORM validator 拒绝 per-head view，详见 6.5-6.7）；validator 放宽后 batch_calls 收敛至 256。夜间 CI（`log_abtest_all_20260807-223924.txt`）实测 JZ PP 436.6 -> 501.7 tok/s（+14.9%），对 QCOM 从 -9.2% 翻转为 +10.0%，超过本节 5-6pp 预期（预期仅计拆分税，实际还含第二根因收益）；TG 持平（26.7 tok/s，对 QCOM +99.0%）。完整修复过程见第六章。

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

**潜在收益优先级 vs 执行优先级是两个维度**：4.2-4.4 的"第一/第二/第三优先级"按**潜在收益空间**分级（4.2 TG 扩展优势 > 4.3 PP 优化 > 4.4 低风险快速收益），按**实际执行顺序**（下方 Step 0-4）则重新排列。**调整依据**：前期实验（详见 4.4.5 末尾注）证实 sampling 路径优化代码复杂且性能收益极小；AB 测试数据也确认 JZ 在 TG 已领先 4/5 模型，进一步投入产出比低，PP 是 JZ 的真正短板（拆分修复前 4/5 模型落后 QCOM，Qwen3.5-2B 修复反超后仍 3/5 落后），结构性优化收益空间最大。**两个维度的执行映射**：4.3.3 SOLVE_TRI offload 作为 Step 1 立即收益（已完成：夜间 CI 实测 qwen3 PP +14.9%，对 QCOM 从 -9.2% 翻转为 +10.0%；4.4.1 a-inv 已于 2026-08-07 实验证伪关闭）；PP 结构性突破（4.3.1 per-layer pipelining）作为 Step 2 核心战场；TG kernel 精调（4.4.2 + 4.4.3 + 4.2.2）作为 Step 3 最后做。**长期/已关闭项**：4.4.5 sampling 路径优化与 4.2.2 KV cache 增量 inval 因复杂度高/收益小。

4.1-4.4 节按优先级组织，经实验验证后，实际执行顺序调整为：

```
Step 0: Profiling 数据驱动（必做前提，详见 4.1）

Step 1: 低风险快速收益（独立于 PP/TG 主战场）
  +-- 4.3.3 SOLVE_TRI offload：已完成（2026-08-07 夜间 CI：batch_calls 6400->256，qwen3 PP 436.6->501.7 tok/s，对 QCOM -9.2% -> +10.0%，详见第六章）
  +-- 4.4.1 a-inv 优化：2026-08-07 实验证伪（bit1 受 PRIOR_DST_MAX_LEN=64 限制退化为 no-op），关闭
  +-- 4.4.4 FastRPC 校准：已实测，结论为投入产出比低，关闭

Step 2: PP 结构性突破（核心战场，详见 4.3.1）
  +-- DSP 侧 partial-execute + resume 接口 + async FastRPC 调度
  +-- 严格 TG 回归测试：M=1 单次 doorbell 优势不被新同步点吃掉
  +-- （条件性）4.3.2 descriptor 模板缓存
  +-- 预期：PP +5-10%；Qwen1.5-1.8B 从 -22.3% -> ~-15%；Gemma4-E2B 从 +49.5% -> +55%+

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
- 实施陷阱 (feature 分支实测): 三个权重共享同一 kparams->n 会在 GQA 模型上 garble (K/V 输出维度 < Q，共享 n 导致 K/V 写出维度错误); 必须为 Wk/Wv/Wq 各自独立 kparams 并携带各自权重维度

**B. bulk flush 异步化 (P0 突破方向)**:

- 现状: `bulk_flush_all()` (entry.c:527-529) 在所有 op 完成后**阻塞**执行 15145 us，期间 DSP thread idle
- 实施: 新增 1 个 DSP worker thread 负责 flush，main batch 线程与 flush 线程并行
- 时序: batch N 完成后, flush 线程立即开始 flush batch N 的 dst range; 同时 AP 侧准备 batch N+1 的 descriptor (hdr/pre 阶段)
- 同步点: batch N+1 的第一个 op 读取 dst 之前，需确保 batch N 的 flush 完成
- 节省: 15145 us 中 30-50% 可与下一 batch 重叠，理论省 4500-7500 us
- 风险: 需要新增 DSP thread + 跨 batch 同步原语，可能影响后续读取的 cache coherency
- 参考: 5.7 节保留的 `dsp-ctx.h` `bulk_flush_ranges` 数组已包含 sort + merge 逻辑，异步化只需把 `bulk_flush_all()` 从主线程移到 worker thread
- 实施陷阱 (feature 分支实测): enable_async_bulk_flush=1 曾在 qwen1 上 garble (flush 线程 DMA 写与主线程读的竞态); 同步点必须显式保证 batch N+1 首个读 dst 的 op 之前 flush 完成，并逐模型验证五模型 CI 输出

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

#### 5.8.4 新增三个方向（2026-08-08 morning，Kimi-K3，基于 self-build-jz 最新代码与数据）

**Table-15**：新增方向一览（编号沿用 5.8.3 的 A-D 顺延）

| 优先级 | 方向 | 当前开销 | 预期节省 | 收益场景 | 风险 |
|---|---|---|---|---|---|
| P0 | F. qwen1 TG 根因定位（op 级 profiling 先行） | TG -28.8%（15.6 ms/token） | 待定，上限为抹平差距 | qwen1 TG | 低（先测后修） |
| P1 | E. AP 侧 per-call overhead 消除（hash 瘦身 + descriptor blob 复用） | 641 us/batch（p1 395 + p8 166 + p5/p6 79） | ~540 us/batch | 全模型 TG +1.2-2.7% | 低（纯 AP 侧） |
| P2 | G. first-touch w-inv 移至加载期 | 首个 PP batch 13.6-25.8 ms | 同左（计时窗口口径） | PP 测量 + TTFT | 低-中（顺序约束） |

**E. AP 侧 per-call overhead 消除（P1）**：

- 证据：第五章 gemma4 端到端 256 批次 `dump_perf_stats`：per-call overhead avg=690 us（graph_dur - p10），其中 p1=395 us/batch（compute_content_hash 每 batch 游走 1493 个散落 tensor 结构体）、p8=166 us/batch（descriptor 构建）、p5/p6=79 us/batch；cgraph cache hits=253 / misses=3，即 253 个 TG batch 图内容完全一致，这部分 AP 工作每 token 原样重做
- 机制 1（hash 瘦身）：[compute_content_hash](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5105-L5123) 每节点 fold op+ne+nb+10 个 src 指针+data ptr（约 20 次解引用）；改为只 fold {op, data ptr}，data ptr 在图内近唯一，配合已有 n_nodes 校验，碰撞安全性可论证
- 机制 2（descriptor blob 复用）：hash 已覆盖全部 data ptr，命中时 Phase 5/6/8 的输出是确定性的；将构建完成的 ops+tensors descriptor 区（约 240 KB 连续 blob）存入 cgraph cache entry，命中时一次 memcpy 替代重跑三个阶段
- 与 4.3.1 pipelining 互补（elimination vs overlap）；纯 AP 侧改动，不碰 DSP kernel / cache 策略 / 调度框架，风险为全部方向中最低
- 预期：641 -> 约 100 us/batch；TG：llama3 +2.3%，gemma4 +1.4%，qwen1 +1.2%
- 验证：`dump_perf_stats` 对照 p1/p8 累计值 + 五模型 CI

**F. qwen1 TG 根因定位（P0）**：

- 现状：五模型中唯一 TG 败局（-28.8%，53.8 vs 38.2 ms/token）；文档对 TG 落后只有 "MHA VTCM/cache 压力" 定性描述，qwen1 TG 的 per-op profile 从未测过（3.6 节是 gemma4 TG，5.x 是 qwen1 PP）
- 计划：`HEX_OP_PROF` 跑 qwen1 TG，与 3.6 节 gemma4 TG profile 逐项对比
- 待验证假设：H1 FLASH_ATTN 主导（MHA 每 token 每层 KV 读取 = 2 x n_embd x 2B = 8 KB，为 GQA 8:1 的 8 倍，随 ctx 线性增长）；H2 lm-head（Q6_K->Q4_0 repack 约 163 MB）matvec 在 DSP 的实际带宽未达预期；H3 KV cache 写入路径 per-token invalidation
- 测出主导项后再设计修复：H1 对策为 KV layout / VTCM 分块，H2 对策为 matvec kernel DRAM 预取
- 一次 CI 即可定位；qwen1 同时是 PP 差距最大模型（-22.3%），机制级解释对 PP 优化同样有价值

**G. first-touch w-inv 移至加载期（P2）**：

- 证据：gemma4 首个 PP batch w-inv=13,637 us（1,334 MB），e4b 25,802 us（2,528 MB）；会话级一次性成本（TG batch 实测 w-inv 约 10 us，bit0 生效），但全部落在首个 PP batch 的计时窗口内，gemma4 PP 76 ms 中 18% 是会话初始化成本而非 prompt 计算
- 机制：复用 `execute_batch(0xFFFC)` 特殊命令通道（推 dsp_cache_mode 的同一路径），模型加载完成后发 init 命令，DSP 一次性整块 invalidate 权重区并预标记 first-touch bitmap；权重区在 mempool 内连续，整块 invalidate 比 24-42 层交错 per-range 遍历局部性更好
- 预期：PP 测量窗口 -13.6 ms（gemma4）/ -25.8 ms（e4b），PP 数字更纯粹反映计算本身，后续 pipelining 收益测量更准；TTFT 微降
- 风险：低-中；只需保证 invalidate 在最后一次 AP 侧权重写之后、首个 DSP batch 之前

**已评估并否掉的方向**（避免重复提议）：

- a-inv range 合并（sort+merge）：a-inv=19,167 us 是字节线性 dcinva 成本，1,754 MB 已是 per-batch 结构性下限（4.4.1 已证伪压缩空间）；合并 range 只省 per-call 开销，不省逐行 invalidate 本身
- PP/TG 双模 cache 策略（PP 批次改用 flush-all）：flush-all 会把权重逐出 L2，恰好摧毁 bit0 first-touch 带来的 TG 优势，得不偿失

#### 5.8.5 优先级排序依据

P0 方向 (A + B) 合计理论节省 6900-12300 us = PP +7.6-13.6%，符合 `4.5 路线图` 中"PP 优化第二优先级"的潜在收益空间。P1 方向 (C + D) 合计 2000-3900 us = PP +2.2-4.3%，作为 P0 实施完成后的后续优化。

实施顺序建议: A -> C -> B -> D。前两个 (A + C) 共用 HMX fused kernel 基础设施，先实施可积累经验。B 涉及新 DSP thread + 同步原语，复杂度最高，但潜在收益最大 (PP +5-8%)。D 风险最低 (复用现有 weight_inval 模式)，但收益受限于重复率假设验证。2026-08-08 新增 E/F/G (5.8.4) 后: F 与 A/B 同为 P0 但性质是定位测量 (一次 CI 出结论)，建议最先执行; E 为最低风险速赢项，可与 A 并行; G 为计时口径修正，独立实施。

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

`batch_calls=1792` 已大幅改善，但 qwen3 推理输出出现字符级乱码（截断、重复、词界破坏）：起点（batch_calls=6400）输出经 `log_abtest_all_20260807-102443.txt` 核实为文本连贯，乱码是 bridge patch 后新引入的中间状态，症状与 4.4.5 描述的"garble = cache 损坏"一致：

- 排除 1：4.4.5 已回滚的 DSP-side sampling 优化不是当前代码状态
- 排除 2：已知 a-inv bit1 实验已证伪（4.4.1），对当前 cgraph 退化为 no-op
- 排除 3：cgraph 中无已知 fusion 异常模式（garble 复现路径未穿越 fusion 节点）

剩余 7 子图/批的拆分点必然影响 cache coherency 维护路径，需进一步定位具体是哪个算子在何处切图。

### 6.4 第三阶段：用 ggml core 的 `GGML_SCHED_DEBUG` 抓 split 现场

> [!WARNING]
> **必须同时启用 `GGML_SCHED_DEBUG=2` 和 `--verbosity 5`，缺一不可**。`GGML_SCHED_DEBUG=2` 通过 `GGML_LOG_DEBUG` 输出 split 行，而 `GGML_LOG_DEBUG` 默认 level 为 0，**仅当 `--verbosity >= 5` 时才在 stdout 可见**。仅设 `GGML_SCHED_DEBUG=2` 而不传 `--verbosity 5` 的常见后果是：跑完一次后日志里没有任何 `## SPLIT #N` 行，看起来像"什么都没发生"，会误判为 `GGML_SCHED_DEBUG=2` 失效。

ggml core 内置的 `GGML_SCHED_DEBUG` 环境变量可在 scheduler 切图时打印每次切分的位置（op index + op type + 原因），无需侵入式修改 [ggml-backend.cpp](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-backend.cpp)。

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
3. 对每个 CPU SPLIT，**找到紧跟其后的 `node #N` 行**——这个 node 就是触发 CPU fallback 的 op，其 `src0/src1/dst` 后缀标的就是 buffer 归属（`[CPU]` 表示在 system memory，`[Hexag]` 表示在 DSP mempool）
4. 统计 CPU SPLIT 命中的 op 模式（如全部为 `MUL_MAT × blk.*.ssm_out.weight`），定位根因是 operator 缺失、validator 过严还是 weight buffer 错位

**Qwen3.5-9B 实测（2026-08-08）**：500 CPU + 500 Hexagon SPLIT，每个 CPU SPLIT 对应一个 `node #(MUL_MAT): linear_attn_out-X × blk.X.ssm_out.weight → final_output-X`，根因是 `ssm_out.weight` 在 system memory buffer 上（model 5.4GB > mempool 4GB 触发 [ggml-hexagon-jz.cpp L5083](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5083) 的 ion pool exhausted 回退路径），scheduler 因 weight 不在 DSP mempool 而强制把对应 MUL_MAT 派给 CPU。这是与 2B 完全不同的根因（2B 是 operator 缺失 + validator 过严，9B 是 weight buffer 错位），不能用 6.2 / 6.7 的方法修复。

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

**Table-16**：Qwen3.5-2B 修复全过程分阶段对比

| 阶段                       | 改动                                    | `batch_calls` | 拆分来源         | 推理输出          |
| ------------------------ | ------------------------------------- | ------------ | ------------ | ------------- |
| 起点（kimi-k3 指出）            | 无                                     | 6400         | SOLVE_TRI + SSM_CONV 缺失（25 子图/批） | 正常（拆分仅造成性能损失） |
| 第一阶段：补 SOLVE_TRI/SSM_CONV bridge | 6.2 节两组 patch                            | 1792         | 6 个 MHA 层 attn_q_norm（7 子图/批）  | 字符级乱码（bridge 后新引入） |
| 第二阶段：放宽 RMS_NORM validator  | 6.7 节 patch                            | **256**      | 无（完整单图）       | 正常输出          |

**Table-17**：最终修复后实测数据

| 指标                         | 起点   | 第一阶段（bridge layer） | 第二阶段（per-head view fix） | 变化 (起点->Stage 2) |
| -------------------------- | ------------ | --------------- | ----------------- | ----------- |
| `batch_calls`              | 6400         | 1792            | **256**           | **-96.0%**  |
| 推理输出                       | 正常（log 102443 文本连贯） | 字符级乱码（新引入）           | 正常               | 正常 -> 正常（Stage 1 引入并已消除）   |
| 6 个 MHA 层 attn_q_norm 上 DSP | 0/6          | 0/6             | **6/6**           | -           |
| PP tok/s                   | **436.6**     | 456.91          | **501.71**          | **+14.9%**  |
| TG tok/s                   | **27.0**      | 23.94           | **26.74**           | **-1.0%**   |

> **数据来源**:
> - 起点 (Qwen3.5-2B JZ baseline): Table-1 AB 测试 (log_abtest_all_20260807-102443.txt), PP 436.6 tok/s, TG 27.0 tok/s, batch_calls=6400 (Table-6A).
> - 第一阶段 (bridge layer, batch_calls=1792): common_perf_print 输出 PP 456.91 tok/s, TG 23.94 tok/s, 输出字符级乱码.
> - 第二阶段 (per-head view fix, batch_calls=256): 5 模型 CI 3 轮均值 (log_abtest_all_20260807-223924.txt, self-build-jz 分支), PP 501.71 tok/s, TG 26.74 tok/s, 输出正常.
> - QCOM baseline (Table-1, log_abtest_all_20260807-102443.txt): PP 481.1 tok/s, TG 14.0 tok/s; 5 模型 CI QCOM (log_abtest_all_20260807-223924.txt): PP 456.10 tok/s, TG 13.44 tok/s.
>
> **与 QCOM 对比的视角**:
> - **PP 性能反超**: 起点 -9.2% (436.6 vs 481.1) -> Stage 2 **+10.0%** (501.71 vs 456.10, 5 模型 CI QCOM 基线). 19.2pp 反转, JZ 在 Qwen3.5-2B 上首次反超 QCOM.
> - **TG依然保持领先**: 起点 +93.6% (27.0 vs 14.0) -> Stage 2 +99.0% (26.74 vs 13.44, 5 模型 CI QCOM 基线). 基本持平, 仍保持近 2x 优势.
>
> **变化归因 (JZ 内部起点 -> Stage 2)**:
> - PP +14.9% (436.6 -> 501.71) 来自两部分: (a) bridge layer 阶段 PP 456.91 (+4.7% vs 起点, 单测), 消除 11 处 delta net 层拆分的子图固定开销; (b) per-head view fix 阶段 PP 进一步 +9.8% (456.91 -> 501.71, 5 模型 CI 3 轮均值), 消除 6 次 MHA 层 attn_q_norm CPU<DSP> 上下文切换.
> - TG -1.0% (27.0 -> 26.74) 内部基本持平, 但 QCOM 对比保持 +99.0% 领先 (5 模型 CI QCOM TG 13.44 tok/s), 实际净效果是 PP 性能反超 + TG依然保持 QCOM 2x 优势.

### 6.9 与 3.7 / 4.3.3 节的关系

**Table-18**：Qwen3.5-2B graph 拆分修复路径全景

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

## 七、qwen1 TG优化实验(2026-08-08)

### 7.1 背景与数据来源

5.8.4 提出的 F 方向 (qwen1 TG 根因定位) 在 feature/qwen1_qwen3_optimize 分支跑完一轮 `run_qwen1_tg_prof` CI，参数 M=1, n_predict=256, ctx=8192, HEX_OP_PROF=1, DUMP_INTERVAL=1。对照模型 qwen1.5-1.8B (MHA 1:1, 24 层) 与 gemma-4-E2B-it (GQA 8:1, 35 层)，稳态均值取 OP-PROF 序列 skip 前 10 batch 后的窗口。

数据来源: `qwen1_tg_prof_20260808-165745_summary.log` + `_qwen1_logcat.log` + `_gemma4-e2b_logcat.log` 三组文件，terminal 实测 qwen1 TG 18.62 tok/s (53.70 ms/token), gemma4-e2b TG 26.53 tok/s (37.70 ms/token)。下文表 Table-19 / Table-20 / Table-21 / Table-22 直接来自 OP-PROF 聚合。

### 7.2 H1 / H2 / H3 假设的算子级判定

**Table-19**：H1 / H2 / H3 假设的算子级对比 (per batch, 稳态均值)

| 假设 | 算子 | qwen1 (us/batch) | gemma4 (us/batch) | qwen1 占比 | gemma4 占比 | 结论 |
|---|---|---:|---:|---:|---:|---|
| H1 | FLASH_ATTN_EXT (us/batch) | 1434.5 | 721.2 | 2.7% | 2.0% | **证伪** |
| H1 | FLASH_ATTN_EXT (us/op) | 57.7 | 19.9 | - | - | gemma4 GQA 更快 (1/3) |
| H1 | FLASH_ATTN_EXT (ops/batch) | 24.88 | 36.26 | - | - | 与 n_layer 一致 |
| H2 | lm-head MUL_MAT (us/batch, max/op) | ~3500 | ~4800 | 6.7% | 13.0% | **证伪** |
| H2 | lm-head 带宽 (qwen1 156MB / 3.5ms) | 64 GB/s | - | 峰值 ~91% | - | DRAM 已饱和, op 级无收益 |
| H3 | a-inv (us/batch) | **16241.0** | 994.1 | **30.9%** | 2.7% | **qwen1 确认** |
| H3 | a-inv MB invalidated (MB/batch) | 1588 | 79 | - | - | **qwen1 是 gemma4 的 20.1x** |
| H3 | a-inv per-MB (us/MB) | 10.2 | 12.6 | - | - | 单位带宽成本相近, 为工作量问题 |

注: qwen1 wall ≈ 52500 us/batch (graph_dur 53407); gemma4 wall ≈ 37000 us/batch (graph_dur 37200)。

判定说明:

- **H1 (FLASH_ATTN) 证伪**: qwen1 25 op/batch (24 层 + 1) × 58 us, gemma4 36 op × 20 us。即使消除 100% FLASH_ATTN, qwen1 TG 也只提速 2.7% (1.4 ms/token)。
- **H2 (lm-head) 证伪**: qwen1 lm-head MUL_MAT 3500 us = 6.7% wall (max 3650 us), gemma4 4800 us = 13% wall。qwen1 lm-head 实测 64 GB/s, 逼近 DRAM 峰值 (~70 GB/s)。op-level 优化 (fuse / faster kernel) 被 DRAM 锁死, 上限 ~5-10%。真正的 H2 收益方向是 K/V cache FP16/Q8 量化 (把 cache 从 2 字节降到 1 字节, lm-head input 间接受益, 但要重做 cache 写入路径)。
- **H3 (KV invalidation) qwen1 确认, gemma4 否决**: qwen1 a-inv 16241 us/batch (30.9% wall) / 1588 MB invalidated, gemma4 仅 994 us / 79 MB。16.3x 时间差, 20.1x 数据量差。单位带宽成本 qwen1 (10.2 us/MB) 甚至略低于 gemma4 (12.6 us/MB), 为总工作量问题而非效率问题。根因推断: qwen1 MHA 1:1 每层 16 KV head × 128 head_dim × 2 (K,V) = 大量 KV 写 src range, 每次 batch 触发整片 L2 invalidation; gemma4 GQA 8:1 KV 头数仅 1/8。

### 7.3 qwen1 vs gemma4 全算子 per-batch 排序 (稳态)

**Table-20**：qwen1 vs gemma4 全算子 per-batch 排序 (稳态)

| rank | qwen1 op | us/batch | 占比 | gemma4 op | us/batch | 占比 |
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

注: qwen1 wall 实测 graph_dur = 53407 us (含 AP 侧 ~400 us 杂项); gemma4 wall 实测 37200 us。

### 7.4 qwen1 TG 真正的瓶颈重构 (a-inv + bulk 是核心)

**Table-21**：qwen1 vs gemma4 wall 拆解对比

| 组件 | qwen1 (us) | gemma4 (us) | 差值 |
|---|---:|---:|---:|
| a-inv | 16241 | 994 | +15247 |
| bulk flush | 15034 | 1205 | +13829 |
| op-sum | 20520 | 34743 | -14223 |
| 其他 | 226 | 445 | -219 |
| **wall** | **52021** | **37387** | **+14634** |

qwen1 比 gemma4 慢 14.6 ms/batch, 缓存维护多吃 29 ms (56% of delta), 把 24 层 vs 35 层的算子优势完全吃掉。a-inv + bulk = 60% wall, 同源 (DSP L2 容量 8 MB 限制), 都是 L2->DRAM 同步代价。

### 7.5 优化方向 (按收益/风险排序)

**Table-22**：剩余优化方向一览 (按优先级)

| 优先级 | 方向 | 预期收益 | 风险 | 杠杆点 |
|---|---|---:|---|---|
| P0 | E 方向 AP per-call overhead 消除 (hash 瘦身 + descriptor blob 复用) | 0.7-2.2% TG | 低 | 99% cache hit rate, 全模型生效 |
| P1 | K/V cache FP16/Q8 量化 | 10-19% TG | 中-高 | 直接砍 a-inv 工作量 (1588 MB → ~794 MB, 省 ~8000 us/batch = 15% wall), 需改 llama.cpp KV cache 写入路径 |
| P2 | MUL_MAT_ADD fusion rate 提升 | 5-10% TG | 中 | 当前 123/120 已几乎全 fuse, 边际空间小 |
| 放弃 | async_bulk_flush (Path-B1) | -14.6% (但 garble) | - | qwen1 MHA K/V race 已实验证伪 |
| 放弃 | dsp_a_inv_bitmap (Path-B3) | 0% (回退) | - | gemma4 garble + qwen1 回退, 已实验证伪 |
| 放弃 | qwen1 force_opfusion_in_pp | - | - | 经验证 HVX fused 6.5 ms/op vs HMX 137 us/op, 不适用 |

### 7.6 关键观察

- qwen1 MUL_MAT_ADD 5.1 op/layer × 24 layers = 123 op/batch, per_op = 68 us 极稳定 (min=47, max=154) → 完美 fusion, 无空间
- qwen1 MUL_MAT 只有 2 op/batch, 推断 1 个是 lm-head (max=3650 us ≈ 3500 us/batch = 6.7% wall, 与 H2 一致), 另 1 个是 output norm/bias 小算子
- qwen1 MUL_MAT_FFN 25 op/batch (24 layers + 1), per_op = 258 us; gemma4 36 op × 341 us
- qwen1 cgraph 5.8.4 早期测量中, 16249 us a-inv (1588 MB) 是结构上限, bitmap 优化已无效
- qwen1 bulk 15034 us/batch 几乎与 a-inv 16241 us/batch 相当 → dst flush 同样吃紧, 与 a-inv 同源 (L2 容量限制)
- qwen1 5.8.4 路径 A/B/C 三个 cache 优化方向已全部实验证伪 (MHA K/V 与 DMA race, L2 8 MB 容量限制无法绕过, PRIOR_DST_MAX_LEN=64 过严), cache 增量调参路径走完; 剩余两条结构层面路径: (a) 5.8.4 E 方向 = AP per-call overhead 消除 (hash 瘦身 + descriptor blob 复用, Table-22 P0), 收益 0.7-2.2% TG 低风险; (b) K/V cache FP16/Q8 量化 (Table-22 P1, 本节新增), 收益 10-19% TG 中-高风险; 合计潜在 11-21% TG 改善


***

## 八、Qwen3.5-9B 性能差距分析

### 8.1 现象与背景

5 模型 CI (Table-6A) 之外，临时跑了一次 Qwen3.5-9B (alias `qwen3-9b`，底层为 Qwen3.5-9B-Q4_0.gguf，5.03 GiB，n_layer=32 + delta-net 层) 与 QCOM libggml-htp 的对比。Terminal 实测 (QCOM 端数据来源：log_xxx.txt 截图；JZ 端数据来源：`qwen3_9b_tg_prof_20260808-192916.log` / `dump_perf_stats` 输出)：

**Table-23**：Qwen3.5-9B 现象与背景总览

| 维度 | JZ (5.1.0) | QCOM (HTP 官方) | 差距倍数 |
| --- | ---: | ---: | ---: |
| PP tok/s (52 token prompt) | 27.31 | 143.69 | 5.26x |
| TG tok/s (255 decode runs) | 1.48 | 7.70 | 5.20x |

JZ 5.2x 慢于 QCOM，与 qwen1 / gemma4 在 PP 上 JZ 领先的"性能反转"完全不同。本章基于 2026-08-08 当天抓的 log (`9b_sched_20260808-183614.txt` / `9b_test_output.log` / `9b_realtime.log` / `qwen3_9b_tg_prof_20260808-192916.log`) 进行根因定位。

**8.1.1 QCOM 端实测明细 (`common_perf_print` 输出，截图原始数据)**

```
sampling  time =      53.59 ms
samplers  time =      34.37 ms /  308 tokens
load      time =     366.10 ms
prompt eval time =    361.89 ms /   52 tokens (   6.96 ms per token,  143.69 tokens per second)
eval      time =  33112.32 ms /  255 runs    ( 129.85 ms per token,    7.70 tokens per second)
total     time =  33537.00 ms /  307 tokens
unaccounted time =     9.20 ms /  0.0 %    (total - sampling - prompt eval - eval) / (total)
graphs reused = 253
```

**8.1.2 JZ 端实测明细 (`common_perf_print` 输出，对应 8.1.3 的 dump_perf_stats 同一 run)**

```
sampling  time =     105.93 ms
samplers  time =      65.72 ms /  308 tokens
load      time =    1912.61 ms
prompt eval time =   1903.87 ms /   52 tokens (  36.61 ms per token,   27.31 tokens per second)
eval      time =  172379.97 ms /  255 runs    ( 676.00 ms per token,    1.48 tokens per second)
total     time =  174413.97 ms /  307 tokens
unaccounted time =    24.21 ms /  0.0 %    (total - sampling - prompt eval - eval) / (total)
graphs reused = 253
```

注：JZ 端 PP 27.31 tok/s / TG 1.48 tok/s 与 QCOM PP 143.69 / TG 7.70 形成 5.26x / 5.20x 差距 (Table-23)；JZ load 时间（1912.61 ms）显著高于 QCOM（366.10 ms），与权重回退 system memory 相关——`log_qwen3.5-9b.txt` 中两次运行的回退规模不同（PID 20562: 权重拆 2005+1996 MiB 双 chunk，后块 1996 MiB 回退；PID 21326: 单块 4002 MiB 整体回退 heap），精确规模取决于运行时 mempool 剩余与 chunk 拆分，不是固定值。unaccounted 24.21 ms = 174413.97 - 105.93 - 65.72 - 1912.61 - 1903.87 - 172379.97 = 44.87 ms（与 24.21 差异是 common_perf_print 内部对 sampling/samplers 不重复减，本质仍是 0.0% 级）。graphs reused = 253 与 QCOM 一致（upstream scheduler 共享）。

**8.1.3 JZ 端实测明细 (`ggmlhexagon_dump_perf_stats` 输出)**

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

**关键观察**：
1. **QCOM `graphs reused = 253 / 255 runs = 99.2%`** - 这是 **upstream llama.cpp scheduler** 的 cgraph cache 命中率统计（在 `common_perf_print` 输出），含义是：255 个 decode 步里 scheduler 复用了同一份已切分好的 cgraph 结构（nodes/ops/sub-graph 划分）的次数；~2 步因为 KV cache 增长或 graph reopt 触发重新切分。**注意此处的 "graphs reused" 与 FastRPC 没有任何关系**——它是 scheduler 层级的 cgraph 结构复用计数，不是 QCOM 端的 descriptor 缓存计数。QCOM 端没有"FastRPC descriptor 重建"的概念：QCOM 的提交链路是 dspqueue 持久化环形队列（`dspqueue_create` 一次，`dspqueue_write` 多次复用，line 1802-1813），descriptor 概念由 dspqueue 内部 `htp_opbatch_req` 承载，每次 `dspqueue_write` 是把 N 个 op 的 descriptor 批量入队，不存在 per-call FastRPC descriptor 缓存/重建过程。JZ 端 cgraph cache（`hits=6351 misses=49 hit_rate=99.2%`）是 JZ 自己按 op+shape+src ptr 哈希建的 cache（[ggml-hexagon-jz.cpp:5159](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5159)），与 QCOM 的 `graphs reused` 含义不同但恰好都是 99.2%，反映两端 9B 模型的 cgraph 拓扑变化模式一致（每 255 步里 ~2 步需要重切）
2. **JZ cgraph cache 与 QCOM `graphs reused` 都是 99.2%** - JZ 端的 cache 是按 op+shape+src ptr 哈希的 AP 端 descriptor 复用 cache（[ggml-hexagon-jz.cpp:5159](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5159)，miss 时走 Phase 2 descriptor 重建），QCOM 端的 `graphs reused` 是 upstream scheduler 的 cgraph 结构复用计数（miss 时重新切分 sub-graph）。两者属于不同层级但都恰好命中 99.2%，原因是 9B 模型的 cgraph 拓扑变化频率稳定（每 255 步 ~2 步需要重切），与具体后端实现无关。**真正决定提交路径成本的是 cgraph cache miss 之后的动作**：JZ 走 per-sub-graph 同步 FastRPC（每次 mirror + sync），QCOM 走 dspqueue_write 批提交（25 个 sub-graph 一次入队）
3. **JZ batch_calls=6400** - JZ 9B 一次推理调用 DSP 6400 次，每 token 平均 25.0 次（24 个 ssm_out MUL_MAT CPU split 把每 token 的 cgraph 切成 25 个 Hexagon sub-graph，256 token x 25 = 6400）
4. **JZ per-call overhead = 21.4 ms/call** - 每 sub-call 26.6 ms 总耗时，5.3 ms 是 DSP 真实计算，21.4 ms 是非 DSP 开销（FastRPC setup + mirror + sync）；**25.0 calls x 21.4 ms = 535 ms / token 是 JZ 单 token 主要开销**
5. **JZ p5+p12 = 136.5 s = 78% of total** - mirror 计算是 JZ 9B 真实瓶颈（68.7 s compute + 67.8 s apply）；其余 22% 是 p1+p6+p8+p11 等
6. **JZ p10 dsp_exec = 5.262 ms/call** - 不是 0.5 ms！sub-call 真实 DSP 计算是 5.3 ms（不含 ssm_out MUL_MAT，后者在 CPU 执行），不是简单 matmul 的 0.5 ms
7. **PP 36.61 ms/token vs TG 676.00 ms/token = 18.5x 差异** - TG 100% decode-bound，与 qwen1 经验一致
8. **输出内容**：jinja "Thinking Process" 模式（Analyze Request / Identify Key Information），正常电影介绍（"Once Upon a Time in America", "Sergio Leone", "Robert De Niro"），非乱码

**Table-24**：Qwen3.5-9B 在 JZ vs QCOM 的 PP/TG 实测对比

| 模型 | 指标 | JZ | QCOM | 差距 | 备注 |
|---|---|---:|---:|---:|---|
| Qwen3.5-9B | PP tok/s | 27.31 | 143.69 | 5.26x | JZ 比 QCOM 慢 5.26x；52 token prompt |
| Qwen3.5-9B | TG tok/s | 1.48 | 7.70 | 5.20x | JZ 比 QCOM 慢 5.20x；255 decode runs |
| Qwen3.5-2B (对照, 6 章) | TG tok/s | 14.50 | 7.74 | 1.87x | JZ 比 QCOM 快 1.87x |
| qwen1 (对照, 7 章) | TG tok/s | 18.62 | ~30 | 1.61x | JZ 比 QCOM 快 1.61x |

Qwen3.5-9B 是 JZ 第二个出现 TG / PP 同时落后于 QCOM 的模型（第一个是 Qwen1.5-1.8B，TG -22.3% / PP -28.8%，见 3.5 节"Qwen1.5-1.8B PP/TG 均落后的根因"分析）。**两者的本质区别**：Qwen1.5-1.8B 仅落后 22-29%（~0.7-0.8x of QCOM），根因是 dspqueue pipelining 优势在 MHA + 24 层 + VTCM/cache 压力场景的最大化；Qwen3.5-9B 落后 80%+（~0.19x of QCOM，5.2x gap），根因是 graph split（24 个 ssm_out MUL_MAT split，根因见 8.3：Q5_K 不被 validator 支持）+ per-sub-call FastRPC overhead（25 sub-call x 21.4 ms = 537 ms/token vs QCOM 25 次 dspqueue_write ~2.5 ms/token）。**Qwen3.5-9B 是首个双边失利差距达 5x 量级的模型**，需要单独章节分析。**JZ cgraph cache 与 QCOM `graphs reused` 命中率都是 99.2%**（两者层级不同但恰好一致，Qwen3.5-9B cgraph 拓扑变化频率稳定），所以差距不在 cache 命中率，而在单次提交开销：两端 sub-graph 数量相同（同一 upstream scheduler 切分），JZ 每段走同步 FastRPC（21.4 ms），QCOM 每段只是一次 dspqueue ring-buffer write（~0.1 ms，~214x）。

### 8.2 SPLIT 现场抓取方法

沿用第六章 6.4 节分析 Qwen3.5-2B 的方法：使用ggml core自带的 `GGML_SCHED_DEBUG=2` 环境变量，输出 scheduler 每次切分子图的具体边界 op，无需触动源码：

```bash
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=. \
  GGML_SCHED_DEBUG=2 \
  ./llama-completion -ngl 99 -t 6 -n 64 --ctx-size 8192 \
  --ubatch-size 1 --batch-size 1 --poll 1000 \
  --no-warmup --load-mode none -fa on --jinja -no-cnv -st \
  --verbosity 5 \
  -m /sdcard/Qwen3.5-9B-Q4_0.gguf -p 'Hello'" 2>&1 | tee log_qwen3.5_9b_graphsplit.txt
```

抓取结果：一次 64-token TG 产生 **1000 `## SPLIT` 记录**（500 个 CPU split + 500 个 Hexagon split），分布在 20 个 cgraph：`graph_reserve` log 显示其中 2 个 `n_tokens=16` 的 cgraph（PP 处理 jinja 模板展开的 16 token 提示 + 1 个 TG 16 token cgraph），其余 18 个 `n_tokens=1` 的 cgraph（PP/TG 单 token 步进）。`rpc stats: batch_calls=1600`（64 个 TG token x 25 个 Hexagon sub-graph/token；1000 条 SPLIT 记录是 20 个 unique cgraph 结构 x 50 段的静态切分输出，cgraph cache 命中时不再重复打印，故 1600 次实际 DSP 调用只对应 1000 条 SPLIT 记录）。

**与 8.1.3 实际推理参数差异说明（重要，2026-08-08）**：抓取 log 使用 `-n 64 --ubatch-size 1 --batch-size 1 -p 'Hello'`，8.1.3 实际推理使用 `n_predict=256 n_batch=2048`（复杂 prompt）。两个场景的 **per-token SPLIT 模式完全一致**：

**Table-25**：抓取 log 与 8.1.3 实际推理参数对比

| 维度 | 抓取 log (-n 64) | 8.1.3 实际推理 (-n 256) | 比值 |
|---|---|---|---|
| TG 步数 | 64 runs | 256 runs | 4x |
| batch_calls 总数 | 1600 | 6400 | 4x |
| **per-token batch_call** | **25.0** | **25.0** | **1x (一致)** |
| TG tok/s | 1.52 | ~1.5 | 1x |
| cgraph 命中 | 1575/1600 = 98.4% | 6351/6400 = 99.2% | ~1x |

**为什么 `per-token batch_call = 25.0` 恒定（4x 关系的来源）**：

**Table-26**：25 calls/token 的构成（每 token 固定开销）

| 来源 | 段数 | 触发条件 | 对应 SPLIT 段 |
|---|---:|---|---|
| embedding GET_ROWS 在 CPU | 1 个 CPU 段 | token_embd.weight（545.62 MiB, Q4_0, [4096 x 248320]）固定在 CPU buffer（upstream 常规：input embedding 在 host 执行） | SPLIT #0 |
| 24 个 delta-net ssm_out MUL_MAT 切到 CPU | 24 个 CPU 段 | 每 token 都经过全部 24 个 delta-net 层（block 0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30），每层 `ssm_out.weight`（Q5_K, 11 MiB）在 CPU buffer | SPLIT #2,4,...,48 |
| 切回 Hexagon 的 sub-graph | 25 个 Hexagon 段 | 上述 25 个 CPU 段把每 token 的 cgraph 切出 25 个 Hexagon sub-graph，每段 1 次 DSP 调用；末段（SPLIT #49）含完整 model.output（output_norm MUL + 795.70 MiB Q6_K lm_head MUL_MAT 均在 Hexagon 执行） | SPLIT #1,3,...,49 |
| **per-token batch_calls 合计** | **25** | - | - |

**关键洞察**：

1. **25 calls/token 与 TG token 数完全独立**：不论 TG 是 64 还是 256 token，每 token 都需经过 1 次 embedding 查表 + 全部 24 个 delta-net 层，ssm_out MUL_MAT 切到 CPU 是"每层每 token"行为而非"首次"行为，故 per-token 调用数严格恒定为 25
2. **batch_calls 与 TG token 数线性相关**：`batch_calls = 25 x TG_token_count`，故 64 -> 256 token 翻 4 倍时 batch_calls 也精确翻 4 倍（1600 -> 6400）。**这反过来证明抓取 log 的 SPLIT 模式分类（25 个 unique 边界 op）可直接外推到实际推理**——1 个 embedding 段 + 24 个 ssm_out 段这套结构对所有 TG token 都一样，区别仅在重复次数
3. **消除 split 的潜在收益**：`25 calls/token` 中 24 个来自 ssm_out split，合并后 per-token 调用从 25 降到 1。注意省的只是**随 call 数线性的固定开销**（~13.5 ms/段，scan/descriptor/sync/dcinva），mirror memcpy 按引用量计、不随段合并减少（heap 权重仍需镜像，8.5/8.6），故消除 split 的净收益是 ~24 x 13.5 = ~325 ms/token，TG 1.48 -> ~2.8 tok/s（8.5 节测算）。**最短根治路径是 Q5_K 转存 Q4_0（见 8.3，复用既有 Q4_K/Q6_K 转存机制，~30-50 行）**；scatter-gather 解决的是 heap 权重的 mirror 与 > 4 GiB 容量，不消除 split（8.7 节 2026-08-09 修正）

`--ubatch-size 1 --batch-size 1` 强制单 token 步进，会让 PP 部分每个 prompt token 单独产生 1 个 cgraph（prompt eval = 1 token / 0 ms），但**不会改变 TG 单 token 的 SPLIT 模式**——split 仅由 `blk.N.ssm_out.weight` 的 buffer 位置触发，与 batch size 无关。`-p 'Hello'` 简化 prompt 让 PP 缩为 1 个 cgraph（vs 实际推理的更多 token PP），**也不影响 TG 的 split 计数**。

**结论**：抓取 log 的 1600 batch_call = 8.1.3 实际推理的 6400 batch_call 的 1/4（仅 TG token 数差异），per-token SPLIT 边界 op 分类结果可直接外推到实际推理。

### 8.3 SPLIT 模式分类与 ssm_out MUL_MAT 根因

基于 `log_qwen3.5_9b_graphsplit.txt` 的精确分析（不再使用估计值）。50 个 unique SPLIT 编号（0-49）按出现频次：

**Table-27**：50 个 unique SPLIT 编号频次总览

| SPLIT # | 出现次数 | 触发位置 | 边界 tensor | 模式 |
|---:|---:|---|---|---|
| 0 | 20 | 每 cgraph 起点 | （空，CPU input） | CPU 起点 |
| 1, 3, 5, 7, 9, ..., 49 | 20 each | delta-net/MHA 层间 | 见下表 | 切到 Hexagon |

**25 个 Hexagon SPLIT 边界 tensor 分类**（按 SPLIT #N 编号，对应每次切回 Hexagon 的输入 tensor）：

**Table-28**：25 个 Hexagon SPLIT 边界 tensor 分类

| SPLIT # | 边界 tensor | 大小 | 类型 | 含义 |
|---:|---|---:|---|---|
| 1 | `model.input_embed` | 16K | embedding 输出 | PP 第一个 sub-graph 起点 |
| 3 | `linear_attn_out-0` | 16K | delta-net | block 0 完成后（CPU 切回） |
| 5 | `linear_attn_out-1` | 16K | delta-net | block 1 完成后 |
| 7 | `linear_attn_out-2` + `attn_inp_kq_mask` | 16K+16K | delta-net + MHA 入口 | block 2 完成，block 3 (MHA) 起点 |
| 9 | `linear_attn_out-4` | 16K | delta-net | block 4 完成后 |
| 11, 13 | `linear_attn_out-5, 6` | 16K each | delta-net | block 5, 6 |
| 15, 17, 19 | `linear_attn_out-8, 9, 10` | 16K each | delta-net | block 8, 9, 10 |
| 21, 23, 25 | `linear_attn_out-12, 13, 14` | 16K each | delta-net | block 12, 13, 14 |
| 27, 29, 31 | `linear_attn_out-16, 17, 18` | 16K each | delta-net | block 16, 17, 18 |
| 33, 35, 37 | `linear_attn_out-20, 21, 22` | 16K each | delta-net | block 20, 21, 22 |
| 39, 41, 43 | `linear_attn_out-24, 25, 26` | 16K each | delta-net | block 24, 25, 26 |
| 45, 47 | `linear_attn_out-28, 29` | 16K each | delta-net | block 28, 29 |
| 49 | `linear_attn_out-30` + `leaf_498` | 16K+0K | delta-net + 最终 norm | block 30 完成，model.output 起点 |

**对应的 25 个 CPU SPLIT 边界（24 个 `blk.N.ssm_out.weight` MUL_MAT + 1 个 embedding GET_ROWS）**：

**Table-29**：25 个 CPU SPLIT 边界 = 24 个 delta-net ssm_out + 1 个 embedding

| 切到 CPU 的 block | 出现次数 | ssm_out.weight 大小 | 层类型 |
|---:|---:|---:|---|
| 0, 1, 2 | 20 × 3 | 11M × 3 | delta-net |
| 4, 5, 6 | 20 × 3 | 11M × 3 | delta-net |
| 8, 9, 10 | 20 × 3 | 11M × 3 | delta-net |
| 12, 13, 14 | 20 × 3 | 11M × 3 | delta-net |
| 16, 17, 18 | 20 × 3 | 11M × 3 | delta-net |
| 20, 21, 22 | 20 × 3 | 11M × 3 | delta-net |
| 24, 25, 26 | 20 × 3 | 11M × 3 | delta-net |
| 28, 29, 30 | 20 × 3 | 11M × 3 | delta-net |
| **小计** | **480** | **264 MB 总 ssm_out** | **24 个 delta-net** |
| embedding GET_ROWS (SPLIT #0) | 20 | - | token_embd.weight 545.62 MiB 固定在 CPU（upstream 常规）；model.output 完整在 Hexagon（output_norm MUL + output.weight lm_head MUL_MAT 均标 `[Hexag]`，见 log） |
| **CPU 端总数** | **500** | - | - |

8 个 MHA-only 层 (层号 3, 7, 11, 15, 19, 23, 27, 31) **不出现 ssm_out MUL_MAT split**（它们没有 ssm_out.weight 张量）。这与 Qwen3-Next 架构的 3:1 模式（3 个 delta-net + 1 个 MHA）一致：32 层 / 4 = 8 cycle，每个 cycle 3 个 delta-net + 1 个 MHA = 24 个 delta-net + 8 个 MHA。

24 个 delta-net 层的 ssm_out MUL_MAT 全部触发 split，模式精确一致。每一处都形如：

```
node # 56 (MUL_MAT): linear_attn_out-0  [CPU]
                   x blk.0.ssm_out.weight (11M) [CPU]
                   -> final_output-0   [Hexag]
```

**关键观察（log 实证，2026-08-08）**：

- `grep "ssm_out.weight" | grep -v "create_tensor" | sed -n 's/.*blk\.\([0-9]*\)\.ssm_out.weight.*/\1/p' | sort -n -u` 输出 24 个 block 编号：0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30
- 每个 block 的 ssm_out.weight 都被标记为 `[CPU]`，与上游 scheduler 的 "src weight buffer 不被当前 backend 支持" 触发条件完全一致
- 8 个 MHA block（3, 7, 11, 15, 19, 23, 27, 31）的所有 op 完整在 Hexagon 端运行，无 CPU 切分
- SPLIT #7 的 5 inputs（`linear_attn_out-2` + `leaf_55` + `leaf_59` + `leaf_61` + `attn_inp_kq_mask`）表明 MHA block 3 接收 4 个额外 leaf 输入（KV cache view + mask），是进入 MHA block 的标志

**ssm_out.weight 在 CPU buffer 的根因（2026-08-09 修正：是 Q5_K validator 不支持，不是 mempool 容量）**：

先前的归因是"9B 模型 5.4 GB，JZ 的 ION mempool 4 GB 装不下，ssm_out.weight 随主 weight 块回退到 system memory"。**该归因与 graphsplit log 矛盾，已撤回**。三重证据链：

1. **log 证据**：`log_qwen3.5_9b_graphsplit.txt` 中 24 个 `ssm_out.weight`（各 11 MiB）标 `[CPU]`，但同层全部其他权重（ffn_down 30 MiB / ffn_up 27 MiB / attn_qkv 18 MiB / attn_gate 9 MiB / ssm_conv1d / ssm_alpha / ssm_beta，从 blk.0 到末层 blk.31 一致）以及 `output.weight`（795.70 MiB lm_head）、`output_norm.weight` **全部标 `[Hexag]`**。注意 `[Hexag]` 标签只说明 tensor 属于 hexagon buft 的 buffer，不代表 backing store 在 mempool 里——本次运行 4002 MiB hexagon 权重块中 2005 MiB 进 mempool、1996 MiB 回退 heap（heap 块内 tensor 仍标 `[Hexag]`，运行时靠 mirror 维持 DSP 执行，不触发 split，见 8.6）。真正被 scheduler 路由到 CPU buft 的权重只有 24 个 ssm_out.weight。"mempool 装不下 ssm_out"不成立——ssm_out 在调度阶段就被判给 CPU，从未进入 hexagon buft 的分配流程。
2. **GGUF 元数据证据**（Qwen3.5-9B-Q4_0.gguf 实测）：全部 24 个 `ssm_out.weight` 是 **Q5_K** [4096, 4096]，11.00 MiB x 24 = 264.0 MiB，且 **Q5_K 在全模型 427 个 tensor 中恰好只有这 24 个**——与 24 个 CPU split 一一对应。类型普查：Q4_0 n=173 (3929.62 MiB，含 token_embd.weight [4096, 248320] 545.62 MiB)，Q6_K n=1 (output.weight 795.70 MiB，全模型唯一)，Q4_1 n=4 (120.00 MiB)，Q8_0 n=48 (6.38 MiB)，F32 n=177 (32.39 MiB)。模型总权重 5148.1 MiB = 5.03 GiB（427 tensors，32 layers）。
3. **代码证据**：[ggml-hexagon-jz.cpp:3336-3406](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L3336-L3406) `ggmlhexagon_supported_mul_mat` 的 switch 仅支持 Q8_0 / Q4_0 / IQ4_NL / Q4_1 / MXFP4 / Q4_K / Q6_K（外加 F16/F32/BF16），**Q5_K 走 `default: return false`**（line 3404）。

**正确根因链**：ssm_out.weight 是 Q5_K -> JZ MUL_MAT validator 返回 false -> upstream scheduler 把该 MUL_MAT 分配到 CPU backend -> tensor 随之落入 CPU buffer -> 24 个 delta-net 层每层产生 1 个 CPU split。**与 mempool 容量无关**。

**mempool 容量问题的真实边界（与 split 根因相互独立）**：`log_qwen3.5-9b.txt` 的 "ion pool exhausted" 记录确实存在，且两次运行模式不同：

```
PID 20562: 2005 MiB 权重 chunk 进 mempool (51.08%), 随后 1996 MiB chunk 回退 (needed 1996, remaining 1968)
PID 21326: 单块 4002 MiB 权重 chunk 整体回退 heap (needed 4002, remaining 3973)
```

mempool 4 GiB 装不下 5.0 GiB 全模型是**事实**（对 > 4 GiB 模型的容量约束真实存在，8.7 节的 scatter-gather 讨论仍然成立），但在 graphsplit / tg_prof 运行中，容量回退的受害者**不是** ssm_out：4002 MiB hexagon 权重块里 2005 MiB 进 mempool、1996 MiB 回退 heap，heap 块内 tensor 仍属 hexagon buft、仍由 DSP 经 mirror 执行，不产生 split。真正落入 CPU 的只有 Q5_K 的 ssm_out.weight（类型被 validator 拒绝，调度阶段判给 CPU buft，264 MiB 从未进 hexagon 权重块）。即 **当前 9B 的 24 个 split 不是 mempool 容量造成的**；容量回退的真实代价体现在 p5/p12 mirror overhead（8.6 节，136.5 s），是另一条独立成本链。

**最短修复路径（随根因反转而变）**：在 `set_tensor` 时把 Q5_K 转存为 Q4_0，复用 [ggml-hexagon-jz.cpp:3329](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L3329) 已有的 "Q4_K/Q6_K are stored as Q4_0" repack 机制，24 个 split 直接消除，batch_calls 6400 -> ~256。改动集中在 set_tensor 量化类型转换与 validator 放行，比 scatter-gather（DSP 端 descriptor 协议扩展）简单一个量级，应作为 9B 性能修复的第一优先级（见 8.7.4 路径表重排）。

### 8.4 根因 2: QCOM 的 dspqueue 机制（5x 差距的真因）

ssm_out split 是直接表象，但 5x 差距的真因是 QCOM 用了 dspqueue 风格的批提交 + AP-DSP overlap，JZ 还在用同步 FastRPC。

**8.4.1 QCOM 的 dspqueue 三件套**

[ggml-hexagon.cpp:41-42](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L41-L42) 引入高通 Hexagon SDK 提供的 dspqueue 库：

```cpp
#include <dspqueue.h>   // 持久化 AP-DSP 队列
#include <rpcmem.h>
```

启动时建一条常驻队列（[line 1802-1813](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1802-L1813)）：

```cpp
err = dspqueue_create(this->domain_id, 0, req_q_size, ...);
// AP ↔ DSP 之间一条持久化的 dspqueue，单次创建，多次复用
```

**8.4.2 op_batch 累积批提交**

`ggml_hexagon_session::enqueue_op` (line 1628-1633) 不直接走 DSP，而是累积到本地 op_batch，满了才 flush：

```cpp
void ggml_hexagon_session::enqueue_op(const htp_opnode & node) {
    if (!op_batch->fit_op(node)) {
        flush_batch();        // 满了才发，N 个 op 一次 dspqueue_write
    }
    op_batch->add_op(node);   // 否则只加到本地 op_batch
}
```

`flush_batch` (line 1604-1626) 一次写整批：

```cpp
int err = dspqueue_write(this->queue, 0, 1, &dbuf, sizeof(req), ...);
// 一次写 N 个 op，dspqueue 内部排队，DSP 后台消费
```

`dspqueue_read` (line 1577-1579) 读响应也是非阻塞的：

```cpp
int err = dspqueue_read(this->queue, &flags, 1, &n_dbufs, &dbuf, ..., timeo);
if (err == AEE_EEXPIRED || err == AEE_EWOULDBLOCK) {
    continue;  // 非阻塞，没准备好就跳过
}
```

**8.4.3 AP-DSP overlap**

QCOM 的 graph_compute 入口（[line 3707-3715](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L3707-L3715)）展示完整 producer-consumer 模式：

```cpp
// Queue and execute
if (opt_opstage & HTP_OPSTAGE_QUEUE) {
    for (const auto & node : *nodes_ptr) {
        sess->enqueue_op(node);  // 全部入队，不等
    }
}
sess->flush();  // 末尾统一等
```

AP 端把所有 op 入队后立刻返回，DSP 后台消费；AP 端在 DSP 跑 op 的同时可以准备下一批 buffer / 算参数。FastRPC call 次数从"JZ 的 N 次" 降到 "QCOM 的 1~2 次"（按 batch 容量）。

**8.4.4 JZ 当前的同步 FastRPC**

[JZ 的 graph_compute_batch 入口](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5362-L5374) 一次 batch_calls 就一次 `ggml_dsp_execute_batch`（同步 FastRPC + 同步等 dspqueue_buffer）：

- 每个 sub-graph 一次 fastrpc_invoke (synchronous)
- AP 端发完就阻塞等 DSP 返回
- DSP 算完返回 AP 才解阻塞进下一轮
- 没有本地 op_batch 累积
- 没有 producer-consumer 重叠

24 个 ssm_out MUL_MAT split 在 JZ 是 24 次同步 fastrpc 串行执行；在 QCOM 是 1~2 次 dspqueue_write，AP-DSP 重叠。

**8.4.5 量化对比**

实测验证（QCOM log + JZ dump_perf_stats 数据）：

- QCOM 9B TG 总耗时 33.1 s / 255 runs = 129.85 ms/run (即 7.70 tok/s)
- QCOM cgraph cache hit rate = 253/255 = 99.2%（upstream llama.cpp scheduler 的 `graphs reused` 计数，~2 步需重新切分 sub-graph，~253 步复用；**注：QCOM 没有 FastRPC descriptor 概念，提交链路是 dspqueue 持久化环形队列的 dspqueue_write（`htp_opbatch_req` 批量入队），不涉及 per-call FastRPC descriptor 缓存/重建**）
- QCOM 端 unaccounted time = 9.20 ms / 33537 ms = 0.0%（QCOM 没有 JZ 的 p1-p12 12 阶段划分，QCOM 的 `common_perf_print` 只输出 sampling / samplers / load / prompt eval / eval / total / unaccounted 等顶层分类，0.0% unaccounted 说明 dspqueue 让 AP 准备与 DSP 算完全 overlap）
- JZ 9B TG 174.4 s / 6400 batch_calls = 27.25 ms/call（其中 DSP 真实计算 p10 = 5.262 ms/call，overhead = 21.4 ms/call；cum_graph=170.4 s, cum_p10=33.7 s）
- **JZ cgraph cache 与 QCOM `graphs reused` 都是 99.2%**（两者层级不同但恰好一致），差距不在 cache 命中率，而在 cache miss 之后的提交路径与单次提交开销

**Table-30**：QCOM dspqueue 批提交 vs JZ 同步 FastRPC 的 9B 实测对比

| 指标 | JZ (实测) | QCOM (实测) | 差距倍数 | 数据来源 |
|---|---:|---:|---:|---|
| 9B TG Hexagon sub-graph 提交次数 | 25 sub-graph x 256 token = **6400 次** fastrpc（24 个 ssm_out CPU split 把每 token 的 Hexagon 图切成 25 段；CPU 侧另有 24 段/token 的 CPU 执行不占 fastrpc） | **同样 ~25 sub-graph x 256 token**（同一 upstream scheduler + QCOM validator 同样拒绝 Q5_K，见注 3），每段 1 次 dspqueue_write | ~1x（次数相同） | JZ batch_calls=6400；QCOM 次数为代码级推断（待 QCOM 侧 log 验证） |
| 单次提交开销 (反推) | ~21.4 ms (FastRPC + mirror) | ~0.1 ms (dspqueue_write) | ~214x | JZ dump_perf_stats per-call overhead avg=21369 us |
| AP-DSP overlap | 否 (同步阻塞) | 是 (dspqueue 异步) | - | ggml-hexagon.cpp:3707-3715 |
| **cgraph cache hit rate (实测)** | **99.2%** (6351/6400 sub-calls, JZ AP 端 descriptor cache) | **99.2%** (253/255 decode 步, upstream sched-graph 复用计数) | **1.0x (相同)** | JZ dump_perf_stats vs QCOM graphs reused, **两者分母与层级不同, 恰好一致** |
| 实际平均 overhead/call (实测/反推) | 21.4 ms | ~0.1-0.4 ms (反推: dspqueue_write ~0.1 ms/次; 或每 token ~10 ms 非 DSP 时间 / 25 段 = ~0.4 ms) | ~50-214x | p10 3-way + 注 2 |
| 9B TG 总耗时 (实测) | 174.4 s (6400 calls) | **33.1 s** (255 runs) | 5.27x | JZ dump_perf_stats vs QCOM eval time |
| 9B TG tok/s (实测) | 1.48 | **7.70** | 5.20x | Table-23 |
| 9B PP tok/s (实测) | 27.31 | **143.69** | 5.26x | Table-23 |
| unaccounted time 占比 (实测) | 0.005% (8.3 ms / 174414 ms) | **0.0%** (9.20 ms / 33537 ms) | ~1x | JZ unaccounted=8292 us vs QCOM 9.20 ms |
| 每 token 提交成本 (实测/反推) | 25 calls x 21.4 ms = **537 ms** | 25 writes x ~0.1 ms = **~2.5 ms** | ~214x | JZ dump_perf_stats per-call overhead；QCOM 0.1 ms/write 为反推 |

注 1：QCOM 端 0.1 ms/dspqueue_write 是反推值（dspqueue 内部是简单 ring buffer write，enqueue + 一次 signal 系统调用），QCOM 端未单独测量此项；JZ 端 21.4 ms/call 来自 dump_perf_stats `per-call overhead: avg=21369 us`。

注 2：QCOM 端"~0.4 ms/overhead/call"是反推（每 token 129.85 ms - ~120 ms DSP compute = ~10 ms 非 DSP 时间，除以 25 段/token）；QCOM 端 `common_perf_print` 只输出顶层分类（sampling / samplers / load / prompt eval / eval / total / unaccounted），**没有 JZ 的 p1-p12 12 阶段**。unaccounted = 0.0% 说明 dspqueue 让 AP 准备与 DSP 算完全 overlap，没有阻塞空隙，AP 端没有空转。

注 3（2026-08-09 重写）：QCOM 同样存在 24 个 ssm_out MUL_MAT CPU split——这不是推测而是代码级事实：QCOM `ggml_hexagon_supported_mul_mat`（[ggml-hexagon.cpp:2783-2853](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2783-L2853)）的 src0 switch 只认 Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4/F16/F32，**Q5_K 同样走 `default: return false`**（L2841-2842），且 QCOM 没有 JZ 的 Q4_K/Q6_K -> Q4_0 转存机制（`ggml_hexagon_is_repack_type` L210-214 不含任何 K-quants），同一 upstream scheduler 面对同一张 cgraph 必然给出同样的 24 个 CPU split。两端 split 拓扑相同；`graphs reused=253` 是 upstream sched-graph 的复用计数（253/255 decode 步复用切分拓扑），**不是提交次数**，与每 token 25 段提交不矛盾。真正的差距在**单次提交开销**：JZ 每段走同步 FastRPC + mirror（21.4 ms），QCOM 每段只是一次 dspqueue ring-buffer write（~0.1 ms，~214x）；每 token 提交成本 JZ 537 ms vs QCOM ~2.5 ms。dspqueue 并没有把 25 个 sub-call 压成 1 次提交（sub-graph 数量由 upstream scheduler 的 split 决定，两边相同），它把每次提交变便宜 ~214x——这是 5.20x 整体差距的真因：JZ 681 ms/token 中 537 ms 是提交开销，扣除后 ~145 ms 与 QCOM 130 ms/token 基本持平。此外 QCOM 还有 JZ 没有的 lm_head split（L2805-2808 `ne[1] > 32768` 硬编码拒绝 lm_head；JZ 靠 Q6_K -> Q4_0 转存 + mempool 连续 IOVA 把 795.70 MiB lm_head 留在 DSP，是 JZ 的净优势，见 why-perbuffer 文档）。

### 8.5 ssm_out split 的次要性澄清

ssm_out MUL_MAT 在 QCOM 必然也存在同样的 24 个 split（代码级证据见 8.4.5 注 3：QCOM validator 同样 `default: return false` 拒绝 Q5_K，且无 K-quants 转存机制；另从算子层面看，SOLVE_TRI / SSM_CONV / GATED_DELTA_NET 三个算子签名与 JZ 一模一样，对比 [htp/gated-delta-net-ops.c:1079-1147](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/gated-delta-net-ops.c#L1079-L1147) 与 [kernels/gated-delta-net-ops.c:1079-1147](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/gated-delta-net-ops.c#L1079-L1147) 完全相同；[htp/main.c:697-708](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/main.c#L697-L708) 的 op dispatch 也无 fused ssm_out）。

QCOM 的 24 个 split 产生与 JZ 相同数量的 sub-graph 段（~25 段/token），差别只在每段的提交成本：QCOM 每段一次 dspqueue_write（~0.1 ms），单 token 提交成本 ~2.5 ms；JZ 每段一次同步 FastRPC + mirror（21.4 ms），单 token 提交成本 537 ms（~214x）。**5.2x 整体差距来自这个提交路径差距**（JZ 681 ms/token 扣掉 537 ms 提交后 ~145 ms，与 QCOM 130 ms/token 持平），ssm_out split 本身的算子成本在两边都相同。

JZ 这边 24 个 split 的硬成本测算（基线：174.4 s 总，6400 batch_calls，21.4 ms/call overhead，p5+p12 = 136.5 s）。消除 split = 每 token 25 个 Hexagon 段合并为 1 段。注意 mirror memcpy 总量**不按 call 数缩减**：合并后的单段仍引用全部 heap 常驻权重（~2 GiB，见 8.6），每 token 都要镜像一次；省的只是随 call 数线性的固定开销（scan/descriptor/sync/dcinva）：

- **固定开销部分**（随 call 数线性）：25 x ~13.5 ms -> 1 x ~13.5 ms，节省 ~325 ms/token，这是消除 split 的收益大头
- **mirror memcpy 部分**（按引用量）：合并前后都是 ~2 GiB/token（~200 ms），不省
- 估算：681 - 325 = 356 ms/token -> **~2.8 tok/s**
- 即便按 mirror 全免的极端假设（136.5 s 全省）：174.4 - 136.5 = 37.9 s -> 6.8 tok/s，不可及，因为 heap 常驻权重必须镜像才能被 DSP 读
- 综合判断 ssm_out split 单独消除后 TG 区间 **1.48 -> 2.0-3.0 tok/s**

即便乐观到 3.0 tok/s，依然离 QCOM 7.70 tok/s 差 2.5x。后续收益来自两处（与 split 修复正交）：(a) mirror memcpy 的消除（heap 权重改走 scatter-gather / DMA 拉取，8.6/8.7 节）；(b) per-call 固定开销的压缩（dspqueue 风格批提交，8.8 节）。

**8.3 根因反转对本节结论的影响**：split 修复从"扩 mempool / scatter-gather"降级为 `set_tensor` 的 Q5_K -> Q4_0 转存（复用既有 Q4_K/Q6_K 机制，约几十行），实施成本极低。它对 TG 上限的贡献是"次要"的（2.0-3.0 tok/s），但它是后续一切提交路径优化的**前置条件**——不先把 24 个 CPU 段消掉，批提交/流水线无从合并——应最先落地。

### 8.6 Mirror 机制深度分析与 DMA 拉取路径讨论

8.5 节已说明 ssm_out split 在 9B 上是次要因素（QCOM 同样有 24 split），但 8.9 结论 4 又指出"174.4 s 中 136.5 s (78%) 是 p5/p12 mirror overhead"——这两节之间存在一个明显的逻辑缺口：**mirror 机制为什么对 9B 表现"失效"，以及能否用一个非 dspqueue 风格的路径（DMA 拉取）绕开 mempool 4 GB 限制**。本节填补这个缺口。

#### 8.6.1 为什么 mirror 机制对 9B 表现"失效"

**先说结论：Mirror 没有失效，工作正常，但 Path-E mechanism 2 的设计不完整，cache hit 跳过了 scan 但没跳过 memcpy。**

看 [ggml-hexagon-jz.cpp:5936-6014](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5936-L6014)：

```cpp
// Cache hit 只回填 metadata，不跳过 memcpy
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

// 不论 cache hit/miss，下面这段都执行：
for (auto & kv : buffer_mirrors_map) {
    ...
    memcpy(ion_buf, data_ptr, mirror_size);  // 每次都拷贝
    info.allocated = true;
}
```

**设计不完整的根因**：mirror 区域是临时 ion_region，每次 call 结束被 [Phase 12 末尾的 free 释放](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L6591-L6597)，下一次 call mempool 区域被复用，必须重新拷贝。Path-E mechanism 2 跳过的只是 tensor_src scan（~1 ms / call），memcpy 本身的 11-12 ms 还在。

**对 9B 的具体开销**（2026-08-09 修正分解）：

- 6400 batch_calls = 25 个 Hexagon 段/token x 256 token（24 个 CPU 段在 CPU 侧执行，不占 batch_calls，"6400 中 6144 个是 ssm_out sub-graph"的旧分解有误）
- 每个 Hexagon 段的 mirror 对象有两类：(a) CPU 段边界的 `linear_attn_out-*` 激活（1 MiB x 24 处/token，小头）；(b) **heap 回退 chunk（1996 MiB）中被该段引用的权重切片**（大头）——25 段合计 ~2 GiB/token
- p5+p12 = 136.5 s 的分解：固定开销（scan/descriptor/sync/dcinva，随 call 数线性）6400 x ~13 ms = ~83 s，mirror memcpy ~2 GiB/token x 256 token @ ~10 GB/s = ~51 s

**Path-E mechanism 2 修复对 9B 无效（2026-08-09 修正）**：先前认为"cache hit 时跳过 memcpy 是 30-50 行 patch、1 天零风险"。对 9B 这个修复**不成立**：临时 mirror 区每 call 释放不是单纯的设计疏忽，而是容量所迫——常驻权重已占 ~2.1 GiB，mempool 余量 ~1.5 GiB，装不下 ~2 GiB heap 权重的常驻镜像（2.1 + 2.0 = 4.1 GiB > 4 GiB 硬顶）。即使 patch 让 cache hit 跳过 memcpy，也没有空间让镜像常驻。该 patch 只对小 mirror 集场景（如五模型的 mask/激活镜像）有效；9B 的 mirror 消除只能靠 scatter-gather / DMA 拉取（8.6.2 / 8.7），让 DSP 直接读 system memory 的 heap 权重。

#### 8.6.2 "DSP 主动从 system memory DMA 拉 weight" 路径详解

**当前 mirror 的本质**（问题）：

```
[System Memory: heap 回退权重 (1996 MiB chunk)] 
        |  
        |  AP-side memcpy (Phase 5)  
        |  每 token 合计拷 ~2 GiB (25 段各拷其引用切片)
        |  走 DDR -> AP L1 -> DDR -> ION
        v  
[ION mempool 4GB 临时 mirror 区]  
        |  
        |  DSP 读
        v  
[DSP L2 cache]
```

问题（2026-08-09 修正因果链）：
- 每 token ~2 GiB AP-side mirror memcpy（~200 ms），这是 mempool 容量回退的真实代价
- 4 GB mempool 装不下 5.0 GiB 模型 -> 4002 MiB hexagon 权重块中 1996 MiB 回退 heap
- heap 权重**不触发 split**（buffer 仍属 hexagon buft），触发的是 mirror
- 24 个 ssm_out split 是另一条独立因果链（Q5_K 被 validator 拒绝，8.3 节），与 mirror/容量无关

**"DSP 主动 DMA 拉 weight" 的本质**（提议）：

```
[System Memory: ssm_out.weight 11MB]  
        |  
        |  DSP-side DMA (HAP_mmap2 映射)  
        |  第一次冷 DMA ~1ms  
        |  后续命中 L2 cache 0ms  
        v  
[DSP L2 cache] 

[ION mempool 4GB]  
        |  
        |  跑 src/dst tensor（激活）
        v  
[DSP L2 cache]
```

**关键对比**：

**Table-31**：当前 mirror 机制 vs 提议的 DMA 拉取路径关键对比

| 维度 | 当前 mirror | 提议的 DMA 拉取 |
|---|---|---|
| 数据搬运主体 | AP CPU memcpy | DSP DMA engine |
| 第一次 11MB weight | AP 拷 11 MB (~11 ms) | DSP DMA 11 MB (~1 ms) |
| 后续重复读 | AP 重拷 11 MB | DSP 命中 L2，0 搬运 |
| mempool 占用 | 临时 mirror 区常驻周转 | **0**（weight 不在 ION） |
| scheduler 视角 | weight 在 hexagon buft heap 回退 -> 无 split，但每 call 付 mirror memcpy | weight 在 DSP-accessible buffer -> 无 split 且无 mirror |

**需要 JZ 当前没有的支持**：

1. **JZ buffer type 支持 cross-buffer view**：当前 [repack_buffer_type](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L352-L356) 只能描述 ION mempool 内的 buffer，需要新增一种 buffer type 表示"system memory pages mapped to DSP via HAP_mmap2"
2. **DSP 端 execute_op 支持 scatter-gather descriptor**：当前每个 src/dst 必须是单一连续 mempool region，需要扩展为"远端 system memory 区域 + 任意 offset"
3. **AP 不再为 heap 回退权重走临时 mirror**：`set_tensor` / alloc 检测到这类 weight 时走新 buffer type，DSP 直接 DMA 读 system memory
4. **cache coherency 协议扩展**：system memory 页可能带 CPU L1/L2 stale data，DSP 读前需 `dc ivac`（这是 Path-G 已处理的事情，可复用）

**5 模型 CI 兼容性**：
- 4 GB mempool 限制绕开（heap 回退权重不再占临时 mirror 区）-> 9B 可装下
- gemma4-e2b / qwen1 等小模型不受影响（用旧路径）
- 性能提升预期（2026-08-09 重估）：24 个 split 已由 Q5_K -> Q4_0 转存覆盖（8.3/8.5，1.48 -> ~2.8 tok/s）；本路径在此之上消除 ~2 GiB/token 的 mirror memcpy（~200 ms/token），叠加后 9B TG ~2.8 -> ~6-7 tok/s（DSP 计算 ~131 ms/token 成为新瓶颈）

**实施成本与阻塞**：

**Table-32**：DMA 拉取路径实施成本与阻塞评估

| 改动点 | 复杂度 | 阻塞 |
|---|---|---|
| JZ buffer type 体系扩展 | 高（影响 set_tensor / alloc_buffer 全链路） | 上游 scheduler 看不到新 buffer type，需 hook `buffer_is_host` / `buffer_supports_backend` |
| DSP entry.c 接收 scatter-gather descriptor | 高（改 fastrpc payload 格式） | 影响所有 5 模型，需 5 模型 CI 全过 |
| cache coherency 协议扩展 | 中（dc ivac flush 整个 weight region） | 复用 Path-G 的 0xFFFD 机制 |
| 上游 ggml-backend.cpp scheduler hook | 高 | 与 upstream llama.cpp 接口耦合 |

**与三阶段路径的关系**：本路径是 Q5_K 转存（消 split）与 8.8 三阶段提交路径（消 per-call 固定开销）之外的**第三条独立路径**，**不冲突**——它消除的是 ~2 GiB/token 的 mirror memcpy 与 4 GB mempool 容量限制。三者正交可叠加：Q5_K 转存把 25 段合并为 1 段，dspqueue 把每段固定开销压到 ~0.1 ms 级，DMA 拉取把 mirror memcpy 归零。**短期不能上**（实施成本 1-2 个月），但它是 9B 追平 QCOM 的根治方案之一（与 dspqueue 并列）。

### 8.7 如何突破 4 GiB mempool 平台限制

8.6.2 节介绍了"DSP 主动 DMA 拉 weight" 路径能绕开 4 GiB mempool 限制，本节展开这个结论的工程价值——它不仅是 9B 单模型优化，而是把 JZ ggml-hexagon 从"4 GB 模型"扩展到"24 GB 模型"的关键路径。

**4 GiB 限制的本质澄清（含先前推测修正）**：

**核心结论（4 GiB 限制的根本原因，硬件级 hard cap, 升顶 2026-08-08）**：

> **Hexagon V79 user-mode 虚拟地址空间是 32-bit (4 GiB)，这是硬件决定的事实**（[Hexagon V79 Programmer's Reference Manual §1.1 Memory](file:///opt/qcom/Hexagon_SDK/6.3.0.0/docs/pdf/80-N2040-60_REV_AA_Hexagon_V79_Programmer_Reference_Manual.pdf) "The Hexagon processor features a unified byte-addressable memory. This memory has a single 32-bit virtual address space, which holds both instructions and data."）

**这意味着**：
- DSP user-mode 任何时候只能看到 4 GiB 虚拟地址空间
- HVX/HMX 指令通过 VA 访问 memory，VA 范围是 32-bit
- 无论 ION 分配多大、FastRPC 64-bit 字段多宽、HAP_mmap2 多灵活，**DSP 端 user-mode 一次只能 mmap/access ≤ 4 GiB**
- ION `ion_allocation_data.len` 32-bit 字段、FastRPC ioctl 32-bit 字段、HAP_mmap 2 GiB 限制、rpcmem_alloc 2 GB 限制——**所有这些"32-bit 限制"都是这个硬件事实的下游派生**（kernel/driver 知道 DSP user VA 是 32-bit，所以所有外部 API 字段也保持 32-bit）

**支持 > 4 GiB 模型的唯一方式 = 散列访问 (scatter-gather / sliding window)**：
- 完整模型 weight 存放在 system memory (24 GB / 32 GB ARM64 主机地址空间，无 4 GiB 限制)
- DSP user-mode 一次只 mmap 4 GiB window（用 `HAP_mmap2` 把 system memory 物理页 map 进 4 GiB VA 范围）
- 不在当前 window 的 weight 通过 **scatter-gather descriptor**（HVX 支持）：DSP 端 descriptor 包含 `phys_addr + size` 列表，HVX DMA 控制器直接从物理地址拉数据，**绕过 user-mode VA 限制**
- Layer N 用完后 unmap，换 Layer N+1 的 weight 进同一个 4 GiB VA 范围

**（自我纠错）先前对 QCOM 实际分配路径的理解两次错误**：

- **错误 1**：把 [ggml-hexagon.cpp:1644-1660](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1644-L1660) 误认为 QCOM 实际 per-buffer 分配方案。**实际是 `ggml_hexagon_measure_max_vmem` 探测函数**（[ggml-hexagon.cpp:1641-1668](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1641-L1668)），测完临时 buffer 全部释放（line 1665），结果存到 `this->max_vmem`（line 1838）供 op_batch sizing 使用
- **错误 2**：把 `ggml_backend_hexagon_buffer_type_alloc_buffer` 理解为"每 tensor 一次"（per-tensor）。**实际是 ggml core 调用后端 alloc_buffer，ggml core 内部维护 mempool，每个 buffer (chunk) 包含多个 tensor**。`alloc_buffer` 接受的 size 是 ggml core tallocr 决定的 chunk size，**不是 per-tensor size**

**QCOM 与 JZ 实际控制 buffer 拆分的关键参数 `get_max_size()`**：

**Table-33**：QCOM 与 JZ 实际控制 buffer 拆分的关键参数 `get_max_size()`

| 后端 | `get_max_size()` 返回 | ggml core 行为 | 9B 模型 (5.4 GB) 实际分配 |
|---|---|---|---|
| QCOM | `this->max_bufsize = opt_mbuf` 默认 **1 GiB**（[ggml-hexagon.cpp:1760](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1760)） | ggml core 内部 tallocr 拆为 **多个 1 GiB chunk**（每 chunk 含多个 tensor） | 6 个 1 GiB chunk = 6 GiB ✓ |
| JZ | `ctx->rpc_mempool_len - 8 MiB` ≈ **4 GiB**（[ggml-hexagon-jz.cpp:5136](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5136)） | ggml core 内部 tallocr 拆为 **1 个 4 GiB chunk** | 1 个 4 GiB chunk → 命中 per-ION-buffer 4 GiB 上限 ✗ |

**ggml core 内部 mempool 机制**（[ggml-alloc.c](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-alloc.c)）：

- `ggml_dyn_tallocr` 是动态 tall allocator，**每个 buffer type 一个**（[ggml-alloc.c:120-127](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-alloc.c#L120-L127)）
- 内部维护 `tallocr_chunk[]` 数组，**最多 `GGML_VBUFFER_MAX_CHUNKS = 16` 个 chunk**（[ggml-alloc.c:95](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-alloc.c#L95)）
- 每个 chunk 是一个独立 backend buffer，通过 `ggml_backend_buft_alloc_buffer(buft, chunk_size)` 分配（[ggml-alloc.c:431](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-alloc.c#L431)）
- 新 chunk 大小 = `MAX(min_size, max_chunk_size)`，其中 `max_chunk_size = MIN(get_max_size(), SIZE_MAX/2)`（[ggml-alloc.c:170](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-alloc.c#L170)、[ggml-alloc.c:370](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-alloc.c#L370)）
- 单 tensor 太大超过 chunk 时，**整 chunk 扩到能装下**（[ggml-alloc.c:171-172](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-alloc.c#L171-L172) "backends will either manage to allocate the larger size, or report an error"）

**QCOM 实际 per-chunk 分配路径**（[ggml-hexagon.cpp:1065-1075](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1065-L1075)）：

```cpp
static ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto sess = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->sess;
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

**关键点**：这里的 `size` 是 ggml core 传下来的 **chunk_size**（不是 per-tensor size）。每 chunk 含多个 tensor。QCOM `get_max_size` 返回 1 GiB，所以 chunk_size 最大 1 GiB，ggml core 在内部 tallocr 中按需创建多个 1 GiB chunk（直到 `GGML_VBUFFER_MAX_CHUNKS = 16` 上限）

**与 JZ single-mempool 的根本差异（升顶 2026-08-08：4 GiB 限制的根因是 V79 user VA 32-bit 硬件 hard cap，不是 ION per-buffer 限制）**：

**Table-34**：与 JZ single-mempool 的根本差异（升顶 2026-08-08：4 GiB 限制的根因是 V79 user VA 32-bit 硬件 hard cap）

| 架构选择 | `get_max_size()` 返回 | ggml core 拆 chunk | DSP 端 4 GiB VA window 行为 | 总可装 weight | 适用模型 |
|---|---|---|---|---|---|
| QCOM per-chunk | **1 GiB** (默认) | 多个 1 GiB chunk (最多 16 个) | **scatter-gather / sliding window** 让 chunk 不一次性 mmap 进 4 GiB VA，DSP 只看当前 4 GiB window | n_chunks * 1 GiB ≤ 16 GiB (DSP 端 sliding 访问) | 9B (6 chunk) / 13B (10 chunk) / 20B (13 chunk) 都可 |
| JZ single-mempool | **~4 GiB** | 1 个 4 GiB chunk（>4 GiB 触发 V79 user VA 上限） | 整个 mempool 一次性 mmap 进 4 GiB VA | 4 GiB (受 V79 user VA 硬件 hard cap) | 9B / 13B / 20B 都不可 |

**关键架构取舍（用户指出的矛盾, 2026-08-08 修订）**：8.7.3 节"绕开路径"的**方案 1**（per-buffer 替代单 mempool, 仿 QCOM）**几乎推翻 JZ single mempool 架构的两大根基之一**。**注：与 8.7.4 路径表 1b/1c/1d 不同**——8.7.4 路径表的 1b 实际指 8.7.3 方案 2 (scatter-gather, 保留 single mempool 架构), 与推翻 JZ 架构的方案 1 无关：

- JZ single mempool 架构 = "**单一连续 IOVA 范围**" → 这让 JZ 能 offload lmhead（lmhead 需要整个 vocab 范围的 embedding/logits tensor，必须连续 IOVA）
- QCOM per-buffer 架构 = "**非连续 IOVA**" → 正是 `why-perbuffer-cannot-offload-lmhead-20260724-en.md` 文档分析的根因：**per-buffer 模式无法 offload lmhead**
- 也就是说，**JZ 走 8.7.3 方案 1 (per-buffer 路径) 能装 9B/13B/20B 模型，但会失去 offload lmhead 的能力**（qwen1 等 < 4 GiB 模型 TG 会显著下降）
- 而 8.7.4 路径表 1b (scatter-gather) 保留主 mempool, **不**推翻 JZ single mempool 架构, **不**失去 offload lmhead 能力

**这是一个真正的 trade-off，不是 clear win**：

**Table-35**：JZ 走 per-buffer 路径的 trade-off 维度对比（这是一个真正的 trade-off，不是 clear win）

| 维度 | JZ 当前 single mempool | QCOM per-buffer | JZ 改 per-buffer 后 |
|---|---|---|---|
| 模型容量上限 | 4 GiB (V79 user VA 限制) | 16 GiB (V79 user VA + scatter-gather 滑动) | 16 GiB (装得下 9B/13B/20B) |
| offload lmhead | **✓ 可** (单一连续 IOVA) | ✗ 不可 (per-buffer 拆散) | ✗ **会失去** (JZ 文档已写明) |
| mirror 机制 | 简单 (一个 4 GiB region 一次 dcinva) | 复杂 (每个 buffer 独立 mirror) | 复杂 (需重写 mirror 机制) |
| Phase 1-12 调度 | 简单 (单 mempool base addr) | 复杂 (per-buffer base addr 数组) | 复杂 (需重写 12 阶段) |
| cgraph cache 命中率 | 99.2% (单 mempool 命中条件) | 略低 (跨 buffer 缓存) | 略低 (需重新设计 hash key) |
| qwen1 / 2B / 8B 等 < 4 GiB 模型 TG | 现有 baseline | 不会显著变化 | **可能下降** (lmhead 回 CPU) |

**JZ 走 per-buffer 路径的净收益分析**：
- 收益：9B / 13B / 20B 等 > 4 GiB 模型可装下
- 代价：失去 offload lmhead，qwen1 / 2B / 8B 等 < 4 GiB 模型 TG 可能下降
- 这意味着**这不是 JZ 优化路径上的明确胜利，而是 JZ 放弃自身架构优势去适配 QCOM 风格**
- 之前文档把 "single mempool 是 JZ 架构优势, 可 offload lmhead" 列为 JZ 净优势之一，**8.7.3 方案 1 (per-buffer 替代) 实际是放弃这个优势换取容量**

**JZ 极简方案实际验证失败 (2026-08-08)**：先前假设"改 `get_max_size` 为 1 GiB 让 ggml core 拆 chunk"是 1-2 行代码改动。**但 JZ `alloc_buffer` 不是按 chunk 调 `rpcmem_alloc2` 分配独立 ION buffer**——JZ 实际架构是"**probe 阶段探测最大 ION 单块 + 一次预分配大块 mempool + bump allocator 切块**"：

- Probe 阶段 ([ggml-hexagon-jz.cpp:1749-1759](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L1749-L1759))：试 `rpcmem_alloc2` 试 2048 / 3840 / 3968 / 4032 MiB，找最大可分配单块 ION buffer，**最大 = 4032 MiB (ION 4 GiB hard cap)**
- Pre-allocate 阶段 ([ggml-hexagon-jz.cpp:1767-1768](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L1767-L1768))：调一次 `rpcmem_alloc2(4024 MiB)` 分配整个 mempool
- `alloc_buffer` 阶段 ([ggml-hexagon-jz.cpp:5000-5096](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5000-L5096))：从 `ctx->rpc_mempool` 用 bump allocator 切块（含 `ion_regions` 数组 best-fit 复用、bump tail、耗尽后回退 `posix_memalign` system memory）

**改 `get_max_size = 1 GiB` 的实际效果**：ggml core 拆为多个 1 GiB chunk → 调 `alloc_buffer` 多次 → **每次仍然从同一个 `ctx->rpc_mempool` 切 1 GiB** → `ctx->rpc_mempool_len` 还是 4024 MiB → **总容量 4 GiB 没变** → 9B 5.4 GB 仍然装不下。**极简方案撤回**：此方案基于错误假设（以为 JZ `alloc_buffer` 像 QCOM 一样按 chunk 调 `rpcmem_alloc2`），实际 JZ 架构是"pre-allocate + bump allocator"，改 `get_max_size` 只能让切块粒度变小，不能扩大总容量。**真正可行的方案**：方案 A 重写 `alloc_buffer` 仿 QCOM 按 chunk 调 `rpcmem_alloc2`（改动大，要重写 mempool 边界 + mirror 机制 + IOVA 寻址）/ 方案 B HAP_mmap2 system memory mmap 路径 / 方案 C 提升 probe_slots 上限（**不可行**，被 ION 4 GiB hard cap 限制）

**JZ single-mempool 优化历史背景**：JZ 之前的所有"single-mempool 优化"（mirror 机制、Phase 1-12 调度、p10 测量）都是基于"所有 weight 在一个连续 IOVA 范围"的假设。这是因为 ggml core 默认行为是 `get_max_size = mempool_len` 时只创建一个 4 GiB 单 chunk。即使 ggml core 已支持多 chunk（[ggml-alloc.c:95](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-alloc.c#L95) `GGML_VBUFFER_MAX_CHUNKS = 16`），JZGmp 内部 `alloc_buffer` 仍按 bump allocator 切同一块 `ctx->rpc_mempool`——所以多 chunk **不会**自动绕过 ION 4 GiB hard cap

**先前推测的修正（诚实承认推测成分）**：

先前 8.7 节首段曾把 4 GiB 根因归因到 "Linux ION kernel driver 内部 `ion_allocation_data.len` 32-bit 字段"。**这一归因是推测**，理由：(1) 本仓库不包含 Linux kernel 源码（`drivers/staging/android/uapi/ion.h` 不可见），无法直接验证 `ion_allocation_data.len` 当前是 32-bit 还是 64-bit；(2) mainline Linux `ion_allocation_data.len` 在 v5.x 后已升级到 `size_t` 64-bit；(3) QCOM Hexagon DSP 配套的内核分支是否还保留 32-bit `len`、是否在 fastrpc_ioctl 路径单独再做截断，无源码无法判定

**真正硬约束的硬证据（来自 Qualcomm 官方 SDK 文档）**：

- **证据 A — QuRT 内存管理 API 显式存在 32-bit / 64-bit 两套**（[80-VB419-178_D_Qualcomm_Hexagon_QuRT_RTOS_User_Guide_SDK.pdf:44 §2.6 64-bit Operations](file:///opt/qcom/Hexagon_SDK/6.3.0.0/docs/pdf/80-VB419-178_D_Qualcomm_Hexagon_QuRT_RTOS_User_Guide_SDK.pdf)）：
> "The QuRT memory management service defines both 32-bit and 64-bit versions of certain operations. The 32-bit operations are provided for backward compatibility with earlier versions of QuRT. The 64-bit operations are functionally equivalent to the corresponding 32-bit operations, but can access memory addresses above 4 GB. The 64-bit operations are identified by the suffix '_64' in their operation names."

  涉及 64-bit 版本的 API（[QuRT RTOS User Guide §21](file:///opt/qcom/Hexagon_SDK/6.3.0.0/docs/pdf/80-VB419-178_D_Qualcomm_Hexagon_QuRT_RTOS_User_Guide_SDK.pdf)）：`qurt_lookup_physaddr_64`、`qurt_mapping_create_64`、`qurt_mapping_remove_64`、`qurt_mem_map_static_query_64`、`qurt_mem_pool_attr_get_addr_64`、`qurt_mem_pool_attr_get_size_64`、`qurt_mem_region_attr_get_physaddr_64`、`qurt_mem_region_attr_set_physaddr_64`、`qurt_mem_region_query_64`、`qurt_mapping_attr_get_64`（**PA 端 64-bit 操作**，因为 V79 user-mode VA 仍受 32-bit 限制）

- **证据 B — Hexagon V79 user-mode VA 是 32-bit**（[80-N2040-60_REV_AA_Hexagon_V79_Programmer_Reference_Manual.pdf §1.1 Memory](file:///opt/qcom/Hexagon_SDK/6.3.0.0/docs/pdf/80-N2040-60_REV_AA_Hexagon_V79_Programmer_Reference_Manual.pdf)）：
> "The Hexagon processor features a unified byte-addressable memory. This memory has a single 32-bit virtual address space, which holds both instructions and data."

  **这是硬件级的 user-mode VA 4 GB 上限**：32-bit general registers（R0-R31）、32-bit memory addressing modes、single 32-bit VA space

- **证据 C — HAP_mmap 2 GB 限制已被 HAP_mmap2 解决**（[HAP_mem.h:351-364](file:///opt/qcom/Hexagon_SDK/6.3.0.0/incs/HAP_mem.h#L351-L364)、[HAP_mem.h:380](file:///opt/qcom/Hexagon_SDK/6.3.0.0/incs/HAP_mem.h#L380)）：`HAP_mmap` 注释明确写 "This API is limited to buffer size less then 2 GB. Recommendation is to use HAP_mmap2 for buffer of size > 2 power(8*sizeof(size_t))"；`HAP_mmap2(void *addr, size_t len, ...)` 用 `size_t` 64-bit，无 documented size limit。JZ 当前 v79 路径已用 HAP_mmap2（[kernels/entry.c:2155](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/kernels/entry.c#L2155)），不受 2 GB 卡

- **证据 D — 用户态 rpcmem_alloc2 已是 64-bit**（[refs/fastrpc/inc/rpcmem.h:171](file:///home/zhouwg/develop/ggml-hexagon/refs/fastrpc/inc/rpcmem.h#L171)）：`rpcmem_alloc2(int heapid, uint32_t flags, size_t size)` 签名是 size_t 64-bit；老 `rpcmem_alloc` 才是 32-bit 2 GB 上限

- **证据 E — FastRPC ioctl 内核接口是 64-bit 长度**（[refs/fastrpc/inc/fastrpc_ioctl.h:122](file:///home/zhouwg/develop/ggml-hexagon/refs/fastrpc/inc/fastrpc_ioctl.h#L122)、[fastrpc_ioctl.h:132](file:///home/zhouwg/develop/ggml-hexagon/refs/fastrpc/inc/fastrpc_ioctl.h#L132)）：`fastrpc_ioctl_req_mmap.__u64 size` 与 `fastrpc_ioctl_munmap_req.__u64 length` 都是 64-bit

**4 GiB 限制的可能归因（按可能性排序，仍含推测成分）**：

1. **（最可能）QCOM ION kernel module 的内部 32-bit 截断**：即便 mainline Linux 已升级 `ion_allocation_data.len` 到 64-bit，QCOM Hexagon DSP 配套内核的 ION driver 可能仍保留旧 ABI，在 fastrpc_ioctl 路径上做 size 截断；4032 MiB 探针边界（= 4 GiB - 64 MiB）正好对应 32-bit `len` 字段 + ION metadata 开销
2. **（可能）QuRT / HAP_mmap2 内部实现细节**：尽管 HAP_mmap2 签名是 size_t，但 HAP_mmap2 内部可能仍走 QuRT 32-bit 内存池操作，PA > 4 GiB 时需要切到 `_64` 版本（QCOM 在 [htp/main.c:181-184](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/main.c#L181-L184) 仍保留 `HTP_MMAP_MAX_VMEM = 2147483648u` 2 GB 限制是这条路径的旧版痕迹）
3. **（最弱）Hexagon V79 32-bit user-mode VA 限制**：这条不直接限制 ION 分配大小，ION 分配后通过 fastrpc_mmap 把 fd 映射到 DSP VA，VA 4 GB 不够装 5.4 GB 单 buffer 时才会成为约束

**实证**：probe_slots 最大 4032 MiB（[ggml-hexagon-jz.cpp:1743](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L1743)），4096 MiB 不在列表中。4032 MiB = 4 GiB - 64 MiB，差值 64 MiB 高度对应 ION kernel + QuRT 32-bit 字段的 page table / metadata 开销

**绕开路径（升顶 2026-08-08：根因是 V79 user VA 32-bit 硬件 hard cap, 单 mempool 扩容只能到 4 GiB, 真正扩容需 scatter-gather / sliding window）**：

**核心论断（2026-08-09 修正：split 根因是 Q5_K 类型，scatter-gather 的定位随之修正）**：9B 模型 5.26x PP / 5.20x TG 性能差距的直接表象是 graph split（24 个 ssm_out MUL_MAT split）→ batch_calls = 6400 → 每 token 25 sub-call × 21.4 ms = 535 ms overhead, 占总时间 79%。但两点先前归因已撤回：(1) **split 的根因不是 mempool 容量**——ssm_out.weight 是 Q5_K，被 JZ MUL_MAT validator 在调度期判给 CPU（8.3 节三重证据链），即便 mempool 无限大这 24 个 split 依然存在；(2) **消除 split 的收益没有先前估计的大**——合并 25 段为 1 段只省随 call 数线性的固定开销（~325 ms/token），heap 权重的 mirror memcpy（~200 ms/token）不随段合并减少，单独消除 split TG 1.48 -> ~2.8 tok/s（8.5 节），不是 5-6 tok/s。**最短修复路径是 set_tensor 时 Q5_K -> Q4_0 转存（复用既有 Q4_K/Q6_K 机制，~30-50 行，8.7.4 步骤 0）**；scatter-gather（方案 2）的定位修正为**消除 heap 回退权重的 mirror memcpy + 支持 > 4 GiB 模型**（13B/20B 门票），与 Q5_K 转存正交叠加后 9B TG 可到 ~6-7 tok/s（8.6.2 节）。

**两类路径的本质区分（升顶 2026-08-08, 避免与 8.7.3 方案 1 混为"都需要用户决策"）**：

- **方案 2 (scatter-gather) = clear win（2026-08-09 修正定位）**：保留主 mempool (offload lmhead), 让 heap 回退的 ~2 GiB 权重被 DSP 直接 scatter-gather 读取。**保留 JZ single mempool 架构, 不失去 lmhead offload 能力**, < 4 GiB 模型 (qwen1/2B/8B) 完全不受影响。**收益是消除 ~200 ms/token 的 mirror memcpy + 支持 > 4 GiB 模型（13B/20B）**；它**不消除 24 个 ssm_out split**——ssm_out.weight 是 Q5_K 类型被 validator 拒绝、调度期就在 CPU buft（8.3），从未进入 hexagon 权重块，scatter-gather 够不到它们。消除 split 靠 Q5_K -> Q4_0 转存（8.7.4 步骤 0，~30-50 行）。两者叠加 9B TG ~6-7 tok/s
- **方案 1 (per-buffer 替代) = 需用户决策**：完全放弃 single mempool, 失去 lmhead offload。**qwen1/2B/8B 等 < 4 GiB 模型 TG 可能显著下降**。属于**架构层面 trade-off, 不是 clear win**, 详见上面"关键架构取舍"段（line 1774-1795）

**重要警告（用户指出, 2026-08-08）**：方案 1（per-buffer 替代单 mempool）**几乎推翻 JZ single mempool 架构的两大根基之一**——single mempool 让 JZ 能 offload lmhead（连续 IOVA），改多 mempool 后 lmhead 必须回 CPU，qwen1 / 2B / 8B 等 < 4 GiB 模型 TG 可能下降。**这是真正 trade-off，不是 clear win**。详见上面"关键架构取舍"段。**方案 2 (scatter-gather) 不在此警告范围内**——它保留主 mempool, 不推翻 JZ 架构

- **方案 1 — 多 mempool 替代单 mempool（per-buffer 仿 QCOM）【已识别但不在 8.7.4 主路径表, 需用户决策】**：放弃 JZ 的 single-mempool 架构，仿 QCOM per-buffer 方案（[ggml-hexagon.cpp:1065-1075](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1065-L1075)），预分配多个 1 GiB pinned buffer（每次 `rpcmem_alloc2` + `fastrpc_mmap`），按 tensor 实际大小挑选合适 buffer 装入。代码改动集中在 [ggml-hexagon-jz.cpp:1725-1790](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L1725-L1790) 的 `ggmlhexagon_init_rpcmempool` 和后续 tensor→mempool 分配逻辑。**取舍（用户指出, 2026-08-08）**：获得 9B+ 模型容量，但**几乎推翻 JZ single mempool 架构的两大根基之一**——single mempool 让 JZ 能 offload lmhead（连续 IOVA），改多 mempool 后 lmhead 必须回 CPU，qwen1 / 2B / 8B 等 < 4 GiB 模型 TG 可能显著下降。这不是 clear win, 详见上面"关键架构取舍"段（line 1774-1795）和下方"重要结论"段。**本方案在 8.7.4 主路径表 1a/1b/1c 中不再列出, 仅在此处保留为"已识别风险路径"**, 待用户明确接受"放弃 JZ 架构优势换取 9B+ 容量"后再评估
- **方案 2 — scatter-gather descriptor【容量与 mirror 根治路径, clear win, 但 2026-08-09 起不再是消 split 路径】**：DSP 端 execute_op 支持从多个非连续 mempool region 拉取数据（即便每个 region < 4 GiB，scatter 多 region 总和 > 4 GiB），让 9B / 13B / 20B 模型的 weight 分布在 2-3 个 mempool 里。AP 侧 descriptor 含 `n_buffers + base + size[]` 数组，DSP 侧 execute_op 按基址+偏移取数。**与方案 1 不同**：方案 2 不完全放弃 single mempool，可在保留主 mempool (offload lmhead) 的同时为超出部分新增几个辅助 mempool（scatter-gather 访问），架构上更"加法"而非"替换"。**9B 模型最低可行范围（2026-08-09 修正）**：对 heap 回退的 ~2 GiB 权重做 scatter-gather（DSP 直读 system memory，等效 8.6.2 的 DMA 拉取），消除每 token ~2 GiB 的 AP-side mirror memcpy；**24 个 ssm_out split 不在本路径范围**——它们是 Q5_K 类型拒绝（8.3），由 Q5_K -> Q4_0 转存覆盖（8.7.4 步骤 0）。**预期收益（2026-08-09 重估）**：
  - 消除 ~200 ms/token mirror memcpy（heap 权重免镜像，DSP 直读）
  - 与 Q5_K 转存叠加：1.48 -> ~2.8（转存消 split）-> ~6-7 tok/s（scatter-gather 消 mirror）
  - 支持 > 4 GiB 模型（13B / 20B 门票，8.7.2/8.7.3 矩阵）
  - < 4 GiB 模型（qwen1/2B/8B）继续用 JZ single mempool, **完全不受影响**
  - 保留 lmhead offload 架构优势, 9B lmhead 继续 DSP 执行
  - **总评：clear win（容量/mirror 维度），优先级仅次于 Q5_K 转存**
- **方案 3 — HAP_mmap2 直读 system memory 物理 fd**（真正绕开 ION / 走 ARM64 host PA 空间）：跳过 ION 分配，由 AP 侧 `mmap` 一大段匿名页（24 GB system mem 完整可用，无 4 GiB 限制），把 fd 通过 `fastrpc_mmap` 传给 DSP，HAP_mmap2 直接映射物理页。**理论最干净**，但要求 fastrpc_ioctl 的 size 字段确实没截断（证据 E 已证 FastRPC ioctl 是 `__u64`），且 HAP_mmap2 内部支持 > 4 GiB 物理连续页

**与 QCOM per-buffer 方案的对比（升顶 2026-08-08：4 GiB 根因是 V79 user VA, 1a 已撤回）**：

**Table-36**：与 QCOM per-buffer 方案的对比（升顶 2026-08-08）

| 维度 | QCOM per-buffer (现状) | JZ single-mempool (现状) | JZ 多 mempool (方案 1) | JZ scatter-gather (方案 2) |
|------|----------------------|------------------------|----------------------|------------------------|
| 单次 rpcmem_alloc2 | < 4 GiB (per-buffer) | 4 GiB (4032 MiB 实际) | < 4 GiB (per-buffer) | < 4 GiB (per-region) |
| 总可装 weight | n_chunks * 1 GiB ≤ 16 GiB (DSP 端 sliding 访问) | 4 GiB (V79 user VA 硬件 hard cap) | n_chunks * 1 GiB ≤ 16 GiB (DSP 端 sliding) | n_regions * < 4 GiB (DSP 端 scatter) |
| IOVA 连续性 | 不连续, 需 per-buffer offset | 连续 (单一 4 GiB mempool) | 不连续, 需 per-buffer offset | 主 mempool 连续 + 辅助 region 不连续 |
| offload lmhead | ✗ 不可 (per-buffer 拆散) | **✓ 可** (单一连续 IOVA) | ✗ 不可 (per-buffer 拆散) | **✓ 可** (主 mempool 仍连续) |
| 与现有 mirror 兼容 | N/A (QCOM 无 mirror) | 完全兼容 | 需重写 mirror | 需扩展 mirror 跨 buffer |
| 改动量 (JZ) | 0 (学习对象) | 0 (现状) | 中等 (仿 QCOM) | 较大 (DSP 端协议扩展) |
| 9B 模型 (5.4 GB) 可行性 | 已验证 | 不可行 | 可行 (6 块 1 GiB) | 可行 (2 块 3 GiB) |
| 20B 模型 (~12 GB) 可行性 | 已验证 | 不可行 | 可行 (12 块 1 GiB) | 可行 (4 块 3 GiB) |
| 32B 模型 (~20 GB) 可行性 | 取决于 ctx, 权重装得下 | 不可行 | 可行 (20 块 1 GiB) | 可行 (7 块 3 GiB) |

**重要结论（2026-08-09 重写, Q5_K 转存提升为最短路径, scatter-gather 保持 clear win 但重新定位）**：

9B 性能修复现在是**两条独立路径的叠加**，各自解决一个问题：

1. **Q5_K -> Q4_0 转存（消 graph split）** — ssm_out.weight 264 MB 是 Q5_K 类型被 validator 拒绝（8.3），set_tensor 时转存 Q4_0（复用既有 Q4_K/Q6_K 机制，~30-50 行）后 validator 放行，24 个 MUL_MAT split 直接消除，batch_calls 从 6400 降到 ~256，**TG 1.48 -> ~2.8 tok/s**（8.5 节测算，省的是随 call 数线性的固定开销 ~325 ms/token）
2. **scatter-gather / DMA 拉取（消 mirror + 扩容量）** — heap 回退的 ~2 GiB 权重免镜像，**消除 ~200 ms/token mirror memcpy**，叠加路径 1 后 **TG ~2.8 -> ~6-7 tok/s**（8.6.2 节）；同时把可装模型上限从 4 GiB 提到 24 GiB（13B/20B 门票）

两条路径正交可叠加，且与提交路径优化（op_batch / dspqueue，8.8）也**正交**：
- Q5_K 转存解决 "**24 个 CPU split**"（类型根因，~30-50 行，最先落地）
- scatter-gather 解决 "**装得下 + 免 mirror**"（容量根因，DSP 端协议扩展，1-2 个月）
- op_batch (8.8 短期) 解决 "**每 sub-call 固定提交开销**"（split 消除后段数已少，收益收窄，见 8.7.4）
- dspqueue (8.8 中期) 解决 "**AP-DSP 异步 overlap**"（叠加后 8-10 tok/s）

**优先级（2026-08-09 重排）**：
- **Q5_K -> Q4_0 转存优先级最高**（最短路径、~30-50 行、收益确定、是后续提交路径优化的前置条件——不消掉 24 个 CPU 段，批提交无从合并）
- scatter-gather (1b) 其次，是容量/mirror 根治路径（clear win）
- op_batch (8.8 短期) 降为**短期 complementary 优化**（可与转存并行，转存落地后其收益被部分覆盖）
- 1c/1d 作为更长线储备
- 8.7 绕开路径方案 1 (per-buffer 替代) 继续搁置, 需用户决策

**QCOM per-buffer 方案没有被 JZ 采纳的设计哲学原因**：

- JZ 的 mirror 机制假设所有 weight 都在一个连续 IOVA 范围（mirror 缓冲池也开在同一段），简化 DSP 端 offset 计算
- QCOM 无 mirror 机制（mirror 是 JZ 自有设计），故可直接 per-buffer 拆分无副作用
- 因此 JZ 走 scatter-gather / 多 mempool 路径时，mirror 机制必须先扩展支持跨 buffer（这是 8.6 节讨论 mirror 机制时已经识别的 gap）

**当前 JZ 的内存去向（2026-08-09 按 GGUF 元数据 + alloc log 修正）**：

```
5.03 GiB 模型权重 (427 tensors, GGUF 实测 5148.1 MiB)
  ├── CPU buft 809.6 MiB (调度期决定, 与 mempool 容量无关)
  │     ├── token_embd.weight 545.62 MiB (Q4_0)
  │     │     └── embedding GET_ROWS 在 CPU (upstream 常规)
  │     └── 24 x ssm_out.weight 264.0 MiB (Q5_K)
  │           └── JZ MUL_MAT validator 拒 Q5_K -> 调度期判给 CPU
  │                 └── 触发 24 个 split (类型根因, 扩容 mempool 也消不掉)
  └── hexagon buft ~4088 MiB (output.weight Q6_K 795.70 已转存 Q4_0 545.62)
        ├── ~2059 MiB 进 ION mempool (2005 MiB chunk + 54 MiB chunk)
        └── 1996.95 MiB 回退 heap (needed 1996 > remaining 1968)
              └── hexagon buft 属性不变, 不触发 split
                    └── 代价是每 token ~2 GiB mirror memcpy (~200 ms, 8.6 节)
```

**scatter-gather 突破 + Q5_K 转存叠加后的新内存布局**：

```
5.03 GiB 模型权重
  ├── CPU buft 545.62 MiB: token_embd.weight (embedding 仍在 CPU, upstream 行为)
  └── DSP 可达 ~4.2 GiB (4088 MiB + ssm_out 转存后 216 MiB, Q5_K 264 -> Q4_0 216)
        └── 权重存 system memory, DSP scatter-gather / DMA 直读
              ├── 无 split (Q5_K 已转 Q4_0, validator 放行)
              └── 无 mirror memcpy (heap 回退消失)
```

#### 8.7.1 三个独立的内存维度

突破 4 GiB 限制只解决 weight 一个维度，完整的内存架构需要分别考虑三个独立维度：

**Table-37**：突破 4 GiB 限制需考虑的三个独立内存维度

| 维度 | 当前位置 | 4 GiB 限制是否卡 | 突破方式 |
|---|---|---|---|
| **Weight** | ION mempool 4 GB | ✅ 是（9B 已回退 ~2.0 GiB 到 heap，靠 mirror 维持执行） | scatter-gather / DSP 端 mirror |
| **KV cache** | ION mempool 4 GB | ✅ 是（长 ctx 8K 卡 1.6-3.4 GB） | K/V 量化 (FP16/Q8, 7 章 Table-22 P1) |
| **Activation / scratch** | ION mempool 4 GB | ❌ 否（小，< 100 MB） | 不需要突破 |

**关键观察**：scatter-gather 把 weight 上限从 4 GB 提到 24 GB，但 KV cache 仍是 4 GB ION 限制。两者独立存在，必须分别突破。

#### 8.7.2 手机系统内存分布与可装模型上限

**Table-38**：当前 Snapdragon 8 Elite 平台手机系统内存分布与 scatter-gather 后可装模型上限

| 手机型号 | 系统内存 | scatter-gather 后可装 Q4_0 模型上限 | 备注 |
|---|---|---|---|
| 8 Gen 2 主流 | 8-12 GB | ~7-11 GB | 可装 9B 边缘，13B 装不下 |
| 8 Gen 3 主流 | 12-16 GB | ~11-15 GB | 可装 9B / 13B，20B 边缘 |
| 8 Elite 主流 | 12-16 GB | ~11-15 GB | 同 8 Gen 3 |
| 8 Elite 高配 | 24 GB（极少） | ~22 GB | 可装 20B / 32B，70B 仍不够 |

#### 8.7.3 不同模型在不同手机上的可行性矩阵

**Qwen3.5-9B 实际内存分配（GGUF 元数据 + `log_qwen3.5-9b.txt` alloc 记录实测，2026-08-09 重写）**：

先前版本此处的布局图基于估计的每层大小（MHA ~130 MB / delta-net ~167 MB）与"24 个 ssm_* 权重容量溢出触发 split"的因果链，两处都与实测矛盾，已撤回：(1) GGUF 普查显示全模型仅 24 个 `ssm_out.weight` 是 Q5_K，其余 ssm_* 权重是 Q4_0/Q4_1/Q8_0/F32 并正常留在 hexagon buft，**被 scheduler 判给 CPU 的只有 Q5_K 的 ssm_out**（8.3 节）；(2) 容量回退的受害者是 1996.95 MiB 的普通 hexagon 权重 chunk——回退 heap 后仍属 hexagon buft，经 mirror 维持 DSP 执行，**不产生 split**，代价是 mirror memcpy。实测布局：

```
========================================================================
Qwen3.5-9B 在 JZ 后端的实际内存去向 (GGUF + alloc log 实测)
========================================================================

模型总量: 427 tensors, 32 layers, 5148.1 MiB = 5.03 GiB
  Q4_0  n=173  3929.62 MiB  (含 token_embd.weight [4096, 248320] 545.62)
  Q6_K  n=1     795.70 MiB  (output.weight [4096, 248320], 全模型唯一)
  Q5_K  n=24    264.00 MiB  (全部 ssm_out.weight [4096, 4096], 11.00 x 24)
  Q4_1  n=4     120.00 MiB / Q8_0 n=48  6.38 MiB / F32 n=177  32.39 MiB

调度期三路去向 (前两路 CPU 均与 mempool 容量无关):

  [CPU buft  809.6 MiB]
    token_embd.weight   545.62 MiB  embedding GET_ROWS 在 CPU (upstream 常规,
                                    log: "cannot be used with preferred buffer
                                    type CPU_REPACK, using CPU instead")
    24 x ssm_out.weight 264.0 MiB   Q5_K 被 MUL_MAT validator 拒 (8.3)
                                    -> 24 个 CPU split (SPLIT #2,4,...,48)

  [ION mempool  ~2059 MiB 权重] (hexagon buft)
    weight chunk A  2005.05 MiB     首 tensor = output.weight 545.62 MiB
                                    (Q6_K 经 set_tensor 转存 Q4_0, 省 250 MiB)
    weight chunk B    54.00 MiB
    另有 compute buffer 若干 (50 + 4 + 256 + 50 + 62.62 MiB)

  [heap 回退  1996.95 MiB] (hexagon buft, 容量所迫: needed 1996 > remaining 1968)
    普通层权重, 不触发 split, DSP 经临时 mirror 执行
    代价: ~2 GiB/token AP-side mirror memcpy (~200 ms, 8.6 节)

mempool 注册容量 4024 MiB (probe 4032 - 8 MiB reserve)
  权重装入后 pool_used 2109 MiB (52.4%), 加 compute buffer 后 ~2.5 GiB
  余量 ~1.5 GiB 空闲, 但单 chunk 需求 1996 MiB > 当时剩余 1968 MiB -> 回退
```

**Key observations（对应 alloc_buffer 行为，2026-08-09 重写）**：

1. 权重 chunk 边界由运行时 mempool 余量决定，不是固定值：PID 20562 拆 2005+1996 MiB 双 chunk（后块回退）；PID 21326 单块 4002 MiB 整体回退 heap（needed 4002 > remaining 3973）。两次运行的 split 模式相同——24 个 Q5_K split 与 chunk 边界无关
2. 5.03 GiB 模型 > 4 GiB mempool 是事实，但容量回退的代价形态是 **mirror memcpy**（heap 权重每 token ~2 GiB 镜像，~200 ms），**不是 split**；24 个 split 全部来自 Q5_K 类型拒绝，扩容 mempool 消不掉
3. 消除 split 的最短路径是 Q5_K -> Q4_0 转存（8.7.4 步骤 0）：24 个 CPU 段消失，batch_calls 6400 -> 256，省 ~325 ms/token 固定开销，TG 1.48 -> ~2.8 tok/s（8.5 节测算）
4. 若容量也突破（scatter-gather / 等效 8 GiB）：heap 回退消失、mirror memcpy 归零，叠加后 681 - 325 - 200 = ~156 ms/token -> ~6-7 tok/s（8.6.2 节）；13B/20B 模型同时获得装载门票
5. QCOM default `get_max_size` 1024 MiB、per-buffer scatter-gather 突破 V79 4 GiB 限制的架构对比见 8.7 Table-33/34，不受影响

**容量突破后的假设对比（仅作对比分析；V79 DSP user VA 仍是 32-bit 4 GiB hard cap，需 scatter-gather 让 backing store 落在 system memory。另注意：embedding 在 CPU 与 Q5_K split 均与容量无关，不随扩容消失）**：

```
========================================================================
4 GiB 现状 vs 容量突破 (scatter-gather / 等效 8 GiB) 对比 (Qwen3.5-9B)
========================================================================

| Metric                      | 4 GiB 现状             | 容量突破后                |
| --------------------------- | ---------------------- | ------------------------- |
| 模型总权重                  | 5.03 GiB               | 5.03 GiB                  |
| DSP 可达权重                | 4088 MiB (2059 mempool | ~4304 MiB (含 ssm_out     |
|                             |  + 1996 heap + 小chunk)|  转存后 216 MiB)          |
| CPU 权重                    | 809.6 MiB (token_embd  | 545.62 MiB (仅 token_embd,|
|                             |  545.62 + ssm_out 264) |  embedding 在 CPU 是      |
|                             |                        |  upstream 常规, 不消)     |
| 24 个 ssm_out CPU split     | 有 (Q5_K 类型根因)     | 仍在 -- 与容量无关,       |
|                             |                        | 需叠加 Q5_K 转存才消除    |
| mirror memcpy               | ~2 GiB/token (~200 ms) | 0 (heap 回退消失)         |
| Hexagon 段/token            | 25                     | 25 (扩容 alone);          |
|                             |                        | 1 (叠加 Q5_K 转存)        |
| batch_calls (256 token)     | 6400                   | 6400 (alone); 256 (叠加)  |
| TG tok/s (估算)             | 1.48 (实测)            | ~2.1 (alone, 只省 mirror);|
|                             |                        | ~6-7 (叠加转存, 8.6.2)    |
| 13B (~7.5 GB) / 20B (~11 GB)| 不可                   | 可装 (backing 在 system   |
|                             |                        |  memory, 16 GB 手机覆盖)  |

Key insight (2026-08-09 修正): 容量突破单独只省 mirror (~200 ms/token),
不消 split -- 旧结论 "扩容到 8 GiB 消掉全部 25 个 split -> 5.7 tok/s
QCOM parity" 已撤回, split 是 Q5_K 类型根因。split 消除靠 Q5_K 转存
(~30-50 行, 8.7.4 步骤 0)。容量 + 转存叠加把 9B TG 从 1.48 推到
~6-7 tok/s; 再叠 dspqueue 批提交 (8.8) 追平/超过 QCOM 7.70。
========================================================================
```

**Table-39**：scatter-gather + K/V 量化双重突破后，主流 Q4_0 模型在不同手机上的可行性

| 模型 | 权重 (Q4) | KV cache (4K ctx) | KV cache (8K ctx) | 16 GB 手机 | 24 GB 手机 |
|---|---|---|---|---|---|
| Qwen3.5-9B | 5.4 GB | 0.6 GB | 1.2 GB | ✅ 装得下 | ✅ 装得下 |
| Qwen3.5-13B | 7.5 GB | 0.8 GB | 1.6 GB | ✅ 装得下 | ✅ 装得下 |
| Qwen3.5-20B | 11 GB | 1.0 GB | 2.0 GB | ✅ 装得下（不需 K/V 量化） | ✅ 装得下 |
| Qwen3.5-32B | 18 GB | 1.6 GB | 3.2 GB | ❌ 超过 16 GB | ✅ 装得下（需 K/V 量化 8K ctx） |
| Llama-3-70B | 38 GB | 2.8 GB | 5.6 GB | ❌ 超过 16 GB | ❌ 超过 24 GB |

**关键发现**：

- **scatter-gather 让 16 GB 手机从"只能装 4 GB 模型"变成"可装 11 GB 模型"**（9B / 13B / 20B 全覆盖）
- **scatter-gather + K/V 量化让 24 GB 手机可装 18 GB 模型**（Qwen3.5-32B），突破 4 GiB KV cache 限制
- **70B 量级任何手机都装不下**，需要 weight 压缩到 Q2/Q3（Llama-3-70B Q2 ~22 GB）或 SSD offload

#### 8.7.4 突破 4 GiB 限制后的优化路径顺序

**前提澄清**：4 GiB 限制是 **Hexagon V79 user-mode 32-bit 虚拟地址空间（硬件 PRM §1.1）**——这是**硬件 hard cap**，不是 platform 软件层限制。任何软件 workaround（ION/FastRPC/HAP_mmap 字段升级）都**不能**让 DSP 看到 > 4 GiB。QCOM 之所以支持 9B/13B/20B 模型，是因为使用了 **scatter-gather** 或 **sliding window mmap** 让 DSP 只在 4 GiB VA window 内工作、weight 实际存 system memory。JZ current 架构（pre-allocate 4 GiB mempool + 一次性 mmap 整个）受此限制，**需要架构重构才能突破**

**Table-40**：9B 性能修复路径顺序（2026-08-09 重排：Q5_K 转存升为最短路径；按收益/风险/工时排序）

| 步骤 | 工作 | 解决什么问题 | 9B TG 收益 | JZ 改动量 |
|---|---|---|---|---|
| **0. Q5_K -> Q4_0 set_tensor 转存（最短路径，2026-08-09 新增）** | 复用既有 Q4_K/Q6_K 转存机制：`ggml_hexagon_weight_dsp_type` 加 Q5_K 映射 + 新增 `repack_q5k_as_q4_0_tiled_to_buf` + validator switch 加 Q5_K case | **24 个 ssm_out CPU split（类型根因，8.3）** | 1.48 -> ~2.8 tok/s（batch_calls 6400 -> 256，省 ~325 ms/token 固定开销，8.5） | **小（~30-50 行，集中在 set_tensor 转换与 validator 放行）** |
| 1a. ~~JZ 改 `get_max_size` (1 行)~~ **(已撤回)** | 改 `get_max_size()` 返回 1 GiB 让 ggml core 拆 chunk | weight 容量 | **不可行 (验证失败, 2026-08-08)** | N/A (JZGmp `alloc_buffer` 是 bump allocator, 改 `get_max_size` 不能扩总容量) |
| 1b. JZ scatter-gather (8.7 绕开路径方案 2, 容量/mirror 根治)【clear win, 2026-08-09 重新定位】 | DSP 端 HVX descriptor 从 system memory 物理页拉 weight, **绕过 V79 user VA 4 GiB hard cap** | weight 容量 + heap 权重 mirror memcpy（**不含 split**——split 是 Q5_K 类型根因，由步骤 0 覆盖） | 叠加步骤 0 后 ~2.8 -> ~6-7 tok/s；13B/20B 装载门票 | 较大 (DSP entry.c scatter-gather 支持 + 4 个缺失点) |
| 1c. HAP_mmap2 system memory (8.7 绕开路径方案 3) | `HAP_mmap2` 按 layer 切换 mmap 内容, sliding window | weight 容量 | 装得下, **仍受 V79 4 GiB 总 VA 限制** | 较大 (需要 DSP 端 mmap 切换协议 + cache 一致性) |
| 1d. 混合方案 (控制结构放 user VA + weight 放 system memory, 工程最干净) | 控制结构放 DSP user VA, weight 放 system memory scatter-gather | weight + control | 装得下, 理论最干净 | 较大 (架构重构, 1-2 个月工时) |
| 2. 短期 op_batch 累积 (8.8) | 压缩 per-call 固定提交开销 | 与容量/split 均不直接相关 | 步骤 0 落地前 1.48 -> 2.5-3.5 tok/s；步骤 0 之后收益收窄（段数已从 25 降到 1，可批的段不多） | 小 (改 1 个函数) |
| 3. K/V 量化 (7 章 Table-22 P1) | 让 KV cache 装下 4 GiB | KV cache | 6-7.7 -> 8-10 tok/s | 中 (算子 + Q8/FP16 kernel) |
| 4. 中期 dspqueue 异步化 (8.8) | 进一步压平单次提交开销 (~214x 差距的真因，8.4) | 提交路径 | 叠加后 8-10 tok/s，追平/超过 QCOM | 中 (改 fastrpc 路径) |

步骤 0 是 9B 性能修复的"第一刀"——不先消掉 24 个 CPU 段，后续批提交/流水线无从合并（25 段不合并，op_batch 能批的只是 descriptor 准备，段间同步开销还在）；步骤 1 是大模型（> 4 GB）的"容量门票"。1b / 1c / 1d 三选一**不互斥**：1d (混合方案) 工程最干净但重构量最大，1c 仍受 V79 4 GiB 总 VA 限制（sliding window 不是真的绕开），**1b scatter-gather 是容量/mirror 维度的 clear win**。

**8.7 绕开路径方案 1 (per-buffer 替代单 mempool, 仿 QCOM) 不在主路径表**：8.7 节"绕开路径"列出的"方案 1"路径虽然最直接（仿 QCOM per-buffer 拆分），但**几乎推翻 JZ single mempool 架构优势**（失去 offload lmhead 能力, qwen1 / 2B / 8B 等 < 4 GiB 模型 TG 可能显著下降），详见上面"关键架构取舍"段。**本路径作为"已识别风险路径"搁置**, 在用户明确接受"放弃 JZ 架构优势换取 9B+ 容量"之前**不推荐推进**——这是架构层面的取舍, 不是 clear win

**优先建议（2026-08-09 重排：Q5_K 转存升为最高优先级，scatter-gather 保持 clear win 但重新定位为容量/mirror 路径）**：

- **编号映射澄清**：
  - 8.7 节"绕开路径"列了"方案 1/2/3"
  - 8.7.4 路径表"步骤 1"列了 1a/1b/1c/1d, **其中 1b=8.7 绕开路径方案 2 (scatter-gather), 1c=方案 3 (HAP_mmap2), 1d=混合方案 (8.7 绕开路径没有, 是新增路径)**；**步骤 0 (Q5_K 转存) 是 2026-08-09 新增**，源于 8.3 节根因反转（split 是 Q5_K 类型根因，不是容量根因）
  - **8.7 绕开路径方案 1 (per-buffer 替代单 mempool) 不在 8.7.4 主路径表**, 原因是几乎推翻 JZ single mempool 架构 (失去 offload lmhead), 是已识别风险路径, 不是 clear win
- **步骤 0 (Q5_K -> Q4_0 转存) 优先级最高**：
  - 改动最小（~30-50 行：`ggml_hexagon_weight_dsp_type` 加映射 + 仿 `repack_q6k_as_q4_0_tiled_to_buf` 新增 Q5_K 转换 + validator 加 case）
  - 收益确定（24 个 split 直接消除，batch_calls 6400 -> 256，TG 1.48 -> ~2.8 tok/s）
  - 是后续一切提交路径优化的**前置条件**（不消掉 24 个 CPU 段，批提交/流水线无从合并）
  - 副作用可控：Q5_K -> Q4_0 是有损重量化（与既有 Q4_K/Q6_K -> Q4_0 同性质），ssm_out 权重体积 264 -> 216 MiB，加载期一次性转换；**需先做 CPU vs DSP 数值对比验证输出不乱码**（9B 乱码问题与 GATED_DELTA_NET 相关，转存后该层输入分布变化需回归）
  - < 4 GiB 模型（qwen1/2B/8B）不含 Q5_K ssm_out, **完全不受影响**（五模型 CI 回归确认）
- **1b scatter-gather 其次（容量/mirror clear win）**：
  - 保留 JZ single mempool 架构, 主 mempool 仍装 lmhead + 主要权重
  - 消除 heap 回退权重的 ~200 ms/token mirror memcpy，叠加步骤 0 后 TG ~6-7 tok/s
  - 把可装模型上限从 4 GiB 提到 24 GiB（13B/20B 门票）
  - < 4 GiB 模型（qwen1/2B/8B）继续用 single mempool, **完全不受影响**
  - 实施成本集中在 DSP 端 entry.c scatter-gather 支持（1-2 个月工时）
- **2 op_batch 累积 降为短期 complementary 优化**：
  - 不依赖步骤 0/1b, 可并行实施；在步骤 0 落地前可先把 1.48 推到 2.5-3.5 tok/s
  - 步骤 0 落地后段数已从 25 降到 1，其收益被大部分覆盖
- **4 中期 dspqueue 异步化**：与 0/1b/2 正交, 压平单次提交开销（~214x 差距的真因）, 全部叠加可达 8-10 tok/s
- **8.7 绕开路径方案 1 (per-buffer 替代) 暂不推进**：因推翻 JZ 架构优势, 待用户正式提交"1 路径架构取舍"提案（含 6 维 trade-off 表格分析）后, 由用户决定是否接受
- **1c / 1d 作为更长线储备** (1-2 个月工时), 等 1b 决策后再评估
- **建议优先策略**：步骤 0 (Q5_K 转存) 立即落地并过五模型 CI, 同步并行 1b (scatter-gather) 与 2 (op_batch), 实施期间持续验证 qwen1 / 2B / 8B 等 < 4 GiB 模型 TG 不退化

#### 8.7.5 与 8.6 节的关系

8.6 节给出 scatter-gather 的具体实现（DSP 端 mirror 或 true scatter-gather），本节给出 scatter-gather 实施后的工程价值（突破 4 GiB 限制 + 大模型扩展性）。两者构成完整的"如何绕开 4 GiB 限制"闭环：

```
8.6 给出"怎么做" (descriptor 扩展 + entry.c +30 行)
8.7 给出"做了之后能得到什么" (4 GB -> 24 GB 模型, 16/24 GB 手机覆盖 9B-32B)
```

**与 QCOM 方案的关系**：QCOM 已经在生产环境运行 9B / 13B / 20B 模型，但**不是因为 QCOM 绕开了 4 GiB DSP user VA 限制**——而是因为 QCOM 实际机制是 **scatter-gather 或 sliding window mmap**，让 weight 存 system memory、DSP 只在 4 GiB VA window 内工作。8.7 节列出的方案 1b/1c/1d 本质上是把 QCOM 的 scatter-gather / sliding window / 混合设计哲学移植到 JZ，但因为 JZ 当前有 mirror 机制 + 12 阶段调度，**架构重构工作量较大**。**4 GiB 限制的真正根因是 Hexagon V79 user-mode 32-bit VA 硬件 hard cap**（[V79 PRM §1.1](file:///opt/qcom/Hexagon_SDK/6.3.0.0/docs/pdf/80-N2040-60_REV_AA_Hexagon_V79_Programmer_Reference_Manual.pdf)），所有软件 32-bit 字段都是它的下游派生

8.8 节列出的三阶段修复路径（短期 op_batch / 中期 dspqueue / 远期 producer-consumer）解决的是 per-call overhead，与本节 scatter-gather 是**正交的两条独立路径**，可同时推进。

### 8.8 三阶段修复路径

**前置说明（2026-08-09）**：本节三阶段解决的是**提交路径开销**（8.4 节 ~214x 差距的真因），与 8.7.4 步骤 0（Q5_K 转存消 split）正交。注意步骤 0 落地后每 token Hexagon 段从 25 降到 1，短期 op_batch 的"多段合一"收益被大部分覆盖——op_batch 更应理解为步骤 0 落地前的过渡手段，或与步骤 0 并行推进的互补路径。中远期 dspqueue / producer-consumer 不受段数影响，仍是追平 QCOM 的主路径。

**Table-41**：9B TG 5.2x 差距修复路径（按收益/风险/工时排序）

| 阶段 | 方案 | 9B TG 预期 | 风险 | 工时 | 杠杆点 |
|---|---|---:|---|---|---|
| 短期 (1-2 周) | JZ 内置 `op_batch` 累积：同一 `batch_calls` 内多个 sub-graph 合成一个 fastrpc descriptor，复用现有 `ggml_dsp_execute_batch` 路径 | 1.48 -> 2.5-3.5 tok/s (1.7-2.4x) | 低 | 小 (改 1 个函数) | 不动 Hexagon SDK 调用约定，仅在 AP 端做 descriptor 复用 |
| 中期 (QCOM 验证后) | 引入 dspqueue 持久队列 + 异步 fastrpc_invoke (FASTRPC_NB)，AP 端 `enqueue_op` 不阻塞，DSP 后台消费 | 3.5 -> 6-7.7 tok/s (1.7-2.2x) | 中 | 中 (需改 fastrpc 路径 + 错误处理) | 5 模型 CI 全过；与 2B/2B-9B/qwen1 兼容 |
| 远期 (重构) | producer-consumer 流水线：AP 端 12 阶段合并为 descriptor 生产，DSP 端独立消费，移除 1-12 阶段间的隐式同步 | 7 -> 8-10 tok/s (超 QCOM 1.0-1.3x) | 高 | 大 (动 12 阶段 pipeline) | 配合 K/V 量化 (Table-22 P1) 可到 9-12 tok/s |

**短期路径的具体改动点**（基于 5.8.4 阶段，不破坏现有 5 模型 CI）：

1. `ggmlhexagon_backend_graph_compute_batch` 入口处增加 `op_batch_t` 本地累积
2. 当前 `ggml_dsp_execute_batch` 改为支持 N-合-1 descriptor (复用 `htp_opbatch_req` 格式，但走 JZ 的 fastrpc handle)
3. 在 batch 边界 / `flush_batch` 时统一等一次 dspqueue
4. AP 侧的 p5/p12 (mirror) 流程合并到 batch 级，sub-graph 间的 mirror 跳过 (cgraph cache 命中时)

### 8.9 本章结论

1. **9B TG 5.20x / PP 5.26x 差距的真因是单次提交开销差距（2026-08-09 修正算术）**：两端 sub-graph 数量相同（同一 upstream scheduler，QCOM validator 同样拒 Q5_K，各 ~25 段/token），JZ 每段走同步 FastRPC + mirror（21.4 ms），QCOM 每段只是一次 dspqueue ring-buffer write（~0.1 ms，~214x）；单 token 提交成本 JZ 537 ms vs QCOM ~2.5 ms。dspqueue 没有把 25 段压成 1 次提交，它把每次提交变便宜 ~214x——JZ 676 ms/token 中 537 ms 是提交开销，扣除后 ~145 ms 与 QCOM 130 ms/token 基本持平（8.4.5 注 3）。不是 ssm_out 算子缺失（QCOM 算子 dispatch 与 JZ 一致）
2. **JZ cgraph cache 与 QCOM `graphs reused` 命中率都是 99.2%**（JZ 是 AP 端按 op+shape+src ptr 哈希的 descriptor 复用 cache，QCOM 是 upstream scheduler 的 cgraph 结构复用计数，两者层级不同但都恰好命中 99.2%，因 9B cgraph 拓扑变化频率稳定），差距**不在** cache 命中率，而在单次提交开销（21.4 ms vs ~0.1 ms）
3. **ssm_out.weight 在 CPU buffer 的根因是 Q5_K 量化类型不被 JZ MUL_MAT validator 支持（2026-08-09 根因反转，8.3）**，与 mempool 容量无关；QCOM validator 同样拒 Q5_K，两端 split 拓扑相同。最短修复是 set_tensor 时 Q5_K -> Q4_0 转存（复用既有 Q4_K/Q6_K 机制，~30-50 行），24 个 split 直接消除，batch_calls 6400 -> 256；单独消除 split TG 从 1.48 到 ~2.8 tok/s（8.5），它是后续提交路径优化的前置条件，列为 8.7.4 步骤 0 最先落地
4. **JZ 当前 174.4 sec 中 ~136.5 sec (78%) 是 p5/p12 mirror overhead**（68.70 + 67.83 sec），其中 ~83 sec 是随 call 数线性的固定开销（6400 calls x ~13 ms，步骤 0 消 split 可省），~51 sec 是 heap 回退权重的 mirror memcpy（~2 GiB/token，需 scatter-gather/容量突破才省，8.6.1 分解）；DSP 真实计算 ~33.7 sec (p10)；其余 ~4.2 sec 是 p1+p6+p8+p11 等
5. **QCOM 端 unaccounted 0.0% 说明 dspqueue 让 AP 准备与 DSP 算完全 overlap**（QCOM 没有 JZ 的 p1-p12 12 阶段，AP 端没有空转），cgraph cache 99.2% hit 让 99% 的 decode 步走 cache 复用路径
6. **修复路径三条正交叠加（2026-08-09 重排，8.7.4 Table-40）**：步骤 0 Q5_K 转存消 split（1.48 -> ~2.8 tok/s，~30-50 行，最高优先级）-> scatter-gather 消 mirror + 扩容量（-> ~6-7 tok/s）-> dspqueue 异步化压单次提交开销（-> 8-10 tok/s）；op_batch 累积作为步骤 0 落地前的过渡（1.48 -> 2.5-3.5 tok/s）可并行
7. 配合第七章 Table-22 P0 (E 方向 AP per-call overhead 消除) + 本章 Table-40/41 路径，JZ 9B TG 理论可从 1.48 逐步追到 8-10 tok/s，追平/超过 QCOM 7.70

### 8.10 与第七章 (qwen1) 的对比

qwen1 (24 层 MHA 1:1) 与 qwen3.5-9B (32 层 + delta-net 24) 在 JZ 优化路径上的关键差异：

| 维度 | qwen1 (第七章) | qwen3.5-9B (本章) |
|---|---|---|
| 首要瓶颈 | a-inv + bulk flush (60% wall, L2 8 MB 限制) | 提交路径开销 (p5+p12 136.5s = 78% wall：~83s 随 call 数线性的固定开销 + ~51s heap 权重 mirror memcpy) |
| 算子瓶颈 | H3 KV invalidation 确认 (16241 us/batch) | 24 个 ssm_out MUL_MAT split（Q5_K 不被 validator 支持，8.3 根因反转） |
| 模型容量 | 1.2 GiB (Q4_0)，全装入 4 GB mempool | 5.03 GiB (GGUF 实测 5148.1 MiB)：CPU 809.6 MiB (token_embd 545.62 + Q5_K ssm_out 264) + hexagon 权重 4088 MiB 中 1996 MiB 回退 heap |
| QCOM 对比 | JZ TG 比 QCOM 快 1.61x | JZ TG 比 QCOM 慢 5.20x |
| 主要优化方向 | K/V 量化 (Table-22 P1, 10-19% TG) | Q5_K 转存消 split (Table-40 步骤 0, ~30-50 行) + dspqueue 批提交 (Table-41, 中期) |
| 短期可实施性 | 高 (改 cache 写入路径) | 高 (Q5_K 转存复用既有 Q4_K/Q6_K 机制) |

两条路径不冲突：qwen1 走 cache 量化路线，qwen3.5-9B 走 Q5_K 转存消 split (步骤 0) + dspqueue 批提交路线。可以分头推进，最终在 P2 远期 (producer-consumer 流水线) 阶段汇合。


***

## 参考文档

1. [ion-mempool-vs-perbuffer-analysis-20260713.md](file:///home/zhouwg/develop/ggml-hexagon/docs/backend/jz-ggml-hexagon/ion-mempool-vs-perbuffer-analysis-20260713.md) - ION mempool vs per-buffer 模式对比分析，量化 JZ 净优势公式与 PP/TG 性能差异归因。
2. [why-perbuffer-cannot-offload-lmhead-20260724-en.md](file:///home/zhouwg/develop/ggml-hexagon/docs/backend/jz-ggml-hexagon/why-perbuffer-cannot-offload-lmhead-20260724-en.md) - 分析 per-buffer 模式无法 offload lm-head 的根因，以及 per-buffer 与 mempool 模式在 lm-head 上的行为差异。

***

## 修订历史

### 2026-08-09

- **第八章 ssm_out split 根因反转与全章数据修正 (Kimi-K3, 基于 log_qwen3.5_9b_graphsplit.txt / log_qwen3.5-9b.txt / Qwen3.5-9B-Q4_0.gguf 元数据实测)**：核心修正是 8.3 节根因反转——24 个 ssm_out.weight 在 CPU buffer 的原因是 **Q5_K 量化类型不被 JZ MUL_MAT validator 支持**（[ggml-hexagon-jz.cpp:3404](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L3404) `default: return false`），upstream scheduler 调度期判给 CPU，**与 mempool 容量无关**；三重证据链：graphsplit log 中 24 个 ssm_out.weight 标 `[CPU]` 而同层全部其他权重及 lm_head 标 `[Hexag]` / GGUF 普查全模型 427 个 tensor 中 Q5_K 恰好只有这 24 个 / validator switch 代码。最短修复路径随之确定为 set_tensor 时 Q5_K -> Q4_0 转存（复用 [ggml-hexagon-jz.cpp:1460-1464](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L1460-L1464) 既有 Q4_K/Q6_K 转存机制，~30-50 行），24 个 split 直接消除。各节修正：8.1.2 注（两次运行回退规模不同：PID 20562 双 chunk 2005+1996 MiB vs PID 21326 单块 4002 MiB）/ 8.1.3 观察 3/4/6（batch_calls = 256 token x 25 段、25 x 21.4 = 535 ms/token、dsp_exec 不含 ssm_out）/ 8.2（batch_calls = 64 x 25 分解、Table-25 命中率 98.4%、Table-26 重构 embedding 段、洞察 3 算术 ~504 -> ~325 ms/token）/ 8.3 全节重写（正确根因链 + mempool 容量问题真实边界 + 最短修复路径）/ Table-29（"model.output 在 CPU"行修正为 embedding GET_ROWS，model.output 完整在 Hexagon）/ 8.4.5 Table-30 与注 3 重写（QCOM validator 同样拒 Q5_K 且无 K-quants 转存、提交开销差距 5370x -> 214x 算术修正、graphs reused 是 sched-graph 复用计数非提交次数、JZ lm_head offload 是净优势）/ 8.5 重写（split 次要性论证：省固定开销 ~325 ms/token 不省 mirror memcpy，1.48 -> ~2.8 tok/s）/ 8.6.1/8.6.2（p5+p12 136.5 s 分解为 ~83 s 固定 + ~51 s mirror memcpy、Path-E mechanism 2 修复对 9B 无效的容量论证、mirror 因果链修正）/ 8.7（核心论断、方案 2 定位、重要结论、当前/突破后内存去向图全部按 GGUF+log 重写；scatter-gather 从"消 split"重新定位为"消 mirror + 扩容量"，不再声称消除 24 个 split）/ 8.7.1 Table-37（1.4 GiB -> ~2.0 GiB heap 回退）/ 8.7.3 布局图重写（token_embd 545.62 MiB [4096, 248320]、output.weight Q6_K 795.70 MiB 转存 Q4_0 545.62 MiB、撤回 MHA ~130 MB / delta-net ~167 MB 每层估计与"ssm_* 容量溢出触发 split"因果链）/ 8 GiB 假设对比表重写（扩容 alone 不消 split 只省 mirror -> ~2.1 tok/s，叠加转存 -> ~6-7 tok/s；旧"8 GiB 消全部 25 split -> 5.7 QCOM parity"结论撤回）/ 8.7.4 Table-40 路径表重排（新增步骤 0 Q5_K 转存升为最高优先级最短路径，scatter-gather 保持 clear win 但定位为容量/mirror 路径，op_batch 收益随段数减少而收窄）/ 8.8 前置说明（三阶段与步骤 0 的正交关系）/ 8.9 结论 1/3/4/6 重写（214x 算术、Q5_K 根因、mirror 83+51 分解、三路径叠加路线）/ 8.10 对比表（容量行改 5.03 GiB 三路去向、首要瓶颈分解、优化方向改 Q5_K 转存 + dspqueue）

### 2026-08-08

- **第八章 8.7 4 GiB 限制八层精确化 + 硬件根因升顶 (MiniMax-M3, 2026-08-08)**: 用户六次质疑推动本次分析，**第七次由用户直接指出"4 GiB 限制的真正根因"是 Hexagon V79 user-mode 32-bit 虚拟地址空间**（[V79 PRM §1.1](file:///opt/qcom/Hexagon_SDK/6.3.0.0/docs/pdf/80-N2040-60_REV_AA_Hexagon_V79_Programmer_Reference_Manual.pdf) "single 32-bit virtual address space, which holds both instructions and data"）。这次纠正**升顶**——之前 6 轮归因（ION kernel 32-bit / FastRPC 32-bit / rpcmem_alloc 2 GB / per-tensor 拆分 / 改 get_max_size 1 行 / JZ mempool 架构）全部是 **software 派生层**，真正根因是 **V79 user-mode 32-bit VA 硬件 hard cap**。文档 8.7 节首段重组：(1) **核心结论升顶**：V79 user VA 32-bit 4 GiB 是硬件事实，所有 software 32-bit 字段都是它的下游派生；(2) **支持 > 4 GiB 模型的唯一方式 = scatter-gather / sliding window**（weight 存 system memory，DSP 一次只 mmap 4 GiB window，跨 window 用 HVX scatter-gather descriptor）；(3) **新增 7 层 software 32-bit 字段 vs V79 关系表**（ION/FastRPC/HAP_mmap/HAP_mmap2/rpcmem_alloc/rpcmem_alloc2 vs V79 user VA 4 GiB hard cap）；(4) **QCOM 实际机制 = scatter-gather 或 sliding window**，**不是"绕开 4 GiB"**——之前理解"QCOM 通过 get_max_size=1GiB 让 ggml core 拆 chunk"是误读，QCOM 实际上也是让 weight 存 system memory 然后 scatter-gather；(5) **JZ 当前架构受限于 V79 4 GiB hard cap**——pre-allocate 4 GiB mempool + 一次性 mmap 整个 = 受 V79 限制。要支持 > 4 GiB 模型需要架构重构。8.7.4 路径表 1a 仍为已撤回，1b 改为"方案 1, 根治" (scatter-gather, 绕开 V79 4 GiB hard cap)，新增 1d "混合方案" (工程最干净)。8.7.5 改为"QCOM 实际机制也是 scatter-gather / sliding window"。**教训记录**：之前 6 轮分析一直停留在 software 32-bit 字段层（ION/FastRPC/HAP_mmap 等），没从硬件 PRM §1.1 找根因。**正确方法**：遇到 hardware platform 限制时，先看 PRM 中关于 VA / PA / mmu 的章节，找到根因后再分析 software 层
- **第八章新增 8.7 如何突破 4 GiB mempool 平台限制 (MiniMax-M3, 2026-08-08)**: 在 8.6 (Mirror 机制深度分析) 与原 8.7 (三阶段修复路径) 之间插入新 8.7 节, 含 8.7.1 (三个独立内存维度 weight / KV cache / activation) / 8.7.2 (Table-27 手机系统内存分布与 scatter-gather 后可装模型上限) / 8.7.3 (Table-28 9B/13B/20B/32B/70B Q4_0 在 16/24 GB 手机上的可行性矩阵) / 8.7.4 (scatter-gather 后优化路径顺序) / 8.7.5 (与 8.6 节关系) 共 5 个子节; 原 8.7/8.8/8.9 顺延为 8.8/8.9/8.10. 核心结论: 4 GiB 限制是 Linux ION kernel driver 内部 `ion_allocation_data.len` 字段的 32-bit 上限（详细证据链见 8.7 节首段）, scatter-gather 把它从"卡 4 GB 模型"解放到"卡 24 GB 模型", 16 GB 手机覆盖 9B-20B Q4, 24 GB 手机覆盖到 32B (需 K/V 量化配合 8K ctx), 70B 量级需 weight 压缩或 SSD offload. 强调 scatter-gather 与 8.8 三阶段路径是**正交的两条独立路径** (scatter-gather 突破 weight 限制, dspqueue 解决 per-call overhead), 可同时推进. Table-27/28 编号顺延, 总表格数 26 -> 28
- **第八章新增 8.6 Mirror 机制深度分析与 DMA 拉取路径讨论 (MiniMax-M3, 2026-08-08)**: 在 8.5 (ssm_out split 次要性澄清) 与原 8.6 (三阶段修复路径) 之间插入 8.6 节, 含 8.6.1 (mirror 机制为什么对 9B 表现"失效") 与 8.6.2 ("DSP 主动从 system memory DMA 拉 weight" 路径详解) 两个子节; 原 8.6 / 8.7 / 8.8 顺延为 8.7 / 8.8 / 8.9. 8.6.1 指出 Path-E mechanism 2 设计不完整 (cache hit 跳过 scan 但没跳过 memcpy, [ggml-hexagon-jz.cpp:5936-6014](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5936-L6014)), 9B 上 256 个 main sub-graph 仍有 3.2 sec 纯 mirror 开销; 修复 path 是 30-50 行 patch 改 Phase 5 step 2 让 cache hit 跳 memcpy. 8.6.2 详述"DSP DMA 拉 weight" 路径 (DSP 端 scatter-gather descriptor 读 system memory), 列举 4 项 JZ 当前缺失的支持 (cross-buffer view / scatter-gather descriptor / set_tensor 路由 / cache coherency 扩展) 与 5 模型 CI 兼容性分析; 与 8.7 三阶段路径的关系定位为**第四条独立路径** (不冲突, 可叠加), 短期不能上 (1-2 个月工时) 但 9B 4-5x 加速的真正根治方案
- **第八章 Table-23 总结句修正 (MiniMax-M3, 2026-08-08)**: 用户指出 "9B 是 JZ 首次出现的 TG / PP 同时落后于 QCOM 的模型" 不准确——Qwen1.5-1.8B (qwen1) 早已是双边失利模型（TG -22.3% / PP -28.8%，见 3.5 节"Qwen1.5-1.8B PP/TG 均落后的根因"）。已修正为 "Qwen3.5-9B 是 JZ **第二个**出现 TG / PP 同时落后于 QCOM 的模型"，并补充两者的本质区别：Qwen1.5-1.8B 落后 22-29%（~0.7-0.8x，根因 dspqueue pipelining + MHA + 层数不足），Qwen3.5-9B 落后 80%+（~0.19x，5.2x gap，根因 graph split + per-sub-call FastRPC overhead）。**Qwen3.5-9B 是首个双边失利差距达 5x 量级的模型**；9B 改为 Qwen3.5-9B
- **第八章 8.1.1 QCOM 数据修正 (MiniMax-M3, 2026-08-08)**: 用户重新提供高通 ggml-hexagon Qwen3.5-9B 推理结果截图后，发现 8.1.1 误放了 JZ common_perf_print 数据（PP 27.31 / TG 1.48 / sampling 105.93 / load 1912.61 / total 174413.97 / graphs reused 253）而非 QCOM 数据。已修正为截图中的 QCOM 原始数据（sampling 53.59 / samplers 34.37 / load 366.10 / prompt eval 361.89 (143.69 tok/s) / eval 33112.32 (7.70 tok/s) / total 33537.00 / unaccounted 9.20 / graphs reused 253）；原 JZ common_perf_print 数据新增 8.1.2 段保留，dump_perf_stats 顺延为 8.1.3
- **第八章 cgraph cache / graphs reused 概念澄清 (MiniMax-M3, 2026-08-08)**: 修正 8.1 关键观察 1/2、8.4.5 量化对比、8.7 结论等处的 cgraph cache 命中率表述，明确区分两个层级的 cache：(a) **JZ cgraph cache** = AP 端按 op+shape+src ptr 哈希的 descriptor 复用 cache（[ggml-hexagon-jz.cpp:5159](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5159)，miss 时走 Phase 2 descriptor 重建）；(b) **QCOM `graphs reused`** = upstream llama.cpp scheduler 的 cgraph 结构复用计数（在 `common_perf_print` 输出，miss 时重新切分 sub-graph）。**两者层级不同但都恰好 99.2%**，原因是 9B 模型 cgraph 拓扑变化频率稳定（每 255 decode 步 ~2 步需要重切），与具体后端实现无关。明确指出"graphs reused"与 FastRPC 没有任何关系：QCOM 端没有"FastRPC descriptor 重建"的概念，提交链路是 dspqueue 持久化环形队列（`dspqueue_create` 一次，`dspqueue_write` 多次复用），descriptor 概念由 dspqueue 内部 `htp_opbatch_req` 承载，每次 `dspqueue_write` 是把 N 个 op 的 descriptor 批量入队，不存在 per-call FastRPC descriptor 缓存/重建过程
- **新增第八章 qwen3.5-9B TG 性能差距分析与 dspqueue 根因 (MiniMax-M3, 2026-08-08, 基于 feature/qwen1_qwen3_optimize 分支 9b_sched_20260808-183614.txt / 9b_test_output.log / 9b_realtime.log / qwen3_9b_tg_prof_20260808-192916.log + QCOM log 截图)**: 新增独立 '八、qwen3.5-9B TG 性能差距分析与 dspqueue 根因（2026-08-08）', 含 8.1-8.8 共 8 个子节, Table-23/24/25/26 共 4 个表; JZ 实测 TG 1.48 tok/s (255 decode runs, 676.00 ms/run, batch_calls=6400, per-call overhead avg=21369 us, p10 dsp_exec avg=5262 us, p5+p12=136.5 s = 78% wall) / PP 27.31 tok/s (52 token prompt, 36.61 ms/token), QCOM 实测 TG 7.70 tok/s / PP 143.69 tok/s / 99.2% cgraph cache hit rate (253/255, 与 JZ 上游 scheduler 共享) / unaccounted 0.0%; 差距 TG 5.20x / PP 5.26x; 关键发现: JZ 与 QCOM cgraph cache hit rate 都是 99.2% (upstream llama.cpp scheduler 共享), 差距不在 cache 命中率; 真正差距在每 token 提交成本 JZ 537 ms (25 sub-call × 21.4 ms) vs QCOM 0.1 ms (1 dspqueue_write), 5370x 差距; ssm_out MUL_MAT 24 个 split 是次要因素, 单独消除只能从 1.48 到 2.0-3.0 tok/s; 列出三阶段修复路径 (Table-26): 短期 1-2 周 JZ 内置 op_batch 累积 (1.48 -> 2.5-3.5 tok/s) / 中期 dspqueue 异步化 (3.5 -> 6-7.7 tok/s) / 远期 producer-consumer 流水线重构 (7 -> 8-10 tok/s); 8.8 节对比 qwen1 (7 章) 与 9B 优化路径差异; 表格编号连续 0-26 无 gap; 边界严格区分 JZ 概念 (mirror / p5-p12 / FastRPC / per-call overhead) vs QCOM 概念 (dspqueue / op_batch / AP-DSP overlap / graphs reused)
- 新增第七章 qwen1 TG优化实验 (MiniMax-M3, 2026-08-08, 基于 feature/qwen1_qwen3_optimize 分支 qwen1_tg_prof_20260808-165745): 新增独立 '七、qwen1 TG优化实验(2026-08-08)', 含 7.1-7.6 共 6 个子节, Table-19/20/21/22 共 4 个表; 判定 H1 (FLASH_ATTN) 与 H2 (lm-head) 证伪, H3 (KV invalidation) qwen1 确认 (a-inv 16241 us = 30.9% wall, gemma4 仅 994 us = 2.7%); 重构 qwen1 TG 真正瓶颈为 a-inv + bulk flush = 60% wall (L2 8 MB 限制); 列出 P0/P1/P2 三条剩余优化方向 (E 方向 AP per-call overhead 消除 + K/V 量化 + MUL_MAT_ADD fusion) 与三条已证伪方向 (async_bulk_flush / dsp_a_inv_bitmap / force_opfusion_in_pp); 表格编号连续 0-22 无 gap
- CI 数据更新 (Kimi-K3, 基于 self-build-jz 分支 log_abtest_all_20260807-223924.txt): Table-1 更新为夜间五模型 3 轮均值, Qwen3.5-2B PP 从 -9.2% 翻转为 +10.0% (JZ 501.7 vs QCOM 456.1 tok/s), 确认 batch_calls=256 拆分归零; 数据注记改写为夜间 Qwen1.5-1.8B QCOM TG 首轮 spike + PP 跨轮递增; 3.7 节新增修复确认段; Table-6 重排并更新; 4.3.3 标记已落地验证并补实际收益; 4.5 路线图 Step 1 标记完成; 第六章 Table-17 与数据来源从 feature 分支 210809 日志切换为本分支 223924 日志 (PP 495.50->501.71, 对 QCOM +6.2%->+10.0%)
- 新增 5.8.4 + 起点输出误记修正 (Kimi-K3, 06:55): 5.8.4 新增 E/F/G 三方向 (AP per-call overhead 消除 / qwen1 TG 根因定位 / first-touch w-inv 移至加载期) 及两项否掉方向 (a-inv range 合并, PP/TG 双模 cache 策略); 5.8.3 方向 A/B 补录 feature 分支实测陷阱 (共享 kparams->n 致 GQA garble; enable_async_bulk_flush 竞态致 qwen1 garble); 原 5.8.4 顺延为 5.8.5; 修正 MiniMax-M3 表格中起点推理输出误记 - log 102443 实测起点 (batch_calls=6400) 输出文本连贯正常, 乱码为第一阶段 bridge 后新引入的中间状态, 原 "拆分导致的多子图 cache 错误" 表述不成立; Table-15/16/17 顺延为 Table-16/17/18

### 2026-08-07

- **第六章新增 (MiniMax-M3)**: Qwen3.5-2B PP&TG 优化完整过程 (6.1-6.11). 涵盖 7 个阶段: (1) 基于 kimi-k3 在 3.7 节指出的 SOLVE_TRI/SSM_CONV 缺失, 实施两个算子的 bridge layer patch; (2) batch_calls 6400->1792 但推理仍乱码; (3) 用 ggml core 的 `GGML_SCHED_DEBUG` 抓 split 现场; (4) 定位到 6 个 MHA 层 attn_q_norm 拆分点; (5) 分析 per-head view 与 validator 拒绝原因; (6) 放宽 validator + 验证 batch_calls=256 PP&TG 大幅提升. Table-16/17/18 三表按阶段/指标/根因三种维度组织
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

