# 为什么高通 ggml-hexagon 无法经济地将 lm-head 卸载到 DSP(2026-07-24)

> 这里的"无法"指当前 per-buffer 设计下"不经济 / 尚未实现",并非绝对架构上不可能。高通理论上可通过引入"常驻共享 buffer"类(见第 10 节)来支持;问题在于今天尚未实现,而一旦实现就相当于向 single mempool 设计收敛。

> 英文版:[why-perbuffer-cannot-offload-lmhead-20260724-en.md](why-perbuffer-cannot-offload-lmhead-20260724-en.md)(作者:MiniMax-M3, Kimi-K3, GLM-5.2)

## 1. 背景:lm-head 是 TG 的最大瓶颈

TG(token 生成)在两个实现中都是内存带宽受限的:每个 token 都要从 DRAM 重读全部权重。lm-head 矩阵(以本文基准所用 gemma-4-E2B 为例:262144 x 1536,Q4\_K)负责把隐藏状态映射到词表空间,在 CPU 上执行约需 30 ms/token,是单一最大的 TG 开销。lm-head 的具体形状与类型随模型而异(如 qwen3.5-2B 为 Q6\_K,见第 9.3 节);此处以 gemma4 数字作为贯穿示例。

JZ ggml-hexagon与高通ggml-hexagon此前都设有 `ne[1] > 32768` 的限制,把量化大权重矩阵挡在 DSP 之外,因此 lm-head 在两个实现中都只能跑在 CPU 上。JZ ggml-hexagon移除了这个限制,配合 Q4\_K -> Q4\_0 repack 与 first-touch 权重失效机制,把 lm-head 卸载到了 DSP;高通ggml-hexagon至今仍保留该限制。这正是 JZ ggml-hexagon TG(26.91 tok/s)反超高通ggml-hexagon TG(24.91 tok/s)的直接原因(多轮均值,见 [ion-mempool-vs-perbuffer-analysis-20260713.md](file:///home/zhouwg/develop/ggml-hexagon/docs/backend/jz-ggml-hexagon/ion-mempool-vs-perbuffer-analysis-20260713.md))。

## 2. per-buffer ION 的每 buffer 固定成本

高通ggml-hexagon的 `ggml_hexagon_shared_buffer`(见 [`ggml-hexagon.cpp`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp))为每个 buffer 独立持有:

- ION fd(`rpcmem_alloc2` 分配,见 `alloc()` 方法)
- `fastrpc_mmap`(内核 SMMU 映射,见 `alloc()` 方法)
- DSP 侧 `htp_buf_desc[]` 表项 + `bi` 间接寻址(见 `ggml_hexagon_opqueue::add_buffer()`)
- AP-DSP 生命周期协调(alloc / munmap / destroy,见 `free()` 方法)

一个 214MB 的 lm-head 作为单个 buffer,要在整个会话期间持有所有这些资源:内核 fd 表项、ION 句柄、SMMU 映射全程占用。这是 per-buffer API 设计的"对称性"要求:不存在"特大 buffer 特例"。

JZ ggml-hexagon 的 single mempool(见 [`ggmlhexagon_init_rpcmempool()`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp)):初始化时一次 mmap,lm-head 只是池内一个偏移区间,零重复成本。

## 3. dspqueue 大 buffer 的每 batch 重复开销

高通ggml-hexagon的 dspqueue 队列深度为 16(`opt_opqueue = 16`,见 [`ggml-hexagon.cpp`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp) 中的 `opt_opqueue`),即最多 16 个 batch 同时 in-flight。每个 batch 都通过 `ggml_hexagon_opqueue::add_buffer()` 重新注册所需的 `htp_buf_desc`(fd / base / size)。

一旦 214MB lm-head 进入 dspqueue 路径:

- 每个 token 的 op-batch 都调用 `add_buffer()` 重新注册 lm-head,其 `htp_buf_desc[]` 表项每 batch 重填一次
- `ggml_hexagon_opqueue::push()` 每 batch 携带 `dbuf`,214MB buffer 的 fd/size 信息被反复传输
- DSP 侧 [`prep_op_bufs()`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/main.c) 按 fd 复用已有映射;当映射总量超过 `max_vmem` 预算时,驱逐未使用的映射并重新 `HAP_mmap`。214MB buffer 独占一大块 vmem 预算,在预算压力下会产生真实的重复 mmap/munmap 开销
- buffer 生命周期跟随 tensor(`ggml_hexagon_shared_buffer` 挂在 `buffer->context` 上)而非 dspqueue,但每 batch 仍必须重新声明引用

理论上高通ggml-hexagon可以把 lm-head 放在 dspqueue 之外(纯 ION),但这会破坏统一路径:"buffer 必须在 batch 描述符中注册才能被 DSP 算子访问"。

JZ ggml-hexagon 的 single mempool 是"全有或全无":池 mmap 一次,所有张量天然可访问,lm-head 无需任何特殊处理。

## 4. 一刀切、不区分权重与激活的缓存维护

高通ggml-hexagon每 batch 的缓存维护分两层,两层都无法区分张量角色(权重 vs 激活):

### 4.1 描述符包 flags(作用于几 KB,而非张量数据)

`ggml_hexagon_opqueue::push()` 中,dspqueue 数据包携带的是统一固定的 flags:

```cpp
dbuf.flags = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
```

注意:`dbuf` 只是 batch 描述符块(buf/tensor/op 描述符,专用共享块中的几 KB,见 [`ggml_hexagon_opqueue::push()`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp)),张量数据 buffer 从不携带 dspqueue flags。它们在 alloc 时通过 `fastrpc_mmap` 一次映射,DSP skel 在首次使用时按 fd 做 mmap(`prep_op_bufs`)。这部分 flags 的实际刷写/失效逻辑隐藏在闭源 Hexagon DSP 驱动内部。

### 4.2 张量数据:DSP 侧全缓存刷写(一刀切的根源)

张量数据真正的缓存维护发生在 DSP 侧 [`process_opbatch()`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/htp/main.c) 中:batch 开始和结束时各执行一次全 D-cache flush+invalidate:

```c
qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
```

这种全缓存清扫是彻底的"一刀切":

- 无法区分权重与激活:整个 D-cache 每 batch 被 flush+invalidate 两次
- 没有任何机制能表达"权重只在加载时写一次,首次触碰后就不再 invalidate":first-touch 权重优化在整个设计中无从落地

### 4.3 对比:JZ ggml-hexagon 能区分权重与激活的 cache 优化

JZ ggml-hexagon 的 user-space 缓存优化分三个互相正交的机制,按作用域分别命名为:全局机制、per-tensor 机制、per-batch 机制。三者均在 AP 端设置,无优先级关系,作用对象相互独立。

| 机制 | 字段 | 类型 | 控制对象 | 优化目的 | 函数 |
|---|---|---|---|---|---|
| **全局机制** | `dsp_cache_mode` | 4-bit 开关位掩码 | DSP 端 cache flush 行为 | first-touch / dcinva skip / bulk dst flush | `ggmlhexagon_init_dsp` |
| **per-tensor 机制** | `td->flags` | per-tensor 角色标识 | tensor 角色(weight/mirrored/normal) | 区分 weight 让 first-touch 路径生效 | `ggmlhexagon_backend_graph_compute_batch` |
| **per-batch 机制** | `ion_sync_mode` | 3 值模式选择 | AP 端 cache coherency 机制(CVAC vs ion_sync) | 跳过 manual DC CVAC,整池 kernel sync | `ggmlhexagon_backend_graph_compute_batch` (Phase 6.5/7.5) |

下面逐项展开:

**全局机制:AP 端 `dsp_cache_mode`** (见 [`ggml-hexagon-jz.cpp`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp) `struct hexagon_appcfg_t` 中 [line 402-405](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L402-L405)) 是 4-bit 开关位掩码,作为整体在 `ggmlhexagon_init_dsp` 推给 DSP,控制**DSP 端**的 cache flush 行为。**默认 `dsp_cache_mode = 5` (0b0101)**,即 bit 0 (first-touch) 与 bit 2 (批量 dst flush) 默认同时使能;bit 0 / bit 2 之外的 bit 需手动开启。

- bit 0 (0x1): first-touch 路径**使能开关**。开启后,`td->flags=2` 的 weight tensor 走 first-touch 路径,加载时写入一次,DSP 首次触碰后便永久跳过 invalidate。JZ 侧实测:消除约 9.2 ms/token 的冗余权重重复 invalidate(lm-head 常驻后每 token 权重流量约 1.9GB,该数字是 bit0 关闭 vs 开启的对比实测,覆盖全部权重)
- bit 1 (0x2): 跳过前序 dst 的 dcinva
- bit 2 (0x4): batch 末尾批量 dst flush
- bit 3 (0x8): 选择性批量 flush:跳过同 batch 内仍被后续算子消费的 dst

**per-tensor 机制:AP 端 `td->flags`** (见 `ggml-hexagon-jz.cpp` 中 [line 5793](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5793), [line 5799](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5799), [line 5802](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5802), [line 5827](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5827)) 是 per-tensor 角色标识,通过 kernel_params 推给 DSP 消费:

- flags = 2: weight 角色(首触后跳过 cache flush)
- flags = 1: mirrored(可与同 batch 内后续 op 共享 dst)
- flags = 0: 普通(每 batch 正常 cache 维护)

**per-batch 机制:AP 端 `ion_sync_mode`** (见 `ggml-hexagon-jz.cpp` 中 [line 394](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L394) 定义, [line 5868-5873](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5868-L5873) 控制) 是 AP 端 **cache coherency 机制选择**,决定 Phase 6.5 (flush) 与 Phase 7.5 (invalidate) 用哪种方式同步:

- mode = 0: DC CVAC/CIVAC + DMA_BUF_IOCTL_SYNC 两者都用(最稳,但非默认)
- mode = 1: ion_sync only(代码默认) — 跳过 manual DC CVAC/CIVAC,只调一次 `ioctl(DMA_BUF_IOCTL_SYNC_IOCTL)`,**整池 kernel 同步**;同时跳过 Phase 6.5/7.5 中的 per-tensor/cgraph range 扫描([line 5897-5910](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L5897-L5910), [line 6045-6050](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon-jz.cpp#L6045-L6050))——这些扫描在 mode=1 时是 pure overhead
- mode = 2: DC CVAC/CIVAC only — 手动 cache 维护,不用 kernel sync

全局机制与 per-tensor 机制关注**DSP 端** cache 行为(role-aware first-touch),per-batch 机制关注**AP 端** cache coherency 路径选择(整池 ioctl vs per-tensor 扫描)。三套机制作用域不同,正交,可独立配置。

## 5. 32768 guard 是设计权衡的证据

高通ggml-hexagon在 [`ggml_hexagon_supported_mul_mat()`](file:///home/zhouwg/develop/ggml-hexagon/ggml/src/ggml-hexagon/ggml-hexagon.cpp) 中保留了明确的 lm-head 拒绝逻辑:

```cpp
// hardcoded limit to refuse the lm-head for now
if (src0->ne[1] > 32768) {
    return false;
}
```

注释直接写明 "refuse the lm-head"。这更应理解为:per-buffer 设计目前还不能以可接受的成本容纳一个 214MB 常驻权重。需注意这是从代码与注释得出的推断,并非明确的设计意图声明;"for now" 的措辞表明这是已知限制,而非刻意的永久取舍。

- 32K 行是 per-buffer API 的"实用上限"(单 buffer 几十 MB 量级,fd/mmap 成本尚可接受)
- 214MB lm-head 远超此限;若强行卸载到 DSP,会被 per-buffer 生命周期成本拖垮

JZ ggml-hexagon 的版本没有这个 guard,lm-head 得以自然卸载到 DSP。

## 6. 一句话总结

高通ggml-hexagon的 per-buffer 设计隐含一个假设:"每个 buffer 都是独立的、生命周期短、缓存策略一刀切的小对象"。lm-head 恰好相反:巨大、常驻整个会话、需要专门的缓存策略。要支持 lm-head,需要在 dspqueue、驱动、buffer 描述符三层做"破坏对称性"的改动,而本质上就是向 single mempool 设计收敛。

JZ ggml-hexagon 的 single mempool 不是"激进设计":它让 lm-head 卸载到 DSP 这件事水到渠成——lm-head 只是池内的一个区间,没有独立生命周期,没有 per-buffer 成本,也不需要单独的缓存策略。

## 7. "劣势变优势"的反转

高通ggml-hexagon把缓存维护交给 batch 边界上的一刀切操作(驱动处理的描述符包 + DSP 侧全缓存刷写),JZ ggml-hexagon看似"被迫自己管理缓存";但恰恰因为用户态看得见张量角色、能区分权重与激活、能决定 first-touch 行为,214MB lm-head 才变得可优化。

与高通ggml-hexagon的对比:

- 表面优势:缓存维护由框架统一包办,代码更少
- 实际代价:DSP 侧全缓存 flush+invalidate 每 batch 一刀切,不区分权重与激活,first-touch 权重优化无从实现
- 表面劣势:JZ ggml-hexagon 必须自己实现 `ion_sync` + `dsp_cache_mode`
- 实际收益:区分权重与激活的策略消除了约 9.2 ms/token 的冗余权重重复 invalidate(JZ 侧实测,覆盖全部权重)

这是经典的层次设计反转:抽象层越高,优化空间越小;抽象层越低,策略灵活性越大,优化空间越大。

## 8. PP/TG 性能差异解读

以下数据为多轮均值(同一部 Snapdragon 8 Elite 手机,详见 [ion-mempool-vs-perbuffer-analysis-20260713.md](file:///home/zhouwg/develop/ggml-hexagon/docs/backend/jz-ggml-hexagon/ion-mempool-vs-perbuffer-analysis-20260713.md)):

- PP(686.46 vs 435.14 tok/s):生命周期开销 + IOVA 局部性的累积效应
- TG(26.91 vs 24.91 tok/s):直接取决于 lm-head 是否卸载到 DSP
- TG 差距较小(+8%)并非主要由 dspqueue 流水线贡献:dspqueue 的 in-flight batches 主要在 PP 场景(每 token 多 op 可并行调度)发挥价值;TG 受 token 间严格串行依赖(下一个 token 依赖上一个 token)限制,dspqueue 在 TG 中能做的加速仅限 per-token 内的 descriptor prep 与 DSP compute overlap,贡献天然被封顶。TG 差距更直接地反映 lm-head 是否卸载到 DSP + role-aware cache 管理的差异

## 9. 层数决定 PP/TG 反超阈值

JZ 与高通的 PP/TG 差距并非固定值,而是随模型层数线性变化:

```
JZ 净优势 = per_layer_dsp_saving x n_layers + fixed_lmhead_saving - dspqueue_overlap_advantage
           └── 随层数线性增长 ──┘   └─ 固定 ─┘   └── 固定优势,不随层数增长 ──┘
```

其中 `fixed_lmhead_saving` 指常驻 lm-head 卸载 + 其约 9.2 ms/token 的 first-touch 节省(per-layer 与 fixed 的区别见第 9.2 节)。

### 9.1 高通 dspqueue overlap 优势的两个限制

1. **LLM 特性限制**: TG 严格串行(每个 token 依赖前一个 token)。dspqueue 的 16 个 in-flight batches 在 TG 中只能做 per-token 内的 descriptor prep 与 DSP compute overlap,贡献天然被封顶。

2. **无法 offload lm-head 限制**: 高通的 lm-head 留在 CPU(32768 guard)。PP 时最后一个 op(lm-head)在 CPU 上执行,DSP 空闲等待,**打破了 dspqueue 流水线**。这在 PP 比 TG 更严重,因为 PP 时 lm_head 的 m=batch_size,CPU 计算时间更长,DSP 空闲时间更大。

### 9.2 JZ 优势随层数放大

JZ 的 per-layer DSP 优势(role-aware first-touch cache 节省 + mempool 零开销 + HMX pipeline)随层数线性累积。高通的 dspqueue overlap 是固定优势,不随层数放大。

注意:约 9.2 ms/token 的 first-touch 节省**不是 per-layer 量**——它是含常驻 lm-head 的整图每 token 总量(lm-head 常驻时每 token 权重流量约 1.9GB),由单个常驻 lm-head 主导,是固定节省,与层数无关,不应计入 per-layer 项。per-layer 节省(每层权重只写一次、随后跳过)真实存在但要小得多。正确拆分如下:

```
JZ 净优势 = per_layer_dsp_saving x n_layers   (per-layer first-touch,随层数增长)
         + fixed lm-head saving               (约 9.2 ms/token,常驻 lm-head)
         - dspqueue_overlap_advantage         (固定,不随层数增长)
```

当 `n_layers x per_layer_saving > dspqueue_overlap` 时,JZ 在 PP 上也反超。反超阈值取决于模型结构。

### 9.3 三个模型实测对比 (Snapdragon 8 Elite, 2026-07-29)

模型结构数据从 GGUF 文件头验证:

| 模型 | 层数 | Attention | hidden | lm_head 类型 | lm_head 来源 | vocab | PP JZ vs QCOM | TG JZ vs QCOM |
|---|---|---|---|---|---|---|---|---|
| gemma4-E2B | 35 | GQA 8:1 | 1536 | Q4_K | tied (token_embd) | 262K | JZ 赢 | JZ 赢 |
| qwen3.5-2B | 25 | GQA 4:1 | 2048 | Q6_K | tied (token_embd) | 248K | 高通赢 | JZ 赢 (1.8x) |
| qwen1.5-1.8b | 24 | MHA 1:1 | 2048 | Q6_K | 独立 output.weight | 152K | 高通赢 | 高通赢 |

TG 数据 (JZ vs 高通, tok/s):
- gemma4-E2B: 27.2 vs ~24.9 (JZ +9%)
- qwen3.5-2B: 27.39 vs 13.65 (JZ 1.8x, Q6_K -> Q4_0 repack offload 后)
- qwen1.5-1.8b: ~19 vs 24.12 (JZ -21%, MHA corner case)

关键观察:
- gemma4 (35 层, GQA 8:1): per-layer 累积跨过 PP 和 TG 两个阈值,JZ 全面反超
- qwen3.5 (25 层, GQA 4:1): 跨过 TG 阈值但未跨过 PP 阈值,dspqueue overlap 在 PP 仍占优
- qwen1.5 (24 层, MHA 1:1): MHA 的大 K/V 矩阵 [2048,2048] 导致 VTCM 压力,降低 DSP per-layer 效率,两个阈值都未跨过。MHA 是早期模型,现代模型都用 GQA,JZ 在 GQA 模型上优势明显

### 9.4 结论

JZ 的架构优势随模型层数增长。层数越多,JZ 的 per-layer DSP 优势累积越大,最终在高通固定的 dspqueue overlap 优势之上反超。层数 30+ 的现代 GQA 模型(如 gemma4)从 JZ 架构获益最大。

## 10. 对高通ggml-hexagon的启示

如果高通ggml-hexagon未来改进:

- 把 dspqueue 拆成两类:"常驻共享 buffer" + "每 batch 临时 buffer"
- 给驱动/DSP skel 增加"权重角色"概念
- 允许 buffer 独立于 dspqueue 生命周期存在

但这本质上是把 per-buffer 重新设计为"少量 buffer + 池",也就是 single ION mempool 路线。

## 修订历史

### 2026-08-04: 文档准确性修正

作者:DeepSeek-V4-Flash

- 澄清标题"无法"为"当前不经济 / 尚未实现",并非绝对架构上不可能(见标题下注释)。
- 澄清约 9.2 ms/token 的 first-touch 节省是整图每 token 固定总量(由常驻 lm-head 主导),而非 per-layer 量;据此更新第 9 节公式与第 9.2 节拆分。
- 将 gemma4 lm-head 数字(262144 x 1536,Q4\_K)标注为贯穿示例,而非通用值。
- 将第 5 节对 32768 guard 的解读从"承认"弱化为"从代码与注释得出的推断"。
