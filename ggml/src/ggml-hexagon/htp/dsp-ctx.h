/*
 * 2024-2026 The ggml authors
 *
 * this file is part of jz's ggml-hexagon
 *
 * GitHub:   - https://github.com/zhouwg/ggml-hexagon
 */
#ifndef GGMLDSP_CTX_H
#define GGMLDSP_CTX_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

/* Alignment requirements */
#define HEX_BATCH_ALIGN     128
#define HEX_TENSOR_ALIGN    128
#define HEX_OP_ALIGN        128

// Forward declarations for types used in dsp_context
struct hmx_queue;
struct htp_context;

typedef struct dsptensor dsptensor;

struct dsptensor {
   int32_t type;
   int32_t ne[4];
   int32_t nb[4];
   int32_t op;
   int32_t op_params[16];
   int32_t flags;
   void * data;
   int data_len;
};

typedef struct dsp_op_desc dsp_op_desc;
struct dsp_op_desc {
   int32_t opcode;
   int32_t params[16];
   int32_t src0_idx;
   int32_t src1_idx;
   int32_t src2_idx;
   int32_t src3_idx;
   int32_t dst_idx;
};

/*
 * Shared memory batch descriptor for ION-based multi-op offload.
 *
 * Layout in ION mempool:
 *   [hex_batch_hdr]
 *   [hex_op_desc[0..n_ops-1]]
 *   [hex_tensor_desc[0..n_tensors-1]]
 *
 * All data_offset fields are byte offsets from the ION mempool base.
 * DSP side accesses data as: g_ion_dsp_base + tensor->data_offset
 */

/* Tensor descriptor - uses offset instead of pointer */
typedef struct hex_tensor_desc {
    int32_t  type;            /* ggml_type */
    int32_t  ne[4];           /* element counts per dimension */
    int32_t  nb[4];           /* strides (bytes) per dimension */
    int32_t  op_params[16];   /* operation-specific parameters */
    uint32_t flags;           /* 0=ION tensor, 1=mirrored (heap), 2=weight (skip cache flush) */
    uint32_t data_offset;     /* byte offset of data in ION mempool */
    uint32_t data_len;        /* data length in bytes */
} hex_tensor_desc;

/* Op descriptor - references tensors by index.
 * Mirrors Qualcomm htp_op_desc layout: src_idx[HTP_OP_MAX_INPUTS=6] + dst_idx[HTP_OP_MAX_OUTPUTS=4]. */
typedef struct hex_op_desc {
    int32_t opcode;          /* GGML_OP_XXX */
    int32_t params[16];      /* operation parameters */
    int32_t kernel_params[32]; /* precomputed kernel params (e.g. htp_mm_kernel_params for MUL_MAT) */
    int32_t src_idx[6];      /* indices into tensor table (-1 = none); mirrors htp_op_desc.src[6] */
    int32_t dst_idx[4];      /* multi-output support (e.g. QKV fusion), -1 = unused */
    int32_t htp_opcode;      /* Direct HTP opcode for fused ops (0 = use ggml_op_to_htp_op) */
} hex_op_desc;

/* Batch header - entry point for DSP to find everything */
typedef struct hex_batch_hdr {
    uint32_t n_ops;             /* number of ops */
    uint32_t n_tensors;         /* number of tensors */
    uint32_t ops_offset;        /* offset from hdr start -> hex_op_desc[] */
    uint32_t tensors_offset;    /* offset from hdr start -> hex_tensor_desc[] */
    uint32_t total_size;        /* total size of this batch region (hdr + ops + tensors) */
    uint32_t reserved;          /* padding / future use */
} hex_batch_hdr;

// DSP session context: bundles all per-session state.
// Allocated in ggml_dsp_open, freed in ggml_dsp_close.
struct dsp_context {
    // Configuration
    int thread_counts;
    int dump_diag_info;

    // VTCM
    void * vtcm_base;
    size_t vtcm_size;
    unsigned int compute_res_ctx_id;
    volatile int vtcm_needs_release;
    volatile int vtcm_valid;
    int thread_prio;

    // Power
    int power_ctx;
    void * hexagon_power_ctx;

    // HMX
    int hmx_available;
    struct hmx_queue * hmx_queue;

    // ION
    void * ion_dsp_base;
    size_t ion_dsp_size;

    // DSP-side entry.c cache optimization bitmask. Pushed by AP at init via
    // execute_batch(0xFFFC) special mode (no IDL change). All three bits are
    //   are wired into ggml_dsp_execute_batch(); dsp_cache_mode=0 is behaviorally
    // identical to baseline 29c1cf196.
    //   bit 0 (0x1): first-touch weight bitmap    - skip dcinva for repack weights (flags==2) after first access
    //   bit 1 (0x2): skip dcinva for prior dst     - DSP's own dst writes stay in L2; next op's src read skips dcinva
    //   bit 2 (0x4): bulk dst flush at batch end   - collect/sort/merge dst ranges, flush once per region
    //   bit 3..31  : reserved for future use
    uint32_t dsp_cache_mode;

    // DSP-side bit 0 (first-touch weight bitmap) trace enable. Pushed by AP at
    // init via the same execute_batch(0xFFFC) special mode as dsp_cache_mode
    // (bit 16 of the same payload word, so the special-mode encoding is
    //   payload = (dsp_cache_trace_bit0 << 16) | (dsp_cache_mode & 0x7u)
    // ). When non-zero, INVAL_SRC_IF_NEEDED emits one [DSP-CACHE-TRACE-BIT0]
    // log line per bit 0 decision (SKIP or INVAL), with op index, src index,
    // weight address, weight length, current ctx id, and qurt_timer tick count.
    // Default 0 (off) so production perf is unaffected. Set to 1 only when
    // diagnosing the bit 0 stale L2 read bug (llama3 33% prompt-repeat rate
    // observed 2026-07-10). Once the bug is root-caused this can be removed.
    uint32_t dsp_cache_trace_bit0;

    // DSP-side bit 1 (skip dcinva for prior dst) trace enable. Pushed by AP at
    // init via bit 17 of the same execute_batch(0xFFFC) payload word, so the
    // special-mode encoding is
    //   payload = (dsp_cache_trace_bit1 << 17) | (dsp_cache_trace_bit0 << 16)
    //           | (dsp_cache_mode & 0x7u)
    // When non-zero, INVAL_SRC_IF_NEEDED emits one [DSP-CACHE-TRACE-BIT1] log
    // line per bit 1 decision (SKIP if prior_dst_contains_src, INVAL otherwise)
    // with the same op/src/ptr/len fields as the bit 0 trace. Default 0 (off)
    // so production perf is unaffected. Set to 1 only when diagnosing why
    // dsp_cache_mode 5/6/7 garble on the new matmul pipeline (upstream commit
    // 81ff7abe5). Pair with dsp_cache_trace_bit0 to localize the stale-L2-read
    // culprit to a specific bit/op combination.
    uint32_t dsp_cache_trace_bit1;

    // htp_context for calling Qualcomm's execute_op.
    struct htp_context * htp_ctx;
};

#ifdef  __cplusplus
}
#endif

#endif /* GGMLDSP_CTX_H */
