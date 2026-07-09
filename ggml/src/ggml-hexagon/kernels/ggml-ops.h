#ifndef GGMLDSP_OPS_H
#define GGMLDSP_OPS_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <remote.h>
#include <AEEStdDef.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

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

typedef struct dsp_opbatch_req dsp_opbatch_req;
struct dsp_opbatch_req {
   int32_t n_tensors;
   int32_t n_ops;
   dsptensor* tensors;
   int tensors_len;
   dsp_op_desc* ops;
   int ops_len;
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

/* Alignment requirements */
#define HEX_BATCH_ALIGN     128
#define HEX_TENSOR_ALIGN    128
#define HEX_OP_ALIGN        128

// DSP session context: bundles all per-session state.
// Allocated in ggmlop_dsp_open, freed in ggmlop_dsp_close.
struct dsp_context {
    // Configuration
    int thread_counts;
    int mulmat_algotype;
    int offload_cgraph_type;
    int dump_diag_info;

    // Work data
    void * work_data;
    size_t work_size;
    unsigned int work_mutex;  // qurt_mutex_t: protects work_data allocation

    // VTCM
    void * vtcm_base;
    size_t vtcm_size;
    unsigned int compute_res_ctx_id;
    volatile int vtcm_needs_release;
    volatile int vtcm_valid;

    // Power
    int power_ctx;
    void * hexagon_power_ctx;

    // HMX
    int hmx_available;
    struct hmx_queue * hmx_queue;

    // ION
    void * ion_dsp_base;
    size_t ion_dsp_size;

    // FP16 weight cache: uses ION shared memory tail region for caching
    // converted FP16 weight tiles (avoids repeated Q4_0->FP16 conversion)
    void * ion_cache_base;
    size_t ion_cache_size;
    size_t ion_cache_offset;

    // DSP-side entry.c cache optimization bitmask. Pushed by AP at init via
    // execute_batch(0xFFFC) special mode (no IDL change). All three bits are
    // wired into ggmlop_dsp_execute_batch(); dsp_cache_mode=0 is behaviorally
    // identical to baseline 29c1cf196.
    //   bit 0 (0x1): first-touch weight bitmap    - skip dcinva for repack weights (flags==2) after first access
    //   bit 1 (0x2): skip dcinva for prior dst     - DSP's own dst writes stay in L2; next op's src read skips dcinva
    //   bit 2 (0x4): bulk dst flush at batch end   - collect/sort/merge dst ranges, flush once per region
    //   bit 3..31  : reserved for future use
    uint32_t dsp_cache_mode;

    // htp_context for calling Qualcomm's execute_op.
    struct htp_context * htp_ctx;
};

extern struct dsp_context *g_dsp_ctx;

int ggmlop_dsp_add(remote_handle64 _h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) ;
int ggmlop_dsp_sub(remote_handle64 _h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) ;
int ggmlop_dsp_mul(remote_handle64 _h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) ;
int ggmlop_dsp_div(remote_handle64 _h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) ;
int ggmlop_dsp_mulmat(remote_handle64 _h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) ;
int ggmlop_dsp_rmsnorm(remote_handle64 _h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) ;
int ggmlop_dsp_rope(remote_handle64 _h, const dsptensor* src0, const dsptensor* src1, const dsptensor* src2, dsptensor* dst) ;
int ggmlop_dsp_softmax(remote_handle64 _h, const dsptensor* src0, const dsptensor* src1, const dsptensor* src2, dsptensor* dst) ;
int ggmlop_dsp_silu(remote_handle64 _h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) ;
int ggmlop_dsp_scale(remote_handle64 _h, const dsptensor* src0, dsptensor* dst) ;
int ggmlop_dsp_cpy(remote_handle64 _h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) ;
int ggmlop_dsp_getrows(remote_handle64 _h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) ;
int ggmlop_dsp_concat(remote_handle64 _h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) ;
int ggmlop_dsp_repeat(remote_handle64 _h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) ;
int ggmlop_dsp_diag_mask_inf(remote_handle64 _h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) ;
int ggmlop_dsp_flash_attn(remote_handle64 h, const dsptensor * q, const dsptensor * k, const dsptensor * v, const dsptensor * mask, dsptensor * dst);
int ggmlop_dsp_test_hmx(remote_handle64 h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst);

#ifdef  __cplusplus
}
#endif

#endif /* GGMLDSP_OPS_H */
