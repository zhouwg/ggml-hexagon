#ifndef HEX_BATCH_H
#define HEX_BATCH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

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

/* Op descriptor - references tensors by index */
typedef struct hex_op_desc {
    int32_t opcode;          /* GGML_OP_XXX */
    int32_t params[16];      /* operation parameters */
    int32_t src0_idx;        /* index into tensor table (-1 = none) */
    int32_t src1_idx;
    int32_t src2_idx;
    int32_t dst_idx;
} hex_op_desc;

/* Batch header - entry point for DSP to find everything */
typedef struct hex_batch_hdr {
    uint32_t n_ops;              /* number of ops */
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

#ifdef __cplusplus
}
#endif

#endif /* HEX_BATCH_H */
