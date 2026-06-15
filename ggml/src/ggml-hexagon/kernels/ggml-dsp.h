#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "HAP_perf.h"
#include "HAP_farf.h"
#include "HAP_power.h"
#include "HAP_vtcm_mgr.h"
#include "HAP_compute_res.h"

#include "qurt.h"
#include "AEEStdErr.h"
#include "hexagon_types.h"
#include "hexagon_protos.h"

#include "skel.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define ggml_tensor         dsptensor

#define GGML_MAX_DIMS       4

#define ALIGN_128_BYTE      128

#define VLEN                128

#define QK4_0               32
#define QK8_0               32

#define GGML_UNUSED(x)      (void)(x)

#define UNUSED              GGML_UNUSED

#define GGML_PAD(x, n)      (((x) + (n) - 1) & ~((n) - 1))

#define GGML_ABORT(...)     ggml_abort(__FILE__, __LINE__, __VA_ARGS__)

#define GGML_ASSERT(x)      if (!(x)) GGML_ABORT("GGML_ASSERT(%s) failed", #x)

#define MIN(a, b)           ((a) < (b) ? (a) : (b))
#define MAX(a, b)           ((a) > (b) ? (a) : (b))
#define ggml_min(a, b)      ((a) < (b) ? (a) : (b))
#define ggml_max(a, b)      ((a) > (b) ? (a) : (b))

#if UINTPTR_MAX == 0xFFFFFFFF
#define GGML_MEM_ALIGN      4
#else
#define GGML_MEM_ALIGN      16
#endif

#define BLOCK_SIZE                  (8 * 1024 / VLEN)  // vector chunks
#define L2FETCH_AHEAD               (BLOCK_SIZE)
#define HVX_INLINE_ALWAYS           inline __attribute__((unused,always_inline))
#define VSIZE_BYTES                 128
#define VSIZE_WORDS                 VSIZE_BYTES/4

#define SIZEOF_FP32                 (4)
#define SIZEOF_FP16                 (2)
#define VLEN_FP32                   (VSIZE_BYTES / SIZEOF_FP32)
#define VLEN_FP16                   (VSIZE_BYTES / SIZEOF_FP16)

#define VTCM_BLOCK_ROWS             16
#define VTCM_BLOCK_COLS             16

#define GGML_API                    extern

#ifdef __cplusplus
// restrict not standard in C++
#    if defined(__GNUC__)
#        define GGML_RESTRICT       __restrict__
#    elif defined(__clang__)
#        define GGML_RESTRICT       __restrict
#    elif defined(_MSC_VER)
#        define GGML_RESTRICT       __restrict
#    else
#        define GGML_RESTRICT
#    endif
#else
#    if defined (_MSC_VER) && (__STDC_VERSION__ < 201112L)
#        define GGML_RESTRICT       __restrict
#    else
#        define GGML_RESTRICT       restrict
#    endif
#endif

#ifndef __cplusplus
#ifndef static_assert
        #if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
            #define static_assert(cond, msg) _Static_assert(cond, msg)
        #else
            #define static_assert(cond, msg) struct global_scope_noop_trick
        #endif
#endif
#endif // __cplusplus


//NPU performance will be slower when enable GGMLHEXAGON_DEBUG
#ifdef NDEBUG
#define GGMLHEXAGON_DEBUG                                   0
#else
#define GGMLHEXAGON_DEBUG                                   1
#endif

#define GGMLHEXAGON_LOGBUF_LEN                              4096
#define GGMLHEXAGON_TMPBUF_LEN                              256
#if GGMLHEXAGON_DEBUG
#define GGMLHEXAGON_LOG_DEBUG(...)                          ggmlhexagon_log_internal(GGMLHEXAGON_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLDSP_LOG_DEBUG(...)                              ggmlhexagon_log_internal(GGMLHEXAGON_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#else
#define GGMLHEXAGON_LOG_DEBUG(...)
#define GGMLDSP_LOG_DEBUG(...)
#endif

#define GGMLDSP_LOG_INFO(...)                              ggmlhexagon_log_always(GGMLHEXAGON_LOG_LEVEL_INFO, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLDSP_LOG_WARN(...)                              ggmlhexagon_log_always(GGMLHEXAGON_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLDSP_LOG_ERROR(...)                             ggmlhexagon_log_always(GGMLHEXAGON_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

#define GGMLHEXAGON_LOG_INFO(...)                          ggmlhexagon_log_always(GGMLHEXAGON_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLHEXAGON_LOG_WARN(...)                          ggmlhexagon_log_always(GGMLHEXAGON_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLHEXAGON_LOG_ERROR(...)                         ggmlhexagon_log_always(GGMLHEXAGON_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

#define GGML_TENSOR_LOCALS_1(type, prefix, pointer, array) \
    const type prefix##0 = (pointer)->array[0]; \
    GGML_UNUSED(prefix##0);
#define GGML_TENSOR_LOCALS_2(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_1    (type, prefix, pointer, array) \
    const type prefix##1 = (pointer)->array[1]; \
    GGML_UNUSED(prefix##1);
#define GGML_TENSOR_LOCALS_3(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_2    (type, prefix, pointer, array) \
    const type prefix##2 = (pointer)->array[2]; \
    GGML_UNUSED(prefix##2);
#define GGML_TENSOR_LOCALS(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_3  (type, prefix, pointer, array) \
    const type prefix##3 = (pointer)->array[3]; \
    GGML_UNUSED(prefix##3);

#define GGML_TENSOR_UNARY_OP_LOCALS \
    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne) \
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

#define GGML_TENSOR_BINARY_OP_LOCALS \
    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne) \
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

#define GGML_TENSOR_BINARY_OP_LOCALS01 \
    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

enum ggmlhexagon_log_level {
    GGMLHEXAGON_LOG_LEVEL_NONE  = 0,
    GGMLHEXAGON_LOG_LEVEL_DEBUG = 1,
    GGMLHEXAGON_LOG_LEVEL_WARN  = 2,
    GGMLHEXAGON_LOG_LEVEL_ERROR = 3,
    GGMLHEXAGON_LOG_LEVEL_INFO  = 4,
};

enum ggml_op {
    GGML_OP_NONE = 0,

    GGML_OP_DUP,
    GGML_OP_ADD,
    GGML_OP_ADD_ID,
    GGML_OP_ADD1,
    GGML_OP_ACC,
    GGML_OP_SUB,
    GGML_OP_MUL,
    GGML_OP_DIV,
    GGML_OP_SQR,
    GGML_OP_SQRT,
    GGML_OP_LOG,
    GGML_OP_SIN,
    GGML_OP_COS,
    GGML_OP_SUM,
    GGML_OP_SUM_ROWS,
    GGML_OP_CUMSUM,
    GGML_OP_MEAN,
    GGML_OP_ARGMAX,
    GGML_OP_COUNT_EQUAL,
    GGML_OP_REPEAT,
    GGML_OP_REPEAT_BACK,
    GGML_OP_CONCAT,
    GGML_OP_SILU_BACK,
    GGML_OP_NORM, // normalize
    GGML_OP_RMS_NORM,
    GGML_OP_RMS_NORM_BACK,
    GGML_OP_GROUP_NORM,
    GGML_OP_L2_NORM,

    GGML_OP_MUL_MAT,
    GGML_OP_MUL_MAT_ID,
    GGML_OP_OUT_PROD,

    GGML_OP_SCALE,
    GGML_OP_SET,
    GGML_OP_CPY,
    GGML_OP_CONT,
    GGML_OP_RESHAPE,
    GGML_OP_VIEW,
    GGML_OP_PERMUTE,
    GGML_OP_TRANSPOSE,
    GGML_OP_GET_ROWS,
    GGML_OP_GET_ROWS_BACK,
    GGML_OP_SET_ROWS,
    GGML_OP_DIAG,
    GGML_OP_DIAG_MASK_INF,
    GGML_OP_DIAG_MASK_ZERO,
    GGML_OP_SOFT_MAX,
    GGML_OP_SOFT_MAX_BACK,
    GGML_OP_ROPE,
    GGML_OP_ROPE_BACK,
    GGML_OP_CLAMP,
    GGML_OP_CONV_TRANSPOSE_1D,
    GGML_OP_IM2COL,
    GGML_OP_IM2COL_BACK,
    GGML_OP_IM2COL_3D,
    GGML_OP_CONV_2D,
    GGML_OP_CONV_3D,
    GGML_OP_CONV_2D_DW,
    GGML_OP_CONV_TRANSPOSE_2D,
    GGML_OP_POOL_1D,
    GGML_OP_POOL_2D,
    GGML_OP_POOL_2D_BACK,
    GGML_OP_UPSCALE,
    GGML_OP_PAD,
    GGML_OP_PAD_REFLECT_1D,
    GGML_OP_ROLL,
    GGML_OP_ARANGE,
    GGML_OP_TIMESTEP_EMBEDDING,
    GGML_OP_ARGSORT,
    GGML_OP_TOP_K,
    GGML_OP_LEAKY_RELU,
    GGML_OP_TRI,
    GGML_OP_FILL,

    GGML_OP_FLASH_ATTN_EXT,
    GGML_OP_FLASH_ATTN_BACK,
    GGML_OP_SSM_CONV,
    GGML_OP_SSM_SCAN,
    GGML_OP_WIN_PART,
    GGML_OP_WIN_UNPART,
    GGML_OP_GET_REL_POS,
    GGML_OP_ADD_REL_POS,
    GGML_OP_RWKV_WKV6,
    GGML_OP_GATED_LINEAR_ATTN,
    GGML_OP_RWKV_WKV7,
    GGML_OP_SOLVE_TRI,
    GGML_OP_GATED_DELTA_NET,

    GGML_OP_UNARY,

    GGML_OP_MAP_CUSTOM1,
    GGML_OP_MAP_CUSTOM2,
    GGML_OP_MAP_CUSTOM3,

    GGML_OP_CUSTOM,

    GGML_OP_CROSS_ENTROPY_LOSS,
    GGML_OP_CROSS_ENTROPY_LOSS_BACK,
    GGML_OP_OPT_STEP_ADAMW,
    GGML_OP_OPT_STEP_SGD,

    GGML_OP_GLU,

    GGML_OP_COUNT,
};

enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 = 5, support has been removed
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_MXFP4   = 39,
    GGML_TYPE_NVFP4   = 40,
    GGML_TYPE_Q1_0    = 41,
};

typedef double      ggml_float;

GGML_API int64_t ggml_time_ms(void);
GGML_API int64_t ggml_time_us(void);

GGML_API size_t ggml_nbytes(const struct ggml_tensor * tensor);
GGML_API int64_t ggml_nrows(const struct ggml_tensor * tensor);
GGML_API int64_t ggml_nelements(const struct ggml_tensor * tensor);
GGML_API int ggml_n_dims(const struct ggml_tensor * tensor);
GGML_API bool ggml_is_contiguous(const struct ggml_tensor * tensor);
GGML_API void ggml_abort(const char * file, int line, const char * fmt, ...);
GGML_API bool ggml_can_repeat(const struct ggml_tensor * t0, const struct ggml_tensor * t1);
GGML_API bool ggml_are_same_shape(const struct ggml_tensor * t0, const struct ggml_tensor * t1);

GGML_API void ggmlhexagon_dump_tensor_elements(const ggml_tensor * tensor);
GGML_API void ggmlhexagon_dump_tensor(const ggml_tensor * tensor, int dump_tensor_data);
GGML_API void ggmlhexagon_log_internal(int level, const char *file, const char *func, int line, const char *format, ...);
GGML_API void ggmlhexagon_log_always(int level, const char *file, const char *func, int line, const char *format, ...);

GGML_API int ggmlop_get_thread_counts(void);
GGML_API int ggmlop_get_mulmat_algotype(void);
GGML_API unsigned int ggmlop_get_compute_res_ctx_id(void);
GGML_API int ggmlop_is_hmx_available(void);
GGML_API void * ggmlop_get_work_data(size_t size);
GGML_API void * ggmlop_get_vtcm_pool(size_t * size);
GGML_API int ggmlop_ensure_vtcm_available(void);  // Ensure VTCM resource is available (for cache mode)
GGML_API uint16_t ggml_compute_fp32_to_fp16(float f);
GGML_API float ggml_compute_fp16_to_fp32(uint16_t h);

GGML_API const char * ggml_op_name(enum ggml_op op);
GGML_API const char * ggml_get_ggml_type_name(enum ggml_type type);

GGML_API size_t ggmlhexagon_get_op_index(const struct ggml_tensor * tensor);
GGML_API void ggmlhexagon_get_opkey(enum ggml_op op, const struct ggml_tensor * src0, const struct ggml_tensor * src1, char * buf, size_t buf_size);

static inline int ggml_blck_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
            return 32;
        case GGML_TYPE_Q2_K:
            return 256;
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_K:
            return 256;
        case GGML_TYPE_I8:
            return 1;
        case GGML_TYPE_BF16:
        case GGML_TYPE_F16:
        case GGML_TYPE_F32:
        default:
            return 1;
    }
}

static inline size_t ggml_type_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return sizeof(float);
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            return sizeof(uint16_t);
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
            return sizeof(uint16_t) + QK4_0/2;
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
            return sizeof(uint16_t) + QK4_0/2 + QK4_0/2;
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
            return sizeof(uint16_t) + QK8_0;
        case GGML_TYPE_I8:
            return sizeof(int8_t);
        default:
            return sizeof(float);
    }
}

static inline size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    return ggml_type_size(type) * ne / ggml_blck_size(type);
}

#ifdef  __cplusplus
}
#endif
