/*
 * Copyright (c) 2025 The ggml authors
 *
 * Qualcomm Hexagon SDK and reference tech guides could be found at:
 * https://developer.qualcomm.com/software/hexagon-dsp-sdk/tools
 *
 * this single-source-file or self-contained file is implementation of ggml-dsp:
 *    - a customized tiny ggml running on Qualcomm Hexagon cDSP
 *    - ported from original ggml
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#include "ggml-dsp.h"

void ggmlhexagon_log_internal(int level, const char *file, const char *func, int line, const char *format, ...) {
#if !GGMLHEXAGON_DEBUG
    return;
#endif
    static char s_ggmlhexagon_log_internal_buf[GGMLHEXAGON_LOGBUF_LEN];
    va_list args;
    va_start(args, format);
    int len_prefix = snprintf(s_ggmlhexagon_log_internal_buf, GGMLHEXAGON_LOGBUF_LEN, "[%s, %d]: ",
                              func, line);
    int len = vsnprintf(s_ggmlhexagon_log_internal_buf + len_prefix,
                        GGMLHEXAGON_LOGBUF_LEN - len_prefix, format, args);
    if (len < (GGMLHEXAGON_LOGBUF_LEN - len_prefix)) {
        FARF(ALWAYS, "%s\n", s_ggmlhexagon_log_internal_buf);
    }
    va_end(args);
}

void ggmlhexagon_log_always(int level, const char *file, const char *func, int line, const char *format, ...) {
    static char s_ggmlhexagon_log_internal_buf[GGMLHEXAGON_LOGBUF_LEN];
    va_list args;
    va_start(args, format);
    int len_prefix = snprintf(s_ggmlhexagon_log_internal_buf, GGMLHEXAGON_LOGBUF_LEN, "[%s, %d]: ",
                              func, line);
    int len = vsnprintf(s_ggmlhexagon_log_internal_buf + len_prefix,
                        GGMLHEXAGON_LOGBUF_LEN - len_prefix, format, args);
    if (len < (GGMLHEXAGON_LOGBUF_LEN - len_prefix)) {
        FARF(ALWAYS, "%s\n", s_ggmlhexagon_log_internal_buf);
    }
    va_end(args);
}

void ggmlhexagon_dump_tensor_elements(const ggml_tensor * tensor) {
#if !GGMLHEXAGON_DEBUG
    return;
#endif
    float value = 0;
    char tmpbuf[GGMLHEXAGON_LOGBUF_LEN];
    size_t buflen = 0;
    if (tensor->type == GGML_TYPE_F32) {
        memset(tmpbuf, 0, GGMLHEXAGON_LOGBUF_LEN);
        for (int h = 0; h < tensor->ne[3]; h++) {
            for (int i = 0; i < tensor->ne[2]; i++) {
                for (int j = 0; j < tensor->ne[1]; j++) {
                    for (int k = 0; k < tensor->ne[0]; k++) {
                        value = ((float *) tensor->data)[h * tensor->ne[2] + i * tensor->ne[1] +
                                                         j * tensor->ne[0] + k];
                        buflen += snprintf(tmpbuf + buflen, GGMLHEXAGON_LOGBUF_LEN - buflen, "%-4.2f\t", value);
                    }
                    buflen += snprintf(tmpbuf + buflen, GGMLHEXAGON_LOGBUF_LEN - buflen, "\n");
                }
            }
        }
        GGMLHEXAGON_LOG_DEBUG("\n%s\n", tmpbuf);
    }

    GGMLHEXAGON_LOG_DEBUG("\n");
}

void ggmlhexagon_dump_tensor(const ggml_tensor * tensor, int dump_tensor_data) {
    GGMLHEXAGON_LOG_DEBUG("ne = %5d x %5d x %5d x %5d , nb = (%5zi, %5zi, %5zi, %5zi)\n",
         tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
         tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]);

    if ((1 == dump_tensor_data) && (ggml_nbytes(tensor) < 320)) {
        ggmlhexagon_dump_tensor_elements(tensor);
    }
}

static inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline float fp32_from_bits(uint32_t w) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

uint16_t ggml_compute_fp32_to_fp16(float f) {
    const float scale_to_inf = fp32_from_bits(0x77800000U);
    const float scale_to_zero = fp32_from_bits(0x08800000U);
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & 0x80000000U;
    uint32_t bias = shl1_w & 0xFF000000U;
    if (bias < 0x71000000U) {
        bias = 0x71000000U;
    }

    base = fp32_from_bits((bias >> 1) + 0x07800000U) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & 0x00007C00U;
    const uint32_t mantissa_bits = bits & 0x00000FFFU;
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > 0xFF000000U ? 0x7E00U : nonsign);
}

float ggml_compute_fp16_to_fp32(uint16_t h) {
    const uint32_t w = (uint32_t)h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

size_t ggml_nbytes(const struct ggml_tensor * tensor) {
    size_t nbytes;
    const size_t blck_size = 1;
    if (blck_size == 1) {
        nbytes = 0;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
        nbytes += tensor->nb[GGML_MAX_DIMS - 1];
    } else {
        nbytes = tensor->ne[0]*tensor->nb[0]/blck_size;
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }

    return nbytes;
}

bool ggml_is_empty(const struct ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] == 0) {
            return true;
        }
    }
    return false;
}

bool ggml_can_repeat(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return ggml_is_empty(t0) ? ggml_is_empty(t1) :
           (t1->ne[0]%t0->ne[0] == 0) &&
           (t1->ne[1]%t0->ne[1] == 0) &&
           (t1->ne[2]%t0->ne[2] == 0) &&
           (t1->ne[3]%t0->ne[3] == 0);
}

bool ggml_are_same_shape(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");
    return
            (t0->ne[0] == t1->ne[0]) &&
            (t0->ne[1] == t1->ne[1]) &&
            (t0->ne[2] == t1->ne[2]) &&
            (t0->ne[3] == t1->ne[3]);
}

int64_t ggml_nrows(const struct ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

bool ggml_is_transposed(const struct ggml_tensor * tensor) {
    return tensor->nb[0] > tensor->nb[1];
}

bool ggml_is_contiguous_n(const struct ggml_tensor * tensor, int n) {
    size_t next_nb = 4;
    if (tensor->ne[0] != 1 && tensor->nb[0] != next_nb) {
        return false;
    }
    next_nb *= tensor->ne[0];
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        if (tensor->ne[i] != 1) {
            if (i > n) {
                if (tensor->nb[i] != next_nb) {
                    return false;
                }
                next_nb *= tensor->ne[i];
            } else {
                // this dimension does not need to be contiguous
                next_nb = tensor->ne[i]*tensor->nb[i];
            }
        }
    }
    return true;
}

int64_t ggml_nelements(const struct ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

static bool ggml_is_contiguous_0(const struct ggml_tensor * tensor) {
    return ggml_is_contiguous_n(tensor, 0);
}

bool ggml_is_contiguous(const struct ggml_tensor * tensor) {
    return ggml_is_contiguous_0(tensor);
}

int ggml_n_dims(const struct ggml_tensor * tensor) {
    for (int i = GGML_MAX_DIMS - 1; i >= 1; --i) {
        if (tensor->ne[i] > 1) {
            return i + 1;
        }
    }
    return 1;
}

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    GGMLHEXAGON_LOG_DEBUG("enter ggml_abort");
    abort();
}

static inline uint64 hexagon_perf_get_time_us(void) {
    unsigned long long count;
    asm volatile (" %0 = c31:30 " : "=r"(count));
    return (uint64)(count) * 10ull / 192ull;
}

int64_t ggml_time_ms(void) {
    return hexagon_perf_get_time_us() * 1000;
}

int64_t ggml_time_us(void) {
    return hexagon_perf_get_time_us();
}

const char * ggml_get_ggml_type_name(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:   return "f32";
        case GGML_TYPE_F16:   return "f16";
        case GGML_TYPE_Q4_0:  return "q4_0";
        case GGML_TYPE_Q4_1:  return "q4_1";
        case GGML_TYPE_Q5_0:  return "q5_0";
        case GGML_TYPE_Q5_1:  return "q5_1";
        case GGML_TYPE_Q8_0:  return "q8_0";
        case GGML_TYPE_Q8_1:  return "q8_1";
        case GGML_TYPE_Q2_K:  return "q2_k";
        case GGML_TYPE_Q3_K:  return "q3_k";
        case GGML_TYPE_Q4_K:  return "q4_k";
        case GGML_TYPE_Q5_K:  return "q5_k";
        case GGML_TYPE_Q6_K:  return "q6_k";
        case GGML_TYPE_Q8_K:  return "q8_k";
        case GGML_TYPE_IQ2_XXS: return "iq2_xxs";
        case GGML_TYPE_IQ2_XS:  return "iq2_xs";
        case GGML_TYPE_IQ3_XXS: return "iq3_xxs";
        case GGML_TYPE_IQ1_S:   return "iq1_s";
        case GGML_TYPE_IQ4_NL:  return "iq4_nl";
        case GGML_TYPE_IQ3_S:   return "iq3_s";
        case GGML_TYPE_IQ2_S:   return "iq2_s";
        case GGML_TYPE_IQ4_XS:  return "iq4_xs";
        case GGML_TYPE_I8:    return "i8";
        case GGML_TYPE_I16:   return "i16";
        case GGML_TYPE_I32:   return "i32";
        case GGML_TYPE_I64:   return "i64";
        case GGML_TYPE_F64:   return "f64";
        case GGML_TYPE_IQ1_M:  return "iq1_m";
        case GGML_TYPE_BF16:  return "bf16";
        case GGML_TYPE_MXFP4: return "mxfp4";
        case GGML_TYPE_NVFP4: return "nvfp4";
        case GGML_TYPE_Q1_0:  return "q1_0";
        default:              return "unknown";
    }
}

const char * ggml_op_name(enum ggml_op op) {
    switch (op) {
        case GGML_OP_NONE: return "NONE";
        case GGML_OP_DUP: return "DUP";
        case GGML_OP_ADD: return "ADD";
        case GGML_OP_ADD_ID: return "ADD_ID";
        case GGML_OP_ADD1: return "ADD1";
        case GGML_OP_ACC: return "ACC";
        case GGML_OP_SUB: return "SUB";
        case GGML_OP_MUL: return "MUL";
        case GGML_OP_DIV: return "DIV";
        case GGML_OP_SQR: return "SQR";
        case GGML_OP_SQRT: return "SQRT";
        case GGML_OP_LOG: return "LOG";
        case GGML_OP_SIN: return "SIN";
        case GGML_OP_COS: return "COS";
        case GGML_OP_SUM: return "SUM";
        case GGML_OP_SUM_ROWS: return "SUM_ROWS";
        case GGML_OP_CUMSUM: return "CUMSUM";
        case GGML_OP_MEAN: return "MEAN";
        case GGML_OP_ARGMAX: return "ARGMAX";
        case GGML_OP_COUNT_EQUAL: return "COUNT_EQUAL";
        case GGML_OP_REPEAT: return "REPEAT";
        case GGML_OP_REPEAT_BACK: return "REPEAT_BACK";
        case GGML_OP_CONCAT: return "CONCAT";
        case GGML_OP_SILU_BACK: return "SILU_BACK";
        case GGML_OP_NORM: return "NORM";
        case GGML_OP_RMS_NORM: return "RMS_NORM";
        case GGML_OP_RMS_NORM_BACK: return "RMS_NORM_BACK";
        case GGML_OP_GROUP_NORM: return "GROUP_NORM";
        case GGML_OP_L2_NORM: return "L2_NORM";
        case GGML_OP_MUL_MAT: return "MUL_MAT";
        case GGML_OP_MUL_MAT_ID: return "MUL_MAT_ID";
        case GGML_OP_OUT_PROD: return "OUT_PROD";
        case GGML_OP_SCALE: return "SCALE";
        case GGML_OP_SET: return "SET";
        case GGML_OP_CPY: return "CPY";
        case GGML_OP_CONT: return "CONT";
        case GGML_OP_RESHAPE: return "RESHAPE";
        case GGML_OP_VIEW: return "VIEW";
        case GGML_OP_PERMUTE: return "PERMUTE";
        case GGML_OP_TRANSPOSE: return "TRANSPOSE";
        case GGML_OP_GET_ROWS: return "GET_ROWS";
        case GGML_OP_GET_ROWS_BACK: return "GET_ROWS_BACK";
        case GGML_OP_SET_ROWS: return "SET_ROWS";
        case GGML_OP_DIAG: return "DIAG";
        case GGML_OP_DIAG_MASK_INF: return "DIAG_MASK_INF";
        case GGML_OP_DIAG_MASK_ZERO: return "DIAG_MASK_ZERO";
        case GGML_OP_SOFT_MAX: return "SOFT_MAX";
        case GGML_OP_SOFT_MAX_BACK: return "SOFT_MAX_BACK";
        case GGML_OP_ROPE: return "ROPE";
        case GGML_OP_ROPE_BACK: return "ROPE_BACK";
        case GGML_OP_CLAMP: return "CLAMP";
        case GGML_OP_CONV_TRANSPOSE_1D: return "CONV_TRANSPOSE_1D";
        case GGML_OP_IM2COL: return "IM2COL";
        case GGML_OP_IM2COL_BACK: return "IM2COL_BACK";
        case GGML_OP_IM2COL_3D: return "IM2COL_3D";
        case GGML_OP_CONV_2D: return "CONV_2D";
        case GGML_OP_CONV_3D: return "CONV_3D";
        case GGML_OP_CONV_2D_DW: return "CONV_2D_DW";
        case GGML_OP_CONV_TRANSPOSE_2D: return "CONV_TRANSPOSE_2D";
        case GGML_OP_POOL_1D: return "POOL_1D";
        case GGML_OP_POOL_2D: return "POOL_2D";
        case GGML_OP_POOL_2D_BACK: return "POOL_2D_BACK";
        case GGML_OP_UPSCALE: return "UPSCALE";
        case GGML_OP_PAD: return "PAD";
        case GGML_OP_PAD_REFLECT_1D: return "PAD_REFLECT_1D";
        case GGML_OP_ROLL: return "ROLL";
        case GGML_OP_ARANGE: return "ARANGE";
        case GGML_OP_TIMESTEP_EMBEDDING: return "TIMESTEP_EMBEDDING";
        case GGML_OP_ARGSORT: return "ARGSORT";
        case GGML_OP_TOP_K: return "TOP_K";
        case GGML_OP_LEAKY_RELU: return "LEAKY_RELU";
        case GGML_OP_TRI: return "TRI";
        case GGML_OP_FILL: return "FILL";
        case GGML_OP_FLASH_ATTN_EXT: return "FLASH_ATTN_EXT";
        case GGML_OP_FLASH_ATTN_BACK: return "FLASH_ATTN_BACK";
        case GGML_OP_SSM_CONV: return "SSM_CONV";
        case GGML_OP_SSM_SCAN: return "SSM_SCAN";
        case GGML_OP_WIN_PART: return "WIN_PART";
        case GGML_OP_WIN_UNPART: return "WIN_UNPART";
        case GGML_OP_GET_REL_POS: return "GET_REL_POS";
        case GGML_OP_ADD_REL_POS: return "ADD_REL_POS";
        case GGML_OP_RWKV_WKV6: return "RWKV_WKV6";
        case GGML_OP_GATED_LINEAR_ATTN: return "GATED_LINEAR_ATTN";
        case GGML_OP_RWKV_WKV7: return "RWKV_WKV7";
        case GGML_OP_SOLVE_TRI: return "SOLVE_TRI";
        case GGML_OP_GATED_DELTA_NET: return "GATED_DELTA_NET";
        case GGML_OP_UNARY: return "UNARY";
        case GGML_OP_MAP_CUSTOM1: return "MAP_CUSTOM1";
        case GGML_OP_MAP_CUSTOM2: return "MAP_CUSTOM2";
        case GGML_OP_MAP_CUSTOM3: return "MAP_CUSTOM3";
        case GGML_OP_CUSTOM: return "CUSTOM";
        case GGML_OP_CROSS_ENTROPY_LOSS: return "CROSS_ENTROPY_LOSS";
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK: return "CROSS_ENTROPY_LOSS_BACK";
        case GGML_OP_OPT_STEP_ADAMW: return "OPT_STEP_ADAMW";
        case GGML_OP_OPT_STEP_SGD: return "OPT_STEP_SGD";
        case GGML_OP_GLU: return "GLU";
        case GGML_OP_COUNT: return "COUNT";
        default: return "UNKNOWN";
    }
}

static void ggmlhexagon_append_tensor_dimensions(const struct ggml_tensor * tensor, char * output, size_t output_size) {
    char buffer[GGMLHEXAGON_TMPBUF_LEN] = {0};
    const char * type_name = ggml_get_ggml_type_name(tensor->type);
    int len = 0;
    switch (ggml_n_dims(tensor)) {
        case 1:
            len = snprintf(buffer, sizeof(buffer), "%ldx1%s", (long)tensor->ne[0], type_name);
            break;
        case 2:
            len = snprintf(buffer, sizeof(buffer), "%ldx%ld%s", (long)tensor->ne[0], (long)tensor->ne[1], type_name);
            break;
        case 3:
            len = snprintf(buffer, sizeof(buffer), "%ldx%ldx%ld%s", (long)tensor->ne[0], (long)tensor->ne[1],
                           (long)tensor->ne[2], type_name);
            break;
        case 4:
        default:
            len = snprintf(buffer, sizeof(buffer), "%ldx%ldx%ldx%ld%s", (long)tensor->ne[0], (long)tensor->ne[1],
                           (long)tensor->ne[2], (long)tensor->ne[3], type_name);
            break;
    }
    if (len > 0 && len < (int)sizeof(buffer) && (size_t)len < output_size) {
        strncat(output, buffer, output_size - strlen(output) - 1);
    }
}

size_t ggmlhexagon_get_op_index(const struct ggml_tensor * tensor) {
    return (size_t)tensor->op;
}

void ggmlhexagon_get_opkey(enum ggml_op op, const struct ggml_tensor * src0, const struct ggml_tensor * src1, char * buf, size_t buf_size) {
    // Format: "ADDf32_4096x4096f32_4096x4096f32"
    // i.e., "<op_name><type>_<ne0>x<ne1>...<type>_<ne0>x<ne1>...<type>"

    if (!buf || buf_size == 0) {
        return;
    }

    buf[0] = '\0';

    // Get operation name
    const char * op_name = ggml_op_name(op);
    size_t len = strlen(buf);

    if (len < buf_size) {
        strncat(buf, op_name, buf_size - len - 1);
    }

    // Get src0 type (e.g., "f32")
    const char * src0_type_name = ggml_get_ggml_type_name((enum ggml_type)src0->type);
    len = strlen(buf);
    if (len < buf_size) {
        strncat(buf, src0_type_name, buf_size - len - 1);
    }

    // Get src0 dimensions (e.g., "4096x4096")
    char src0_dims[GGMLHEXAGON_TMPBUF_LEN] = {0};
    int ndims = ggml_n_dims(src0);
    if (ndims == 1) {
        snprintf(src0_dims, sizeof(src0_dims), "%ldx1", (long)src0->ne[0]);
    } else if (ndims == 2) {
        snprintf(src0_dims, sizeof(src0_dims), "%ldx%ld", (long)src0->ne[0], (long)src0->ne[1]);
    } else if (ndims == 3) {
        snprintf(src0_dims, sizeof(src0_dims), "%ldx%ldx%ld", (long)src0->ne[0], (long)src0->ne[1], (long)src0->ne[2]);
    } else {
        snprintf(src0_dims, sizeof(src0_dims), "%ldx%ldx%ldx%ld", (long)src0->ne[0], (long)src0->ne[1], (long)src0->ne[2], (long)src0->ne[3]);
    }
    len = strlen(buf);
    if (len < buf_size) {
        strncat(buf, "_", buf_size - len - 1);
    }
    len = strlen(buf);
    if (len < buf_size) {
        strncat(buf, src0_dims, buf_size - len - 1);
    }

    // Get src1 type and dimensions
    const char * src1_type_name = ggml_get_ggml_type_name((enum ggml_type)src1->type);
    len = strlen(buf);
    if (len < buf_size) {
        strncat(buf, src1_type_name, buf_size - len - 1);
    }

    char src1_dims[GGMLHEXAGON_TMPBUF_LEN] = {0};
    ndims = ggml_n_dims(src1);
    if (ndims == 1) {
        snprintf(src1_dims, sizeof(src1_dims), "%ldx1", (long)src1->ne[0]);
    } else if (ndims == 2) {
        snprintf(src1_dims, sizeof(src1_dims), "%ldx%ld", (long)src1->ne[0], (long)src1->ne[1]);
    } else if (ndims == 3) {
        snprintf(src1_dims, sizeof(src1_dims), "%ldx%ldx%ld", (long)src1->ne[0], (long)src1->ne[1], (long)src1->ne[2]);
    } else {
        snprintf(src1_dims, sizeof(src1_dims), "%ldx%ldx%ldx%ld", (long)src1->ne[0], (long)src1->ne[1], (long)src1->ne[2], (long)src1->ne[3]);
    }
    len = strlen(buf);
    if (len < buf_size) {
        strncat(buf, "_", buf_size - len - 1);
    }
    len = strlen(buf);
    if (len < buf_size) {
        strncat(buf, src1_dims, buf_size - len - 1);
    }
}
