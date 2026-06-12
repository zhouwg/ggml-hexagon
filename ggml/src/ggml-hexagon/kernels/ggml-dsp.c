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
