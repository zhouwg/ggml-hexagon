#include "ggml-dsp.h"
#include <string.h>
#include <math.h>

// DIAG_MASK_INF: set upper triangular elements to -infinity.
// Used in attention mechanism before SOFT_MAX.
// src0 = input tensor (square or rectangular matrix per row group)
// dst  = output tensor with upper triangle masked
// op_params[0] = n_past (number of past positions to mask)

int ggmlop_dsp_diag_mask_inf(remote_handle64 h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) {
    GGML_UNUSED(h); GGML_UNUSED(src1);

    uint64_t begin_time = ggml_time_us();

    const int64_t ne0 = dst->ne[0];  // cols (sequence length)
    const int64_t ne1 = dst->ne[1];  // rows (heads * ...)
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];

    int32_t n_past = 0;
    memcpy(&n_past, dst->op_params, sizeof(int32_t));

    float * pd = (float *)dst->data;
    const float * ps = (const float *)src0->data;

    // If src0 != dst, copy first
    if (ps != pd) {
        memcpy(pd, ps, (size_t)(ne0 * ne1 * ne2 * ne3) * sizeof(float));
    }

    // Mask upper triangular: for row j, mask column i if i > n_past + j
    // This matches the reference implementation in ggml-cpu/ops.cpp
    const int64_t nz = ne2 * ne3;
    for (int64_t k = 0; k < nz; ++k) {
        for (int64_t j = 0; j < ne1; ++j) {
            for (int64_t i = n_past; i < ne0; ++i) {
                if (i > n_past + j) {
                    pd[k * ne0 * ne1 + j * ne0 + i] = -INFINITY;
                }
            }
        }
    }

    int64_t end_time = ggml_time_us();
    GGMLHEXAGON_LOG_INFO("elapse time of DIAG_MASK_INF is %lld us (ne0=%ld, ne1=%ld, n_past=%d)",
                         (long long)(end_time - begin_time), (long)ne0, (long)ne1, n_past);
    return 0;
}
