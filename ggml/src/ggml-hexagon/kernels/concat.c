#include "ggml-dsp.h"

// CONCAT: concatenate two tensors along a given axis.
// src0 = first input, src1 = second input
// dst  = concatenated output
// Axis in dst->op_params[0]

int ggmlop_dsp_concat(remote_handle64 h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) {
    GGML_UNUSED(h);

    uint64_t begin_time = ggml_time_us();

    int axis = 0;
    memcpy(&axis, dst->op_params, sizeof(int));

    const int32_t type = src0->type;
    const size_t es = (type == GGML_TYPE_F32) ? sizeof(float) :
                       (type == GGML_TYPE_F16) ? sizeof(uint16_t) :
                       (type == GGML_TYPE_I32) ? sizeof(int32_t) :
                       (type == GGML_TYPE_I16) ? sizeof(int16_t) : sizeof(float);

    // Size of one full inner block (all dims < axis)
    size_t inner_size = es;
    for (int d = 0; d < axis; ++d) inner_size *= dst->ne[d];

    const int64_t n0 = src0->ne[axis]; // slices from src0 along axis
    const int64_t n1 = src1->ne[axis]; // slices from src1 along axis

    char * pdst = (char *)dst->data;

    // Iterate over all outer dimension combinations (dims > axis)
    for (int64_t i3 = 0; i3 < dst->ne[3]; ++i3) {
        if (axis >= 3 && i3 > 0) break; // axis=3: only one outer position
        for (int64_t i2 = 0; i2 < dst->ne[2]; ++i2) {
            if (axis >= 2 && i2 > 0) break; // axis=2: only dim3 as outer
            for (int64_t i1 = 0; i1 < dst->ne[1]; ++i1) {
                if (axis >= 1 && i1 > 0) break; // axis=1: only dims2,3 as outer

                // Compute source pointers for this outer position
                const char * p0 = (const char *)src0->data;
                const char * p1 = (const char *)src1->data;

                if (axis < 3) { p0 += i3 * src0->nb[3]; p1 += i3 * src1->nb[3]; }
                if (axis < 2) { p0 += i2 * src0->nb[2]; p1 += i2 * src1->nb[2]; }
                if (axis < 1) { p0 += i1 * src0->nb[1]; p1 += i1 * src1->nb[1]; }

                char * pd = pdst;
                if (axis < 3) pd += i3 * dst->nb[3];
                if (axis < 2) pd += i2 * dst->nb[2];
                if (axis < 1) pd += i1 * dst->nb[1];

                // Copy n0 slices from src0
                for (int64_t s = 0; s < n0; ++s) {
                    memcpy(pd, p0, inner_size);
                    if (axis == 0) { p0 += inner_size; pd += inner_size; }
                    else { p0 += src0->nb[axis]; pd += dst->nb[axis]; }
                }
                // Copy n1 slices from src1
                for (int64_t s = 0; s < n1; ++s) {
                    memcpy(pd, p1, inner_size);
                    if (axis == 0) { p1 += inner_size; pd += inner_size; }
                    else { p1 += src1->nb[axis]; pd += dst->nb[axis]; }
                }
            }
        }
    }

    int64_t end_time = ggml_time_us();
    GGMLHEXAGON_LOG_INFO("elapse time of CONCAT is %lld us (axis=%d)",
                         (long long)(end_time - begin_time), axis);
    return 0;
}
