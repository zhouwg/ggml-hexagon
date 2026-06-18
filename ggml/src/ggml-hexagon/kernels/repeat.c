#include "ggml-dsp.h"

// REPEAT: broadcast/repeat tensor to match larger shape.
// src0 = input tensor (smaller), dst = output tensor (larger, repeated)
// For each element in dst[i0,i1,i2,i3], copy from src0[i0%ne0, i1%ne1, i2%ne2, i3%ne3]

int ggmlop_dsp_repeat(remote_handle64 h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) {
    GGML_UNUSED(h); GGML_UNUSED(src1);

    uint64_t begin_time = ggml_time_us();

    const int64_t ne0 = src0->ne[0], ne1 = src0->ne[1], ne2 = src0->ne[2], ne3 = src0->ne[3];
    const int64_t dne0 = dst->ne[0], dne1 = dst->ne[1], dne2 = dst->ne[2], dne3 = dst->ne[3];

    const int32_t type = src0->type;
    const size_t es = (type == GGML_TYPE_F32) ? sizeof(float) :
                       (type == GGML_TYPE_F16) ? sizeof(uint16_t) :
                       (type == GGML_TYPE_I32) ? sizeof(int32_t) :
                       (type == GGML_TYPE_I16) ? sizeof(int16_t) : 1;

    const char * ps = (const char *)src0->data;
    char       * pd = (char *)dst->data;

    // Generic 4D repeat using modulo indexing
    // Optimize: process one "row" (dim 0) at a time, tiling from source
    for (int64_t i3 = 0; i3 < dne3; ++i3) {
        const int64_t s3 = i3 % ne3;
        for (int64_t i2 = 0; i2 < dne2; ++i2) {
            const int64_t s2 = i2 % ne2;
            for (int64_t i1 = 0; i1 < dne1; ++i1) {
                const int64_t s1 = i1 % ne1;
                const char * base_src = ps + s3 * src0->nb[3] + s2 * src0->nb[2] + s1 * src0->nb[1];
                char       * base_dst = pd + i3 * dst->nb[3] + i2 * dst->nb[2] + i1 * dst->nb[1];

                // Tile the source row (ne0 elements) across the destination row (dne0 elements)
                for (int64_t off = 0; off < dne0; off += ne0) {
                    size_t copy_len = (off + ne0 <= dne0) ? ne0 : (dne0 - off);
                    memcpy(base_dst + off * es, base_src + (off % ne0) * es, copy_len * es);
                }
            }
        }
    }

    int64_t end_time = ggml_time_us();
    GGMLHEXAGON_LOG_INFO("elapse time of REPEAT is %lld us",
                         (long long)(end_time - begin_time));
    return 0;
}
