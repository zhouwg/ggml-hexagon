#include "ggml-dsp.h"
#include <string.h>

int ggmlop_dsp_cpy(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s\n", __func__);
    uint64_t begin_time = ggml_time_us();

    const int64_t n = ggml_nelements(dst);
    if (n <= 0 || !src0->data || !dst->data) {
        int64_t end_time = ggml_time_us();
        GGMLHEXAGON_LOG_INFO("elapse time of CPY is %lld us (empty)", (long long)(end_time - begin_time));
        return 0;
    }

    const int64_t ne0 = dst->ne[0], ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2], ne3 = dst->ne[3];

    // Determine element sizes
    size_t src_elem_size = (src0->type == GGML_TYPE_F32) ? sizeof(float) : sizeof(uint16_t);
    size_t dst_elem_size = (dst->type == GGML_TYPE_F32) ? sizeof(float) : sizeof(uint16_t);

    // Check if both src and dst are row-major contiguous
    bool src_contig = (src0->nb[0] == src_elem_size &&
                       src0->nb[1] == src0->ne[0] * src_elem_size &&
                       src0->nb[2] >= src0->ne[1] * src0->nb[1] &&
                       src0->nb[3] >= src0->ne[2] * src0->nb[2]);
    bool dst_contig = (dst->nb[0] == dst_elem_size &&
                       dst->nb[1] == dst->ne[0] * dst_elem_size &&
                       dst->nb[2] >= dst->ne[1] * dst->nb[1] &&
                       dst->nb[3] >= dst->ne[2] * dst->nb[2]);

    if (src0->type == dst->type && src_contig && dst_contig) {
        // Fast path: both contiguous, same type -> raw byte memcpy
        memcpy(dst->data, src0->data, n * src_elem_size);
    } else if (!src_contig || !dst_contig) {
        // Slow path: one or both are non-contiguous; use stride-based element copy
        if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
            uint16_t * raw_dst = (uint16_t *)dst->data;
            uint16_t * raw_src = (uint16_t *)src0->data;
            for (int64_t i = 0; i < n; ++i) {
                int64_t i0 = i % ne0;
                int64_t r  = i / ne0;
                int64_t i1 = r % ne1;
                int64_t r2 = r / ne1;
                int64_t i2 = r2 % ne2;
                int64_t i3 = r2 / ne2;

                int64_t soff = i0*src0->nb[0] + i1*src0->nb[1] + i2*src0->nb[2] + i3*src0->nb[3];
                int64_t doff = i0*dst->nb[0]   + i1*dst->nb[1]   + i2*dst->nb[2]   + i3*dst->nb[3];
                raw_dst[doff >> 1] = raw_src[soff >> 1];
            }
        } else if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            float * raw_dst = (float *)dst->data;
            float * raw_src = (float *)src0->data;
            for (int64_t i = 0; i < n; ++i) {
                int64_t i0 = i % ne0;
                int64_t r  = i / ne0;
                int64_t i1 = r % ne1;
                int64_t r2 = r / ne1;
                int64_t i2 = r2 % ne2;
                int64_t i3 = r2 / ne2;

                int64_t soff = i0*src0->nb[0] + i1*src0->nb[1] + i2*src0->nb[2] + i3*src0->nb[3];
                int64_t doff = i0*dst->nb[0]   + i1*dst->nb[1]   + i2*dst->nb[2]   + i3*dst->nb[3];
                raw_dst[doff >> 2] = raw_src[soff >> 2];
            }
        } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            float * raw_dst = (float *)dst->data;
            uint16_t * raw_src = (uint16_t *)src0->data;
            for (int64_t i = 0; i < n; ++i) {
                int64_t i0 = i % ne0;
                int64_t r  = i / ne0;
                int64_t i1 = r % ne1;
                int64_t r2 = r / ne1;
                int64_t i2 = r2 % ne2;
                int64_t i3 = r2 / ne2;

                int64_t soff = i0*src0->nb[0] + i1*src0->nb[1] + i2*src0->nb[2] + i3*src0->nb[3];
                int64_t doff = i0*dst->nb[0]   + i1*dst->nb[1]   + i2*dst->nb[2]   + i3*dst->nb[3];
                raw_dst[doff >> 2] = ggml_compute_fp16_to_fp32(raw_src[soff >> 1]);
            }
        } else if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            uint16_t * raw_dst = (uint16_t *)dst->data;
            float * raw_src = (float *)src0->data;
            for (int64_t i = 0; i < n; ++i) {
                int64_t i0 = i % ne0;
                int64_t r  = i / ne0;
                int64_t i1 = r % ne1;
                int64_t r2 = r / ne1;
                int64_t i2 = r2 % ne2;
                int64_t i3 = r2 / ne2;

                int64_t soff = i0*src0->nb[0] + i1*src0->nb[1] + i2*src0->nb[2] + i3*src0->nb[3];
                int64_t doff = i0*dst->nb[0]   + i1*dst->nb[1]   + i2*dst->nb[2]   + i3*dst->nb[3];
                raw_dst[doff >> 1] = ggml_compute_fp32_to_fp16(raw_src[soff >> 2]);
            }
        }
    } else {
        // Contiguous but different types: convert via element loop
        if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            uint16_t * raw_src = (uint16_t *)src0->data;
            float * raw_dst = (float *)dst->data;
            for (int64_t i = 0; i < n; ++i)
                raw_dst[i] = ggml_compute_fp16_to_fp32(raw_src[i]);
        } else if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            float * raw_src = (float *)src0->data;
            uint16_t * raw_dst = (uint16_t *)dst->data;
            for (int64_t i = 0; i < n; ++i)
                raw_dst[i] = ggml_compute_fp32_to_fp16(raw_src[i]);
        }
    }

    int64_t end_time = ggml_time_us();
    GGMLHEXAGON_LOG_INFO("elapse time of CPY is %lld us", (long long)(end_time - begin_time));
    GGMLHEXAGON_LOG_DEBUG("leave %s\n", __func__);
    return 0;
}
