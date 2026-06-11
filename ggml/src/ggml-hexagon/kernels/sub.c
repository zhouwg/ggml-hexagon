#include "ggml-dsp.h"

static inline int ggml_nelements_dsptensor(const struct dsptensor * tensor) {
    int n = 1;
    for (int i = 0; i < 4; i++) {
        n *= tensor->ne[i];
    }
    return n;
}

int ggmlop_dsp_sub(remote_handle64 h, const struct dsptensor * src0, const struct dsptensor * src1, struct dsptensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__);

    const int n = ggml_nelements_dsptensor(src0);

    if (src0->type == GGML_TYPE_F32) {
        float * dst_ptr  = (float *)dst->data;
        float * src0_ptr = (float *)src0->data;
        float * src1_ptr = (float *)src1->data;

        for (int i = 0; i < n; ++i) {
            dst_ptr[i] = src0_ptr[i] - src1_ptr[i];
        }
    } else if (src0->type == GGML_TYPE_F16) {
        uint16_t * dst_ptr  = (uint16_t *)dst->data;
        uint16_t * src0_ptr = (uint16_t *)src0->data;
        uint16_t * src1_ptr = (uint16_t *)src1->data;

        for (int i = 0; i < n; ++i) {
            float f0 = ggml_compute_fp16_to_fp32(src0_ptr[i]);
            float f1 = ggml_compute_fp16_to_fp32(src1_ptr[i]);
            dst_ptr[i] = ggml_compute_fp32_to_fp16(f0 - f1);
        }
    }

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__);
    return 0;
}