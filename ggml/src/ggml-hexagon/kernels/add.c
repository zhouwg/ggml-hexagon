#include "ggml-dsp.h"

static inline void l2fetch(const void * p, uint32_t stride,
                           uint32_t width, uint32_t height,
                           uint32_t dir) {
    uint64_t control = HEXAGON_V64_CREATE_H(dir, stride, width, height);
    __asm__ __volatile__ (" l2fetch(%0,%1) " : :"r"(p),"r"(control));
}

static inline void ggmlhexagon_dsp_add_f32(const int n, float * GGML_RESTRICT z, const float * GGML_RESTRICT x, const float * GGML_RESTRICT y) {
    for (size_t i = 0; i < n; ++i) {
        z[i] = x[i] + y[i];
    }
}


static void ggml_compute_forward_add_f32(
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );
    uint64_t start_time = ggml_time_us();

    ggmlhexagon_dump_tensor(src0, 1);
    ggmlhexagon_dump_tensor(src1, 1);
    ggmlhexagon_dump_tensor(dst, 1);

    const int n = ggml_nelements(src0);

    if (src0->type == GGML_TYPE_F16) {
        uint16_t * dst_ptr  = (uint16_t *)dst->data;
        uint16_t * src0_ptr = (uint16_t *)src0->data;
        uint16_t * src1_ptr = (uint16_t *)src1->data;

        for (int i = 0; i < n; ++i) {
            float f0 = ggml_compute_fp16_to_fp32(src0_ptr[i]);
            float f1 = ggml_compute_fp16_to_fp32(src1_ptr[i]);
            float f_result = f0 + f1;
            dst_ptr[i] = ggml_compute_fp32_to_fp16(f_result);
        }
    } else {
        float * dst_ptr  = (float *)dst->data;
        float * src0_ptr = (float *)src0->data;
        float * src1_ptr = (float *)src1->data;

        for (int i = 0; i < n; ++i) {
            dst_ptr[i] = src0_ptr[i] + src1_ptr[i];
        }
    }

    uint64_t end_time = ggml_time_us();
    uint64_t duration = (end_time - start_time);
    GGMLHEXAGON_LOG_DEBUG("duration %llu us", duration);
#if !GGMLHEXAGON_DEBUG
    UNUSED(duration);
#endif

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
}

//FIXME: why failed with test-backend-ops when disable ion rpc mempool
int ggmlop_dsp_add(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s\n", __func__);
    ggml_compute_forward_add_f32(src0, src1, dst);
    GGMLHEXAGON_LOG_DEBUG("leave %s\n", __func__);
    return 0;
}
