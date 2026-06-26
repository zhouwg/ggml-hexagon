#include "ggml-dsp.h"
#include <string.h>

int ggmlop_dsp_cpy(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    uint64_t begin_time = ggml_time_us();

    const int64_t n = ggml_nelements(dst);
    if (n <= 0 || !src0->data || !dst->data) {
        int64_t end_time = ggml_time_us();
        GGMLHEXAGON_LOG_INFO("elapse time of CPY is %lld us (empty)", (long long)(end_time - begin_time));
        return 0;
    }

    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    const int64_t ne0  = dst->ne[0],  ne1  = dst->ne[1],  ne2  = dst->ne[2],  ne3  = dst->ne[3];
    const size_t  nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t  nb0  = dst->nb[0],  nb1  = dst->nb[1],  nb2  = dst->nb[2],  nb3  = dst->nb[3];

    const size_t src_es = (src0->type == GGML_TYPE_F32) ? sizeof(float) : sizeof(uint16_t);
    const size_t dst_es = (dst->type == GGML_TYPE_F32) ? sizeof(float) : sizeof(uint16_t);

    const bool src0_contig = (nb00 == src_es) &&
                             (nb01 == (size_t)ne00 * nb00) &&
                             (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    const bool dst_contig  = (nb0  == dst_es) &&
                             (nb1  == (size_t)ne0  * nb0) &&
                             (nb2  == (size_t)ne1  * nb1) &&
                             (nb3  == (size_t)ne2  * nb2);

    // Both contiguous: flat linear copy (matches CPU ggml_compute_forward_dup_same_cont)
    if (src0_contig && dst_contig) {
        const int64_t total = ne00 * ne01 * ne02 * ne03;
        if (src0->type == dst->type) {
            memcpy(dst->data, src0->data, (size_t)total * src_es);
        } else if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            const int64_t total = ne00 * ne01 * ne02 * ne03;
            const float   * s = (const float   *)src0->data;
            uint16_t * d = (uint16_t *)dst->data;
            for (int64_t i = 0; i < total; i++) {
                d[i] = ggml_compute_fp32_to_fp16(s[i]);
            }
        } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            const int64_t total = ne00 * ne01 * ne02 * ne03;
            const uint16_t * s = (const uint16_t *)src0->data;
            float    * d = (float    *)dst->data;
            for (int64_t i = 0; i < total; i++) {
                d[i] = ggml_compute_fp16_to_fp32(s[i]);
            }
        }
    }
    // dst contiguous: iterate src0 dims, linear write to dst
    else if (dst_contig) {
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            float * dst_ptr = (float *)dst->data;
            size_t id = 0;
            for (int64_t i03 = 0; i03 < ne03; i03++) {
                for (int64_t i02 = 0; i02 < ne02; i02++) {
                    for (int64_t i01 = 0; i01 < ne01; i01++) {
                        for (int64_t i00 = 0; i00 < ne00; i00++) {
                            const float * s = (const float *)((const char *)src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                            dst_ptr[id++] = *s;
                        }
                    }
                }
            }
        } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
            uint16_t * dst_ptr = (uint16_t *)dst->data;
            size_t id = 0;
            for (int64_t i03 = 0; i03 < ne03; i03++) {
                for (int64_t i02 = 0; i02 < ne02; i02++) {
                    for (int64_t i01 = 0; i01 < ne01; i01++) {
                        for (int64_t i00 = 0; i00 < ne00; i00++) {
                            const uint16_t * s = (const uint16_t *)((const char *)src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                            dst_ptr[id++] = *s;
                        }
                    }
                }
            }
        } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            float * dst_ptr = (float *)dst->data;
            size_t id = 0;
            for (int64_t i03 = 0; i03 < ne03; i03++) {
                for (int64_t i02 = 0; i02 < ne02; i02++) {
                    for (int64_t i01 = 0; i01 < ne01; i01++) {
                        for (int64_t i00 = 0; i00 < ne00; i00++) {
                            const uint16_t * s = (const uint16_t *)((const char *)src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                            dst_ptr[id++] = ggml_compute_fp16_to_fp32(*s);
                        }
                    }
                }
            }
        } else if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            uint16_t * dst_ptr = (uint16_t *)dst->data;
            size_t id = 0;
            for (int64_t i03 = 0; i03 < ne03; i03++) {
                for (int64_t i02 = 0; i02 < ne02; i02++) {
                    for (int64_t i01 = 0; i01 < ne01; i01++) {
                        for (int64_t i00 = 0; i00 < ne00; i00++) {
                            const float * s = (const float *)((const char *)src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                            dst_ptr[id++] = ggml_compute_fp32_to_fp16(*s);
                        }
                    }
                }
            }
        }
    }
    // General path: both non-contiguous
    else {
        int64_t i10 = 0, i11 = 0, i12 = 0, i13 = 0;

        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            for (int64_t i03 = 0; i03 < ne03; i03++) {
                for (int64_t i02 = 0; i02 < ne02; i02++) {
                    for (int64_t i01 = 0; i01 < ne01; i01++) {
                        for (int64_t i00 = 0; i00 < ne00; i00++) {
                            const float * s = (const float *)((const char *)src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                            float * d = (float *)((char *)dst->data + i10*nb0 + i11*nb1 + i12*nb2 + i13*nb3);
                            *d = *s;
                            if (++i10 == ne0) { i10 = 0; if (++i11 == ne1) { i11 = 0; if (++i12 == ne2) { i12 = 0; if (++i13 == ne3) { i13 = 0; } } } }
                        }
                    }
                }
            }
        } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
            for (int64_t i03 = 0; i03 < ne03; i03++) {
                for (int64_t i02 = 0; i02 < ne02; i02++) {
                    for (int64_t i01 = 0; i01 < ne01; i01++) {
                        for (int64_t i00 = 0; i00 < ne00; i00++) {
                            const uint16_t * s = (const uint16_t *)((const char *)src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                            uint16_t * d = (uint16_t *)((char *)dst->data + i10*nb0 + i11*nb1 + i12*nb2 + i13*nb3);
                            *d = *s;
                            if (++i10 == ne0) { i10 = 0; if (++i11 == ne1) { i11 = 0; if (++i12 == ne2) { i12 = 0; if (++i13 == ne3) { i13 = 0; } } } }
                        }
                    }
                }
            }
        } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            for (int64_t i03 = 0; i03 < ne03; i03++) {
                for (int64_t i02 = 0; i02 < ne02; i02++) {
                    for (int64_t i01 = 0; i01 < ne01; i01++) {
                        for (int64_t i00 = 0; i00 < ne00; i00++) {
                            const uint16_t * s = (const uint16_t *)((const char *)src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                            float * d = (float *)((char *)dst->data + i10*nb0 + i11*nb1 + i12*nb2 + i13*nb3);
                            *d = ggml_compute_fp16_to_fp32(*s);
                            if (++i10 == ne0) { i10 = 0; if (++i11 == ne1) { i11 = 0; if (++i12 == ne2) { i12 = 0; if (++i13 == ne3) { i13 = 0; } } } }
                        }
                    }
                }
            }
        } else if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            for (int64_t i03 = 0; i03 < ne03; i03++) {
                for (int64_t i02 = 0; i02 < ne02; i02++) {
                    for (int64_t i01 = 0; i01 < ne01; i01++) {
                        for (int64_t i00 = 0; i00 < ne00; i00++) {
                            const float * s = (const float *)((const char *)src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                            uint16_t * d = (uint16_t *)((char *)dst->data + i10*nb0 + i11*nb1 + i12*nb2 + i13*nb3);
                            *d = ggml_compute_fp32_to_fp16(*s);
                            if (++i10 == ne0) { i10 = 0; if (++i11 == ne1) { i11 = 0; if (++i12 == ne2) { i12 = 0; if (++i13 == ne3) { i13 = 0; } } } }
                        }
                    }
                }
            }
        }
    }

    int64_t end_time = ggml_time_us();
    GGMLHEXAGON_LOG_INFO("elapse time of CPY is %lld us", (long long)(end_time - begin_time));
    return 0;
}
