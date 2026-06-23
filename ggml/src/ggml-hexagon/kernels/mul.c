#include "ggml-dsp.h"
#include "worker_pool.h"
#include "../htp/hvx-base.h"  // for hvx_vec_mul_f16_f16

typedef struct {
    const ggml_tensor * src0;
    const ggml_tensor * src1;
    ggml_tensor * dst;
    int64_t start_idx;
    int64_t end_idx;
    worker_synctoken_t *synctoken;
} mul_thread_data_t;

static HVX_INLINE_ALWAYS void l2fetch(const void * p, uint32_t stride,
                           uint32_t width, uint32_t height,
                           uint32_t dir) {
    uint64_t control = HEXAGON_V64_CREATE_H(dir, stride, width, height);
    __asm__ __volatile__ (" l2fetch(%0,%1) " : :"r"(p),"r"(control));
}

/* HVX-accelerated F32 element-wise multiply.
 * NOTE: z and x may alias (in-place operation z==x is allowed).
 * Do NOT use GGML_RESTRICT here -- the compiler may reorder reads/writes
 * assuming no aliasing, which produces wrong results when z==x. */
static inline void ggml_mul_f32_hvx(const int n, float * z,
                                     const float * x,
                                     const float * y) {
    const size_t FLOATS_PER_VECTOR = 128 / sizeof(float);
    const size_t block = n / FLOATS_PER_VECTOR;
    const size_t left  = n % FLOATS_PER_VECTOR;

    if ((((uintptr_t)z | (uintptr_t)x | (uintptr_t)y) % ALIGN_128_BYTE) != 0) {
        for (int i = 0; i < n; ++i) z[i] = x[i] * y[i];
        return;
    }

    HVX_Vector * va = (HVX_Vector *)x;
    HVX_Vector * vb = (HVX_Vector *)y;
    HVX_Vector * vc = (HVX_Vector *)z;

    int fetch_counts = 1;
    if (0 == (n % (128 * 8)))
        fetch_counts = 8;
    else if (0 == (n % (128 * 4)))
        fetch_counts = 4;
    else if (0 == (n % (128 * 2)))
        fetch_counts = 2;

    for (size_t i = 0; i < block; i += fetch_counts) {
        l2fetch((void*)((uint8_t*)va + VLEN * fetch_counts), VLEN, VLEN * fetch_counts, 1, 0);
        l2fetch((void*)((uint8_t*)vb + VLEN * fetch_counts), VLEN, VLEN * fetch_counts, 1, 0);

        #pragma unroll(4)
        for (size_t j = 0; j < (size_t)fetch_counts && i + j < block; j++) {
#if __HEXAGON_ARCH__ >= 79
            *vc++ = Q6_Vsf_vmpy_VsfVsf(*va++, *vb++);
#else
            *vc++ = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(*va++, *vb++));
#endif
        }
    }

    if (left > 0) {
        const size_t off = block * FLOATS_PER_VECTOR;
        for (size_t i = 0; i < left; ++i)
            z[i + off] = x[i + off] * y[i + off];
    }
}

/* Scalar F16 multiply: fp16->fp32->mul->fp16 for each element.
 * Matches the precision of the reference implementation. */
static inline void ggml_mul_f16_scalar(const int n, uint16_t * z,
                                        const uint16_t * x,
                                        const uint16_t * y) {
    for (int i = 0; i < n; ++i) {
        float f0 = ggml_compute_fp16_to_fp32(x[i]);
        float f1 = ggml_compute_fp16_to_fp32(y[i]);
        z[i] = ggml_compute_fp32_to_fp16(f0 * f1);
    }
}

/* Broadcast-aware row index computation.
 * Decompose row index using dst dimensions, then modulo for src broadcast. */
static inline void mul_compute_row_ptrs(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst,
        int64_t ir,
        const uint8_t ** src0_row, const uint8_t ** src1_row, uint8_t ** dst_row) {
    const int64_t ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];

    const int64_t i3 = ir / (ne2 * ne1);
    const int64_t i2 = (ir - i3 * ne2 * ne1) / ne1;
    const int64_t i1 = ir - i3 * ne2 * ne1 - i2 * ne1;

    const int64_t s0_3 = i3 % src0->ne[3];
    const int64_t s0_2 = i2 % src0->ne[2];
    const int64_t s0_1 = i1 % src0->ne[1];

    const int64_t s1_3 = i3 % src1->ne[3];
    const int64_t s1_2 = i2 % src1->ne[2];
    const int64_t s1_1 = i1 % src1->ne[1];

    *dst_row  = (uint8_t *)dst->data  + i3*dst->nb[3]  + i2*dst->nb[2]  + i1*dst->nb[1];
    *src0_row = (const uint8_t *)src0->data + s0_3*src0->nb[3] + s0_2*src0->nb[2] + s0_1*src0->nb[1];
    *src1_row = (const uint8_t *)src1->data + s1_3*src1->nb[3] + s1_2*src1->nb[2] + s1_1*src1->nb[1];
}

static void mul_thread_func_vtcm(void * data) {
    mul_thread_data_t * tdata = (mul_thread_data_t *) data;
    const ggml_tensor * src0 = tdata->src0;
    const ggml_tensor * src1 = tdata->src1;
    ggml_tensor * dst = tdata->dst;
    const int64_t start_idx = tdata->start_idx;
    const int64_t end_idx = tdata->end_idx;

    const int64_t ne0 = dst->ne[0], ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2], ne3 = dst->ne[3];
    const int64_t nr  = ne1 * ne2 * ne3;

    /* row_based: no broadcast in dim 0, can process full rows */
    const bool row_based = (src0->ne[0] == ne0 && src1->ne[0] == ne0 &&
                            src0->nb[0] == dst->nb[0] && src1->nb[0] == dst->nb[0]);

    if (src0->type == GGML_TYPE_F16) {
        if (row_based) {
            const int64_t ir0 = start_idx / ne0;
            const int64_t ir1 = (end_idx + ne0 - 1) / ne0;

            for (int64_t ir = ir0; ir < ir1 && ir < nr; ++ir) {
                const uint8_t * src0_row, * src1_row;
                uint8_t * dst_row;
                mul_compute_row_ptrs(src0, src1, dst, ir, &src0_row, &src1_row, &dst_row);

                ggml_mul_f16_scalar(ne0, (uint16_t *)dst_row,
                                    (const uint16_t *)src0_row,
                                    (const uint16_t *)src1_row);
            }
        } else {
            for (int64_t i = start_idx; i < end_idx; ++i) {
                int64_t i0 = i % ne0;
                int64_t r  = i / ne0;
                int64_t i1 = r % ne1;
                int64_t r2 = r / ne1;
                int64_t i2 = r2 % ne2;
                int64_t i3 = r2 / ne2;

                int64_t s0_0 = i0 % src0->ne[0], s0_1 = i1 % src0->ne[1];
                int64_t s0_2 = i2 % src0->ne[2], s0_3 = i3 % src0->ne[3];
                int64_t s1_0 = i0 % src1->ne[0], s1_1 = i1 % src1->ne[1];
                int64_t s1_2 = i2 % src1->ne[2], s1_3 = i3 % src1->ne[3];

                int64_t off0 = s0_0*src0->nb[0] + s0_1*src0->nb[1] + s0_2*src0->nb[2] + s0_3*src0->nb[3];
                int64_t off1 = s1_0*src1->nb[0] + s1_1*src1->nb[1] + s1_2*src1->nb[2] + s1_3*src1->nb[3];
                int64_t offd = i0*dst->nb[0]   + i1*dst->nb[1]   + i2*dst->nb[2]   + i3*dst->nb[3];

                float f0 = ggml_compute_fp16_to_fp32(*(const uint16_t *)((const uint8_t *)src0->data + off0));
                float f1 = ggml_compute_fp16_to_fp32(*(const uint16_t *)((const uint8_t *)src1->data + off1));
                *(uint16_t *)((uint8_t *)dst->data + offd) = ggml_compute_fp32_to_fp16(f0 * f1);
            }
        }
    } else {
        if (row_based) {
            const int64_t ir0 = start_idx / ne0;
            const int64_t ir1 = (end_idx + ne0 - 1) / ne0;

            for (int64_t ir = ir0; ir < ir1 && ir < nr; ++ir) {
                const uint8_t * src0_row, * src1_row;
                uint8_t * dst_row;
                mul_compute_row_ptrs(src0, src1, dst, ir, &src0_row, &src1_row, &dst_row);

                ggml_mul_f32_hvx(ne0, (float *)dst_row,
                                  (const float *)src0_row,
                                  (const float *)src1_row);
            }
        } else {
            for (int64_t i = start_idx; i < end_idx; ++i) {
                int64_t i0 = i % ne0;
                int64_t r  = i / ne0;
                int64_t i1 = r % ne1;
                int64_t r2 = r / ne1;
                int64_t i2 = r2 % ne2;
                int64_t i3 = r2 / ne2;

                int64_t s0_0 = i0 % src0->ne[0], s0_1 = i1 % src0->ne[1];
                int64_t s0_2 = i2 % src0->ne[2], s0_3 = i3 % src0->ne[3];
                int64_t s1_0 = i0 % src1->ne[0], s1_1 = i1 % src1->ne[1];
                int64_t s1_2 = i2 % src1->ne[2], s1_3 = i3 % src1->ne[3];

                int64_t off0 = s0_0*src0->nb[0] + s0_1*src0->nb[1] + s0_2*src0->nb[2] + s0_3*src0->nb[3];
                int64_t off1 = s1_0*src1->nb[0] + s1_1*src1->nb[1] + s1_2*src1->nb[2] + s1_3*src1->nb[3];
                int64_t offd = i0*dst->nb[0]   + i1*dst->nb[1]   + i2*dst->nb[2]   + i3*dst->nb[3];

                *(float *)((uint8_t *)dst->data + offd) = *(const float *)((const uint8_t *)src0->data + off0) * *(const float *)((const uint8_t *)src1->data + off1);
            }
        }
    }

    if (tdata->synctoken != NULL) {
        worker_pool_synctoken_jobdone(tdata->synctoken);
    }
}

/* Single-threaded MUL */
static void ggml_mul_singlethread(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t ne0 = dst->ne[0], ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2], ne3 = dst->ne[3];
    const int64_t nr  = ne1 * ne2 * ne3;

    const bool row_based = (src0->ne[0] == ne0 && src1->ne[0] == ne0 &&
                            src0->nb[0] == dst->nb[0] && src1->nb[0] == dst->nb[0]);

    if (src0->type == GGML_TYPE_F16) {
        if (row_based) {
            for (int64_t ir = 0; ir < nr; ++ir) {
                const uint8_t * src0_row, * src1_row;
                uint8_t * dst_row;
                mul_compute_row_ptrs(src0, src1, dst, ir, &src0_row, &src1_row, &dst_row);

                ggml_mul_f16_scalar(ne0, (uint16_t *)dst_row,
                                    (const uint16_t *)src0_row,
                                    (const uint16_t *)src1_row);
            }
        } else {
            const int64_t n = ggml_nelements(dst);
            for (int64_t i = 0; i < n; ++i) {
                int64_t i0 = i % ne0;
                int64_t r  = i / ne0;
                int64_t i1 = r % ne1;
                int64_t r2 = r / ne1;
                int64_t i2 = r2 % ne2;
                int64_t i3 = r2 / ne2;

                int64_t s0_0 = i0 % src0->ne[0], s0_1 = i1 % src0->ne[1];
                int64_t s0_2 = i2 % src0->ne[2], s0_3 = i3 % src0->ne[3];
                int64_t s1_0 = i0 % src1->ne[0], s1_1 = i1 % src1->ne[1];
                int64_t s1_2 = i2 % src1->ne[2], s1_3 = i3 % src1->ne[3];

                int64_t off0 = s0_0*src0->nb[0] + s0_1*src0->nb[1] + s0_2*src0->nb[2] + s0_3*src0->nb[3];
                int64_t off1 = s1_0*src1->nb[0] + s1_1*src1->nb[1] + s1_2*src1->nb[2] + s1_3*src1->nb[3];
                int64_t offd = i0*dst->nb[0]   + i1*dst->nb[1]   + i2*dst->nb[2]   + i3*dst->nb[3];

                float f0 = ggml_compute_fp16_to_fp32(*(const uint16_t *)((const uint8_t *)src0->data + off0));
                float f1 = ggml_compute_fp16_to_fp32(*(const uint16_t *)((const uint8_t *)src1->data + off1));
                *(uint16_t *)((uint8_t *)dst->data + offd) = ggml_compute_fp32_to_fp16(f0 * f1);
            }
        }
    } else {
        if (row_based) {
            for (int64_t ir = 0; ir < nr; ++ir) {
                const uint8_t * src0_row, * src1_row;
                uint8_t * dst_row;
                mul_compute_row_ptrs(src0, src1, dst, ir, &src0_row, &src1_row, &dst_row);

                ggml_mul_f32_hvx(ne0, (float *)dst_row,
                                  (const float *)src0_row,
                                  (const float *)src1_row);
            }
        } else {
            const int64_t n = ggml_nelements(dst);
            for (int64_t i = 0; i < n; ++i) {
                int64_t i0 = i % ne0;
                int64_t r  = i / ne0;
                int64_t i1 = r % ne1;
                int64_t r2 = r / ne1;
                int64_t i2 = r2 % ne2;
                int64_t i3 = r2 / ne2;

                int64_t s0_0 = i0 % src0->ne[0], s0_1 = i1 % src0->ne[1];
                int64_t s0_2 = i2 % src0->ne[2], s0_3 = i3 % src0->ne[3];
                int64_t s1_0 = i0 % src1->ne[0], s1_1 = i1 % src1->ne[1];
                int64_t s1_2 = i2 % src1->ne[2], s1_3 = i3 % src1->ne[3];

                int64_t off0 = s0_0*src0->nb[0] + s0_1*src0->nb[1] + s0_2*src0->nb[2] + s0_3*src0->nb[3];
                int64_t off1 = s1_0*src1->nb[0] + s1_1*src1->nb[1] + s1_2*src1->nb[2] + s1_3*src1->nb[3];
                int64_t offd = i0*dst->nb[0]   + i1*dst->nb[1]   + i2*dst->nb[2]   + i3*dst->nb[3];

                *(float *)((uint8_t *)dst->data + offd) = *(const float *)((const uint8_t *)src0->data + off0) * *(const float *)((const uint8_t *)src1->data + off1);
            }
        }
    }
}

static int ggmlop_dsp_mul_singlethread(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_UNUSED(h);
    ggml_mul_singlethread(src0, src1, dst);
    return 0;
}

static int ggmlop_dsp_mul_multithread(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_UNUSED(h);

    const int64_t ne0 = dst->ne[0];
    const int64_t nr  = dst->ne[1] * dst->ne[2] * dst->ne[3];
    const int64_t n   = ne0 * nr;
    int num_threads = num_workers;

    if (src0->type == GGML_TYPE_F32) {
        num_threads = 2;
    } else if (src0->type == GGML_TYPE_F16) {
        num_threads = ggml_min(num_workers, 6);
    } else {
        num_threads = num_workers;
    }

    if (num_threads <= 1 || n < num_threads * 512 || nr < num_threads) {
        return ggmlop_dsp_mul_singlethread(h, src0, src1, dst);
    }

    worker_synctoken_t synctoken;
    worker_pool_synctoken_init(&synctoken, num_threads - 1);

    mul_thread_data_t tdata[num_threads];
    /* Divide work by rows (not flat elements) to prevent row overlap.
     * If ne_per_thread is not a multiple of ne0, two adjacent threads
     * can process the same row, causing a race condition. */
    const int64_t rows_per_thread = (nr + num_threads - 1) / num_threads;
    int64_t start_row = 0;

    for (int i = 0; i < num_threads - 1; ++i) {
        int64_t end_row = start_row + rows_per_thread;
        if (end_row > nr)
            end_row = nr;

        tdata[i].src0 = src0;
        tdata[i].src1 = src1;
        tdata[i].dst = dst;
        tdata[i].start_idx = start_row * ne0;
        tdata[i].end_idx = end_row * ne0;
        tdata[i].synctoken = &synctoken;

        worker_pool_job_t job;
        job.fptr = mul_thread_func_vtcm;
        job.dptr = &tdata[i];
        worker_pool_submit(NULL, job);

        start_row = end_row;
    }

    tdata[num_threads - 1].src0 = src0;
    tdata[num_threads - 1].src1 = src1;
    tdata[num_threads - 1].dst = dst;
    tdata[num_threads - 1].start_idx = start_row * ne0;
    tdata[num_threads - 1].end_idx = nr * ne0;
    tdata[num_threads - 1].synctoken = NULL;

    mul_thread_func_vtcm(&tdata[num_threads - 1]);

    worker_pool_synctoken_wait(&synctoken);

    return 0;
}

int ggmlop_dsp_mul(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    char tempbuf[256];
    ggmlhexagon_get_opkey(GGML_OP_MUL, src0, src1, tempbuf, 256);
    int64_t begin_time = ggml_time_us();

    if (ggmlop_get_thread_counts() > 1) {
        ggmlop_dsp_mul_multithread(h, src0, src1, dst);
    } else {
        ggmlop_dsp_mul_singlethread(h, src0, src1, dst);
    }

    int64_t end_time = ggml_time_us();
    GGMLHEXAGON_LOG_INFO("elapse time of %s is %lld us", tempbuf, (long long)(end_time - begin_time));
    return 0;
}
