#include "ggml-dsp.h"
#include "worker_pool.h"

#define VSIZE_BYTES 128
#define VSIZE_WORDS VSIZE_BYTES/4

#define SIZEOF_FP32 (4)
#define SIZEOF_FP16 (2)
#define VLEN_FP32   (VSIZE_BYTES / SIZEOF_FP32)
#define VLEN_FP16   (VSIZE_BYTES / SIZEOF_FP16)

#define VTCM_BLOCK_ROWS 16
#define VTCM_BLOCK_COLS 16

typedef union {
    HVX_Vector v;
    uint8_t    b[VSIZE_BYTES];
    uint16_t   h[VLEN_FP16];
    uint32_t   w[VLEN_FP32];
    __fp16     fp16[VLEN_FP16];
    float      fp32[VLEN_FP32];
} HVX_VectorAlias;

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

static inline float ggml_compute_fp16_to_fp32(uint16_t h) {
    const uint32_t w = (uint32_t)h << 16;
    const uint32_t sign = w & 0x80000000U;
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = 0xE0U << 23;
    const float exp_scale = fp32_from_bits(0x7800000U);
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = 126U << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = 1U << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

static inline float hvx_vec_get_f32(HVX_Vector v) {
    HVX_VectorAlias va;
    va.v = v;
    return va.fp32[0];
}

static inline HVX_Vector hvx_vec_reduce_sum_n_f32(HVX_Vector in, unsigned int n) {
    unsigned int total = n * 4;
    unsigned int width = 4;

    HVX_Vector sum = in, sum_t;
    while (width < total) {
        sum_t = Q6_V_vror_VR(sum, width);
        sum   = Q6_Vsf_vadd_VsfVsf(sum, sum_t);
        width = width << 1;
    }
    return sum;
}

static inline HVX_Vector hvx_vec_reduce_sum_f32(HVX_Vector in) {
    return hvx_vec_reduce_sum_n_f32(in, VLEN_FP32);
}

static void vec_dot_f32_hvx(int n, float *GGML_RESTRICT s, const float *GGML_RESTRICT x, const float *GGML_RESTRICT y) {
    const HVX_Vector * restrict vx = (const HVX_Vector *) x;
    const HVX_Vector * restrict vy = (const HVX_Vector *) y;

    uint32_t nvec = n / VLEN_FP32;
    uint32_t nloe = n % VLEN_FP32;

    HVX_Vector rsum = Q6_V_vsplat_R(0);

    uint32_t i = 0;

    #pragma unroll(4)
    for (i = 0; i < nvec; i++) {
        HVX_Vector prod = Q6_Vsf_vmpy_VsfVsf(vx[i], vy[i]);
        rsum = Q6_Vsf_vadd_VsfVsf(rsum, prod);
    }

    if (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector x_sf = Q6_V_vand_QV(bmask, vx[i]);
        HVX_Vector y_sf = Q6_V_vand_QV(bmask, vy[i]);
        HVX_Vector prod = Q6_Vsf_vmpy_VsfVsf(x_sf, y_sf);
        rsum = Q6_Vsf_vadd_VsfVsf(rsum, prod);
    }

    *s = hvx_vec_get_f32(hvx_vec_reduce_sum_f32(rsum));
}

static void vec_dot_f32(int n, float *GGML_RESTRICT s, size_t bs, const float *GGML_RESTRICT x,
                    size_t bx, const float *GGML_RESTRICT y, size_t by, int nrc) {
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(nrc);

    if (n >= VLEN_FP32 && ((uintptr_t)x & 0x7F) == 0 && ((uintptr_t)y & 0x7F) == 0) {
        vec_dot_f32_hvx(n, s, x, y);
        return;
    }

    ggml_float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float) (x[i] * y[i]);
    }
    *s = sumf;
}

static void vec_dot_f16_f32(int n, float *GGML_RESTRICT s, size_t bs, const uint16_t *GGML_RESTRICT x,
                    size_t bx, const float *GGML_RESTRICT y, size_t by, int nrc) {
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(nrc);

    ggml_float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        float va = ggml_compute_fp16_to_fp32(x[i]);
        float vb = y[i];
        sumf += (ggml_float) (va * vb);
    }
    *s = sumf;
}

static void vec_dot_f16_f16(int n, float *GGML_RESTRICT s, size_t bs, const uint16_t *GGML_RESTRICT x,
                    size_t bx, const uint16_t *GGML_RESTRICT y, size_t by, int nrc) {
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(nrc);

    ggml_float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        float va = ggml_compute_fp16_to_fp32(x[i]);
        float vb = ggml_compute_fp16_to_fp32(y[i]);
        sumf += (ggml_float) (va * vb);
    }
    *s = sumf;
}

typedef struct {
    const ggml_tensor *src0;
    const ggml_tensor *src1;
    ggml_tensor *dst;
    enum ggml_type type;
    int32_t num_rows_per_vec_dot;
    int32_t ir0_start;
    int32_t ir0_end;
    int32_t ir1_start;
    int32_t ir1_end;
    worker_synctoken_t *synctoken;
} mulmat_thread_data_t;

static void ggml_compute_forward_mul_mat_one_chunk(const ggml_tensor *src0, const ggml_tensor *src1,
                                                   struct ggml_tensor *dst,
                                                   const enum ggml_type type,
                                                   const int32_t num_rows_per_vec_dot,
                                                   const int32_t ir0_start, const int32_t ir0_end,
                                                   const int32_t ir1_start, const int32_t ir1_end) {
    const bool src1_cont = ggml_is_contiguous(src1);

    const int32_t ne00 = src0->ne[0];
    const int32_t ne01 = src0->ne[1];
    const int32_t ne02 = src0->ne[2];
    const int32_t ne03 = src0->ne[3];

    const int32_t ne10 = src1->ne[0];
    const int32_t ne11 = src1->ne[1];
    const int32_t ne12 = src1->ne[2];
    const int32_t ne13 = src1->ne[3];

    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    const size_t nb11 = src1->nb[1];
    const size_t nb12 = src1->nb[2];
    const size_t nb13 = src1->nb[3];

    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];
    const size_t nb0 = dst->nb[0];

    const int32_t r2 = ne12 / ne02;
    const int32_t r3 = ne13 / ne03;

    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const void * wdata = src1->data;
    const size_t row_size = 4 * ne10;

    const int32_t blck_0 = 16;
    const int32_t blck_1 = 16;

    const size_t src1_col_stride = src1_cont ? row_size : nb11;

    float tmp[32];

    for (int32_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int32_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int32_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                const int32_t i13 = (ir1 / (ne12 * ne11));
                const int32_t i12 = (ir1 - i13 * ne12 * ne11) / ne11;
                const int32_t i11 = (ir1 - i13 * ne12 * ne11 - i12 * ne11);

                const int32_t i03 = i13 / r3;
                const int32_t i02 = i12 / r2;

                const char * src0_row = (const char*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                const char * src1_col = (const char*)wdata +
                    (src1_cont
                     ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                     : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                float * dst_col = (float*)((char*)dst->data + (i11 * nb1 + i12 * nb2 + i13 * nb3));

                for (int32_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    if (type == GGML_TYPE_F16) {
                        vec_dot_f16_f32(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0),
                                       (uintptr_t)(src0_row + ir0 * nb01), (num_rows_per_vec_dot > 1 ? nb01 : 0),
                                       (float*)src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                    } else {
                        vec_dot_f32(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0),
                                    (float*)(src0_row + ir0 * nb01), (num_rows_per_vec_dot > 1 ? nb01 : 0),
                                    (float*)src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                    }
                }

                for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
                }
            }
        }
    }
}

static void mulmat_thread_func(void * data) {
    mulmat_thread_data_t * tdata = (mulmat_thread_data_t *) data;

    ggml_compute_forward_mul_mat_one_chunk(
        tdata->src0, tdata->src1, tdata->dst,
        tdata->type, tdata->num_rows_per_vec_dot,
        tdata->ir0_start, tdata->ir0_end,
        tdata->ir1_start, tdata->ir1_end
    );

    if (tdata->synctoken != NULL) {
        worker_pool_synctoken_jobdone(tdata->synctoken);
    }
}

static int ggmlop_dsp_mulmat_singlethread(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );

    dst->ne[0] = src0->ne[1];
    dst->ne[1] = src1->ne[1];
    dst->ne[2] = src1->ne[2];
    dst->ne[3] = src1->ne[3];

    dst->nb[0] = 4;
    dst->nb[1] = dst->nb[0] * dst->ne[0];
    dst->nb[2] = dst->nb[1] * dst->ne[1];
    dst->nb[3] = dst->nb[2] * dst->ne[2];

    const int32_t ne0 = src0->ne[1];
    const int32_t ne1 = src1->ne[1];
    const int32_t ne2 = src1->ne[2];
    const int32_t ne3 = src1->ne[3];

    const int32_t nr0 = ne0;
    const int32_t nr1 = ne1 * ne2 * ne3;

    int chunk_size = 16;
    if (nr0 == 1 || nr1 == 1) {
        chunk_size = 64;
    }

    int32_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int32_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    nchunk0 = 1;
    nchunk1 = 1;

    const int32_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int32_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    int current_chunk = 0;

    while (current_chunk < nchunk0 * nchunk1) {
        const int32_t ith0 = current_chunk % nchunk0;
        const int32_t ith1 = current_chunk / nchunk0;

        const int32_t ir0_start = dr0 * ith0;
        const int32_t ir0_end = MIN(ir0_start + dr0, nr0);

        const int32_t ir1_start = dr1 * ith1;
        const int32_t ir1_end = MIN(ir1_start + dr1, nr1);

        int32_t num_rows_per_vec_dot = 1;

        if ((nr0 % 2 != 0) || (ne1 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            num_rows_per_vec_dot = 1;
        }

        ggml_compute_forward_mul_mat_one_chunk(src0, src1, dst, src0->type, num_rows_per_vec_dot,
                                               ir0_start, ir0_end, ir1_start, ir1_end);

        if (1 >= nchunk0 * nchunk1) {
            break;
        }
        current_chunk++;
    }

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return 0;
}

static int ggmlop_dsp_mulmat_multithread(remote_handle64 h, const struct dsptensor * src0, const struct dsptensor * src1, dsptensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );

    dst->ne[0] = src0->ne[1];
    dst->ne[1] = src1->ne[1];
    dst->ne[2] = src1->ne[2];
    dst->ne[3] = src1->ne[3];

    dst->nb[0] = 4;
    dst->nb[1] = dst->nb[0] * dst->ne[0];
    dst->nb[2] = dst->nb[1] * dst->ne[1];
    dst->nb[3] = dst->nb[2] * dst->ne[2];

    const int32_t ne0 = src0->ne[1];
    const int32_t ne1 = src1->ne[1];
    const int32_t ne2 = src1->ne[2];
    const int32_t ne3 = src1->ne[3];

    const int32_t nr0 = ne0;
    const int32_t nr1 = ne1 * ne2 * ne3;

    unsigned int n_threads = num_workers;
    if (n_threads < 1) n_threads = 1;
    if (n_threads > 8) n_threads = 8;

    const int32_t rows_per_thread = (nr1 + n_threads - 1) / n_threads;

    mulmat_thread_data_t thread_data[MAX_NUM_WORKERS];
    worker_synctoken_t synctoken;

    worker_pool_synctoken_init(&synctoken, n_threads - 1);

    for (unsigned int t = 0; t < n_threads; t++) {
        const int32_t ir1_start = t * rows_per_thread;
        const int32_t ir1_end = MIN(ir1_start + rows_per_thread, nr1);

        thread_data[t].src0 = src0;
        thread_data[t].src1 = src1;
        thread_data[t].dst = dst;
        thread_data[t].type = src0->type;
        thread_data[t].num_rows_per_vec_dot = 1;
        thread_data[t].ir0_start = 0;
        thread_data[t].ir0_end = nr0;
        thread_data[t].ir1_start = ir1_start;
        thread_data[t].ir1_end = ir1_end;
        thread_data[t].synctoken = (t == 0) ? NULL : &synctoken;

        if (t == 0) {
            mulmat_thread_func(&thread_data[t]);
        } else {
            worker_pool_job_t job;
            job.fptr = mulmat_thread_func;
            job.dptr = &thread_data[t];
            worker_pool_submit(NULL, job);
        }
    }

    worker_pool_synctoken_wait(&synctoken);

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return 0;
}

typedef struct {
    const ggml_tensor *src0;
    const ggml_tensor *src1;
    ggml_tensor *dst;
    enum ggml_type type;
    int32_t num_rows_per_vec_dot;
    int32_t ir0_start;
    int32_t ir0_end;
    int32_t ir1_start;
    int32_t ir1_end;
    uint8_t *vtcm_buf;
    size_t vtcm_size;
    worker_synctoken_t *synctoken;
} mulmat_thread_data_vtcm_t;

static void ggml_compute_forward_mul_mat_vtcm_chunk(const ggml_tensor *src0, const ggml_tensor *src1,
                                                    struct ggml_tensor *dst,
                                                    const enum ggml_type type,
                                                    const int32_t num_rows_per_vec_dot,
                                                    const int32_t ir0_start, const int32_t ir0_end,
                                                    const int32_t ir1_start, const int32_t ir1_end,
                                                    uint8_t *vtcm_buf, size_t vtcm_size) {
    const bool src1_cont = ggml_is_contiguous(src1);

    const int32_t ne00 = src0->ne[0];
    const int32_t ne01 = src0->ne[1];
    const int32_t ne02 = src0->ne[2];
    const int32_t ne03 = src0->ne[3];

    const int32_t ne10 = src1->ne[0];
    const int32_t ne11 = src1->ne[1];
    const int32_t ne12 = src1->ne[2];
    const int32_t ne13 = src1->ne[3];

    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    const size_t nb11 = src1->nb[1];
    const size_t nb12 = src1->nb[2];
    const size_t nb13 = src1->nb[3];

    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];
    const size_t nb0 = dst->nb[0];

    const int32_t r2 = ne12 / ne02;
    const int32_t r3 = ne13 / ne03;

    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const void * wdata = src1->data;
    const size_t row_size_f32 = 4 * ne10;
    const size_t row_size_f16 = 2 * ne10;

    const int32_t blck_0 = VTCM_BLOCK_ROWS;
    const int32_t blck_1 = VTCM_BLOCK_COLS;

    const bool src1_is_f16 = src1->type == GGML_TYPE_F16;
    const size_t row_size = src1_is_f16 ? row_size_f16 : row_size_f32;
    const size_t src1_col_stride = src1_cont ? row_size : nb11;

    const size_t max_rows_in_vtcm = (vtcm_size / sizeof(float)) / ne00;
    const int32_t rows_per_vtcm_block = MIN(max_rows_in_vtcm, VTCM_BLOCK_ROWS);

    float tmp[32];

    for (int32_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int32_t iir0_base = ir0_start; iir0_base < ir0_end; iir0_base += rows_per_vtcm_block) {
            const int32_t iir0_end = MIN(iir0_base + rows_per_vtcm_block, ir0_end);

            for (int32_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                const int32_t i13 = (ir1 / (ne12 * ne11));
                const int32_t i12 = (ir1 - i13 * ne12 * ne11) / ne11;
                const int32_t i11 = (ir1 - i13 * ne12 * ne11 - i12 * ne11);

                const int32_t i03 = i13 / r3;
                const int32_t i02 = i12 / r2;

                const char * src0_row = (const char*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                const char * src1_col = (const char*)wdata +
                    (src1_cont
                     ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                     : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                float * dst_col = (float*)((char*)dst->data + (i11 * nb1 + i12 * nb2 + i13 * nb3));

                for (int32_t iir0 = iir0_base; iir0 < iir0_end; iir0 += blck_0) {
                    const int32_t block_rows = MIN(iir0 + blck_0, iir0_end) - iir0;
                    const size_t copy_size = block_rows * nb01;

                    memcpy(vtcm_buf, src0_row + iir0 * nb01, copy_size);

                    for (int32_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < iir0_end; ir0 += num_rows_per_vec_dot) {
                        if (type == GGML_TYPE_F16 && src1_is_f16) {
                            vec_dot_f16_f16(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0),
                                           (uintptr_t)(vtcm_buf + (ir0 - iir0) * nb01), (num_rows_per_vec_dot > 1 ? nb01 : 0),
                                           (uint16_t*)src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                        } else if (type == GGML_TYPE_F16) {
                            vec_dot_f16_f32(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0),
                                           (uintptr_t)(vtcm_buf + (ir0 - iir0) * nb01), (num_rows_per_vec_dot > 1 ? nb01 : 0),
                                           (float*)src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                        } else {
                            vec_dot_f32(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0),
                                        (float*)(vtcm_buf + (ir0 - iir0) * nb01), (num_rows_per_vec_dot > 1 ? nb01 : 0),
                                        (float*)src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                        }
                    }

                    for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                        memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), block_rows * sizeof(float));
                    }
                }
            }
        }
    }
}

static void mulmat_thread_func_vtcm(void * data) {
    mulmat_thread_data_vtcm_t * tdata = (mulmat_thread_data_vtcm_t *) data;

    ggml_compute_forward_mul_mat_vtcm_chunk(
        tdata->src0, tdata->src1, tdata->dst,
        tdata->type, tdata->num_rows_per_vec_dot,
        tdata->ir0_start, tdata->ir0_end,
        tdata->ir1_start, tdata->ir1_end,
        tdata->vtcm_buf, tdata->vtcm_size
    );

    if (tdata->synctoken != NULL) {
        worker_pool_synctoken_jobdone(tdata->synctoken);
    }
}

static int ggmlop_dsp_mulmat_multithread_vtcm(remote_handle64 h, const struct dsptensor * src0, const struct dsptensor * src1, dsptensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );

    dst->ne[0] = src0->ne[1];
    dst->ne[1] = src1->ne[1];
    dst->ne[2] = src1->ne[2];
    dst->ne[3] = src1->ne[3];

    dst->nb[0] = 4;
    dst->nb[1] = dst->nb[0] * dst->ne[0];
    dst->nb[2] = dst->nb[1] * dst->ne[1];
    dst->nb[3] = dst->nb[2] * dst->ne[2];

    const int32_t ne0 = src0->ne[1];
    const int32_t ne1 = src1->ne[1];
    const int32_t ne2 = src1->ne[2];
    const int32_t ne3 = src1->ne[3];

    const int32_t nr0 = ne0;
    const int32_t nr1 = ne1 * ne2 * ne3;

    unsigned int n_threads = num_workers;
    if (n_threads < 1) n_threads = 1;
    if (n_threads > 8) n_threads = 8;

    const size_t vtcm_per_thread = 64 * 1024;
    const size_t total_vtcm = vtcm_per_thread * n_threads;

    void *vtcm_base = HAP_request_VTCM(total_vtcm, 0);
    if (vtcm_base == NULL) {
        GGMLHEXAGON_LOG_DEBUG("%s: VTCM allocation failed, falling back to non-VTCM", __func__);
        return ggmlop_dsp_mulmat_multithread(h, src0, src1, dst);
    }

    const int32_t rows_per_thread = (nr1 + n_threads - 1) / n_threads;

    mulmat_thread_data_vtcm_t thread_data[MAX_NUM_WORKERS];
    worker_synctoken_t synctoken;

    worker_pool_synctoken_init(&synctoken, n_threads - 1);

    for (unsigned int t = 0; t < n_threads; t++) {
        const int32_t ir1_start = t * rows_per_thread;
        const int32_t ir1_end = MIN(ir1_start + rows_per_thread, nr1);

        thread_data[t].src0 = src0;
        thread_data[t].src1 = src1;
        thread_data[t].dst = dst;
        thread_data[t].type = src0->type;
        thread_data[t].num_rows_per_vec_dot = 1;
        thread_data[t].ir0_start = 0;
        thread_data[t].ir0_end = nr0;
        thread_data[t].ir1_start = ir1_start;
        thread_data[t].ir1_end = ir1_end;
        thread_data[t].vtcm_buf = (uint8_t *)vtcm_base + t * vtcm_per_thread;
        thread_data[t].vtcm_size = vtcm_per_thread;
        thread_data[t].synctoken = (t == 0) ? NULL : &synctoken;

        if (t == 0) {
            mulmat_thread_func_vtcm(&thread_data[t]);
        } else {
            worker_pool_job_t job;
            job.fptr = mulmat_thread_func_vtcm;
            job.dptr = &thread_data[t];
            worker_pool_submit(NULL, job);
        }
    }

    worker_pool_synctoken_wait(&synctoken);

    HAP_release_VTCM(vtcm_base);

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return 0;
}

int ggmlop_dsp_mulmat(remote_handle64 h, const struct dsptensor * src0, const struct dsptensor * src1, dsptensor * dst) {
    if (ggmlop_get_thread_counts() > 1 && num_workers > 1) {
        return ggmlop_dsp_mulmat_multithread_vtcm(h, src0, src1, dst);
    } else {
        return ggmlop_dsp_mulmat_singlethread(h, src0, src1, dst);
    }
}