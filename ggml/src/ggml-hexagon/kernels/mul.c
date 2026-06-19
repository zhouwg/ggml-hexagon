#include "ggml-dsp.h"
#include "worker_pool.h"

typedef struct {
    const ggml_tensor * src0;
    const ggml_tensor * src1;
    ggml_tensor * dst;
    int64_t start_idx;
    int64_t end_idx;
    worker_synctoken_t *synctoken;
} mul_thread_data_t;

static void mul_thread_func(void * data) {
    mul_thread_data_t * tdata = (mul_thread_data_t *) data;
    const ggml_tensor * src0 = tdata->src0;
    const ggml_tensor * src1 = tdata->src1;
    ggml_tensor * dst = tdata->dst;
    const int64_t start_idx = tdata->start_idx;
    const int64_t end_idx = tdata->end_idx;

    const int64_t ne0  = dst->ne[0];
    const int64_t ne1  = dst->ne[1];
    const int64_t ne2  = dst->ne[2];
    const int64_t ne3  = dst->ne[3];
    const int64_t nb0  = dst->nb[0];
    const int64_t nb1  = dst->nb[1];
    const int64_t nb2  = dst->nb[2];
    const int64_t nb3  = dst->nb[3];

    const int64_t nr  = ne1 * ne2 * ne3;
    const int64_t ir0 = start_idx / ne0;
    const int64_t ir1 = (end_idx + ne0 - 1) / ne0;

    if (src0->type == GGML_TYPE_F16) {
        uint16_t *       dst_data  = (uint16_t *)dst->data;
        const uint16_t * src0_data = (const uint16_t *)src0->data;
        const uint16_t * src1_data = (const uint16_t *)src1->data;

        const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
        const int64_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
        const int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
        const int64_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];

        bool src1_contig_rows = (ne10 == ne00 || ne10 == 1) &&
                                (nb10 == sizeof(uint16_t)) &&
                                (ne11 == 1 && ne12 == 1 && ne13 == 1);

        for (int64_t ir = ir0; ir < ir1 && ir < nr; ++ir) {
            const int64_t i03 = ir / (ne02 * ne01);
            const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            const int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;

            uint16_t *       dst_row  = (uint16_t *)((uint8_t *)dst_data  + i03 * nb3  + i02 * nb2  + i01 * nb1);
            const uint16_t * src0_row = (const uint16_t *)((const uint8_t *)src0_data + i03 * nb03 + i02 * nb02 + i01 * nb01);
            const uint16_t * src1_row = (const uint16_t *)((const uint8_t *)src1_data + i13 * nb13 + i12 * nb12 + i11 * nb11);

            int64_t row_start = ir * ne0;
            int64_t eff_start = (start_idx > row_start) ? start_idx - row_start : 0;
            int64_t eff_end   = ((row_start + ne0) < end_idx) ? ne0 : end_idx - row_start;

            if (src1_contig_rows) {
                int64_t nr0 = ne00 / ne10;
                for (int64_t r = 0; r < nr0; ++r) {
                    int64_t seg_start = r * ne10;
                    int64_t seg_end   = seg_start + ne10;
                    if (seg_end <= eff_start || seg_start >= eff_end) continue;
                    int64_t js = (seg_start > eff_start) ? seg_start : eff_start;
                    int64_t je = (seg_end < eff_end) ? seg_end : eff_end;
                    for (int64_t j = js; j < je; ++j) {
                        float f0 = ggml_compute_fp16_to_fp32(src0_row[j]);
                        float f1 = ggml_compute_fp16_to_fp32(src1_row[j - seg_start]);
                        dst_row[j] = ggml_compute_fp32_to_fp16(f0 * f1);
                    }
                }
            } else {
                for (int64_t j = eff_start; j < eff_end; ++j) {
                    int64_t j10 = j % ne10;
                    float f0 = ggml_compute_fp16_to_fp32(src0_row[j]);
                    float f1 = ggml_compute_fp16_to_fp32(src1_row[j10]);
                    dst_row[j] = ggml_compute_fp32_to_fp16(f0 * f1);
                }
            }
        }
    } else {
        float *       dst_data  = (float *)dst->data;
        const float * src0_data = (const float *)src0->data;
        const float * src1_data = (const float *)src1->data;

        const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
        const int64_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
        const int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
        const int64_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];

        bool src1_contig_rows = (ne10 == ne00 || ne10 == 1) &&
                                (nb10 == sizeof(float)) &&
                                (ne11 == 1 && ne12 == 1 && ne13 == 1);

        const int64_t nr  = ne1 * ne2 * ne3;
        const int64_t ir0_mt = start_idx / ne0;
        const int64_t ir1_mt = (end_idx + ne0 - 1) / ne0;

        for (int64_t ir = ir0_mt; ir < ir1_mt && ir < nr; ++ir) {
            const int64_t i03 = ir / (ne02 * ne01);
            const int64_t i02 = (ir - i03*ne02*ne01) / ne01;
            const int64_t i01 = ir - i03*ne02*ne01 - i02*ne01;

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;

            float *       dst_row  = (float *)((uint8_t *)dst_data  + i03*nb3  + i02*nb2  + i01*nb1);
            const float * src0_row = (float *)((const uint8_t *)src0_data + i03*nb03 + i02*nb02 + i01*nb01);
            const float * src1_row = (float *)((const uint8_t *)src1_data + i13*nb13 + i12*nb12 + i11*nb11);

            int64_t row_start = ir * ne0;
            int64_t eff_start = (start_idx > row_start) ? start_idx - row_start : 0;
            int64_t eff_end   = ((row_start + ne0) < end_idx) ? ne0 : end_idx - row_start;

            if (src1_contig_rows) {
                int64_t nr0 = ne00 / ne10;
                for (int64_t r = 0; r < nr0; ++r) {
                    int64_t seg_start = r * ne10;
                    int64_t seg_end   = seg_start + ne10;
                    if (seg_end <= eff_start || seg_start >= eff_end) continue;
                    int64_t js = (seg_start > eff_start) ? seg_start : eff_start;
                    int64_t je = (seg_end < eff_end) ? seg_end : eff_end;
                    for (int64_t j = js; j < je; ++j)
                        dst_row[j] = src0_row[j] * src1_row[j - seg_start];
                }
            } else {
                for (int64_t j = eff_start; j < eff_end; ++j)
                    dst_row[j] = src0_row[j] * src1_row[j % ne10];
            }
        }
    }

    if (tdata->synctoken != NULL) {
        worker_pool_synctoken_jobdone(tdata->synctoken);
    }
}

/* Single-threaded row-iteration MUL — matches CPU binary-ops.cpp apply_binary_op */
static void ggml_mul_singlethread(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t ne0  = dst->ne[0], ne1  = dst->ne[1], ne2  = dst->ne[2], ne3  = dst->ne[3];
    const int64_t nb0  = dst->nb[0], nb1  = dst->nb[1], nb2  = dst->nb[2], nb3  = dst->nb[3];

    if (src0->type == GGML_TYPE_F16) {
        uint16_t *       dst_data  = (uint16_t *)dst->data;
        const uint16_t * src0_data = (const uint16_t *)src0->data;
        const uint16_t * src1_data = (const uint16_t *)src1->data;

        const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
        const int64_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
        const int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
        const int64_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];

        bool src1_contig_rows = (ne10 == ne00 || ne10 == 1) &&
                                (nb10 == sizeof(uint16_t)) &&
                                (ne11 == 1 && ne12 == 1 && ne13 == 1);

        for (int64_t i01 = 0; i01 < ne1; ++i01) {
            for (int64_t i02 = 0; i02 < ne2; ++i02) {
                for (int64_t i03 = 0; i03 < ne3; ++i03) {
                    const int64_t i13 = i03 % ne13;
                    const int64_t i12 = i02 % ne12;
                    const int64_t i11 = i01 % ne11;

                    uint16_t *       dst_row  = (uint16_t *)((uint8_t *)dst_data  + i03*nb3  + i02*nb2  + i01*nb1);
                    const uint16_t * src0_row = (const uint16_t *)((const uint8_t *)src0_data + i03*nb03 + i02*nb02 + i01*nb01);
                    const uint16_t * src1_row = (const uint16_t *)((const uint8_t *)src1_data + i13*nb13 + i12*nb12 + i11*nb11);

                    if (src1_contig_rows) {
                        int64_t nr0 = ne00 / ne10;
                        for (int64_t r = 0; r < nr0; ++r) {
                            for (int64_t j = 0; j < ne10; ++j) {
                                float f0 = ggml_compute_fp16_to_fp32(src0_row[r*ne10 + j]);
                                float f1 = ggml_compute_fp16_to_fp32(src1_row[j]);
                                dst_row[r*ne10 + j] = ggml_compute_fp32_to_fp16(f0 * f1);
                            }
                        }
                    } else {
                        for (int64_t j = 0; j < ne0; ++j) {
                            int64_t j10 = j % ne10;
                            float f0 = ggml_compute_fp16_to_fp32(src0_row[j]);
                            float f1 = ggml_compute_fp16_to_fp32(src1_row[j10]);
                            dst_row[j] = ggml_compute_fp32_to_fp16(f0 * f1);
                        }
                    }
                }
            }
        }
    } else {
        float *       dst_data  = (float *)dst->data;
        const float * src0_data = (const float *)src0->data;
        const float * src1_data = (const float *)src1->data;

        const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
        const int64_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
        const int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
        const int64_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];

        bool src1_contig_rows = (ne10 == ne00 || ne10 == 1) &&
                                (nb10 == sizeof(float)) &&
                                (ne11 == 1 && ne12 == 1 && ne13 == 1);

        for (int64_t i01 = 0; i01 < ne1; ++i01) {
            for (int64_t i02 = 0; i02 < ne2; ++i02) {
                for (int64_t i03 = 0; i03 < ne3; ++i03) {
                    const int64_t i13 = i03 % ne13;
                    const int64_t i12 = i02 % ne12;
                    const int64_t i11 = i01 % ne11;

                    float *       dst_row  = (float *)((uint8_t *)dst_data  + i03*nb3  + i02*nb2  + i01*nb1);
                    const float * src0_row = (float *)((const uint8_t *)src0_data + i03*nb03 + i02*nb02 + i01*nb01);
                    const float * src1_row = (float *)((const uint8_t *)src1_data + i13*nb13 + i12*nb12 + i11*nb11);

                    if (src1_contig_rows) {
                        int64_t nr0 = ne00 / ne10;
                        for (int64_t r = 0; r < nr0; ++r) {
                            for (int64_t j = 0; j < ne10; ++j)
                                dst_row[r*ne10 + j] = src0_row[r*ne10 + j] * src1_row[j];
                        }
                    } else {
                        for (int64_t j = 0; j < ne0; ++j)
                            dst_row[j] = src0_row[j] * src1_row[j % ne10];
                    }
                }
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

    const int64_t n = ggml_nelements(src0);
    int num_threads = num_workers;

    if (src0->type == GGML_TYPE_F32) {
        num_threads = 2;
    } else if (src0->type == GGML_TYPE_F16) {
        num_threads = ggml_min(num_workers, 6);
    } else {
        num_threads = num_workers;
    }

    if (num_threads <= 1 || n < num_threads * 512) {
        return ggmlop_dsp_mul_singlethread(h, src0, src1, dst);
    }

    worker_synctoken_t synctoken;
    worker_pool_synctoken_init(&synctoken, num_threads - 1);

    mul_thread_data_t tdata[num_threads];
    const int64_t ne_per_thread = ((n + num_threads - 1) / num_threads + 127) & ~127;
    int64_t start_idx = 0;

    for (int i = 0; i < num_threads - 1; ++i) {
        int64_t end_idx = start_idx + ne_per_thread;
        if (end_idx > n)
            end_idx = n;

        tdata[i].src0 = src0;
        tdata[i].src1 = src1;
        tdata[i].dst = dst;
        tdata[i].start_idx = start_idx;
        tdata[i].end_idx = end_idx;
        tdata[i].synctoken = &synctoken;

        worker_pool_job_t job;
        job.fptr = mul_thread_func;
        job.dptr = &tdata[i];
        worker_pool_submit(NULL, job);

        start_idx = end_idx;
    }

    tdata[num_threads - 1].src0 = src0;
    tdata[num_threads - 1].src1 = src1;
    tdata[num_threads - 1].dst = dst;
    tdata[num_threads - 1].start_idx = start_idx;
    tdata[num_threads - 1].end_idx = n;
    tdata[num_threads - 1].synctoken = NULL;

    mul_thread_func(&tdata[num_threads - 1]);

    worker_pool_synctoken_wait(&synctoken);

    return 0;
}

int ggmlop_dsp_mul(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    /* For now, use single-threaded row-iteration.
     * Multi-thread requires guarantee that dst does not alias src0. */
    return ggmlop_dsp_mul_singlethread(h, src0, src1, dst);
}
