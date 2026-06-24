#include "ggml-dsp.h"
#include "worker_pool.h"

typedef struct {
    const ggml_tensor * src0;
    const ggml_tensor * src1;
    ggml_tensor * dst;
    int64_t start_idx;
    int64_t end_idx;
    worker_synctoken_t *synctoken;
} div_thread_data_t;

static inline void ggml_div_f32_scalar(const int n, float * z, const float * x, const float * y) {
    #pragma unroll(4)
    for (int i = 0; i < n; ++i)
        z[i] = x[i] / y[i];
}

static void div_thread_func(void * data) {
    div_thread_data_t * tdata = (div_thread_data_t *) data;
    const ggml_tensor * src0 = tdata->src0;
    const ggml_tensor * src1 = tdata->src1;
    ggml_tensor * dst = tdata->dst;
    const int64_t start_idx = tdata->start_idx;
    const int64_t end_idx = tdata->end_idx;

    const int64_t ne0 = dst->ne[0], ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2], ne3 = dst->ne[3];

    if (src0->type == GGML_TYPE_F16) {
        for (int64_t i = start_idx; i < end_idx; ++i) {
            int64_t i0 = i % ne0;
            int64_t r  = i / ne0;
            int64_t i1 = r % ne1;
            int64_t r2 = r / ne1;
            int64_t i2 = r2 % ne2;
            int64_t i3 = r2 / ne2;

            int64_t s0_0 = i0 % src0->ne[0];
            int64_t s0_1 = i1 % src0->ne[1];
            int64_t s0_2 = i2 % src0->ne[2];
            int64_t s0_3 = i3 % src0->ne[3];

            int64_t s1_0 = i0 % src1->ne[0];
            int64_t s1_1 = i1 % src1->ne[1];
            int64_t s1_2 = i2 % src1->ne[2];
            int64_t s1_3 = i3 % src1->ne[3];

            int64_t off0 = s0_0*src0->nb[0] + s0_1*src0->nb[1] + s0_2*src0->nb[2] + s0_3*src0->nb[3];
            int64_t off1 = s1_0*src1->nb[0] + s1_1*src1->nb[1] + s1_2*src1->nb[2] + s1_3*src1->nb[3];
            int64_t offd = i0*dst->nb[0]   + i1*dst->nb[1]   + i2*dst->nb[2]   + i3*dst->nb[3];

            float f0 = ggml_compute_fp16_to_fp32(*(const uint16_t *)((const uint8_t *)src0->data + off0));
            float f1 = ggml_compute_fp16_to_fp32(*(const uint16_t *)((const uint8_t *)src1->data + off1));
            *(uint16_t *)((uint8_t *)dst->data + offd) = ggml_compute_fp32_to_fp16(f0 / f1);
        }
    } else {
        bool need_broadcast = (src0->ne[0] != ne0 || src0->ne[1] != ne1 ||
                               src0->ne[2] != ne2 || src0->ne[3] != ne3 ||
                               src1->ne[0] != ne0 || src1->ne[1] != ne1 ||
                               src1->ne[2] != ne2 || src1->ne[3] != ne3);

        bool contig = !need_broadcast &&
                      (src0->nb[0] == sizeof(float) && src0->nb[1] == src0->ne[0]*sizeof(float) &&
                       src1->nb[0] == sizeof(float) && src1->nb[1] == src1->ne[0]*sizeof(float) &&
                       dst->nb[0] == sizeof(float) && dst->nb[1] == dst->ne[0]*sizeof(float));

        if (contig) {
            const int n = end_idx - start_idx;
            float * dst_ptr  = (float *)dst->data;
            float * src0_ptr = (float *)src0->data;
            float * src1_ptr = (float *)src1->data;
            ggml_div_f32_scalar(n, dst_ptr + start_idx, src0_ptr + start_idx, src1_ptr + start_idx);
        } else {
            for (int64_t i = start_idx; i < end_idx; ++i) {
                int64_t i0 = i % ne0;
                int64_t r  = i / ne0;
                int64_t i1 = r % ne1;
                int64_t r2 = r / ne1;
                int64_t i2 = r2 % ne2;
                int64_t i3 = r2 / ne2;

                int64_t s0_0 = i0 % src0->ne[0];
                int64_t s0_1 = i1 % src0->ne[1];
                int64_t s0_2 = i2 % src0->ne[2];
                int64_t s0_3 = i3 % src0->ne[3];

                int64_t s1_0 = i0 % src1->ne[0];
                int64_t s1_1 = i1 % src1->ne[1];
                int64_t s1_2 = i2 % src1->ne[2];
                int64_t s1_3 = i3 % src1->ne[3];

                int64_t off0 = s0_0*src0->nb[0] + s0_1*src0->nb[1] + s0_2*src0->nb[2] + s0_3*src0->nb[3];
                int64_t off1 = s1_0*src1->nb[0] + s1_1*src1->nb[1] + s1_2*src1->nb[2] + s1_3*src1->nb[3];
                int64_t offd = i0*dst->nb[0]   + i1*dst->nb[1]   + i2*dst->nb[2]   + i3*dst->nb[3];

                *(float *)((uint8_t *)dst->data + offd) = *(const float *)((const uint8_t *)src0->data + off0) / *(const float *)((const uint8_t *)src1->data + off1);
            }
        }
    }

    if (tdata->synctoken != NULL) {
        worker_pool_synctoken_jobdone(tdata->synctoken);
    }
}

static void ggml_compute_forward_div_f32(
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );
    uint64_t start_time = ggml_time_us();

    const int64_t n = ggml_nelements(dst);
    const int64_t ne0 = dst->ne[0], ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2], ne3 = dst->ne[3];

    if (src0->type == GGML_TYPE_F16) {
        for (int64_t i = 0; i < n; ++i) {
            int64_t i0 = i % ne0;
            int64_t r  = i / ne0;
            int64_t i1 = r % ne1;
            int64_t r2 = r / ne1;
            int64_t i2 = r2 % ne2;
            int64_t i3 = r2 / ne2;

            int64_t s0_0 = i0 % src0->ne[0];
            int64_t s0_1 = i1 % src0->ne[1];
            int64_t s0_2 = i2 % src0->ne[2];
            int64_t s0_3 = i3 % src0->ne[3];

            int64_t s1_0 = i0 % src1->ne[0];
            int64_t s1_1 = i1 % src1->ne[1];
            int64_t s1_2 = i2 % src1->ne[2];
            int64_t s1_3 = i3 % src1->ne[3];

            int64_t off0 = s0_0*src0->nb[0] + s0_1*src0->nb[1] + s0_2*src0->nb[2] + s0_3*src0->nb[3];
            int64_t off1 = s1_0*src1->nb[0] + s1_1*src1->nb[1] + s1_2*src1->nb[2] + s1_3*src1->nb[3];
            int64_t offd = i0*dst->nb[0]   + i1*dst->nb[1]   + i2*dst->nb[2]   + i3*dst->nb[3];

            float f0 = ggml_compute_fp16_to_fp32(*(const uint16_t *)((const uint8_t *)src0->data + off0));
            float f1 = ggml_compute_fp16_to_fp32(*(const uint16_t *)((const uint8_t *)src1->data + off1));
            *(uint16_t *)((uint8_t *)dst->data + offd) = ggml_compute_fp32_to_fp16(f0 / f1);
        }
    } else {
        bool need_broadcast = (src0->ne[0] != ne0 || src0->ne[1] != ne1 ||
                               src0->ne[2] != ne2 || src0->ne[3] != ne3 ||
                               src1->ne[0] != ne0 || src1->ne[1] != ne1 ||
                               src1->ne[2] != ne2 || src1->ne[3] != ne3);

        bool contig = !need_broadcast &&
                      (src0->nb[0] == sizeof(float) && src0->nb[1] == src0->ne[0]*sizeof(float) &&
                       src1->nb[0] == sizeof(float) && src1->nb[1] == src1->ne[0]*sizeof(float) &&
                       dst->nb[0] == sizeof(float) && dst->nb[1] == dst->ne[0]*sizeof(float));

        if (contig) {
            float * dst_ptr  = (float *)dst->data;
            float * src0_ptr = (float *)src0->data;
            float * src1_ptr = (float *)src1->data;
            ggml_div_f32_scalar(n, dst_ptr, src0_ptr, src1_ptr);
        } else {
            for (int64_t i = 0; i < n; ++i) {
                int64_t i0 = i % ne0;
                int64_t r  = i / ne0;
                int64_t i1 = r % ne1;
                int64_t r2 = r / ne1;
                int64_t i2 = r2 % ne2;
                int64_t i3 = r2 / ne2;

                int64_t s0_0 = i0 % src0->ne[0];
                int64_t s0_1 = i1 % src0->ne[1];
                int64_t s0_2 = i2 % src0->ne[2];
                int64_t s0_3 = i3 % src0->ne[3];

                int64_t s1_0 = i0 % src1->ne[0];
                int64_t s1_1 = i1 % src1->ne[1];
                int64_t s1_2 = i2 % src1->ne[2];
                int64_t s1_3 = i3 % src1->ne[3];

                int64_t off0 = s0_0*src0->nb[0] + s0_1*src0->nb[1] + s0_2*src0->nb[2] + s0_3*src0->nb[3];
                int64_t off1 = s1_0*src1->nb[0] + s1_1*src1->nb[1] + s1_2*src1->nb[2] + s1_3*src1->nb[3];
                int64_t offd = i0*dst->nb[0]   + i1*dst->nb[1]   + i2*dst->nb[2]   + i3*dst->nb[3];

                *(float *)((uint8_t *)dst->data + offd) = *(const float *)((const uint8_t *)src0->data + off0) / *(const float *)((const uint8_t *)src1->data + off1);
            }
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

static int ggmlop_dsp_div_singlethread(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );
    ggml_compute_forward_div_f32(src0, src1, dst);
    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return 0;
}

static int ggmlop_dsp_div_multithread(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );

    const int64_t n = ggml_nelements(dst);
    int num_threads = num_workers;

    if (src0->type == GGML_TYPE_F32) {
        num_threads = 2;
    } else if (src0->type == GGML_TYPE_F16) {
        num_threads = ggml_min(num_workers, 6);
    } else {
        num_threads = num_workers;
    }

    if (num_threads <= 1 || n < num_threads * 512) {
        return ggmlop_dsp_div_singlethread(h, src0, src1, dst);
    }

    worker_synctoken_t synctoken;
    worker_pool_synctoken_init(&synctoken, num_threads - 1);

    div_thread_data_t tdata[num_threads];
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
        job.fptr = div_thread_func;
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

    div_thread_func(&tdata[num_threads - 1]);

    worker_pool_synctoken_wait(&synctoken);

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return 0;
}

int ggmlop_dsp_div(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s\n", __func__);
    char tempbuf[256];
    ggml_get_opkey(GGML_OP_DIV, src0, src1, tempbuf, 256);

    int64_t begin_time = ggml_time_us();
    if (ggmlop_get_thread_counts() > 1) {
        ggmlop_dsp_div_multithread(h, src0, src1, dst);
    } else {
        ggmlop_dsp_div_singlethread(h, src0, src1, dst);
    }
    int64_t end_time = ggml_time_us();
    GGMLHEXAGON_LOG_INFO("elapse time of %s is %lld us", tempbuf, (long long)(end_time - begin_time));
    GGMLHEXAGON_LOG_DEBUG("leave %s\n", __func__);
    return 0;
}
