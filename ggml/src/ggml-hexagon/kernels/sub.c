#include "ggml-dsp.h"
#include "worker_pool.h"

typedef struct {
    const ggml_tensor * src0;
    const ggml_tensor * src1;
    ggml_tensor * dst;
    int64_t start_idx;
    int64_t end_idx;
    worker_synctoken_t *synctoken;
} sub_thread_data_t;

static void sub_thread_func(void * data) {
    sub_thread_data_t * tdata = (sub_thread_data_t *) data;
    const ggml_tensor * src0 = tdata->src0;
    const ggml_tensor * src1 = tdata->src1;
    ggml_tensor * dst = tdata->dst;
    const int64_t start_idx = tdata->start_idx;
    const int64_t end_idx = tdata->end_idx;

    const int64_t ne0 = dst->ne[0], ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2], ne3 = dst->ne[3];

    if (src0->type == GGML_TYPE_F16) {
        uint16_t * dst_ptr  = (uint16_t *)dst->data;
        uint16_t * src0_ptr = (uint16_t *)src0->data;
        uint16_t * src1_ptr = (uint16_t *)src1->data;

        for (int64_t i = start_idx; i < end_idx; ++i) {
            int64_t i0 = i % ne0;
            int64_t r  = i / ne0;
            int64_t i1 = r % ne1;
            int64_t r2 = r / ne1;
            int64_t i2 = r2 % ne2;
            int64_t i3 = r2 / ne2;

            // Broadcast-aware coordinate mapping via modulo
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

            float f0 = ggml_compute_fp16_to_fp32(src0_ptr[off0 >> 1]);
            float f1 = ggml_compute_fp16_to_fp32(src1_ptr[off1 >> 1]);
            dst_ptr[i] = ggml_compute_fp32_to_fp16(f0 - f1);
        }
    } else {
        float * dst_ptr  = (float *)dst->data;
        float * src0_ptr = (float *)src0->data;
        float * src1_ptr = (float *)src1->data;

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

            dst_ptr[i] = src0_ptr[off0 >> 2] - src1_ptr[off1 >> 2];
        }
    }

    if (tdata->synctoken != NULL) {
        worker_pool_synctoken_jobdone(tdata->synctoken);
    }
}

static void ggml_compute_forward_sub_f32(
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );
    uint64_t start_time = ggml_time_us();

    const int64_t n = ggml_nelements(dst);
    const int64_t ne0 = dst->ne[0], ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2], ne3 = dst->ne[3];

    if (src0->type == GGML_TYPE_F16) {
        uint16_t * dst_ptr  = (uint16_t *)dst->data;
        uint16_t * src0_ptr = (uint16_t *)src0->data;
        uint16_t * src1_ptr = (uint16_t *)src1->data;

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

            float f0 = ggml_compute_fp16_to_fp32(src0_ptr[off0 >> 1]);
            float f1 = ggml_compute_fp16_to_fp32(src1_ptr[off1 >> 1]);
            dst_ptr[i] = ggml_compute_fp32_to_fp16(f0 - f1);
        }
    } else {
        float * dst_ptr  = (float *)dst->data;
        float * src0_ptr = (float *)src0->data;
        float * src1_ptr = (float *)src1->data;

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

            dst_ptr[i] = src0_ptr[off0 >> 2] - src1_ptr[off1 >> 2];
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

static int ggmlop_dsp_sub_singlethread(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );
    ggml_compute_forward_sub_f32(src0, src1, dst);
    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return 0;
}

static int ggmlop_dsp_sub_multithread(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );

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
        return ggmlop_dsp_sub_singlethread(h, src0, src1, dst);
    }

    worker_synctoken_t synctoken;
    worker_pool_synctoken_init(&synctoken, num_threads - 1);

    sub_thread_data_t tdata[num_threads];
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
        job.fptr = sub_thread_func;
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

    sub_thread_func(&tdata[num_threads - 1]);

    worker_pool_synctoken_wait(&synctoken);

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return 0;
}

int ggmlop_dsp_sub(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s\n", __func__);
    char tempbuf[256];
    ggmlhexagon_get_opkey(GGML_OP_SUB, src0, src1, tempbuf, 256);

    int64_t begin_time = ggml_time_us();
    if (ggmlop_get_thread_counts() > 1) {
        ggmlop_dsp_sub_multithread(h, src0, src1, dst);
    } else {
        ggmlop_dsp_sub_singlethread(h, src0, src1, dst);
    }
    int64_t end_time = ggml_time_us();
    GGMLHEXAGON_LOG_INFO("elapse time of %s is %lld us", tempbuf, (long long)(end_time - begin_time));
    GGMLHEXAGON_LOG_DEBUG("leave %s\n", __func__);
    return 0;
}
