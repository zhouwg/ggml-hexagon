#include "ggml-dsp.h"
#include "worker_pool.h"
#include <math.h>
#include <string.h>

typedef struct {
    const ggml_tensor * src0;
    ggml_tensor * dst;
    float scale;
    int64_t start_idx;
    int64_t end_idx;
    worker_synctoken_t *synctoken;
} scale_thread_data_t;

static void scale_thread_func(void * data) {
    scale_thread_data_t * tdata = (scale_thread_data_t *) data;
    const ggml_tensor * src0 = tdata->src0;
    ggml_tensor * dst = tdata->dst;
    const float scale = tdata->scale;
    const int64_t start_idx = tdata->start_idx;
    const int64_t end_idx = tdata->end_idx;

    if (src0->type == GGML_TYPE_F16) {
        uint16_t * dst_ptr  = (uint16_t *)dst->data;
        uint16_t * src0_ptr = (uint16_t *)src0->data;

        for (int64_t i = start_idx; i < end_idx; ++i) {
            float f = ggml_compute_fp16_to_fp32(src0_ptr[i]);
            dst_ptr[i] = ggml_compute_fp32_to_fp16(f * scale);
        }
    } else {
        float * dst_ptr  = (float *)dst->data;
        float * src0_ptr = (float *)src0->data;

        for (int64_t i = start_idx; i < end_idx; ++i) {
            dst_ptr[i] = src0_ptr[i] * scale;
        }
    }

    if (tdata->synctoken != NULL) {
        worker_pool_synctoken_jobdone(tdata->synctoken);
    }
}

static void ggml_compute_forward_scale_f32(
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst,
        float scale) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );
    uint64_t start_time = ggml_time_us();

    const int64_t n = ggml_nelements(dst);

    if (src0->type == GGML_TYPE_F16) {
        uint16_t * dst_ptr  = (uint16_t *)dst->data;
        uint16_t * src0_ptr = (uint16_t *)src0->data;

        for (int64_t i = 0; i < n; ++i) {
            float f = ggml_compute_fp16_to_fp32(src0_ptr[i]);
            dst_ptr[i] = ggml_compute_fp32_to_fp16(f * scale);
        }
    } else {
        float * dst_ptr  = (float *)dst->data;
        float * src0_ptr = (float *)src0->data;

        for (int64_t i = 0; i < n; ++i) {
            dst_ptr[i] = src0_ptr[i] * scale;
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

static int ggmlop_dsp_scale_singlethread(remote_handle64 h, const ggml_tensor * src0, ggml_tensor * dst, float scale) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );
    ggml_compute_forward_scale_f32(src0, dst, scale);
    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return 0;
}

static int ggmlop_dsp_scale_multithread(remote_handle64 h, const ggml_tensor * src0, ggml_tensor * dst, float scale) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );

    const int64_t n = ggml_nelements(src0);
    int num_threads = num_workers;

    if (num_threads <= 1 || n < num_threads * 512) {
        return ggmlop_dsp_scale_singlethread(h, src0, dst, scale);
    }

    worker_synctoken_t synctoken;
    worker_pool_synctoken_init(&synctoken, num_threads - 1);

    scale_thread_data_t tdata[num_threads];
    const int64_t ne_per_thread = ((n + num_threads - 1) / num_threads + 127) & ~127;
    int64_t start_idx = 0;

    for (int i = 0; i < num_threads - 1; ++i) {
        int64_t end_idx = start_idx + ne_per_thread;
        if (end_idx > n)
            end_idx = n;

        tdata[i].src0 = src0;
        tdata[i].dst = dst;
        tdata[i].scale = scale;
        tdata[i].start_idx = start_idx;
        tdata[i].end_idx = end_idx;
        tdata[i].synctoken = &synctoken;

        worker_pool_job_t job;
        job.fptr = scale_thread_func;
        job.dptr = &tdata[i];
        worker_pool_submit(NULL, job);

        start_idx = end_idx;
    }

    tdata[num_threads - 1].src0 = src0;
    tdata[num_threads - 1].dst = dst;
    tdata[num_threads - 1].scale = scale;
    tdata[num_threads - 1].start_idx = start_idx;
    tdata[num_threads - 1].end_idx = n;
    tdata[num_threads - 1].synctoken = NULL;

    scale_thread_func(&tdata[num_threads - 1]);

    worker_pool_synctoken_wait(&synctoken);

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return 0;
}

int ggmlop_dsp_scale(remote_handle64 h, const ggml_tensor * src0, ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s\n", __func__);

    // Scale factor is stored in op_params[0]
    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    char tempbuf[256];
    ggmlhexagon_get_opkey(GGML_OP_SCALE, src0, NULL, tempbuf, 256);

    int64_t begin_time = ggml_time_us();
    if (ggmlop_get_thread_counts() > 1) {
        ggmlop_dsp_scale_multithread(h, src0, dst, scale);
    } else {
        ggmlop_dsp_scale_singlethread(h, src0, dst, scale);
    }
    int64_t end_time = ggml_time_us();
    GGMLHEXAGON_LOG_INFO("elapse time of %s is %lld us", tempbuf, (long long)(end_time - begin_time));
    GGMLHEXAGON_LOG_DEBUG("leave %s\n", __func__);
    return 0;
}
