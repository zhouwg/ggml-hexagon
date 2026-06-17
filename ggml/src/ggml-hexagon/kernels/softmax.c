#include "ggml-dsp.h"
#include "worker_pool.h"
#include <math.h>

// ============================================================
// SOFT_MAX - Softmax (F32 scalar implementation)
// Basic case: no mask (src1=NULL), no sinks (src2=NULL)
// op_params[0] = scale (float), op_params[1] = max_bias (float)
// ============================================================

static void soft_max_row_f32(int64_t ne00, float * dp, const float * sp, float scale) {
    // step 1: find max
    float max_val = -INFINITY;
    for (int64_t i = 0; i < ne00; ++i) {
        if (sp[i] > max_val) max_val = sp[i];
    }

    // step 2: exp(x - max) and sum
    float sum = 0.0f;
    for (int64_t i = 0; i < ne00; ++i) {
        float val = expf((sp[i] - max_val) * scale);
        dp[i] = val;
        sum += val;
    }

    // step 3: normalize
    float inv_sum = 1.0f / sum;
    for (int64_t i = 0; i < ne00; ++i) {
        dp[i] *= inv_sum;
    }
}

typedef struct {
    const ggml_tensor * src0;
    ggml_tensor * dst;
    int64_t start_idx;
    int64_t end_idx;
    float scale;
    worker_synctoken_t *synctoken;
} softmax_thread_data_t;

static void softmax_thread_func(void * data) {
    softmax_thread_data_t * tdata = (softmax_thread_data_t *) data;
    const ggml_tensor * src0 = tdata->src0;
    ggml_tensor * dst = tdata->dst;
    int64_t start_idx = tdata->start_idx;
    int64_t end_idx = tdata->end_idx;
    float scale = tdata->scale;

    int64_t ne00 = src0->ne[0];
    size_t nb01 = src0->nb[1];
    size_t nb1 = dst->nb[1];

    for (int64_t idx = start_idx; idx < end_idx; ++idx) {
        const float * sp = (const float *)((const uint8_t *)src0->data + idx * nb01);
        float * dp = (float *)((uint8_t *)dst->data + idx * nb1);
        soft_max_row_f32(ne00, dp, sp, scale);
    }

    if (tdata->synctoken != NULL) {
        worker_pool_synctoken_jobdone(tdata->synctoken);
    }
}

static void ggml_compute_forward_soft_max_f32(
        const ggml_tensor * src0,
        ggml_tensor * dst) {

    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__);
    int64_t start_time = ggml_time_us();

    float scale = 1.0f;
    memcpy(&scale, dst->op_params, sizeof(float));
    // max_bias at op_params+1 (ignored for now, used in ALiBi)

    int64_t ne00 = src0->ne[0];
    int64_t nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    if (ggmlop_get_thread_counts() > 1 && nrows >= ggmlop_get_thread_counts() * 2) {
        int num_threads = ggmlop_get_thread_counts();
        if (num_threads > nrows) num_threads = nrows;

        worker_synctoken_t synctoken;
        worker_pool_synctoken_init(&synctoken, num_threads - 1);

        softmax_thread_data_t tdata[num_threads];
        int64_t rows_per_thread = (nrows + num_threads - 1) / num_threads;
        int64_t idx = 0;

        for (int i = 0; i < num_threads - 1; ++i) {
            int64_t end_idx = idx + rows_per_thread;
            if (end_idx > nrows) end_idx = nrows;

            tdata[i].src0 = src0;
            tdata[i].dst = dst;
            tdata[i].start_idx = idx;
            tdata[i].end_idx = end_idx;
            tdata[i].scale = scale;
            tdata[i].synctoken = &synctoken;

            worker_pool_job_t job;
            job.fptr = softmax_thread_func;
            job.dptr = &tdata[i];
            worker_pool_submit(NULL, job);

            idx = end_idx;
        }

        tdata[num_threads - 1].src0 = src0;
        tdata[num_threads - 1].dst = dst;
        tdata[num_threads - 1].start_idx = idx;
        tdata[num_threads - 1].end_idx = nrows;
        tdata[num_threads - 1].scale = scale;
        tdata[num_threads - 1].synctoken = NULL;

        softmax_thread_func(&tdata[num_threads - 1]);

        worker_pool_synctoken_wait(&synctoken);
    } else {
        size_t nb01 = src0->nb[1];
        size_t nb1 = dst->nb[1];
        for (int64_t r = 0; r < nrows; ++r) {
            const float * sp = (const float *)((const uint8_t *)src0->data + r * nb01);
            float * dp = (float *)((uint8_t *)dst->data + r * nb1);
            soft_max_row_f32(ne00, dp, sp, scale);
        }
    }

    int64_t end_time = ggml_time_us();
    GGMLHEXAGON_LOG_INFO("SOFT_MAX elapse %lld us (ne00=%lld, nrows=%lld, scale=%f)",
                         (long long)(end_time - start_time),
                         (long long)ne00, (long long)nrows, scale);
    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__);
}

int ggmlop_dsp_softmax(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_UNUSED(h);
    GGML_UNUSED(src1);  // mask tensor, not handled yet
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__);

    int64_t begin_time = ggml_time_us();

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        GGMLHEXAGON_LOG_ERROR("SOFT_MAX: unsupported type src0=%d dst=%d", src0->type, dst->type);
        return AEE_EUNSUPPORTED;
    }

    ggml_compute_forward_soft_max_f32(src0, dst);

    int64_t end_time = ggml_time_us();
    GGMLHEXAGON_LOG_INFO("elapse time of SOFT_MAX is %lld us", (long long)(end_time - begin_time));

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__);
    return 0;
}
