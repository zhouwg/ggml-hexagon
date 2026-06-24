#include "ggml-dsp.h"
#include "worker_pool.h"

// RMS_NORM: dst[i] = src0[i] * scale, where scale = 1.0f / sqrtf(mean_sq + eps)
// mean_sq = sum(src0[i]^2) / ne00
// eps is stored in op_params[0] as float (unary op: src1 == NULL)

static inline void rmsnorm_f32_scalar(const int n, float * y, const float * x, float eps) {
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        sum_sq += (double)(x[i] * x[i]);
    }
    float mean_sq = (float)(sum_sq / n);
    // Guard against NaN when input is all-zero and eps is very small:
    // scale = 1/sqrt(mean_sq + eps); if mean_sq=0 and eps≈0, scale→inf,
    // then y[i]=0*inf = NaN per IEEE 754.
    float denom = mean_sq + eps;
    float scale = (denom > 0.0f) ? (1.0f / sqrtf(denom)) : 0.0f;

    for (int i = 0; i < n; ++i) {
        y[i] = x[i] * scale;
    }
}

typedef struct {
    const ggml_tensor * src0;
    ggml_tensor * dst;
    int64_t start_idx;
    int64_t end_idx;
    float eps;
    worker_synctoken_t *synctoken;
} rmsnorm_thread_data_t;

static void rmsnorm_thread_func(void * data) {
    rmsnorm_thread_data_t * tdata = (rmsnorm_thread_data_t *) data;
    const ggml_tensor * src0 = tdata->src0;
    ggml_tensor * dst = tdata->dst;
    int64_t start_idx = tdata->start_idx;
    int64_t end_idx = tdata->end_idx;
    float eps = tdata->eps;

    int64_t ne00 = src0->ne[0];
    int64_t ne01 = src0->ne[1];
    int64_t ne02 = src0->ne[2];
    int64_t ne03 = src0->ne[3];
    size_t nb01 = src0->nb[1];
    size_t nb02 = src0->nb[2];
    size_t nb03 = src0->nb[3];
    size_t nb1  = dst->nb[1];
    size_t nb2  = dst->nb[2];
    size_t nb3  = dst->nb[3];

    // iterate rows using 3-level index to handle non-contiguous strides
    int64_t row = 0;
    for (int64_t i3 = 0; i3 < ne03; i3++) {
        for (int64_t i2 = 0; i2 < ne02; i2++) {
            for (int64_t i1 = 0; i1 < ne01; i1++) {
                if (row < start_idx) { row++; continue; }
                if (row >= end_idx) goto done;
                const float * x = (const float *)((const uint8_t *)src0->data + i1*nb01 + i2*nb02 + i3*nb03);
                float * y = (float *)((uint8_t *)dst->data + i1*nb1 + i2*nb2 + i3*nb3);
                rmsnorm_f32_scalar(ne00, y, x, eps);
                row++;
            }
        }
    }
done:

    if (tdata->synctoken != NULL) {
        worker_pool_synctoken_jobdone(tdata->synctoken);
    }
}

static void ggml_compute_forward_rms_norm_f32(
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {

    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__);

    int64_t start_time = ggml_time_us();

    ggml_dump_tensor(src0, 1);
    ggml_dump_tensor(dst, 1);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    int64_t ne00 = src0->ne[0];
    int64_t nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    GGMLHEXAGON_LOG_INFO("RMS_NORM: src0 ne=[%lld,%lld,%lld,%lld] nb=[%d,%d,%d,%d] data=%p data_len=%d",
                         (long long)src0->ne[0], (long long)src0->ne[1],
                         (long long)src0->ne[2], (long long)src0->ne[3],
                         src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                         src0->data, src0->data_len);
    GGMLHEXAGON_LOG_INFO("RMS_NORM: dst  ne=[%lld,%lld,%lld,%lld] nb=[%d,%d,%d,%d] data=%p data_len=%d",
                         (long long)dst->ne[0], (long long)dst->ne[1],
                         (long long)dst->ne[2], (long long)dst->ne[3],
                         dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                         dst->data, dst->data_len);
    GGMLHEXAGON_LOG_INFO("RMS_NORM: eps=%f, contiguous=%d, nrows=%lld",
                         eps, ggml_is_contiguous(src0), (long long)nrows);

    if (ggmlop_get_thread_counts() > 1 && nrows >= ggmlop_get_thread_counts() * 2) {
        int num_threads = ggmlop_get_thread_counts();
        if (num_threads > nrows) num_threads = nrows;

        worker_synctoken_t synctoken;
        worker_pool_synctoken_init(&synctoken, num_threads - 1);

        rmsnorm_thread_data_t tdata[num_threads];
        int64_t rows_per_thread = (nrows + num_threads - 1) / num_threads;
        int64_t idx = 0;

        for (int i = 0; i < num_threads - 1; ++i) {
            int64_t end_idx = idx + rows_per_thread;
            if (end_idx > nrows) end_idx = nrows;

            tdata[i].src0 = src0;
            tdata[i].dst = dst;
            tdata[i].start_idx = idx;
            tdata[i].end_idx = end_idx;
            tdata[i].eps = eps;
            tdata[i].synctoken = &synctoken;

            worker_pool_job_t job;
            job.fptr = rmsnorm_thread_func;
            job.dptr = &tdata[i];
            worker_pool_submit(NULL, job);

            idx = end_idx;
        }

        tdata[num_threads - 1].src0 = src0;
        tdata[num_threads - 1].dst = dst;
        tdata[num_threads - 1].start_idx = idx;
        tdata[num_threads - 1].end_idx = nrows;
        tdata[num_threads - 1].eps = eps;
        tdata[num_threads - 1].synctoken = NULL;

        rmsnorm_thread_func(&tdata[num_threads - 1]);

        worker_pool_synctoken_wait(&synctoken);
    } else {
        // single-threaded: iterate all rows with proper stride computation
        int64_t ne01 = src0->ne[1];
        int64_t ne02 = src0->ne[2];
        int64_t ne03 = src0->ne[3];
        size_t nb01 = src0->nb[1];
        size_t nb02 = src0->nb[2];
        size_t nb03 = src0->nb[3];
        size_t nb1  = dst->nb[1];
        size_t nb2  = dst->nb[2];
        size_t nb3  = dst->nb[3];

        for (int64_t i3 = 0; i3 < ne03; i3++) {
            for (int64_t i2 = 0; i2 < ne02; i2++) {
                for (int64_t i1 = 0; i1 < ne01; i1++) {
                    const float * x = (const float *)((const uint8_t *)src0->data + i1*nb01 + i2*nb02 + i3*nb03);
                    float * y = (float *)((uint8_t *)dst->data + i1*nb1 + i2*nb2 + i3*nb3);
                    rmsnorm_f32_scalar(ne00, y, x, eps);
                }
            }
        }
    }

    int64_t end_time = ggml_time_us();
    int64_t duration = end_time - start_time;
    GGMLHEXAGON_LOG_INFO("RMS_NORM elapse %lld us (ne00=%lld, nrows=%lld, eps=%f)",
                         (long long)duration, (long long)ne00, (long long)nrows, eps);

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__);
}

int ggmlop_dsp_rmsnorm(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_UNUSED(h);
    GGML_UNUSED(src1);  // unary op, no src1
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__);

    int64_t begin_time = ggml_time_us();

    if (src0->type != GGML_TYPE_F32) {
        GGMLHEXAGON_LOG_ERROR("RMS_NORM: unsupported src0 type %d", src0->type);
        return AEE_EUNSUPPORTED;
    }

    ggml_compute_forward_rms_norm_f32(src0, dst);

    int64_t end_time = ggml_time_us();
    GGMLHEXAGON_LOG_INFO("RMS_NORM elapse %lld us (ne00=%lld, nrows=%lld)",
                         (long long)(end_time - begin_time),
                         (long long)src0->ne[0],
                         (long long)(src0->ne[1] * src0->ne[2] * src0->ne[3]));

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__);
    return 0;
}
