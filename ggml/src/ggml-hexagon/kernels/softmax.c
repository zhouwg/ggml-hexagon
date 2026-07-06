#include "ggml-dsp.h"
#include "worker_pool.h"
#include <math.h>

// ============================================================
// SOFT_MAX - Softmax (F32 scalar implementation)
// Supports: scale, mask (src1, F16 or F32), ALiBi (max_bias),
//           sinks (src2, F32, one value per head)
// op_params[0] = scale (float), op_params[1] = max_bias (float)
// ============================================================

static void soft_max_row_f32_masked(
        int64_t ne00,
        float * dp, const float * sp,
        const void * mp, int mp_type,
        float scale, float slope,
        float sink) {
    // step 1: apply scale and mask, find max
    float max_val = -INFINITY;
    if (mp) {
        if (mp_type == GGML_TYPE_F16) {
            const uint16_t * mp16 = (const uint16_t *)mp;
            for (int64_t i = 0; i < ne00; ++i) {
                float v = sp[i] * scale + slope * ggml_compute_fp16_to_fp32(mp16[i]);
                dp[i] = v;
                if (v > max_val) max_val = v;
            }
        } else {
            const float * mp32 = (const float *)mp;
            for (int64_t i = 0; i < ne00; ++i) {
                float v = sp[i] * scale + slope * mp32[i];
                dp[i] = v;
                if (v > max_val) max_val = v;
            }
        }
    } else {
        for (int64_t i = 0; i < ne00; ++i) {
            float v = sp[i] * scale;
            dp[i] = v;
            if (v > max_val) max_val = v;
        }
    }

    // sinks: adjust max to include sink value
    if (!isnan(sink)) {
        if (sink > max_val) max_val = sink;
    }

    // step 2: exp(x - max) and sum
    float sum = 0.0f;
    for (int64_t i = 0; i < ne00; ++i) {
        float val = expf(dp[i] - max_val);
        dp[i] = val;
        sum += val;
    }

    // sinks: add exp(sink - max) to sum
    if (!isnan(sink)) {
        sum += expf(sink - max_val);
    }

    // step 3: normalize
    float inv_sum = 1.0f / sum;
    for (int64_t i = 0; i < ne00; ++i) {
        dp[i] *= inv_sum;
    }
}

typedef struct {
    const ggml_tensor * src0;
    const ggml_tensor * src1;
    const ggml_tensor * src2;
    ggml_tensor * dst;
    int64_t start_idx;
    int64_t end_idx;
    float scale;
    float max_bias;
    worker_synctoken_t *synctoken;
} softmax_thread_data_t;

static void softmax_thread_func(void * data) {
    softmax_thread_data_t * tdata = (softmax_thread_data_t *) data;
    const ggml_tensor * src0 = tdata->src0;
    const ggml_tensor * src1 = tdata->src1;
    const ggml_tensor * src2 = tdata->src2;
    ggml_tensor * dst = tdata->dst;
    int64_t start_idx = tdata->start_idx;
    int64_t end_idx = tdata->end_idx;
    float scale = tdata->scale;
    float max_bias = tdata->max_bias;

    int64_t ne00 = src0->ne[0];
    int64_t ne01 = src0->ne[1];
    int64_t ne02 = src0->ne[2];
    size_t nb01 = src0->nb[1];
    size_t nb02 = src0->nb[2];
    size_t nb03 = src0->nb[3];
    size_t nb1 = dst->nb[1];
    size_t nb2 = dst->nb[2];
    size_t nb3 = dst->nb[3];

    // mask broadcast dims
    int64_t ne12 = src1 ? src1->ne[2] : 1;
    int64_t ne13 = src1 ? src1->ne[3] : 1;
    size_t nb11 = src1 ? src1->nb[1] : 0;
    size_t nb12 = src1 ? src1->nb[2] : 0;
    size_t nb13 = src1 ? src1->nb[3] : 0;

    // sinks: one F32 value per head (ne02 dimension)
    const float * sk = src2 ? (const float *)src2->data : NULL;

    // ALiBi slope
    uint32_t n_head = ne02;
    uint32_t n_head_log2 = 1;
    while (n_head_log2 * 2 <= n_head) n_head_log2 <<= 1;
    float m0 = (max_bias > 0.0f) ? powf(2.0f, -max_bias / n_head_log2) : 1.0f;
    float m1 = (max_bias > 0.0f) ? powf(2.0f, -(max_bias / 2.0f) / n_head_log2) : 1.0f;

    for (int64_t idx = start_idx; idx < end_idx; ++idx) {
        int64_t i01 = idx % ne01;
        int64_t i02 = (idx / ne01) % ne02;
        int64_t i03 = idx / (ne01 * ne02);

        const float * sp = (const float *)((const uint8_t *)src0->data + i01*nb01 + i02*nb02 + i03*nb03);
        float * dp = (float *)((uint8_t *)dst->data + i01*nb1 + i02*nb2 + i03*nb3);

        const void * mp = NULL;
        if (src1) {
            int64_t i11 = i01;
            int64_t i12 = i02 % ne12;
            int64_t i13 = i03 % ne13;
            mp = (const void *)((const uint8_t *)src1->data + i11*nb11 + i12*nb12 + i13*nb13);
        }

        float slope = 1.0f;
        if (max_bias > 0.0f) {
            uint32_t h = i02;
            slope = (h < n_head_log2) ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1);
        }

        float sink = sk ? sk[i02] : NAN;
        soft_max_row_f32_masked(ne00, dp, sp, mp, src1 ? src1->type : GGML_TYPE_F32, scale, slope, sink);
    }

    if (tdata->synctoken != NULL) {
        worker_pool_synctoken_jobdone(tdata->synctoken);
    }
}

static void ggml_compute_forward_soft_max_f32(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * src2,
        ggml_tensor * dst) {

    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__);
    int64_t start_time = ggml_time_us();

    float scale = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale,    dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, dst->op_params + 1, sizeof(float));

    int64_t ne00 = src0->ne[0];
    int64_t nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    // sinks: one F32 value per head (ne02 dimension)
    const float * sk = src2 ? (const float *)src2->data : NULL;

    if (g_dsp_ctx->thread_counts > 1 && nrows >= g_dsp_ctx->thread_counts * 2) {
        int num_threads = g_dsp_ctx->thread_counts;
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
            tdata[i].src1 = src1;
            tdata[i].src2 = src2;
            tdata[i].dst = dst;
            tdata[i].start_idx = idx;
            tdata[i].end_idx = end_idx;
            tdata[i].scale = scale;
            tdata[i].max_bias = max_bias;
            tdata[i].synctoken = &synctoken;

            worker_pool_job_t job;
            job.fptr = softmax_thread_func;
            job.dptr = &tdata[i];
            worker_pool_submit(NULL, job);

            idx = end_idx;
        }

        tdata[num_threads - 1].src0 = src0;
        tdata[num_threads - 1].src1 = src1;
        tdata[num_threads - 1].src2 = src2;
        tdata[num_threads - 1].dst = dst;
        tdata[num_threads - 1].start_idx = idx;
        tdata[num_threads - 1].end_idx = nrows;
        tdata[num_threads - 1].scale = scale;
        tdata[num_threads - 1].max_bias = max_bias;
        tdata[num_threads - 1].synctoken = NULL;

        softmax_thread_func(&tdata[num_threads - 1]);

        worker_pool_synctoken_wait(&synctoken);
    } else {
        // single-thread path
        int64_t ne01 = src0->ne[1];
        int64_t ne02 = src0->ne[2];
        size_t nb01 = src0->nb[1];
        size_t nb02 = src0->nb[2];
        size_t nb03 = src0->nb[3];
        size_t nb1 = dst->nb[1];
        size_t nb2 = dst->nb[2];
        size_t nb3 = dst->nb[3];

        int64_t ne12 = src1 ? src1->ne[2] : 1;
        int64_t ne13 = src1 ? src1->ne[3] : 1;
        size_t nb11 = src1 ? src1->nb[1] : 0;
        size_t nb12 = src1 ? src1->nb[2] : 0;
        size_t nb13 = src1 ? src1->nb[3] : 0;

        uint32_t n_head = ne02;
        uint32_t n_head_log2 = 1;
        while (n_head_log2 * 2 <= n_head) n_head_log2 <<= 1;
        float m0 = (max_bias > 0.0f) ? powf(2.0f, -max_bias / n_head_log2) : 1.0f;
        float m1 = (max_bias > 0.0f) ? powf(2.0f, -(max_bias / 2.0f) / n_head_log2) : 1.0f;

        for (int64_t i03 = 0; i03 < src0->ne[3]; ++i03) {
            for (int64_t i02 = 0; i02 < ne02; ++i02) {
                for (int64_t i01 = 0; i01 < ne01; ++i01) {
                    const float * sp = (const float *)((const uint8_t *)src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                    float * dp = (float *)((uint8_t *)dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                    const void * mp = NULL;
                    if (src1) {
                        int64_t i11 = i01;
                        int64_t i12 = i02 % ne12;
                        int64_t i13 = i03 % ne13;
                        mp = (const void *)((const uint8_t *)src1->data + i11*nb11 + i12*nb12 + i13*nb13);
                    }

                    float slope = 1.0f;
                    if (max_bias > 0.0f) {
                        uint32_t h = i02;
                        slope = (h < n_head_log2) ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1);
                    }

                    float sink = sk ? sk[i02] : NAN;
                    soft_max_row_f32_masked(ne00, dp, sp, mp, src1 ? src1->type : GGML_TYPE_F32, scale, slope, sink);
                }
            }
        }
    }

    int64_t end_time = ggml_time_us();
    GGMLHEXAGON_LOG_DEBUG("SOFT_MAX elapse %lld us (ne00=%lld, nrows=%lld, scale=%f, mask=%d, sinks=%d)",
                         (long long)(end_time - start_time),
                         (long long)ne00, (long long)nrows, scale, src1 ? 1 : 0, src2 ? 1 : 0);
    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__);
}

int ggmlop_dsp_softmax(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, ggml_tensor * dst) {
    GGML_UNUSED(h);
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__);

    int64_t begin_time = ggml_time_us();

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        GGMLHEXAGON_LOG_ERROR("SOFT_MAX: unsupported type src0=%d dst=%d", src0->type, dst->type);
        return AEE_EUNSUPPORTED;
    }

    if (src2 && src2->type != GGML_TYPE_F32) {
        GGMLHEXAGON_LOG_ERROR("SOFT_MAX: unsupported sinks type src2=%d", src2->type);
        return AEE_EUNSUPPORTED;
    }

    ggml_compute_forward_soft_max_f32(src0, src1, src2, dst);

    int64_t end_time = ggml_time_us();
    GGMLHEXAGON_LOG_INFO("elapse time of SOFT_MAX is %lld us", (long long)(end_time - begin_time));

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__);
    return 0;
}
