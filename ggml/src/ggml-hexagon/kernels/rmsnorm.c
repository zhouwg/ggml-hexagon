#include "ggml-dsp.h"
#include "worker_pool.h"
#include "../htp/hvx-base.h"
#include "../htp/hvx-reduce.h"
#include "../htp/hvx-sqrt.h"

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

// HVX-accelerated RMS_NORM for f32.
// Requires: 128-byte aligned src/dst, n % VLEN_FP32 == 0.
// Uses qf32 (48-bit mantissa) accumulation for higher precision than plain f32.
static void rmsnorm_f32_hvx(const int n, float * restrict y,
                             const float * restrict x, float eps) {
    const HVX_Vector * restrict v_src = (const HVX_Vector *) x;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) y;
    const int nvec = n / VLEN_FP32;

    // Phase 1: sum of squares via qf32 accumulation
    HVX_Vector sum_v = Q6_V_vsplat_R(0);
    HVX_Vector eps_v = hvx_vec_splat_f32(eps);

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, Q6_Vqf32_vmpy_VsfVsf(v1, v1));
    }

    // Reduce vector sum to a single lane
    sum_v = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_v));

    // mean = sum / n; scale = rsqrt(mean + eps)
    HVX_Vector denom_v    = hvx_vec_splat_f32(1.0f / (float)n);
    HVX_Vector mean_eps_v = Q6_Vqf32_vadd_Vqf32Vsf(
                                Q6_Vqf32_vmpy_VsfVsf(sum_v, denom_v), eps_v);
    HVX_Vector scale_v    = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(mean_eps_v));

    // Phase 2: dst[i] = src[i] * scale
    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        v_dst[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_src[i], scale_v));
    }
}

// Global path-selection counters. Updated by dispatch (runs on worker threads
// where FARF does not reach the DU log). Read out in ggmlop_dsp_rmsnorm which
// runs on the FastRPC thread where FARF output is visible in the DU log.
static int g_rmsnorm_hvx_calls    = 0;
static int g_rmsnorm_scalar_calls = 0;
static int g_rmsnorm_diag_logged  = 0;

static inline void rmsnorm_f32_dispatch(const int n, float * y,
                                         const float * x, float eps) {
    const int use_hvx = ggml_get_dsp_use_hvx()
        && ((uintptr_t)x & 127) == 0
        && ((uintptr_t)y & 127) == 0
        && (n % VLEN_FP32) == 0;

    if (use_hvx) {
        rmsnorm_f32_hvx(n, y, x, eps);
        g_rmsnorm_hvx_calls++;
    } else {
        rmsnorm_f32_scalar(n, y, x, eps);
        g_rmsnorm_scalar_calls++;
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
                rmsnorm_f32_dispatch(ne00, y, x, eps);
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

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    int64_t ne00 = src0->ne[0];
    int64_t nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    GGMLHEXAGON_LOG_DEBUG("RMS_NORM: src0 ne=[%lld,%lld,%lld,%lld] nrows=%lld eps=%f",
                          (long long)src0->ne[0], (long long)src0->ne[1],
                          (long long)src0->ne[2], (long long)src0->ne[3],
                          (long long)nrows, eps);

    if (g_dsp_ctx->thread_counts > 1 && nrows >= g_dsp_ctx->thread_counts * 2) {
        int num_threads = g_dsp_ctx->thread_counts;
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
                    rmsnorm_f32_dispatch(ne00, y, x, eps);
                }
            }
        }
    }

    int64_t end_time = ggml_time_us();
    int64_t duration = end_time - start_time;
    GGMLHEXAGON_LOG_DEBUG("RMS_NORM elapse %lld us (ne00=%lld, nrows=%lld, eps=%f)",
                         (long long)duration, (long long)ne00, (long long)nrows, eps);

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__);
}

int ggmlop_dsp_rmsnorm(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_UNUSED(h);
    GGML_UNUSED(src1);  // unary op, no src1
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__);

    int ne00 = src0->ne[0];
    int nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    int x_algn = ((uintptr_t)src0->data & 127) == 0;
    int y_algn = ((uintptr_t)dst->data  & 127) == 0;
    int n_ok   = (ne00 % VLEN_FP32) == 0;
    int would_hvx = ggml_get_dsp_use_hvx() && x_algn && y_algn && n_ok;

    int hvx_before = g_rmsnorm_hvx_calls;
    int sca_before = g_rmsnorm_scalar_calls;

    GGMLHEXAGON_LOG_INFO("RMSNORM ENTRY: ne=[%d,%d,%d,%d] nrows=%d hvx=%d x_al=%d y_al=%d n_ok=%d",
                         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                         nrows, would_hvx, x_algn, y_algn, n_ok);

    int64_t begin_time = ggml_time_us();

    if (src0->type != GGML_TYPE_F32) {
        GGMLHEXAGON_LOG_ERROR("RMS_NORM: unsupported src0 type %d", src0->type);
        return AEE_EUNSUPPORTED;
    }

    FARF(ALWAYS, "RMSNORM COMPUTE BEGIN\n");
    ggml_compute_forward_rms_norm_f32(src0, dst);
    FARF(ALWAYS, "RMSNORM COMPUTE END hvx=%d sca=%d\n",
         g_rmsnorm_hvx_calls - hvx_before, g_rmsnorm_scalar_calls - sca_before);

    int64_t end_time = ggml_time_us();
    int hd = g_rmsnorm_hvx_calls    - hvx_before;
    int sd = g_rmsnorm_scalar_calls - sca_before;

    GGMLHEXAGON_LOG_INFO("RMSNORM DONE: us=%lld hvx_rows=%d sca_rows=%d (tot hvx=%d sca=%d)",
                         (long long)(end_time - begin_time), hd, sd,
                         g_rmsnorm_hvx_calls, g_rmsnorm_scalar_calls);
    g_rmsnorm_diag_logged++;

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__);
    return 0;
}
