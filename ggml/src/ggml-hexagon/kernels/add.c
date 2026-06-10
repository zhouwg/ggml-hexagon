#include "ggml-dsp.h"
#include "worker_pool.h"

static inline void l2fetch(const void * p, uint32_t stride,
                           uint32_t width, uint32_t height,
                           uint32_t dir) {
    uint64_t control = HEXAGON_V64_CREATE_H(dir, stride, width, height);
    __asm__ __volatile__ (" l2fetch(%0,%1) " : :"r"(p),"r"(control));
}

static inline void ggmlhexagon_dsp_add_f32(const int n, float * GGML_RESTRICT z, const float * GGML_RESTRICT x, const float * GGML_RESTRICT y) {
    for (size_t i = 0; i < n; ++i) {
        z[i] = x[i] + y[i];
    }
}

typedef struct {
    const ggml_tensor * src0;
    const ggml_tensor * src1;
    ggml_tensor * dst;
    int64_t start_idx;
    int64_t end_idx;
    worker_synctoken_t *synctoken;
} add_thread_data_t;

static void add_thread_func(void * data) {
    add_thread_data_t * tdata = (add_thread_data_t *) data;
    const ggml_tensor * src0 = tdata->src0;
    const ggml_tensor * src1 = tdata->src1;
    ggml_tensor * dst = tdata->dst;
    const int64_t start_idx = tdata->start_idx;
    const int64_t end_idx = tdata->end_idx;

    if (src0->type == GGML_TYPE_F16) {
        uint16_t * dst_ptr  = (uint16_t *)dst->data;
        uint16_t * src0_ptr = (uint16_t *)src0->data;
        uint16_t * src1_ptr = (uint16_t *)src1->data;

        for (int64_t i = start_idx; i < end_idx; ++i) {
            float f0 = ggml_compute_fp16_to_fp32(src0_ptr[i]);
            float f1 = ggml_compute_fp16_to_fp32(src1_ptr[i]);
            float f_result = f0 + f1;
            dst_ptr[i] = ggml_compute_fp32_to_fp16(f_result);
        }
    } else {
        float * dst_ptr  = (float *)dst->data;
        float * src0_ptr = (float *)src0->data;
        float * src1_ptr = (float *)src1->data;

        for (int64_t i = start_idx; i < end_idx; ++i) {
            dst_ptr[i] = src0_ptr[i] + src1_ptr[i];
        }
    }

    if (tdata->synctoken != NULL) {
        worker_pool_synctoken_jobdone(tdata->synctoken);
    }
}


static void ggml_compute_forward_add_f32(
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );
    uint64_t start_time = ggml_time_us();

    ggmlhexagon_dump_tensor(src0, 1);
    ggmlhexagon_dump_tensor(src1, 1);
    ggmlhexagon_dump_tensor(dst, 1);

    const int n = ggml_nelements(src0);

    if (src0->type == GGML_TYPE_F16) {
        uint16_t * dst_ptr  = (uint16_t *)dst->data;
        uint16_t * src0_ptr = (uint16_t *)src0->data;
        uint16_t * src1_ptr = (uint16_t *)src1->data;

        for (int i = 0; i < n; ++i) {
            float f0 = ggml_compute_fp16_to_fp32(src0_ptr[i]);
            float f1 = ggml_compute_fp16_to_fp32(src1_ptr[i]);
            float f_result = f0 + f1;
            dst_ptr[i] = ggml_compute_fp32_to_fp16(f_result);
        }
    } else {
        float * dst_ptr  = (float *)dst->data;
        float * src0_ptr = (float *)src0->data;
        float * src1_ptr = (float *)src1->data;

        for (int i = 0; i < n; ++i) {
            dst_ptr[i] = src0_ptr[i] + src1_ptr[i];
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

static int ggmlop_dsp_add_singlethread(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );
    ggml_compute_forward_add_f32(src0, src1, dst);
    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return 0;
}

static int ggmlop_dsp_add_multithread(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );

    const int64_t n = ggml_nelements(src0);
    const int num_threads = num_workers;

    if (num_threads <= 1 || n < num_threads) {
        return ggmlop_dsp_add_singlethread(h, src0, src1, dst);
    }

    worker_synctoken_t synctoken;
    worker_pool_synctoken_init(&synctoken, num_threads - 1);

    add_thread_data_t tdata[num_threads];
    const int64_t ne_per_thread = n / num_threads;
    int64_t start_idx = 0;

    for (int i = 0; i < num_threads - 1; ++i) {
        int64_t end_idx = start_idx + ne_per_thread;
        tdata[i].src0 = src0;
        tdata[i].src1 = src1;
        tdata[i].dst = dst;
        tdata[i].start_idx = start_idx;
        tdata[i].end_idx = end_idx;
        tdata[i].synctoken = &synctoken;

        worker_pool_job_t job;
        job.fptr = add_thread_func;
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

    add_thread_func(&tdata[num_threads - 1]);

    worker_pool_synctoken_wait(&synctoken);

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return 0;
}

//FIXME: why failed with test-backend-ops when disable ion rpc mempool
int ggmlop_dsp_add(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s\n", __func__);
    if (ggmlop_get_thread_counts() > 1) {
        ggmlop_dsp_add_multithread(h, src0, src1, dst);
    } else {
        ggmlop_dsp_add_singlethread(h, src0, src1, dst);
    }
    GGMLHEXAGON_LOG_DEBUG("leave %s\n", __func__);
    return 0;
}
