#include "ggml-dsp.h"
#include "worker_pool.h"

// GET_ROWS: extract rows from quantized weight matrix by index.
// src0 = quantized weight (q4_K/q5_K), shape [ne00, ne01, ne02, ne03]
// src1 = row indices (i32),         shape [ne10, ne11=ne02, ne12=ne03]
// dst  = dequantized output (f32),  shape [ne00, ne10, ne11, ne12]
//
// For each index i in src1, copy & dequantize row src0[row_idx] into dst[:,i]

// ---- Thread data for parallel GET_ROWS ----
typedef struct {
    const ggml_tensor * src0;
    const ggml_tensor * src1;
    ggml_tensor       * dst;
    int64_t start_idx;
    int64_t end_idx;
    worker_synctoken_t * synctoken;
} getrows_thread_data_t;

static void getrows_thread_func(void * data) {
    getrows_thread_data_t * tdata = (getrows_thread_data_t *) data;
    const ggml_tensor * src0 = tdata->src0;
    const ggml_tensor * src1 = tdata->src1;
    ggml_tensor       * dst  = tdata->dst;

    const int64_t ne00  = src0->ne[0];
    const int64_t ne01  = src0->ne[1];
    const int64_t ne10  = src1->ne[0];
    const int64_t ne11  = src1->ne[1];
    const size_t  nb01  = src0->nb[1];
    const size_t  nb02  = src0->nb[2];
    const size_t  nb03  = src0->nb[3];
    const size_t  nb10  = src1->nb[0];
    const size_t  nb11  = src1->nb[1];
    const size_t  nb12  = src1->nb[2];
    const size_t  nb1   = dst->nb[1];
    const size_t  nb2   = dst->nb[2];
    const size_t  nb3   = dst->nb[3];

    const enum ggml_type type = src0->type;

    for (int64_t i = tdata->start_idx; i < tdata->end_idx; ++i) {
        // decompose flat index i into (i10, i11, i12)
        const int64_t i12 = i / (ne11 * ne10);
        const int64_t i11 = (i - i12 * ne11 * ne10) / ne10;
        const int64_t i10 = i - i12 * ne11 * ne10 - i11 * ne10;

        // read row index from src1
        const int32_t row_idx = *(const int32_t *)(
            (const char *)src1->data + i10 * nb10 + i11 * nb11 + i12 * nb12);

        if (row_idx < 0 || row_idx >= ne01) {
            GGMLHEXAGON_LOG_ERROR("GET_ROWS: row_idx %ld out of range [0, %ld]",
                                  (long)row_idx, (long)ne01);
            continue;
        }

        // source row pointer (quantized data)
        const void * src_row = (const char *)src0->data
                               + row_idx * nb01 + i11 * nb02 + i12 * nb03;

        // destination row pointer (float output)
        float * dst_row = (float *)((char *)dst->data
                                    + i10 * nb1 + i11 * nb2 + i12 * nb3);

        // dequantize based on type
        switch (type) {
            case GGML_TYPE_Q4_K:
                dequantize_row_q4_K((const block_q4_K *)src_row, dst_row, (int)ne00);
                break;
            case GGML_TYPE_Q5_K:
                dequantize_row_q5_K((const block_q5_K *)src_row, dst_row, (int)ne00);
                break;
            default:
                GGMLHEXAGON_LOG_ERROR("GET_ROWS: unsupported type %d (%s)",
                                      type, ggml_get_type_traits(type)->type_name);
                memset(dst_row, 0, ne00 * sizeof(float));
                break;
        }
    }

    if (tdata->synctoken) {
        worker_pool_synctoken_jobdone(tdata->synctoken);
    }
}

// ---- Public entry point ----

int ggmlop_dsp_getrows(remote_handle64 h,
                         const ggml_tensor * src0,
                         const ggml_tensor * src1,
                         ggml_tensor * dst) {
    GGML_UNUSED(h);

    const int64_t nr = src1->ne[0] * src1->ne[1] * src1->ne[2];

    GGMLHEXAGON_LOG_DEBUG("GET_ROWS: src0 type=%s ne=[%ld,%ld,%ld,%ld] -> "
                          "dst f32 ne=[%ld,%ld,%ld,%ld], nr=%ld rows",
                          ggml_get_type_traits(src0->type)->type_name,
                          (long)src0->ne[0], (long)src0->ne[1],
                          (long)src0->ne[2], (long)src0->ne[3],
                          (long)dst->ne[0], (long)dst->ne[1],
                          (long)dst->ne[2], (long)dst->ne[3],
                          (long)nr);

    int num_threads = g_dsp_ctx->thread_counts;
    if (num_threads > nr) num_threads = (int)nr;
    if (num_threads <= 0) num_threads = 1;

    if (num_threads > 1 && nr >= num_threads * 2) {
        worker_synctoken_t synctoken;
        worker_pool_synctoken_init(&synctoken, num_threads - 1);

        getrows_thread_data_t tdata[num_threads];
        int64_t rows_per_thread = (nr + num_threads - 1) / num_threads;
        int64_t idx = 0;

        for (int t = 0; t < num_threads - 1; ++t) {
            int64_t end_idx = idx + rows_per_thread;
            if (end_idx > nr) end_idx = nr;

            tdata[t].src0      = src0;
            tdata[t].src1      = src1;
            tdata[t].dst       = dst;
            tdata[t].start_idx = idx;
            tdata[t].end_idx   = end_idx;
            tdata[t].synctoken = &synctoken;

            worker_pool_job_t job;
            job.fptr = getrows_thread_func;
            job.dptr = &tdata[t];
            worker_pool_submit(NULL, job);

            idx = end_idx;
        }

        // last chunk on calling thread
        tdata[num_threads - 1].src0      = src0;
        tdata[num_threads - 1].src1      = src1;
        tdata[num_threads - 1].dst       = dst;
        tdata[num_threads - 1].start_idx = idx;
        tdata[num_threads - 1].end_idx   = nr;
        tdata[num_threads - 1].synctoken = NULL;

        getrows_thread_func(&tdata[num_threads - 1]);

        worker_pool_synctoken_wait(&synctoken);
    } else {
        // single-threaded fallback
        getrows_thread_data_t single;
        single.src0      = src0;
        single.src1      = src1;
        single.dst       = dst;
        single.start_idx = 0;
        single.end_idx   = nr;
        single.synctoken = NULL;
        getrows_thread_func(&single);
    }

    return 0;
}
