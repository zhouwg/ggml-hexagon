#include "ggml-dsp.h"
#include "worker_pool.h"
#include "sgemm.h"
#include <stdlib.h>
#include <string.h>
#include "../htp/hex-dma.h"    // for Qualcomm's official DMA async transfers

#define HMX_FP16_TILE_N_ROWS 32
#define HMX_FP16_TILE_N_COLS 32
#define HMX_FP16_TILE_N_ELMS 1024
#define HMX_FP16_TILE_SIZE (HMX_FP16_TILE_N_ELMS * sizeof(__fp16))

// FP16 weight cache: stores converted FP16 tiles in ION shared memory tail region
// Keyed by src0->data pointer (stable ION address for same weight tensor)
// Uses ggmlop_cache_mempool_alloc() which allocates from ION tail region
#define FP16_WEIGHT_CACHE_MAX_ENTRIES 64
// Only cache weights whose FP16 tile size exceeds this threshold (bytes)
// Small weights convert quickly and waste cache space; large weights benefit most
#define FP16_WEIGHT_CACHE_MIN_SIZE  (18 * 1024 * 1024)  // 18 MB

typedef struct {
    void *   src0_data;     // key: src0->data pointer (ION DSP address)
    void *   fp16_ptr;      // ION-based FP16 tile buffer (from ggmlop_cache_mempool_alloc)
    uint32_t fp16_size;     // size of FP16 tile buffer in bytes
    int32_t  M;             // weight columns at cache time
    int32_t  K;             // inner dimension at cache time
    int      type;          // weight type at cache time
} fp16_weight_cache_entry_t;

static fp16_weight_cache_entry_t g_fp16_weight_cache[FP16_WEIGHT_CACHE_MAX_ENTRIES];
static int g_fp16_weight_cache_count = 0;

static inline HVX_Vector hvx_vec_repl(HVX_Vector v, const uint8_t * ctrl) {
    return Q6_V_vdelta_VV(v, hvx_vmem(ctrl));
}

static inline HVX_Vector hvx_vec_repl_f16(HVX_Vector v) {
    // vdelta control to replicate first two bytes across all lanes
    static const uint8_t __attribute__((aligned(128))) repl[128] = {
        0x00, 0x00, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
        0x10, 0x10, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
        0x20, 0x20, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
        0x10, 0x10, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
        0x40, 0x40, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
        0x10, 0x10, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
        0x20, 0x20, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
        0x10, 0x10, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02, 0x08, 0x08, 0x02, 0x02, 0x04, 0x04, 0x02, 0x02,
    };
    return hvx_vec_repl(v, repl);
}

// hmx_ceil_div, hex_align_up, hex_align_down are provided by hex-common.h (via hex-dma.h)

static inline HVX_Vector hvx_vec_f32_to_f16_shuff(HVX_Vector v0, HVX_Vector v1) {
#if __HVX_ARCH__ >= 81
    HVX_Vector q0 = Q6_Vqf32_equals_Vsf(v0);
    HVX_Vector q1 = Q6_Vqf32_equals_Vsf(v1);
#else
    const HVX_Vector zero = Q6_V_vzero();
    HVX_Vector q0 = Q6_Vqf32_vadd_VsfVsf(v0, zero);
    HVX_Vector q1 = Q6_Vqf32_vadd_VsfVsf(v1, zero);
#endif
    return Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(q1, q0));
}

static inline HVX_Vector hvx_vec_f32_to_f16(HVX_Vector v0, HVX_Vector v1) {
    HVX_Vector v = Q6_Vh_vdeal_Vh(hvx_vec_f32_to_f16_shuff(v0, v1));

#if __HVX_ARCH__ < 79
    // replace NaNs with -INF, older arches produce NaNs for (-INF + 0.0)
    const HVX_Vector neg_inf = hvx_vec_splat_f16(-INFINITY);
    HVX_VectorPred nan = hvx_vec_is_nan_f16(v);
    v = Q6_V_vmux_QVV(nan, neg_inf, v);
#endif

    return v;
}

static fp16_weight_cache_entry_t * fp16_weight_cache_lookup(void * src0_data) {
    for (int i = 0; i < g_fp16_weight_cache_count; i++) {
        if (g_fp16_weight_cache[i].src0_data == src0_data) {
            return &g_fp16_weight_cache[i];
        }
    }
    return NULL;
}

// Called from entry.c when batch_size == 0xFFFE (cache reset)
void ggmlop_dsp_fp16_cache_reset(void) {
    g_fp16_weight_cache_count = 0;
}

static fp16_weight_cache_entry_t * fp16_weight_cache_insert(void * src0_data, uint32_t fp16_size, int32_t M, int32_t K, int type) {
    if (g_fp16_weight_cache_count >= FP16_WEIGHT_CACHE_MAX_ENTRIES) {
        GGMLHEXAGON_LOG_INFO("FP16 weight cache: table full (%d entries), cannot cache",
                             FP16_WEIGHT_CACHE_MAX_ENTRIES);
        return NULL;
    }
    // Allocate from ION cache region (no size limit, ION tail has ~1.5GB)
    void * ptr = ggmlop_cache_mempool_alloc(fp16_size);
    if (!ptr) {
        GGMLHEXAGON_LOG_INFO("FP16 weight cache: ION alloc(%u) failed, cannot cache", fp16_size);
        return NULL;
    }
    fp16_weight_cache_entry_t * entry = &g_fp16_weight_cache[g_fp16_weight_cache_count++];
    entry->src0_data = src0_data;
    entry->fp16_ptr  = ptr;
    entry->fp16_size = fp16_size;
    entry->M         = M;
    entry->K         = K;
    entry->type      = type;
    GGMLHEXAGON_LOG_INFO("FP16 weight cache: ION alloc %u bytes at %p for src0=%p (M=%d, K=%d, type=%d)",
                         fp16_size, ptr, src0_data, M, K, type);
    return entry;
}

// vscatter offsets for writing FP16 values directly into column-pair interleaved tile format.
// word[i] = i*128 maps column-pair i to byte offset i*128 in the tile.
static const int32_t hmx_transpose_scatter_offsets[32] __attribute__((aligned(VLEN))) = {
    0 * 128,  1 * 128,  2 * 128,  3 * 128,  4 * 128,  5 * 128,  6 * 128,  7 * 128,
    8 * 128,  9 * 128, 10 * 128, 11 * 128, 12 * 128, 13 * 128, 14 * 128, 15 * 128,
   16 * 128, 17 * 128, 18 * 128, 19 * 128, 20 * 128, 21 * 128, 22 * 128, 23 * 128,
   24 * 128, 25 * 128, 26 * 128, 27 * 128, 28 * 128, 29 * 128, 30 * 128, 31 * 128,
};

// IQ4_NL dequantization LUT: maps 4-bit index to fp16 value
static const __fp16 iq4_nl_to_fp16_lut[64] __attribute__((aligned(VLEN))) = {
    -127, 0, -104, 0, -83, 0, -65, 0, -49, 0, -35, 0, -22, 0, -10, 0,
       1, 0,   13, 0,  25, 0,  38, 0,  53, 0,  69, 0,  89, 0, 113, 0,
};

typedef struct {
    const ggml_tensor *src0;
    const ggml_tensor *src1;
    ggml_tensor *dst;
    enum ggml_type type;
    enum ggml_type vec_dot_type;
    int32_t num_rows_per_vec_dot;
    int32_t ir0_start;
    int32_t ir0_end;
    int32_t ir1_start;
    int32_t ir1_end;
    const void *wdata;
    worker_synctoken_t *synctoken;
} mulmat_thread_data_t;

static void ggml_compute_forward_mul_mat_one_chunk(const ggml_tensor *src0, const ggml_tensor *src1,
                                                   struct ggml_tensor *dst,
                                                   const enum ggml_type type,
                                                   const enum ggml_type vec_dot_type,
                                                   const int32_t num_rows_per_vec_dot,
                                                   const int32_t ir0_start, const int32_t ir0_end,
                                                   const int32_t ir1_start, const int32_t ir1_end,
                                                   const void * wdata_precomputed) {
    const bool src1_cont = ggml_is_contiguous(src1);

    const int32_t ne00 = src0->ne[0];
    const int32_t ne01 = src0->ne[1];
    const int32_t ne02 = src0->ne[2];
    const int32_t ne03 = src0->ne[3];

    const int32_t ne10 = src1->ne[0];
    const int32_t ne11 = src1->ne[1];
    const int32_t ne12 = src1->ne[2];
    const int32_t ne13 = src1->ne[3];

    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    const size_t nb11 = src1->nb[1];
    const size_t nb12 = src1->nb[2];
    const size_t nb13 = src1->nb[3];

    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];
    const size_t nb0 = dst->nb[0];

    const int32_t r2 = ne12 / ne02;
    const int32_t r3 = ne13 / ne03;

    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const size_t row_size = ggml_row_size(vec_dot_type, ne10);

    const int32_t blck_0 = 16;
    const int32_t blck_1 = 16;

    const void * wdata;
    if (wdata_precomputed != NULL) {
        wdata = wdata_precomputed;
    } else if (src1->type != vec_dot_type) {
        const size_t nbw1 = row_size;
        const size_t nbw2 = nbw1 * ne11;
        const size_t nbw3 = nbw2 * ne12;
        const size_t q8_size = nbw3 * ne13;
        void * q8_data = ggmlop_get_work_data(q8_size);
        if (q8_data != NULL) {
            const struct ggml_type_traits_dsp * quant_traits = ggml_get_type_traits_dsp(vec_dot_type);
            if (quant_traits->from_float) {
                for (int i13 = 0; i13 < ne13; ++i13) {
                    for (int i12 = 0; i12 < ne12; ++i12) {
                        for (int i11 = 0; i11 < ne11; ++i11) {
                            const float * src_row = (const float*)((const char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);
                            void * dst_row = (void*)((char*)q8_data + i13 * nbw3 + i12 * nbw2 + i11 * nbw1);
                            quant_traits->from_float(src_row, dst_row, ne10);
                        }
                    }
                }
            }
            wdata = q8_data;
        } else {
            wdata = src1->data;
        }
    } else {
        wdata = src1->data;
    }

    const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    const struct ggml_type_traits_dsp * traits = ggml_get_type_traits_dsp(type);
    const ggml_vec_dot_t vec_dot_fn = traits->vec_dot;

    for (int32_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int32_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int32_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                const int32_t i13 = (ir1 / (ne12 * ne11));
                const int32_t i12 = (ir1 - i13 * ne12 * ne11) / ne11;
                const int32_t i11 = (ir1 - i13 * ne12 * ne11 - i12 * ne11);

                const int32_t i03 = i13 / r3;
                const int32_t i02 = i12 / r2;

                const char * src0_row = (const char*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                const char * src1_col = (const char*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                     ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                     : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                float * dst_col = (float*)((char*)dst->data + (i11 * nb1 + i12 * nb2 + i13 * nb3));

                const int32_t block_rows = MIN(iir0 + blck_0, ir0_end) - iir0;

                if (num_rows_per_vec_dot == 1 && vec_dot_fn) {
                    for (int32_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0++) {
                        if (ir0 + 1 < ir0_end) {
                            l2fetch(src0_row + (ir0 + 1) * nb01, nb01, nb01, 1, 0);
                        }
                        vec_dot_fn(ne00, &dst_col[ir0], 0,
                                    src0_row + ir0 * nb01, 0,
                                    src1_col, 0, 1);
                    }
                } else {
                    float tmp[32];
                    for (int32_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                        const int32_t row_idx = ir0 - iir0;
                        if (vec_dot_fn) {
                            vec_dot_fn(ne00, &tmp[row_idx], 0,
                                            src0_row + ir0 * nb01, 0,
                                            src1_col, 0, 1);
                        }
                    }
                    for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                        memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), block_rows * sizeof(float));
                    }
                }
            }
        }
    }
}

static void mulmat_thread_func(void * data) {
    mulmat_thread_data_t * tdata = (mulmat_thread_data_t *) data;

    ggml_compute_forward_mul_mat_one_chunk(
        tdata->src0, tdata->src1, tdata->dst,
        tdata->type, tdata->vec_dot_type,
        tdata->num_rows_per_vec_dot,
        tdata->ir0_start, tdata->ir0_end,
        tdata->ir1_start, tdata->ir1_end,
        tdata->wdata
    );

    if (tdata->synctoken != NULL) {
        worker_pool_synctoken_jobdone(tdata->synctoken);
    }
}

static int ggmlop_dsp_mulmat_singlethread(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );

    dst->ne[0] = src0->ne[1];
    dst->ne[1] = src1->ne[1];
    dst->ne[2] = src1->ne[2];
    dst->ne[3] = src1->ne[3];

    dst->nb[0] = 4;
    dst->nb[1] = dst->nb[0] * dst->ne[0];
    dst->nb[2] = dst->nb[1] * dst->ne[1];
    dst->nb[3] = dst->nb[2] * dst->ne[2];

    const int32_t ne0 = src0->ne[1];
    const int32_t ne1 = src1->ne[1];
    const int32_t ne2 = src1->ne[2];
    const int32_t ne3 = src1->ne[3];

    const int32_t nr0 = ne0;
    const int32_t nr1 = ne1 * ne2 * ne3;

    const enum ggml_type vec_dot_type = ggml_get_type_traits(src0->type)->vec_dot_type;

    int chunk_size = 16;
    if (nr0 == 1 || nr1 == 1) {
        chunk_size = 64;
    }

    int32_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int32_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    nchunk0 = 1;
    nchunk1 = 1;

    const int32_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int32_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    int current_chunk = 0;

    while (current_chunk < nchunk0 * nchunk1) {
        const int32_t ith0 = current_chunk % nchunk0;
        const int32_t ith1 = current_chunk / nchunk0;

        const int32_t ir0_start = dr0 * ith0;
        const int32_t ir0_end = MIN(ir0_start + dr0, nr0);

        const int32_t ir1_start = dr1 * ith1;
        const int32_t ir1_end = MIN(ir1_start + dr1, nr1);

        int32_t num_rows_per_vec_dot = 1;

        if ((nr0 % 2 != 0) || (ne1 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            num_rows_per_vec_dot = 1;
        }

        ggml_compute_forward_mul_mat_one_chunk(src0, src1, dst, src0->type, vec_dot_type, num_rows_per_vec_dot,
                                               ir0_start, ir0_end, ir1_start, ir1_end, NULL);

        if (1 >= nchunk0 * nchunk1) {
            break;
        }
        current_chunk++;
    }

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return 0;
}

static int ggmlop_dsp_mulmat_multithread(remote_handle64 h, const struct dsptensor * src0, const struct dsptensor * src1, dsptensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );

    dst->ne[0] = src0->ne[1];
    dst->ne[1] = src1->ne[1];
    dst->ne[2] = src1->ne[2];
    dst->ne[3] = src1->ne[3];

    dst->nb[0] = 4;
    dst->nb[1] = dst->nb[0] * dst->ne[0];
    dst->nb[2] = dst->nb[1] * dst->ne[1];
    dst->nb[3] = dst->nb[2] * dst->ne[2];

    const int32_t ne0 = src0->ne[1];
    const int32_t ne1 = src1->ne[1];
    const int32_t ne2 = src1->ne[2];
    const int32_t ne3 = src1->ne[3];

    const int32_t nr0 = ne0;
    const int32_t nr1 = ne1 * ne2 * ne3;

    const enum ggml_type vec_dot_type = ggml_get_type_traits(src0->type)->vec_dot_type;

    const void * wdata = src1->data;
    if (src1->type != vec_dot_type) {
        const size_t nbw1 = ggml_row_size(vec_dot_type, src1->ne[0]);
        const size_t nbw2 = nbw1 * src1->ne[1];
        const size_t nbw3 = nbw2 * src1->ne[2];
        const size_t q8_size = nbw3 * src1->ne[3];
        void * q8_data = ggmlop_get_work_data(q8_size);
        if (q8_data != NULL) {
            const struct ggml_type_traits_dsp * quant_traits = ggml_get_type_traits_dsp(vec_dot_type);
            if (quant_traits->from_float) {
                for (int i13 = 0; i13 < src1->ne[3]; ++i13) {
                    for (int i12 = 0; i12 < src1->ne[2]; ++i12) {
                        for (int i11 = 0; i11 < src1->ne[1]; ++i11) {
                            const float * src_row = (const float*)((const char*)src1->data + i13 * src1->nb[3] + i12 * src1->nb[2] + i11 * src1->nb[1]);
                            void * dst_row = (void*)((char*)q8_data + i13 * nbw3 + i12 * nbw2 + i11 * nbw1);
                            quant_traits->from_float(src_row, dst_row, src1->ne[0]);
                        }
                    }
                }
            }
            wdata = q8_data;
        } else {
            GGMLHEXAGON_LOG_ERROR("Failed to allocate work data for mulmat");
            return -1;
        }
    }

    unsigned int n_threads = num_workers;
    if (n_threads < 1) n_threads = 1;
    if (n_threads > 8) n_threads = 8;

    GGMLHEXAGON_LOG_DEBUG("mulmat multithread: num_workers=%u, n_threads=%u, nr1=%d", num_workers, n_threads, nr1);

    if (n_threads == 1) {
        GGMLHEXAGON_LOG_WARN("WARNING: Running single-threaded! num_workers=%u", num_workers);
    }

    const int32_t rows_per_thread = (nr1 + n_threads - 1) / n_threads;

    mulmat_thread_data_t thread_data[MAX_NUM_WORKERS];
    worker_synctoken_t synctoken;

    worker_pool_synctoken_init(&synctoken, n_threads - 1);

    for (unsigned int t = 0; t < n_threads; t++) {
        const int32_t ir1_start = t * rows_per_thread;
        const int32_t ir1_end = MIN(ir1_start + rows_per_thread, nr1);

        thread_data[t].src0 = src0;
        thread_data[t].src1 = src1;
        thread_data[t].dst = dst;
        thread_data[t].type = src0->type;
        thread_data[t].vec_dot_type = vec_dot_type;
        thread_data[t].num_rows_per_vec_dot = 1;
        thread_data[t].ir0_start = 0;
        thread_data[t].ir0_end = nr0;
        thread_data[t].ir1_start = ir1_start;
        thread_data[t].ir1_end = ir1_end;
        thread_data[t].wdata = wdata;
        thread_data[t].synctoken = (t == 0) ? NULL : &synctoken;

        if (t == 0) {
            mulmat_thread_func(&thread_data[t]);
        } else {
            worker_pool_job_t job;
            job.fptr = mulmat_thread_func;
            job.dptr = &thread_data[t];
            worker_pool_submit(NULL, job);
        }
    }

    worker_pool_synctoken_wait(&synctoken);

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return 0;
}

typedef struct {
    const ggml_tensor *src0;
    const ggml_tensor *src1;
    ggml_tensor *dst;
    enum ggml_type type;
    int32_t num_rows_per_vec_dot;
    int32_t ir0_start;
    int32_t ir0_end;
    int32_t ir1_start;
    int32_t ir1_end;
    uint8_t *vtcm_buf;
    size_t vtcm_size;
    dma_queue *dma;
    worker_synctoken_t *synctoken;
} mulmat_thread_data_vtcm_t;

static void ggml_compute_forward_mul_mat_vtcm_chunk(const ggml_tensor *src0, const ggml_tensor *src1,
                                                    struct ggml_tensor *dst,
                                                    const enum ggml_type type,
                                                    const int32_t num_rows_per_vec_dot,
                                                    const int32_t ir0_start, const int32_t ir0_end,
                                                    const int32_t ir1_start, const int32_t ir1_end,
                                                    uint8_t *vtcm_buf, size_t vtcm_size,
                                                    dma_queue *dma) {
    const bool src1_cont = ggml_is_contiguous(src1);

    const int32_t ne00 = src0->ne[0];
    const int32_t ne01 = src0->ne[1];
    const int32_t ne02 = src0->ne[2];
    const int32_t ne03 = src0->ne[3];

    const int32_t ne10 = src1->ne[0];
    const int32_t ne11 = src1->ne[1];
    const int32_t ne12 = src1->ne[2];
    const int32_t ne13 = src1->ne[3];

    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    const size_t nb11 = src1->nb[1];
    const size_t nb12 = src1->nb[2];
    const size_t nb13 = src1->nb[3];

    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];
    const size_t nb0 = dst->nb[0];

    const int32_t r2 = ne12 / ne02;
    const int32_t r3 = ne13 / ne03;

    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const enum ggml_type vec_dot_type = ggml_get_type_traits(type)->vec_dot_type;
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);

    const void * wdata = (src1->type == vec_dot_type) ? src1->data : NULL;

    if (wdata == NULL) {
        const size_t nbw1 = row_size;
        const size_t nbw2 = nbw1 * ne11;
        const size_t nbw3 = nbw2 * ne12;
        const size_t q8_size = nbw3 * ne13;
        void * q8_data = ggmlop_get_work_data(q8_size);
        if (q8_data != NULL) {
            const struct ggml_type_traits_dsp * quant_traits = ggml_get_type_traits_dsp(vec_dot_type);
            if (quant_traits->from_float) {
                for (int i13 = 0; i13 < ne13; ++i13) {
                    for (int i12 = 0; i12 < ne12; ++i12) {
                        for (int i11 = 0; i11 < ne11; ++i11) {
                            const float * src_row = (const float*)((const char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);
                            void * dst_row = (void*)((char*)q8_data + i13 * nbw3 + i12 * nbw2 + i11 * nbw1);
                            quant_traits->from_float(src_row, dst_row, ne10);
                        }
                    }
                }
            }
            wdata = q8_data;
        } else {
            wdata = src1->data;
        }
    }

    const int32_t blck_0 = VTCM_BLOCK_ROWS;
    const int32_t blck_1 = VTCM_BLOCK_COLS;

    const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    const struct ggml_type_traits_dsp * traits = ggml_get_type_traits_dsp(type);
    const ggml_vec_dot_t vec_dot_fn = traits->vec_dot;

    const size_t max_rows_in_vtcm = (vtcm_size / sizeof(float)) / ne00;
    const int32_t rows_per_vtcm_block = MIN(max_rows_in_vtcm, VTCM_BLOCK_ROWS);

    for (int32_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int32_t iir0_base = ir0_start; iir0_base < ir0_end; iir0_base += rows_per_vtcm_block) {
            const int32_t iir0_end = MIN(iir0_base + rows_per_vtcm_block, ir0_end);

            for (int32_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                const int32_t i13 = (ir1 / (ne12 * ne11));
                const int32_t i12 = (ir1 - i13 * ne12 * ne11) / ne11;
                const int32_t i11 = (ir1 - i13 * ne12 * ne11 - i12 * ne11);

                const int32_t i03 = i13 / r3;
                const int32_t i02 = i12 / r2;

                const char * src0_row = (const char*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                const char * src1_col = (const char*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                     ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                     : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                float * dst_col = (float*)((char*)dst->data + (i11 * nb1 + i12 * nb2 + i13 * nb3));

                for (int32_t iir0 = iir0_base; iir0 < iir0_end; iir0 += blck_0) {
                    const int32_t block_rows = MIN(iir0 + blck_0, iir0_end) - iir0;
                    const size_t copy_size = block_rows * nb01;

                    // Use DMA for src0 row copy from DDR to VTCM
                    if (dma) {
                        dma_queue_push_ddr_to_vtcm(dma,
                            dma_make_ptr(vtcm_buf, src0_row + iir0 * nb01),
                            nb01, nb01, block_rows);
                        dma_queue_pop(dma);
                    } else {
                        memcpy(vtcm_buf, src0_row + iir0 * nb01, copy_size);
                    }

                    if (num_rows_per_vec_dot == 1 && vec_dot_fn) {
                        for (int32_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < iir0_end; ir0++) {
                            const int32_t row_idx = ir0 - iir0;
                            vec_dot_fn(ne00, &dst_col[ir0], 0,
                                        vtcm_buf + row_idx * nb01, 0,
                                        src1_col, 0, 1);
                        }
                    } else {
                        float tmp[32];
                        for (int32_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < iir0_end; ir0 += num_rows_per_vec_dot) {
                            const int32_t row_idx = ir0 - iir0;
                            if (vec_dot_fn) {
                                vec_dot_fn(ne00, &tmp[row_idx], 0,
                                                vtcm_buf + row_idx * nb01, 0,
                                                src1_col, 0, 1);
                            }
                        }
                        for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                            memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), block_rows * sizeof(float));
                        }
                    }
                }
            }
        }
    }
}

static void mulmat_thread_func_vtcm(void * data) {
    mulmat_thread_data_vtcm_t * tdata = (mulmat_thread_data_vtcm_t *) data;

    ggml_compute_forward_mul_mat_vtcm_chunk(
        tdata->src0, tdata->src1, tdata->dst,
        tdata->type, tdata->num_rows_per_vec_dot,
        tdata->ir0_start, tdata->ir0_end,
        tdata->ir1_start, tdata->ir1_end,
        tdata->vtcm_buf, tdata->vtcm_size,
        tdata->dma
    );

    if (tdata->synctoken != NULL) {
        worker_pool_synctoken_jobdone(tdata->synctoken);
    }
}

static int ggmlop_dsp_mulmat_multithread_vtcm(remote_handle64 h, const struct dsptensor * src0, const struct dsptensor * src1, dsptensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );

    dst->ne[0] = src0->ne[1];
    dst->ne[1] = src1->ne[1];
    dst->ne[2] = src1->ne[2];
    dst->ne[3] = src1->ne[3];

    dst->nb[0] = 4;
    dst->nb[1] = dst->nb[0] * dst->ne[0];
    dst->nb[2] = dst->nb[1] * dst->ne[1];
    dst->nb[3] = dst->nb[2] * dst->ne[2];

    const int32_t ne0 = src0->ne[1];
    const int32_t ne1 = src1->ne[1];
    const int32_t ne2 = src1->ne[2];
    const int32_t ne3 = src1->ne[3];

    const int32_t nr0 = ne0;
    const int32_t nr1 = ne1 * ne2 * ne3;

    unsigned int n_threads = num_workers;
    if (n_threads < 1) n_threads = 1;
    if (n_threads > 8) n_threads = 8;

    // In cache mode, VTCM must be acquired before each use
    int vtcm_err = ggmlop_ensure_vtcm_available();
    if (vtcm_err != 0) {
        GGMLHEXAGON_LOG_INFO("%s: VTCM ensure failed (%d), falling back to multithread-without-vtcm",
                             __func__, vtcm_err);
        return ggmlop_dsp_mulmat_multithread(h, src0, src1, dst);
    }

    // Use pre-allocated VTCM pool instead of HAP_request_VTCM
    // (VTCM pool is allocated at init time via HAP_compute_res_acquire)
    size_t pool_size = 0;
    void *vtcm_base = ggmlop_get_vtcm_pool(&pool_size);
    if (vtcm_base == NULL) {
        GGMLHEXAGON_LOG_INFO("%s: VTCM pool unavailable, falling back to multithread-without-vtcm",
                             __func__);
        return ggmlop_dsp_mulmat_multithread(h, src0, src1, dst);
    }

    // Calculate vtcm_per_thread as the largest power-of-2 that fits all threads
    // This ensures alignment safety (DMA/HVX friendly) and maximizes VTCM utilization
    size_t vtcm_per_thread = 64 * 1024;  // minimum 64KB
    while (vtcm_per_thread * 2 * n_threads <= pool_size) {
        vtcm_per_thread *= 2;
    }

    const int32_t rows_per_thread = (nr1 + n_threads - 1) / n_threads;

    // Create DMA queues for each thread
    dma_queue *dma_queues[MAX_NUM_WORKERS];
    for (unsigned int t = 0; t < n_threads; t++) {
        dma_queues[t] = dma_queue_create(16);
    }

    mulmat_thread_data_vtcm_t thread_data[MAX_NUM_WORKERS];
    worker_synctoken_t synctoken;

    worker_pool_synctoken_init(&synctoken, n_threads - 1);

    for (unsigned int t = 0; t < n_threads; t++) {
        const int32_t ir1_start = t * rows_per_thread;
        const int32_t ir1_end = MIN(ir1_start + rows_per_thread, nr1);

        thread_data[t].src0 = src0;
        thread_data[t].src1 = src1;
        thread_data[t].dst = dst;
        thread_data[t].type = src0->type;
        thread_data[t].num_rows_per_vec_dot = 1;
        thread_data[t].ir0_start = 0;
        thread_data[t].ir0_end = nr0;
        thread_data[t].ir1_start = ir1_start;
        thread_data[t].ir1_end = ir1_end;
        thread_data[t].vtcm_buf = (uint8_t *)vtcm_base + t * vtcm_per_thread;
        thread_data[t].vtcm_size = vtcm_per_thread;
        thread_data[t].dma = dma_queues[t];
        thread_data[t].synctoken = (t == 0) ? NULL : &synctoken;

        if (t == 0) {
            mulmat_thread_func_vtcm(&thread_data[t]);
        } else {
            worker_pool_job_t job;
            job.fptr = mulmat_thread_func_vtcm;
            job.dptr = &thread_data[t];
            worker_pool_submit(NULL, job);
        }
    }

    worker_pool_synctoken_wait(&synctoken);

    // Flush and delete DMA queues
    for (unsigned int t = 0; t < n_threads; t++) {
        dma_queue_flush(dma_queues[t]);
        dma_queue_delete(dma_queues[t]);
    }

    // VTCM pool is pre-allocated, no need to release

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return 0;
}

// Transfer activation chunk from fp32 to fp16 tiles
// Uses FP16 Crouton layout (interleaved format for activation)
// Reference: htp/hmx-matmul-ops.c transfer_activation_chunk_fp32_to_fp16
//
// Input data layout in VTCM buffer (after column-major to row-major conversion):
// - Buffer has n_rows rows, each row has n_cols elements
// - n_rows = N (activation columns, batch size)
// - n_cols = K (inner dimension)
// - src[row][col] = src[row * row_stride + col]
//
// FP16 Crouton layout for activation (interleaved format from hvx_vec_f32_to_f16_shuff):
// - Each tile is 32x32 fp16 elements (2048 bytes)
// - Organized as 16 row pairs, each pair has 64 fp16
// - Within each row pair: interleaved format
// - tile[(r1/2) * 64 + j*2 + 0] = row0 data
// - tile[(r1/2) * 64 + j*2 + 1] = row1 data
void transfer_activation_chunk_fp32_to_fp16(__fp16 *restrict vtcm_dst, const float *restrict src,
                                                   int n_rows, int n_cols, int row_stride) {
    // n_rows = N (activation columns in VTCM buffer)
    // n_cols = K (inner dimension, elements per row)
    // row_stride = K (stride in VTCM buffer)
    const int n_rows_padded = ((n_rows + HMX_FP16_TILE_N_ROWS - 1) / HMX_FP16_TILE_N_ROWS) * HMX_FP16_TILE_N_ROWS;
    const int n_rows_tiled  = n_rows_padded;  // HVX processes all rows including padding (padding rows read uninitialized VTCM fp32, but we zero-init the fp32 buffer)
    const int n_tiles_per_row = n_cols / HMX_FP16_TILE_N_COLS;

    // Zero-initialize padding rows in the fp32 source buffer so HVX can safely convert them to fp16 0.0
    for (int r = n_rows; r < n_rows_padded; ++r) {
        memset((void *)(src + r * row_stride), 0, n_cols * sizeof(float));
    }

    int r = 0;

    // Process all rows (including padding) using HVX vector operations
    #pragma unroll(2)
    for (r = 0; r < n_rows_tiled; r += 2) {
        int r0 = r / HMX_FP16_TILE_N_ROWS;  // tile row index
        int r1 = r % HMX_FP16_TILE_N_ROWS;  // intra-tile row idx

        const HVX_Vector *pv_in0 = (const HVX_Vector *) (src + (r + 0) * row_stride);
        const HVX_Vector *pv_in1 = (const HVX_Vector *) (src + (r + 1) * row_stride);
        for (int c = 0; c < n_cols; c += 32) {
            HVX_Vector v0 = *pv_in0++;
            HVX_Vector v1 = *pv_in1++;

            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            int c0       = c / HMX_FP16_TILE_N_COLS;  // tile column index
            int tile_idx = r0 * n_tiles_per_row + c0;

            __fp16 *tile_base = vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS;
            HVX_Vector *tile_hvx = (HVX_Vector *)tile_base;
            tile_hvx[r1 / 2] = v_out;
        }
    }
}

// Transfer activation chunk from f16 to f16 tiles
// Uses FP16 Crouton layout (interleaved format for activation, same as hvx_vec_f32_to_f16_shuff)
// Reference: htp/hmx-matmul-ops.c transfer_activation_chunk_fp32_to_fp16
static void transfer_activation_chunk_f16_to_f16_tiles(__fp16 *restrict vtcm_dst, const __fp16 *restrict src,
                                                        int n_rows, int k, int row_stride) {
    const int n_rows_padded = ((n_rows + HMX_FP16_TILE_N_ROWS - 1) / HMX_FP16_TILE_N_ROWS) * HMX_FP16_TILE_N_ROWS;
    const int n_tiles_per_row = k / HMX_FP16_TILE_N_COLS;

    // Process all rows (including padded)
    for (int r = 0; r < n_rows_padded; r += 2) {
        int r0 = r / HMX_FP16_TILE_N_ROWS;  // tile row index
        int r1 = r % HMX_FP16_TILE_N_ROWS;  // intra-tile row idx

        const __fp16 *src_row0 = (r < n_rows) ? src + (r + 0) * row_stride : NULL;
        const __fp16 *src_row1 = (r + 1 < n_rows) ? src + (r + 1) * row_stride : NULL;

        for (int c = 0; c < k; c += HMX_FP16_TILE_N_COLS) {
            int c0 = c / HMX_FP16_TILE_N_COLS;  // tile column index
            int tile_idx = r0 * n_tiles_per_row + c0;

            __fp16 *tile_base = vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS;

            // FP16 Crouton layout (interleaved format):
            for (int i = 0; i < HMX_FP16_TILE_N_COLS; ++i) {
                tile_base[(r1 / 2) * 64 + i * 2] =
                    (src_row0) ? src_row0[c + i] : (__fp16)0;
            }
            for (int i = 0; i < HMX_FP16_TILE_N_COLS; ++i) {
                tile_base[(r1 / 2) * 64 + i * 2 + 1] =
                    (src_row1) ? src_row1[c + i] : (__fp16)0;
            }
        }
    }
    __asm__ __volatile__("" ::: "memory");
}

// HVX-accelerated F16 activation transfer using vscatter for row-pair interleaved tile format
static void transfer_activation_chunk_f16_to_f16_tiles_hvx(__fp16 *restrict vtcm_dst, const __fp16 *restrict src,
                                                            int n_rows, int k, int row_stride) {
    const int n_row_tiles = (n_rows + HMX_FP16_TILE_N_ROWS - 1) / HMX_FP16_TILE_N_ROWS;
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;

    const HVX_Vector v_scat_base = hvx_vmem(hmx_transpose_scatter_offsets);
    const HVX_Vector v_scat_step = Q6_V_vsplat_R(4);
    const HVX_VectorPred q_mask64 = Q6_Q_vsetq_R(64);
    const size_t single_region = (size_t)(HMX_FP16_TILE_SIZE - 1);

    for (int t = 0; t < n_row_tiles * k_tiles; ++t) {
        int rt = t / k_tiles;
        int kt = t % k_tiles;
        __fp16 *tile_base = vtcm_dst + t * HMX_FP16_TILE_N_ELMS;

        for (int i = 0; i < HMX_FP16_TILE_N_ROWS; i += 2) {
            int row0 = rt * HMX_FP16_TILE_N_ROWS + i;
            int row1 = row0 + 1;
            const __fp16 *r0 = (row0 < n_rows) ? src + row0 * row_stride + kt * HMX_FP16_TILE_N_COLS : NULL;
            const __fp16 *r1 = (row1 < n_rows) ? src + row1 * row_stride + kt * HMX_FP16_TILE_N_COLS : NULL;

            HVX_Vector v0 = r0 ? hvx_vmemu(r0) : Q6_V_vzero();
            HVX_Vector v1 = r1 ? hvx_vmemu(r1) : Q6_V_vzero();

            const HVX_Vector v_off0 = Q6_Vw_vadd_VwVw(v_scat_base, Q6_V_vsplat_R(i * 4));
            const HVX_Vector v_off1 = Q6_Vw_vadd_VwVw(v_off0, v_scat_step);
            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, single_region, v_off0, v0);
            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, single_region, v_off1, v1);
        }
    }
}

// Convert weight chunk from fp32 to fp16 tiles
// Uses FP16 Crouton layout (column-pair interleaved format for weight)
// Reference: htp/hmx-matmul-ops.c convert_f16_weight_to_fp16_tiles_task,
//           htp/hmx-utils.h hmx_interleave_rows_to_tiles
//
// Weight VTCM buffer layout: row-major format (after memcpy from column-major src0)
// - Buffer stores weight [K, M] as row-major: buf[m * K + k] = weight[k, m]
// - Each row has K elements, total M rows
// - src[row * K + col] = element at (row, col) where row=m, col=k
//
// Weight tiles layout: organized by column tiles (M dimension)
// - Each tile contains 32 rows of weight data (M dimension)
// - Tile index: ct * n_dot_tiles + kt, where ct is column tile index (M dimension)
// - This matches core_dot_chunk_fp16's access: weight + c * n_dot_tiles * TILE_SIZE
//
// FP16 Crouton layout for weight (column-pair interleaved format):
// - Each tile is 32x32 fp16 elements (2048 bytes)
// - Organized as 16 column pairs, each pair has 64 fp16
// - Within each column pair: interleaved format
// - tile[(j/2)*64 + i*2 + (j%2)] = tile[i, j]
static void convert_weight_f32_to_fp16_tiles_hvx(__fp16 *restrict vtcm_dst, const float *restrict src,
                                                   int n_cols, int k, int col_stride) {
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int n_col_tiles = (n_cols + HMX_FP16_TILE_N_COLS - 1) / HMX_FP16_TILE_N_COLS;

    for (int t = 0; t < n_col_tiles * k_tiles; ++t) {
        int ct = t / k_tiles;
        int kt = t % k_tiles;
        __fp16 *tile_base = vtcm_dst + t * HMX_FP16_TILE_N_ELMS;

        for (int i = 0; i < HMX_FP16_TILE_N_ROWS; i += 2) {
            int m_idx0 = ct * HMX_FP16_TILE_N_COLS + i;
            int m_idx1 = m_idx0 + 1;
            const float *r0 = (m_idx0 < n_cols) ? src + m_idx0 * col_stride + kt * HMX_FP16_TILE_N_COLS : NULL;
            const float *r1 = (m_idx1 < n_cols) ? src + m_idx1 * col_stride + kt * HMX_FP16_TILE_N_COLS : NULL;

            HVX_Vector v0 = r0 ? *(const HVX_Vector *)r0 : Q6_V_vzero();
            HVX_Vector v1 = r1 ? *(const HVX_Vector *)r1 : Q6_V_vzero();
            HVX_Vector v_fp16 = hvx_vec_f32_to_f16(v0, v1);

            // v_fp16 has 64 FP16: [row0[0..31], row1[0..31]] (after vdeal)
            __fp16 tmp[64] __attribute__((aligned(128)));
            *(HVX_Vector *)tmp = v_fp16;

            // Column-pair interleaved format: tile[(j/2)*64 + i*2 + (j%2)]
            for (int j = 0; j < HMX_FP16_TILE_N_COLS; ++j) {
                tile_base[(j / 2) * 64 + i * 2 + (j % 2)] = tmp[j];
                tile_base[(j / 2) * 64 + (i + 1) * 2 + (j % 2)] = tmp[32 + j];
            }
        }
    }
    __asm__ __volatile__("" ::: "memory");
}

static void convert_weight_f32_to_fp16_tiles(__fp16 *restrict vtcm_dst, const float *restrict src,
                                              int n_cols, int k, int col_stride) {
    // CRITICAL FIX: vtcm_weight_fp32_buf has [M, K] layout after copying from src0
    // - Copy loop: for (i = 0; i < M_cols; ++i) memcpy(vtcm_buf + i * K, src0 + i * src0_stride, K * sizeof(float))
    // - This creates [M, K] layout in vtcm_buf: vtcm_buf[m * K + k] = weight[m, k]
    // - We need to read weight[m, k] and store in tile[m, k] layout
    //
    // n_cols = M (output dimension)
    // k = K (inner dimension)
    // col_stride = K (stride in vtcm_buf, which is [M, K] layout)
    //
    // Weight tiles layout: organized by column tiles (M dimension)
    // - Each tile contains 32 rows of weight data (M dimension)
    // - Tile rows correspond to M dimension, columns to K dimension
    // - tile[i, j] should contain weight[m, k] where m = ct*32+i, k = kt*32+j

    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int n_col_tiles = (n_cols + HMX_FP16_TILE_N_COLS - 1) / HMX_FP16_TILE_N_COLS;
    const int n_tot_tiles = n_col_tiles * k_tiles;

    // Process all tiles
    for (int t = 0; t < n_tot_tiles; ++t) {
        int ct = t / k_tiles;  // column tile index (M dimension)
        int kt = t % k_tiles;  // K tile index (inner dimension)

        __fp16 *tile_base = vtcm_dst + t * HMX_FP16_TILE_N_ELMS;

        for (int i = 0; i < HMX_FP16_TILE_N_ROWS; ++i) {  // 32 rows per tile (M dimension)
            int m_idx = ct * HMX_FP16_TILE_N_COLS + i;  // global M index
            for (int j = 0; j < HMX_FP16_TILE_N_COLS; ++j) {  // 32 columns per tile (K dimension)
                int k_idx = kt * HMX_FP16_TILE_N_COLS + j;  // global K index

                // Read weight[m, k] from vtcm_buf [M, K] layout
                // vtcm_buf[m * K + k] = weight[m, k]
                // So we read: src[m_idx * col_stride + k_idx]
                float val = (m_idx < n_cols && k_idx < k) ?
                            src[m_idx * col_stride + k_idx] : 0.0f;

                // Column-pair interleaved format: tile[(j/2)*64 + i*2 + (j%2)]
                tile_base[(j / 2) * 64 + i * 2 + (j % 2)] = (__fp16)val;
            }
        }
    }
    __asm__ __volatile__("" ::: "memory");
}

// Transfer weight chunk from f16 to f16 tiles
// Uses FP16 Crouton layout (column-pair interleaved format for weight)
// Reference: htp/hmx-matmul-ops.c convert_f16_weight_to_fp16_tiles_task
static void transfer_weight_chunk_f16_to_f16_tiles(__fp16 *restrict vtcm_dst, const __fp16 *restrict src,
                                                    int n_cols, int k, int row_stride) {
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int n_col_tiles = (n_cols + HMX_FP16_TILE_N_COLS - 1) / HMX_FP16_TILE_N_COLS;
    const int n_tot_tiles = n_col_tiles * k_tiles;

    // Process all tiles (matching test-hmx.c implementation)
    for (int t = 0; t < n_tot_tiles; ++t) {
        int ct = t / k_tiles;  // N tile index (column tile)
        int kt = t % k_tiles;  // K tile index (row tile)

        __fp16 *tile_base = vtcm_dst + t * HMX_FP16_TILE_N_ELMS;

        for (int i = 0; i < HMX_FP16_TILE_N_ROWS; ++i) {  // 32 rows per tile (M dimension)
            int row_idx = ct * HMX_FP16_TILE_N_COLS + i;  // global M index (row index in VTCM buffer)
            for (int j = 0; j < HMX_FP16_TILE_N_COLS; ++j) {  // 32 columns per tile (K dimension)
                int col_idx = kt * HMX_FP16_TILE_N_COLS + j;  // global K index (column index in VTCM buffer)

                // Read from src: src[row_idx * row_stride + col_idx]
                // VTCM buffer layout: [M, K] row-major format
                __fp16 val = (row_idx < n_cols && col_idx < k) ?
                             src[row_idx * row_stride + col_idx] : (__fp16)0;

                // Column-pair interleaved format: tile[(j/2)*64 + i*2 + (j%2)]
                tile_base[(j / 2) * 64 + i * 2 + (j % 2)] = val;
            }
        }
    }
    __asm__ __volatile__("" ::: "memory");
}

// HVX-accelerated F16 weight transfer using vscatter for column-pair interleaved tile format
static void transfer_weight_chunk_f16_to_f16_tiles_hvx(__fp16 *restrict vtcm_dst, const __fp16 *restrict src,
                                                        int n_cols, int k, int row_stride) {
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int n_col_tiles = (n_cols + HMX_FP16_TILE_N_COLS - 1) / HMX_FP16_TILE_N_COLS;

    const HVX_Vector v_scat_base = hvx_vmem(hmx_transpose_scatter_offsets);
    const HVX_Vector v_scat_step = Q6_V_vsplat_R(4);
    const HVX_VectorPred q_mask64 = Q6_Q_vsetq_R(64);
    const size_t single_region = (size_t)(HMX_FP16_TILE_SIZE - 1);

    for (int t = 0; t < n_col_tiles * k_tiles; ++t) {
        int ct = t / k_tiles;
        int kt = t % k_tiles;
        __fp16 *tile_base = vtcm_dst + t * HMX_FP16_TILE_N_ELMS;

        HVX_Vector v_off = v_scat_base;
        for (int i = 0; i < HMX_FP16_TILE_N_ROWS; i += 2) {
            int row0 = ct * HMX_FP16_TILE_N_ROWS + i;
            int row1 = row0 + 1;
            const __fp16 *r0 = (row0 < n_cols) ? src + row0 * row_stride + kt * HMX_FP16_TILE_N_COLS : NULL;
            const __fp16 *r1 = (row1 < n_cols) ? src + row1 * row_stride + kt * HMX_FP16_TILE_N_COLS : NULL;

            HVX_Vector v0 = r0 ? hvx_vmemu(r0) : Q6_V_vzero();
            HVX_Vector v1 = r1 ? hvx_vmemu(r1) : Q6_V_vzero();

            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, single_region, v_off, v0);
            v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, single_region, v_off, v1);
            v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
        }
        // Ensure vscatter completion for this tile (like Qualcomm reference code)
        (void) *(volatile HVX_Vector *)(tile_base);
    }
    // Final fence: read from last tile
    if (n_col_tiles * k_tiles > 0) {
        (void) *(volatile HVX_Vector *)(vtcm_dst + (n_col_tiles * k_tiles - 1) * HMX_FP16_TILE_N_ELMS);
    }
}

void core_dot_chunk_fp16(__fp16 *restrict output, const __fp16 *restrict activation,
                                const __fp16 *restrict weight, const __fp16 *restrict scales,
                                int n_row_tiles, int n_col_tiles, int n_dot_tiles) {
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);

    Q6_bias_mxmem2_A((void *)scales);

    for (int r = 0; r < n_row_tiles; ++r) {
        for (int c = 0; c < n_col_tiles; ++c) {
            Q6_mxclracc_hf();

            const __fp16 *row_tiles = activation + r * n_dot_tiles * HMX_FP16_TILE_N_ELMS;
            const __fp16 *col_tiles = weight + c * n_dot_tiles * HMX_FP16_TILE_N_ELMS;

            for (int k = 0, k_block; k < n_dot_tiles; k += k_block) {
                k_block = (n_dot_tiles - k) > 32 ? 32 : (n_dot_tiles - k);
                const uint32_t range = 2048u * (uint32_t)k_block - 1;  // CRITICAL: range = tile_size * k_block - 1

                Q6_activation_hf_mxmem_RR_deep((unsigned int)row_tiles, range);
                Q6_weight_hf_mxmem_RR((unsigned int)col_tiles, range);

                row_tiles += k_block * HMX_FP16_TILE_N_ELMS;
                col_tiles += k_block * HMX_FP16_TILE_N_ELMS;
            }

            __fp16 *out_tile = output + (r * n_col_tiles + c) * HMX_FP16_TILE_N_ELMS;
            Q6_mxmem_AR_after_hf(out_tile, 0);
        }
    }
}

// Transfer output chunk from fp16 tiles to fp32
// Uses FP16 Crouton layout (interleaved format for output, same as activation)
// Reference: htp/hmx-matmul-ops.c transfer_output_chunk_fp16_to_fp32
//
// HMX output tiles use the same interleaved format as activation:
// - Each tile is 32x32 fp16 elements (2048 bytes)
// - Organized as 16 row pairs, each pair has 64 fp16
// - Within each row pair: interleaved format (from hvx_vec_f32_to_f16_shuff)
// - tile[(r1/2)*64 + j*2 + 0] = row0 data
// - tile[(r1/2)*64 + j*2 + 1] = row1 data
//
// Parameters:
// - dst: output chunk pointer (points to dst[nr, mc])
// - src: HMX output tiles (chunk layout, relative indices)
// - n_rows: chunk rows (N dimension)
// - n_cols: chunk cols (M dimension)
// - col_stride: M (dst row count)
void transfer_output_chunk_fp16_to_fp32(float *restrict dst, const __fp16 *restrict src,
                                                int n_rows, int n_cols, int col_stride) {
    // HMX output uses interleaved format (same layout as activation):
    // output_tile[r, j] is at tile[(r/2)*64 + j*2 + (r%2)]
    // We read output_tile[r, j] and write to dst[r * col_stride + (c + j)]

    const int n_row_tiles = (n_rows + HMX_FP16_TILE_N_ROWS - 1) / HMX_FP16_TILE_N_ROWS;
    const int n_col_tiles = (n_cols + HMX_FP16_TILE_N_COLS - 1) / HMX_FP16_TILE_N_COLS;

    // Process all rows in pairs (interleaved format stores row pairs)
    for (int r = 0; r < n_rows; r += 2) {
        int r0 = r / HMX_FP16_TILE_N_ROWS;  // chunk-relative N tile index
        int intra_tile_row = r % HMX_FP16_TILE_N_ROWS;  // intra-tile row index (0-31)
        int row_pair = intra_tile_row / 2;  // row pair index (0-15)
        // For row r (even): offset = row_pair * 64
        // For row r+1 (odd): offset = row_pair * 64 + 32

        for (int c = 0; c < n_cols; c += HMX_FP16_TILE_N_COLS) {
            int c0 = c / HMX_FP16_TILE_N_COLS;  // chunk-relative M tile index
            int tile_idx = r0 * n_col_tiles + c0;  // chunk-relative tile index
            const __fp16 *tile = src + tile_idx * HMX_FP16_TILE_N_ELMS;

            // Interleaved format: tile[(row_pair)*64 + j*2 + row_offset]
            // - row r (even): row_offset = 0 -> tile[row_pair * 64 + j*2]
            // - row r+1 (odd): row_offset = 1 -> tile[row_pair * 64 + j*2 + 1]
            int j_max = (c + HMX_FP16_TILE_N_COLS <= n_cols) ? HMX_FP16_TILE_N_COLS : (n_cols - c);
            for (int j = 0; j < j_max; ++j) {
                dst[(c + j) + r * col_stride] = (float)tile[row_pair * 64 + j * 2];
            }
            if (r + 1 < n_rows) {
                for (int j = 0; j < j_max; ++j) {
                    dst[(c + j) + (r + 1) * col_stride] = (float)tile[row_pair * 64 + j * 2 + 1];
                }
            }
        }
    }
}

// Helper: convert float to __fp16 via ggml_fp16_t bit pattern
// ggml_compute_fp32_to_fp16 returns uint16_t (raw FP16 bits), not __fp16
static inline void fp32_to_fp16_store(__fp16 *dst, float val) {
    ggml_fp16_t bits = ggml_compute_fp32_to_fp16(val);
    memcpy(dst, &bits, sizeof(__fp16));
}

static void dequantize_q4_0_to_f16_tiles(__fp16 *restrict vtcm_dst, const block_q4_0 *restrict src,
                                         int n_cols, int k) {
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int n_col_tiles = (n_cols + HMX_FP16_TILE_N_COLS - 1) / HMX_FP16_TILE_N_COLS;
    const int n_tot_tiles = n_col_tiles * k_tiles;
    const int nb_per_col = k / QK4_0;

    // Process all tiles (matching convert_weight_f32_to_fp16_tiles structure)
    for (int t = 0; t < n_tot_tiles; ++t) {
        int ct = t / k_tiles;  // N tile index (column tile)
        int kt = t % k_tiles;  // K tile index (row tile)

        __fp16 *tile_base = vtcm_dst + t * HMX_FP16_TILE_N_ELMS;

        for (int i = 0; i < HMX_FP16_TILE_N_ROWS; ++i) {  // 32 rows per tile (N dimension)
            int row_global = ct * HMX_FP16_TILE_N_ROWS + i;  // global N index
            const block_q4_0 *row_blocks = (row_global < n_cols) ?
                                           src + row_global * nb_per_col : NULL;

            for (int j = 0; j < HMX_FP16_TILE_N_COLS; ++j) {  // 32 columns per tile (K dimension)
                int col_global = kt * HMX_FP16_TILE_N_COLS + j;  // global K index
                int block_idx = col_global / QK4_0;
                int elem_idx = col_global % QK4_0;

                float val = 0.0f;
                if (row_blocks && col_global < k) {
                    float d = ggml_compute_fp16_to_fp32(row_blocks[block_idx].d);
                    // Q4_0: qs[j] lower nibble -> element j, upper nibble -> element j+16
                    int8_t q;
                    if (elem_idx < 16) {
                        q = (row_blocks[block_idx].qs[elem_idx] & 0x0F);
                    } else {
                        q = (row_blocks[block_idx].qs[elem_idx - 16] >> 4);
                    }
                    val = (q - 8) * d;
                }

                // Column-pair interleaved format: tile[(j/2)*64 + i*2 + (j%2)]
                fp32_to_fp16_store(&tile_base[(j / 2) * 64 + i * 2 + (j % 2)], val);
            }
        }
    }
}

static void dequantize_q4_1_to_f16_tiles(__fp16 *restrict vtcm_dst, const block_q4_1 *restrict src,
                                         int n_cols, int k) {
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int n_col_tiles = (n_cols + HMX_FP16_TILE_N_COLS - 1) / HMX_FP16_TILE_N_COLS;
    const int n_tot_tiles = n_col_tiles * k_tiles;
    const int nb_per_col = k / QK4_1;

    // Process all tiles (matching convert_weight_f32_to_fp16_tiles structure)
    for (int t = 0; t < n_tot_tiles; ++t) {
        int ct = t / k_tiles;  // N tile index (column tile)
        int kt = t % k_tiles;  // K tile index (row tile)

        __fp16 *tile_base = vtcm_dst + t * HMX_FP16_TILE_N_ELMS;

        for (int i = 0; i < HMX_FP16_TILE_N_ROWS; ++i) {  // 32 rows per tile (N dimension)
            int row_global = ct * HMX_FP16_TILE_N_ROWS + i;  // global N index
            const block_q4_1 *row_blocks = (row_global < n_cols) ?
                                           src + row_global * nb_per_col : NULL;

            for (int j = 0; j < HMX_FP16_TILE_N_COLS; ++j) {  // 32 columns per tile (K dimension)
                int col_global = kt * HMX_FP16_TILE_N_COLS + j;  // global K index
                int block_idx = col_global / QK4_1;
                int elem_idx = col_global % QK4_1;

                float val = 0.0f;
                if (row_blocks && col_global < k) {
                    float d = ggml_compute_fp16_to_fp32(row_blocks[block_idx].d);
                    float m = ggml_compute_fp16_to_fp32(row_blocks[block_idx].m);
                    // Q4_1: qs[j] lower nibble -> element j, upper nibble -> element j+16
                    int8_t q;
                    if (elem_idx < 16) {
                        q = (row_blocks[block_idx].qs[elem_idx] & 0x0F);
                    } else {
                        q = (row_blocks[block_idx].qs[elem_idx - 16] >> 4);
                    }
                    val = q * d + m;
                }

                // Column-pair interleaved format: tile[(j/2)*64 + i*2 + (j%2)]
                fp32_to_fp16_store(&tile_base[(j / 2) * 64 + i * 2 + (j % 2)], val);
            }
        }
    }
}

static void dequantize_q8_0_to_f16_tiles(__fp16 *restrict vtcm_dst, const block_q8_0 *restrict src,
                                         int n_cols, int k) {
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int n_col_tiles = (n_cols + HMX_FP16_TILE_N_COLS - 1) / HMX_FP16_TILE_N_COLS;
    const int n_tot_tiles = n_col_tiles * k_tiles;
    const int nb_per_col = k / QK8_0;

    // Process all tiles (matching convert_weight_f32_to_fp16_tiles structure)
    for (int t = 0; t < n_tot_tiles; ++t) {
        int ct = t / k_tiles;  // N tile index (column tile)
        int kt = t % k_tiles;  // K tile index (row tile)

        __fp16 *tile_base = vtcm_dst + t * HMX_FP16_TILE_N_ELMS;

        for (int i = 0; i < HMX_FP16_TILE_N_ROWS; ++i) {  // 32 rows per tile (N dimension)
            int row_global = ct * HMX_FP16_TILE_N_ROWS + i;  // global N index
            const block_q8_0 *row_blocks = (row_global < n_cols) ?
                                           src + row_global * nb_per_col : NULL;

            for (int j = 0; j < HMX_FP16_TILE_N_COLS; ++j) {  // 32 columns per tile (K dimension)
                int col_global = kt * HMX_FP16_TILE_N_COLS + j;  // global K index
                int block_idx = col_global / QK8_0;
                int elem_idx = col_global % QK8_0;

                float val = 0.0f;
                if (row_blocks && col_global < k) {
                    float d = ggml_compute_fp16_to_fp32(row_blocks[block_idx].d);
                    val = row_blocks[block_idx].qs[elem_idx] * d;
                }

                // Column-pair interleaved format: tile[(j/2)*64 + i*2 + (j%2)]
                fp32_to_fp16_store(&tile_base[(j / 2) * 64 + i * 2 + (j % 2)], val);
            }
        }
    }
}

static void dequantize_q5_0_to_f16_tiles(__fp16 *restrict vtcm_dst, const block_q5_0 *restrict src,
                                         int n_cols, int k) {
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int n_col_tiles = (n_cols + HMX_FP16_TILE_N_COLS - 1) / HMX_FP16_TILE_N_COLS;
    const int n_tot_tiles = n_col_tiles * k_tiles;
    const int nb_per_col = k / QK5_0;

    // Process all tiles (matching convert_weight_f32_to_fp16_tiles structure)
    for (int t = 0; t < n_tot_tiles; ++t) {
        int ct = t / k_tiles;  // N tile index (column tile)
        int kt = t % k_tiles;  // K tile index (row tile)

        __fp16 *tile_base = vtcm_dst + t * HMX_FP16_TILE_N_ELMS;

        for (int i = 0; i < HMX_FP16_TILE_N_ROWS; ++i) {  // 32 rows per tile (N dimension)
            int row_global = ct * HMX_FP16_TILE_N_ROWS + i;  // global N index
            const block_q5_0 *row_blocks = (row_global < n_cols) ?
                                           src + row_global * nb_per_col : NULL;

            for (int j = 0; j < HMX_FP16_TILE_N_COLS; ++j) {  // 32 columns per tile (K dimension)
                int col_global = kt * HMX_FP16_TILE_N_COLS + j;  // global K index
                int block_idx = col_global / QK5_0;
                int elem_idx = col_global % QK5_0;

                float val = 0.0f;
                if (row_blocks && col_global < k) {
                    float d = ggml_compute_fp16_to_fp32(row_blocks[block_idx].d);
                    uint32_t qh;
                    memcpy(&qh, row_blocks[block_idx].qh, sizeof(qh));
                    // Q5_0: qs[j] lower nibble -> element j, upper nibble -> element j+16
                    // qh bits: bit j -> element j (5th bit), bit (j+16) -> element j+16 (5th bit)
                    int8_t q;
                    if (elem_idx < 16) {
                        q = (row_blocks[block_idx].qs[elem_idx] & 0x0F) | (((qh >> elem_idx) & 1) << 4);
                    } else {
                        q = (row_blocks[block_idx].qs[elem_idx - 16] >> 4) | (((qh >> elem_idx) & 1) << 4);
                    }
                    val = (q - 16) * d;
                }

                // Column-pair interleaved format: tile[(j/2)*64 + i*2 + (j%2)]
                fp32_to_fp16_store(&tile_base[(j / 2) * 64 + i * 2 + (j % 2)], val);
            }
        }
    }
}

static void dequantize_iq4_nl_to_f16_tiles(__fp16 *restrict vtcm_dst, const block_iq4_nl *restrict src,
                                            int n_cols, int k) {
    static const int8_t kvalues_iq4nl[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
    };
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int n_col_tiles = (n_cols + HMX_FP16_TILE_N_COLS - 1) / HMX_FP16_TILE_N_COLS;
    const int n_tot_tiles = n_col_tiles * k_tiles;
    const int nb_per_col = k / QK4_NL;

    // Process all tiles (matching convert_weight_f32_to_fp16_tiles structure)
    for (int t = 0; t < n_tot_tiles; ++t) {
        int ct = t / k_tiles;  // N tile index (column tile)
        int kt = t % k_tiles;  // K tile index (row tile)

        __fp16 *tile_base = vtcm_dst + t * HMX_FP16_TILE_N_ELMS;

        for (int i = 0; i < HMX_FP16_TILE_N_ROWS; ++i) {  // 32 rows per tile (N dimension)
            int row_global = ct * HMX_FP16_TILE_N_ROWS + i;  // global N index
            const block_iq4_nl *row_blocks = (row_global < n_cols) ?
                                              src + row_global * nb_per_col : NULL;

            for (int j = 0; j < HMX_FP16_TILE_N_COLS; ++j) {  // 32 columns per tile (K dimension)
                int col_global = kt * HMX_FP16_TILE_N_COLS + j;  // global K index
                int block_idx = col_global / QK4_NL;
                int elem_idx = col_global % QK4_NL;

                float val = 0.0f;
                if (row_blocks && col_global < k) {
                    float d = ggml_compute_fp16_to_fp32(row_blocks[block_idx].d);
                    // IQ4_NL: qs[j] lower nibble -> element j, upper nibble -> element j+16
                    int8_t q;
                    if (elem_idx < 16) {
                        q = (row_blocks[block_idx].qs[elem_idx] & 0x0F);
                    } else {
                        q = (row_blocks[block_idx].qs[elem_idx - 16] >> 4);
                    }
                    val = kvalues_iq4nl[q] * d;
                }

                // Column-pair interleaved format: tile[(j/2)*64 + i*2 + (j%2)]
                fp32_to_fp16_store(&tile_base[(j / 2) * 64 + i * 2 + (j % 2)], val);
            }
        }
    }
}

// BF16 weight: convert each element BF16 -> FP32 -> FP16, store in column-pair interleaved tile format
static void convert_weight_bf16_to_fp16_tiles(__fp16 *restrict vtcm_dst, const ggml_bf16_t *restrict src,
                                               int n_cols, int k, int row_stride) {
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int n_col_tiles = (n_cols + HMX_FP16_TILE_N_COLS - 1) / HMX_FP16_TILE_N_COLS;
    const int n_tot_tiles = n_col_tiles * k_tiles;

    // Process all tiles (matching convert_weight_f32_to_fp16_tiles structure)
    for (int t = 0; t < n_tot_tiles; ++t) {
        int ct = t / k_tiles;  // N tile index (column tile)
        int kt = t % k_tiles;  // K tile index (row tile)

        __fp16 *tile_base = vtcm_dst + t * HMX_FP16_TILE_N_ELMS;

        for (int i = 0; i < HMX_FP16_TILE_N_ROWS; ++i) {  // 32 rows per tile (N dimension)
            int row_global = ct * HMX_FP16_TILE_N_ROWS + i;  // global N index
            const ggml_bf16_t *row = (row_global < n_cols) ?
                                     src + row_global * row_stride + kt * HMX_FP16_TILE_N_COLS : NULL;

            for (int j = 0; j < HMX_FP16_TILE_N_COLS; ++j) {  // 32 columns per tile (K dimension)
                float val = 0.0f;
                if (row) {
                    val = ggml_compute_bf16_to_fp32(row[j]);
                }
                // Column-pair interleaved format: tile[(j/2)*64 + i*2 + (j%2)]
                fp32_to_fp16_store(&tile_base[(j / 2) * 64 + i * 2 + (j % 2)], val);
            }
        }
    }
}

// ============================================================
// HVX-accelerated dequantize-to-FP16-tiles functions
// ============================================================

// Q8_0: dequantize one block (32 int8) to 32 FP16 values in first 64 bytes of HVX_Vector
static inline HVX_Vector dequantize_q8_0_block_to_fp16_hvx(const block_q8_0 *b) {
    HVX_Vector vq = hvx_vmemu(b->qs);
    HVX_Vector v_scales = hvx_vec_repl_f16(hvx_vmemu(&b->d));
    HVX_Vector v0 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(vq));
    HVX_Vector v_hf = Q6_Vhf_equals_Vh(v0);
    return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v_hf, v_scales));
}

static void dequantize_q8_0_to_f16_tiles_hvx(__fp16 *restrict vtcm_dst, const block_q8_0 *restrict src,
                                              int n_cols, int k) {
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int n_col_tiles = (n_cols + HMX_FP16_TILE_N_COLS - 1) / HMX_FP16_TILE_N_COLS;
    const int nb_per_col = k / QK8_0;

    const HVX_Vector v_scat_base = hvx_vmem(hmx_transpose_scatter_offsets);
    const HVX_Vector v_scat_step = Q6_V_vsplat_R(4);
    const HVX_VectorPred q_mask64 = Q6_Q_vsetq_R(64);
    const size_t single_region = (size_t)(HMX_FP16_TILE_SIZE - 1);

    for (int t = 0; t < n_col_tiles * k_tiles; ++t) {
        int ct = t / k_tiles;
        int kt = t % k_tiles;
        __fp16 *tile_base = vtcm_dst + t * HMX_FP16_TILE_N_ELMS;

        for (int i = 0; i < HMX_FP16_TILE_N_ROWS; i += 2) {
            int row0 = ct * HMX_FP16_TILE_N_ROWS + i;
            int row1 = row0 + 1;
            const block_q8_0 *row0_blocks = (row0 < n_cols) ? src + row0 * nb_per_col + kt : NULL;
            const block_q8_0 *row1_blocks = (row1 < n_cols) ? src + row1 * nb_per_col + kt : NULL;

            HVX_Vector v0 = row0_blocks ? dequantize_q8_0_block_to_fp16_hvx(row0_blocks) : Q6_V_vzero();
            HVX_Vector v1 = row1_blocks ? dequantize_q8_0_block_to_fp16_hvx(row1_blocks) : Q6_V_vzero();

            const HVX_Vector v_off0 = Q6_Vw_vadd_VwVw(v_scat_base, Q6_V_vsplat_R(i * 4));
            const HVX_Vector v_off1 = Q6_Vw_vadd_VwVw(v_off0, v_scat_step);
            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, single_region, v_off0, v0);
            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, single_region, v_off1, v1);
        }
    }
}

// Q4_0: dequantize one block (18 bytes: 2-byte scale + 16 packed nibbles) to 32 FP16
// qs[j] lower nibble -> element j, upper nibble -> element j+16
static inline HVX_Vector dequantize_q4_0_block_to_fp16_hvx(const block_q4_0 *b) {
    HVX_Vector vq = hvx_vmemu(b->qs);
    HVX_Vector v_scales = hvx_vec_repl_f16(hvx_vmemu(&b->d));

    // Extract lower nibbles (elements 0..15) and upper nibbles (elements 16..31)
    const HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector v_lo = Q6_V_vand_VV(vq, mask_h4);
    HVX_Vector v_hi = Q6_Vub_vlsr_VubR(vq, 4);

    // Subtract 8 from each nibble
    const HVX_Vector i8 = Q6_Vb_vsplat_R(8);
    v_lo = Q6_Vb_vsub_VbVb(v_lo, i8);
    v_hi = Q6_Vb_vsub_VbVb(v_hi, i8);

    // Unpack int8 -> int16 (lo half only, 16 int8 -> 16 int16 in first 32 bytes)
    HVX_Vector v_lo16 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(v_lo));
    HVX_Vector v_hi16 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(v_hi));

    // int16 -> fp16 -> multiply by scale
    v_lo16 = Q6_Vhf_equals_Vh(v_lo16);
    v_hi16 = Q6_Vhf_equals_Vh(v_hi16);
    v_lo16 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v_lo16, v_scales));
    v_hi16 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v_hi16, v_scales));

    // Combine: lo in first 32 bytes, hi in second 32 bytes
    // Mask both with p32 first (zero out garbage in bytes 32-127),
    // then rotate hi's first 32 bytes to bytes 32-63, then OR
    const HVX_VectorPred p32 = Q6_Q_vsetq_R(32);
    HVX_Vector v_lo_masked = Q6_V_vand_QV(p32, v_lo16);
    HVX_Vector v_hi_rotated = Q6_V_vror_VR(Q6_V_vand_QV(p32, v_hi16), 96);
    HVX_Vector result = Q6_V_vor_VV(v_lo_masked, v_hi_rotated);
    return result;
}

static void dequantize_q4_0_to_f16_tiles_hvx(__fp16 *restrict vtcm_dst, const block_q4_0 *restrict src,
                                              int n_cols, int k) {
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int n_col_tiles = (n_cols + HMX_FP16_TILE_N_COLS - 1) / HMX_FP16_TILE_N_COLS;
    const int nb_per_col = k / QK4_0;

    const HVX_Vector v_scat_base = hvx_vmem(hmx_transpose_scatter_offsets);
    const HVX_Vector v_scat_step = Q6_V_vsplat_R(4);
    const HVX_VectorPred q_mask64 = Q6_Q_vsetq_R(64);
    const size_t single_region = (size_t)(HMX_FP16_TILE_SIZE - 1);

    for (int t = 0; t < n_col_tiles * k_tiles; ++t) {
        int ct = t / k_tiles;
        int kt = t % k_tiles;
        __fp16 *tile_base = vtcm_dst + t * HMX_FP16_TILE_N_ELMS;

        for (int i = 0; i < HMX_FP16_TILE_N_ROWS; i += 2) {
            int row0 = ct * HMX_FP16_TILE_N_ROWS + i;
            int row1 = row0 + 1;
            const block_q4_0 *row0_blocks = (row0 < n_cols) ? src + row0 * nb_per_col + kt : NULL;
            const block_q4_0 *row1_blocks = (row1 < n_cols) ? src + row1 * nb_per_col + kt : NULL;

            HVX_Vector v0 = row0_blocks ? dequantize_q4_0_block_to_fp16_hvx(row0_blocks) : Q6_V_vzero();
            HVX_Vector v1 = row1_blocks ? dequantize_q4_0_block_to_fp16_hvx(row1_blocks) : Q6_V_vzero();

            const HVX_Vector v_off0 = Q6_Vw_vadd_VwVw(v_scat_base, Q6_V_vsplat_R(i * 4));
            const HVX_Vector v_off1 = Q6_Vw_vadd_VwVw(v_off0, v_scat_step);
            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, single_region, v_off0, v0);
            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, single_region, v_off1, v1);
        }
    }
}

// x4x2 format: dequantize one 32-element sub-block to 32 FP16
// x4x2 layout: [quants: K/2 bytes][scales: nb*16 bytes] per row
// Each 256-element logical block has 8 sub-blocks of 32 elements
// Sub-blocks 0-3 use low nibble, sub-blocks 4-7 use high nibble
static inline HVX_Vector dequantize_x4x2_q4_0_group_hvx(const uint8_t *packed_32, bool upper_nibbles, const __fp16 *scale) {
    HVX_Vector vq = hvx_vmemu(packed_32);
    const HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    const HVX_Vector i8 = Q6_Vb_vsplat_R(8);
    HVX_Vector v_scales = hvx_vec_repl_f16(hvx_vmemu(scale));

    HVX_Vector v_quants = Q6_Vub_vlsr_VubR(vq, 4 * upper_nibbles);
    v_quants = Q6_V_vand_VV(v_quants, mask_h4);

    HVX_Vector v_int8 = Q6_Vb_vsub_VbVb(v_quants, i8);
    HVX_Vector v0 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(v_int8));
    HVX_Vector v_hf = Q6_Vhf_equals_Vh(v0);

    return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v_hf, v_scales));
}

// x4x2 format: dequantize Q4_0x4x2 weight chunk to FP16 tiles
// src points to x4x2 data (quants first, then scales per row)
// row_stride is the same as Q4_0 row stride (same total size)
static void dequantize_x4x2_q4_0_to_f16_tiles_hvx(__fp16 *restrict vtcm_dst, const uint8_t *restrict src,
                                                     int n_cols, int k, size_t row_stride) {
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int n_col_tiles = (n_cols + HMX_FP16_TILE_N_COLS - 1) / HMX_FP16_TILE_N_COLS;
    const int qrow_size = k / 2;       // quants region size per row
    const int dblk_size = 8 * 2;       // scales per 256-element block: 8 * fp16
    const int scale_step = (int)sizeof(__fp16);  // 2 bytes per scale

    const HVX_Vector v_scat_base = hvx_vmem(hmx_transpose_scatter_offsets);
    const HVX_Vector v_scat_step = Q6_V_vsplat_R(4);
    const HVX_VectorPred q_mask64 = Q6_Q_vsetq_R(64);
    const size_t single_region = (size_t)(HMX_FP16_TILE_SIZE - 1);

    for (int t = 0; t < n_col_tiles * k_tiles; ++t) {
        int ct = t / k_tiles;
        int kt = t % k_tiles;
        __fp16 *tile_base = vtcm_dst + t * HMX_FP16_TILE_N_ELMS;

        // x4x2 addressing: determine which 256-element block and sub-block
        unsigned blk_idx   = (kt * 32) / 256;
        unsigned sub_blk   = (kt * 32) % 256 / 32;
        bool upper         = (sub_blk >= 4);
        unsigned packed_off = blk_idx * 128 + (upper ? (sub_blk - 4) : sub_blk) * 32;
        unsigned scale_off  = qrow_size + blk_idx * dblk_size + sub_blk * scale_step;

        HVX_Vector v_off = v_scat_base;
        for (int i = 0; i < HMX_FP16_TILE_N_ROWS; i += 2) {
            int row0 = ct * HMX_FP16_TILE_N_ROWS + i;
            int row1 = row0 + 1;

            HVX_Vector v0 = Q6_V_vzero();
            HVX_Vector v1 = Q6_V_vzero();

            if (row0 < n_cols) {
                const uint8_t *r0 = src + row0 * row_stride;
                v0 = dequantize_x4x2_q4_0_group_hvx(r0 + packed_off, upper, (const __fp16 *)(r0 + scale_off));
            }
            if (row1 < n_cols) {
                const uint8_t *r1 = src + row1 * row_stride;
                v1 = dequantize_x4x2_q4_0_group_hvx(r1 + packed_off, upper, (const __fp16 *)(r1 + scale_off));
            }

            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, single_region, v_off, v0);
            v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, single_region, v_off, v1);
            v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
        }
    }
}

// Q4_1: dequantize one block to 32 FP16 (has both scale d and offset m)
static inline HVX_Vector dequantize_q4_1_block_to_fp16_hvx(const block_q4_1 *b) {
    HVX_Vector vq = hvx_vmemu(b->qs);

    // Load d and m: they are adjacent ggml_fp16_t values
    HVX_Vector v_dm = hvx_vmemu(&b->d);
    HVX_Vector v_scales = hvx_vec_repl_f16(v_dm);                     // replicate d
    HVX_Vector v_offsets = hvx_vec_repl_f16(Q6_V_vror_VR(v_dm, 2));  // replicate m

    // Extract nibbles same as Q4_0
    const HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector v_lo = Q6_V_vand_VV(vq, mask_h4);
    HVX_Vector v_hi = Q6_Vub_vlsr_VubR(vq, 4);

    // Unpack int8 -> int16 (no subtraction for Q4_1)
    HVX_Vector v_lo16 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(v_lo));
    HVX_Vector v_hi16 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(v_hi));

    // int16 -> fp16 -> q*d + m
    v_lo16 = Q6_Vhf_equals_Vh(v_lo16);
    v_hi16 = Q6_Vhf_equals_Vh(v_hi16);
    v_lo16 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(v_lo16, v_scales), v_offsets));
    v_hi16 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(v_hi16, v_scales), v_offsets));

    // Combine lo and hi
    const HVX_VectorPred p32 = Q6_Q_vsetq_R(32);
    HVX_Vector v_lo_masked = Q6_V_vand_QV(p32, v_lo16);
    HVX_Vector v_hi_rotated = Q6_V_vror_VR(Q6_V_vand_QV(p32, v_hi16), 96);
    HVX_Vector result = Q6_V_vor_VV(v_lo_masked, v_hi_rotated);
    return result;
}

static void dequantize_q4_1_to_f16_tiles_hvx(__fp16 *restrict vtcm_dst, const block_q4_1 *restrict src,
                                              int n_cols, int k) {
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int n_col_tiles = (n_cols + HMX_FP16_TILE_N_COLS - 1) / HMX_FP16_TILE_N_COLS;
    const int nb_per_col = k / QK4_1;

    const HVX_Vector v_scat_base = hvx_vmem(hmx_transpose_scatter_offsets);
    const HVX_Vector v_scat_step = Q6_V_vsplat_R(4);
    const HVX_VectorPred q_mask64 = Q6_Q_vsetq_R(64);
    const size_t single_region = (size_t)(HMX_FP16_TILE_SIZE - 1);

    for (int t = 0; t < n_col_tiles * k_tiles; ++t) {
        int ct = t / k_tiles;
        int kt = t % k_tiles;
        __fp16 *tile_base = vtcm_dst + t * HMX_FP16_TILE_N_ELMS;

        for (int i = 0; i < HMX_FP16_TILE_N_ROWS; i += 2) {
            int row0 = ct * HMX_FP16_TILE_N_ROWS + i;
            int row1 = row0 + 1;
            const block_q4_1 *row0_blocks = (row0 < n_cols) ? src + row0 * nb_per_col + kt : NULL;
            const block_q4_1 *row1_blocks = (row1 < n_cols) ? src + row1 * nb_per_col + kt : NULL;

            HVX_Vector v0 = row0_blocks ? dequantize_q4_1_block_to_fp16_hvx(row0_blocks) : Q6_V_vzero();
            HVX_Vector v1 = row1_blocks ? dequantize_q4_1_block_to_fp16_hvx(row1_blocks) : Q6_V_vzero();

            const HVX_Vector v_off0 = Q6_Vw_vadd_VwVw(v_scat_base, Q6_V_vsplat_R(i * 4));
            const HVX_Vector v_off1 = Q6_Vw_vadd_VwVw(v_off0, v_scat_step);
            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, single_region, v_off0, v0);
            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, single_region, v_off1, v1);
        }
    }
}

// Q5_0: fully HVX approach for nibble extraction + qh bit correction
// qh bits are expanded to fp16 mask using bit manipulation on HVX
static inline HVX_Vector dequantize_q5_0_block_to_fp16_hvx(const block_q5_0 *b) {
    HVX_Vector vq = hvx_vmemu(b->qs);
    HVX_Vector v_scales = hvx_vec_repl_f16(hvx_vmemu(&b->d));

    // Extract nibbles same as Q4_0, subtract 16 instead of 8
    const HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector v_lo = Q6_V_vand_VV(vq, mask_h4);
    HVX_Vector v_hi = Q6_Vub_vlsr_VubR(vq, 4);

    const HVX_Vector i16 = Q6_Vb_vsplat_R(16);
    v_lo = Q6_Vb_vsub_VbVb(v_lo, i16);
    v_hi = Q6_Vb_vsub_VbVb(v_hi, i16);

    HVX_Vector v_lo16 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(v_lo));
    HVX_Vector v_hi16 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(v_hi));

    v_lo16 = Q6_Vhf_equals_Vh(v_lo16);
    v_hi16 = Q6_Vhf_equals_Vh(v_hi16);
    v_lo16 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v_lo16, v_scales));
    v_hi16 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v_hi16, v_scales));

    // Combine lo and hi into 32 fp16 values in first 64 bytes
    const HVX_VectorPred p32 = Q6_Q_vsetq_R(32);
    HVX_Vector v_lo_masked = Q6_V_vand_QV(p32, v_lo16);
    HVX_Vector v_hi_rotated = Q6_V_vror_VR(Q6_V_vand_QV(p32, v_hi16), 96);
    HVX_Vector result = Q6_V_vor_VV(v_lo_masked, v_hi_rotated);

    // HVX qh bit correction: expand qh bits to fp16 mask, multiply by d*16, add to result
    // qh is 4 bytes (32 bits), each bit corresponds to one element's 5th bit
    // Strategy: load qh as 4 bytes, expand each bit to a byte mask (0x00 or 0xFF),
    //           convert to fp16 (0.0 or NaN->1.0), multiply by d*16, add to result
    uint32_t qh;
    memcpy(&qh, b->qh, sizeof(qh));

    // Build qh mask vector using scalar bit extraction into a temp buffer
    // This is still scalar but avoids the store-modify-load round-trip on the result
    __fp16 qh_add[32] __attribute__((aligned(128)));
    float d = ggml_compute_fp16_to_fp32(b->d);
    float d16 = d * 16.0f;
    for (int j = 0; j < 32; ++j) {
        qh_add[j] = ((qh >> j) & 1) ? d16 : 0.0f;
    }

    // Add qh correction using HVX
    // Convert result (Vhf) to Vqf16 via multiply-by-1.0, then add qh correction
    HVX_Vector v_qh_add = hvx_vmem(qh_add);
    HVX_Vector v_one = Q6_Vh_vsplat_R(0x3C00);  // fp16 1.0
    result = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(result, v_one), v_qh_add));

    return result;
}

static void dequantize_q5_0_to_f16_tiles_hvx(__fp16 *restrict vtcm_dst, const block_q5_0 *restrict src,
                                              int n_cols, int k) {
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int n_col_tiles = (n_cols + HMX_FP16_TILE_N_COLS - 1) / HMX_FP16_TILE_N_COLS;
    const int nb_per_col = k / QK5_0;

    const HVX_Vector v_scat_base = hvx_vmem(hmx_transpose_scatter_offsets);
    const HVX_Vector v_scat_step = Q6_V_vsplat_R(4);
    const HVX_VectorPred q_mask64 = Q6_Q_vsetq_R(64);
    const size_t single_region = (size_t)(HMX_FP16_TILE_SIZE - 1);

    for (int t = 0; t < n_col_tiles * k_tiles; ++t) {
        int ct = t / k_tiles;
        int kt = t % k_tiles;
        __fp16 *tile_base = vtcm_dst + t * HMX_FP16_TILE_N_ELMS;

        for (int i = 0; i < HMX_FP16_TILE_N_ROWS; i += 2) {
            int row0 = ct * HMX_FP16_TILE_N_ROWS + i;
            int row1 = row0 + 1;
            const block_q5_0 *row0_blocks = (row0 < n_cols) ? src + row0 * nb_per_col + kt : NULL;
            const block_q5_0 *row1_blocks = (row1 < n_cols) ? src + row1 * nb_per_col + kt : NULL;

            HVX_Vector v0 = row0_blocks ? dequantize_q5_0_block_to_fp16_hvx(row0_blocks) : Q6_V_vzero();
            HVX_Vector v1 = row1_blocks ? dequantize_q5_0_block_to_fp16_hvx(row1_blocks) : Q6_V_vzero();

            const HVX_Vector v_off0 = Q6_Vw_vadd_VwVw(v_scat_base, Q6_V_vsplat_R(i * 4));
            const HVX_Vector v_off1 = Q6_Vw_vadd_VwVw(v_off0, v_scat_step);
            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, single_region, v_off0, v0);
            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, single_region, v_off1, v1);
        }
    }
}

// IQ4_NL: dequantize one block using vlut32 LUT
static inline HVX_Vector dequantize_iq4_nl_block_to_fp16_hvx(const block_iq4_nl *b, const HVX_Vector vlut) {
    HVX_Vector vq = hvx_vmemu(b->qs);
    HVX_Vector v_scales = hvx_vec_repl_f16(hvx_vmemu(&b->d));

    // Extract lower and upper nibbles
    const HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector v_lo = Q6_V_vand_VV(vq, mask_h4);
    HVX_Vector v_hi = Q6_Vub_vlsr_VubR(vq, 4);

    // vlut32 byte lookup: each nibble index -> fp16 value (2 bytes)
    // vshuff interleaves bytes for vlut32
    v_lo = Q6_Vb_vshuff_Vb(v_lo);
    v_hi = Q6_Vb_vshuff_Vb(v_hi);

    HVX_VectorPair vp_lo = Q6_Wh_vlut16_VbVhR(v_lo, vlut, 0);
    HVX_VectorPair vp_hi = Q6_Wh_vlut16_VbVhR(v_hi, vlut, 0);
    HVX_Vector v_lo16 = Q6_V_lo_W(vp_lo);
    HVX_Vector v_hi16 = Q6_V_lo_W(vp_hi);

    // Multiply by scale
    v_lo16 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v_lo16, v_scales));
    v_hi16 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v_hi16, v_scales));

    // Combine lo and hi
    const HVX_VectorPred p32 = Q6_Q_vsetq_R(32);
    HVX_Vector v_lo_masked = Q6_V_vand_QV(p32, v_lo16);
    HVX_Vector v_hi_rotated = Q6_V_vror_VR(Q6_V_vand_QV(p32, v_hi16), 96);
    HVX_Vector result = Q6_V_vor_VV(v_lo_masked, v_hi_rotated);
    return result;
}

static void dequantize_iq4_nl_to_f16_tiles_hvx(__fp16 *restrict vtcm_dst, const block_iq4_nl *restrict src,
                                                int n_cols, int k) {
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int n_col_tiles = (n_cols + HMX_FP16_TILE_N_COLS - 1) / HMX_FP16_TILE_N_COLS;
    const int nb_per_col = k / QK4_NL;

    const HVX_Vector v_scat_base = hvx_vmem(hmx_transpose_scatter_offsets);
    const HVX_Vector v_scat_step = Q6_V_vsplat_R(4);
    const HVX_VectorPred q_mask64 = Q6_Q_vsetq_R(64);
    const size_t single_region = (size_t)(HMX_FP16_TILE_SIZE - 1);
    const HVX_Vector vlut = hvx_vmem(iq4_nl_to_fp16_lut);

    for (int t = 0; t < n_col_tiles * k_tiles; ++t) {
        int ct = t / k_tiles;
        int kt = t % k_tiles;
        __fp16 *tile_base = vtcm_dst + t * HMX_FP16_TILE_N_ELMS;

        for (int i = 0; i < HMX_FP16_TILE_N_ROWS; i += 2) {
            int row0 = ct * HMX_FP16_TILE_N_ROWS + i;
            int row1 = row0 + 1;
            const block_iq4_nl *row0_blocks = (row0 < n_cols) ? src + row0 * nb_per_col + kt : NULL;
            const block_iq4_nl *row1_blocks = (row1 < n_cols) ? src + row1 * nb_per_col + kt : NULL;

            HVX_Vector v0 = row0_blocks ? dequantize_iq4_nl_block_to_fp16_hvx(row0_blocks, vlut) : Q6_V_vzero();
            HVX_Vector v1 = row1_blocks ? dequantize_iq4_nl_block_to_fp16_hvx(row1_blocks, vlut) : Q6_V_vzero();

            const HVX_Vector v_off0 = Q6_Vw_vadd_VwVw(v_scat_base, Q6_V_vsplat_R(i * 4));
            const HVX_Vector v_off1 = Q6_Vw_vadd_VwVw(v_off0, v_scat_step);
            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, single_region, v_off0, v0);
            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, single_region, v_off1, v1);
        }
    }
}

// MXFP4 dequantization LUT: maps 4-bit index to fp16 kvalue
// kvalues_mxfp4 = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}
static const __fp16 mxfp4_to_fp16_lut[64] __attribute__((aligned(VLEN))) = {
     0, 0,  1, 0,  2, 0,  3, 0,  4, 0,  6, 0,  8, 0, 12, 0,
     0, 0, -1, 0, -2, 0, -3, 0, -4, 0, -6, 0, -8, 0,-12, 0,
};

static inline float mxfp4_e8m0_to_fp32_half(uint8_t x) {
    uint32_t bits;
    if (x < 2) {
        bits = 0x00200000 << x;
    } else {
        bits = (uint32_t)(x - 1) << 23;
    }
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

static void dequantize_mxfp4_to_f16_tiles(__fp16 *restrict vtcm_dst, const block_mxfp4 *restrict src,
                                           int n_cols, int k) {
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int n_col_tiles = (n_cols + HMX_FP16_TILE_N_COLS - 1) / HMX_FP16_TILE_N_COLS;
    const int n_tot_tiles = n_col_tiles * k_tiles;
    const int nb_per_col = k / QK_MXFP4;

    for (int t = 0; t < n_tot_tiles; ++t) {
        int ct = t / k_tiles;
        int kt = t % k_tiles;

        __fp16 *tile_base = vtcm_dst + t * HMX_FP16_TILE_N_ELMS;

        for (int i = 0; i < HMX_FP16_TILE_N_ROWS; ++i) {
            int row_global = ct * HMX_FP16_TILE_N_ROWS + i;
            const block_mxfp4 *row_blocks = (row_global < n_cols) ?
                                             src + row_global * nb_per_col : NULL;

            for (int j = 0; j < HMX_FP16_TILE_N_COLS; ++j) {
                int col_global = kt * HMX_FP16_TILE_N_COLS + j;
                int block_idx = col_global / QK_MXFP4;
                int elem_idx = col_global % QK_MXFP4;

                float val = 0.0f;
                if (row_blocks && col_global < k) {
                    float d = mxfp4_e8m0_to_fp32_half(row_blocks[block_idx].e);
                    int8_t q;
                    if (elem_idx < 16) {
                        q = (row_blocks[block_idx].qs[elem_idx] & 0x0F);
                    } else {
                        q = (row_blocks[block_idx].qs[elem_idx - 16] >> 4);
                    }
                    val = kvalues_mxfp4[q] * d;
                }

                fp32_to_fp16_store(&tile_base[(j / 2) * 64 + i * 2 + (j % 2)], val);
            }
        }
    }
}

// MXFP4: dequantize one block using vlut16 LUT
static inline HVX_Vector dequantize_mxfp4_block_to_fp16_hvx(const block_mxfp4 *b, const HVX_Vector vlut) {
    HVX_Vector vq = Q6_V_vand_QV(Q6_Q_vsetq_R(16), hvx_vmemu(b->qs));

    // Build FP16 scale from e8m0: 0.5 * 2^(e-127) = 2^(e-128)
    // FP16 exponent = e - 128 + 15 = e - 113
    uint16_t fp16_exp = (uint16_t)b->e;
    int16_t fp16_biased = (int16_t)fp16_exp - 113;
    if (fp16_biased < 0) fp16_biased = 0;
    if (fp16_biased > 30) fp16_biased = 30;
    uint16_t fp16_bits = (uint16_t)fp16_biased << 10;
    __fp16 fp16_scale;
    memcpy(&fp16_scale, &fp16_bits, sizeof(__fp16));
    HVX_Vector v_scales = hvx_vec_repl_f16(hvx_vmemu(&fp16_scale));

    // Extract lower and upper nibbles
    const HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector v_lo = Q6_V_vand_VV(vq, mask_h4);
    HVX_Vector v_hi = Q6_Vub_vlsr_VubR(vq, 4);

    // vlut16 byte lookup: each nibble index -> fp16 value (2 bytes)
    v_lo = Q6_Vb_vshuff_Vb(v_lo);
    v_hi = Q6_Vb_vshuff_Vb(v_hi);

    HVX_VectorPair vp_lo = Q6_Wh_vlut16_VbVhR(v_lo, vlut, 0);
    HVX_VectorPair vp_hi = Q6_Wh_vlut16_VbVhR(v_hi, vlut, 0);
    HVX_Vector v_lo16 = Q6_V_lo_W(vp_lo);
    HVX_Vector v_hi16 = Q6_V_lo_W(vp_hi);

    // Multiply by scale
    v_lo16 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v_lo16, v_scales));
    v_hi16 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v_hi16, v_scales));

    // Combine lo and hi
    const HVX_VectorPred p32 = Q6_Q_vsetq_R(32);
    HVX_Vector v_lo_masked = Q6_V_vand_QV(p32, v_lo16);
    HVX_Vector v_hi_rotated = Q6_V_vror_VR(Q6_V_vand_QV(p32, v_hi16), 96);
    HVX_Vector result = Q6_V_vor_VV(v_lo_masked, v_hi_rotated);
    return result;
}

static void dequantize_mxfp4_to_f16_tiles_hvx(__fp16 *restrict vtcm_dst, const block_mxfp4 *restrict src,
                                               int n_cols, int k) {
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int n_col_tiles = (n_cols + HMX_FP16_TILE_N_COLS - 1) / HMX_FP16_TILE_N_COLS;
    const int nb_per_col = k / QK_MXFP4;

    const HVX_Vector v_scat_base = hvx_vmem(hmx_transpose_scatter_offsets);
    const HVX_Vector v_scat_step = Q6_V_vsplat_R(4);
    const HVX_VectorPred q_mask64 = Q6_Q_vsetq_R(64);
    const size_t single_region = (size_t)(HMX_FP16_TILE_SIZE - 1);
    const HVX_Vector vlut = hvx_vmem(mxfp4_to_fp16_lut);

    for (int t = 0; t < n_col_tiles * k_tiles; ++t) {
        int ct = t / k_tiles;
        int kt = t % k_tiles;
        __fp16 *tile_base = vtcm_dst + t * HMX_FP16_TILE_N_ELMS;

        for (int i = 0; i < HMX_FP16_TILE_N_ROWS; i += 2) {
            int row0 = ct * HMX_FP16_TILE_N_ROWS + i;
            int row1 = row0 + 1;
            const block_mxfp4 *row0_blocks = (row0 < n_cols) ? src + row0 * nb_per_col + kt : NULL;
            const block_mxfp4 *row1_blocks = (row1 < n_cols) ? src + row1 * nb_per_col + kt : NULL;

            HVX_Vector v0 = row0_blocks ? dequantize_mxfp4_block_to_fp16_hvx(row0_blocks, vlut) : Q6_V_vzero();
            HVX_Vector v1 = row1_blocks ? dequantize_mxfp4_block_to_fp16_hvx(row1_blocks, vlut) : Q6_V_vzero();

            const HVX_Vector v_off0 = Q6_Vw_vadd_VwVw(v_scat_base, Q6_V_vsplat_R(i * 4));
            const HVX_Vector v_off1 = Q6_Vw_vadd_VwVw(v_off0, v_scat_step);
            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, single_region, v_off0, v0);
            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, single_region, v_off1, v1);
        }
    }
}

// BF16: convert each element BF16 -> FP32 -> FP16, store in column-pair interleaved tile format
static void convert_weight_bf16_to_fp16_tiles_hvx(__fp16 *restrict vtcm_dst, const ggml_bf16_t *restrict src,
                                                   int n_cols, int k, int row_stride) {
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int n_col_tiles = (n_cols + HMX_FP16_TILE_N_COLS - 1) / HMX_FP16_TILE_N_COLS;

    const HVX_Vector v_scat_base = hvx_vmem(hmx_transpose_scatter_offsets);
    const HVX_Vector v_scat_step = Q6_V_vsplat_R(4);
    const HVX_VectorPred q_mask64 = Q6_Q_vsetq_R(64);
    const size_t single_region = (size_t)(HMX_FP16_TILE_SIZE - 1);

    for (int t = 0; t < n_col_tiles * k_tiles; ++t) {
        int ct = t / k_tiles;
        int kt = t % k_tiles;
        __fp16 *tile_base = vtcm_dst + t * HMX_FP16_TILE_N_ELMS;

        for (int i = 0; i < HMX_FP16_TILE_N_ROWS; i += 2) {
            int row0 = ct * HMX_FP16_TILE_N_ROWS + i;
            int row1 = row0 + 1;
            const ggml_bf16_t *r0 = (row0 < n_cols) ? src + row0 * row_stride + kt * HMX_FP16_TILE_N_COLS : NULL;
            const ggml_bf16_t *r1 = (row1 < n_cols) ? src + row1 * row_stride + kt * HMX_FP16_TILE_N_COLS : NULL;

            HVX_Vector v0, v1;
            if (r0) {
                // BF16 -> FP32 using vshuff+vasl pattern (same as ggml_bf16_to_fp32_row_hvx)
                HVX_Vector vbf0 = hvx_vmemu(r0);
                HVX_Vector v_shuf0 = Q6_Vh_vshuff_Vh(vbf0);
                HVX_Vector vf0_lo = Q6_Vw_vasl_VwR(v_shuf0, 16);
                HVX_Vector vf0_hi = Q6_Vw_vasl_VwR(Q6_Vw_vasr_VwR(v_shuf0, 16), 16);
                v0 = hvx_vec_f32_to_f16(vf0_lo, vf0_hi);
            } else {
                v0 = Q6_V_vzero();
            }

            if (r1) {
                HVX_Vector vbf1 = hvx_vmemu(r1);
                HVX_Vector v_shuf1 = Q6_Vh_vshuff_Vh(vbf1);
                HVX_Vector vf1_lo = Q6_Vw_vasl_VwR(v_shuf1, 16);
                HVX_Vector vf1_hi = Q6_Vw_vasl_VwR(Q6_Vw_vasr_VwR(v_shuf1, 16), 16);
                v1 = hvx_vec_f32_to_f16(vf1_lo, vf1_hi);
            } else {
                v1 = Q6_V_vzero();
            }

            const HVX_Vector v_off0 = Q6_Vw_vadd_VwVw(v_scat_base, Q6_V_vsplat_R(i * 4));
            const HVX_Vector v_off1 = Q6_Vw_vadd_VwVw(v_off0, v_scat_step);
            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, single_region, v_off0, v0);
            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, single_region, v_off1, v1);
        }
    }
}

// Activation BF16: convert to FP16 tiles using HVX (row-pair interleaved format)
static void convert_activation_bf16_to_fp16_tiles_hvx(__fp16 *restrict vtcm_dst, const ggml_bf16_t *restrict src,
                                                       int n_rows, int k, int row_stride) {
    const int n_row_tiles = (n_rows + HMX_FP16_TILE_N_ROWS - 1) / HMX_FP16_TILE_N_ROWS;
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;

    const HVX_Vector v_scat_base = hvx_vmem(hmx_transpose_scatter_offsets);
    const HVX_Vector v_scat_step = Q6_V_vsplat_R(4);
    const HVX_VectorPred q_mask64 = Q6_Q_vsetq_R(64);
    const size_t single_region = (size_t)(HMX_FP16_TILE_SIZE - 1);

    for (int t = 0; t < n_row_tiles * k_tiles; ++t) {
        int rt = t / k_tiles;
        int kt = t % k_tiles;
        __fp16 *tile_base = vtcm_dst + t * HMX_FP16_TILE_N_ELMS;

        for (int i = 0; i < HMX_FP16_TILE_N_ROWS; i += 2) {
            int row0 = rt * HMX_FP16_TILE_N_ROWS + i;
            int row1 = row0 + 1;
            const ggml_bf16_t *r0 = (row0 < n_rows) ? src + row0 * row_stride + kt * HMX_FP16_TILE_N_COLS : NULL;
            const ggml_bf16_t *r1 = (row1 < n_rows) ? src + row1 * row_stride + kt * HMX_FP16_TILE_N_COLS : NULL;

            HVX_Vector v0, v1;
            if (r0) {
                HVX_Vector vbf0 = hvx_vmemu(r0);
                HVX_Vector v_shuf0 = Q6_Vh_vshuff_Vh(vbf0);
                HVX_Vector vf0_lo = Q6_Vw_vasl_VwR(v_shuf0, 16);
                HVX_Vector vf0_hi = Q6_Vw_vasl_VwR(Q6_Vw_vasr_VwR(v_shuf0, 16), 16);
                v0 = hvx_vec_f32_to_f16(vf0_lo, vf0_hi);
            } else {
                v0 = Q6_V_vzero();
            }

            if (r1) {
                HVX_Vector vbf1 = hvx_vmemu(r1);
                HVX_Vector v_shuf1 = Q6_Vh_vshuff_Vh(vbf1);
                HVX_Vector vf1_lo = Q6_Vw_vasl_VwR(v_shuf1, 16);
                HVX_Vector vf1_hi = Q6_Vw_vasl_VwR(Q6_Vw_vasr_VwR(v_shuf1, 16), 16);
                v1 = hvx_vec_f32_to_f16(vf1_lo, vf1_hi);
            } else {
                v1 = Q6_V_vzero();
            }

            const HVX_Vector v_off0 = Q6_Vw_vadd_VwVw(v_scat_base, Q6_V_vsplat_R(i * 4));
            const HVX_Vector v_off1 = Q6_Vw_vadd_VwVw(v_off0, v_scat_step);
            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, single_region, v_off0, v0);
            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, single_region, v_off1, v1);
        }
    }
}

// ============================================================
// Parallel data conversion helpers for VTCM+HMX
// ============================================================

// Range-aware output writeback: only processes rows [start_row, end_row)
static void transfer_output_chunk_fp16_to_fp32_range(float *restrict dst, const __fp16 *restrict src,
                                                      int n_rows, int n_cols, int col_stride,
                                                      int start_row, int end_row) {
    const int n_col_tiles = (n_cols + HMX_FP16_TILE_N_COLS - 1) / HMX_FP16_TILE_N_COLS;

    // Round start_row down to even for row-pair alignment
    int sr = start_row & ~1;
    for (int r = sr; r < end_row; r += 2) {
        if (r < start_row) continue;
        int r0 = r / HMX_FP16_TILE_N_ROWS;
        int intra_tile_row = r % HMX_FP16_TILE_N_ROWS;
        int row_pair = intra_tile_row / 2;

        for (int c = 0; c < n_cols; c += HMX_FP16_TILE_N_COLS) {
            int c0 = c / HMX_FP16_TILE_N_COLS;
            int tile_idx = r0 * n_col_tiles + c0;
            const __fp16 *tile = src + tile_idx * HMX_FP16_TILE_N_ELMS;

            if (r >= start_row) {
                int j_max = (c + HMX_FP16_TILE_N_COLS <= n_cols) ? HMX_FP16_TILE_N_COLS : (n_cols - c);
                for (int j = 0; j < j_max; ++j) {
                    dst[(c + j) + r * col_stride] = (float)tile[row_pair * 64 + j * 2];
                }
            }
            if (r + 1 < end_row && r + 1 >= start_row && r + 1 < n_rows) {
                int j_max = (c + HMX_FP16_TILE_N_COLS <= n_cols) ? HMX_FP16_TILE_N_COLS : (n_cols - c);
                for (int j = 0; j < j_max; ++j) {
                    dst[(c + j) + (r + 1) * col_stride] = (float)tile[row_pair * 64 + j * 2 + 1];
                }
            }
        }
    }
}

// Range-aware activation fp32->fp16: only processes rows [start_row, end_row)
// start_row and end_row must be even and tile-aligned (multiples of 32)
// IMPORTANT: padding rows (n_rows..n_rows_padded) must be zero-initialized in src before calling this
static void transfer_activation_chunk_fp32_to_fp16_range(__fp16 *restrict vtcm_dst, const float *restrict src,
                                                          int n_rows, int n_cols, int row_stride,
                                                          int start_row, int end_row) {
    const int n_rows_padded = ((n_rows + HMX_FP16_TILE_N_ROWS - 1) / HMX_FP16_TILE_N_ROWS) * HMX_FP16_TILE_N_ROWS;
    const int n_tiles_per_row = n_cols / HMX_FP16_TILE_N_COLS;

    int r = start_row;

    // HVX path for all rows in range (padding rows have been zero-initialized)
    int hvx_end = (end_row < n_rows_padded) ? end_row : n_rows_padded;
    #pragma unroll(2)
    for (; r < hvx_end; r += 2) {
        int r0 = r / HMX_FP16_TILE_N_ROWS;
        int r1 = r % HMX_FP16_TILE_N_ROWS;

        const HVX_Vector *pv_in0 = (const HVX_Vector *) (src + (r + 0) * row_stride);
        const HVX_Vector *pv_in1 = (const HVX_Vector *) (src + (r + 1) * row_stride);
        for (int c = 0; c < n_cols; c += 32) {
            HVX_Vector v0 = *pv_in0++;
            HVX_Vector v1 = *pv_in1++;

            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            int c0       = c / HMX_FP16_TILE_N_COLS;
            int tile_idx = r0 * n_tiles_per_row + c0;

            __fp16 *tile_base = vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS;
            HVX_Vector *tile_hvx = (HVX_Vector *)tile_base;
            tile_hvx[r1 / 2] = v_out;
        }
    }
}

// Range-aware activation f16->f16 tiles: only processes rows [start_row, end_row)
static void transfer_activation_chunk_f16_to_f16_tiles_range(__fp16 *restrict vtcm_dst, const __fp16 *restrict src,
                                                              int n_rows, int k, int row_stride,
                                                              int start_row, int end_row) {
    const int n_rows_padded = ((n_rows + HMX_FP16_TILE_N_ROWS - 1) / HMX_FP16_TILE_N_ROWS) * HMX_FP16_TILE_N_ROWS;
    const int n_tiles_per_row = k / HMX_FP16_TILE_N_COLS;

    int sr = start_row & ~1;  // round down to even
    int er = (end_row + 1) & ~1;  // round up to even
    if (er > n_rows_padded) er = n_rows_padded;

    for (int r = sr; r < er; r += 2) {
        if (r < start_row) continue;
        int r0 = r / HMX_FP16_TILE_N_ROWS;
        int r1 = r % HMX_FP16_TILE_N_ROWS;

        const __fp16 *src_row0 = (r < n_rows) ? src + (r + 0) * row_stride : NULL;
        const __fp16 *src_row1 = (r + 1 < n_rows) ? src + (r + 1) * row_stride : NULL;

        for (int c = 0; c < k; c += HMX_FP16_TILE_N_COLS) {
            int c0 = c / HMX_FP16_TILE_N_COLS;
            int tile_idx = r0 * n_tiles_per_row + c0;

            __fp16 *tile_base = vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS;

            for (int i = 0; i < HMX_FP16_TILE_N_COLS; ++i) {
                tile_base[(r1 / 2) * 64 + i * 2] =
                    (src_row0 && (c + i) < k) ? src_row0[c + i] : (__fp16)0;
            }
            for (int i = 0; i < HMX_FP16_TILE_N_COLS; ++i) {
                tile_base[(r1 / 2) * 64 + i * 2 + 1] =
                    (src_row1 && (c + i) < k) ? src_row1[c + i] : (__fp16)0;
            }
        }
    }
}

// Worker for parallel memcpy of fp32 rows (activation or weight)
typedef struct {
    float       *dst;
    const float *src;
    int          k;             // elements per row
    int          src_stride;    // source row stride (in float elements)
    int          start_row;
    int          end_row;
    worker_synctoken_t *synctoken;
} memcpy_rows_task_t;

static void memcpy_rows_worker(void *data) {
    memcpy_rows_task_t *t = (memcpy_rows_task_t *)data;
    for (int i = t->start_row; i < t->end_row; i++) {
        memcpy(t->dst + i * t->k, t->src + i * t->src_stride, t->k * sizeof(float));
    }
    __asm__ __volatile__("" ::: "memory");
    if (t->synctoken) worker_pool_synctoken_jobdone(t->synctoken);
}

// Worker for parallel output writeback
typedef struct {
    float        *dst;
    const __fp16 *src;
    int           n_rows;
    int           n_cols;
    int           col_stride;
    int           start_row;
    int           end_row;
    worker_synctoken_t *synctoken;
} output_wb_task_t;

static void output_wb_worker(void *data) {
    output_wb_task_t *t = (output_wb_task_t *)data;
    transfer_output_chunk_fp16_to_fp32_range(
        t->dst, t->src, t->n_rows, t->n_cols, t->col_stride,
        t->start_row, t->end_row);
    if (t->synctoken) worker_pool_synctoken_jobdone(t->synctoken);
}

// Worker for parallel activation fp32->fp16 conversion
typedef struct {
    __fp16       *vtcm_dst;
    const float  *src;
    int           n_rows;
    int           n_cols;
    int           row_stride;
    int           start_row;
    int           end_row;
    worker_synctoken_t *synctoken;
} act_convert_task_t;

static void act_convert_worker(void *data) {
    act_convert_task_t *t = (act_convert_task_t *)data;
    transfer_activation_chunk_fp32_to_fp16_range(
        t->vtcm_dst, t->src, t->n_rows, t->n_cols, t->row_stride,
        t->start_row, t->end_row);
    if (t->synctoken) worker_pool_synctoken_jobdone(t->synctoken);
}

// Worker for parallel activation f16->f16 tiles conversion
typedef struct {
    __fp16       *vtcm_dst;
    const __fp16 *src;
    int           n_rows;
    int           k;
    int           row_stride;
    int           start_row;
    int           end_row;
    worker_synctoken_t *synctoken;
} act_f16_convert_task_t;

static void act_f16_convert_worker(void *data) {
    act_f16_convert_task_t *t = (act_f16_convert_task_t *)data;
    transfer_activation_chunk_f16_to_f16_tiles_range(
        t->vtcm_dst, t->src, t->n_rows, t->k, t->row_stride,
        t->start_row, t->end_row);
    if (t->synctoken) worker_pool_synctoken_jobdone(t->synctoken);
}

// Helper: split tile-aligned rows across workers
static void split_tile_rows(int total_rows, int n_threads,
                            int start_rows[], int end_rows[]) {
    const int tile_rows = HMX_FP16_TILE_N_ROWS;
    int total_tiles = (total_rows + tile_rows - 1) / tile_rows;
    int tiles_per_thread = (total_tiles + n_threads - 1) / n_threads;
    for (int t = 0; t < n_threads; t++) {
        int tile_start = t * tiles_per_thread;
        int tile_end   = MIN((t + 1) * tiles_per_thread, total_tiles);
        start_rows[t] = tile_start * tile_rows;
        end_rows[t]   = MIN(tile_end * tile_rows, total_rows);
    }
}

// Helper: submit parallel memcpy of fp32 rows and wait
static void parallel_memcpy_rows(float *dst, const float *src,
                                 int n_rows, int k, int src_stride,
                                 int n_threads) {
    if (n_rows <= 0 || n_threads <= 1) {
        for (int i = 0; i < n_rows; i++) {
            memcpy(dst + i * k, src + i * src_stride, k * sizeof(float));
        }
        __asm__ __volatile__("" ::: "memory");
        return;
    }

    int sr[MAX_NUM_WORKERS], er[MAX_NUM_WORKERS];
    split_tile_rows(n_rows, n_threads, sr, er);

    memcpy_rows_task_t tasks[MAX_NUM_WORKERS];
    worker_synctoken_t synctoken;
    worker_pool_synctoken_init(&synctoken, n_threads - 1);

    for (int t = 0; t < n_threads; t++) {
        if (sr[t] >= er[t]) {
            if (t > 0) worker_pool_synctoken_jobdone(&synctoken);
            continue;
        }
        tasks[t] = (memcpy_rows_task_t){
            .dst = dst, .src = src, .k = k, .src_stride = src_stride,
            .start_row = sr[t], .end_row = er[t],
            .synctoken = (t == 0) ? NULL : &synctoken,
        };
        if (t == 0) {
            memcpy_rows_worker(&tasks[t]);
        } else {
            worker_pool_job_t job = { memcpy_rows_worker, &tasks[t] };
            worker_pool_submit(NULL, job);
        }
    }
    worker_pool_synctoken_wait(&synctoken);
}

// Helper: submit parallel activation fp32->fp16 conversion and wait
static void parallel_act_convert_fp32(__fp16 *vtcm_dst, const float *src,
                                      int n_rows, int n_cols, int row_stride,
                                      int n_threads) {
    // Zero-initialize padding rows so HVX can safely convert them to fp16 0.0
    const int n_rows_padded = ((n_rows + HMX_FP16_TILE_N_ROWS - 1) / HMX_FP16_TILE_N_ROWS) * HMX_FP16_TILE_N_ROWS;
    for (int r = n_rows; r < n_rows_padded; ++r) {
        memset((void *)(src + r * row_stride), 0, n_cols * sizeof(float));
    }

    if (n_rows <= 0 || n_threads <= 1) {
        transfer_activation_chunk_fp32_to_fp16(vtcm_dst, src, n_rows, n_cols, row_stride);
        return;
    }

    int sr[MAX_NUM_WORKERS], er[MAX_NUM_WORKERS];
    split_tile_rows(n_rows, n_threads, sr, er);

    act_convert_task_t tasks[MAX_NUM_WORKERS];
    worker_synctoken_t synctoken;
    worker_pool_synctoken_init(&synctoken, n_threads - 1);

    for (int t = 0; t < n_threads; t++) {
        if (sr[t] >= er[t]) {
            if (t > 0) worker_pool_synctoken_jobdone(&synctoken);
            continue;
        }
        tasks[t] = (act_convert_task_t){
            .vtcm_dst = vtcm_dst, .src = src,
            .n_rows = n_rows, .n_cols = n_cols, .row_stride = row_stride,
            .start_row = sr[t], .end_row = er[t],
            .synctoken = (t == 0) ? NULL : &synctoken,
        };
        if (t == 0) {
            act_convert_worker(&tasks[t]);
        } else {
            worker_pool_job_t job = { act_convert_worker, &tasks[t] };
            worker_pool_submit(NULL, job);
        }
    }
    worker_pool_synctoken_wait(&synctoken);
}

// Helper: submit parallel activation f16->f16 tiles conversion and wait
static void parallel_act_convert_f16(__fp16 *vtcm_dst, const __fp16 *src,
                                     int n_rows, int k, int row_stride,
                                     int n_threads) {
    if (n_rows <= 0 || n_threads <= 1) {
        transfer_activation_chunk_f16_to_f16_tiles(vtcm_dst, src, n_rows, k, row_stride);
        return;
    }

    int sr[MAX_NUM_WORKERS], er[MAX_NUM_WORKERS];
    split_tile_rows(n_rows, n_threads, sr, er);

    act_f16_convert_task_t tasks[MAX_NUM_WORKERS];
    worker_synctoken_t synctoken;
    worker_pool_synctoken_init(&synctoken, n_threads - 1);

    for (int t = 0; t < n_threads; t++) {
        if (sr[t] >= er[t]) {
            if (t > 0) worker_pool_synctoken_jobdone(&synctoken);
            continue;
        }
        tasks[t] = (act_f16_convert_task_t){
            .vtcm_dst = vtcm_dst, .src = src,
            .n_rows = n_rows, .k = k, .row_stride = row_stride,
            .start_row = sr[t], .end_row = er[t],
            .synctoken = (t == 0) ? NULL : &synctoken,
        };
        if (t == 0) {
            act_f16_convert_worker(&tasks[t]);
        } else {
            worker_pool_job_t job = { act_f16_convert_worker, &tasks[t] };
            worker_pool_submit(NULL, job);
        }
    }
    worker_pool_synctoken_wait(&synctoken);
}

// Helper: submit parallel output writeback and wait
static void parallel_output_writeback(float *dst, const __fp16 *src,
                                      int n_rows, int n_cols, int col_stride,
                                      int n_threads) {
    if (n_rows <= 0 || n_threads <= 1) {
        transfer_output_chunk_fp16_to_fp32(dst, src, n_rows, n_cols, col_stride);
        return;
    }

    int sr[MAX_NUM_WORKERS], er[MAX_NUM_WORKERS];
    split_tile_rows(n_rows, n_threads, sr, er);

    output_wb_task_t tasks[MAX_NUM_WORKERS];
    worker_synctoken_t synctoken;
    worker_pool_synctoken_init(&synctoken, n_threads - 1);

    for (int t = 0; t < n_threads; t++) {
        if (sr[t] >= er[t]) {
            if (t > 0) worker_pool_synctoken_jobdone(&synctoken);
            continue;
        }
        tasks[t] = (output_wb_task_t){
            .dst = dst, .src = src,
            .n_rows = n_rows, .n_cols = n_cols, .col_stride = col_stride,
            .start_row = sr[t], .end_row = er[t],
            .synctoken = (t == 0) ? NULL : &synctoken,
        };
        if (t == 0) {
            output_wb_worker(&tasks[t]);
        } else {
            worker_pool_job_t job = { output_wb_worker, &tasks[t] };
            worker_pool_submit(NULL, job);
        }
    }
    worker_pool_synctoken_wait(&synctoken);
}

// HMX weight dequantization: common params for all src0 types
struct hmx_weight_dequant_params {
    __fp16     *vtcm_weight;       // destination: FP16 tiles in VTCM
    const void *weight_chunk;     // source: weight data in DDR
    float      *vtcm_fp32_buf;    // VTCM FP32 buffer (F32 weight DMA target)
    dma_queue  *dma;              // DMA queue (F32 weight async transfer)
    int         M_cols;           // number of weight columns in this chunk
    int         K;                // inner dimension
    size_t      row_stride;       // src0 row stride in bytes
};

typedef void (*hmx_weight_dequant_fn)(const struct hmx_weight_dequant_params *p);

struct hmx_weight_traits {
    enum ggml_type type;
    hmx_weight_dequant_fn dequant;       // generic implementation
    hmx_weight_dequant_fn dequant_hvx;   // HVX implementation (NULL if only HVX)
};

static void wrap_dequant_f16(const struct hmx_weight_dequant_params *p) {
    transfer_weight_chunk_f16_to_f16_tiles(p->vtcm_weight, (const __fp16 *)p->weight_chunk,
                                           p->M_cols, p->K, p->row_stride / sizeof(__fp16));
}
static void wrap_dequant_f16_hvx(const struct hmx_weight_dequant_params *p) {
    transfer_weight_chunk_f16_to_f16_tiles_hvx(p->vtcm_weight, (const __fp16 *)p->weight_chunk,
                                                p->M_cols, p->K, p->row_stride / sizeof(__fp16));
}

static void wrap_dequant_f32(const struct hmx_weight_dequant_params *p) {
    dma_queue_push_ddr_to_vtcm(p->dma,
        dma_make_ptr(p->vtcm_fp32_buf, (const float *)p->weight_chunk),
        p->K * sizeof(float), p->row_stride, p->M_cols);
    dma_queue_pop(p->dma);
    convert_weight_f32_to_fp16_tiles(p->vtcm_weight, p->vtcm_fp32_buf, p->M_cols, p->K, p->K);
}
static void wrap_dequant_f32_hvx(const struct hmx_weight_dequant_params *p) {
    dma_queue_push_ddr_to_vtcm(p->dma,
        dma_make_ptr(p->vtcm_fp32_buf, (const float *)p->weight_chunk),
        p->K * sizeof(float), p->row_stride, p->M_cols);
    dma_queue_pop(p->dma);
    convert_weight_f32_to_fp16_tiles_hvx(p->vtcm_weight, p->vtcm_fp32_buf, p->M_cols, p->K, p->K);
}

static void wrap_dequant_bf16(const struct hmx_weight_dequant_params *p) {
    convert_weight_bf16_to_fp16_tiles(p->vtcm_weight, (const ggml_bf16_t *)p->weight_chunk,
                                      p->M_cols, p->K, p->row_stride / sizeof(ggml_bf16_t));
}
static void wrap_dequant_bf16_hvx(const struct hmx_weight_dequant_params *p) {
    convert_weight_bf16_to_fp16_tiles_hvx(p->vtcm_weight, (const ggml_bf16_t *)p->weight_chunk,
                                           p->M_cols, p->K, p->row_stride / sizeof(ggml_bf16_t));
}

static void wrap_dequant_q4_0(const struct hmx_weight_dequant_params *p) {
    dequantize_q4_0_to_f16_tiles(p->vtcm_weight, (const block_q4_0 *)p->weight_chunk, p->M_cols, p->K);
}
static void wrap_dequant_q4_0_hvx(const struct hmx_weight_dequant_params *p) {
    dequantize_q4_0_to_f16_tiles_hvx(p->vtcm_weight, (const block_q4_0 *)p->weight_chunk, p->M_cols, p->K);
}

static void wrap_dequant_q4_1(const struct hmx_weight_dequant_params *p) {
    dequantize_q4_1_to_f16_tiles(p->vtcm_weight, (const block_q4_1 *)p->weight_chunk, p->M_cols, p->K);
}
static void wrap_dequant_q4_1_hvx(const struct hmx_weight_dequant_params *p) {
    dequantize_q4_1_to_f16_tiles_hvx(p->vtcm_weight, (const block_q4_1 *)p->weight_chunk, p->M_cols, p->K);
}

static void wrap_dequant_q5_0(const struct hmx_weight_dequant_params *p) {
    dequantize_q5_0_to_f16_tiles(p->vtcm_weight, (const block_q5_0 *)p->weight_chunk, p->M_cols, p->K);
}
static void wrap_dequant_q5_0_hvx(const struct hmx_weight_dequant_params *p) {
    dequantize_q5_0_to_f16_tiles_hvx(p->vtcm_weight, (const block_q5_0 *)p->weight_chunk, p->M_cols, p->K);
}

static void wrap_dequant_q8_0(const struct hmx_weight_dequant_params *p) {
    dequantize_q8_0_to_f16_tiles(p->vtcm_weight, (const block_q8_0 *)p->weight_chunk, p->M_cols, p->K);
}
static void wrap_dequant_q8_0_hvx(const struct hmx_weight_dequant_params *p) {
    dequantize_q8_0_to_f16_tiles_hvx(p->vtcm_weight, (const block_q8_0 *)p->weight_chunk, p->M_cols, p->K);
}

static void wrap_dequant_iq4_nl(const struct hmx_weight_dequant_params *p) {
    dequantize_iq4_nl_to_f16_tiles(p->vtcm_weight, (const block_iq4_nl *)p->weight_chunk, p->M_cols, p->K);
}
static void wrap_dequant_iq4_nl_hvx(const struct hmx_weight_dequant_params *p) {
    dequantize_iq4_nl_to_f16_tiles_hvx(p->vtcm_weight, (const block_iq4_nl *)p->weight_chunk, p->M_cols, p->K);
}

static void wrap_dequant_x4x2_q4_0_hvx(const struct hmx_weight_dequant_params *p) {
    dequantize_x4x2_q4_0_to_f16_tiles_hvx(p->vtcm_weight, (const uint8_t *)p->weight_chunk,
                                            p->M_cols, p->K, p->row_stride);
}

static void wrap_dequant_mxfp4(const struct hmx_weight_dequant_params *p) {
    dequantize_mxfp4_to_f16_tiles(p->vtcm_weight, (const block_mxfp4 *)p->weight_chunk, p->M_cols, p->K);
}
static void wrap_dequant_mxfp4_hvx(const struct hmx_weight_dequant_params *p) {
    dequantize_mxfp4_to_f16_tiles_hvx(p->vtcm_weight, (const block_mxfp4 *)p->weight_chunk, p->M_cols, p->K);
}

static const struct hmx_weight_traits hmx_weight_traits_table[] = {
    { GGML_TYPE_F16,      wrap_dequant_f16,          wrap_dequant_f16_hvx },
    { GGML_TYPE_F32,      wrap_dequant_f32,          wrap_dequant_f32_hvx },
    { GGML_TYPE_BF16,     wrap_dequant_bf16,         wrap_dequant_bf16_hvx },
    { GGML_TYPE_Q4_0,     wrap_dequant_q4_0,         wrap_dequant_q4_0_hvx },
    { GGML_TYPE_Q4_1,     wrap_dequant_q4_1,         wrap_dequant_q4_1_hvx },
    { GGML_TYPE_Q5_0,     wrap_dequant_q5_0,         wrap_dequant_q5_0_hvx },
    { GGML_TYPE_Q8_0,     wrap_dequant_q8_0,         wrap_dequant_q8_0_hvx },
    { GGML_TYPE_IQ4_NL,   wrap_dequant_iq4_nl,       wrap_dequant_iq4_nl_hvx },
    { GGML_TYPE_MXFP4,    wrap_dequant_mxfp4,        wrap_dequant_mxfp4_hvx },
    { GGML_TYPE_Q4_0x4x2, NULL,                      wrap_dequant_x4x2_q4_0_hvx },
};
#define HMX_WEIGHT_TRAITS_TABLE_SIZE (sizeof(hmx_weight_traits_table) / sizeof(hmx_weight_traits_table[0]))

static const struct hmx_weight_traits * hmx_weight_traits_lookup(enum ggml_type type) {
    for (int i = 0; i < (int)HMX_WEIGHT_TRAITS_TABLE_SIZE; i++) {
        if (hmx_weight_traits_table[i].type == type) {
            return &hmx_weight_traits_table[i];
        }
    }
    return NULL;
}

int ggmlop_dsp_mulmat_hmx(remote_handle64 h, const struct dsptensor * src0, const struct dsptensor * src1, dsptensor * dst) {
    // Early type checks before acquiring any resources
    // src0 (weight) types supported by HMX path
    const struct hmx_weight_traits *wt = hmx_weight_traits_lookup(src0->type);
    if (wt == NULL || src0->type == GGML_TYPE_F16) {
        GGMLHEXAGON_LOG_INFO("src0 type %d not supported by HMX, falling back to VTCM multithread mode\n", src0->type);
        return ggmlop_dsp_mulmat_multithread_vtcm(h, src0, src1, dst);
    }

    // src1 (activation) types supported by HMX path
    if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_BF16) {
        GGMLHEXAGON_LOG_INFO("src1 type %d not supported by HMX, falling back to VTCM multithread mode\n", src1->type);
        return ggmlop_dsp_mulmat_multithread_vtcm(h, src0, src1, dst);
    }

    unsigned int compute_res_ctx_id = ggmlop_get_compute_res_ctx_id();
    int hmx_locked = 0;
    if (compute_res_ctx_id != 0) {
        int lock_result = HAP_compute_res_hmx_lock(compute_res_ctx_id);
        if (lock_result != 0) {
            GGMLHEXAGON_LOG_INFO("HMX lock failed (%d), falling back to VTCM multithread mode\n", lock_result);
            return ggmlop_dsp_mulmat_multithread_vtcm(h, src0, src1, dst);
        }
        hmx_locked = 1;
    } else {
        GGMLHEXAGON_LOG_INFO("falling back to VTCM multithread mode\n");
        return ggmlop_dsp_mulmat_multithread_vtcm(h, src0, src1, dst);
    }

    // Ensure VTCM resource is available (for cache mode)
    int vtcm_err = ggmlop_ensure_vtcm_available();
    if (vtcm_err != 0) {
        if (hmx_locked) {
            HAP_compute_res_hmx_unlock(compute_res_ctx_id);
        }
        GGMLHEXAGON_LOG_INFO("VTCM ensure failed (%d), falling back to VTCM multithread mode\n", vtcm_err);
        return ggmlop_dsp_mulmat_multithread_vtcm(h, src0, src1, dst);
    }

    dst->ne[0] = src0->ne[1];
    dst->ne[1] = src1->ne[1];
    dst->ne[2] = src1->ne[2];
    dst->ne[3] = src1->ne[3];

    dst->nb[0] = 4;
    dst->nb[1] = dst->nb[0] * dst->ne[0];
    dst->nb[2] = dst->nb[1] * dst->ne[1];
    dst->nb[3] = dst->nb[2] * dst->ne[2];

    // CRITICAL FIX: Align with ggml_mul_mat definition
    // ggml_mul_mat(src0, src1): src0=weight[K,M], src1=activation[K,N], dst=[M,N]
    // src0->ne[0] = K (inner dimension), src0->ne[1] = M (weight columns)
    // src1->ne[0] = K (inner dimension), src1->ne[1] = N (activation columns)
    const int32_t M = src0->ne[1];  // weight columns (output dimension)
    const int32_t K = src0->ne[0];  // inner dimension
    const int32_t N = src1->ne[1];  // activation columns (batch size)

    GGMLHEXAGON_LOG_INFO("HMX matmul: src0(weight)[K=%d, M=%d], src1(activation)[K=%d, N=%d], dst[M=%d, N=%d]",
                         K, M, K, N, M, N);
    //GGMLHEXAGON_LOG_DEBUG("src0 type=%d, src1 type=%d", src0->type, src1->type);

    if (K % HMX_FP16_TILE_N_COLS != 0) {
        if (hmx_locked) {
            HAP_compute_res_hmx_unlock(compute_res_ctx_id);
        }
        GGMLHEXAGON_LOG_INFO("K=%d not 32-aligned, falling back to VTCM multithread mode\n", K);
        return ggmlop_dsp_mulmat_multithread_vtcm(h, src0, src1, dst);
    }

    size_t vtcm_size = 0;
    void * vtcm_base = ggmlop_get_vtcm_pool(&vtcm_size);
    if (vtcm_base == NULL) {
        if (hmx_locked) {
            HAP_compute_res_hmx_unlock(compute_res_ctx_id);
        }
        GGMLHEXAGON_LOG_INFO("falling back to VTCM multithread mode\n");
        return ggmlop_dsp_mulmat_multithread(h, src0, src1, dst);
    }

    if ((uintptr_t)vtcm_base % HMX_FP16_TILE_SIZE != 0) {
        if (hmx_locked) {
            HAP_compute_res_hmx_unlock(compute_res_ctx_id);
        }
        GGMLHEXAGON_LOG_INFO("falling back to VTCM multithread mode\n");
        return ggmlop_dsp_mulmat_multithread_vtcm(h, src0, src1, dst);
    }

    // VTCM layout calculation with M-dimension chunking
    // src0 = weight [K, M], src1 = activation [K, N]
    // We chunk both M (weight columns) and N (activation columns)
    //
    // For F32 weight: weight_fp32_buf + weight_tiles + reusable_buf + act_tiles + scales <= vtcm_size
    //   weight_fp32_buf = M_chunk * K * 4  (fp32 input for weight conversion)
    //   weight_tiles    = M_chunk * K * 2  (fp16 tiles)
    //   act_tiles       = N_chunk * K * 2  (fp16 tiles)
    //   output_tiles    = M_chunk * N_chunk * 2 (fp16, time-shared with act_fp32_buf)
    //   reusable_buf    = max(act_fp32_buf, output_tiles)
    //   act_fp32_buf    = N_chunk * K * 4
    //
    // For quantized/BF16 weight: no weight_fp32_buf needed (dequantize directly to fp16 tiles)

    const bool src0_needs_fp32_buf = (src0->type == GGML_TYPE_F32);
    const bool src1_needs_fp32_buf = (src1->type == GGML_TYPE_F32);

    const size_t vec_dot_size = K * sizeof(__fp16);
    const size_t scales_size  = 256;

    // Sweep M_chunk from max down to find a fit
    // M_chunk is rounded up to tile boundary for VTCM allocation
    const size_t M_padded = ((size_t)M + HMX_FP16_TILE_N_COLS - 1) / HMX_FP16_TILE_N_COLS * HMX_FP16_TILE_N_COLS;
    size_t M_chunk_n_cols = 0;
    size_t N_chunk_n_rows = 0;

    for (size_t mc = M_padded; mc >= HMX_FP16_TILE_N_COLS; mc -= HMX_FP16_TILE_N_COLS) {
        const size_t w_fp32  = src0_needs_fp32_buf ? hex_align_up(mc * K * sizeof(float), HMX_FP16_TILE_SIZE) : 0;
        const size_t w_tiles = hex_align_up(mc * vec_dot_size, HMX_FP16_TILE_SIZE);
        const size_t remain  = vtcm_size - w_fp32 - w_tiles - scales_size;
        if (remain <= 0) continue;

        // N * K * 2 + max(act_fp32_buf, output_tiles) <= remain
        // act_fp32_buf is needed only when src1 is F32
        const size_t act_fp32_per_n = src1_needs_fp32_buf ? K * sizeof(float) : 0;
        const size_t per_n_act  = K * sizeof(__fp16);  // act_tiles
        const size_t per_n_reusable = (act_fp32_per_n > mc * sizeof(__fp16)) ? act_fp32_per_n : mc * sizeof(__fp16);
        const size_t per_n = per_n_act + per_n_reusable;

        size_t nc = (per_n > 0) ? hex_align_down(remain / per_n, HMX_FP16_TILE_N_ROWS) : HMX_FP16_TILE_N_ROWS;
        if (nc == 0) nc = HMX_FP16_TILE_N_ROWS;

        // Clamp N_chunk to N (allow non-32-aligned N, pad to tile boundary)
        if (nc > (size_t)N) nc = (size_t)N;
        if (nc == 0 && N > 0) nc = HMX_FP16_TILE_N_ROWS;

        // Verify it actually fits (use padded tile counts for VTCM allocation)
        const size_t nc_padded = ((nc + HMX_FP16_TILE_N_ROWS - 1) / HMX_FP16_TILE_N_ROWS) * HMX_FP16_TILE_N_ROWS;
        const size_t a_fp32   = src1_needs_fp32_buf ? hex_align_up(nc_padded * K * sizeof(float), HMX_FP16_TILE_SIZE) : 0;
        const size_t a_tiles  = hex_align_up(nc_padded * vec_dot_size, HMX_FP16_TILE_SIZE);
        const size_t o_tiles  = hex_align_up(nc_padded * mc * sizeof(__fp16), HMX_FP16_TILE_SIZE);
        const size_t reusable = (a_fp32 > o_tiles) ? a_fp32 : o_tiles;
        const size_t total    = w_fp32 + w_tiles + a_tiles + reusable + scales_size;

        if (total <= vtcm_size) {
            M_chunk_n_cols = mc;
            N_chunk_n_rows = nc;
            break;
        }
    }

    if (M_chunk_n_cols == 0) {
        if (hmx_locked) {
            HAP_compute_res_hmx_unlock(compute_res_ctx_id);
        }
        GGMLHEXAGON_LOG_INFO("Cannot fit even one tile in VTCM, falling back to VTCM multithread mode\n");
        return ggmlop_dsp_mulmat_multithread_vtcm(h, src0, src1, dst);
    }

    // Recompute exact sizes for chosen chunks (use padded sizes for VTCM allocation)
    const size_t M_chunk_padded = ((M_chunk_n_cols + HMX_FP16_TILE_N_COLS - 1) / HMX_FP16_TILE_N_COLS) * HMX_FP16_TILE_N_COLS;
    const size_t N_chunk_padded = ((N_chunk_n_rows + HMX_FP16_TILE_N_ROWS - 1) / HMX_FP16_TILE_N_ROWS) * HMX_FP16_TILE_N_ROWS;
    const size_t weight_fp32_buf_size = src0_needs_fp32_buf ? hex_align_up(M_chunk_padded * K * sizeof(float), HMX_FP16_TILE_SIZE) : 0;
    const size_t weight_area_size     = hex_align_up(M_chunk_padded * vec_dot_size, HMX_FP16_TILE_SIZE);
    const size_t act_fp32_buf_size    = src1_needs_fp32_buf ? hex_align_up(N_chunk_padded * K * sizeof(float), HMX_FP16_TILE_SIZE) : 0;
    const size_t act_area_size        = hex_align_up(N_chunk_padded * vec_dot_size, HMX_FP16_TILE_SIZE);
    const size_t output_area_size     = hex_align_up(N_chunk_padded * M_chunk_padded * sizeof(__fp16), HMX_FP16_TILE_SIZE);
    const size_t reusable_buf_size    = (act_fp32_buf_size > output_area_size) ? act_fp32_buf_size : output_area_size;
    const size_t total_vtcm_needed    = act_area_size + weight_area_size + reusable_buf_size + weight_fp32_buf_size + scales_size;

    const char * src0_type_name = (src0->type == GGML_TYPE_Q4_0x4x2) ? "Q4_0x4x2" : ggml_get_type_traits((enum ggml_type)src0->type)->type_name;
    GGMLHEXAGON_LOG_INFO("VTCM check: (src0 %s)M=%d, N=%d, K=%d, vtcm_size=%zu, M_chunk=%zu, N_chunk=%zu, total_needed=%zu (act=%zu, weight=%zu, reusable=%zu, weight_fp32=%zu, scales=%zu)",
                         src0_type_name,
                         M, N, K, vtcm_size, M_chunk_n_cols, N_chunk_n_rows, total_vtcm_needed, act_area_size, weight_area_size, reusable_buf_size, weight_fp32_buf_size, scales_size);

    GGMLHEXAGON_LOG_INFO("begin real vtcm + hmx");
    uint8_t *vtcm_ptr = (uint8_t *)vtcm_base;
    __fp16 *vtcm_activation = (__fp16 *) vtcm_ptr;  // activation tiles (interleaved format)
    vtcm_ptr += act_area_size;
    __fp16 *vtcm_weight = (__fp16 *) vtcm_ptr;      // weight tiles (interleaved format)
    vtcm_ptr += weight_area_size;
    // Reusable buffer: used as act_fp32_buf during activation conversion,
    // then as output_area during HMX computation
    union {
        float *fp32;
        __fp16 *fp16;
    } reusable_buf;
    reusable_buf.fp32 = (float *) vtcm_ptr;
    reusable_buf.fp16 = (__fp16 *) vtcm_ptr;
    vtcm_ptr += reusable_buf_size;
    float *vtcm_weight_fp32_buf = (float *) vtcm_ptr;
    vtcm_ptr += weight_fp32_buf_size;
    __fp16 *vtcm_scales = (__fp16 *) vtcm_ptr;

    HVX_Vector v_scale = Q6_V_vsplat_R(0x3c00);
    volatile HVX_Vector *pv_scales = (volatile HVX_Vector *) vtcm_scales;
    pv_scales[0] = v_scale;
    pv_scales[1] = Q6_V_vzero();

    const size_t n_dot_tiles = K / HMX_FP16_TILE_N_COLS;

    const bool src1_is_f16 = (src1->type == GGML_TYPE_F16);  // activation type

    const size_t src0_row_stride = src0->nb[1];  // weight stride
    const size_t src1_row_stride = src1->nb[1];  // activation stride

    // Resolve weight dequantization function from lookup table
    const int use_hvx = ggml_get_dsp_use_hvx();
    const hmx_weight_dequant_fn weight_dequant_fn =
        (use_hvx && wt->dequant_hvx) ? wt->dequant_hvx :
        (wt->dequant ? wt->dequant : wt->dequant_hvx);

    // Create DMA queue for async data transfers
    dma_queue *dma = dma_queue_create(16);

    // Outer loop: iterate over M (weight columns)
    // Inner loop: iterate over N (activation columns)
    // Weight uses column-pair interleaved format, Activation uses row-pair interleaved format

    // FP16 weight cache: lookup by src0->data pointer (stable ION address)
    const int cache_enabled = src0->op_params[0];
    fp16_weight_cache_entry_t * cache_entry = NULL;
    int cache_is_hit = 0;

    if (cache_enabled) {
        cache_entry = fp16_weight_cache_lookup(src0->data);
        if (cache_entry && cache_entry->M == M && cache_entry->K == K && cache_entry->type == src0->type) {
            cache_is_hit = 1;
            GGMLHEXAGON_LOG_INFO("FP16 weight cache: HIT, src0=%p, size=%u", src0->data, cache_entry->fp16_size);
        } else {
            cache_entry = NULL;  // mismatch, treat as miss
            GGMLHEXAGON_LOG_INFO("FP16 weight cache: MISS, src0=%p, M=%d, K=%d, type=%d",
                                 src0->data, M, K, src0->type);
        }
    }

    for (size_t mc = 0; mc < M; mc += M_chunk_n_cols) {
        const size_t M_cols = (M - mc) > M_chunk_n_cols ? M_chunk_n_cols : (M - mc);
        const size_t M_col_tiles = (M_cols + HMX_FP16_TILE_N_COLS - 1) / HMX_FP16_TILE_N_COLS;

        // Compute cache offset for this M chunk
        const size_t chunk_cache_offset = (mc / HMX_FP16_TILE_N_COLS) * n_dot_tiles * HMX_FP16_TILE_SIZE;
        const size_t chunk_cache_size = M_col_tiles * n_dot_tiles * HMX_FP16_TILE_SIZE;

        // Check if we can read from FP16 weight cache
        if (cache_is_hit && cache_entry) {
            // Cache hit: copy FP16 tiles from ION cache to VTCM
            // ION cache is in DSP L2 cache from previous write, no invalidate needed
            memcpy(vtcm_weight, (uint8_t *)cache_entry->fp16_ptr + chunk_cache_offset, chunk_cache_size);
        } else {
            // Cache miss: convert weight chunk (src0) to fp16 tiles via lookup table dispatch
            struct hmx_weight_dequant_params wparams = {
                .vtcm_weight   = vtcm_weight,
                .weight_chunk  = (const char *)src0->data + mc * src0_row_stride,
                .vtcm_fp32_buf = vtcm_weight_fp32_buf,
                .dma           = dma,
                .M_cols        = (int)M_cols,
                .K             = K,
                .row_stride    = src0_row_stride,
            };
            weight_dequant_fn(&wparams);

            // Write converted FP16 tiles to ION cache region
            if (cache_enabled && !cache_is_hit && !cache_entry) {
                // First chunk: allocate cache and copy this chunk
                const size_t M_padded_cache = ((size_t)M + HMX_FP16_TILE_N_COLS - 1) / HMX_FP16_TILE_N_COLS * HMX_FP16_TILE_N_COLS;
                const uint32_t total_fp16_size = (uint32_t)((M_padded_cache / HMX_FP16_TILE_N_COLS) * n_dot_tiles * HMX_FP16_TILE_SIZE);
                // Only cache weights above minimum size threshold
                if (total_fp16_size >= FP16_WEIGHT_CACHE_MIN_SIZE) {
                    cache_entry = fp16_weight_cache_insert(src0->data, total_fp16_size, M, K, src0->type);
                }
                if (cache_entry) {
                    memcpy((uint8_t *)cache_entry->fp16_ptr + chunk_cache_offset, vtcm_weight, chunk_cache_size);
                    ggmlop_dsp_cache_flush_range((uint8_t *)cache_entry->fp16_ptr + chunk_cache_offset, chunk_cache_size);
                }
            } else if (cache_enabled && !cache_is_hit && cache_entry) {
                // Subsequent chunks: just copy to already-allocated cache
                memcpy((uint8_t *)cache_entry->fp16_ptr + chunk_cache_offset, vtcm_weight, chunk_cache_size);
                ggmlop_dsp_cache_flush_range((uint8_t *)cache_entry->fp16_ptr + chunk_cache_offset, chunk_cache_size);
            }
        }

        // Pipeline: use DMA to prefetch next activation while HMX computes current
        // For the first N-chunk, we must prepare activation synchronously
        bool act_dma_pending = false;

        for (size_t nr = 0; nr < N; nr += N_chunk_n_rows) {
            const size_t N_rows = (N - nr) > N_chunk_n_rows ? N_chunk_n_rows : (N - nr);
            const size_t N_row_tiles = ((N_rows + HMX_FP16_TILE_N_ROWS - 1) / HMX_FP16_TILE_N_ROWS);

            // Convert activation chunk (src1) to fp16 tiles
            if (src1_is_f16) {
                const __fp16 *act_chunk = (const __fp16 *)((const char *)src1->data + nr * src1_row_stride);
                if (use_hvx) {
                    transfer_activation_chunk_f16_to_f16_tiles_hvx(vtcm_activation, act_chunk, N_rows, K, src1_row_stride / sizeof(__fp16));
                } else {
                    transfer_activation_chunk_f16_to_f16_tiles(vtcm_activation, act_chunk, N_rows, K, src1_row_stride / sizeof(__fp16));
                }
            } else if (src1->type == GGML_TYPE_BF16) {
                const ggml_bf16_t *act_chunk = (const ggml_bf16_t *)((const char *)src1->data + nr * src1_row_stride);
                if (use_hvx) {
                    convert_activation_bf16_to_fp16_tiles_hvx(vtcm_activation, act_chunk, N_rows, K, src1_row_stride / sizeof(ggml_bf16_t));
                } else {
                    // BF16 -> FP16: convert each element directly
                    // Reuse activation tile format (row-pair interleaved)
                    const int k_tiles = K / HMX_FP16_TILE_N_COLS;
                    const int n_row_tiles = (N_rows + HMX_FP16_TILE_N_ROWS - 1) / HMX_FP16_TILE_N_ROWS;
                    for (int rt = 0; rt < n_row_tiles * k_tiles; ++rt) {
                        int ct = rt / k_tiles;
                        int kt = rt % k_tiles;
                        __fp16 *tile_base = vtcm_activation + rt * HMX_FP16_TILE_N_ELMS;
                        for (int i = 0; i < HMX_FP16_TILE_N_ROWS; i += 2) {
                            int row0 = ct * HMX_FP16_TILE_N_ROWS + i;
                            int row1 = row0 + 1;
                            const ggml_bf16_t *r0 = (row0 < N_rows) ? act_chunk + row0 * (src1_row_stride / sizeof(ggml_bf16_t)) + kt * HMX_FP16_TILE_N_COLS : NULL;
                            const ggml_bf16_t *r1 = (row1 < N_rows) ? act_chunk + row1 * (src1_row_stride / sizeof(ggml_bf16_t)) + kt * HMX_FP16_TILE_N_COLS : NULL;
                            for (int j = 0; j < HMX_FP16_TILE_N_COLS; ++j) {
                                float v0 = r0 ? ggml_compute_bf16_to_fp32(r0[j]) : 0.0f;
                                float v1 = r1 ? ggml_compute_bf16_to_fp32(r1[j]) : 0.0f;
                                // Row-pair interleaved: tile[j*64 + i*2] = row0[j], tile[j*64 + i*2 + 1] = row1[j]
                                fp32_to_fp16_store(&tile_base[j * 64 + i * 2], v0);
                                fp32_to_fp16_store(&tile_base[j * 64 + i * 2 + 1], v1);
                            }
                        }
                    }
                }
            } else if (src1->type == GGML_TYPE_F32) {
                // Wait for pending DMA (from previous iteration's prefetch) or do sync copy
                if (act_dma_pending) {
                    dma_queue_pop(dma);  // wait for DMA completion
                    act_dma_pending = false;
                } else {
                    // First chunk: DMA push and wait immediately
                    const float *act_chunk = (const float *)((const char *)src1->data + nr * src1_row_stride);
                    dma_queue_push_ddr_to_vtcm(dma,
                        dma_make_ptr(reusable_buf.fp32, act_chunk),
                        K * sizeof(float), src1_row_stride, N_rows);
                    dma_queue_pop(dma);
                }

                // Convert from fp32 buffer to fp16 tiles (row-pair interleaved format)
                transfer_activation_chunk_fp32_to_fp16(vtcm_activation, reusable_buf.fp32, N_rows, K, K);
            }

            // Ensure all HVX vscatter writes to VTCM are visible to HMX
            // Read from the last tile of each buffer to force vscatter completion
            {
                const int n_act_tiles = N_row_tiles * n_dot_tiles;
                const int n_wt_tiles = M_col_tiles * n_dot_tiles;
                if (n_act_tiles > 0) {
                    (void) *(volatile HVX_Vector *)(vtcm_activation + (n_act_tiles - 1) * HMX_FP16_TILE_N_ELMS);
                }
                if (n_wt_tiles > 0) {
                    (void) *(volatile HVX_Vector *)(vtcm_weight + (n_wt_tiles - 1) * HMX_FP16_TILE_N_ELMS);
                }
            }

            // HMX computation
            core_dot_chunk_fp16(reusable_buf.fp16, vtcm_activation, vtcm_weight, vtcm_scales, N_row_tiles, M_col_tiles, n_dot_tiles);

            // Copy output to dst (must complete before DMA prefetch overwrites reusable_buf)
            float *output_chunk = (float *)((char *)dst->data + mc * dst->nb[0] + nr * dst->nb[1]);
            transfer_output_chunk_fp16_to_fp32(output_chunk, reusable_buf.fp16, N_rows, M_cols, M);

            // Prefetch next activation chunk via DMA (overlaps with next iteration's compute)
            // NOTE: this must be after output writeback since reusable_buf is shared
            // Only F32 activation uses DMA prefetch (BF16/F16 don't need fp32 buffer)
            size_t nr_next = nr + N_chunk_n_rows;
            if (nr_next < N && src1_needs_fp32_buf) {
                const float *act_chunk_next = (const float *)((const char *)src1->data + nr_next * src1_row_stride);
                dma_queue_push_ddr_to_vtcm(dma,
                    dma_make_ptr(reusable_buf.fp32, act_chunk_next),
                    K * sizeof(float), src1_row_stride,
                    (N - nr_next) > N_chunk_n_rows ? N_chunk_n_rows : (N - nr_next));
                act_dma_pending = true;
            }
        }
    }

    dma_queue_flush(dma);
    dma_queue_delete(dma);

    if (hmx_locked) {
        HAP_compute_res_hmx_unlock(compute_res_ctx_id);
    }
    GGMLHEXAGON_LOG_INFO("end real vtcm + hmx");

    return 0;
}

// Thread data for sgemm multithread
typedef struct {
    struct ggmldsp_compute_params cparams;
    struct sgemm_params s_params;
    worker_synctoken_t *synctoken;
} sgemm_thread_data_t;

static void sgemm_thread_func(void * data) {
    sgemm_thread_data_t * tdata = (sgemm_thread_data_t *)data;
    ggmldsp_llamafile_sgemm(&tdata->cparams, &tdata->s_params);
    if (tdata->synctoken) worker_pool_synctoken_jobdone(tdata->synctoken);
}

static int ggmlop_dsp_mulmat_sgemm(remote_handle64 h, const struct dsptensor * src0, const struct dsptensor * src1, dsptensor * dst) {
    const enum ggml_type type = src0->type;
    const enum ggml_type vec_dot_type = ggml_get_type_traits(type)->vec_dot_type;
    const size_t blck_size = ggml_blck_size(type);

    // Check if sgemm supports this type combination
    bool supported = false;
    if (type == GGML_TYPE_F32 && vec_dot_type == GGML_TYPE_F32) {
        supported = true;
    } else if (type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q5_0) {
        supported = true;
    } else if (type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) {
        supported = true;
    } else if (type == GGML_TYPE_IQ4_NL) {
        supported = true;
    }
    if (!supported) {
        GGMLHEXAGON_LOG_INFO("sgemm: type %d not supported, fallback", type);
        goto fallback;
    }

    // For F32/F16/BF16, k must be multiple of 32 (HVX_Vector holds 32 floats)
    if ((type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) && (src0->ne[0] % 32 != 0)) {
        GGMLHEXAGON_LOG_INFO("sgemm: k=%d not multiple of 32, fallback", src0->ne[0]);
        goto fallback;
    }

    dst->ne[0] = src0->ne[1];
    dst->ne[1] = src1->ne[1];
    dst->ne[2] = src1->ne[2];
    dst->ne[3] = src1->ne[3];

    dst->nb[0] = 4;
    dst->nb[1] = dst->nb[0] * dst->ne[0];
    dst->nb[2] = dst->nb[1] * dst->ne[1];
    dst->nb[3] = dst->nb[2] * dst->ne[2];

    const int32_t ne00 = src0->ne[0];
    const int32_t ne01 = src0->ne[1];
    const int32_t ne11 = src1->ne[1];
    const int32_t ne12 = src1->ne[2];
    const int32_t ne13 = src1->ne[3];

    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];
    const size_t nb11 = src1->nb[1];
    const size_t nb12 = src1->nb[2];
    const size_t nb13 = src1->nb[3];
    const size_t nb1  = dst->nb[1];
    const size_t nb2  = dst->nb[2];
    const size_t nb3  = dst->nb[3];

    const int32_t r2 = ne12 / src0->ne[2];
    const int32_t r3 = ne13 / src0->ne[3];

    const size_t type_size = ggml_type_size(type);
    const size_t vec_dot_type_size = ggml_type_size(vec_dot_type);

    // For F16/BF16: pre-convert to F32, then use F32 sgemm
    // This avoids the F32->F16->F32 round-trip through quantize
    bool use_f32_sgemm = (type == GGML_TYPE_F16 || type == GGML_TYPE_BF16);
    float * f32_A = NULL;
    float * f32_B = NULL;

    if (use_f32_sgemm) {
        const size_t f32_A_size = (size_t)ne01 * ne00 * sizeof(float);
        const size_t f32_B_size = (size_t)ne11 * ne00 * sizeof(float) * ne12 * ne13;
        f32_A = (float *)ggmlop_get_work_data(f32_A_size + f32_B_size);
        if (f32_A == NULL) {
            GGMLHEXAGON_LOG_INFO("sgemm: F16/BF16 work buffer alloc failed, fallback");
            goto fallback;
        }
        f32_B = f32_A + (size_t)ne01 * ne00;

        // Convert src0 (A) from F16/BF16 to F32
        for (int i = 0; i < ne01; ++i) {
            const void * src_row = (const char *)src0->data + i * nb01;
            if (type == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row_hvx((const ggml_fp16_t *)src_row, f32_A + i * ne00, ne00);
            } else {
                ggml_bf16_to_fp32_row_hvx((const ggml_bf16_t *)src_row, f32_A + i * ne00, ne00);
            }
        }

        // Convert src1 (B) to F32
        for (int i13 = 0; i13 < ne13; ++i13) {
            for (int i12 = 0; i12 < ne12; ++i12) {
                for (int i11 = 0; i11 < ne11; ++i11) {
                    const void * src_row = (const char *)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11;
                    float * dst_row = f32_B + ((i13 * ne12 + i12) * ne11 + i11) * ne00;
                    if (src1->type == GGML_TYPE_F32) {
                        memcpy(dst_row, src_row, ne00 * sizeof(float));
                    } else if (src1->type == GGML_TYPE_F16) {
                        ggml_fp16_to_fp32_row_hvx((const ggml_fp16_t *)src_row, dst_row, ne00);
                    } else if (src1->type == GGML_TYPE_BF16) {
                        ggml_bf16_to_fp32_row_hvx((const ggml_bf16_t *)src_row, dst_row, ne00);
                    }
                }
            }
        }
    }

    // Quantize src1 to vec_dot_type if needed (for quantized types only)
    // F16/BF16 skip this step - they convert directly to F32 instead
    const void * wdata = src1->data;
    if (!use_f32_sgemm && src1->type != vec_dot_type) {
        const size_t row_size = ggml_row_size(vec_dot_type, ne00);
        const size_t q8_size = row_size * ne11 * ne12 * ne13;
        void * q8_data = ggmlop_get_work_data(q8_size);
        if (q8_data != NULL) {
            const struct ggml_type_traits_dsp * quant_traits = ggml_get_type_traits_dsp(vec_dot_type);
            if (quant_traits->from_float) {
                for (int i13 = 0; i13 < ne13; ++i13) {
                    for (int i12 = 0; i12 < ne12; ++i12) {
                        for (int i11 = 0; i11 < ne11; ++i11) {
                            const float * src_row = (const float*)((const char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);
                            void * dst_row = (void*)((char*)q8_data + i13 * ne12 * ne11 * row_size + i12 * ne11 * row_size + i11 * row_size);
                            quant_traits->from_float(src_row, dst_row, ne00);
                        }
                    }
                }
            }
            wdata = q8_data;
        }
    }

    // Compute sgemm_params
    const size_t row_size = ggml_row_size(vec_dot_type, ne00);

    struct sgemm_params s_params;
    s_params.m     = ne01;
    s_params.n     = ne11;
    s_params.ldc   = nb1 / sizeof(float);
    s_params.Ctype = GGML_TYPE_F32;

    if (use_f32_sgemm) {
        // F16/BF16: use pre-converted F32 buffers
        s_params.k     = ne00;
        s_params.lda   = ne00;
        s_params.ldb   = ne00;
        s_params.Atype = GGML_TYPE_F32;
        s_params.Btype = GGML_TYPE_F32;
    } else {
        s_params.k     = ne00 / blck_size;
        s_params.lda   = nb01 / type_size;
        s_params.ldb   = row_size / vec_dot_type_size;
        s_params.Atype = type;
        s_params.Btype = vec_dot_type;
    }

    // VTCM buffering for quantized types
    // sgemm is designed for CPU cache; on DSP without VTCM, every load hits DDR.
    // Buffering A and B in VTCM gives HVX loads ~10x lower latency.
    // Must use DMA (not memcpy) for DDR->VTCM transfer to avoid cache coherence issues.
    bool use_vtcm = false;
    void * vtcm_A = NULL;
    void * vtcm_B = NULL;
    dma_queue * vtcm_dma = NULL;
    const size_t A_data_size = ne01 * nb01;
    const size_t B_data_size = ne11 * row_size;

    if (type != GGML_TYPE_F32) {
        int vtcm_err = ggmlop_ensure_vtcm_available();
        if (vtcm_err == 0) {
            size_t vtcm_pool_size = 0;
            void * vtcm_base = ggmlop_get_vtcm_pool(&vtcm_pool_size);
            if (vtcm_base != NULL && A_data_size + B_data_size <= vtcm_pool_size) {
                // VTCM buffering disabled: causes 5-6x slowdown due to
                // sgemm's tiled access pattern not benefiting from VTCM
                // (unlike vec_dot which is sequential and benefits greatly)
                // use_vtcm = true;
                GGMLHEXAGON_LOG_INFO("sgemm: VTCM available but disabled (A=%zu B=%zu), using DDR",
                                     A_data_size, B_data_size);
            }
        }
        if (!use_vtcm) {
            GGMLHEXAGON_LOG_INFO("sgemm: VTCM unavailable or too small (A=%zu B=%zu), using DDR",
                                 A_data_size, B_data_size);
        }
    }

    // Multi-threaded sgemm: distribute work across threads via ith/nth
    unsigned int n_threads = num_workers;
    if (n_threads < 1) n_threads = 1;
    if (n_threads > MAX_NUM_WORKERS) n_threads = MAX_NUM_WORKERS;

    for (int i13 = 0; i13 < ne13; ++i13) {
        for (int i12 = 0; i12 < ne12; ++i12) {
            if (use_f32_sgemm) {
                // F16/BF16: use pre-converted F32 buffers
                s_params.A = f32_A;
                s_params.B = f32_B + (i12 * ne11 + i13 * ne12 * ne11) * ne00;
            } else {
                const void * A_src = (const char *)src0->data + (i12 / r2) * nb02 + (i13 / r3) * nb03;
                const void * B_src = (const char *)wdata + (i12 * ne11 + i13 * ne12 * ne11) * row_size;

                if (use_vtcm) {
                    dma_queue_push_ddr_to_vtcm(vtcm_dma,
                        dma_make_ptr(vtcm_A, A_src), nb01, nb01, ne01);
                    dma_queue_pop(vtcm_dma);
                    dma_queue_push_ddr_to_vtcm(vtcm_dma,
                        dma_make_ptr(vtcm_B, B_src), row_size, row_size, ne11);
                    dma_queue_pop(vtcm_dma);
                    s_params.A = vtcm_A;
                    s_params.B = vtcm_B;
                } else {
                    s_params.A = A_src;
                    s_params.B = B_src;
                }
            }
            s_params.C = (char *)dst->data + i12 * nb2 + i13 * nb3;

            if (n_threads <= 1) {
                struct ggmldsp_compute_params cparams = {0, 1};
                ggmldsp_llamafile_sgemm(&cparams, &s_params);
            } else {
                sgemm_thread_data_t thread_data[MAX_NUM_WORKERS];
                worker_synctoken_t synctoken;
                worker_pool_synctoken_init(&synctoken, n_threads - 1);

                for (unsigned int t = 0; t < n_threads; t++) {
                    thread_data[t].cparams.ith = t;
                    thread_data[t].cparams.nth = n_threads;
                    thread_data[t].s_params = s_params;
                    thread_data[t].synctoken = (t == 0) ? NULL : &synctoken;

                    if (t == 0) {
                        sgemm_thread_func(&thread_data[t]);
                    } else {
                        worker_pool_job_t job;
                        job.fptr = sgemm_thread_func;
                        job.dptr = &thread_data[t];
                        worker_pool_submit(NULL, job);
                    }
                }

                worker_pool_synctoken_wait(&synctoken);
            }
        }
    }

    if (vtcm_dma) {
        dma_queue_flush(vtcm_dma);
        dma_queue_delete(vtcm_dma);
    }

    return 0;

fallback:
    if (ggmlop_get_thread_counts() > 1) {
        return ggmlop_dsp_mulmat_multithread_vtcm(h, src0, src1, dst);
    } else {
        return ggmlop_dsp_mulmat_singlethread(h, src0, src1, dst);
    }
}

// mulmat dispatch table: maps algotype to implementation function + description
typedef int (*mulmat_fn_t)(remote_handle64, const struct dsptensor *, const struct dsptensor *, dsptensor *);

struct mulmat_dispatch_entry {
    int          algotype;
    mulmat_fn_t  fn;
    const char * desc;
    int          log_use_hvx;
};

static const struct mulmat_dispatch_entry mulmat_dispatch_table[] = {
    { 30, ggmlop_dsp_mulmat_hmx,              "HMX x4x2 mode", 1 },
    { 32, ggmlop_dsp_mulmat_hmx,              "HMX mode",      1 },
    { 31, ggmlop_dsp_mulmat_sgemm,            "sgemm mode",    0 },
    { 33, ggmlop_dsp_mulmat_multithread_vtcm, "MT_VTCM mode",  0 },
};
#define MULMAT_DISPATCH_TABLE_SIZE (sizeof(mulmat_dispatch_table) / sizeof(mulmat_dispatch_table[0]))

int ggmlop_dsp_mulmat(remote_handle64 h, const struct dsptensor * src0, const struct dsptensor * src1, dsptensor * dst) {
    char tempbuf[256];
    int  mulmat_algo = ggmlop_get_mulmat_algotype();
    ggml_get_opkey(GGML_OP_MUL_MAT, src0, src1, tempbuf, 256);
    int64_t begin_time = ggml_time_us();

    mulmat_fn_t fn = NULL;
    const char * desc = NULL;
    int log_use_hvx = 0;

    for (int i = 0; i < (int)MULMAT_DISPATCH_TABLE_SIZE; i++) {
        if (mulmat_dispatch_table[i].algotype == mulmat_algo) {
            fn = mulmat_dispatch_table[i].fn;
            desc = mulmat_dispatch_table[i].desc;
            log_use_hvx = mulmat_dispatch_table[i].log_use_hvx;
            break;
        }
    }

    // algotype=0 (default): dispatch based on thread count
    if (fn == NULL) {
        if (ggmlop_get_thread_counts() > 1) {
            fn = ggmlop_dsp_mulmat_multithread;
            desc = "MT_HVX mode";
        } else {
            fn = ggmlop_dsp_mulmat_singlethread;
            desc = "singlethread mode";
        }
    }

    if (log_use_hvx) {
        GGMLHEXAGON_LOG_INFO("mulmat using %s(ggml_dsp_use_hvx=%d)", desc, ggml_get_dsp_use_hvx());
    } else {
        GGMLHEXAGON_LOG_INFO("mulmat using %s", desc);
    }

    int ret = fn(h, src0, src1, dst);

    int64_t end_time = ggml_time_us();
    GGMLHEXAGON_LOG_INFO("elapse time of %s is %lld us", tempbuf, (long long)(end_time - begin_time));
    GGMLHEXAGON_LOG_DEBUG("leave %s\n", __func__);
    return ret;
}
