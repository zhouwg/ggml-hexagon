#include "ggml-dsp.h"

// GET_ROWS: extract rows from quantized weight matrix by index.
// src0 = quantized weight (q4_K/q5_K), shape [ne00, ne01, ne02, ne03]
// src1 = row indices (i32),         shape [ne10, ne11=ne02, ne12=ne03]
// dst  = dequantized output (f32),  shape [ne00, ne10, ne11, ne12]
//
// For each index i in src1, copy & dequantize row src0[row_idx] into dst[:,i]

// ---- K-quant constants (must match ggml-common.h) ----

#define QK_K        256
#define K_SCALE_SIZE 12

// ---- fp16 conversion (reuse existing helper) ----

static inline float dsp_fp16_to_fp32(uint16_t h) {
    return ggml_compute_fp16_to_fp32(h);
}

// ---- scale/min unpacking for K-type quantization ----

static inline void get_scale_min_k4(int j, const uint8_t * q,
                                     uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// ---- Quantized block structures (DSP-local definitions) ----

typedef struct {
    uint16_t d;             // super-block scale (fp16)
    uint16_t dmin;          // super-block min (fp16)
    uint8_t scales[K_SCALE_SIZE]; // packed 6-bit scales/mins
    uint8_t qs[QK_K/2];     // 4-bit quants
} dsp_block_q4_K;

typedef struct {
    uint16_t d;             // super-block scale (fp16)
    uint16_t dmin;          // super-block min (fp16)
    uint8_t scales[K_SCALE_SIZE]; // packed 6-bit scales/mins
    uint8_t qh[QK_K/8];     // high bit for 5-bit quants
    uint8_t qs[QK_K/2];     // low 4-bit quants
} dsp_block_q5_K;

// ---- Dequantization functions ----

static void dequantize_row_q4_K(const dsp_block_q4_K * x, float * y, int k) {
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t * q  = x[i].qs;
        const float   d   = dsp_fp16_to_fp32(x[i].d);
        const float min  = dsp_fp16_to_fp32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

static void dequantize_row_q5_K(const dsp_block_q5_K * x, float * y, int k) {
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t * ql = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const float   d   = dsp_fp16_to_fp32(x[i].d);
        const float min  = dsp_fp16_to_fp32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l)
                *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l)
                *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

// ---- Public entry point (uses dsptensor like all other DSP ops) ----

int ggmlop_dsp_getrows(remote_handle64 h,
                       const dsptensor * src0,
                       const dsptensor * src1,
                       dsptensor * dst) {
    GGML_UNUSED(h);

    uint64_t begin_time = ggml_time_us();

    const int64_t ne00  = src0->ne[0];
    const int64_t ne01  = src0->ne[1];
    const int64_t ne10  = src1->ne[0];
    const int64_t ne11  = src1->ne[1];
    const int64_t ne12  = src1->ne[2];
    const size_t  nb01  = src0->nb[1];
    const size_t  nb02  = src0->nb[2];
    const size_t  nb03  = src0->nb[3];
    const size_t  nb10  = src1->nb[0];
    const size_t  nb11  = src1->nb[1];
    const size_t  nb12  = src1->nb[2];
    const size_t  nb1   = dst->nb[1];
    const size_t  nb2   = dst->nb[2];
    const size_t  nb3   = dst->nb[3];

    const int32_t type = src0->type;

    const int64_t nr = ne10 * ne11 * ne12;

    GGMLHEXAGON_LOG_DEBUG("GET_ROWS: src0 type=%d ne=[%ld,%ld,%ld,%ld] -> "
                          "dst f32 ne=[%ld,%ld,%ld,%ld], nr=%ld rows",
                          type,
                          (long)ne00, (long)ne01,
                          (long)src0->ne[2], (long)src0->ne[3],
                          (long)dst->ne[0], (long)dst->ne[1],
                          (long)dst->ne[2], (long)dst->ne[3],
                          (long)nr);

    for (int64_t i = 0; i < nr; ++i) {
        // decompose flat index i into (i10, i11, i12)
        const int64_t i12 = i / (ne11 * ne10);
        const int64_t i11 = (i - i12 * ne11 * ne10) / ne10;
        const int64_t i10 = i - i12 * ne11 * ne10 - i11 * ne10;

        // read row index from src1
        const int32_t row_idx = *(const int32_t *)(
            (const char *)src1->data + i10 * nb10 + i11 * nb11 + i12 * nb12);

        if (row_idx < 0 || row_idx >= ne01) {
            GGMLHEXAGON_LOG_ERROR("GET_ROWS: row_idx %d out of range [0, %ld]",
                                  row_idx, (long)ne01);
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
                dequantize_row_q4_K((const dsp_block_q4_K *)src_row, dst_row, (int)ne00);
                break;
            case GGML_TYPE_Q5_K:
                dequantize_row_q5_K((const dsp_block_q5_K *)src_row, dst_row, (int)ne00);
                break;
            default:
                GGMLHEXAGON_LOG_ERROR("GET_ROWS: unsupported type %d", type);
                memset(dst_row, 0, ne00 * sizeof(float));
                break;
        }
    }

    int64_t end_time = ggml_time_us();
    GGMLHEXAGON_LOG_INFO("elapse time of GET_ROWS is %lld us (%ld rows)",
                         (long long)(end_time - begin_time), (long)nr);
    return 0;
}
