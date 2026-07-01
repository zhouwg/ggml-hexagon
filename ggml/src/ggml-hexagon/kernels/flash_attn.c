/*
 * flash_attn.c - HVX Flash Attention (Phase 2: HVX-vectorized inner loop)
 *
 * Ported from htp/flash-attn-ops.c into the JZ kernels/ framework.
 *
 * The dispatch skeleton (per-Q-row driver, multi-threading via worker_pool)
 * is unchanged from the Phase 1 scalar version. The hot inner loop now runs
 * a vectorized path:
 *   - One Q row in VTCM (fp16; fp32 Q is converted in-place on the way in)
 *   - For each K block of FA_BLOCK_SIZE rows:
 *       - DMA K block and V block from DDR to per-thread VTCM scratch
 *       - Compute 32 QK scores per HVX vector via hvx_dot_f16_f16_aa_rx32
 *       - Apply mask / logit-softcap / ALiBi per lane
 *       - Online softmax: max + exp + sum, rescale VKQ32
 *       - P * V accumulation via hvx_mad_f32_f16_aa(_rx2)
 *   - Normalize by 1/S and write back to dst
 *
 * If the VTCM pool is too small for the per-thread scratch, the operator
 * falls back to the original scalar reference path so behaviour is
 * preserved on resource-constrained devices.
 */

#include "ggml-dsp.h"
#include "worker_pool.h"

#include <HAP_farf.h>
#include <HAP_perf.h>
#include <hexagon_types.h>
#include <math.h>
#include <string.h>

#include "../htp/hex-dma.h"
#include "../htp/hex-fastdiv.h"
#include "../htp/hex-utils.h"
#include "../htp/hvx-base.h"
#include "../htp/hvx-reduce.h"
#include "../htp/hvx-copy.h"
#include "../htp/hvx-scale.h"
#include "../htp/hvx-flash-attn.h"
#include "../htp/hvx-sigmoid.h"

#define FA_BLOCK_SIZE  (32 * 2)
#define FA_MAX_HEADS   512

#if __HVX_ARCH__ < 79
#define FA_OP_ADD_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#define FA_OP_SUB_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b))
#define FA_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#else
#define FA_OP_ADD_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#define FA_OP_SUB_F32(a, b) Q6_Vsf_vsub_VsfVsf(a, b)
#define FA_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#endif

// f32 -> f16 with explicit store; htp's compiler-stalling variant.
static __attribute__((noinline)) void fa_hvx_f32_to_f16_a(void *ptr, HVX_Vector v0, HVX_Vector v1)
{
    *(HVX_Vector *) ptr = hvx_vec_f32_to_f16(v0, v1);
}

// Q (fp16) dot K (fp16) -> single fp32 sum, scaled by s.
static inline void fa_hvx_dot_f16_f16_aa(float * restrict r,
                                        const void * restrict x,
                                        const void * restrict y,
                                        unsigned int n, float s) {
    const HVX_Vector * restrict vx = (const HVX_Vector * restrict) x;
    const HVX_Vector * restrict vy = (const HVX_Vector * restrict) y;

    uint32_t nvec = n / VLEN_FP16;
    uint32_t nloe = n % VLEN_FP16;

    HVX_VectorPair rsum_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    for (uint32_t i = 0; i < nvec; ++i) {
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, vx[i], vy[i]);
    }
    if (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p,
                                        Q6_V_vand_QV(bmask, vx[nvec]),
                                        Q6_V_vand_QV(bmask, vy[nvec]));
    }

    HVX_Vector rsum = FA_OP_ADD_F32(Q6_V_lo_W(rsum_p), Q6_V_hi_W(rsum_p));
    rsum = FA_OP_MUL_F32(hvx_vec_splat_f32(s), hvx_vec_reduce_sum_f32(rsum));
    hvx_vec_store_u(r, 4, rsum);
}

// 4x unrolled dot product: y is one K row in VTCM, x is 4 strided K rows in VTCM.
// Returns 4-lane HVX_Vector with 4 dot products of (Q . x[i]) * s.
static inline HVX_Vector fa_hvx_dot_f16_f16_aa_rx4(const void * restrict y,
                                                   const uint8_t * restrict x,
                                                   const size_t stride_x,
                                                   const size_t nvec,
                                                   const size_t nloe) {
    const HVX_Vector * restrict vx0 = (const HVX_Vector * restrict) x;
    const HVX_Vector * restrict vx1 = (const HVX_Vector * restrict) (x + stride_x);
    const HVX_Vector * restrict vx2 = (const HVX_Vector * restrict) (x + stride_x * 2);
    const HVX_Vector * restrict vx3 = (const HVX_Vector * restrict) (x + stride_x * 3);
    const HVX_Vector * restrict vy  = (const HVX_Vector * restrict) y;

    HVX_VectorPair rsum0_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_VectorPair rsum1_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_VectorPair rsum2_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_VectorPair rsum3_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));

    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector y_v  = vy[i];
        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, vx0[i], y_v);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, vx1[i], y_v);
        rsum2_p = hvx_vec_mpyacc_f32_f16(rsum2_p, vx2[i], y_v);
        rsum3_p = hvx_vec_mpyacc_f32_f16(rsum3_p, vx3[i], y_v);
    }
    if (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector y_v  = Q6_V_vand_QV(bmask, vy[nvec]);
        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, Q6_V_vand_QV(bmask, vx0[nvec]), y_v);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, Q6_V_vand_QV(bmask, vx1[nvec]), y_v);
        rsum2_p = hvx_vec_mpyacc_f32_f16(rsum2_p, Q6_V_vand_QV(bmask, vx2[nvec]), y_v);
        rsum3_p = hvx_vec_mpyacc_f32_f16(rsum3_p, Q6_V_vand_QV(bmask, vx3[nvec]), y_v);
    }

    HVX_Vector rsum0 = FA_OP_ADD_F32(Q6_V_lo_W(rsum0_p), Q6_V_hi_W(rsum0_p));
    HVX_Vector rsum1 = FA_OP_ADD_F32(Q6_V_lo_W(rsum1_p), Q6_V_hi_W(rsum1_p));
    HVX_Vector rsum2 = FA_OP_ADD_F32(Q6_V_lo_W(rsum2_p), Q6_V_hi_W(rsum2_p));
    HVX_Vector rsum3 = FA_OP_ADD_F32(Q6_V_lo_W(rsum3_p), Q6_V_hi_W(rsum3_p));

    HVX_Vector_x4 rsum0123 = { .v = { rsum0, rsum1, rsum2, rsum3 } };
    return hvx_vec_reduce_sum_f32x4(rsum0123);
}

// 32-row dot product: y is one Q row, x is 32 strided K rows in VTCM.
// Returns 32-lane HVX_Vector of scaled QK scores.
static inline HVX_Vector fa_hvx_dot_f16_f16_aa_rx32(const void * restrict y,
                                                    const uint8_t * restrict x,
                                                    const size_t stride_x,
                                                    const size_t n,
                                                    float s) {
    const size_t nvec = n / VLEN_FP16;
    const size_t nloe = n % VLEN_FP16;

    HVX_Vector sums = Q6_V_vzero();
    const size_t stride_x_4 = stride_x * 4;
    for (uint32_t j = 0; j < VLEN_FP32; j += 4) {
        HVX_Vector sums_x4 = fa_hvx_dot_f16_f16_aa_rx4(y, x, stride_x, nvec, nloe);
        HVX_VectorPred pred = Q6_Q_vsetq_R(j * SIZEOF_FP32);
        sums = Q6_V_vmux_QVV(pred, sums, sums_x4);
        x += stride_x_4;
    }
    return FA_OP_MUL_F32(hvx_vec_splat_f32(s), sums);
}

// y (F32) += x (F16) * s (F16)
static inline void fa_hvx_mad_f32_f16_aa(float * restrict y, const void * restrict x,
                                         const __fp16 * restrict s, int n) {
    const HVX_Vector * restrict vx0 = (const HVX_Vector *) x;
    HVX_VectorPair * restrict vy_p = (HVX_VectorPair *) y;
    HVX_Vector * restrict vy = (HVX_Vector *) y;

    uint32_t nvec = n / VLEN_FP16;
    uint32_t nloe = n % VLEN_FP16;

    HVX_Vector S0 = hvx_vec_splat_f16(*s);
    for (uint32_t i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
    }
    if (nloe) {
        HVX_VectorPair xy_p = vy_p[nvec];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[nvec]), S0);
        HVX_Vector xy = Q6_V_lo_W(xy_p);
        uint32_t i = 2 * nvec;

        if (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32;
            ++i;
            xy = Q6_V_hi_W(xy_p);
        }
        if (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

// y (F32) += x0 (F16) * s0 (F16) + x1 (F16) * s1 (F16)
static inline void fa_hvx_mad_f32_f16_aa_rx2(float * restrict y, const void * restrict x0, const void * restrict x1,
                                             const __fp16 * restrict s0, const __fp16 * restrict s1, int n) {
    const HVX_Vector * restrict vx0 = (const HVX_Vector *) x0;
    const HVX_Vector * restrict vx1 = (const HVX_Vector *) x1;
    HVX_VectorPair * restrict vy_p = (HVX_VectorPair *) y;
    HVX_Vector * restrict vy = (HVX_Vector *) y;

    uint32_t nvec = n / VLEN_FP16;
    uint32_t nloe = n % VLEN_FP16;

    HVX_Vector S0 = hvx_vec_splat_f16(*s0);
    HVX_Vector S1 = hvx_vec_splat_f16(*s1);
    for (uint32_t i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx1[i]), S1);
    }
    if (nloe) {
        HVX_VectorPair xy_p = vy_p[nvec];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[nvec]), S0);
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx1[nvec]), S1);
        HVX_Vector xy = Q6_V_lo_W(xy_p);
        uint32_t i = 2 * nvec;

        if (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32;
            ++i;
            xy = Q6_V_hi_W(xy_p);
        }
        if (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

// y (F32) *= vs (broadcast scalar)
static inline void fa_hvx_scale_vec_f32_aa(uint8_t * restrict dst,
                                           const uint8_t * restrict src,
                                           const int n, HVX_Vector vs) {
    const HVX_Vector * restrict vsrc = (const HVX_Vector * restrict) src;
    HVX_Vector * restrict vdst = (HVX_Vector * restrict) dst;

    const uint32_t nvec = n / VLEN_FP32;
    const uint32_t nloe = n % VLEN_FP32;
    for (uint32_t i = 0; i < nvec; ++i) {
        vdst[i] = FA_OP_MUL_F32(vsrc[i], vs);
    }
    if (nloe) {
        hvx_vec_store_a(&vdst[nvec], nloe * sizeof(float),
                        FA_OP_MUL_F32(vsrc[nvec], vs));
    }
}

// Block-stride for the online-softmax scan over K rows.
struct fa_thread {
    const dsptensor * q;
    const dsptensor * k;
    const dsptensor * v;
    const dsptensor * mask;
    dsptensor       * dst;
    const int32_t  * op_params;

    uint32_t neq1, neq2, neq3;
    uint32_t nek1, nek2, nek3;
    uint32_t nev1, nev2, nev3;
    uint32_t nbq1, nbq2, nbq3;
    uint32_t nbk1, nbk2, nbk3;
    uint32_t nbv1, nbv2, nbv3;
    uint32_t ne1, ne2, ne3;
    uint32_t nb1, nb2, nb3;

    uint32_t n_rows;
    uint32_t row_lo;
    uint32_t row_hi;

    float    scale;
    float    max_bias;
    float    logit_softcap;
    uint32_t n_head;
    uint32_t n_head_log2;
    float    m0;
    float    m1;
    float    slopes[FA_MAX_HEADS];

    int      is_q_fp32;

    // Per-op scratch (DDR). Pre-allocated by the driver and shared by all
    // workers. They are NOT thread-safe on a per-row basis: each row is
    // assigned to exactly one worker, so the writer pattern is safe as long
    // as we never call ggmlop_get_work_data from inside fa_row_*.
    void   * scratch;     // holds qk_buf + out_buf
    float  * qk_buf;      // size = nek1
    float  * out_buf;     // size = v->ne[0]
    uint32_t scratch_ne[2];
};

// Per-thread VTCM scratch (one slice per worker). Allocated by the driver
// from ggmlop_get_vtcm_pool and passed down via fa_thread_ctx.
struct fa_thread_ctx {
    struct fa_thread * t;
    uint8_t * spad_q;     // one Q row (fp16, padded)
    uint8_t * spad_k;     // FA_BLOCK_SIZE K rows (fp16, padded)
    uint8_t * spad_v;     // FA_BLOCK_SIZE V rows (fp16, padded)
    uint8_t * spad_m;     // FA_BLOCK_SIZE mask lanes (fp16, padded)
    float   * spad_a;     // DV fp32 output accumulator
    HVX_Vector * spad_s;  // FA_BLOCK_SIZE / VLEN_FP32 scratch for QK scores
    dma_queue * dma;
    size_t size_q_padded;
    size_t size_k_padded;
    size_t size_v_padded;
    size_t size_m_padded;
    // Scalar-path scratch (DDR). Each worker has its own slice; assigned
    // by the driver to avoid concurrent reads/writes of out[d] across
    // workers computing different rows.
    void   * scratch;
    float  * qk_buf;      // size = nek1
    float  * out_buf;     // size = v->ne[0]
};

// Scalar reference path: used when VTCM is not available.
static void fa_row_scalar(struct fa_thread_ctx * ctx, uint32_t iq1, uint32_t iq2, uint32_t iq3) {
    struct fa_thread * t = ctx->t;
    const uint32_t DK = t->k->ne[0];
    const uint32_t DV = t->v->ne[0];

    const uint32_t ik2 = (t->neq2 && t->k->ne[2]) ? iq2 / (t->neq2 / t->k->ne[2]) : 0;
    const uint32_t ik3 = (t->neq3 && t->k->ne[3]) ? iq3 / (t->neq3 / t->k->ne[3]) : 0;
    const uint32_t iv2 = (t->neq2 && t->v->ne[2]) ? iq2 / (t->neq2 / t->v->ne[2]) : 0;
    const uint32_t iv3 = (t->neq3 && t->v->ne[3]) ? iq3 / (t->neq3 / t->v->ne[3]) : 0;

    const uint8_t * q_row = (const uint8_t *) t->q->data + iq1 * t->nbq1 + iq2 * t->nbq2 + iq3 * t->nbq3;
    const float slope = t->max_bias > 0.0f ? t->slopes[iq2] : 0.0f;
    GGMLHEXAGON_LOG_INFO("flash_attn scalar trace: iq1=%u iq2=%u iq3=%u ik2=%u iv2=%u q_row=0x%x nbq1=%u nbq2=%u nbk1=%u nbk2=%u nbv1=%u nbv2=%u",
                         iq1, iq2, iq3, ik2, iv2, (unsigned)((const uint8_t *)q_row - (const uint8_t *)t->q->data),
                         t->nbq1, t->nbq2, t->nbk1, t->nbk2, t->nbv1, t->nbv2);

    // Scratch is pre-allocated by the driver and stored per-worker in ctx.
    // Never call ggmlop_get_work_data from here, it is not thread-safe.
    float * qk = ctx->qk_buf;
    float * out = ctx->out_buf;
    if (!qk || !out) return;
    memset(out, 0, DV * sizeof(float));

    float m_old = -INFINITY;
    float l_old = 0.0f;

    const uint32_t n_blocks = (t->nek1 + FA_BLOCK_SIZE - 1) / FA_BLOCK_SIZE;
    int printed_first_block = 0;
    for (uint32_t ib = 0; ib < n_blocks; ++ib) {
        const uint32_t ic0 = ib * FA_BLOCK_SIZE;
        const uint32_t icN = MIN(ic0 + FA_BLOCK_SIZE, t->nek1);

        for (uint32_t j = ic0; j < icN; ++j) {
            const uint8_t * k_row = (const uint8_t *) t->k->data + j * t->nbk1 + ik2 * t->nbk2 + ik3 * t->nbk3;
            float s = 0.0f;
            if (t->is_q_fp32) {
                const float * qf = (const float *) q_row;
                const __fp16 * kh = (const __fp16 *) k_row;
                for (uint32_t d = 0; d < DK; ++d) s += qf[d] * (float) kh[d];
            } else {
                const __fp16 * qh = (const __fp16 *) q_row;
                const __fp16 * kh = (const __fp16 *) k_row;
                for (uint32_t d = 0; d < DK; ++d) s += (float) qh[d] * (float) kh[d];
            }
            s *= t->scale;
            if (t->logit_softcap != 0.0f) {
                s = tanhf(s * t->logit_softcap) / t->logit_softcap;
            }
            if (t->max_bias > 0.0f) {
                s += -(float)(int32_t)(j) * slope;
            }
            if (t->mask) {
                const uint32_t im2 = (t->mask->ne[2] > 0) ? (iq2 % t->mask->ne[2]) : 0;
                const uint32_t im3 = (t->mask->ne[3] > 0) ? (iq3 % t->mask->ne[3]) : 0;
                const __fp16 * mp = (const __fp16 *) ((const uint8_t *) t->mask->data +
                                                     iq1 * t->mask->nb[1] +
                                                     im2 * t->mask->nb[2] +
                                                     im3 * t->mask->nb[3]);
                const __fp16 mj = mp[j];
                s += (float) mj;
            }
            qk[j] = s;
        }

        float m_block = -INFINITY;
        for (uint32_t j = ic0; j < icN; ++j) {
            if (qk[j] > m_block) m_block = qk[j];
        }
        const float m_new = MAX(m_old, m_block);
        const float alpha = expf(m_old - m_new);

        if (!printed_first_block && iq2 == 0) {
            printed_first_block = 1;
            GGMLHEXAGON_LOG_INFO("flash_attn scalar: iq1=%u qk[0..3]={%.4f %.4f %.4f %.4f} qk_max=%.4f m_new=%.4f",
                                 iq1,
                                 (double)qk[0], (double)qk[1], (double)qk[2], (double)qk[3],
                                 (double)m_new);
        }

        float l_block = 0.0f;
        for (uint32_t j = ic0; j < icN; ++j) {
            const float p = expf(qk[j] - m_new);
            qk[j] = p;
            l_block += p;
        }
        const float l_new = l_old * alpha + l_block;

        for (uint32_t d = 0; d < DV; ++d) out[d] *= alpha;
        for (uint32_t j = ic0; j < icN; ++j) {
            const uint8_t * v_row = (const uint8_t *) t->v->data + j * t->nbv1 + iv2 * t->nbv2 + iv3 * t->nbv3;
            const __fp16 * vh = (const __fp16 *) v_row;
            const float pj = qk[j];
            for (uint32_t d = 0; d < DV; ++d) {
                out[d] += pj * (float) vh[d];
            }
        }

        m_old = m_new;
        l_old = l_new;
    }

    const float inv_l = (l_old > 0.0f) ? (1.0f / l_old) : 0.0f;
    // dst->ne = [DV, neq2, neq1, neq3] (logical permute(0,2,1,3) of natural
    // [DV, neq1, neq2, neq3]). Memory layout is unchanged from natural, so
    // dst[d, iq2, iq1, iq3] lives at offset iq3*nb3 + iq1*nb2 + iq2*nb1.
    const uint8_t * dst_base = (const uint8_t *) t->dst->data + iq2 * t->nb1 + iq1 * t->nb2 + iq3 * t->nb3;
    if (t->dst->type == GGML_TYPE_F32) {
        float * dst_row = (float *) dst_base;
        for (uint32_t d = 0; d < DV; ++d) dst_row[d] = out[d] * inv_l;
        GGMLHEXAGON_LOG_INFO("flash_attn scalar: iq1=%u iq2=%u iq3=%u inv_l=%.6f out[0..3]={%.4f %.4f %.4f %.4f} dst[0..3]={%.4f %.4f %.4f %.4f}",
                             iq1, iq2, iq3, (double)inv_l,
                             (double)out[0], (double)out[1], (double)out[2], (double)out[3],
                             (double)dst_row[0], (double)dst_row[1], (double)dst_row[2], (double)dst_row[3]);
    } else if (t->dst->type == GGML_TYPE_F16) {
        __fp16 * dst_row = (__fp16 *) dst_base;
        for (uint32_t d = 0; d < DV; ++d) dst_row[d] = (__fp16)(out[d] * inv_l);
        GGMLHEXAGON_LOG_INFO("flash_attn scalar: iq1=%u iq2=%u iq3=%u inv_l=%.6f out[0..3]={%.4f %.4f %.4f %.4f} dst16[0..3]={%.4f %.4f %.4f %.4f}",
                             iq1, iq2, iq3, (double)inv_l,
                             (double)out[0], (double)out[1], (double)out[2], (double)out[3],
                             (double)dst_row[0], (double)dst_row[1], (double)dst_row[2], (double)dst_row[3]);
    } else {
        GGMLHEXAGON_LOG_ERROR("flash_attn: unsupported dst type %d", t->dst->type);
    }
}

// Per-thread VTCM scratch (one slice per worker). Allocated by the driver
// from ggmlop_get_vtcm_pool and passed down via fa_thread_ctx.
// (struct fa_thread_ctx is defined earlier in this file)

// HVX-vectorized path. Each Q row runs entirely in VTCM. K and V blocks are
// pulled from DDR via synchronous DMA per block.
static void fa_row_hvx(struct fa_thread_ctx * c, uint32_t iq1, uint32_t iq2, uint32_t iq3) {
    struct fa_thread * t = c->t;
    const uint32_t DK = t->k->ne[0];
    const uint32_t DV = t->v->ne[0];

    const uint32_t ik2 = (t->neq2 && t->k->ne[2]) ? iq2 / (t->neq2 / t->k->ne[2]) : 0;
    const uint32_t ik3 = (t->neq3 && t->k->ne[3]) ? iq3 / (t->neq3 / t->k->ne[3]) : 0;
    const uint32_t iv2 = (t->neq2 && t->v->ne[2]) ? iq2 / (t->neq2 / t->v->ne[2]) : 0;
    const uint32_t iv3 = (t->neq3 && t->v->ne[3]) ? iq3 / (t->neq3 / t->v->ne[3]) : 0;

    // 1. Fetch Q row into VTCM and normalize to fp16.
    //    Do not stage through ggmlop_get_work_data: that global is not
    //    thread-safe and concurrent worker calls can free the buffer out
    //    from under the others. Q.data lives in ION which the DMA engine
    //    can read directly.
    const uint8_t * q_row_ptr = (const uint8_t *) t->q->data + iq1 * t->nbq1 + iq2 * t->nbq2 + iq3 * t->nbq3;
    if (t->is_q_fp32) {
        dma_queue_push_ddr_to_vtcm(c->dma,
                                   dma_make_ptr(c->spad_q, (void *) q_row_ptr),
                                   c->size_q_padded,
                                   (size_t) DK * sizeof(float),
                                   1);
        dma_queue_pop(c->dma);
        hvx_copy_f16_f32_aa(c->spad_q, c->spad_q, DK);
    } else {
        dma_queue_push_ddr_to_vtcm(c->dma,
                                   dma_make_ptr(c->spad_q, q_row_ptr),
                                   c->size_q_padded,
                                   (size_t) DK * sizeof(__fp16),
                                   1);
        dma_queue_pop(c->dma);
    }

    const float slope = t->max_bias > 0.0f ? t->slopes[iq2] : 0.0f;
    const HVX_Vector logit_cap = hvx_vec_splat_f32(t->logit_softcap);

    // 2. Online softmax accumulators (broadcast across all DV lanes).
    HVX_Vector S_vec = hvx_vec_splat_f32(0.0f);
    HVX_Vector M_vec = hvx_vec_splat_f32(-INFINITY);

    // Clear VKQ32 accumulator.
    hvx_splat_f32_a(c->spad_a, 0, DV);
    float * VKQ32 = (float *) c->spad_a;

    // Mask base for this Q row.
    const __fp16 * mp_base = NULL;
    if (t->mask) {
        const uint32_t im2 = (t->mask->ne[2] > 0) ? (iq2 % t->mask->ne[2]) : 0;
        const uint32_t im3 = (t->mask->ne[3] > 0) ? (iq3 % t->mask->ne[3]) : 0;
        mp_base = (const __fp16 *) ((const uint8_t *) t->mask->data +
                                    iq1 * t->mask->nb[1] +
                                    im2 * t->mask->nb[2] +
                                    im3 * t->mask->nb[3]);
    }

    // 3. Process K in blocks of FA_BLOCK_SIZE rows.
    const uint32_t n_blocks = (t->nek1 + FA_BLOCK_SIZE - 1) / FA_BLOCK_SIZE;
    for (uint32_t ib = 0; ib < n_blocks; ++ib) {
        const uint32_t ic_start = ib * FA_BLOCK_SIZE;
        const uint32_t block_size = MIN(FA_BLOCK_SIZE, t->nek1 - ic_start);

        // DMA K block (contiguous rows from DDR).
        const uint8_t * k_src = (const uint8_t *) t->k->data + (ic_start * t->nbk1 + ik2 * t->nbk2 + ik3 * t->nbk3);
        dma_queue_push_ddr_to_vtcm(c->dma,
                                   dma_make_ptr(c->spad_k, k_src),
                                   c->size_k_padded,
                                   t->nbk1,
                                   block_size);
        dma_queue_pop(c->dma);

        // DMA V block.
        const uint8_t * v_src = (const uint8_t *) t->v->data + (ic_start * t->nbv1 + iv2 * t->nbv2 + iv3 * t->nbv3);
        dma_queue_push_ddr_to_vtcm(c->dma,
                                   dma_make_ptr(c->spad_v, v_src),
                                   c->size_v_padded,
                                   t->nbv1,
                                   block_size);
        dma_queue_pop(c->dma);

        // Zero-pad the unused tail rows of K and V blocks. The HVX inner loop
        // always processes FA_BLOCK_SIZE lanes (via 32-lane HVX vectors); if
        // block_size is not a multiple of 32, the last HVX vector would read
        // uninitialized scratch for the invalid lanes, contaminating the dot
        // product and the final output. For full blocks this is a no-op.
        if (block_size < FA_BLOCK_SIZE) {
            const uint32_t tail_rows = FA_BLOCK_SIZE - block_size;
            const size_t tail_k_bytes = (size_t) tail_rows * c->size_k_padded;
            const size_t tail_v_bytes = (size_t) tail_rows * c->size_v_padded;
            memset(c->spad_k + (size_t) block_size * c->size_k_padded, 0, tail_k_bytes);
            memset(c->spad_v + (size_t) block_size * c->size_v_padded, 0, tail_v_bytes);
        }

        // Pre-DMA mask block into VTCM (cache mode: HVX can then read it
        // safely regardless of source alignment). Mask is contiguous in K
        // (mask->nb[0] == sizeof(__fp16)) so a single 1D DMA suffices.
        if (t->mask) {
            const uint8_t * m_src = (const uint8_t *) (mp_base + ic_start);
            dma_queue_push_ddr_to_vtcm(c->dma,
                                       dma_make_ptr(c->spad_m, m_src),
                                       c->size_m_padded,
                                       block_size * sizeof(__fp16),
                                       1);
            dma_queue_pop(c->dma);
        }

        // 3a. Compute QK scores for each 32-lane sub-block within this K block.
        HVX_Vector sb_scores[FA_BLOCK_SIZE / VLEN_FP32];
        HVX_Vector v_max = hvx_vec_splat_f32(-INFINITY);
        uint32_t ic = 0;
        for (uint32_t iv = 0; ic < block_size; ic += VLEN_FP32, ++iv) {
            HVX_Vector scores = fa_hvx_dot_f16_f16_aa_rx32(c->spad_q,
                                                           c->spad_k + ic * c->size_k_padded,
                                                           c->size_k_padded,
                                                           DK,
                                                           t->scale);
            if (t->logit_softcap != 0.0f) {
                scores = FA_OP_MUL_F32(hvx_vec_tanh_f32(scores), logit_cap);
            }
            if (t->mask) {
                const __fp16 * mp = (const __fp16 *) c->spad_m + ic;
                HVX_Vector m_vals_f16 = *(const HVX_UVector *) mp;
                // Mask out any lanes past the end of this K block (the VTCM
                // padding area is uninitialized). Invalid lanes are also
                // forced to -INFINITY below for the score itself.
                uint32_t valid_lanes_m = block_size - ic;
                if (valid_lanes_m < VLEN_FP32) {
                    HVX_VectorPred mp_pred = Q6_Q_vsetq_R(valid_lanes_m * 2); // fp16 lane = 2 bytes
                    m_vals_f16 = Q6_V_vmux_QVV(mp_pred, m_vals_f16, Q6_V_vzero());
                }
                // Clamp -INFINITY (0xFC00) to -65504.0f to avoid NaN in VhfVhf mpy on v79.
                HVX_Vector vinf = Q6_Vh_vsplat_R(0xFC00);
                HVX_Vector vmin = Q6_Vh_vsplat_R(0xFBFF);
                HVX_VectorPred is_inf = Q6_Q_vcmp_eq_VhVh(m_vals_f16, vinf);
                m_vals_f16 = Q6_V_vmux_QVV(is_inf, vmin, m_vals_f16);
                // Add raw mask (no slope scaling). ALiBi is applied separately below.
                // hvx_vec_f16_to_f32 already performs the Q6_Vh_vshuff_Vh internally,
                // so its Q6_V_lo_W(p) holds fp32 lane 0..31 (matching scores).
                HVX_VectorPair m_vals_f32_pair = hvx_vec_f16_to_f32(m_vals_f16);
                HVX_Vector add_val_lo = Q6_V_lo_W(m_vals_f32_pair);
                scores = FA_OP_ADD_F32(scores, add_val_lo);
            }
            if (t->max_bias > 0.0f) {
                // ALiBi: bias[j] = -j * slope, applied to each lane.
                static const float ramp_32[32] __attribute__((aligned(128))) = {
                    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
                    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                    16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,
                    24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f
                };
                HVX_Vector v_ramp = hvx_vmem(ramp_32);
                HVX_Vector j_base = hvx_vec_splat_f32((float) ic_start);
                HVX_Vector v_j = FA_OP_ADD_F32(j_base, v_ramp);
                HVX_Vector bias = FA_OP_MUL_F32(v_j, hvx_vec_splat_f32(-slope));
                scores = FA_OP_ADD_F32(scores, bias);
            }

            // Mask out invalid lanes for leftover handling.
            uint32_t valid_lanes = block_size - ic;
            if (valid_lanes < VLEN_FP32) {
                HVX_VectorPred valid_pred = Q6_Q_vsetq_R(valid_lanes * 4);
                scores = Q6_V_vmux_QVV(valid_pred, scores, hvx_vec_splat_f32(-INFINITY));
            }
            sb_scores[iv] = scores;
            v_max = hvx_vec_reduce_max2_f32(scores, v_max);
        }

        // 3b. Online softmax: M_new, alpha=exp(M_old - M_new), rescale VKQ32.
        HVX_Vector M_new_vec = HVX_VMAX_F32(v_max, M_vec);
        HVX_Vector diff_vec = FA_OP_SUB_F32(M_vec, M_new_vec);
        HVX_Vector ms_vec = hvx_vec_exp_f32(diff_vec);
        M_vec = M_new_vec;

        fa_hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (const uint8_t *) VKQ32, DV, ms_vec);

        // 3c. exp(s - M) into P, then accumulate p_j * V[j] using HVX MAD.
        HVX_Vector p_sum_vec = hvx_vec_splat_f32(0.0f);
        uint32_t ic2 = 0;
        for (uint32_t iv = 0; ic2 < block_size; ic2 += VLEN_FP32, ++iv) {
            HVX_Vector scores = sb_scores[iv];
            HVX_Vector scores_shifted = FA_OP_SUB_F32(scores, M_vec);
            HVX_Vector P = hvx_vec_exp_f32(scores_shifted);
            p_sum_vec = FA_OP_ADD_F32(p_sum_vec, P);

            // Convert P fp32 -> fp16 lane pairs for MAD.
            __fp16 __attribute__((aligned(VLEN))) p_arr[VLEN_FP16];
            fa_hvx_f32_to_f16_a(p_arr, P, hvx_vec_splat_f32(0));

            float __attribute__((aligned(128))) P_arr[VLEN_FP32];
            hvx_vec_store_a(P_arr, 128, P);

            for (uint32_t j = 0; j < VLEN_FP32; j += 2) {
                const uint32_t cur_ic = ic2 + j;
                if (cur_ic >= block_size) break;
                if (cur_ic + 1 == block_size) {
                    // Odd leftover, single row MAD.
                    if (P_arr[j] != 0.0f) {
                        const uint8_t * v_ptr = c->spad_v + cur_ic * c->size_v_padded;
                        fa_hvx_mad_f32_f16_aa(VKQ32, v_ptr, (p_arr + j), DV);
                    }
                    break;
                }
                if (P_arr[j] == 0.0f && P_arr[j + 1] == 0.0f) {
                    continue;
                }
                const uint8_t * v_ptr = c->spad_v + cur_ic * c->size_v_padded;
                fa_hvx_mad_f32_f16_aa_rx2(VKQ32, v_ptr, v_ptr + c->size_v_padded,
                                           (p_arr + j), (p_arr + j + 1), DV);
            }
        }

        p_sum_vec = hvx_vec_reduce_sum_f32(p_sum_vec);
        S_vec = FA_OP_ADD_F32(FA_OP_MUL_F32(S_vec, ms_vec), p_sum_vec);
    }

    // 4. Read S scalar and apply final normalization.
    const float S = hvx_vec_get_f32(S_vec);
    const float S_inv = (S == 0.0f) ? 0.0f : 1.0f / S;
    hvx_scale_f32_aa((uint8_t *) VKQ32, (const uint8_t *) VKQ32, DV, S_inv);

    // 5. Store to dst. dst logical shape = [DV, neq2, neq1, neq3], but the
    //    underlying memory layout is the natural [DV, neq1, neq2, neq3] from
    //    ggml_flash_attn_ext. So dst[d, iq2, iq1, iq3] is at offset
    //    iq3*nb3 + iq1*nb2 + iq2*nb1 (nb1 = DV*sizeof(float), nb2 = neq2*DV*4).
    const int i1 = iq1;
    const int i2 = iq2;
    const int i3 = iq3;
    uint8_t * dst_ptr = (uint8_t *) t->dst->data + i2 * t->nb1 + i1 * t->nb2 + i3 * t->nb3;
    if (t->dst->type == GGML_TYPE_F32) {
        // Aligned copy back through VTCM (unaligned dst).
        uint8_t * src_a = (uint8_t *) c->spad_a;
        for (uint32_t d = 0; d < DV; ++d) {
            ((float *) dst_ptr)[d] = ((float *) src_a)[d];
        }
    } else if (t->dst->type == GGML_TYPE_F16) {
        uint8_t * src_a = (uint8_t *) c->spad_a;
        for (uint32_t d = 0; d < DV; ++d) {
            ((__fp16 *) dst_ptr)[d] = (__fp16) ((float *) src_a)[d];
        }
    } else {
        GGMLHEXAGON_LOG_ERROR("flash_attn: unsupported dst type %d", t->dst->type);
    }
}

static void fa_thread_main(unsigned int nth, unsigned int ith, void * data) {
    // data is the fa_thread_ctx for this worker (one per worker).
    struct fa_thread_ctx * ctx = (struct fa_thread_ctx *) data;
    struct fa_thread * t = ctx->t;
    const uint32_t dr = (t->n_rows + nth - 1) / nth;
    const uint32_t ir0 = dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, t->n_rows);
    if (ir0 >= ir1) return;

    for (uint32_t ir = ir0; ir < ir1; ++ir) {
        const uint32_t iq3 = ir / (t->neq1 * t->neq2);
        const uint32_t iq2 = (ir - iq3 * t->neq1 * t->neq2) / t->neq1;
        const uint32_t iq1 = ir - iq3 * t->neq1 * t->neq2 - iq2 * t->neq1;
        if (ctx->spad_q != NULL) {
            fa_row_hvx(ctx, iq1, iq2, iq3);
        } else {
            fa_row_scalar(ctx, iq1, iq2, iq3);
        }
    }
}

// Per-worker dispatch record for worker_pool_submit.
struct fa_dispatch {
    unsigned int         nth;
    unsigned int         ith;
    struct fa_thread_ctx * ctx;
    worker_synctoken_t   * token;  // NULL for ith == 0 (main thread)
};

static void fa_dispatch_trampoline(void * p) {
    struct fa_dispatch * d = (struct fa_dispatch *) p;
    fa_thread_main(d->nth, d->ith, d->ctx);
    if (d->token) {
        worker_pool_synctoken_jobdone(d->token);
    }
}

int ggmlop_dsp_flash_attn(remote_handle64 h,
                          const dsptensor * q,
                          const dsptensor * k,
                          const dsptensor * v,
                          const dsptensor * mask,
                          dsptensor * dst) {
    (void) h;

    if (!q || !k || !v || !dst) {
        GGMLHEXAGON_LOG_ERROR("flash_attn: required tensor missing (q=%p k=%p v=%p dst=%p)",
                              q, k, v, dst);
        return AEE_EBADPARM;
    }
    GGMLHEXAGON_LOG_INFO("flash_attn: q[n0=%d,n1=%d,n2=%d,n3=%d,nb0=%d,nb1=%d,type=%d,data=%p] "
                          "k[n0=%d,n1=%d,n2=%d,n3=%d,nb0=%d,nb1=%d,type=%d,data=%p] "
                          "v[n0=%d,n1=%d,n2=%d,n3=%d,nb0=%d,nb1=%d,type=%d,data=%p] "
                          "dst[n0=%d,n1=%d,n2=%d,n3=%d,nb0=%d,nb1=%d,type=%d,data=%p] mask=%p",
                          q->ne[0], q->ne[1], q->ne[2], q->ne[3], q->nb[0], q->nb[1], q->type, q->data,
                          k->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[0], k->nb[1], k->type, k->data,
                          v->ne[0], v->ne[1], v->ne[2], v->ne[3], v->nb[0], v->nb[1], v->type, v->data,
                          dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], dst->nb[0], dst->nb[1], dst->type, dst->data,
                          mask);
    {
        const float * q0 = (const float *) q->data;
        const __fp16 * k0 = (const __fp16 *) k->data;
        const __fp16 * v0 = (const __fp16 *) v->data;
        GGMLHEXAGON_LOG_INFO("flash_attn: q0[0..3]={%.4f %.4f %.4f %.4f} k0[0..3]={%.4f %.4f %.4f %.4f} v0[0..3]={%.4f %.4f %.4f %.4f}",
                              (double)q0[0], (double)q0[1], (double)q0[2], (double)q0[3],
                              (double)k0[0], (double)k0[1], (double)k0[2], (double)k0[3],
                              (double)v0[0], (double)v0[1], (double)v0[2], (double)v0[3]);
    }
    if (q->ne[0] != k->ne[0]) {
        GGMLHEXAGON_LOG_ERROR("flash_attn: Q/K head_dim mismatch (%d vs %d)", (int)q->ne[0], (int)k->ne[0]);
        return AEE_EBADPARM;
    }
    if (k->ne[1] != v->ne[1] || k->ne[2] != v->ne[2] || k->ne[3] != v->ne[3]) {
        GGMLHEXAGON_LOG_ERROR("flash_attn: K/V shape mismatch");
        return AEE_EBADPARM;
    }
    if (q->type != GGML_TYPE_F16 && q->type != GGML_TYPE_F32) {
        GGMLHEXAGON_LOG_ERROR("flash_attn: Q must be f16 or f32 (got %d)", q->type);
        return AEE_EBADPARM;
    }
    if (k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) {
        GGMLHEXAGON_LOG_ERROR("flash_attn: K/V must be f16 (got K=%d V=%d)", k->type, v->type);
        return AEE_EBADPARM;
    }
    if (q->ne[2] > FA_MAX_HEADS) {
        GGMLHEXAGON_LOG_ERROR("flash_attn: too many heads (%d > %d)", (int)q->ne[2], FA_MAX_HEADS);
        return AEE_EUNSUPPORTED;
    }
    if (mask && mask->type != GGML_TYPE_F16) {
        GGMLHEXAGON_LOG_ERROR("flash_attn: mask must be f16 (got %d)", mask->type);
        return AEE_EBADPARM;
    }
    if (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32) {
        GGMLHEXAGON_LOG_ERROR("flash_attn: dst must be f16 or f32 (got %d)", dst->type);
        return AEE_EBADPARM;
    }

    // ---- 1. Set up shared per-op thread context ----
    static struct fa_thread t;
    memset(&t, 0, sizeof(t));
    t.q = q; t.k = k; t.v = v; t.mask = mask; t.dst = dst;
    t.op_params = dst->op_params;
    t.is_q_fp32 = (q->type == GGML_TYPE_F32);

    t.neq1 = q->ne[1]; t.neq2 = q->ne[2]; t.neq3 = q->ne[3];
    t.nek1 = k->ne[1]; t.nek2 = k->ne[2]; t.nek3 = k->ne[3];
    t.nev1 = v->ne[1]; t.nev2 = v->ne[2]; t.nev3 = v->ne[3];

    t.nbq1 = q->nb[1]; t.nbq2 = q->nb[2]; t.nbq3 = q->nb[3];
    t.nbk1 = k->nb[1]; t.nbk2 = k->nb[2]; t.nbk3 = k->nb[3];
    t.nbv1 = v->nb[1]; t.nbv2 = v->nb[2]; t.nbv3 = v->nb[3];
    t.ne1 = dst->ne[1]; t.ne2 = dst->ne[2]; t.ne3 = dst->ne[3];
    t.nb1 = dst->nb[1]; t.nb2 = dst->nb[2]; t.nb3 = dst->nb[3];

    t.n_rows = t.neq1 * t.neq2 * t.neq3;
    t.n_head = t.neq2;

    t.scale = 1.0f;
    t.max_bias = 0.0f;
    t.logit_softcap = 0.0f;
    if (t.op_params) {
        memcpy(&t.scale,         t.op_params + 0, sizeof(float));
        memcpy(&t.max_bias,      t.op_params + 1, sizeof(float));
        memcpy(&t.logit_softcap, t.op_params + 2, sizeof(float));
    }
    GGMLHEXAGON_LOG_INFO("flash_attn: scale=%.6f max_bias=%.6f logit_softcap=%.6f mask=%p t.op_params=%p",
                         (double)t.scale, (double)t.max_bias, (double)t.logit_softcap,
                         t.mask, (void *)t.op_params);
    if (t.logit_softcap != 0.0f) {
        t.scale /= t.logit_softcap;
    }

    t.n_head_log2 = 1u << (uint32_t) floorf(log2f((float) t.n_head));
    t.m0 = powf(2.0f, -(t.max_bias)         / (float) t.n_head_log2);
    t.m1 = powf(2.0f, -(t.max_bias / 2.0f)  / (float) t.n_head_log2);
    if (t.max_bias > 0.0f) {
        for (uint32_t h = 0; h < t.n_head; ++h) {
            t.slopes[h] = alibi_slope(h, t.n_head_log2, t.m0, t.m1);
        }
    }

    // ---- 2. Per-worker VTCM scratch and DMA queues ----
    unsigned int nth = (unsigned int) ggmlop_get_thread_counts();
    if (nth < 1) nth = 1;
    if (nth > MAX_NUM_WORKERS) nth = MAX_NUM_WORKERS;
    if ((uint32_t) nth > t.n_rows) nth = (unsigned int) t.n_rows;
    if (nth == 0) nth = 1;

    size_t pool_size = 0;
    uint8_t * vtcm_base = (uint8_t *) ggmlop_get_vtcm_pool(&pool_size);

    const uint32_t DK = k->ne[0];
    const uint32_t DV = v->ne[0];
    const size_t q_elem_bytes = (q->type == GGML_TYPE_F32) ? sizeof(float) : sizeof(__fp16);
    const size_t size_q_padded = hex_round_up((size_t) DK * q_elem_bytes, 128);
    const size_t size_k_padded = hex_round_up(DK * sizeof(__fp16), 128);
    const size_t size_v_padded = hex_round_up(DV * sizeof(__fp16), 128);
    const size_t size_a_padded = hex_round_up(DV * sizeof(float), 128);
    // Mask scratch is one block's worth of fp16 lanes (FA_BLOCK_SIZE = 64).
    const size_t size_m_padded = hex_round_up(FA_BLOCK_SIZE * sizeof(__fp16), 128);
    const size_t per_thread = size_q_padded
                            + FA_BLOCK_SIZE * size_k_padded
                            + FA_BLOCK_SIZE * size_v_padded
                            + size_m_padded
                            + size_a_padded;

    // Decide HVX vs scalar based on VTCM pool availability and per-thread budget.
    // VTCM is acquired at batch entry (per-batch, not per-op).
    int use_hvx = 0;
    if (vtcm_base != NULL && pool_size >= per_thread * nth) {
        use_hvx = 1;
    }
    if (!use_hvx) {
        // Fall back to scalar. Per-worker ctx is unused.
        static struct fa_thread_ctx ctxs[MAX_NUM_WORKERS];
        static struct fa_dispatch disp[MAX_NUM_WORKERS];
        static worker_synctoken_t token;
        worker_pool_synctoken_init(&token, nth - 1);

        const uint32_t DV = v->ne[0];
        const size_t qk_bytes   = (size_t) t.nek1 * sizeof(float);
        const size_t out_bytes  = (size_t) DV      * sizeof(float);
        const size_t per_worker = qk_bytes + out_bytes;
        const size_t total_scratch = per_worker * MAX_NUM_WORKERS;
        t.scratch     = NULL;
        t.qk_buf      = NULL;
        t.out_buf     = NULL;
        t.scratch_ne[0] = t.nek1;
        t.scratch_ne[1] = DV;
        if (per_worker == 0) {
            GGMLHEXAGON_LOG_ERROR("flash_attn: empty per-worker scratch");
            return AEE_EBADPARM;
        }
        t.scratch = memalign(128, total_scratch);
        if (t.scratch == NULL) {
            GGMLHEXAGON_LOG_ERROR("flash_attn: failed to allocate scratch (%zu bytes)", total_scratch);
            return AEE_ENOMEMORY;
        }

        for (unsigned i = 0; i < nth; ++i) {
            uint8_t * worker_base = (uint8_t *) t.scratch + i * per_worker;
            memset(&ctxs[i], 0, sizeof(ctxs[i]));
            ctxs[i].t       = &t;
            ctxs[i].scratch = worker_base;
            ctxs[i].qk_buf  = (float *) worker_base;
            ctxs[i].out_buf = (float *) (worker_base + qk_bytes);

            disp[i].nth   = nth;
            disp[i].ith   = i;
            disp[i].ctx   = &ctxs[i];
            disp[i].token = (i == 0) ? NULL : &token;
            if (i == 0) continue;
            worker_pool_job_t job = { fa_dispatch_trampoline, &disp[i] };
            if (worker_pool_submit(NULL, job) != AEE_SUCCESS) {
                fa_dispatch_trampoline(&disp[i]);
            }
        }
        fa_thread_main(nth, 0, &ctxs[0]);
        if (nth > 1) worker_pool_synctoken_wait(&token);
        if (t.scratch) { free(t.scratch); t.scratch = NULL; t.qk_buf = NULL; t.out_buf = NULL; }
        return AEE_SUCCESS;
    }

    // HVX path: split VTCM pool among workers, create per-worker DMA queues.
    size_t vtcm_per_thread = per_thread;
    // Round up to a nice power-of-2 so each thread's slice is 128-byte aligned.
    while (vtcm_per_thread & (vtcm_per_thread - 1)) vtcm_per_thread++;
    // Don't grow beyond what fits.
    while (vtcm_per_thread * 2 * nth <= pool_size) vtcm_per_thread *= 2;
    if (vtcm_per_thread < per_thread) vtcm_per_thread = per_thread;

    static struct fa_thread_ctx ctxs[MAX_NUM_WORKERS];
    static dma_queue * dma_queues[MAX_NUM_WORKERS];
    static struct fa_dispatch disp[MAX_NUM_WORKERS];
    static worker_synctoken_t token;
    worker_pool_synctoken_init(&token, nth - 1);

    for (unsigned i = 0; i < nth; ++i) {
        uint8_t * base = vtcm_base + i * vtcm_per_thread;
        ctxs[i].t = &t;
        ctxs[i].spad_q = base;
        ctxs[i].spad_k = base + size_q_padded;
        ctxs[i].spad_v = ctxs[i].spad_k + FA_BLOCK_SIZE * size_k_padded;
        ctxs[i].spad_m = ctxs[i].spad_v + FA_BLOCK_SIZE * size_v_padded;
        ctxs[i].spad_a = (float *) (ctxs[i].spad_m + size_m_padded);
        ctxs[i].spad_s = NULL;  // unused in current path
        ctxs[i].size_q_padded = size_q_padded;
        ctxs[i].size_k_padded = size_k_padded;
        ctxs[i].size_v_padded = size_v_padded;
        ctxs[i].size_m_padded = size_m_padded;
        if (dma_queues[i] == NULL) {
            dma_queues[i] = dma_queue_create(8);
        }
        ctxs[i].dma = dma_queues[i];

        disp[i].nth = nth;
        disp[i].ith = i;
        disp[i].ctx = &ctxs[i];
        disp[i].token = (i == 0) ? NULL : &token;
        if (i == 0) continue;
        worker_pool_job_t job = { fa_dispatch_trampoline, &disp[i] };
        if (worker_pool_submit(NULL, job) != AEE_SUCCESS) {
            fa_dispatch_trampoline(&disp[i]);
        }
    }
    fa_thread_main(nth, 0, &ctxs[0]);
    if (nth > 1) worker_pool_synctoken_wait(&token);

    for (unsigned i = 0; i < nth; ++i) {
        if (dma_queues[i] != NULL) {
            dma_queue_flush(dma_queues[i]);
            dma_queue_delete(dma_queues[i]);
            dma_queues[i] = NULL;
        }
    }

    return AEE_SUCCESS;
}
