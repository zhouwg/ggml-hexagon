#include "ggml-dsp.h"
#include "worker_pool.h"
#include <math.h>

// ============================================================
// ROPE - Rotary Position Embedding (F32 scalar implementation)
// Supports: GGML_ROPE_TYPE_NORMAL, GGML_ROPE_TYPE_NEOX
// Ternary op: src0(input f32), src1(positions i32), src2(freq_factors f32, optional)
// ============================================================

static void rope_yarn_ramp(float low, float high, int i0) {
    // unused in core computation, kept for API compat
    (void)low; (void)high; (void)i0;
}

static void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dims[2], int64_t i0,
    float ext_factor, float mscale,
    float * cos_theta, float * sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float y = (i0 / 2 - corr_dims[0]) / (corr_dims[1] - corr_dims[0] > 0.001f ? corr_dims[1] - corr_dims[0] : 0.001f);
        float ramp_mix = (1.0f < (1.0f - y) ? 1.0f : (0.0f > (1.0f - y) ? 0.0f : (1.0f - y))) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

static void rope_cache_init(
    int64_t pos, float freq_base, float freq_scale, const float * freq_factors,
    float corr_dims[2], int64_t ne0, float ext_factor, float attn_factor,
    float * cache, float sin_sign, float theta_scale) {
    float theta = powf(freq_base, (float)(pos * 2) / (float)ne0);
    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0/2] : 1.0f;
        rope_yarn(theta/ff, freq_scale, corr_dims, i0, ext_factor, attn_factor,
                  &cache[i0 + 0], &cache[i0 + 1]);
        cache[i0 + 1] *= sin_sign;
        theta *= theta_scale;
    }
}

static void rotate_pairs_f32(int64_t n, int64_t n_offset, const float * cache,
                             const float * src_data, float * dst_data) {
    for (int64_t i0 = 0; i0 < n; i0 += 2) {
        int64_t ic = i0 / (n_offset == 1 ? 1 : 2); // hack for NORMAL mode
        float cos_theta = cache[i0 + 0];
        float sin_theta = cache[i0 + 1];
        float x0 = src_data[ic];
        float x1 = src_data[ic + n_offset];
        dst_data[ic]       = x0 * cos_theta - x1 * sin_theta;
        dst_data[ic + n_offset] = x0 * sin_theta + x1 * cos_theta;
    }
}

static void ggml_compute_forward_rope_f32(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * src2,
        ggml_tensor * dst) {

    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__);
    int64_t start_time = ggml_time_us();

    // --- parse op_params ---
    // [0]=n_past, [1]=n_dims, [2]=mode, [3]=n_ctx, [4]=n_ctx_orig
    // [5]=freq_base(f), [6]=freq_scale(f), [7]=ext_factor(f),
    // [8]=attn_factor(f), [9]=beta_fast(f), [10]=beta_slow(f)
    // [11..14]=sections[4](i32)
    const int32_t * iparams = (const int32_t *)dst->op_params;
    int n_dims     = iparams[1];
    int mode       = iparams[2];
    int n_ctx_orig = iparams[4];

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   iparams + 5, sizeof(float));
    memcpy(&freq_scale,  iparams + 6, sizeof(float));
    memcpy(&ext_factor,  iparams + 7, sizeof(float));
    memcpy(&attn_factor, iparams + 8, sizeof(float));
    memcpy(&beta_fast,   iparams + 9, sizeof(float));
    memcpy(&beta_slow,   iparams + 10, sizeof(float));

    int64_t ne0 = src0->ne[0];  // embedding dim per head
    int64_t ne1 = src0->ne[1];  // heads
    int64_t ne2 = src0->ne[2];  // seq len
    int64_t ne3 = src0->ne[3];  // batch

    size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    size_t nb1  = dst->nb[1],   nb2  = dst->nb[2],   nb3  = dst->nb[3];

    GGML_ASSERT(n_dims <= ne0 && n_dims % 2 == 0);

    float theta_scale = powf(freq_base, -2.0f / (float)n_dims);

    float corr_dims[2] = {0.0f, 0.0f};
    // simplified corr_dims (full YaRN not needed for basic LLM inference)

    const float * freq_factors = NULL;
    if (src2 != NULL) {
        freq_factors = (const float *)src2->data;
    }

    const int32_t * pos = (const int32_t *)src1->data;

    // allocate temp cache for cos/sin values
    float * cache = (float *)malloc(ne0 * sizeof(float));
    if (!cache) {
        GGMLHEXAGON_LOG_ERROR("ROPE: failed to alloc cache (%lld bytes)", (long long)(ne0 * sizeof(float)));
        return;
    }

    int64_t last_i2 = -1;

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            if (last_i2 != i2) {
                int64_t p = pos[i2];
                rope_cache_init(p, freq_base, freq_scale, freq_factors,
                                corr_dims, ne0, ext_factor, attn_factor,
                                cache, 1.0f /* sin_sign */, theta_scale);
                last_i2 = i2;
            }

            for (int64_t i1 = 0; i1 < ne1; i1++) {
                const float * src_row = (const float *)((const uint8_t *)src0->data + i3*nb03 + i2*nb02 + i1*nb01);
                float * dst_row = (float *)((uint8_t *)dst->data + i3*nb3 + i2*nb2 + i1*nb1);

                switch (mode) {
                    case 0: // GGML_ROPE_TYPE_NORMAL
                        rotate_pairs_f32(n_dims, 1, cache, src_row, dst_row);
                        break;
                    case 2: // GGML_ROPE_TYPE_NEOX
                        rotate_pairs_f32(n_dims, n_dims/2, cache, src_row, dst_row);
                        break;
                    default:
                        GGMLHEXAGON_LOG_WARN("ROPE: unsupported mode %d, falling back to NEOX", mode);
                        rotate_pairs_f32(n_dims, n_dims/2, cache, src_row, dst_row);
                        break;
                }

                // copy remaining channels unchanged
                for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
                    dst_row[i0]     = src_row[i0];
                    dst_row[i0 + 1] = src_row[i0 + 1];
                }
            }
        }
    }

    free(cache);

    int64_t end_time = ggml_time_us();
    GGMLHEXAGON_LOG_INFO("ROPE elapse %lld us (ne0=%lld, ne1=%lld, ne2=%lld, mode=%d)",
                         (long long)(end_time - start_time),
                         (long long)ne0, (long long)ne1, (long long)ne2, mode);
    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__);
}

int ggmlop_dsp_rope(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_UNUSED(h);
    // ROPE is a ternary op (src0=input, src1=positions, src2=freq_factors).
    // Current batch infra only supports binary ops; freq_factors (src2) is
    // not commonly used in standard LLM inference, so treat as NULL for now.
    const ggml_tensor * src2 = NULL;
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__);

    int64_t begin_time = ggml_time_us();

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
        GGMLHEXAGON_LOG_ERROR("ROPE: unsupported types src0=%d src1=%d", src0->type, src1->type);
        return AEE_EUNSUPPORTED;
    }
    if (src2 && src2->type != GGML_TYPE_F32) {
        GGMLHEXAGON_LOG_ERROR("ROPE: unsupported src2 type %d", src2->type);
        return AEE_EUNSUPPORTED;
    }

    ggml_compute_forward_rope_f32(src0, src1, src2, dst);

    int64_t end_time = ggml_time_us();
    GGMLHEXAGON_LOG_INFO("elapse time of ROPE is %lld us", (long long)(end_time - begin_time));

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__);
    return 0;
}
