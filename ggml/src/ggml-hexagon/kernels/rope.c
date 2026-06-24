#include "ggml-dsp.h"
#include "worker_pool.h"
#include <math.h>
#include <string.h>

// ============================================================
// ROPE - Rotary Position Embedding
// Strictly aligned with ggml-cpu/ops.cpp reference implementation
// Ternary op: src0(input), src1(positions i32), src2(freq_factors f32, optional)
// Supports: NORMAL(0), NEOX(2), MROPE(8), VISION(24)
// ============================================================

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

static float rope_yarn_ramp(float low, float high, int i0) {
    float y = (i0 / 2 - low) / (high - low > 0.001f ? high - low : 0.001f);
    float r = 1.0f - y;
    return (r < 0.0f) ? 0.0f : ((r > 1.0f) ? 1.0f : r);
}

static void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dims[2], int64_t i0,
    float ext_factor, float mscale,
    float * cos_theta, float * sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], (int)i0) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

static void rope_yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base,
                                 float beta_fast, float beta_slow, float dims[2]) {
    float start = floorf(n_dims * logf(n_ctx_orig / (beta_fast * 2.0f * M_PI)) / (2.0f * logf(freq_base)));
    float end   = ceilf(n_dims * logf(n_ctx_orig / (beta_slow * 2.0f * M_PI)) / (2.0f * logf(freq_base)));
    dims[0] = (start > 0.0f) ? start : 0.0f;
    dims[1] = (end < (float)(n_dims - 1)) ? end : (float)(n_dims - 1);
}

// Standard ROPE cache init (NORMAL / NEOX modes)
static void rope_cache_init(
    float theta_base, float freq_scale, const float * freq_factors,
    float corr_dims[2], int64_t ne0, float ext_factor, float attn_factor,
    float * cache, float sin_sign, float theta_scale) {
    float theta = theta_base;
    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0/2] : 1.0f;
        rope_yarn(theta/ff, freq_scale, corr_dims, i0, ext_factor, attn_factor,
                  &cache[i0 + 0], &cache[i0 + 1]);
        cache[i0 + 1] *= sin_sign;
        theta *= theta_scale;
    }
}

// MROPE cache init (MROPE / VISION modes) — uses 4 position IDs and sections
static void ggml_mrope_cache_init(
     float theta_base_t, float theta_base_h, float theta_base_w, float theta_base_e,
     int sections[4], int is_imrope, int is_vision,
     float freq_scale, const float * freq_factors, float corr_dims[2],
     int64_t ne0, float ext_factor, float mscale,
     float * cache, float sin_sign, float theta_scale) {
    float theta_t = theta_base_t;
    float theta_h = theta_base_h;
    float theta_w = theta_base_w;
    float theta_e = theta_base_e;
    int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    int sec_w = sections[1] + sections[0];
    int sec_e = sections[2] + sec_w;

    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0/2] : 1.0f;

        int sector = (i0 / 2) % sect_dims;
        // Only reset theta at section boundaries for independent-sections mode (VISION)
        // MROPE uses continuous accumulation across all sectors
        if (is_vision) {
            if (sector == 0) {
                theta_t = theta_base_t;
            } else if (sector == sections[0]) {
                theta_h = theta_base_h;
            } else if (sector == sec_w) {
                theta_w = theta_base_w;
            } else if (sector == sec_e) {
                theta_e = theta_base_e;
            }
        }

        float theta = theta_t;
        if (is_imrope) {
            // interleaved mrope (Qwen3VL style)
            if (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } else if (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } else if (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } else {
                theta = theta_e;
            }
        } else {
            // standard mrope / vision
            if (sector >= sections[0] && sector < sec_w) {
                theta = theta_h;
            } else if (sector >= sec_w && sector < sec_w + sections[2]) {
                theta = theta_w;
            } else if (sector >= sec_w + sections[2]) {
                theta = theta_e;
            }
        }

        rope_yarn(theta/ff, freq_scale, corr_dims, i0, ext_factor, mscale,
                  &cache[i0 + 0], &cache[i0 + 1]);
        cache[i0 + 1] *= sin_sign;

        theta_t *= theta_scale;
        theta_h *= theta_scale;
        theta_w *= theta_scale;
        theta_e *= theta_scale;
    }
}

// Rotate pairs: n = number of elements to rotate, n_offset = stride between pair members
// scale: 1 for NORMAL (ic=i0), 2 for others (ic=i0/2)
static void rotate_pairs_f32(int64_t n, int64_t n_offset, const float * cache,
                             const float * src_data, float * dst_data) {
    for (int64_t i0 = 0; i0 < n; i0 += 2) {
        int64_t ic = (n_offset == 1) ? i0 : (i0 / 2);
        float cos_theta = cache[i0 + 0];
        float sin_theta = cache[i0 + 1];
        float x0 = src_data[ic];
        float x1 = src_data[ic + n_offset];
        dst_data[ic]       = x0 * cos_theta - x1 * sin_theta;
        dst_data[ic + n_offset] = x0 * sin_theta + x1 * cos_theta;
    }
}

static void rotate_pairs_f16(int64_t n, int64_t n_offset, const float * cache,
                             const uint16_t * src_data, uint16_t * dst_data) {
    for (int64_t i0 = 0; i0 < n; i0 += 2) {
        int64_t ic = (n_offset == 1) ? i0 : (i0 / 2);
        float cos_theta = cache[i0 + 0];
        float sin_theta = cache[i0 + 1];
        float x0 = ggml_compute_fp16_to_fp32(src_data[ic]);
        float x1 = ggml_compute_fp16_to_fp32(src_data[ic + n_offset]);
        dst_data[ic]             = ggml_compute_fp32_to_fp16(x0 * cos_theta - x1 * sin_theta);
        dst_data[ic + n_offset] = ggml_compute_fp32_to_fp16(x0 * sin_theta + x1 * cos_theta);
    }
}

static void ggml_compute_forward_rope_f32(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * src2,
        ggml_tensor * dst) {

    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__);
    int64_t start_time = ggml_time_us();

    // --- parse op_params (reference aligned) ---
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

    int sections[4];
    memcpy(sections, iparams + 11, sizeof(int) * 4);

    int64_t ne0 = src0->ne[0];  // embedding dim per head
    int64_t ne1 = src0->ne[1];  // heads
    int64_t ne2 = src0->ne[2];  // seq len
    int64_t ne3 = src0->ne[3];  // batch

    size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    size_t nb1  = dst->nb[1],   nb2  = dst->nb[2],   nb3  = dst->nb[3];

    // // diagnostic logging (disabled, keep for future debugging)
    // GGMLHEXAGON_LOG_WARN("ROPE diag: n_dims=%d mode=%d n_ctx_orig=%d freq_base=%.1f freq_scale=%.6f",
    //                      n_dims, mode, n_ctx_orig, freq_base, freq_scale);
    // GGMLHEXAGON_LOG_WARN("ROPE diag: src0 type=%d ne=[%d,%d,%d,%d] nb=[%d,%d,%d,%d] data=%p",
    //                      src0->type, (int)src0->ne[0], (int)src0->ne[1], (int)src0->ne[2], (int)src0->ne[3],
    //                      (int)src0->nb[0], (int)src0->nb[1], (int)src0->nb[2], (int)src0->nb[3], src0->data);
    // GGMLHEXAGON_LOG_WARN("ROPE diag: src1 type=%d ne=[%d,%d,%d,%d] nb=[%d,%d,%d,%d] data=%p data_len=%d",
    //                      src1->type, (int)src1->ne[0], (int)src1->ne[1], (int)src1->ne[2], (int)src1->ne[3],
    //                      (int)src1->nb[0], (int)src1->nb[1], (int)src1->nb[2], (int)src1->nb[3], src1->data, src1->data_len);
    // GGMLHEXAGON_LOG_WARN("ROPE diag: dst type=%d ne=[%d,%d,%d,%d] nb=[%d,%d,%d,%d] data=%p",
    //                      dst->type, (int)dst->ne[0], (int)dst->ne[1], (int)dst->ne[2], (int)dst->ne[3],
    //                      (int)dst->nb[0], (int)dst->nb[1], (int)dst->nb[2], (int)dst->nb[3], dst->data);
    // {
    //     const int32_t * pos = (const int32_t *)src1->data;
    //     int n_pos = (int)ne2;
    //     if (n_pos > 8) n_pos = 8;
    //     GGMLHEXAGON_LOG_WARN("ROPE diag: pos[0..%d] = [%d, %d, %d, %d, %d, %d, %d, %d]",
    //                          n_pos,
    //                          n_pos > 0 ? pos[0] : -1, n_pos > 1 ? pos[1] : -1,
    //                          n_pos > 2 ? pos[2] : -1, n_pos > 3 ? pos[3] : -1,
    //                          n_pos > 4 ? pos[4] : -1, n_pos > 5 ? pos[5] : -1,
    //                          n_pos > 6 ? pos[6] : -1, n_pos > 7 ? pos[7] : -1);
    //     const float * sf = (const float *)src0->data;
    //     GGMLHEXAGON_LOG_WARN("ROPE diag: src0 f32=[%.4f, %.4f, %.4f, %.4f]", sf[0], sf[1], sf[2], sf[3]);
    // }

    GGML_ASSERT(n_dims <= ne0 && n_dims % 2 == 0);

    float theta_scale = powf(freq_base, -2.0f / (float)n_dims);

    float corr_dims[2] = {0.0f, 0.0f};
    rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const float * freq_factors = NULL;
    if (src2 != NULL) {
        freq_factors = (const float *)src2->data;
    }

    const int32_t * pos = (const int32_t *)src1->data;

    // mode detection (reference aligned)
    const int is_imrope = (mode == 40);  // GGML_ROPE_TYPE_IMROPE
    const int mrope_used = (mode & 8);   // GGML_ROPE_TYPE_MROPE bit
    const int is_vision  = (mode == 24); // GGML_ROPE_TYPE_VISION

    if (is_vision) {
        GGML_ASSERT(n_dims == ne0 / 2);
    }

    // static cache buffer to avoid malloc/free on DSP (heap alloc can cause
    // corruption or cache-coherency issues in frequently-called DSP functions)
    #define ROPE_CACHE_MAX_NE0 4096
    static float s_rope_cache[ROPE_CACHE_MAX_NE0];
    float * cache = (ne0 <= ROPE_CACHE_MAX_NE0) ? s_rope_cache : (float *)malloc(ne0 * sizeof(float));
    if (!cache) {
        GGMLHEXAGON_LOG_ERROR("ROPE: failed to alloc cache (%lld bytes)", (long long)(ne0 * sizeof(float)));
        return;
    }

    int64_t last_i2 = -1;

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            if (last_i2 != i2) {
                if (!mrope_used) {
                    // NORMAL / NEOX: single position ID
                    int64_t p = pos[i2];
                    rope_cache_init((float)p, freq_scale, freq_factors,
                                    corr_dims, ne0, ext_factor, attn_factor,
                                    cache, 1.0f /* sin_sign */, theta_scale);
                } else {
                    // MROPE / VISION / IMROPE: 4 position IDs (t, h, w, e)
                    int64_t p_t = pos[i2];
                    int64_t p_h = pos[i2 + ne2];
                    int64_t p_w = pos[i2 + ne2 * 2];
                    int64_t p_e = pos[i2 + ne2 * 3];
                    ggml_mrope_cache_init(
                        (float)p_t, (float)p_h, (float)p_w, (float)p_e,
                        sections, is_imrope, is_vision,
                        freq_scale, freq_factors, corr_dims, ne0,
                        ext_factor, attn_factor, cache, 1.0f, theta_scale);
                }
                last_i2 = i2;
            }

            for (int64_t i1 = 0; i1 < ne1; i1++) {
                size_t src_off = (size_t)(i3*nb03 + i2*nb02 + i1*nb01);
                size_t dst_off = (size_t)(i3*nb3  + i2*nb2  + i1*nb1);

                switch (mode) {
                    case 0: // GGML_ROPE_TYPE_NORMAL
                        if (src0->type == GGML_TYPE_F16) {
                            rotate_pairs_f16(n_dims, 1, cache,
                                (const uint16_t *)((const uint8_t *)src0->data + src_off),
                                (uint16_t *)((uint8_t *)dst->data + dst_off));
                        } else {
                            rotate_pairs_f32(n_dims, 1, cache,
                                (const float *)((const uint8_t *)src0->data + src_off),
                                (float *)((uint8_t *)dst->data + dst_off));
                        }
                        break;
                    case 2: // GGML_ROPE_TYPE_NEOX
                    case 8: // GGML_ROPE_TYPE_MROPE
                    case 40:// GGML_ROPE_TYPE_IMROPE
                        if (src0->type == GGML_TYPE_F16) {
                            rotate_pairs_f16(n_dims, n_dims/2, cache,
                                (const uint16_t *)((const uint8_t *)src0->data + src_off),
                                (uint16_t *)((uint8_t *)dst->data + dst_off));
                        } else {
                            rotate_pairs_f32(n_dims, n_dims/2, cache,
                                (const float *)((const uint8_t *)src0->data + src_off),
                                (float *)((uint8_t *)dst->data + dst_off));
                        }
                        break;
                    case 24: // GGML_ROPE_TYPE_VISION
                        if (src0->type == GGML_TYPE_F16) {
                            rotate_pairs_f16(ne0, n_dims, cache,
                                (const uint16_t *)((const uint8_t *)src0->data + src_off),
                                (uint16_t *)((uint8_t *)dst->data + dst_off));
                        } else {
                            rotate_pairs_f32(ne0, n_dims, cache,
                                (const float *)((const uint8_t *)src0->data + src_off),
                                (float *)((uint8_t *)dst->data + dst_off));
                        }
                        break;
                    default:
                        GGMLHEXAGON_LOG_WARN("ROPE: unsupported mode %d", mode);
                        break;
                }

                // copy remaining channels unchanged (skip for VISION)
                if (!is_vision) {
                    for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
                        if (src0->type == GGML_TYPE_F16) {
                            uint16_t * d = (uint16_t *)((uint8_t *)dst->data + dst_off + i0 * sizeof(uint16_t));
                            const uint16_t * s = (const uint16_t *)((const uint8_t *)src0->data + src_off + i0 * sizeof(uint16_t));
                            d[0] = s[0]; d[1] = s[1];
                        } else {
                            float * d = (float *)((uint8_t *)dst->data + dst_off + i0 * sizeof(float));
                            const float * s = (const float *)((const uint8_t *)src0->data + src_off + i0 * sizeof(float));
                            d[0] = s[0]; d[1] = s[1];
                        }
                    }
                }
            }
        }
    }

    // Detailed pair verification: log first 2 rotation pairs with cos/sin
    // NOTE: disabled - flawed for in-place ops (src0->data == dst->data)
    // if (mode == 2 && src0->type == GGML_TYPE_F32) {
    //     const float * sf = (const float *)src0->data;
    //     const float * df = (const float *)dst->data;
    //     int hd = n_dims / 2;
    //     GGMLHEXAGON_LOG_WARN("ROPE pair0: x0=%.6f x1=%.6f cos=%.6f sin=%.6f -> dst0=%.6f dst1=%.6f",
    //                          sf[0], sf[hd], cache[0], cache[1], df[0], df[hd]);
    //     GGMLHEXAGON_LOG_WARN("ROPE pair1: x0=%.6f x1=%.6f cos=%.6f sin=%.6f -> dst0=%.6f dst1=%.6f",
    //                          sf[1], sf[hd+1], cache[2], cache[3], df[1], df[hd+1]);
    // }

    if (cache != s_rope_cache) free(cache);

    int64_t end_time = ggml_time_us();
    GGMLHEXAGON_LOG_INFO("ROPE elapse %lld us (ne0=%lld, ne1=%lld, ne2=%lld, mode=%d, type=%d)",
                         (long long)(end_time - start_time),
                         (long long)ne0, (long long)ne1, (long long)ne2, mode, src0->type);
    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__);
}

int ggmlop_dsp_rope(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, ggml_tensor * dst) {
    GGML_UNUSED(h);
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__);

    int64_t begin_time = ggml_time_us();

    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        GGMLHEXAGON_LOG_ERROR("ROPE: unsupported src0 type %d", src0->type);
        return AEE_EUNSUPPORTED;
    }
    if (src1->type != GGML_TYPE_I32) {
        GGMLHEXAGON_LOG_ERROR("ROPE: unsupported src1 type %d", src1->type);
        return AEE_EUNSUPPORTED;
    }

    ggml_compute_forward_rope_f32(src0, src1, src2, dst);

    int64_t end_time = ggml_time_us();
    GGMLHEXAGON_LOG_INFO("elapse time of ROPE is %lld us", (long long)(end_time - begin_time));

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__);
    return 0;
}
