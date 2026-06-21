#include "ggml-dsp.h"
#include "worker_pool.h"
#include "../htp/hvx-base.h"   // for Qualcomm's official hvx_vec_f32_to_f16 with vdeal
#include "../htp/hvx-reduce.h" // for Qualcomm's official hvx_vec_reduce_max_f32
#include "../htp/hex-dma.h"    // for Qualcomm's official DMA async transfers

union ui32f { int32_t i; float f; };

#define HMX_FP16_TILE_N_ROWS 32
#define HMX_FP16_TILE_N_COLS 32
#define HMX_FP16_TILE_N_ELMS 1024
#define HMX_FP16_TILE_SIZE (HMX_FP16_TILE_N_ELMS * sizeof(__fp16))

static HVX_INLINE_ALWAYS void l2fetch(const void * p, uint32_t stride,
                           uint32_t width, uint32_t height,
                           uint32_t dir) {
    uint64_t control = HEXAGON_V64_CREATE_H(dir, stride, width, height);
    __asm__ __volatile__ (" l2fetch(%0,%1) " : :"r"(p),"r"(control));
}

// create a vector of floats from a float
static __attribute__((always_inline)) HVX_Vector create_sfv_from_sf(float value) {
    union ui32f cvt;
    cvt.f = value;
    HVX_Vector tmp = Q6_V_vsplat_R(cvt.i);
    return tmp;
}

// create a vector of qf32's from a float
static __attribute__((always_inline)) HVX_Vector create_qf32v_from_sf(float value) {
    HVX_Vector tmp = Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_vsplat_R(0), create_sfv_from_sf(value));
    return tmp;
}

// convert qf32 vector to float vector
static __attribute__((always_inline)) HVX_Vector convert_qf32v_to_fltv(HVX_Vector vect) {
    HVX_Vector tmp = Q6_Vsf_equals_Vqf32(vect);
    return tmp;
}

static inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline float fp32_from_bits(uint32_t w) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline float ggml_compute_bf16_to_fp32(ggml_bf16_t h) {
    union { float f; uint32_t i; } u;
    u.i = (uint32_t)h.bits << 16;
    return u.f;
}

static inline ggml_bf16_t ggml_compute_fp32_to_bf16(float f) {
    union { float f; uint32_t i; } u;
    u.f = f;
    // Round to nearest even
    uint32_t rounding_bias = 0x0000FFFF + ((u.i >> 16) & 1);
    ggml_bf16_t h;
    h.bits = (uint16_t)((u.i + rounding_bias) >> 16);
    return h;
}

static inline float ggml_e8m0_to_fp32_half(uint8_t x) {
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

static inline float ggml_ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || x == 0x7F) {
        return 0.0f;
    }
    int   exp = (x >> 3) & 0xF;
    int   man = x & 0x7;
    float raw;
    if (exp == 0) {
        raw = ldexpf((float)man, -9);
    } else {
        raw = ldexpf(1.0f + (float)man / 8.0f, exp - 7);
    }
    return raw * 0.5f;
}

// Quantize F32 to BF16
static void quantize_row_bf16_generic(const float * GGML_RESTRICT x, ggml_bf16_t * GGML_RESTRICT y, int n) {
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_compute_fp32_to_bf16(x[i]);
    }
}

void quantize_row_bf16_hvx(const float * GGML_RESTRICT x, ggml_bf16_t * GGML_RESTRICT y, int n) {
    const int fp32_per_vec = VLEN / sizeof(float);  // 32

    // scalar fallback for small or unaligned cases
    if (n < fp32_per_vec || ((uintptr_t)x & 0x7F) != 0 || ((uintptr_t)y & 0x3F) != 0) {
        for (int i = 0; i < n; ++i) {
            y[i] = ggml_compute_fp32_to_bf16(x[i]);
        }
        return;
    }

    const int nvec = n / fp32_per_vec;
    const int nloe = n % fp32_per_vec;

    const HVX_Vector * restrict vx = (const HVX_Vector *)x;

    // BF16 = upper 16 bits of each FP32 value
    // Process one FP32 vector (32 floats) at a time -> 32 BF16 values (64 bytes)
    for (int i = 0; i < nvec; ++i) {
        HVX_Vector v = vx[i];

        // Shift right by 16 bits: moves upper 16 bits (BF16) to lower 16 bits of each 32-bit word
        HVX_Vector s = Q6_Vuw_vlsr_VuwR(v, 16);

        // vshuff swaps even/odd halfwords within each 32-bit word
        // After shift+shuff: even halfwords = BF16 values, odd halfwords = zeros
        s = Q6_Vh_vshuff_Vh(s);

        // vdeal packs even halfwords into first 64 bytes, odd into last 64 bytes
        s = Q6_Vh_vdeal_Vh(s);

        // First 64 bytes contain 32 BF16 values
        hvx_vec_store_u(y + i * fp32_per_vec, fp32_per_vec * sizeof(ggml_bf16_t), s);
    }

    if (nloe > 0) {
        const float * tail_x = x + nvec * fp32_per_vec;
        ggml_bf16_t * tail_y = y + nvec * fp32_per_vec;
        for (int i = 0; i < nloe; ++i) {
            tail_y[i] = ggml_compute_fp32_to_bf16(tail_x[i]);
        }
    }
}

static int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    return (int)(fval + (fval >= 0 ? 0.5f : -0.5f));
}

static inline float horizontal_sum_safe(HVX_Vector vec) {
    float __attribute__((aligned(VLEN))) buffer[32];
    float sum = 0.0f;
    union {
        HVX_Vector v;
        float f[32];
    } converter;
    converter.v = vec;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        sum += converter.f[i];
    }
    return sum;
}

static inline float horizontal_sum_hvx_1(HVX_Vector vec) {
    HVX_VectorPair shuffled = Q6_W_vshuff_VVR(vec, vec, 64);
    HVX_Vector sum = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(shuffled), Q6_V_hi_W(shuffled));

    shuffled = Q6_W_vshuff_VVR(sum, sum, 32);
    sum = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(shuffled), Q6_V_hi_W(shuffled));

    shuffled = Q6_W_vshuff_VVR(sum, sum, 16);
    sum = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(shuffled), Q6_V_hi_W(shuffled));

    shuffled = Q6_W_vshuff_VVR(sum, sum, 8);
    sum = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(shuffled), Q6_V_hi_W(shuffled));

    shuffled = Q6_W_vshuff_VVR(sum, sum, 4);
    sum = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(shuffled), Q6_V_hi_W(shuffled));

    int32_t result = Q6_R_vextract_VR(sum, 0);
    float f_result;
    memcpy(&f_result, &result, sizeof(float));
    return f_result;
}

static inline float horizontal_sum_hvx_2(HVX_Vector v) {
#if defined(v68) || defined(v69) || defined(v73) || defined(v75)
  v = Q6_Vqf32_vadd_VsfVsf(v, Q6_V_vror_VR(v, 64));
  v = Q6_Vqf32_vadd_Vqf32Vqf32(v, Q6_V_vror_VR(v, 32));
  v = Q6_Vqf32_vadd_Vqf32Vqf32(v, Q6_V_vror_VR(v, 16));
  v = Q6_Vqf32_vadd_Vqf32Vqf32(v, Q6_V_vror_VR(v, 8));
  v = Q6_Vqf32_vadd_Vqf32Vqf32(v, Q6_V_vror_VR(v, 4));
  v = Q6_Vsf_equals_Vqf32(v);
#else
  v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v, Q6_V_vror_VR(v, 64)));
  v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v, Q6_V_vror_VR(v, 32)));
  v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v, Q6_V_vror_VR(v, 16)));
  v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v, Q6_V_vror_VR(v, 8)));
  v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v, Q6_V_vror_VR(v, 4)));
#endif
  return *((float*)&v);
}

static void vec_dot_f32_hvx_impl(int n, float *GGML_RESTRICT s, const float *GGML_RESTRICT x, const float *GGML_RESTRICT y) {
    const HVX_Vector * restrict vx = (const HVX_Vector *) x;
    const HVX_Vector * restrict vy = (const HVX_Vector *) y;

    uint32_t nvec = n / VLEN_FP32;
    uint32_t nloe = n % VLEN_FP32;

    HVX_Vector rsum = Q6_V_vsplat_R(0);

    uint32_t i = 0;

    #pragma unroll(4)
    for (i = 0; i < nvec; i++) {
        HVX_Vector prod = Q6_Vsf_vmpy_VsfVsf(vx[i], vy[i]);
        rsum = Q6_Vsf_vadd_VsfVsf(rsum, prod);
    }

    if (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector x_sf = Q6_V_vand_QV(bmask, vx[i]);
        HVX_Vector y_sf = Q6_V_vand_QV(bmask, vy[i]);
        HVX_Vector prod = Q6_Vsf_vmpy_VsfVsf(x_sf, y_sf);
        rsum = Q6_Vsf_vadd_VsfVsf(rsum, prod);
    }

    *s = hvx_vec_get_f32(hvx_vec_reduce_sum_f32(rsum));
}

static inline void vec_dot_f32_hvx_impl_me(int n, float * GGML_RESTRICT s, const float * GGML_RESTRICT x, const float * GGML_RESTRICT y) {
    float sumf    = 0.0f;

    if ((((uintptr_t)x | (uintptr_t)y) % ALIGN_128_BYTE) != 0) {
        GGMLDSP_LOG_DEBUG("memaddress mismatch alignment 128 bytes x:%p y:%p", x, y);
        #pragma unroll
        for (int i = 0; i < n; ++i) {
            sumf += (ggml_float) (x[i] * y[i]);
        }
        *s = sumf;
        return;
    }

    const int FLOATS_PER_VECTOR = 128 / sizeof(float);
    const int block             = n / FLOATS_PER_VECTOR;
    const size_t left           = n % FLOATS_PER_VECTOR;
    const size_t blocks         = block * FLOATS_PER_VECTOR;

#if defined(v68) || defined(v69) || defined(v73) || defined(v75)
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) != 0) {
        FARF(ALWAYS, "failed hvx lock\n");
        return;
    }
#endif

    HVX_Vector * va;
    HVX_Vector * vb;

    va = (HVX_Vector *)x;
    vb = (HVX_Vector *)y;

    HVX_Vector sout, temp, qf32;
    qf32 =  create_qf32v_from_sf(0.0f);
    int fetch_counts = 1;
        if (0 == (n % (128 * 32)))
            fetch_counts = 32;
        else if (0 == (n % (128 * 24)))
            fetch_counts = 24;
        else if (0 == (n % (128 * 16)))
            fetch_counts = 16;
        else if (0 == (n % (128 * 8)))
            fetch_counts = 8;
        else if (0 == (n % (128 * 4)))
            fetch_counts = 4;
        else if (0 == (n % (128 * 2)))
            fetch_counts = 2;
        else
            fetch_counts = 1;

    for (size_t i = 0; i < block; i+= fetch_counts) {
        l2fetch(va + VLEN * fetch_counts, VLEN, VLEN * fetch_counts, 1, 0);
        l2fetch(vb + VLEN * fetch_counts, VLEN, VLEN * fetch_counts, 1, 0);
        #pragma unroll
        for (size_t j = 0; j < fetch_counts; j++) {
            temp = Q6_Vqf32_vmpy_VsfVsf(*va++, *vb++);
            qf32 = Q6_Vqf32_vadd_Vqf32Vqf32(qf32, temp);
        }
    }
    sout = Q6_Vsf_equals_Vqf32(qf32);

    //sumf = horizontal_sum_hvx_1(sout);
    sumf = horizontal_sum_hvx_2(sout);
    //sumf = horizontal_sum_safe(sout);

#if defined(v68) || defined(v69) || defined(v73) || defined(v75)
    qurt_hvx_unlock();
#endif
    if (left > 0) {
        #pragma unroll
        for (size_t i = 0; i < left; i++) {
            sumf += (ggml_float)(x[i + blocks]*y[i + blocks]);
        }
    }

    *s = sumf;
}

void vec_dot_f32_hvx(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, size_t bx,
                    const void *GGML_RESTRICT vy, size_t by, int nrc) {
    const float *GGML_RESTRICT x = (const float *)vx;
    const float *GGML_RESTRICT y = (const float *)vy;
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(nrc);

    if (n >= VLEN_FP32 && ((uintptr_t)x & 0x7F) == 0 && ((uintptr_t)y & 0x7F) == 0) {
        vec_dot_f32_hvx_impl(n, s, x, y);
        vec_dot_f32_hvx_impl_me(n, s, x, y);
        return;
    }

    ggml_float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float) (x[i] * y[i]);
    }
    *s = sumf;
}

void vec_dot_f32_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, size_t bx,
                    const void *GGML_RESTRICT vy, size_t by, int nrc) {
    const float *GGML_RESTRICT x = (const float *)vx;
    const float *GGML_RESTRICT y = (const float *)vy;
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(nrc);

    ggml_float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float) (x[i] * y[i]);
    }
    *s = sumf;
}

void vec_dot_f16_f32_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, size_t bx,
                    const void *GGML_RESTRICT vy, size_t by, int nrc) {
    const uint16_t *GGML_RESTRICT x = (const uint16_t *)vx;
    const float *GGML_RESTRICT y = (const float *)vy;
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(nrc);

    ggml_float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        float va = ggml_compute_fp16_to_fp32(x[i]);
        float vb = y[i];
        sumf += (ggml_float) (va * vb);
    }
    *s = sumf;
}

void vec_dot_f16_f16_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, size_t bx,
                    const void *GGML_RESTRICT vy, size_t by, int nrc) {
    const uint16_t *GGML_RESTRICT x = (const uint16_t *)vx;
    const uint16_t *GGML_RESTRICT y = (const uint16_t *)vy;
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(nrc);

    ggml_float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        float va = ggml_compute_fp16_to_fp32(x[i]);
        float vb = ggml_compute_fp16_to_fp32(y[i]);
        sumf += (ggml_float) (va * vb);
    }
    *s = sumf;
}

void vec_dot_f16_f16_hvx(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, size_t bx,
                    const void *GGML_RESTRICT vy, size_t by, int nrc) {
    const uint16_t *GGML_RESTRICT x = (const uint16_t *)vx;
    const uint16_t *GGML_RESTRICT y = (const uint16_t *)vy;
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(nrc);

    const int fp16_per_vec = VLEN / sizeof(uint16_t); // 64
    const int nvec = n / fp16_per_vec;
    const int nloe = n % fp16_per_vec;

    float sumf = 0.0f;

    if (nvec > 0 && ((uintptr_t)x & 0x7F) == 0 && ((uintptr_t)y & 0x7F) == 0) {
        const HVX_Vector * restrict vx = (const HVX_Vector *)x;
        const HVX_Vector * restrict vy = (const HVX_Vector *)y;

        HVX_VectorPair acc = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());

        for (int i = 0; i < nvec; ++i) {
            HVX_Vector vx_shuf = Q6_Vh_vshuff_Vh(vx[i]);
            HVX_Vector vy_shuf = Q6_Vh_vshuff_Vh(vy[i]);
            acc = hvx_vec_mpyacc_f32_f16(acc, vx_shuf, vy_shuf);
        }

        // horizontal sum of acc
        HVX_Vector acc_lo = Q6_V_lo_W(acc);
        HVX_Vector acc_hi = Q6_V_hi_W(acc);
        HVX_Vector sum_v = Q6_Vsf_vadd_VsfVsf(acc_lo, acc_hi);
        sumf = horizontal_sum_hvx_2(sum_v);

        if (nloe > 0) {
            const int base = nvec * fp16_per_vec;
            for (int i = 0; i < nloe; ++i) {
                float va = ggml_compute_fp16_to_fp32(x[base + i]);
                float vb = ggml_compute_fp16_to_fp32(y[base + i]);
                sumf += (va * vb);
            }
        }
    } else {
        for (int i = 0; i < n; ++i) {
            float va = ggml_compute_fp16_to_fp32(x[i]);
            float vb = ggml_compute_fp16_to_fp32(y[i]);
            sumf += (va * vb);
        }
    }

    *s = sumf;
}


#define GGML_Q4_0_BLCK_SZ (sizeof(uint16_t) + QK4_0/2)
#define GGML_Q4_1_BLCK_SZ (sizeof(uint16_t) + sizeof(uint16_t) + QK4_1/2)
#define GGML_Q5_0_BLCK_SZ (sizeof(uint16_t) + sizeof(uint32_t) + QK5_0/2)
#define GGML_Q5_1_BLCK_SZ (2*sizeof(uint16_t) + sizeof(uint32_t) + QK5_1/2)
#define GGML_Q8_0_BLCK_SZ (sizeof(uint16_t) + QK8_0)
#define GGML_Q8_1_BLCK_SZ (sizeof(uint16_t) + sizeof(uint16_t) + QK8_1)

static void vec_dot_q4_0_f32_generic(int n, float *GGML_RESTRICT s, size_t bs, const block_q4_0 *GGML_RESTRICT x,
                    size_t bx, const float *GGML_RESTRICT y, size_t by, int nrc) {
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(nrc);

    ggml_float sumf = 0.0;
    const int nb = n / QK4_0;

    for (int i = 0; i < nb; ++i) {
        const float d = ggml_compute_fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK4_0 / 2; ++j) {
            const int8_t q0 = (x[i].qs[j] & 0x0F) - 8;
            const int8_t q1 = (x[i].qs[j] >> 4) - 8;
            sumf += (ggml_float)(q0 * d * y[i * QK4_0 + 2 * j]);
            sumf += (ggml_float)(q1 * d * y[i * QK4_0 + 2 * j + 1]);
        }
    }
    *s = sumf;
}

static void vec_dot_q4_0_f32_hvx(int n, float *GGML_RESTRICT s, size_t bs, const block_q4_0 *GGML_RESTRICT x,
                    size_t bx, const float *GGML_RESTRICT y, size_t by, int nrc) {
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(nrc);

    const int nb = n / QK4_0;
    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const float d = ggml_compute_fp16_to_fp32(x[i].d);
        const uint8_t * qs_ptr = x[i].qs;
        const float   * y_ptr  = y + i * QK4_0;

        // Dequantize q4_0 to f32 using scalar (block is only 18 bytes, not vector-aligned)
        float dq[QK4_0];
        for (int j = 0; j < QK4_0 / 2; ++j) {
            dq[2 * j]     = (float)((qs_ptr[j] & 0x0F) - 8) * d;
            dq[2 * j + 1] = (float)((qs_ptr[j] >> 4) - 8) * d;
        }

        // Dot product with f32 y using HVX if aligned
        if (((uintptr_t)y_ptr & 0x7F) == 0 && QK4_0 >= VLEN_FP32) {
            const HVX_Vector * restrict vy = (const HVX_Vector *)y_ptr;
            HVX_Vector * restrict vdq = (HVX_Vector *)dq;
            HVX_Vector rsum = Q6_V_vsplat_R(0);
            for (int j = 0; j < QK4_0 / VLEN_FP32; ++j) {
                HVX_Vector prod = Q6_Vsf_vmpy_VsfVsf(vdq[j], vy[j]);
                rsum = Q6_Vsf_vadd_VsfVsf(rsum, prod);
            }
            sumf += hvx_vec_get_f32(hvx_vec_reduce_sum_f32(rsum));
        } else {
            float block_sum = 0.0f;
            for (int j = 0; j < QK4_0; ++j) {
                block_sum += dq[j] * y_ptr[j];
            }
            sumf += block_sum;
        }
    }

    *s = sumf;
}

static void vec_dot_q8_0_f32_generic(int n, float *GGML_RESTRICT s, size_t bs, const block_q8_0 *GGML_RESTRICT x,
                    size_t bx, const float *GGML_RESTRICT y, size_t by, int nrc) {
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(nrc);

    ggml_float sumf = 0.0;
    const int nb = n / QK8_0;

    for (int i = 0; i < nb; ++i) {
        const float d = ggml_compute_fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK8_0; ++j) {
            sumf += (ggml_float)(x[i].qs[j] * d * y[i * QK8_0 + j]);
        }
    }
    *s = sumf;
}

static void vec_dot_q8_0_f32_hvx(int n, float *GGML_RESTRICT s, size_t bs, const block_q8_0 *GGML_RESTRICT x,
                    size_t bx, const float *GGML_RESTRICT y, size_t by, int nrc) {
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(nrc);

    const int nb = n / QK8_0;
    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const float d = ggml_compute_fp16_to_fp32(x[i].d);
        const int8_t * qs_ptr = x[i].qs;
        const float  * y_ptr  = y + i * QK8_0;

        // Dequantize q8_0 to f32
        float dq[QK8_0];
        for (int j = 0; j < QK8_0; ++j) {
            dq[j] = (float)qs_ptr[j] * d;
        }

        // Dot product with f32 y using HVX if aligned
        if (((uintptr_t)y_ptr & 0x7F) == 0 && QK8_0 >= VLEN_FP32) {
            const HVX_Vector * restrict vy = (const HVX_Vector *)y_ptr;
            HVX_Vector * restrict vdq = (HVX_Vector *)dq;
            HVX_Vector rsum = Q6_V_vsplat_R(0);
            for (int j = 0; j < QK8_0 / VLEN_FP32; ++j) {
                HVX_Vector prod = Q6_Vsf_vmpy_VsfVsf(vdq[j], vy[j]);
                rsum = Q6_Vsf_vadd_VsfVsf(rsum, prod);
            }
            sumf += hvx_vec_get_f32(hvx_vec_reduce_sum_f32(rsum));
        } else {
            float block_sum = 0.0f;
            for (int j = 0; j < QK8_0; ++j) {
                block_sum += dq[j] * y_ptr[j];
            }
            sumf += block_sum;
        }
    }

    *s = sumf;
}

void vec_dot_q4_0_q8_0_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, size_t bx,
                    const void *GGML_RESTRICT vy, size_t by, int nrc) {
    const block_q4_0 *GGML_RESTRICT x = (const block_q4_0 *)vx;
    const block_q8_0 *GGML_RESTRICT y = (const block_q8_0 *)vy;
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(nrc);

    const int qk = QK4_0;
    const int nb = n / qk;

    float sumf = 0;
    for (int ib = 0; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F) - 8;
            const int v1 = (x[ib].qs[j] >>   4) - 8;

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        const float d = ggml_compute_fp16_to_fp32(x[ib].d) * ggml_compute_fp16_to_fp32(y[ib].d);
        sumf += (float)(sumi0 + sumi1) * d;
    }
    *s = sumf;
}

void vec_dot_q4_0_q8_0_generic_hvx(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, size_t bx,
                    const void *GGML_RESTRICT vy, size_t by, int nrc) {
    const block_q4_0 *GGML_RESTRICT x = (const block_q4_0 *)vx;
    const block_q8_0 *GGML_RESTRICT y = (const block_q8_0 *)vy;
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(nrc);

    const int qk = QK4_0;
    const int nb = n / qk;

    float sumf = 0;

    const HVX_Vector vmask = Q6_Vb_vsplat_R(0x0F);
    const HVX_Vector voff  = Q6_Vb_vsplat_R(8);
    const HVX_VectorPred p16 = Q6_Q_vsetq_R(16);
    const HVX_VectorPred p32 = Q6_Q_vsetq_R(32);

    for (int ib = 0; ib < nb; ++ib) {
        // Use HVX_UVector for unaligned load (block qs at offset 2, not 128-byte aligned)
        HVX_Vector qs_raw = Q6_V_vand_QV(p16, *(const HVX_UVector *)x[ib].qs);

        // Extract low nibbles: (qs & 0x0F) - 8
        HVX_Vector lo_nib = Q6_V_vand_VV(qs_raw, vmask);
        HVX_Vector lo_val = Q6_Vb_vsub_VbVb(lo_nib, voff);

        // Extract high nibbles: (qs >> 4) - 8
        HVX_Vector hi_nib = Q6_Vub_vlsr_VubR(qs_raw, 4);
        HVX_Vector hi_val = Q6_Vb_vsub_VbVb(hi_nib, voff);

        // Load q8 values: first 16 bytes and next 16 bytes
        HVX_Vector q8_lo = Q6_V_vand_QV(p16, *(const HVX_UVector *)y[ib].qs);
        HVX_Vector q8_hi = Q6_V_vand_QV(p16, *(const HVX_UVector *)(y[ib].qs + 16));

        // vrmpy: for each 4-byte group, sum of signed byte products -> int32
        HVX_Vector rsum_lo = Q6_Vw_vrmpy_VbVb(lo_val, q8_lo);
        HVX_Vector rsum_hi = Q6_Vw_vrmpy_VbVb(hi_val, q8_hi);

        // Horizontal sum of 4 int32 values from each
        int32_t __attribute__((aligned(128))) tmp_lo[32];
        int32_t __attribute__((aligned(128))) tmp_hi[32];
        *(HVX_Vector *)tmp_lo = rsum_lo;
        *(HVX_Vector *)tmp_hi = rsum_hi;

        int32_t sumi = 0;
        for (int j = 0; j < 4; ++j) {
            sumi += tmp_lo[j] + tmp_hi[j];
        }

        const float d = ggml_compute_fp16_to_fp32(x[ib].d) * ggml_compute_fp16_to_fp32(y[ib].d);
        sumf += (float)sumi * d;
    }

    *s = sumf;
}

void vec_dot_q8_0_q8_0_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, size_t bx,
                    const void *GGML_RESTRICT vy, size_t by, int nrc) {
    const block_q8_0 *GGML_RESTRICT x = (const block_q8_0 *)vx;
    const block_q8_0 *GGML_RESTRICT y = (const block_q8_0 *)vy;
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(nrc);

    const int qk = QK8_0;
    const int nb = n / qk;

    float sumf = 0;
    for (int ib = 0; ib < nb; ++ib) {
        int sumi = 0;

        for (int j = 0; j < qk; ++j) {
            sumi += (x[ib].qs[j] * y[ib].qs[j]);
        }

        const float d = ggml_compute_fp16_to_fp32(x[ib].d) * ggml_compute_fp16_to_fp32(y[ib].d);
        sumf += (float)sumi * d;
    }
    *s = sumf;
}

void vec_dot_q8_0_q8_0_hvx(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, size_t bx,
                    const void *GGML_RESTRICT vy, size_t by, int nrc) {
    const block_q8_0 *GGML_RESTRICT x = (const block_q8_0 *)vx;
    const block_q8_0 *GGML_RESTRICT y = (const block_q8_0 *)vy;
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(nrc);

    const int qk = QK8_0;
    const int nb = n / qk;

    float sumf = 0;

    const HVX_VectorPred p32 = Q6_Q_vsetq_R(32);

    for (int ib = 0; ib < nb; ++ib) {
        // Use HVX_UVector for unaligned load (block qs at offset 2, not 128-byte aligned)
        HVX_Vector vx = Q6_V_vand_QV(p32, *(const HVX_UVector *)x[ib].qs);
        HVX_Vector vy = Q6_V_vand_QV(p32, *(const HVX_UVector *)y[ib].qs);

        // vrmpy: for each 4-byte group, sum of signed byte products -> int32
        HVX_Vector rsum = Q6_Vw_vrmpy_VbVb(vx, vy);

        // Horizontal sum of 8 int32 values
        int32_t sumi = 0;
        int32_t __attribute__((aligned(128))) tmp[32];
        *(HVX_Vector *)tmp = rsum;
        for (int j = 0; j < 8; ++j) {
            sumi += tmp[j];
        }

        const float d = ggml_compute_fp16_to_fp32(x[ib].d) * ggml_compute_fp16_to_fp32(y[ib].d);
        sumf += (float)sumi * d;
    }

    *s = sumf;
}

void vec_dot_q4_1_q8_1_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, size_t bx,
                    const void *GGML_RESTRICT vy, size_t by, int nrc) {
    const block_q4_1 *GGML_RESTRICT x = (const block_q4_1 *)vx;
    const block_q8_1 *GGML_RESTRICT y = (const block_q8_1 *)vy;
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(nrc);

    const int qk = QK4_1;
    const int nb = n / qk;

    float sumf = 0;
    for (int ib = 0; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F);
            const int v1 = (x[ib].qs[j] >>   4);

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        const float d  = ggml_compute_fp16_to_fp32(x[ib].d) * ggml_compute_fp16_to_fp32(y[ib].d);
        const float m  = ggml_compute_fp16_to_fp32(x[ib].m) * ggml_compute_fp16_to_fp32(y[ib].s);
        sumf += d * (sumi0 + sumi1) + m;
    }
    *s = sumf;
}

void vec_dot_q4_1_q8_1_hvx(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, size_t bx,
                    const void *GGML_RESTRICT vy, size_t by, int nrc) {
    const block_q4_1 *GGML_RESTRICT x = (const block_q4_1 *)vx;
    const block_q8_1 *GGML_RESTRICT y = (const block_q8_1 *)vy;
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(nrc);

    const int qk = QK4_1;
    const int nb = n / qk;

    float sumf = 0;

    const HVX_Vector vmask = Q6_Vb_vsplat_R(0x0F);
    const HVX_VectorPred p16 = Q6_Q_vsetq_R(16);

    for (int ib = 0; ib < nb; ++ib) {
        // Use HVX_UVector for unaligned load
        HVX_Vector qs_raw = Q6_V_vand_QV(p16, *(const HVX_UVector *)x[ib].qs);

        // Extract low nibbles: qs & 0x0F (no offset subtraction for q4_1)
        HVX_Vector lo_nib = Q6_V_vand_VV(qs_raw, vmask);

        // Extract high nibbles: qs >> 4
        HVX_Vector hi_nib = Q6_Vub_vlsr_VubR(qs_raw, 4);

        // Load q8 values: first 16 bytes and next 16 bytes
        HVX_Vector q8_lo = Q6_V_vand_QV(p16, *(const HVX_UVector *)y[ib].qs);
        HVX_Vector q8_hi = Q6_V_vand_QV(p16, *(const HVX_UVector *)(y[ib].qs + 16));

        // vrmpy: for each 4-byte group, sum of unsigned*signed byte products -> int32
        // q4_1 nibbles are unsigned (0-15), q8_1 values are signed
        HVX_Vector rsum_lo = Q6_Vw_vrmpy_VubVb(lo_nib, q8_lo);
        HVX_Vector rsum_hi = Q6_Vw_vrmpy_VubVb(hi_nib, q8_hi);

        // Horizontal sum of 4 int32 values from each
        int32_t __attribute__((aligned(128))) tmp_lo[32];
        int32_t __attribute__((aligned(128))) tmp_hi[32];
        *(HVX_Vector *)tmp_lo = rsum_lo;
        *(HVX_Vector *)tmp_hi = rsum_hi;

        int32_t sumi = 0;
        for (int j = 0; j < 4; ++j) {
            sumi += tmp_lo[j] + tmp_hi[j];
        }

        // Q4_1 formula: sumf += d_x * d_y * sumi + m_x * s_y
        const float d = ggml_compute_fp16_to_fp32(x[ib].d) * ggml_compute_fp16_to_fp32(y[ib].d);
        const float m = ggml_compute_fp16_to_fp32(x[ib].m) * ggml_compute_fp16_to_fp32(y[ib].s);
        sumf += d * sumi + m;
    }

    *s = sumf;
}

void vec_dot_q5_0_q8_0_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, size_t bx,
                    const void *GGML_RESTRICT vy, size_t by, int nrc) {
    const block_q5_0 *GGML_RESTRICT x = (const block_q5_0 *)vx;
    const block_q8_0 *GGML_RESTRICT y = (const block_q8_0 *)vy;
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(nrc);

    const int qk = QK5_0;
    const int nb = n / qk;

    float sumf = 0;

    for (int ib = 0; ib < nb; ++ib) {
        uint32_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));

        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh & (1u << (j + 0 ))) >> (j + 0 )) << 4;
            const uint8_t xh_1 = ((qh & (1u << (j + 16))) >> (j + 12));

            const int32_t x0 = (int8_t)(((x[ib].qs[j] & 0x0F) | xh_0) - 16);
            const int32_t x1 = (int8_t)(((x[ib].qs[j] >>   4) | xh_1) - 16);

            sumi0 += (x0 * y[ib].qs[j]);
            sumi1 += (x1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += ggml_compute_fp16_to_fp32(x[ib].d) * ggml_compute_fp16_to_fp32(y[ib].d) * sumi;
    }

    *s = sumf;
}

void vec_dot_q5_1_q8_1_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, size_t bx,
                    const void *GGML_RESTRICT vy, size_t by, int nrc) {
    const block_q5_1 *GGML_RESTRICT x = (const block_q5_1 *)vx;
    const block_q8_1 *GGML_RESTRICT y = (const block_q8_1 *)vy;
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(nrc);

    const int qk = QK5_1;
    const int nb = n / qk;

    float sumf = 0;

    for (int ib = 0; ib < nb; ++ib) {
        uint32_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));

        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            const int32_t x0 = (x[ib].qs[j] & 0xF) | xh_0;
            const int32_t x1 = (x[ib].qs[j] >>  4) | xh_1;

            sumi0 += (x0 * y[ib].qs[j]);
            sumi1 += (x1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += ggml_compute_fp16_to_fp32(x[ib].d) * ggml_compute_fp16_to_fp32(y[ib].d) * sumi
              + ggml_compute_fp16_to_fp32(x[ib].m) * ggml_compute_fp16_to_fp32(y[ib].s);
    }

    *s = sumf;
}

void vec_dot_iq4_nl_q8_0_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, size_t bx,
                    const void *GGML_RESTRICT vy, size_t by, int nrc) {
    const block_iq4_nl *GGML_RESTRICT x = (const block_iq4_nl *)vx;
    const block_q8_0 *GGML_RESTRICT y = (const block_q8_0 *)vy;
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(nrc);

    const int qk = QK4_NL;
    const int nb = n / qk;

    float sumf = 0;

    for (int ib = 0; ib < nb; ++ib) {
        const float d = ggml_compute_fp16_to_fp32(y[ib].d) * ggml_compute_fp16_to_fp32(x[ib].d);
        int sumi1 = 0, sumi2 = 0;
        for (int j = 0; j < qk/2; ++j) {
            sumi1 += y[ib].qs[j+  0] * kvalues_iq4nl[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j+qk/2] * kvalues_iq4nl[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }

    *s = sumf;
}

// BF16 dot product: convert both to F32 and accumulate
void vec_dot_bf16_bf16_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx,
                               size_t bx, const void *GGML_RESTRICT vy, size_t by, int nrc) {
    const ggml_bf16_t *GGML_RESTRICT x = (const ggml_bf16_t *)vx;
    const ggml_bf16_t *GGML_RESTRICT y = (const ggml_bf16_t *)vy;
    UNUSED(bs); UNUSED(bx); UNUSED(by); UNUSED(nrc);
    float sumf = 0;
    for (int i = 0; i < n; ++i) {
        sumf += ggml_compute_bf16_to_fp32(x[i]) * ggml_compute_bf16_to_fp32(y[i]);
    }
    *s = sumf;
}

// Q6_K x Q8_K dot product
void vec_dot_q6_K_q8_K_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx,
                                size_t bx, const void *GGML_RESTRICT vy, size_t by, int nrc) {
    const block_q6_K *GGML_RESTRICT x = (const block_q6_K *)vx;
    const block_q8_K *GGML_RESTRICT y = (const block_q8_K *)vy;
    UNUSED(bs); UNUSED(bx); UNUSED(by); UNUSED(nrc);
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums[8];
    int32_t aux32[8];
    memset(sums, 0, 8 * sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8 * sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l +  0] = (int8_t)((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a  += 128;
            q4 += 64;
            qh += 32;
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = ggml_compute_fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

// Q4_K x Q8_K dot product
void vec_dot_q4_K_q8_K_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx,
                                size_t bx, const void *GGML_RESTRICT vy, size_t by, int nrc) {
    const block_q4_K *GGML_RESTRICT x = (const block_q4_K *)vx;
    const block_q8_K *GGML_RESTRICT y = (const block_q8_K *)vy;
    UNUSED(bs); UNUSED(bx); UNUSED(by); UNUSED(nrc);
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];
    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums[8];
    int32_t aux32[8];
    memset(sums, 0, 8 * sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8 * sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            a += 32; q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = ggml_compute_fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = ggml_compute_fp16_to_fp32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

// Q2_K x Q8_K dot product
void vec_dot_q2_K_q8_K_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx,
                                size_t bx, const void *GGML_RESTRICT vy, size_t by, int nrc) {
    const block_q2_K *GGML_RESTRICT x = (const block_q2_K *)vx;
    const block_q8_K *GGML_RESTRICT y = (const block_q8_K *)vy;
    UNUSED(bs); UNUSED(bx); UNUSED(by); UNUSED(nrc);
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        const uint8_t * GGML_RESTRICT sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < 16; ++j) {
            summs += y[i].bsums[j] * (sc[j] >> 4);
        }

        const float dall = y[i].d * ggml_compute_fp16_to_fp32(x[i].d);
        const float dmin = y[i].d * ggml_compute_fp16_to_fp32(x[i].dmin);

        int isum = 0;
        int is = 0;
        for (int k = 0; k < QK_K/128; ++k) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                int d = sc[is++] & 0xF;
                int isuml = 0;
                for (int l = 0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                d = sc[is++] & 0xF;
                isuml = 0;
                for (int l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * isum - dmin * summs;
    }
    *s = sumf;
}

// Q3_K x Q8_K dot product
void vec_dot_q3_K_q8_K_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx,
                                size_t bx, const void *GGML_RESTRICT vy, size_t by, int nrc) {
    const block_q3_K *GGML_RESTRICT x = (const block_q3_K *)vx;
    const block_q8_K *GGML_RESTRICT y = (const block_q8_K *)vy;
    UNUSED(bs); UNUSED(bx); UNUSED(by); UNUSED(nrc);
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x03030303;
    static const uint32_t kmask2 = 0x0f0f0f0f;

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums[8];
    int32_t aux32[8];
    memset(sums, 0, 8 * sizeof(float));

    uint32_t auxs[4];
    const int8_t * scales = (const int8_t*)auxs;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint8_t * GGML_RESTRICT hm = x[i].hmask;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8 * sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K/128; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            q3 += 32;
        }
        a = aux8;

        memcpy(auxs, x[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = ggml_compute_fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

// Q5_K x Q8_K dot product
void vec_dot_q5_K_q8_K_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx,
                                size_t bx, const void *GGML_RESTRICT vy, size_t by, int nrc) {
    const block_q5_K *GGML_RESTRICT x = (const block_q5_K *)vx;
    const block_q8_K *GGML_RESTRICT y = (const block_q8_K *)vy;
    UNUSED(bs); UNUSED(bx); UNUSED(by); UNUSED(nrc);
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];
    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums[8];
    int32_t aux32[8];
    memset(sums, 0, 8 * sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const uint8_t * GGML_RESTRICT hm = x[i].qh;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8 * sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = ggml_compute_fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = ggml_compute_fp16_to_fp32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

// MXFP4 x Q8_0 dot product
void vec_dot_mxfp4_q8_0_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, size_t bx,
                                 const void *GGML_RESTRICT vy, size_t by, int nrc) {
    const block_mxfp4 *GGML_RESTRICT x = (const block_mxfp4 *)vx;
    const block_q8_0 *GGML_RESTRICT y = (const block_q8_0 *)vy;
    UNUSED(bs); UNUSED(bx); UNUSED(by); UNUSED(nrc);
    assert(n % QK_MXFP4 == 0);

    const int nb = n / QK_MXFP4;
    float sumf = 0;

    for (int ib = 0; ib < nb; ++ib) {
        const float d = ggml_compute_fp16_to_fp32(y[ib].d) * ggml_e8m0_to_fp32_half(x[ib].e);

        int sumi1 = 0;
        int sumi2 = 0;
        for (int j = 0; j < QK_MXFP4/2; ++j) {
            sumi1 += y[ib].qs[j +          0] * kvalues_mxfp4[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j + QK_MXFP4/2] * kvalues_mxfp4[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}

// NVFP4 x Q8_0 dot product
void vec_dot_nvfp4_q8_0_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, size_t bx,
                                 const void *GGML_RESTRICT vy, size_t by, int nrc) {
    const block_nvfp4 *GGML_RESTRICT x = (const block_nvfp4 *)vx;
    const block_q8_0 *GGML_RESTRICT y = (const block_q8_0 *)vy;
    UNUSED(bs); UNUSED(bx); UNUSED(by); UNUSED(nrc);
    assert(n % QK_NVFP4 == 0);

    const int nb = n / QK_NVFP4;
    float sumf = 0;

    for (int ib = 0; ib < nb; ++ib) {
        for (int s_idx = 0; s_idx < 4; ++s_idx) {
            const float d = ggml_ue4m3_to_fp32(x[ib].d[s_idx]);
            const int q8_block = s_idx / 2;
            const int q8_off   = (s_idx % 2) * QK_NVFP4_SUB;
            const float dy = ggml_compute_fp16_to_fp32(y[2*ib + q8_block].d);

            int sumi_lo = 0, sumi_hi = 0;
            for (int j = 0; j < QK_NVFP4_SUB/2; ++j) {
                const uint8_t qv = x[ib].qs[s_idx*(QK_NVFP4_SUB/2) + j];
                sumi_lo += y[2*ib + q8_block].qs[q8_off + j +               0] * kvalues_mxfp4[qv & 0xf];
                sumi_hi += y[2*ib + q8_block].qs[q8_off + j + QK_NVFP4_SUB/2] * kvalues_mxfp4[qv >>  4];
            }
            sumf += dy * d * (sumi_lo + sumi_hi);
        }
    }
    *s = sumf;
}

// IQ4_XS x Q8_K dot product
void vec_dot_iq4_xs_q8_K_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, size_t bx,
                                  const void *GGML_RESTRICT vy, size_t by, int nrc) {
    const block_iq4_xs *GGML_RESTRICT x = (const block_iq4_xs *)vx;
    const block_q8_K *GGML_RESTRICT y = (const block_q8_K *)vy;
    UNUSED(bs); UNUSED(bx); UNUSED(by); UNUSED(nrc);
    assert(n % QK_K == 0);

    const int nb = n / QK_K;

    float sumf = 0;
    for (int ibl = 0; ibl < nb; ++ibl) {
        const float d4d8 = ggml_compute_fp16_to_fp32(x[ibl].d) * y[ibl].d;
        uint16_t h = x[ibl].scales_h;
        const uint8_t * qs = x[ibl].qs;
        const int8_t  * q8 = y[ibl].qs;
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const uint8_t ls1 = (x[ibl].scales_l[ib/2] & 0xf) | ((h << 4) & 0x30);
            const uint8_t ls2 = (x[ibl].scales_l[ib/2] >>  4) | ((h << 2) & 0x30);
            h >>= 4;
            const float d1 = d4d8 * (ls1 - 32);
            const float d2 = d4d8 * (ls2 - 32);
            int sumi1 = 0, sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j+16] * kvalues_iq4nl[qs[j] >>  4];
            }
            sumf += d1 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
            sumi1 = sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j+16] * kvalues_iq4nl[qs[j] >>  4];
            }
            sumf += d2 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
        }
    }
    *s = sumf;
}

// IQ3_XXS x Q8_K dot product
void vec_dot_iq3_xxs_q8_K_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, size_t bx,
                                   const void *GGML_RESTRICT vy, size_t by, int nrc) {
    const block_iq3_xxs *GGML_RESTRICT x = (const block_iq3_xxs *)vx;
    const block_q8_K *GGML_RESTRICT y = (const block_q8_K *)vy;
    UNUSED(bs); UNUSED(bx); UNUSED(by); UNUSED(nrc);
    assert(n % QK_K == 0);

    const int nb = n / QK_K;
    uint32_t aux32;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = ggml_compute_fp16_to_fp32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint8_t * GGML_RESTRICT gas = x[i].qs + QK_K/4;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(&aux32, gas, sizeof(uint32_t)); gas += sizeof(uint32_t);
            const uint32_t ls = 2*(aux32 >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*l+0]);
                const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*l+1]);
                const uint8_t  signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            q3 += 8;
            bsum += sumi * ls;
        }
        sumf += d * bsum;
    }
    *s = 0.25f * sumf;
}

// IQ2_XXS x Q8_K dot product
void vec_dot_iq2_xxs_q8_K_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, size_t bx,
                                   const void *GGML_RESTRICT vy, size_t by, int nrc) {
    const block_iq2_xxs *GGML_RESTRICT x = (const block_iq2_xxs *)vx;
    const block_q8_K *GGML_RESTRICT y = (const block_q8_K *)vy;
    UNUSED(bs); UNUSED(bx); UNUSED(by); UNUSED(nrc);
    assert(n % QK_K == 0);

    const int nb = n / QK_K;
    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = ggml_compute_fp16_to_fp32(x[i].d) * y[i].d;
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(aux32, q2, 2*sizeof(uint32_t));
            q2 += 4;
            const uint32_t ls = 2*(aux32[1] >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;
}

// IQ2_XS x Q8_K dot product
void vec_dot_iq2_xs_q8_K_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, size_t bx,
                                  const void *GGML_RESTRICT vy, size_t by, int nrc) {
    const block_iq2_xs *GGML_RESTRICT x = (const block_iq2_xs *)vx;
    const block_q8_K *GGML_RESTRICT y = (const block_q8_K *)vy;
    UNUSED(bs); UNUSED(bx); UNUSED(by); UNUSED(nrc);
    assert(n % QK_K == 0);

    const int nb = n / QK_K;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = ggml_compute_fp16_to_fp32(x[i].d) * y[i].d;
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        const uint8_t  * GGML_RESTRICT sc = x[i].scales;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            const uint16_t ls1 = 2*(sc[ib32] & 0xf) + 1;
            const uint16_t ls2 = 2*(sc[ib32] >>  4) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 2; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
                const uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls1;
            sumi = 0;
            for (int l = 2; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
                const uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls2;
            q2 += 4;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;
}

// IQ2_S x Q8_K dot product
void vec_dot_iq2_s_q8_K_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, size_t bx,
                                 const void *GGML_RESTRICT vy, size_t by, int nrc) {
    const block_iq2_s *GGML_RESTRICT x = (const block_iq2_s *)vx;
    const block_q8_K *GGML_RESTRICT y = (const block_q8_K *)vy;
    UNUSED(bs); UNUSED(bx); UNUSED(by); UNUSED(nrc);
    assert(n % QK_K == 0);

    const int nb = n / QK_K;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = ggml_compute_fp16_to_fp32(x[i].d) * y[i].d;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;
        const uint8_t  * GGML_RESTRICT qs = x[i].qs;
        const uint8_t  * GGML_RESTRICT qh = x[i].qh;
        const uint8_t  * GGML_RESTRICT signs = qs + QK_K/8;

        int bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            int ls1 = 1 + 2*(x[i].scales[ib32] & 0xf);
            int ls2 = 1 + 2*(x[i].scales[ib32] >>  4);
            int sumi1 = 0, sumi2 = 0;
            for (int l = 0; l < 2; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    sumi1 += q8[j] * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            for (int l = 2; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    sumi2 += q8[j] * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += ls1 * sumi1 + ls2 * sumi2;
            qs += 4;
            signs += 4;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;
}

// IQ1_S x Q8_K dot product
void vec_dot_iq1_s_q8_K_generic(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, size_t bx,
                                 const void *GGML_RESTRICT vy, size_t by, int nrc) {
    const block_iq1_s *GGML_RESTRICT x = (const block_iq1_s *)vx;
    const block_q8_K *GGML_RESTRICT y = (const block_q8_K *)vy;
    UNUSED(bs); UNUSED(bx); UNUSED(by); UNUSED(nrc);
    assert(n % QK_K == 0);

    const int nb = n / QK_K;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;
        const uint8_t  * GGML_RESTRICT qs = x[i].qs;
        const uint16_t * GGML_RESTRICT qh = x[i].qh;

        int sumi = 0, sumi1 = 0;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const int ls = 2*((qh[ib] >> 12) & 7) + 1;
            const int delta = qh[ib] & 0x8000 ? -1 : 1;
            int lsum = 0;
            for (int l = 0; l < 4; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + (qs[l] | (((qh[ib] >> 3*l) & 7) << 8)));
                for (int j = 0; j < 8; ++j) {
                    lsum += q8[j] * grid[j];
                }
                q8 += 8;
            }
            sumi  += ls * lsum;
            sumi1 += ls * delta * (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]);
            qs += 4;
        }

        sumf += ggml_compute_fp16_to_fp32(x[i].d) * y[i].d * (sumi + IQ1S_DELTA * sumi1);
    }
    *s = sumf;
}

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

static void quantize_row_q8_0_generic(const float * x, block_q8_0 * y, int n) {
    const int nb = n / QK8_0;
    for (int i = 0; i < nb; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; ++j) {
            const float v = x[i * QK8_0 + j];
            amax = MAX(amax, fabsf(v));
        }
        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;
        y[i].d = ggml_compute_fp32_to_fp16(d);
        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = x[i * QK8_0 + j] * id;
            y[i].qs[j] = roundf(x0);
        }
    }
}

void quantize_row_q8_0_hvx(const float * x, block_q8_0 * y, int n) {
    const int nb = n / QK8_0;

    for (int i = 0; i < nb; ++i) {
        const float * src = x + i * QK8_0;
        int8_t * dst_qs = y[i].qs;

        // Compute amax using HVX if aligned
        float amax = 0.0f;
        if (((uintptr_t)src & 0x7F) == 0) {
            const HVX_Vector * restrict vsrc = (const HVX_Vector *)src;
            HVX_Vector vabs_max = Q6_V_vsplat_R(0);
            for (int j = 0; j < QK8_0 / VLEN_FP32; ++j) {
                HVX_Vector v = vsrc[j];
                HVX_Vector vabs = hvx_vec_abs_f32(v);
                vabs_max = Q6_Vsf_vmax_VsfVsf(vabs_max, vabs);
            }
            vabs_max = hvx_vec_reduce_max_f32(vabs_max);
            amax = hvx_vec_get_f32(vabs_max);
        } else {
            for (int j = 0; j < QK8_0; ++j) {
                amax = MAX(amax, fabsf(src[j]));
            }
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;
        y[i].d = ggml_compute_fp32_to_fp16(d);

        // Quantize using scalar (fp16 intermediate in HVX path causes precision issues)
        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = src[j] * id;
            dst_qs[j] = roundf(x0);
        }
    }
}

static void quantize_row_q8_1_generic(const float * x, block_q8_1 * y, int n) {
    const int nb = n / QK8_1;

    for (int i = 0; i < nb; ++i) {
        const float * src = x + i * QK8_1;

        float amax = 0.0f;
        for (int j = 0; j < QK8_1; ++j) {
            amax = MAX(amax, fabsf(src[j]));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = ggml_compute_fp32_to_fp16(d);

        int sum = 0;
        for (int j = 0; j < QK8_1 / 2; ++j) {
            const float v0 = src[j] * id;
            const float v1 = src[QK8_1 / 2 + j] * id;

            y[i].qs[j] = roundf(v0);
            y[i].qs[QK8_1 / 2 + j] = roundf(v1);

            sum += y[i].qs[j];
            sum += y[i].qs[QK8_1 / 2 + j];
        }

        y[i].s = ggml_compute_fp32_to_fp16(sum * d);
    }
}

void quantize_row_q8_1_hvx(const float * x, block_q8_1 * y, int n) {
    const int nb = n / QK8_1;

    for (int i = 0; i < nb; ++i) {
        const float * src = x + i * QK8_1;
        int8_t * dst_qs = y[i].qs;

        // Compute amax using HVX if aligned
        float amax = 0.0f;
        if (((uintptr_t)src & 0x7F) == 0) {
            const HVX_Vector * restrict vsrc = (const HVX_Vector *)src;
            HVX_Vector vabs_max = Q6_V_vsplat_R(0);
            for (int j = 0; j < QK8_1 / VLEN_FP32; ++j) {
                HVX_Vector v = vsrc[j];
                HVX_Vector vabs = hvx_vec_abs_f32(v);
                vabs_max = Q6_Vsf_vmax_VsfVsf(vabs_max, vabs);
            }
            vabs_max = hvx_vec_reduce_max_f32(vabs_max);
            amax = hvx_vec_get_f32(vabs_max);
        } else {
            for (int j = 0; j < QK8_1; ++j) {
                amax = MAX(amax, fabsf(src[j]));
            }
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;
        y[i].d = ggml_compute_fp32_to_fp16(d);

        int sum = 0;
        for (int j = 0; j < QK8_1; ++j) {
            const float x0 = src[j] * id;
            const int8_t v = (int8_t)roundf(x0);
            dst_qs[j] = v;
            sum += v;
        }

        y[i].s = ggml_compute_fp32_to_fp16(sum * d);
    }
}

static void quantize_row_q5_0_generic(const float * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int n) {
    const int nb = n / QK5_0;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        float max  = 0.0f;

        for (int j = 0; j < QK5_0; j++) {
            const float v = x[i*QK5_0 + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -16;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = ggml_compute_fp32_to_fp16(d);

        uint32_t qh = 0;

        for (int j = 0; j < QK5_0/2; ++j) {
            const float x0 = x[i*QK5_0 + 0      + j]*id;
            const float x1 = x[i*QK5_0 + QK5_0/2 + j]*id;

            const uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
            const uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

            y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
        }

        memcpy(&y[i].qh, &qh, sizeof(qh));
    }
}

void quantize_row_q5_0_hvx(const float * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int n) {
    const int nb = n / QK5_0;

    for (int i = 0; i < nb; i++) {
        const float * src = x + i * QK5_0;

        // Compute amax using HVX if aligned
        float amax = 0.0f;
        float max  = 0.0f;
        if (((uintptr_t)src & 0x7F) == 0) {
            const HVX_Vector * restrict vsrc = (const HVX_Vector *)src;
            HVX_Vector vabs_max = Q6_V_vsplat_R(0);
            HVX_Vector vmax_val = Q6_V_vsplat_R(0);
            for (int j = 0; j < QK5_0 / VLEN_FP32; ++j) {
                HVX_Vector v = vsrc[j];
                HVX_Vector vabs = hvx_vec_abs_f32(v);
                HVX_VectorPred p = Q6_Q_vcmp_gt_VsfVsf(vabs, vabs_max);
                vabs_max = Q6_V_vmux_QVV(p, vabs, vabs_max);
                vmax_val = Q6_V_vmux_QVV(p, v, vmax_val);
            }
            vabs_max = hvx_vec_reduce_max_f32(vabs_max);
            amax = hvx_vec_get_f32(vabs_max);
            // Find the value corresponding to amax
            for (int j = 0; j < QK5_0; ++j) {
                if (fabsf(src[j]) == amax) { max = src[j]; break; }
            }
        } else {
            for (int j = 0; j < QK5_0; j++) {
                const float v = src[j];
                if (amax < fabsf(v)) {
                    amax = fabsf(v);
                    max  = v;
                }
            }
        }

        const float d  = max / -16;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = ggml_compute_fp32_to_fp16(d);

        uint32_t qh = 0;

        for (int j = 0; j < QK5_0/2; ++j) {
            const float x0 = src[0      + j]*id;
            const float x1 = src[QK5_0/2 + j]*id;

            const uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
            const uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

            y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
        }

        memcpy(&y[i].qh, &qh, sizeof(qh));
    }
}

static void quantize_row_q5_1_generic(const float * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int n) {
    const int nb = n / QK5_1;

    for (int i = 0; i < nb; i++) {
        float min = 1e30f;
        float max = -1e30f;

        for (int j = 0; j < QK5_1; j++) {
            const float v = x[i*QK5_1 + j];
            if (v < min) min = v;
            if (v > max) max = v;
        }

        const float d  = (max - min) / ((1 << 5) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = ggml_compute_fp32_to_fp16(d);
        y[i].m = ggml_compute_fp32_to_fp16(min);

        uint32_t qh = 0;

        for (int j = 0; j < QK5_1/2; ++j) {
            const float x0 = (x[i*QK5_1 + 0      + j] - min)*id;
            const float x1 = (x[i*QK5_1 + QK5_1/2 + j] - min)*id;

            const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
            const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

            y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
        }

        memcpy(&y[i].qh, &qh, sizeof(y[i].qh));
    }
}

void quantize_row_q5_1_hvx(const float * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int n) {
    const int nb = n / QK5_1;

    for (int i = 0; i < nb; i++) {
        const float * src = x + i * QK5_1;

        // Compute min/max using HVX if aligned
        float min = 1e30f;
        float max = -1e30f;
        if (((uintptr_t)src & 0x7F) == 0) {
            const HVX_Vector * restrict vsrc = (const HVX_Vector *)src;
            HVX_Vector vmin = hvx_vec_splat_f32(1e30f);
            HVX_Vector vmax = hvx_vec_splat_f32(-1e30f);
            for (int j = 0; j < QK5_1 / VLEN_FP32; ++j) {
                HVX_Vector v = vsrc[j];
                vmin = Q6_Vsf_vmin_VsfVsf(vmin, v);
                vmax = Q6_Vsf_vmax_VsfVsf(vmax, v);
            }
            vmin = hvx_vec_reduce_min_f32(vmin);
            vmax = hvx_vec_reduce_max_f32(vmax);
            min = hvx_vec_get_f32(vmin);
            max = hvx_vec_get_f32(vmax);
        } else {
            for (int j = 0; j < QK5_1; j++) {
                const float v = src[j];
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }

        const float d  = (max - min) / ((1 << 5) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = ggml_compute_fp32_to_fp16(d);
        y[i].m = ggml_compute_fp32_to_fp16(min);

        uint32_t qh = 0;

        for (int j = 0; j < QK5_1/2; ++j) {
            const float x0 = (src[0      + j] - min)*id;
            const float x1 = (src[QK5_1/2 + j] - min)*id;

            const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
            const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

            y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
        }

        memcpy(&y[i].qh, &qh, sizeof(y[i].qh));
    }
}

static void quantize_row_iq4_nl_generic(const float * GGML_RESTRICT x, block_iq4_nl * GGML_RESTRICT y, int n) {
    const int nb = n / QK4_NL;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK4_NL; ++j) {
            const float v = x[i * QK4_NL + j];
            amax = MAX(amax, fabsf(v));
        }
        const float d = amax / ((1 << 4) - 1);
        const float id = d ? 1.0f/d : 0.0f;
        y[i].d = ggml_compute_fp32_to_fp16(d);
        for (int j = 0; j < QK4_NL/2; ++j) {
            const float v0 = x[i*QK4_NL + 0         + j] * id;
            const float v1 = x[i*QK4_NL + QK4_NL/2  + j] * id;
            const uint8_t vi0 = MIN(15, (int8_t)(v0 + 8.5f));
            const uint8_t vi1 = MIN(15, (int8_t)(v1 + 8.5f));
            y[i].qs[j] = vi0 | (vi1 << 4);
        }
    }
}

void quantize_row_iq4_nl_hvx(const float * GGML_RESTRICT x, block_iq4_nl * GGML_RESTRICT y, int n) {
    const int nb = n / QK4_NL;

    for (int i = 0; i < nb; i++) {
        const float * src = x + i * QK4_NL;

        // Compute amax using HVX if aligned
        float amax = 0.0f;
        if (((uintptr_t)src & 0x7F) == 0) {
            const HVX_Vector * restrict vsrc = (const HVX_Vector *)src;
            HVX_Vector vabs_max = Q6_V_vsplat_R(0);
            for (int j = 0; j < QK4_NL / VLEN_FP32; ++j) {
                HVX_Vector v = vsrc[j];
                HVX_Vector vabs = hvx_vec_abs_f32(v);
                vabs_max = Q6_Vsf_vmax_VsfVsf(vabs_max, vabs);
            }
            vabs_max = hvx_vec_reduce_max_f32(vabs_max);
            amax = hvx_vec_get_f32(vabs_max);
        } else {
            for (int j = 0; j < QK4_NL; ++j) {
                amax = MAX(amax, fabsf(src[j]));
            }
        }

        const float d  = amax / ((1 << 4) - 1);
        const float id = d ? 1.0f/d : 0.0f;
        y[i].d = ggml_compute_fp32_to_fp16(d);

        for (int j = 0; j < QK4_NL/2; ++j) {
            const float v0 = src[0         + j] * id;
            const float v1 = src[QK4_NL/2  + j] * id;
            const uint8_t vi0 = MIN(15, (int8_t)(v0 + 8.5f));
            const uint8_t vi1 = MIN(15, (int8_t)(v1 + 8.5f));
            y[i].qs[j] = vi0 | (vi1 << 4);
        }
    }
}

void quantize_f32_to_f16_row_hvx(const float * GGML_RESTRICT x, uint16_t * GGML_RESTRICT y, int n) {
    const int fp32_per_vec = VLEN / sizeof(float);  // 32
    const int fp16_per_vec = VLEN / sizeof(uint16_t); // 64

    // scalar fallback for small or unaligned cases
    if (n < fp16_per_vec || ((uintptr_t)x & 0x3F) != 0 || ((uintptr_t)y & 0x7F) != 0) {
        for (int i = 0; i < n; ++i) {
            y[i] = ggml_compute_fp32_to_fp16(x[i]);
        }
        return;
    }

    const int nvec = n / fp16_per_vec;
    const int nloe = n % fp16_per_vec;

    const HVX_Vector * restrict vx = (const HVX_Vector *)x;
    HVX_Vector * restrict vy = (HVX_Vector *)y;

    for (int i = 0; i < nvec; ++i) {
        HVX_Vector v0 = vx[2 * i];
        HVX_Vector v1 = vx[2 * i + 1];
        vy[i] = hvx_vec_f32_to_f16(v0, v1);
    }

    if (nloe > 0) {
        const float * tail_x = x + nvec * fp16_per_vec;
        uint16_t * tail_y = y + nvec * fp16_per_vec;
        for (int i = 0; i < nloe; ++i) {
            tail_y[i] = ggml_compute_fp32_to_fp16(tail_x[i]);
        }
    }
}

// Quantize F32 to Q8_K (256-element super-block)
static void quantize_row_q8_K_generic(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int n) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    for (int i = 0; i < nb; i++) {
        float amax = 0;
        float max  = 0;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax; max = x[j];
            }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, QK_K);
            x += QK_K;
            continue;
        }
        const float iscale = -127.f / max;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int(iscale * x[j]);
            y[i].qs[j] = MIN(127, v);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j*16 + ii];
            }
            y[i].bsums[j] = sum;
        }
        y[i].d = 1.0f / iscale;
        x += QK_K;
    }
}

void quantize_row_q8_K_hvx(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int n) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    for (int i = 0; i < nb; i++) {
        const float * src = x + i * QK_K;

        // Compute amax and max using HVX if aligned
        float amax = 0;
        float max  = 0;
        if (((uintptr_t)src & 0x7F) == 0) {
            const HVX_Vector * restrict vsrc = (const HVX_Vector *)src;
            HVX_Vector vabs_max = Q6_V_vsplat_R(0);
            HVX_Vector vmax_val = Q6_V_vsplat_R(0);
            for (int j = 0; j < QK_K / VLEN_FP32; ++j) {
                HVX_Vector v = vsrc[j];
                HVX_Vector vabs = hvx_vec_abs_f32(v);
                HVX_VectorPred p = Q6_Q_vcmp_gt_VsfVsf(vabs, vabs_max);
                vabs_max = Q6_V_vmux_QVV(p, vabs, vabs_max);
                vmax_val = Q6_V_vmux_QVV(p, v, vmax_val);
            }
            vabs_max = hvx_vec_reduce_max_f32(vabs_max);
            amax = hvx_vec_get_f32(vabs_max);
            // find the value corresponding to amax
            for (int j = 0; j < QK_K; ++j) {
                if (fabsf(src[j]) == amax) { max = src[j]; break; }
            }
        } else {
            for (int j = 0; j < QK_K; ++j) {
                float ax = fabsf(src[j]);
                if (ax > amax) {
                    amax = ax; max = src[j];
                }
            }
        }

        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, QK_K);
            continue;
        }

        const float iscale = -127.f / max;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int(iscale * src[j]);
            y[i].qs[j] = MIN(127, v);
        }

        for (int j = 0; j < QK_K/16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j*16 + ii];
            }
            y[i].bsums[j] = sum;
        }

        y[i].d = 1.0f / iscale;
    }
}

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

    float tmp[32];

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

                for (int32_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    const int32_t row_idx = ir0 - iir0;

                    const struct ggml_type_traits_dsp * traits = ggml_get_type_traits_dsp(type);
                    if (traits->vec_dot) {
                        traits->vec_dot(ne00, &tmp[row_idx], 0,
                                        src0_row + ir0 * nb01, 0,
                                        src1_col, 0, 1);
                    }
                }

                for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
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
    //GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );

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

    //GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
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

    const size_t max_rows_in_vtcm = (vtcm_size / sizeof(float)) / ne00;
    const int32_t rows_per_vtcm_block = MIN(max_rows_in_vtcm, VTCM_BLOCK_ROWS);

    float tmp[32];

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

                    for (int32_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < iir0_end; ir0 += num_rows_per_vec_dot) {
                        const int32_t row_idx = ir0 - iir0;

                        const struct ggml_type_traits_dsp * traits = ggml_get_type_traits_dsp(type);
                        if (traits->vec_dot) {
                            traits->vec_dot(ne00, &tmp[row_idx], 0,
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
    const int n_rows_tiled  = (n_rows / HMX_FP16_TILE_N_ROWS) * HMX_FP16_TILE_N_ROWS;
    const int n_tiles_per_row = n_cols / HMX_FP16_TILE_N_COLS;

    int r = 0;

    // Process tiled rows using HVX vector operations (like official backend)
    // Reference: htp/hmx-matmul-ops.c transfer_activation_chunk_fp32_to_fp16
    #pragma unroll(2)
    for (r = 0; r < n_rows_tiled; r += 2) {
        int r0 = r / HMX_FP16_TILE_N_ROWS;  // tile row index
        int r1 = r % HMX_FP16_TILE_N_ROWS;  // intra-tile row idx

        // Read from src: src[row * row_stride + col]
        // row is N dimension (r), col is K dimension
        const HVX_Vector *pv_in0 = (const HVX_Vector *) (src + (r + 0) * row_stride);
        const HVX_Vector *pv_in1 = (const HVX_Vector *) (src + (r + 1) * row_stride);
        for (int c = 0; c < n_cols; c += 32) {
            HVX_Vector v0 = *pv_in0++;
            HVX_Vector v1 = *pv_in1++;

            // Use HVX vector operation for fp32->fp16 conversion (same as official backend)
            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

            // compute output position
            int c0       = c / HMX_FP16_TILE_N_COLS;  // tile column index
            int tile_idx = r0 * n_tiles_per_row + c0;

            // CRITICAL: hvx_vec_f32_to_f16_shuff produces interleaved format:
            // [row0[0], row1[0], row0[1], row1[1], ...]
            // Each row pair occupies 64 fp16 elements (128 bytes) at position r1/2
            __fp16 *tile_base = vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS;
            HVX_Vector *tile_hvx = (HVX_Vector *)tile_base;
            tile_hvx[r1 / 2] = v_out;
        }
    }

    // Process remaining padded rows using scalar operations
    for (; r < n_rows_padded; r += 2) {
        int r0 = r / HMX_FP16_TILE_N_ROWS;  // tile row index
        int r1 = r % HMX_FP16_TILE_N_ROWS;  // intra-tile row idx

        const bool row0_valid = r       < n_rows;
        const bool row1_valid = (r + 1) < n_rows;

        const float *src_row0 = row0_valid ? src + (r + 0) * row_stride : NULL;
        const float *src_row1 = row1_valid ? src + (r + 1) * row_stride : NULL;

        for (int c = 0; c < n_cols; c += HMX_FP16_TILE_N_COLS) {
            int c0 = c / HMX_FP16_TILE_N_COLS;  // tile column index
            int tile_idx = r0 * n_tiles_per_row + c0;

            __fp16 *tile_base = vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS;

            // FP16 Crouton layout (interleaved format, matching hvx_vec_f32_to_f16_shuff):
            // Each row pair position (r1/2) holds 64 fp16 elements:
            // - Even positions: row0 data
            // - Odd positions: row1 data
            for (int i = 0; i < HMX_FP16_TILE_N_COLS; ++i) {
                tile_base[(r1 / 2) * 64 + i * 2] =
                    (src_row0) ? (__fp16)src_row0[c + i] : (__fp16)0;
            }
            for (int i = 0; i < HMX_FP16_TILE_N_COLS; ++i) {
                tile_base[(r1 / 2) * 64 + i * 2 + 1] =
                    (src_row1) ? (__fp16)src_row1[c + i] : (__fp16)0;
            }
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
    const int n_col_tiles = n_cols / HMX_FP16_TILE_N_COLS;
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
    const int n_col_tiles = n_cols / HMX_FP16_TILE_N_COLS;
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
    const int n_col_tiles = n_cols / HMX_FP16_TILE_N_COLS;

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
            for (int j = 0; j < HMX_FP16_TILE_N_COLS; ++j) {
                dst[(c + j) + r * col_stride] = (float)tile[row_pair * 64 + j * 2];
            }
            if (r + 1 < n_rows) {
                for (int j = 0; j < HMX_FP16_TILE_N_COLS; ++j) {
                    dst[(c + j) + (r + 1) * col_stride] = (float)tile[row_pair * 64 + j * 2 + 1];
                }
            }
        }
    }
}

static void dequantize_q4_0_to_f16_tiles(__fp16 *restrict vtcm_dst, const block_q4_0 *restrict src,
                                         int n_cols, int k) {
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int n_col_tiles = n_cols / HMX_FP16_TILE_N_COLS;
    const int n_tot_tiles = n_col_tiles * k_tiles;
    const int nb_per_col = k / QK4_0;

    // Process all tiles (matching convert_weight_f32_to_fp16_tiles structure)
    for (int t = 0; t < n_tot_tiles; ++t) {
        int ct = t / k_tiles;  // N tile index (column tile)
        int kt = t % k_tiles;  // K tile index (row tile)

        __fp16 *tile_base = vtcm_dst + t * HMX_FP16_TILE_N_ELMS;

        for (int i = 0; i < HMX_FP16_TILE_N_ROWS; ++i) {  // 32 rows per tile (N dimension)
            int row_global = ct * HMX_FP16_TILE_N_ROWS + i;  // global N index
            const block_q4_0 *col_blocks = (row_global < n_cols) ?
                                           src + row_global * nb_per_col + kt * (HMX_FP16_TILE_N_COLS / QK4_0) : NULL;

            for (int j = 0; j < HMX_FP16_TILE_N_COLS; ++j) {  // 32 columns per tile (K dimension)
                int col_global = kt * HMX_FP16_TILE_N_COLS + j;  // global K index
                int block_idx = col_global / QK4_0;
                int elem_idx = col_global % QK4_0;

                float val = 0.0f;
                if (col_blocks && col_global < k) {
                    float d = ggml_compute_fp16_to_fp32(col_blocks[block_idx].d);
                    int8_t q = (col_blocks[block_idx].qs[elem_idx / 2] >> ((elem_idx % 2) * 4)) & 0x0F;
                    val = (q - 8) * d;
                }

                // Column-pair interleaved format: tile[(j/2)*64 + i*2 + (j%2)]
                tile_base[(j / 2) * 64 + i * 2 + (j % 2)] = ggml_compute_fp32_to_fp16(val);
            }
        }
    }
}

static void dequantize_q4_1_to_f16_tiles(__fp16 *restrict vtcm_dst, const block_q4_1 *restrict src,
                                         int n_cols, int k) {
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int n_col_tiles = n_cols / HMX_FP16_TILE_N_COLS;
    const int n_tot_tiles = n_col_tiles * k_tiles;
    const int nb_per_col = k / QK4_1;

    // Process all tiles (matching convert_weight_f32_to_fp16_tiles structure)
    for (int t = 0; t < n_tot_tiles; ++t) {
        int ct = t / k_tiles;  // N tile index (column tile)
        int kt = t % k_tiles;  // K tile index (row tile)

        __fp16 *tile_base = vtcm_dst + t * HMX_FP16_TILE_N_ELMS;

        for (int i = 0; i < HMX_FP16_TILE_N_ROWS; ++i) {  // 32 rows per tile (N dimension)
            int row_global = ct * HMX_FP16_TILE_N_ROWS + i;  // global N index
            const block_q4_1 *col_blocks = (row_global < n_cols) ?
                                           src + row_global * nb_per_col + kt * (HMX_FP16_TILE_N_COLS / QK4_1) : NULL;

            for (int j = 0; j < HMX_FP16_TILE_N_COLS; ++j) {  // 32 columns per tile (K dimension)
                int col_global = kt * HMX_FP16_TILE_N_COLS + j;  // global K index
                int block_idx = col_global / QK4_1;
                int elem_idx = col_global % QK4_1;

                float val = 0.0f;
                if (col_blocks && col_global < k) {
                    float d = ggml_compute_fp16_to_fp32(col_blocks[block_idx].d);
                    float m = ggml_compute_fp16_to_fp32(col_blocks[block_idx].m);
                    int8_t q = (col_blocks[block_idx].qs[elem_idx / 2] >> ((elem_idx % 2) * 4)) & 0x0F;
                    val = q * d + m;
                }

                // Column-pair interleaved format: tile[(j/2)*64 + i*2 + (j%2)]
                tile_base[(j / 2) * 64 + i * 2 + (j % 2)] = ggml_compute_fp32_to_fp16(val);
            }
        }
    }
}

static void dequantize_q8_0_to_f16_tiles(__fp16 *restrict vtcm_dst, const block_q8_0 *restrict src,
                                         int n_cols, int k) {
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int n_col_tiles = n_cols / HMX_FP16_TILE_N_COLS;
    const int n_tot_tiles = n_col_tiles * k_tiles;
    const int nb_per_col = k / QK8_0;

    // Process all tiles (matching convert_weight_f32_to_fp16_tiles structure)
    for (int t = 0; t < n_tot_tiles; ++t) {
        int ct = t / k_tiles;  // N tile index (column tile)
        int kt = t % k_tiles;  // K tile index (row tile)

        __fp16 *tile_base = vtcm_dst + t * HMX_FP16_TILE_N_ELMS;

        for (int i = 0; i < HMX_FP16_TILE_N_ROWS; ++i) {  // 32 rows per tile (N dimension)
            int row_global = ct * HMX_FP16_TILE_N_ROWS + i;  // global N index
            const block_q8_0 *col_blocks = (row_global < n_cols) ?
                                           src + row_global * nb_per_col + kt * (HMX_FP16_TILE_N_COLS / QK8_0) : NULL;

            for (int j = 0; j < HMX_FP16_TILE_N_COLS; ++j) {  // 32 columns per tile (K dimension)
                int col_global = kt * HMX_FP16_TILE_N_COLS + j;  // global K index
                int block_idx = col_global / QK8_0;
                int elem_idx = col_global % QK8_0;

                float val = 0.0f;
                if (col_blocks && col_global < k) {
                    float d = ggml_compute_fp16_to_fp32(col_blocks[block_idx].d);
                    val = col_blocks[block_idx].qs[elem_idx] * d;
                }

                // Column-pair interleaved format: tile[(j/2)*64 + i*2 + (j%2)]
                tile_base[(j / 2) * 64 + i * 2 + (j % 2)] = ggml_compute_fp32_to_fp16(val);
            }
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
    const int n_col_tiles = n_cols / HMX_FP16_TILE_N_COLS;

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
                for (int j = 0; j < HMX_FP16_TILE_N_COLS; ++j) {
                    dst[(c + j) + r * col_stride] = (float)tile[row_pair * 64 + j * 2];
                }
            }
            if (r + 1 < end_row && r + 1 >= start_row && r + 1 < n_rows) {
                for (int j = 0; j < HMX_FP16_TILE_N_COLS; ++j) {
                    dst[(c + j) + (r + 1) * col_stride] = (float)tile[row_pair * 64 + j * 2 + 1];
                }
            }
        }
    }
}

// Range-aware activation fp32->fp16: only processes rows [start_row, end_row)
// start_row and end_row must be even and tile-aligned (multiples of 32)
static void transfer_activation_chunk_fp32_to_fp16_range(__fp16 *restrict vtcm_dst, const float *restrict src,
                                                          int n_rows, int n_cols, int row_stride,
                                                          int start_row, int end_row) {
    const int n_rows_tiled  = (n_rows / HMX_FP16_TILE_N_ROWS) * HMX_FP16_TILE_N_ROWS;
    const int n_tiles_per_row = n_cols / HMX_FP16_TILE_N_COLS;

    int r = start_row;

    // HVX path for tiled rows in range
    if (r < n_rows_tiled) {
        int hvx_end = (end_row < n_rows_tiled) ? end_row : n_rows_tiled;
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

    // Scalar path for remaining padded rows in range
    const int n_rows_padded = ((n_rows + HMX_FP16_TILE_N_ROWS - 1) / HMX_FP16_TILE_N_ROWS) * HMX_FP16_TILE_N_ROWS;
    if (r < end_row && r >= n_rows_tiled) {
        int scalar_end = (end_row < n_rows_padded) ? end_row : n_rows_padded;
        for (; r < scalar_end; r += 2) {
            int r0 = r / HMX_FP16_TILE_N_ROWS;
            int r1 = r % HMX_FP16_TILE_N_ROWS;

            const bool row0_valid = r       < n_rows;
            const bool row1_valid = (r + 1) < n_rows;

            const float *src_row0 = row0_valid ? src + (r + 0) * row_stride : NULL;
            const float *src_row1 = row1_valid ? src + (r + 1) * row_stride : NULL;

            for (int c = 0; c < n_cols; c += HMX_FP16_TILE_N_COLS) {
                int c0 = c / HMX_FP16_TILE_N_COLS;
                int tile_idx = r0 * n_tiles_per_row + c0;

                __fp16 *tile_base = vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS;

                for (int i = 0; i < HMX_FP16_TILE_N_COLS; ++i) {
                    tile_base[(r1 / 2) * 64 + i * 2] =
                        (src_row0) ? (__fp16)src_row0[c + i] : (__fp16)0;
                }
                for (int i = 0; i < HMX_FP16_TILE_N_COLS; ++i) {
                    tile_base[(r1 / 2) * 64 + i * 2 + 1] =
                        (src_row1) ? (__fp16)src_row1[c + i] : (__fp16)0;
                }
            }
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

int ggmlop_dsp_mulmat_hmx(remote_handle64 h, const struct dsptensor * src0, const struct dsptensor * src1, dsptensor * dst) {
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

    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        if (hmx_locked) {
            HAP_compute_res_hmx_unlock(compute_res_ctx_id);
        }
        GGMLHEXAGON_LOG_INFO("falling back to VTCM multithread mode\n");
        return ggmlop_dsp_mulmat_multithread_vtcm(h, src0, src1, dst);
    }

    if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16 &&
        src1->type != GGML_TYPE_Q4_0 && src1->type != GGML_TYPE_Q4_1 && src1->type != GGML_TYPE_Q8_0) {
        if (hmx_locked) {
            HAP_compute_res_hmx_unlock(compute_res_ctx_id);
        }
        GGMLHEXAGON_LOG_INFO("falling back to VTCM multithread mode\n");
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
    GGMLHEXAGON_LOG_DEBUG("src0 type=%d, src1 type=%d", src0->type, src1->type);

    if (K % HMX_FP16_TILE_N_COLS != 0 || M % HMX_FP16_TILE_N_COLS != 0 || N % 32 != 0) {
        if (hmx_locked) {
            HAP_compute_res_hmx_unlock(compute_res_ctx_id);
        }
        GGMLHEXAGON_LOG_INFO("K=%d or M=%d or N=%d not 32-aligned, falling back to VTCM multithread mode\n", K, M, N);
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
    // Budget: weight_fp32_buf + weight_tiles + reusable_buf + act_tiles + scales <= vtcm_size
    //   weight_fp32_buf = M_chunk * K * 4  (fp32 input for weight conversion)
    //   weight_tiles    = M_chunk * K * 2  (fp16 tiles)
    //   act_tiles       = N_chunk * K * 2  (fp16 tiles)
    //   output_tiles    = M_chunk * N_chunk * 2 (fp16, time-shared with act_fp32_buf)
    //   reusable_buf    = max(act_fp32_buf, output_tiles)
    //   act_fp32_buf    = N_chunk * K * 4

    const size_t vec_dot_size = K * sizeof(__fp16);
    const size_t scales_size  = 256;

    // Sweep M_chunk from max down to find a fit
    const size_t M_aligned = hex_align_down((size_t)M, HMX_FP16_TILE_N_COLS);
    size_t M_chunk_n_cols = 0;
    size_t N_chunk_n_rows = 0;

    for (size_t mc = M_aligned; mc >= HMX_FP16_TILE_N_COLS; mc -= HMX_FP16_TILE_N_COLS) {
        const size_t w_fp32  = hex_align_up(mc * K * sizeof(float), HMX_FP16_TILE_SIZE);
        const size_t w_tiles = hex_align_up(mc * vec_dot_size, HMX_FP16_TILE_SIZE);
        const size_t remain  = vtcm_size - w_fp32 - w_tiles - scales_size;
        if (remain <= 0) continue;

        // N * K * 2 + max(N * K * 4, mc * N * 2) <= remain
        // When K*4 >= mc*2 (i.e. K*2 >= mc), act_fp32_buf dominates:
        //   N * K * 6 <= remain  =>  N = remain / (K * 6)
        // Otherwise output dominates:
        //   N * (K * 2 + mc * 2) <= remain  =>  N = remain / (K * 2 + mc * 2)
        const size_t per_n = (K * (size_t)4 >= mc * 2) ? K * 6 : K * 2 + mc * 2;
        size_t nc = hex_align_down(remain / per_n, HMX_FP16_TILE_N_ROWS);
        if (nc == 0) nc = HMX_FP16_TILE_N_ROWS;

        // Clamp N_chunk to N
        if (nc > (size_t)N) nc = hex_align_down((size_t)N, HMX_FP16_TILE_N_ROWS);
        if (nc == 0 && N > 0) nc = HMX_FP16_TILE_N_ROWS;

        // Verify it actually fits
        const size_t a_fp32   = hex_align_up(nc * K * sizeof(float), HMX_FP16_TILE_SIZE);
        const size_t a_tiles  = hex_align_up(nc * vec_dot_size, HMX_FP16_TILE_SIZE);
        const size_t o_tiles  = hex_align_up(nc * mc * sizeof(__fp16), HMX_FP16_TILE_SIZE);
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

    // Recompute exact sizes for chosen chunks
    const size_t weight_fp32_buf_size = hex_align_up(M_chunk_n_cols * K * sizeof(float), HMX_FP16_TILE_SIZE);
    const size_t weight_area_size     = hex_align_up(M_chunk_n_cols * vec_dot_size, HMX_FP16_TILE_SIZE);
    const size_t act_fp32_buf_size    = hex_align_up(N_chunk_n_rows * K * sizeof(float), HMX_FP16_TILE_SIZE);
    const size_t act_area_size        = hex_align_up(N_chunk_n_rows * vec_dot_size, HMX_FP16_TILE_SIZE);
    const size_t output_area_size     = hex_align_up(N_chunk_n_rows * M_chunk_n_cols * sizeof(__fp16), HMX_FP16_TILE_SIZE);
    const size_t reusable_buf_size    = (act_fp32_buf_size > output_area_size) ? act_fp32_buf_size : output_area_size;
    const size_t total_vtcm_needed    = act_area_size + weight_area_size + reusable_buf_size + weight_fp32_buf_size + scales_size;

    GGMLHEXAGON_LOG_INFO("VTCM check: M=%d, N=%d, K=%d, vtcm_size=%zu, M_chunk=%zu, N_chunk=%zu, total_needed=%zu (act=%zu, weight=%zu, reusable=%zu, weight_fp32=%zu, scales=%zu)",
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

    const bool src0_is_f16 = (src0->type == GGML_TYPE_F16);  // weight type
    const bool src1_is_f16 = (src1->type == GGML_TYPE_F16);  // activation type

    const size_t src0_row_stride = src0->nb[1];  // weight stride
    const size_t src1_row_stride = src1->nb[1];  // activation stride

    // Create DMA queue for async data transfers
    dma_queue *dma = dma_queue_create(16);

    // Outer loop: iterate over M (weight columns)
    // Inner loop: iterate over N (activation columns)
    // Weight uses column-pair interleaved format, Activation uses row-pair interleaved format

    for (size_t mc = 0; mc < M; mc += M_chunk_n_cols) {
        const size_t M_cols = (M - mc) > M_chunk_n_cols ? M_chunk_n_cols : (M - mc);
        const size_t M_col_tiles = M_cols / HMX_FP16_TILE_N_COLS;

        // Convert weight chunk (src0) to fp16 tiles using interleaved format
        if (src0_is_f16) {
            const __fp16 *weight_chunk = (const __fp16 *)((const char *)src0->data + mc * src0_row_stride);
            transfer_weight_chunk_f16_to_f16_tiles(vtcm_weight, weight_chunk, M_cols, K, src0_row_stride / sizeof(__fp16));
        } else if (src0->type == GGML_TYPE_F32) {
            const float *weight_chunk = (const float *)((const char *)src0->data + mc * src0_row_stride);

            // DMA async: push weight fp32 rows from DDR to VTCM
            dma_queue_push_ddr_to_vtcm(dma,
                dma_make_ptr(vtcm_weight_fp32_buf, weight_chunk),
                K * sizeof(float), src0_row_stride, M_cols);
            dma_queue_pop(dma);  // wait for DMA completion

            // Convert from VTCM fp32 buffer to fp16 tiles (interleaved format)
            convert_weight_f32_to_fp16_tiles(vtcm_weight, vtcm_weight_fp32_buf, M_cols, K, K);
        } else if (src0->type == GGML_TYPE_Q4_0) {
            const block_q4_0 *weight_chunk = (const block_q4_0 *)((const char *)src0->data + mc * src0_row_stride);
            dequantize_q4_0_to_f16_tiles(vtcm_weight, weight_chunk, M_cols, K);
        } else if (src0->type == GGML_TYPE_Q4_1) {
            const block_q4_1 *weight_chunk = (const block_q4_1 *)((const char *)src0->data + mc * src0_row_stride);
            dequantize_q4_1_to_f16_tiles(vtcm_weight, weight_chunk, M_cols, K);
        } else if (src0->type == GGML_TYPE_Q8_0) {
            const block_q8_0 *weight_chunk = (const block_q8_0 *)((const char *)src0->data + mc * src0_row_stride);
            dequantize_q8_0_to_f16_tiles(vtcm_weight, weight_chunk, M_cols, K);
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
                transfer_activation_chunk_f16_to_f16_tiles(vtcm_activation, act_chunk, N_rows, K, src1_row_stride / sizeof(__fp16));
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

            // HMX computation
            core_dot_chunk_fp16(reusable_buf.fp16, vtcm_activation, vtcm_weight, vtcm_scales, N_row_tiles, M_col_tiles, n_dot_tiles);

            // Copy output to dst (must complete before DMA prefetch overwrites reusable_buf)
            float *output_chunk = (float *)((char *)dst->data + mc * dst->nb[0] + nr * dst->nb[1]);
            transfer_output_chunk_fp16_to_fp32(output_chunk, reusable_buf.fp16, N_rows, M_cols, M);

            // Prefetch next activation chunk via DMA (overlaps with next iteration's compute)
            // NOTE: this must be after output writeback since reusable_buf is shared
            size_t nr_next = nr + N_chunk_n_rows;
            if (nr_next < N && !src1_is_f16) {
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

int ggmlop_dsp_mulmat(remote_handle64 h, const struct dsptensor * src0, const struct dsptensor * src1, dsptensor * dst) {
    int  ret = 0;
    char tempbuf[256];
    int  mulmat_algo = ggmlop_get_mulmat_algotype();
    ggmlhexagon_get_opkey(GGML_OP_MUL_MAT, src0, src1, tempbuf, 256);
    int64_t begin_time = ggml_time_us();
    if (mulmat_algo == 32) {
        GGMLHEXAGON_LOG_INFO("mulmat using HMX mode");
        ret = ggmlop_dsp_mulmat_hmx(h, src0, src1, dst);
    } else if (ggmlop_get_thread_counts() > 1) {
        GGMLHEXAGON_LOG_INFO("mulmat using MT_VTCM mode");
        ret= ggmlop_dsp_mulmat_multithread_vtcm(h, src0, src1, dst);
    } else {
        GGMLHEXAGON_LOG_INFO("mulmat using singlethread mode");
        ret = ggmlop_dsp_mulmat_singlethread(h, src0, src1, dst);
    }
    int64_t end_time = ggml_time_us();
    GGMLHEXAGON_LOG_INFO("elapse time of %s is %lld us", tempbuf, (long long)(end_time - begin_time));
    GGMLHEXAGON_LOG_DEBUG("leave %s\n", __func__);
    return ret;
}

