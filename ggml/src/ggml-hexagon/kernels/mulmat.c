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

static void vec_dot_f32_hvx_ai(int n, float *GGML_RESTRICT s, const float *GGML_RESTRICT x, const float *GGML_RESTRICT y) {
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

inline void vec_dot_f32_hvx_me(int n, float * GGML_RESTRICT s, const float * GGML_RESTRICT x, const float * GGML_RESTRICT y) {
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

static void vec_dot_f32(int n, float *GGML_RESTRICT s, size_t bs, const float *GGML_RESTRICT x,
                    size_t bx, const float *GGML_RESTRICT y, size_t by, int nrc) {
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(nrc);

    if (n >= VLEN_FP32 && ((uintptr_t)x & 0x7F) == 0 && ((uintptr_t)y & 0x7F) == 0) {
        vec_dot_f32_hvx_ai(n, s, x, y);
        //vec_dot_f32_hvx_me(n, s, x, y);
        return;
    }

    ggml_float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float) (x[i] * y[i]);
    }
    *s = sumf;
}

static void vec_dot_f16_f32(int n, float *GGML_RESTRICT s, size_t bs, const uint16_t *GGML_RESTRICT x,
                    size_t bx, const float *GGML_RESTRICT y, size_t by, int nrc) {
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

#if 0
static void vec_dot_f16_f16(int n, float *GGML_RESTRICT s, size_t bs, const uint16_t *GGML_RESTRICT x,
                    size_t bx, const uint16_t *GGML_RESTRICT y, size_t by, int nrc) {
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
#endif

static void vec_dot_f16_f16(int n, float *GGML_RESTRICT s, size_t bs, const uint16_t *GGML_RESTRICT x,
                    size_t bx, const uint16_t *GGML_RESTRICT y, size_t by, int nrc) {
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

#define QK4_0 32
#define QK4_1 32
#define QK8_0 32
#define QK8_1 32
#define QK5_0 32
#define QK5_1 32
#define QK4_NL 32
#define QK_K   256
#define K_SCALE_SIZE 12
#define QK_MXFP4 32
#define QK_NVFP4 64
#define QK_NVFP4_SUB 16

#define GGML_Q4_0_BLCK_SZ (sizeof(uint16_t) + QK4_0/2)
#define GGML_Q4_1_BLCK_SZ (sizeof(uint16_t) + sizeof(uint16_t) + QK4_1/2)
#define GGML_Q5_0_BLCK_SZ (sizeof(uint16_t) + sizeof(uint32_t) + QK5_0/2)
#define GGML_Q5_1_BLCK_SZ (2*sizeof(uint16_t) + sizeof(uint32_t) + QK5_1/2)
#define GGML_Q8_0_BLCK_SZ (sizeof(uint16_t) + QK8_0)
#define GGML_Q8_1_BLCK_SZ (sizeof(uint16_t) + sizeof(uint16_t) + QK8_1)

static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

typedef struct {
    uint16_t d;
    uint8_t qs[QK4_0 / 2];
} block_q4_0;

typedef struct {
    uint16_t d;
    uint16_t m;
    uint8_t qs[QK4_1 / 2];
} block_q4_1;

typedef struct {
    uint16_t d;
    int8_t qs[QK8_0];
} block_q8_0;

typedef struct {
    uint16_t d;
    uint16_t s;
    int8_t qs[QK8_1];
} block_q8_1;

typedef struct {
    uint16_t d;
    uint8_t qh[4];
    uint8_t qs[QK5_0 / 2];
} block_q5_0;

typedef struct {
    uint16_t d;
    uint16_t m;
    uint8_t qh[4];
    uint8_t qs[QK5_1 / 2];
} block_q5_1;

typedef struct {
    uint16_t d;
    uint8_t qs[QK4_NL / 2];
} block_iq4_nl;

// BF16 type
typedef struct { uint16_t bits; } ggml_bf16_t;

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

// K-quant structures (super-block size = QK_K = 256)
typedef struct {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16
} block_q8_K;

typedef struct {
    uint16_t d;     // super-block scale for quantized scales
    uint16_t dmin;  // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4-bit quants
} block_q4_K;

typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    uint16_t d;              // super-block scale
} block_q6_K;

typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants (2-bit)
    uint16_t d;              // super-block scale for quantized scales
    uint16_t dmin;           // super-block scale for quantized mins
} block_q2_K;

typedef struct {
    uint8_t hmask[QK_K/8];  // quants - high bit
    uint8_t qs[QK_K/4];     // quants - low 2 bits
    uint8_t scales[12];     // scales, quantized with 6 bits
    uint16_t d;             // super-block scale
} block_q3_K;

typedef struct {
    uint16_t d;             // super-block scale for quantized scales
    uint16_t dmin;          // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];    // quants, high bit
    uint8_t qs[QK_K/2];    // quants, low 4 bits
} block_q5_K;

// MXFP4: 32-element block with E8M0 shared exponent
typedef struct {
    uint8_t e;              // E8M0 shared exponent
    uint8_t qs[QK_MXFP4/2]; // packed 4-bit E2M1 values
} block_mxfp4;

// NVFP4: 64-element block with per-sub-block UE4M3 scales
typedef struct {
    uint8_t d[QK_NVFP4/QK_NVFP4_SUB]; // UE4M3 scales (4 bytes)
    uint8_t qs[QK_NVFP4/2];           // packed 4-bit E2M1 values (32 bytes)
} block_nvfp4;

// IQ4_XS: 256-element super-block with non-linear 4-bit quantization
typedef struct {
    uint16_t d;             // super-block scale
    uint16_t scales_h;      // high bits of per-32-element scales
    uint8_t  scales_l[QK_K/64]; // low bits of per-32-element scales
    uint8_t  qs[QK_K/2];    // 4-bit quants
} block_iq4_xs;

// IQ3_XXS: 256-element super-block with 3-bit quantization
typedef struct {
    uint16_t d;             // super-block scale
    uint8_t  qs[3*QK_K/8]; // 3-bit quants (96 bytes)
} block_iq3_xxs;

// IQ2_XXS: 256-element super-block with 2-bit quantization
typedef struct {
    uint16_t d;             // super-block scale
    uint16_t qs[QK_K/8];   // 2-bit quants (64 bytes)
} block_iq2_xxs;

// IQ2_XS: 256-element super-block with 2-bit quantization + per-32-element scales
typedef struct {
    uint16_t d;             // super-block scale
    uint16_t qs[QK_K/8];   // 2-bit quants (64 bytes)
    uint8_t  scales[QK_K/32]; // per-32-element scales (8 bytes)
} block_iq2_xs;

// IQ2_S: 256-element super-block with 2-bit quantization + per-32-element scales + signs
typedef struct {
    uint16_t d;             // super-block scale
    uint8_t  qs[QK_K/4];   // 2-bit quants + signs (64 bytes)
    uint8_t  qh[QK_K/32];  // high bits for grid lookup (8 bytes)
    uint8_t  scales[QK_K/32]; // per-32-element scales (8 bytes)
} block_iq2_s;

// IQ1_S: 256-element super-block with 1-bit quantization
typedef struct {
    uint16_t d;             // super-block scale
    uint8_t  qs[QK_K/8];   // 1-bit quants (32 bytes)
    uint16_t qh[QK_K/32];  // high bits for grid lookup + scales + delta (16 bytes)
} block_iq1_s;

static const uint64_t iq2xs_grid[512] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x080808082b080808,
    0x080808082b08082b, 0x080808082b081919, 0x080808082b082b08, 0x080808082b190819,
    0x080808082b191908, 0x080808082b192b19, 0x080808082b2b0808, 0x0808081908080819,
    0x0808081908081908, 0x080808190808192b, 0x0808081908082b19, 0x0808081908190808,
    0x080808190819082b, 0x0808081908191919, 0x0808081908192b08, 0x0808081908192b2b,
    0x08080819082b0819, 0x08080819082b1908, 0x0808081919080808, 0x080808191908082b,
    0x0808081919081919, 0x0808081919082b08, 0x0808081919190819, 0x0808081919191908,
    0x08080819192b0808, 0x08080819192b2b08, 0x080808192b080819, 0x080808192b081908,
    0x080808192b190808, 0x0808082b08080808, 0x0808082b0808082b, 0x0808082b08081919,
    0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908, 0x0808082b082b0808,
    0x0808082b19080819, 0x0808082b19081908, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b082b2b, 0x0808190808080819, 0x0808190808081908,
    0x080819080808192b, 0x0808190808082b19, 0x0808190808190808, 0x080819080819082b,
    0x0808190808191919, 0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908,
    0x0808190819080808, 0x080819081908082b, 0x0808190819081919, 0x0808190819082b08,
    0x0808190819190819, 0x0808190819191908, 0x080819081919192b, 0x08081908192b0808,
    0x080819082b080819, 0x080819082b081908, 0x080819082b190808, 0x0808191908080808,
    0x080819190808082b, 0x0808191908081919, 0x0808191908082b08, 0x0808191908190819,
    0x0808191908191908, 0x08081919082b0808, 0x0808191919080819, 0x0808191919081908,
    0x0808191919190808, 0x08081919192b0819, 0x080819192b080808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b08190808, 0x0808192b082b192b, 0x0808192b19080808,
    0x0808192b1908082b, 0x0808192b2b081908, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808082b2b, 0x08082b0808190819,
    0x08082b0808191908, 0x08082b08082b0808, 0x08082b08082b1919, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b0819192b08, 0x08082b082b080808,
    0x08082b082b2b0808, 0x08082b082b2b2b2b, 0x08082b1908080819, 0x08082b1908081908,
    0x08082b1908190808, 0x08082b1919080808, 0x08082b192b080819, 0x08082b192b082b19,
    0x08082b2b08080808, 0x08082b2b082b0808, 0x08082b2b082b2b08, 0x08082b2b2b19192b,
    0x08082b2b2b2b0808, 0x0819080808080819, 0x0819080808081908, 0x081908080808192b,
    0x0819080808082b19, 0x0819080808190808, 0x081908080819082b, 0x0819080808191919,
    0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908, 0x0819080819080808,
    0x081908081908082b, 0x0819080819081919, 0x0819080819082b08, 0x0819080819190819,
    0x0819080819191908, 0x08190808192b0808, 0x08190808192b2b2b, 0x081908082b080819,
    0x081908082b081908, 0x081908082b190808, 0x0819081908080808, 0x081908190808082b,
    0x0819081908081919, 0x0819081908082b08, 0x0819081908190819, 0x0819081908191908,
    0x08190819082b0808, 0x0819081919080819, 0x0819081919081908, 0x0819081919190808,
    0x081908192b080808, 0x081908192b191908, 0x081908192b19192b, 0x0819082b08080819,
    0x0819082b08081908, 0x0819082b0808192b, 0x0819082b08190808, 0x0819082b19080808,
    0x0819082b192b0808, 0x0819190808080808, 0x081919080808082b, 0x0819190808081919,
    0x0819190808082b08, 0x0819190808190819, 0x0819190808191908, 0x08191908082b0808,
    0x0819190819080819, 0x0819190819081908, 0x0819190819082b19, 0x0819190819190808,
    0x08191908192b1908, 0x081919082b080808, 0x0819191908080819, 0x0819191908081908,
    0x0819191908190808, 0x0819191919080808, 0x0819192b08080808, 0x0819192b08191908,
    0x0819192b19082b19, 0x08192b0808080819, 0x08192b0808081908, 0x08192b0808190808,
    0x08192b080819082b, 0x08192b0819080808, 0x08192b0819191908, 0x08192b082b08192b,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b19192b192b, 0x08192b2b19190819,
    0x08192b2b2b2b2b19, 0x082b080808080808, 0x082b08080808082b, 0x082b080808081919,
    0x082b080808082b08, 0x082b080808082b2b, 0x082b080808190819, 0x082b080808191908,
    0x082b0808082b0808, 0x082b080819080819, 0x082b080819081908, 0x082b080819190808,
    0x082b08082b080808, 0x082b08082b2b0808, 0x082b081908080819, 0x082b081908081908,
    0x082b081908190808, 0x082b081919080808, 0x082b081919082b08, 0x082b0819192b1919,
    0x082b082b08080808, 0x082b082b082b082b, 0x082b082b2b080808, 0x082b082b2b2b2b08,
    0x082b190808080819, 0x082b190808081908, 0x082b190808190808, 0x082b1908082b2b19,
    0x082b190819080808, 0x082b191908080808, 0x082b191919080819, 0x082b19191919082b,
    0x082b19192b192b19, 0x082b192b08080819, 0x082b192b08192b2b, 0x082b192b2b2b192b,
    0x082b2b0808080808, 0x082b2b0808082b08, 0x082b2b0808082b2b, 0x082b2b08082b0808,
    0x082b2b0819191919, 0x082b2b082b082b08, 0x082b2b082b2b082b, 0x082b2b19192b2b08,
    0x082b2b192b190808, 0x082b2b2b08082b08, 0x082b2b2b082b0808, 0x082b2b2b2b08082b,
    0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819, 0x1908080808081908,
    0x190808080808192b, 0x1908080808082b19, 0x1908080808190808, 0x190808080819082b,
    0x1908080808191919, 0x1908080808192b08, 0x19080808082b0819, 0x19080808082b1908,
    0x1908080819080808, 0x190808081908082b, 0x1908080819081919, 0x1908080819082b08,
    0x1908080819082b2b, 0x1908080819190819, 0x1908080819191908, 0x19080808192b0808,
    0x19080808192b1919, 0x190808082b080819, 0x190808082b081908, 0x190808082b190808,
    0x1908081908080808, 0x190808190808082b, 0x1908081908081919, 0x1908081908082b08,
    0x1908081908190819, 0x1908081908191908, 0x19080819082b0808, 0x1908081919080819,
    0x1908081919081908, 0x1908081919190808, 0x190808192b080808, 0x190808192b081919,
    0x190808192b2b082b, 0x1908082b08080819, 0x1908082b08081908, 0x1908082b08190808,
    0x1908082b0819082b, 0x1908082b082b2b19, 0x1908082b19080808, 0x1908190808080808,
    0x190819080808082b, 0x1908190808081919, 0x1908190808082b08, 0x1908190808190819,
    0x1908190808191908, 0x1908190808192b19, 0x19081908082b0808, 0x1908190819080819,
    0x1908190819081908, 0x1908190819190808, 0x190819082b080808, 0x190819082b191908,
    0x1908191908080819, 0x1908191908081908, 0x1908191908190808, 0x19081919082b1908,
    0x1908191919080808, 0x190819192b192b2b, 0x1908192b08080808, 0x1908192b08082b2b,
    0x1908192b19081908, 0x1908192b19190808, 0x19082b0808080819, 0x19082b0808081908,
    0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919, 0x19082b0819191908,
    0x19082b08192b082b, 0x19082b1908080808, 0x19082b1908190819, 0x19082b1919081908,
    0x19082b1919190808, 0x19082b19192b2b19, 0x19082b2b08081908, 0x1919080808080808,
    0x191908080808082b, 0x1919080808081919, 0x1919080808082b08, 0x1919080808190819,
    0x1919080808191908, 0x19190808082b0808, 0x19190808082b2b08, 0x1919080819080819,
    0x1919080819081908, 0x1919080819190808, 0x191908082b080808, 0x1919081908080819,
    0x1919081908081908, 0x1919081908190808, 0x1919081908191919, 0x1919081919080808,
    0x191908191908082b, 0x1919082b08080808, 0x1919082b19081908, 0x1919082b2b2b2b2b,
    0x1919190808080819, 0x1919190808081908, 0x1919190808190808, 0x19191908082b0819,
    0x1919190819080808, 0x19191908192b0808, 0x191919082b080819, 0x191919082b2b0819,
    0x1919191908080808, 0x1919191908082b08, 0x191919192b080808, 0x191919192b082b08,
    0x1919192b082b0819, 0x1919192b192b2b08, 0x1919192b2b2b0819, 0x19192b0808080808,
    0x19192b0808191908, 0x19192b0819080819, 0x19192b0819190808, 0x19192b082b192b19,
    0x19192b1908192b2b, 0x19192b1919080808, 0x19192b191908082b, 0x19192b2b2b081919,
    0x192b080808080819, 0x192b080808081908, 0x192b080808190808, 0x192b080819080808,
    0x192b080819191908, 0x192b0808192b082b, 0x192b08082b08192b, 0x192b08082b2b2b19,
    0x192b081908080808, 0x192b082b082b1908, 0x192b082b19082b2b, 0x192b082b2b19082b,
    0x192b190808080808, 0x192b19080819192b, 0x192b191908190808, 0x192b191919080808,
    0x192b191919081919, 0x192b19192b2b1908, 0x192b2b0808080819, 0x192b2b08192b2b2b,
    0x192b2b19082b1919, 0x192b2b2b0808192b, 0x192b2b2b19191908, 0x192b2b2b192b082b,
    0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08,
    0x2b08080808190819, 0x2b08080808191908, 0x2b080808082b0808, 0x2b080808082b2b2b,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808082b080808,
    0x2b0808082b08082b, 0x2b0808082b2b2b08, 0x2b0808082b2b2b2b, 0x2b08081908080819,
    0x2b08081908081908, 0x2b0808190808192b, 0x2b08081908190808, 0x2b08081919080808,
    0x2b08081919190819, 0x2b08081919192b19, 0x2b08082b08080808, 0x2b08082b082b0808,
    0x2b08082b2b080808, 0x2b08082b2b08082b, 0x2b08082b2b2b0808, 0x2b08082b2b2b2b08,
    0x2b08190808080819, 0x2b08190808081908, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190819080808, 0x2b081908192b0808, 0x2b0819082b082b19,
    0x2b08191908080808, 0x2b08191919081908, 0x2b0819192b2b1919, 0x2b08192b08192b08,
    0x2b08192b192b2b2b, 0x2b082b0808080808, 0x2b082b0808082b08, 0x2b082b08082b1919,
    0x2b082b0819192b2b, 0x2b082b082b080808, 0x2b082b082b08082b, 0x2b082b082b2b2b08,
    0x2b082b190808192b, 0x2b082b2b082b082b, 0x2b082b2b2b080808, 0x2b082b2b2b082b08,
    0x2b082b2b2b19192b, 0x2b082b2b2b2b2b08, 0x2b19080808080819, 0x2b19080808081908,
    0x2b19080808190808, 0x2b19080819080808, 0x2b1908081919192b, 0x2b1908082b081908,
    0x2b19081908080808, 0x2b190819082b082b, 0x2b190819192b1908, 0x2b19082b1919192b,
    0x2b19082b2b082b19, 0x2b19190808080808, 0x2b19190808081919, 0x2b19190819081908,
    0x2b19190819190808, 0x2b19190819192b08, 0x2b191919082b2b19, 0x2b1919192b190808,
    0x2b1919192b19082b, 0x2b19192b19080819, 0x2b192b0819190819, 0x2b192b082b2b192b,
    0x2b192b1919082b19, 0x2b192b2b08191919, 0x2b192b2b192b0808, 0x2b2b080808080808,
    0x2b2b08080808082b, 0x2b2b080808082b08, 0x2b2b080808082b2b, 0x2b2b0808082b0808,
    0x2b2b0808082b2b2b, 0x2b2b08082b2b0808, 0x2b2b081919190819, 0x2b2b081919192b19,
    0x2b2b08192b2b192b, 0x2b2b082b08080808, 0x2b2b082b0808082b, 0x2b2b082b08082b08,
    0x2b2b082b082b2b2b, 0x2b2b082b2b080808, 0x2b2b082b2b2b0808, 0x2b2b190819080808,
    0x2b2b19082b191919, 0x2b2b192b192b1919, 0x2b2b192b2b192b08, 0x2b2b2b0808082b2b,
    0x2b2b2b08082b0808, 0x2b2b2b08082b082b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b0808,
    0x2b2b2b082b2b2b08, 0x2b2b2b1908081908, 0x2b2b2b192b081908, 0x2b2b2b192b08192b,
    0x2b2b2b2b082b2b08, 0x2b2b2b2b082b2b2b, 0x2b2b2b2b2b190819, 0x2b2b2b2b2b2b2b2b,
};

static const uint64_t iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

static const uint8_t kmask_iq2xs[8] = {
    1, 2, 4, 8, 16, 32, 64, 128
};

static const uint8_t ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static const uint32_t iq3xxs_grid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

static const uint64_t iq2s_grid[1024] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x08080808192b192b,
    0x08080808192b2b19, 0x080808082b080808, 0x080808082b08082b, 0x080808082b081919,
    0x080808082b082b08, 0x080808082b190819, 0x080808082b191908, 0x080808082b2b0808,
    0x080808082b2b1919, 0x080808082b2b2b2b, 0x0808081908080819, 0x0808081908081908,
    0x080808190808192b, 0x0808081908082b19, 0x0808081908190808, 0x080808190819082b,
    0x0808081908191919, 0x0808081908192b08, 0x08080819082b0819, 0x08080819082b1908,
    0x0808081919080808, 0x080808191908082b, 0x0808081919081919, 0x0808081919082b08,
    0x0808081919190819, 0x0808081919191908, 0x080808191919192b, 0x0808081919192b19,
    0x08080819192b0808, 0x08080819192b1919, 0x08080819192b2b08, 0x080808192b080819,
    0x080808192b081908, 0x080808192b190808, 0x080808192b19082b, 0x080808192b191919,
    0x080808192b2b0819, 0x080808192b2b1908, 0x0808082b08080808, 0x0808082b0808082b,
    0x0808082b08081919, 0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908,
    0x0808082b082b0808, 0x0808082b082b2b2b, 0x0808082b19080819, 0x0808082b19081908,
    0x0808082b1908192b, 0x0808082b19082b19, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b081919, 0x0808082b2b082b2b, 0x0808082b2b191908,
    0x0808082b2b2b082b, 0x0808190808080819, 0x0808190808081908, 0x080819080808192b,
    0x0808190808082b19, 0x0808190808190808, 0x080819080819082b, 0x0808190808191919,
    0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908, 0x08081908082b192b,
    0x08081908082b2b19, 0x0808190819080808, 0x080819081908082b, 0x0808190819081919,
    0x0808190819082b08, 0x0808190819082b2b, 0x0808190819190819, 0x0808190819191908,
    0x080819081919192b, 0x0808190819192b19, 0x08081908192b0808, 0x08081908192b082b,
    0x08081908192b1919, 0x080819082b080819, 0x080819082b081908, 0x080819082b08192b,
    0x080819082b082b19, 0x080819082b190808, 0x080819082b191919, 0x080819082b192b08,
    0x080819082b2b0819, 0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b,
    0x0808191908081919, 0x0808191908082b08, 0x0808191908082b2b, 0x0808191908190819,
    0x0808191908191908, 0x080819190819192b, 0x0808191908192b19, 0x08081919082b0808,
    0x08081919082b1919, 0x08081919082b2b08, 0x0808191919080819, 0x0808191919081908,
    0x080819191908192b, 0x0808191919082b19, 0x0808191919190808, 0x080819191919082b,
    0x0808191919191919, 0x0808191919192b08, 0x08081919192b0819, 0x08081919192b1908,
    0x080819192b080808, 0x080819192b08082b, 0x080819192b081919, 0x080819192b082b08,
    0x080819192b190819, 0x080819192b191908, 0x080819192b2b0808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b0808192b, 0x0808192b08082b19, 0x0808192b08190808,
    0x0808192b08191919, 0x0808192b19080808, 0x0808192b19081919, 0x0808192b19082b08,
    0x0808192b19190819, 0x0808192b19191908, 0x0808192b192b0808, 0x0808192b2b080819,
    0x0808192b2b081908, 0x0808192b2b190808, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808190819, 0x08082b0808191908,
    0x08082b080819192b, 0x08082b0808192b19, 0x08082b08082b0808, 0x08082b08082b1919,
    0x08082b08082b2b2b, 0x08082b0819080819, 0x08082b0819081908, 0x08082b081908192b,
    0x08082b0819082b19, 0x08082b0819190808, 0x08082b081919082b, 0x08082b0819191919,
    0x08082b0819192b08, 0x08082b08192b0819, 0x08082b08192b1908, 0x08082b082b080808,
    0x08082b082b081919, 0x08082b082b191908, 0x08082b082b2b2b2b, 0x08082b1908080819,
    0x08082b1908081908, 0x08082b1908190808, 0x08082b190819082b, 0x08082b1908191919,
    0x08082b1908192b08, 0x08082b19082b0819, 0x08082b1919080808, 0x08082b1919081919,
    0x08082b1919082b08, 0x08082b1919190819, 0x08082b1919191908, 0x08082b19192b0808,
    0x08082b192b080819, 0x08082b192b190808, 0x08082b2b08080808, 0x08082b2b08190819,
    0x08082b2b08191908, 0x08082b2b082b082b, 0x08082b2b082b2b08, 0x08082b2b082b2b2b,
    0x08082b2b19190808, 0x08082b2b2b192b19, 0x0819080808080819, 0x0819080808081908,
    0x081908080808192b, 0x0819080808082b19, 0x0819080808190808, 0x081908080819082b,
    0x0819080808191919, 0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908,
    0x08190808082b192b, 0x0819080819080808, 0x081908081908082b, 0x0819080819081919,
    0x0819080819082b08, 0x0819080819190819, 0x0819080819191908, 0x081908081919192b,
    0x0819080819192b19, 0x08190808192b0808, 0x08190808192b082b, 0x08190808192b1919,
    0x08190808192b2b08, 0x081908082b080819, 0x081908082b081908, 0x081908082b08192b,
    0x081908082b190808, 0x081908082b191919, 0x081908082b192b08, 0x081908082b2b0819,
    0x081908082b2b1908, 0x0819081908080808, 0x081908190808082b, 0x0819081908081919,
    0x0819081908082b08, 0x0819081908082b2b, 0x0819081908190819, 0x0819081908191908,
    0x081908190819192b, 0x0819081908192b19, 0x08190819082b0808, 0x08190819082b082b,
    0x08190819082b1919, 0x08190819082b2b08, 0x0819081919080819, 0x0819081919081908,
    0x081908191908192b, 0x0819081919082b19, 0x0819081919190808, 0x081908191919082b,
    0x0819081919191919, 0x0819081919192b08, 0x08190819192b0819, 0x08190819192b1908,
    0x081908192b080808, 0x081908192b08082b, 0x081908192b081919, 0x081908192b082b08,
    0x081908192b190819, 0x081908192b191908, 0x0819082b08080819, 0x0819082b08081908,
    0x0819082b08082b19, 0x0819082b08190808, 0x0819082b08191919, 0x0819082b082b0819,
    0x0819082b082b1908, 0x0819082b19080808, 0x0819082b19081919, 0x0819082b19190819,
    0x0819082b19191908, 0x0819082b2b080819, 0x0819082b2b081908, 0x0819082b2b190808,
    0x0819190808080808, 0x081919080808082b, 0x0819190808081919, 0x0819190808082b08,
    0x0819190808190819, 0x0819190808191908, 0x081919080819192b, 0x0819190808192b19,
    0x08191908082b0808, 0x08191908082b1919, 0x08191908082b2b08, 0x0819190819080819,
    0x0819190819081908, 0x081919081908192b, 0x0819190819082b19, 0x0819190819190808,
    0x081919081919082b, 0x0819190819191919, 0x0819190819192b08, 0x08191908192b0819,
    0x08191908192b1908, 0x081919082b080808, 0x081919082b08082b, 0x081919082b081919,
    0x081919082b082b08, 0x081919082b190819, 0x081919082b191908, 0x081919082b2b0808,
    0x0819191908080819, 0x0819191908081908, 0x081919190808192b, 0x0819191908082b19,
    0x0819191908190808, 0x081919190819082b, 0x0819191908191919, 0x0819191908192b08,
    0x08191919082b0819, 0x08191919082b1908, 0x0819191919080808, 0x081919191908082b,
    0x0819191919081919, 0x0819191919082b08, 0x0819191919190819, 0x0819191919191908,
    0x08191919192b0808, 0x081919192b080819, 0x081919192b081908, 0x081919192b190808,
    0x0819192b08080808, 0x0819192b08081919, 0x0819192b08082b08, 0x0819192b08190819,
    0x0819192b08191908, 0x0819192b082b0808, 0x0819192b19080819, 0x0819192b19081908,
    0x0819192b19190808, 0x0819192b2b080808, 0x0819192b2b2b2b2b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b080808192b, 0x08192b0808082b19, 0x08192b0808190808,
    0x08192b0808191919, 0x08192b0808192b08, 0x08192b08082b0819, 0x08192b0819080808,
    0x08192b081908082b, 0x08192b0819081919, 0x08192b0819082b08, 0x08192b0819190819,
    0x08192b0819191908, 0x08192b08192b0808, 0x08192b082b080819, 0x08192b082b081908,
    0x08192b1908080808, 0x08192b190808082b, 0x08192b1908081919, 0x08192b1908082b08,
    0x08192b1908190819, 0x08192b1908191908, 0x08192b19082b0808, 0x08192b1919080819,
    0x08192b1919081908, 0x08192b1919190808, 0x08192b19192b2b19, 0x08192b192b2b082b,
    0x08192b2b08081908, 0x08192b2b08190808, 0x08192b2b19080808, 0x08192b2b1919192b,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808081919, 0x082b080808082b08,
    0x082b080808190819, 0x082b080808191908, 0x082b08080819192b, 0x082b080808192b19,
    0x082b0808082b0808, 0x082b0808082b1919, 0x082b0808082b2b2b, 0x082b080819080819,
    0x082b080819081908, 0x082b080819190808, 0x082b08081919082b, 0x082b080819191919,
    0x082b0808192b1908, 0x082b08082b080808, 0x082b08082b082b2b, 0x082b08082b191908,
    0x082b08082b2b2b2b, 0x082b081908080819, 0x082b081908081908, 0x082b081908190808,
    0x082b08190819082b, 0x082b081908191919, 0x082b0819082b0819, 0x082b081919080808,
    0x082b08191908082b, 0x082b081919081919, 0x082b081919190819, 0x082b081919191908,
    0x082b0819192b0808, 0x082b08192b080819, 0x082b08192b081908, 0x082b08192b190808,
    0x082b082b08080808, 0x082b082b08082b2b, 0x082b082b082b082b, 0x082b082b082b2b08,
    0x082b082b082b2b2b, 0x082b082b19081908, 0x082b082b19190808, 0x082b082b2b082b08,
    0x082b082b2b082b2b, 0x082b082b2b2b2b08, 0x082b190808080819, 0x082b190808081908,
    0x082b19080808192b, 0x082b190808082b19, 0x082b190808190808, 0x082b190808191919,
    0x082b190808192b08, 0x082b1908082b0819, 0x082b1908082b1908, 0x082b190819080808,
    0x082b19081908082b, 0x082b190819081919, 0x082b190819082b08, 0x082b190819190819,
    0x082b190819191908, 0x082b1908192b0808, 0x082b19082b080819, 0x082b19082b081908,
    0x082b19082b190808, 0x082b191908080808, 0x082b191908081919, 0x082b191908082b08,
    0x082b191908190819, 0x082b191908191908, 0x082b1919082b0808, 0x082b191919080819,
    0x082b191919081908, 0x082b191919190808, 0x082b1919192b192b, 0x082b19192b080808,
    0x082b192b08080819, 0x082b192b08081908, 0x082b192b08190808, 0x082b192b19080808,
    0x082b192b19192b19, 0x082b2b0808080808, 0x082b2b0808081919, 0x082b2b0808190819,
    0x082b2b0808191908, 0x082b2b0819080819, 0x082b2b0819081908, 0x082b2b0819190808,
    0x082b2b082b082b2b, 0x082b2b082b2b2b2b, 0x082b2b1908080819, 0x082b2b1908081908,
    0x082b2b1908190808, 0x082b2b192b191919, 0x082b2b2b08082b2b, 0x082b2b2b082b082b,
    0x082b2b2b192b1908, 0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819,
    0x1908080808081908, 0x190808080808192b, 0x1908080808082b19, 0x1908080808190808,
    0x190808080819082b, 0x1908080808191919, 0x1908080808192b08, 0x1908080808192b2b,
    0x19080808082b0819, 0x19080808082b1908, 0x19080808082b192b, 0x1908080819080808,
    0x190808081908082b, 0x1908080819081919, 0x1908080819082b08, 0x1908080819082b2b,
    0x1908080819190819, 0x1908080819191908, 0x190808081919192b, 0x1908080819192b19,
    0x19080808192b0808, 0x19080808192b082b, 0x19080808192b1919, 0x190808082b080819,
    0x190808082b081908, 0x190808082b190808, 0x190808082b191919, 0x190808082b192b08,
    0x190808082b2b0819, 0x190808082b2b1908, 0x1908081908080808, 0x190808190808082b,
    0x1908081908081919, 0x1908081908082b08, 0x1908081908190819, 0x1908081908191908,
    0x190808190819192b, 0x1908081908192b19, 0x19080819082b0808, 0x19080819082b082b,
    0x19080819082b1919, 0x1908081919080819, 0x1908081919081908, 0x190808191908192b,
    0x1908081919082b19, 0x1908081919190808, 0x190808191919082b, 0x1908081919191919,
    0x1908081919192b08, 0x19080819192b0819, 0x19080819192b1908, 0x190808192b080808,
    0x190808192b08082b, 0x190808192b081919, 0x190808192b082b08, 0x190808192b190819,
    0x190808192b191908, 0x190808192b2b0808, 0x1908082b08080819, 0x1908082b08081908,
    0x1908082b08190808, 0x1908082b0819082b, 0x1908082b08191919, 0x1908082b08192b08,
    0x1908082b082b1908, 0x1908082b19080808, 0x1908082b19081919, 0x1908082b19082b08,
    0x1908082b19190819, 0x1908082b19191908, 0x1908082b192b0808, 0x1908082b2b080819,
    0x1908082b2b081908, 0x1908190808080808, 0x190819080808082b, 0x1908190808081919,
    0x1908190808082b08, 0x1908190808082b2b, 0x1908190808190819, 0x1908190808191908,
    0x190819080819192b, 0x1908190808192b19, 0x19081908082b0808, 0x19081908082b082b,
    0x19081908082b1919, 0x19081908082b2b08, 0x1908190819080819, 0x1908190819081908,
    0x190819081908192b, 0x1908190819082b19, 0x1908190819190808, 0x190819081919082b,
    0x1908190819191919, 0x1908190819192b08, 0x19081908192b0819, 0x19081908192b1908,
    0x190819082b080808, 0x190819082b08082b, 0x190819082b081919, 0x190819082b082b08,
    0x190819082b190819, 0x190819082b191908, 0x190819082b2b0808, 0x1908191908080819,
    0x1908191908081908, 0x190819190808192b, 0x1908191908082b19, 0x1908191908190808,
    0x190819190819082b, 0x1908191908191919, 0x1908191908192b08, 0x19081919082b0819,
    0x19081919082b1908, 0x1908191919080808, 0x190819191908082b, 0x1908191919081919,
    0x1908191919082b08, 0x1908191919190819, 0x1908191919191908, 0x19081919192b0808,
    0x19081919192b2b2b, 0x190819192b080819, 0x190819192b081908, 0x190819192b190808,
    0x1908192b08080808, 0x1908192b0808082b, 0x1908192b08081919, 0x1908192b08082b08,
    0x1908192b08190819, 0x1908192b08191908, 0x1908192b082b0808, 0x1908192b19080819,
    0x1908192b19081908, 0x1908192b19190808, 0x1908192b2b080808, 0x1908192b2b2b1919,
    0x19082b0808080819, 0x19082b0808081908, 0x19082b0808082b19, 0x19082b0808190808,
    0x19082b080819082b, 0x19082b0808191919, 0x19082b0808192b08, 0x19082b08082b0819,
    0x19082b08082b1908, 0x19082b0819080808, 0x19082b081908082b, 0x19082b0819081919,
    0x19082b0819082b08, 0x19082b0819190819, 0x19082b0819191908, 0x19082b08192b0808,
    0x19082b082b081908, 0x19082b082b190808, 0x19082b1908080808, 0x19082b190808082b,
    0x19082b1908081919, 0x19082b1908082b08, 0x19082b1908190819, 0x19082b1908191908,
    0x19082b19082b0808, 0x19082b1919080819, 0x19082b1919081908, 0x19082b1919190808,
    0x19082b192b080808, 0x19082b192b19192b, 0x19082b2b08080819, 0x19082b2b08081908,
    0x19082b2b08190808, 0x19082b2b19080808, 0x1919080808080808, 0x191908080808082b,
    0x1919080808081919, 0x1919080808082b08, 0x1919080808190819, 0x1919080808191908,
    0x191908080819192b, 0x1919080808192b19, 0x19190808082b0808, 0x19190808082b082b,
    0x19190808082b1919, 0x19190808082b2b08, 0x1919080819080819, 0x1919080819081908,
    0x191908081908192b, 0x1919080819082b19, 0x1919080819190808, 0x191908081919082b,
    0x1919080819191919, 0x1919080819192b08, 0x19190808192b0819, 0x19190808192b1908,
    0x191908082b080808, 0x191908082b08082b, 0x191908082b081919, 0x191908082b082b08,
    0x191908082b190819, 0x191908082b191908, 0x1919081908080819, 0x1919081908081908,
    0x191908190808192b, 0x1919081908082b19, 0x1919081908190808, 0x191908190819082b,
    0x1919081908191919, 0x1919081908192b08, 0x19190819082b0819, 0x19190819082b1908,
    0x1919081919080808, 0x191908191908082b, 0x1919081919081919, 0x1919081919082b08,
    0x1919081919190819, 0x1919081919191908, 0x19190819192b0808, 0x191908192b080819,
    0x191908192b081908, 0x191908192b190808, 0x1919082b08080808, 0x1919082b08081919,
    0x1919082b08082b08, 0x1919082b08190819, 0x1919082b08191908, 0x1919082b082b0808,
    0x1919082b19080819, 0x1919082b19081908, 0x1919082b19190808, 0x1919082b192b2b19,
    0x1919082b2b080808, 0x1919190808080819, 0x1919190808081908, 0x191919080808192b,
    0x1919190808082b19, 0x1919190808190808, 0x191919080819082b, 0x1919190808191919,
    0x1919190808192b08, 0x19191908082b0819, 0x19191908082b1908, 0x1919190819080808,
    0x191919081908082b, 0x1919190819081919, 0x1919190819082b08, 0x1919190819190819,
    0x1919190819191908, 0x19191908192b0808, 0x191919082b080819, 0x191919082b081908,
    0x191919082b190808, 0x1919191908080808, 0x191919190808082b, 0x1919191908081919,
    0x1919191908082b08, 0x1919191908190819, 0x1919191908191908, 0x19191919082b0808,
    0x1919191919080819, 0x1919191919081908, 0x1919191919190808, 0x191919192b080808,
    0x1919192b08080819, 0x1919192b08081908, 0x1919192b08190808, 0x1919192b082b192b,
    0x1919192b19080808, 0x19192b0808080808, 0x19192b080808082b, 0x19192b0808081919,
    0x19192b0808082b08, 0x19192b0808190819, 0x19192b0808191908, 0x19192b08082b0808,
    0x19192b0819080819, 0x19192b0819081908, 0x19192b0819190808, 0x19192b0819192b2b,
    0x19192b082b080808, 0x19192b1908080819, 0x19192b1908081908, 0x19192b1908190808,
    0x19192b1919080808, 0x19192b2b08080808, 0x19192b2b08192b19, 0x19192b2b2b081919,
    0x19192b2b2b2b2b08, 0x192b080808080819, 0x192b080808081908, 0x192b08080808192b,
    0x192b080808190808, 0x192b08080819082b, 0x192b080808191919, 0x192b080808192b08,
    0x192b0808082b0819, 0x192b0808082b1908, 0x192b080819080808, 0x192b080819081919,
    0x192b080819082b08, 0x192b080819190819, 0x192b080819191908, 0x192b0808192b0808,
    0x192b08082b081908, 0x192b08082b190808, 0x192b081908080808, 0x192b08190808082b,
    0x192b081908081919, 0x192b081908082b08, 0x192b081908190819, 0x192b081908191908,
    0x192b0819082b0808, 0x192b081919080819, 0x192b081919081908, 0x192b081919190808,
    0x192b08192b080808, 0x192b08192b192b19, 0x192b082b08081908, 0x192b082b08190808,
    0x192b082b19080808, 0x192b082b1919192b, 0x192b082b2b2b0819, 0x192b190808080808,
    0x192b190808081919, 0x192b190808082b08, 0x192b190808190819, 0x192b190808191908,
    0x192b1908082b0808, 0x192b190819080819, 0x192b190819081908, 0x192b190819190808,
    0x192b19082b080808, 0x192b191908080819, 0x192b191908081908, 0x192b191908190808,
    0x192b191919080808, 0x192b191919082b2b, 0x192b1919192b2b08, 0x192b19192b19082b,
    0x192b192b08080808, 0x192b192b2b191908, 0x192b2b0808080819, 0x192b2b0808081908,
    0x192b2b0808190808, 0x192b2b08192b1919, 0x192b2b082b192b08, 0x192b2b1908080808,
    0x192b2b19082b2b2b, 0x192b2b2b1908082b, 0x192b2b2b2b2b0819, 0x2b08080808080808,
    0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08, 0x2b08080808190819,
    0x2b08080808191908, 0x2b08080808192b19, 0x2b080808082b0808, 0x2b080808082b1919,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808081919082b,
    0x2b08080819191919, 0x2b08080819192b08, 0x2b080808192b0819, 0x2b0808082b080808,
    0x2b0808082b081919, 0x2b0808082b190819, 0x2b0808082b191908, 0x2b08081908080819,
    0x2b08081908081908, 0x2b08081908082b19, 0x2b08081908190808, 0x2b0808190819082b,
    0x2b08081908191919, 0x2b08081908192b08, 0x2b080819082b0819, 0x2b080819082b1908,
    0x2b08081919080808, 0x2b0808191908082b, 0x2b08081919081919, 0x2b08081919082b08,
    0x2b08081919190819, 0x2b08081919191908, 0x2b0808192b080819, 0x2b0808192b081908,
    0x2b0808192b190808, 0x2b0808192b2b2b19, 0x2b08082b08080808, 0x2b08082b08081919,
    0x2b08082b08082b2b, 0x2b08082b08190819, 0x2b08082b08191908, 0x2b08082b19080819,
    0x2b08082b19081908, 0x2b08082b19190808, 0x2b08190808080819, 0x2b08190808081908,
    0x2b0819080808192b, 0x2b08190808082b19, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190808192b08, 0x2b081908082b0819, 0x2b08190819080808,
    0x2b0819081908082b, 0x2b08190819081919, 0x2b08190819082b08, 0x2b08190819190819,
    0x2b08190819191908, 0x2b081908192b0808, 0x2b0819082b080819, 0x2b0819082b081908,
    0x2b0819082b190808, 0x2b08191908080808, 0x2b0819190808082b, 0x2b08191908081919,
    0x2b08191908082b08, 0x2b08191908190819, 0x2b08191908191908, 0x2b081919082b0808,
    0x2b08191919080819, 0x2b08191919081908, 0x2b08191919190808, 0x2b0819192b080808,
    0x2b0819192b082b2b, 0x2b08192b08080819, 0x2b08192b08081908, 0x2b08192b08190808,
    0x2b08192b082b2b19, 0x2b08192b19080808, 0x2b082b0808080808, 0x2b082b0808081919,
    0x2b082b0808190819, 0x2b082b0808191908, 0x2b082b0819080819, 0x2b082b0819081908,
    0x2b082b0819190808, 0x2b082b082b2b082b, 0x2b082b1908080819, 0x2b082b1908081908,
    0x2b082b1919080808, 0x2b082b19192b1919, 0x2b082b2b082b082b, 0x2b082b2b19192b08,
    0x2b082b2b19192b2b, 0x2b082b2b2b08082b, 0x2b082b2b2b2b082b, 0x2b19080808080819,
    0x2b19080808081908, 0x2b19080808082b19, 0x2b19080808190808, 0x2b1908080819082b,
    0x2b19080808191919, 0x2b19080808192b08, 0x2b190808082b1908, 0x2b19080819080808,
    0x2b1908081908082b, 0x2b19080819081919, 0x2b19080819082b08, 0x2b19080819190819,
    0x2b19080819191908, 0x2b190808192b0808, 0x2b1908082b080819, 0x2b1908082b081908,
    0x2b1908082b190808, 0x2b19081908080808, 0x2b19081908081919, 0x2b19081908190819,
    0x2b19081908191908, 0x2b19081919080819, 0x2b19081919081908, 0x2b19081919190808,
    0x2b19081919192b2b, 0x2b19082b08080819, 0x2b19082b08081908, 0x2b19082b08190808,
    0x2b19082b19080808, 0x2b19082b2b2b192b, 0x2b19190808080808, 0x2b1919080808082b,
    0x2b19190808081919, 0x2b19190808082b08, 0x2b19190808190819, 0x2b19190808191908,
    0x2b191908082b0808, 0x2b19190819080819, 0x2b19190819081908, 0x2b19190819190808,
    0x2b1919082b080808, 0x2b1919082b19192b, 0x2b19191908080819, 0x2b19191908081908,
    0x2b19191908190808, 0x2b19191919080808, 0x2b1919192b192b08, 0x2b1919192b2b0819,
    0x2b19192b08080808, 0x2b19192b1908192b, 0x2b19192b192b1908, 0x2b192b0808080819,
    0x2b192b0808081908, 0x2b192b0808190808, 0x2b192b08082b192b, 0x2b192b0819080808,
    0x2b192b082b2b2b19, 0x2b192b1908080808, 0x2b192b1919082b19, 0x2b192b191919082b,
    0x2b192b2b2b190808, 0x2b2b080808080808, 0x2b2b080808081919, 0x2b2b080808082b2b,
    0x2b2b080808191908, 0x2b2b0808082b082b, 0x2b2b0808082b2b2b, 0x2b2b080819080819,
    0x2b2b080819081908, 0x2b2b080819190808, 0x2b2b08082b2b082b, 0x2b2b08082b2b2b2b,
    0x2b2b081919080808, 0x2b2b0819192b1919, 0x2b2b082b0808082b, 0x2b2b082b08082b2b,
    0x2b2b082b082b082b, 0x2b2b082b082b2b08, 0x2b2b082b082b2b2b, 0x2b2b082b2b08082b,
    0x2b2b082b2b082b08, 0x2b2b082b2b082b2b, 0x2b2b082b2b2b2b08, 0x2b2b190808080819,
    0x2b2b190808081908, 0x2b2b190808190808, 0x2b2b190819080808, 0x2b2b19082b082b19,
    0x2b2b19082b2b1908, 0x2b2b191908080808, 0x2b2b191908192b19, 0x2b2b192b19190819,
    0x2b2b2b0808082b2b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b082b, 0x2b2b2b1919191908,
    0x2b2b2b192b08192b, 0x2b2b2b2b08082b08, 0x2b2b2b2b08082b2b, 0x2b2b2b2b082b0808,
    0x2b2b2b2b082b082b, 0x2b2b2b2b082b2b08, 0x2b2b2b2b2b082b08, 0x2b2b2b2b2b2b2b2b,
};

#define IQ1S_DELTA 0.125f

static const uint64_t iq1s_grid[2048] = {
    0xffffffffffffffff, 0xffffffffffffff01, 0xffffffffffff0000, 0xffffffffffff01ff,
    0xffffffffffff0101, 0xffffffffff00ff00, 0xffffffffff000000, 0xffffffffff01ffff,
    0xffffffffff01ff01, 0xffffffffff0101ff, 0xffffffffff010101, 0xffffffff00ff0000,
    0xffffffff0000ff00, 0xffffffff000000ff, 0xffffffff00000001, 0xffffffff00010000,
    0xffffffff01ffffff, 0xffffffff01ffff01, 0xffffffff01ff01ff, 0xffffffff01ff0101,
    0xffffffff01000000, 0xffffffff0101ffff, 0xffffffff0101ff01, 0xffffffff010101ff,
    0xffffffff01010101, 0xffffff00ffff00ff, 0xffffff00ffff0000, 0xffffff00ff00ff00,
    0xffffff00ff0000ff, 0xffffff00ff000001, 0xffffff00ff000100, 0xffffff00ff000101,
    0xffffff00ff010000, 0xffffff0000ffff00, 0xffffff0000ff0001, 0xffffff0000ff0100,
    0xffffff000000ff01, 0xffffff0000000000, 0xffffff0000000101, 0xffffff000001ff00,
    0xffffff00000100ff, 0xffffff0000010001, 0xffffff00000101ff, 0xffffff0001ff0000,
    0xffffff000100ff00, 0xffffff00010000ff, 0xffffff0001000001, 0xffffff0001010000,
    0xffffff01ffffffff, 0xffffff01ffffff01, 0xffffff01ffff01ff, 0xffffff01ffff0101,
    0xffffff01ff000000, 0xffffff01ff01ffff, 0xffffff01ff01ff01, 0xffffff01ff0101ff,
    0xffffff01ff010101, 0xffffff0100ff0000, 0xffffff010000ff00, 0xffffff0100000100,
    0xffffff01000100ff, 0xffffff0100010100, 0xffffff0101ffffff, 0xffffff0101ffff01,
    0xffffff0101ff01ff, 0xffffff0101ff0101, 0xffffff010100ff00, 0xffffff0101000000,
    0xffffff0101000100, 0xffffff010101ffff, 0xffffff010101ff01, 0xffffff01010101ff,
    0xffffff0101010101, 0xffff00ffff00ff00, 0xffff00ffff0000ff, 0xffff00ffff000001,
    0xffff00ffff010000, 0xffff00ff00ffff00, 0xffff00ff00ff0100, 0xffff00ff00000000,
    0xffff00ff00000101, 0xffff00ff000100ff, 0xffff00ff00010000, 0xffff00ff0100ff00,
    0xffff00ff01000100, 0xffff00ff01010000, 0xffff0000ffffff00, 0xffff0000ffff00ff,
    0xffff0000ffff0000, 0xffff0000ffff0001, 0xffff0000ff000000, 0xffff0000ff0001ff,
    0xffff0000ff000101, 0xffff0000ff010100, 0xffff000000ffffff, 0xffff000000ff0000,
    0xffff000000ff0101, 0xffff00000000ffff, 0xffff00000000ff00, 0xffff0000000000ff,
    0xffff000000000000, 0xffff000000000001, 0xffff000000000100, 0xffff00000001ffff,
    0xffff00000001ff01, 0xffff000000010000, 0xffff0000000101ff, 0xffff000000010101,
    0xffff000001ffff00, 0xffff00000100ff00, 0xffff000001000000, 0xffff0000010001ff,
    0xffff000001000101, 0xffff00000101ff00, 0xffff0000010100ff, 0xffff000001010000,
    0xffff000001010001, 0xffff000001010100, 0xffff0001ff0000ff, 0xffff0001ff000100,
    0xffff000100ffff00, 0xffff000100ff00ff, 0xffff00010000ffff, 0xffff00010000ff01,
    0xffff000100000000, 0xffff0001000001ff, 0xffff00010001ffff, 0xffff00010001ff00,
    0xffff000100010001, 0xffff000100010100, 0xffff000101ff0000, 0xffff00010100ff00,
    0xffff0001010000ff, 0xffff000101000100, 0xffff01ffffffffff, 0xffff01ffffffff01,
    0xffff01ffffff01ff, 0xffff01ffffff0101, 0xffff01ffff000000, 0xffff01ffff01ffff,
    0xffff01ffff01ff01, 0xffff01ffff0101ff, 0xffff01ffff010101, 0xffff01ff00ff0000,
    0xffff01ff0000ff00, 0xffff01ff00000001, 0xffff01ff00010000, 0xffff01ff01ffffff,
    0xffff01ff01ffff01, 0xffff01ff01ff01ff, 0xffff01ff01ff0101, 0xffff01ff01000000,
    0xffff01ff0101ffff, 0xffff01ff0101ff01, 0xffff01ff010101ff, 0xffff01ff01010101,
    0xffff0100ffff0000, 0xffff0100ff00ff00, 0xffff0100ff0000ff, 0xffff0100ff000100,
    0xffff0100ff0100ff, 0xffff0100ff010000, 0xffff010000ffff00, 0xffff01000000ffff,
    0xffff01000000ff00, 0xffff010000000000, 0xffff01000001ff00, 0xffff0100000100ff,
    0xffff010000010100, 0xffff01000100ff00, 0xffff0100010000ff, 0xffff010001000001,
    0xffff010001000100, 0xffff010001010000, 0xffff0101ffffffff, 0xffff0101ffffff01,
    0xffff0101ffff01ff, 0xffff0101ffff0101, 0xffff0101ff000000, 0xffff0101ff01ffff,
    0xffff0101ff01ff01, 0xffff0101ff0101ff, 0xffff0101ff010101, 0xffff010100ff0000,
    0xffff01010000ff00, 0xffff010100000100, 0xffff01010001ff00, 0xffff010100010000,
    0xffff010101ffffff, 0xffff010101ffff01, 0xffff010101ff0000, 0xffff010101ff01ff,
    0xffff010101ff0101, 0xffff010101000000, 0xffff01010101ffff, 0xffff01010101ff01,
    0xffff0101010101ff, 0xffff010101010101, 0xff00ffffff00ffff, 0xff00ffffff00ff00,
    0xff00ffffff0000ff, 0xff00ffffff000100, 0xff00ffffff0100ff, 0xff00ffffff010000,
    0xff00ffff00ffff00, 0xff00ffff00ff00ff, 0xff00ffff0000ffff, 0xff00ffff00000000,
    0xff00ffff000001ff, 0xff00ffff0001ff00, 0xff00ffff000100ff, 0xff00ffff00010000,
    0xff00ffff00010100, 0xff00ffff0100ff00, 0xff00ffff010000ff, 0xff00ffff01000001,
    0xff00ffff0101ff00, 0xff00ffff01010000, 0xff00ff00ffffff00, 0xff00ff00ffff00ff,
    0xff00ff00ffff0001, 0xff00ff00ffff0100, 0xff00ff00ff00ffff, 0xff00ff00ff00ff01,
    0xff00ff00ff000000, 0xff00ff00ff0001ff, 0xff00ff00ff01ff00, 0xff00ff00ff0100ff,
    0xff00ff00ff010100, 0xff00ff0000ff0000, 0xff00ff0000ff0101, 0xff00ff000000ffff,
    0xff00ff000000ff00, 0xff00ff000000ff01, 0xff00ff00000000ff, 0xff00ff0000000000,
    0xff00ff0000000001, 0xff00ff0000000100, 0xff00ff000001ffff, 0xff00ff0000010000,
    0xff00ff0001ff00ff, 0xff00ff000100ff01, 0xff00ff0001000000, 0xff00ff000101ff00,
    0xff00ff00010100ff, 0xff00ff01ff00ff00, 0xff00ff01ff0000ff, 0xff00ff01ff000001,
    0xff00ff01ff010000, 0xff00ff0100ffffff, 0xff00ff0100ff0001, 0xff00ff0100ff0100,
    0xff00ff010000ff01, 0xff00ff0100000000, 0xff00ff01000001ff, 0xff00ff0100000101,
    0xff00ff01000100ff, 0xff00ff0100010001, 0xff00ff0101ff0000, 0xff00ff010100ff00,
    0xff00ff01010000ff, 0xff00ff0101000001, 0xff00ff0101010000, 0xff0000ffffffff00,
    0xff0000ffffff0001, 0xff0000ffffff0100, 0xff0000ffff0000ff, 0xff0000ffff000000,
    0xff0000ffff0001ff, 0xff0000ffff000100, 0xff0000ffff01ff00, 0xff0000ffff010001,
    0xff0000ff00ffff00, 0xff0000ff00ff0000, 0xff0000ff00ff0001, 0xff0000ff00ff01ff,
    0xff0000ff00ff0101, 0xff0000ff0000ff00, 0xff0000ff000000ff, 0xff0000ff00000000,
    0xff0000ff00000001, 0xff0000ff00000100, 0xff0000ff0001ff01, 0xff0000ff00010000,
    0xff0000ff000101ff, 0xff0000ff01ff00ff, 0xff0000ff01ff0100, 0xff0000ff0100ffff,
    0xff0000ff010000ff, 0xff0000ff01000000, 0xff0000ff010001ff, 0xff0000ff01000100,
    0xff0000ff01000101, 0xff0000ff0101ff00, 0xff0000ff010100ff, 0xff0000ff01010000,
    0xff0000ff01010100, 0xff000000ffffff01, 0xff000000ffff0000, 0xff000000ffff0101,
    0xff000000ff00ff00, 0xff000000ff0000ff, 0xff000000ff000000, 0xff000000ff000001,
    0xff000000ff000100, 0xff000000ff01ffff, 0xff000000ff01ff01, 0xff000000ff010000,
    0xff000000ff0101ff, 0xff000000ff010101, 0xff00000000ffff00, 0xff00000000ff00ff,
    0xff00000000ff0000, 0xff00000000ff0001, 0xff0000000000ff00, 0xff0000000000ff01,
    0xff000000000000ff, 0xff00000000000000, 0xff00000000000001, 0xff00000000000100,
    0xff00000000000101, 0xff0000000001ff00, 0xff000000000100ff, 0xff00000000010000,
    0xff00000000010001, 0xff00000000010100, 0xff00000001ffffff, 0xff00000001ffff01,
    0xff00000001ff00ff, 0xff00000001ff0000, 0xff00000001ff01ff, 0xff00000001ff0101,
    0xff0000000100ffff, 0xff0000000100ff00, 0xff000000010000ff, 0xff00000001000000,
    0xff00000001000001, 0xff00000001000100, 0xff00000001000101, 0xff0000000101ffff,
    0xff0000000101ff01, 0xff00000001010000, 0xff000001ffffff00, 0xff000001ffff00ff,
    0xff000001ffff0000, 0xff000001ffff0001, 0xff000001ff000000, 0xff000001ff000001,
    0xff000001ff0001ff, 0xff000001ff000101, 0xff000001ff01ff00, 0xff000001ff010001,
    0xff00000100ffffff, 0xff00000100ffff01, 0xff00000100ff00ff, 0xff00000100ff0000,
    0xff00000100ff01ff, 0xff00000100ff0101, 0xff0000010000ff00, 0xff00000100000000,
    0xff00000100000001, 0xff000001000001ff, 0xff00000100000100, 0xff0000010001ff00,
    0xff000001000100ff, 0xff00000100010000, 0xff000001000101ff, 0xff00000100010100,
    0xff00000100010101, 0xff00000101ff0001, 0xff00000101ff0101, 0xff0000010100ff01,
    0xff00000101000000, 0xff000001010100ff, 0xff00000101010100, 0xff0001ffff00ff00,
    0xff0001ffff000001, 0xff0001ffff010000, 0xff0001ff00ffff00, 0xff0001ff00ff00ff,
    0xff0001ff00ff0001, 0xff0001ff00ff0100, 0xff0001ff0000ffff, 0xff0001ff00000000,
    0xff0001ff000001ff, 0xff0001ff00000101, 0xff0001ff0001ffff, 0xff0001ff0001ff00,
    0xff0001ff000100ff, 0xff0001ff00010001, 0xff0001ff00010100, 0xff0001ff01ff0000,
    0xff0001ff0100ff00, 0xff0001ff010000ff, 0xff0001ff01010000, 0xff000100ff00ffff,
    0xff000100ff00ff01, 0xff000100ff000000, 0xff000100ff000101, 0xff000100ff01ff00,
    0xff000100ff010000, 0xff00010000ffff01, 0xff00010000ff00ff, 0xff00010000ff0000,
    0xff00010000ff01ff, 0xff0001000000ff00, 0xff000100000000ff, 0xff00010000000000,
    0xff00010000000001, 0xff00010000000100, 0xff00010000000101, 0xff0001000001ffff,
    0xff00010000010000, 0xff00010000010101, 0xff00010001ff0100, 0xff0001000100ff00,
    0xff0001000100ff01, 0xff00010001000000, 0xff000100010001ff, 0xff0001000101ff00,
    0xff00010001010001, 0xff00010001010100, 0xff000101ffff0100, 0xff000101ff000001,
    0xff000101ff0100ff, 0xff000101ff010001, 0xff00010100ff00ff, 0xff00010100ff0001,
    0xff00010100ff0100, 0xff0001010000ffff, 0xff0001010000ff01, 0xff00010100000000,
    0xff000101000001ff, 0xff0001010001ff00, 0xff00010100010001, 0xff00010100010100,
    0xff00010101ff0000, 0xff0001010100ff00, 0xff00010101000001, 0xff00010101000101,
    0xff01ffffffffffff, 0xff01ffffffffff01, 0xff01ffffffff01ff, 0xff01ffffffff0101,
    0xff01ffffff000000, 0xff01ffffff01ffff, 0xff01ffffff01ff01, 0xff01ffffff010000,
    0xff01ffffff0101ff, 0xff01ffffff010101, 0xff01ffff00ff0000, 0xff01ffff0000ff00,
    0xff01ffff00000100, 0xff01ffff0001ff00, 0xff01ffff00010000, 0xff01ffff01ffffff,
    0xff01ffff01ffff01, 0xff01ffff01ff01ff, 0xff01ffff01ff0101, 0xff01ffff01000000,
    0xff01ffff0101ffff, 0xff01ffff0101ff01, 0xff01ffff01010000, 0xff01ffff010101ff,
    0xff01ffff01010101, 0xff01ff00ffff0000, 0xff01ff00ff00ff00, 0xff01ff00ff0000ff,
    0xff01ff00ff000100, 0xff01ff00ff010000, 0xff01ff0000ffff01, 0xff01ff0000ff00ff,
    0xff01ff0000ff0100, 0xff01ff0000000000, 0xff01ff00000001ff, 0xff01ff0000000101,
    0xff01ff000001ff00, 0xff01ff00000100ff, 0xff01ff0000010000, 0xff01ff0000010001,
    0xff01ff0001ff0000, 0xff01ff000100ffff, 0xff01ff0001000001, 0xff01ff0001000100,
    0xff01ff0001010000, 0xff01ff01ffffff00, 0xff01ff01ffff01ff, 0xff01ff01ffff0101,
    0xff01ff01ff00ff00, 0xff01ff01ff000000, 0xff01ff01ff01ffff, 0xff01ff01ff01ff01,
    0xff01ff01ff0101ff, 0xff01ff01ff010101, 0xff01ff0100ff0000, 0xff01ff010000ff00,
    0xff01ff0100000001, 0xff01ff0100000100, 0xff01ff0100010000, 0xff01ff0101ffff00,
    0xff01ff0101ff01ff, 0xff01ff0101ff0101, 0xff01ff010100ff00, 0xff01ff0101000000,
    0xff01ff010101ffff, 0xff01ff010101ff01, 0xff01ff01010101ff, 0xff01ff0101010101,
    0xff0100ffffff0000, 0xff0100ffff0000ff, 0xff0100ffff000001, 0xff0100ffff000100,
    0xff0100ffff010000, 0xff0100ff00ff00ff, 0xff0100ff00ff0000, 0xff0100ff00ff0001,
    0xff0100ff00ff0100, 0xff0100ff0000ff01, 0xff0100ff00000000, 0xff0100ff000001ff,
    0xff0100ff00000101, 0xff0100ff00010001, 0xff0100ff01ff0000, 0xff0100ff0100ff00,
    0xff0100ff010000ff, 0xff0100ff01000100, 0xff0100ff0101ff00, 0xff0100ff01010000,
    0xff010000ffff0100, 0xff010000ff000000, 0xff010000ff01ff00, 0xff010000ff010100,
    0xff01000000ffffff, 0xff01000000ff0000, 0xff01000000ff01ff, 0xff0100000000ff00,
    0xff010000000000ff, 0xff01000000000000, 0xff01000000000100, 0xff0100000001ff01,
    0xff01000000010000, 0xff010000000101ff, 0xff01000001ff0100, 0xff0100000100ffff,
    0xff010000010000ff, 0xff01000001000000, 0xff010000010001ff, 0xff01000001000101,
    0xff0100000101ff00, 0xff010000010100ff, 0xff01000001010001, 0xff01000001010100,
    0xff010001ffff0000, 0xff010001ff00ffff, 0xff010001ff00ff01, 0xff010001ff000100,
    0xff010001ff010000, 0xff01000100ffff00, 0xff01000100ff0100, 0xff01000100000000,
    0xff0100010001ffff, 0xff0100010001ff00, 0xff01000100010100, 0xff01000101ff00ff,
    0xff01000101ff0001, 0xff0100010100ffff, 0xff01000101000101, 0xff0101ffffffffff,
    0xff0101ffffffff01, 0xff0101ffffff01ff, 0xff0101ffffff0101, 0xff0101ffff000000,
    0xff0101ffff01ffff, 0xff0101ffff01ff01, 0xff0101ffff0101ff, 0xff0101ffff010101,
    0xff0101ff00ff0000, 0xff0101ff0000ff00, 0xff0101ff000000ff, 0xff0101ff00010000,
    0xff0101ff01ffffff, 0xff0101ff01ffff01, 0xff0101ff01ff01ff, 0xff0101ff01ff0101,
    0xff0101ff0101ffff, 0xff0101ff0101ff01, 0xff0101ff010101ff, 0xff0101ff01010101,
    0xff010100ffff0100, 0xff010100ff00ff00, 0xff010100ff0000ff, 0xff010100ff000100,
    0xff010100ff010000, 0xff01010000ff0001, 0xff01010000ff0100, 0xff0101000000ff01,
    0xff01010000000000, 0xff0101000001ff00, 0xff010100000100ff, 0xff01010000010001,
    0xff01010000010100, 0xff01010001ff0000, 0xff0101000100ffff, 0xff01010001000001,
    0xff01010001000100, 0xff010100010100ff, 0xff01010001010000, 0xff010101ffffffff,
    0xff010101ffffff01, 0xff010101ffff01ff, 0xff010101ffff0101, 0xff010101ff01ffff,
    0xff010101ff01ff01, 0xff010101ff0101ff, 0xff010101ff010101, 0xff01010100ff0000,
    0xff0101010000ff00, 0xff01010100000001, 0xff01010100000100, 0xff01010100010000,
    0xff01010101ffffff, 0xff01010101ffff01, 0xff01010101ff01ff, 0xff01010101ff0101,
    0xff01010101000000, 0xff0101010101ffff, 0xff0101010101ff01, 0xff010101010101ff,
    0xff01010101010101, 0x00ffffffffff0000, 0x00ffffffff00ff00, 0x00ffffffff000001,
    0x00ffffffff010000, 0x00ffffff00ff0100, 0x00ffffff0000ff01, 0x00ffffff00000000,
    0x00ffffff000001ff, 0x00ffffff00000101, 0x00ffffff0001ff00, 0x00ffffff000100ff,
    0x00ffffff00010001, 0x00ffffff010000ff, 0x00ffffff01000100, 0x00ffffff0101ff00,
    0x00ffffff01010001, 0x00ffff00ffffffff, 0x00ffff00ffffff00, 0x00ffff00ffff00ff,
    0x00ffff00ffff0001, 0x00ffff00ffff0100, 0x00ffff00ff00ff01, 0x00ffff00ff000000,
    0x00ffff00ff000001, 0x00ffff00ff0001ff, 0x00ffff00ff000101, 0x00ffff00ff01ff00,
    0x00ffff00ff010001, 0x00ffff00ff010100, 0x00ffff0000ff0000, 0x00ffff0000ff01ff,
    0x00ffff0000ff0101, 0x00ffff000000ff00, 0x00ffff00000000ff, 0x00ffff0000000000,
    0x00ffff0000000001, 0x00ffff0000000100, 0x00ffff0000000101, 0x00ffff0000010000,
    0x00ffff00000101ff, 0x00ffff0000010101, 0x00ffff0001ffff00, 0x00ffff0001ff00ff,
    0x00ffff0001ff0001, 0x00ffff000100ffff, 0x00ffff000100ff01, 0x00ffff0001000000,
    0x00ffff000101ffff, 0x00ffff000101ff00, 0x00ffff000101ff01, 0x00ffff01ffff0000,
    0x00ffff01ff00ff00, 0x00ffff01ff0000ff, 0x00ffff01ff000001, 0x00ffff01ff010000,
    0x00ffff0100ffff00, 0x00ffff010000ff01, 0x00ffff0100000000, 0x00ffff0100000101,
    0x00ffff01000100ff, 0x00ffff0100010100, 0x00ffff0101ff0100, 0x00ffff01010000ff,
    0x00ffff0101010000, 0x00ff00ffffffff00, 0x00ff00ffff000000, 0x00ff00ffff000100,
    0x00ff00ffff010100, 0x00ff00ff00ff0000, 0x00ff00ff00ff01ff, 0x00ff00ff00ff0101,
    0x00ff00ff0000ff00, 0x00ff00ff000000ff, 0x00ff00ff00000000, 0x00ff00ff00000001,
    0x00ff00ff0001ff00, 0x00ff00ff0001ff01, 0x00ff00ff00010000, 0x00ff00ff000101ff,
    0x00ff00ff00010101, 0x00ff00ff01ffff00, 0x00ff00ff01ff0001, 0x00ff00ff01ff0100,
    0x00ff00ff0100ffff, 0x00ff00ff0100ff01, 0x00ff00ff01000000, 0x00ff00ff0101ffff,
    0x00ff00ff0101ff00, 0x00ff00ff01010100, 0x00ff0000ffffff00, 0x00ff0000ffffff01,
    0x00ff0000ffff0000, 0x00ff0000ffff0101, 0x00ff0000ff00ff00, 0x00ff0000ff0000ff,
    0x00ff0000ff000000, 0x00ff0000ff000001, 0x00ff0000ff000100, 0x00ff0000ff01ffff,
    0x00ff0000ff010000, 0x00ff0000ff010101, 0x00ff000000ffff00, 0x00ff000000ff00ff,
    0x00ff000000ff0000, 0x00ff000000ff0001, 0x00ff000000ff0100, 0x00ff00000000ffff,
    0x00ff00000000ff00, 0x00ff0000000000ff, 0x00ff000000000000, 0x00ff000000000001,
    0x00ff0000000001ff, 0x00ff000000000100, 0x00ff00000001ff00, 0x00ff0000000100ff,
    0x00ff000000010000, 0x00ff000000010001, 0x00ff000000010100, 0x00ff000001ffff01,
    0x00ff000001ff00ff, 0x00ff000001ff0000, 0x00ff000001ff01ff, 0x00ff00000100ff00,
    0x00ff0000010000ff, 0x00ff000001000000, 0x00ff000001000001, 0x00ff000001000100,
    0x00ff000001000101, 0x00ff000001010000, 0x00ff0000010101ff, 0x00ff000001010101,
    0x00ff0001ffffff00, 0x00ff0001ffff0000, 0x00ff0001ffff0100, 0x00ff0001ff0000ff,
    0x00ff0001ff000000, 0x00ff0001ff0001ff, 0x00ff0001ff000101, 0x00ff0001ff01ff00,
    0x00ff0001ff0100ff, 0x00ff0001ff010100, 0x00ff000100ffffff, 0x00ff000100ffff01,
    0x00ff000100ff0000, 0x00ff000100ff01ff, 0x00ff00010000ffff, 0x00ff00010000ff00,
    0x00ff00010000ff01, 0x00ff000100000000, 0x00ff000100000001, 0x00ff000100000100,
    0x00ff00010001ff01, 0x00ff000100010000, 0x00ff0001000101ff, 0x00ff000101ffff00,
    0x00ff000101ff0000, 0x00ff000101ff0101, 0x00ff0001010000ff, 0x00ff000101000000,
    0x00ff00010101ff00, 0x00ff0001010100ff, 0x00ff000101010001, 0x00ff01ffffff0000,
    0x00ff01ffff00ff00, 0x00ff01ffff000000, 0x00ff01ffff000101, 0x00ff01ffff010000,
    0x00ff01ff00ffff01, 0x00ff01ff00ff0100, 0x00ff01ff0000ffff, 0x00ff01ff00000000,
    0x00ff01ff000001ff, 0x00ff01ff0001ff00, 0x00ff01ff000100ff, 0x00ff01ff00010001,
    0x00ff01ff00010100, 0x00ff01ff01ff0000, 0x00ff01ff0100ff00, 0x00ff01ff010000ff,
    0x00ff01ff01000001, 0x00ff01ff01000100, 0x00ff01ff01010000, 0x00ff0100ffffff00,
    0x00ff0100ffff0000, 0x00ff0100ffff0001, 0x00ff0100ffff0101, 0x00ff0100ff00ffff,
    0x00ff0100ff0000ff, 0x00ff0100ff000000, 0x00ff0100ff0001ff, 0x00ff0100ff01ff00,
    0x00ff0100ff0100ff, 0x00ff0100ff010001, 0x00ff010000ffffff, 0x00ff010000ff0000,
    0x00ff010000ff0101, 0x00ff01000000ff00, 0x00ff01000000ff01, 0x00ff0100000000ff,
    0x00ff010000000000, 0x00ff010000000001, 0x00ff010000000100, 0x00ff01000001ffff,
    0x00ff01000001ff01, 0x00ff010000010000, 0x00ff010000010001, 0x00ff010000010101,
    0x00ff010001ff0001, 0x00ff010001ff0100, 0x00ff01000100ff01, 0x00ff010001000000,
    0x00ff010001000001, 0x00ff0100010001ff, 0x00ff01000101ff00, 0x00ff0100010100ff,
    0x00ff010001010001, 0x00ff010001010100, 0x00ff0101ff000001, 0x00ff010100ff00ff,
    0x00ff010100ff0001, 0x00ff010100ff0100, 0x00ff010100000000, 0x00ff0101000001ff,
    0x00ff010100000101, 0x00ff0101000100ff, 0x00ff010100010100, 0x00ff0101010000ff,
    0x00ff010101010000, 0x0000ffffffffff00, 0x0000ffffffff00ff, 0x0000ffffffff0000,
    0x0000ffffffff0001, 0x0000ffffffff0100, 0x0000ffffff00ff01, 0x0000ffffff000000,
    0x0000ffffff000101, 0x0000ffffff01ff00, 0x0000ffffff0100ff, 0x0000ffffff010100,
    0x0000ffff00ffffff, 0x0000ffff00ff0000, 0x0000ffff00ff01ff, 0x0000ffff0000ff00,
    0x0000ffff000000ff, 0x0000ffff00000000, 0x0000ffff00000001, 0x0000ffff00000100,
    0x0000ffff00010000, 0x0000ffff000101ff, 0x0000ffff01ff0001, 0x0000ffff01ff0100,
    0x0000ffff01000000, 0x0000ffff010001ff, 0x0000ffff0101ffff, 0x0000ffff0101ff00,
    0x0000ffff01010001, 0x0000ffff01010100, 0x0000ff00ffff0000, 0x0000ff00ffff01ff,
    0x0000ff00ffff0100, 0x0000ff00ffff0101, 0x0000ff00ff00ff00, 0x0000ff00ff0000ff,
    0x0000ff00ff000000, 0x0000ff00ff000001, 0x0000ff00ff0001ff, 0x0000ff00ff000100,
    0x0000ff00ff01ffff, 0x0000ff00ff010000, 0x0000ff00ff010001, 0x0000ff00ff0101ff,
    0x0000ff00ff010101, 0x0000ff0000ffff00, 0x0000ff0000ff00ff, 0x0000ff0000ff0000,
    0x0000ff0000ff0001, 0x0000ff0000ff0100, 0x0000ff000000ffff, 0x0000ff000000ff00,
    0x0000ff000000ff01, 0x0000ff00000000ff, 0x0000ff0000000000, 0x0000ff0000000001,
    0x0000ff00000001ff, 0x0000ff0000000100, 0x0000ff0000000101, 0x0000ff000001ff00,
    0x0000ff00000100ff, 0x0000ff0000010000, 0x0000ff0000010001, 0x0000ff0000010100,
    0x0000ff0001ffff01, 0x0000ff0001ff0000, 0x0000ff000100ff00, 0x0000ff00010000ff,
    0x0000ff0001000000, 0x0000ff0001000001, 0x0000ff0001000100, 0x0000ff000101ffff,
    0x0000ff0001010000, 0x0000ff0001010101, 0x0000ff01ffffff00, 0x0000ff01ffff0001,
    0x0000ff01ff00ff01, 0x0000ff01ff000000, 0x0000ff01ff000101, 0x0000ff01ff01ff00,
    0x0000ff01ff0100ff, 0x0000ff0100ffff01, 0x0000ff0100ff0000, 0x0000ff0100ff0101,
    0x0000ff010000ff00, 0x0000ff01000000ff, 0x0000ff0100000000, 0x0000ff0100000001,
    0x0000ff0100000100, 0x0000ff010001ff01, 0x0000ff0100010000, 0x0000ff0101ff0000,
    0x0000ff010100ffff, 0x0000ff010100ff01, 0x0000ff0101000000, 0x0000ff0101000100,
    0x0000ff0101000101, 0x0000ff01010100ff, 0x000000ffffff00ff, 0x000000ffffff0000,
    0x000000ffff00ff00, 0x000000ffff0000ff, 0x000000ffff000000, 0x000000ffff000001,
    0x000000ffff0001ff, 0x000000ffff000100, 0x000000ffff01ff00, 0x000000ffff010000,
    0x000000ffff0101ff, 0x000000ffff010101, 0x000000ff00ffff00, 0x000000ff00ff00ff,
    0x000000ff00ff0000, 0x000000ff00ff0001, 0x000000ff00ff0100, 0x000000ff00ff0101,
    0x000000ff0000ffff, 0x000000ff0000ff00, 0x000000ff000000ff, 0x000000ff00000000,
    0x000000ff00000001, 0x000000ff000001ff, 0x000000ff00000100, 0x000000ff00000101,
    0x000000ff0001ff00, 0x000000ff0001ff01, 0x000000ff000100ff, 0x000000ff00010000,
    0x000000ff00010001, 0x000000ff00010100, 0x000000ff01ffffff, 0x000000ff01ff01ff,
    0x000000ff01ff0101, 0x000000ff0100ff00, 0x000000ff010000ff, 0x000000ff01000000,
    0x000000ff01000001, 0x000000ff01000100, 0x000000ff0101ff00, 0x000000ff010100ff,
    0x000000ff01010000, 0x000000ff01010101, 0x00000000ffffff00, 0x00000000ffffff01,
    0x00000000ffff00ff, 0x00000000ffff0000, 0x00000000ffff0001, 0x00000000ffff0100,
    0x00000000ff00ffff, 0x00000000ff00ff00, 0x00000000ff00ff01, 0x00000000ff0000ff,
    0x00000000ff000000, 0x00000000ff000001, 0x00000000ff000100, 0x00000000ff000101,
    0x00000000ff01ff00, 0x00000000ff0100ff, 0x00000000ff010000, 0x00000000ff010001,
    0x00000000ff010100, 0x0000000000ffffff, 0x0000000000ffff00, 0x0000000000ffff01,
    0x0000000000ff00ff, 0x0000000000ff0000, 0x0000000000ff0001, 0x0000000000ff01ff,
    0x0000000000ff0100, 0x000000000000ffff, 0x000000000000ff00, 0x000000000000ff01,
    0x00000000000000ff, 0x0000000000000000, 0x0000000000000001, 0x00000000000001ff,
    0x0000000000000100, 0x0000000000000101, 0x000000000001ffff, 0x000000000001ff00,
    0x00000000000100ff, 0x0000000000010000, 0x0000000000010001, 0x00000000000101ff,
    0x0000000000010100, 0x0000000000010101, 0x0000000001ffff00, 0x0000000001ff00ff,
    0x0000000001ff0000, 0x0000000001ff0100, 0x0000000001ff0101, 0x000000000100ffff,
    0x000000000100ff00, 0x00000000010000ff, 0x0000000001000000, 0x0000000001000001,
    0x00000000010001ff, 0x0000000001000100, 0x000000000101ff00, 0x00000000010100ff,
    0x0000000001010000, 0x0000000001010001, 0x0000000001010100, 0x00000001ffffffff,
    0x00000001ffffff00, 0x00000001ffffff01, 0x00000001ffff00ff, 0x00000001ffff0001,
    0x00000001ffff01ff, 0x00000001ffff0100, 0x00000001ff00ff00, 0x00000001ff0000ff,
    0x00000001ff000000, 0x00000001ff0001ff, 0x00000001ff000100, 0x00000001ff01ffff,
    0x00000001ff01ff00, 0x00000001ff01ff01, 0x00000001ff0100ff, 0x00000001ff010000,
    0x00000001ff010001, 0x00000001ff0101ff, 0x00000001ff010100, 0x0000000100ffff00,
    0x0000000100ff0000, 0x0000000100ff0001, 0x0000000100ff01ff, 0x0000000100ff0100,
    0x0000000100ff0101, 0x000000010000ffff, 0x000000010000ff00, 0x000000010000ff01,
    0x00000001000000ff, 0x0000000100000000, 0x0000000100000001, 0x00000001000001ff,
    0x0000000100000100, 0x0000000100000101, 0x000000010001ff00, 0x00000001000100ff,
    0x0000000100010000, 0x0000000100010100, 0x0000000101ffff01, 0x0000000101ff0000,
    0x0000000101ff0001, 0x0000000101ff01ff, 0x0000000101ff0100, 0x0000000101ff0101,
    0x000000010100ff00, 0x0000000101000000, 0x0000000101000101, 0x000000010101ff01,
    0x0000000101010000, 0x0000000101010001, 0x00000001010101ff, 0x0000000101010100,
    0x000001ffffff00ff, 0x000001ffffff0000, 0x000001ffffff0001, 0x000001ffffff0100,
    0x000001ffff00ffff, 0x000001ffff000000, 0x000001ffff0001ff, 0x000001ffff01ff00,
    0x000001ffff010101, 0x000001ff00ff0000, 0x000001ff00ff01ff, 0x000001ff00ff0101,
    0x000001ff0000ff00, 0x000001ff000000ff, 0x000001ff00000000, 0x000001ff00000001,
    0x000001ff000001ff, 0x000001ff00000100, 0x000001ff0001ffff, 0x000001ff0001ff01,
    0x000001ff000100ff, 0x000001ff00010000, 0x000001ff01ffff01, 0x000001ff01ff0100,
    0x000001ff0100ffff, 0x000001ff0100ff01, 0x000001ff01000000, 0x000001ff010001ff,
    0x000001ff0101ff00, 0x000001ff01010100, 0x00000100ffffff00, 0x00000100ffffff01,
    0x00000100ffff0000, 0x00000100ffff0101, 0x00000100ff00ff00, 0x00000100ff0000ff,
    0x00000100ff000000, 0x00000100ff000001, 0x00000100ff000100, 0x00000100ff010000,
    0x0000010000ffff00, 0x0000010000ff00ff, 0x0000010000ff0000, 0x0000010000ff0001,
    0x0000010000ff0100, 0x000001000000ffff, 0x000001000000ff00, 0x000001000000ff01,
    0x00000100000000ff, 0x0000010000000000, 0x0000010000000001, 0x00000100000001ff,
    0x0000010000000100, 0x0000010000000101, 0x000001000001ff00, 0x00000100000100ff,
    0x0000010000010000, 0x0000010000010001, 0x0000010000010100, 0x0000010001ffff00,
    0x0000010001ff0000, 0x0000010001ff0100, 0x000001000100ff00, 0x00000100010000ff,
    0x0000010001000000, 0x0000010001000001, 0x00000100010001ff, 0x0000010001000100,
    0x0000010001010000, 0x00000101ffff00ff, 0x00000101ffff01ff, 0x00000101ff000000,
    0x00000101ff000101, 0x00000101ff01ffff, 0x00000101ff010000, 0x00000101ff010001,
    0x00000101ff010100, 0x0000010100ff0000, 0x0000010100ff01ff, 0x0000010100ff0100,
    0x000001010000ff00, 0x0000010100000000, 0x0000010100000001, 0x00000101000001ff,
    0x0000010100000100, 0x000001010001ff01, 0x0000010100010000, 0x00000101000101ff,
    0x0000010100010101, 0x0000010101ffff00, 0x0000010101ff0101, 0x000001010100ff01,
    0x0000010101000000, 0x0000010101000001, 0x00000101010001ff, 0x0000010101000101,
    0x000001010101ff00, 0x0001ffffffff0000, 0x0001ffffff0000ff, 0x0001ffffff000001,
    0x0001ffffff000100, 0x0001ffffff010000, 0x0001ffff00ff00ff, 0x0001ffff0000ffff,
    0x0001ffff00000000, 0x0001ffff00000001, 0x0001ffff000001ff, 0x0001ffff00000101,
    0x0001ffff0001ff00, 0x0001ffff000100ff, 0x0001ffff00010001, 0x0001ffff00010100,
    0x0001ffff01ffff00, 0x0001ffff01000001, 0x0001ffff01010000, 0x0001ff00ffffff00,
    0x0001ff00ffff00ff, 0x0001ff00ffff0001, 0x0001ff00ffff0100, 0x0001ff00ff00ff01,
    0x0001ff00ff000000, 0x0001ff00ff01ff00, 0x0001ff00ff01ff01, 0x0001ff00ff010001,
    0x0001ff00ff010100, 0x0001ff0000ff0000, 0x0001ff0000ff0100, 0x0001ff000000ff00,
    0x0001ff0000000000, 0x0001ff0000000001, 0x0001ff0000000100, 0x0001ff0000010000,
    0x0001ff0000010001, 0x0001ff0000010101, 0x0001ff0001ff00ff, 0x0001ff0001ff0101,
    0x0001ff000100ff01, 0x0001ff0001000000, 0x0001ff000101ff00, 0x0001ff0001010001,
    0x0001ff0001010100, 0x0001ff01ff00ff00, 0x0001ff01ff000001, 0x0001ff01ff000100,
    0x0001ff0100ffffff, 0x0001ff0100ffff00, 0x0001ff0100ff0001, 0x0001ff0100000000,
    0x0001ff0100000001, 0x0001ff01000001ff, 0x0001ff010001ffff, 0x0001ff0101ff0000,
    0x0001ff010100ff00, 0x0001ff0101000001, 0x0001ff0101010000, 0x000100ffff00ff00,
    0x000100ffff00ff01, 0x000100ffff000000, 0x000100ffff000001, 0x000100ffff000101,
    0x000100ffff01ff00, 0x000100ffff010001, 0x000100ffff010100, 0x000100ff00ffffff,
    0x000100ff00ffff01, 0x000100ff00ff0000, 0x000100ff00ff01ff, 0x000100ff00ff0101,
    0x000100ff0000ff00, 0x000100ff000000ff, 0x000100ff00000000, 0x000100ff00000001,
    0x000100ff00000100, 0x000100ff00000101, 0x000100ff0001ffff, 0x000100ff0001ff01,
    0x000100ff00010000, 0x000100ff01ff00ff, 0x000100ff01ff0000, 0x000100ff01ff0100,
    0x000100ff0100ffff, 0x000100ff0100ff01, 0x000100ff010000ff, 0x000100ff01000000,
    0x000100ff01000001, 0x000100ff010001ff, 0x000100ff01000101, 0x000100ff0101ff00,
    0x000100ff010100ff, 0x000100ff01010100, 0x00010000ffff0000, 0x00010000ffff01ff,
    0x00010000ffff0101, 0x00010000ff00ff00, 0x00010000ff000000, 0x00010000ff000001,
    0x00010000ff000100, 0x0001000000ff00ff, 0x0001000000ff0000, 0x0001000000ff0001,
    0x0001000000ff0100, 0x000100000000ffff, 0x000100000000ff00, 0x00010000000000ff,
    0x0001000000000000, 0x0001000000000001, 0x0001000000000100, 0x000100000001ff00,
    0x00010000000100ff, 0x0001000000010000, 0x0001000000010001, 0x0001000000010100,
    0x0001000001ff0001, 0x0001000001ff0100, 0x0001000001ff0101, 0x000100000100ff00,
    0x0001000001000000, 0x0001000001000001, 0x0001000001000100, 0x0001000001000101,
    0x000100000101ff01, 0x0001000001010000, 0x0001000001010001, 0x00010000010101ff,
    0x00010001ffffff01, 0x00010001ffff0100, 0x00010001ff000000, 0x00010001ff01ffff,
    0x00010001ff010001, 0x00010001ff0101ff, 0x00010001ff010100, 0x0001000100ffffff,
    0x0001000100ff0000, 0x0001000100ff01ff, 0x0001000100ff0101, 0x000100010000ff00,
    0x00010001000000ff, 0x0001000100000000, 0x0001000100000001, 0x00010001000001ff,
    0x0001000100000101, 0x000100010001ffff, 0x0001000100010000, 0x00010001000101ff,
    0x0001000101ffffff, 0x0001000101ffff01, 0x0001000101ff0000, 0x0001000101ff0101,
    0x00010001010000ff, 0x0001000101000001, 0x00010001010001ff, 0x0001000101000100,
    0x000100010101ffff, 0x00010001010100ff, 0x0001000101010001, 0x0001000101010101,
    0x000101ffff000001, 0x000101ffff000100, 0x000101ffff010000, 0x000101ff00ffff00,
    0x000101ff0000ff01, 0x000101ff00000000, 0x000101ff00000101, 0x000101ff0001ff00,
    0x000101ff00010100, 0x000101ff01ff0000, 0x000101ff0100ff00, 0x000101ff010001ff,
    0x000101ff01010001, 0x00010100ffffff00, 0x00010100ffff00ff, 0x00010100ff00ffff,
    0x00010100ff000000, 0x00010100ff01ff00, 0x00010100ff0100ff, 0x00010100ff010001,
    0x00010100ff010100, 0x0001010000ffffff, 0x0001010000ffff00, 0x0001010000ff0000,
    0x0001010000ff0001, 0x0001010000ff01ff, 0x000101000000ff00, 0x00010100000000ff,
    0x0001010000000000, 0x0001010000000001, 0x0001010000000100, 0x000101000001ffff,
    0x0001010000010000, 0x0001010000010101, 0x0001010001ffff01, 0x0001010001ff00ff,
    0x0001010001ff0101, 0x0001010001000000, 0x000101000101ff00, 0x00010100010100ff,
    0x0001010001010000, 0x0001010001010100, 0x00010101ff00ff00, 0x00010101ff000001,
    0x00010101ff0001ff, 0x0001010100ffff00, 0x0001010100ff00ff, 0x0001010100ff0100,
    0x000101010000ffff, 0x0001010100000000, 0x00010101000001ff, 0x0001010100000101,
    0x00010101000100ff, 0x0001010100010000, 0x0001010100010100, 0x0001010101ff0001,
    0x00010101010000ff, 0x00010101010001ff, 0x0001010101000101, 0x0001010101010001,
    0x01ffffffffffffff, 0x01ffffffffffff01, 0x01ffffffffff01ff, 0x01ffffffffff0101,
    0x01ffffffff01ffff, 0x01ffffffff01ff01, 0x01ffffffff0101ff, 0x01ffffffff010101,
    0x01ffffff00ff0000, 0x01ffffff0000ffff, 0x01ffffff0000ff00, 0x01ffffff000000ff,
    0x01ffffff00000001, 0x01ffffff00000100, 0x01ffffff00010000, 0x01ffffff01ffffff,
    0x01ffffff01ffff01, 0x01ffffff01ff01ff, 0x01ffffff01ff0101, 0x01ffffff01000000,
    0x01ffffff0101ffff, 0x01ffffff0101ff01, 0x01ffffff010101ff, 0x01ffffff01010101,
    0x01ffff00ffff0000, 0x01ffff00ff00ff00, 0x01ffff00ff0000ff, 0x01ffff00ff000001,
    0x01ffff00ff000100, 0x01ffff00ff010000, 0x01ffff0000ffff00, 0x01ffff0000ff00ff,
    0x01ffff0000ff0100, 0x01ffff000000ffff, 0x01ffff000000ff01, 0x01ffff0000000000,
    0x01ffff0000000001, 0x01ffff00000001ff, 0x01ffff0000000100, 0x01ffff00000100ff,
    0x01ffff0000010001, 0x01ffff0000010100, 0x01ffff0001ff0000, 0x01ffff0001ff0100,
    0x01ffff00010000ff, 0x01ffff0001000001, 0x01ffff0001000100, 0x01ffff0001010000,
    0x01ffff01ffffffff, 0x01ffff01ffffff01, 0x01ffff01ffff01ff, 0x01ffff01ffff0101,
    0x01ffff01ff000000, 0x01ffff01ff01ffff, 0x01ffff01ff01ff01, 0x01ffff01ff0101ff,
    0x01ffff01ff010101, 0x01ffff010000ff00, 0x01ffff01000000ff, 0x01ffff0100000100,
    0x01ffff0100010000, 0x01ffff0101ffffff, 0x01ffff0101ffff01, 0x01ffff0101ff01ff,
    0x01ffff0101ff0101, 0x01ffff0101000000, 0x01ffff010101ffff, 0x01ffff010101ff01,
    0x01ffff01010101ff, 0x01ffff0101010101, 0x01ff00ffff0000ff, 0x01ff00ffff000100,
    0x01ff00ff00ffff00, 0x01ff00ff00ff00ff, 0x01ff00ff0000ff00, 0x01ff00ff00000000,
    0x01ff00ff00000101, 0x01ff00ff0001ff00, 0x01ff00ff000100ff, 0x01ff00ff00010100,
    0x01ff00ff010000ff, 0x01ff00ff01000100, 0x01ff0000ffffff00, 0x01ff0000ffff0100,
    0x01ff0000ff00ff01, 0x01ff0000ff000000, 0x01ff0000ff000101, 0x01ff0000ff010001,
    0x01ff0000ff010100, 0x01ff000000ffffff, 0x01ff000000ffff00, 0x01ff000000ff0000,
    0x01ff000000ff01ff, 0x01ff00000000ff00, 0x01ff0000000000ff, 0x01ff000000000000,
    0x01ff000000000001, 0x01ff000000000100, 0x01ff000000000101, 0x01ff000000010000,
    0x01ff000000010001, 0x01ff0000000101ff, 0x01ff000000010101, 0x01ff000001ffff00,
    0x01ff000001ff00ff, 0x01ff000001ff0001, 0x01ff000001ff0100, 0x01ff00000100ffff,
    0x01ff00000100ff01, 0x01ff000001000000, 0x01ff0000010001ff, 0x01ff000001010001,
    0x01ff0001ff00ff00, 0x01ff0001ff000001, 0x01ff0001ff000100, 0x01ff0001ff010000,
    0x01ff000100ffff00, 0x01ff000100ff00ff, 0x01ff000100ff0100, 0x01ff000100ff0101,
    0x01ff00010000ffff, 0x01ff000100000000, 0x01ff000100000100, 0x01ff000100000101,
    0x01ff00010001ff00, 0x01ff000100010001, 0x01ff000100010101, 0x01ff000101ff0000,
    0x01ff00010100ff00, 0x01ff000101000101, 0x01ff0001010100ff, 0x01ff01ffffffffff,
    0x01ff01ffffffff01, 0x01ff01ffffff01ff, 0x01ff01ffffff0101, 0x01ff01ffff000000,
    0x01ff01ffff01ffff, 0x01ff01ffff01ff01, 0x01ff01ffff0101ff, 0x01ff01ffff010101,
    0x01ff01ff00ffff00, 0x01ff01ff00ff0000, 0x01ff01ff0000ff00, 0x01ff01ff000000ff,
    0x01ff01ff00000100, 0x01ff01ff00010000, 0x01ff01ff00010100, 0x01ff01ff01ffffff,
    0x01ff01ff01ffff01, 0x01ff01ff01ff01ff, 0x01ff01ff01ff0101, 0x01ff01ff01000000,
    0x01ff01ff0101ffff, 0x01ff01ff0101ff01, 0x01ff01ff010101ff, 0x01ff01ff01010101,
    0x01ff0100ffff0000, 0x01ff0100ffff0001, 0x01ff0100ff00ff00, 0x01ff0100ff0000ff,
    0x01ff0100ff000001, 0x01ff0100ff010000, 0x01ff010000ffff00, 0x01ff010000ff00ff,
    0x01ff010000ff0001, 0x01ff010000ff0100, 0x01ff01000000ffff, 0x01ff01000000ff01,
    0x01ff010000000000, 0x01ff010000000101, 0x01ff01000001ff00, 0x01ff0100000100ff,
    0x01ff010001ff0000, 0x01ff010001000001, 0x01ff010001000100, 0x01ff010001010000,
    0x01ff0101ffffffff, 0x01ff0101ffffff01, 0x01ff0101ffff01ff, 0x01ff0101ffff0101,
    0x01ff0101ff000000, 0x01ff0101ff01ffff, 0x01ff0101ff01ff01, 0x01ff0101ff0101ff,
    0x01ff0101ff010101, 0x01ff010100ff0000, 0x01ff01010000ff00, 0x01ff0101000000ff,
    0x01ff010100000001, 0x01ff010101ffffff, 0x01ff010101ffff01, 0x01ff010101ff01ff,
    0x01ff010101ff0101, 0x01ff010101000000, 0x01ff01010101ffff, 0x01ff01010101ff01,
    0x01ff0101010101ff, 0x01ff010101010101, 0x0100ffffffff0000, 0x0100ffffff00ff00,
    0x0100ffffff000001, 0x0100ffffff0001ff, 0x0100ffffff000100, 0x0100ffffff010000,
    0x0100ffff00ffff00, 0x0100ffff00ff0001, 0x0100ffff00ff0100, 0x0100ffff00000000,
    0x0100ffff000001ff, 0x0100ffff00000101, 0x0100ffff00010100, 0x0100ffff00010101,
    0x0100ffff01ff0000, 0x0100ffff0100ff00, 0x0100ffff010000ff, 0x0100ffff01000001,
    0x0100ffff01000100, 0x0100ffff01010000, 0x0100ff00ffffff00, 0x0100ff00ffff00ff,
    0x0100ff00ffff0001, 0x0100ff00ffff0100, 0x0100ff00ff00ffff, 0x0100ff00ff000000,
    0x0100ff00ff0001ff, 0x0100ff00ff000101, 0x0100ff00ff01ff00, 0x0100ff00ff0100ff,
    0x0100ff00ff010001, 0x0100ff00ff010100, 0x0100ff0000ffffff, 0x0100ff0000ff0000,
    0x0100ff000000ffff, 0x0100ff000000ff00, 0x0100ff00000000ff, 0x0100ff0000000000,
    0x0100ff0000000001, 0x0100ff0000000100, 0x0100ff000001ff01, 0x0100ff0000010000,
    0x0100ff0001ff00ff, 0x0100ff0001ff0001, 0x0100ff000100ff01, 0x0100ff0001000000,
    0x0100ff00010001ff, 0x0100ff000101ff00, 0x0100ff00010100ff, 0x0100ff0001010001,
    0x0100ff0001010100, 0x0100ff01ffff0000, 0x0100ff01ff00ff00, 0x0100ff01ff0000ff,
    0x0100ff01ff000100, 0x0100ff01ff010000, 0x0100ff0100ff00ff, 0x0100ff0100ff0001,
    0x0100ff0100ff0100, 0x0100ff010000ffff, 0x0100ff010000ff01, 0x0100ff0100000000,
    0x0100ff01000001ff, 0x0100ff0100010001, 0x0100ff0100010100, 0x0100ff0101ff0000,
    0x0100ff01010000ff, 0x0100ff0101000001, 0x0100ff0101010100, 0x010000ffffffff00,
    0x010000ffffff00ff, 0x010000ffffff0001, 0x010000ffff00ffff, 0x010000ffff000000,
    0x010000ffff0001ff, 0x010000ffff010001, 0x010000ff00ffffff, 0x010000ff00ff0101,
    0x010000ff0000ff00, 0x010000ff000000ff, 0x010000ff00000000, 0x010000ff00000001,
    0x010000ff000001ff, 0x010000ff00000100, 0x010000ff0001ffff, 0x010000ff0001ff00,
    0x010000ff0001ff01, 0x010000ff00010000, 0x010000ff01ff00ff, 0x010000ff01ff0001,
    0x010000ff0100ff01, 0x010000ff010000ff, 0x010000ff01000000, 0x010000ff010001ff,
    0x010000ff0101ff00, 0x010000ff01010100, 0x01000000ffffffff, 0x01000000ffff0000,
    0x01000000ffff01ff, 0x01000000ffff0101, 0x01000000ff00ffff, 0x01000000ff00ff00,
    0x01000000ff0000ff, 0x01000000ff000000, 0x01000000ff000001, 0x01000000ff000100,
    0x01000000ff01ff00, 0x01000000ff010000, 0x01000000ff010100, 0x01000000ff010101,
    0x0100000000ffff00, 0x0100000000ff00ff, 0x0100000000ff0000, 0x0100000000ff0001,
    0x0100000000ff0100, 0x010000000000ffff, 0x010000000000ff00, 0x010000000000ff01,
    0x01000000000000ff, 0x0100000000000000, 0x0100000000000001, 0x01000000000001ff,
    0x0100000000000100, 0x0100000000000101, 0x010000000001ff00, 0x01000000000100ff,
    0x0100000000010000, 0x0100000000010001, 0x0100000000010100, 0x0100000001ffff00,
    0x0100000001ff0000, 0x0100000001ff01ff, 0x010000000100ff00, 0x010000000100ff01,
    0x01000000010000ff, 0x0100000001000000, 0x0100000001000001, 0x0100000001000100,
    0x0100000001000101, 0x010000000101ffff, 0x010000000101ff01, 0x0100000001010000,
    0x01000000010101ff, 0x0100000001010101, 0x01000001ffffff00, 0x01000001ffff00ff,
    0x01000001ff00ffff, 0x01000001ff000000, 0x01000001ff000100, 0x01000001ff01ffff,
    0x01000001ff010001, 0x01000001ff010100, 0x0100000100ff0000, 0x0100000100ff01ff,
    0x0100000100ff0100, 0x010000010000ff00, 0x010000010000ff01, 0x0100000100000000,
    0x0100000100000001, 0x0100000100000100, 0x0100000100010000, 0x01000001000101ff,
    0x0100000101ffff01, 0x0100000101ff00ff, 0x0100000101ff0100, 0x0100000101ff0101,
    0x010000010100ff01, 0x01000001010000ff, 0x0100000101000000, 0x01000001010100ff,
    0x0100000101010001, 0x0100000101010100, 0x010001ffffff0000, 0x010001ffff000001,
    0x010001ffff000100, 0x010001ffff010000, 0x010001ff00ffff00, 0x010001ff00ff0001,
    0x010001ff0000ffff, 0x010001ff0000ff01, 0x010001ff00000000, 0x010001ff00000001,
    0x010001ff00000101, 0x010001ff000100ff, 0x010001ff00010000, 0x010001ff01ff0000,
    0x010001ff0100ff00, 0x010001ff01000001, 0x010001ff01000100, 0x010001ff01010000,
    0x01000100ffff00ff, 0x01000100ffff0001, 0x01000100ffff0100, 0x01000100ff00ffff,
    0x01000100ff00ff01, 0x01000100ff000000, 0x01000100ff0001ff, 0x01000100ff000101,
    0x01000100ff01ffff, 0x01000100ff01ff00, 0x01000100ff0100ff, 0x01000100ff010001,
    0x0100010000ffffff, 0x0100010000ffff01, 0x0100010000ff0000, 0x0100010000ff01ff,
    0x0100010000ff0101, 0x010001000000ff00, 0x01000100000000ff, 0x0100010000000000,
    0x0100010000000001, 0x0100010000000100, 0x010001000001ff01, 0x0100010000010000,
    0x0100010000010001, 0x0100010000010101, 0x0100010001ffff00, 0x0100010001ff00ff,
    0x010001000100ffff, 0x010001000100ff01, 0x0100010001000000, 0x0100010001000101,
    0x010001000101ff00, 0x0100010001010001, 0x01000101ffff0000, 0x01000101ff000000,
    0x01000101ff010000, 0x0100010100ff00ff, 0x0100010100ff0001, 0x0100010100ff0100,
    0x010001010000ffff, 0x0100010100000000, 0x01000101000001ff, 0x010001010001ff00,
    0x0100010101ff0000, 0x010001010100ff00, 0x01000101010000ff, 0x0100010101000000,
    0x0100010101000001, 0x0101ffffffffffff, 0x0101ffffffffff01, 0x0101ffffffff01ff,
    0x0101ffffffff0101, 0x0101ffffff000000, 0x0101ffffff01ffff, 0x0101ffffff01ff01,
    0x0101ffffff0101ff, 0x0101ffffff010101, 0x0101ffff00ff0000, 0x0101ffff0000ff00,
    0x0101ffff000000ff, 0x0101ffff00000001, 0x0101ffff00000100, 0x0101ffff01ffffff,
    0x0101ffff01ffff01, 0x0101ffff01ff01ff, 0x0101ffff01ff0101, 0x0101ffff01000000,
    0x0101ffff0101ffff, 0x0101ffff0101ff01, 0x0101ffff010101ff, 0x0101ffff01010101,
    0x0101ff00ffff0000, 0x0101ff00ffff0100, 0x0101ff00ff00ff00, 0x0101ff00ff0000ff,
    0x0101ff00ff000001, 0x0101ff00ff000100, 0x0101ff00ff000101, 0x0101ff0000ff0001,
    0x0101ff0000ff0100, 0x0101ff000000ff00, 0x0101ff0000000000, 0x0101ff00000001ff,
    0x0101ff0000000101, 0x0101ff000001ff00, 0x0101ff00000100ff, 0x0101ff0001ff0000,
    0x0101ff000100ffff, 0x0101ff000100ff01, 0x0101ff0001000001, 0x0101ff0001000100,
    0x0101ff01ffffff01, 0x0101ff01ffff01ff, 0x0101ff01ffff0101, 0x0101ff01ff00ffff,
    0x0101ff01ff000100, 0x0101ff01ff01ff01, 0x0101ff01ff0101ff, 0x0101ff01ff010101,
    0x0101ff0100ff0000, 0x0101ff010000ff00, 0x0101ff0100000001, 0x0101ff0100000100,
    0x0101ff0100010000, 0x0101ff0101ffffff, 0x0101ff0101ffff01, 0x0101ff0101ff01ff,
    0x0101ff0101ff0101, 0x0101ff0101000000, 0x0101ff010101ffff, 0x0101ff010101ff01,
    0x0101ff01010101ff, 0x0101ff0101010101, 0x010100ffff000100, 0x010100ffff010000,
    0x010100ff00ffff00, 0x010100ff00ff00ff, 0x010100ff0000ffff, 0x010100ff000000ff,
    0x010100ff00000000, 0x010100ff000001ff, 0x010100ff00000101, 0x010100ff0001ff00,
    0x010100ff00010000, 0x010100ff00010001, 0x010100ff000101ff, 0x010100ff00010100,
    0x010100ff01ff0000, 0x01010000ffff0001, 0x01010000ffff0100, 0x01010000ff00ffff,
    0x01010000ff00ff01, 0x01010000ff000000, 0x01010000ff0001ff, 0x01010000ff010001,
    0x01010000ff010100, 0x0101000000ffff01, 0x0101000000ff0000, 0x010100000000ff00,
    0x01010000000000ff, 0x0101000000000000, 0x0101000000000001, 0x0101000000000100,
    0x0101000000010000, 0x0101000000010101, 0x0101000001ffff00, 0x0101000001ff00ff,
    0x0101000001ff0000, 0x0101000001ff0001, 0x0101000001ff0100, 0x010100000100ff01,
    0x0101000001000000, 0x01010000010001ff, 0x01010001ffff0000, 0x01010001ff00ff00,
    0x01010001ff000001, 0x01010001ff000101, 0x01010001ff01ff00, 0x01010001ff010000,
    0x0101000100ff00ff, 0x0101000100ff0001, 0x0101000100ff0101, 0x010100010000ff01,
    0x0101000100000000, 0x0101000100000001, 0x01010001000001ff, 0x010100010001ffff,
    0x010100010001ff01, 0x0101000101ff0001, 0x010100010100ffff, 0x0101000101000000,
    0x0101000101000001, 0x0101000101000100, 0x010100010101ff00, 0x01010001010100ff,
    0x0101000101010001, 0x010101ffffffffff, 0x010101ffffffff01, 0x010101ffffff01ff,
    0x010101ffffff0101, 0x010101ffff01ffff, 0x010101ffff01ff01, 0x010101ffff0101ff,
    0x010101ffff010101, 0x010101ff0000ff00, 0x010101ff000000ff, 0x010101ff00000001,
    0x010101ff00000100, 0x010101ff01ffffff, 0x010101ff01ffff01, 0x010101ff01ff01ff,
    0x010101ff01ff0101, 0x010101ff01000000, 0x010101ff0101ffff, 0x010101ff0101ff01,
    0x010101ff010101ff, 0x010101ff01010101, 0x01010100ffff0000, 0x01010100ff0000ff,
    0x01010100ff000100, 0x01010100ff01ff00, 0x01010100ff010000, 0x0101010000ffff00,
    0x010101000000ffff, 0x0101010000000000, 0x0101010000000101, 0x010101000001ff00,
    0x0101010000010001, 0x0101010000010100, 0x010101000100ffff, 0x0101010001000001,
    0x01010101ffffffff, 0x01010101ffffff01, 0x01010101ffff01ff, 0x01010101ffff0101,
    0x01010101ff01ffff, 0x01010101ff01ff01, 0x01010101ff0101ff, 0x01010101ff010101,
    0x010101010000ff00, 0x01010101000000ff, 0x0101010100000001, 0x0101010101ffffff,
    0x0101010101ffff01, 0x0101010101ff01ff, 0x0101010101ff0101, 0x0101010101000000,
    0x010101010101ffff, 0x010101010101ff01, 0x01010101010101ff, 0x0101010101010101,
};

static const int8_t kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

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

#if 0
static void vec_dot_q4_0_f32(int n, float *GGML_RESTRICT s, size_t bs, const block_q4_0 *GGML_RESTRICT x,
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
#endif

static void vec_dot_q4_0_f32(int n, float *GGML_RESTRICT s, size_t bs, const block_q4_0 *GGML_RESTRICT x,
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

#if 0
static void vec_dot_q8_0_f32(int n, float *GGML_RESTRICT s, size_t bs, const block_q8_0 *GGML_RESTRICT x,
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
#endif

static void vec_dot_q8_0_f32(int n, float *GGML_RESTRICT s, size_t bs, const block_q8_0 *GGML_RESTRICT x,
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

#if 0
static void vec_dot_q4_0_q8_0(int n, float *GGML_RESTRICT s, size_t bs, const block_q4_0 *GGML_RESTRICT x,
                    size_t bx, const block_q8_0 *GGML_RESTRICT y, size_t by, int nrc) {
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
#endif

static void vec_dot_q4_0_q8_0(int n, float *GGML_RESTRICT s, size_t bs, const block_q4_0 *GGML_RESTRICT x,
                    size_t bx, const block_q8_0 *GGML_RESTRICT y, size_t by, int nrc) {
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

#if 0
static void vec_dot_q8_0_q8_0(int n, float *GGML_RESTRICT s, size_t bs, const block_q8_0 *GGML_RESTRICT x,
                    size_t bx, const block_q8_0 *GGML_RESTRICT y, size_t by, int nrc) {
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
#endif

static void vec_dot_q8_0_q8_0(int n, float *GGML_RESTRICT s, size_t bs, const block_q8_0 *GGML_RESTRICT x,
                    size_t bx, const block_q8_0 *GGML_RESTRICT y, size_t by, int nrc) {
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

static void vec_dot_q4_1_q8_1(int n, float *GGML_RESTRICT s, size_t bs, const block_q4_1 *GGML_RESTRICT x,
                    size_t bx, const block_q8_1 *GGML_RESTRICT y, size_t by, int nrc) {
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

static void vec_dot_q5_0_q8_0(int n, float *GGML_RESTRICT s, size_t bs, const block_q5_0 *GGML_RESTRICT x,
                    size_t bx, const block_q8_0 *GGML_RESTRICT y, size_t by, int nrc) {
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

static void vec_dot_q5_1_q8_1(int n, float *GGML_RESTRICT s, size_t bs, const block_q5_1 *GGML_RESTRICT x,
                    size_t bx, const block_q8_1 *GGML_RESTRICT y, size_t by, int nrc) {
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

static void vec_dot_iq4_nl_q8_0(int n, float *GGML_RESTRICT s, size_t bs, const block_iq4_nl *GGML_RESTRICT x,
                    size_t bx, const block_q8_0 *GGML_RESTRICT y, size_t by, int nrc) {
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
static void vec_dot_bf16_bf16(int n, float *GGML_RESTRICT s, size_t bs, const ggml_bf16_t *GGML_RESTRICT x,
                               size_t bx, const ggml_bf16_t *GGML_RESTRICT y, size_t by, int nrc) {
    UNUSED(bs); UNUSED(bx); UNUSED(by); UNUSED(nrc);
    float sumf = 0;
    for (int i = 0; i < n; ++i) {
        sumf += ggml_compute_bf16_to_fp32(x[i]) * ggml_compute_bf16_to_fp32(y[i]);
    }
    *s = sumf;
}

// Q6_K x Q8_K dot product
static void vec_dot_q6_K_q8_K(int n, float *GGML_RESTRICT s, size_t bs, const block_q6_K *GGML_RESTRICT x,
                                size_t bx, const block_q8_K *GGML_RESTRICT y, size_t by, int nrc) {
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
static void vec_dot_q4_K_q8_K(int n, float *GGML_RESTRICT s, size_t bs, const block_q4_K *GGML_RESTRICT x,
                                size_t bx, const block_q8_K *GGML_RESTRICT y, size_t by, int nrc) {
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
static void vec_dot_q2_K_q8_K(int n, float *GGML_RESTRICT s, size_t bs, const block_q2_K *GGML_RESTRICT x,
                                size_t bx, const block_q8_K *GGML_RESTRICT y, size_t by, int nrc) {
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
static void vec_dot_q3_K_q8_K(int n, float *GGML_RESTRICT s, size_t bs, const block_q3_K *GGML_RESTRICT x,
                                size_t bx, const block_q8_K *GGML_RESTRICT y, size_t by, int nrc) {
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
static void vec_dot_q5_K_q8_K(int n, float *GGML_RESTRICT s, size_t bs, const block_q5_K *GGML_RESTRICT x,
                                size_t bx, const block_q8_K *GGML_RESTRICT y, size_t by, int nrc) {
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
static void vec_dot_mxfp4_q8_0(int n, float *GGML_RESTRICT s, size_t bs, const block_mxfp4 *GGML_RESTRICT x,
                                 size_t bx, const block_q8_0 *GGML_RESTRICT y, size_t by, int nrc) {
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
static void vec_dot_nvfp4_q8_0(int n, float *GGML_RESTRICT s, size_t bs, const block_nvfp4 *GGML_RESTRICT x,
                                 size_t bx, const block_q8_0 *GGML_RESTRICT y, size_t by, int nrc) {
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
static void vec_dot_iq4_xs_q8_K(int n, float *GGML_RESTRICT s, size_t bs, const block_iq4_xs *GGML_RESTRICT x,
                                  size_t bx, const block_q8_K *GGML_RESTRICT y, size_t by, int nrc) {
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
static void vec_dot_iq3_xxs_q8_K(int n, float *GGML_RESTRICT s, size_t bs, const block_iq3_xxs *GGML_RESTRICT x,
                                   size_t bx, const block_q8_K *GGML_RESTRICT y, size_t by, int nrc) {
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
static void vec_dot_iq2_xxs_q8_K(int n, float *GGML_RESTRICT s, size_t bs, const block_iq2_xxs *GGML_RESTRICT x,
                                   size_t bx, const block_q8_K *GGML_RESTRICT y, size_t by, int nrc) {
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
static void vec_dot_iq2_xs_q8_K(int n, float *GGML_RESTRICT s, size_t bs, const block_iq2_xs *GGML_RESTRICT x,
                                  size_t bx, const block_q8_K *GGML_RESTRICT y, size_t by, int nrc) {
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
static void vec_dot_iq2_s_q8_K(int n, float *GGML_RESTRICT s, size_t bs, const block_iq2_s *GGML_RESTRICT x,
                                 size_t bx, const block_q8_K *GGML_RESTRICT y, size_t by, int nrc) {
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
static void vec_dot_iq1_s_q8_K(int n, float *GGML_RESTRICT s, size_t bs, const block_iq1_s *GGML_RESTRICT x,
                                 size_t bx, const block_q8_K *GGML_RESTRICT y, size_t by, int nrc) {
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

#if 0
static void quantize_row_q8_0(const float * x, block_q8_0 * y, int n) {
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
#endif

static void quantize_row_q8_0(const float * x, block_q8_0 * y, int n) {
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

static void quantize_row_q8_1(const float * x, block_q8_1 * y, int n) {
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

static void quantize_row_q5_0(const float * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int n) {
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

static void quantize_row_q5_1(const float * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int n) {
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

static void quantize_row_iq4_nl(const float * GGML_RESTRICT x, block_iq4_nl * GGML_RESTRICT y, int n) {
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

static void quantize_f32_to_f16_row_hvx(const float * GGML_RESTRICT x, uint16_t * GGML_RESTRICT y, int n) {
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

static int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    return (int)(fval + (fval >= 0 ? 0.5f : -0.5f));
}

// Quantize F32 to Q8_K (256-element super-block)
static void quantize_row_q8_K(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int n) {
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

// Quantize F32 to BF16
static void quantize_row_bf16(const float * GGML_RESTRICT x, ggml_bf16_t * GGML_RESTRICT y, int n) {
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_compute_fp32_to_bf16(x[i]);
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
            if (vec_dot_type == GGML_TYPE_F16) {
                for (int i13 = 0; i13 < ne13; ++i13) {
                    for (int i12 = 0; i12 < ne12; ++i12) {
                        for (int i11 = 0; i11 < ne11; ++i11) {
                            const float * src_row = (const float*)((const char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);
                            uint16_t * dst_row = (uint16_t*)((char*)q8_data + i13 * nbw3 + i12 * nbw2 + i11 * nbw1);
                            quantize_f32_to_f16_row_hvx(src_row, dst_row, ne10);
                        }
                    }
                }
            } else if (vec_dot_type == GGML_TYPE_Q8_1) {
                for (int i13 = 0; i13 < ne13; ++i13) {
                    for (int i12 = 0; i12 < ne12; ++i12) {
                        for (int i11 = 0; i11 < ne11; ++i11) {
                            const float * src_row = (const float*)((const char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);
                            block_q8_1 * dst_row = (block_q8_1*)((char*)q8_data + i13 * nbw3 + i12 * nbw2 + i11 * nbw1);
                            quantize_row_q8_1(src_row, dst_row, ne10);
                        }
                    }
                }
            } else {
                for (int i13 = 0; i13 < ne13; ++i13) {
                    for (int i12 = 0; i12 < ne12; ++i12) {
                        for (int i11 = 0; i11 < ne11; ++i11) {
                            const float * src_row = (const float*)((const char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);
                            block_q8_0 * dst_row = (block_q8_0*)((char*)q8_data + i13 * nbw3 + i12 * nbw2 + i11 * nbw1);
                            quantize_row_q8_0(src_row, dst_row, ne10);
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

                    if (type == GGML_TYPE_F16) {
                        if (vec_dot_type == GGML_TYPE_F16) {
                            vec_dot_f16_f16(ne00, &tmp[row_idx], 0,
                                           (const uint16_t*)(src0_row + ir0 * nb01), 0,
                                           (uint16_t*)src1_col, 0, 1);
                        } else {
                            vec_dot_f16_f32(ne00, &tmp[row_idx], 0,
                                           (const uint16_t*)(src0_row + ir0 * nb01), 0,
                                           (float*)src1_col, 0, 1);
                        }
                    } else if (type == GGML_TYPE_Q4_0) {
                        const block_q4_0 * q4_row = (const block_q4_0*)(src0_row + ir0 * nb01);
                        const block_q8_0 * q8_col = (const block_q8_0*)src1_col;
                        vec_dot_q4_0_q8_0(ne00, &tmp[row_idx], 0, q4_row, 0, q8_col, 0, 1);
                    } else if (type == GGML_TYPE_Q4_1) {
                        const block_q4_1 * q4_row = (const block_q4_1*)(src0_row + ir0 * nb01);
                        const block_q8_1 * q8_col = (const block_q8_1*)src1_col;
                        vec_dot_q4_1_q8_1(ne00, &tmp[row_idx], 0, q4_row, 0, q8_col, 0, 1);
                    } else if (type == GGML_TYPE_Q5_0) {
                        const block_q5_0 * q5_row = (const block_q5_0*)(src0_row + ir0 * nb01);
                        const block_q8_0 * q8_col = (const block_q8_0*)src1_col;
                        vec_dot_q5_0_q8_0(ne00, &tmp[row_idx], 0, q5_row, 0, q8_col, 0, 1);
                    } else if (type == GGML_TYPE_Q5_1) {
                        const block_q5_1 * q5_row = (const block_q5_1*)(src0_row + ir0 * nb01);
                        const block_q8_1 * q8_col = (const block_q8_1*)src1_col;
                        vec_dot_q5_1_q8_1(ne00, &tmp[row_idx], 0, q5_row, 0, q8_col, 0, 1);
                    } else if (type == GGML_TYPE_IQ4_NL) {
                        const block_iq4_nl * iq4_row = (const block_iq4_nl*)(src0_row + ir0 * nb01);
                        const block_q8_0 * q8_col = (const block_q8_0*)src1_col;
                        vec_dot_iq4_nl_q8_0(ne00, &tmp[row_idx], 0, iq4_row, 0, q8_col, 0, 1);
                    } else if (type == GGML_TYPE_Q8_0) {
                        const block_q8_0 * q8_row = (const block_q8_0*)(src0_row + ir0 * nb01);
                        const block_q8_0 * q8_col = (const block_q8_0*)src1_col;
                        vec_dot_q8_0_q8_0(ne00, &tmp[row_idx], 0, q8_row, 0, q8_col, 0, 1);
                    } else if (type == GGML_TYPE_BF16) {
                        const ggml_bf16_t * bf16_row = (const ggml_bf16_t*)(src0_row + ir0 * nb01);
                        const ggml_bf16_t * bf16_col = (const ggml_bf16_t*)src1_col;
                        vec_dot_bf16_bf16(ne00, &tmp[row_idx], 0, bf16_row, 0, bf16_col, 0, 1);
                    } else if (type == GGML_TYPE_Q4_K) {
                        const block_q4_K * q4_row = (const block_q4_K*)(src0_row + ir0 * nb01);
                        const block_q8_K * q8_col = (const block_q8_K*)src1_col;
                        vec_dot_q4_K_q8_K(ne00, &tmp[row_idx], 0, q4_row, 0, q8_col, 0, 1);
                    } else if (type == GGML_TYPE_Q6_K) {
                        const block_q6_K * q6_row = (const block_q6_K*)(src0_row + ir0 * nb01);
                        const block_q8_K * q8_col = (const block_q8_K*)src1_col;
                        vec_dot_q6_K_q8_K(ne00, &tmp[row_idx], 0, q6_row, 0, q8_col, 0, 1);
                    } else if (type == GGML_TYPE_Q2_K) {
                        const block_q2_K * q2_row = (const block_q2_K*)(src0_row + ir0 * nb01);
                        const block_q8_K * q8_col = (const block_q8_K*)src1_col;
                        vec_dot_q2_K_q8_K(ne00, &tmp[row_idx], 0, q2_row, 0, q8_col, 0, 1);
                    } else if (type == GGML_TYPE_Q3_K) {
                        const block_q3_K * q3_row = (const block_q3_K*)(src0_row + ir0 * nb01);
                        const block_q8_K * q8_col = (const block_q8_K*)src1_col;
                        vec_dot_q3_K_q8_K(ne00, &tmp[row_idx], 0, q3_row, 0, q8_col, 0, 1);
                    } else if (type == GGML_TYPE_Q5_K) {
                        const block_q5_K * q5_row = (const block_q5_K*)(src0_row + ir0 * nb01);
                        const block_q8_K * q8_col = (const block_q8_K*)src1_col;
                        vec_dot_q5_K_q8_K(ne00, &tmp[row_idx], 0, q5_row, 0, q8_col, 0, 1);
                    } else if (type == GGML_TYPE_MXFP4) {
                        const block_mxfp4 * mxfp4_row = (const block_mxfp4*)(src0_row + ir0 * nb01);
                        const block_q8_0 * q8_col = (const block_q8_0*)src1_col;
                        vec_dot_mxfp4_q8_0(ne00, &tmp[row_idx], 0, mxfp4_row, 0, q8_col, 0, 1);
                    } else if (type == GGML_TYPE_NVFP4) {
                        const block_nvfp4 * nvfp4_row = (const block_nvfp4*)(src0_row + ir0 * nb01);
                        const block_q8_0 * q8_col = (const block_q8_0*)src1_col;
                        vec_dot_nvfp4_q8_0(ne00, &tmp[row_idx], 0, nvfp4_row, 0, q8_col, 0, 1);
                    } else if (type == GGML_TYPE_IQ4_XS) {
                        const block_iq4_xs * iq4xs_row = (const block_iq4_xs*)(src0_row + ir0 * nb01);
                        const block_q8_K * q8_col = (const block_q8_K*)src1_col;
                        vec_dot_iq4_xs_q8_K(ne00, &tmp[row_idx], 0, iq4xs_row, 0, q8_col, 0, 1);
                    } else if (type == GGML_TYPE_IQ3_XXS) {
                        const block_iq3_xxs * iq3xxs_row = (const block_iq3_xxs*)(src0_row + ir0 * nb01);
                        const block_q8_K * q8_col = (const block_q8_K*)src1_col;
                        vec_dot_iq3_xxs_q8_K(ne00, &tmp[row_idx], 0, iq3xxs_row, 0, q8_col, 0, 1);
                    } else if (type == GGML_TYPE_IQ2_XXS) {
                        const block_iq2_xxs * iq2xxs_row = (const block_iq2_xxs*)(src0_row + ir0 * nb01);
                        const block_q8_K * q8_col = (const block_q8_K*)src1_col;
                        vec_dot_iq2_xxs_q8_K(ne00, &tmp[row_idx], 0, iq2xxs_row, 0, q8_col, 0, 1);
                    } else if (type == GGML_TYPE_IQ2_XS) {
                        const block_iq2_xs * iq2xs_row = (const block_iq2_xs*)(src0_row + ir0 * nb01);
                        const block_q8_K * q8_col = (const block_q8_K*)src1_col;
                        vec_dot_iq2_xs_q8_K(ne00, &tmp[row_idx], 0, iq2xs_row, 0, q8_col, 0, 1);
                    } else if (type == GGML_TYPE_IQ2_S) {
                        const block_iq2_s * iq2s_row = (const block_iq2_s*)(src0_row + ir0 * nb01);
                        const block_q8_K * q8_col = (const block_q8_K*)src1_col;
                        vec_dot_iq2_s_q8_K(ne00, &tmp[row_idx], 0, iq2s_row, 0, q8_col, 0, 1);
                    } else if (type == GGML_TYPE_IQ1_S) {
                        const block_iq1_s * iq1s_row = (const block_iq1_s*)(src0_row + ir0 * nb01);
                        const block_q8_K * q8_col = (const block_q8_K*)src1_col;
                        vec_dot_iq1_s_q8_K(ne00, &tmp[row_idx], 0, iq1s_row, 0, q8_col, 0, 1);
                    } else {
                        vec_dot_f32(ne00, &tmp[row_idx], 0,
                                    (const float*)(src0_row + ir0 * nb01), 0,
                                    (float*)src1_col, 0, 1);
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

    const enum ggml_type vec_dot_type = ggml_vec_dot_type(src0->type);

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

    const enum ggml_type vec_dot_type = ggml_vec_dot_type(src0->type);

    const void * wdata = src1->data;
    if (src1->type != vec_dot_type) {
        const size_t nbw1 = ggml_row_size(vec_dot_type, src1->ne[0]);
        const size_t nbw2 = nbw1 * src1->ne[1];
        const size_t nbw3 = nbw2 * src1->ne[2];
        const size_t q8_size = nbw3 * src1->ne[3];
        void * q8_data = ggmlop_get_work_data(q8_size);
        if (q8_data != NULL) {
            if (vec_dot_type == GGML_TYPE_F16) {
                for (int i13 = 0; i13 < src1->ne[3]; ++i13) {
                    for (int i12 = 0; i12 < src1->ne[2]; ++i12) {
                        for (int i11 = 0; i11 < src1->ne[1]; ++i11) {
                            const float * src_row = (const float*)((const char*)src1->data + i13 * src1->nb[3] + i12 * src1->nb[2] + i11 * src1->nb[1]);
                            uint16_t * dst_row = (uint16_t*)((char*)q8_data + i13 * nbw3 + i12 * nbw2 + i11 * nbw1);
                            quantize_f32_to_f16_row_hvx(src_row, dst_row, src1->ne[0]);
                        }
                    }
                }
            } else if (vec_dot_type == GGML_TYPE_Q8_1) {
                for (int i13 = 0; i13 < src1->ne[3]; ++i13) {
                    for (int i12 = 0; i12 < src1->ne[2]; ++i12) {
                        for (int i11 = 0; i11 < src1->ne[1]; ++i11) {
                            const float * src_row = (const float*)((const char*)src1->data + i13 * src1->nb[3] + i12 * src1->nb[2] + i11 * src1->nb[1]);
                            block_q8_1 * dst_row = (block_q8_1*)((char*)q8_data + i13 * nbw3 + i12 * nbw2 + i11 * nbw1);
                            quantize_row_q8_1(src_row, dst_row, src1->ne[0]);
                        }
                    }
                }
            } else if (vec_dot_type == GGML_TYPE_Q8_K) {
                for (int i13 = 0; i13 < src1->ne[3]; ++i13) {
                    for (int i12 = 0; i12 < src1->ne[2]; ++i12) {
                        for (int i11 = 0; i11 < src1->ne[1]; ++i11) {
                            const float * src_row = (const float*)((const char*)src1->data + i13 * src1->nb[3] + i12 * src1->nb[2] + i11 * src1->nb[1]);
                            block_q8_K * dst_row = (block_q8_K*)((char*)q8_data + i13 * nbw3 + i12 * nbw2 + i11 * nbw1);
                            quantize_row_q8_K(src_row, dst_row, src1->ne[0]);
                        }
                    }
                }
            } else if (vec_dot_type == GGML_TYPE_BF16) {
                for (int i13 = 0; i13 < src1->ne[3]; ++i13) {
                    for (int i12 = 0; i12 < src1->ne[2]; ++i12) {
                        for (int i11 = 0; i11 < src1->ne[1]; ++i11) {
                            const float * src_row = (const float*)((const char*)src1->data + i13 * src1->nb[3] + i12 * src1->nb[2] + i11 * src1->nb[1]);
                            ggml_bf16_t * dst_row = (ggml_bf16_t*)((char*)q8_data + i13 * nbw3 + i12 * nbw2 + i11 * nbw1);
                            quantize_row_bf16(src_row, dst_row, src1->ne[0]);
                        }
                    }
                }
            } else {
                for (int i13 = 0; i13 < src1->ne[3]; ++i13) {
                    for (int i12 = 0; i12 < src1->ne[2]; ++i12) {
                        for (int i11 = 0; i11 < src1->ne[1]; ++i11) {
                            const float * src_row = (const float*)((const char*)src1->data + i13 * src1->nb[3] + i12 * src1->nb[2] + i11 * src1->nb[1]);
                            block_q8_0 * dst_row = (block_q8_0*)((char*)q8_data + i13 * nbw3 + i12 * nbw2 + i11 * nbw1);
                            quantize_row_q8_0(src_row, dst_row, src1->ne[0]);
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

    const enum ggml_type vec_dot_type = ggml_vec_dot_type(type);
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);

    const void * wdata = (src1->type == vec_dot_type) ? src1->data : NULL;

    if (wdata == NULL) {
        const size_t nbw1 = row_size;
        const size_t nbw2 = nbw1 * ne11;
        const size_t nbw3 = nbw2 * ne12;
        const size_t q8_size = nbw3 * ne13;
        void * q8_data = ggmlop_get_work_data(q8_size);
        if (q8_data != NULL) {
            if (vec_dot_type == GGML_TYPE_F16) {
                for (int i13 = 0; i13 < ne13; ++i13) {
                    for (int i12 = 0; i12 < ne12; ++i12) {
                        for (int i11 = 0; i11 < ne11; ++i11) {
                            const float * src_row = (const float*)((const char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);
                            uint16_t * dst_row = (uint16_t*)((char*)q8_data + i13 * nbw3 + i12 * nbw2 + i11 * nbw1);
                            quantize_f32_to_f16_row_hvx(src_row, dst_row, ne10);
                        }
                    }
                }
            } else if (vec_dot_type == GGML_TYPE_Q8_1) {
                for (int i13 = 0; i13 < ne13; ++i13) {
                    for (int i12 = 0; i12 < ne12; ++i12) {
                        for (int i11 = 0; i11 < ne11; ++i11) {
                            const float * src_row = (const float*)((const char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);
                            block_q8_1 * dst_row = (block_q8_1*)((char*)q8_data + i13 * nbw3 + i12 * nbw2 + i11 * nbw1);
                            quantize_row_q8_1(src_row, dst_row, ne10);
                        }
                    }
                }
            } else if (vec_dot_type == GGML_TYPE_Q8_K) {
                for (int i13 = 0; i13 < ne13; ++i13) {
                    for (int i12 = 0; i12 < ne12; ++i12) {
                        for (int i11 = 0; i11 < ne11; ++i11) {
                            const float * src_row = (const float*)((const char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);
                            block_q8_K * dst_row = (block_q8_K*)((char*)q8_data + i13 * nbw3 + i12 * nbw2 + i11 * nbw1);
                            quantize_row_q8_K(src_row, dst_row, ne10);
                        }
                    }
                }
            } else if (vec_dot_type == GGML_TYPE_BF16) {
                for (int i13 = 0; i13 < ne13; ++i13) {
                    for (int i12 = 0; i12 < ne12; ++i12) {
                        for (int i11 = 0; i11 < ne11; ++i11) {
                            const float * src_row = (const float*)((const char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);
                            ggml_bf16_t * dst_row = (ggml_bf16_t*)((char*)q8_data + i13 * nbw3 + i12 * nbw2 + i11 * nbw1);
                            quantize_row_bf16(src_row, dst_row, ne10);
                        }
                    }
                }
            } else {
                for (int i13 = 0; i13 < ne13; ++i13) {
                    for (int i12 = 0; i12 < ne12; ++i12) {
                        for (int i11 = 0; i11 < ne11; ++i11) {
                            const float * src_row = (const float*)((const char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);
                            block_q8_0 * dst_row = (block_q8_0*)((char*)q8_data + i13 * nbw3 + i12 * nbw2 + i11 * nbw1);
                            quantize_row_q8_0(src_row, dst_row, ne10);
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

                        if (type == GGML_TYPE_F16) {
                            vec_dot_f16_f16(ne00, &tmp[row_idx], 0,
                                           (const uint16_t*)(vtcm_buf + row_idx * nb01), 0,
                                           (uint16_t*)src1_col, 0, 1);
                        } else if (type == GGML_TYPE_Q4_0) {
                            const block_q4_0 * q4_row = (const block_q4_0*)(vtcm_buf + row_idx * nb01);
                            const block_q8_0 * q8_col = (const block_q8_0*)src1_col;
                            vec_dot_q4_0_q8_0(ne00, &tmp[row_idx], 0, q4_row, 0, q8_col, 0, 1);
                        } else if (type == GGML_TYPE_Q4_1) {
                            const block_q4_1 * q4_row = (const block_q4_1*)(vtcm_buf + row_idx * nb01);
                            const block_q8_1 * q8_col = (const block_q8_1*)src1_col;
                            vec_dot_q4_1_q8_1(ne00, &tmp[row_idx], 0, q4_row, 0, q8_col, 0, 1);
                        } else if (type == GGML_TYPE_Q5_0) {
                            const block_q5_0 * q5_row = (const block_q5_0*)(vtcm_buf + row_idx * nb01);
                            const block_q8_0 * q8_col = (const block_q8_0*)src1_col;
                            vec_dot_q5_0_q8_0(ne00, &tmp[row_idx], 0, q5_row, 0, q8_col, 0, 1);
                        } else if (type == GGML_TYPE_Q5_1) {
                            const block_q5_1 * q5_row = (const block_q5_1*)(vtcm_buf + row_idx * nb01);
                            const block_q8_1 * q8_col = (const block_q8_1*)src1_col;
                            vec_dot_q5_1_q8_1(ne00, &tmp[row_idx], 0, q5_row, 0, q8_col, 0, 1);
                        } else if (type == GGML_TYPE_IQ4_NL) {
                            const block_iq4_nl * iq4_row = (const block_iq4_nl*)(vtcm_buf + row_idx * nb01);
                            const block_q8_0 * q8_col = (const block_q8_0*)src1_col;
                            vec_dot_iq4_nl_q8_0(ne00, &tmp[row_idx], 0, iq4_row, 0, q8_col, 0, 1);
                        } else if (type == GGML_TYPE_Q8_0) {
                            const block_q8_0 * q8_row = (const block_q8_0*)(vtcm_buf + row_idx * nb01);
                            const block_q8_0 * q8_col = (const block_q8_0*)src1_col;
                            vec_dot_q8_0_q8_0(ne00, &tmp[row_idx], 0, q8_row, 0, q8_col, 0, 1);
                        } else if (type == GGML_TYPE_BF16) {
                            const ggml_bf16_t * bf16_row = (const ggml_bf16_t*)(vtcm_buf + row_idx * nb01);
                            const ggml_bf16_t * bf16_col = (const ggml_bf16_t*)src1_col;
                            vec_dot_bf16_bf16(ne00, &tmp[row_idx], 0, bf16_row, 0, bf16_col, 0, 1);
                        } else if (type == GGML_TYPE_Q4_K) {
                            const block_q4_K * q4_row = (const block_q4_K*)(vtcm_buf + row_idx * nb01);
                            const block_q8_K * q8_col = (const block_q8_K*)src1_col;
                            vec_dot_q4_K_q8_K(ne00, &tmp[row_idx], 0, q4_row, 0, q8_col, 0, 1);
                        } else if (type == GGML_TYPE_Q6_K) {
                            const block_q6_K * q6_row = (const block_q6_K*)(vtcm_buf + row_idx * nb01);
                            const block_q8_K * q8_col = (const block_q8_K*)src1_col;
                            vec_dot_q6_K_q8_K(ne00, &tmp[row_idx], 0, q6_row, 0, q8_col, 0, 1);
                        } else if (type == GGML_TYPE_Q2_K) {
                            const block_q2_K * q2_row = (const block_q2_K*)(vtcm_buf + row_idx * nb01);
                            const block_q8_K * q8_col = (const block_q8_K*)src1_col;
                            vec_dot_q2_K_q8_K(ne00, &tmp[row_idx], 0, q2_row, 0, q8_col, 0, 1);
                        } else if (type == GGML_TYPE_Q3_K) {
                            const block_q3_K * q3_row = (const block_q3_K*)(vtcm_buf + row_idx * nb01);
                            const block_q8_K * q8_col = (const block_q8_K*)src1_col;
                            vec_dot_q3_K_q8_K(ne00, &tmp[row_idx], 0, q3_row, 0, q8_col, 0, 1);
                        } else if (type == GGML_TYPE_Q5_K) {
                            const block_q5_K * q5_row = (const block_q5_K*)(vtcm_buf + row_idx * nb01);
                            const block_q8_K * q8_col = (const block_q8_K*)src1_col;
                            vec_dot_q5_K_q8_K(ne00, &tmp[row_idx], 0, q5_row, 0, q8_col, 0, 1);
                        } else if (type == GGML_TYPE_MXFP4) {
                            const block_mxfp4 * mxfp4_row = (const block_mxfp4*)(vtcm_buf + row_idx * nb01);
                            const block_q8_0 * q8_col = (const block_q8_0*)src1_col;
                            vec_dot_mxfp4_q8_0(ne00, &tmp[row_idx], 0, mxfp4_row, 0, q8_col, 0, 1);
                        } else if (type == GGML_TYPE_NVFP4) {
                            const block_nvfp4 * nvfp4_row = (const block_nvfp4*)(vtcm_buf + row_idx * nb01);
                            const block_q8_0 * q8_col = (const block_q8_0*)src1_col;
                            vec_dot_nvfp4_q8_0(ne00, &tmp[row_idx], 0, nvfp4_row, 0, q8_col, 0, 1);
                        } else if (type == GGML_TYPE_IQ4_XS) {
                            const block_iq4_xs * iq4xs_row = (const block_iq4_xs*)(vtcm_buf + row_idx * nb01);
                            const block_q8_K * q8_col = (const block_q8_K*)src1_col;
                            vec_dot_iq4_xs_q8_K(ne00, &tmp[row_idx], 0, iq4xs_row, 0, q8_col, 0, 1);
                        } else if (type == GGML_TYPE_IQ3_XXS) {
                            const block_iq3_xxs * iq3xxs_row = (const block_iq3_xxs*)(vtcm_buf + row_idx * nb01);
                            const block_q8_K * q8_col = (const block_q8_K*)src1_col;
                            vec_dot_iq3_xxs_q8_K(ne00, &tmp[row_idx], 0, iq3xxs_row, 0, q8_col, 0, 1);
                        } else if (type == GGML_TYPE_IQ2_XXS) {
                            const block_iq2_xxs * iq2xxs_row = (const block_iq2_xxs*)(vtcm_buf + row_idx * nb01);
                            const block_q8_K * q8_col = (const block_q8_K*)src1_col;
                            vec_dot_iq2_xxs_q8_K(ne00, &tmp[row_idx], 0, iq2xxs_row, 0, q8_col, 0, 1);
                        } else if (type == GGML_TYPE_IQ2_XS) {
                            const block_iq2_xs * iq2xs_row = (const block_iq2_xs*)(vtcm_buf + row_idx * nb01);
                            const block_q8_K * q8_col = (const block_q8_K*)src1_col;
                            vec_dot_iq2_xs_q8_K(ne00, &tmp[row_idx], 0, iq2xs_row, 0, q8_col, 0, 1);
                        } else if (type == GGML_TYPE_IQ2_S) {
                            const block_iq2_s * iq2s_row = (const block_iq2_s*)(vtcm_buf + row_idx * nb01);
                            const block_q8_K * q8_col = (const block_q8_K*)src1_col;
                            vec_dot_iq2_s_q8_K(ne00, &tmp[row_idx], 0, iq2s_row, 0, q8_col, 0, 1);
                        } else if (type == GGML_TYPE_IQ1_S) {
                            const block_iq1_s * iq1s_row = (const block_iq1_s*)(vtcm_buf + row_idx * nb01);
                            const block_q8_K * q8_col = (const block_q8_K*)src1_col;
                            vec_dot_iq1_s_q8_K(ne00, &tmp[row_idx], 0, iq1s_row, 0, q8_col, 0, 1);
                        } else {
                            vec_dot_f32(ne00, &tmp[row_idx], 0,
                                        (const float*)(vtcm_buf + row_idx * nb01), 0,
                                        (float*)src1_col, 0, 1);
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
    GGMLHEXAGON_LOG_INFO("src0 type=%d, src1 type=%d", src0->type, src1->type);

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
        //ret= ggmlop_dsp_mulmat_multithread(h, src0, src1, dst);
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
