#include "ggml-dsp.h"
#include "worker_pool.h"
#include "../htp/hvx-base.h"  // for official hvx_vec_f32_to_f16 with vdeal
#include "../htp/hex-dma.h"   // for official DMA async transfers

union ui32f { int32_t i; float f; };

#define HMX_FP16_TILE_N_ROWS 32
#define HMX_FP16_TILE_N_COLS 32
#define HMX_FP16_TILE_N_ELMS 1024
#define HMX_FP16_TILE_SIZE (HMX_FP16_TILE_N_ELMS * sizeof(__fp16))

// Forward declarations
int ggmlop_dsp_mulmat_vtcm_hmx(remote_handle64 h, const struct dsptensor * src0, const struct dsptensor * src1, dsptensor * dst);

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

static inline HVX_Vector hvx_vec_reduce_sum_n_f32(HVX_Vector in, unsigned int n) {
    unsigned int total = n * 4;
    unsigned int width = 4;

    HVX_Vector sum = in, sum_t;
    while (width < total) {
        sum_t = Q6_V_vror_VR(sum, width);
        sum   = Q6_Vsf_vadd_VsfVsf(sum, sum_t);
        width = width << 1;
    }
    return sum;
}

static inline HVX_Vector hvx_vec_reduce_sum_f32(HVX_Vector in) {
    return hvx_vec_reduce_sum_n_f32(in, VLEN_FP32);
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

    // HVX accelerated path: process 64 fp16 + 64 f32 per iteration
    // Requires: n >= 64, both pointers 128-byte aligned
    if (n >= VLEN_FP16 && ((uintptr_t)x & 0x7F) == 0 && ((uintptr_t)y & 0x7F) == 0) {
        const HVX_Vector * restrict px = (const HVX_Vector *) x;
        const HVX_Vector * restrict py = (const HVX_Vector *) y;

        uint32_t nvec = n / VLEN_FP16;
        uint32_t nloe = n % VLEN_FP16;

        HVX_Vector rsum = Q6_V_vsplat_R(0);

        uint32_t i = 0;
        #pragma unroll(2)
        for (i = 0; i < nvec; i++) {
            // Load 64 fp16 from x, convert to f32 pair (64 floats)
            HVX_Vector vx = px[i];
            HVX_VectorPair vxf = hvx_vec_f16_to_f32(vx);

            // Load 64 f32 from y (2 HVX vectors)
            HVX_Vector vy0 = py[2 * i];
            HVX_Vector vy1 = py[2 * i + 1];

            // f32 multiply-accumulate
            HVX_Vector prod0 = Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(vxf), vy0);
            HVX_Vector prod1 = Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(vxf), vy1);

            rsum = Q6_Vsf_vadd_VsfVsf(rsum, prod0);
            rsum = Q6_Vsf_vadd_VsfVsf(rsum, prod1);
        }

        if (nloe) {
            // Handle remaining elements (< 64) with scalar tail
            float sumf = hvx_vec_get_f32(hvx_vec_reduce_sum_f32(rsum));
            int base = nvec * VLEN_FP16;
            for (uint32_t j = 0; j < nloe; ++j) {
                float va = ggml_compute_fp16_to_fp32(x[base + j]);
                sumf += va * y[base + j];
            }
            *s = sumf;
            return;
        }

        *s = hvx_vec_get_f32(hvx_vec_reduce_sum_f32(rsum));
        return;
    }

    // Scalar fallback
    ggml_float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        float va = ggml_compute_fp16_to_fp32(x[i]);
        float vb = y[i];
        sumf += (ggml_float) (va * vb);
    }
    *s = sumf;
}

static void vec_dot_f16_f16(int n, float *GGML_RESTRICT s, size_t bs, const uint16_t *GGML_RESTRICT x,
                    size_t bx, const uint16_t *GGML_RESTRICT y, size_t by, int nrc) {
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(nrc);

    // HVX accelerated path: process 64 fp16 + 64 fp16 per iteration
    // Uses qf32 accumulator for precision (fp16*fp16 -> qf32 mac)
    // Requires: n >= 64, both pointers 128-byte aligned
    if (n >= VLEN_FP16 && ((uintptr_t)x & 0x7F) == 0 && ((uintptr_t)y & 0x7F) == 0) {
        const HVX_Vector * restrict px = (const HVX_Vector *) x;
        const HVX_Vector * restrict py = (const HVX_Vector *) y;

        uint32_t nvec = n / VLEN_FP16;
        uint32_t nloe = n % VLEN_FP16;

        // Use qf32 accumulator for better precision with f16*f16 products
        HVX_Vector acc_lo = create_qf32v_from_sf(0.0f);
        HVX_Vector acc_hi = create_qf32v_from_sf(0.0f);

        #pragma unroll(2)
        for (uint32_t i = 0; i < nvec; i++) {
            // Load 64 fp16 from each input
            HVX_Vector vx = px[i];
            HVX_Vector vy = py[i];

            // fp16 * fp16 -> qf32 pair (high precision multiply)
            HVX_VectorPair prod = Q6_Wqf32_vmpy_VhfVhf(vx, vy);

            // Accumulate in qf32 domain
            acc_lo = Q6_Vqf32_vadd_Vqf32Vqf32(acc_lo, Q6_V_lo_W(prod));
            acc_hi = Q6_Vqf32_vadd_Vqf32Vqf32(acc_hi, Q6_V_hi_W(prod));
        }

        // Convert qf32 accumulators to sf and horizontal reduce
        HVX_Vector sf_lo = Q6_Vsf_equals_Vqf32(acc_lo);
        HVX_Vector sf_hi = Q6_Vsf_equals_Vqf32(acc_hi);

        float sumf = hvx_vec_get_f32(hvx_vec_reduce_sum_f32(sf_lo))
                   + hvx_vec_get_f32(hvx_vec_reduce_sum_f32(sf_hi));

        if (nloe) {
            int base = nvec * VLEN_FP16;
            for (uint32_t j = 0; j < nloe; ++j) {
                float va = ggml_compute_fp16_to_fp32(x[base + j]);
                float vb = ggml_compute_fp16_to_fp32(y[base + j]);
                sumf += va * vb;
            }
        }

        *s = sumf;
        return;
    }

    // Scalar fallback
    ggml_float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        float va = ggml_compute_fp16_to_fp32(x[i]);
        float vb = ggml_compute_fp16_to_fp32(y[i]);
        sumf += (ggml_float) (va * vb);
    }
    *s = sumf;
}

#define QK4_0 32
#define QK4_1 32
#define QK8_0 32

#define GGML_Q4_0_BLCK_SZ (sizeof(uint16_t) + QK4_0/2)
#define GGML_Q4_1_BLCK_SZ (sizeof(uint16_t) + sizeof(uint16_t) + QK4_1/2)
#define GGML_Q8_0_BLCK_SZ (sizeof(uint16_t) + QK8_0)

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
    worker_synctoken_t *synctoken;
} mulmat_thread_data_t;

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

// HVX-accelerated: convert one row of n floats to fp16
// Processes 64 floats per iteration using hvx_vec_f32_to_f16
static inline void quantize_f32_to_f16_row_hvx(const float * restrict src,
                                                uint16_t * restrict dst, int n) {
    if (n >= VLEN_FP16 && ((uintptr_t)src & 0x7F) == 0 && ((uintptr_t)dst & 0x7F) == 0) {
        const HVX_Vector * restrict ps = (const HVX_Vector *) src;
        HVX_Vector * restrict pd = (HVX_Vector *) dst;
        uint32_t nvec = n / VLEN_FP16;
        #pragma unroll(2)
        for (uint32_t i = 0; i < nvec; i++) {
            // Load 64 floats as 2 HVX vectors, convert to 64 fp16 in 1 HVX vector
            HVX_Vector v0 = ps[2 * i];
            HVX_Vector v1 = ps[2 * i + 1];
            pd[i] = hvx_vec_f32_to_f16(v0, v1);
        }
        // Scalar tail
        int tail_base = nvec * VLEN_FP16;
        for (int j = tail_base; j < n; ++j) {
            dst[j] = ggml_compute_fp32_to_fp16(src[j]);
        }
    } else {
        // Fallback: scalar conversion
        for (int i = 0; i < n; ++i) {
            dst[i] = ggml_compute_fp32_to_fp16(src[i]);
        }
    }
}

static enum ggml_type ggml_vec_dot_type(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return GGML_TYPE_F32;
        case GGML_TYPE_F16:
            return GGML_TYPE_F16;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
            return GGML_TYPE_Q8_0;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_1:
            return GGML_TYPE_Q8_1;
        case GGML_TYPE_Q8_0:
            return GGML_TYPE_Q8_0;
        case GGML_TYPE_Q8_1:
            return GGML_TYPE_Q8_1;
        default:
            return GGML_TYPE_F32;
    }
}



static void ggml_compute_forward_mul_mat_one_chunk(const ggml_tensor *src0, const ggml_tensor *src1,
                                                   struct ggml_tensor *dst,
                                                   const enum ggml_type type,
                                                   const enum ggml_type vec_dot_type,
                                                   const int32_t num_rows_per_vec_dot,
                                                   const int32_t ir0_start, const int32_t ir0_end,
                                                   const int32_t ir1_start, const int32_t ir1_end) {
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

    const void * wdata = src1->data;
    if (src1->type != vec_dot_type) {
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
        }
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
                    } else if (type == GGML_TYPE_Q8_0) {
                        const block_q8_0 * q8_row = (const block_q8_0*)(src0_row + ir0 * nb01);
                        const block_q8_0 * q8_col = (const block_q8_0*)src1_col;
                        vec_dot_q8_0_q8_0(ne00, &tmp[row_idx], 0, q8_row, 0, q8_col, 0, 1);
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
        tdata->ir1_start, tdata->ir1_end
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
                                               ir0_start, ir0_end, ir1_start, ir1_end);

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
                            for (int i = 0; i < src1->ne[0]; ++i) {
                                dst_row[i] = ggml_compute_fp32_to_fp16(src_row[i]);
                            }
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

    FARF(HIGH, "mulmat multithread: num_workers=%u, n_threads=%u, nr1=%d", num_workers, n_threads, nr1);

    if (n_threads == 1) {
        FARF(HIGH, "WARNING: Running single-threaded! num_workers=%u", num_workers);
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
                        } else if (type == GGML_TYPE_Q8_0) {
                            const block_q8_0 * q8_row = (const block_q8_0*)(vtcm_buf + row_idx * nb01);
                            const block_q8_0 * q8_col = (const block_q8_0*)src1_col;
                            vec_dot_q8_0_q8_0(ne00, &tmp[row_idx], 0, q8_row, 0, q8_col, 0, 1);
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

    size_t vtcm_size = 0;
    void *vtcm_base = ggmlop_get_vtcm_pool(&vtcm_size);

    //const size_t vtcm_per_thread = 64 * 1024;
    //const size_t total_vtcm = vtcm_per_thread * n_threads;
    size_t vtcm_per_thread = vtcm_size / n_threads;
    size_t total_vtcm      = vtcm_size;

    if (vtcm_base == NULL) {
        GGMLHEXAGON_LOG_DEBUG("%s: VTCM allocation failed, falling back to non-VTCM", __func__);
        return ggmlop_dsp_mulmat_multithread(h, src0, src1, dst);
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

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return 0;
}

int ggmlop_dsp_mulmat(remote_handle64 h, const struct dsptensor * src0, const struct dsptensor * src1, dsptensor * dst) {
    int  ret = 0;
    char tempbuf[256];
    int  mulmat_algo = ggmlop_get_mulmat_algotype();
    ggmlhexagon_get_opkey(GGML_OP_MUL_MAT, src0, src1, tempbuf, 256);
    int64_t begin_time = ggml_time_us();
    if (mulmat_algo == 32) {
        GGMLHEXAGON_LOG_DEBUG("mulmat using VTCM+HMX mode");
        ret = ggmlop_dsp_mulmat_vtcm_hmx(h, src0, src1, dst);
    } else if (ggmlop_get_thread_counts() > 1) {
        GGMLHEXAGON_LOG_DEBUG("mulmat using multithread mode");
        ret= ggmlop_dsp_mulmat_multithread(h, src0, src1, dst);
    } else {
        GGMLHEXAGON_LOG_DEBUG("mulmat using singlethread mode");
        ret = ggmlop_dsp_mulmat_singlethread(h, src0, src1, dst);
    }
    int64_t end_time = ggml_time_us();
    GGMLHEXAGON_LOG_INFO("elapse time of %s is %lld us", tempbuf, (long long)(end_time - begin_time));
    GGMLHEXAGON_LOG_DEBUG("leave %s\n", __func__);
    return ret;
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
    // Process all tiles (matching convert_weight_f32_to_fp16_tiles structure)
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int n_col_tiles = n_cols / HMX_FP16_TILE_N_COLS;
    const int n_tot_tiles = n_col_tiles * k_tiles;
    const int nb_per_col = k / QK4_0;

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
    // Process all tiles (matching convert_weight_f32_to_fp16_tiles structure)
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int n_col_tiles = n_cols / HMX_FP16_TILE_N_COLS;
    const int n_tot_tiles = n_col_tiles * k_tiles;
    const int nb_per_col = k / QK8_0;

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

// Worker for parallel memcpy of rows
typedef struct {
    float       *dst;
    const float *src;
    int          k;
    int          src_stride;
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

// Range-aware output writeback: only processes rows [start_row, end_row)
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

int ggmlop_dsp_mulmat_vtcm_hmx(remote_handle64 h, const struct dsptensor * src0, const struct dsptensor * src1, dsptensor * dst) {
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
