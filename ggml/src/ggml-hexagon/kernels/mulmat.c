#include "ggml-dsp.h"
#include "worker_pool.h"

union ui32f { int32_t i; float f; };

#define HMX_FP16_TILE_N_ROWS 32
#define HMX_FP16_TILE_N_COLS 32
#define HMX_FP16_TILE_N_ELMS 1024
#define HMX_FP16_TILE_SIZE (HMX_FP16_TILE_N_ELMS * sizeof(__fp16))

static inline size_t hex_align_down(size_t x, size_t align) {
    return (x / align) * align;
}

static inline size_t hex_align_up(size_t x, size_t align) {
    return ((x + align - 1) / align) * align;
}

typedef union {
    HVX_Vector v;
    uint8_t    b[VSIZE_BYTES];
    uint16_t   h[VLEN_FP16];
    uint32_t   w[VLEN_FP32];
    __fp16     fp16[VLEN_FP16];
    float      fp32[VLEN_FP32];
} HVX_VectorAlias;

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

static inline float hvx_vec_get_f32(HVX_Vector v) {
    HVX_VectorAlias va;
    va.v = v;
    return va.fp32[0];
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
                            for (int i = 0; i < ne10; ++i) {
                                dst_row[i] = ggml_compute_fp32_to_fp16(src_row[i]);
                            }
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
                        if (vec_dot_type == GGML_TYPE_F16 && src1->type != GGML_TYPE_F16) {
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
    worker_synctoken_t *synctoken;
} mulmat_thread_data_vtcm_t;

static void ggml_compute_forward_mul_mat_vtcm_chunk(const ggml_tensor *src0, const ggml_tensor *src1,
                                                    struct ggml_tensor *dst,
                                                    const enum ggml_type type,
                                                    const int32_t num_rows_per_vec_dot,
                                                    const int32_t ir0_start, const int32_t ir0_end,
                                                    const int32_t ir1_start, const int32_t ir1_end,
                                                    uint8_t *vtcm_buf, size_t vtcm_size) {
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
                            for (int i = 0; i < ne10; ++i) {
                                dst_row[i] = ggml_compute_fp32_to_fp16(src_row[i]);
                            }
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

                    memcpy(vtcm_buf, src0_row + iir0 * nb01, copy_size);

                    for (int32_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < iir0_end; ir0 += num_rows_per_vec_dot) {
                        const int32_t row_idx = ir0 - iir0;

                        if (type == GGML_TYPE_F16 && vec_dot_type == GGML_TYPE_F16) {
                            vec_dot_f16_f16(ne00, &tmp[row_idx], 0,
                                           (const uint16_t*)(vtcm_buf + row_idx * nb01), 0,
                                           (uint16_t*)src1_col, 0, 1);
                        } else if (type == GGML_TYPE_F16) {
                            vec_dot_f16_f32(ne00, &tmp[row_idx], 0,
                                           (const uint16_t*)(vtcm_buf + row_idx * nb01), 0,
                                           (float*)src1_col, 0, 1);
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
        tdata->vtcm_buf, tdata->vtcm_size
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

    const size_t vtcm_per_thread = 64 * 1024;
    const size_t total_vtcm = vtcm_per_thread * n_threads;

    void *vtcm_base = HAP_request_VTCM(total_vtcm, 0);
    if (vtcm_base == NULL) {
        GGMLHEXAGON_LOG_DEBUG("%s: VTCM allocation failed, falling back to non-VTCM", __func__);
        return ggmlop_dsp_mulmat_multithread(h, src0, src1, dst);
    }

    const int32_t rows_per_thread = (nr1 + n_threads - 1) / n_threads;

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

    HAP_release_VTCM(vtcm_base);

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return 0;
}

int ggmlop_dsp_mulmat(remote_handle64 h, const struct dsptensor * src0, const struct dsptensor * src1, dsptensor * dst) {
    int mulmat_algo = ggmlop_get_mulmat_algotype();
    if (mulmat_algo == 32) {
        GGMLHEXAGON_LOG_DEBUG("mulmat using VTCM+HMX mode");
        return ggmlop_dsp_mulmat_vtcm_hmx(h, src0, src1, dst);
    } else if (ggmlop_get_thread_counts() > 1) {
        GGMLHEXAGON_LOG_DEBUG("mulmat using multithread mode");
        return ggmlop_dsp_mulmat_multithread(h, src0, src1, dst);
    } else {
        GGMLHEXAGON_LOG_DEBUG("mulmat using singlethread mode");
        return ggmlop_dsp_mulmat_singlethread(h, src0, src1, dst);
    }
}

static void transfer_activation_chunk_fp32_to_fp16(__fp16 *restrict vtcm_dst, const float *restrict src,
                                                   int n_rows, int k, int row_stride) {
    const int n_tiles_per_row = (k + HMX_FP16_TILE_N_COLS - 1) / HMX_FP16_TILE_N_COLS;

    for (int r = 0; r < n_rows; r += 2) {
        for (int t = 0; t < n_tiles_per_row; ++t) {
            int tile_idx = (r / HMX_FP16_TILE_N_ROWS) * n_tiles_per_row + t;
            int r_in_tile = r % HMX_FP16_TILE_N_ROWS;

            const float *row0 = src + r * row_stride + t * HMX_FP16_TILE_N_COLS;
            const float *row1 = src + (r + 1) * row_stride + t * HMX_FP16_TILE_N_COLS;

            for (int i = 0; i < HMX_FP16_TILE_N_COLS; ++i) {
                vtcm_dst[tile_idx * HMX_FP16_TILE_N_ELMS + r_in_tile * HMX_FP16_TILE_N_COLS + i] =
                    ggml_compute_fp32_to_fp16(row0[i]);
                vtcm_dst[tile_idx * HMX_FP16_TILE_N_ELMS + (r_in_tile + 1) * HMX_FP16_TILE_N_COLS + i] =
                    ggml_compute_fp32_to_fp16(row1[i]);
            }
        }
    }
}

static void transfer_activation_chunk_f16_to_f16_tiles(__fp16 *restrict vtcm_dst, const __fp16 *restrict src,
                                                        int n_rows, int k, int row_stride) {
    const int n_tiles_per_row = k / HMX_FP16_TILE_N_COLS;

    for (int r = 0; r < n_rows; r += 2) {
        for (int t = 0; t < n_tiles_per_row; ++t) {
            int tile_idx = (r / HMX_FP16_TILE_N_ROWS) * n_tiles_per_row + t;
            int r_in_tile = r % HMX_FP16_TILE_N_ROWS;

            const __fp16 *row0 = src + r * row_stride + t * HMX_FP16_TILE_N_COLS;
            const __fp16 *row1 = src + (r + 1) * row_stride + t * HMX_FP16_TILE_N_COLS;

            for (int i = 0; i < HMX_FP16_TILE_N_COLS; ++i) {
                vtcm_dst[tile_idx * HMX_FP16_TILE_N_ELMS + r_in_tile * HMX_FP16_TILE_N_COLS + i] = row0[i];
                vtcm_dst[tile_idx * HMX_FP16_TILE_N_ELMS + (r_in_tile + 1) * HMX_FP16_TILE_N_COLS + i] = row1[i];
            }
        }
    }
}

static void convert_weight_f32_to_fp16_tiles(__fp16 *restrict vtcm_dst, const float *restrict src,
                                              int n_cols, int k, int row_stride) {
    const int n_tiles_per_col = n_cols / HMX_FP16_TILE_N_COLS;
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;

    for (int c = 0; c < n_cols; c += 2) {
        for (int kt = 0; kt < k_tiles; ++kt) {
            int tile_idx = (c / HMX_FP16_TILE_N_ROWS) * k_tiles + kt;
            int c_in_tile = c % HMX_FP16_TILE_N_ROWS;

            const float *col0 = src + c * row_stride + kt * HMX_FP16_TILE_N_COLS;
            const float *col1 = src + (c + 1) * row_stride + kt * HMX_FP16_TILE_N_COLS;

            for (int i = 0; i < HMX_FP16_TILE_N_COLS; ++i) {
                vtcm_dst[tile_idx * HMX_FP16_TILE_N_ELMS + c_in_tile * HMX_FP16_TILE_N_COLS + i] =
                    ggml_compute_fp32_to_fp16(col0[i]);
                vtcm_dst[tile_idx * HMX_FP16_TILE_N_ELMS + (c_in_tile + 1) * HMX_FP16_TILE_N_COLS + i] =
                    ggml_compute_fp32_to_fp16(col1[i]);
            }
        }
    }
}

static void transfer_weight_chunk_f16_to_f16_tiles(__fp16 *restrict vtcm_dst, const __fp16 *restrict src,
                                                    int n_cols, int k, int row_stride) {
    const int n_tiles_per_col = n_cols / HMX_FP16_TILE_N_COLS;
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;

    for (int c = 0; c < n_cols; c += 2) {
        for (int kt = 0; kt < k_tiles; ++kt) {
            int tile_idx = (c / HMX_FP16_TILE_N_ROWS) * k_tiles + kt;
            int c_in_tile = c % HMX_FP16_TILE_N_ROWS;

            const __fp16 *col0 = src + c * row_stride + kt * HMX_FP16_TILE_N_COLS;
            const __fp16 *col1 = src + (c + 1) * row_stride + kt * HMX_FP16_TILE_N_COLS;

            for (int i = 0; i < HMX_FP16_TILE_N_COLS; ++i) {
                vtcm_dst[tile_idx * HMX_FP16_TILE_N_ELMS + c_in_tile * HMX_FP16_TILE_N_COLS + i] = col0[i];
                vtcm_dst[tile_idx * HMX_FP16_TILE_N_ELMS + (c_in_tile + 1) * HMX_FP16_TILE_N_COLS + i] = col1[i];
            }
        }
    }
}

static void core_dot_chunk_fp16(__fp16 *restrict output, const __fp16 *restrict activation,
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
                const uint32_t range = (uint32_t)(k_block - 1);

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

static void transfer_output_chunk_fp16_to_fp32(float *restrict dst, const __fp16 *restrict src,
                                                int n_rows, int n_cols, int row_stride) {
    const int n_row_tiles = (n_rows + HMX_FP16_TILE_N_ROWS - 1) / HMX_FP16_TILE_N_ROWS;
    const int n_col_tiles = n_cols / HMX_FP16_TILE_N_COLS;

    for (int r = 0; r < n_rows; ++r) {
        int tile_r = r / HMX_FP16_TILE_N_ROWS;
        int r_in_tile = r % HMX_FP16_TILE_N_ROWS;

        for (int c = 0; c < n_cols; ++c) {
            int tile_c = c / HMX_FP16_TILE_N_COLS;
            int c_in_tile = c % HMX_FP16_TILE_N_COLS;

            int tile_idx = tile_r * n_col_tiles + tile_c;
            const __fp16 *tile = src + tile_idx * HMX_FP16_TILE_N_ELMS;
            dst[r * row_stride + c] = ggml_compute_fp16_to_fp32(tile[r_in_tile * HMX_FP16_TILE_N_COLS + c_in_tile]);
        }
    }
}

static void dequantize_q4_0_to_f16_tiles(__fp16 *restrict vtcm_dst, const block_q4_0 *restrict src,
                                         int n_cols, int k) {
    const int n_tiles_per_col = n_cols / HMX_FP16_TILE_N_COLS;
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int nb_per_col = k / QK4_0;
    const int nb_per_tile = HMX_FP16_TILE_N_COLS / QK4_0;

    for (int c = 0; c < n_cols; c += 2) {
        for (int kt = 0; kt < k_tiles; ++kt) {
            int tile_idx = (c / HMX_FP16_TILE_N_ROWS) * k_tiles + kt;
            int c_in_tile = c % HMX_FP16_TILE_N_ROWS;

            const block_q4_0 *col0_blocks = src + c * nb_per_col + kt * nb_per_tile;
            const block_q4_0 *col1_blocks = src + (c + 1) * nb_per_col + kt * nb_per_tile;

            for (int b = 0; b < nb_per_tile; ++b) {
                float d0 = ggml_compute_fp16_to_fp32(col0_blocks[b].d);
                float d1 = ggml_compute_fp16_to_fp32(col1_blocks[b].d);

                for (int i = 0; i < QK4_0 / 2; ++i) {
                    int8_t q0 = (col0_blocks[b].qs[i] & 0x0F) - 8;
                    int8_t q1 = (col0_blocks[b].qs[i] >> 4) - 8;
                    int8_t q2 = (col1_blocks[b].qs[i] & 0x0F) - 8;
                    int8_t q3 = (col1_blocks[b].qs[i] >> 4) - 8;

                    int idx = tile_idx * HMX_FP16_TILE_N_ELMS + c_in_tile * HMX_FP16_TILE_N_COLS + b * QK4_0;
                    vtcm_dst[idx + 2 * i] = ggml_compute_fp32_to_fp16(q0 * d0);
                    vtcm_dst[idx + 2 * i + 1] = ggml_compute_fp32_to_fp16(q1 * d0);

                    idx = tile_idx * HMX_FP16_TILE_N_ELMS + (c_in_tile + 1) * HMX_FP16_TILE_N_COLS + b * QK4_0;
                    vtcm_dst[idx + 2 * i] = ggml_compute_fp32_to_fp16(q2 * d1);
                    vtcm_dst[idx + 2 * i + 1] = ggml_compute_fp32_to_fp16(q3 * d1);
                }
            }
        }
    }
}

static void dequantize_q4_1_to_f16_tiles(__fp16 *restrict vtcm_dst, const block_q4_1 *restrict src,
                                         int n_cols, int k) {
    const int n_tiles_per_col = n_cols / HMX_FP16_TILE_N_COLS;
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int nb_per_col = k / QK4_1;
    const int nb_per_tile = HMX_FP16_TILE_N_COLS / QK4_1;

    for (int c = 0; c < n_cols; c += 2) {
        for (int kt = 0; kt < k_tiles; ++kt) {
            int tile_idx = (c / HMX_FP16_TILE_N_ROWS) * k_tiles + kt;
            int c_in_tile = c % HMX_FP16_TILE_N_ROWS;

            const block_q4_1 *col0_blocks = src + c * nb_per_col + kt * nb_per_tile;
            const block_q4_1 *col1_blocks = src + (c + 1) * nb_per_col + kt * nb_per_tile;

            for (int b = 0; b < nb_per_tile; ++b) {
                float d0 = ggml_compute_fp16_to_fp32(col0_blocks[b].d);
                float m0 = ggml_compute_fp16_to_fp32(col0_blocks[b].m);
                float d1 = ggml_compute_fp16_to_fp32(col1_blocks[b].d);
                float m1 = ggml_compute_fp16_to_fp32(col1_blocks[b].m);

                for (int i = 0; i < QK4_1 / 2; ++i) {
                    int8_t q0 = (col0_blocks[b].qs[i] & 0x0F);
                    int8_t q1 = (col0_blocks[b].qs[i] >> 4);
                    int8_t q2 = (col1_blocks[b].qs[i] & 0x0F);
                    int8_t q3 = (col1_blocks[b].qs[i] >> 4);

                    int idx = tile_idx * HMX_FP16_TILE_N_ELMS + c_in_tile * HMX_FP16_TILE_N_COLS + b * QK4_1;
                    vtcm_dst[idx + 2 * i] = ggml_compute_fp32_to_fp16(q0 * d0 + m0);
                    vtcm_dst[idx + 2 * i + 1] = ggml_compute_fp32_to_fp16(q1 * d0 + m0);

                    idx = tile_idx * HMX_FP16_TILE_N_ELMS + (c_in_tile + 1) * HMX_FP16_TILE_N_COLS + b * QK4_1;
                    vtcm_dst[idx + 2 * i] = ggml_compute_fp32_to_fp16(q2 * d1 + m1);
                    vtcm_dst[idx + 2 * i + 1] = ggml_compute_fp32_to_fp16(q3 * d1 + m1);
                }
            }
        }
    }
}

static void dequantize_q8_0_to_f16_tiles(__fp16 *restrict vtcm_dst, const block_q8_0 *restrict src,
                                         int n_cols, int k) {
    const int n_tiles_per_col = n_cols / HMX_FP16_TILE_N_COLS;
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int nb_per_col = k / QK8_0;
    const int nb_per_tile = HMX_FP16_TILE_N_COLS / QK8_0;

    for (int c = 0; c < n_cols; c += 2) {
        for (int kt = 0; kt < k_tiles; ++kt) {
            int tile_idx = (c / HMX_FP16_TILE_N_ROWS) * k_tiles + kt;
            int c_in_tile = c % HMX_FP16_TILE_N_ROWS;

            const block_q8_0 *col0_blocks = src + c * nb_per_col + kt * nb_per_tile;
            const block_q8_0 *col1_blocks = src + (c + 1) * nb_per_col + kt * nb_per_tile;

            for (int b = 0; b < nb_per_tile; ++b) {
                float d0 = ggml_compute_fp16_to_fp32(col0_blocks[b].d);
                float d1 = ggml_compute_fp16_to_fp32(col1_blocks[b].d);

                for (int i = 0; i < QK8_0; ++i) {
                    int idx = tile_idx * HMX_FP16_TILE_N_ELMS + c_in_tile * HMX_FP16_TILE_N_COLS + b * QK8_0 + i;
                    vtcm_dst[idx] = ggml_compute_fp32_to_fp16(col0_blocks[b].qs[i] * d0);

                    idx = tile_idx * HMX_FP16_TILE_N_ELMS + (c_in_tile + 1) * HMX_FP16_TILE_N_COLS + b * QK8_0 + i;
                    vtcm_dst[idx] = ggml_compute_fp32_to_fp16(col1_blocks[b].qs[i] * d1);
                }
            }
        }
    }
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

    const int32_t m = src0->ne[1];
    const int32_t k = src0->ne[0];
    const int32_t n = src1->ne[1];

    if (k % HMX_FP16_TILE_N_COLS != 0 || n % HMX_FP16_TILE_N_COLS != 0) {
        if (hmx_locked) {
            HAP_compute_res_hmx_unlock(compute_res_ctx_id);
        }
        //GGMLHEXAGON_LOG_INFO("falling back to VTCM multithread mode\n");
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

    const size_t vec_dot_size = k * sizeof(__fp16);
    const size_t max_m_chunk = (vtcm_size / 4) / vec_dot_size;
    const size_t m_chunk = ((max_m_chunk / HMX_FP16_TILE_N_ROWS) * HMX_FP16_TILE_N_ROWS);
    const size_t m_chunk_n_rows = (m_chunk == 0) ? HMX_FP16_TILE_N_ROWS : m_chunk;

    const size_t n_chunk_n_cols = hex_align_down((size_t)n, HMX_FP16_TILE_N_COLS);

    const size_t act_area_size    = hex_align_up(m_chunk_n_rows * vec_dot_size, HMX_FP16_TILE_SIZE);
    const size_t weight_area_size = hex_align_up(n_chunk_n_cols * vec_dot_size, HMX_FP16_TILE_SIZE);
    const size_t output_area_size = hex_align_up(m_chunk_n_rows * n_chunk_n_cols * sizeof(__fp16), HMX_FP16_TILE_SIZE);
    const size_t scales_size      = 256;

    const size_t total_vtcm_needed = act_area_size + weight_area_size + output_area_size + scales_size;
    if (total_vtcm_needed > vtcm_size) {
        if (hmx_locked) {
            HAP_compute_res_hmx_unlock(compute_res_ctx_id);
        }
        return ggmlop_dsp_mulmat_multithread_vtcm(h, src0, src1, dst);
    }

    uint8_t *vtcm_ptr = (uint8_t *)vtcm_base;
    __fp16 *vtcm_activation = (__fp16 *) vtcm_ptr;
    vtcm_ptr += act_area_size;
    __fp16 *vtcm_weight = (__fp16 *) vtcm_ptr;
    vtcm_ptr += weight_area_size;
    __fp16 *vtcm_output = (__fp16 *) vtcm_ptr;
    vtcm_ptr += output_area_size;
    __fp16 *vtcm_scales = (__fp16 *) vtcm_ptr;

    HVX_Vector v_scale = Q6_V_vsplat_R(0x3c00);
    volatile HVX_Vector *pv_scales = (volatile HVX_Vector *) vtcm_scales;
    pv_scales[0] = v_scale;
    pv_scales[1] = Q6_V_vzero();

    const size_t n_dot_tiles = k / HMX_FP16_TILE_N_COLS;

    const bool src0_is_f16 = (src0->type == GGML_TYPE_F16);
    const bool src1_is_f16 = (src1->type == GGML_TYPE_F16);

    const size_t src0_row_stride = src0->nb[1];
    const size_t src1_row_stride = src1->nb[1];

    for (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
        const size_t n_rows = (m - mr) > m_chunk_n_rows ? m_chunk_n_rows : (m - mr);
        const size_t n_row_tiles = ((n_rows + HMX_FP16_TILE_N_ROWS - 1) / HMX_FP16_TILE_N_ROWS);

        if (src0_is_f16) {
            const __fp16 *activation_chunk = (const __fp16 *)((const char *)src0->data + mr * src0_row_stride);
            transfer_activation_chunk_f16_to_f16_tiles(vtcm_activation, activation_chunk, n_rows, k, src0_row_stride / sizeof(__fp16));
        } else {
            const float *activation_chunk = (const float *)((const char *)src0->data + mr * src0_row_stride);
            transfer_activation_chunk_fp32_to_fp16(vtcm_activation, activation_chunk, n_rows, k, src0_row_stride / sizeof(float));
        }

        for (size_t nc = 0; nc < n; nc += n_chunk_n_cols) {
            const size_t n_cols = (n - nc) > n_chunk_n_cols ? n_chunk_n_cols : (n - nc);
            const size_t n_col_tiles = n_cols / HMX_FP16_TILE_N_COLS;

            if (src1->type == GGML_TYPE_F16) {
                const __fp16 *weight_chunk = (const __fp16 *)((const char *)src1->data + nc * src1_row_stride);
                transfer_weight_chunk_f16_to_f16_tiles(vtcm_weight, weight_chunk, n_cols, k, src1_row_stride / sizeof(__fp16));
            } else if (src1->type == GGML_TYPE_F32) {
                const float *weight_chunk = (const float *)((const char *)src1->data + nc * src1_row_stride);
                convert_weight_f32_to_fp16_tiles(vtcm_weight, weight_chunk, n_cols, k, src1_row_stride / sizeof(float));
            } else if (src1->type == GGML_TYPE_Q4_0) {
                const block_q4_0 *weight_chunk = (const block_q4_0 *)((const char *)src1->data + nc * src1_row_stride);
                dequantize_q4_0_to_f16_tiles(vtcm_weight, weight_chunk, n_cols, k);
            } else if (src1->type == GGML_TYPE_Q4_1) {
                const block_q4_1 *weight_chunk = (const block_q4_1 *)((const char *)src1->data + nc * src1_row_stride);
                dequantize_q4_1_to_f16_tiles(vtcm_weight, weight_chunk, n_cols, k);
            } else if (src1->type == GGML_TYPE_Q8_0) {
                const block_q8_0 *weight_chunk = (const block_q8_0 *)((const char *)src1->data + nc * src1_row_stride);
                dequantize_q8_0_to_f16_tiles(vtcm_weight, weight_chunk, n_cols, k);
            }

            core_dot_chunk_fp16(vtcm_output, vtcm_activation, vtcm_weight, vtcm_scales, n_row_tiles, n_col_tiles, n_dot_tiles);

            float *output_chunk = (float *)((char *)dst->data + mr * dst->nb[1] + nc * dst->nb[0]);
            transfer_output_chunk_fp16_to_fp32(output_chunk, vtcm_output, n_rows, n_cols, n);
        }
    }

    if (hmx_locked) {
        HAP_compute_res_hmx_unlock(compute_res_ctx_id);
    }

    return 0;
}
