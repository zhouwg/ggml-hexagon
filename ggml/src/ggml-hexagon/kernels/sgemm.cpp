// Copyright 2024 Mozilla Foundation
// Copyright 2025 ggml-hexagon contributors
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//
//                   _   _          ___ _      _   ___
//                  | |_(_)_ _ _  _| _ ) |    /_\ / __|
//                  |  _| | ' \ || | _ \ |__ / _ \\__ \.
//                   \__|_|_||_\_, |___/____/_/ \_\___/
//                             |__/
//
//                    BASIC LINEAR ALGEBRA SUBPROGRAMS
//
//
// This file implements multithreaded CPU matrix multiplication for the
// common contiguous use case C = A^T * B. These kernels are designed to
// have excellent performance[1] for matrices that fit in the CPU cache
// without imposing any overhead such as cache filling or malloc calls.
//
// This implementation does not guarantee any upper bound with rounding
// errors, which grow along with k. Our goal's to maximally exploit the
// hardware for performance, and then use whatever resources remain for
// improving numerical accuracy.
//
// [1] J. Tunney, 'LLaMA Now Goes Faster on CPUs', Mar. 2024. [Online].
//     Available: https://justine.lol/matmul/. [Accessed: 29-Mar-2024].

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#include "sgemm.h"
#include "ggml-dsp.h"

#include <type_traits>

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((__noinline__))
#endif

#if defined(__HVX__)
#define VECTOR_REGISTERS 32
#else
#define VECTOR_REGISTERS 16
#endif

namespace {

inline float unhalf(ggml_fp16_t d) {
    return GGML_FP16_TO_FP32(d);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED ARITHMETIC OPERATIONS (HVX)

#if defined(__HVX__)

inline HVX_Vector add(HVX_Vector x, HVX_Vector y) {
    HVX_Vector qf32 = Q6_Vqf32_vadd_VsfVsf(x, y);
    return Q6_Vsf_equals_Vqf32(qf32);
}

inline HVX_Vector sub(HVX_Vector x, HVX_Vector y) {
    HVX_Vector qf32 = Q6_Vqf32_vsub_VsfVsf(x, y);
    return Q6_Vsf_equals_Vqf32(qf32);
}

inline HVX_Vector mul(HVX_Vector x, HVX_Vector y) {
    HVX_Vector qf32 = Q6_Vqf32_vmpy_VsfVsf(x, y);
    return Q6_Vsf_equals_Vqf32(qf32);
}

#endif // __HVX__

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED FUSED MULTIPLY ADD

template <typename T, typename U>
inline U madd(T a, T b, U c) {
    return add(mul(a, b), c);
}

#if defined(__HVX__)
template <>
inline HVX_Vector madd(HVX_Vector a, HVX_Vector b, HVX_Vector c) {
    return Q6_Vsf_equals_Vqf32(
        Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(a, b), c));
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED HORIZONTAL SUM

#if defined(__HVX__)
static inline float hsum_hvx_fast(HVX_Vector x) {
    // qf32 accumulate with vror for horizontal reduction
#if defined(v68) || defined(v69) || defined(v73) || defined(v75)
    x = Q6_Vqf32_vadd_VsfVsf(x, Q6_V_vror_VR(x, 64));
    x = Q6_Vqf32_vadd_Vqf32Vqf32(x, Q6_V_vror_VR(x, 32));
    x = Q6_Vqf32_vadd_Vqf32Vqf32(x, Q6_V_vror_VR(x, 16));
    x = Q6_Vqf32_vadd_Vqf32Vqf32(x, Q6_V_vror_VR(x, 8));
    x = Q6_Vqf32_vadd_Vqf32Vqf32(x, Q6_V_vror_VR(x, 4));
    x = Q6_Vsf_equals_Vqf32(x);
#else
    x = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(x, Q6_V_vror_VR(x, 64)));
    x = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(x, Q6_V_vror_VR(x, 32)));
    x = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(x, Q6_V_vror_VR(x, 16)));
    x = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(x, Q6_V_vror_VR(x, 8)));
    x = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(x, Q6_V_vror_VR(x, 4)));
#endif
    return *((float*)&x);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED MEMORY LOADING

template <typename T, typename U> T load(const U *);

#if defined(__HVX__)
template <> inline HVX_Vector load(const float *p) {
    return *((HVX_Vector*)p);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// TILING HELPERS

template <int M>
static inline int64_t SGEMM_BLOCK_SIZE(size_t m) {
    const int64_t NB_BLOC_M = (m + M - 1) / M;
    return (m % NB_BLOC_M == 0) ? m / NB_BLOC_M : (m / NB_BLOC_M) + 1;
}

static constexpr inline int64_t BLOC_POS(int64_t ib, int64_t ibN, int64_t bloc_size) {
    return ib < ibN ? ib * bloc_size : ibN * bloc_size + (ib - ibN) * (bloc_size - 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// FLOATING POINT MATRIX MULTIPLICATION
//
// tinyBLAS_Fast: based on tinyBLASFast from refs/sgemm.cpp
// Uses BLOC_POS tiling for cache-friendly access pattern.
// KN=32 for HVX_Vector (128 bytes / 4 bytes per float = 32 floats).

template <int KN, typename D, typename V, typename TA, typename TB, typename TC>
class tinyBLAS_Fast {
  public:
    tinyBLAS_Fast(const ggmldsp_compute_params * params, int64_t k,
             const TA *A, int64_t lda,
             const TB *B, int64_t ldb,
             TC *C, int64_t ldc)
        : ith(params->ith), nth(params->nth), A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc) {
    }

    bool matmul(int64_t m, int64_t n) {
        if (k % KN != 0)
            return false;
#if VECTOR_REGISTERS == 32
        if (m % 16 == 0 && (m/16 >= nth)) {
            const int64_t SIZE_N = SGEMM_BLOCK_SIZE<6>(n);
            mnpack<4, 6, 4>(m, n, SIZE_N, 12);
            return true;
        }
        if (m % 8 == 0 ) {
            const int64_t SIZE_N = SGEMM_BLOCK_SIZE<6>(n);
            mnpack<4, 6, 2>(m, n, SIZE_N, 12);
            return true;
        }
        if (m % 4 == 0) {
            const int64_t SIZE_N = SGEMM_BLOCK_SIZE<6>(n);
            mnpack<4, 6, 1>(m, n, SIZE_N, 12);
            return true;
        }
#else
        if (m % 16 == 0 && (m/16 >= nth)) {
            const int64_t SIZE_N = SGEMM_BLOCK_SIZE<3>(n);
            mnpack<4, 3, 4>(m, n, SIZE_N, 24);
            return true;
        }
        if (m % 8 == 0 ) {
            const int64_t SIZE_N = SGEMM_BLOCK_SIZE<3>(n);
            mnpack<4, 3, 2>(m, n, SIZE_N, 24);
            return true;
        }
        if (m % 4 == 0) {
            const int64_t SIZE_N = SGEMM_BLOCK_SIZE<3>(n);
            mnpack<4, 3, 1>(m, n, SIZE_N, 24);
            return true;
        }
#endif
        return false;
    }

  private:
    template <int RM, int RN, int BM>
    inline void mnpack(int64_t m, int64_t n, int64_t SIZE_N, int64_t BN) {
        if (SIZE_N == RN) {
            return gemm<RM, RN, BM>(m, n, BN);
        }
        if constexpr (RN > 1) {
            return mnpack<RM, RN-1, BM>(m, n, SIZE_N, BN);
        } else {
            GGML_ASSERT(false);
        }
    }

    template <int RM, int RN>
    inline void gemm_bloc(int64_t ii, int64_t jj) {
        D Cv[RN][RM] = {};
        for (int64_t l = 0; l < k; l += KN) {
            if constexpr (RM <= RN) {
                V Av[RM];
                for (int64_t i = 0; i < RM; ++i) {
                    Av[i] = load<V>(A + lda * (ii + i) + l);
                }
                for (int64_t j = 0; j < RN; ++j) {
                    V Bv = load<V>(B + ldb * (jj + j) + l);
                    for (int64_t i = 0; i < RM; ++i) {
                        Cv[j][i] = madd(Av[i], Bv, Cv[j][i]);
                    }
                }
            } else {
                V Bv[RN];
                for (int64_t j = 0; j < RN; ++j) {
                    Bv[j] = load<V>(B + ldb * (jj + j) + l);
                }
                for (int64_t i = 0; i < RM; ++i) {
                    V Av = load<V>(A + lda * (ii + i) + l);
                    for (int64_t j = 0; j < RN; ++j) {
                        Cv[j][i] = madd(Av, Bv[j], Cv[j][i]);
                    }
                }
            }
        }
        for (int64_t j = 0; j < RN; ++j)
            for (int64_t i = 0; i < RM; ++i) {
                C[ldc * (jj + j) + (ii + i)] = hsum_hvx_fast(Cv[j][i]);
            }
    }

    template <int RM, int RN, int BM>
    NOINLINE void gemm(int64_t m, int64_t n, int64_t BN) {
        GGML_ASSERT(m % (RM * BM) == 0);
        const int64_t ytiles = m / (RM * BM);
        const int64_t xtiles = (n + RN -1) / RN;
        const int64_t jj_RN = (xtiles - (xtiles * RN - n));

        const int64_t NB_BN = xtiles < BN ? 1 : (xtiles + BN / 2) / BN;
        const int64_t SIZE_BN = xtiles % NB_BN == 0 ? xtiles / NB_BN : xtiles / NB_BN + 1;
        const int64_t jj_BN = (NB_BN - (NB_BN * SIZE_BN - xtiles));
        const int64_t nb_job = ytiles * NB_BN;

        // static duty-based thread distribution (no threadpool needed)
        int64_t duty = (nb_job + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        if (end > nb_job)
            end = nb_job;

        for (int64_t job = start; job < end; ++job) {
            const int64_t ii = (job % ytiles) * RM * BM;
            const int64_t jb =  job / ytiles;
            const int64_t jr0 = BLOC_POS(jb  , jj_BN, SIZE_BN);
            const int64_t jrN = BLOC_POS(jb+1, jj_BN, SIZE_BN);

            const int64_t jj0 = BLOC_POS(jr0, jj_RN, RN);
            const int64_t jj2 = BLOC_POS(jrN, jj_RN, RN);
            const int64_t jj1 = jj2 < jj_RN * RN ? jj2 : jj_RN * RN;

            for (int64_t bi = 0; bi < BM * RM; bi += RM) {
                int64_t jj = jj0;
                for (; jj < jj1; jj += RN) {
                    gemm_bloc<RM, RN>(ii + bi, jj);
                }
                if constexpr (RN > 1) {
                    for (; jj < jj2; jj += RN - 1) {
                        gemm_bloc<RM, RN-1>(ii + bi, jj);
                    }
                }
                GGML_ASSERT(jj == jj2);
            }
        }
    }

    const int ith;
    const int nth;
    const TA *const A;
    const TB *const B;
    TC *const C;
    const int64_t k;
    const int64_t lda;
    const int64_t ldb;
    const int64_t ldc;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// QUANTIZED MATRIX MULTIPLICATION (HVX)
//
// tinyBLAS_Q0_Fast: based on tinyBLASSpecial/tinyBLASFast pattern
// Uses BLOC_POS tiling instead of ARM mnpack recursive approach.
//
// HVX: Q6_Vw_vrmpy_VbVb processes 32 int8 pairs -> 8 int32 results
//   One call covers the full block -> hsum -> 1 int32
//
// block_q8_0.qs has 32 int8 values (QK8_0 = 32)
// block_q4_0.qs has 16 bytes (32 nibbles, QK4_0 = 32)
// block_q5_0.qs has 16 bytes (32 nibbles, QK5_0 = 32)

#if defined(__HVX__)

// Load 32 bytes from memory into an HVX_Vector (upper 96 bytes zeroed)
static inline HVX_Vector load_32bytes(const void *p) {
    int8_t __attribute__((aligned(128))) buf[128] = {};
    memcpy(buf, p, 32);
    return *(HVX_Vector *)buf;
}

// Reduce first 8 int32 values of an HVX_Vector to a single int32
static inline int32_t hsum_i32_8(HVX_Vector v) {
    int32_t __attribute__((aligned(128))) tmp[32];
    *(HVX_Vector *)tmp = v;
    return tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
}

template <typename TA>
class tinyBLAS_Q0_Fast {
  public:
    tinyBLAS_Q0_Fast(const ggmldsp_compute_params * params, int64_t k,
                     const TA *A, int64_t lda,
                     const block_q8_0 *B, int64_t ldb,
                     float *C, int64_t ldc)
        : ith(params->ith), nth(params->nth), A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc) {
    }

    void matmul(int64_t m, int64_t n) {
#if VECTOR_REGISTERS == 32
        if (m % 16 == 0 && (m/16 >= nth)) {
            const int64_t SIZE_N = SGEMM_BLOCK_SIZE<6>(n);
            mnpack<4, 6, 4>(m, n, SIZE_N, 12);
            return;
        }
        if (m % 8 == 0 ) {
            const int64_t SIZE_N = SGEMM_BLOCK_SIZE<6>(n);
            mnpack<4, 6, 2>(m, n, SIZE_N, 12);
            return;
        }
        if (m % 4 == 0) {
            const int64_t SIZE_N = SGEMM_BLOCK_SIZE<6>(n);
            mnpack<4, 6, 1>(m, n, SIZE_N, 12);
            return;
        }
#else
        if (m % 16 == 0 && (m/16 >= nth)) {
            const int64_t SIZE_N = SGEMM_BLOCK_SIZE<3>(n);
            mnpack<4, 3, 4>(m, n, SIZE_N, 24);
            return;
        }
        if (m % 8 == 0 ) {
            const int64_t SIZE_N = SGEMM_BLOCK_SIZE<3>(n);
            mnpack<4, 3, 2>(m, n, SIZE_N, 24);
            return;
        }
        if (m % 4 == 0) {
            const int64_t SIZE_N = SGEMM_BLOCK_SIZE<3>(n);
            mnpack<4, 3, 1>(m, n, SIZE_N, 24);
            return;
        }
#endif
    }

  private:
    template <int RM, int RN, int BM>
    inline void mnpack(int64_t m, int64_t n, int64_t SIZE_N, int64_t BN) {
        if (SIZE_N == RN) {
            return gemm<RM, RN, BM>(m, n, BN);
        }
        if constexpr (RN > 1) {
            return mnpack<RM, RN-1, BM>(m, n, SIZE_N, BN);
        }
    }

    template <int RM, int RN>
    inline void gemm_bloc(int64_t ii, int64_t jj) {
        float Cv[RN][RM] = {};
        for (int64_t l = 0; l < k; ++l) {
            // Load A blocks first, reuse across all B columns
            float dA[RM];
            HVX_Vector Avec[RM];
            for (int64_t i = 0; i < RM; ++i) {
                dA[i] = unhalf(A[lda * (ii + i) + l].d);
                Avec[i] = load_a(A + lda * (ii + i) + l);
            }
            // Stream B blocks, reuse A blocks
            for (int64_t j = 0; j < RN; ++j) {
                float dB = unhalf(B[ldb * (jj + j) + l].d);
                HVX_Vector bvec = load_q8(B + ldb * (jj + j) + l);
                for (int64_t i = 0; i < RM; ++i) {
                    HVX_Vector rsum = Q6_Vw_vrmpy_VbVb(Avec[i], bvec);
                    int32_t sumi = hsum_i32_8(rsum);
                    Cv[j][i] += dA[i] * dB * (float)sumi;
                }
            }
        }
        for (int64_t j = 0; j < RN; ++j)
            for (int64_t i = 0; i < RM; ++i)
                C[ldc * (jj + j) + (ii + i)] = Cv[j][i];
    }

    template <int RM, int RN, int BM>
    NOINLINE void gemm(int64_t m, int64_t n, int64_t BN) {
        GGML_ASSERT(m % (RM * BM) == 0);
        const int64_t ytiles = m / (RM * BM);
        const int64_t xtiles = (n + RN -1) / RN;
        const int64_t jj_RN = (xtiles - (xtiles * RN - n));

        const int64_t NB_BN = xtiles < BN ? 1 : (xtiles + BN / 2) / BN;
        const int64_t SIZE_BN = xtiles % NB_BN == 0 ? xtiles / NB_BN : xtiles / NB_BN + 1;
        const int64_t jj_BN = (NB_BN - (NB_BN * SIZE_BN - xtiles));
        const int64_t nb_job = ytiles * NB_BN;

        // static duty-based thread distribution
        int64_t duty = (nb_job + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        if (end > nb_job)
            end = nb_job;

        for (int64_t job = start; job < end; ++job) {
            const int64_t ii = (job % ytiles) * RM * BM;
            const int64_t jb =  job / ytiles;
            const int64_t jr0 = BLOC_POS(jb  , jj_BN, SIZE_BN);
            const int64_t jrN = BLOC_POS(jb+1, jj_BN, SIZE_BN);

            const int64_t jj0 = BLOC_POS(jr0, jj_RN, RN);
            const int64_t jj2 = BLOC_POS(jrN, jj_RN, RN);
            const int64_t jj1 = jj2 < jj_RN * RN ? jj2 : jj_RN * RN;

            for (int64_t bi = 0; bi < BM * RM; bi += RM) {
                int64_t jj = jj0;
                for (; jj < jj1; jj += RN) {
                    gemm_bloc<RM, RN>(ii + bi, jj);
                }
                if constexpr (RN > 1) {
                    for (; jj < jj2; jj += RN - 1) {
                        gemm_bloc<RM, RN-1>(ii + bi, jj);
                    }
                }
                GGML_ASSERT(jj == jj2);
            }
        }
    }

    // Load Q8_0 block qs values (32 int8) into HVX_Vector
    inline HVX_Vector load_q8(const block_q8_0 *b) {
        return load_32bytes(b->qs);
    }

    // Load and dequantize A block for different quantized types

    // Q8_0: direct load (already int8)
    inline HVX_Vector load_a(const block_q8_0 *b) {
        return load_32bytes(b->qs);
    }

    // Q4_0: dequantize nibbles to int8 (nibble - 8)
    inline HVX_Vector load_a(const block_q4_0 *b) {
        int8_t __attribute__((aligned(128))) dequant[128] = {};
        const uint8_t *qs = b->qs;
        for (int j = 0; j < 16; ++j) {
            dequant[j]      = (qs[j] & 0x0F) - 8;
            dequant[j + 16] = (qs[j] >> 4)   - 8;
        }
        return *(HVX_Vector *)dequant;
    }

    // Q5_0: dequantize nibbles + qh bits to int8
    inline HVX_Vector load_a(const block_q5_0 *b) {
        int8_t __attribute__((aligned(128))) dequant[128] = {};
        const uint8_t *qs = b->qs;
        uint32_t qh;
        memcpy(&qh, b->qh, sizeof(qh));
        for (int j = 0; j < 16; ++j) {
            int8_t lo = (qs[j] & 0x0F) - 16;
            int8_t hi = (qs[j] >> 4)   - 16;
            lo += (qh >> j)        & 1 ? 16 : 0;
            hi += (qh >> (j + 16)) & 1 ? 16 : 0;
            dequant[j]      = lo;
            dequant[j + 16] = hi;
        }
        return *(HVX_Vector *)dequant;
    }

    const int ith;
    const int nth;
    const TA *const A;
    const block_q8_0 *const B;
    float *const C;
    const int64_t k;
    const int64_t lda;
    const int64_t ldb;
    const int64_t ldc;
};

#endif // __HVX__

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////
// Main entry point

bool ggmldsp_llamafile_sgemm(const struct ggmldsp_compute_params * params, struct sgemm_params * param) {
    int64_t m      = param->m;
    int64_t n      = param->n;
    int64_t k      = param->k;
    const void * A = param->A;
    int64_t lda    = param->lda;
    const void * B = param->B;
    int64_t ldb    = param->ldb;
    void * C       = param->C;
    int64_t ldc    = param->ldc;
    int Atype      = param->Atype;
    int Btype      = param->Btype;
    int Ctype      = param->Ctype;

    assert(m >= 0);
    assert(n >= 0);
    assert(k >= 0);

    if (n < 2)
        return false;

    if (Ctype != GGML_TYPE_F32)
        return false;

    switch (Atype) {

    case GGML_TYPE_F32: {
        if (Btype != GGML_TYPE_F32)
            return false;
#if defined(__HVX__)
        tinyBLAS_Fast<32, HVX_Vector, HVX_Vector, float, float, float> tb{ params,
            k, (const float *)A, lda,
            (const float *)B, ldb,
            (float *)C, ldc};
        return tb.matmul(m, n);
#else
        return false;
#endif
    }

    case GGML_TYPE_Q8_0: {
        if (Btype != GGML_TYPE_Q8_0)
           return false;
#if defined(__HVX__)
        tinyBLAS_Q0_Fast<block_q8_0> tb{ params,
            k, (const block_q8_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc};
        tb.matmul(m, n);
        return true;
#else
        return false;
#endif
    }

    case GGML_TYPE_Q4_0: {
        if (Btype != GGML_TYPE_Q8_0)
            return false;
#if defined(__HVX__)
        tinyBLAS_Q0_Fast<block_q4_0> tb{ params,
            k, (const block_q4_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc};
        tb.matmul(m, n);
        return true;
#else
        return false;
#endif
    }

    case GGML_TYPE_Q5_0: {
        if (Btype != GGML_TYPE_Q8_0)
            return false;
#if defined(__HVX__)
        tinyBLAS_Q0_Fast<block_q5_0> tb{ params,
            k, (const block_q5_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc};
        tb.matmul(m, n);
        return true;
#else
        return false;
#endif
    }

    default:
        return false;
    }
}
