#pragma once
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * llamafile-style sgemm for Hexagon DSP
 *
 * C = A^T * B
 *
 * @param m is rows in A and C
 * @param n is cols in B and C
 * @param k is cols in A and rows in B (in blocks)
 * @param A is first input matrix (always transposed)
 * @param lda is row stride of A (in blocks)
 * @param B is second input matrix (never transposed)
 * @param ldb is row stride of B (in blocks)
 * @param C is output matrix
 * @param ldc is row stride of C
 * @param ith is thread id
 * @param nth is number of threads
 * @param Atype is GGML data type of A
 * @param Btype is GGML data type of B
 * @param Ctype is GGML data type of C
 * @return true if this function was able to service the matmul request
 */

struct sgemm_params {
    int64_t m;
    int64_t n;
    int64_t k;
    const void * A;
    int64_t lda;
    const void * B;
    int64_t ldb;
    void * C;
    int64_t ldc;
    int Atype;
    int Btype;
    int Ctype;
};

struct ggmldsp_compute_params {
    int ith;
    int nth;
};

bool ggmldsp_llamafile_sgemm(const struct ggmldsp_compute_params * params, struct sgemm_params * s_params);

#ifdef __cplusplus
}
#endif
