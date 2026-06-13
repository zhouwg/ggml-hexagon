#include <hexagon_types.h>
#include <HAP_power.h>
#include <HAP_dcvs.h>
#include <HAP_mem.h>
#include <HAP_compute_res.h>
#include <assert.h>
#include "ggml-dsp.h"

// HVX memory access macros (matching htp/hvx-base.h)
#define hvx_vmem(A)   *((HVX_Vector *)(A))
#define hvx_vmemu(A)  *((HVX_UVector *)(A))

// HMX tile constants
#define HMX_FP16_TILE_N_ROWS  32
#define HMX_FP16_TILE_N_COLS  32
#define HMX_FP16_TILE_N_ELMS  (HMX_FP16_TILE_N_ROWS * HMX_FP16_TILE_N_COLS)  // 1024
#define HMX_FP16_TILE_SIZE    (HMX_FP16_TILE_N_ELMS * sizeof(__fp16))          // 2048

// --- HMX helper functions from htp/hmx-utils.h ---

// Initialise aligned 256-byte area with scale vector + zero padding.
static inline void hmx_init_column_scales(void *out_scales, HVX_Vector v_scale) {
    volatile HVX_Vector *pv = (HVX_Vector *) out_scales;
    pv[0] = v_scale;
    pv[1] = Q6_V_vzero();
}

// vscatter offsets for fused dequant+transpose: write K-values directly to [K][N] tile.
// word[i] = i*128 maps K-row-pair i to byte offset i*128.
// Column offset (n*4) is added at runtime.  Entries 0..15 cover one tile (region 2047);
// entries 16..31 cover the next adjacent tile (region 4095) — pick region size at the
// call site to scatter into one tile (masked) or two contiguous tiles (unmasked).
static const int32_t hmx_transpose_scatter_offsets[32] __attribute__((aligned(VLEN))) = {
    0 * 128,  1 * 128,  2 * 128,  3 * 128,  4 * 128,  5 * 128,  6 * 128,  7 * 128,  8 * 128,  9 * 128,  10 * 128,
    11 * 128, 12 * 128, 13 * 128, 14 * 128, 15 * 128, 16 * 128, 17 * 128, 18 * 128, 19 * 128, 20 * 128, 21 * 128,
    22 * 128, 23 * 128, 24 * 128, 25 * 128, 26 * 128, 27 * 128, 28 * 128, 29 * 128, 30 * 128, 31 * 128,
};

// Scatter row-major FP16 data (in VTCM scratch) into transposed [K][N] tiles.
// vtcm_src: [n_cols][src_stride] row-major fp16 (only first k elements per row are used)
// vtcm_dst: [n_col_tiles][n_k_tiles][HMX_FP16_TILE_N_ELMS] tile-major interleaved fp16
// Processes rows [start_row, end_row) for multi-thread slicing.
// Full range: start_row=0, end_row=n_cols.
static inline void hmx_interleave_rows_to_tiles(__fp16 * restrict vtcm_dst,
                                            const __fp16 * restrict vtcm_src,
                                            int n_cols,
                                            int k,
                                            int src_stride,
                                            int start_row,
                                            int end_row) {
    assert(k % HMX_FP16_TILE_N_COLS == 0);

    const int            n_k_tiles     = k / HMX_FP16_TILE_N_COLS;
    const HVX_Vector     v_scat_base   = hvx_vmem(hmx_transpose_scatter_offsets);
    const HVX_Vector     v_scat_step   = Q6_V_vsplat_R(4);
    const HVX_VectorPred q_mask64      = Q6_Q_vsetq_R(64);
    // Each hvx_vmemu load brings 64 fp16 = 128 bytes covering 2 adjacent K-tiles.
    // When n_k_tiles is even, scatter into 2 K-tiles per call (region 4095, no mask)
    // using the upper half of hmx_transpose_scatter_offsets.  Tail one K-tile (when
    // n_k_tiles is odd) falls back to single-tile masked scatter.
    const bool           pair_scatter  = (n_k_tiles & 1) == 0;
    const size_t         pair_region   = (size_t) (2 * HMX_FP16_TILE_SIZE - 1);
    const size_t         single_region = (size_t) (HMX_FP16_TILE_SIZE - 1);
    __builtin_assume(k > 0);
    __builtin_assume(end_row > start_row);

    if (pair_scatter) {
        // Step c by 64 fp16 (two K-tiles per scatter), advance dst by 2 tiles per iter.
        const int    c_step      = 2 * HMX_FP16_TILE_N_COLS;
        const size_t c_byte_step = (size_t) c_step * sizeof(__fp16);
        const size_t dst_step    = 2 * (size_t) HMX_FP16_TILE_N_ELMS;
        const int    n_c_iters   = k / c_step;

        for (int r = start_row; r < end_row; r += 2) {
            const int        ct             = r / HMX_FP16_TILE_N_ROWS;
            const int        local_r        = r % HMX_FP16_TILE_N_ROWS;
            const bool       next_row_valid = (r + 1) < end_row && (r + 1) < n_cols;
            const HVX_Vector v_off0         = Q6_Vw_vadd_VwVw(v_scat_base, Q6_V_vsplat_R(local_r * 4));
            const HVX_Vector v_off1         = Q6_Vw_vadd_VwVw(v_off0, v_scat_step);

            __fp16 * tile_base = vtcm_dst + (size_t) ct * n_k_tiles * HMX_FP16_TILE_N_ELMS;
            const uint8_t * p0 = (const uint8_t *) (vtcm_src + r * src_stride);
            const uint8_t * p1 = next_row_valid ? (const uint8_t *) (vtcm_src + (r + 1) * src_stride) : NULL;

            assert(hex_is_aligned(p0, 128));
            assert(hex_is_aligned(p1, 128));
            assert(c_byte_step % 128 == 0);

            if (p1) {
                for (int i = 0; i < n_c_iters; ++i) {
                    HVX_Vector v0 = hvx_vmem(p0); p0 += c_byte_step;
                    HVX_Vector v1 = hvx_vmem(p1); p1 += c_byte_step;
                    Q6_vscatter_RMVwV((size_t) tile_base, pair_region, v_off0, v0);
                    Q6_vscatter_RMVwV((size_t) tile_base, pair_region, v_off1, v1);
                    tile_base += dst_step;
                }
            } else {
                const HVX_Vector vzero = Q6_V_vzero();
                for (int i = 0; i < n_c_iters; ++i) {
                    HVX_Vector v0 = hvx_vmem(p0); p0 += c_byte_step;
                    Q6_vscatter_RMVwV((size_t) tile_base, pair_region, v_off0, v0);
                    Q6_vscatter_RMVwV((size_t) tile_base, pair_region, v_off1, vzero);
                    tile_base += dst_step;
                }
            }
        }
    } else {
        // Fallback: scatter one K-tile per call (region 2047, masked).
        const int    c_step      = HMX_FP16_TILE_N_COLS;
        const size_t c_byte_step = (size_t) c_step * sizeof(__fp16);
        const size_t dst_step    = (size_t) HMX_FP16_TILE_N_ELMS;
        const int    n_c_iters   = k / c_step;

        for (int r = start_row; r < end_row; r += 2) {
            const int        ct             = r / HMX_FP16_TILE_N_ROWS;
            const int        local_r        = r % HMX_FP16_TILE_N_ROWS;
            const bool       next_row_valid = (r + 1) < end_row && (r + 1) < n_cols;
            const HVX_Vector v_off0         = Q6_Vw_vadd_VwVw(v_scat_base, Q6_V_vsplat_R(local_r * 4));
            const HVX_Vector v_off1         = Q6_Vw_vadd_VwVw(v_off0, v_scat_step);

            __fp16 * tile_base = vtcm_dst + (size_t) ct * n_k_tiles * HMX_FP16_TILE_N_ELMS;
            const uint8_t * p0 = (const uint8_t *) (vtcm_src + r * src_stride);
            const uint8_t * p1 = next_row_valid ? (const uint8_t *) (vtcm_src + (r + 1) * src_stride) : NULL;

            if (p1) {
                for (int i = 0; i < n_c_iters; ++i) {
                    HVX_Vector v0 = hvx_vmemu(p0); p0 += c_byte_step;
                    HVX_Vector v1 = hvx_vmemu(p1); p1 += c_byte_step;
                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_base, single_region, v_off0, v0);
                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_base, single_region, v_off1, v1);
                    tile_base += dst_step;
                }
            } else {
                const HVX_Vector vzero = Q6_V_vzero();
                for (int i = 0; i < n_c_iters; ++i) {
                    HVX_Vector v0 = hvx_vmemu(p0); p0 += c_byte_step;
                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_base, single_region, v_off0, v0);
                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_base, single_region, v_off1, vzero);
                    tile_base += dst_step;
                }
            }
        }
    }
}

static int g_thread_counts                  = 1;

// Forward declarations for mulmat.c functions
// HVX memory access macros (matching htp/hvx-base.h)
void transfer_activation_chunk_fp32_to_fp16(__fp16 *restrict vtcm_dst, const float *restrict src,
                                            int n_rows, int k, int row_stride);
void convert_weight_f32_to_fp16_tiles(__fp16 *restrict vtcm_dst, const float *restrict src,
                                      int n_cols, int k, int row_stride);
void core_dot_chunk_fp16(__fp16 *restrict output, const __fp16 *restrict activation,
                        const __fp16 *restrict weights, const __fp16 *restrict scales,
                        int n_row_tiles, int n_col_tiles, int n_dot_tiles);
void transfer_output_chunk_fp16_to_fp32(float *restrict dst, const __fp16 *restrict src,
                                         int n_rows, int n_cols, int row_stride);

// Test HMX function - inline HMX implementation matching official code
int ggmlop_dsp_test_hmx(remote_handle64 h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) {
    GGMLHEXAGON_LOG_INFO("==============enter %s (inline HMX)===========\n", __func__);

    // Performance measurement
    char tempbuf[256];
    ggmlhexagon_get_opkey(GGML_OP_MUL_MAT, src0, src1, tempbuf, 256);
    int64_t begin_time = ggml_time_us();

    if (!src0 || !src1 || !dst) {
        GGMLHEXAGON_LOG_ERROR("invalid input tensors");
        return AEE_EBADPARM;
    }

    const int32_t K = src0->ne[0];
    const int32_t M = src0->ne[1];
    const int32_t N = src1->ne[1];

    GGMLHEXAGON_LOG_INFO("src0: ne[0]=%d (K), ne[1]=%d (M)\n", K, M);
    GGMLHEXAGON_LOG_INFO("src1: ne[0]=%d (K), ne[1]=%d (N)\n", K, N);

    // Set dst dimensions
    dst->ne[0] = N;
    dst->ne[1] = M;
    dst->nb[0] = sizeof(float);
    dst->nb[1] = dst->nb[0] * dst->ne[0];

    // Get VTCM
    size_t vtcm_size = 0;
    void * vtcm_base = ggmlop_get_vtcm_pool(&vtcm_size);
    if (vtcm_base == NULL) {
        GGMLHEXAGON_LOG_ERROR("No VTCM pool available\n");
        return AEE_EFAILED;
    }

    GGMLHEXAGON_LOG_INFO("VTCM pool: base=%p, size=%zu bytes\n", vtcm_base, vtcm_size);

    // Check alignment (HMX requires 32-aligned dimensions)
    if (K % 32 != 0 || N % 32 != 0 || M % 32 != 0) {
        GGMLHEXAGON_LOG_ERROR("HMX dimensions must be 32-aligned: M=%d, N=%d, K=%d\n", M, N, K);
        return AEE_EBADPARM;
    }

    // Calculate tile counts
    const int n_row_tiles = M / 32;
    const int n_col_tiles = N / 32;
    const int n_dot_tiles = K / 32;

    // VTCM layout - matching official implementation
    const size_t vec_dot_size = K * sizeof(__fp16);
    const size_t act_area_size = ((M * vec_dot_size + 2047) & ~2047);
    const size_t weight_area_size = ((N * vec_dot_size + 2047) & ~2047);
    const size_t output_area_size = ((M * N * sizeof(__fp16) + 2047) & ~2047);
    const size_t scales_size = 256;
    // Temp buffer needs to hold fp32 data for both activation (M*K) and weight (N*K)
    // Use max(M*K, N*K) * sizeof(float) for each, total 2 * max * sizeof(float)
    const size_t max_rows = (M > N) ? M : N;
    const size_t temp_buf_size = ((max_rows * K * sizeof(float) + 2047) & ~2047);  // Temp for fp32->fp16 conversion

    GGMLHEXAGON_LOG_INFO("VTCM sizes: act=%zu, weight=%zu, output=%zu, scales=%zu, temp=%zu\n",
                         act_area_size, weight_area_size, output_area_size, scales_size, temp_buf_size);

    // Allocate VTCM regions
    uint8_t *vtcm_ptr = (uint8_t *)vtcm_base;
    __fp16 *vtcm_activation = (__fp16 *)vtcm_ptr;
    vtcm_ptr += act_area_size;
    __fp16 *vtcm_weight = (__fp16 *)vtcm_ptr;
    vtcm_ptr += weight_area_size;
    __fp16 *vtcm_output = (__fp16 *)vtcm_ptr;
    vtcm_ptr += output_area_size;

    // Align vtcm_temp to 128 bytes for HVX vector access (hmx_interleave_cols_to_tiles uses HVX_Vector*)
    uintptr_t temp_addr = ((uintptr_t)vtcm_ptr + 127) & ~127;
    vtcm_ptr = (uint8_t *)temp_addr;
    __fp16 *vtcm_temp = (__fp16 *)vtcm_ptr;  // Temp buffer for fp32->fp16 conversion
    vtcm_ptr += temp_buf_size;

    // Align scales to 256 bytes
    uintptr_t scales_addr = ((uintptr_t)vtcm_ptr + 255) & ~255;
    __fp16 *vtcm_scales = (__fp16 *)scales_addr;

    GGMLHEXAGON_LOG_INFO("VTCM layout: act=%p, weight=%p, output=%p, temp=%p (aligned), scales=%p\n",
                         vtcm_activation, vtcm_weight, vtcm_output, vtcm_temp, vtcm_scales);

    // Acquire HMX lock
    unsigned int compute_res_ctx_id = ggmlop_get_compute_res_ctx_id();
    GGMLHEXAGON_LOG_INFO("compute_res_ctx_id=%u\n", compute_res_ctx_id);
    
    int lock_result = HAP_compute_res_hmx_lock(compute_res_ctx_id);
    GGMLHEXAGON_LOG_INFO("HMX lock result: %d\n", lock_result);
    if (lock_result != 0) {
        GGMLHEXAGON_LOG_ERROR("HMX lock failed\n");
        return AEE_EFAILED;
    }

    // Initialize scales (scale=1.0, bias=0.0) using OFFICIAL function
    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0

    // Verify scales initialization
    GGMLHEXAGON_LOG_INFO("DEBUG: scales[0]=%f (scale), scales[64]=%f (bias)\n",
                         ggml_compute_fp16_to_fp32(vtcm_scales[0]),
                         ggml_compute_fp16_to_fp32(vtcm_scales[64]));

    // Memory barrier to ensure VTCM writes are visible before HMX access
    __asm__ __volatile__("" ::: "memory");

    // ====== STEP 1: Transfer activation ======
    // Use scalar reads from DDR, then scalar conversion to fp16 tiles
    const float *src0_data = (const float *)src0->data;
    const size_t src0_stride = src0->nb[1] / sizeof(float);
    const float *src1_data = (const float *)src1->data;
    const size_t src1_stride = src1->nb[1] / sizeof(float);

    GGMLHEXAGON_LOG_INFO("Transferring activation: M=%d, K=%d, stride=%zu, first_val=%f\n",
                         M, K, src0_stride, src0_data[0]);

    // Transfer activation using mulmat.c function (matches core_dot_chunk_fp16 expectations)
    transfer_activation_chunk_fp32_to_fp16(vtcm_activation, src0_data, M, K, src0_stride);

    // Transfer weight using OFFICIAL hmx_interleave_rows_to_tiles function
    // First convert fp32 to fp16, then use official function for scatter format
    for (int i = 0; i < N * K; i++) {
        ((__fp16 *)vtcm_temp)[i] = ggml_compute_fp32_to_fp16(src1_data[i]);
    }
    
    // hmx_interleave_rows_to_tiles parameters:
    // vtcm_dst: output tile array
    // vtcm_src: input fp16 data (row-major)
    // n_cols: number of columns (N)
    // k: number of rows (K)
    // src_stride: stride in fp16 elements (K)
    // start_row, end_row: row range to process
    hmx_interleave_rows_to_tiles(vtcm_weight, vtcm_temp, N, K, K, 0, N);

    // Debug: check weight data layout after OFFICIAL interleave
    GGMLHEXAGON_LOG_INFO("DEBUG: After OFFICIAL weight interleave: weight[0..3]=%f,%f,%f,%f\n",
                         ggml_compute_fp16_to_fp32(vtcm_weight[0]),
                         ggml_compute_fp16_to_fp32(vtcm_weight[1]),
                         ggml_compute_fp16_to_fp32(vtcm_weight[2]),
                         ggml_compute_fp16_to_fp32(vtcm_weight[3]));

    // Initialize output
    memset(vtcm_output, 0, output_area_size);

    // Memory barrier after data transfers to ensure VTCM coherency before HMX
    __asm__ __volatile__("" ::: "memory");

    // Execute HMX matrix multiplication using official core_dot_chunk_fp16 from mulmat.c
    GGMLHEXAGON_LOG_INFO("Performing HMX matrix multiplication: row_tiles=%d, col_tiles=%d, dot_tiles=%d\n",
                         n_row_tiles, n_col_tiles, n_dot_tiles);

    GGMLHEXAGON_LOG_INFO("VTCM addresses: act=%p, weight=%p, output=%p, scales=%p\n",
                         vtcm_activation, vtcm_weight, vtcm_output, vtcm_scales);

    // Verify data before HMX
    GGMLHEXAGON_LOG_INFO("Before HMX: act[0]=%f, weight[0]=%f, output[0]=%f\n",
                         ggml_compute_fp16_to_fp32(vtcm_activation[0]),
                         ggml_compute_fp16_to_fp32(vtcm_weight[0]),
                         ggml_compute_fp16_to_fp32(vtcm_output[0]));

    core_dot_chunk_fp16(vtcm_output, vtcm_activation, vtcm_weight, vtcm_scales,
                        n_row_tiles, n_col_tiles, n_dot_tiles);

    // Memory barrier
    __asm__ __volatile__("" ::: "memory");

    // Check result after HMX
    GGMLHEXAGON_LOG_INFO("After HMX: output[0]=%f, output[1]=%f, output[2]=%f, output[3]=%f\n",
                         ggml_compute_fp16_to_fp32(vtcm_output[0]),
                         ggml_compute_fp16_to_fp32(vtcm_output[1]),
                         ggml_compute_fp16_to_fp32(vtcm_output[2]),
                         ggml_compute_fp16_to_fp32(vtcm_output[3]));

    // ====== STEP 4: Transfer result back using mulmat.c function ======
    float *dst_data = (float *)dst->data;
    const size_t dst_stride = dst->nb[1] / sizeof(float);

    transfer_output_chunk_fp16_to_fp32(dst_data, vtcm_output, M, N, dst_stride);

    // Release HMX lock
    HAP_compute_res_hmx_unlock(compute_res_ctx_id);

    // Verify
    float expected = (float)K * 2.0f;
    GGMLHEXAGON_LOG_INFO("HMX test: expected=%f, result[0]=%f\n", expected, dst_data[0]);

    // Performance measurement
    int64_t end_time = ggml_time_us();
    GGMLHEXAGON_LOG_INFO("elapse time of %s is %lld us", tempbuf, (long long)(end_time - begin_time));

    GGMLHEXAGON_LOG_INFO("==============leave %s===========\n", __func__);
    return AEE_SUCCESS;
}
