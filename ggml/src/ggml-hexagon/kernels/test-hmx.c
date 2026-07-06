#include <hexagon_types.h>
#include <HAP_power.h>
#include <HAP_dcvs.h>
#include <HAP_mem.h>
#include <HAP_compute_res.h>
#include <assert.h>
#include "ggml-dsp.h"
#include "../htp/hvx-base.h"  // for official hvx_vec_f32_to_f16 with vdeal

// HMX tile constants
#define HMX_FP16_TILE_N_ROWS  32
#define HMX_FP16_TILE_N_COLS  32
#define HMX_FP16_TILE_N_ELMS  (HMX_FP16_TILE_N_ROWS * HMX_FP16_TILE_N_COLS)  // 1024
#define HMX_FP16_TILE_SIZE    (HMX_FP16_TILE_N_ELMS * sizeof(__fp16))          // 2048

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

// Initialise aligned 256-byte area with scale vector + zero padding.
// Reference: htp/hmx-utils.h hmx_init_column_scales
static inline void hmx_init_column_scales(void *out_scales, HVX_Vector v_scale) {
    volatile HVX_Vector *pv = (volatile HVX_Vector *) out_scales;
    pv[0] = v_scale;  // 128 bytes (64 fp16 elements) - scale
    pv[1] = Q6_V_vzero();  // 128 bytes (64 fp16 elements) - bias
}

// Transfer activation chunk from fp32 to fp16 tiles using scalar approach
// Uses FP16 Crouton layout (separated format for activation)
// Reference: htp/hmx-matmul-ops.c transfer_activation_chunk_fp32_to_fp16
//
// Activation matrix input: [M][K] row-major format
// HMX tile output: [M_row_tiles][K_tiles] with FP16 Crouton layout
//
// FP16 Crouton layout for activation (separated format):
// - Each tile is 32x32 fp16 elements (2048 bytes)
// - Organized as 16 row pairs, each pair has 64 fp16
// - Within each row pair: separated format
// - tile[(r1/2) * 64 + i] = row0 data (first 32 fp16)
// - tile[(r1/2) * 64 + 32 + i] = row1 data (last 32 fp16)
static void transfer_activation_chunk_fp32_to_fp16_scalar(__fp16 *restrict vtcm_dst, const float *restrict src,
                                                          int n_rows, int k, int row_stride) {
    const int n_rows_padded = ((n_rows + HMX_FP16_TILE_N_ROWS - 1) / HMX_FP16_TILE_N_ROWS) * HMX_FP16_TILE_N_ROWS;
    const int n_tiles_per_row = k / HMX_FP16_TILE_N_COLS;

    // Process all rows (including padded)
    for (int r = 0; r < n_rows_padded; r += 2) {
        int r0 = r / HMX_FP16_TILE_N_ROWS;  // tile row index
        int r1 = r % HMX_FP16_TILE_N_ROWS;  // intra-tile row idx

        const float *src_row0 = (r < n_rows) ? src + (r + 0) * row_stride : NULL;
        const float *src_row1 = (r + 1 < n_rows) ? src + (r + 1) * row_stride : NULL;

        for (int c = 0; c < k; c += HMX_FP16_TILE_N_COLS) {
            int c0 = c / HMX_FP16_TILE_N_COLS;  // tile column index
            int tile_idx = r0 * n_tiles_per_row + c0;

            __fp16 *tile_base = vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS;

            // FP16 Crouton layout (separated format):
            // Each row pair position (r1/2) holds 64 fp16 elements:
            // - First 32 fp16: row0 data
            // - Last 32 fp16: row1 data
            for (int i = 0; i < HMX_FP16_TILE_N_COLS; ++i) {
                tile_base[(r1 / 2) * 64 + i] =
                    (src_row0) ? (__fp16)src_row0[c + i] : (__fp16)0;
            }
            for (int i = 0; i < HMX_FP16_TILE_N_COLS; ++i) {
                tile_base[(r1 / 2) * 64 + 32 + i] =
                    (src_row1) ? (__fp16)src_row1[c + i] : (__fp16)0;
            }
        }
    }

    // Memory barrier to ensure data is written to VTCM
    __asm__ __volatile__("" ::: "memory");
}

// Convert weight chunk from fp32 to fp16 tiles using HVX (matching official implementation)
// Uses HVX vscatter and hvx_vec_f32_to_f16 with vdeal shuffle
static void convert_weight_f32_to_fp16_tiles_hvx(__fp16 *restrict vtcm_dst, const float *restrict vtcm_src,
                                                  int n_cols, int k, int row_stride) {
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int n_col_tiles = n_cols / HMX_FP16_TILE_N_COLS;
    const int n_tot_tiles = n_col_tiles * k_tiles;

    GGMLHEXAGON_LOG_INFO("HVX weight conversion: n_cols=%d, k=%d, k_tiles=%d, n_col_tiles=%d, n_tot_tiles=%d, row_stride=%d",
                         n_cols, k, k_tiles, n_col_tiles, n_tot_tiles, row_stride);

    const HVX_Vector v_scat_base = hvx_vmem(hmx_transpose_scatter_offsets);
    const HVX_Vector v_scat_step = Q6_V_vsplat_R(4);
    const HVX_VectorPred q_mask64 = Q6_Q_vsetq_R(64);

    for (int t = 0; t < n_tot_tiles; ++t) {
        int ct = t / k_tiles;  // N tile index
        int kt = t % k_tiles;  // K tile index

        __fp16 *tile_base = vtcm_dst + t * HMX_FP16_TILE_N_ELMS;
        int byte_off = kt * 32 * sizeof(float);

        GGMLHEXAGON_LOG_INFO("HVX weight tile %d: ct=%d, kt=%d, tile_base=%p, byte_off=%d",
                             t, ct, kt, tile_base, byte_off);

        HVX_Vector v_off = v_scat_base;
        for (int r = 0; r < HMX_FP16_TILE_N_ROWS; r += 2) {
            int row0 = ct * HMX_FP16_TILE_N_COLS + r;
            int row1 = row0 + 1;

            const uint8_t *r0 = (const uint8_t *)vtcm_src + row0 * row_stride;
            const uint8_t *r1 = (const uint8_t *)vtcm_src + row1 * row_stride;

            GGMLHEXAGON_LOG_INFO("HVX weight tile %d, row pair %d: row0=%d, row1=%d, r0=%p, r1=%p",
                                 t, r/2, row0, row1, r0, r1);

            HVX_Vector v0_f32 = hvx_vmemu((const float *)(r0 + byte_off));
            HVX_Vector v1_f32 = (row1 < n_cols) ? hvx_vmemu((const float *)(r1 + byte_off)) : Q6_V_vzero();

            // DEBUG: Check HVX read data
            float *v0_ptr = (float *)&v0_f32;
            float *v1_ptr = (float *)&v1_f32;
            GGMLHEXAGON_LOG_INFO("HVX weight tile %d, row pair %d: v0_f32[0..3]=%f,%f,%f,%f, v1_f32[0..3]=%f,%f,%f,%f",
                                 t, r/2, v0_ptr[0], v0_ptr[1], v0_ptr[2], v0_ptr[3],
                                 v1_ptr[0], v1_ptr[1], v1_ptr[2], v1_ptr[3]);

            // Use hvx_vec_f32_to_f16 with vdeal shuffle (matching official implementation)
            HVX_Vector v_out = hvx_vec_f32_to_f16(v0_f32, v1_f32);

            // DEBUG: Check HVX conversion result
            __fp16 *v_out_ptr = (__fp16 *)&v_out;
            GGMLHEXAGON_LOG_INFO("HVX weight tile %d, row pair %d: v_out[0..7]=%f,%f,%f,%f,%f,%f,%f,%f",
                                 t, r/2,
                                 (float)v_out_ptr[0], (float)v_out_ptr[1], (float)v_out_ptr[2], (float)v_out_ptr[3],
                                 (float)v_out_ptr[4], (float)v_out_ptr[5], (float)v_out_ptr[6], (float)v_out_ptr[7]);

            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HMX_FP16_TILE_SIZE - 1, v_off, v_out);
            v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);

            HVX_Vector v_out_hi = Q6_V_vror_VR(v_out, 64);
            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HMX_FP16_TILE_SIZE - 1, v_off, v_out_hi);
            v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
        }
        (void) *(volatile HVX_Vector *)(tile_base);

        // DEBUG: Check tile data after vscatter
        GGMLHEXAGON_LOG_INFO("HVX weight tile %d after vscatter: tile[0..7]=%f,%f,%f,%f,%f,%f,%f,%f",
                             t,
                             (float)tile_base[0], (float)tile_base[1], (float)tile_base[2], (float)tile_base[3],
                             (float)tile_base[4], (float)tile_base[5], (float)tile_base[6], (float)tile_base[7]);
    }

    // Final memory barrier and volatile read to ensure all vscatter writes are committed to VTCM
    __asm__ __volatile__("" ::: "memory");
    if (n_tot_tiles > 0) {
        (void) *(volatile HVX_Vector *)(vtcm_dst + (n_tot_tiles - 1) * HMX_FP16_TILE_N_ELMS);
    }
    __asm__ __volatile__("" ::: "memory");
}

// Convert weight chunk from fp32 to fp16 tiles (matching official quantize_f32_weight_to_fp16_tiles_task)
// Uses HVX vscatter to produce the correct tile layout for HMX
// NOTE: This function may fail when src points to RPC shared memory instead of VTCM
static void convert_weight_f32_to_fp16_tiles(__fp16 *restrict vtcm_dst, const float *restrict src,
                                              int n_cols, int k, int row_stride) {
    const int k_tiles = k / HMX_FP16_TILE_N_COLS;
    const int n_col_tiles = n_cols / HMX_FP16_TILE_N_COLS;
    const int n_tot_tiles = n_col_tiles * k_tiles;

    const HVX_Vector v_scat_base = hvx_vmem(hmx_transpose_scatter_offsets);
    const HVX_Vector v_scat_step = Q6_V_vsplat_R(4);
    const HVX_VectorPred q_mask64 = Q6_Q_vsetq_R(64);

    // Use uint8_t* for byte-level offset calculation (matching official code)
    const uint8_t *src_bytes = (const uint8_t *)src;

    for (int t = 0; t < n_tot_tiles; ++t) {
        int ct = t / k_tiles;  // column tile index
        int kt = t % k_tiles;  // k tile index

        __fp16 *tile_base = vtcm_dst + t * HMX_FP16_TILE_N_ELMS;
        int byte_off = kt * 32 * sizeof(float);

        HVX_Vector v_off = v_scat_base;
        for (int r = 0; r < HMX_FP16_TILE_N_ROWS; r += 2) {
            int row0 = ct * HMX_FP16_TILE_N_COLS + r;
            int row1 = row0 + 1;

            // Use byte-level offset (row_stride is already in bytes)
            const uint8_t *r0 = src_bytes + row0 * row_stride;
            const uint8_t *r1 = src_bytes + row1 * row_stride;

            // Read 32 floats from each row using HVX
            HVX_Vector v0_f32 = hvx_vmemu((const float *)(r0 + byte_off));
            HVX_Vector v1_f32 = (row1 < n_cols) ? hvx_vmemu((const float *)(r1 + byte_off)) : Q6_V_vzero();

            // Convert to interleaved fp16
            HVX_Vector v_out = hvx_vec_f32_to_f16(v0_f32, v1_f32);

            // Scatter write using HVX vscatter (same pattern as official code)
            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HMX_FP16_TILE_SIZE - 1, v_off, v_out);
            v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);

            // Second half of the 32-element row pair
            HVX_Vector v_out_hi = Q6_V_vror_VR(v_out, 64);
            Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HMX_FP16_TILE_SIZE - 1, v_off, v_out_hi);
            v_off = Q6_Vw_vadd_VwVw(v_off, v_scat_step);
        }

        // Memory barrier
        (void) *(volatile HVX_Vector *)(tile_base);
    }
}

// Software matrix multiplication for verification
static void software_matmul_fp32(float *restrict output, const float *restrict A,
                                  const float *restrict B, int M, int N, int K,
                                  int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            output[i * ldc + j] = sum;
        }
    }
}

// HMX helper functions (matching official Qualcomm implementation)
// Load activation and weight tiles separately (matching official implementation)
static inline void hmx_load_activation_fp16(const __fp16 *row_tiles, size_t range) {
    Q6_activation_hf_mxmem_RR_deep((unsigned int)row_tiles, range);
}

static inline void hmx_load_weight_fp16(const __fp16 *col_tiles, size_t range) {
    Q6_weight_hf_mxmem_RR((unsigned int)col_tiles, range);
}

// HMX output instruction - using official Qualcomm single instruction
// Reference: Q6_mxmem_AR_after_hf(out_tile, 0) corresponds to mxmem(Rs,Rt):after.hf=acc
static inline void hmx_consume_accumulator_fp16(__fp16 *out) {
    Q6_mxmem_AR_after_hf(out, 0);
}

static inline void hmx_set_output_scales(const void *scales) {
    Q6_bias_mxmem2_A((void *)scales);
}

// Simple HMX instruction test using inline assembly
// This function tests the basic HMX instruction flow: load -> compute -> store
// with manually constructed data to verify the instruction execution path.
static void test_hmx_instruction_flow(void) {
    GGMLHEXAGON_LOG_INFO("=== enter HMX Instruction Flow Test ===");

    // Ensure VTCM resource is available (for cache mode)
    int vtcm_err = ggmlop_ensure_vtcm_available();
    if (vtcm_err != 0) {
        GGMLHEXAGON_LOG_ERROR("Failed to ensure VTCM availability: %d", vtcm_err);
        return;
    }

    // Get VTCM pool
    size_t vtcm_size = 0;
    void *vtcm_base = ggmlop_get_vtcm_pool(&vtcm_size);
    if (vtcm_base == NULL) {
        GGMLHEXAGON_LOG_ERROR("No VTCM pool available");
        return;
    }

    // Check if we have enough VTCM for the test (2048*3 + 256 = 6400 bytes)
    if (vtcm_size < 6400) {
        GGMLHEXAGON_LOG_ERROR("VTCM insufficient for test: needed=6400, available=%zu", vtcm_size);
        return;
    }

    // Allocate aligned buffers from VTCM pool
    uint8_t *vtcm_ptr = (uint8_t *)vtcm_base;
    __fp16 *activation_tile = (__fp16 *)vtcm_ptr;
    vtcm_ptr += 2048;
    __fp16 *weight_tile = (__fp16 *)vtcm_ptr;
    vtcm_ptr += 2048;
    __fp16 *output_tile = (__fp16 *)vtcm_ptr;
    vtcm_ptr += 2048;
    // Align scales to 256 bytes
    uintptr_t scales_addr = ((uintptr_t)vtcm_ptr + 255) & ~255;
    __fp16 *scales = (__fp16 *)scales_addr;

    GGMLHEXAGON_LOG_INFO("VTCM buffers: act=%p, weight=%p, output=%p, scales=%p",
                         activation_tile, weight_tile, output_tile, scales);

    // Initialize activation tile (32x32 fp16, all 1.0)
    // Using separated format: first 32 fp16 = row0, next 32 fp16 = row1, etc.
    for (int i = 0; i < 16; i++) {  // 16 row pairs
        for (int j = 0; j < 32; j++) {  // 32 columns per row
            // Row 0 of pair i
            activation_tile[i * 64 + j] = (__fp16)1.0f;
            // Row 1 of pair i
            activation_tile[i * 64 + 32 + j] = (__fp16)1.0f;
        }
    }

    // Initialize weight tile (32x32 fp16, all 1.0)
    // Using interleaved format: even indices from row0, odd indices from row1
    for (int i = 0; i < 32; i++) {  // 32 rows
        for (int j = 0; j < 32; j++) {  // 32 columns
            // Interleaved: tile[(i/2)*64 + j*2 + (i%2)]
            weight_tile[(i / 2) * 64 + j * 2 + (i % 2)] = (__fp16)1.0f;
        }
    }

    // Initialize scales: scale=1.0 (fp16 0x3c00), bias=0.0
    // Using volatile HVX Vector write (matching official implementation)
    volatile HVX_Vector *pv = (volatile HVX_Vector *)scales;
    HVX_Vector v_scale = Q6_V_vsplat_R(0x3c00);  // fp16 1.0
    pv[0] = v_scale;  // 128 bytes (64 fp16 elements) - scale
    pv[1] = Q6_V_vzero();  // 128 bytes (64 fp16 elements) - bias

    // Memory barrier
    __asm__ __volatile__("" ::: "memory");

    // Verify initialization
    GGMLHEXAGON_LOG_INFO("Activation tile[0..3]=%f,%f,%f,%f",
                         (float)activation_tile[0], (float)activation_tile[1],
                         (float)activation_tile[2], (float)activation_tile[3]);
    GGMLHEXAGON_LOG_INFO("Weight tile[0..3]=%f,%f,%f,%f",
                         (float)weight_tile[0], (float)weight_tile[1],
                         (float)weight_tile[2], (float)weight_tile[3]);
    GGMLHEXAGON_LOG_INFO("Scales[0]=%f, Scales[64]=%f",
                         (float)scales[0], (float)scales[64]);

    // Clear output tile
    for (int i = 0; i < 1024; i++) {
        output_tile[i] = (__fp16)0.0f;
    }
    __asm__ __volatile__("" ::: "memory");

    // Acquire HMX lock
    unsigned int compute_res_ctx_id = g_dsp_ctx->compute_res_ctx_id;
    GGMLHEXAGON_LOG_INFO("compute_res_ctx_id=%u", compute_res_ctx_id);

    int lock_result = HAP_compute_res_hmx_lock(compute_res_ctx_id);
    GGMLHEXAGON_LOG_INFO("HMX lock result: %d", lock_result);
    if (lock_result != 0) {
        GGMLHEXAGON_LOG_ERROR("HMX lock failed (%d)", lock_result);
        return;
    }

    GGMLHEXAGON_LOG_INFO("Starting HMX computation...");
    int64_t begin_time = ggml_time_us();

    // Execute HMX computation using Q6 intrinsics
    // Step 1: Set scales (must be done before tile loading)
    Q6_bias_mxmem2_A((void *)scales);

    // Step 2: Clear accumulator
    Q6_mxclracc_hf();

    // Step 3: Load activation tile (range = 2048 * 1 - 1 = 2047)
    Q6_activation_hf_mxmem_RR_deep((unsigned int)activation_tile, 2047);

    // Step 4: Load weight tile (range = 2047)
    Q6_weight_hf_mxmem_RR((unsigned int)weight_tile, 2047);

    // Step 5: Store result to output tile
    Q6_mxmem_AR_after_hf(output_tile, 0);

    // Memory barrier
    __asm__ __volatile__("" ::: "memory");

    int64_t end_time = ggml_time_us();
    GGMLHEXAGON_LOG_INFO("elapse time of HMX computation is %lld us",  (long long)(end_time - begin_time));
    GGMLHEXAGON_LOG_INFO("HMX computation completed");

    // Release HMX lock
    HAP_compute_res_hmx_unlock(compute_res_ctx_id);
    GGMLHEXAGON_LOG_INFO("HMX unlocked");

    // Read back and print results
    GGMLHEXAGON_LOG_INFO("Output tile[0..7]=%f,%f,%f,%f,%f,%f,%f,%f",
                         (float)output_tile[0], (float)output_tile[1],
                         (float)output_tile[2], (float)output_tile[3],
                         (float)output_tile[4], (float)output_tile[5],
                         (float)output_tile[6], (float)output_tile[7]);

    // Expected result: 32 * 1.0 * 1.0 = 32.0
    // Check if result is correct
    float expected = 32.0f;
    float actual = (float)output_tile[0];
    if (actual == expected) {
        GGMLHEXAGON_LOG_INFO("HMX test PASSED: expected=%f, actual=%f", expected, actual);
    } else if (isnan(actual)) {
        GGMLHEXAGON_LOG_ERROR("HMX test FAILED: result is NaN");
    } else {
        GGMLHEXAGON_LOG_ERROR("HMX test FAILED: expected=%f, actual=%f", expected, actual);
    }
    GGMLHEXAGON_LOG_INFO("=== leave HMX Instruction Flow Test ===");
}

// Core HMX dot product computation (matching official Qualcomm implementation)
static void core_dot_chunk_fp16(__fp16 *restrict output, const __fp16 *restrict activation,
                                const __fp16 *restrict weight, const __fp16 *restrict scales,
                                int n_row_tiles, int n_col_tiles, int n_dot_tiles) {
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);

    // Set scales (must be done before tile loading)
    hmx_set_output_scales(scales);

    for (int r = 0; r < n_row_tiles; ++r) {
        for (int c = 0; c < n_col_tiles; ++c) {
            // Clear accumulator before each output tile (matching official implementation)
            Q6_mxclracc_hf();

            const __fp16 *row_tiles = activation + r * n_dot_tiles * HMX_FP16_TILE_N_ELMS;
            const __fp16 *col_tiles = weight + c * n_dot_tiles * HMX_FP16_TILE_N_ELMS;

            // Load tiles in batches of up to 32 (matching official implementation)
            for (int k = 0, k_block; k < n_dot_tiles; k += k_block) {
                k_block = (n_dot_tiles - k) > 32 ? 32 : (n_dot_tiles - k);
                const uint32_t range = 2048u * (uint32_t)k_block - 1;
                hmx_load_activation_fp16(row_tiles, range);
                hmx_load_weight_fp16(col_tiles, range);
                row_tiles += k_block * HMX_FP16_TILE_N_ELMS;
                col_tiles += k_block * HMX_FP16_TILE_N_ELMS;
            }

            __fp16 *out_tile = output + (r * n_col_tiles + c) * HMX_FP16_TILE_N_ELMS;
            hmx_consume_accumulator_fp16(out_tile);
        }
    }
}

// Transfer output chunk from fp16 tiles to fp32 (column-major format)
// dst is [M, N] stored as column-major: dst[col * M + row]
static void transfer_output_chunk_fp16_to_fp32_colmajor(float *restrict dst, const __fp16 *restrict vtcm_src,
                                                        int m, int n) {
    // m = M (dst rows), n = N (dst columns)
    // Output tile layout: [n_row_tiles][n_col_tiles] where n_row_tiles = N/32, n_col_tiles = M/32
    // Each tile is 32x32 fp16 elements in separated format
    // tile_idx must match core_dot_chunk_fp16's calculation
    // core_dot: tile_idx = r * n_col_tiles + c, where r is N tile, c is M tile
    // So output tiles are arranged by N dimension (rows), then M dimension (columns)

    const int m_tiles = m / HMX_FP16_TILE_N_ROWS;  // M tiles = n_col_tiles in core_dot
    const int n_tiles = n / HMX_FP16_TILE_N_COLS;  // N tiles = n_row_tiles in core_dot

    for (int rt = 0; rt < n_tiles; ++rt) {  // rt is N tile (corresponds to r in core_dot)
        for (int ct = 0; ct < m_tiles; ++ct) {  // ct is M tile (corresponds to c in core_dot)
            // tile_idx = rt * m_tiles + ct = r * n_col_tiles + c
            const __fp16 *tile = vtcm_src + (rt * m_tiles + ct) * HMX_FP16_TILE_N_ELMS;

            // Process each element in the tile
            for (int i = 0; i < HMX_FP16_TILE_N_ROWS; ++i) {  // 32 rows (N dimension within tile)
                int row = rt * HMX_FP16_TILE_N_ROWS + i;  // global N index
                for (int j = 0; j < HMX_FP16_TILE_N_COLS; ++j) {  // 32 columns (M dimension within tile)
                    int col = ct * HMX_FP16_TILE_N_COLS + j;  // global M index

                    // Separated format: tile[(i/2) * 64 + (i%2) * 32 + j]
                    int tile_idx = (i / 2) * 64 + (i % 2) * 32 + j;
                    __fp16 val_fp16 = tile[tile_idx];
                    float val = (float)val_fp16;

                    // Write to column-major dst: dst[col * m + row]
                    dst[col * m + row] = val;
                }
            }
        }
    }
}

// Test HMX function - inline HMX implementation matching official code
int ggmlop_dsp_test_hmx(remote_handle64 h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) {
    test_hmx_instruction_flow();

    GGMLHEXAGON_LOG_INFO("==============enter %s (self-contained HMX)===========\n", __func__);

    char tempbuf[256];
    ggml_get_opkey(GGML_OP_MUL_MAT, src0, src1, tempbuf, 256);
    int64_t begin_time = ggml_time_us();

    if (!src0 || !src1 || !dst) {
        GGMLHEXAGON_LOG_ERROR("invalid input tensors");
        return AEE_EBADPARM;
    }

    const int32_t K = src0->ne[0];
    const int32_t M = src0->ne[1];
    const int32_t N = src1->ne[1];

    GGMLHEXAGON_LOG_INFO("Matrix dimensions: M=%d, N=%d, K=%d", M, N, K);

    // Set dst dimensions - matching ggml_mul_mat definition
    // dst = [M, N] stored as column-major: dst->ne[0]=M, dst->ne[1]=N
    dst->ne[0] = M;
    dst->ne[1] = N;
    dst->nb[0] = sizeof(float);
    dst->nb[1] = dst->nb[0] * dst->ne[0];

    // Ensure VTCM resource is available (for cache mode)
    int vtcm_err = ggmlop_ensure_vtcm_available();
    if (vtcm_err != 0) {
        GGMLHEXAGON_LOG_ERROR("Failed to ensure VTCM availability: %d", vtcm_err);
        return AEE_EFAILED;
    }

    // Get VTCM
    size_t vtcm_size = 0;
    void * vtcm_base = ggmlop_get_vtcm_pool(&vtcm_size);
    if (vtcm_base == NULL) {
        GGMLHEXAGON_LOG_ERROR("No VTCM pool available");
        return AEE_EFAILED;
    }

    GGMLHEXAGON_LOG_INFO("VTCM pool: base=%p, size=%zu bytes", vtcm_base, vtcm_size);

    // Check alignment (HMX requires 32-aligned dimensions)
    if (K % 32 != 0 || N % 32 != 0 || M % 32 != 0) {
        GGMLHEXAGON_LOG_ERROR("HMX dimensions must be 32-aligned: M=%d, N=%d, K=%d", M, N, K);
        return AEE_EBADPARM;
    }

    // Calculate tile counts
    // matching ggml_mul_mat parameter convention
    // src0 = weight [K, M], src1 = activation [K, N], dst = [M, N]
    // n_row_tiles = N / 32 (activation row tiles, corresponds to dst N dimension)
    // n_col_tiles = M / 32 (weight column tiles, corresponds to dst M dimension)
    // n_dot_tiles = K / 32 (inner dimension tiles)
    const int n_row_tiles = N / 32;
    const int n_col_tiles = M / 32;
    const int n_dot_tiles = K / 32;

    GGMLHEXAGON_LOG_INFO("Tile counts: row=%d, col=%d, dot=%d", n_row_tiles, n_col_tiles, n_dot_tiles);

    // VTCM layout - matching official implementation
    // matching ggml_mul_mat parameter convention
    // src0 = weight [K, M], src1 = activation [K, N], dst = [M, N]
    const size_t vec_dot_size = K * sizeof(__fp16);
    const size_t act_area_size = ((N * vec_dot_size + 2047) & ~2047);  // activation [K, N]
    const size_t weight_area_size = ((M * vec_dot_size + 2047) & ~2047);  // weight [K, M]
    const size_t output_area_size = ((M * N * sizeof(__fp16) + 2047) & ~2047);
    const size_t scales_size = 256;
    // Extra buffer for weight fp32 data (needed for HVX vscatter to work correctly)
    const size_t weight_fp32_buf_size = ((M * K * sizeof(float) + 2047) & ~2047);  // weight [K, M]
    // Extra buffer for activation fp32 data (needed for HVX Vector read to work correctly)
    const size_t act_fp32_buf_size = ((N * K * sizeof(float) + 2047) & ~2047);  // activation [K, N]

    GGMLHEXAGON_LOG_INFO("VTCM sizes: act=%zu, weight=%zu, output=%zu, scales=%zu, weight_fp32_buf=%zu, act_fp32_buf=%zu",
                         act_area_size, weight_area_size, output_area_size, scales_size, weight_fp32_buf_size, act_fp32_buf_size);

    // Check VTCM capacity
    size_t total_needed = act_area_size + weight_area_size + output_area_size + scales_size + weight_fp32_buf_size + act_fp32_buf_size;
    if (total_needed > vtcm_size) {
        GGMLHEXAGON_LOG_ERROR("VTCM insufficient: needed=%zu, available=%zu", total_needed, vtcm_size);
        return AEE_EFAILED;
    }

    // Allocate VTCM regions
    uint8_t *vtcm_ptr = (uint8_t *)vtcm_base;
    __fp16 *vtcm_activation = (__fp16 *)vtcm_ptr;
    vtcm_ptr += act_area_size;
    __fp16 *vtcm_weight = (__fp16 *)vtcm_ptr;
    vtcm_ptr += weight_area_size;
    __fp16 *vtcm_output = (__fp16 *)vtcm_ptr;
    vtcm_ptr += output_area_size;
    // Activation fp32 buffer for HVX Vector read
    float *vtcm_act_fp32_buf = (float *)vtcm_ptr;
    vtcm_ptr += act_fp32_buf_size;
    // Weight fp32 buffer for HVX vscatter
    float *vtcm_weight_fp32_buf = (float *)vtcm_ptr;
    vtcm_ptr += weight_fp32_buf_size;

    // Align scales to 256 bytes
    uintptr_t scales_addr = ((uintptr_t)vtcm_ptr + 255) & ~255;
    __fp16 *vtcm_scales = (__fp16 *)scales_addr;

    GGMLHEXAGON_LOG_INFO("VTCM layout: act=%p, weight=%p, output=%p, act_fp32_buf=%p, weight_fp32_buf=%p, scales=%p",
                         vtcm_activation, vtcm_weight, vtcm_output, vtcm_act_fp32_buf, vtcm_weight_fp32_buf, vtcm_scales);

    // ====== STEP 1: Transfer weight (src0) ======
    // src0 is weight [K, M]
    // Matching ggml_mul_mat parameter convention
    const float *src0_data = (const float *)src0->data;
    const size_t src0_stride = src0->nb[1] / sizeof(float);

    GGMLHEXAGON_LOG_INFO("Transferring weight: M=%d, K=%d, stride=%zu, first_val=%f",
                         M, K, src0_stride, src0_data[0]);

    // First copy weight data from RPC memory to VTCM buffer (HMX needs data in VTCM)
    // weight matrix [K, M] in column-major: src0_data[m * stride + k] = weight[k, m]
    // We copy to row-major buffer: vtcm_weight_fp32_buf[m * K + k] = weight[k, m]
    for (int i = 0; i < M; ++i) {
        memcpy(vtcm_weight_fp32_buf + i * K, src0_data + i * src0_stride, K * sizeof(float));
    }
    __asm__ __volatile__("" ::: "memory");

    // DEBUG: Check weight_fp32_buf data after memcpy
    GGMLHEXAGON_LOG_INFO("DEBUG: weight_fp32_buf[0..3]=%f,%f,%f,%f (after memcpy to VTCM)",
                         vtcm_weight_fp32_buf[0], vtcm_weight_fp32_buf[1],
                         vtcm_weight_fp32_buf[2], vtcm_weight_fp32_buf[3]);

    // Convert weight from fp32 to fp16 tiles using scalar approach
    // weight tile rows correspond to M dimension, columns to K dimension
    // Matching mulmat.c convert_weight_f32_to_fp16_tiles implementation
    // Total weight tiles = n_col_tiles * n_dot_tiles = (M / 32) * (K / 32)
    for (int t = 0; t < n_col_tiles * n_dot_tiles; ++t) {
        int ct = t / n_dot_tiles;  // M tile index (output dimension)
        int kt = t % n_dot_tiles;  // K tile index (inner dimension)

        __fp16 *tile_base = vtcm_weight + t * HMX_FP16_TILE_N_ELMS;

        for (int i = 0; i < 32; i++) {  // 32 rows per tile (M dimension)
            int row_global = ct * 32 + i;  // global M index
            for (int j = 0; j < 32; j++) {  // 32 columns per tile (K dimension)
                int col_global = kt * 32 + j;  // global K index
                float val = (row_global < M && col_global < K) ?
                            vtcm_weight_fp32_buf[row_global * K + col_global] : 0.0f;
                // Interleaved: tile[(i/2)*64 + j*2 + (i%2)]
                // i = M dimension (tile row), j = K dimension (tile column)
                tile_base[(i / 2) * 64 + j * 2 + (i % 2)] = (__fp16)val;
            }
        }
    }
    __asm__ __volatile__("" ::: "memory");

    // DEBUG: Check weight data
    GGMLHEXAGON_LOG_INFO("DEBUG: After weight transfer: weight[0..3]=%f,%f,%f,%f",
                         (float)vtcm_weight[0],
                         (float)vtcm_weight[1],
                         (float)vtcm_weight[2],
                         (float)vtcm_weight[3]);

    // ====== STEP 2: Transfer activation (src1) ======
    // src1 is activation [K, N]
    // Matching ggml_mul_mat parameter convention
    const float *src1_data = (const float *)src1->data;
    const size_t src1_stride = src1->nb[1] / sizeof(float);

    GGMLHEXAGON_LOG_INFO("Transferring activation: N=%d, K=%d, stride=%zu, first_val=%f",
                         N, K, src1_stride, src1_data[0]);

    // First copy activation data from RPC memory to VTCM buffer (HVX needs data in VTCM)
    // activation matrix [K, N] in column-major: src1_data[n * stride + k] = activation[k, n]
    // We copy to row-major buffer: vtcm_act_fp32_buf[n * K + k] = activation[k, n]
    for (int i = 0; i < N; ++i) {
        memcpy(vtcm_act_fp32_buf + i * K, src1_data + i * src1_stride, K * sizeof(float));
    }
    __asm__ __volatile__("" ::: "memory");

    // DEBUG: Check activation fp32 data after memcpy
    GGMLHEXAGON_LOG_INFO("DEBUG: act_fp32_buf[0..3]=%f,%f,%f,%f (after memcpy to VTCM)",
                         vtcm_act_fp32_buf[0], vtcm_act_fp32_buf[1],
                         vtcm_act_fp32_buf[2], vtcm_act_fp32_buf[3]);

    // Now convert from VTCM fp32 buffer to fp16 tiles using scalar approach
    // activation tile rows correspond to N dimension, columns to K dimension
    transfer_activation_chunk_fp32_to_fp16_scalar(vtcm_activation, vtcm_act_fp32_buf, N, K, K);

    // ====== STEP 3: Initialize scales (AFTER weight, matching test_hmx_instruction_flow order) ======
    // Initialize scales: scale=1.0, bias=0.0 using HVX vector (matching official backend)
#if 1
    //right and precise
    // CRITICAL: Use HVX vector write for HMX instruction compatibility
    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16
#else
    //right and not precise:1.0 offset
    //CRITICAL: Use (__fp16) cast for VTCM fp16 write (matching test_hmx_instruction_flow)
    for (int i = 0; i < 64; ++i) {
        vtcm_scales[i] = (__fp16)1.0f;  // scale
    }
    for (int i = 64; i < 128; ++i) {
        vtcm_scales[i] = (__fp16)0.0f;  // bias
    }
#endif

    // Memory barrier after HVX write/scalar write
    __asm__ __volatile__("" ::: "memory");

    // Verify scales initialization
    GGMLHEXAGON_LOG_INFO("DEBUG: scales[0]=%f (scale), scales[64]=%f (bias)",
                         (float)vtcm_scales[0],
                         (float)vtcm_scales[64]);

    // ====== STEP 4: Initialize output (AFTER scales, matching test_hmx_instruction_flow order) ======
    memset(vtcm_output, 0, output_area_size);

    // Memory barrier before HMX lock
    __asm__ __volatile__("" ::: "memory");

    // Acquire HMX lock (ALL VTCM initialization must be done BEFORE this)
    unsigned int compute_res_ctx_id = g_dsp_ctx->compute_res_ctx_id;
    GGMLHEXAGON_LOG_INFO("compute_res_ctx_id=%u", compute_res_ctx_id);

    int lock_result = HAP_compute_res_hmx_lock(compute_res_ctx_id);
    GGMLHEXAGON_LOG_INFO("HMX lock result: %d", lock_result);
    if (lock_result != 0) {
        GGMLHEXAGON_LOG_ERROR("HMX lock failed (%d)", lock_result);
        return AEE_EFAILED;
    }

    // ====== STEP 3: Software matrix multiplication test for verification ======
    float *sw_output = (float *)malloc(M * N * sizeof(float));
    if (sw_output) {
        software_matmul_fp32(sw_output, src0_data, src1_data, M, N, K,
                            src0_stride, src1_stride, N);
        GGMLHEXAGON_LOG_INFO("DEBUG: Software matmul result[0]=%f", sw_output[0]);
        free(sw_output);
    }

    // ====== STEP 4: Execute HMX matrix multiplication ======
    GGMLHEXAGON_LOG_INFO("Performing HMX matrix multiplication");

    // DEBUG: Print fp16 hex values to check format
    GGMLHEXAGON_LOG_INFO("DEBUG: act[0] hex=0x%04x, weight[0] hex=0x%04x, scales[0] hex=0x%04x",
                         (unsigned int)*(uint16_t*)&vtcm_activation[0],
                         (unsigned int)*(uint16_t*)&vtcm_weight[0],
                         (unsigned int)*(uint16_t*)&vtcm_scales[0]);

    GGMLHEXAGON_LOG_INFO("Before HMX: act[0]=%f, weight[0]=%f, output[0]=%f",
                         (float)vtcm_activation[0],
                         (float)vtcm_weight[0],
                         (float)vtcm_output[0]);

    core_dot_chunk_fp16(vtcm_output, vtcm_activation, vtcm_weight, vtcm_scales,
                        n_row_tiles, n_col_tiles, n_dot_tiles);

    // Memory barrier
    __asm__ __volatile__("" ::: "memory");

    // Check result after HMX
    GGMLHEXAGON_LOG_INFO("After HMX: output[0]=%f, output[1]=%f, output[2]=%f, output[3]=%f",
                         (float)vtcm_output[0],
                         (float)vtcm_output[1],
                         (float)vtcm_output[2],
                         (float)vtcm_output[3]);

    // ====== STEP 5: Transfer result back ======
    float *dst_data = (float *)dst->data;

    transfer_output_chunk_fp16_to_fp32_colmajor(dst_data, vtcm_output, M, N);

    // Release HMX lock
    HAP_compute_res_hmx_unlock(compute_res_ctx_id);

    // Verify - note: HMX F16 accumulation has hardware precision characteristics
    // Expected: dst[i] = K * src0_val * src1_val
    // For 64x64x64 with src0=0.5, src1=1.0: expected = 64 * 0.5 * 1.0 = 32.0
    float src0_val = src0_data[0];
    float src1_val = src1_data[0];
    float expected = (float)K * src0_val * src1_val;
    GGMLHEXAGON_LOG_INFO("HMX test: expected=%f (K=%d, src0=%f, src1=%f), result[0]=%f",
                         expected, K, src0_val, src1_val, dst_data[0]);

    // Performance measurement
    int64_t end_time = ggml_time_us();
    GGMLHEXAGON_LOG_INFO("elapse time of %s is %lld us", tempbuf, (long long)(end_time - begin_time));

    GGMLHEXAGON_LOG_INFO("==============leave %s===========", __func__);
    return AEE_SUCCESS;
}
