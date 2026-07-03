#include <hexagon_types.h>
#include <HAP_power.h>
#include <HAP_dcvs.h>
#include <HAP_mem.h>
#include <HAP_compute_res.h>

#include "ggml-dsp.h"
#include "ggml-ops.h"
#include "../htp/htp-ctx.h"
#include "../htp/matmul-ops.h"

static int g_thread_counts                  = 1;
static int g_mulmat_algotype                = 0;
static int g_offload_cgraph_type            = 2;
static int g_dump_diag_info                 = 0;
static void * g_work_data                   = NULL;
static size_t g_work_size                   = 0;

static void * g_vtcm_base                   = NULL;
static size_t g_vtcm_size                   = 0;
static unsigned int g_compute_res_ctx_id    = 0;
static int g_power_ctx                      = 0;
static int g_hmx_available                  = 0;
static struct hmx_queue * g_hmx_queue        = NULL;  // Async HMX queue (created when HMX is available)
static volatile int g_vtcm_needs_release    = 0;  // For cache mode VTCM management
static volatile int g_vtcm_valid            = 0;  // VTCM resource is currently valid/available

// htp_context for calling Qualcomm's execute_op.
// Shares our already-acquired VTCM/HMX resources; worker_pool and dma queues
// are initialized in ggmlop_dsp_open.
static struct htp_context g_htp_ctx;

static void * g_hexagon_power_ctx           = NULL;
static void * g_ion_dsp_base                = NULL;
static size_t g_ion_dsp_size                = 0;     // ION total size (bytes)

// FP16 weight cache: uses ION shared memory tail region for caching
// converted FP16 weight tiles (avoids repeated Q4_0->FP16 conversion)
// Cache region: [g_ion_cache_base, g_ion_dsp_base + g_ion_size)
// Grows from cache_base forward (monotonic bump allocator)
static void * g_ion_cache_base          = NULL;  // DSP VA of cache region start
static size_t g_ion_cache_size          = 0;     // cache region size in bytes
static size_t g_ion_cache_offset        = 0;     // monotonic allocation offset within cache

#define MAX_WORK_SIZE                       (1024 * 1024 * 1024)
#define DEFAULT_VTCM_SIZE                   (8 * 1024 * 1024)

// ===========================================================================
// Qualcomm execute_op dispatch (moved from htp/main.c)
// All op_xxx functions are exported from htp/*.c (non-static, declared in
// htp-ctx.h). We only need this dispatch wrapper + a translation layer.
// ===========================================================================
static int execute_op(struct htp_ops_context * octx) {
    switch (octx->op) {
        case HTP_OP_MUL_MAT:
        case HTP_OP_MUL_MAT_ADD:
            return op_matmul(octx);
        case HTP_OP_MUL_MAT_ID:
            return op_matmul_id(octx);
        case HTP_OP_MUL_MAT_QKV:
            return op_matmul_qkv(octx);
        case HTP_OP_MUL_MAT_FFN:
            return op_matmul_ffn(octx);
        case HTP_OP_MUL:
        case HTP_OP_ADD:
        case HTP_OP_SUB:
        case HTP_OP_DIV:
        case HTP_OP_ADD_ID:
            return op_binary(octx);
        case HTP_OP_NORM:
        case HTP_OP_RMS_NORM:
        case HTP_OP_RMS_NORM_MUL:
        case HTP_OP_SCALE:
        case HTP_OP_SQR:
        case HTP_OP_SQRT:
        case HTP_OP_UNARY_SOFTPLUS:
        case HTP_OP_UNARY_SIGMOID:
        case HTP_OP_UNARY_NEG:
        case HTP_OP_UNARY_EXP:
        case HTP_OP_UNARY_TANH:
        case HTP_OP_L2_NORM:
            return op_unary(octx);
        case HTP_OP_UNARY_SILU:
        case HTP_OP_UNARY_GELU:
        case HTP_OP_GLU_SWIGLU:
        case HTP_OP_GLU_SWIGLU_OAI:
        case HTP_OP_GLU_GEGLU:
            return op_activations(octx);
        case HTP_OP_SOFTMAX:
            return op_softmax(octx);
        case HTP_OP_ROPE:
            return op_rope(octx);
        // case HTP_OP_FLASH_ATTN_EXT:
        //     return op_flash_attn_ext(octx);
        case HTP_OP_SET_ROWS:
            return op_set_rows(octx);
        case HTP_OP_GET_ROWS:
            return op_get_rows(octx);
        case HTP_OP_SUM_ROWS:
            return op_sum_rows(octx);
        case HTP_OP_CPY:
            return op_cpy(octx);
        case HTP_OP_REPEAT:
            return op_repeat(octx);
        case HTP_OP_ARGSORT:
            return op_argsort(octx);
        case HTP_OP_SSM_CONV:
            return op_ssm_conv(octx);
        case HTP_OP_CUMSUM:
            return op_cumsum(octx);
        case HTP_OP_FILL:
            return op_fill(octx);
        case HTP_OP_DIAG:
            return op_diag(octx);
        case HTP_OP_SOLVE_TRI:
            return op_solve_tri(octx);
        case HTP_OP_PAD:
            return op_pad(octx);
        case HTP_OP_CONCAT:
            return op_concat(octx);
        case HTP_OP_GATED_DELTA_NET:
            return op_gated_delta_net(octx);
        case HTP_OP_TRI:
            return op_tri(octx);
        case HTP_OP_INVALID:
            break;
    }
    FARF(ERROR, "Unknown Op %u", octx->op);
    return -1;
}

// ---------------------------------------------------------------------------
// Translation layer: dsptensor -> htp_tensor, GGML_OP -> HTP_OP
// ---------------------------------------------------------------------------

// Hexagon DSP is 32-bit address space: pointer fits in uint32_t.
// htp_tensor.data is uint32_t offset, but Qualcomm's prep_tensor replaces
// it with actual pointer. We set it directly to the pointer value and mark
// HTP_TENSOR_FLUSHED so proc_op_req skips L2 flush (we handle cache ourselves).
static inline void dsptensor_to_htp_tensor(const dsptensor * dt,
                                            struct htp_tensor * ht) {
    ht->data  = (uint32_t)(uintptr_t)dt->data;
    ht->size  = (uint32_t)dt->data_len;
    ht->flags = HTP_TENSOR_FLUSHED;
    ht->type  = (uint16_t)dt->type;
    ht->bi    = 0;
    ht->ne[0] = (uint32_t)dt->ne[0];
    ht->ne[1] = (uint32_t)dt->ne[1];
    ht->ne[2] = (uint32_t)dt->ne[2];
    ht->ne[3] = (uint32_t)dt->ne[3];
    ht->nb[0] = (uint32_t)dt->nb[0];
    ht->nb[1] = (uint32_t)dt->nb[1];
    ht->nb[2] = (uint32_t)dt->nb[2];
    ht->nb[3] = (uint32_t)dt->nb[3];
}

// Map GGML opcode to HTP opcode. Returns 0 on success, -1 if unsupported.
static int ggml_op_to_htp_op(int32_t ggml_op, const int32_t * op_params,
                             enum htp_op_code * htp_op) {
    switch (ggml_op) {
        case GGML_OP_ADD:      *htp_op = HTP_OP_ADD;         return 0;
        case GGML_OP_SUB:      *htp_op = HTP_OP_SUB;         return 0;
        case GGML_OP_MUL:      *htp_op = HTP_OP_MUL;         return 0;
        case GGML_OP_DIV:      *htp_op = HTP_OP_DIV;         return 0;
        case GGML_OP_MUL_MAT:  *htp_op = HTP_OP_MUL_MAT;     return 0;
        case GGML_OP_RMS_NORM: *htp_op = HTP_OP_RMS_NORM;    return 0;
        case GGML_OP_ROPE:     *htp_op = HTP_OP_ROPE;        return 0;
        default:
            FARF(ERROR, "ggml_op_to_htp_op: unsupported ggml_op %d", ggml_op);
            return -1;
    }
}

// Build htp_ops_context from our dsptensor structures, ready for execute_op.
// Mirrors proc_op_req in htp/main.c: unconditionally copy op_params, and copy
// kernel_params when available (non-NULL). For dsptensor-based callers (no
// kernel_params), pass NULL and the memset-zero state is preserved.
static void build_htp_octx(
    struct htp_ops_context * octx,
    enum htp_op_code htp_op,
    const int32_t * op_params,
    const int32_t * kernel_params,
    const dsptensor * src0, const dsptensor * src1,
    const dsptensor * src2, const dsptensor * src3,
    const dsptensor * dst,
    struct htp_tensor src_ht[HTP_OP_MAX_INPUTS],
    struct htp_tensor * dst_ht) {

    memset(octx, 0, sizeof(*octx));
    octx->ctx = &g_htp_ctx;
    octx->op  = htp_op;
    // Mirror proc_op_req: unconditional copy (op_params is always provided)
    memcpy(octx->op_params, op_params, sizeof(octx->op_params));
    if (kernel_params) {
        memcpy(octx->kernel_params, kernel_params, sizeof(octx->kernel_params));
    }

    const dsptensor * srcs[HTP_OP_MAX_INPUTS] = {src0, src1, src2, src3, NULL, NULL};
    for (int i = 0; i < HTP_OP_MAX_INPUTS; i++) {
        if (srcs[i]) {
            dsptensor_to_htp_tensor(srcs[i], &src_ht[i]);
            octx->src[i] = &src_ht[i];
        } else {
            octx->src[i] = NULL;
        }
    }

    if (dst) {
        dsptensor_to_htp_tensor(dst, dst_ht);
        octx->dsts[0] = dst_ht;
    } else {
        octx->dsts[0] = NULL;
    }

    octx->n_threads = (uint32_t)g_thread_counts;
}

// Compute htp_mm_kernel_params on DSP side for MUL_MAT.
// Mirrors ggml_hexagon_precompute_hvx_mm_params (F32/F16 paths only).
// HMX and quantized paths are not yet supported.
static int build_mm_kernel_params(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;
    if (!src0 || !src1 || !dst) return -1;

    struct htp_mm_kernel_params * kparams =
        (struct htp_mm_kernel_params *) octx->kernel_params;
    memset(kparams, 0, sizeof(*kparams));

    const int wtype = src0->type;
    const uint32_t ne02 = src0->ne[2];
    const uint32_t ne03 = src0->ne[3];
    const uint32_t ne10 = src1->ne[0];
    const uint32_t ne11 = src1->ne[1];
    const uint32_t ne12 = src1->ne[2];
    const uint32_t ne13 = src1->ne[3];
    const uint32_t src1_nrows = ne11 * ne12 * ne13;

    kparams->n_hmx       = 0;
    kparams->n_threads   = octx->n_threads;
    kparams->n_prefetch  = 16;

    const bool is_batched  = (ne02 > 1) || (ne03 > 1);
    const bool is_permuted = (src0->nb[0] > src0->nb[1] || src0->nb[1] > src0->nb[2] || src0->nb[2] > src0->nb[3]) ||
                             (src1->nb[0] > src1->nb[1] || src1->nb[1] > src1->nb[2] || src1->nb[2] > src1->nb[3]);

    size_t vtcm_src0_size = 0, vtcm_src1_size = 0, vtcm_dst_size = 0;

    if (wtype == HTP_TYPE_F32) {
        size_t vtcm_size = htp_mm_hvx_get_vtcm_sizes(
            HTP_MM_KERNEL_HVX_F32_F32_VTCM, wtype, ne10, src1_nrows, octx->n_threads,
            dst->nb[1], src0->nb[1], src1->nb[1], 16,
            &vtcm_src0_size, &vtcm_src1_size, &vtcm_dst_size);

        if (!is_batched && !is_permuted && vtcm_size <= g_vtcm_size) {
            kparams->kernel_type    = HTP_MM_KERNEL_HVX_F32_F32_VTCM;
            kparams->src1_row_size  = hex_round_up(ne10 * 4, 128);
        } else {
            kparams->kernel_type    = HTP_MM_KERNEL_HVX_F32_F32_DDR;
            kparams->src1_row_size  = src1->nb[1];
            vtcm_size = htp_mm_hvx_get_vtcm_sizes(
                kparams->kernel_type, wtype, ne10, src1_nrows, octx->n_threads,
                dst->nb[1], src0->nb[1], src1->nb[1], 16,
                &vtcm_src0_size, &vtcm_src1_size, &vtcm_dst_size);
        }
        kparams->vtcm_size      = (int32_t) vtcm_size;
        kparams->vtcm_src0_size = (int32_t) vtcm_src0_size;
        kparams->vtcm_src1_size = (int32_t) vtcm_src1_size;
        kparams->vtcm_dst_size  = (int32_t) vtcm_dst_size;
    } else if (wtype == HTP_TYPE_F16) {
        size_t vtcm_size = htp_mm_hvx_get_vtcm_sizes(
            HTP_MM_KERNEL_HVX_F16_F16_VTCM, wtype, ne10, src1_nrows, octx->n_threads,
            dst->nb[1], src0->nb[1], src1->nb[1], 16,
            &vtcm_src0_size, &vtcm_src1_size, &vtcm_dst_size);

        if (!is_batched && !is_permuted && vtcm_size <= g_vtcm_size) {
            kparams->kernel_type    = HTP_MM_KERNEL_HVX_F16_F16_VTCM;
            kparams->src1_row_size  = hex_round_up(ne10 * 2, 128);
        } else {
            if (src1->type == HTP_TYPE_F32) {
                kparams->kernel_type = HTP_MM_KERNEL_HVX_F16_F32_DDR;
            } else {
                kparams->kernel_type = HTP_MM_KERNEL_HVX_F16_F16_DDR;
            }
            kparams->src1_row_size  = src1->nb[1];
            vtcm_size = htp_mm_hvx_get_vtcm_sizes(
                kparams->kernel_type, wtype, ne10, src1_nrows, octx->n_threads,
                dst->nb[1], src0->nb[1], src1->nb[1], 16,
                &vtcm_src0_size, &vtcm_src1_size, &vtcm_dst_size);
        }
        kparams->vtcm_size      = (int32_t) vtcm_size;
        kparams->vtcm_src0_size = (int32_t) vtcm_src0_size;
        kparams->vtcm_src1_size = (int32_t) vtcm_src1_size;
        kparams->vtcm_dst_size  = (int32_t) vtcm_dst_size;
    } else {
        // Quantized HVX path (Q4_0, Q4_1, Q5_0, Q8_0, IQ4_NL, MXFP4)
        kparams->tile_size         = (int32_t) htp_mm_get_weight_tile_size(wtype);
        kparams->aligned_tile_size = (int32_t) htp_mm_get_weight_aligned_tile_size(wtype);

        const bool k_align   = (ne10 % 32 == 0);
        const bool try_tiled = k_align && kparams->tile_size > 0;
        bool tiled_ok = false;

        if (try_tiled) {
            kparams->src1_row_size = (int32_t)((wtype == HTP_TYPE_Q4_1)
                ? htp_mm_q8_1_tiled_row_size(ne10)
                : htp_mm_q8_0_tiled_row_size(ne10));
            kparams->kernel_type = (src1_nrows < octx->n_threads)
                ? HTP_MM_KERNEL_HVX_QUANT_BLOCK
                : HTP_MM_KERNEL_HVX_QUANT_ROW;

            const uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
            uint32_t best_n_prefetch = 2;
            size_t vs0 = 0, vs1 = 0, vd = 0;
            size_t total_size = 0;
            for (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                total_size = htp_mm_hvx_get_vtcm_sizes(
                    kparams->kernel_type, wtype, ne10, src1_nrows, octx->n_threads,
                    dst->nb[1], src0->nb[1], src1->nb[1], d,
                    &vs0, &vs1, &vd);
                if (total_size <= g_vtcm_size) {
                    best_n_prefetch = d;
                    break;
                }
            }
            if (best_n_prefetch == 2 && total_size > g_vtcm_size) {
                total_size = htp_mm_hvx_get_vtcm_sizes(
                    kparams->kernel_type, wtype, ne10, src1_nrows, octx->n_threads,
                    dst->nb[1], src0->nb[1], src1->nb[1], 2,
                    &vs0, &vs1, &vd);
            }
            kparams->n_prefetch = (int32_t) best_n_prefetch;

            if (total_size <= g_vtcm_size) {
                kparams->vtcm_size      = (int32_t) total_size;
                kparams->vtcm_src0_size = (int32_t) vs0;
                kparams->vtcm_src1_size = (int32_t) vs1;
                kparams->vtcm_dst_size  = (int32_t) vd;
                tiled_ok = true;
            }
        }

        if (!tiled_ok) {
            kparams->src1_row_size = (int32_t)((wtype == HTP_TYPE_Q4_1)
                ? htp_mm_q8_1_flat_row_size(ne10)
                : htp_mm_q8_0_flat_row_size(ne10));
            kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT;

            size_t vs0 = 0, vs1 = 0, vd = 0;
            const size_t total_size = htp_mm_hvx_get_vtcm_sizes(
                kparams->kernel_type, wtype, ne10, src1_nrows, octx->n_threads,
                dst->nb[1], src0->nb[1], src1->nb[1], 16,
                &vs0, &vs1, &vd);

            kparams->n_prefetch     = 16;
            kparams->vtcm_size      = (int32_t) total_size;
            kparams->vtcm_src0_size = (int32_t) vs0;
            kparams->vtcm_src1_size = (int32_t) vs1;
            kparams->vtcm_dst_size  = (int32_t) vd;
        }
    }

    kparams->div_ne12_ne1 = init_fastdiv_values(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_values(ne11);
    kparams->div_r2       = init_fastdiv_values(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_values(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne11     = init_fastdiv_values(ne11);

    return 0;
}

// Stub: FP16 weight cache was managed by kernels/mulmat.c (removed from build).
// The ION cache region setup is preserved, but no cache entries are populated.
void ggmlop_dsp_fp16_cache_reset(void) {
    // no-op: cache is reset via g_ion_cache_offset in ggmlop_dsp_execute_batch_ion
}

static int power_on_hvx_hmx(void) {
    HAP_power_request_t req;

    /* Set client class */
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_apptype;
    req.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
    if (HAP_power_set((void *)&g_power_ctx, &req) != 0) {
        GGMLHEXAGON_LOG_ERROR("HAP_power_set apptype failed");
        return -1;
    }

    /* DCVS performance mode */
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_DCVS_v3;
    req.dcvs_v3.set_dcvs_enable = 1;
    req.dcvs_v3.dcvs_enable = 0;  // disable DVFS, pin to fixed frequency for stable performance
    req.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
    req.dcvs_v3.set_bus_params = 1;
    req.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.set_core_params = 1;
    req.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.set_sleep_disable = 1;
    req.dcvs_v3.sleep_disable = 1;

    GGMLHEXAGON_LOG_INFO("__HVX_ARCH__ = %d\n", __HVX_ARCH__);

    // v79 architecture requires protected bus corners setting
#if __HEXAGON_ARCH__ >= 79
    HAP_set_dcvs_v3_protected_bus_corners(&req, 1);
#endif

    if (HAP_power_set((void *)&g_power_ctx, &req) != 0) {
        GGMLHEXAGON_LOG_ERROR("HAP_power_set DCVS failed");
        return -2;
    }

    /* Power up HVX */
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_HVX;
    req.hvx.power_up = 1;
    if (HAP_power_set((void *)&g_power_ctx, &req) != 0) {
        GGMLHEXAGON_LOG_ERROR("HAP_power_set HVX failed");
        return -3;
    }

    /* Power up HMX with v2 settings for v75+ architecture */
#if __HVX_ARCH__ >= 75
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_HMX_v2;
    req.hmx_v2.set_power = 1;
    req.hmx_v2.power_up = 1;
    req.hmx_v2.set_clock = 1;
    req.hmx_v2.target_corner = HAP_DCVS_EXP_VCORNER_MAX;
    req.hmx_v2.min_corner = HAP_DCVS_EXP_VCORNER_MAX;
    req.hmx_v2.max_corner = HAP_DCVS_EXP_VCORNER_MAX;
    req.hmx_v2.perf_mode = HAP_CLK_PERF_HIGH;
    GGMLHEXAGON_LOG_INFO("Setting HMX clock with HMX_v2 for v75+ architecture");
    if (HAP_power_set((void *)&g_power_ctx, &req) != 0) {
        GGMLHEXAGON_LOG_ERROR("HAP_power_set HMX_v2 failed, continuing without HMX");
        return -4;
    }
#else
    /* Power up HMX (legacy for older architectures) */
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_HMX;
    req.hmx.power_up = 1;
    if (HAP_power_set((void *)&g_power_ctx, &req) != 0) {
        GGMLHEXAGON_LOG_ERROR("HAP_power_set HMX failed, continuing without HMX");
        return -4;
    }
#endif

    GGMLHEXAGON_LOG_INFO("HAP_power_set for HVX and HMX succeeded");
    return 0;
}


static int vtcm_release_callback(unsigned int rctx, void * state) {
    // Async notification only: flag that another session wants VTCM.
    // Do NOT clear g_vtcm_valid here - the current batch keeps running
    // and releases VTCM at the batch boundary (matches Qualcomm htp/main.c).
    g_vtcm_needs_release = 1;
    return 0;
}

int ggmlop_dsp_open(const char * uri, remote_handle64 * handle) {
    void * tptr = NULL;
    GGMLHEXAGON_LOG_INFO("uri %s", uri);
    tptr = (void *)malloc(1);
    GGML_ASSERT(NULL != tptr);
    *handle = (remote_handle64)tptr;

    unsigned int api_version = qurt_api_version();
    FARF(ALWAYS, "qurt_api_version            = 0x%x", api_version);
    FARF(ALWAYS, "qurt_hvx_units              = 0x%d", qurt_hvx_get_units());
    qurt_arch_version_t  vers;
    qurt_sysenv_get_arch_version(&vers);
    FARF(ALWAYS, "qurt_arch_version           = 0x%x", vers.arch_version);
    qurt_sysenv_app_heap_t aheap;
    qurt_sysenv_get_app_heap(&aheap);
    GGMLDSP_LOG_DEBUG("aheap.heap_base=0x%x, aheap.heap_limit=0x%x", aheap.heap_base, aheap.heap_limit);
    qurt_sysenv_max_hthreads_t mhwt;
    qurt_sysenv_get_max_hw_threads(&mhwt);
    FARF(ALWAYS, "qurt_hardware_thread_counts = %d", mhwt.max_hthreads);
     g_thread_counts = mhwt.max_hthreads;

    /* Step 1: Power up HVX and HMX */
    int power_result = power_on_hvx_hmx();
    if (power_result != 0) {
        GGMLHEXAGON_LOG_INFO("power_on_hvx_hmx failed (%d), continuing without HMX", power_result);
        g_hmx_available = 0;
    } else {
        g_hmx_available = 1;
    }

    /* Step 2: Query VTCM size and allocate resources */
    unsigned int vtcm_size_query = 0;
    unsigned int availBlockSize;
    unsigned int totalBlocksize;
    compute_res_vtcm_page_t availBlock;
    compute_res_vtcm_page_t totalBlock;
    int result = 0;
    result = HAP_compute_res_query_VTCM(0, &vtcm_size_query, &totalBlock, &availBlockSize, &availBlock);
    GGMLHEXAGON_LOG_INFO("VTCM total = %u bytes\n", vtcm_size_query);
    printf("Querying VTCM before acquiring resources:\n");
    printf("Compute resource query return %d, totalBlocksize %d, availBlockSize %d\n",
                                 result, vtcm_size_query, availBlockSize);
    printf("Compute resource query ctd, valid page sizes in total table: %d, valid page sizes in avail table: %d\n",
                                 totalBlock.page_list_len, availBlock.page_list_len);
    printf("Compute resource query ctd, (Size, num pages); total (0x%x, %d) Avail (0x%x, %d, 0x%x, %d)\n",
                                totalBlock.page_list[0].page_size,
                                totalBlock.page_list[0].num_pages,
                                availBlock.page_list[0].page_size,
                                availBlock.page_list[0].num_pages,
                                availBlock.page_list[1].page_size,
                                availBlock.page_list[1].num_pages);

    /* Step 3: Acquire compute resources (including VTCM and HMX) */
    compute_res_attr_t attr;
    unsigned int vtcm_size_to_use = (DEFAULT_VTCM_SIZE < vtcm_size_query) ? DEFAULT_VTCM_SIZE : vtcm_size_query;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_serialize(&attr, 0);
    HAP_compute_res_attr_set_cache_mode(&attr, 1);  // Enable cache mode (matching official implementation)
    HAP_compute_res_attr_set_vtcm_param_v2(&attr, vtcm_size_to_use, vtcm_size_to_use, vtcm_size_to_use); // single page (matching official implementation)
    HAP_compute_res_attr_set_release_callback(&attr, vtcm_release_callback, NULL);  // Enable release callback for cache mode
    HAP_compute_res_attr_set_hmx_param(&attr, 1);
    // Allocate VTCM for scratch pads
    g_compute_res_ctx_id = HAP_compute_res_acquire(&attr, 1000000);
    if (g_compute_res_ctx_id == 0) {
        GGMLHEXAGON_LOG_ERROR("HAP_compute_res_acquire failed, no VTCM available\n");
    } else {
        /* Using VTCM acquired via HAP_compute_res */
        void * vtcm_ptr = NULL;
        unsigned int vtcm_ptr_size = 0;
        if (HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &vtcm_ptr, &vtcm_ptr_size) != 0) {
            GGMLHEXAGON_LOG_INFO("HAP_compute_res_attr_get_vtcm_ptr_v2 failed\n");
            HAP_compute_res_release(g_compute_res_ctx_id);
            g_compute_res_ctx_id = 0;
        } else {
            g_vtcm_base = vtcm_ptr;
            g_vtcm_size = vtcm_ptr_size;
            GGMLHEXAGON_LOG_INFO("allocated VTCM pool via compute_res: %zu bytes at %p\n", g_vtcm_size, g_vtcm_base);

            //clear the VTCM region
            // TEMPORARILY DISABLED FOR DEBUGGING - memset(g_vtcm_base, 0, g_vtcm_size);
            // NOTE: HMX lock is managed per-operation in mulmat.c, not here
            //HAP_compute_res_hmx_lock(g_compute_res_ctx_id);
        }
    }

    /* Step 3.5: Create async HMX queue for pipeline overlap (DMA/HVX/HMX) */
    if (g_hmx_available && g_compute_res_ctx_id != 0) {
        if (g_hmx_queue != NULL) {
            GGMLHEXAGON_LOG_INFO("hmx_queue already exists, deleting old one\n");
            hmx_queue_delete(g_hmx_queue);
            g_hmx_queue = NULL;
        }
        g_hmx_queue = hmx_queue_create(16, g_compute_res_ctx_id);
        if (g_hmx_queue) {
            GGMLHEXAGON_LOG_INFO("async HMX queue created (capacity %u, rctx %u)\n",
                                 hmx_queue_capacity(g_hmx_queue), g_compute_res_ctx_id);
        } else {
            GGMLHEXAGON_LOG_INFO("hmx_queue_create failed, HMX path will run synchronously\n");
        }
    } else {
        GGMLHEXAGON_LOG_INFO("HMX not available (hmx=%d, rctx=%u), skipping hmx_queue creation\n",
                             g_hmx_available, g_compute_res_ctx_id);
    }

    /* Step 4: probe DSP memory for information only (no allocation) */
    {
        struct HAP_mem_stats mem_stats;
        memset(&mem_stats, 0, sizeof(mem_stats));
        int ret = HAP_mem_get_stats(&mem_stats);
        if (ret == 0) {
            FARF(ALWAYS, "DSP HAP_mem_stats: bytes_free=%llu, bytes_used=%llu, seg_free=%llu, seg_used=%llu",
                 (unsigned long long)mem_stats.bytes_free, (unsigned long long)mem_stats.bytes_used,
                 (unsigned long long)mem_stats.seg_free, (unsigned long long)mem_stats.seg_used);
        } else {
            FARF(ALWAYS, "HAP_mem_get_stats failed: %d", ret);
        }

        // Probe available DSP heap (information only, no allocation)
        size_t max_avail_mb = 0;
        for (int mb = 2048; mb >= 16; mb -= 16) {
            void * ptr = malloc((size_t)mb * 1024 * 1024);
            if (ptr) {
                FARF(ALWAYS, "DSP malloc probe: %d MB succeeded at %p", mb, ptr);
                free(ptr);
                max_avail_mb = mb;
                break;
            }
        }
        if (max_avail_mb == 0) {
            FARF(ALWAYS, "DSP malloc probe: even 16 MB failed!");
        } else {
            FARF(ALWAYS, "DSP malloc probe: max available = %zu MB (for work data only, cache uses ION)",
                 max_avail_mb);
        }
    }

    return 0;
}

int ggmlop_dsp_close(remote_handle64 handle) {
    if (handle)
        free((void*)handle);

    if (g_work_data != NULL) {
        free(g_work_data);
        g_work_data = NULL;
        g_work_size = 0;
    }

    // Cleanup htp_context resources (worker_pool + dma queues)
    if (g_htp_ctx.worker_pool) {
        worker_pool_release(&g_htp_ctx.worker_pool);
        g_htp_ctx.worker_pool = NULL;
    }
    for (int i = 0; i < HTP_MAX_NTHREADS; i++) {
        if (g_htp_ctx.dma[i]) {
            dma_queue_delete(g_htp_ctx.dma[i]);
            g_htp_ctx.dma[i] = NULL;
        }
    }

    if (g_hmx_queue != NULL) {
        hmx_queue_delete(g_hmx_queue);
        g_hmx_queue = NULL;
        GGMLHEXAGON_LOG_INFO("released async HMX queue");
    }

    if (g_compute_res_ctx_id != 0) {
        HAP_compute_res_release_cached(g_compute_res_ctx_id);
        // NOTE: HMX lock is managed per-operation in mulmat.c, not here
        // HAP_compute_res_hmx_unlock(g_compute_res_ctx_id);

        HAP_compute_res_release(g_compute_res_ctx_id);
        g_compute_res_ctx_id = 0;
        g_vtcm_base = NULL;
        g_vtcm_size = 0;
        GGMLHEXAGON_LOG_INFO("released compute resources");
    }

    return 0;
}

static AEEResult set_power_boost(remote_handle64 handle, uint32 on) {
    AEEResult res = AEE_SUCCESS;
    //Clear the structure to only update the selected fields
    HAP_power_request_t request = {0};
    void* rpcperf_ctx = (void*) handle;

    if(on) {
        request.type = HAP_power_set_DCVS_v3;
        request.dcvs_v3.set_dcvs_enable = TRUE;
        request.dcvs_v3.dcvs_enable = FALSE;  // keep DVFS disabled, only re-assert max corners
        request.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
        request.dcvs_v3.set_bus_params = TRUE;
        request.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
        request.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
        request.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
        request.dcvs_v3.set_core_params = TRUE;
        request.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;
        request.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
        request.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
        request.dcvs_v3.set_sleep_disable = TRUE;
        request.dcvs_v3.sleep_disable = TRUE;
        res = HAP_power_set(rpcperf_ctx, &request);
    } else {
        //These commands are to reset the voting done previously
        request.type = HAP_power_set_DCVS_v3;
        request.dcvs_v3.set_core_params = TRUE;
        res = HAP_power_set(rpcperf_ctx, &request);
    }
    if (res == HAP_POWER_ERR_UNKNOWN) {
        FARF(ERROR, "HAP_power_set FAILED, result 0x%x: Unknown\n", res);
        res = AEE_EUNKNOWN;
    } else if (res == HAP_POWER_ERR_INVALID_PARAM) {
        FARF(ERROR, "HAP_power_set FAILED, result 0x%x: Invalid Param\n", res);
        res = AEE_EBADPARM;
    } else if (res == HAP_POWER_ERR_UNSUPPORTED_API) {
        FARF(ERROR, "HAP_power_set FAILED, result 0x%x: Unsupported API\n", res);
        res = AEE_EUNSUPPORTED;
    }

    if(res != AEE_SUCCESS) {
        FARF(ERROR, "HAP_power_set FAILED! Attempting with HAP_power_set_DCVS_v2. This will reset the powerboost request.\n");
        HAP_power_request_t request = {0};
        request.type = HAP_power_set_DCVS_v2;
        res = HAP_power_set(rpcperf_ctx, &request);
        if(res != AEE_SUCCESS) {
            FARF(ERROR, "HAP_power_set FAILED, result 0x%x\n", res);
            res = AEE_EUNKNOWN;
        }
    }
    return res;
}

AEEResult hap_probe_dsp(remote_handle64 h) {
    int retVal = 0;

    unsigned int max_mips       = 0;
    unsigned int max_bus_bw     = 0;
    int client_class            = 0;
    unsigned int clk_freq_hz    = 0;
    boolean dcvs_enabled;
    void * context_ptr = NULL;

    HAP_power_response_t response;

    /*
     * HAP_utils_create_context : Creates a user client context
     * The client created with this API should be destroyed using
     * HAP_utils_destroy_context API.
     *
     * returns: void* ptr representing a unique context for the client
     */
    context_ptr = g_hexagon_power_ctx;

    /*
     * HAP_power_get : Queries the DSP for current performance levels
     * Input Parameters :
     *     context - this parameter is ignored and can be NULL for HAP_power_get function
     *     response - The power response for the system represented by HAP_power_response_t
     *
     * returns:  0 on success, non-zero error code in case of failure
     */
    /*
     * HAP_power_get_max_mips : Returns the maximum MIPS supported
     * output : max_mips
     */
    memset(&response, 0, sizeof(HAP_power_response_t));
    response.type = HAP_power_get_max_mips;
    retVal = HAP_power_get(context_ptr, &response);
    if (retVal!=AEE_SUCCESS) {
        FARF(ERROR, "Unable to get the maximum MIPS supported");
        return AEE_EFAILED;
    }

    max_mips = response.max_mips;
    /*
     * HAP_power_get_max_bus_bw : Returns the maximum bus bandwidth supported
     * output : max_bus_bw
     */
    memset(&response, 0, sizeof(HAP_power_response_t));
    response.type = HAP_power_get_max_bus_bw;
    retVal = HAP_power_get(context_ptr, &response);
    if (retVal!=AEE_SUCCESS) {
        FARF(ERROR, "Unable to get the maximum bus bandwidth supported");
        return AEE_EFAILED;
    }

    max_bus_bw = response.max_bus_bw;
    /*
     * HAP_power_get_client_class : Returns the client class:
     *     0x00 - Unknown Client Class
     *     0x01 - Audio Client Class
     *     0x02 - Voice Client Class
     *     0x04 - Compute Client Class
     *     0x08 - Camera Streaming with 1 HVX Client Class
     *     0x10 - Camera Streaming with 2 HVX Client Class
     *
     * output : client_class
     */
    memset(&response, 0, sizeof(HAP_power_response_t));
    response.type = HAP_power_get_client_class;
    retVal = HAP_power_get(context_ptr, &response);
    if (retVal!=AEE_SUCCESS) {
        FARF(ERROR, "Unable to get the client class");
        return AEE_EFAILED;
    }

    client_class = response.client_class;
    /*
     * HAP_power_get_clk_Freq : Returns the Core Clock Frequency
     * output : clk_freq_hz
     */
    memset(&response, 0, sizeof(HAP_power_response_t));
    response.type = HAP_power_get_clk_Freq;
    retVal = HAP_power_get(context_ptr, &response);
    if (retVal!=AEE_SUCCESS) {
        FARF(ERROR, "Unable to get the DSP core clock frequency");
        return AEE_EFAILED;
    }

    clk_freq_hz = response.clkFreqHz;
    /*
     * HAP_power_get_dcvsEnabled : Returns the DCVS status : 0 - disabled; 1 - enabled
     * output : dcvs_enabled
     */
    memset(&response, 0, sizeof(HAP_power_response_t));
    response.type = HAP_power_get_dcvsEnabled;
    retVal = HAP_power_get(context_ptr, &response);
    if (retVal!=AEE_SUCCESS) {
        FARF(ERROR, "Unable to get the DCVS status");
        return AEE_EFAILED;
    }

    dcvs_enabled = response.dcvsEnabled;
    printf("\nMaximum MIPS of DSP:             %u"
                 "\nMaximum Bus Bandwidth supported: %u Bytes/second(%u MiB/s)"
                 "\nClient Class:                    %x"
                 "\nCore clock frequency of the DSP: %u"
                 "\nDCVS status:                     %d\n\n",
                  max_mips, max_bus_bw, max_bus_bw >> 20, client_class, clk_freq_hz, dcvs_enabled);

    return AEE_SUCCESS;
}

AEEResult ggmlop_dsp_setclocks(remote_handle64 handle, int32 diag_info, int32 offload_cgraph_type, int32 mulmat_algo, int32 thread_counts) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__);

    GGMLHEXAGON_LOG_INFO("user specified thread_counts %d", thread_counts);
    if (thread_counts <= g_thread_counts) {
        g_thread_counts = thread_counts;
    }

    g_mulmat_algotype = mulmat_algo;
    GGMLHEXAGON_LOG_INFO("mulmat_algotype set to %d (0=HVX multithread,31=sgemm,32=HMX,33=VTCM multithread)", g_mulmat_algotype);
    g_offload_cgraph_type = offload_cgraph_type;
    GGMLHEXAGON_LOG_INFO("switch option %d", diag_info);
    g_dump_diag_info      = diag_info;

    printf("\n");
    printf("real thread_counts:             %d\n", g_thread_counts);
    printf("mulmat_algotype:                %d\n", g_mulmat_algotype);
    printf("offload_cgraph_type:            %d\n", offload_cgraph_type);
    printf("dump_diag_info:                 %d\n\n", g_dump_diag_info);

    // diag_info is now used for dump_diag_info (log control), so force HVX on
    ggml_type_traits_dsp_init(1);
    GGMLHEXAGON_LOG_INFO("ggml_dsp_use_hvx %d", 1);

    // Initialize htp_context for calling Qualcomm's execute_op.
    // Shares our already-acquired VTCM and HMX queue.
    if (g_thread_counts >= 1) {
        memset(&g_htp_ctx, 0, sizeof(g_htp_ctx));
        g_htp_ctx.vtcm_base      = (uint8_t *)g_vtcm_base;
        g_htp_ctx.vtcm_size      = g_vtcm_size;
        g_htp_ctx.vtcm_rctx      = g_compute_res_ctx_id;
        g_htp_ctx.hmx_queue      = g_hmx_queue;
        g_htp_ctx.n_threads      = (uint32_t)g_thread_counts;
        g_htp_ctx.hmx_enabled    = g_hmx_available ? true : false;

        AEEResult wp = worker_pool_init(&g_htp_ctx.worker_pool, (uint32_t)g_thread_counts);
        FARF(ALWAYS, "htp_ctx worker_pool_init returned %d (n_threads=%d)", wp, g_thread_counts);

        for (int i = 0; i < g_thread_counts; i++) {
            g_htp_ctx.dma[i] = dma_queue_create(256);
        }
        FARF(ALWAYS, "htp_ctx dma_queue created x%d", g_thread_counts);
    }

    g_hexagon_power_ctx = (void *)(handle);

    // Test VTCM memory read/write (must ensure VTCM is available in cache mode)
    if (g_vtcm_base != NULL) {
        // Ensure VTCM resource is available before accessing
        if (ggmlop_ensure_vtcm_available() == 0) {
            uint8_t *weight = (uint8_t *)g_vtcm_base;
            uint8_t *active = (uint8_t *)g_vtcm_base + 256;
            // Write test patterns
            memset(weight, 0xaa, 128);
            memset(active, 0xbb, 128);
            // Verify write
            if (weight[0] == 0xaa && active[0] == 0xbb) {
                GGMLHEXAGON_LOG_INFO("VTCM read/write test PASSED: weight[0]=0x%02x, active[0]=0x%02x", weight[0], active[0]);
            } else {
                GGMLHEXAGON_LOG_ERROR("VTCM read/write test FAILED: weight[0]=0x%02x, active[0]=0x%02x", weight[0], active[0]);
            }
        } else {
            GGMLHEXAGON_LOG_WARN("VTCM not available (cache mode), skipping VTCM test");
        }
    } else {
        GGMLHEXAGON_LOG_WARN("VTCM not available, skipping VTCM test");
    }

    hap_probe_dsp(handle);

    //set_power_boost(handle, 1);  //not needed

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return AEE_SUCCESS;
}

int ggmlop_get_mulmat_algotype(void) {
    return g_mulmat_algotype;
}

int ggmlop_get_thread_counts(void) {
    return g_thread_counts;
}

int ggmlop_get_offload_cgraph_type(void) {
    return g_offload_cgraph_type;
}

unsigned int ggmlop_get_compute_res_ctx_id(void) {
    return g_compute_res_ctx_id;
}

int ggmlop_is_hmx_available(void) {
    return g_hmx_available;
}

int ggmlop_is_dumpdiag_enabled(void) {
    return g_dump_diag_info;
}

struct hmx_queue * ggmlop_get_hmx_queue(void) {
    return g_hmx_queue;
}

bool ggmlop_is_ion_mode(void) {
    return g_ion_dsp_base != NULL;
}

void * ggmlop_get_work_data(size_t size) {
    // All callers (mulmat dispatch, flash-attn driver) invoke this from the
    // main thread before spawning workers, so the returned pointer stays
    // valid during worker execution.
    if (g_work_data == NULL || g_work_size < size) {
        if (g_work_data != NULL) {
            free(g_work_data);
        }
        size = (size > MAX_WORK_SIZE) ? MAX_WORK_SIZE : size;
        g_work_data = memalign(128, size);
        if (g_work_data != NULL) {
            g_work_size = size;
        }
    }
    return g_work_data;
}

void * ggmlop_get_vtcm_pool(size_t * size) {
    if (size != NULL) {
        *size = g_vtcm_size;
    }
    return g_vtcm_base;
}

// Allocate from the ION-based FP16 weight cache region
// Returns pointer to allocated region in ION shared memory, or NULL if full
void * ggmlop_cache_mempool_alloc(size_t size) {
    if (!g_ion_cache_base || size == 0) {
        return NULL;
    }
    // Align to 128 bytes (cache line size)
    size = (size + 127) & ~(size_t)127;
    if (g_ion_cache_offset + size > g_ion_cache_size) {
        FARF(ALWAYS, "ION cache: full, cannot allocate %zu bytes (offset=%zu, size=%zu)",
             size, g_ion_cache_offset, g_ion_cache_size);
        return NULL;
    }
    void * ptr = (char *)g_ion_cache_base + g_ion_cache_offset;
    g_ion_cache_offset += size;
    return ptr;
}


// Acquire VTCM for the current batch/op (cache mode).
// Called once at batch entry (ggmlop_dsp_execute_batch_ion) or at per-op entry
// (ggmlop_dsp_execute_task). Per-op mulmat/flash_attn code no longer calls this.
// If already valid, returns 0 immediately (cheap check).
// If needs_release was flagged by the release callback, release first, then re-acquire.
int ggmlop_ensure_vtcm_available(void) {
    if (g_compute_res_ctx_id == 0) {
        // compute_res acquire failed at init. VTCM is available only if the
        // legacy HAP_request_VTCM fallback succeeded (g_vtcm_base != NULL).
        // On unsigned PDs (e.g. domain 7) both paths fail and g_vtcm_base is NULL.
        return (g_vtcm_base != NULL) ? 0 : -1;
    }

    // Already valid - batch is running, keep using VTCM until batch boundary.
    // The release callback only sets needs_release; the actual release happens
    // in ggmlop_dsp_execute_batch_ion after the batch loop (lazy release).
    if (g_vtcm_valid) {
        return 0;
    }

    // First acquire or re-acquire after a lazy release at the previous batch boundary
    if (g_vtcm_needs_release) {
        GGMLHEXAGON_LOG_INFO("VTCM re-acquire (cache mode, batch boundary)");
        g_vtcm_needs_release = 0;
        HAP_compute_res_release_cached(g_compute_res_ctx_id);
    } else {
        GGMLHEXAGON_LOG_INFO("VTCM first acquire (cache mode)");
    }

    int err = HAP_compute_res_acquire_cached(g_compute_res_ctx_id, 1000000);
    if (err != 0) {
        GGMLHEXAGON_LOG_ERROR("Failed to acquire VTCM: 0x%08x", err);
        return -1;
    }
    g_vtcm_valid = 1;
    // Lower our priority so other sessions (e.g. QNN) can preempt and receive
    // release callbacks. Matches Qualcomm htp/main.c vtcm_acquire.
    HAP_compute_res_update_priority(g_compute_res_ctx_id,
                                    qurt_thread_get_priority(qurt_thread_get_id()) + 10);
    GGMLHEXAGON_LOG_INFO("VTCM acquired successfully");
    return 0;
}

// Release VTCM if the release callback flagged it (lazy release at batch boundary).
// Called after ggmlop_dsp_execute_batch_ion finishes its op loop.
static void ggmlop_vtcm_lazy_release(void) {
    if (g_compute_res_ctx_id != 0 && g_vtcm_needs_release) {
        g_vtcm_needs_release = 0;
        g_vtcm_valid = 0;
        HAP_compute_res_release_cached(g_compute_res_ctx_id);
        GGMLHEXAGON_LOG_INFO("VTCM released (lazy, batch boundary)");
    }
}

int ggmlop_dsp_execute_task(remote_handle64 h, int32 ggml_op, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__);

    if (!src0 || !dst) {
        GGMLHEXAGON_LOG_ERROR("invalid input: src0=%p, dst=%p", src0, dst);
        return AEE_EBADPARM;
    }

    GGMLHEXAGON_LOG_DEBUG("executing op type %d", ggml_op);

    // GGML_OP_NONE: register ION mempool on DSP side.
    // AP passes metadata: [0]=fd, [1..2]=size (bytes), [3]=size_mb, [4..5]=DSP VA from logcat
    // Strategy: use HAP_mmap2(fd) to get a DSP-user-space-accessible VA,
    //            same as QCOM's htp_iface_mmap() in htp/main.c.
    if (ggml_op == GGML_OP_NONE) {
        if (src0 && src0->data) {
            if (2 != g_offload_cgraph_type) {
                uint32_t * meta = (uint32_t *)src0->data;
                int32_t fd = (int32_t)meta[0];
                uint64_t size = ((uint64_t)(uint32_t)meta[2] << 32) | (uint64_t)(uint32_t)meta[1];
                int32_t size_mb = (int32_t)meta[3];
                g_ion_dsp_base = src0->data;
                GGMLHEXAGON_LOG_INFO("offload_cgraph_type=%d, registered ION DSP base: %p, data_len=%llu, fd=%d, size=%llubytes(%dMB)",
                                 g_offload_cgraph_type,
                                 g_ion_dsp_base, size, fd, (unsigned long long)size, size_mb);
            }
        } else {
            g_ion_dsp_base = NULL;
            GGMLHEXAGON_LOG_ERROR("GGML_OP_NONE: no src0 data");
        }
        return AEE_SUCCESS;
    }

    /* Per-op path: acquire VTCM once here (cheap if already valid).
     * mulmat/flash_attn no longer call ensure internally. */
    if (ggmlop_ensure_vtcm_available() != 0) {
        GGMLHEXAGON_LOG_ERROR("VTCM acquire failed for op %d", ggml_op);
        return AEE_EFAILED;
    }

    // Translation layer: map GGML op to HTP op, build octx, call execute_op
    enum htp_op_code htp_op;
    if (ggml_op_to_htp_op(ggml_op, dst->op_params, &htp_op) != 0) {
        GGMLHEXAGON_LOG_ERROR("unsupported op type: %d", ggml_op);
        return AEE_EUNSUPPORTED;
    }

    struct htp_ops_context octx;
    struct htp_tensor src_ht[HTP_OP_MAX_INPUTS];
    struct htp_tensor dst_ht;

    build_htp_octx(&octx, htp_op, dst->op_params, NULL,
                   src0, src1, NULL, NULL, dst, src_ht, &dst_ht);

    if (htp_op == HTP_OP_MUL_MAT) {
        if (build_mm_kernel_params(&octx) != 0) {
            return AEE_EFAILED;
        }
    }

    int op_ret = execute_op(&octx);

    octx.src0_spad.src = NULL;
    octx.src1_spad.src = NULL;
    octx.src2_spad.src = NULL;
    octx.src3_spad.src = NULL;
    octx.dst_spad.src  = NULL;

    if (op_ret != HTP_STATUS_OK) {
        GGMLHEXAGON_LOG_ERROR("execute_op returned %d (htp_op=%d)", op_ret, htp_op);
        return AEE_EFAILED;
    }

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__);
    return AEE_SUCCESS;
}


AEEResult ggmlop_dsp_execute_batch(remote_handle64 h, const dsp_opbatch_req* req) {
    //GGMLHEXAGON_LOG_DEBUG("enter %s", __func__);

    if (!req) {
        GGMLHEXAGON_LOG_ERROR("invalid input: req=%p", req);
        return AEE_EBADPARM;
    }

    if (req->n_tensors == 0 || req->n_ops == 0) {
        GGMLHEXAGON_LOG_ERROR("empty batch: n_tensors=%d, n_ops=%d", req->n_tensors, req->n_ops);
        return AEE_EBADPARM;
    }

    // req->tensors[] are dsptensor structs with data pointers already
    // translated from AP VA to DSP VA by FastRPC (same as per-op path).
    // No need for manual base+offset calculation or fd lookup.
    if (1 == g_dump_diag_info) {
        GGMLHEXAGON_LOG_INFO("batch: %d tensors, %d ops", req->n_tensors, req->n_ops);
    }

    // dispatch each op using pre-translated dsptensor pointers
    for (int i = 0; i < req->n_ops; i++) {
        const dsp_op_desc * op = &req->ops[i];

        if (op->src0_idx < 0 || op->src0_idx >= req->n_tensors ||
            op->dst_idx < 0  || op->dst_idx >= req->n_tensors) {
            GGMLHEXAGON_LOG_ERROR("op %d: invalid tensor indices src0=%d src1=%d dst=%d",
                                  i, op->src0_idx, op->src1_idx, op->dst_idx);
            return AEE_EBADPARM;
        }

        const dsptensor * src0_dt = &req->tensors[op->src0_idx];
        const dsptensor * src1_dt = (op->src1_idx >= 0) ? &req->tensors[op->src1_idx] : NULL;
        const dsptensor * src2_dt = (op->src2_idx >= 0) ? &req->tensors[op->src2_idx] : NULL;
        const dsptensor * src3_dt = (op->src3_idx >= 0) ? &req->tensors[op->src3_idx] : NULL;
        const dsptensor * dst_dt  = &req->tensors[op->dst_idx];

        if (1 == g_dump_diag_info) {
            // log tensor details and sample data for debugging
            GGMLHEXAGON_LOG_INFO("batch op %d: opcode=%d(%s), src0[t%d] data=%p ne=[%d,%d,%d,%d] nb=[%d,%d,%d,%d] type=%d len=%d",
                                 i, op->opcode, ggml_op_name(op->opcode),
                                 op->src0_idx, src0_dt->data,
                                 src0_dt->ne[0], src0_dt->ne[1], src0_dt->ne[2], src0_dt->ne[3],
                                 src0_dt->nb[0], src0_dt->nb[1], src0_dt->nb[2], src0_dt->nb[3],
                                 src0_dt->type, src0_dt->data_len);
            if (src1_dt) {
                GGMLHEXAGON_LOG_INFO("  src1[t%d] data=%p ne=[%d,%d,%d,%d] type=%d len=%d",
                                     op->src1_idx, src1_dt->data,
                                     src1_dt->ne[0], src1_dt->ne[1], src1_dt->ne[2], src1_dt->ne[3],
                                     src1_dt->type, src1_dt->data_len);
            }
            if (src2_dt) {
                GGMLHEXAGON_LOG_INFO("  src2[t%d] data=%p ne=[%d,%d,%d,%d] type=%d len=%d",
                                     op->src2_idx, src2_dt->data,
                                     src2_dt->ne[0], src2_dt->ne[1], src2_dt->ne[2], src2_dt->ne[3],
                                     src2_dt->type, src2_dt->data_len);
            }
            GGMLHEXAGON_LOG_INFO("  dst[t%d]  data=%p ne=[%d,%d,%d,%d] type=%d len=%d",
                                 op->dst_idx, dst_dt->data,
                                 dst_dt->ne[0], dst_dt->ne[1], dst_dt->ne[2], dst_dt->ne[3],
                                 dst_dt->type, dst_dt->data_len);

            // sample first few float values from src0 (for f32/f16 tensors)
            if (src0_dt->data && src0_dt->data_len >= 16) {
                const float * fdata = (const float *)src0_dt->data;
                GGMLHEXAGON_LOG_INFO("  src0 sample before: [%f, %f, %f, %f]",
                                     fdata[0], fdata[1], fdata[2], fdata[3]);
            }
        }

        // Translation layer: map GGML op to HTP op, build octx, call execute_op
        enum htp_op_code htp_op;
        if (ggml_op_to_htp_op(op->opcode, op->params, &htp_op) != 0) {
            GGMLHEXAGON_LOG_ERROR("batch op %d: unsupported opcode %d", i, op->opcode);
            return AEE_EUNSUPPORTED;
        }

        struct htp_ops_context octx;
        struct htp_tensor src_ht[HTP_OP_MAX_INPUTS];
        struct htp_tensor dst_ht;

        build_htp_octx(&octx, htp_op, op->params, NULL,
                       src0_dt, src1_dt, src2_dt, src3_dt,
                       dst_dt, src_ht, &dst_ht);

        if (htp_op == HTP_OP_MUL_MAT) {
            if (build_mm_kernel_params(&octx) != 0) {
                return AEE_EFAILED;
            }
        }

        int op_ret = execute_op(&octx);

        octx.src0_spad.src = NULL;
        octx.src1_spad.src = NULL;
        octx.src2_spad.src = NULL;
        octx.src3_spad.src = NULL;
        octx.dst_spad.src  = NULL;

        if (op_ret != HTP_STATUS_OK) {
            GGMLHEXAGON_LOG_ERROR("batch op %d: execute_op returned %d (htp_op=%d)",
                                  i, op_ret, htp_op);
            return AEE_EFAILED;
        }

        if (1 == g_dump_diag_info) {
            // sample dst after op execution
            if (dst_dt->data && dst_dt->data_len >= 16) {
                const float * fdata = (const float *)dst_dt->data;
                GGMLHEXAGON_LOG_INFO("  dst sample after: [%f, %f, %f, %f]",
                                 fdata[0], fdata[1], fdata[2], fdata[3]);
            }
        }
    }

    // [Direction-3 debug] ensure all DSP memory writes (especially HMX/DMA) are visible
    // before returning to FastRPC, which will copy data back to AP side.
    // Use same pattern as test-hmx.c: compiler barrier + volatile read to flush stores.
    __asm__ __volatile__("" ::: "memory");
    // force a volatile read on dst of last op to ensure writeback is committed
    if (req->n_ops > 0 && req->ops[req->n_ops - 1].dst_idx >= 0) {
        const dsptensor * last_dst = &req->tensors[req->ops[req->n_ops - 1].dst_idx];
        if (last_dst->data && last_dst->data_len >= 4) {
            (void) *(volatile const int *)(last_dst->data);
        }
    }
    __asm__ __volatile__("" ::: "memory");

    //GGMLHEXAGON_LOG_DEBUG("leave %s (dsp_execute_batch)", __func__);
    return AEE_SUCCESS;
}

// FastRPC IDL per-op methods (referenced by skel.c case 3/4)
int ggmlop_dsp_add(remote_handle64 h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) {
    GGML_UNUSED(h);
    enum htp_op_code htp_op = HTP_OP_ADD;
    struct htp_ops_context octx;
    struct htp_tensor src_ht[HTP_OP_MAX_INPUTS];
    struct htp_tensor dst_ht;
    build_htp_octx(&octx, htp_op, dst->op_params, NULL, src0, src1, NULL, NULL, dst, src_ht, &dst_ht);
    int ret = execute_op(&octx);
    octx.src0_spad.src = NULL; octx.src1_spad.src = NULL;
    octx.src2_spad.src = NULL; octx.src3_spad.src = NULL; octx.dst_spad.src = NULL;
    return ret;
}

int ggmlop_dsp_mulmat(remote_handle64 h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) {
    GGML_UNUSED(h);
    enum htp_op_code htp_op = HTP_OP_MUL_MAT;
    struct htp_ops_context octx;
    struct htp_tensor src_ht[HTP_OP_MAX_INPUTS];
    struct htp_tensor dst_ht;
    build_htp_octx(&octx, htp_op, dst->op_params, NULL, src0, src1, NULL, NULL, dst, src_ht, &dst_ht);
    if (build_mm_kernel_params(&octx) != 0) {
        return AEE_EFAILED;
    }
    int ret = execute_op(&octx);
    octx.src0_spad.src = NULL; octx.src1_spad.src = NULL;
    octx.src2_spad.src = NULL; octx.src3_spad.src = NULL; octx.dst_spad.src = NULL;
    return ret;
}

/*
 * ION-based batch execution: reads batch descriptor from shared ION memory.
 * FastRPC only passes 2 scalars (offset, size) - all data is in the mempool.
 *
 * Probe mode: when batch_size == 0, performs bidirectional ION memory test.
 */
AEEResult ggmlop_dsp_execute_batch_ion(remote_handle64 h, uint32_t batch_offset, uint32_t batch_size) {
    if (g_ion_dsp_base == NULL) {
        GGMLHEXAGON_LOG_ERROR("ION base not registered");
        return AEE_EBADPARM;
    }

    const char * base = (const char *)g_ion_dsp_base;

    /* Probe mode: verify bidirectional ION access */
    if (batch_size == 0) {
        GGMLHEXAGON_LOG_INFO("[DSP-PROBE] testing ION R/W at base=%p", g_ion_dsp_base);

        // Step 1: Read what AP wrote (AP->DSP direction)
        // Invalidate DSP cache before reading
        Q6_dccleaninva_A((void *)base);
        uint8_t ap_val = ((const uint8_t *)base)[0];
        GGMLHEXAGON_LOG_INFO("[DSP-PROBE] AP->DSP: read base+0 = 0x%02x", ap_val);

        // Step 2: Write pattern for AP to verify (DSP->AP direction)
        memset((void *)base, 0xAB, 16);
        memset((void *)(base + 64), 0xCD, 16);
        // Flush DSP L2 cache so AP can see the written data (ION is non-coherent)
        Q6_dccleaninva_A((void *)base);
        Q6_dccleaninva_A((void *)(base + 64));
        __asm__ __volatile__("" ::: "memory");
        return AEE_SUCCESS;
    }

    /* Cache setup mode: batch_size == 0xFFFF, batch_offset = cache_offset in ION */
    if (batch_size == 0xFFFF) {
        uint32_t cache_offset = batch_offset;
        if (cache_offset > 0 && g_ion_dsp_base != NULL && g_ion_dsp_size > 0) {
            g_ion_cache_base = (char *)g_ion_dsp_base + cache_offset;
            g_ion_cache_size = g_ion_dsp_size - (size_t)cache_offset;
            g_ion_cache_offset = 0;
            GGMLHEXAGON_LOG_INFO("[DSP-CACHE] FP16 weight cache: base=%p, offset=0x%x, size=%zu MB",
                                 g_ion_cache_base, cache_offset, g_ion_cache_size / (1024*1024));
        } else {
            g_ion_cache_base = NULL;
            g_ion_cache_size = 0;
            GGMLHEXAGON_LOG_WARN("[DSP-CACHE] no cache region (cache_offset=%u, ion_size=%zu)",
                                 cache_offset, g_ion_dsp_size);
        }
        return AEE_SUCCESS;
    }

    /* Cache reset mode: batch_size == 0xFFFE, clear FP16 weight cache */
    if (batch_size == 0xFFFE) {
        ggmlop_dsp_fp16_cache_reset();
        g_ion_cache_offset = 0;
        GGMLHEXAGON_LOG_INFO("[DSP-CACHE] FP16 weight cache reset");
        return AEE_SUCCESS;
    }

    /* Normal batch execution */
    /* Invalidate DSP cache for the batch descriptor before reading.
     * ION is non-coherent: AP reuses the mempool and writes a new batch
     * at the same offset, so DSP must invalidate to fetch fresh data.
     * Use dcinva (invalidate only) instead of dccleaninva (clean+invalidate):
     * dccleaninva would write back stale DSP cache lines to DRAM, overwriting
     * the fresh data AP just flushed via DC CVAC. */
    ggmlop_dsp_cache_inval_range((void *)(base + batch_offset), batch_size);
    const hex_batch_hdr * hdr = (const hex_batch_hdr *)(base + batch_offset);

    if (hdr->n_ops == 0 || hdr->n_tensors == 0) {
        GGMLHEXAGON_LOG_ERROR("empty ion-batch: n_ops=%u n_tensors=%u", hdr->n_ops, hdr->n_tensors);
        return AEE_EBADPARM;
    }

    const hex_op_desc * ops = (const hex_op_desc *)((const char *)hdr + hdr->ops_offset);
    const hex_tensor_desc * tens = (const hex_tensor_desc *)((const char *)hdr + hdr->tensors_offset);

    /* Per-batch VTCM acquire (matches Qualcomm htp/main.c opbatch pattern):
     * acquire once here, all ops in the batch share it, release at batch
     * boundary. Per-op mulmat/flash_attn no longer call ensure themselves. */
    if (ggmlop_ensure_vtcm_available() != 0) {
        GGMLHEXAGON_LOG_ERROR("VTCM acquire failed at batch entry, aborting batch");
        return AEE_EFAILED;
    }

    FARF(ALWAYS, "ion-batch: start n_ops=%u n_tensors=%u", hdr->n_ops, hdr->n_tensors);

    for (uint32_t i = 0; i < hdr->n_ops; i++) {
        const hex_op_desc * op = &ops[i];

        dsptensor src0_dt, src1_dt_buf, src2_dt_buf, src3_dt_buf, dst_dt;
        const dsptensor *src1_dt_ptr = NULL, *src2_dt_ptr = NULL, *src3_dt_ptr = NULL;

        /* Build src0 from hex_tensor_desc using ION base + offset */
        const hex_tensor_desc * t0 = &tens[op->src0_idx];
        memset(&src0_dt, 0, sizeof(src0_dt));
        src0_dt.type     = t0->type;
        memcpy(src0_dt.ne, t0->ne, sizeof(src0_dt.ne));
        memcpy(src0_dt.nb, t0->nb, sizeof(src0_dt.nb));
        memcpy(src0_dt.op_params, t0->op_params, sizeof(src0_dt.op_params));
        src0_dt.flags    = t0->flags;
        src0_dt.data     = (void *)(base + t0->data_offset);
        src0_dt.data_len = t0->data_len;

        if (1 == g_dump_diag_info) {
            /* DSP-side DIAG: dump first 4 f32 values from src0 data (BEFORE dcinva) */
            if (src0_dt.data && src0_dt.data_len >= 16) {
                const float * fv = (const float *)src0_dt.data;
                GGMLHEXAGON_LOG_INFO("[DSP-DIAG] op%u src0 PRE-INVAL off=0x%x ptr=%p f32=[%.4f, %.4f, %.4f, %.4f]",
                                 i, t0->data_offset, src0_dt.data, fv[0], fv[1], fv[2], fv[3]);
            }
        }

        if (op->src1_idx >= 0) {
            const hex_tensor_desc * t1 = &tens[op->src1_idx];
            memset(&src1_dt_buf, 0, sizeof(src1_dt_buf));
            src1_dt_buf.type     = t1->type;
            memcpy(src1_dt_buf.ne, t1->ne, sizeof(src1_dt_buf.ne));
            memcpy(src1_dt_buf.nb, t1->nb, sizeof(src1_dt_buf.nb));
            memcpy(src1_dt_buf.op_params, t1->op_params, sizeof(src1_dt_buf.op_params));
            src1_dt_buf.flags    = t1->flags;
            src1_dt_buf.data     = (void *)(base + t1->data_offset);
            src1_dt_buf.data_len = t1->data_len;
            src1_dt_ptr = &src1_dt_buf;

            if (1 == g_dump_diag_info) {
                if (src1_dt_buf.data && src1_dt_buf.data_len >= 16 && src1_dt_buf.type == 0) {
                    const float * fv = (const float *)src1_dt_buf.data;
                    GGMLHEXAGON_LOG_INFO("[DSP-DIAG] op%u src1 off=0x%x ptr=%p f32=[%.4f, %.4f, %.4f, %.4f] ne=[%d,%d,%d,%d]",
                                     i, t1->data_offset, src1_dt_buf.data, fv[0], fv[1], fv[2], fv[3],
                                     (int)src1_dt_buf.ne[0], (int)src1_dt_buf.ne[1], (int)src1_dt_buf.ne[2], (int)src1_dt_buf.ne[3]);
                }
            }
        }
        if (op->src2_idx >= 0) {
            const hex_tensor_desc * t2 = &tens[op->src2_idx];
            memset(&src2_dt_buf, 0, sizeof(src2_dt_buf));
            src2_dt_buf.type     = t2->type;
            memcpy(src2_dt_buf.ne, t2->ne, sizeof(src2_dt_buf.ne));
            memcpy(src2_dt_buf.nb, t2->nb, sizeof(src2_dt_buf.nb));
            memcpy(src2_dt_buf.op_params, t2->op_params, sizeof(src2_dt_buf.op_params));
            src2_dt_buf.flags    = t2->flags;
            src2_dt_buf.data     = (void *)(base + t2->data_offset);
            src2_dt_buf.data_len = t2->data_len;
            src2_dt_ptr = &src2_dt_buf;
        }
        if (op->src3_idx >= 0) {
            const hex_tensor_desc * t3 = &tens[op->src3_idx];
            memset(&src3_dt_buf, 0, sizeof(src3_dt_buf));
            src3_dt_buf.type     = t3->type;
            memcpy(src3_dt_buf.ne, t3->ne, sizeof(src3_dt_buf.ne));
            memcpy(src3_dt_buf.nb, t3->nb, sizeof(src3_dt_buf.nb));
            memcpy(src3_dt_buf.op_params, t3->op_params, sizeof(src3_dt_buf.op_params));
            src3_dt_buf.flags    = t3->flags;
            src3_dt_buf.data     = (void *)(base + t3->data_offset);
            src3_dt_buf.data_len = t3->data_len;
            src3_dt_ptr = &src3_dt_buf;
        }

        const hex_tensor_desc * td = &tens[op->dst_idx];
        memset(&dst_dt, 0, sizeof(dst_dt));
        dst_dt.type     = td->type;
        memcpy(dst_dt.ne, td->ne, sizeof(dst_dt.ne));
        memcpy(dst_dt.nb, td->nb, sizeof(dst_dt.nb));
        memcpy(dst_dt.op_params, td->op_params, sizeof(dst_dt.op_params));
        // Always override with op-level params (from node->op_params).
        // Confirmed: node->op_params is correct for all ops, but dst tensor's
        // op_params can be zero (ROPE, SOFT_MAX) or stale (SCALE in-place reuse).
        memcpy(dst_dt.op_params, op->params, sizeof(dst_dt.op_params));
        dst_dt.flags    = td->flags;
        dst_dt.data     = (void *)(base + td->data_offset);
        dst_dt.data_len = td->data_len;

        /* Cache maintenance for non-coherent ION memory:
         * - Invalidate DSP cache before reading src (AP wrote data into ION)
         * - Always invalidate, even for weights: ION region reuse means the
         *   same address may hold different data from a previous allocation
         * - Use dcinva (invalidate only), not dccleaninva: AP already flushed
         *   fresh src to DRAM via DC CVAC, so dccleaninva would write back
         *   stale DSP cache lines and clobber the fresh DRAM data. */
        ggmlop_dsp_cache_inval_range(src0_dt.data, src0_dt.data_len);
        if (src1_dt_ptr) ggmlop_dsp_cache_inval_range(src1_dt_buf.data, src1_dt_buf.data_len);
        if (src2_dt_ptr) ggmlop_dsp_cache_inval_range(src2_dt_buf.data, src2_dt_buf.data_len);
        if (src3_dt_ptr) ggmlop_dsp_cache_inval_range(src3_dt_buf.data, src3_dt_buf.data_len);

        if (1 == g_dump_diag_info) {
            /* DSP-side DIAG: dump first 4 f32 values from src0 data (AFTER dcinva).
             * Compare with PRE-INVAL values to detect stale cache lines. */
            if (src0_dt.data && src0_dt.data_len >= 16) {
                const float * fv = (const float *)src0_dt.data;
                float eps_f;
                memcpy(&eps_f, dst_dt.op_params, sizeof(float));
                GGMLHEXAGON_LOG_INFO("[DSP-DIAG] op%u src0 POST-INVAL off=0x%x ptr=%p f32=[%.4f, %.4f, %.4f, %.4f] eps=%f ne=[%d,%d,%d,%d]",
                                 i, t0->data_offset, src0_dt.data, fv[0], fv[1], fv[2], fv[3], eps_f,
                                 (int)src0_dt.ne[0], (int)src0_dt.ne[1], (int)src0_dt.ne[2], (int)src0_dt.ne[3]);
            }
        }

        FARF(ALWAYS, "ion-batch: op %u/%u opc=%d", i, hdr->n_ops, op->opcode);

        // Translation layer: map GGML op to HTP op, build octx, call execute_op
        enum htp_op_code htp_op;
        if (ggml_op_to_htp_op(op->opcode, op->params, &htp_op) != 0) {
            GGMLHEXAGON_LOG_ERROR("ion-op %u: unsupported opcode %d", i, op->opcode);
            return AEE_EUNSUPPORTED;
        }

        struct htp_ops_context octx;
        struct htp_tensor src_ht[HTP_OP_MAX_INPUTS];
        struct htp_tensor dst_ht;

        build_htp_octx(&octx, htp_op, op->params, op->kernel_params,
                       &src0_dt, src1_dt_ptr, src2_dt_ptr, src3_dt_ptr,
                       &dst_dt, src_ht, &dst_ht);

        if (htp_op == HTP_OP_MUL_MAT) {
            /* kernel_params already copied in build_htp_octx.
             * Fall back to DSP-side computation only when AP didn't precompute
             * (kernel_type == 0, e.g. per-op FastRPC path). */
            const int32_t kp_kernel_type = octx.kernel_params[0];
            if (kp_kernel_type == 0) {
                if (build_mm_kernel_params(&octx) != 0) {
                    return AEE_EFAILED;
                }
            }
        }

        /* F32 MUL_MAT diagnostic: dump src0 row 0/16, src1 row 0, dst[16] BEFORE execute_op.
         * Case 1 (m=32,n=14,k=64): src0 nb[1]=256, so row 16 = +1024 floats. */
        if (htp_op == HTP_OP_MUL_MAT && src0_dt.type == 0 /*F32*/ &&
            src0_dt.data && src0_dt.data_len >= (size_t)(17 * 256) &&
            src1_dt_ptr && src1_dt_buf.data && src1_dt_buf.data_len >= 16 &&
            dst_dt.data && dst_dt.data_len >= (size_t)(17 * 4)) {
            const float * s0  = (const float *) src0_dt.data;
            const float * s1  = (const float *) src1_dt_buf.data;
            const float * dp  = (const float *) dst_dt.data;
            const uint32_t s0_row16_off = src0_dt.nb[1] * 16 / 4;
            /* htp_mm_kernel_params layout (see matmul-ops.h):
             *   [0]kernel_type [1]pipeline [2]m_chunk [3]n_chunk [4]n_threads
             *   [5]n_act_threads [6]n_hmx [7]n_prefetch [8]tile_size [9]aligned_tile_size
             *   [10]src1_row_size [11]vtcm_size [12]vtcm_src0_size [13]vtcm_src1_size
             *   [14]vtcm_src2_size [15]vtcm_src3_size [16]vtcm_dst_size */
            GGMLHEXAGON_LOG_ERROR("[DSP-MM-PRE] op%u kp_type=%d s0r0=[%.4f,%.4f,%.4f,%.4f] s0r16=[%.4f,%.4f,%.4f,%.4f] s1r0=[%.4f,%.4f,%.4f,%.4f] dst16=[%.4f,%.4f,%.4f,%.4f] nb=[%u,%u,%u,%u] ne=[%u,%u,%u,%u]",
                i, octx.kernel_params[0],
                s0[0], s0[1], s0[2], s0[3],
                s0[s0_row16_off+0], s0[s0_row16_off+1], s0[s0_row16_off+2], s0[s0_row16_off+3],
                s1[0], s1[1], s1[2], s1[3],
                dp[16], dp[17], dp[18], dp[19],
                src0_dt.nb[0], src0_dt.nb[1], src0_dt.nb[2], src0_dt.nb[3],
                src0_dt.ne[0], src0_dt.ne[1], src0_dt.ne[2], src0_dt.ne[3]);
            GGMLHEXAGON_LOG_ERROR("[DSP-MM-KP]  op%u ktype=%d pipe=%d mch=%d nch=%d nthr=%d nact=%d nhmx=%d npf=%d src1rs=%d vtcm_sz=%d src0_sz=%d src1_sz=%d dst_sz=%d",
                i,
                octx.kernel_params[0],  /* kernel_type */
                octx.kernel_params[1],  /* pipeline */
                octx.kernel_params[2],  /* m_chunk */
                octx.kernel_params[3],  /* n_chunk */
                octx.kernel_params[4],  /* n_threads */
                octx.kernel_params[5],  /* n_act_threads */
                octx.kernel_params[6],  /* n_hmx */
                octx.kernel_params[7],  /* n_prefetch */
                octx.kernel_params[10], /* src1_row_size */
                octx.kernel_params[11], /* vtcm_size */
                octx.kernel_params[12], /* vtcm_src0_size */
                octx.kernel_params[13], /* vtcm_src1_size */
                octx.kernel_params[16]);/* vtcm_dst_size */
        }

        int op_ret = execute_op(&octx);

        /* F32 MUL_MAT diagnostic: dump dst[0..3] and dst[16..19] AFTER execute_op.
         * Locates whether NaN at index 16 originates in execute_op. */
        if (htp_op == HTP_OP_MUL_MAT && src0_dt.type == 0 /*F32*/ &&
            dst_dt.data && dst_dt.data_len >= (size_t)(20 * 4)) {
            const float * dp = (const float *) dst_dt.data;
            GGMLHEXAGON_LOG_ERROR("[DSP-MM-POST] op%u d[0..3]=[%.4f,%.4f,%.4f,%.4f] d[16..19]=[%.4f,%.4f,%.4f,%.4f]",
                i, dp[0], dp[1], dp[2], dp[3], dp[16], dp[17], dp[18], dp[19]);
        }

        // Clear spad refs (matches proc_op_req post-execute cleanup)
        octx.src0_spad.src = NULL;
        octx.src1_spad.src = NULL;
        octx.src2_spad.src = NULL;
        octx.src3_spad.src = NULL;
        octx.dst_spad.src  = NULL;

        if (op_ret != HTP_STATUS_OK) {
            const char * st_name =
                (op_ret == HTP_STATUS_INTERNAL_ERR)   ? "INTERNAL_ERR"   :
                (op_ret == HTP_STATUS_NO_SUPPORT)    ? "NO_SUPPORT"     :
                (op_ret == HTP_STATUS_INVAL_PARAMS)  ? "INVAL_PARAMS"   :
                (op_ret == HTP_STATUS_VTCM_TOO_SMALL) ? "VTCM_TOO_SMALL" : "UNKNOWN";
            GGMLHEXAGON_LOG_ERROR("ion-op %u: execute_op returned %d/%s (htp_op=%d)",
                                  i, op_ret, st_name, htp_op);
            return AEE_EFAILED;
        }

        FARF(ALWAYS, "ion-batch: op %u done, flushing %zuB", i, dst_dt.data_len);

        /* Flush DSP cache after writing dst (so AP can read from DRAM) */
        ggmlop_dsp_cache_flush_range(dst_dt.data, dst_dt.data_len);

        if (1 == g_dump_diag_info) {
            /* DSP-side DIAG: dump first 4 f32 values from dst data */
            if (dst_dt.data && dst_dt.data_len >= 16) {
                const float * fv = (const float *)dst_dt.data;
                GGMLHEXAGON_LOG_INFO("[DSP-DIAG] op%u dst  off=0x%x ptr=%p f32=[%.4f, %.4f, %.4f, %.4f]",
                                 i, tens[op->dst_idx].data_offset, dst_dt.data, fv[0], fv[1], fv[2], fv[3]);
            }
        }
    }

    FARF(ALWAYS, "ion-batch: all %u ops done", hdr->n_ops);

    __asm__ __volatile__("" ::: "memory");
    if (hdr->n_ops > 0 && ops[hdr->n_ops - 1].dst_idx < hdr->n_tensors) {
        uint32_t last_off = tens[ops[hdr->n_ops - 1].dst_idx].data_offset;
        if (batch_size > last_off + 4)
            (void) *(volatile const int *)(base + last_off);
    }
    __asm__ __volatile__("" ::: "memory");

    /* Lazy VTCM release: if the release callback flagged us during the batch,
     * release now so other sessions (QNN/another GGML session) can use VTCM.
     * If not flagged, keep it cached for the next batch (avoids re-acquire). */
    ggmlop_vtcm_lazy_release();

    return AEE_SUCCESS;
}

AEEResult ggmlop_dsp_register_ion(remote_handle64 h, uint32_t ion_fd, uint32_t size_lo, uint32_t size_hi) {
    (void)h;
    int32_t fd = (int32_t)ion_fd;
    uint64_t size = ((uint64_t)size_hi << 32) | (uint64_t)size_lo;

    GGMLHEXAGON_LOG_INFO("[ION-REG] fd=%d, size=%llu bytes (%dMB)",
                         fd, (unsigned long long)size, (int32_t)(size >> 20));

#if __HVX_ARCH__ > 73
    void * va = HAP_mmap2(NULL, (size_t)size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#else
    void * va = HAP_mmap(NULL, (size_t)size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#endif

    if (va == (void *)-1) {
        g_ion_dsp_base = NULL;
        GGMLHEXAGON_LOG_ERROR("[ION-REG] HAP_mmap2 FAILED: returned -1 (fd=%d, size=%llu)", fd, (unsigned long long)size);
        return AEE_EFAILED;
    }

    g_ion_dsp_base = va;
    g_ion_dsp_size = (size_t)size;
    GGMLHEXAGON_LOG_INFO("[ION-REG] HAP_mmap2 OK: va=%p (fd=%d, size=%zuMB)", va, fd, g_ion_dsp_size / (1024*1024));

    // FP16 weight cache region will be set up via NONE op with cache metadata
    // (see GGML_OP_NONE handling in ggmlop_dsp_execute_task)
    return AEE_SUCCESS;
}
