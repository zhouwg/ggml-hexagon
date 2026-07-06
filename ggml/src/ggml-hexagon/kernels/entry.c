#include <hexagon_types.h>
#include <HAP_power.h>
#include <HAP_dcvs.h>
#include <HAP_mem.h>
#include <HAP_compute_res.h>
#include <math.h>
#include <qurt_mutex.h>

#include "ggml-dsp.h"
#include "ggml-ops.h"
#include "hmx-queue.h"
#include "../htp/htp-ctx.h"
#include "../htp/matmul-ops.h"
#include "../htp/flash-attn-ops.h"

#define MAX_WORK_SIZE                       (1024 * 1024 * 1024)
#define DEFAULT_VTCM_SIZE                   (8 * 1024 * 1024)

struct dsp_context *g_dsp_ctx = NULL;

bool ggmlop_is_ion_mode(void) {
    return g_dsp_ctx->ion_dsp_base != NULL;
}

void * ggmlop_get_work_data(size_t size) {
    qurt_mutex_lock((qurt_mutex_t *)&g_dsp_ctx->work_mutex);
    if (g_dsp_ctx->work_data == NULL || g_dsp_ctx->work_size < size) {
        if (g_dsp_ctx->work_data != NULL) {
            free(g_dsp_ctx->work_data);
        }
        if (size > MAX_WORK_SIZE) {
            size = MAX_WORK_SIZE;
        }
        g_dsp_ctx->work_data = memalign(128, size);
        if (g_dsp_ctx->work_data != NULL) {
            g_dsp_ctx->work_size = size;
        }
    }
    void * result = g_dsp_ctx->work_data;
    qurt_mutex_unlock((qurt_mutex_t *)&g_dsp_ctx->work_mutex);
    return result;
}

void * ggmlop_get_vtcm_pool(size_t * size) {
    if (size != NULL) {
        *size = g_dsp_ctx->vtcm_size;
    }
    return g_dsp_ctx->vtcm_base;
}

// Allocate from the ION-based FP16 weight cache region
void * ggmlop_cache_mempool_alloc(size_t size) {
    if (!g_dsp_ctx->ion_cache_base || size == 0) {
        return NULL;
    }
    size = (size + 127) & ~(size_t)127;
    if (g_dsp_ctx->ion_cache_offset + size > g_dsp_ctx->ion_cache_size) {
        FARF(ALWAYS, "ION cache: full, cannot allocate %zu bytes (offset=%zu, size=%zu)",
             size, g_dsp_ctx->ion_cache_offset, g_dsp_ctx->ion_cache_size);
        return NULL;
    }
    void * ptr = (char *)g_dsp_ctx->ion_cache_base + g_dsp_ctx->ion_cache_offset;
    g_dsp_ctx->ion_cache_offset += size;
    return ptr;
}

// VTCM release strategy:
//   0 = lazy release: release only when release callback fires (default).
//       Suitable when batches are small (1 op/batch) and overhead of
//       per-batch acquire/release would dominate.
//   1 = per-batch release: release after every batch (Qualcomm pattern).
//       Suitable when batches are large (many ops/batch) and overhead
//       is amortized. Use after all-ops-offload reduces batch count.
#define DSP_VTCM_PER_BATCH_RELEASE 0

// Per-batch VTCM acquire:
// Called at batch entry. If VTCM is not valid, acquire it via cached API.
// Drops priority so other sessions can preempt and send release callbacks.
static void dsp_vtcm_acquire(void) {
    if (!g_dsp_ctx->vtcm_valid) {
        int err = HAP_compute_res_acquire_cached(g_dsp_ctx->compute_res_ctx_id, 1000000u);
        if (err != 0) {
            GGMLHEXAGON_LOG_ERROR("Failed to acquire VTCM: 0x%08x", (unsigned)err);
            return;
        }
        g_dsp_ctx->vtcm_needs_release = 0;
        g_dsp_ctx->vtcm_valid = 1;
        HAP_compute_res_update_priority(g_dsp_ctx->compute_res_ctx_id,
                                        qurt_thread_get_priority(qurt_thread_get_id()) + 10);
        GGMLHEXAGON_LOG_INFO("VTCM acquired");
    }
}

// Per-batch VTCM release.
// When DSP_VTCM_PER_BATCH_RELEASE=1: always releases (Qualcomm pattern).
// When DSP_VTCM_PER_BATCH_RELEASE=0: lazy release, only when the release
// callback fired (holds VTCM across batches to avoid re-acquire overhead).
static void dsp_vtcm_release(void) {
#if DSP_VTCM_PER_BATCH_RELEASE
    if (g_dsp_ctx->vtcm_valid) {
        g_dsp_ctx->vtcm_valid = 0;
        g_dsp_ctx->vtcm_needs_release = 0;
        HAP_compute_res_release_cached(g_dsp_ctx->compute_res_ctx_id);
        GGMLHEXAGON_LOG_INFO("VTCM released");
    }
#else
    if (g_dsp_ctx->compute_res_ctx_id != 0 && g_dsp_ctx->vtcm_needs_release) {
        g_dsp_ctx->vtcm_needs_release = 0;
        g_dsp_ctx->vtcm_valid = 0;
        HAP_compute_res_release_cached(g_dsp_ctx->compute_res_ctx_id);
        GGMLHEXAGON_LOG_INFO("VTCM released (lazy, batch boundary)");
    }
#endif
}

// Legacy wrapper: kept for external callers (test-hmx.c, per-op path).
// Acquires VTCM if not already valid. Returns 0 on success, -1 on failure.
int ggmlop_ensure_vtcm_available(void) {
    if (g_dsp_ctx->compute_res_ctx_id == 0) {
        return (g_dsp_ctx->vtcm_base != NULL) ? 0 : -1;
    }
    if (g_dsp_ctx->vtcm_valid) {
        return 0;
    }
    int err = HAP_compute_res_acquire_cached(g_dsp_ctx->compute_res_ctx_id, 1000000);
    if (err != 0) {
        GGMLHEXAGON_LOG_ERROR("Failed to acquire VTCM: 0x%08x", err);
        return -1;
    }
    g_dsp_ctx->vtcm_valid = 1;
    g_dsp_ctx->vtcm_needs_release = 0;
    HAP_compute_res_update_priority(g_dsp_ctx->compute_res_ctx_id,
                                    qurt_thread_get_priority(qurt_thread_get_id()) + 10);
    GGMLHEXAGON_LOG_INFO("VTCM acquired successfully");
    return 0;
}

static int power_on_hvx_hmx(void) {
    HAP_power_request_t req;

    /* Set client class */
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_apptype;
    req.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
    if (HAP_power_set((void *)&g_dsp_ctx->power_ctx, &req) != 0) {
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

    if (HAP_power_set((void *)&g_dsp_ctx->power_ctx, &req) != 0) {
        GGMLHEXAGON_LOG_ERROR("HAP_power_set DCVS failed");
        return -2;
    }

    /* Power up HVX */
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_HVX;
    req.hvx.power_up = 1;
    if (HAP_power_set((void *)&g_dsp_ctx->power_ctx, &req) != 0) {
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
    if (HAP_power_set((void *)&g_dsp_ctx->power_ctx, &req) != 0) {
        GGMLHEXAGON_LOG_ERROR("HAP_power_set HMX_v2 failed, continuing without HMX");
        return -4;
    }
#else
    /* Power up HMX (legacy for older architectures) */
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_HMX;
    req.hmx.power_up = 1;
    if (HAP_power_set((void *)&g_dsp_ctx->power_ctx, &req) != 0) {
        GGMLHEXAGON_LOG_ERROR("HAP_power_set HMX failed, continuing without HMX");
        return -4;
    }
#endif

    GGMLHEXAGON_LOG_INFO("HAP_power_set for HVX and HMX succeeded");
    return 0;
}


static int vtcm_release_callback(unsigned int rctx, void * state) {
    // Async notification only: flag that another session wants VTCM.
    // Do NOT clear g_dsp_ctx->vtcm_valid here - the current batch keeps running
    // and releases VTCM at the batch boundary.
    g_dsp_ctx->vtcm_needs_release = 1;
    return 0;
}

int ggmlop_dsp_open(const char * uri, remote_handle64 * handle) {
    struct dsp_context * ctx = NULL;

    // Guard against double initialization
    if (g_dsp_ctx != NULL) {
        GGMLHEXAGON_LOG_ERROR("ggmlop_dsp_open: g_dsp_ctx already initialized");
        return AEE_EITEMBUSY;
    }

    ctx = (struct dsp_context *)calloc(1, sizeof(struct dsp_context));
    GGML_ASSERT(NULL != ctx);
    ctx->thread_counts      = 1;
    ctx->offload_cgraph_type = 2;
    ctx->htp_ctx = (struct htp_context *)calloc(1, sizeof(struct htp_context));
    GGML_ASSERT(NULL != ctx->htp_ctx);
    qurt_mutex_init((qurt_mutex_t *)&ctx->work_mutex);
    g_dsp_ctx = ctx;
    *handle = (remote_handle64)ctx;
    GGMLHEXAGON_LOG_INFO("uri %s", uri);

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
     g_dsp_ctx->thread_counts = mhwt.max_hthreads;

    /* Step 1: Power up HVX and HMX */
    int power_result = power_on_hvx_hmx();
    if (power_result != 0) {
        GGMLHEXAGON_LOG_INFO("power_on_hvx_hmx failed (%d), continuing without HMX", power_result);
        g_dsp_ctx->hmx_available = 0;
    } else {
        g_dsp_ctx->hmx_available = 1;
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
    if (totalBlock.page_list_len >= 1 && availBlock.page_list_len >= 2) {
        printf("Compute resource query ctd, (Size, num pages); total (0x%x, %d) Avail (0x%x, %d, 0x%x, %d)\n",
                                    totalBlock.page_list[0].page_size,
                                    totalBlock.page_list[0].num_pages,
                                    availBlock.page_list[0].page_size,
                                    availBlock.page_list[0].num_pages,
                                    availBlock.page_list[1].page_size,
                                    availBlock.page_list[1].num_pages);
    } else if (totalBlock.page_list_len >= 1 && availBlock.page_list_len >= 1) {
        printf("Compute resource query ctd, (Size, num pages); total (0x%x, %d) Avail (0x%x, %d)\n",
                                    totalBlock.page_list[0].page_size,
                                    totalBlock.page_list[0].num_pages,
                                    availBlock.page_list[0].page_size,
                                    availBlock.page_list[0].num_pages);
    } else {
        printf("Compute resource query ctd, no page list data available\n");
    }

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
    g_dsp_ctx->compute_res_ctx_id = HAP_compute_res_acquire(&attr, 1000000);
    if (g_dsp_ctx->compute_res_ctx_id == 0) {
        GGMLHEXAGON_LOG_ERROR("HAP_compute_res_acquire failed, no VTCM available\n");
    } else {
        /* Using VTCM acquired via HAP_compute_res */
        void * vtcm_ptr = NULL;
        unsigned int vtcm_ptr_size = 0;
        if (HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &vtcm_ptr, &vtcm_ptr_size) != 0) {
            GGMLHEXAGON_LOG_INFO("HAP_compute_res_attr_get_vtcm_ptr_v2 failed\n");
            HAP_compute_res_release(g_dsp_ctx->compute_res_ctx_id);
            g_dsp_ctx->compute_res_ctx_id = 0;
        } else {
            g_dsp_ctx->vtcm_base = vtcm_ptr;
            g_dsp_ctx->vtcm_size = vtcm_ptr_size;
            GGMLHEXAGON_LOG_INFO("allocated VTCM pool via compute_res: %zu bytes at %p\n", g_dsp_ctx->vtcm_size, g_dsp_ctx->vtcm_base);

            //clear the VTCM region
            // TEMPORARILY DISABLED FOR DEBUGGING - memset(g_dsp_ctx->vtcm_base, 0, g_dsp_ctx->vtcm_size);
            // NOTE: HMX lock is managed per-operation in mulmat.c, not here
            //HAP_compute_res_hmx_lock(g_dsp_ctx->compute_res_ctx_id);
        }
    }

    /* Step 3.5: Create async HMX queue for pipeline overlap (DMA/HVX/HMX) */
    if (g_dsp_ctx->hmx_available && g_dsp_ctx->compute_res_ctx_id != 0) {
        if (g_dsp_ctx->hmx_queue != NULL) {
            GGMLHEXAGON_LOG_INFO("hmx_queue already exists, deleting old one\n");
            hmx_queue_delete(g_dsp_ctx->hmx_queue);
            g_dsp_ctx->hmx_queue = NULL;
        }
        g_dsp_ctx->hmx_queue = hmx_queue_create(16, g_dsp_ctx->compute_res_ctx_id);
        if (g_dsp_ctx->hmx_queue) {
            GGMLHEXAGON_LOG_INFO("async HMX queue created (capacity %u, rctx %u)\n",
                                 hmx_queue_capacity(g_dsp_ctx->hmx_queue), g_dsp_ctx->compute_res_ctx_id);
        } else {
            GGMLHEXAGON_LOG_INFO("hmx_queue_create failed, HMX path will run synchronously\n");
        }
    } else {
        GGMLHEXAGON_LOG_INFO("HMX not available (hmx=%d, rctx=%u), skipping hmx_queue creation\n",
                             g_dsp_ctx->hmx_available, g_dsp_ctx->compute_res_ctx_id);
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
    struct dsp_context * ctx = (struct dsp_context *)handle;
    if (!ctx) return 0;

    if (ctx->work_data != NULL) {
        free(ctx->work_data);
        ctx->work_data = NULL;
        ctx->work_size = 0;
    }

    // Cleanup htp_context resources (worker_pool + dma queues)
    if (ctx->htp_ctx) {
        if (ctx->htp_ctx->worker_pool) {
            worker_pool_release(&ctx->htp_ctx->worker_pool);
            ctx->htp_ctx->worker_pool = NULL;
        }
        for (int i = 0; i < HTP_MAX_NTHREADS; i++) {
            if (ctx->htp_ctx->dma[i]) {
                dma_queue_delete(ctx->htp_ctx->dma[i]);
                ctx->htp_ctx->dma[i] = NULL;
            }
        }
        free(ctx->htp_ctx);
        ctx->htp_ctx = NULL;
    }

    if (ctx->hmx_queue != NULL) {
        hmx_queue_delete(ctx->hmx_queue);
        ctx->hmx_queue = NULL;
        GGMLHEXAGON_LOG_INFO("released async HMX queue");
    }

    if (ctx->compute_res_ctx_id != 0) {
        HAP_compute_res_release_cached(ctx->compute_res_ctx_id);
        // NOTE: HMX lock is managed per-operation in mulmat.c, not here
        // HAP_compute_res_hmx_unlock(ctx->compute_res_ctx_id);

        HAP_compute_res_release(ctx->compute_res_ctx_id);
        ctx->compute_res_ctx_id = 0;
        ctx->vtcm_base = NULL;
        ctx->vtcm_size = 0;
        GGMLHEXAGON_LOG_INFO("released compute resources");
    }

    g_dsp_ctx = NULL;
    qurt_mutex_destroy((qurt_mutex_t *)&ctx->work_mutex);
    free(ctx);
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
    context_ptr = g_dsp_ctx->hexagon_power_ctx;

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
    if (thread_counts <= g_dsp_ctx->thread_counts) {
        g_dsp_ctx->thread_counts = thread_counts;
    }

    g_dsp_ctx->mulmat_algotype = mulmat_algo;
    GGMLHEXAGON_LOG_INFO("mulmat_algotype set to %d (0=HVX multithread,31=sgemm,32=HMX,33=VTCM multithread)", g_dsp_ctx->mulmat_algotype);
    g_dsp_ctx->offload_cgraph_type = offload_cgraph_type;
    GGMLHEXAGON_LOG_INFO("switch option %d", diag_info);
    g_dsp_ctx->dump_diag_info      = diag_info;

    printf("\n");
    printf("real thread_counts:             %d\n", g_dsp_ctx->thread_counts);
    printf("mulmat_algotype:                %d\n", g_dsp_ctx->mulmat_algotype);
    printf("offload_cgraph_type:            %d\n", offload_cgraph_type);
    printf("dump_diag_info:                 %d\n\n", g_dsp_ctx->dump_diag_info);

    // diag_info is now used for dump_diag_info (log control), so force HVX on
    ggml_type_traits_dsp_init(1);
    GGMLHEXAGON_LOG_INFO("ggml_dsp_use_hvx %d", 1);

    // Initialize htp_context for calling Qualcomm's execute_op.
    // Shares our already-acquired VTCM and HMX queue.
    if (g_dsp_ctx->thread_counts >= 1) {
        memset(g_dsp_ctx->htp_ctx, 0, sizeof(*g_dsp_ctx->htp_ctx));
        g_dsp_ctx->htp_ctx->vtcm_base      = (uint8_t *)g_dsp_ctx->vtcm_base;
        g_dsp_ctx->htp_ctx->vtcm_size      = g_dsp_ctx->vtcm_size;
        g_dsp_ctx->htp_ctx->vtcm_rctx      = g_dsp_ctx->compute_res_ctx_id;
        g_dsp_ctx->htp_ctx->hmx_queue      = g_dsp_ctx->hmx_queue;
        g_dsp_ctx->htp_ctx->n_threads      = (uint32_t)g_dsp_ctx->thread_counts;
        g_dsp_ctx->htp_ctx->hmx_enabled    = g_dsp_ctx->hmx_available ? true : false;

        AEEResult wp = worker_pool_init(&g_dsp_ctx->htp_ctx->worker_pool, (uint32_t)g_dsp_ctx->thread_counts);
        FARF(ALWAYS, "htp_ctx worker_pool_init returned %d (n_threads=%d)", wp, g_dsp_ctx->thread_counts);

        for (int i = 0; i < g_dsp_ctx->thread_counts; i++) {
            g_dsp_ctx->htp_ctx->dma[i] = dma_queue_create(256);
        }
        FARF(ALWAYS, "htp_ctx dma_queue created x%d", g_dsp_ctx->thread_counts);
    }

    g_dsp_ctx->hexagon_power_ctx = (void *)(handle);

    // Test VTCM memory read/write (must ensure VTCM is available in cache mode)
    if (g_dsp_ctx->vtcm_base != NULL) {
        // Ensure VTCM resource is available before accessing
        if (ggmlop_ensure_vtcm_available() == 0) {
            uint8_t *weight = (uint8_t *)g_dsp_ctx->vtcm_base;
            uint8_t *active = (uint8_t *)g_dsp_ctx->vtcm_base + 256;
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

    set_power_boost(handle, 1);

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
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
        g_dsp_ctx->ion_dsp_base = NULL;
        GGMLHEXAGON_LOG_ERROR("[ION-REG] HAP_mmap2 FAILED: returned -1 (fd=%d, size=%llu)", fd, (unsigned long long)size);
        return AEE_EFAILED;
    }

    g_dsp_ctx->ion_dsp_base = va;
    g_dsp_ctx->ion_dsp_size = (size_t)size;
    GGMLHEXAGON_LOG_INFO("[ION-REG] HAP_mmap2 OK: va=%p (fd=%d, size=%zuMB)", va, fd, g_dsp_ctx->ion_dsp_size / (1024*1024));

    // FP16 weight cache region will be set up via NONE op with cache metadata
    // (see GGML_OP_NONE handling in ggmlop_dsp_execute_task)
    return AEE_SUCCESS;
}

// begin translation layer {

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
        case HTP_OP_FLASH_ATTN_EXT:
            return op_flash_attn_ext(octx);
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
// For GGML_OP_UNARY, op_params[0] selects the unary sub-op.
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
        case GGML_OP_FLASH_ATTN_EXT: *htp_op = HTP_OP_FLASH_ATTN_EXT; return 0;
        case GGML_OP_SOFT_MAX: *htp_op = HTP_OP_SOFTMAX;     return 0;
        case GGML_OP_SCALE:   *htp_op = HTP_OP_SCALE;       return 0;
        case GGML_OP_CONCAT:  *htp_op = HTP_OP_CONCAT;      return 0;
        case GGML_OP_CPY:     *htp_op = HTP_OP_CPY;         return 0;
        case GGML_OP_GET_ROWS: *htp_op = HTP_OP_GET_ROWS;   return 0;
        case GGML_OP_SET_ROWS: *htp_op = HTP_OP_SET_ROWS;   return 0;
        case GGML_OP_SUM_ROWS: *htp_op = HTP_OP_SUM_ROWS;   return 0;
        case GGML_OP_CONT:    *htp_op = HTP_OP_CPY;         return 0;
        case GGML_OP_REPEAT:  *htp_op = HTP_OP_REPEAT;       return 0;
        case GGML_OP_NORM:    *htp_op = HTP_OP_NORM;        return 0;
        case GGML_OP_L2_NORM: *htp_op = HTP_OP_L2_NORM;     return 0;
        case GGML_OP_SQR:     *htp_op = HTP_OP_SQR;         return 0;
        case GGML_OP_SQRT:    *htp_op = HTP_OP_SQRT;        return 0;
        case GGML_OP_ARGSORT: *htp_op = HTP_OP_ARGSORT;     return 0;
        case GGML_OP_PAD:     *htp_op = HTP_OP_PAD;         return 0;
        case GGML_OP_CUMSUM:  *htp_op = HTP_OP_CUMSUM;      return 0;
        case GGML_OP_FILL:    *htp_op = HTP_OP_FILL;        return 0;
        case GGML_OP_DIAG:    *htp_op = HTP_OP_DIAG;        return 0;
        case GGML_OP_TRI:     *htp_op = HTP_OP_TRI;         return 0;
        case GGML_OP_UNARY: {
            if (!op_params) {
                FARF(ERROR, "ggml_op_to_htp_op: UNARY missing op_params");
                return -1;
            }
            switch (op_params[0]) {
                case GGML_UNARY_OP_NEG:      *htp_op = HTP_OP_UNARY_NEG;      return 0;
                case GGML_UNARY_OP_TANH:     *htp_op = HTP_OP_UNARY_TANH;     return 0;
                case GGML_UNARY_OP_SIGMOID:  *htp_op = HTP_OP_UNARY_SIGMOID;  return 0;
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_QUICK: *htp_op = HTP_OP_UNARY_GELU;     return 0;
                case GGML_UNARY_OP_SILU:      *htp_op = HTP_OP_UNARY_SILU;     return 0;
                case GGML_UNARY_OP_EXP:       *htp_op = HTP_OP_UNARY_EXP;      return 0;
                case GGML_UNARY_OP_SOFTPLUS: *htp_op = HTP_OP_UNARY_SOFTPLUS; return 0;
                default:
                    FARF(ERROR, "ggml_op_to_htp_op: unsupported unary_op %d", op_params[0]);
                    return -1;
            }
        }
        case GGML_OP_GLU: {
            if (!op_params) {
                FARF(ERROR, "ggml_op_to_htp_op: GLU missing op_params");
                return -1;
            }
            switch (op_params[0]) {
                case GGML_GLU_OP_SWIGLU:     *htp_op = HTP_OP_GLU_SWIGLU;     return 0;
                case GGML_GLU_OP_SWIGLU_OAI: *htp_op = HTP_OP_GLU_SWIGLU_OAI; return 0;
                case GGML_GLU_OP_GEGLU:      *htp_op = HTP_OP_GLU_GEGLU;      return 0;
                default:
                    FARF(ERROR, "ggml_op_to_htp_op: unsupported glu_op %d", op_params[0]);
                    return -1;
            }
        }
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
    const dsptensor * const dsts[HTP_OP_MAX_OUTPUTS],
    struct htp_tensor src_ht[HTP_OP_MAX_INPUTS],
    struct htp_tensor dst_ht[HTP_OP_MAX_OUTPUTS]) {

    memset(octx, 0, sizeof(*octx));
    octx->ctx = g_dsp_ctx->htp_ctx;
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

    for (int i = 0; i < HTP_OP_MAX_OUTPUTS; i++) {
        if (dsts[i]) {
            dsptensor_to_htp_tensor(dsts[i], &dst_ht[i]);
            octx->dsts[i] = &dst_ht[i];
        } else {
            octx->dsts[i] = NULL;
        }
    }

    octx->n_threads = (uint32_t)g_dsp_ctx->thread_counts;
}

// Try HMX precompute (simple 2D path). Mirrors ggml_hexagon_precompute_hmx_mm_params
// without the grouped batched path (we don't use MUL_MAT_ID).
// Returns true on success, false to fall back to HVX.
static bool build_mm_hmx_params(struct htp_ops_context * octx,
                                struct htp_mm_kernel_params * kparams) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];

    const int      wtype = src0->type;
    const uint32_t ne00  = src0->ne[0];
    const uint32_t ne01  = src0->ne[1];
    const uint32_t ne02  = src0->ne[2];
    const uint32_t ne03  = src0->ne[3];
    const uint32_t ne10  = src1->ne[0];
    const uint32_t ne11  = src1->ne[1];
    const uint32_t ne12  = src1->ne[2];
    const uint32_t ne13  = src1->ne[3];

    const bool is_repack = (wtype == HTP_TYPE_Q4_0 || wtype == HTP_TYPE_Q4_1 ||
                            wtype == HTP_TYPE_Q8_0 || wtype == HTP_TYPE_IQ4_NL ||
                            wtype == HTP_TYPE_MXFP4);
    const bool is_hmx_wtype = (wtype == HTP_TYPE_F16 || wtype == HTP_TYPE_F32 || is_repack);
    if (!is_hmx_wtype) return false;

    const bool is_batched = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    const int ne00_padded = is_repack ? hex_round_up(ne00, 32) : (int) ne00;
    const int ne01_padded = is_repack ? hex_round_up(ne01, 32) : (int) ne01;
    const int ne11_padded = hex_round_up(ne11, 32);

    // Eligibility (mirrors ggml_hexagon_matmul_is_hmx_eligible)
    if (ne01_padded % 32 != 0) return false;
    if (ne00 % 32 != 0) return false;
    if (is_batched && wtype != HTP_TYPE_F16) return false;
    if (src0->nb[0] > src0->nb[1] || src1->nb[0] > src1->nb[1]) return false;
    if (ne11 <= HTP_MM_HMX_MIN_NROWS) return false;

    const uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size(wtype);
    const bool     pipeline          = htp_mm_hmx_pipeline(ne11);
    const int      n_threads         = (int) octx->n_threads;
    const size_t   vtcm_budget       = g_dsp_ctx->vtcm_size;

    size_t best_mblocks       = SIZE_MAX;
    int    best_act_threads   = 0;
    size_t best_m_chunk       = 0;
    size_t best_n_chunk       = 0;
    size_t best_vtcm_size     = 0;

    int act_threads = n_threads;
    while (act_threads >= 1) {
        const size_t act_f32_size = hex_align_up(
            (size_t) act_threads * HTP_MM_DMA_ACT_MULTIPLIER * ne00_padded * sizeof(float),
            HTP_MM_HMX_TILE_SIZE);
        const size_t overhead = 256 + act_f32_size;

        size_t cost_n = 0, cost_m = 0, cost_mn = 0;
        htp_mm_hmx_get_2d_chunk_costs(wtype, ne00_padded, pipeline, aligned_tile_size,
                                      &cost_n, &cost_m, &cost_mn);

        size_t m_chunk_cand = 0, n_chunk_cand = 0, vtcm_size_cand = 0;
        if (htp_mm_hmx_compute_chunks(vtcm_budget, overhead, cost_n, cost_m, cost_mn,
                                      (size_t) ne11_padded, (size_t) ne01_padded,
                                      (size_t) ne01_padded * HTP_MM_HMX_COST_W_DEQUANT,
                                      (size_t) ne11 * HTP_MM_HMX_COST_A_CONVERT,
                                      &m_chunk_cand, &n_chunk_cand, &vtcm_size_cand) == 0) {
            size_t exact_size = htp_mm_hmx_get_2d_vtcm_size(
                wtype, ne00_padded, m_chunk_cand, n_chunk_cand, pipeline,
                act_threads, aligned_tile_size);
            if (exact_size <= vtcm_budget) {
                size_t mblocks = ((size_t) ne11 + m_chunk_cand - 1) / m_chunk_cand;
                if (mblocks < best_mblocks ||
                    (mblocks == best_mblocks && act_threads > best_act_threads)) {
                    best_mblocks     = mblocks;
                    best_act_threads = act_threads;
                    best_m_chunk     = m_chunk_cand;
                    best_n_chunk     = n_chunk_cand;
                    best_vtcm_size   = exact_size;
                }
            }
        }
        if (act_threads == 1) break;
        act_threads /= 2;
    }

    if (best_act_threads == 0) return false;

    kparams->n_hmx             = 1;
    kparams->pipeline           = pipeline ? 1 : 0;
    kparams->m_chunk            = (int32_t) best_m_chunk;
    kparams->n_chunk            = (int32_t) best_n_chunk;
    kparams->n_threads          = n_threads;
    kparams->n_act_threads      = best_act_threads;
    kparams->tile_size          = (int32_t) htp_mm_get_weight_tile_size(wtype);
    kparams->aligned_tile_size  = (int32_t) aligned_tile_size;
    kparams->src1_row_size      = (int32_t)((wtype == HTP_TYPE_Q4_1)
                                            ? htp_mm_q8_1_tiled_row_size(ne10)
                                            : htp_mm_q8_0_tiled_row_size(ne10));
    kparams->vtcm_size          = (int32_t) best_vtcm_size;
    kparams->vtcm_src0_size     = 0;
    kparams->vtcm_src1_size     = 0;
    kparams->vtcm_dst_size      = 0;
    kparams->n_prefetch         = 16;
    kparams->kernel_type        = is_batched ? HTP_MM_KERNEL_HMX_F16_BATCHED
                                             : HTP_MM_KERNEL_HMX_2D;
    return true;
}

// Compute htp_mm_kernel_params on DSP side for MUL_MAT.
// Tries HMX first (if available), falls back to HVX F32/F16/quantized paths.
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

    // Try HMX first (mirrors ggml_hexagon_precompute_matmul_params: HMX-first, HVX-fallback)
    if (g_dsp_ctx->hmx_available && build_mm_hmx_params(octx, kparams)) {
        goto mm_finalize;
    }

    const bool is_batched  = (ne02 > 1) || (ne03 > 1);
    const bool is_permuted = (src0->nb[0] > src0->nb[1] || src0->nb[1] > src0->nb[2] || src0->nb[2] > src0->nb[3]) ||
                             (src1->nb[0] > src1->nb[1] || src1->nb[1] > src1->nb[2] || src1->nb[2] > src1->nb[3]);

    size_t vtcm_src0_size = 0, vtcm_src1_size = 0, vtcm_dst_size = 0;

    if (wtype == HTP_TYPE_F32) {
        size_t vtcm_size = htp_mm_hvx_get_vtcm_sizes(
            HTP_MM_KERNEL_HVX_F32_F32_VTCM, wtype, ne10, src1_nrows, octx->n_threads,
            dst->nb[1], src0->nb[1], src1->nb[1], 16,
            &vtcm_src0_size, &vtcm_src1_size, &vtcm_dst_size);

        if (!is_batched && !is_permuted && vtcm_size <= g_dsp_ctx->vtcm_size) {
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

        if (!is_batched && !is_permuted && vtcm_size <= g_dsp_ctx->vtcm_size) {
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
                if (total_size <= g_dsp_ctx->vtcm_size) {
                    best_n_prefetch = d;
                    break;
                }
            }
            if (best_n_prefetch == 2 && total_size > g_dsp_ctx->vtcm_size) {
                total_size = htp_mm_hvx_get_vtcm_sizes(
                    kparams->kernel_type, wtype, ne10, src1_nrows, octx->n_threads,
                    dst->nb[1], src0->nb[1], src1->nb[1], 2,
                    &vs0, &vs1, &vd);
            }
            kparams->n_prefetch = (int32_t) best_n_prefetch;

            if (total_size <= g_dsp_ctx->vtcm_size) {
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

mm_finalize:
    kparams->div_ne12_ne1 = init_fastdiv_values(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_values(ne11);
    kparams->div_r2       = init_fastdiv_values(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_values(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne11     = init_fastdiv_values(ne11);

    return 0;
}

// Build htp_fa_kernel_params on DSP side for FLASH_ATTN_EXT.
// Mirrors ggml_hexagon_precompute_flash_attn_params on AP side, using
// DSP-side globals (g_dsp_ctx->vtcm_size, g_dsp_ctx->thread_counts, g_dsp_ctx->hmx_available).
static int build_fa_kernel_params(struct htp_ops_context * octx) {
    const struct htp_tensor * q  = octx->src[0];
    const struct htp_tensor * k  = octx->src[1];
    const struct htp_tensor * v  = octx->src[2];
    const struct htp_tensor * mask = octx->src[3];
    const struct htp_tensor * dst = octx->dst;
    if (!q || !k || !v || !dst) return -1;
    FARF(ALWAYS, "build_fa: DK=%u DV=%u neq1=%u nek1=%u G=%u ktype=%d vtype=%d hmx=%d",
         q->ne[0], v->ne[0], q->ne[1], k->ne[1], q->ne[2]/k->ne[2],
         k->type, v->type, g_dsp_ctx->hmx_available);

    struct htp_fa_kernel_params * kparams =
        (struct htp_fa_kernel_params *) octx->kernel_params;
    memset(kparams, 0, sizeof(*kparams));

    const uint32_t DK = q->ne[0];
    const uint32_t DV = v->ne[0];
    const uint32_t neq1 = q->ne[1];
    const uint32_t nek1 = k->ne[1];
    const uint32_t n_kv_heads = k->ne[2];
    const uint32_t G = q->ne[2] / n_kv_heads;

    float scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&scale,         &octx->op_params[0], sizeof(float));
    memcpy(&max_bias,      &octx->op_params[1], sizeof(float));
    memcpy(&logit_softcap, &octx->op_params[2], sizeof(float));
    if (logit_softcap != 0.0f) scale /= logit_softcap;

    kparams->scale         = scale;
    kparams->max_bias      = max_bias;
    kparams->logit_softcap = logit_softcap;
    kparams->is_q_fp32     = (q->type == HTP_TYPE_F32) ? 1 : 0;
    kparams->is_dst_fp32   = (dst->type == HTP_TYPE_F32) ? 1 : 0;
    kparams->G             = G;

    // ALiBi: find largest power of 2 <= n_head, then compute slope bases.
    // AP uses std::pow(2, -x); here we use 2^x = exp(x * ln2) to avoid powf.
    // Always computed (matches AP): when max_bias = 0, m0 = m1 = 1.0.
    const float ln2 = 0.6931471805599453f;
    uint32_t n_head_log2 = 1;
    while (n_head_log2 * 2u <= q->ne[2]) n_head_log2 *= 2;
    kparams->n_head_log2 = n_head_log2;
    kparams->m0 = expf(-ln2 * max_bias / (float)n_head_log2);
    kparams->m1 = expf(-ln2 * (max_bias * 0.5f) / (float)n_head_log2);

    // HMX eligibility: k/v F16, DK/DV divisible by 64, enough tokens.
    bool hmx_eligible = false;
    if (g_dsp_ctx->hmx_available && k->type == HTP_TYPE_F16 && v->type == HTP_TYPE_F16) {
        if (DK % 64 == 0 && DV % 64 == 0 && !(DK <= 128 && neq1 < 5)) {
            hmx_eligible = true;
        }
    }

    if (hmx_eligible) {
        size_t Br = 0, Bc = 0;
        int ret = hmx_fa_find_chunk_size(&Br, &Bc, G, DK, DV, neq1, nek1,
                                         g_dsp_ctx->vtcm_size, g_dsp_ctx->thread_counts);
        if (ret == 0) {
            kparams->kernel_type = HTP_FA_KERNEL_HMX;
            kparams->Br          = (uint16_t)Br;
            kparams->Bc          = (uint16_t)Bc;
            kparams->n_kv_blocks = (uint16_t)((nek1 + Bc - 1) / Bc);
            kparams->n_threads   = (kparams->n_kv_blocks >= 3 && g_dsp_ctx->thread_counts >= 2)
                                    ? (uint8_t)g_dsp_ctx->thread_counts : 1;
            kparams->u.hmx.g_br      = hex_align_up(G * Br, 32);
            kparams->u.hmx.pipeline  = (kparams->n_kv_blocks >= 3 && g_dsp_ctx->thread_counts >= 2) ? 1 : 0;
            kparams->vtcm_size       = (uint32_t)hmx_fa_compute_vtcm_usage(
                G, DK, DV, Br, Bc, kparams->n_threads, kparams->u.hmx.pipeline != 0);

            const size_t row_vec_bytes = hex_align_up(Bc * sizeof(uint16_t), 256);
            kparams->u.hmx.row_buf_stride = row_vec_bytes / 128;
            const size_t m_line_bytes = hex_align_up(Bc * sizeof(uint16_t), 128);
            kparams->u.hmx.mask_buf_row_stride = m_line_bytes / sizeof(uint16_t);
            kparams->u.hmx.mask_broadcast = (mask && mask->ne[2] == 1) ? 1 : 0;
            kparams->u.hmx.div_G = init_fastdiv_values(G);
            if (mask) {
                kparams->src3_div2 = init_fastdiv_values(mask->ne[2]);
                kparams->src3_div3 = init_fastdiv_values(mask->ne[3]);
            }
            kparams->qrows = 0;
            kparams->qrows_per_thread = 0;
            return 0;
        }
    }

    // Fallback to HVX
    kparams->kernel_type    = HTP_FA_KERNEL_HVX;
    kparams->Br             = 1;
    kparams->Bc             = 64;
    kparams->n_kv_blocks    = (uint16_t)((k->ne[1] + 64 - 1) / 64);
    kparams->n_threads      = (uint8_t)g_dsp_ctx->thread_counts;
    kparams->vtcm_size      = (uint32_t)hvx_fa_compute_vtcm_usage(
        DK, DV, kparams->is_q_fp32 != 0, mask != NULL, g_dsp_ctx->thread_counts);

    kparams->u.hvx.size_q_row_padded = hex_round_up(q->ne[0] * (kparams->is_q_fp32 ? 4 : 2), 128);
    kparams->u.hvx.size_k_row_padded = hex_round_up(k->ne[0] * 2, 128);
    kparams->u.hvx.size_v_row_padded = hex_round_up(v->ne[0] * 2, 128);
    kparams->u.hvx.src0_div21     = init_fastdiv_values(q->ne[2] * q->ne[1]);
    kparams->u.hvx.src0_div1      = init_fastdiv_values(q->ne[1]);
    kparams->u.hvx.broadcast_rk2   = init_fastdiv_values(q->ne[2] / k->ne[2]);
    kparams->u.hvx.broadcast_rk3   = init_fastdiv_values(q->ne[3] / k->ne[3]);
    kparams->u.hvx.broadcast_rv2   = init_fastdiv_values(q->ne[2] / v->ne[2]);
    kparams->u.hvx.broadcast_rv3   = init_fastdiv_values(q->ne[3] / v->ne[3]);
    if (mask) {
        kparams->src3_div2 = init_fastdiv_values(mask->ne[2]);
        kparams->src3_div3 = init_fastdiv_values(mask->ne[3]);
    }
    kparams->qrows           = q->ne[1] * q->ne[2] * q->ne[3];
    kparams->qrows_per_thread = (kparams->qrows + g_dsp_ctx->thread_counts - 1) / g_dsp_ctx->thread_counts;
    return 0;
}
// end translation layer }

/*
 * Per-op FastRPC call
 */
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
        if (src0 && src0->data && src1 && src1->data) {
            if (2 != g_dsp_ctx->offload_cgraph_type) {
                // src0: ION pool base; src1: metadata [fd, size_lo, size_hi, size_mb]
                uint32_t * meta = (uint32_t *)src1->data;
                int32_t fd = (int32_t)meta[0];
                uint64_t size = ((uint64_t)(uint32_t)meta[2] << 32) | (uint64_t)(uint32_t)meta[1];
                int32_t size_mb = (int32_t)meta[3];
                g_dsp_ctx->ion_dsp_base = src0->data;
                GGMLHEXAGON_LOG_INFO("offload_cgraph_type=%d, registered ION DSP base: %p, data_len=%llu, fd=%d, size=%llubytes(%dMB)",
                                 g_dsp_ctx->offload_cgraph_type,
                                 g_dsp_ctx->ion_dsp_base, size, fd, (unsigned long long)size, size_mb);
            }
        } else {
            g_dsp_ctx->ion_dsp_base = NULL;
            GGMLHEXAGON_LOG_ERROR("GGML_OP_NONE: no src0 or src1 data");
        }
        return AEE_SUCCESS;
    }

    /* Per-op path: acquire VTCM once here (cheap if already valid).
     * mulmat/flash_attn no longer call ensure internally. */
    if (ggmlop_ensure_vtcm_available() != 0) {
        GGMLHEXAGON_LOG_ERROR("VTCM acquire failed for op %d", ggml_op);
        return AEE_EFAILED;
    }

    /* Dual-channel dispatch: mulmat_algotype == 29 uses Qualcomm execute_op,
     * other values use self-built kernels (ggmlop_dsp_*). */
    if ((g_dsp_ctx->mulmat_algotype != 29) || (g_dsp_ctx->offload_cgraph_type != 2)) {
        switch (ggml_op) {
            case GGML_OP_ADD:      ggmlop_dsp_add(h, src0, src1, dst); break;
            case GGML_OP_SUB:      ggmlop_dsp_sub(h, src0, src1, dst); break;
            case GGML_OP_MUL:      ggmlop_dsp_mul(h, src0, src1, dst); break;
            case GGML_OP_DIV:      ggmlop_dsp_div(h, src0, src1, dst); break;
            case GGML_OP_MUL_MAT:  ggmlop_dsp_mulmat(h, src0, src1, dst); break;
            case GGML_OP_RMS_NORM: ggmlop_dsp_rmsnorm(h, src0, src1, dst); break;
            case GGML_OP_ROPE:     ggmlop_dsp_rope(h, src0, src1, NULL, dst); break;
            case GGML_OP_SOFT_MAX: ggmlop_dsp_softmax(h, src0, src1, NULL, dst); break;
            case GGML_OP_UNARY:    ggmlop_dsp_silu(h, src0, src1, dst); break;
            case GGML_OP_SCALE:    ggmlop_dsp_scale(h, src0, dst); break;
            case GGML_OP_CPY:      ggmlop_dsp_cpy(h, src0, src1, dst); break;
            case GGML_OP_FLASH_ATTN_EXT:
                GGMLHEXAGON_LOG_ERROR("FLASH_ATTN_EXT not supported via per-task path; use batch");
                return AEE_EUNSUPPORTED;
            default:
                GGMLHEXAGON_LOG_ERROR("unsupported op type: %d", ggml_op);
                return AEE_EUNSUPPORTED;
        }

        GGMLHEXAGON_LOG_DEBUG("leave %s", __func__);
        return AEE_SUCCESS;
    }

    // Translation layer: map GGML op to HTP op, build octx, call execute_op
    enum htp_op_code htp_op;
    if (ggml_op_to_htp_op(ggml_op, dst->op_params, &htp_op) != 0) {
        GGMLHEXAGON_LOG_ERROR("unsupported op type: %d", ggml_op);
        return AEE_EUNSUPPORTED;
    }

    struct htp_ops_context octx;
    struct htp_tensor src_ht[HTP_OP_MAX_INPUTS];
    struct htp_tensor dst_ht[HTP_OP_MAX_OUTPUTS];
    const dsptensor * dsts[HTP_OP_MAX_OUTPUTS] = {dst};

    build_htp_octx(&octx, htp_op, dst->op_params, NULL,
                   src0, src1, NULL, NULL, dsts, src_ht, dst_ht);

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

/*
 * ION-based batch execution: reads batch descriptor from shared ION memory.
 * FastRPC only passes 2 scalars (offset, size) - all data is in the mempool.
 *
 * Probe mode: when batch_size == 0, performs bidirectional ION memory test.
 */
AEEResult ggmlop_dsp_execute_batch(remote_handle64 h, uint32_t batch_offset, uint32_t batch_size) {
    if (g_dsp_ctx->ion_dsp_base == NULL) {
        GGMLHEXAGON_LOG_ERROR("ION base not registered");
        return AEE_EBADPARM;
    }

    const char * base = (const char *)g_dsp_ctx->ion_dsp_base;

    /* Probe mode: verify bidirectional ION access */
    if (batch_size == 0) {
        GGMLHEXAGON_LOG_INFO("[DSP-PROBE] testing ION R/W at base=%p", g_dsp_ctx->ion_dsp_base);

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
        if (cache_offset > 0 && g_dsp_ctx->ion_dsp_base != NULL && g_dsp_ctx->ion_dsp_size > 0) {
            g_dsp_ctx->ion_cache_base = (char *)g_dsp_ctx->ion_dsp_base + cache_offset;
            g_dsp_ctx->ion_cache_size = g_dsp_ctx->ion_dsp_size - (size_t)cache_offset;
            g_dsp_ctx->ion_cache_offset = 0;
            GGMLHEXAGON_LOG_INFO("[DSP-CACHE] FP16 weight cache: base=%p, offset=0x%x, size=%zu MB",
                                 g_dsp_ctx->ion_cache_base, cache_offset, g_dsp_ctx->ion_cache_size / (1024*1024));
        } else {
            g_dsp_ctx->ion_cache_base = NULL;
            g_dsp_ctx->ion_cache_size = 0;
            GGMLHEXAGON_LOG_WARN("[DSP-CACHE] no cache region (cache_offset=%u, ion_size=%zu)",
                                 cache_offset, g_dsp_ctx->ion_dsp_size);
        }
        return AEE_SUCCESS;
    }

    /* Cache reset mode: batch_size == 0xFFFE, clear FP16 weight cache */
    if (batch_size == 0xFFFE) {
        ggmlop_dsp_fp16_cache_reset();
        g_dsp_ctx->ion_cache_offset = 0;
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

    /* Per-batch VTCM acquire:
     * acquire at batch start, release at batch end. Every batch does
     * acquire+release to be cooperative with other DSP sessions. */
    dsp_vtcm_acquire();
    if (!g_dsp_ctx->vtcm_valid) {
        GGMLHEXAGON_LOG_ERROR("VTCM acquire failed at batch entry, aborting batch");
        return AEE_EFAILED;
    }

    GGMLHEXAGON_LOG_INFO("ion-batch: start n_ops=%u n_tensors=%u", hdr->n_ops, hdr->n_tensors);

    /* Bulk dst flush: collect all dst ranges during the loop, then merge
     * and flush once after all ops complete. This avoids redundant
     * dccleaninva calls when multiple ops share the same dst tensor or
     * have adjacent dst ranges. Max 4 dsts per op. */
    struct { void *data; size_t len; } dst_flush_ranges[HTP_OP_MAX_OUTPUTS * 256];
    uint32_t n_dst_ranges = 0;

    for (uint32_t i = 0; i < hdr->n_ops; i++) {
        const hex_op_desc * op = &ops[i];

        dsptensor src0_dt, src1_dt_buf, src2_dt_buf, src3_dt_buf;
        dsptensor dst_dt_buf[HTP_OP_MAX_OUTPUTS];
        const dsptensor *src1_dt_ptr = NULL, *src2_dt_ptr = NULL, *src3_dt_ptr = NULL;
        const dsptensor * dst_dt_ptrs[HTP_OP_MAX_OUTPUTS] = {NULL};

        /* Build src0 from hex_tensor_desc using ION base + offset */
        const hex_tensor_desc * t0 = &tens[op->src_idx[0]];
        memset(&src0_dt, 0, sizeof(src0_dt));
        src0_dt.type     = t0->type;
        memcpy(src0_dt.ne, t0->ne, sizeof(src0_dt.ne));
        memcpy(src0_dt.nb, t0->nb, sizeof(src0_dt.nb));
        memcpy(src0_dt.op_params, t0->op_params, sizeof(src0_dt.op_params));
        src0_dt.flags    = t0->flags;
        src0_dt.data     = (void *)(base + t0->data_offset);
        src0_dt.data_len = t0->data_len;

        if (1 == g_dsp_ctx->dump_diag_info) {
            /* DSP-side DIAG: dump first 4 f32 values from src0 data (BEFORE dcinva) */
            if (src0_dt.data && src0_dt.data_len >= 16) {
                const float * fv = (const float *)src0_dt.data;
                GGMLHEXAGON_LOG_INFO("[DSP-DIAG] op%u src0 PRE-INVAL off=0x%x ptr=%p f32=[%.4f, %.4f, %.4f, %.4f]",
                                 i, t0->data_offset, src0_dt.data, fv[0], fv[1], fv[2], fv[3]);
            }
        }

        if (op->src_idx[1] >= 0) {
            const hex_tensor_desc * t1 = &tens[op->src_idx[1]];
            memset(&src1_dt_buf, 0, sizeof(src1_dt_buf));
            src1_dt_buf.type     = t1->type;
            memcpy(src1_dt_buf.ne, t1->ne, sizeof(src1_dt_buf.ne));
            memcpy(src1_dt_buf.nb, t1->nb, sizeof(src1_dt_buf.nb));
            memcpy(src1_dt_buf.op_params, t1->op_params, sizeof(src1_dt_buf.op_params));
            src1_dt_buf.flags    = t1->flags;
            src1_dt_buf.data     = (void *)(base + t1->data_offset);
            src1_dt_buf.data_len = t1->data_len;
            src1_dt_ptr = &src1_dt_buf;

            if (1 == g_dsp_ctx->dump_diag_info) {
                if (src1_dt_buf.data && src1_dt_buf.data_len >= 16 && src1_dt_buf.type == 0) {
                    const float * fv = (const float *)src1_dt_buf.data;
                    GGMLHEXAGON_LOG_INFO("[DSP-DIAG] op%u src1 off=0x%x ptr=%p f32=[%.4f, %.4f, %.4f, %.4f] ne=[%d,%d,%d,%d]",
                                     i, t1->data_offset, src1_dt_buf.data, fv[0], fv[1], fv[2], fv[3],
                                     (int)src1_dt_buf.ne[0], (int)src1_dt_buf.ne[1], (int)src1_dt_buf.ne[2], (int)src1_dt_buf.ne[3]);
                }
            }
        }
        if (op->src_idx[2] >= 0) {
            const hex_tensor_desc * t2 = &tens[op->src_idx[2]];
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
        if (op->src_idx[3] >= 0) {
            const hex_tensor_desc * t3 = &tens[op->src_idx[3]];
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

        /* Read all dst outputs (QKV/FFN fusion uses dst_idx[0..2] / dst_idx[0..1]).
         * Single-output ops only use dst_idx[0]; the rest are -1 (NULL). */
        for (int k = 0; k < HTP_OP_MAX_OUTPUTS; k++) {
            if (op->dst_idx[k] < 0) {
                dst_dt_ptrs[k] = NULL;
                continue;
            }
            const hex_tensor_desc * td = &tens[op->dst_idx[k]];
            memset(&dst_dt_buf[k], 0, sizeof(dst_dt_buf[k]));
            dst_dt_buf[k].type     = td->type;
            memcpy(dst_dt_buf[k].ne, td->ne, sizeof(dst_dt_buf[k].ne));
            memcpy(dst_dt_buf[k].nb, td->nb, sizeof(dst_dt_buf[k].nb));
            memcpy(dst_dt_buf[k].op_params, td->op_params, sizeof(dst_dt_buf[k].op_params));
            // Always override with op-level params (from node->op_params).
            // Confirmed: node->op_params is correct for all ops, but dst tensor's
            // op_params can be zero (ROPE, SOFT_MAX) or stale (SCALE in-place reuse).
            memcpy(dst_dt_buf[k].op_params, op->params, sizeof(dst_dt_buf[k].op_params));
            dst_dt_buf[k].flags    = td->flags;
            dst_dt_buf[k].data     = (void *)(base + td->data_offset);
            dst_dt_buf[k].data_len = td->data_len;
            dst_dt_ptrs[k] = &dst_dt_buf[k];
        }

        /* Cache maintenance for non-coherent ION memory:
         * - Invalidate DSP cache before reading src (AP wrote data into ION)
         * - Use dcinva (invalidate only), not dccleaninva: AP already flushed
         *   fresh src to DRAM via DC CVAC, so dccleaninva would write back
         *   stale DSP cache lines and clobber the fresh DRAM data. */
        ggmlop_dsp_cache_inval_range(src0_dt.data, src0_dt.data_len);
        if (src1_dt_ptr) {
            ggmlop_dsp_cache_inval_range(src1_dt_buf.data, src1_dt_buf.data_len);
        }
        if (src2_dt_ptr) {
            ggmlop_dsp_cache_inval_range(src2_dt_buf.data, src2_dt_buf.data_len);
        }
        if (src3_dt_ptr) {
            ggmlop_dsp_cache_inval_range(src3_dt_buf.data, src3_dt_buf.data_len);
        }

        if (1 == g_dsp_ctx->dump_diag_info) {
            /* DSP-side DIAG: dump first 4 f32 values from src0 data (AFTER dcinva).
             * Compare with PRE-INVAL values to detect stale cache lines. */
            if (src0_dt.data && src0_dt.data_len >= 16) {
                const float * fv = (const float *)src0_dt.data;
                float eps_f;
                memcpy(&eps_f, dst_dt_buf[0].op_params, sizeof(float));
                GGMLHEXAGON_LOG_INFO("[DSP-DIAG] op%u src0 POST-INVAL off=0x%x ptr=%p f32=[%.4f, %.4f, %.4f, %.4f] eps=%f ne=[%d,%d,%d,%d]",
                                 i, t0->data_offset, src0_dt.data, fv[0], fv[1], fv[2], fv[3], eps_f,
                                 (int)src0_dt.ne[0], (int)src0_dt.ne[1], (int)src0_dt.ne[2], (int)src0_dt.ne[3]);
            }
        }

        GGMLHEXAGON_LOG_INFO("ion-batch: op %u/%u opc=%d", i, hdr->n_ops, op->opcode);

        /* Dual-channel dispatch: mulmat_algotype == 29 uses Qualcomm execute_op,
         * other values use self-built kernels (ggmlop_dsp_*). */
        if (g_dsp_ctx->mulmat_algotype != 29) {
            int op_ret = 0;
            switch (op->opcode) {
                case GGML_OP_ADD:      op_ret = ggmlop_dsp_add(h, &src0_dt, src1_dt_ptr, &dst_dt_buf[0]); break;
                case GGML_OP_SUB:      op_ret = ggmlop_dsp_sub(h, &src0_dt, src1_dt_ptr, &dst_dt_buf[0]); break;
                case GGML_OP_MUL:      op_ret = ggmlop_dsp_mul(h, &src0_dt, src1_dt_ptr, &dst_dt_buf[0]); break;
                case GGML_OP_DIV:      op_ret = ggmlop_dsp_div(h, &src0_dt, src1_dt_ptr, &dst_dt_buf[0]); break;
                case GGML_OP_MUL_MAT:  op_ret = ggmlop_dsp_mulmat(h, &src0_dt, src1_dt_ptr, &dst_dt_buf[0]); break;
                case GGML_OP_RMS_NORM: op_ret = ggmlop_dsp_rmsnorm(h, &src0_dt, src1_dt_ptr, &dst_dt_buf[0]); break;
                case GGML_OP_ROPE:     op_ret = ggmlop_dsp_rope(h, &src0_dt, src1_dt_ptr, src2_dt_ptr, &dst_dt_buf[0]); break;
                case GGML_OP_SOFT_MAX: op_ret = ggmlop_dsp_softmax(h, &src0_dt, src1_dt_ptr, src2_dt_ptr, &dst_dt_buf[0]); break;
                case GGML_OP_UNARY:    op_ret = ggmlop_dsp_silu(h, &src0_dt, src1_dt_ptr, &dst_dt_buf[0]); break;
                case GGML_OP_SCALE:    op_ret = ggmlop_dsp_scale(h, &src0_dt, &dst_dt_buf[0]); break;
                case GGML_OP_CPY:      op_ret = ggmlop_dsp_cpy(h, &src0_dt, src1_dt_ptr, &dst_dt_buf[0]); break;
                case GGML_OP_CONCAT:   op_ret = ggmlop_dsp_concat(h, &src0_dt, src1_dt_ptr, &dst_dt_buf[0]); break;
                case GGML_OP_REPEAT:   op_ret = ggmlop_dsp_repeat(h, &src0_dt, src1_dt_ptr, &dst_dt_buf[0]); break;
                case GGML_OP_DIAG_MASK_INF: op_ret = ggmlop_dsp_diag_mask_inf(h, &src0_dt, src1_dt_ptr, &dst_dt_buf[0]); break;
                case GGML_OP_FLASH_ATTN_EXT: op_ret = ggmlop_dsp_flash_attn(h, &src0_dt, src1_dt_ptr, src2_dt_ptr, src3_dt_ptr, &dst_dt_buf[0]); break;
                default:
                    GGMLHEXAGON_LOG_ERROR("ion-op %u: unsupported opcode %d", i, op->opcode);
                    return AEE_EUNSUPPORTED;
            }
            if (op_ret != 0) {
                GGMLHEXAGON_LOG_ERROR("ion-op %u: self-built kernel returned %d (opcode=%d)",
                                      i, op_ret, op->opcode);
                return AEE_EFAILED;
            }

            GGMLHEXAGON_LOG_INFO("ion-batch: op %u done, collecting dst range", i);
            if (n_dst_ranges < HTP_OP_MAX_OUTPUTS * 256) {
                dst_flush_ranges[n_dst_ranges].data = dst_dt_buf[0].data;
                dst_flush_ranges[n_dst_ranges].len  = dst_dt_buf[0].data_len;
                n_dst_ranges++;
            }
            continue;
        }

        // Translation layer: map GGML op to HTP op, build octx, call execute_op.
        // For fused ops, AP sets htp_opcode directly (skip ggml_op_to_htp_op).
        enum htp_op_code htp_op;
        if (op->htp_opcode != 0) {
            htp_op = (enum htp_op_code) op->htp_opcode;
        } else if (ggml_op_to_htp_op(op->opcode, op->params, &htp_op) != 0) {
            GGMLHEXAGON_LOG_ERROR("ion-op %u: unsupported opcode %d", i, op->opcode);
            return AEE_EUNSUPPORTED;
        }

        struct htp_ops_context octx;
        struct htp_tensor src_ht[HTP_OP_MAX_INPUTS];
        struct htp_tensor dst_ht[HTP_OP_MAX_OUTPUTS];

        build_htp_octx(&octx, htp_op, op->params, op->kernel_params,
                       &src0_dt, src1_dt_ptr, src2_dt_ptr, src3_dt_ptr,
                       dst_dt_ptrs, src_ht, dst_ht);

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

        if (htp_op == HTP_OP_FLASH_ATTN_EXT) {
            /* htp_fa_kernel_params.kernel_type is at offset 0.
             * HTP_FA_KERNEL_UNSUPPORTED = 0 means AP didn't precompute. */
            const int32_t kp_kernel_type = octx.kernel_params[0];
            if (kp_kernel_type == HTP_FA_KERNEL_UNSUPPORTED) {
                if (build_fa_kernel_params(&octx) != 0) {
                    return AEE_EFAILED;
                }
            }
        }

        /* F32 MUL_MAT diagnostic: dump src0 row 0/16, src1 row 0, dst[16] BEFORE execute_op.
         * Case 1 (m=32,n=14,k=64): src0 nb[1]=256, so row 16 = +1024 floats. */
        if (htp_op == HTP_OP_MUL_MAT && src0_dt.type == 0 /*F32*/ &&
            src0_dt.data && src0_dt.data_len >= (size_t)(17 * 256) &&
            src1_dt_ptr && src1_dt_buf.data && src1_dt_buf.data_len >= 16 &&
            dst_dt_ptrs[0] && dst_dt_buf[0].data && dst_dt_buf[0].data_len >= (size_t)(17 * 4)) {
            const float * s0  = (const float *) src0_dt.data;
            const float * s1  = (const float *) src1_dt_buf.data;
            const float * dp  = (const float *) dst_dt_buf[0].data;
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
            dst_dt_ptrs[0] && dst_dt_buf[0].data && dst_dt_buf[0].data_len >= (size_t)(20 * 4)) {
            const float * dp = (const float *) dst_dt_buf[0].data;
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

        GGMLHEXAGON_LOG_INFO("ion-batch: op %u done, collecting dst ranges", i);

        /* Collect dst ranges for bulk flush after the loop */
        for (int k = 0; k < HTP_OP_MAX_OUTPUTS; k++) {
            if (dst_dt_ptrs[k] && n_dst_ranges < HTP_OP_MAX_OUTPUTS * 256) {
                dst_flush_ranges[n_dst_ranges].data = dst_dt_buf[k].data;
                dst_flush_ranges[n_dst_ranges].len  = dst_dt_buf[k].data_len;
                n_dst_ranges++;
            }
        }

        if (1 == g_dsp_ctx->dump_diag_info) {
            /* DSP-side DIAG: dump first 4 f32 values from each dst output */
            for (int k = 0; k < HTP_OP_MAX_OUTPUTS; k++) {
                if (dst_dt_ptrs[k] && dst_dt_buf[k].data && dst_dt_buf[k].data_len >= 16) {
                    const float * fv = (const float *)dst_dt_buf[k].data;
                    GGMLHEXAGON_LOG_INFO("[DSP-DIAG] op%u dst[%d] off=0x%x ptr=%p f32=[%.4f, %.4f, %.4f, %.4f]",
                                     i, k, tens[op->dst_idx[k]].data_offset, dst_dt_buf[k].data, fv[0], fv[1], fv[2], fv[3]);
                }
            }
        }
    }

    GGMLHEXAGON_LOG_INFO("ion-batch: all %u ops done, bulk flushing %u dst ranges", hdr->n_ops, n_dst_ranges);

    /* Bulk flush: merge overlapping/adjacent dst ranges and flush once per
     * contiguous region. This avoids per-op dccleaninva overhead when ops
     * share the same output tensor or write to adjacent ION regions. */
    if (n_dst_ranges > 0) {
        /* Sort by starting address */
        for (uint32_t j = 0; j < n_dst_ranges; j++) {
            for (uint32_t k = j + 1; k < n_dst_ranges; k++) {
                if (dst_flush_ranges[j].data > dst_flush_ranges[k].data) {
                    void * tmp_data = dst_flush_ranges[j].data;
                    size_t tmp_len  = dst_flush_ranges[j].len;
                    dst_flush_ranges[j].data = dst_flush_ranges[k].data;
                    dst_flush_ranges[j].len  = dst_flush_ranges[k].len;
                    dst_flush_ranges[k].data = tmp_data;
                    dst_flush_ranges[k].len  = tmp_len;
                }
            }
        }
        /* Merge and flush */
        void * merge_start = dst_flush_ranges[0].data;
        size_t merge_end   = (size_t)merge_start + dst_flush_ranges[0].len;
        uint32_t n_flushed = 0;
        for (uint32_t j = 1; j <= n_dst_ranges; j++) {
            size_t next_start = (j < n_dst_ranges) ? (size_t)dst_flush_ranges[j].data : merge_end + 128;
            if (next_start <= merge_end + 64) {
                /* Overlap or adjacent (within 64B cache line): merge */
                size_t next_end = (j < n_dst_ranges) ? next_start + dst_flush_ranges[j].len : merge_end;
                if (next_end > merge_end) merge_end = next_end;
            } else {
                /* Gap: flush current merged range */
                ggmlop_dsp_cache_flush_range(merge_start, merge_end - (size_t)merge_start);
                n_flushed++;
                if (j < n_dst_ranges) {
                    merge_start = dst_flush_ranges[j].data;
                    merge_end   = (size_t)merge_start + dst_flush_ranges[j].len;
                }
            }
        }
        GGMLHEXAGON_LOG_INFO("ion-batch: bulk flush done, %u ranges -> %u flushes", n_dst_ranges, n_flushed);
    }

    __asm__ __volatile__("" ::: "memory");
    if (hdr->n_ops > 0 && ops[hdr->n_ops - 1].dst_idx[0] < hdr->n_tensors) {
        uint32_t last_off = tens[ops[hdr->n_ops - 1].dst_idx[0]].data_offset;
        if (batch_size > last_off + 4)
            (void) *(volatile const int *)(base + last_off);
    }
    __asm__ __volatile__("" ::: "memory");

    /* Per-batch VTCM release: controlled by DSP_VTCM_PER_BATCH_RELEASE.
     * 0 (default): lazy release, only when callback fires.
     * 1: always release (Qualcomm pattern, for large-batch mode). */
    dsp_vtcm_release();

    return AEE_SUCCESS;
}
