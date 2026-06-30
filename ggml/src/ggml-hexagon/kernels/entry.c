#include <hexagon_types.h>
#include <HAP_power.h>
#include <HAP_dcvs.h>
#include <HAP_mem.h>
#include <HAP_compute_res.h>

#include "ggml-dsp.h"
#include "ggml-ops.h"
#include "worker_pool.h"
#include "hmx-queue.h"

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
        GGMLHEXAGON_LOG_INFO("HAP_compute_res_acquire failed, falling back to HAP_request_VTCM\n");
        /* Fallback to legacy VTCM allocation */
        g_vtcm_base = HAP_request_VTCM(DEFAULT_VTCM_SIZE, 0);
        if (g_vtcm_base != NULL) {
            g_vtcm_size = DEFAULT_VTCM_SIZE;
            GGMLHEXAGON_LOG_INFO("allocated VTCM pool via HAP_request_VTCM: %zu bytes at %p\n", g_vtcm_size, g_vtcm_base);
        } else {
            GGMLHEXAGON_LOG_INFO("failed to allocate VTCM pool, will allocate on demand\n");
        }
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
    } else if (g_vtcm_base != NULL) {
        HAP_release_VTCM(g_vtcm_base);
        g_vtcm_base = NULL;
        g_vtcm_size = 0;
        GGMLHEXAGON_LOG_INFO("released VTCM pool via HAP_request_VTCM");
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

    if (g_thread_counts >= 1) {
        AEEResult result = worker_pool_reinit_with_threads(g_thread_counts);
        FARF(HIGH, "worker_pool_reinit_with_threads returned %d", result);
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

    switch (ggml_op) {
        case GGML_OP_SUB:
            GGMLHEXAGON_LOG_DEBUG("executing GGML_OP_SUB task");
            ggmlop_dsp_sub(h, src0, src1, dst);
            break;
        case GGML_OP_ADD:
            GGMLHEXAGON_LOG_DEBUG("executing GGML_OP_ADD task");
            ggmlop_dsp_add(h, src0, src1, dst);
            break;
        case GGML_OP_MUL:
            GGMLHEXAGON_LOG_DEBUG("executing GGML_OP_MUL task");
            ggmlop_dsp_mul(h, src0, src1, dst);
            break;
        case GGML_OP_DIV:
            GGMLHEXAGON_LOG_DEBUG("executing GGML_OP_DIV task");
            ggmlop_dsp_div(h, src0, src1, dst);
            break;
        case GGML_OP_MUL_MAT:
            GGMLHEXAGON_LOG_DEBUG("executing GGML_OP_MUL_MAT task");
            ggmlop_dsp_mulmat(h, src0, src1, dst);
            break;
        case GGML_OP_RMS_NORM:
            GGMLHEXAGON_LOG_DEBUG("executing GGML_OP_RMS_NORM task");
            ggmlop_dsp_rmsnorm(h, src0, src1, dst);
            break;
        case GGML_OP_ROPE:
            GGMLHEXAGON_LOG_DEBUG("executing GGML_OP_ROPE task");
            ggmlop_dsp_rope(h, src0, src1, NULL, dst);
            break;
        case GGML_OP_SOFT_MAX:
            GGMLHEXAGON_LOG_DEBUG("executing GGML_OP_SOFT_MAX task");
            ggmlop_dsp_softmax(h, src0, src1, NULL, dst);
            break;
        case GGML_OP_UNARY:
            GGMLHEXAGON_LOG_DEBUG("executing GGML_OP_UNARY (SILU) task");
            ggmlop_dsp_silu(h, src0, src1, dst);
            break;
        case GGML_OP_SCALE:
            GGMLHEXAGON_LOG_DEBUG("executing GGML_OP_SCALE task");
            ggmlop_dsp_scale(h, src0, dst);
            break;
        case GGML_OP_CPY:
            GGMLHEXAGON_LOG_DEBUG("executing GGML_OP_CPY task");
            ggmlop_dsp_cpy(h, src0, src1, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            // Not supported through the per-task path (requires 4 src tensors).
            // Use the batch path (ggmlop_dsp_execute_batch / _ion) instead.
            GGMLHEXAGON_LOG_ERROR("FLASH_ATTN_EXT not supported via per-task path; use batch");
            return AEE_EUNSUPPORTED;
        case 168:  // Test HMX operation
            GGMLHEXAGON_LOG_INFO("executing TEST_HMX task (op=168)");
            GGMLHEXAGON_LOG_INFO("src0: data=%p, ne[0]=%d, ne[1]=%d", src0->data, src0->ne[0], src0->ne[1]);
            GGMLHEXAGON_LOG_INFO("src1: data=%p, ne[0]=%d, ne[1]=%d", src1->data, src1->ne[0], src1->ne[1]);
            ggmlop_dsp_test_hmx(h, src0, src1, dst);
            GGMLHEXAGON_LOG_INFO("TEST_HMX task completed");
            break;
        default:
            GGMLHEXAGON_LOG_ERROR("unsupported op type: %d", ggml_op);
            return AEE_EUNSUPPORTED;
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

        int op_ret = 0;
        switch (op->opcode) {
            case GGML_OP_SUB:
                ggmlop_dsp_sub(h, src0_dt, src1_dt, dst_dt);
                break;
            case GGML_OP_ADD:
                ggmlop_dsp_add(h, src0_dt, src1_dt, dst_dt);
                break;
            case GGML_OP_MUL:
                ggmlop_dsp_mul(h, src0_dt, src1_dt, dst_dt);
                break;
            case GGML_OP_DIV:
                ggmlop_dsp_div(h, src0_dt, src1_dt, dst_dt);
                break;
            case GGML_OP_MUL_MAT:
                op_ret = ggmlop_dsp_mulmat(h, src0_dt, src1_dt, dst_dt);
                break;
            case GGML_OP_RMS_NORM:
                op_ret = ggmlop_dsp_rmsnorm(h, src0_dt, src1_dt, dst_dt);
                break;
            case GGML_OP_ROPE:
                op_ret = ggmlop_dsp_rope(h, src0_dt, src1_dt, src2_dt, dst_dt);
                break;
            case GGML_OP_SOFT_MAX:
                op_ret = ggmlop_dsp_softmax(h, src0_dt, src1_dt, src2_dt, dst_dt);
                break;
            case GGML_OP_UNARY:
                op_ret = ggmlop_dsp_silu(h, src0_dt, src1_dt, dst_dt);
                break;
            case GGML_OP_SCALE:
                op_ret = ggmlop_dsp_scale(h, src0_dt, dst_dt);
                break;
            case GGML_OP_CPY:
                op_ret = ggmlop_dsp_cpy(h, src0_dt, src1_dt, dst_dt);
                break;
            case GGML_OP_FLASH_ATTN_EXT:
                // Q=src0, K=src1, V=src2, mask=src3 (optional)
                op_ret = ggmlop_dsp_flash_attn(h, src0_dt, src1_dt, src2_dt, src3_dt, dst_dt);
                break;
            default:
                GGMLHEXAGON_LOG_ERROR("batch op %d: unsupported opcode %d", i, op->opcode);
                return AEE_EUNSUPPORTED;
        }
        if (op_ret != 0) {
            GGMLHEXAGON_LOG_ERROR("batch op %d (%s) failed with ret=%d", i, ggml_op_name(op->opcode), op_ret);
            return op_ret;
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
     * Use dcinva (invalidate-only) instead of dccleaninva: the descriptor
     * region may have been a previous op's dst, and clean+invalidate would
     * write stale dirty data back to DRAM, corrupting the fresh descriptor. */
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

    /* Pass 1: invalidate all src tensor data before executing any op.
     * Replaces per-op dcinva+syncht with one batched syncht.
     * dcinva (invalidate-only) avoids writing stale dirty lines back to DRAM
     * when ION regions are reused across batches. */
    for (uint32_t i = 0; i < hdr->n_ops; i++) {
        const hex_op_desc * op = &ops[i];
        ggmlop_dsp_cache_inval_range_nosync(
            (void *)(base + tens[op->src0_idx].data_offset),
            tens[op->src0_idx].data_len);
        if (op->src1_idx >= 0) {
            ggmlop_dsp_cache_inval_range_nosync(
                (void *)(base + tens[op->src1_idx].data_offset),
                tens[op->src1_idx].data_len);
        }
        if (op->src2_idx >= 0) {
            ggmlop_dsp_cache_inval_range_nosync(
                (void *)(base + tens[op->src2_idx].data_offset),
                tens[op->src2_idx].data_len);
        }
        if (op->src3_idx >= 0) {
            ggmlop_dsp_cache_inval_range_nosync(
                (void *)(base + tens[op->src3_idx].data_offset),
                tens[op->src3_idx].data_len);
        }
    }
    __asm__ __volatile__("syncht\n");

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
            /* DSP-side DIAG: dump first 4 f32 values from src0 data */
            if (src0_dt.data && src0_dt.data_len >= 16) {
                const float * fv = (const float *)src0_dt.data;
                GGMLHEXAGON_LOG_INFO("[DSP-DIAG] op%u src0 off=0x%x ptr=%p f32=[%.4f, %.4f, %.4f, %.4f]",
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

        /* src cache invalidation is done in Pass 1 above (batched syncht) */

        int op_ret = 0;
        switch (op->opcode) {
            case GGML_OP_SUB:
                ggmlop_dsp_sub(h, &src0_dt, src1_dt_ptr, &dst_dt); break;
            case GGML_OP_ADD:
                ggmlop_dsp_add(h, &src0_dt, src1_dt_ptr, &dst_dt); break;
            case GGML_OP_MUL:
                ggmlop_dsp_mul(h, &src0_dt, src1_dt_ptr, &dst_dt); break;
            case GGML_OP_DIV:
                ggmlop_dsp_div(h, &src0_dt, src1_dt_ptr, &dst_dt); break;
            case GGML_OP_MUL_MAT:
                op_ret = ggmlop_dsp_mulmat(h, &src0_dt, src1_dt_ptr, &dst_dt); break;
            case GGML_OP_RMS_NORM:
                op_ret = ggmlop_dsp_rmsnorm(h, &src0_dt, src1_dt_ptr, &dst_dt); break;
            case GGML_OP_ROPE:
                op_ret = ggmlop_dsp_rope(h, &src0_dt, src1_dt_ptr, src2_dt_ptr, &dst_dt); break;
            case GGML_OP_SOFT_MAX:
                op_ret = ggmlop_dsp_softmax(h, &src0_dt, src1_dt_ptr, src2_dt_ptr, &dst_dt); break;
            case GGML_OP_UNARY:
                op_ret = ggmlop_dsp_silu(h, &src0_dt, src1_dt_ptr, &dst_dt); break;
            case GGML_OP_SCALE:
                op_ret = ggmlop_dsp_scale(h, &src0_dt, &dst_dt); break;
            case GGML_OP_CPY:
                op_ret = ggmlop_dsp_cpy(h, &src0_dt, src1_dt_ptr, &dst_dt); break;
            case GGML_OP_CONCAT:
                op_ret = ggmlop_dsp_concat(h, &src0_dt, src1_dt_ptr, &dst_dt); break;
            case GGML_OP_REPEAT:
                op_ret = ggmlop_dsp_repeat(h, &src0_dt, src1_dt_ptr, &dst_dt); break;
            case GGML_OP_DIAG_MASK_INF:
                op_ret = ggmlop_dsp_diag_mask_inf(h, &src0_dt, src1_dt_ptr, &dst_dt); break;
            case GGML_OP_FLASH_ATTN_EXT:
                // Q=src0, K=src1, V=src2, mask=src3 (optional, may be NULL).
                op_ret = ggmlop_dsp_flash_attn(h, &src0_dt, src1_dt_ptr, src2_dt_ptr, src3_dt_ptr, &dst_dt); break;
            default:
                GGMLHEXAGON_LOG_ERROR("ion-op %u: unsupported opcode %d", i, op->opcode);
                return AEE_EUNSUPPORTED;
        }
        if (op_ret != 0) return op_ret;

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

    /* Last op's cache_flush_range already issued syncht, ensuring all dst
     * writebacks complete before AP reads from DRAM. */

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
