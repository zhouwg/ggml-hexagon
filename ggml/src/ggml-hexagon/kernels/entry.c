#include <hexagon_types.h>
#include <HAP_power.h>
#include <HAP_dcvs.h>
#include <HAP_mem.h>
#include <HAP_compute_res.h>
#include "ggml-dsp.h"
#include "worker_pool.h"

static int g_thread_counts                  = 1;
static int g_mulmat_algotype                = 0;
static void * g_work_data                   = NULL;
static size_t g_work_size                   = 0;

static void * g_vtcm_base                   = NULL;
static size_t g_vtcm_size                   = 0;
static unsigned int g_compute_res_ctx_id    = 0;
static int g_power_ctx                      = 0;
static int g_hmx_available                  = 0;
static volatile int g_vtcm_needs_release    = 0;  // For cache mode VTCM management
static volatile int g_vtcm_valid            = 0;  // VTCM resource is currently valid/available

static void * g_hexagon_power_ctx           = NULL;
static void * g_ion_dsp_base                = NULL;

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
    req.dcvs_v3.dcvs_enable = 1;
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
    g_vtcm_needs_release = 1;
    g_vtcm_valid = 0;
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
        request.dcvs_v3.dcvs_enable = TRUE;
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
                 "\nDCVS status:                     %d",
                  max_mips, max_bus_bw, max_bus_bw >> 20, client_class, clk_freq_hz, dcvs_enabled);

}

AEEResult ggmlop_dsp_setclocks(remote_handle64 handle, int32 power_level, int32 latency, int32 mulmat_algo, int32 thread_counts) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__);

    GGMLHEXAGON_LOG_INFO("user specified thread_counts %d", thread_counts);
    if (thread_counts <= g_thread_counts) {
        g_thread_counts = thread_counts;
    }
    GGMLHEXAGON_LOG_INFO("real thread_counts %d", g_thread_counts);

    g_mulmat_algotype = mulmat_algo;
    GGMLHEXAGON_LOG_INFO("mulmat_algotype %d", g_mulmat_algotype);
    FARF(ALWAYS, "mulmat_algotype set to %d (0=auto, 32=VTCM+HMX, 33=VTCM multithread)", g_mulmat_algotype);

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

    //set_power_boost(handle, 1);

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return AEE_SUCCESS;
}

int ggmlop_get_mulmat_algotype(void) {
    return g_mulmat_algotype;
}

int ggmlop_get_thread_counts(void) {
    return g_thread_counts;
}

unsigned int ggmlop_get_compute_res_ctx_id(void) {
    return g_compute_res_ctx_id;
}

int ggmlop_is_hmx_available(void) {
    return g_hmx_available;
}

void * ggmlop_get_work_data(size_t size) {
    if (g_work_data == NULL || g_work_size < size) {
        if (g_work_data != NULL) {
            free(g_work_data);
        }
        size = (size > MAX_WORK_SIZE) ? MAX_WORK_SIZE : size;
        g_work_data = malloc(size);
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


// Ensure VTCM resource is available (for cache mode)
// Must be called before using VTCM in each operation
int ggmlop_ensure_vtcm_available(void) {
    if (g_compute_res_ctx_id == 0) {
        // Not using compute_res, VTCM is always available
        return 0;
    }

    // In cache mode, VTCM needs to be acquired before each use
    if (!g_vtcm_valid || g_vtcm_needs_release) {
        // VTCM needs to be acquired
        if (g_vtcm_needs_release) {
            GGMLHEXAGON_LOG_INFO("VTCM needs re-acquire (cache mode)");
            g_vtcm_needs_release = 0;
            // Release cached VTCM first
            HAP_compute_res_release_cached(g_compute_res_ctx_id);
        } else {
            GGMLHEXAGON_LOG_INFO("VTCM first acquire (cache mode)");
        }

        // Acquire VTCM with timeout
        int err = HAP_compute_res_acquire_cached(g_compute_res_ctx_id, 1000000);
        if (err != 0) {
            GGMLHEXAGON_LOG_ERROR("Failed to acquire VTCM: 0x%08x", err);
            return -1;
        }
        g_vtcm_valid = 1;
        GGMLHEXAGON_LOG_INFO("VTCM acquired successfully");
    }

    return 0;
}

int ggmlop_dsp_execute_task(remote_handle64 h, int32 ggml_op, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__);

    if (!src0 || !dst) {
        GGMLHEXAGON_LOG_ERROR("invalid input: src0=%p, dst=%p", src0, dst);
        return AEE_EBADPARM;
    }

    GGMLHEXAGON_LOG_INFO("executing op type %d", ggml_op);

    // GGML_OP_NONE: register ION mempool base VA from AP side.
    // FastRPC has already translated src0->data to DSP VA.
    // src1 contains metadata: meta_data[0]=fd, [1..3]=size (low32, high32, size_mb)
    if (ggml_op == GGML_OP_NONE) {
        if (src0 && src0->data) {
            g_ion_dsp_base = src0->data;
            int32_t data_len = src0->data_len;

            int32_t fd = 0;
            uint64_t size_bytes = 0;
            int32_t size_in_mb = 0;
            if (src1 && src1->data) {
                uint32_t * meta_data = (uint32_t *)src1->data;
                fd = (int32_t)meta_data[0];
                uint32_t size_low  = meta_data[1];
                uint32_t size_high = meta_data[2];
                size_bytes = ((uint64_t)size_high << 32) | (uint64_t)size_low;
                size_in_mb = meta_data[3];
            }

            GGMLHEXAGON_LOG_INFO("registered ION DSP base: %p, data_len=%d, fd=%d, size=%llubytes(%dMB)",
                                 g_ion_dsp_base, data_len, fd, (unsigned long long)size_bytes, size_in_mb);
        }
        return AEE_SUCCESS;
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
            ggmlop_dsp_rope(h, src0, src1, dst);
            break;
        case GGML_OP_SOFT_MAX:
            GGMLHEXAGON_LOG_DEBUG("executing GGML_OP_SOFT_MAX task");
            ggmlop_dsp_softmax(h, src0, src1, dst);
            break;
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
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__);

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
    GGMLHEXAGON_LOG_INFO("batch: %d tensors, %d ops", req->n_tensors, req->n_ops);

    // dispatch each op using pre-translated dsptensor pointers
    for (int i = 0; i < req->n_ops; i++) {
        const dsp_op_desc * op = &req->ops[i];

        if (op->src0_idx < 0 || op->src0_idx >= req->n_tensors ||
            op->dst_idx < 0  || op->dst_idx >= req->n_tensors) {
            GGMLHEXAGON_LOG_ERROR("op %d: invalid tensor indices src0=%d src1=%d dst=%d",
                                  i, op->src0_idx, op->src1_idx, op->dst_idx);
            return AEE_EBADPARM;
        }

        const dsptensor * src0 = &req->tensors[op->src0_idx];
        const dsptensor * src1 = (op->src1_idx >= 0) ? &req->tensors[op->src1_idx] : NULL;
        const dsptensor * dst  = &req->tensors[op->dst_idx];

        // log tensor details and sample data for debugging
        GGMLHEXAGON_LOG_INFO("batch op %d: opcode=%d(%s), src0[t%d] data=%p ne=[%d,%d,%d,%d] nb=[%d,%d,%d,%d] type=%d len=%d",
                             i, op->opcode, ggml_op_name(op->opcode),
                             op->src0_idx, src0->data,
                             src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                             src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                             src0->type, src0->data_len);
        if (src1) {
            GGMLHEXAGON_LOG_INFO("  src1[t%d] data=%p ne=[%d,%d,%d,%d] type=%d len=%d",
                                 op->src1_idx, src1->data,
                                 src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                                 src1->type, src1->data_len);
        }
        GGMLHEXAGON_LOG_INFO("  dst[t%d]  data=%p ne=[%d,%d,%d,%d] type=%d len=%d",
                             op->dst_idx, dst->data,
                             dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                             dst->type, dst->data_len);

        // sample first few float values from src0 (for f32/f16 tensors)
        if (src0->data && src0->data_len >= 16) {
            const float * fdata = (const float *)src0->data;
            GGMLHEXAGON_LOG_INFO("  src0 sample before: [%f, %f, %f, %f]",
                                 fdata[0], fdata[1], fdata[2], fdata[3]);
        }

        switch (op->opcode) {
            case GGML_OP_SUB:
                ggmlop_dsp_sub(h, src0, src1, dst);
                break;
            case GGML_OP_ADD:
                ggmlop_dsp_add(h, src0, src1, dst);
                break;
            case GGML_OP_MUL:
                ggmlop_dsp_mul(h, src0, src1, dst);
                break;
            case GGML_OP_DIV:
                ggmlop_dsp_div(h, src0, src1, dst);
                break;
            case GGML_OP_MUL_MAT:
                ggmlop_dsp_mulmat(h, src0, src1, dst);
                break;
            case GGML_OP_RMS_NORM:
                ggmlop_dsp_rmsnorm(h, src0, src1, dst);
                break;
            case GGML_OP_ROPE:
                ggmlop_dsp_rope(h, src0, src1, dst);
                break;
            case GGML_OP_SOFT_MAX:
                ggmlop_dsp_softmax(h, src0, src1, dst);
                break;
            default:
                GGMLHEXAGON_LOG_ERROR("batch op %d: unsupported opcode %d", i, op->opcode);
                return AEE_EUNSUPPORTED;
        }

        // sample dst after op execution
        if (dst->data && dst->data_len >= 16) {
            const float * fdata = (const float *)dst->data;
            GGMLHEXAGON_LOG_INFO("  dst sample after: [%f, %f, %f, %f]",
                                 fdata[0], fdata[1], fdata[2], fdata[3]);
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

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__);
    return AEE_SUCCESS;
}
