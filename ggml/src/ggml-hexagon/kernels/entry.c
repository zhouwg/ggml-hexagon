#include "ggml-dsp.h"
#include "worker_pool.h"
#include "HAP_compute_res.h"
#include "HAP_power.h"

static int g_thread_counts = 1;
static int g_mulmat_algotype = 0;
static void * g_work_data = NULL;
static size_t g_work_size = 0;

static void * g_vtcm_base = NULL;
static size_t g_vtcm_size = 0;
static unsigned int g_compute_res_ctx_id = 0;
static int g_power_ctx = 0;

#define MAX_WORK_SIZE (1024 * 1024 * 1024)
#define DEFAULT_VTCM_SIZE (8 * 1024 * 1024)

static int power_on_hvx_hmx(void) {
    HAP_power_request_t req;

    /* Set client class */
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_apptype;
    req.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
    if (HAP_power_set((void *)&g_power_ctx, &req) != 0) {
        GGMLHEXAGON_LOG_DEBUG("HAP_power_set apptype failed");
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
    if (HAP_power_set((void *)&g_power_ctx, &req) != 0) {
        GGMLHEXAGON_LOG_DEBUG("HAP_power_set DCVS failed");
        return -2;
    }

    /* Power up HVX */
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_HVX;
    req.hvx.power_up = 1;
    if (HAP_power_set((void *)&g_power_ctx, &req) != 0) {
        GGMLHEXAGON_LOG_DEBUG("HAP_power_set HVX failed");
        return -3;
    }

    /* Power up HMX */
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_HMX;
    req.hmx.power_up = 1;
    if (HAP_power_set((void *)&g_power_ctx, &req) != 0) {
        GGMLHEXAGON_LOG_DEBUG("HAP_power_set HMX failed");
        return -4;
    }

    GGMLHEXAGON_LOG_DEBUG("HAP_power_set for HVX and HMX succeeded");
    return 0;
}

int ggmlop_dsp_open(const char * uri, remote_handle64 * handle) {
    void * tptr = NULL;
    GGMLHEXAGON_LOG_DEBUG("uri %s", uri);
    tptr = (void *)malloc(1);
    GGML_ASSERT(NULL != tptr);
    *handle = (remote_handle64)tptr;

    qurt_sysenv_max_hthreads_t mhwt;
    qurt_sysenv_get_max_hw_threads(&mhwt);
    GGMLHEXAGON_LOG_DEBUG("max hardware threads counts=%d", mhwt.max_hthreads);
    g_thread_counts = mhwt.max_hthreads;

    /* Step 1: Power up HVX and HMX */
    if (power_on_hvx_hmx() != 0) {
        GGMLHEXAGON_LOG_DEBUG("power_on_hvx_hmx failed, continuing without HMX");
    }

    /* Step 2: Query VTCM size and allocate resources */
    unsigned int vtcm_size_query = 0;
    HAP_compute_res_query_VTCM(0, &vtcm_size_query, NULL, NULL, NULL);
    GGMLHEXAGON_LOG_DEBUG("VTCM total = %u bytes", vtcm_size_query);

    /* Step 3: Acquire compute resources (including VTCM and HMX) */
    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);

    unsigned int vtcm_size_to_use = (DEFAULT_VTCM_SIZE < vtcm_size_query) ? DEFAULT_VTCM_SIZE : vtcm_size_query;
    HAP_compute_res_attr_set_vtcm_param(&attr, vtcm_size_to_use, 1);
    HAP_compute_res_attr_set_hmx_param(&attr, 1);

    g_compute_res_ctx_id = HAP_compute_res_acquire(&attr, 100000);
    if (g_compute_res_ctx_id == 0) {
        GGMLHEXAGON_LOG_DEBUG("HAP_compute_res_acquire failed, falling back to HAP_request_VTCM");
        /* Fallback to legacy VTCM allocation */
        g_vtcm_base = HAP_request_VTCM(DEFAULT_VTCM_SIZE, 0);
        if (g_vtcm_base != NULL) {
            g_vtcm_size = DEFAULT_VTCM_SIZE;
            GGMLHEXAGON_LOG_DEBUG("allocated VTCM pool via HAP_request_VTCM: %zu bytes at %p", g_vtcm_size, g_vtcm_base);
        } else {
            GGMLHEXAGON_LOG_DEBUG("failed to allocate VTCM pool, will allocate on demand");
        }
    } else {
        /* Using VTCM acquired via HAP_compute_res */
        void * vtcm_ptr = NULL;
        unsigned int vtcm_ptr_size = 0;
        if (HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &vtcm_ptr, &vtcm_ptr_size) != 0) {
            GGMLHEXAGON_LOG_DEBUG("HAP_compute_res_attr_get_vtcm_ptr_v2 failed");
            HAP_compute_res_release(g_compute_res_ctx_id);
            g_compute_res_ctx_id = 0;
        } else {
            g_vtcm_base = vtcm_ptr;
            g_vtcm_size = vtcm_ptr_size;
            GGMLHEXAGON_LOG_DEBUG("allocated VTCM pool via compute_res: %zu bytes at %p", g_vtcm_size, g_vtcm_base);
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
        /* Release compute resources */
        HAP_compute_res_release(g_compute_res_ctx_id);
        g_compute_res_ctx_id = 0;
        g_vtcm_base = NULL;
        g_vtcm_size = 0;
        GGMLHEXAGON_LOG_DEBUG("released compute resources");
    } else if (g_vtcm_base != NULL) {
        HAP_release_VTCM(g_vtcm_base);
        g_vtcm_base = NULL;
        g_vtcm_size = 0;
        GGMLHEXAGON_LOG_DEBUG("released VTCM pool via HAP_request_VTCM");
    }

    return 0;
}

AEEResult ggmlop_dsp_setclocks(remote_handle64 handle, int32 power_level, int32 latency, int32 mulmat_algo, int32 thread_counts) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__);

    GGMLHEXAGON_LOG_DEBUG("user specified thread_counts %d", thread_counts);
    if (thread_counts <= g_thread_counts) {
        g_thread_counts = thread_counts;
    }
    GGMLHEXAGON_LOG_DEBUG("real thread_counts %d", g_thread_counts);

    g_mulmat_algotype = mulmat_algo;
    GGMLHEXAGON_LOG_DEBUG("mulmat_algotype %d", g_mulmat_algotype);
    FARF(ALWAYS, "mulmat_algotype set to %d (0=auto, 32=VTCM+HMX, 33=VTCM multithread)", g_mulmat_algotype);

    if (g_thread_counts >= 1) {
        AEEResult result = worker_pool_reinit_with_threads(g_thread_counts);
        FARF(HIGH, "worker_pool_reinit_with_threads returned %d", result);
    }

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

int ggmlop_dsp_execute_task(remote_handle64 h, int32 ggml_op, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__);

    if (!src0 || !dst) {
        return AEE_EBADPARM;
    }

    GGMLHEXAGON_LOG_DEBUG("executing op type %d", ggml_op);

    switch ((enum ggml_op)ggml_op) {
        case GGML_OP_SUB:
            GGMLHEXAGON_LOG_DEBUG("executing GGML_OP_SUB task");
            ggmlop_dsp_sub(h, src0, src1, dst);
            break;
        case GGML_OP_ADD:
            GGMLHEXAGON_LOG_DEBUG("executing GGML_OP_ADD task");
            ggmlop_dsp_add(h, src0, src1, dst);
            break;
        case GGML_OP_MUL_MAT:
            GGMLHEXAGON_LOG_DEBUG("executing GGML_OP_MUL_MAT task");
            ggmlop_dsp_mulmat(h, src0, src1, dst);
            break;
        default:
            return AEE_EUNSUPPORTED;
    }

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__);
    return AEE_SUCCESS;
}
