#include "ggml-dsp.h"

static int32 g_thread_counts = 1;
static void * g_work_data = NULL;
static size_t g_work_size = 0;

#define MAX_WORK_SIZE (1024 * 1024 * 64)

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

    return 0;
}

int ggmlop_dsp_close(remote_handle64 handle) {
    if (handle)
        free((void*)handle);

    return 0;
}

AEEResult ggmlop_dsp_setclocks(remote_handle64 handle, int32 power_level, int32 latency, int32 mulmat_algo, int32 thread_counts) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__);

    GGMLHEXAGON_LOG_DEBUG("user specified thread_counts %d", thread_counts);
    if (thread_counts > 1)
        g_thread_counts = (thread_counts > g_thread_counts) ? g_thread_counts : thread_counts;
    else
        g_thread_counts = 1;
    GGMLHEXAGON_LOG_DEBUG("real thread_counts %d", g_thread_counts);

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return AEE_SUCCESS;
}

int ggmlop_get_thread_counts(void) {
    return g_thread_counts;
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
