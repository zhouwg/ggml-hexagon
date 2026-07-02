/*
 * Copyright (c) 2024-2026 The ggml authors
 *
 * Qualcomm Hexagon SDK and reference tech guides could be found at:
 * https://developer.qualcomm.com/software/hexagon-dsp-sdk/tools
 *
 * this single-source-file or self-contained implementation of ggml-hexagon backend has 5 sections:
 * section-1  forward/prototype declaration, global vars, macros, data structures
 * section-2  internal troubleshooting function/class
 * section-3  general helper function
 * section-4  cDSP helper function
 * section-5  implementation of ggml-hexagon backend according to specification in ggml backend subsystem
 *
 * this is a practical implementation(although mulmat's performance is slower than Qualcomm's official
 * ggml-hexagon backend at the moment), can expand other ggml ops easily & accordingly.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stddef.h>
#include <inttypes.h>
#include <math.h>
#include <time.h>

#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <map>
#include <set>
#include <tuple>
#include <queue>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <memory>
#include <regex>
#include <random>
#include <functional>
#include <unordered_map>
#include <condition_variable>
#include <unordered_set>
#include <utility>
#include <future>
#include <algorithm>

#if defined(__ANDROID__) || defined(__linux__)
#include <unistd.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <cerrno>
#include <stdatomic.h>
#endif

#if defined(__ANDROID__)
#include "android/log.h"

#include "rpcmem.h"
#include "remote.h"
#include "AEEStdErr.h"
#include "htp-drv.h"
#include "HAP_power.h"
#include "HAP_farf.h"
#endif

#include "ggml-hexagon.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "kernels/skel.h"
#include "kernels/ggml-ops.h"

// =================================================================================================
//  section-1: forward/prototype declaration, global vars, macros, data structures
// =================================================================================================
class  hexagon_profiler;
struct ggml_backend_hexagon_context;

#ifdef NDEBUG
#define GGMLHEXAGON_DEBUG                               0
#else
#define GGMLHEXAGON_DEBUG                               1
#endif

#ifndef PROJECT_NAME
#define PROJECT_NAME                                    "ggml-hexagon"
#endif

#define GGMLHEXAGON_LOGBUF_LEN                          4096
#define GGMLHEXAGON_TMPBUF_LEN                          256

#define GGMLHEXAGON_LOG_ERROR(...)                      ggmlhexagon_log_internal(GGML_LOG_LEVEL_ERROR, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLHEXAGON_LOG_WARN(...)                       ggmlhexagon_log_internal(GGML_LOG_LEVEL_WARN , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

#if !defined (DISABLE_ALL_LOG)
#define GGMLHEXAGON_LOG_INFO(...)                       ggmlhexagon_log_internal(GGML_LOG_LEVEL_INFO , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLHEXAGON_LOG_VERBOSE(...)                    ggmlhexagon_log_internal(GGML_LOG_LEVEL_CONT , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#else
//manually disable all verbose logs in ggml-hexagon/CMakeLists.txt to
//make compare NPU performance through llama-bench more clear
#define GGMLHEXAGON_LOG_INFO(...)
#define GGMLHEXAGON_LOG_VERBOSE(...)
#endif

#if GGMLHEXAGON_DEBUG
#define GGMLHEXAGON_LOG_DEBUG(...)                      ggmlhexagon_log_internal(GGML_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#else
#define GGMLHEXAGON_LOG_DEBUG(...)
#endif

#define RPCMEM_DEFAULT_FLAGS                            1
#define RPCMEM_HEAP_ID_SYSTEM                           25
#define SIZE_IN_MB                                      (1 << 20)
#define STATUS_CONTEXT                                  0x12345678

#define GGMLHEXAGON_MAX_OPS_PER_TASK                    16
#define GGMLHEXAGON_MAX_TENSORS_PER_TASK                32

#if !defined (_WINDOWS)
#pragma weak remote_system_request
#pragma weak remote_session_control
#pragma weak remote_handle_control
#pragma weak remote_handle64_control
#pragma weak fastrpc_mmap
#pragma weak fastrpc_munmap
#endif

// =================================================================================================
//  section-1: data type, data structure, global vars
// =================================================================================================
using pfn_rpc_mem_init                          = void (*)(void);
using pfn_rpc_mem_deinit                        = void (*)(void);
using pfn_rpc_mem_alloc                         = void *(*)(int, uint32_t, int);
using pfn_rpc_mem_free                          = void (*)(void *);
using pfn_rpc_mem_to_fd                         = int (*)(void *);

typedef int  (* notify_callback_fn)(void * context, int domain, int session, remote_rpc_status_flags_t status);
typedef int  (* ggmlhexagon_op_func_t)(remote_handle64 handle, const dsptensor * src0, const dsptensor * src1, dsptensor * dst);

// Forward declaration
static int                   test_hmx_ap(ggml_backend_hexagon_context * ctx);
static bool                  ggml_backend_hexagon_buffer_is_host(ggml_backend_buffer_type_t buft);
static size_t                ggml_backend_hexagon_buffer_type_get_alignment(ggml_backend_buffer_type_t buft);
static size_t                ggml_backend_hexagon_buffer_type_get_max_size(ggml_backend_buffer_type_t buft);
static const char *          ggml_backend_hexagon_buffer_type_name(ggml_backend_buffer_type_t buft);
static int                   ggmlhexagon_probe_dspinfo(ggml_backend_hexagon_context * ctx);
static ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);

struct ggmlhexagon_task {
    int32 op_type;
    ggml_tensor * src0;
    ggml_tensor * src1;
    ggml_tensor * dst;
};

enum qcom_dsp_type {
    HEXAGON_ADSP    = 0,
    HEXAGON_MDSP    = 1,
    HEXAGON_SDSP    = 2,
    HEXAGON_CDSP    = 3,
    HEXAGON_CDSP1   = 4,
};

enum qcom_htp_arch {
    NONE = 0,
    V68 = 68,
    V69 = 69,
    V73 = 73,
    V75 = 75,
    V79 = 79,
    V81 = 81,
};

enum qcom_chipset_soc_model {
    UNKNOWN_SM = 0,
    SM7450 = 41,  // v69, 7 Gen1
    SM8350 = 30,  // v68, 888
    SM8450 = 36,  // v69, SD 8 Gen 1
    SM8475 = 42,  // v69, SD 8+ Gen 1
    SM8550 = 43,  // v73, SD 8 Gen 2
    SM8650 = 57,  // v75, SD 8 Gen 3
    SM8750 = 69,  // v79, SD 8 Elite
    SM8850 = 73,  // v81, SD 8 Elite Gen 5
};

struct qcom_socinfo {
    uint32_t soc_model;
    size_t htp_arch;
    size_t vtcm_size_in_mb;
    char soc_desc[GGML_MAX_NAME];
};

// ION pool region tracking for free-space management.
// Each region records an allocated area within the ION pool.
// When free_buffer is called, the region is marked as not-in-use.
// Free regions can be reused by best-fit allocation.
struct ion_pool_region {
    size_t offset;      // byte offset from ION pool base
    size_t size;        // allocation size in bytes
    bool   in_use;      // true if currently allocated
};

struct ggml_backend_hexagon_context {
    int device;
    char name[GGML_MAX_NAME];
    char desc[GGML_MAX_NAME];
    char lib[GGML_MAX_NAME];
    struct ggml_backend * backend;
    struct qcom_socinfo           socinfo;

    int n_threads;

    //Hexagon resource management for the general approach through Hexagaon cDSP
    size_t rpc_mempool_capacity;
    size_t rpc_mempool_len;
    size_t rpc_mempool_usage;
    size_t rpc_mempool_cache_offset;  // ION offset where FP16 cache region starts
    bool   weights_dirty;             // set by set_tensor/memset_tensor, cleared by Phase 6.5
    void * rpc_mempool;
    int rpc_mempool_handle;
    void * rpc_mempool_dsp_base;   // DSP-side VA from fastrpc_mmap() (NOT from FastRPC pointer translation)
    std::vector<ion_pool_region> ion_regions;  // region tracking for ION pool free-space management
    remote_handle64 ggmlop_handle;
    int domain_id;
    int session_id;

    // FastRPC call statistics
    uint64_t rpc_batch_call_count;   // total ggmlop_dsp_execute_batch_ion calls
    int64_t  cumulative_p7_us;       // cumulative FastRPC time (p7 phase)
    int64_t  cumulative_graph_us;   // cumulative graph inference duration
    int64_t  last_graph_end_us;     // wall clock of last graph end (to measure gap)

    // buffer type owned by this context (each device has its own buft)
    struct ggml_backend_buffer_type buffer_type;
    char buft_name[GGML_MAX_NAME];  // "hexagon-ion-buffer-<name>", unique per device

    // per-device hardware caps (probed at init, used by supports_op)
    bool has_vtcm;  // domain has VTCM pages available
    bool has_hvx;   // domain has HVX support

    ggml_backend_hexagon_context(int dev_id, ggml_backend_dev_t dev);
    ~ggml_backend_hexagon_context();
};

struct hexagon_op_caps {
    bool supported;
    ggml_op op;
    const size_t input_param_count;
    const char * hexagon_op_name;
    ggmlhexagon_op_func_t dsp_op_func;
};

struct hexagon_appcfg_t {
    int enable_perf;            // enable/disable perf of a specified ggml op
    int enable_profiler;        // enable/disable profiler feature
    int dump_debug_info;        // enable/disable dump debug info for troubleshooting issues on AP side
    int enable_q_mulmat;        // enable/disable offload quantized mulmat
    int profiler_duration;      // threshold of duration in profiler, per seconds
    int profiler_counts;        // threshold of counts in profiler
    int thread_counts;          // thread_counts on cDSP side
    int mulmat_algotype;        // algorithm type of mulmat on cDSP side
    int mulmat_min_n;           // minimum N (batch size) to offload quantized MUL_MAT to DSP
    int offload_cgraph_type;    // offload type on AP side
    int dump_diag_info;         // enable/disable dump diag info for troubleshooting issues on cDSP side
    int ggml_dsp_use_hvx;       // enable/disable HVX-optimized quantize_row & vec_dot on cDSP side
    int ndev;                   // number of Hexagon devices (PDs), from GGML_HEXAGON_NDEV env

    const char * cfgfilename;
    const char * runtime_libpath;
    char version[GGMLHEXAGON_TMPBUF_LEN];
    std::string enabled_ops;    // comma-separated list of ops to offload (empty = all supported ops)
    std::string enabled_types;  // comma-separated list of weight types to offload for MUL_MAT (empty = all supported types)
};

static struct hexagon_appcfg_t g_hexagon_appcfg = {
        .enable_perf            = 1,
        .enable_profiler        = 0,
        .dump_debug_info        = 0,
        .enable_q_mulmat        = 1,
        .profiler_duration      = 5,    //seconds
        .profiler_counts        = 100,
        .thread_counts          = 4,
        .mulmat_algotype        = 0,
        .mulmat_min_n           = 32,
        .offload_cgraph_type    = 0,
        .dump_diag_info         = 0,
        .ggml_dsp_use_hvx       = 1,
        .ndev                   = 1,
        .cfgfilename            = "ggml-hexagon.cfg",
#if defined(__ANDROID__)
        .runtime_libpath        = "/data/local/tmp/",
#endif
        .version                = {"0.99.3"},
};

//supported Snapdragon devices with Hexagon DSP
static struct qcom_socinfo g_hexagon_soc_info_table[] = {
        /* Qualcomm SnapDragon 8 Gen 1 */
        {
                .soc_model         = SM8450,
                .htp_arch          = V69,
                .vtcm_size_in_mb   = 8,
                .soc_desc          = "Qualcomm SnapDragon 8 Gen 1"},

        /* Qualcomm SnapDragon 8 Gen 1+ */
        {
                .soc_model         = SM8475,
                .htp_arch          = V69,
                .vtcm_size_in_mb   = 8,
                .soc_desc          = "Qualcomm SnapDragon 8 Gen 1+"},

        /* Qualcomm SnapDragon 8 Gen 2 */
        {
                .soc_model         = SM8550,
                .htp_arch          = V73,
                .vtcm_size_in_mb   = 8,
                .soc_desc          = "Qualcomm SnapDragon 8 Gen 2"},

        /* Qualcomm SnapDragon 8 Gen 3 */
        {
                .soc_model         = SM8650,
                .htp_arch          = V75,
                .vtcm_size_in_mb   = 8,
                .soc_desc          = "Qualcomm SnapDragon 8 Gen 3 "},

        /* Qualcomm SnapDragon 8 Gen 4 */
        {
                .soc_model         = SM8750,
                .htp_arch          = V79,
                .vtcm_size_in_mb   = 8,
                .soc_desc          = "Qualcomm SnapDragon 8 Elite"},

        /* Qualcomm SnapDragon 8 Gen 5 */
        {
                .soc_model         = SM8850,
                .htp_arch          = V81,
                .vtcm_size_in_mb   = 8,
                .soc_desc          = "Qualcomm SnapDragon 8 Elite Gen5"},
};

// Contexts are dynamically allocated in ggml_backend_hexagon_reg() so that
// the constructor can perform DSP initialization (ala qcom's ggml_hexagon_session).
// g_hexagon_mgr holds owning pointers for legacy by-index lookups (e.g. devname).
static struct ggml_backend_hexagon_context * g_hexagon_mgr[GGML_HEXAGON_MAX_DEVICES] = { nullptr };

// Track tensors repacked in set_tensor to skip Phase 4.5 and mulmat_min_n check
static std::unordered_set<const void *> g_set_tensor_repacked;

static domain hexagon_supported_domains[] = {
        {ADSP_DOMAIN_ID, ADSP_DOMAIN},
        {MDSP_DOMAIN_ID, MDSP_DOMAIN},
        {SDSP_DOMAIN_ID, SDSP_DOMAIN},
        {CDSP_DOMAIN_ID, CDSP_DOMAIN},
        {CDSP1_DOMAIN_ID, CDSP1_DOMAIN}
};

// Supported ggml ops by HWACCEL_CDSP.
// Used by both per-op mode (offload_cgraph_type=0) and ION-batch mode (offload_cgraph_type=2).
// Only marks op type as supported - shape/size/type restrictions are enforced in supports_op.
static constexpr const hexagon_op_caps ggmlhexagon_k_op_caps[] = {
    {true,  GGML_OP_NONE,     0, nullptr, nullptr},
    {false, GGML_OP_DUP,      0, nullptr, nullptr},
    {true,  GGML_OP_ADD,      2, "ggmlop_dsp_add",     ggmlop_dsp_add},
    {false, GGML_OP_ADD_ID,   0, nullptr, nullptr},
    {false, GGML_OP_ADD1,     0, nullptr, nullptr},
    {false, GGML_OP_ACC,      0, nullptr, nullptr},
    {true,  GGML_OP_SUB,      2, "ggmlop_dsp_sub",      nullptr},
    {true,  GGML_OP_MUL,      2, "ggmlop_dsp_mul",      nullptr},
    {true,  GGML_OP_DIV,      2, "ggmlop_dsp_div",      nullptr},
    {false, GGML_OP_SQR,      0, nullptr, nullptr},
    {false, GGML_OP_SQRT,     0, nullptr, nullptr},
    {false, GGML_OP_LOG,      0, nullptr, nullptr},
    {false, GGML_OP_SIN,      0, nullptr, nullptr},
    {false, GGML_OP_COS,      0, nullptr, nullptr},
    {false, GGML_OP_SUM,      0, nullptr, nullptr},
    {false, GGML_OP_SUM_ROWS, 0, nullptr, nullptr},
    {false, GGML_OP_CUMSUM,   0, nullptr, nullptr},
    {false, GGML_OP_MEAN,     0, nullptr, nullptr},
    {false, GGML_OP_ARGMAX,   0, nullptr, nullptr},
    {false, GGML_OP_COUNT_EQUAL, 0, nullptr, nullptr},
    {true,  GGML_OP_REPEAT,   2, "ggmlop_dsp_repeat",   nullptr},
    {false, GGML_OP_REPEAT_BACK, 0, nullptr, nullptr},
    {true,  GGML_OP_CONCAT,   2, "ggmlop_dsp_concat",   nullptr},
    {false, GGML_OP_SILU_BACK, 0, nullptr, nullptr},
    {false, GGML_OP_NORM,     0, nullptr, nullptr},
    {true,  GGML_OP_RMS_NORM, 1, "ggmlop_dsp_rmsnorm", nullptr},
    {false, GGML_OP_RMS_NORM_BACK, 0, nullptr, nullptr},
    {false, GGML_OP_GROUP_NORM, 0, nullptr, nullptr},
    {false, GGML_OP_L2_NORM,  0, nullptr, nullptr},
    {true,  GGML_OP_MUL_MAT,  2, "ggmlop_dsp_mulmat",   ggmlop_dsp_mulmat},
    {false, GGML_OP_MUL_MAT_ID, 0, nullptr, nullptr},
    {false, GGML_OP_OUT_PROD, 0, nullptr, nullptr},
    {true,  GGML_OP_SCALE,    1, "ggmlop_dsp_scale", nullptr},
    {false, GGML_OP_SET,      0, nullptr, nullptr},
    {true,  GGML_OP_CPY,      2, "ggmlop_dsp_cpy",      nullptr},
    {false, GGML_OP_CONT,     0, nullptr, nullptr},
    {false, GGML_OP_RESHAPE,  0, nullptr, nullptr},
    {false, GGML_OP_VIEW,     0, nullptr, nullptr},
    {false, GGML_OP_PERMUTE,  0, nullptr, nullptr},
    {false, GGML_OP_TRANSPOSE, 0, nullptr, nullptr},
    {false, GGML_OP_GET_ROWS, 0, nullptr, nullptr},
    {false, GGML_OP_GET_ROWS_BACK, 0, nullptr, nullptr},
    {false, GGML_OP_SET_ROWS, 0, nullptr, nullptr},
    {false, GGML_OP_DIAG,     0, nullptr, nullptr},
    {true,  GGML_OP_DIAG_MASK_INF, 2, "ggmlop_dsp_diag_mask_inf", nullptr},
    {false, GGML_OP_DIAG_MASK_ZERO, 0, nullptr, nullptr},
    {true,  GGML_OP_SOFT_MAX, 2, "ggmlop_dsp_softmax", nullptr},
    {false, GGML_OP_SOFT_MAX_BACK, 0, nullptr, nullptr},
    {true,  GGML_OP_ROPE,     3, "ggmlop_dsp_rope",     nullptr},
    {false, GGML_OP_ROPE_BACK, 0, nullptr, nullptr},
    {false, GGML_OP_CLAMP,    0, nullptr, nullptr},
    {false, GGML_OP_CONV_TRANSPOSE_1D, 0, nullptr, nullptr},
    {false, GGML_OP_IM2COL,   0, nullptr, nullptr},
    {false, GGML_OP_IM2COL_BACK, 0, nullptr, nullptr},
    {false, GGML_OP_IM2COL_3D, 0, nullptr, nullptr},
    {false, GGML_OP_COL2IM_1D, 0, nullptr, nullptr},
    {false, GGML_OP_CONV_2D,  0, nullptr, nullptr},
    {false, GGML_OP_CONV_3D,  0, nullptr, nullptr},
    {false, GGML_OP_CONV_2D_DW, 0, nullptr, nullptr},
    {false, GGML_OP_CONV_TRANSPOSE_2D, 0, nullptr, nullptr},
    {false, GGML_OP_POOL_1D,  0, nullptr, nullptr},
    {false, GGML_OP_POOL_2D,  0, nullptr, nullptr},
    {false, GGML_OP_POOL_2D_BACK, 0, nullptr, nullptr},
    {false, GGML_OP_UPSCALE,  0, nullptr, nullptr},
    {false, GGML_OP_PAD,      0, nullptr, nullptr},
    {false, GGML_OP_PAD_REFLECT_1D, 0, nullptr, nullptr},
    {false, GGML_OP_ROLL,     0, nullptr, nullptr},
    {false, GGML_OP_ARANGE,   0, nullptr, nullptr},
    {false, GGML_OP_TIMESTEP_EMBEDDING, 0, nullptr, nullptr},
    {false, GGML_OP_ARGSORT,  0, nullptr, nullptr},
    {false, GGML_OP_TOP_K,    0, nullptr, nullptr},
    {false, GGML_OP_LEAKY_RELU, 0, nullptr, nullptr},
    {false, GGML_OP_TRI,      0, nullptr, nullptr},
    {false, GGML_OP_FILL,     0, nullptr, nullptr},
    {true,  GGML_OP_FLASH_ATTN_EXT, 4, "ggmlop_dsp_flash_attn", nullptr},
    {false, GGML_OP_FLASH_ATTN_BACK, 0, nullptr, nullptr},
    {false, GGML_OP_SSM_CONV, 0, nullptr, nullptr},
    {false, GGML_OP_SSM_SCAN, 0, nullptr, nullptr},
    {false, GGML_OP_WIN_PART, 0, nullptr, nullptr},
    {false, GGML_OP_WIN_UNPART, 0, nullptr, nullptr},
    {false, GGML_OP_GET_REL_POS, 0, nullptr, nullptr},
    {false, GGML_OP_ADD_REL_POS, 0, nullptr, nullptr},
    {false, GGML_OP_RWKV_WKV6, 0, nullptr, nullptr},
    {false, GGML_OP_GATED_LINEAR_ATTN, 0, nullptr, nullptr},
    {false, GGML_OP_RWKV_WKV7, 0, nullptr, nullptr},
    {false, GGML_OP_SOLVE_TRI, 0, nullptr, nullptr},
    {false, GGML_OP_GATED_DELTA_NET, 0, nullptr, nullptr},
    {true,  GGML_OP_UNARY,    1, "ggmlop_dsp_silu", nullptr},
    {false, GGML_OP_MAP_CUSTOM1, 0, nullptr, nullptr},
    {false, GGML_OP_MAP_CUSTOM2, 0, nullptr, nullptr},
    {false, GGML_OP_MAP_CUSTOM3, 0, nullptr, nullptr},
    {false, GGML_OP_CUSTOM,   0, nullptr, nullptr},
    {false, GGML_OP_CROSS_ENTROPY_LOSS, 0, nullptr, nullptr},
    {false, GGML_OP_CROSS_ENTROPY_LOSS_BACK, 0, nullptr, nullptr},
    {false, GGML_OP_OPT_STEP_ADAMW, 0, nullptr, nullptr},
    {false, GGML_OP_OPT_STEP_SGD, 0, nullptr, nullptr},
    {false, GGML_OP_GLU,      0, nullptr, nullptr},
};

// =================================================================================================
//  section-2: ggml-hexagon internal troubleshooting and profiler function/class
// =================================================================================================
static void ggmlhexagon_get_timestring(char * p_currenttime) {
    if (nullptr == p_currenttime)
        return;

    auto time_to_string = [](const std::chrono::system_clock::time_point & tp)->std::string {
        auto as_time_t = std::chrono::system_clock::to_time_t(tp);
        struct tm tm;

        localtime_r(&as_time_t, &tm);

        std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch());
        char buf[GGMLHEXAGON_TMPBUF_LEN];
        memset(buf, 0, GGMLHEXAGON_TMPBUF_LEN);
        snprintf(buf, sizeof(buf), "%04d-%02d-%02d,%02d:%02d:%02d",
                 tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
        GGML_UNUSED(ms);
        return buf;
    };

    std::chrono::system_clock::time_point tp = std::chrono::system_clock::now();
    snprintf(p_currenttime, GGMLHEXAGON_TMPBUF_LEN, "%s", time_to_string(tp).c_str());
}

static void ggmlhexagon_log_internal(ggml_log_level level, const char * file, const char * func, int line, const char * format, ...) {
    static std::mutex ggmlhexagon_log_internal_mutex;
    static char s_ggmlhexagon_log_internal_buf[GGMLHEXAGON_LOGBUF_LEN];

    GGML_UNUSED(file);

    if (0 == g_hexagon_appcfg.dump_debug_info) {
        if (level != GGML_LOG_LEVEL_CONT) {
            return;
        }
    }

    {
        std::lock_guard<std::mutex> lock(ggmlhexagon_log_internal_mutex);
        va_list args;
        va_start(args, format);
        int len_prefix = snprintf(s_ggmlhexagon_log_internal_buf, GGMLHEXAGON_LOGBUF_LEN, "[%s, %d]: ", func, line);
        int len = vsnprintf(s_ggmlhexagon_log_internal_buf + len_prefix, GGMLHEXAGON_LOGBUF_LEN - len_prefix, format, args);
        if (len < (GGMLHEXAGON_LOGBUF_LEN - len_prefix)) {
#if (defined __ANDROID__) || (defined ANDROID)
            __android_log_print(ANDROID_LOG_INFO, PROJECT_NAME, "%s\n", s_ggmlhexagon_log_internal_buf);
            if (GGML_LOG_LEVEL_INFO == level) {
                printf("%s\n", s_ggmlhexagon_log_internal_buf);
            }
#else
            //for Snapdragon based WoA(Windows on ARM) device or Linux
            printf("%s\n", s_ggmlhexagon_log_internal_buf);
#endif
        }
        va_end(args);
    }
}

// ---- ARM64 Cache Maintenance for Non-Coherent ION ----
// DMA_BUF_IOCTL_SYNC: bracket CPU access to dma-buf fd so the kernel
// can flush/invalidate caches.  This goes through the kernel (EL1)
// so it works even when EL0 DC CVAC/CIVAC is trapped by hypervisor.
// Correct ioctl number: _IOW('b', 0, struct { __u64 flags; }) = 0x40086200
#define DMA_BUF_IOCTL_SYNC_IOCTL  0x40086200u

static int ion_sync_for_direction(int fd, int direction) {
#if defined(__ANDROID__) || defined(__linux__)
    if (fd <= 0) return -1;
    // Flag definitions from Linux kernel include/uapi/linux/dma-buf.h:
    //   DMA_BUF_SYNC_READ  = (1 << 0) = 1
    //   DMA_BUF_SYNC_WRITE = (2 << 0) = 2
    //   DMA_BUF_SYNC_START = (1 << 2) = 4
    //   DMA_BUF_SYNC_END   = (2 << 2) = 8
    {
        static const uint64_t DMA_BUF_SYNC_READ  = (1u << 0);
        static const uint64_t DMA_BUF_SYNC_WRITE = (2u << 0);
        static const uint64_t DMA_BUF_SYNC_START = (1u << 2);
        static const uint64_t DMA_BUF_SYNC_END   = (2u << 2);
        uint64_t rw = (direction == 1) ? DMA_BUF_SYNC_WRITE : DMA_BUF_SYNC_READ;
        struct { uint64_t flags; } s;
        s.flags = DMA_BUF_SYNC_START | rw;
        int r = ioctl(fd, DMA_BUF_IOCTL_SYNC_IOCTL, &s);
        if (r == 0) {
            s.flags = DMA_BUF_SYNC_END | rw;
            ioctl(fd, DMA_BUF_IOCTL_SYNC_IOCTL, &s);
            static int logged = 0;
            if (!logged) { GGMLHEXAGON_LOG_WARN("DMA_BUF_IOCTL_SYNC(%s) OK fd=%d (kernel cache sync)", direction ? "WRITE" : "READ", fd); logged = 1; }
            return 0;
        }
        static int logged_fail = 0;
        if (!logged_fail) { GGMLHEXAGON_LOG_WARN("DMA_BUF_IOCTL_SYNC(%s) FAILED fd=%d errno=%d (%s)", direction ? "WRITE" : "READ", fd, errno, strerror(errno)); logged_fail = 1; }
    }
    {
        struct ion_sync_data { int fd; unsigned int flags; unsigned int pad; };
        struct ion_sync_data sync = { .fd = fd, .flags = (unsigned int)direction };
        int r = ioctl(fd, _IOWR('I', 7, struct ion_sync_data), &sync);
        if (r == 0) {
            static int logged = 0;
            if (!logged) { GGMLHEXAGON_LOG_WARN("ION_IOC_SYNC(%s) fallback OK fd=%d", direction ? "WRITE" : "READ", fd); logged = 1; }
            return 0;
        }
        static int logged_fail2 = 0;
        if (!logged_fail2) { GGMLHEXAGON_LOG_WARN("ION_IOC_SYNC(%s) fallback FAILED fd=%d errno=%d (%s)", direction ? "WRITE" : "READ", fd, errno, strerror(errno)); logged_fail2 = 1; }
    }
#else
    (void)fd; (void)direction;
#endif
    return -1;
}

static inline void cpu_dcache_flush_range(ggml_backend_hexagon_context * backend_ctx, int ion_fd, const void * p, size_t size) {
#if 1
    // range-based DC CVAC
    if (size == 0) return;
    {
        const char * start = (const char *)p;
        const char * end   = start + size;
        const size_t line_size = 64;
        const char * addr = (const char *)((uintptr_t)start & ~(line_size - 1));
        for (; addr < end; addr += line_size) {
            __asm__ volatile("dc cvac, %0" : : "r"((const void *)addr) : "memory");
        }
        __asm__ volatile("dsb ish" ::: "memory");
    }
#else
    // whole-pool DC CVAC
    if (backend_ctx && backend_ctx->rpc_mempool && backend_ctx->rpc_mempool_len > 0) {
        const char * start = (const char *)backend_ctx->rpc_mempool;
        const char * end   = start + backend_ctx->rpc_mempool_len;
        const size_t line_size = 64;
        const char * addr = (const char *)((uintptr_t)start & ~(line_size - 1));
        for (; addr < end; addr += line_size) {
            __asm__ volatile("dc cvac, %0" : : "r"((const void *)addr) : "memory");
        }
        __asm__ volatile("dsb ish" ::: "memory");
    }
#endif
    if (ion_fd > 0) ion_sync_for_direction(ion_fd, 1);
}

static inline void cpu_dcache_inval_range(ggml_backend_hexagon_context * backend_ctx, int ion_fd, const void * p, size_t size) {
#if 1
    // range-based DC CIVAC
    if (size == 0) return;
    {
        const char * start = (const char *)p;
        const char * end   = start + size;
        const size_t line_size = 64;
        const char * addr = (const char *)((uintptr_t)start & ~(line_size - 1));
        for (; addr < end; addr += line_size) {
            __asm__ volatile("dc civac, %0" : : "r"((const void *)addr) : "memory");
        }
        __asm__ volatile("dsb ish" ::: "memory");
        __asm__ volatile("isb" ::: "memory");
    }
#else
    // whole-pool DC CIVAC
    if (backend_ctx && backend_ctx->rpc_mempool && backend_ctx->rpc_mempool_len > 0) {
        const char * start = (const char *)backend_ctx->rpc_mempool;
        const char * end   = start + backend_ctx->rpc_mempool_len;
        const size_t line_size = 64;
        const char * addr = (const char *)((uintptr_t)start & ~(line_size - 1));
        for (; addr < end; addr += line_size) {
            __asm__ volatile("dc civac, %0" : : "r"((const void *)addr) : "memory");
        }
        __asm__ volatile("dsb ish" ::: "memory");
        __asm__ volatile("isb" ::: "memory");
    }
#endif
    if (ion_fd > 0) ion_sync_for_direction(ion_fd, 0);
}

static bool ggmlhexagon_use_ion_mempool() {
    return true;
}

static void ggmlhexagon_get_processname(char * p_name) {
    if (nullptr == p_name)
        return;

    char tmpbuf[GGMLHEXAGON_TMPBUF_LEN];
    memset(tmpbuf, 0, GGMLHEXAGON_TMPBUF_LEN);
#if defined(__ANDROID__) || defined(__linux__)
    int result = readlink("/proc/self/exe", tmpbuf, GGMLHEXAGON_TMPBUF_LEN - 1);
    if (result < 0) {
        GGMLHEXAGON_LOG_WARN("failed to get process name, reason:%s", strerror(errno));
        return;
    }
    GGMLHEXAGON_LOG_DEBUG("process name %s", tmpbuf);
    const char * realname = strrchr(tmpbuf, '/') + 1;
    GGMLHEXAGON_LOG_DEBUG("process name %s", realname);
    snprintf(p_name, GGMLHEXAGON_TMPBUF_LEN, "%s", realname);
#endif
}

static bool ggmlhexagon_is_llamabench_running() {
    char processname[GGMLHEXAGON_TMPBUF_LEN];
    memset(processname, 0, GGMLHEXAGON_TMPBUF_LEN);

    ggmlhexagon_get_processname(processname);
    if (0 != processname[0] && 0 != processname[1] && 0 != processname[10]) {
        if (0 == memcmp(processname, "llama-bench", strlen("llama-bench"))) {
            return true;
        }
        if (0 == memcmp(processname, "test-thread-safety", strlen("test-thread-safety"))) {
            return true;
        }
    }
    return false;
}

static void ggmlhexagon_print_tensors_info(const char * func_name, const ggml_backend_hexagon_context * ctx,
                const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (nullptr != func_name && nullptr != ctx) {
        GGMLHEXAGON_LOG_VERBOSE("call %s in dev %s\n", func_name, ctx->name);
    }
    if (nullptr != src0) {
        GGMLHEXAGON_LOG_VERBOSE(
                "%-6s: type = %i (%s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi, %5zi)",
                src0->name,
                src0->type, ggml_type_name(src0->type), src0->ne[0], src0->ne[1], src0->ne[2],
                src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
    }
    if (nullptr != src1) {
        GGMLHEXAGON_LOG_VERBOSE(
                "%-6s: type = %i (%s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi, %5zi)",
                src1->name,
                src1->type, ggml_type_name(src1->type), src1->ne[0], src1->ne[1], src1->ne[2],
                src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
    }
    GGMLHEXAGON_LOG_VERBOSE("%-6s: type = %i (%s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi, %5zi)",
                      dst->name,
                      dst->type, ggml_type_name(dst->type), dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                      dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
    GGMLHEXAGON_LOG_VERBOSE("\n");
}

static void ggmlhexagon_dump_op_info(const struct ggml_tensor * tensor) {
    //skip sanity check of params because of performance concern
    if (0 == g_hexagon_appcfg.dump_debug_info)
        return;

    const struct ggml_tensor * src0 = tensor->src[0];
    struct ggml_tensor       * src1 = tensor->src[1];
    struct ggml_tensor       * dst  = const_cast<ggml_tensor *>(tensor);
    GGMLHEXAGON_LOG_VERBOSE("op name:%s, tensor type:%s", ggml_op_name(tensor->op), ggml_type_name(tensor->type));
    ggmlhexagon_print_tensors_info(nullptr, nullptr, src0, src1, dst);
}

static void ggmlhexagon_dump_tensor_elements(const ggml_tensor * tensor) {
    float value = 0;
    std::ostringstream tmposs;
    if (tensor->type == GGML_TYPE_F32) {
        for (int h = 0; h < tensor->ne[3]; h++) {
            for (int i = 0; i < tensor->ne[2]; i++) {
                for (int j = 0; j < tensor->ne[1]; j++) {
                    for (int k = 0; k < tensor->ne[0]; k++) {
                        value = ((float *) tensor->data)[h * tensor->ne[2] + i * tensor->ne[1] +
                                                         j * tensor->ne[0] + k];
                        tmposs << std::setw(8) << std::fixed << std::setprecision(2) << value
                               << " ";
                    }
                    if (strlen(tmposs.str().c_str()) <= (GGMLHEXAGON_LOGBUF_LEN - 96)) {
                        GGMLHEXAGON_LOG_VERBOSE("%s\n", tmposs.str().c_str());
                    }
                    tmposs.clear();
                    tmposs.str("");
                }
            }
        }
    }

    GGMLHEXAGON_LOG_VERBOSE("\n");
}

static void ggmlhexagon_dump_tensor(const ggml_tensor * tensor, const char * name) {
    GGMLHEXAGON_LOG_DEBUG("dump ggml tensor %s(%s)\n", name, tensor->name);
    GGMLHEXAGON_LOG_DEBUG("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64", nb = (%5zi, %5zi, %5zi, %5zi)\n",
                      name,
                      tensor->type, ggml_type_name(tensor->type),
                      tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                      tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[2]);
    ggmlhexagon_dump_tensor_elements(tensor);

    GGMLHEXAGON_LOG_DEBUG("\n");
}

//a simple high-cohesion and low-coupling class to collect necessary profiler data and visualize NPU performance accordingly
class hexagon_profiler {
public:
    static hexagon_profiler & get_instance() {
        //make thread-safety without using complex dynamic resource management
        static hexagon_profiler instance;
        return instance;
    }

public:
    void profiler_init(int profiler_threshold_duration, int profiler_threshold_counts) {
        reset();
        //here is not accurate profiler start time because inference wasn't launched at the moment
        _profiler_starttime = ggml_time_us();

        _profiler_threshold_duration = profiler_threshold_duration;
        _profiler_threshold_counts   = profiler_threshold_counts;

        std::string filename = std::string(g_hexagon_appcfg.runtime_libpath) + "/";
        if (g_hexagon_appcfg.thread_counts > 1) {
            //multi-threading feature enabled on cDSP side
            filename = filename + "hexagon_perf_cdsp_mt.dat";
        } else {
            filename = filename + "hexagon_perf_cdsp.dat";
        }
        GGMLHEXAGON_LOG_DEBUG("profiler name:%s", filename.c_str());
        const char * profiler_filename = filename.c_str();
        _fp_profile_file = fopen(profiler_filename, "w");
        if (nullptr == _fp_profile_file) {
            GGMLHEXAGON_LOG_WARN("can't open profiler file %s, reason:%s", profiler_filename, strerror(errno));
            reset();
            return;
        } else {
            size_t written_size = 0;
            char profiler_info[GGMLHEXAGON_TMPBUF_LEN];
            const char * prefix = "### starting hexagon profiler at ";

            written_size = fwrite(prefix, 1, strlen(prefix), _fp_profile_file);
            if (written_size != strlen(prefix)) {
                GGMLHEXAGON_LOG_WARN("write data to file %s failed, reason: %s", profiler_filename, strerror(errno));
                profiler_deinit();
                return;
            }

            memset(profiler_info, 0, GGMLHEXAGON_TMPBUF_LEN);
            ggmlhexagon_get_timestring(profiler_info);
            written_size = fwrite(profiler_info, 1, strlen(profiler_info), _fp_profile_file);
            if (written_size != strlen(profiler_info)) {
                GGMLHEXAGON_LOG_WARN("write data to file %s failed, reason: %s", profiler_filename, strerror(errno));
                profiler_deinit();
                return;
            }
            fprintf(_fp_profile_file, "\n\n");
            fprintf(_fp_profile_file,
                    "#frame     input   max     total     avg         elapse     frame       max        total      avg\n");
            fprintf(_fp_profile_file,
                    "#                                                           inference   inference  inference  inference\n");
            fprintf(_fp_profile_file,
                    "#index     len     i-len   i-len     i-speed     time       time        time       time       time\n");
            fprintf(_fp_profile_file, "\n\n");
        }
        _enable_profiler = true;
    }

    void profiler_deinit() {
        if (nullptr != _fp_profile_file) {
            fclose(_fp_profile_file);
            _fp_profile_file = nullptr;
        }
        reset();
    }

/**
 * \param inference_time          microseconds, inference time for a single GGML op
 * \param inference_input_size    bytes, total input data size for a single GGML op
 * \param inference_output_size   bytes, total output data size for a single GGML op
 */
    void profiler_update_profilerdata(const char * ggml_opname, int inference_time, int inference_input_size, int inference_output_size) {
        if (!_enable_profiler)
            return;

        //1.get the accurate profiler starting time in this function when frame index is 0
        //2.update frame index in this function accordingly
        profiler_update_frameindex();

        int64_t elapse_time = ggml_time_us() - profiler_get_starttime();
        profiler_update_elapsetime(elapse_time);
        if (elapse_time > (_profiler_threshold_duration * SIZE_IN_MB)) {
            //do nothing when elapsed profiler time > profiler_duration in ggml-hexagon.cfg
            return;
        }
        if (profiler_get_frame_index() >= _profiler_threshold_counts) {
            //do nothing when frame_index >= profiler_counts in ggml-hexagon.cfg
            return;
        }

        if (inference_input_size > profiler_get_max_inputsize()) {
            profiler_set_max_inputsize(inference_input_size);
        }

        if (inference_output_size > profiler_get_max_outputsize()) {
            profiler_set_max_outputsize(inference_output_size);
        }

        if (inference_time > profiler_get_max_inferencetime()) {
            profiler_set_max_inferencetime(inference_time);
        }

        profiler_update_total_inputsize(inference_input_size);
        profiler_update_total_outputsize(inference_output_size);
        profiler_update_total_inferencetime(inference_time);
        profiler_update_elapsetime(elapse_time);

        if (10 > _frame_index) {
            //FIXME:why some initial profiler data in llama-cli looks unusual
            //return;
        }

        if (0 == elapse_time) {
            //filter invalid profiler data
            return;
        }

        if (NULL != _fp_profile_file) {
            fprintf(_fp_profile_file, "%-8d  %-6d  %-6d  %-10ld %-11ld %-10ld %-12d %-9d %-11ld %-3ld\n",
                    profiler_get_frame_index(),
                    inference_input_size,
                    profiler_get_max_inputsize(),
                    profiler_get_total_inputputsize(),
                    profiler_get_total_inputputsize() / profiler_get_frame_index(),

                    elapse_time,
                    inference_time,
                    profiler_get_max_inferencetime(),
                    profiler_get_total_inferencetime(),
                    profiler_get_total_inferencetime() / profiler_get_frame_index()
            );
        }

        //print/compare NPU's I/O performance between 8Gen3 and 8Elite , removed in the future
        char bps_string[GGMLHEXAGON_TMPBUF_LEN];
        memset(bps_string, 0, GGMLHEXAGON_TMPBUF_LEN);
        profiler_get_bpsstring(_total_inputsize + _total_outputsize, elapse_time, bps_string);
        GGMLHEXAGON_LOG_VERBOSE("I/O performance:%s", bps_string);
    }

    int profiler_get_frame_index() {
        return _frame_index;
    }

    int profiler_get_threshold_count() {
        return _profiler_threshold_counts;
    }

private:
    void profiler_set_max_inputsize(int input_size) {
        _max_inputsize = input_size;
    }

    void profiler_set_max_outputsize(int output_size) {
        _max_outputsize = output_size;
    }

    void profiler_set_max_inferencetime(int inference_time) {
        _max_inferencetime = inference_time;
    }

    void profiler_update_frameindex() {
        if (0 == _frame_index) {
            _profiler_starttime = ggml_time_us();
        }
        _frame_index += 1;
    }

    void profiler_update_elapsetime(int64_t elapse_time_microseconds) {
        _profiler_elapsetime = elapse_time_microseconds;
    }

    void profiler_update_total_inferencetime(int inference_time) {
        _total_inferencetime += inference_time;
    }

    void profiler_update_total_inputsize(int input_size) {
        _total_inputsize += input_size;
    }

    void profiler_update_total_outputsize(int output_size) {
        _total_outputsize += output_size;
    }

    int profiler_get_max_inputsize() {
        return _max_inputsize;
    }

    int profiler_get_max_outputsize() {
        return _max_outputsize;
    }

    int profiler_get_max_inferencetime() {
        return _max_inferencetime;
    }

    int64_t profiler_get_total_inferencetime() {
        return _total_inferencetime;
    }

    int64_t profiler_get_total_inputputsize() {
        return _total_inputsize;
    }

    //might-be used to calculate total I/O performance in the future
    int64_t profiler_get_total_outputsize() {
        return _total_outputsize;
    }

    int64_t profiler_get_starttime() {
        return _profiler_starttime;
    }

    int64_t profiler_get_elapsedtime() {
        return _profiler_elapsetime;
    }

    void profiler_get_bpsstring(int64_t data_size, int64_t elapse_time_microseconds, char * bps_string) {
        if (nullptr == bps_string) {
            return;
        }

        float bps = 0.0f;
        bps = (data_size * SIZE_IN_MB * 1.0f) / (elapse_time_microseconds * 1.0f);
        if (bps >= SIZE_IN_MB) {
            snprintf(bps_string, GGMLHEXAGON_TMPBUF_LEN, "%.2f MiB/s", ((float) bps) / SIZE_IN_MB);
        } else if (bps >= 1000) {
            snprintf(bps_string, GGMLHEXAGON_TMPBUF_LEN, "%.1f KiB/s", ((float) bps) / 1000);
        } else {
            snprintf(bps_string, GGMLHEXAGON_TMPBUF_LEN, "%.2f B/s", bps);
        }
    }

    void reset() {
        _frame_index         = 0;

        _max_inputsize       = 0;
        _max_outputsize      = 0;
        _max_inferencetime   = 0;

        _total_inputsize     = 0;
        _total_outputsize    = 0;
        _total_inferencetime = 0;

        _profiler_starttime  = 0;
        _profiler_elapsetime = 0;
        _fp_profile_file     = nullptr;
        _enable_profiler     = false;
        _profiler_threshold_duration = 100;
        _profiler_threshold_duration = 5;
    }

private:
    hexagon_profiler() {
        reset();
    }

    hexagon_profiler(const hexagon_profiler &) = delete;

    hexagon_profiler(const hexagon_profiler &&) = delete;

    hexagon_profiler & operator= (const hexagon_profiler &) = delete;

private:
    int _frame_index;

    int _max_inputsize;             //bytes
    int _max_outputsize;            //bytes
    int _max_inferencetime;         //bytes

    int64_t _total_inputsize;       //bytes
    int64_t _total_outputsize;      //bytes
    int64_t _total_inferencetime;   //microsecond

    int64_t _profiler_starttime;    //microsecond
    int64_t _profiler_elapsetime;   //microsecond
    FILE *  _fp_profile_file;

    bool _enable_profiler;
    int  _profiler_threshold_duration; //seconds
    int  _profiler_threshold_counts;
};
static hexagon_profiler & g_hexagon_profiler = hexagon_profiler::get_instance();

//a simple perf class to probe NPU performance
class hexagon_perf {
public:
    hexagon_perf(const std::string & perf_name) : _perf_name(std::move(perf_name)) {}
    hexagon_perf(const std::string & perf_name, const char * op_name, int input_size, int output_size)
               : _perf_name(std::move(perf_name)), _op_name(op_name),
                 _input_size(input_size),
                 _output_size(output_size) {

    }

    void start() {
        if (0 == g_hexagon_appcfg.enable_perf)
            return;
        _begin_time = ggml_time_us();
    }

    //use explicit function calls rather than scoped feature
    void info() {
        if (0 == g_hexagon_appcfg.enable_perf) {
            return;
        }

        _end_time = ggml_time_us();
        _duration = (_end_time - _begin_time);
        //add following judgement will useful for other developers and AI experts although:
        // it breaks the original logic
        // it's not mandatory
        // had to expose two public function in hexagon_profiler class
        if (g_hexagon_profiler.profiler_get_frame_index() <= g_hexagon_profiler.profiler_get_threshold_count()) {
            GGMLHEXAGON_LOG_VERBOSE("inference duration of %s through Hexagon-cDSP: %lld microseconds",
                                    _perf_name.c_str(), _duration);
        }

        //update profiler data
        g_hexagon_profiler.profiler_update_profilerdata(_op_name, _duration, _input_size, _output_size);
    }

private:
    hexagon_perf() = delete;
    hexagon_perf(const hexagon_perf & ) = delete;
    hexagon_perf(const hexagon_perf && ) = delete;
    hexagon_perf & operator= (const hexagon_perf & ) = delete;

private:
    int64_t _begin_time = 0LL;
    int64_t _end_time   = 0LL;
    int64_t _duration   = 0LL;
    std::string _perf_name;
    const char * _op_name;
    int   _input_size   = 0;
    int   _output_size  = 0;
};

//a simple class to load/set running configurations in ggml-hexagon.cfg
class hexagon_appcfg {
public:
    hexagon_appcfg() {}

    void dump(std::function<void(const std::string &, const std::string &, const std::string &)> worker) {
        if (!_load_success) {
            GGMLHEXAGON_LOG_INFO("hexagon cfg file %s not loaded", _cfg_filename.c_str());
            return;
        }
        auto iter = _hexagon_appcfg.begin();
        while (iter != _hexagon_appcfg.end()) {
            auto kv_iter = iter->second.begin();
            while (kv_iter != iter->second.end()) {
                worker(iter->first, kv_iter->first, kv_iter->second);
                ++kv_iter;
            }
            ++iter;
        }
    }

    bool load(const std::string & file_name) {
        if (file_name == "") {
            return false;
        }
        _cfg_filename = file_name;
        std::ifstream in;
        std::string line;
        in.open(file_name.c_str());
        if (not in.is_open()) {
            GGMLHEXAGON_LOG_WARN("can't open file %s", file_name.c_str());
            return false;
        }
        while (getline(in, line)) {
            std::string section, key, value;
            if (not parse_line(line, section, key, value)) {
                continue;
            }
            set_section_keyvalue(section, key, value);
        }
        _load_success = true;
        return true;
    }

    void get_stringvalue(const std::string & section, const std::string & key, std::string & value, std::string default_value) {
        value = default_value;
        if (_hexagon_appcfg.find(section) == _hexagon_appcfg.end()) {
            return;
        }
        if (_hexagon_appcfg[section].find(key) == _hexagon_appcfg[section].end()) {
            return;
        }
        value = _hexagon_appcfg[section][key];
    }

    void get_intvalue(const std::string & section, const std::string & key, int & value, int default_value) {
        value = default_value;
        if (_hexagon_appcfg.find(section) == _hexagon_appcfg.end()) {
            return;
        }
        if (_hexagon_appcfg[section].find(key) == _hexagon_appcfg[section].end()) {
            return;
        }
        value = atol(_hexagon_appcfg[section][key].c_str());
    }

    bool modify_hexagon_config(std::string & cfg_filename, int new_mulmat_algotype) {
        std::ifstream inputfile(cfg_filename);
        if (!inputfile.is_open()) {
            GGMLHEXAGON_LOG_WARN("can't open file %s", cfg_filename.c_str());
            return false;
        }

        std::string filedata = "";

        std::string line;
        std::string backupline;
        bool is_rewrite = false;
        bool is_founded = false;
        bool is_key = true;
        std::string key;
        std::string value;
        std::string newvalue;
        while (std::getline(inputfile, line)) {
            is_founded = false;
            backupline = line;
            trim(line);
            if (0 == line.rfind("#", 0)) {
                filedata += backupline;
                filedata += "\n";
                continue;
            }

            newvalue = "";
            if (line.rfind("mulmat_algotype", 0) != std::string::npos) {
                //compatiable with previous logic
                if (new_mulmat_algotype >= 0) {
                    is_founded = true;
                    is_rewrite = true;
                    newvalue = std::to_string(new_mulmat_algotype);
                }
            }

            if (is_founded) {
                is_key = true;
                key = "";
                value = "";

                for (size_t i = 0; i < line.size(); ++i) {
                    if (line[i] == '=') {
                        is_key = false;
                        continue;
                    }
                    if (is_key) {
                        key += line[i];
                    } else {
                        value += line[i];
                    }
                }
                trim(key);
                trim(value);
                GGMLHEXAGON_LOG_VERBOSE("key %s value %s\n", key.c_str(), value.c_str());
                GGMLHEXAGON_LOG_VERBOSE("key %s new value %s\n", key.c_str(), newvalue.c_str());
                backupline = key + " = " + newvalue;
            }
            filedata += backupline;
            filedata += "\n";
        }
        inputfile.close();

        if (is_rewrite) {
            std::ofstream outputfile;
            outputfile.open(cfg_filename);
            outputfile.flush();
            outputfile << filedata;
            outputfile.close();
        }
        return true;
    }

private:
    void ltrim(std::string & str) {
        if (str.empty()) return;
        size_t len  = 0;
        const char * temp = str.c_str();
        while (*temp && isblank(*temp)) {
            ++len;
            ++temp;
        }
        if (len > 0) str.erase(0, len);
    }

    void rtrim(std::string & str) {
        if (str.empty()) return;
        size_t len = str.length();
        size_t pos = len;
        while (pos > 0) {
            if (not isblank(str[pos - 1])) {
                break;
            }
            --pos;
        }
        if (pos != len) str.erase(pos);
    }

    void trim(std::string & str) {
        ltrim(str);
        rtrim(str);
    }

    void set_section_keyvalue(std::string & section, std::string & key, std::string & value) {
        if (_hexagon_appcfg.find(section) == _hexagon_appcfg.end()) {
            std::unordered_map<std::string, std::string> kv_map;
            _hexagon_appcfg[section] = kv_map;
        }
        if (key != "" && value != "") _hexagon_appcfg[section][key] = value;
    }

    bool parse_line(std::string & line, std::string & section, std::string & key, std::string & value) {
        static std::string cur_section = "";
        std::string nodes[2] = {"#", ";"};
        for (int i = 0; i < 2; ++i) {
            std::string::size_type pos = line.find(nodes[i]);
            if (pos != std::string::npos) line.erase(pos);
        }
        trim(line);
        if (line == "") return false;
        if (line[0] == '[' && line[line.size() - 1] == ']') {
            section = line.substr(1, line.size() - 2);
            trim(section);
            cur_section = section;
            return false;
        }
        if (cur_section == "") return false;
        bool is_key = true;
        for (size_t i = 0; i < line.size(); ++i) {
            if (line[i] == '=') {
                is_key = false;
                continue;
            }
            if (is_key) {
                key += line[i];
            } else {
                value += line[i];
            }
        }
        section = cur_section;
        trim(key);
        trim(value);

        //"1.00" -> 1.00
        if (value.front() == '"' && value.back() == '"') {
            value.erase(0, 1); // erase the first character "
            value.erase(value.size() - 1); // erase the last character "
        }

        return true;
    }

private:
    hexagon_appcfg(const hexagon_appcfg & ) = delete;
    hexagon_appcfg(const hexagon_appcfg && ) = delete;
    hexagon_appcfg & operator= (const hexagon_appcfg & ) = delete;

private:
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> _hexagon_appcfg;
    bool _load_success = false;
    std::string _cfg_filename;
};

// =================================================================================================
//  section-3: general helper function
// =================================================================================================
static const char * ggmlhexagon_get_socmodel_desc(uint32_t soc_model) {
    switch (soc_model) {
        case SM7450:
            return "SM7450";
        case SM8350:
            return "SM8350";
        case SM8450:
            return "SM8450";
        case SM8475:
            return "SM8475";
        case SM8550:
            return "SM8550";
        case SM8650:
            return "SM8650";
        case SM8750:
            return "SM8750";
        case SM8850:
            return "SM8850";
        default:
            return "unknown";
    }
}

//0x68 -> 68, 0x69 -> 69, 0x73 -> 73, 0x75 -> 75, 0x79 -> 79, 0x81 -> 81
static size_t ggmlhexagon_htparch_hex_to_decimal(size_t htp_arch) {
    //naive algorithm
    int a = htp_arch / 16;
    int b = htp_arch % 16;
    return a * 10 + b;
}

static const char * ggmlhexagon_get_htparch_desc(size_t htp_arch) {
    switch (htp_arch) {
        case V68:
            return "QCOM_HTP_V68";
        case V69:
            return "QCOM_HTP_V69";
        case V73:
            return "QCOM_HTP_V73";
        case V75:
            return "QCOM_HTP_V75";
        case V79:
            return "QCOM_HTP_V79";
        case V81:
            return "QCOM_HTP_V81";
        default:
            return "unknown";
    }
}

static struct qcom_socinfo * ggmlhexagon_get_socinfo_from_socmodel(uint32_t soc_model) {
    size_t items = sizeof(g_hexagon_soc_info_table) / sizeof(g_hexagon_soc_info_table[0]);
    for (size_t idx = 0; idx < items; idx++) {
        if (soc_model == g_hexagon_soc_info_table[idx].soc_model) {
            return &g_hexagon_soc_info_table[idx];
        }
    }
    return nullptr;
}

static struct qcom_socinfo * ggmlhexagon_get_socinfo_from_htparch(size_t htp_arch) {
    size_t items = sizeof(g_hexagon_soc_info_table) / sizeof(g_hexagon_soc_info_table[0]);
    for (size_t idx = 0; idx < items; idx++) {
        if (htp_arch == g_hexagon_soc_info_table[idx].htp_arch) {
            return &g_hexagon_soc_info_table[idx];
        }
    }
    return nullptr;
}

static size_t ggmlhexagon_get_system_total_memory_in_bytes() {
#if defined(__ANDROID__) || defined(__linux__)
    struct sysinfo info = {};
    if (0 == sysinfo(&info)) {
        return (info.totalram + info.totalswap) * info.mem_unit;
    }
    size_t pages      = (size_t)sysconf(_SC_PHYS_PAGES);
    size_t page_size  = (size_t)sysconf(_SC_PAGE_SIZE);

    return pages * page_size;
#endif
}

static size_t ggmlhexagon_get_system_free_memory_in_bytes() {
#if defined(__ANDROID__) || defined(__linux__)
    struct sysinfo info = {};
    if (0 == sysinfo(&info)) {
        return (info.freeram + info.freeswap) * info.mem_unit;
    }
    size_t avail_pages = (size_t)sysconf(_SC_AVPHYS_PAGES);
    size_t page_size   = (size_t)sysconf(_SC_PAGE_SIZE);

    return avail_pages * page_size;
#endif
}

static bool ggmlhexagon_same_types(const ggml_backend_hexagon_context * ctx, const ggml_tensor * op_tensor) {
    GGML_UNUSED(ctx);
    ggml_tensor * src0 = op_tensor->src[0];
    ggml_tensor * src1 = op_tensor->src[1];
    if (nullptr != src1) {
        if (src0->type != op_tensor->type || src1->type != op_tensor->type) {
            return false;
        }
    } else {
        if (src0->type != op_tensor->type) {
            return false;
        }
    }

    if (src0->type != GGML_TYPE_F32)
        return false;

    return true;
}

static const char * ggmlhexagon_get_ggml_type_name(ggml_type type) {
    const auto * traits = ggml_get_type_traits(type);
    return traits->type_name;
}

static void ggmlhexagon_append_tensor_dimensions(const ggml_tensor * tensor, std::string & output) {
    char buffer[GGMLHEXAGON_TMPBUF_LEN] = {};
    const char * type_name = ggmlhexagon_get_ggml_type_name(tensor->type);
    int len = 0;
    switch (ggml_n_dims(tensor)) {
        case 1:
            len = snprintf(buffer, sizeof(buffer), "%ldx1%s", (long)tensor->ne[0], type_name);
            break;
        case 2:
            len = snprintf(buffer, sizeof(buffer), "%ldx%ld%s", (long)tensor->ne[0], (long)tensor->ne[1], type_name);
            break;
        case 3:
            len = snprintf(buffer, sizeof(buffer), "%ldx%ldx%ld%s", (long)tensor->ne[0], (long)tensor->ne[1],
                           (long)tensor->ne[2], type_name);
            break;
        case 4:
        default:
            len = snprintf(buffer, sizeof(buffer), "%ldx%ldx%ldx%ld%s", (long)tensor->ne[0], (long)tensor->ne[1],
                           (long)tensor->ne[2], (long)tensor->ne[3], type_name);
            break;
    }
    GGML_ASSERT(len > 0 && len < (int)sizeof(buffer));
    output.append(buffer, len);
}

static size_t ggmlhexagon_get_op_index(const ggml_tensor * tensor) {
    return tensor->op;
}

static size_t ggmlhexagon_get_op_input_param_count(const ggml_tensor * op) {
    auto op_index = ggmlhexagon_get_op_index(op);
    GGML_ASSERT(op_index < std::size(ggmlhexagon_k_op_caps));
    return ggmlhexagon_k_op_caps[op_index].input_param_count;
}

static void ggmlhexagon_get_opkey_from_op(const ggml_tensor * op, std::string & output) {
    if (op->op == GGML_OP_NONE) {
        output = "GGML_OP_NONE";
        return;
    }
    output += ggml_op_desc(op);
    output += ggmlhexagon_get_ggml_type_name(op->type);
    size_t param_count = ggmlhexagon_get_op_input_param_count(op);
    for (size_t i = 0; i < param_count; ++i) {
        auto * input = op->src[i];
        if (!input) {
            break;
        }
        output += '_';
        ggmlhexagon_append_tensor_dimensions(input, output);
    }
}

static void ggmlhexagon_set_runtime_path(size_t device, const std::string & path) {
    GGML_UNUSED(device);
#if defined(__ANDROID__)
    std::string lib_runtime_path = path + ":/vendor/dsp/cdsp:/vendor/lib64:/vendor/dsp/dsp:/vendor/dsp/images";
    if (0 == setenv("LD_LIBRARY_PATH", lib_runtime_path.c_str(), 1)) {
        GGMLHEXAGON_LOG_DEBUG("setenv LD_LIBRARY_PATH %s successfully", lib_runtime_path.c_str());
    } else {
        GGMLHEXAGON_LOG_ERROR("setenv LD_LIBRARY_PATH %s failure", lib_runtime_path.c_str());
    }

    std::string adsp_runtime_path = path + ";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/vendor/dsp/dsp;/vendor/dsp/images;/dsp";
    if (0 == setenv("ADSP_LIBRARY_PATH", adsp_runtime_path.c_str(), 1)) {
        GGMLHEXAGON_LOG_DEBUG("setenv ADSP_LIBRARY_PATH %s successfully", adsp_runtime_path.c_str());
    } else {
        GGMLHEXAGON_LOG_ERROR("setenv ADSP_LIBRARY_PATH %s failure", adsp_runtime_path.c_str());
    }
#endif
}

static void ggmlhexagon_load_cfg() {
    //this function can be called in various scenarios
    static bool initialized = false;
    if (initialized) {
        GGMLHEXAGON_LOG_DEBUG("hexagon appcfg file already loaded\n");
        return;
    }
    char time_string[GGMLHEXAGON_TMPBUF_LEN];
    memset(time_string, 0, GGMLHEXAGON_TMPBUF_LEN);
    ggmlhexagon_get_timestring(time_string);
    GGMLHEXAGON_LOG_DEBUG("program running start time:%s", time_string);
    std::string cfg_filename = std::string(g_hexagon_appcfg.runtime_libpath) + std::string(g_hexagon_appcfg.cfgfilename);

    hexagon_appcfg hexagoncfg_instance;
    hexagoncfg_instance.load(cfg_filename);
    hexagoncfg_instance.dump([](const std::string & section, const std::string & key, const std::string value) {
        std::ostringstream  tmposs;
        tmposs << "section[" << std::setw(10) << std::left << section << "],[" << std::setw(25) << std::left << key << "] = [" << value << "]";
        GGMLHEXAGON_LOG_VERBOSE("%s", tmposs.str().c_str());
    });
    std::string version; //version of ggml-hexagon
    hexagoncfg_instance.get_stringvalue("general", "version", version, "0.99");
    hexagoncfg_instance.get_intvalue("general", "enable_perf", g_hexagon_appcfg.enable_perf, 1);
    hexagoncfg_instance.get_intvalue("general", "dump_debug_info", g_hexagon_appcfg.dump_debug_info, 0);
    hexagoncfg_instance.get_intvalue("general", "enable_q_mulmat", g_hexagon_appcfg.enable_q_mulmat, 0);
    hexagoncfg_instance.get_intvalue("general", "enable_profiler", g_hexagon_appcfg.enable_profiler, 0);
    hexagoncfg_instance.get_intvalue("general", "profiler_duration", g_hexagon_appcfg.profiler_duration, 5);
    hexagoncfg_instance.get_intvalue("general", "profiler_counts", g_hexagon_appcfg.profiler_counts, 100);

    hexagoncfg_instance.get_intvalue("cdsp", "thread_counts", g_hexagon_appcfg.thread_counts, 4);
    hexagoncfg_instance.get_intvalue("cdsp", "mulmat_algotype", g_hexagon_appcfg.mulmat_algotype, 0);
    hexagoncfg_instance.get_intvalue("cdsp", "mulmat_min_n", g_hexagon_appcfg.mulmat_min_n, 32);
    hexagoncfg_instance.get_intvalue("cdsp", "offload_cgraph_type", g_hexagon_appcfg.offload_cgraph_type, 2);
    hexagoncfg_instance.get_intvalue("cdsp", "dump_diag_info", g_hexagon_appcfg.dump_diag_info, 0);
    hexagoncfg_instance.get_intvalue("cdsp", "ggml_dsp_use_hvx", g_hexagon_appcfg.ggml_dsp_use_hvx, 1);
    hexagoncfg_instance.get_intvalue("cdsp", "ndev", g_hexagon_appcfg.ndev, 1);
    hexagoncfg_instance.get_stringvalue("cdsp", "enabled_ops", g_hexagon_appcfg.enabled_ops, "");
    hexagoncfg_instance.get_stringvalue("cdsp", "enabled_types", g_hexagon_appcfg.enabled_types, "");

    memcpy(g_hexagon_appcfg.version, version.c_str(), strlen(version.c_str()));

    GGMLHEXAGON_LOG_VERBOSE("load hexagon appcfg from %s", cfg_filename.c_str());
    GGMLHEXAGON_LOG_VERBOSE("ggml_hexagon_version=%s", g_hexagon_appcfg.version);
    GGMLHEXAGON_LOG_VERBOSE("runtime libpath=%s", g_hexagon_appcfg.runtime_libpath);
    GGMLHEXAGON_LOG_VERBOSE("enable_perf=%d", g_hexagon_appcfg.enable_perf);
    GGMLHEXAGON_LOG_VERBOSE("enable_profiler=%d", g_hexagon_appcfg.enable_profiler);

    // env var GGML_HEXAGON_NDEV overrides cfg value (for automation/testing)
    const char * str_ndev = getenv("GGML_HEXAGON_NDEV");
    if (str_ndev) {
        int v = atoi(str_ndev);
        if (v > 0 && v <= GGML_HEXAGON_MAX_DEVICES) {
            g_hexagon_appcfg.ndev = v;
        } else {
            GGMLHEXAGON_LOG_WARN("invalid GGML_HEXAGON_NDEV=%d, must be 1..%d, using cfg value %d",
                                 v, GGML_HEXAGON_MAX_DEVICES, g_hexagon_appcfg.ndev);
        }
    }
    if (g_hexagon_appcfg.ndev < 1 || g_hexagon_appcfg.ndev > GGML_HEXAGON_MAX_DEVICES) {
        GGMLHEXAGON_LOG_WARN("invalid ndev=%d from cfg, must be 1..%d, using default 1",
                             g_hexagon_appcfg.ndev, GGML_HEXAGON_MAX_DEVICES);
        g_hexagon_appcfg.ndev = 1;
    }
    GGMLHEXAGON_LOG_INFO("ndev=%d (from cfg, env GGML_HEXAGON_NDEV overrides if set)", g_hexagon_appcfg.ndev);

    ggmlhexagon_set_runtime_path(0, g_hexagon_appcfg.runtime_libpath);

    if (1 == g_hexagon_appcfg.enable_profiler) {
        //make sure this function is called only once
        g_hexagon_profiler.profiler_init(g_hexagon_appcfg.profiler_duration, g_hexagon_appcfg.profiler_counts);
    }

    initialized = true;
}

int ggml_backend_hexagon_get_mulmat_algotype() {
    std::string cfg_filename = std::string(g_hexagon_appcfg.runtime_libpath) + std::string(g_hexagon_appcfg.cfgfilename);
    hexagon_appcfg hexagoncfg_instance;
    hexagoncfg_instance.load(cfg_filename);
    hexagoncfg_instance.get_intvalue("cdsp", "mulmat_algotype", g_hexagon_appcfg.mulmat_algotype, 0);
    return g_hexagon_appcfg.mulmat_algotype;
}

/**
 * troubleshooting peformance of mulmat on cDSP during development stage
 */
void ggml_backend_hexagon_set_mulmat_algotype(int new_mulmat_algotype) {
    if (new_mulmat_algotype < 0) {
        GGMLHEXAGON_LOG_WARN("invalid new_mulmat_algotype");
        return;
    }
    std::string cfg_filename = std::string(g_hexagon_appcfg.runtime_libpath) + std::string(g_hexagon_appcfg.cfgfilename);
    GGMLHEXAGON_LOG_VERBOSE("load hexagon appcfg from %s", cfg_filename.c_str());
    hexagon_appcfg hexagoncfg_instance;
    GGMLHEXAGON_LOG_VERBOSE("set_hexagon_cfg with new_mulmat_algotype %d", new_mulmat_algotype);
    hexagoncfg_instance.modify_hexagon_config(cfg_filename, new_mulmat_algotype);
    hexagoncfg_instance.load(cfg_filename);
    hexagoncfg_instance.dump([](const std::string & section, const std::string & key, const std::string value) {
        std::ostringstream  tmposs;
        tmposs << "section[" << std::setw(10) << std::left << section << "],[" << std::setw(25) << std::left << key << "] = [" << value << "]";
        GGMLHEXAGON_LOG_VERBOSE("%s", tmposs.str().c_str());
    });
}

static bool ggmlhexagon_check_valid_appcfg() {
    if (g_hexagon_appcfg.offload_cgraph_type != 0 && g_hexagon_appcfg.offload_cgraph_type != 2) {
        GGMLHEXAGON_LOG_WARN("invalid offload_cgraph_type %d, reset to 2 (only 0=per-op and 2=ION-batch supported)",
                             g_hexagon_appcfg.offload_cgraph_type);
        g_hexagon_appcfg.offload_cgraph_type = 2;
    }

    if (g_hexagon_appcfg.thread_counts > 6) {
        GGMLHEXAGON_LOG_WARN("invalid thread_counts %d, reset to 6", g_hexagon_appcfg.thread_counts);
        g_hexagon_appcfg.thread_counts = 6;
    }

    if (g_hexagon_appcfg.dump_diag_info > 1) {
        GGMLHEXAGON_LOG_WARN("invalid dump_diag_info %d, reset to 0", g_hexagon_appcfg.dump_diag_info);
        g_hexagon_appcfg.dump_diag_info = 0;
    }

    if (g_hexagon_appcfg.ggml_dsp_use_hvx > 1) {
        GGMLHEXAGON_LOG_WARN("invalid ggml_dsp_use_hvx %d, reset to 1", g_hexagon_appcfg.ggml_dsp_use_hvx);
        g_hexagon_appcfg.ggml_dsp_use_hvx = 1;
    }
    if (g_hexagon_appcfg.enable_q_mulmat > 1) {
        GGMLHEXAGON_LOG_WARN("invalid enable_q_mulmat %d, reset to 1", g_hexagon_appcfg.enable_q_mulmat);
        g_hexagon_appcfg.enable_q_mulmat = 1;
    }

    return true;
}

static void ggmlhexagon_print_running_timestamp(ggml_backend_hexagon_context * ctx) {
    char timestamp[GGMLHEXAGON_TMPBUF_LEN];
    memset(timestamp, 0, GGMLHEXAGON_TMPBUF_LEN);

    if (ggmlhexagon_is_llamabench_running()) {
        //make llama-bench happy
        return;
    }

    GGMLHEXAGON_LOG_INFO("ggml_hexagon_version:             %s", g_hexagon_appcfg.version);
    ggmlhexagon_get_timestring(timestamp);
    GGMLHEXAGON_LOG_INFO("offload quantize GGML_OP_MUL_MAT: %s", g_hexagon_appcfg.enable_q_mulmat ? "YES" : "NO");
    GGMLHEXAGON_LOG_INFO("offload MUL_MAT types:            %s", g_hexagon_appcfg.enabled_types.empty() ? "ALL" : g_hexagon_appcfg.enabled_types.c_str());
    GGMLHEXAGON_LOG_INFO("using rpc ion memory pool:        %s", ggmlhexagon_use_ion_mempool() ? "YES" : "NO");
    GGMLHEXAGON_LOG_INFO("thread_counts with HWACCEL_CDSP:  %d", g_hexagon_appcfg.thread_counts);
    GGMLHEXAGON_LOG_INFO("mulmat algo type on cDSP:         %d", g_hexagon_appcfg.mulmat_algotype);
    GGMLHEXAGON_LOG_INFO("mulmat min N for DSP offload:     %d", g_hexagon_appcfg.mulmat_min_n);
    GGMLHEXAGON_LOG_INFO("offload cgraph type:              %d", g_hexagon_appcfg.offload_cgraph_type);
    GGMLHEXAGON_LOG_INFO("dump diag info:                   %d", g_hexagon_appcfg.dump_diag_info);
    GGMLHEXAGON_LOG_INFO("ggml-dsp use hvx:                 %d", g_hexagon_appcfg.ggml_dsp_use_hvx);
    GGMLHEXAGON_LOG_INFO("enabled_types:                    %s", g_hexagon_appcfg.enabled_types.c_str());
    GGMLHEXAGON_LOG_INFO("enabled_ops:                      %s", g_hexagon_appcfg.enabled_ops.c_str());
    GGMLHEXAGON_LOG_INFO("running timestamp:%s", timestamp);

    if (1 == g_hexagon_appcfg.enable_profiler) {
        //make sure this function is called only once
        g_hexagon_profiler.profiler_deinit();
    }
}

// =================================================================================================
//  section-4: cDSP helper function
// =================================================================================================
static const char * ggmlhexagon_get_dsp_name(int domain_id) {
    switch (domain_id) {
        case HEXAGON_ADSP:
            return "Hexagon-aDSP";
        case HEXAGON_MDSP:
            return "Hexagon-mDSP";
        case HEXAGON_SDSP:
            return "Hexagon-sDSP";
        case HEXAGON_CDSP:
            return "Hexagon-cDSP";
        case HEXAGON_CDSP1:
            return "Hexagon-cDSP1";
        default:
            return "Hexagon-unknown";
    }
}

static int ggmlhexagon_pd_status_notifier_callback(void * context, int domain, int session, remote_rpc_status_flags_t status){
    int error = AEE_SUCCESS;
    switch (status){
        case  FASTRPC_USER_PD_UP:
            GGMLHEXAGON_LOG_DEBUG("PD is up\n");
            break;
        case  FASTRPC_USER_PD_EXIT:
            GGMLHEXAGON_LOG_DEBUG("PD closed\n");
            break;
        case  FASTRPC_USER_PD_FORCE_KILL:
            GGMLHEXAGON_LOG_DEBUG("PD force kill\n");
            break;
        case  FASTRPC_USER_PD_EXCEPTION:
            GGMLHEXAGON_LOG_DEBUG("PD exception\n");
            break;
        case  FASTRPC_DSP_SSR:
            GGMLHEXAGON_LOG_DEBUG("DSP SSR\n");
            break;
        default :
            error =  AEE_EBADITEM;
            break;
    }
    return error;
}

static domain * ggmlhexagon_get_domain(int domain_id) {
    int size = sizeof(hexagon_supported_domains) / sizeof(domain);

    for (int i = 0; i < size; i++) {
        if (hexagon_supported_domains[i].id == domain_id)
            return &hexagon_supported_domains[i];
    }

    return nullptr;
}

static bool ggmlhexagon_is_cdsp(int domain_id) {
    return (domain_id == HEXAGON_CDSP) || (domain_id == HEXAGON_CDSP1);
}

static bool ggmlhexagon_is_valid_domain_id(int domain_id, int compute_only) {
    int size = sizeof(hexagon_supported_domains) / sizeof(domain);

    if (0 != compute_only) {
        return ggmlhexagon_is_cdsp(domain_id);
    }

    for (int i = 0; i < size; i++) {
        if (hexagon_supported_domains[i].id == domain_id)
            return true;
    }

    return false;
}

static int ggmlhexagon_get_domains_info(const char * domain_type, int * num_domains, fastrpc_domain ** domains_info) {
    int hexagon_err = AEE_SUCCESS;
    int ss_info     = 0;
    void * buffer   = nullptr;
#if 0
typedef enum {
    /** Flag to be used to query list of all available domains */
    ALL_DOMAINS,
    NSP,
    LPASS,
    SDSP,
    MODEM,
    HPASS,
} fastrpc_domain_type;
#endif
    //ss_info = strcmp(domain_type, "NSP") ? HPASS: NSP;
    //forward compatible with new SDK
    ss_info = (0 == memcmp(domain_type, "NSP", 3)) ? 1 : 5;
    system_req_payload req;
    memset(&req, 0, sizeof(system_req_payload));
    req.id = FASTRPC_GET_DOMAINS;
    req.sys.domains = nullptr;
    fastrpc_domain * domain = nullptr;

    if (ss_info != 0) {
        req.sys.flags = DOMAINS_LIST_FLAGS_SET_TYPE(req.sys.flags, ss_info);
    } else {
        req.sys.flags =0;
    }

#ifdef _WIN32
    hexagon_err = AEE_EUNSUPPORTED;
    goto bail;
#endif

    hexagon_err = remote_system_request(&req);
    if (hexagon_err != AEE_SUCCESS) {
        GGMLHEXAGON_LOG_DEBUG("failure in remote_system_request call: %d", hexagon_err);
        goto bail;
    }
    //allocate memory for domain-info array
    req.sys.max_domains = req.sys.num_domains;
    buffer = calloc(req.sys.num_domains, sizeof(fastrpc_domain));
    if (nullptr == buffer) {
        hexagon_err = AEE_ENOMEMORY;
        GGMLHEXAGON_LOG_DEBUG("unable to allocate memory for req.sys.domains");
        goto bail;
    }
    req.sys.domains = static_cast<fastrpc_domain *>(buffer);
    hexagon_err = remote_system_request(&req);
    if (hexagon_err != AEE_SUCCESS) {
        GGMLHEXAGON_LOG_DEBUG("failure in remote_system_request call: %d.\n", hexagon_err);
        goto bail;
    }

    for (int i = 0; i < req.sys.num_domains; i++) {
        //verify that only requested type domains were returned
        domain = &req.sys.domains[i];
        if (domain->type != ss_info) {
            hexagon_err = -1;
            GGMLHEXAGON_LOG_DEBUG("incorrect data received from remote_system_request.\n");
            goto bail;
        }
    }
    *domains_info = req.sys.domains;
    *num_domains  = req.sys.num_domains;

bail:
    if (hexagon_err && !req.sys.domains) {
        free(req.sys.domains);
    }
    return hexagon_err;
}

static int ggmlhexagon_get_dsp_support(int * domain) {
    int hexagon_error = AEE_SUCCESS;
    *domain = HEXAGON_CDSP;

    if (remote_handle_control) {
        struct remote_dsp_capability dsp_capability_domain = {HEXAGON_CDSP, DOMAIN_SUPPORT, 0};
        hexagon_error = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_domain, sizeof(struct remote_dsp_capability));
        if ((hexagon_error & 0xFF) == (AEE_EUNSUPPORTEDAPI & 0xFF)) {
            GGMLHEXAGON_LOG_DEBUG("FastRPC Capability API is not supported on this device");
            goto bail;
        }

        if (0 == dsp_capability_domain.capability) {
            dsp_capability_domain.domain       = HEXAGON_ADSP;
            dsp_capability_domain.attribute_ID = DOMAIN_SUPPORT;
            dsp_capability_domain.capability   = 0;
            hexagon_error = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_domain, sizeof(struct remote_dsp_capability));
            if(dsp_capability_domain.capability) {
                *domain = HEXAGON_ADSP;
            }
        }

        if (hexagon_error != AEE_SUCCESS) {
            GGMLHEXAGON_LOG_DEBUG("get_dsp_support failed with error 0x%x", hexagon_error);
            goto bail;
        }
    } else {
        hexagon_error = AEE_EUNSUPPORTEDAPI;
        GGMLHEXAGON_LOG_DEBUG("remote_dsp_capability interface is not supported on this device");
    }

bail:
    return hexagon_error;
}

static int ggmlhexagon_get_vtcm_info(int domain, uint32_t attr, uint32_t * capability) {
    int hexagon_error = AEE_SUCCESS;
    *capability = 0;

    if (attr == VTCM_PAGE || attr == VTCM_COUNT) {
    } else {
        hexagon_error = AEE_EBADPARM;
        GGMLHEXAGON_LOG_DEBUG("unsupported attr, only VTCM_PAGE and VTCM_COUNT supported");
        goto bail;
    }

    if (remote_handle_control) {
        if (domain == HEXAGON_ADSP || domain == HEXAGON_CDSP) {
            /*
            * query the DSP for VTCM information
            * since the ADSP does not have a dedicated VTCM, we expect the output to be 0
            */
            struct remote_dsp_capability dsp_capability_vtcm_dsp;
            dsp_capability_vtcm_dsp.domain       = (uint32_t)domain;
            dsp_capability_vtcm_dsp.attribute_ID = attr;
            dsp_capability_vtcm_dsp.capability   = (uint32_t)0;
            hexagon_error = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_vtcm_dsp, sizeof(struct remote_dsp_capability));
            if ((hexagon_error & 0xFF) == (AEE_EUNSUPPORTEDAPI & 0xFF)) {
                GGMLHEXAGON_LOG_DEBUG("FastRPC Capability API is not supported on this device");
                GGMLHEXAGON_LOG_DEBUG("running the use case without checking the capability");
                hexagon_error = AEE_SUCCESS;
                goto bail;
            } else if (hexagon_error == AEE_SUCCESS) {
                *capability = dsp_capability_vtcm_dsp.capability;
            } else {
                GGMLHEXAGON_LOG_DEBUG("get_vtcm_info failed with error 0x%x", hexagon_error);
                goto bail;
            }
        } else {
            hexagon_error = AEE_EUNSUPPORTED;
            GGMLHEXAGON_LOG_DEBUG("unsupported domain %d", domain);
            goto bail;
        }
    } else {
        hexagon_error = AEE_EUNSUPPORTEDAPI;
        GGMLHEXAGON_LOG_DEBUG("remote_dsp_capability interface is not supported on this device");
    }

bail:
    return hexagon_error;
}

static bool ggmlhexagon_is_unsignedpd_supported(int domain_id) {
    int hexagon_error = AEE_SUCCESS;
    if (remote_handle_control) {
        struct remote_dsp_capability dsp_capability_domain = {static_cast<uint32_t>(domain_id), UNSIGNED_PD_SUPPORT, 0};
        hexagon_error = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_domain, sizeof(struct remote_dsp_capability));
        if ((hexagon_error & 0xFF) == (AEE_EUNSUPPORTEDAPI & 0xFF)) {
            GGMLHEXAGON_LOG_WARN("FastRPC Capability API is not supported on this device. Falling back to signed pd");
            return false;
        }

        if (hexagon_error) {
            GGMLHEXAGON_LOG_WARN("error 0x%x: FastRPC Capability API failed. falling back to signed pd", hexagon_error);
            return false;
        }

        if (dsp_capability_domain.capability == 1) {
            return true;
        }
    } else {
        hexagon_error = AEE_EUNSUPPORTEDAPI;
        GGMLHEXAGON_LOG_WARN("remote_dsp_capability interface is not supported on this device.falling back to signed pd");
        return false;
    }

    return false;
}

static bool ggmlhexagon_get_unsignedpd_support(void) {
    return ggmlhexagon_is_unsignedpd_supported(HEXAGON_CDSP);
}

static bool ggmlhexagon_is_async_fastrpc_supported(int domain) {
    int hexagon_error = AEE_SUCCESS;
    if (remote_handle_control) {
        if (domain == HEXAGON_CDSP) {
            /*
            * Query the DSP for ASYNC_FASTRPC_SUPPORT information
            * Async fastrpc is supported only on CDSP
            */
            struct remote_dsp_capability dsp_capability_async_support;
            dsp_capability_async_support.domain       = (uint32_t)domain;
            dsp_capability_async_support.attribute_ID = ASYNC_FASTRPC_SUPPORT;
            dsp_capability_async_support.capability   = (uint32_t)0;
            hexagon_error = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_async_support, sizeof(struct remote_dsp_capability));
            if ((hexagon_error & 0xFF) == (AEE_EUNSUPPORTEDAPI & 0xFF)) {
                GGMLHEXAGON_LOG_WARN("FastRPC Capability API is not supported on this device");
                hexagon_error = AEE_SUCCESS;
                goto bail;
            } else if (dsp_capability_async_support.capability == 1) {
                return true;
            }

            if (hexagon_error != AEE_SUCCESS){
                GGMLHEXAGON_LOG_WARN("failed with error 0x%x", hexagon_error);
                goto bail;
            }
        } else {
            hexagon_error = AEE_EUNSUPPORTED;
            GGMLHEXAGON_LOG_WARN("async FastRPC is not supported on domain %d", domain);
            goto bail;
        }
    } else {
        hexagon_error = AEE_EUNSUPPORTEDAPI;
        GGMLHEXAGON_LOG_WARN("remote_dsp_capability interface is not supported on this device");
    }

bail:
    return false;
}

static void ggmlhexagon_set_rpc_latency(remote_handle64 handle, int qos, int latency) {
    int hexagon_error = AEE_SUCCESS;
    (void)latency;  // unused: aligning with QCOM reference, only enable is set

    if (remote_handle_control) {
        // Align with QCOM reference: only enable QoS mode, let DSP decide latency.
        struct remote_rpc_control_latency data;
        memset(&data, 0, sizeof(data));
        data.enable = qos;
        hexagon_error = remote_handle64_control(handle, DSPRPC_CONTROL_LATENCY, (void*)&data, sizeof(data));
        if (hexagon_error != AEE_SUCCESS) {
            GGMLHEXAGON_LOG_WARN("failed with error 0x%x", hexagon_error);
            goto bail;
        } else {
            if (ggmlhexagon_is_llamabench_running()) {
                GGMLHEXAGON_LOG_VERBOSE("set rpc qos %d (DSP default latency)\n", qos);
            } else {
                GGMLHEXAGON_LOG_INFO("set rpc qos %d (DSP default latency)\n", qos);
            }
        }
    } else {
        hexagon_error = AEE_EUNSUPPORTEDAPI;
        GGMLHEXAGON_LOG_WARN("remote_dsp_capability interface is not supported on this device");
    }

bail:
    return;
}

/**
 * set FastRPC thread priority (default unchanged at 192)
 * priority values range from 1 to 255, with smaller values representing higher priorities
 * Unprivileged clients: 64 through 254 (cDSP only)
 * Privileged clients:   1  through 254
 *
 * ref:file:///opt/qcom/Hexagon_SDK/6.2.0.1/docs/software/system_integration.html#priority-levels
 */
static int ggmlhexagon_set_priority(int domain, int priority) {
    int err = 0;

    if (priority < 1) {
        priority = 1;
    }
    if (priority > 255) {
        priority = 255;
    }

    if (remote_session_control) {
        struct remote_rpc_thread_params data;
        data.domain     = domain;
        data.prio       = priority;
        data.stack_size = -1;
        err = remote_session_control(FASTRPC_THREAD_PARAMS, (void *)&data, sizeof(data));
        if (err != AEE_SUCCESS) {
            GGMLHEXAGON_LOG_WARN("remote_session_control failed with 0x%x when setting thread priority\n", err);
        } else {
            GGMLHEXAGON_LOG_VERBOSE("thread priority set to %d\n", priority);
        }
    } else {
        GGMLHEXAGON_LOG_WARN("cannot set thread priority\n");
    }
    return err;
}

static bool ggmlhexagon_is_status_notification_supported(int domain) {
    int hexagon_error = AEE_SUCCESS;

    if (remote_handle_control) {
        /*
        * Query the DSP for STATUS_NOTIFICATION_SUPPORT information
        * DSP User PD status notification Support
        */
        struct remote_dsp_capability dsp_capability_status_notification_support;
        dsp_capability_status_notification_support.domain       = (uint32_t)domain;
        dsp_capability_status_notification_support.attribute_ID = STATUS_NOTIFICATION_SUPPORT;
        dsp_capability_status_notification_support.capability   = (uint32_t)0;
        hexagon_error = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_status_notification_support, sizeof(struct remote_dsp_capability));
        if ((hexagon_error & 0xFF) == (AEE_EUNSUPPORTEDAPI & 0xFF)) {
            GGMLHEXAGON_LOG_WARN("FastRPC Capability API is not supported on this device");
            hexagon_error = AEE_SUCCESS;
            goto bail;
        } else if (1 == dsp_capability_status_notification_support.capability) {
            return true;
        }

        if (hexagon_error != AEE_SUCCESS){
            GGMLHEXAGON_LOG_WARN("failed with error 0x%x", hexagon_error);
            goto bail;
        }
    } else {
        hexagon_error = AEE_EUNSUPPORTEDAPI;
        GGMLHEXAGON_LOG_WARN("remote_dsp_capability interface is not supported on this device");
    }

bail:
    return false;
}

static int ggmlhexagon_get_hmx_support_info(int domain, uint32_t attr, uint32_t * capability) {
    int hexagon_error = AEE_SUCCESS;
    *capability = 0;

    if (attr != HMX_SUPPORT_SPATIAL && attr != HMX_SUPPORT_DEPTH) {
        hexagon_error = AEE_EBADPARM;
        GGMLHEXAGON_LOG_WARN("unsupported attr, only HMX_SUPPORT_SPATIAL and HMX_SUPPORT_DEPTH supported");
        goto bail;
    }

    if (remote_handle_control) {
        if (domain == HEXAGON_CDSP) {
            /*
            * Query the DSP for HMX SUPPORT information
            * HMX is supported on CDSP only
            */
            struct remote_dsp_capability dsp_capability_hmx_dsp;
            dsp_capability_hmx_dsp.domain       = (uint32_t)domain;
            dsp_capability_hmx_dsp.attribute_ID = attr;
            dsp_capability_hmx_dsp.capability   = (uint32_t)0;
            hexagon_error = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_hmx_dsp, sizeof(struct remote_dsp_capability));
            if ((hexagon_error & 0xFF) == (AEE_EUNSUPPORTEDAPI & 0xFF)) {
                GGMLHEXAGON_LOG_DEBUG("FastRPC Capability API is not supported on this device");
                hexagon_error = AEE_SUCCESS;
                goto bail;
            }
            else if (hexagon_error == AEE_SUCCESS) {
                *capability = dsp_capability_hmx_dsp.capability;
            } else {
                GGMLHEXAGON_LOG_DEBUG("get_hmx_support_info failed with Error 0x%x", hexagon_error);
                goto bail;
            }
        } else {
            hexagon_error = AEE_EUNSUPPORTED;
            GGMLHEXAGON_LOG_DEBUG("HMX support is not there for domain %d", domain);
            goto bail;
        }
    } else {
        hexagon_error = AEE_EUNSUPPORTEDAPI;
        GGMLHEXAGON_LOG_DEBUG("remote_dsp_capability interface is not supported on this device");
    }

bail:
    return hexagon_error;
}

static int ggmlhexagon_get_hvx_arch_ver(int domain, uint32_t * capability) {
    int hexagon_error = AEE_SUCCESS;
    *capability = 0;
    if(remote_handle_control) {
        /*
        * Query the Hexagon processor architecture version information
        */
        struct remote_dsp_capability dsp_capability_arch_ver;
        dsp_capability_arch_ver.domain       = (uint32_t)domain;
        dsp_capability_arch_ver.attribute_ID = ARCH_VER;
        dsp_capability_arch_ver.capability   = (uint32_t)0;
        hexagon_error = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_arch_ver, sizeof(struct remote_dsp_capability));
        if ((hexagon_error & 0xFF) == (AEE_EUNSUPPORTEDAPI & 0xFF)) {
            GGMLHEXAGON_LOG_DEBUG("FastRPC Capability API is not supported on this device");
            hexagon_error = AEE_SUCCESS;
            goto bail;
        } else if (hexagon_error == AEE_SUCCESS) {
            *capability = dsp_capability_arch_ver.capability & 0xFF;
        } else {
            GGMLHEXAGON_LOG_DEBUG("get_hex_arch_ver failed with error 0x%x", hexagon_error);
            goto bail;
        }
    } else {
        hexagon_error = AEE_EUNSUPPORTEDAPI;
        GGMLHEXAGON_LOG_DEBUG("remote_dsp_capability interface is not supported on this device");
    }

bail:
    return hexagon_error;
}

static int ggmlhexagon_get_hvx_support_info(int domain, uint32_t attr, uint32_t * capability)
{
    int hexagon_error = AEE_SUCCESS;
    *capability = 0;
    if (attr == HVX_SUPPORT_64B) {
        hexagon_error = AEE_EBADPARM;
        GGMLHEXAGON_LOG_DEBUG("latest targets have 128 byte HVX register, use HVX_SUPPORT_128B instead of HVX_SUPPORT_64B");
        goto bail;
    }

    if (attr != HVX_SUPPORT_128B) {
        hexagon_error = AEE_EBADPARM;
        GGMLHEXAGON_LOG_DEBUG("unsupported attr. only HVX_SUPPORT_128B supported");
        goto bail;
    }

    if (remote_handle_control) {
        if (domain == HEXAGON_CDSP) {
            /*
            * Query the DSP for HVX SUPPORT information
            * HVX is supported on CDSP only
            */
            struct remote_dsp_capability dsp_capability_hvx_dsp;
            dsp_capability_hvx_dsp.domain       = (uint32_t)domain;
            dsp_capability_hvx_dsp.attribute_ID = attr;
            dsp_capability_hvx_dsp.capability   = (uint32_t)0;
            hexagon_error = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_hvx_dsp, sizeof(struct remote_dsp_capability));
            if ((hexagon_error & 0xFF)==(AEE_EUNSUPPORTEDAPI & 0xFF)) {
                GGMLHEXAGON_LOG_DEBUG("FastRPC Capability API is not supported on this device");
                hexagon_error = AEE_SUCCESS;
                goto bail;
            } else if (hexagon_error == AEE_SUCCESS) {
                *capability = dsp_capability_hvx_dsp.capability;
            } else {
                GGMLHEXAGON_LOG_DEBUG("failed with error 0x%x", hexagon_error);
                goto bail;
            }
        } else {
            hexagon_error = AEE_EUNSUPPORTED;
            GGMLHEXAGON_LOG_DEBUG("HVX support is not available on domain %d", domain);
            goto bail;
        }
    } else {
        hexagon_error = AEE_EUNSUPPORTEDAPI;
        GGMLHEXAGON_LOG_DEBUG("remote_dsp_capability interface is not supported on this device");
    }

bail:
    return hexagon_error;
}

static int ggmlhexagon_request_status_notifications(int domain_id, void * context, notify_callback_fn call_back_fn) {
    int hexagon_error = AEE_SUCCESS;
    struct remote_rpc_notif_register notif;
    bool status_notification_support;

    notif.context     = context;
    notif.domain      = domain_id;
    notif.notifier_fn = call_back_fn;

    status_notification_support = ggmlhexagon_is_status_notification_supported(domain_id);
    if (status_notification_support) {
        hexagon_error = remote_session_control(FASTRPC_REGISTER_STATUS_NOTIFICATIONS, (void*)&notif, sizeof(notif));
        if (hexagon_error != AEE_SUCCESS) {
            GGMLHEXAGON_LOG_DEBUG("error 0x%x: remote_session_control failed to enable status notifications", hexagon_error);
        }
    } else {
        hexagon_error = AEE_EUNSUPPORTEDAPI;
    }

    return hexagon_error;
}

static int ggmlhexagon_init_rpcmempool(ggml_backend_hexagon_context * ctx) {
    size_t candidate_size   = 0;
    uint8_t * rpc_buffer    = nullptr;
    std::vector<int>        probe_slots;

    int htp_arch = 0;
    htp_arch = ggmlhexagon_probe_dspinfo(ctx);
    if (0 == htp_arch)
        return 1;

    if (nullptr == ctx)
        return 2;
    probe_slots.push_back(1024);
    probe_slots.push_back(1536);
    probe_slots.push_back(2000);
    probe_slots.push_back(2048);
    probe_slots.push_back(1024+2048);
    if (htp_arch > 75) {
        probe_slots.push_back(1024+2048+900);
    } else {
        probe_slots.push_back(1024+2048+200);
    }
    if (2 != g_hexagon_appcfg.offload_cgraph_type) {
        probe_slots.push_back(4096);
    }
    size_t probe_counts     = probe_slots.size();
    for (size_t idx = 0; idx < probe_counts; idx++) {
        rpc_buffer = static_cast<uint8_t *>(rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (probe_slots[idx] * SIZE_IN_MB)));
        if (nullptr == rpc_buffer) {
            GGMLHEXAGON_LOG_DEBUG("alloc rpcmem %d (MiB) failure during probe rpc memory info, reason: %s\n", probe_slots[idx], strerror(errno));
            break;
        } else {
            candidate_size = probe_slots[idx];
            rpcmem_free(rpc_buffer);
            rpc_buffer = nullptr;
        }
    }
    ctx->rpc_mempool_capacity = candidate_size * SIZE_IN_MB;
    GGMLHEXAGON_LOG_DEBUG("rpc memory capacity %ld(%d MiB) for device %d",
                          ctx->rpc_mempool_capacity, ctx->rpc_mempool_capacity / SIZE_IN_MB, ctx->device);
    if (ggmlhexagon_is_llamabench_running()) {
        GGMLHEXAGON_LOG_VERBOSE("capacity of rpc memory %d MiB", ctx->rpc_mempool_capacity / SIZE_IN_MB);
    } else {
        GGMLHEXAGON_LOG_INFO("capacity of rpc memory %d MiB", ctx->rpc_mempool_capacity / SIZE_IN_MB);
    }

    GGML_ASSERT(ctx->rpc_mempool_capacity > (8 * SIZE_IN_MB));
    ctx->rpc_mempool_len = ctx->rpc_mempool_capacity - (8 * SIZE_IN_MB);
    if (2 == g_hexagon_appcfg.offload_cgraph_type) {
        ctx->rpc_mempool = rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, ctx->rpc_mempool_len);
    } else {
        ctx->rpc_mempool = rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS | RPCMEM_TRY_MAP_STATIC, ctx->rpc_mempool_len);
    }
    if (nullptr == ctx->rpc_mempool) {
        GGMLHEXAGON_LOG_WARN("alloc rpc memorypool %ld(%d MiB) failed", ctx->rpc_mempool_len, ctx->rpc_mempool_capacity / SIZE_IN_MB);
        return 2;
    } else {
        GGMLHEXAGON_LOG_DEBUG("alloc rpc memorypool %p successfully %ld(%d MiB)",
                              ctx->rpc_mempool, ctx->rpc_mempool_len,
                              ctx->rpc_mempool_len / SIZE_IN_MB);
    }
    ctx->rpc_mempool_handle = rpcmem_to_fd(ctx->rpc_mempool);
    GGMLHEXAGON_LOG_WARN("rpc mempool handle %d", ctx->rpc_mempool_handle);
    GGMLHEXAGON_LOG_WARN("rpc mempool addr %p", ctx->rpc_mempool);
    GGMLHEXAGON_LOG_WARN("rpc mempool size %lld(%dMB)", ctx->rpc_mempool_len, ctx->rpc_mempool_len/ SIZE_IN_MB);
    if (2 == g_hexagon_appcfg.offload_cgraph_type) {
        // Register ION buffer with FastRPC kernel driver using DELAYED mapping.
        // FASTRPC_MAP_FD_DELAYED: registers fd but does NOT create immediate mapping.
        // Actual DSP-side mapping is deferred until DSP calls HAP_mmap2(fd).
        // Without this registration, invoke() still triggers implicit fd_mmap_create.
        int mmap_err = fastrpc_mmap(ctx->domain_id, ctx->rpc_mempool_handle,
                                     ctx->rpc_mempool, 0, ctx->rpc_mempool_len,
                                     FASTRPC_MAP_FD_DELAYED);
        if (mmap_err != 0) {
            GGMLHEXAGON_LOG_WARN("fastrpc_mmap(DELAYED) returned %d (fd=%d), continuing...",
                                 mmap_err, ctx->rpc_mempool_handle);
        } else {
            GGMLHEXAGON_LOG_WARN("fastrpc_mmap(DELAYED) OK: fd=%d, size=%dMB",
                                 ctx->rpc_mempool_handle, ctx->rpc_mempool_len / SIZE_IN_MB);
        }

        // Register ION pool on DSP side via pure-scalar IDL call.
        // This avoids FastRPC's fdlist_fd_from_buf() scan that triggers
        // implicit fd_mmap_create when dsptensor.data pointers are passed.
        // The DSP will call HAP_mmap2(fd) to get a user-space-accessible VA,
        uint32_t ion_fd = (uint32_t)ctx->rpc_mempool_handle;
        uint32_t size_lo = (uint32_t)(ctx->rpc_mempool_len & 0xFFFFFFFF);
        uint32_t size_hi = (uint32_t)((ctx->rpc_mempool_len >> 32) & 0xFFFFFFFF);

        int reg_err = ggmlop_dsp_register_ion(ctx->ggmlop_handle, ion_fd, size_lo, size_hi);
        if (reg_err != AEE_SUCCESS) {
            GGMLHEXAGON_LOG_ERROR("dsp_register_ion failed: 0x%x", reg_err);
        } else {
            GGMLHEXAGON_LOG_WARN("registered ION base via scalar call: fd=%d, size=%dMB",
                                 ctx->rpc_mempool_handle, ctx->rpc_mempool_len / SIZE_IN_MB);
        }

        // Reserve tail of ION for FP16 weight cache
        // Cache region: [cache_offset, rpc_mempool_len)
        // AP bump allocator must not exceed cache_offset
        // Set to 0 to disable cache; set to e.g. 512 for 512MB cache pool
        // FP16 tiles are ~3.56x larger than Q4_0 weights; cache as many as possible
        const size_t cache_pool_size = (size_t)512 * SIZE_IN_MB;
        if (cache_pool_size > 0 && ctx->rpc_mempool_len > cache_pool_size) {
            ctx->rpc_mempool_cache_offset = ctx->rpc_mempool_len - cache_pool_size;
        } else {
            ctx->rpc_mempool_cache_offset = 0;  // no cache
        }
        GGMLHEXAGON_LOG_WARN("ION layout: total=%zuMB, cache_offset=%zuMB, cache_size=%zuMB, data_region=%zuMB",
                             ctx->rpc_mempool_len / SIZE_IN_MB,
                             ctx->rpc_mempool_cache_offset / SIZE_IN_MB,
                             (ctx->rpc_mempool_len - ctx->rpc_mempool_cache_offset) / SIZE_IN_MB,
                             ctx->rpc_mempool_cache_offset / SIZE_IN_MB);

        // Set up FP16 weight cache region on DSP side
        // Uses special batch_size=0xFFFF to signal cache setup mode
        if (ctx->rpc_mempool_cache_offset > 0) {
            uint32_t cache_offset_lo = (uint32_t)(ctx->rpc_mempool_cache_offset & 0xFFFFFFFF);
            int cache_err = ggmlop_dsp_execute_batch_ion(ctx->ggmlop_handle, cache_offset_lo, 0xFFFF);
            if (cache_err == AEE_SUCCESS) {
                GGMLHEXAGON_LOG_WARN("DSP FP16 weight cache set up: offset=%zuMB, size=%zuMB",
                                     ctx->rpc_mempool_cache_offset / SIZE_IN_MB,
                                     (ctx->rpc_mempool_len - ctx->rpc_mempool_cache_offset) / SIZE_IN_MB);
            } else {
                GGMLHEXAGON_LOG_ERROR("DSP FP16 weight cache setup failed: 0x%x", cache_err);
            }
        }

        // [ION-PROBE] Verify bidirectional ION shared memory access.
        // Call with batch_size=0 → DSP enters probe mode: writes 0xAB at base+0,
        // 0xCD at base+64. AP then reads back to confirm DSP writes are visible.
        {
            int probe_err = ggmlop_dsp_execute_batch_ion(ctx->ggmlop_handle, 0, 0);
            if (probe_err == AEE_SUCCESS && ctx->rpc_mempool) {
                // Invalidate AP-side cache before reading DSP-written data (ION is non-coherent)
                __builtin___clear_cache((char *)ctx->rpc_mempool, (char *)ctx->rpc_mempool + 80);

                const uint8_t * p = (const uint8_t *)ctx->rpc_mempool;
                bool ok_ab = (p[0] == 0xAB && p[1] == 0xAB && p[2] == 0xAB && p[3] == 0xAB);
                bool ok_cd = (p[64] == 0xCD && p[65] == 0xCD && p[66] == 0xCD && p[67] == 0xCD);
                GGMLHEXAGON_LOG_WARN("[AP-PROBE] read back: base+0 = %02x %02x %02x %02x (expect AB) -> %s",
                                     p[0], p[1], p[2], p[3],
                                     ok_ab ? "PASS" : "FAIL");
                GGMLHEXAGON_LOG_WARN("[AP-PROBE] read back: base+64 = %02x %02x %02x %02x (expect CD) -> %s",
                                     p[64], p[65], p[66], p[67],
                                     ok_cd ? "PASS" : "FAIL");
                if (ok_ab && ok_cd) {
                    GGMLHEXAGON_LOG_WARN("=== ION BIDIRECTIONAL R/W VERIFIED: DSP can write, AP can read! ===");
                } else {
                    GGMLHEXAGON_LOG_ERROR("=== ION PROBE FAILED: DSP writes NOT visible on AP side ===");
                }
                // clean up probe patterns
                memset((void *)p, 0, 16);
                memset((void *)(p + 64), 0, 16);
            } else {
                GGMLHEXAGON_LOG_WARN("[AP-PROBE] dsp_execute_batch_ion probe failed: 0x%x", probe_err);
            }

            // [ION-MULTI-INVOKE] Test: verify no repeated mmap/munmap on subsequent invokes.
            // Call dsp_execute_batch_ion N times with different write patterns.
            // Check DSP log for "fastrpc_invoke_fd_mmap_create" — should NOT appear after 1st call.
            {
                const int N_ROUNDS = 5;
                bool multi_ok = true;

                for (int round = 0; round < N_ROUNDS; round++) {
                    uint8_t pattern = (uint8_t)(0xA0 + round);
                    // Write pattern from AP side first (AP→DSP direction)
                    memset((void *)ctx->rpc_mempool, pattern, 16);
                    __builtin___clear_cache((char *)ctx->rpc_mempool,
                                            (char *)ctx->rpc_mempool + 16);

                    int err = ggmlop_dsp_execute_batch_ion(ctx->ggmlop_handle, 0, 0);
                    if (err != AEE_SUCCESS) {
                        GGMLHEXAGON_LOG_ERROR("[MULTI-PROBE] round %d/%d invoke FAILED: 0x%x",
                                              round + 1, N_ROUNDS, err);
                        multi_ok = false;
                        break;
                    }

                    // Read back DSP-written data (DSP→AP direction)
                    __builtin___clear_cache((char *)ctx->rpc_mempool,
                                            (char *)ctx->rpc_mempool + 80);
                    const uint8_t * r = (const uint8_t *)ctx->rpc_mempool;
                    if (r[0] != 0xAB || r[64] != 0xCD) {
                        GGMLHEXAGON_LOG_ERROR("[MULTI-PROBE] round %d/%d data mismatch: "
                                              "base+0=0x%02x base+64=0x%02x",
                                              round + 1, N_ROUNDS, r[0], r[64]);
                        multi_ok = false;
                        break;
                    }

                    GGMLHEXAGON_LOG_WARN("[MULTI-PROBE] round %d/%d PASS (invoke OK, data verified)",
                                         round + 1, N_ROUNDS);

                    // Clean up for next round
                    memset((void *)r, 0, 16);
                    memset((void *)(r + 64), 0, 16);
                }

                if (multi_ok) {
                    GGMLHEXAGON_LOG_WARN("=== MULTI-INVOKE TEST PASSED: %d rounds, NO repeated mmap ===",
                                         N_ROUNDS);
                } else {
                    GGMLHEXAGON_LOG_ERROR("=== MULTI-INVOKE TEST FAILED ===");
                }
            }
        }
    } else {
        remote_register_buf(ctx->rpc_mempool, ctx->rpc_mempool_len, ctx->rpc_mempool_handle);

        // Register ION pool base address on DSP side
        // FastRPC translates dsptensor.data from AP VA to DSP VA automatically
        // Use src1 to pass fd and size information as a special "metadata" tensor
        struct dsptensor ion_base_tensor;
        struct dsptensor ion_meta_tensor;  // metadata tensor for fd and size
        struct dsptensor ion_dst_dummy;
        int32_t dummy = 0;
        int32_t meta_data[16] = {0};  // metadata array

        memset(&ion_base_tensor, 0, sizeof(ion_base_tensor));
        memset(&ion_meta_tensor, 0, sizeof(ion_meta_tensor));
        memset(&ion_dst_dummy, 0, sizeof(ion_dst_dummy));

        // Main tensor: ION buffer base address
        // For remote_register_buf'd ION buffers, FastRPC translates the pointer
        // (not copy data). data_len tells FastRPC how many bytes to validate/map.
        // Must be > sizeof(float) to get real pointer translation, but small enough
        // that stub-layer allocation doesn't fail (data_len * sizeof(float) must fit).
        ion_base_tensor.data = ctx->rpc_mempool;
        ion_base_tensor.data_len = (int)(64 * 1024);  // 64KB: enough for pointer translation, safe for allocation
        ion_base_tensor.type = 0;

        // Metadata tensor: contains fd and size
        // Use a small buffer on stack, FastRPC will copy it to DSP
        meta_data[0] = ctx->rpc_mempool_handle;  // fd
        meta_data[1] = (int32_t)(ctx->rpc_mempool_len & 0xFFFFFFFF);  // size lower 32 bits
        meta_data[2] = (int32_t)((ctx->rpc_mempool_len >> 32) & 0xFFFFFFFF);  // size upper 32 bits
        meta_data[3] = (int32_t)(ctx->rpc_mempool_len >> 20);  // size in MB
        ion_meta_tensor.data = meta_data;
        ion_meta_tensor.data_len = sizeof(meta_data);
        ion_meta_tensor.type = 0;

        ion_dst_dummy.data = &dummy;
        ion_dst_dummy.data_len = sizeof(dummy);

        ggmlop_dsp_execute_task(ctx->ggmlop_handle, GGML_OP_NONE, &ion_base_tensor, &ion_meta_tensor, &ion_dst_dummy);
        GGMLHEXAGON_LOG_INFO("registered ION DSP base, AP VA %p, size=%lldMB, fd=%d",
                             ctx->rpc_mempool, ctx->rpc_mempool_len / SIZE_IN_MB, ctx->rpc_mempool_handle);
    }

    return 0;
}

static void ggmlhexagon_deinit_rpcmempool(ggml_backend_hexagon_context * ctx) {
    if (ctx->rpc_mempool) {
        if (2 != g_hexagon_appcfg.offload_cgraph_type) {
            //deregister rpc memory pool
            remote_register_buf(ctx->rpc_mempool, ctx->rpc_mempool_len, -1);
        }
        GGMLHEXAGON_LOG_DEBUG("free rpc mempool %p", ctx->rpc_mempool);
        rpcmem_free(ctx->rpc_mempool);
        ctx->rpc_mempool = nullptr;
        ctx->rpc_mempool_len = 0;
        ctx->rpc_mempool_capacity = 0;
    }
}

static int ggmlhexagon_probe_dspinfo(ggml_backend_hexagon_context * ctx) {
    if (ctx == nullptr) {
        return 0;
    }
    uint32_t dsp_version = 0;
    int htp_arch         = 0;
    ggmlhexagon_get_hvx_arch_ver(ctx->domain_id, &dsp_version);

    size_t total_mem = ggmlhexagon_get_system_total_memory_in_bytes();
    GGMLHEXAGON_LOG_VERBOSE("system mem size %d MiB", total_mem / SIZE_IN_MB);

    if (dsp_version == 0x68 || dsp_version == 0x69 || dsp_version == 0x73 || dsp_version == 0x75 || dsp_version == 0x79 || dsp_version == 0x81) {
        if (ggmlhexagon_is_llamabench_running()) {
            GGMLHEXAGON_LOG_VERBOSE("dsp arch version 0x%x", dsp_version);
        } else {
            GGMLHEXAGON_LOG_INFO("dsp arch version 0x%x", dsp_version);
        }
        //0x68 -> 68, 0x69 -> 69, 0x73 -> 73, 0x75 -> 75, 0x79 -> 79, 0x81 -> 81
        htp_arch = ggmlhexagon_htparch_hex_to_decimal(dsp_version);
        GGMLHEXAGON_LOG_DEBUG("dsp arch version %d", htp_arch);
        struct qcom_socinfo * socinfo = ggmlhexagon_get_socinfo_from_htparch(htp_arch);
        if (nullptr != socinfo) {
            //got fully description of SoC
            if (ggmlhexagon_is_llamabench_running()) {
                GGMLHEXAGON_LOG_VERBOSE("device info: %s, %s", socinfo->soc_desc, ggmlhexagon_get_htparch_desc(htp_arch));
            } else {
                GGMLHEXAGON_LOG_INFO("device info: %s, %s", socinfo->soc_desc, ggmlhexagon_get_htparch_desc(htp_arch));
            }
        }
    } else {
        GGMLHEXAGON_LOG_WARN("error: dsp arch version 0x%x is not supported", dsp_version);
    }

    uint32_t vtcm_count = 0;
    uint32_t vtcm_page  = 0;
    ggmlhexagon_get_vtcm_info(ctx->domain_id, VTCM_COUNT, &vtcm_count);
    ggmlhexagon_get_vtcm_info(ctx->domain_id, VTCM_PAGE, &vtcm_page);
    ctx->has_vtcm = (vtcm_count > 0 && vtcm_page > 0);

    uint32_t hmx_depth = 0;
    uint32_t hmx_spatial = 0;
    ggmlhexagon_get_hmx_support_info(ctx->domain_id, HMX_SUPPORT_DEPTH, &hmx_depth);
    ggmlhexagon_get_hmx_support_info(ctx->domain_id, HMX_SUPPORT_SPATIAL, &hmx_spatial);

    uint32_t hvx_support_128b = 0;
    ggmlhexagon_get_hvx_support_info(ctx->domain_id, HVX_SUPPORT_128B, &hvx_support_128b);
    ctx->has_hvx = (hvx_support_128b > 0);

    if (ggmlhexagon_is_llamabench_running()) {
        //make llama-bench happy
        GGMLHEXAGON_LOG_VERBOSE("vtcm_count %d", vtcm_count);
        GGMLHEXAGON_LOG_VERBOSE("vtcm_page %d", vtcm_page);
        GGMLHEXAGON_LOG_VERBOSE("hvx_support_128b %d", hvx_support_128b);
        GGMLHEXAGON_LOG_VERBOSE("unsigned pd supported %d", ggmlhexagon_get_unsignedpd_support());
        GGMLHEXAGON_LOG_VERBOSE("async fastrpc supported %d", ggmlhexagon_is_async_fastrpc_supported(ctx->domain_id));
    } else {
        GGMLHEXAGON_LOG_INFO("vtcm_count %d", vtcm_count);
        GGMLHEXAGON_LOG_INFO("vtcm_page %d", vtcm_page);
        GGMLHEXAGON_LOG_INFO("hvx_support_128b %d", hvx_support_128b);
        GGMLHEXAGON_LOG_INFO("unsigned pd supported %d", ggmlhexagon_get_unsignedpd_support());
        GGMLHEXAGON_LOG_INFO("async fastrpc supported %d", ggmlhexagon_is_async_fastrpc_supported(ctx->domain_id));
    }
    GGMLHEXAGON_LOG_INFO("device %d caps: has_vtcm=%d has_hvx=%d", ctx->device, (int)ctx->has_vtcm, (int)ctx->has_hvx);
    return htp_arch;
}

static void ggmlhexagon_deinit_cdsp(ggml_backend_hexagon_context * ctx) {
    int hexagon_error  = AEE_SUCCESS;
    if (ggmlhexagon_is_llamabench_running()) {
        GGMLHEXAGON_LOG_VERBOSE("enter %s", __func__);
    } else {
        GGMLHEXAGON_LOG_INFO("enter %s", __func__);
    }
    if (0 != ctx->ggmlop_handle) {
        hexagon_error = ggmlop_dsp_close(ctx->ggmlop_handle);
        if (AEE_SUCCESS != hexagon_error) {
            GGMLHEXAGON_LOG_WARN("error 0x%x: failed to close ggmlop dsp handle", hexagon_error);
        }
        ctx->ggmlop_handle = 0;
    }

    ggmlhexagon_deinit_rpcmempool(ctx);

    //probe before domain_id is invalidated so AP-side domain queries still work
    ggmlhexagon_probe_dspinfo(ctx);

    ctx->domain_id             = -1;
    if (ggmlhexagon_is_llamabench_running()) {
        GGMLHEXAGON_LOG_VERBOSE("leave %s", __func__);
    } else {
        GGMLHEXAGON_LOG_INFO("leave %s", __func__);
    }
}

static int ggmlhexagon_init_dsp(ggml_backend_hexagon_context * ctx) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    int hexagon_error               = AEE_SUCCESS;

    int htp_arch                    = 0;
    int domain_id                   = HEXAGON_CDSP;

    int unsignedpd_flag             = 1;
    bool is_unsignedpd_enabled      = false;

    domain * my_domain              = NULL;
    const char * uri                = NULL;
    char * ggmlop_domain_uri        = NULL;

    if (nullptr == ctx)
        return 1;
    GGMLHEXAGON_LOG_DEBUG("init Hexagon cDSP with backend %d(%s)", ctx->device, ctx->name);
    if (0 != ctx->ggmlop_handle) {
        GGMLHEXAGON_LOG_DEBUG("already init Hexagon cDSP with backend %d(%s)", ctx->device, ctx->name);
        return 0;
    }
    ctx->ggmlop_handle = 0;

    if (!ggmlhexagon_is_valid_domain_id(domain_id, 0)) {
        hexagon_error = AEE_EBADPARM;
        GGMLHEXAGON_LOG_DEBUG("error 0x%x: invalid domain %d", hexagon_error, domain_id);
        goto bail;
    }

    my_domain = ggmlhexagon_get_domain(domain_id);
    if (nullptr == my_domain) {
        GGMLHEXAGON_LOG_DEBUG("unable to get domain struct %d",  domain_id);
        goto bail;
    }
    uri = my_domain->uri;
    GGMLHEXAGON_LOG_DEBUG("temporary domain uri=%s\n", uri);

    // Reserve new FastRPC session (PD) for additional devices (dev_id > 0)
    // dev_id == 0 reuses the default CDSP PD (session_id=0)
    ctx->session_id = 0;
    if (ctx->device > 0) {
        if (remote_session_control) {
            struct remote_rpc_reserve_new_session n;
            n.domain_name_len  = strlen(CDSP_DOMAIN_NAME);
            n.domain_name      = const_cast<char *>(CDSP_DOMAIN_NAME);
            char sess_name[32];
            snprintf(sess_name, sizeof(sess_name), "Hexagon-cDSP%d", ctx->device);
            n.session_name     = sess_name;
            n.session_name_len = strlen(sess_name);

            int err = remote_session_control(FASTRPC_RESERVE_NEW_SESSION, (void *) &n, sizeof(n));
            if (err != AEE_SUCCESS) {
                GGMLHEXAGON_LOG_WARN("FASTRPC_RESERVE_NEW_SESSION failed for device %d: error 0x%x", ctx->device, err);
                hexagon_error = err;
                goto bail;
            }
            ctx->session_id = n.session_id;
            domain_id       = n.effective_domain_id;
            GGMLHEXAGON_LOG_INFO("reserved new session: device=%d session_id=%d effective_domain_id=%d",
                                 ctx->device, ctx->session_id, domain_id);
        } else {
            GGMLHEXAGON_LOG_WARN("remote_session_control not available, cannot create new PD for device %d", ctx->device);
            hexagon_error = AEE_EUNSUPPORTED;
            goto bail;
        }
    }

    if (1 == unsignedpd_flag) {
        is_unsignedpd_enabled = ggmlhexagon_is_unsignedpd_supported(domain_id);
        if (!is_unsignedpd_enabled) {
            GGMLHEXAGON_LOG_DEBUG("overriding user request for unsigned PD, only signed offload is allowed on domain %d", domain_id);
            unsignedpd_flag = 0;
        }
    }

    ctx->domain_id = domain_id;
    if (ggmlhexagon_is_llamabench_running()) {
        GGMLHEXAGON_LOG_VERBOSE("using Hexagon domain %d(%s)", domain_id, ggmlhexagon_get_dsp_name(domain_id));
        GGMLHEXAGON_LOG_VERBOSE("unsignedpd_enabled %d", is_unsignedpd_enabled);
    } else {
        GGMLHEXAGON_LOG_INFO("using Hexagon domain %d(%s)", domain_id, ggmlhexagon_get_dsp_name(domain_id));
        GGMLHEXAGON_LOG_INFO("unsignedpd_enabled %d", is_unsignedpd_enabled);
    }
    if (is_unsignedpd_enabled) {
        if (remote_session_control) {
            struct remote_rpc_control_unsigned_module data;
            data.enable = 1;
            data.domain = domain_id;
            hexagon_error = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&data, sizeof(data));
            GGMLHEXAGON_LOG_DEBUG("remote_session_control returned %d for configuring unsigned PD success", hexagon_error);
            if (AEE_SUCCESS != hexagon_error) {
                GGMLHEXAGON_LOG_WARN("error 0x%x: remote_session_control failed", hexagon_error);
            }
        } else {
            GGMLHEXAGON_LOG_DEBUG("unsigned PD not supported on this device");
            hexagon_error = AEE_EUNSUPPORTED;
            GGMLHEXAGON_LOG_DEBUG("error 0x%x: remote_session_control interface is not supported on this device", hexagon_error);
        }
    }

    hexagon_error = ggmlhexagon_request_status_notifications(domain_id, (void *)STATUS_CONTEXT, ggmlhexagon_pd_status_notifier_callback);
    if (AEE_SUCCESS != hexagon_error) {
        if (AEE_EUNSUPPORTEDAPI != hexagon_error) {
            GGMLHEXAGON_LOG_WARN("error 0x%x: hexagon_request_status_notifications failed", hexagon_error);
        }
        GGMLHEXAGON_LOG_WARN("error 0x%x: failed to compute on domain %d", hexagon_error, domain_id);
        goto bail;
    }
    //ggmlhexagon_set_priority(domain_id, 160);

    //probe arch and build the versioned dsp skel URI
    htp_arch = ggmlhexagon_probe_dspinfo(ctx);
    GGML_ASSERT(htp_arch != 0);
    char ggmldsp_uri[256];
    snprintf(ggmldsp_uri, sizeof(ggmldsp_uri),
             "file:///libggmldsp-skel-v%u.so?ggmldsp_skel_handle_invoke&_modver=1.0&_idlver=0.0.1",
             htp_arch);

    // For session_id > 0 (new PD), use FASTRPC_GET_URI to obtain the session-specific URI.
    // For session_id == 0 (default CDSP PD), use ggmldsp_uri + domain uri.
    if (ctx->session_id > 0 && remote_session_control) {
        char session_uri[256];
        struct remote_rpc_get_uri u = {};
        u.session_id      = ctx->session_id;
        u.domain_name     = const_cast<char *>(CDSP_DOMAIN_NAME);
        u.domain_name_len = strlen(CDSP_DOMAIN_NAME);
        u.module_uri      = const_cast<char *>(ggmldsp_uri);
        u.module_uri_len  = strlen(ggmldsp_uri);
        u.uri             = session_uri;
        u.uri_len         = sizeof(session_uri);

        int err = remote_session_control(FASTRPC_GET_URI, (void *) &u, sizeof(u));
        if (err == AEE_SUCCESS) {
            ggmlop_domain_uri = strdup(session_uri);
            GGMLHEXAGON_LOG_INFO("session URI for session_id=%d: %s", ctx->session_id, ggmlop_domain_uri);
        } else {
            GGMLHEXAGON_LOG_WARN("FASTRPC_GET_URI failed for session_id=%d: error 0x%x, fallback to %s%s",
                                 ctx->session_id, err, ggmldsp_uri, uri);
        }
    }

    if (NULL == ggmlop_domain_uri) {
        //session_id == 0 or FASTRPC_GET_URI failed
        size_t uri_len = strlen(ggmldsp_uri) + MAX_DOMAIN_NAMELEN;
        ggmlop_domain_uri = (char *)malloc(uri_len);
        if (NULL == ggmlop_domain_uri) {
            goto bail;
        }
        snprintf(ggmlop_domain_uri, uri_len, "%s%s", ggmldsp_uri, uri);
    }

    GGMLHEXAGON_LOG_DEBUG("ggmlop domain uri:%s", ggmlop_domain_uri);
    hexagon_error = ggmlop_dsp_open(ggmlop_domain_uri, &ctx->ggmlop_handle);
    if (AEE_SUCCESS == hexagon_error) {
        if (ggmlhexagon_is_llamabench_running()) {
            GGMLHEXAGON_LOG_VERBOSE("succeed to open domain %d(%s)", domain_id, ggmlhexagon_get_dsp_name(domain_id));
        } else {
            GGMLHEXAGON_LOG_INFO("succeed to open domain %d(%s)", domain_id, ggmlhexagon_get_dsp_name(domain_id));
        }

        ggmlop_dsp_setclocks(ctx->ggmlop_handle, g_hexagon_appcfg.dump_diag_info, g_hexagon_appcfg.offload_cgraph_type, g_hexagon_appcfg.mulmat_algotype, g_hexagon_appcfg.thread_counts);
        ggmlhexagon_set_rpc_latency(ctx->ggmlop_handle, RPC_PM_QOS, 100);
        int result = ggmlhexagon_init_rpcmempool(ctx);
        if (0 != result) {
            GGMLHEXAGON_LOG_INFO("failed to init rpc mempool");
            goto bail;
        }

        //Test HMX functionality after DSP initialization and rpc mempool setup
        //GGMLHEXAGON_LOG_INFO("Running HMX test...");
        //test_hmx_ap(ctx);
    } else {
        GGMLHEXAGON_LOG_INFO("error 0x%x: failed to open domain %d(%s)", hexagon_error, domain_id,
                             ggmlhexagon_get_dsp_name(domain_id));
        goto bail;
    }

    //make sure test-backend-ops get the correct backend name
    snprintf(ctx->name,
             sizeof(ctx->name), "Hexagon-cDSP%d", ctx->device);

    if (NULL != ggmlop_domain_uri) {
        free(ggmlop_domain_uri);
        ggmlop_domain_uri = NULL;
    }
    return 0;

bail:
    if (ggmlop_domain_uri) {
        free(ggmlop_domain_uri);
    }

    ggmlhexagon_deinit_cdsp(ctx);

    return -1;
}

ggml_backend_hexagon_context::ggml_backend_hexagon_context(int dev_id, ggml_backend_dev_t dev)
    : device(dev_id),
      backend(nullptr),
      socinfo{},
      n_threads(6),
      rpc_mempool_capacity(0),
      rpc_mempool_len(0),
      rpc_mempool_usage(0),
      rpc_mempool_cache_offset(0),
      weights_dirty(false),
      rpc_mempool(nullptr),
      rpc_mempool_handle(0),
      rpc_mempool_dsp_base(nullptr),
      ggmlop_handle(0),
      domain_id(HEXAGON_CDSP),
      session_id(0),
      rpc_batch_call_count(0),
      cumulative_p7_us(0),
      cumulative_graph_us(0),
      last_graph_end_us(0),
      buffer_type{},
      has_vtcm(false),
      has_hvx(false) {
    snprintf(name, sizeof(name), "Hexagon-cDSP%d", dev_id);
    snprintf(desc, sizeof(desc), "Qualcomm NPU(cDSP%d)", dev_id);
    snprintf(buft_name, sizeof(buft_name), "hexagon-ion-buffer-%s", name);
    lib[0] = '\0';

    buffer_type.iface.get_name         = ggml_backend_hexagon_buffer_type_name;
    buffer_type.iface.alloc_buffer     = ggml_backend_hexagon_buffer_type_alloc_buffer;
    buffer_type.iface.get_alignment    = ggml_backend_hexagon_buffer_type_get_alignment;
    buffer_type.iface.get_max_size     = ggml_backend_hexagon_buffer_type_get_max_size;
    buffer_type.iface.get_alloc_size   = nullptr;
    buffer_type.iface.is_host          = ggml_backend_hexagon_buffer_is_host;
    buffer_type.device  = dev;
    buffer_type.context = this;

    int result = ggmlhexagon_init_dsp(this);
    if (0 != result) {
        GGMLHEXAGON_LOG_ERROR("init hexagon dsp failure for device %d", dev_id);
    }
}

ggml_backend_hexagon_context::~ggml_backend_hexagon_context() {
    ggmlhexagon_deinit_cdsp(this);
    ggmlhexagon_print_running_timestamp(NULL);
}

static void ggmlhexagon_task_init(struct ggmlhexagon_task * task) {
    memset(task, 0, sizeof(struct ggmlhexagon_task));
}

static int ggmlhexagon_task_add_op(struct ggmlhexagon_task * task, ggml_op op_type, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst) {
    task->op_type = op_type;
    task->src0 = src0;
    task->src1 = src1;
    task->dst = dst;
    return 0;
}

static int ggmlhexagon_task_execute(ggml_backend_hexagon_context * ctx, struct ggmlhexagon_task * task) {
    if (!task->src0 || !task->dst) {
        return AEE_SUCCESS;
    }

    struct dsptensor dsptensor_0;
    struct dsptensor dsptensor_1;
    struct dsptensor dsptensor_2;

    memset(&dsptensor_0, 0, sizeof(dsptensor_0));
    dsptensor_0.data = task->src0->data;
    dsptensor_0.data_len = ggml_nelements(task->src0);
    dsptensor_0.type = task->src0->type;
    dsptensor_0.ne[0] = task->src0->ne[0];
    dsptensor_0.ne[1] = task->src0->ne[1];
    dsptensor_0.ne[2] = task->src0->ne[2];
    dsptensor_0.ne[3] = task->src0->ne[3];
    dsptensor_0.nb[0] = task->src0->nb[0];
    dsptensor_0.nb[1] = task->src0->nb[1];
    dsptensor_0.nb[2] = task->src0->nb[2];
    dsptensor_0.nb[3] = task->src0->nb[3];

    memset(&dsptensor_1, 0, sizeof(dsptensor_1));
    if (task->src1) {
        dsptensor_1.data = task->src1->data;
        dsptensor_1.data_len = ggml_nelements(task->src1);
        dsptensor_1.type = task->src1->type;
        dsptensor_1.ne[0] = task->src1->ne[0];
        dsptensor_1.ne[1] = task->src1->ne[1];
        dsptensor_1.ne[2] = task->src1->ne[2];
        dsptensor_1.ne[3] = task->src1->ne[3];
        dsptensor_1.nb[0] = task->src1->nb[0];
        dsptensor_1.nb[1] = task->src1->nb[1];
        dsptensor_1.nb[2] = task->src1->nb[2];
        dsptensor_1.nb[3] = task->src1->nb[3];
    }

    memset(&dsptensor_2, 0, sizeof(dsptensor_2));
    dsptensor_2.data = task->dst->data;
    dsptensor_2.data_len = ggml_nelements(task->dst);
    dsptensor_2.type = task->dst->type;
    dsptensor_2.ne[0] = task->dst->ne[0];
    dsptensor_2.ne[1] = task->dst->ne[1];
    dsptensor_2.ne[2] = task->dst->ne[2];
    dsptensor_2.ne[3] = task->dst->ne[3];
    dsptensor_2.nb[0] = task->dst->nb[0];
    dsptensor_2.nb[1] = task->dst->nb[1];
    dsptensor_2.nb[2] = task->dst->nb[2];
    dsptensor_2.nb[3] = task->dst->nb[3];
    memcpy(dsptensor_2.op_params, task->dst->op_params, sizeof(dsptensor_2.op_params));

    int hexagon_error = ggmlop_dsp_execute_task(ctx->ggmlop_handle, task->op_type, &dsptensor_0, task->src1 ? &dsptensor_1 : NULL, &dsptensor_2);
    if (AEE_SUCCESS != hexagon_error) {
        GGMLHEXAGON_LOG_WARN("ggmlop_dsp_execute_task failed: %d", hexagon_error);
    }

    return hexagon_error;
}

// ==================== HMX Test Functions ====================
// Test HMX matrix multiplication with known values
// Uses RPC mempool to simulate real offload mulmat behavior
// src0: MxK matrix, all values = 0.5f
// src1: KxN matrix, all values = 1.0f
// Expected result: dst = src0^T * src1, each element = K * 0.5 * 1.0 = 0.5*K
// Note: HMX F16 accumulation has hardware precision characteristics
static int test_hmx_ap(ggml_backend_hexagon_context * ctx) {
    // Use 64x64x64 matrices for HMX test (larger than 32x32)
    // HMX requires all dimensions to be 32-aligned
    int     sizex               = 64;  // K dimension
    int     sizey               = 64;  // M dimension (rows of src0)
    int     sizez               = 64;  // N dimension (rows of src1)

    GGMLHEXAGON_LOG_INFO("DEBUG: sizex=%d, sizey=%d, sizez=%d\n", sizex, sizey, sizez);

    // Calculate aligned sizes (128-byte alignment for RPC mempool)
    size_t src0_size = sizex * sizey * sizeof(float);
    size_t src1_size = sizex * sizez * sizeof(float);
    size_t dst_size = sizey * sizez * sizeof(float);

    size_t src0_size_aligned = ((src0_size + 127) / 128) * 128;
    size_t src1_size_aligned = ((src1_size + 127) / 128) * 128;
    size_t dst_size_aligned = ((dst_size + 127) / 128) * 128;

    // Allocate from RPC mempool (simulating real offload behavior)
    if (nullptr == ctx->rpc_mempool) {
        GGMLHEXAGON_LOG_INFO("rpc_mempool not initialized, skipping HMX test\n");
        return 2;
    }

    size_t aligned_offset = ((ctx->rpc_mempool_usage + 127) / 128) * 128;
    if (aligned_offset + src0_size_aligned + src1_size_aligned + dst_size_aligned > ctx->rpc_mempool_len) {
        GGMLHEXAGON_LOG_INFO("rpc_mempool exhausted, skipping HMX test\n");
        return 2;
    }

    float * src0_data = (float *)((char *)ctx->rpc_mempool + aligned_offset);
    aligned_offset += src0_size_aligned;
    float * src1_data = (float *)((char *)ctx->rpc_mempool + aligned_offset);
    aligned_offset += src1_size_aligned;
    float * dst_data = (float *)((char *)ctx->rpc_mempool + aligned_offset);
    ctx->rpc_mempool_usage = aligned_offset + dst_size_aligned;

    GGMLHEXAGON_LOG_INFO("HMX test allocated from rpc_mempool: src0=%p, src1=%p, dst=%p\n",
                         (void *)src0_data, (void *)src1_data, (void *)dst_data);

    // Initialize tensors with known values
    // Using 0.5 and 1.0 to test HMX F16 accumulation precision
    for (int i = 0; i < sizex * sizey; i++) {
        src0_data[i] = 0.5f;
    }
    for (int i = 0; i < sizex * sizez; i++) {
        src1_data[i] = 1.0f;
    }
    memset(dst_data, 0, dst_size);

    GGMLHEXAGON_LOG_INFO("HMX test: src0(%dx%d), src1(%dx%d), dst(%dx%d)\n",
                         sizex, sizey, sizex, sizez, sizez, sizey);
    GGMLHEXAGON_LOG_INFO("src0 first 4 elements: %.2f %.2f %.2f %.2f\n",
                         src0_data[0], src0_data[1], src0_data[2], src0_data[3]);
    GGMLHEXAGON_LOG_INFO("src1 first 4 elements: %.2f %.2f %.2f %.2f\n",
                         src1_data[0], src1_data[1], src1_data[2], src1_data[3]);

    // Construct dsptensor structures
    struct dsptensor dsptensor_0;
    struct dsptensor dsptensor_1;
    struct dsptensor dsptensor_2;

    memset(&dsptensor_0, 0, sizeof(dsptensor_0));
    dsptensor_0.data = (void *)src0_data;
    dsptensor_0.data_len = src0_size;
    dsptensor_0.type = GGML_TYPE_F32;
    dsptensor_0.ne[0] = sizex;
    dsptensor_0.ne[1] = sizey;
    dsptensor_0.ne[2] = 1;
    dsptensor_0.ne[3] = 1;
    dsptensor_0.nb[0] = sizeof(float);
    dsptensor_0.nb[1] = dsptensor_0.nb[0] * dsptensor_0.ne[0];
    dsptensor_0.nb[2] = dsptensor_0.nb[1] * dsptensor_0.ne[1];
    dsptensor_0.nb[3] = dsptensor_0.nb[2] * dsptensor_0.ne[2];

    memset(&dsptensor_1, 0, sizeof(dsptensor_1));
    dsptensor_1.data = (void *)src1_data;
    dsptensor_1.data_len = src1_size;
    dsptensor_1.type = GGML_TYPE_F32;
    dsptensor_1.ne[0] = sizex;
    dsptensor_1.ne[1] = sizez;
    dsptensor_1.ne[2] = 1;
    dsptensor_1.ne[3] = 1;
    dsptensor_1.nb[0] = sizeof(float);
    dsptensor_1.nb[1] = dsptensor_1.nb[0] * dsptensor_1.ne[0];
    dsptensor_1.nb[2] = dsptensor_1.nb[1] * dsptensor_1.ne[1];
    dsptensor_1.nb[3] = dsptensor_1.nb[2] * dsptensor_1.ne[2];

    memset(&dsptensor_2, 0, sizeof(dsptensor_2));
    dsptensor_2.data = (void *)dst_data;
    dsptensor_2.data_len = dst_size;
    dsptensor_2.type = GGML_TYPE_F32;
    dsptensor_2.ne[0] = sizez;
    dsptensor_2.ne[1] = sizey;
    dsptensor_2.ne[2] = 1;
    dsptensor_2.ne[3] = 1;
    dsptensor_2.nb[0] = sizeof(float);
    dsptensor_2.nb[1] = dsptensor_2.nb[0] * dsptensor_2.ne[0];
    dsptensor_2.nb[2] = dsptensor_2.nb[1] * dsptensor_2.ne[1];
    dsptensor_2.nb[3] = dsptensor_2.nb[2] * dsptensor_2.ne[2];

    // Execute task on DSP using op_type = 168 (test HMX)
    GGMLHEXAGON_LOG_INFO("calling ggmlop_dsp_execute_task with op_type=168\n");
    int hexagon_error = ggmlop_dsp_execute_task(ctx->ggmlop_handle, 168, &dsptensor_0, &dsptensor_1, &dsptensor_2);
    if (AEE_SUCCESS != hexagon_error) {
        GGMLHEXAGON_LOG_WARN("ggmlop_dsp_execute_task failed: %d", hexagon_error);
    }

    // Dump result (first row)
    GGMLHEXAGON_LOG_INFO("dst first row: ");
    for (int j = 0; j < sizez; j++) {
        GGMLHEXAGON_LOG_INFO("%.2f ", dst_data[j]);
    }
    GGMLHEXAGON_LOG_INFO("\n");

    // Verify result: each element should be 64 * 0.5f * 1.0f = 32.0f
    // Note: HMX F16 accumulation may have small precision differences
    float expected = (float)sizex * 0.5f;  // 32.0f
    int errors = 0;
    int nan_count = 0;
    int inf_count = 0;

    for (int i = 0; i < sizey * sizez; i++) {
        float val = dst_data[i];

        // Check for NaN
        if (isnan(val)) {
            if (nan_count < 5) {
                GGMLHEXAGON_LOG_INFO("ERROR: dst[%d] = NaN", i);
            }
            nan_count++;
            errors++;
        }
        // Check for Infinity
        else if (isinf(val)) {
            if (inf_count < 5) {
                GGMLHEXAGON_LOG_INFO("ERROR: dst[%d] = %s\n", i, val > 0 ? "Inf" : "-Inf");
            }
            inf_count++;
            errors++;
        }
        // Check for value mismatch (allow 1% tolerance for HMX F16 precision)
        else if (fabs(val - expected) > expected * 0.01f) {
            if (errors - nan_count - inf_count < 5) {
                GGMLHEXAGON_LOG_INFO("ERROR: dst[%d] = %.6f, expected %.6f (diff=%.6f)\n", i, val, expected, fabs(val - expected));
            }
            errors++;
        }

        if (errors >= 20) {
            GGMLHEXAGON_LOG_INFO("... (more errors truncated)\n");
            break;
        }
    }

    if (errors == 0) {
        GGMLHEXAGON_LOG_INFO("HMX test PASSED: all %d elements = %.2f\n", sizey * sizez, expected);
    } else {
        GGMLHEXAGON_LOG_INFO("HMX test FAILED: %d errors out of %d elements\n", errors, sizey * sizez);
        if (nan_count > 0) {
            GGMLHEXAGON_LOG_INFO("  - NaN count: %d\n", nan_count);
        }
        if (inf_count > 0) {
            GGMLHEXAGON_LOG_INFO("  - Infinity count: %d\n", inf_count);
        }
        if (errors > nan_count + inf_count) {
            GGMLHEXAGON_LOG_INFO("  - Value mismatch count: %d\n", errors - nan_count - inf_count);
        }
    }

    return errors == 0 ? 0 : 1;
}

static bool ggmlhexagon_compute_forward(ggml_backend_hexagon_context * ctx, struct ggml_tensor * op) {
    struct dsptensor dsptensor_0;
    struct dsptensor dsptensor_1;
    struct dsptensor dsptensor_2;
    std::string op_name;
    const char * ggml_opname = ggml_op_name(op->op);
    ggmlhexagon_get_opkey_from_op(op, op_name);

    int hexagon_error               = AEE_SUCCESS;
    ggmlhexagon_op_func_t op_func   = nullptr;
    size_t input_tensor_count       = 2;

    ggml_tensor * src0  = op->src[0];
    ggml_tensor * src1  = op->src[1];
    ggml_tensor * dst   = op;

    int input_size = ggml_nbytes(src0);
    if (nullptr != src1)
        input_size += ggml_nbytes(src1);
    hexagon_perf op_perf(op_name, ggml_opname, input_size, ggml_nbytes(dst));
    op_perf.start();

    input_tensor_count  =  ggmlhexagon_k_op_caps[ggmlhexagon_get_op_index(op)].input_param_count;
    op_func             =  ggmlhexagon_k_op_caps[ggmlhexagon_get_op_index(op)].dsp_op_func;

    if (nullptr != op_func) {
        std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
        dsptensor_0.data        = src0->data;
        dsptensor_0.data_len    = ggml_nbytes(src0);
        dsptensor_0.type        = src0->type;

        dsptensor_0.ne[0] = src0->ne[0];
        dsptensor_0.ne[1] = src0->ne[1];
        dsptensor_0.ne[2] = src0->ne[2];
        dsptensor_0.ne[3] = src0->ne[3];

        dsptensor_0.nb[0] = src0->nb[0];
        dsptensor_0.nb[1] = src0->nb[1];
        dsptensor_0.nb[2] = src0->nb[2];
        dsptensor_0.nb[3] = src0->nb[3];

        if (2 == input_tensor_count) {
            GGML_ASSERT(nullptr != src1);
            dsptensor_1.data        = src1->data;
            dsptensor_1.type        = src1->type;
            dsptensor_1.data_len    = ggml_nbytes(src1);

            dsptensor_1.ne[0] = src1->ne[0];
            dsptensor_1.ne[1] = src1->ne[1];
            dsptensor_1.ne[2] = src1->ne[2];
            dsptensor_1.ne[3] = src1->ne[3];

            dsptensor_1.nb[0] = src1->nb[0];
            dsptensor_1.nb[1] = src1->nb[1];
            dsptensor_1.nb[2] = src1->nb[2];
            dsptensor_1.nb[3] = src1->nb[3];
        }

        dsptensor_2.data        = dst->data;
        dsptensor_2.data_len    = ggml_nbytes(dst);
        dsptensor_2.type        = dst->type;

        dsptensor_2.ne[0] = dst->ne[0];
        dsptensor_2.ne[1] = dst->ne[1];
        dsptensor_2.ne[2] = dst->ne[2];
        dsptensor_2.ne[3] = dst->ne[3];

        dsptensor_2.nb[0] = dst->nb[0];
        dsptensor_2.nb[1] = dst->nb[1];
        dsptensor_2.nb[2] = dst->nb[2];
        dsptensor_2.nb[3] = dst->nb[3];

        memcpy(dsptensor_2.op_params, dst->op_params, GGML_MAX_OP_PARAMS / sizeof(int32_t));
        std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<size_t, std::nano> duration = end_time - start_time;
        GGMLHEXAGON_LOG_DEBUG("pack duration %llu ns", duration.count());

        hexagon_error = op_func(ctx->ggmlop_handle, &dsptensor_0, &dsptensor_1, &dsptensor_2);
        if (AEE_SUCCESS != hexagon_error) {
            GGMLHEXAGON_LOG_WARN("ggmlop %s computation fail on cdsp", ggml_op_name(op->op));
            return false;
        }
    } else if (ggmlhexagon_k_op_caps[ggmlhexagon_get_op_index(op)].supported) {
        struct ggmlhexagon_task task;
        ggmlhexagon_task_init(&task);

        int ret = ggmlhexagon_task_add_op(&task, op->op, src0, src1, dst);
        if (ret != 0) {
            GGMLHEXAGON_LOG_WARN("failed to add op to task");
            return false;
        }

        hexagon_error = ggmlhexagon_task_execute(ctx, &task);
        if (AEE_SUCCESS != hexagon_error) {
            GGMLHEXAGON_LOG_WARN("ggmlop %s computation fail on cdsp via dsp task", ggml_op_name(op->op));
        }
    } else {
        GGMLHEXAGON_LOG_DEBUG("op GGML_OP_%s not supported on cDSP", ggml_op_name(op->op));
        return false;
    }

    op_perf.info();
    return true;
}

// =================================================================================================
//  section-5: implementation of ggml-hexagon backend according to specification in ggml backend subsystem
// =================================================================================================
// Check if a ggml_type is allowed by the enabled_types config filter
// Returns true if the type is in the enabled list, or if the list is empty (all types allowed)
// Only applies to quantized types and F16/BF16; F32 is always allowed
static bool ggmlhexagon_type_is_enabled(enum ggml_type type) {
    if (g_hexagon_appcfg.enabled_types.empty()) {
        return true;
    }
    // F32 is always allowed
    if (type == GGML_TYPE_F32) {
        return true;
    }
    const char * type_name = ggml_type_name(type);
    if (type_name == NULL) {
        return false;
    }
    // Check if type_name appears as a whole word in the comma-separated list
    const std::string & list = g_hexagon_appcfg.enabled_types;
    size_t pos = 0;
    while (pos < list.size()) {
        size_t end = list.find(',', pos);
        if (end == std::string::npos) end = list.size();
        std::string token = list.substr(pos, end - pos);
        // trim whitespace
        size_t start = token.find_first_not_of(" \t");
        size_t last = token.find_last_not_of(" \t");
        if (start != std::string::npos && last != std::string::npos) {
            token = token.substr(start, last - start + 1);
        }
        // "all" keyword enables all types
        if (token.size() == 3 &&
            tolower((unsigned char)token[0]) == 'a' &&
            tolower((unsigned char)token[1]) == 'l' &&
            tolower((unsigned char)token[2]) == 'l') {
            return true;
        }
        // case-insensitive compare (ggml_type_name returns lowercase, cfg may use Q4_0 or q4_0)
        if (token.size() == strlen(type_name)) {
            bool match = true;
            for (size_t i = 0; i < token.size(); ++i) {
                if (tolower((unsigned char)token[i]) != tolower((unsigned char)type_name[i])) {
                    match = false;
                    break;
                }
            }
            if (match) return true;
        }
        pos = end + 1;
    }
    return false;
}

// ref: ggml_hexagon_supported_mul_mat in Qualcomm's official ggml-hexagon backend
static bool ggmlhexagon_supported_mul_mat(const struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    ggmlhexagon_dump_op_info(dst);
    const int64_t m = src0->ne[1];
    const int64_t k = src0->ne[0];
    const int64_t n = src1->ne[1];
    const uint32_t src0_rank    = ggml_n_dims(src0);
    const uint32_t src1_rank    = ggml_n_dims(src1);
    GGMLHEXAGON_LOG_DEBUG("MUL_MAT check: m=%lld, n=%lld, k=%lld, src0_rank=%d, src1_rank=%d", (long long)m, (long long)n, (long long)k, src0_rank, src1_rank);

    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
        return false;
    }

    if (!ggmlhexagon_type_is_enabled(src0->type)) {
        return false;
    }

    if (g_hexagon_appcfg.mulmat_min_n >= n) {
        GGMLHEXAGON_LOG_DEBUG("MUL_MAT quantized N=%lld <= %d, keep on CPU\n", (long long)n, g_hexagon_appcfg.mulmat_min_n);
        return false;
    }

    switch (src0->type) {
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_MXFP4:
#if 0
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_S:
#endif
            if (src0->ne[0] % 32) {
                return false;
            }

            if (ggml_nrows(src0) > 16 * 1024) {
                return false;  // typically the lm-head which would be too large for VTCM
            }

            if (ggml_nrows(src1) > 1024 || src1->ne[2] != 1 || src1->ne[3] != 1) {
                return false;  // no huge batches or broadcasting (for now)
            }

            if (src1->ne[2] != 1 || src1->ne[3] != 1) {
                return false;  // no broadcasting (for now)
            }

            // src0 (weights) must be repacked
            //if (src0->buffer && !ggml_backend_buffer_is_hexagon_repack(src0->buffer)) {
            //    return false;
            //}
            if (1 == g_hexagon_appcfg.enable_q_mulmat) {
                // N-based dispatch: for quantized MUL_MAT, only offload to DSP
                // when N > mulmat_min_n (PP phase) where HMX pipeline is efficient.
                // For N <= mulmat_min_n (decode/small PP), CPU ARM NEON SDOT is
                // faster than any DSP path due to tile waste and per-op overhead.
                // Exception: if weight was repacked to x4x2 in set_tensor,
                // t->data is no longer Q4_0 so CPU GEMV cannot read it -
                // must always offload to DSP.
                if (g_set_tensor_repacked.count(src0->data)) {
                    return true;
                }
                return true;
            } else {
                return false;
            }
            break;

        case GGML_TYPE_F16:
            if (src0->nb[1] < src0->nb[0]) {
                GGMLHEXAGON_LOG_WARN("permuted F16 src0 not supported\n");
                return false;
            }
            if (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                GGMLHEXAGON_LOG_WARN("src1 broadcasting not supported\n");
                return false;
            }
            if (ggml_nrows(src1) > 1024) {
                //return false;  // no huge batches (for now)
            }
            break;

        case GGML_TYPE_F32:
            if (src1->type != GGML_TYPE_F32) {
                return false;
            }
            if (src0->nb[1] < src0->nb[0]) {
                GGMLHEXAGON_LOG_WARN("permuted F32 src0 not supported\n");
                return false;
            }
            if (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                GGMLHEXAGON_LOG_WARN("src1 broadcasting not supported\n");
                return false;
            }
            if (ggml_nrows(src1) > 1024) {
                //GGMLHEXAGON_LOG_WARN("no huge batches");
                return false;  // no huge batches (for now)
            }

            break;

        default:
            return false;
    }

    return true;
}

// Decide whether a FLASH_ATTN_EXT node can be offloaded to the DSP.
// Mirrors the shape/type checks in the JZ kernel (kernels/flash_attn.c) so
// that the AP-side scheduler agrees with what the DSP will actually accept.
static bool ggmlhexagon_supported_flash_attn(const struct ggml_tensor * dst) {
    const struct ggml_tensor * q   = dst->src[0];
    const struct ggml_tensor * k   = dst->src[1];
    const struct ggml_tensor * v   = dst->src[2];
    const struct ggml_tensor * msk = dst->src[3];

    if (!q || !k || !v) {
        return false;
    }
    // Q / K / V shape consistency.
    if (q->ne[0] != k->ne[0]) {
        return false;
    }
    if (k->ne[1] != v->ne[1] || k->ne[2] != v->ne[2] || k->ne[3] != v->ne[3]) {
        return false;
    }
    // Q can be f16 or f32; K/V must be f16 in the JZ kernel.
    if (q->type != GGML_TYPE_F16 && q->type != GGML_TYPE_F32) {
        return false;
    }
    if (k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) {
        return false;
    }
    // dst must be f16 or f32.
    if (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32) {
        return false;
    }
    // Mask (optional) must be f16 if present.
    if (msk && msk->type != GGML_TYPE_F16) {
        return false;
    }
    // Sinks (dst->src[4]) are not yet implemented in the JZ FA kernel.
    if (dst->src[4] != NULL) {
        return false;
    }
    // n_head must fit the per-op slopes scratch in the JZ kernel.
    if (q->ne[2] > 512) {
        return false;
    }
    return true;
}

// True for metadata-only ops that never execute on cDSP.
// Tests iterate every tensor in the graph and call supports_op on each;
// view/reshape/permute parents must be reported as supported.
static bool ggmlhexagon_is_metadata_op(enum ggml_op op) {
    switch (op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_REPEAT:
        //case GGML_OP_CONT:
            return true;
        default:
            return false;
    }
}

// Check if an op is allowed by the enabled_ops config filter
// Returns true if the op is in the enabled list, or if the list is empty (all ops allowed)
static bool ggmlhexagon_op_is_enabled(enum ggml_op op) {
    if (ggmlhexagon_is_metadata_op(op)) {
        return true;
    }
    if (g_hexagon_appcfg.enabled_ops.empty()) {
        return true;
    }
    const char * op_name = ggml_op_name(op);
    // Check if op_name appears as a whole word in the comma-separated list
    const std::string & list = g_hexagon_appcfg.enabled_ops;
    size_t pos = 0;
    while (pos < list.size()) {
        size_t end = list.find(',', pos);
        if (end == std::string::npos) end = list.size();
        std::string token = list.substr(pos, end - pos);
        // trim whitespace
        size_t start = token.find_first_not_of(" \t");
        size_t last = token.find_last_not_of(" \t");
        if (start != std::string::npos && last != std::string::npos) {
            token = token.substr(start, last - start + 1);
        }
        // "all" keyword enables all ops
        if (token.size() == 3 &&
            tolower((unsigned char)token[0]) == 'a' &&
            tolower((unsigned char)token[1]) == 'l' &&
            tolower((unsigned char)token[2]) == 'l') {
            return true;
        }
        // case-insensitive compare
        if (token.size() == strlen(op_name)) {
            bool match = true;
            for (size_t i = 0; i < token.size(); ++i) {
                if (tolower((unsigned char)token[i]) != tolower((unsigned char)op_name[i])) {
                    match = false;
                    break;
                }
            }
            if (match) return true;
        }
        pos = end + 1;
    }
    return false;
}

static bool ggmlhexagon_supports_op_none(ggml_backend_dev_t dev, const struct ggml_tensor * op_tensor) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op_tensor);
    return false;
}

static bool ggmlhexagon_can_handle_op_through_cdsp(ggml_backend_dev_t dev, const struct ggml_tensor * op_tensor) {
    if (ggmlhexagon_is_metadata_op(op_tensor->op)) {
        return true;
    }

    if (!ggmlhexagon_op_is_enabled(op_tensor->op)) {
        return false;
    }

    if (!ggmlhexagon_k_op_caps[ggmlhexagon_get_op_index(op_tensor)].supported) {
        return false;
    }

    ggml_backend_hexagon_context * ctx = (ggml_backend_hexagon_context *)dev->context;
    const ggml_tensor * src0 = op_tensor->src[0];
    const ggml_tensor * src1 = op_tensor->src[1];
    const int src0_rank      = ggml_n_dims(src0);
    const int64_t ne00       = src0->ne[0];
    int src1_rank            = 0;
    if (nullptr != src1) {
        src1_rank = ggml_n_dims(src1);
    }

    // Device-aware MUL_MAT dispatch: quantized MUL_MAT needs VTCM for the fast
    // path, but the DSP has a non-VTCM fallback. We always accept (weights are
    // local to this device), but log when the device lacks VTCM so the fallback
    // is expected.
    if (op_tensor->op == GGML_OP_MUL_MAT && src0 && ggml_is_quantized(src0->type)) {
        if (!ctx->has_vtcm) {
            GGMLHEXAGON_LOG_DEBUG("device %d: quantized MUL_MAT without VTCM, will use non-VTCM fallback", ctx->device);
        }
    }

    switch (op_tensor->op) {
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        {
            ggmlhexagon_dump_op_info(op_tensor);
            if (!ggml_are_same_shape(src0, src1)) {
                return false;
            }
            return ((src0->type == GGML_TYPE_F32) || (src0->type == GGML_TYPE_F16));
        }
        case GGML_OP_PERMUTE:
        {
            return true;
        }
        case GGML_OP_MUL_MAT:
        {
            return ggmlhexagon_supported_mul_mat(op_tensor);
        }
        case GGML_OP_SOFT_MAX:{
            if (!ggml_is_contiguous(op_tensor))
                return false;
            if (!ggml_are_same_shape(src0, op_tensor))
                return false;
            return true;
        }
        case GGML_OP_RMS_NORM:
        {
            ggmlhexagon_dump_op_info(op_tensor);
            if (src0->type != GGML_TYPE_F32 || op_tensor->type != GGML_TYPE_F32)
                return false;
            if (!ggml_is_contiguous(src0))
                return false;
            return true;
        }
        case GGML_OP_UNARY:
        {
            const int unary_op = (int)op_tensor->op_params[0];
            if (unary_op != GGML_UNARY_OP_SILU)
                return false;
            if (src0->type != GGML_TYPE_F32)
                return false;
            return true;
        }
        case GGML_OP_SCALE:
        {
            if (src0->type != GGML_TYPE_F32)
                return false;
            return true;
        }
        case GGML_OP_FLASH_ATTN_EXT:
        {
            return ggmlhexagon_supported_flash_attn(op_tensor);
        }
        default:
            break;
    }
    return false;
}

// Relaxed supports_op for cgraph offload mode (offload_cgraph_type==2).
// Uses op-type-specific validation (type consistency, broadcast support, contiguity)
// but omits the strict size threshold (ne00 >= 1024) that limits per-op granularity.
// This allows the scheduler to form larger subgraphs with more ops per batch,
// reducing FastRPC call overhead (the dominant cost).
static bool ggmlhexagon_can_handle_op_through_cdsp_ion(ggml_backend_dev_t dev, const struct ggml_tensor * op_tensor) {
    if (ggmlhexagon_is_metadata_op(op_tensor->op)) {
        return true;
    }

    if (!ggmlhexagon_op_is_enabled(op_tensor->op)) {
        // Log once per op type so the user can see what enabled_ops keeps on CPU.
        // Without this, the "unsupported_nodes=0" log in graph_compute_ion is
        // misleading: it only reflects ops that survived this scheduler filter.
        static std::unordered_set<int> logged_filtered;
        if (logged_filtered.find((int)op_tensor->op) == logged_filtered.end()) {
            logged_filtered.insert((int)op_tensor->op);
            GGMLHEXAGON_LOG_INFO("op %s filtered by enabled_ops (kept on CPU)",
                                 ggml_op_name(op_tensor->op));
        }
        return false;
    }

    if (!ggmlhexagon_k_op_caps[ggmlhexagon_get_op_index(op_tensor)].supported) {
        static std::unordered_set<int> logged_unsupported;
        if (logged_unsupported.find((int)op_tensor->op) == logged_unsupported.end()) {
            logged_unsupported.insert((int)op_tensor->op);
            GGMLHEXAGON_LOG_INFO("op %s not supported by op_caps (kept on CPU)",
                                 ggml_op_name(op_tensor->op));
        }
        return false;
    }

    ggml_backend_hexagon_context * ctx = (ggml_backend_hexagon_context *)dev->context;
    const ggml_tensor * src0 = op_tensor->src[0];
    const ggml_tensor * src1 = (op_tensor->src[1] != nullptr) ? op_tensor->src[1] : nullptr;
    const ggml_tensor * dst  = op_tensor;

    // Device-aware: log when quantized MUL_MAT runs on a VTCM-less device
    if (op_tensor->op == GGML_OP_MUL_MAT && src0 && ggml_is_quantized(src0->type)) {
        if (!ctx->has_vtcm) {
            GGMLHEXAGON_LOG_DEBUG("device %d: quantized MUL_MAT without VTCM, will use non-VTCM fallback", ctx->device);
        }
    }

    switch (op_tensor->op) {
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        {
            // Type consistency: all operands must be same type (f32 or f16)
            if (src0->type == GGML_TYPE_F32) {
                if (!src1 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)
                    return false;
            } else if (src0->type == GGML_TYPE_F16) {
                if (!src1 || src1->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F16)
                    return false;
            } else {
                return false;
            }
            // dst shape must match src0
            if (!ggml_are_same_shape(src0, dst)) {
                return false;
            }
            // Allow broadcasting of src1 into src0, but reject permuted src1
            if (!ggml_can_repeat(src1, src0) || ggml_is_permuted(src1)) {
                return false;
            }
            return true;
        }
        case GGML_OP_MUL:
        {
            // Binary element-wise: same rules as ADD/SUB
            if (src0->type == GGML_TYPE_F32) {
                if (!src1 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)
                    return false;
            } else if (src0->type == GGML_TYPE_F16) {
                if (!src1 || src1->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F16)
                    return false;
            } else return false;
            if (!ggml_are_same_shape(src0, dst)) return false;
            if (!ggml_can_repeat(src1, src0) || ggml_is_permuted(src1))
                return false;
            return true;
        }
        case GGML_OP_DIV:
        {
            // Binary element-wise: same rules as ADD/SUB/MUL
            if (src0->type == GGML_TYPE_F32) {
                if (!src1 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)
                    return false;
            } else if (src0->type == GGML_TYPE_F16) {
                if (!src1 || src1->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F16)
                    return false;
            } else return false;
            if (!ggml_are_same_shape(src0, dst)) return false;
            if (!ggml_can_repeat(src1, src0) || ggml_is_permuted(src1))
                return false;
            return true;
        }
        case GGML_OP_MUL_MAT:
        {
            return ggmlhexagon_supported_mul_mat(op_tensor);
        }
        case GGML_OP_RMS_NORM:
        {
            // Unary op: src0 -> dst, eps in op_params[0]
            if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)
                return false;
            if (!ggml_is_contiguous(src0))
                return false;
            return true;
        }
        case GGML_OP_ROPE:
        {
            // Ternary: src0(f32/f16 input), src1(i32 positions), dst(f32/f16)
            if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16)
                return false;
            if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16)
                return false;
            if (!src1 || src1->type != GGML_TYPE_I32)
                return false;
            return true;
        }
        case GGML_OP_SOFT_MAX:
        {
            // Softmax with optional mask (src1): src0(f32) -> dst(f32)
            // Mask (src1) can be F16 or F32
            if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)
                return false;
            if (src1 != nullptr && src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32)
                return false;
            return true;
        }
        case GGML_OP_UNARY:
        {
            // Unary ops: SILU, RELU, SIGMOID, GELU, etc.
            // The specific unary operation is stored in op_params[0] as ggml_unary_op
            const int unary_op = (int)dst->op_params[0];
            if (unary_op != GGML_UNARY_OP_SILU)
                return false; // only support SILU for now
            if (src0->type != dst->type) return false;
            if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16)
                return false;
            return true;
        }
        case GGML_OP_SCALE:
        {
            // Unary scale: src0 -> dst (same type), scale in op_params[0]
            if (src0->type != dst->type) return false;
            if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16)
                return false;
            return true;
        }
        case GGML_OP_CPY:
        {
            // CPY kernel only supports f16<->f32, not quantized types
            // Copy: src0 -> dst (may involve type conversion f16<->f32)
            if (src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32)
                return false;
            if (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)
                return false;
            return true;
        }
        case GGML_OP_GET_ROWS:
        {
            if (!src1 || src1->type != GGML_TYPE_I32)
                return false;
            if (dst->type != GGML_TYPE_F32)
                return false;
            if (src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K)
                return false;
            return true;
        }
        case GGML_OP_CONCAT:
        {
            if (src0->type != dst->type)
                return false;
            // DSP kernel only supports F32, F16, I32, I16
            if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 &&
                src0->type != GGML_TYPE_I32 && src0->type != GGML_TYPE_I16)
                return false;
            return true;
        }
        case GGML_OP_REPEAT:
        {
            // DSP kernel only supports F32, F16, I32, I16
            if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 &&
                src0->type != GGML_TYPE_I32 && src0->type != GGML_TYPE_I16)
                return false;
            return true;
        }
        case GGML_OP_DIAG_MASK_INF:
        {
            if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)
                return false;
            // Kernel assumes contiguous layout for flat memcpy and indexing
            if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst))
                return false;
            return true;
        }
        case GGML_OP_FLASH_ATTN_EXT:
        {
            return ggmlhexagon_supported_flash_attn(op_tensor);
        }
        default:
            return true; // other ops in table: trust the table entry
    }
}

struct ggml_backend_hexagon_buffer_context {
    ~ggml_backend_hexagon_buffer_context() {
        if (buffer) {
            if (is_ion_buffer) {
                // Mark the ION pool region as free so it can be reused.
                if (backend_ctx && backend_ctx->rpc_mempool) {
                    const char * buf_ptr = (const char *)buffer;
                    const char * pool_base = (const char *)backend_ctx->rpc_mempool;
                    if (buf_ptr >= pool_base && buf_ptr < pool_base + (ptrdiff_t)backend_ctx->rpc_mempool_len) {
                        size_t buf_offset = (size_t)(buf_ptr - pool_base);
                        for (auto & r : backend_ctx->ion_regions) {
                            if (r.in_use && r.offset == buf_offset) {
                                r.in_use = false;
                                GGMLHEXAGON_LOG_WARN("[FREE] device=%d region offset=%zu size=%zu", backend_ctx->device, r.offset, r.size);
                                break;
                            }
                        }
                    }
                }
            } else {
                ggml_aligned_free(buffer, 0);
            }
        }
    }

    void * buffer       = nullptr;
    size_t buffer_size  = 0;
    bool   is_ion_buffer= false;

    struct ggml_backend_hexagon_context * backend_ctx = nullptr;
};

static void ggml_backend_hexagon_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_hexagon_buffer_context * ctx = (ggml_backend_hexagon_buffer_context *)buffer->context;
    delete ctx;
}

static void * ggml_backend_hexagon_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_hexagon_buffer_context * ctx = (ggml_backend_hexagon_buffer_context *)buffer->context;
    return ctx->buffer;
}

static enum ggml_status ggml_backend_hexagon_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_hexagon_buffer_context * ctx = (ggml_backend_hexagon_buffer_context *)buffer->context;
    GGML_UNUSED(tensor);
    GGML_UNUSED(ctx);
    return GGML_STATUS_SUCCESS;
}

static inline size_t hex_round_up(size_t n, size_t m) {
    return (n + m - 1) & ~(m - 1);
}

static void repack_q4_0_q4x4x2(ggml_tensor * t, const void * data, size_t size, void * dst_buf = nullptr) {
    const int QK4_0 = 32;
    const int QK_Q4_0x4x2 = 256;

    int64_t nrows = ggml_nrows(t);

    size_t row_size    = ggml_row_size(t->type, t->ne[0]);
    size_t row_size_pd = ggml_row_size(t->type, hex_round_up(t->ne[0], QK_Q4_0x4x2));
    size_t row_size_rp = row_size_pd;

    const size_t total_tensor_size = (size_t)nrows * row_size;
    const size_t n_bytes_to_copy = size < total_tensor_size ? size : total_tensor_size;

    const int64_t n_full_rows = n_bytes_to_copy / row_size;
    const size_t  n_rem_bytes = n_bytes_to_copy % row_size;

    uint8_t * out_base = dst_buf ? (uint8_t *)dst_buf : (uint8_t *)t->data;

    void * buf_pd = ggml_aligned_malloc(row_size_pd);
    GGML_ASSERT(buf_pd != NULL);

    void * buf_rp = ggml_aligned_malloc(row_size_rp);
    GGML_ASSERT(buf_rp != NULL);

    // src stride: t->nb[1] handles padded (non-contiguous) tensors (e.g. KV cache views);
    // for contiguous tensors nb[1] == row_size. dst stays contiguous (x4x2 repacked output).
    const size_t src_stride = t->nb[1];

    for (int64_t i = 0; i < n_full_rows; i++) {
        const uint8_t * src = (const uint8_t *) data + (i * src_stride);
        uint8_t *       dst = out_base + (i * row_size);

        memcpy(buf_pd, src, row_size);

        const uint8_t * x = (const uint8_t *) buf_pd;
        uint8_t * y = (uint8_t *) buf_rp;

        const int qk = QK_Q4_0x4x2;
        const int nb = t->ne[0] / qk;

        const int dblk_size = 8 * 2;
        const int qblk_size = qk / 2;
        const int qrow_size = t->ne[0] / 2;
        const int q4_blk_sz = QK4_0 / 2 + 2;

        uint8_t * y_q = y + 0;
        uint8_t * y_d = y + qrow_size;

        for (int ib = 0; ib < nb; ib++) {
            uint8_t qs[QK_Q4_0x4x2];

            for (int j = 0; j < 8; j++) {
                const uint8_t * b = x + (ib * 8 + j) * q4_blk_sz + 2;
                for (int k = 0; k < QK4_0 / 2; k++) {
                    qs[j * QK4_0 + k + 0]         = (b[k] & 0x0F);
                    qs[j * QK4_0 + k + QK4_0 / 2] = (b[k] >> 4);
                }
            }

            uint8_t * q = y_q + (ib * qblk_size);
            for (int j = 0; j < qk / 2; j++) {
                q[j] = (qs[j + 128] << 4) | qs[j];
            }

            uint16_t * d = (uint16_t *) (y_d + ib * dblk_size);
            for (int j = 0; j < 8; j++) {
                const uint16_t * scale = (const uint16_t *)(x + (ib * 8 + j) * q4_blk_sz);
                d[j] = *scale;
            }
        }

        memcpy(dst, buf_rp, row_size);
    }

    if (n_rem_bytes > 0) {
        const uint8_t * src = (const uint8_t *) data + (n_full_rows * src_stride);
        uint8_t *       dst = out_base + (n_full_rows * row_size);

        memset(buf_pd, 0, row_size_pd);
        memcpy(buf_pd, src, n_rem_bytes);

        const uint8_t * x = (const uint8_t *) buf_pd;
        uint8_t * y = (uint8_t *) buf_rp;

        const int qk = QK_Q4_0x4x2;
        const int nb = (t->ne[0] + qk - 1) / qk;

        const int dblk_size = 8 * 2;
        const int qblk_size = qk / 2;
        const int qrow_size = t->ne[0] / 2;
        const int q4_blk_sz = QK4_0 / 2 + 2;

        uint8_t * y_q = y + 0;
        uint8_t * y_d = y + qrow_size;

        for (int ib = 0; ib < nb; ib++) {
            uint8_t qs[QK_Q4_0x4x2] = {0};

            for (int j = 0; j < 8 && (ib * 8 + j) * q4_blk_sz < row_size_pd; j++) {
                const uint8_t * b = x + (ib * 8 + j) * q4_blk_sz + 2;
                for (int k = 0; k < QK4_0 / 2; k++) {
                    qs[j * QK4_0 + k + 0]         = (b[k] & 0x0F);
                    qs[j * QK4_0 + k + QK4_0 / 2] = (b[k] >> 4);
                }
            }

            uint8_t * q = y_q + (ib * qblk_size);
            bool partial = (ib == nb - 1);
            for (int j = 0; j < qk / 2; j++) {
                if (partial) {
                    q[j] = (qs[j * 2 + 1] << 4) | qs[j * 2 + 0];
                } else {
                    q[j] = (qs[j + 128] << 4) | qs[j];
                }
            }

            uint16_t * d = (uint16_t *) (y_d + ib * dblk_size);
            for (int j = 0; j < 8 && (ib * 8 + j) * q4_blk_sz < row_size_pd; j++) {
                const uint16_t * scale = (const uint16_t *)(x + (ib * 8 + j) * q4_blk_sz);
                d[j] = *scale;
            }
        }

        memcpy(dst, buf_rp, n_rem_bytes);
    }

    ggml_aligned_free(buf_pd, row_size_pd);
    ggml_aligned_free(buf_rp, row_size_rp);
}

// Inverse of repack_q4_0_q4x4x2: convert x4x2 layout back to Q4_0.
// Used by get_tensor so CPU backends receive canonical Q4_0 bytes
// when is_host returns false and ggml_backend_tensor_copy takes the slow path.
static void repack_q4x4x2_q4_0(const ggml_tensor * t, void * data, size_t size) {
    const int QK4_0 = 32;
    const int QK_Q4_0x4x2 = 256;

    int64_t nrows = ggml_nrows(t);
    size_t  row_size = ggml_row_size(t->type, t->ne[0]);
    size_t  total = (size_t)nrows * row_size;
    int64_t n_full_rows = (size >= total) ? nrows : (int64_t)(size / row_size);

    const int qk         = QK_Q4_0x4x2;
    const int nb         = t->ne[0] / qk;
    const int qblk_size  = qk / 2;          // 128
    const int dblk_size  = 8 * 2;           // 16
    const int qrow_size  = t->ne[0] / 2;
    const int q4_blk_sz  = QK4_0 / 2 + 2;   // 18

    for (int64_t i = 0; i < n_full_rows; i++) {
        const uint8_t * src = (const uint8_t *) t->data + (i * row_size);
        uint8_t *       dst = (uint8_t *) data + (i * row_size);

        const uint8_t * x_q = src;
        const uint8_t * x_d = src + qrow_size;

        for (int ib = 0; ib < nb; ib++) {
            const uint8_t * q = x_q + (ib * qblk_size);

            uint8_t qs[QK_Q4_0x4x2];
            for (int j = 0; j < qk / 2; j++) {
                qs[j]       = q[j] & 0x0F;
                qs[j + 128] = q[j] >> 4;
            }

            const uint16_t * d_src = (const uint16_t *)(x_d + ib * dblk_size);

            for (int j = 0; j < 8; j++) {
                uint8_t * block = dst + (ib * 8 + j) * q4_blk_sz;
                *(uint16_t *)block = d_src[j];
                uint8_t * b = block + 2;
                for (int k = 0; k < QK4_0 / 2; k++) {
                    b[k] = (qs[j * QK4_0 + k + QK4_0 / 2] << 4) | qs[j * QK4_0 + k];
                }
            }
        }
    }
}

static void ggml_backend_hexagon_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                               ggml_tensor * tensor, const void * data,
                                               size_t offset, size_t size) {
    memcpy((char *)tensor->data + offset, data, size);

    // Mark weights dirty so Phase 6.5 flushes them on the next batch.
    // Phase 6.5 normally skips weight tensors (read-only across batches),
    // but set_tensor modifies them.
    ggml_backend_hexagon_buffer_context * bctx =
        (ggml_backend_hexagon_buffer_context *)buffer->context;
    if (bctx && bctx->is_ion_buffer && bctx->backend_ctx) {
        ggml_backend_hexagon_context * hctx = bctx->backend_ctx;
        const char * dp   = (const char *)tensor->data + offset;
        const char * base = (const char *)hctx->rpc_mempool;
        if (dp >= base && dp < base + (ptrdiff_t)hctx->rpc_mempool_len) {
            hctx->weights_dirty = true;
        }
    }
}

static void ggml_backend_hexagon_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                  struct ggml_tensor * tensor,
                                                  uint8_t value, size_t offset, size_t size) {
    memset((char *)tensor->data + offset, value, size);

    ggml_backend_hexagon_buffer_context * bctx =
        (ggml_backend_hexagon_buffer_context *)buffer->context;
    if (bctx && bctx->is_ion_buffer && bctx->backend_ctx) {
        ggml_backend_hexagon_context * hctx = bctx->backend_ctx;
        const char * dp   = (const char *)tensor->data + offset;
        const char * base = (const char *)hctx->rpc_mempool;
        if (dp >= base && dp < base + (ptrdiff_t)hctx->rpc_mempool_len) {
            hctx->weights_dirty = true;
        }
    }
}

static void ggml_backend_hexagon_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                               const ggml_tensor * tensor,
                                               void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    // Inverse repack: if this tensor was repacked to x4x2 in set_tensor,
    // convert back to canonical Q4_0 so CPU backends see the original layout.
    // Only full-tensor reads (offset==0, full size) are handled; partial reads
    // would require block-aligned inverse transform which is not needed today.
    if (g_set_tensor_repacked.count(tensor->data) &&
        offset == 0 && size == ggml_nbytes(tensor)) {
        repack_q4x4x2_q4_0(tensor, data, size);
    } else {
        memcpy(data, (const char *)tensor->data + offset, size);
    }
}

static bool ggml_backend_hexagon_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                               const struct ggml_tensor * src,
                                               struct ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        size_t nbytes = ggml_nbytes(src);
        memcpy(dst->data, src->data, nbytes);
        return true;
    }

    return false;
}

static void ggml_backend_hexagon_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_hexagon_buffer_context * ctx = (ggml_backend_hexagon_buffer_context *)buffer->context;
    memset(ctx->buffer, value, ctx->buffer_size);
}

static void ggml_backend_hexagon_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_UNUSED(buffer);
    for (size_t copy = 0; copy < n_copies; copy++) {
        memcpy((char *)tensor->data + offset + copy * stride_tensor, (const char *)data + copy * stride_data, size);
    }
}

static void ggml_backend_hexagon_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_UNUSED(buffer);
    for (size_t copy = 0; copy < n_copies; copy++) {
        memcpy((char *)data + copy * stride_data, (const char *)tensor->data + offset + copy * stride_tensor, size);
    }
}

static ggml_backend_buffer_i ggml_backend_hexagon_buffer_interface = {
        /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
        /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
        /* .init_tensor     = */ ggml_backend_hexagon_buffer_init_tensor,
        /* .memset_tensor   = */ ggml_backend_hexagon_buffer_memset_tensor,
        /* .set_tensor      = */ ggml_backend_hexagon_buffer_set_tensor,
        /* .get_tensor      = */ ggml_backend_hexagon_buffer_get_tensor,
        /* .set_tensor_2d   = */ ggml_backend_hexagon_buffer_set_tensor_2d,
        /* .get_tensor_2d   = */ ggml_backend_hexagon_buffer_get_tensor_2d,
        /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
        /* .clear           = */ ggml_backend_hexagon_buffer_clear,
        /* .reset           = */ nullptr,
};

static const char * ggml_backend_hexagon_buffer_type_name(ggml_backend_buffer_type_t buft) {
    struct ggml_backend_hexagon_context * ctx = static_cast<ggml_backend_hexagon_context *>(buft->context);
    if (ctx) {
        return ctx->buft_name;
    }
    return "hexagon-ion-buffer";
}

static ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer(
           ggml_backend_buffer_type_t buft, size_t size) {
    struct ggml_backend_hexagon_context * ctx = static_cast<ggml_backend_hexagon_context *>(buft->context);
    GGML_ASSERT(nullptr != ctx);
    GGMLHEXAGON_LOG_WARN("[ALLOC] ENTER device=%d size=%zu bytes (%.2f MiB)", ctx->device, size, (double)size / (1024.0 * 1024.0));
    ggml_backend_hexagon_buffer_context * buffer_ctx = new ggml_backend_hexagon_buffer_context;
    buffer_ctx->backend_ctx = ctx;
    buffer_ctx->is_ion_buffer = true;

    size_t size_page = 0;
#if defined(__ANDROID__) || defined(__linux__)
    size_page = sysconf(_SC_PAGESIZE);
#endif
    size_t size_aligned = size;
    if (0 != (size_aligned % size_page)) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    GGMLHEXAGON_LOG_DEBUG("device %d(%s)", ctx->device, ctx->name);
    GGML_ASSERT(nullptr != ctx->rpc_mempool);
    GGMLHEXAGON_LOG_WARN("device=%d size %ld(%d MiB), rpc_mempool_usage %ld(%d MiB), rpc_mempool_len %ld(%d MiB)",
                          ctx->device, size, size / SIZE_IN_MB, ctx->rpc_mempool_usage, ctx->rpc_mempool_usage / SIZE_IN_MB,
                          ctx->rpc_mempool_len, ctx->rpc_mempool_len / SIZE_IN_MB);

    size_t data_limit = ctx->rpc_mempool_cache_offset > 0 ? ctx->rpc_mempool_cache_offset : ctx->rpc_mempool_len;

    // Try to reuse a free region (best fit)
    size_t best_idx = (size_t)-1;
    size_t best_waste = (size_t)-1;
    for (size_t ri = 0; ri < ctx->ion_regions.size(); ri++) {
        const auto & r = ctx->ion_regions[ri];
        if (!r.in_use && r.size >= size_aligned) {
            size_t waste = r.size - size_aligned;
            if (waste < best_waste) {
                best_waste = waste;
                best_idx = ri;
            }
        }
    }

    if (best_idx != (size_t)-1) {
        // Reuse free region
        auto & r = ctx->ion_regions[best_idx];
        buffer_ctx->buffer      = (char *)ctx->rpc_mempool + r.offset;
        buffer_ctx->buffer_size = size_aligned;  // actual requested size, not region size
        r.in_use = true;
        if (r.offset + size_aligned > ctx->rpc_mempool_usage) {
            ctx->rpc_mempool_usage = r.offset + size_aligned;
        }
        GGMLHEXAGON_LOG_WARN("[ALLOC] device=%d reuse free region: offset=%zu size=%zu (requested=%zu, waste=%zu)",
                             ctx->device, r.offset, r.size, size_aligned, r.size - size_aligned);
    } else {
        // Allocate new region from bump allocator tail
        size_t aligned_offset = ((ctx->rpc_mempool_usage + 127) / 128) * 128;
        if (aligned_offset + size_aligned <= data_limit) {
            buffer_ctx->buffer      = (char *)ctx->rpc_mempool + aligned_offset;
            buffer_ctx->buffer_size = size_aligned;
            ctx->rpc_mempool_usage  = aligned_offset + size_aligned;
            // Record new region
            ion_pool_region new_region;
            new_region.offset = aligned_offset;
            new_region.size   = size_aligned;
            new_region.in_use = true;
            ctx->ion_regions.push_back(new_region);
            GGMLHEXAGON_LOG_WARN("[ALLOC] device=%d new region: offset=%zu size=%zu", ctx->device, aligned_offset, size_aligned);
        } else {
            GGMLHEXAGON_LOG_WARN("device=%d ion pool exhausted: needed %zu MiB, remaining %zu MiB -- falling back to system memory",
                                 ctx->device, size_aligned / SIZE_IN_MB,
                                 (data_limit - ctx->rpc_mempool_usage) / SIZE_IN_MB);
            buffer_ctx->buffer = ggml_aligned_malloc(size_aligned);
            buffer_ctx->buffer_size = size_aligned;
            buffer_ctx->is_ion_buffer = false;
        }
    }

    if (nullptr == buffer_ctx->buffer) {
        GGMLHEXAGON_LOG_WARN("%s: failed to allocate %d MiB\n", __func__, size / SIZE_IN_MB);
        return nullptr;
    } else {
        GGMLHEXAGON_LOG_DEBUG("%s: succeed to allocate %d MiB\n", __func__, size / SIZE_IN_MB);
    }
    // Report allocation result and current mempool state
    if (buffer_ctx->is_ion_buffer) {
        const char * mem_type = "heap";
        const char * data_ptr = (const char *)buffer_ctx->buffer;
        const char * ion_base = (const char *)ctx->rpc_mempool;
        const char * ion_end  = ion_base + ctx->rpc_mempool_len;
        if (data_ptr >= ion_base && data_ptr < ion_end) {
            mem_type = "ION-pool";
        }
        GGMLHEXAGON_LOG_WARN("[ALLOC] device=%d LEAVE size=%zu (%.2f MiB) -> %s, pool_used=%zu/%zu (%.2f%%)",
                             ctx->device, size, (double)size / (1024.0 * 1024.0),
                             mem_type,
                             ctx->rpc_mempool_usage, ctx->rpc_mempool_len,
                             ctx->rpc_mempool_len > 0 ? (double)ctx->rpc_mempool_usage * 100.0 / ctx->rpc_mempool_len : 0.0);
    } else {
        GGMLHEXAGON_LOG_WARN("[ALLOC] device=%d LEAVE size=%zu (%.2f MiB) -> heap", ctx->device, size, (double)size / (1024.0 * 1024.0));
    }
    return ggml_backend_buffer_init(buft, ggml_backend_hexagon_buffer_interface, buffer_ctx, size);
}

static size_t ggml_backend_hexagon_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    //Alignment requirement in bytes
    return 128;
}

static size_t ggml_backend_hexagon_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    struct ggml_backend_hexagon_context * ctx = static_cast<ggml_backend_hexagon_context *>(buft->context);
    GGML_ASSERT(nullptr != ctx);
    GGML_ASSERT(ctx->rpc_mempool_len > (8 * SIZE_IN_MB));
    return ctx->rpc_mempool_len - (8 * SIZE_IN_MB);
}

static bool ggml_backend_buft_is_hexagon(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_hexagon_buffer_type_name;
}

static bool ggml_backend_hexagon_buffer_is_host(ggml_backend_buffer_type_t buft) {
    // Must return true: ION shared memory is system memory (DDR) that both AP
    // and DSP can access via their own VAs. Returning false would prevent the
    // scheduler from falling back unsupported ops (e.g. SET_ROWS on KV cache)
    // to CPU, causing "cannot run the operation" aborts.
    GGML_UNUSED(buft);
    return true;
}

static const char * ggml_backend_hexagon_name(ggml_backend_t backend) {
    ggml_backend_hexagon_context * ctx = (ggml_backend_hexagon_context *) backend->context;
    return ctx->name;
}

static void ggml_backend_hexagon_free(ggml_backend_t backend) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );
    ggml_backend_hexagon_context * ctx = (ggml_backend_hexagon_context *)backend->context;

    // Only free the backend handle. The DSP session is owned by the context
    // (created in ggml_backend_hexagon_reg via the constructor, deinitialized
    // in the destructor). This matches qcom's pattern where freeing a backend
    // does NOT destroy the session, allowing common_fit_params to create/free
    // backends without deinitializing the DSP.
    if (nullptr != ctx->backend) {
        delete backend;
        ctx->backend = nullptr;
    }
    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
}

/*
# offload type on AP side
# 0 = per-op (debug, each op = one FastRPC call)
# 1 = FastRPC-based op-batch (experimental, support has been removed)
# 2 = ION-based op-batch (production, data via ion shared memory)
*/
// MODE 0: per-op FastRPC call (debug only, limited to MUL_MAT and ADD)
static enum ggml_status ggmlhexagon_backend_graph_compute_general(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status result         = GGML_STATUS_SUCCESS;
    ggml_backend_hexagon_context * ctx  = (ggml_backend_hexagon_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE
            || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW
            || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }
        // Mode 0 only supports MUL_MAT and ADD for debugging
        if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_ADD) {
            continue;
        }
        bool ok = ggmlhexagon_compute_forward(ctx, node);
        if (!ok) {
            GGMLHEXAGON_LOG_DEBUG("%s: error: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
        }
    }

    return result;
}

// MODE 1: FastRPC-based op-batch (support has been removed)

// MODE 2: ION-based op-batch — packs all ops into ION shared memory,
//         passes only (offset, size) via FastRPC as doorbell.
//         Avoids FastRPC scatter-gather limits entirely.
static enum ggml_status ggmlhexagon_backend_graph_compute_ion(ggml_backend_t backend, struct ggml_cgraph * cgraph) {

    enum ggml_status result         = GGML_STATUS_SUCCESS;
    ggml_backend_hexagon_context * ctx  = (ggml_backend_hexagon_context *)backend->context;
    int64_t begin_time = ggml_time_us();
    int64_t gap_from_prev = ctx->last_graph_end_us ? (begin_time - ctx->last_graph_end_us) : 0;

    // collect supported ops
    std::vector<ggml_tensor *> supported_nodes;
    std::vector<ggml_tensor *> unsupported_nodes;
    GGMLHEXAGON_LOG_WARN("cgraph has %d total nodes (gap_from_prev=%lld us)", cgraph->n_nodes, (long long)gap_from_prev);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        std::string node_name;
        ggmlhexagon_get_opkey_from_op(node, node_name);
        GGMLHEXAGON_LOG_WARN("node[%d]:%s", i, node_name.c_str());

        if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE
            || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW
            || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }

        //TODO: use relaxed batch table to maximize batching
        if (ggmlhexagon_k_op_caps[ggmlhexagon_get_op_index(node)].supported) {
            supported_nodes.push_back(node);
        } else {
            unsupported_nodes.push_back(node);
        }
    }

    if (!unsupported_nodes.empty()) {
        GGMLHEXAGON_LOG_WARN("special: %d unsupported ops skipped:", (int)unsupported_nodes.size());
        for (auto * n : unsupported_nodes) {
            GGMLHEXAGON_LOG_WARN("  node[%s] op=%s(%d), src0=%p src1=%p dst=%p",
                                  n->name ? n->name : "?", ggml_op_name(n->op), n->op,
                                  n->src[0] ? n->src[0]->data : nullptr,
                                  n->src[1] ? n->src[1]->data : nullptr,
                                  n->data);
        }
    }

    if (supported_nodes.empty()) {
        return result;
    }

    // ====================================================================
    // ION-based multi-op offload: pack all ops into ION shared memory,
    // then pass only (offset, size) via FastRPC as doorbell.
    // This avoids FastRPC's scatter-gather limits on large batches.
    // ====================================================================

    size_t saved_mempool_usage = ctx->rpc_mempool_usage;
    if (!ctx->rpc_mempool || ctx->rpc_mempool_len == 0) {
        GGMLHEXAGON_LOG_WARN("special: no ION mempool, falling back to per-op");
        return result;  // let scheduler use per-op path
    }

    const char * ion_base = (const char *)ctx->rpc_mempool;
    const size_t ion_size = ctx->rpc_mempool_len;

    // Track temporary ION regions (mirrors, batch descriptors, repacked weights)
    // for cleanup after Phase 8. Mark them as free (no tail compaction).
    std::vector<size_t> temp_region_indices;

    // FP16 weight cache is keyed by (src0_data, M, K, type). Static weight
    // tensors keep stable ION addresses across graph_compute calls, so cached
    // tiles remain valid and TG can reuse them (first call miss, later hit).
    // Do NOT reset cache here: resetting kills TG hit rate.

    // ---- Phase 1: collect unique tensor objects (per-tensor, not per-buffer) ----
    // Each tensor object gets its own descriptor with correct ne/nb,
    // even if multiple tensors share the same data buffer (in-place or buffer reuse).
    std::unordered_map<ggml_tensor *, int32_t> tensor_index_map;
    std::vector<ggml_tensor *> tensor_src;

    auto get_or_add_tensor_idx = [&](ggml_tensor * t) -> int32_t {
        if (!t) return -1;
        auto it = tensor_index_map.find(t);
        if (it != tensor_index_map.end()) return it->second;
        int32_t idx = (int32_t)tensor_src.size();
        tensor_index_map[t] = idx;
        tensor_src.push_back(t);
        return idx;
    };

    // ---- Phase 2: build op descriptors ----
    std::vector<hex_op_desc> hex_ops;
    for (auto * node : supported_nodes) {
        hex_op_desc op;
        memset(&op, 0, sizeof(op));
        op.opcode   = node->op;
        memcpy(op.params, node->op_params, sizeof(op.params));
        op.src0_idx = get_or_add_tensor_idx(node->src[0]);
        op.src1_idx = (node->src[1]) ? get_or_add_tensor_idx(node->src[1]) : -1;
        op.src2_idx = (node->src[2]) ? get_or_add_tensor_idx(node->src[2]) : -1;
        op.src3_idx = (node->src[3]) ? get_or_add_tensor_idx(node->src[3]) : -1;
        op.dst_idx  = get_or_add_tensor_idx(node);
        hex_ops.push_back(op);
    }

    const uint32_t n_ops     = (uint32_t)hex_ops.size();
    const uint32_t n_tensors = (uint32_t)tensor_src.size();

    GGMLHEXAGON_LOG_DEBUG("ion-batch %u ops, %u unique tensors", n_ops, n_tensors);
    for (size_t i = 0; i < hex_ops.size(); i++) {
        const hex_op_desc & o = hex_ops[i];
        GGMLHEXAGON_LOG_DEBUG("  ion-op[%zu] %s: src0[t%d] src1[t%d] src2[t%d] dst[t%d]",
                              i, ggml_op_name((ggml_op)o.opcode),
                              o.src0_idx, o.src1_idx, o.src2_idx, o.dst_idx);
    }

    // Identify weight tensors: src0 of MUL_MAT that is NOT dst of any op.
    // Weights are read-only across batches; AP never modifies them per batch,
    // so cache flush/invalidate can be skipped for them.
    std::unordered_set<uint32_t> dst_indices;
    std::unordered_set<uint32_t> weight_indices;
    for (const auto & op : hex_ops) {
        dst_indices.insert(op.dst_idx);
    }
    for (const auto & op : hex_ops) {
        if (op.opcode == GGML_OP_MUL_MAT) {
            if (dst_indices.find(op.src0_idx) == dst_indices.end()) {
                weight_indices.insert(op.src0_idx);
                GGMLHEXAGON_LOG_WARN("weight-cache: tensor[%d] identified as weight (type=%d)",
                                     op.src0_idx, (int)tensor_src[op.src0_idx]->type);
            }
        }
    }

    // ---- Phase 3: compute layout sizes ----
    const uint32_t hdr_size      = (uint32_t)sizeof(hex_batch_hdr);          // ~24 bytes
    const uint32_t ops_region    = (uint32_t)(n_ops * sizeof(hex_op_desc));  // ~96*N
    const uint32_t tens_region   = (uint32_t)(n_tensors * sizeof(hex_tensor_desc)); // ~104*M
    // align ops/tensors regions
    const uint32_t ops_offset    = hdr_size;
    const uint32_t tensors_offset = ops_offset + ((ops_region + HEX_OP_ALIGN - 1) & ~(HEX_OP_ALIGN - 1));
    const uint32_t total_desc_size = tensors_offset + tens_region;

    // ---- Phase 4: handle heap tensors -> mirror into ION ----
    int64_t t_p4, t_p6, t_p65, t_p7, t_p75, t_p8;
    int64_t t_prev = ggml_time_us();
    // Two-step approach:
    //   Step 1: Collect unique data pointers and compute max mirror size per buffer
    //   Step 2: Allocate one mirror per unique buffer (not per tensor)
    // This ensures: (a) shared buffers get one mirror with max size,
    //               (b) each tensor descriptor gets correct ne/nb.
    //
    // Cache coherency fix: for in-place ops (src0->data == dst->data), the
    // shared mirror causes Phase 6.5 DC CVAC to pollute the dst cache lines
    // with stale src0 data. After DSP writes the MUL result to DRAM, the CPU
    // cache still holds the old src0 data, so Phase 8 copy-back reads stale
    // data. Fix: allocate a separate dst mirror for in-place ops so that
    // Phase 6.5 only flushes the src0 mirror, and the dst mirror is never
    // flushed (CPU cache has no stale data for it).
    struct ion_mirror {
        int32_t  tensor_idx;
        void *   original_data;
        uint32_t mirror_offset;
        uint32_t data_len;
    };
    std::vector<ion_mirror> mirrors;

    // Step 1: Collect unique data pointers and max sizes
    struct buffer_mirror_info {
        uint32_t mirror_offset;
        uint32_t max_data_len;
        bool     allocated;
    };
    std::unordered_map<void *, buffer_mirror_info> buffer_mirrors_map;

    for (int32_t tidx = 0; tidx < (int32_t)n_tensors; tidx++) {
        ggml_tensor * t = tensor_src[tidx];
        if (!t->data) continue;

        const char * data_ptr = (const char *)t->data;
        if (data_ptr >= ion_base && data_ptr < ion_base + (ptrdiff_t)ion_size) {
            continue;  // already in ION pool
        }

        uint32_t t_size = (uint32_t)ggml_nbytes(t);
        auto it = buffer_mirrors_map.find(t->data);
        if (it == buffer_mirrors_map.end()) {
            buffer_mirrors_map[t->data] = {0, t_size, false};
        } else if (t_size > it->second.max_data_len) {
            it->second.max_data_len = t_size;
        }
    }

    // Step 2: Allocate mirrors for each unique data pointer
    size_t data_limit = ctx->rpc_mempool_cache_offset > 0 ? ctx->rpc_mempool_cache_offset : ion_size;
    for (auto & kv : buffer_mirrors_map) {
        void * data_ptr = kv.first;
        buffer_mirror_info & info = kv.second;
        size_t mirror_size = info.max_data_len;
        size_t aligned_offset = (ctx->rpc_mempool_usage + 127u) & ~127u;

        if (aligned_offset + mirror_size > data_limit) {
            GGMLHEXAGON_LOG_WARN("ion-batch: mempool full for mirror (%zu bytes)", mirror_size);
            continue;
        }

        uint32_t moff = (uint32_t)aligned_offset;
        void * ion_buf = (char *)ctx->rpc_mempool + moff;
        ctx->rpc_mempool_usage = aligned_offset + mirror_size;

        // Record mirror as a temporary ION region
        ion_pool_region mirror_region;
        mirror_region.offset = aligned_offset;
        mirror_region.size   = mirror_size;
        mirror_region.in_use = true;
        ctx->ion_regions.push_back(mirror_region);
        temp_region_indices.push_back(ctx->ion_regions.size() - 1);

        memcpy(ion_buf, data_ptr, mirror_size);

        info.mirror_offset = moff;
        info.allocated = true;

        GGMLHEXAGON_LOG_DEBUG("ion-batch: mirror buffer %p -> ION offset=0x%x (%u bytes)",
                              data_ptr, moff, info.max_data_len);
    }

    // Step 3: Build mirrors list for each tensor (for Phase 6 offset lookup and Phase 8 copy-back)
    for (int32_t tidx = 0; tidx < (int32_t)n_tensors; tidx++) {
        ggml_tensor * t = tensor_src[tidx];
        if (!t->data) continue;

        const char * data_ptr = (const char *)t->data;
        if (data_ptr >= ion_base && data_ptr < ion_base + (ptrdiff_t)ion_size) {
            continue;  // already in ION pool
        }

        auto it = buffer_mirrors_map.find(t->data);
        if (it == buffer_mirrors_map.end() || !it->second.allocated) continue;

        ion_mirror m;
        m.tensor_idx    = tidx;
        m.original_data = t->data;
        m.mirror_offset = it->second.mirror_offset;
        m.data_len      = (uint32_t)ggml_nbytes(t);
        mirrors.push_back(m);
    }

    // ---- Phase 4.5: x4x2 repack into ION (NOT t->data, to preserve Q4_0 for CPU GEMV) ----
    // For mulmat_algotype=30, repack Q4_0 weights to x4x2 format.
    // Must NOT repack t->data in-place because CPU-side GEMV (N<=mulmat_min_n)
    // still reads t->data as Q4_0. Instead, repack into the ION mirror (heap weights)
    // or a separate ION region (ION weights), and point the DSP descriptor there.
    std::vector<std::pair<uint32_t, uint32_t>> repacked_ion_weights; // (offset, length)
    static std::unordered_map<const void *, uint32_t> g_x4x2_ion_offsets;
    if (g_hexagon_appcfg.mulmat_algotype == 30) {
        static std::unordered_set<const void *> g_x4x2_repacked;
        // Clear stale entries from previous graph_compute call.
        // ION regions may have been freed and reused by alloc_buffer,
        // making old repack offsets invalid.
        g_x4x2_repacked.clear();
        g_x4x2_ion_offsets.clear();
        for (uint32_t i = 0; i < n_tensors; i++) {
            ggml_tensor * t = tensor_src[i];
            if (!t || !t->data) continue;
            bool is_quant_weight = weight_indices.count(i) && t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16;
            if (!is_quant_weight || t->type != GGML_TYPE_Q4_0) continue;
            const int32_t K = t->ne[0];
            if (K % 256 != 0 || K <= 0) continue;
            if (g_x4x2_repacked.find(t->data) != g_x4x2_repacked.end()) {
                continue;  // already repacked, g_x4x2_ion_offsets has the offset
            }
            if (g_set_tensor_repacked.count(t->data)) {
                // Already repacked in set_tensor: t->data is x4x2 format.
                // Phase 4 mirror already copied x4x2 data to ION mirror.
                // Skip Phase 4.5 repack - Phase 7 will use the ION mirror offset.
                continue;
            }

            const char * dp = (const char *)t->data;
            size_t repack_size = ggml_nbytes(t);

            if (dp >= ion_base && dp < ion_base + (ptrdiff_t)ion_size) {
                // ION weight: allocate separate ION region for repacked data
                size_t aligned_offset = (ctx->rpc_mempool_usage + 127u) & ~127u;
                if (aligned_offset + repack_size > data_limit) {
                    GGMLHEXAGON_LOG_WARN("x4x2 repack: ION mempool full, skipping tensor[%d]", i);
                    continue;
                }
                uint32_t rp_off = (uint32_t)aligned_offset;
                void * rp_buf = (char *)ctx->rpc_mempool + rp_off;
                ctx->rpc_mempool_usage = aligned_offset + repack_size;
                // Record repack as a temporary ION region
                ion_pool_region repack_region;
                repack_region.offset = aligned_offset;
                repack_region.size   = repack_size;
                repack_region.in_use = true;
                ctx->ion_regions.push_back(repack_region);
                temp_region_indices.push_back(ctx->ion_regions.size() - 1);
                GGMLHEXAGON_LOG_WARN("x4x2 repack: ION tensor[%d] K=%d M=%d -> ION offset=0x%x", i, K, (int)t->ne[1], rp_off);
                repack_q4_0_q4x4x2(t, t->data, repack_size, rp_buf);
                g_x4x2_repacked.insert(t->data);
                g_x4x2_ion_offsets[t->data] = rp_off;
                repacked_ion_weights.push_back({rp_off, (uint32_t)repack_size});
            } else {
                // Heap weight: repack into the ION mirror (overwriting the Q4_0 copy)
                auto bmi = buffer_mirrors_map.find(t->data);
                if (bmi != buffer_mirrors_map.end() && bmi->second.allocated) {
                    void * ion_mirror = (char *)ctx->rpc_mempool + bmi->second.mirror_offset;
                    GGMLHEXAGON_LOG_WARN("x4x2 repack: heap tensor[%d] K=%d M=%d -> mirror offset=0x%x", i, K, (int)t->ne[1], bmi->second.mirror_offset);
                    repack_q4_0_q4x4x2(t, t->data, repack_size, ion_mirror);
                    g_x4x2_repacked.insert(t->data);
                    g_x4x2_ion_offsets[t->data] = bmi->second.mirror_offset;
                } else {
                    GGMLHEXAGON_LOG_WARN("x4x2 repack: heap tensor[%d] has no ION mirror, skipping", i);
                }
            }
        }
    }

    // ---- Phase 5: allocate batch descriptor region in ION mempool ----
    size_t batch_align = HEX_BATCH_ALIGN;
    size_t batch_offset_raw = ctx->rpc_mempool_usage;
    size_t batch_offset_aligned = (batch_offset_raw + batch_align - 1) & ~(batch_align - 1);

    if (batch_offset_aligned + total_desc_size > data_limit) {
        GGMLHEXAGON_LOG_ERROR("ion-batch: mempool full for batch desc (%zu bytes at offset %zu)",
                              total_desc_size, batch_offset_aligned);
        // Free temporary mirror regions before returning
        for (size_t ri : temp_region_indices) {
            ctx->ion_regions[ri].in_use = false;
        }
        return result;
    }

    uint32_t batch_offset = (uint32_t)batch_offset_aligned;
    ctx->rpc_mempool_usage = batch_offset_aligned + total_desc_size;
    // Record batch descriptor as a temporary ION region
    ion_pool_region batch_region;
    batch_region.offset = batch_offset_aligned;
    batch_region.size   = total_desc_size;
    batch_region.in_use = true;
    ctx->ion_regions.push_back(batch_region);
    temp_region_indices.push_back(ctx->ion_regions.size() - 1);

    // ---- Phase 6: build descriptors in local buffer, then memcpy to ION ----
    t_p4 = ggml_time_us() - t_prev; t_prev = ggml_time_us();
    std::vector<uint8_t> local_buf(total_desc_size);
    hex_batch_hdr * hdr = (hex_batch_hdr *)local_buf.data();
    memset(hdr, 0, sizeof(*hdr));
    hdr->n_ops         = n_ops;
    hdr->n_tensors    = n_tensors;
    hdr->ops_offset   = ops_offset;
    hdr->tensors_offset = tensors_offset;
    hdr->total_size   = total_desc_size;

    // write op descriptors
    hex_op_desc * ops_out = (hex_op_desc *)(local_buf.data() + ops_offset);
    memcpy(ops_out, hex_ops.data(), ops_region);

    // write tensor descriptors with computed offsets
    hex_tensor_desc * tens_out = (hex_tensor_desc *)(local_buf.data() + tensors_offset);
    for (uint32_t i = 0; i < n_tensors; i++) {
        ggml_tensor * t = tensor_src[i];
        hex_tensor_desc * td = &tens_out[i];
        memset(td, 0, sizeof(*td));

        td->type = (int32_t)t->type;
        td->ne[0] = (int32_t)t->ne[0]; td->ne[1] = (int32_t)t->ne[1];
        td->ne[2] = (int32_t)t->ne[2]; td->ne[3] = (int32_t)t->ne[3];
        td->nb[0] = (int32_t)t->nb[0]; td->nb[1] = (int32_t)t->nb[1];
        td->nb[2] = (int32_t)t->nb[2]; td->nb[3] = (int32_t)t->nb[3];
        memcpy(td->op_params, t->op_params, sizeof(td->op_params));
        td->data_len = (uint32_t)ggml_nbytes(t);

        const char * data_ptr = (const char *)t->data;
        if (data_ptr >= ion_base && data_ptr < ion_base + (ptrdiff_t)ion_size) {
            // ION tensor: direct offset
            td->data_offset = (uint32_t)(data_ptr - ion_base);
            td->flags = weight_indices.count(i) ? 2 : 0;  // 2=weight (skip cache flush)
        } else {
            // heap tensor: look up ION offset
            auto bmi = buffer_mirrors_map.find(t->data);
            if (bmi != buffer_mirrors_map.end() && bmi->second.allocated) {
                td->data_offset = bmi->second.mirror_offset;
                td->flags = 1;  // writable (mirrored)
            } else {
                td->data_offset = 0;
                td->flags = 0;
                GGMLHEXAGON_LOG_WARN("ion-batch: tensor[%d] is non-ION heap without mirror!", i);
            }
        }

        // FP16 weight cache: for quantized weight tensors, set op_params[0]=1
        // to request DSP-side caching (only when cache region is configured)
        // Only applies to mulmat_algotype=32 (HMX with FP16 cache)
        bool is_quant_weight = weight_indices.count(i) && t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16;
        if (is_quant_weight && ctx->rpc_mempool_cache_offset > 0 && g_hexagon_appcfg.mulmat_algotype == 32) {
            const int32_t M = t->ne[1];  // weight columns
            const int32_t K = t->ne[0];  // inner dimension
            if (K % 32 == 0 && M > 0) {
                td->op_params[0] = 1;  // request DSP-side FP16 cache
            }
        }

        // x4x2: mark tensor descriptor as x4x2 format and use repacked ION offset
        // (repack done in Phase 4.5, data is in ION mirror or separate ION region)
        if (is_quant_weight && g_hexagon_appcfg.mulmat_algotype == 30 && t->type == GGML_TYPE_Q4_0) {
            const int32_t K = t->ne[0];
            if (K % 256 == 0 && K > 0) {
                td->type = 200;  // GGML_TYPE_Q4_0x4x2
                auto it = g_x4x2_ion_offsets.find(t->data);
                if (it != g_x4x2_ion_offsets.end()) {
                    td->data_offset = it->second;
                    td->flags = 1;  // mirrored (needs cache flush)
                }
            }
        }
    }

    // ---- DIAGNOSTIC: dump tensor data locations and sample values ----
    if (1 == g_hexagon_appcfg.dump_diag_info) {
        uint32_t n_mirrored = 0, n_no_mirror = 0;
        for (uint32_t i = 0; i < n_tensors; i++) {
            ggml_tensor * t = tensor_src[i];
            const char * dp = (const char *)t->data;
            const char * location = "???";
            if (dp >= ion_base && dp < ion_base + (ptrdiff_t)ion_size) location = "ION";
            else location = "HEAP";
            uint32_t offset = (dp >= ion_base && dp < ion_base + (ptrdiff_t)ion_size)
                              ? (uint32_t)(dp - ion_base) : 0xFFFFFFFFu;
            const hex_tensor_desc * td = &tens_out[i];
            GGMLHEXAGON_LOG_WARN("DIAG tensor[%d] type=%d ne=[%d,%d,%d,%d] nb=[%d,%d,%d,%d] ptr=%p %s off=0x%x nbytes=%u td_off=0x%x flags=%d",
                                 i, (int)t->type,
                                 (int)t->ne[0], (int)t->ne[1], (int)t->ne[2], (int)t->ne[3],
                                 (int)t->nb[0], (int)t->nb[1], (int)t->nb[2], (int)t->nb[3],
                                 (void *)dp, location, offset, (uint32_t)ggml_nbytes(t),
                                 td->data_offset, td->flags);
            if (td->flags == 1) n_mirrored++;
            if (td->flags == 0 && location[0] == 'H') n_no_mirror++;
            // dump first 4 f32 values from src tensors (if f32 type and has data)
            if (t->data && ggml_nbytes(t) >= 16) {
                const float * fv = (const float *)t->data;
                float op_param0 = 0;
                memcpy(&op_param0, t->op_params, sizeof(float));
                GGMLHEXAGON_LOG_WARN("DIAG   sample[%d] f32=[%.4f, %.4f, %.4f, %.4f] op_params[0]=%.8f",
                                     i, fv[0], fv[1], fv[2], fv[3], op_param0);
            }
        }
        GGMLHEXAGON_LOG_WARN("DIAG summary: mirrored=%u no_mirror=%u mempool_usage=%zu/%zu bytes",
                             n_mirrored, n_no_mirror, ctx->rpc_mempool_usage, ctx->rpc_mempool_len);
    }

    // copy entire batch descriptor to ION mempool
    memcpy((char *)ctx->rpc_mempool + batch_offset, local_buf.data(), total_desc_size);

    GGMLHEXAGON_LOG_DEBUG("ion-batch: submitted offset=0x%x size=%u (%u ops, %u tensors)",
                         batch_offset, total_desc_size, n_ops, n_tensors);

    // ---- Phase 6.5: AP -> DSP cache coherency ----
    t_p6 = ggml_time_us() - t_prev; t_prev = ggml_time_us();
    // Flush CPU cache to DRAM so DSP can read AP-written data.
    // Always do DC CVAC first: DMA_BUF_IOCTL_SYNC may succeed but be a no-op
    // on platforms the kernel considers coherent (7us for 4GB = no actual flush).
    {
        // DC CVAC on ALL ION ranges that DSP will read (src + mirrors + repacked + descriptor)
        uint32_t clean_min = ~0u, clean_max = 0;
        for (uint32_t i = 0; i < n_tensors; i++) {
            ggml_tensor * t = tensor_src[i];
            if (!t || !t->data) continue;
            if (weight_indices.count(i) && !ctx->weights_dirty) continue;
            const char * dp = (const char *)t->data;
            if (dp >= ion_base && dp < ion_base + (ptrdiff_t)ion_size) {
                uint32_t off = (uint32_t)(dp - ion_base);
                uint32_t len = (uint32_t)ggml_nbytes(t);
                if (off < clean_min) clean_min = off;
                if (off + len > clean_max) clean_max = off + len;
            }
        }
        for (const auto & m : mirrors) {
            if (m.mirror_offset < clean_min) clean_min = m.mirror_offset;
            uint32_t end = m.mirror_offset + m.data_len;
            if (end > clean_max) clean_max = end;
        }
        for (const auto & rw : repacked_ion_weights) {
            if (rw.first < clean_min) clean_min = rw.first;
            uint32_t end = rw.first + rw.second;
            if (end > clean_max) clean_max = end;
        }
        if (batch_offset < clean_min) clean_min = batch_offset;
        uint32_t desc_end = batch_offset + total_desc_size;
        if (desc_end > clean_max) clean_max = desc_end;

        // Also flush non-op tensors in cgraph not in tensor_src (e.g., test sentinels).
        // Without this, Phase 7.5 DC CIVAC can invalidate cache lines containing
        // unflushed sentinel data, causing sentinel mismatch.
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * t = cgraph->nodes[i];
            if (!t || !t->data) continue;
            const char * dp = (const char *)t->data;
            if (dp >= ion_base && dp < ion_base + (ptrdiff_t)ion_size) {
                uint32_t off = (uint32_t)(dp - ion_base);
                uint32_t len = (uint32_t)ggml_nbytes(t);
                if (off < clean_min) clean_min = off;
                if (off + len > clean_max) clean_max = off + len;
            }
        }

        if (clean_max > clean_min) {
            cpu_dcache_flush_range(ctx, 0, (char *)ctx->rpc_mempool + clean_min, clean_max - clean_min);
            GGMLHEXAGON_LOG_DEBUG("ion-batch: phase6.5 DC CVAC [0x%x, 0x%x] (%u bytes)",
                                  clean_min, clean_max, clean_max - clean_min);
        }
        // Also try DMA_BUF_IOCTL_SYNC as extra safeguard
        int ion_fd = ctx->rpc_mempool_handle;
        if (ion_fd > 0) ion_sync_for_direction(ion_fd, 1);

        ctx->weights_dirty = false;
    }

    // AP-side PRE-CALL diagnostic: log first op's src0 first 4 floats after DC CVAC.
    // Compare with [DSP-DIAG] POST-INVAL to pinpoint cache coherency issues.
    if (n_ops > 0) {
        const hex_op_desc & first_op = hex_ops[0];
        uint32_t s0_idx = first_op.src0_idx;
        if (s0_idx < n_tensors) {
            ggml_tensor * s0_t = tensor_src[s0_idx];
            if (s0_t && s0_t->data) {
                const char * dp = (const char *)s0_t->data;
                uint32_t s0_off = 0xFFFFFFFFu;
                if (dp >= ion_base && dp < ion_base + (ptrdiff_t)ion_size) {
                    s0_off = (uint32_t)(dp - ion_base);
                } else {
                    for (const auto & m : mirrors) {
                        if ((uint32_t)m.tensor_idx == s0_idx) { s0_off = m.mirror_offset; break; }
                    }
                }
                if (s0_off != 0xFFFFFFFFu) {
                    const float * fv = (const float *)((const char *)ctx->rpc_mempool + s0_off);
                    GGMLHEXAGON_LOG_WARN("[AP-PRE] batch first-op src0[tensor%u]: ION_off=0x%x f32=[%.4f, %.4f, %.4f, %.4f]",
                                         s0_idx, s0_off, fv[0], fv[1], fv[2], fv[3]);
                }
            }
        }
    }

    // ---- Phase 7: FastRPC doorbell call (only 2 scalars!) ----
    t_p65 = ggml_time_us() - t_prev; t_prev = ggml_time_us();
    ctx->rpc_batch_call_count++;
    GGMLHEXAGON_LOG_WARN("batch_call #%llu n_ops=%u", ctx->rpc_batch_call_count, n_ops);
    int hexagon_error = ggmlop_dsp_execute_batch_ion(ctx->ggmlop_handle, batch_offset, total_desc_size);

    if (AEE_SUCCESS != hexagon_error) {
        GGMLHEXAGON_LOG_WARN("ggmlop_dsp_execute_batch_ion failed: 0x%x", hexagon_error);
    }

    // ---- Phase 7.3: Post-invoke AP-side verification ----
    if (hexagon_error == AEE_SUCCESS && n_ops > 0) {
        // Log LAST op's dst tensor for general verification
        const hex_op_desc & last_op = hex_ops[n_ops - 1];
        uint32_t last_dst_idx = last_op.dst_idx;
        if (last_dst_idx < n_tensors) {
            ggml_tensor * dst_tensor = tensor_src[last_dst_idx];
            if (dst_tensor && dst_tensor->data) {
                // Find this tensor's ION offset from mirrors
                uint32_t ion_off = 0;
                for (const auto & m : mirrors) {
                    if ((uint32_t)m.tensor_idx == last_dst_idx) { ion_off = m.mirror_offset; break; }
                }
                // If not mirrored (already in ION), try to compute from data pointer
                if (ion_off == 0 && dst_tensor->data >= (void *)ion_base && dst_tensor->data < (void *)(ion_base + ion_size)) {
                    ion_off = (uint32_t)((const char *)dst_tensor->data - ion_base);
                }
                const char * ion_data = (const char *)ctx->rpc_mempool + ion_off;
                const float * ion_vals = (const float *)ion_data;
                const float * ptr_vals  = (const float *)dst_tensor->data;
                GGMLHEXAGON_LOG_WARN("[AP-POST] batch last-op[%u] dst[tensor%u]: ION_f32=[%.4f, %.4f, %.4f, %.4f] PTR_f32=[%.4f, %.4f, %.4f, %.4f] ion_off=0x%x ptr=%p",
                                     n_ops - 1, last_dst_idx,
                                     ion_vals[0], ion_vals[1], ion_vals[2], ion_vals[3],
                                     ptr_vals[0], ptr_vals[1], ptr_vals[2], ptr_vals[3],
                                     ion_off, (void *)dst_tensor->data);
            }
        }
    }

    // ---- Phase 7.5: invalidate CPU cache for DSP-written ION regions ----
    t_p7 = ggml_time_us() - t_prev; t_prev = ggml_time_us();
    // DSP writes results to DRAM via ION buffer, but CPU cache may still hold
    // stale data.  Always do DC CIVAC first: DMA_BUF_IOCTL_SYNC may be a no-op.
    if (hexagon_error == AEE_SUCCESS) {
        uint32_t inval_min = ~0u, inval_max = 0;
        for (uint32_t oi = 0; oi < n_ops; oi++) {
            const hex_op_desc & cur_op = hex_ops[oi];
            uint32_t dst_idx = cur_op.dst_idx;
            if (dst_idx >= n_tensors) continue;
            ggml_tensor * dst_t = tensor_src[dst_idx];
            if (!dst_t || !dst_t->data) continue;

            uint32_t dst_off = 0xFFFFFFFFu;
            const char * dp = (const char *)dst_t->data;
            if (dp >= ion_base && dp < ion_base + (ptrdiff_t)ion_size) {
                dst_off = (uint32_t)(dp - ion_base);
            } else {
                for (const auto & m : mirrors) {
                    if ((uint32_t)m.tensor_idx == dst_idx) { dst_off = m.mirror_offset; break; }
                }
                if (dst_off == 0xFFFFFFFFu) {
                    auto bmi = buffer_mirrors_map.find(dst_t->data);
                    if (bmi != buffer_mirrors_map.end() && bmi->second.allocated)
                        dst_off = bmi->second.mirror_offset;
                }
            }
            if (dst_off == 0xFFFFFFFFu) continue;

            uint32_t dst_len = (uint32_t)ggml_nbytes(dst_t);
            uint32_t start = dst_off & ~63u;
            uint32_t end   = (dst_off + dst_len + 63u) & ~63u;
            if (start < inval_min) inval_min = start;
            if (end > inval_max) inval_max = end;
        }
        if (inval_max > inval_min) {
            cpu_dcache_inval_range(ctx, 0, (const char *)ctx->rpc_mempool + inval_min, inval_max - inval_min);
            GGMLHEXAGON_LOG_DEBUG("ion-batch: phase7.5 DC CIVAC [0x%x, 0x%x] (%u bytes)",
                                  inval_min, inval_max, inval_max - inval_min);
        }
        // Also try DMA_BUF_IOCTL_SYNC as extra safeguard
        int ion_fd = ctx->rpc_mempool_handle;
        if (ion_fd > 0) ion_sync_for_direction(ion_fd, 0);
    }

    // ---- Phase 7.6: Post-CIVAC verification ----
    // Read dst AFTER DC CIVAC to see what the test framework will actually read.
    // Compare with [AP-POST] (pre-CIVAC, AP cache) and DSP-DIAG dst to pinpoint issues.
    if (hexagon_error == AEE_SUCCESS && n_ops > 0) {
        const hex_op_desc & last_op = hex_ops[n_ops - 1];
        uint32_t last_dst_idx = last_op.dst_idx;
        if (last_dst_idx < n_tensors) {
            ggml_tensor * dst_tensor = tensor_src[last_dst_idx];
            if (dst_tensor && dst_tensor->data) {
                const float * ptr_vals = (const float *)dst_tensor->data;
                GGMLHEXAGON_LOG_WARN("[AP-POST-CIVAC] dst[tensor%u]: PTR_f32=[%.4f, %.4f, %.4f, %.4f]",
                                     last_dst_idx, ptr_vals[0], ptr_vals[1], ptr_vals[2], ptr_vals[3]);
            }
        }
    }

    // Reset bump pointer so next graph_compute reuses the same ION pool region.
    // Without this, rpc_mempool_usage only grows and eventually exhausts the pool,
    // causing mirror alloc failure (data_offset=0 -> DSP corrupts model weights).
    ctx->rpc_mempool_usage = saved_mempool_usage;

    // ---- Phase 8: copy-back mirrored results to heap ----
    t_p75 = ggml_time_us() - t_prev; t_prev = ggml_time_us();
    if (hexagon_error == AEE_SUCCESS && !mirrors.empty()) {
        std::unordered_map<void *, std::pair<uint32_t, uint32_t>> copyback_map;
        for (const auto & m : mirrors) {
            auto it = copyback_map.find(m.original_data);
            if (it == copyback_map.end()) {
                copyback_map[m.original_data] = {m.mirror_offset, m.data_len};
            } else {
                if (m.data_len > it->second.second) {
                    it->second.second = m.data_len;
                }
            }
        }
        for (const auto & kv : copyback_map) {
            void * orig_data = kv.first;
            uint32_t moff = kv.second.first;
            uint32_t max_len = kv.second.second;
            memcpy(orig_data, (const char *)ctx->rpc_mempool + moff, max_len);
        }

        // Post-copy-back verification: check last op's dst tensor
        if (1 == g_hexagon_appcfg.dump_diag_info && n_ops > 0) {
            const hex_op_desc & last_op = hex_ops[n_ops - 1];
            uint32_t last_dst_idx = last_op.dst_idx;
            if (last_dst_idx < n_tensors) {
                ggml_tensor * dst_tensor = tensor_src[last_dst_idx];
                if (dst_tensor && dst_tensor->data && ggml_nbytes(dst_tensor) >= 16) {
                    const float * ptr_vals = (const float *)dst_tensor->data;
                    // Find ION offset
                    uint32_t ion_off = 0;
                    for (const auto & m : mirrors) {
                        if ((uint32_t)m.tensor_idx == last_dst_idx) { ion_off = m.mirror_offset; break; }
                    }
                    if (ion_off == 0 && dst_tensor->data >= (void *)ion_base && dst_tensor->data < (void *)(ion_base + ion_size)) {
                        ion_off = (uint32_t)((const char *)dst_tensor->data - ion_base);
                    }
                    const float * ion_vals = (const float *)((const char *)ctx->rpc_mempool + ion_off);
                    GGMLHEXAGON_LOG_WARN("[POST-COPY] op[%u] dst[t%d]: ION=[%.4f, %.4f, %.4f, %.4f] HEAP=[%.4f, %.4f, %.4f, %.4f] ion_off=0x%x",
                                         n_ops - 1, last_dst_idx,
                                         ion_vals[0], ion_vals[1], ion_vals[2], ion_vals[3],
                                         ptr_vals[0], ptr_vals[1], ptr_vals[2], ptr_vals[3],
                                         ion_off);
                }
            }
        }
    }

    // Free temporary ION regions (mirrors, batch descriptors, repacked weights).
    // These are only needed during this graph_compute call and can be reused
    // in the next call. Mark them as free (no tail compaction).
    for (size_t ri : temp_region_indices) {
        ctx->ion_regions[ri].in_use = false;
    }
    t_p8 = ggml_time_us() - t_prev;
    int64_t end_time = ggml_time_us();
    int64_t graph_dur = end_time - begin_time;
    ctx->cumulative_p7_us    += t_p7;
    ctx->cumulative_graph_us += graph_dur;
    ctx->last_graph_end_us   = end_time;
    GGMLHEXAGON_LOG_WARN("ion-batch timing: p4=%lld p6=%lld p6.5=%lld p7=%lld p7.5=%lld p8=%lld (us) ops=%u",
                         (long long)t_p4, (long long)t_p6, (long long)t_p65,
                         (long long)t_p7, (long long)t_p75, (long long)t_p8, n_ops);
    GGMLHEXAGON_LOG_WARN("graph supported_nodes   %d", supported_nodes.size());
    GGMLHEXAGON_LOG_WARN("graph inference duration %lld microseconds (gap_from_prev=%lld us)", (long long)graph_dur, (long long)gap_from_prev);
    GGMLHEXAGON_LOG_WARN("rpc stats: batch_calls=%llu cum_p7=%lld us cum_graph=%lld us avg_p7=%lld us avg_graph=%lld us",
                         (unsigned long long)ctx->rpc_batch_call_count,
                         (long long)ctx->cumulative_p7_us, (long long)ctx->cumulative_graph_us,
                         ctx->rpc_batch_call_count ? (long long)(ctx->cumulative_p7_us / (int64_t)ctx->rpc_batch_call_count) : 0,
                         ctx->rpc_batch_call_count ? (long long)(ctx->cumulative_graph_us / (int64_t)ctx->rpc_batch_call_count) : 0);

    return result;
}

static const char * ggml_backend_hexagon_device_get_name(ggml_backend_dev_t dev) {
    struct ggml_backend_hexagon_context * ctx = static_cast<ggml_backend_hexagon_context *>(dev->context);
    if (nullptr == ctx) {
        GGMLHEXAGON_LOG_ERROR("pls check why ctx is null");
        return "unknown";
    }
    return ctx->name;
}

static const char * ggml_backend_hexagon_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "Hexagon-cDSP";
}

static void ggml_backend_hexagon_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    struct ggml_backend_hexagon_context * ctx = static_cast<ggml_backend_hexagon_context *>(dev->context);
    if ((nullptr == ctx) || (ctx->device >= GGML_HEXAGON_MAX_DEVICES)) {
        GGMLHEXAGON_LOG_ERROR("pls check params");
        *free = 0;
        *total = 0;
        return;
    }

    GGMLHEXAGON_LOG_WARN("get_memory: enter device=%d domain_id=%d", ctx->device, ctx->domain_id);

    // ggml backend has domain_id == -1 (not a real cDSP PD)
    if (-1 == ctx->domain_id) {
        *total = ggmlhexagon_get_system_total_memory_in_bytes();
        *free = ggmlhexagon_get_system_free_memory_in_bytes();
    } else {
        size_t rpc_ion_memsize = 0;
        size_t rpc_ion_usage   = 0;
        rpc_ion_memsize = ctx->rpc_mempool_capacity;
        rpc_ion_usage   = ctx->rpc_mempool_usage;
        *total = rpc_ion_memsize;
        *free = (rpc_ion_memsize - rpc_ion_usage);
        GGMLHEXAGON_LOG_WARN("get_memory: device %d, rpc memsize %d MiB, usage %d MiB, free %d MiB",
                             ctx->device, rpc_ion_memsize / SIZE_IN_MB,
                             rpc_ion_usage / SIZE_IN_MB, (rpc_ion_memsize - rpc_ion_usage) / SIZE_IN_MB);
    }
}

static enum ggml_backend_dev_type ggml_backend_hexagon_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_hexagon_device_get_props(ggml_backend_dev_t dev,
                                              struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_hexagon_device_get_name(dev);
    props->description = ggml_backend_hexagon_device_get_description(dev);
    props->type        = ggml_backend_hexagon_device_get_type(dev);
    props->device_id   = nullptr;  // no PCI bus id for Hexagon cDSP devices
    ggml_backend_hexagon_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
            /* .async                 = */ false,
            /* .host_buffer           = */ false,
            /* .buffer_from_host_ptr  = */ false,
            /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_hexagon_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGMLHEXAGON_LOG_DEBUG("enter %s\n", __func__);
    int dev_index = 0;

    //case-1: test-backend-ops or other similar scenario: calling ggml_backend_dev_init(dev, reinterpret_cast<const char *>(i)) directly in user's code
    ggmlhexagon_load_cfg();
    if (!ggmlhexagon_check_valid_appcfg()) {
        return nullptr;
    }

    if (nullptr == params) {
        // Derive dev_index from the device's context (g_hexagon_mgr[i]) so each
        // registered Hexagon device initializes its own PD in multi-device mode.
        if (nullptr != dev && nullptr != dev->context) {
            struct ggml_backend_hexagon_context * ctx =
                static_cast<ggml_backend_hexagon_context *>(dev->context);
            dev_index = ctx->device;
        } else {
            dev_index = 0;
        }
    } else {
        GGMLHEXAGON_LOG_VERBOSE("program specified param is not nullptr");
        //user's program calling ggml_backend_hexagon_device_init_backend directly
        dev_index = (int)(intptr_t)params;
        if (dev_index < 0) {
            GGMLHEXAGON_LOG_VERBOSE("it shouldn't happend\n");
            dev_index = 0;
        }
        GGMLHEXAGON_LOG_VERBOSE("program specified dev_index %d\n", dev_index);
    }
    if (dev_index >= GGML_HEXAGON_MAX_DEVICES) {
        GGMLHEXAGON_LOG_ERROR("invalid dev_index %d", dev_index);
        return nullptr;
    }
    GGMLHEXAGON_LOG_DEBUG("dev_index=%d", dev_index);
    ggml_backend_t hexagon_backend = ggml_backend_hexagon_init(dev_index, g_hexagon_appcfg.runtime_libpath);
    GGMLHEXAGON_LOG_DEBUG("leave %s\n", __func__);

    return hexagon_backend;
}

static ggml_backend_buffer_type_t ggml_backend_hexagon_buffer_type(size_t device_index) {
    GGMLHEXAGON_LOG_DEBUG("enter %s, device_index %zu", __func__, device_index);
    if (device_index >= GGML_HEXAGON_MAX_DEVICES || g_hexagon_mgr[device_index] == nullptr) {
        GGMLHEXAGON_LOG_ERROR("ggml_backend_hexagon_buffer_type: device_index %zu out of range or not initialized",
                              device_index);
        return nullptr;
    }
    // buft is owned by the context and initialized in its constructor (no lazy init)
    return &g_hexagon_mgr[device_index]->buffer_type;
}


static ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_hexagon_context * ctx = (ggml_backend_hexagon_context *)dev->context;
    GGMLHEXAGON_LOG_WARN("get_buffer_type: device=%d domain_id=%d buft=%p", ctx->device, ctx->domain_id, (void*)&ctx->buffer_type);
    return &ctx->buffer_type;
}


static bool ggml_backend_hexagon_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (ggml_backend_buft_is_hexagon(buft)) {
        ggml_backend_hexagon_context * dev_ctx  = (ggml_backend_hexagon_context *)dev->context;
        ggml_backend_hexagon_context * buft_ctx = (ggml_backend_hexagon_context *)buft->context;
        return buft_ctx->device == dev_ctx->device;
    }
    // ATTENTION: in ION mempool mode, only support hexagon buffer type (ION memory).
    // Do NOT accept host buffer type, otherwise the scheduler will allocate
    // tensors on the heap, requiring ION mirror + copy-back which has
    // unsolvable cache coherency issues on ARM64 (no DC IVAC in user-space).
    return false;
}

static struct ggml_backend_device_i ggml_backend_hexagon_device_interface = {
        /* .get_name             = */ ggml_backend_hexagon_device_get_name,
        /* .get_description      = */ ggml_backend_hexagon_device_get_description,
        /* .get_memory           = */ ggml_backend_hexagon_device_get_memory,
        /* .get_type             = */ ggml_backend_hexagon_device_get_type,
        /* .get_props            = */ ggml_backend_hexagon_device_get_props,
        /* .init_backend         = */ ggml_backend_hexagon_device_init_backend,
        /* .get_buffer_type      = */ ggml_backend_hexagon_device_get_buffer_type,
        /* .get_host_buffer_type = */ nullptr,
        /* .buffer_from_host_ptr = */ nullptr,
        /* .supports_op          = */ nullptr,
        /* .supports_buft        = */ ggml_backend_hexagon_device_supports_buft,
        /* .offload_op           = */ nullptr,
        /* .event_new            = */ nullptr,
        /* .event_free           = */ nullptr,
        /* .event_synchronize    = */ nullptr,
};

static ggml_backend_i ggml_backend_hexagon_interface = {
        /* .get_name                = */ ggml_backend_hexagon_name,
        /* .free                    = */ ggml_backend_hexagon_free,
        /* .set_tensor_async        = */ nullptr,
        /* .get_tensor_async        = */ nullptr,
        /* .set_tensor_2d_async     = */ nullptr,
        /* .get_tensor_2d_async     = */ nullptr,
        /* .cpy_tensor_async        = */ nullptr,
        /* .synchronize             = */ nullptr,
        /* .graph_plan_create       = */ nullptr,
        /* .graph_plan_free         = */ nullptr,
        /* .graph_plan_update       = */ nullptr,
        /* .graph_plan_compute      = */ nullptr,
        /* .graph_compute           = */ nullptr,
        /* .event_record            = */ nullptr,
        /* .event_wait              = */ nullptr,
        /* .graph_optimize          = */ nullptr,
};

//FIXME: this guid is not make sense
static ggml_guid_t ggml_backend_hexagon_guid() {
    static ggml_guid guid = {
            0x1a, 0x2b, 0x3c, 0x4d, 0x5e, 0x6f, 0x70, 0x81,
            0x92, 0xa3, 0xb4, 0xc5, 0xd6, 0xe7, 0xf8, 0x09
    };
    return &guid;
}

bool ggml_backend_is_hexagon(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_hexagon_guid());
}

static void ggml_backend_hexagon_set_n_threads(ggml_backend_t backend, int n_threads) {
    GGML_ASSERT(ggml_backend_is_hexagon(backend));

    struct ggml_backend_hexagon_context * ctx = (struct ggml_backend_hexagon_context *)backend->context;
    ctx->n_threads = n_threads;
}

int ggml_backend_hexagon_get_device_count() {
    return g_hexagon_appcfg.ndev;
}

struct ggml_backend_hexagon_reg_context {
    std::vector<ggml_backend_dev_t> devices;
    ~ggml_backend_hexagon_reg_context() {
        for (auto * dev : devices) {
            auto * hctx = static_cast<ggml_backend_hexagon_context *>(dev->context);
            delete hctx;
            delete dev;
        }
    }
};

// Owning pointer to the reg context. The framework's ~ggml_backend_registry()
// does not delete reg->context (see FIXME in ggml-backend-reg.cpp), so we rely
// on an atexit handler to release DSP sessions. atexit runs before static
// dtors, so function-local std::mutex objects (e.g. the log mutex) are still
// alive when ~ggml_backend_hexagon_context calls ggmlhexagon_deinit_cdsp.
static ggml_backend_hexagon_reg_context * g_reg_ctx = nullptr;

static void ggml_backend_hexagon_atexit_cleanup() {
    if (g_reg_ctx) {
        delete g_reg_ctx;
        g_reg_ctx = nullptr;
    }
}

static const char * ggml_backend_hexagon_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "Hexagon-cDSP";
}

static size_t ggml_backend_hexagon_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_hexagon_reg_context * ctx = (ggml_backend_hexagon_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_hexagon_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_hexagon_reg_context * ctx = (ggml_backend_hexagon_reg_context *)reg->context;
    GGMLHEXAGON_LOG_WARN("reg_get_device: index=%zu count=%zu", index, ctx->devices.size());
    if (index >= ctx->devices.size()) {
        GGMLHEXAGON_LOG_ERROR("invalid device index %d (count=%zu)", index, ctx->devices.size());
        return nullptr;
    }
    return ctx->devices[index];
}

static void * ggml_backend_hexagon_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);

    if (nullptr == name)
        return nullptr;

    const char * slot_name =  "ggml_backend_set_n_threads";
    if (0 == memcmp(name, slot_name, strlen(slot_name))) {
        return (void *)ggml_backend_hexagon_set_n_threads;
    }

    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_hexagon_reg_interface = {
        /* .get_name          = */ ggml_backend_hexagon_reg_get_name,
        /* .get_device_count  = */ ggml_backend_hexagon_reg_get_device_count,
        /* .get_device        = */ ggml_backend_hexagon_reg_get_device,
        /* .get_proc_address  = */ ggml_backend_hexagon_reg_get_proc_address,
};

ggml_backend_reg_t ggml_backend_hexagon_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__);

    ggmlhexagon_load_cfg();
    if (!ggmlhexagon_check_valid_appcfg()) {
        return nullptr;
    }

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            int ret = htpdrv_init();
            if (AEE_SUCCESS != ret) {
                GGMLHEXAGON_LOG_ERROR("htpdrv_init failed with error %d", ret);
                return nullptr;
            }

            ggml_backend_hexagon_reg_context * ctx = new ggml_backend_hexagon_reg_context;

            if (2 == g_hexagon_appcfg.offload_cgraph_type) {
                ggml_backend_hexagon_device_interface.supports_op = ggmlhexagon_can_handle_op_through_cdsp_ion;
            } else {
                ggml_backend_hexagon_device_interface.supports_op = ggmlhexagon_can_handle_op_through_cdsp;
            }

            int ndev = g_hexagon_appcfg.ndev;
            GGMLHEXAGON_LOG_INFO("registering %d Hexagon device(s), ndev=%d", ndev, g_hexagon_appcfg.ndev);

            for (int i = 0; i < ndev; i++) {
                if (i >= GGML_HEXAGON_MAX_DEVICES) {
                    GGMLHEXAGON_LOG_WARN("ndev=%d exceeds GGML_HEXAGON_MAX_DEVICES=%d, only %d devices registered",
                                         ndev, GGML_HEXAGON_MAX_DEVICES, i);
                    break;
                }

                GGMLHEXAGON_LOG_DEBUG("create backend device for device %d", i);
                // Create device struct first so we can pass it to the context constructor
                // (matches qcom's ggml_hexagon_session pattern: session owns DSP handle,
                //  buft is a member initialized with the owning device pointer).
                ggml_backend_dev_t dev = new ggml_backend_device{
                        /* .iface       = */ ggml_backend_hexagon_device_interface,
                        /* .reg         = */ &reg,
                        /* .context     = */ nullptr  // set below after context creation
                };

                // Constructor performs full DSP init (ggmlhexagon_init_dsp) and
                // initializes buffer_type with context = this, device = dev.
                auto * hctx = new ggml_backend_hexagon_context(i, dev);
                dev->context = hctx;
                g_hexagon_mgr[i] = hctx;

                if (0 == hctx->ggmlop_handle) {
                    GGMLHEXAGON_LOG_INFO("init hexagon dsp failure for device %d", i);
                    // constructor already logged the error; keep the device registered
                    // so get_memory/get_buffer_type do not crash, but return nullptr
                }

                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                    /* .api_version = */ GGML_BACKEND_API_VERSION,
                    /* .iface       = */ ggml_backend_hexagon_reg_interface,
                    /* .context     = */ ctx
            };

            g_reg_ctx = ctx;
            std::atexit(ggml_backend_hexagon_atexit_cleanup);
        }

        initialized = true;
    }
    GGMLHEXAGON_LOG_DEBUG("leave ggml_backend_hexagon_reg");

    return &reg;
}

const char * ggml_backend_hexagon_get_devname(size_t dev_num) {
    // CDSP devices: Hexagon-cDSP0, Hexagon-cDSP1, ...
    if (dev_num < GGML_HEXAGON_MAX_DEVICES && g_hexagon_mgr[dev_num] != nullptr) {
        return g_hexagon_mgr[dev_num]->name;
    }
    return "unknown";
}

ggml_backend_t ggml_backend_hexagon_init(size_t device, const char * runtime_libpath) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__);
    if (nullptr == runtime_libpath)
        return nullptr;

    //case-3: calling ggml_backend_hexagon_init() directly in user's code
    ggmlhexagon_load_cfg();
    if (!ggmlhexagon_check_valid_appcfg()) {
        return nullptr;
    }

    GGMLHEXAGON_LOG_DEBUG("device %d", device);
    GGMLHEXAGON_LOG_DEBUG("runtime libpath %s", runtime_libpath);
    if (device >= GGML_HEXAGON_MAX_DEVICES) {
        GGMLHEXAGON_LOG_ERROR("invalid device %d", device);
        return nullptr;
    }

    if (0 != memcmp(runtime_libpath, g_hexagon_appcfg.runtime_libpath, strlen(g_hexagon_appcfg.runtime_libpath))) {
        //re-setting runtime libpath
        ggmlhexagon_set_runtime_path(device, runtime_libpath);
    }

    if (nullptr != g_hexagon_mgr[device] && nullptr != g_hexagon_mgr[device]->backend) {
        GGMLHEXAGON_LOG_DEBUG("backend %d(%s) already loaded", device,
                         ggml_backend_hexagon_get_devname(device));
        GGMLHEXAGON_LOG_DEBUG("leave %s", __func__);
        return g_hexagon_mgr[device]->backend;
    }

    if (2 == g_hexagon_appcfg.offload_cgraph_type) {
        GGMLHEXAGON_LOG_WARN("using ggmlhexagon_backend_graph_compute_ion (ION-based op-batch)");
        ggml_backend_hexagon_interface.graph_compute = ggmlhexagon_backend_graph_compute_ion;
    } else {
        GGMLHEXAGON_LOG_WARN("using ggmlhexagon_backend_graph_compute_general (per-op)");
        ggml_backend_hexagon_interface.graph_compute = ggmlhexagon_backend_graph_compute_general;
    }
    // Context (and DSP session) was already created by ggml_backend_hexagon_reg().
    // Just attach the backend handle to the existing context.
    ggml_backend_hexagon_context * ctx = g_hexagon_mgr[device];
    if (ctx == nullptr) {
        GGMLHEXAGON_LOG_ERROR("device %zu context not initialized", device);
        return nullptr;
    }
    ggml_backend_t hexagon_backend = new ggml_backend{
            /* .guid      = */ ggml_backend_hexagon_guid(),
            /* .iface     = */ ggml_backend_hexagon_interface,
            /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_hexagon_reg(), device),
            /* .context   = */ ctx
    };

    ctx->backend = hexagon_backend;
    // DSP init already done in ggml_backend_hexagon_context constructor
    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__);

    return hexagon_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)
