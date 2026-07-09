/*
 * 2024-2026 The ggml authors
 *
 * this single-source-file implementation of JZ's ggml-hexagon backend has 8 sections:
 * section-1  forward declarations, global vars, macros
 * section-2  data structures
 * section-3  troubleshooting and profiler
 * section-4  configuration class and helper functions
 * section-5  general helper functions
 * section-6  CDSP helper functions
 * section-7  Qualcomm compatibility layer
 * section-8  backend implementation
 *
 * this is a practical implementation:
 *    in mulmat_algotype=29 path, Qualcomm's operators/kernels in htp directory will be used
 *    in mulmat_algotype=32 path, ggmlop_dsp_* operators/kernels in kernels directory will be used,mulmat's performance is slower than Qualcomm's mulmat and can expand other ops easily & accordingly.
 *
 * Jeff Zhou - zhouwg2000@gmail.com
 * GitHub:   - https://github.com/zhouwg/ggml-hexagon
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
#endif

#if defined(__ANDROID__)
#include "android/log.h"
#endif

#include "rpcmem.h"
#include "remote.h"
#include "AEEStdErr.h"
#include "htp-drv.h"
#include "HAP_power.h"
#include "HAP_farf.h"

#include "ggml-hexagon.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "kernels/skel.h"
#include "kernels/ggml-ops.h"
#include "htp/htp-ops.h"
#include "htp/matmul-ops.h"
#include "htp/flash-attn-ops.h"

// Adapters for the old htp_mm_hvx vtcm_sizes API that was replaced by
// htp_mm_hvx_vtcm_layout_build in upstream htp/matmul-ops.h.
static inline size_t htp_mm_hvx_get_vtcm_sizes(
    int kernel_type, int wtype, uint32_t ne10, uint32_t src1_nrows,
    uint32_t n_threads,
    size_t dst_row_size, size_t src0_row_size, size_t src1_row_size,
    uint32_t n_prefetch,
    size_t * vtcm_src0_size, size_t * vtcm_src1_size, size_t * vtcm_dst_size
) {
    struct htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kernel_type, wtype, ne10, src1_nrows, n_threads,
                                 dst_row_size, src0_row_size, src1_row_size, n_prefetch,
                                 false, false, false);
    *vtcm_src0_size = L.src0_bytes;
    *vtcm_src1_size = L.src1_bytes;
    *vtcm_dst_size  = L.dst_bytes;
    return L.total_bytes;
}

static inline size_t htp_mm_hvx_id_get_vtcm_sizes(
    int wtype, uint32_t ne10, uint32_t src1_nrows, uint32_t n_threads,
    size_t src0_row_size, uint32_t n_prefetch,
    size_t * vtcm_src0_size, size_t * vtcm_src1_size, size_t * vtcm_dst_size
) {
    struct htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, 0, wtype, ne10, src1_nrows, n_threads,
                                 0, src0_row_size, 0, n_prefetch,
                                 true, false, false);
    *vtcm_src0_size = L.src0_bytes;
    *vtcm_src1_size = L.src1_bytes;
    *vtcm_dst_size  = L.dst_bytes;
    return L.total_bytes;
}

// =================================================================================================
//  section-1: forward declarations, global vars, macros
// =================================================================================================
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

#define GGMLHEXAGON_LOG_ALWAYS(...)                     ggmlhexagon_log_always_internal(__FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLHEXAGON_LOG_ERROR(...)                      ggmlhexagon_log_always_internal(__FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLHEXAGON_LOG_WARN(...)                       ggmlhexagon_log_internal(GGML_LOG_LEVEL_WARN , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLHEXAGON_LOG_INFO(...)                       ggmlhexagon_log_internal(GGML_LOG_LEVEL_INFO , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLHEXAGON_LOG_VERBOSE(...)                    ggmlhexagon_log_internal(GGML_LOG_LEVEL_CONT , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

#if GGMLHEXAGON_DEBUG
#define GGMLHEXAGON_LOG_DEBUG(...)                      ggmlhexagon_log_internal(GGML_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#else
#define GGMLHEXAGON_LOG_DEBUG(...)
#endif

#define SIZE_IN_MB                                      (1 << 20)
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

// Forward declarations
struct ggml_backend_hexagon_context;

static bool                  ggmlhexagon_use_ion_mempool(void);
static bool                  ggmlhexagon_is_metadata_op(enum ggml_op op);
static const char *          ggmlhexagon_get_mulmat_algotype_desc(int algotype);
static int                   ggmlhexagon_probe_dspinfo(ggml_backend_hexagon_context * ctx);
static int                   hexagon_probe_invoke_timed(ggml_backend_hexagon_context * ctx);
static bool                  ggml_backend_hexagon_buffer_is_host(ggml_backend_buffer_type_t buft);
static void                  ggmlhexagon_set_runtime_path(size_t device, const std::string & path);
static const char *          ggml_backend_hexagon_buffer_type_name(ggml_backend_buffer_type_t buft);
static size_t                ggml_backend_hexagon_buffer_type_get_max_size(ggml_backend_buffer_type_t buft);
static size_t                ggml_backend_hexagon_buffer_type_get_alignment(ggml_backend_buffer_type_t buft);
static bool                  ggml_backend_hexagon_repack_buffer_is_host(ggml_backend_buffer_type_t buft);
static bool                  ggml_backend_buffer_is_hexagon_repack(const struct ggml_backend_buffer * b);
static bool                  ggmlhexagon_tensor_buffer_is_owned_by(ggml_backend_dev_t dev, const struct ggml_tensor * t);
static bool                  ggmlhexagon_op_buffers_belong_to_dev(ggml_backend_dev_t dev, const struct ggml_tensor * op);
static size_t                ggml_backend_hexagon_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);

static ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);
static bool                  ggml_hexagon_compute_fa_params(const ggml_backend_hexagon_context * ctx,
                                                            const ggml_tensor * node,
                                                            struct htp_fa_kernel_params * kparams);

// =================================================================================================
//  section-2: data structures
// =================================================================================================
struct ggmlhexagon_task {
    int32 op_type;
    ggml_tensor * src0;
    ggml_tensor * src1;
    ggml_tensor * dst;
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

    //Hexagon resource management for the general approach through Hexagon CDSP
    size_t rpc_mempool_capacity;
    size_t rpc_mempool_len;
    size_t rpc_mempool_usage;
    size_t rpc_mempool_cache_offset;    // ION offset where FP16 cache region starts
    size_t rpc_mempool_cache_budget;    // total ION cache size for FP16 weights (= rpc_mempool_len - cache_offset)
    size_t rpc_mempool_cache_usage_est; // estimated cumulative FP16 cache usage, reset per graph
    std::unordered_set<const void*> rpc_mempool_cache_tracked; // weight data pointers already counted
    bool   weights_dirty;               // set by set_tensor/memset_tensor, cleared by Phase 6.5
    void * rpc_mempool;
    int rpc_mempool_handle;
    void * rpc_mempool_dsp_base;        // DSP-side VA from fastrpc_mmap() (NOT from FastRPC pointer translation)
    std::vector<ion_pool_region> ion_regions;  // region tracking for ION pool free-space management
    remote_handle64 ggmlop_handle;
    int domain_id;
    int session_id;

    // FastRPC call statistics
    uint64_t rpc_batch_call_count;   // total ggmlop_dsp_execute_batch calls
    int64_t  cumulative_p7_us;       // cumulative FastRPC time (p7 phase)
    int64_t  cumulative_graph_us;    // cumulative graph inference duration
    int64_t  last_graph_end_us;      // wall clock of last graph end (to measure gap)

    // Per-graph node statistics
    uint32_t max_nodes_per_graph;    // max node count in a single graph
    uint32_t min_nodes_per_graph;    // min node count in a single graph
    uint32_t total_nodes_processed;  // cumulative node count across all graphs

    // Per-call execution time range
    int64_t  min_graph_us;          // shortest single graph execution
    int64_t  max_graph_us;          // longest single graph execution
    uint32_t max_graph_n_nodes;     // cgraph node count when max_graph_us recorded
    uint32_t max_graph_n_ops;       // DSP op count (post-fusion) when max_graph_us recorded
    int64_t  min_p7_us;             // shortest single FastRPC call
    int64_t  max_p7_us;             // longest single FastRPC call

    // AP-side per-phase cumulative time
    int64_t  cum_p1_us;       // Phase 1: collect unique tensor objects
    int64_t  cum_p2_us;       // Phase 2: build op descriptors
    int64_t  cum_p25_us;      // Phase 2.5: op fusion
    int64_t  cum_p3_us;       // Phase 3: compute layout sizes
    int64_t  cum_p4_us;       // Phase 4: tensor mirroring
    int64_t  cum_p45_us;      // Phase 4.5: weight repack (dominant for algotype=29)
    int64_t  cum_p5_us;       // Phase 5: allocate batch descriptor in ION
    int64_t  cum_p6_us;       // Phase 6: descriptor construction
    int64_t  cum_p65_us;      // Phase 6.5: cache flush
    int64_t  cum_p75_us;      // Phase 7.5: cache inval
    int64_t  cum_p8_us;       // Phase 8: ION->heap copy-back

    // Per-call fine-grained profiling (ring buffer, capacity 1024).
    // Captures per-call durations so dump_perf_stats can compute
    // min/p50/p95/max and reveal distribution that cumulative averages hide.
    static constexpr int PERF_HIST_CAP = 1024;
    int      perf_hist_count;     // number of valid samples (<= PERF_HIST_CAP)
    int      perf_hist_idx;       // next ring slot (mod PERF_HIST_CAP)
    int64_t  p1_hist[PERF_HIST_CAP];   // Phase 1: collect unique tensor objects
    int64_t  p2_hist[PERF_HIST_CAP];   // Phase 2: build op descriptors
    int64_t  p25_hist[PERF_HIST_CAP];  // Phase 2.5: op fusion
    int64_t  p3_hist[PERF_HIST_CAP];   // Phase 3: compute layout sizes
    int64_t  p4_hist[PERF_HIST_CAP];   // Phase 4: tensor mirroring
    int64_t  p45_hist[PERF_HIST_CAP];  // Phase 4.5: weight repack (dominant for algotype=29)
    int64_t  p5_hist[PERF_HIST_CAP];   // Phase 5: allocate batch descriptor in ION
    int64_t  p6_hist[PERF_HIST_CAP];   // Phase 6: descriptor construction
    int64_t  p65_hist[PERF_HIST_CAP];  // Phase 6.5: cache flush
    int64_t  p7_hist[PERF_HIST_CAP];   // Phase 7: FastRPC + DSP exec + cache inval (cumulative; see 3-way split below)
    int64_t  p75_hist[PERF_HIST_CAP];  // Phase 7.5: cache inval
    int64_t  p8_hist[PERF_HIST_CAP];   // Phase 8: ION->heap copy-back
    int64_t  graph_us_hist[PERF_HIST_CAP];   // total wall-clock per graph_compute_batch call
    int32_t  n_nodes_hist[PERF_HIST_CAP];    // cgraph->n_nodes at entry
    int32_t  n_ops_hist[PERF_HIST_CAP];      // offloaded DSP ops at FastRPC dispatch
    int64_t  gap_from_prev_hist[PERF_HIST_CAP]; // us between consecutive graph_compute calls (sampler)

    // p7 3-way breakdown: split the FastRPC + DSP exec + cache inval window
    // so we can tell AP-side cache-coherency cost apart from DSP-side work.
    int64_t  cum_p7_rpc_setup_us;  // AP setup before ggmlop_dsp_execute_batch (ioctl / marshalling)
    int64_t  cum_p7_dsp_exec_us;   // pure DSP execution time inside the sync call
    int64_t  cum_p7_civac_us;     // AP cache invalidate after DSP reply
    int64_t  p7_rpc_setup_hist[PERF_HIST_CAP]; // AP setup before ggmlop_dsp_execute_batch (ioctl / marshalling)
    int64_t  p7_dsp_exec_hist[PERF_HIST_CAP];  // pure DSP execution time inside the sync call
    int64_t  p7_civac_hist[PERF_HIST_CAP];     // AP cache invalidate after DSP reply

    // FastRPC transport overhead calibration (measured via probe invokes at init).
    // probe mode (batch_size=0) does minimal DSP work (1 byte read, 32 bytes write,
    // 2 cache line flushes, 2x LOG_INFO), so measured time is an upper bound of
    // pure FastRPC transport overhead (invoke round-trip: AP -> DSP -> AP).
    int64_t  rpc_overhead_min_us;  // shortest probe invoke
    int64_t  rpc_overhead_max_us;  // longest probe invoke
    int64_t  rpc_overhead_sum_us;  // sum of all probe invokes (for avg)
    uint32_t rpc_overhead_count;   // number of probe invokes measured

    // Buffer type owned by this context (each device has its own buft)
    struct ggml_backend_buffer_type buffer_type;
    // Repack buffer type(is_host=false), same ION pool as buffer_type
    struct ggml_backend_buffer_type repack_buffer_type;
    char buft_name[GGML_MAX_NAME];        // "hexagon-ion-buffer-<name>", unique per device
    char repack_buft_name[GGML_MAX_NAME]; // "hexagon-ion-buffer-<name>-REPACK"

    // Per-device hardware caps (probed at init, used by supports_op)
    bool has_vtcm;  // domain has VTCM pages available
    bool has_hvx;   // domain has HVX support
    bool has_hmx;   // domain has HMX support

    // Cached htp_mm_kernel_params per (weight_data, ne11). For TG, the
    // precompute math produces identical results for every token, so we
    // cache the params struct to skip the multi-hundred-microsecond
    // thread/chunk search on subsequent calls.
    std::unordered_map<uintptr_t, struct htp_mm_kernel_params> mm_params_cache;

    // cgraph cache: Phase 1 (tensor dedup) + Phase 2 (hex_ops build) +
    // Phase 2.5 (op fusion) result keyed by content-based cgraph hash.
    // The scheduler's split->graph pointer changes every call, but the
    // underlying node ops/shapes/data ptrs are stable for graph-reuse
    // A FNV-1a hash over {op, ne[4], nb[4], src[0..2] ptr, data ptr}
    // per node gives a 64-bit key that is stable across pointer churn
    // and effectively collision-free (2^-64 false positive).
    // On hit, skip ~38us of Phase 1+2 work. With 17 subgraphs/token and
    // 100% hit rate after warmup, this saves ~646us/token = 1.1% of TG.
    struct cgraph_cache_entry {
        uint64_t content_hash = 0;
        int n_nodes = 0;
        int n_tensors = 0;
        int n_ops = 0;
        std::vector<ggml_tensor *> tensor_src;
        std::vector<ggml_tensor *> supported_nodes;
        std::vector<ggml_tensor *> unsupported_nodes;
        std::vector<hex_op_desc>   hex_ops;
        std::vector<uint32_t>      weight_indices;
    };
    std::unordered_map<uint64_t, cgraph_cache_entry> cgraph_cache;
    uint64_t cgraph_cache_hits   = 0;
    uint64_t cgraph_cache_misses = 0;

    ggml_backend_hexagon_context(int dev_id, ggml_backend_dev_t dev);
    ~ggml_backend_hexagon_context();
};

struct hexagon_op_caps {
    bool supported;
    ggml_op op;
    const size_t input_param_count;
    const char * hexagon_op_name;
};

struct hexagon_appcfg_t {
    int dump_debug_info;        // enable/disable dump debug info for troubleshooting issues on AP side
    int thread_counts;          // thread_counts on CDSP side
    int mulmat_algotype;        // algorithm type of mulmat on CDSP side
    int mulmat_min_n;           // minimum N (batch size) to offload quantized MUL_MAT to DSP
    int offload_cgraph_type;    // offload type on AP side
    int dump_diag_info;         // enable/disable dump diag info for troubleshooting issues on CDSP side
    int ggml_dsp_use_hvx;       // enable/disable HVX-optimized quantize_row & vec_dot on CDSP side
    int ndev;                   // number of Hexagon devices (PDs), from GGML_HEXAGON_NDEV env
    int ion_sync_mode;          // 0=both(DC CVAC+ion_sync, default), 1=ion_sync only, 2=DC CVAC only
    int enable_opfusion;        // 1=enable QKV/FFN op fusion (default), 0=disable (for debugging)
    int fa_select;              // flash attention: 2=HMX->HVX->CPU, 1=HVX->CPU, 0=CPU (default 2)
    int gemv_offload;           // 1=offload GEMV (N=1) to DSP (default, current behavior), 0=keep GEMV on CPU (for debugging)

    const char * cfgfilename;
    const char * runtime_libpath;
    char version[GGMLHEXAGON_TMPBUF_LEN];
    std::string enabled_ops;    // comma-separated list of ops to offload (empty = all supported ops)
    std::string enabled_types;  // comma-separated list of weight types to offload for MUL_MAT (empty = all supported types)
};

static struct hexagon_appcfg_t g_hexagon_appcfg = {
        .dump_debug_info        = 0,
        .thread_counts          = 6,
        .mulmat_algotype        = 0,
        .mulmat_min_n           = 32,
        .offload_cgraph_type    = 0,
        .dump_diag_info         = 0,
        .ggml_dsp_use_hvx       = 1,
        .ndev                   = 1,
        .ion_sync_mode          = 0,
        .enable_opfusion        = 1,
        .fa_select              = 2,
        .gemv_offload           = 1,
        .cfgfilename            = "ggml-hexagon.cfg",
#if defined(__ANDROID__)
        .runtime_libpath        = "/data/local/tmp/",
#endif
        .version                = {"0.99.3.3"},
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

// Track tensors repacked in set_tensor to skip Phase 4.5
static std::unordered_set<const void *> g_set_tensor_repacked;

// Used by both per-op mode (offload_cgraph_type=0) and ION-batch mode (offload_cgraph_type=2).
// Only marks op type as supported - shape/size/type restrictions are enforced in supports_op.
static constexpr const hexagon_op_caps ggmlhexagon_k_op_caps[] = {
    {true,  GGML_OP_NONE,     0, nullptr},
    {false, GGML_OP_DUP,      0, nullptr},
    {true,  GGML_OP_ADD,      2, "ggmlop_dsp_add"},
    {false, GGML_OP_ADD_ID,   0, nullptr},
    {false, GGML_OP_ADD1,     0, nullptr},
    {false, GGML_OP_ACC,      0, nullptr},
    {true,  GGML_OP_SUB,      2, "ggmlop_dsp_sub"},
    {true,  GGML_OP_MUL,      2, "ggmlop_dsp_mul"},
    {true,  GGML_OP_DIV,      2, "ggmlop_dsp_div"},
    {true,  GGML_OP_SQR,      1, "ggmlop_dsp_sqr"},
    {true,  GGML_OP_SQRT,     1, "ggmlop_dsp_sqrt"},
    {false, GGML_OP_LOG,      0, nullptr},
    {false, GGML_OP_SIN,      0, nullptr},
    {false, GGML_OP_COS,      0, nullptr},
    {false, GGML_OP_SUM,      0, nullptr},
    {true,  GGML_OP_SUM_ROWS, 0, nullptr},
    {true,  GGML_OP_CUMSUM,   1, "ggmlop_dsp_cumsum"},
    {false, GGML_OP_MEAN,     0, nullptr},
    {false, GGML_OP_ARGMAX,   0, nullptr},
    {false, GGML_OP_COUNT_EQUAL, 0, nullptr},
    {true,  GGML_OP_REPEAT,   2, "ggmlop_dsp_repeat"},
    {false, GGML_OP_REPEAT_BACK, 0, nullptr},
    {true,  GGML_OP_CONCAT,   2, "ggmlop_dsp_concat"},
    {false, GGML_OP_SILU_BACK, 0, nullptr},
    {true,  GGML_OP_NORM,     1, "ggmlop_dsp_norm"},
    {true,  GGML_OP_RMS_NORM, 1, "ggmlop_dsp_rmsnorm"},
    {false, GGML_OP_RMS_NORM_BACK, 0, nullptr},
    {false, GGML_OP_GROUP_NORM, 0, nullptr},
    {true,  GGML_OP_L2_NORM,  1, "ggmlop_dsp_l2_norm"},
    {true,  GGML_OP_MUL_MAT,  2, "ggmlop_dsp_mulmat"},
    {false, GGML_OP_MUL_MAT_ID, 0, nullptr},
    {false, GGML_OP_OUT_PROD, 0, nullptr},
    {true,  GGML_OP_SCALE,    1, "ggmlop_dsp_scale"},
    {false, GGML_OP_SET,      0, nullptr},
    {true,  GGML_OP_CPY,      2, "ggmlop_dsp_cpy"},
    {true,  GGML_OP_CONT,     0, nullptr},
    {false, GGML_OP_RESHAPE,  0, nullptr},
    {false, GGML_OP_VIEW,     0, nullptr},
    {false, GGML_OP_PERMUTE,  0, nullptr},
    {false, GGML_OP_TRANSPOSE, 0, nullptr},
    {true,  GGML_OP_GET_ROWS, 0, nullptr},
    {false, GGML_OP_GET_ROWS_BACK, 0, nullptr},
    {true,  GGML_OP_SET_ROWS, 0, nullptr},
    {true,  GGML_OP_DIAG,     1, "ggmlop_dsp_diag"},
    {true,  GGML_OP_DIAG_MASK_INF, 2, "ggmlop_dsp_diag_mask_inf"},
    {false, GGML_OP_DIAG_MASK_ZERO, 0, nullptr},
    {true,  GGML_OP_SOFT_MAX, 2, "ggmlop_dsp_softmax"},
    {false, GGML_OP_SOFT_MAX_BACK, 0, nullptr},
    {true,  GGML_OP_ROPE,     3, "ggmlop_dsp_rope"},
    {false, GGML_OP_ROPE_BACK, 0, nullptr},
    {false, GGML_OP_CLAMP,    0, nullptr},
    {false, GGML_OP_CONV_TRANSPOSE_1D, 0, nullptr},
    {false, GGML_OP_IM2COL,   0, nullptr},
    {false, GGML_OP_IM2COL_BACK, 0, nullptr},
    {false, GGML_OP_IM2COL_3D, 0, nullptr},
    {false, GGML_OP_COL2IM_1D, 0, nullptr},
    {false, GGML_OP_CONV_2D,  0, nullptr},
    {false, GGML_OP_CONV_3D,  0, nullptr},
    {false, GGML_OP_CONV_2D_DW, 0, nullptr},
    {false, GGML_OP_CONV_TRANSPOSE_2D, 0, nullptr},
    {false, GGML_OP_POOL_1D,  0, nullptr},
    {false, GGML_OP_POOL_2D,  0, nullptr},
    {false, GGML_OP_POOL_2D_BACK, 0, nullptr},
    {false, GGML_OP_UPSCALE,  0, nullptr},
    {true,  GGML_OP_PAD,      1, "ggmlop_dsp_pad"},
    {false, GGML_OP_PAD_REFLECT_1D, 0, nullptr},
    {false, GGML_OP_ROLL,     0, nullptr},
    {false, GGML_OP_ARANGE,   0, nullptr},
    {false, GGML_OP_TIMESTEP_EMBEDDING, 0, nullptr},
    {true,  GGML_OP_ARGSORT,  1, "ggmlop_dsp_argsort"},
    {false, GGML_OP_TOP_K,    0, nullptr},
    {false, GGML_OP_LEAKY_RELU, 0, nullptr},
    {true,  GGML_OP_TRI,      1, "ggmlop_dsp_tri"},
    {true,  GGML_OP_FILL,     1, "ggmlop_dsp_fill"},
    {true,  GGML_OP_FLASH_ATTN_EXT, 4, "ggmlop_dsp_flash_attn"},
    {false, GGML_OP_FLASH_ATTN_BACK, 0, nullptr},
    {false, GGML_OP_SSM_CONV, 0, nullptr},
    {false, GGML_OP_SSM_SCAN, 0, nullptr},
    {false, GGML_OP_WIN_PART, 0, nullptr},
    {false, GGML_OP_WIN_UNPART, 0, nullptr},
    {false, GGML_OP_GET_REL_POS, 0, nullptr},
    {false, GGML_OP_ADD_REL_POS, 0, nullptr},
    {false, GGML_OP_RWKV_WKV6, 0, nullptr},
    {false, GGML_OP_GATED_LINEAR_ATTN, 0, nullptr},
    {false, GGML_OP_RWKV_WKV7, 0, nullptr},
    {false, GGML_OP_SOLVE_TRI, 0, nullptr},
    {false, GGML_OP_GATED_DELTA_NET, 0, nullptr},
    {true,  GGML_OP_UNARY,    1, "ggmlop_dsp_silu"},
    {false, GGML_OP_MAP_CUSTOM1, 0, nullptr},
    {false, GGML_OP_MAP_CUSTOM2, 0, nullptr},
    {false, GGML_OP_MAP_CUSTOM3, 0, nullptr},
    {false, GGML_OP_CUSTOM,   0, nullptr},
    {false, GGML_OP_CROSS_ENTROPY_LOSS, 0, nullptr},
    {false, GGML_OP_CROSS_ENTROPY_LOSS_BACK, 0, nullptr},
    {false, GGML_OP_OPT_STEP_ADAMW, 0, nullptr},
    {false, GGML_OP_OPT_STEP_SGD, 0, nullptr},
    {true,  GGML_OP_GLU,      1, "ggmlop_dsp_glu"},
};

// =================================================================================================
//  section-3: troubleshooting and profiler
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
            if (GGML_LOG_LEVEL_INFO == level || GGML_LOG_LEVEL_CONT == level) {
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

// Always-emit log channel. Used by GGMLHEXAGON_LOG_ERROR and
// GGMLHEXAGON_LOG_ALWAYS (both bypass the log-level filter).
// Unlike ggmlhexagon_log_internal:
//   - on Android writes ONLY to logcat (via __android_log_print), never to
//     stdout, so per-batch error/always messages won't spam the llama-cli console
//   - on non-Android hosts falls back to stdout
static void ggmlhexagon_log_always_internal(const char * file, const char * func, int line, const char * format, ...) {
    static std::mutex s_log_mutex;
    static char s_log_buf[GGMLHEXAGON_LOGBUF_LEN];

    GGML_UNUSED(file);

    {
        std::lock_guard<std::mutex> lock(s_log_mutex);
        va_list args;
        va_start(args, format);
        int len_prefix = snprintf(s_log_buf, GGMLHEXAGON_LOGBUF_LEN, "[%s, %d]: ", func, line);
        int len = vsnprintf(s_log_buf + len_prefix, GGMLHEXAGON_LOGBUF_LEN - len_prefix, format, args);
        if (len < (GGMLHEXAGON_LOGBUF_LEN - len_prefix)) {
#if (defined __ANDROID__) || (defined ANDROID)
            __android_log_print(ANDROID_LOG_INFO, PROJECT_NAME, "%s\n", s_log_buf);
#else
            printf("%s\n", s_log_buf);
#endif
        }
        va_end(args);
    }
}

// Invoke probe (batch_size=0) and record FastRPC transport overhead timing.
// probe mode does minimal DSP work, so measured time is an upper bound of
// pure FastRPC transport overhead. Used at init only, no per-graph overhead.
static int hexagon_probe_invoke_timed(ggml_backend_hexagon_context * ctx) {
    int64_t t0 = ggml_time_us();
    int err = ggmlop_dsp_execute_batch(ctx->ggmlop_handle, 0, 0);
    int64_t dt = ggml_time_us() - t0;
    ctx->rpc_overhead_sum_us += dt;
    ctx->rpc_overhead_count++;
    if (ctx->rpc_overhead_min_us == 0 || dt < ctx->rpc_overhead_min_us) ctx->rpc_overhead_min_us = dt;
    if (dt > ctx->rpc_overhead_max_us)                                  ctx->rpc_overhead_max_us = dt;
    return err;
}

static void ggmlhexagon_print_running_timestamp(ggml_backend_hexagon_context * ctx) {
    char timestamp[GGMLHEXAGON_TMPBUF_LEN];
    memset(timestamp, 0, GGMLHEXAGON_TMPBUF_LEN);

    GGMLHEXAGON_LOG_VERBOSE("ggml_hexagon_version:             %s", g_hexagon_appcfg.version);
    ggmlhexagon_get_timestring(timestamp);
    if (2 != g_hexagon_appcfg.offload_cgraph_type) { // only support F32, F16, Q4_0 in FastRPC per-op mode
        GGMLHEXAGON_LOG_VERBOSE("offload MUL_MAT types:            %s", "F32, F16, Q4_0");
    } else {
        GGMLHEXAGON_LOG_VERBOSE("offload MUL_MAT types:            %s", g_hexagon_appcfg.enabled_types.empty() ? "ALL" : g_hexagon_appcfg.enabled_types.c_str());
    }
    GGMLHEXAGON_LOG_VERBOSE("using rpc ion memory pool:        %s", ggmlhexagon_use_ion_mempool() ? "YES" : "NO");
    GGMLHEXAGON_LOG_VERBOSE("thread_counts on CDSP:            %d", g_hexagon_appcfg.thread_counts);
    int algotype = g_hexagon_appcfg.mulmat_algotype;
    GGMLHEXAGON_LOG_VERBOSE("mulmat algo type on CDSP:         %d(%s)", algotype, ggmlhexagon_get_mulmat_algotype_desc(algotype));
    GGMLHEXAGON_LOG_VERBOSE("mulmat min N for DSP offload:     %d", g_hexagon_appcfg.mulmat_min_n);
    GGMLHEXAGON_LOG_VERBOSE("offload cgraph type:              %d", g_hexagon_appcfg.offload_cgraph_type);
    GGMLHEXAGON_LOG_VERBOSE("ion_sync_mode:                    %d", g_hexagon_appcfg.ion_sync_mode);
    GGMLHEXAGON_LOG_VERBOSE("dump diag info:                   %d", g_hexagon_appcfg.dump_diag_info);
    GGMLHEXAGON_LOG_VERBOSE("ggml-dsp use hvx:                 %d", g_hexagon_appcfg.ggml_dsp_use_hvx);
    if (NULL != ctx) {
        GGMLHEXAGON_LOG_VERBOSE("ggml-dsp use hmx:                 %d", ctx->has_hmx);
    }
    if (2 != g_hexagon_appcfg.offload_cgraph_type) { // only support MUL_MAT in FastRPC per-op mode
        GGMLHEXAGON_LOG_VERBOSE("enabled_ops:                      %s", "MUL_MAT");
    } else {
        if (g_hexagon_appcfg.mulmat_algotype == 29) { // 29 = Qualcomm execute_op, all ops are enabled
            GGMLHEXAGON_LOG_VERBOSE("enabled_ops:                      %s", "ALL");
        } else {
            GGMLHEXAGON_LOG_VERBOSE("enabled_ops:                      %s", g_hexagon_appcfg.enabled_ops.c_str());
        }
    }
    GGMLHEXAGON_LOG_VERBOSE("running timestamp:%s", timestamp);
}

// Compute min/p50/p95/max over a ring buffer. `count` <= cap, so we sort a
// scratch copy. Sorts are only paid at dump time (once per deinit), so the
// per-call hot path stays branch-free.
static void ggmlhexagon_compute_hist_stats(const int64_t * hist, int count, int64_t & mn, int64_t & p50, int64_t & p95, int64_t & mx) {
    if (count <= 0) { mn = p50 = p95 = mx = 0; return; }
    std::vector<int64_t> tmp(hist, hist + count);
    std::sort(tmp.begin(), tmp.end());
    mn  = tmp.front();
    mx  = tmp.back();
    p50 = tmp[count / 2];
    p95 = tmp[(int)((int64_t)count * 95 / 100)];
    if (p95 >= count) p95 = count - 1;
}

// int32_t variant for n_nodes / n_ops which are stored as int32_t to save
// memory. Returning into int64_t keeps the print site uniform with the
// int64_t variant above.
static void ggmlhexagon_compute_hist_stats_i32(const int32_t * hist, int count, int64_t & mn, int64_t & p50, int64_t & p95, int64_t & mx) {
    if (count <= 0) { mn = p50 = p95 = mx = 0; return; }
    std::vector<int32_t> tmp(hist, hist + count);
    std::sort(tmp.begin(), tmp.end());
    mn  = tmp.front();
    mx  = tmp.back();
    p50 = tmp[count / 2];
    p95 = tmp[(int)((int64_t)count * 95 / 100)];
    if (p95 >= count) p95 = count - 1;
}

// dump accumulated performance statistics collected during graph_compute_batch
static void ggmlhexagon_dump_perf_stats(const ggml_backend_hexagon_context * ctx) {
    if (nullptr == ctx) {
        return;
    }
    GGMLHEXAGON_LOG_VERBOSE("rpc stats: batch_calls=%llu cum_p7=%lld us cum_graph=%lld us avg_p7=%lld us avg_graph=%lld us",
                             (unsigned long long)ctx->rpc_batch_call_count,
                             (long long)ctx->cumulative_p7_us, (long long)ctx->cumulative_graph_us,
                             ctx->rpc_batch_call_count ? (long long)(ctx->cumulative_p7_us / (int64_t)ctx->rpc_batch_call_count) : 0,
                             ctx->rpc_batch_call_count ? (long long)(ctx->cumulative_graph_us / (int64_t)ctx->rpc_batch_call_count) : 0);
    GGMLHEXAGON_LOG_VERBOSE("graph nodes: min=%u max=%u total=%u",
                             ctx->min_nodes_per_graph, ctx->max_nodes_per_graph, ctx->total_nodes_processed);
    GGMLHEXAGON_LOG_VERBOSE("per-call range: graph=[%lld, %lld] us p7=[%lld, %lld] us",
                             (long long)ctx->min_graph_us, (long long)ctx->max_graph_us,
                             (long long)ctx->min_p7_us, (long long)ctx->max_p7_us);
    GGMLHEXAGON_LOG_VERBOSE("max graph detail: dur=%lld us n_nodes=%u n_ops=%u",
                             (long long)ctx->max_graph_us, ctx->max_graph_n_nodes, ctx->max_graph_n_ops);
    GGMLHEXAGON_LOG_VERBOSE("AP phase cumulative: p1=%lld p2=%lld p2.5=%lld p3=%lld p4=%lld p4.5=%lld p5=%lld p6=%lld p6.5=%lld p7.5=%lld p8=%lld us",
                             (long long)ctx->cum_p1_us, (long long)ctx->cum_p2_us,
                             (long long)ctx->cum_p25_us, (long long)ctx->cum_p3_us,
                             (long long)ctx->cum_p4_us, (long long)ctx->cum_p45_us,
                             (long long)ctx->cum_p5_us,
                             (long long)ctx->cum_p6_us, (long long)ctx->cum_p65_us,
                             (long long)ctx->cum_p75_us, (long long)ctx->cum_p8_us);
    // Fine-grained: 3-way p7 split + per-call distribution
    GGMLHEXAGON_LOG_VERBOSE("p7 3-way cumulative: rpc_setup=%lld dsp_exec=%lld civac=%lld us (sum=%lld)",
                             (long long)ctx->cum_p7_rpc_setup_us,
                             (long long)ctx->cum_p7_dsp_exec_us,
                             (long long)ctx->cum_p7_civac_us,
                             (long long)(ctx->cum_p7_rpc_setup_us + ctx->cum_p7_dsp_exec_us + ctx->cum_p7_civac_us));
    GGMLHEXAGON_LOG_VERBOSE("rpc overhead (probe): n=%u min=%lld max=%lld avg=%lld us (upper bound, includes DSP-side cache flush+memset+LOG_INFO)",
                             ctx->rpc_overhead_count,
                             (long long)ctx->rpc_overhead_min_us, (long long)ctx->rpc_overhead_max_us,
                             ctx->rpc_overhead_count ? (long long)(ctx->rpc_overhead_sum_us / (int64_t)ctx->rpc_overhead_count) : 0);
    const uint64_t total_cache_lookups = ctx->cgraph_cache_hits + ctx->cgraph_cache_misses;
    GGMLHEXAGON_LOG_VERBOSE("cgraph cache: hits=%llu misses=%llu (hit_rate=%.1f%%) entries=%zu",
                             (unsigned long long)ctx->cgraph_cache_hits,
                             (unsigned long long)ctx->cgraph_cache_misses,
                             total_cache_lookups ? (100.0 * ctx->cgraph_cache_hits / total_cache_lookups) : 0.0,
                             ctx->cgraph_cache.size());

    // Per-call distribution (min/p50/p95/max) for the last PERF_HIST_CAP calls
    if (ctx->perf_hist_count > 0) {
        int64_t mn, p50, p95, mx;
        const int n = ctx->perf_hist_count;
        GGMLHEXAGON_LOG_ALWAYS("---- per-call distribution over last %d calls (us) ----", n);

        #define DUMP_PHASE_HIST(NAME, ARR, IS_FORGROUND_LOG) do { \
            ggmlhexagon_compute_hist_stats((ARR), n, mn, p50, p95, mx); \
            if (IS_FORGROUND_LOG) { \
                GGMLHEXAGON_LOG_VERBOSE("  %-5s min=%6lld p50=%6lld p95=%6lld max=%6lld", \
                    (NAME), (long long)mn, (long long)p50, (long long)p95, (long long)mx); \
            } else { \
                GGMLHEXAGON_LOG_ALWAYS("  %-5s min=%6lld p50=%6lld p95=%6lld max=%6lld", \
                    (NAME), (long long)mn, (long long)p50, (long long)p95, (long long)mx); \
            } \
        } while (0)
        DUMP_PHASE_HIST("p1",   ctx->p1_hist,   false);
        DUMP_PHASE_HIST("p2",   ctx->p2_hist,   false);
        DUMP_PHASE_HIST("p2.5", ctx->p25_hist,  false);
        DUMP_PHASE_HIST("p3",   ctx->p3_hist,   false);
        DUMP_PHASE_HIST("p4",   ctx->p4_hist,   false);
        DUMP_PHASE_HIST("p4.5", ctx->p45_hist,  false);
        DUMP_PHASE_HIST("p5",   ctx->p5_hist,   false);
        DUMP_PHASE_HIST("p6",   ctx->p6_hist,   false);
        DUMP_PHASE_HIST("p6.5", ctx->p65_hist,  false);
        DUMP_PHASE_HIST("p7",   ctx->p7_hist,   false);
        DUMP_PHASE_HIST("p7.5", ctx->p75_hist,  false);
        DUMP_PHASE_HIST("p8",   ctx->p8_hist,   false);
        DUMP_PHASE_HIST("p7rpc", ctx->p7_rpc_setup_hist, true);
        DUMP_PHASE_HIST("p7dsp", ctx->p7_dsp_exec_hist, true);
        DUMP_PHASE_HIST("p7civ", ctx->p7_civac_hist, true);
        DUMP_PHASE_HIST("graph", ctx->graph_us_hist, true);
        DUMP_PHASE_HIST("gap",   ctx->gap_from_prev_hist, true);
        #undef DUMP_PHASE_HIST
        // n_nodes / n_ops are int32_t in ctx, use the i32 variant
        ggmlhexagon_compute_hist_stats_i32(ctx->n_nodes_hist, n, mn, p50, p95, mx);
        GGMLHEXAGON_LOG_ALWAYS("  %-5s min=%6lld p50=%6lld p95=%6lld max=%6lld",
            "n_node", (long long)mn, (long long)p50, (long long)p95, (long long)mx);
        ggmlhexagon_compute_hist_stats_i32(ctx->n_ops_hist, n, mn, p50, p95, mx);
        GGMLHEXAGON_LOG_ALWAYS("  %-5s min=%6lld p50=%6lld p95=%6lld max=%6lld",
            "n_ops",  (long long)mn, (long long)p50, (long long)p95, (long long)mx);
    }
}

// =================================================================================================
//  section-4: configuration class and helper functions
// =================================================================================================
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
                GGMLHEXAGON_LOG_ALWAYS("key %s value %s\n", key.c_str(), value.c_str());
                GGMLHEXAGON_LOG_ALWAYS("key %s new value %s\n", key.c_str(), newvalue.c_str());
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
        GGMLHEXAGON_LOG_ALWAYS("%s", tmposs.str().c_str());
    });
    std::string version; //version of ggml-hexagon
    hexagoncfg_instance.get_stringvalue("general", "version", version, "0.99");
    hexagoncfg_instance.get_intvalue("general", "dump_debug_info", g_hexagon_appcfg.dump_debug_info, 0);

    hexagoncfg_instance.get_intvalue("cdsp", "thread_counts", g_hexagon_appcfg.thread_counts, 4);
    hexagoncfg_instance.get_intvalue("cdsp", "mulmat_algotype", g_hexagon_appcfg.mulmat_algotype, 0);
    hexagoncfg_instance.get_intvalue("cdsp", "mulmat_min_n", g_hexagon_appcfg.mulmat_min_n, 32);
    hexagoncfg_instance.get_intvalue("cdsp", "offload_cgraph_type", g_hexagon_appcfg.offload_cgraph_type, 2);
    hexagoncfg_instance.get_intvalue("cdsp", "dump_diag_info", g_hexagon_appcfg.dump_diag_info, 0);
    hexagoncfg_instance.get_intvalue("cdsp", "ggml_dsp_use_hvx", g_hexagon_appcfg.ggml_dsp_use_hvx, 1);
    hexagoncfg_instance.get_intvalue("cdsp", "ndev", g_hexagon_appcfg.ndev, 1);
    hexagoncfg_instance.get_intvalue("cdsp", "ion_sync_mode", g_hexagon_appcfg.ion_sync_mode, 0);
    hexagoncfg_instance.get_intvalue("cdsp", "enable_opfusion", g_hexagon_appcfg.enable_opfusion, 1);
    hexagoncfg_instance.get_intvalue("cdsp", "fa_select", g_hexagon_appcfg.fa_select, 2);
    hexagoncfg_instance.get_intvalue("cdsp", "gemv_offload", g_hexagon_appcfg.gemv_offload, 1);
    hexagoncfg_instance.get_stringvalue("cdsp", "enabled_ops", g_hexagon_appcfg.enabled_ops, "");
    hexagoncfg_instance.get_stringvalue("cdsp", "enabled_types", g_hexagon_appcfg.enabled_types, "");

    memcpy(g_hexagon_appcfg.version, version.c_str(), strlen(version.c_str()));

    GGMLHEXAGON_LOG_ALWAYS("load hexagon appcfg from %s", cfg_filename.c_str());
    GGMLHEXAGON_LOG_ALWAYS("ggml_hexagon_version=%s", g_hexagon_appcfg.version);
    GGMLHEXAGON_LOG_ALWAYS("runtime libpath=%s", g_hexagon_appcfg.runtime_libpath);

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

    initialized = true;
}

int ggml_backend_hexagon_get_mulmat_algotype() {
    std::string cfg_filename = std::string(g_hexagon_appcfg.runtime_libpath) + std::string(g_hexagon_appcfg.cfgfilename);
    hexagon_appcfg hexagoncfg_instance;
    hexagoncfg_instance.load(cfg_filename);
    hexagoncfg_instance.get_intvalue("cdsp", "mulmat_algotype", g_hexagon_appcfg.mulmat_algotype, 0);
    return g_hexagon_appcfg.mulmat_algotype;
}

void ggml_backend_hexagon_set_mulmat_algotype(int new_mulmat_algotype) {
    if (new_mulmat_algotype < 0) {
        GGMLHEXAGON_LOG_WARN("invalid new_mulmat_algotype");
        return;
    }
    std::string cfg_filename = std::string(g_hexagon_appcfg.runtime_libpath) + std::string(g_hexagon_appcfg.cfgfilename);
    GGMLHEXAGON_LOG_ALWAYS("load hexagon appcfg from %s", cfg_filename.c_str());
    hexagon_appcfg hexagoncfg_instance;
    GGMLHEXAGON_LOG_ALWAYS("set_hexagon_cfg with new_mulmat_algotype %d", new_mulmat_algotype);
    hexagoncfg_instance.modify_hexagon_config(cfg_filename, new_mulmat_algotype);
    hexagoncfg_instance.load(cfg_filename);
    hexagoncfg_instance.dump([](const std::string & section, const std::string & key, const std::string value) {
        std::ostringstream  tmposs;
        tmposs << "section[" << std::setw(10) << std::left << section << "],[" << std::setw(25) << std::left << key << "] = [" << value << "]";
        GGMLHEXAGON_LOG_ALWAYS("%s", tmposs.str().c_str());
    });
    // ggmlhexagon_load_cfg() is one-shot (static initialized flag). If it was
    // already called before us (e.g. during static init or earlier backend
    // registration), it won't re-read the cfg file. Update the global directly
    // so the new algotype takes effect regardless of call ordering.
    g_hexagon_appcfg.mulmat_algotype = new_mulmat_algotype;
    GGMLHEXAGON_LOG_INFO("mulmat_algotype set to %d (cfg file and global updated)", new_mulmat_algotype);
}

static bool ggmlhexagon_check_valid_appcfg() {
    if (g_hexagon_appcfg.offload_cgraph_type != 0 && g_hexagon_appcfg.offload_cgraph_type != 2) {
        GGMLHEXAGON_LOG_WARN("invalid offload_cgraph_type %d, reset to 2 (only 0=per-op and 2=ION-batch supported)",
                             g_hexagon_appcfg.offload_cgraph_type);
        g_hexagon_appcfg.offload_cgraph_type = 2;
    }

    if (g_hexagon_appcfg.offload_cgraph_type != 2) { // is not ION-based op-batch
        if (g_hexagon_appcfg.mulmat_algotype == 29) { // 29 = Qualcomm execute_op
            GGMLHEXAGON_LOG_WARN("mulmat_algotype can't be 29 when offload_cgraph_type !=2");
            g_hexagon_appcfg.mulmat_algotype = 32;
        }
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

    return true;
}

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

// Check if an op is allowed by the enabled_ops config filter.
// Returns true when:
//   - the op is a metadata op (NONE, NOOP)
//   - mulmat_algotype == 29 (Qualcomm execute_op forces all ops)
//   - enabled_ops is empty or contains "all" or the op name
static bool ggmlhexagon_op_is_enabled(enum ggml_op op) {
    if (ggmlhexagon_is_metadata_op(op)) {
        return true;
    }
    if (g_hexagon_appcfg.mulmat_algotype == 29) { // 29 = Qualcomm execute_op
        if (!g_hexagon_appcfg.enabled_ops.empty()) {
            GGMLHEXAGON_LOG_DEBUG("enabled_ops config ignored: mulmat_algotype=29 forces all ops");
        }
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

// =================================================================================================
//  section-5: general helper functions
// =================================================================================================
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

// ---- ARM64 cache maintenance for non-coherent ION ----
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
    // range-based DC CVAC with 8x loop unrolling (matching QCOM hex_l2flush pattern)
    if (size == 0) return;
    {
        const size_t line_size = 64;
        const char * start = (const char *)((uintptr_t)p & ~(line_size - 1));
        const char * end   = (const char *)p + size;
        // 8x unrolled: 8 cache lines per iteration
        for (; start + line_size * 8 <= end; start += line_size * 8) {
            __asm__ volatile("dc cvac, %0" : : "r"((const void *)(start + line_size * 0)) : "memory");
            __asm__ volatile("dc cvac, %0" : : "r"((const void *)(start + line_size * 1)) : "memory");
            __asm__ volatile("dc cvac, %0" : : "r"((const void *)(start + line_size * 2)) : "memory");
            __asm__ volatile("dc cvac, %0" : : "r"((const void *)(start + line_size * 3)) : "memory");
            __asm__ volatile("dc cvac, %0" : : "r"((const void *)(start + line_size * 4)) : "memory");
            __asm__ volatile("dc cvac, %0" : : "r"((const void *)(start + line_size * 5)) : "memory");
            __asm__ volatile("dc cvac, %0" : : "r"((const void *)(start + line_size * 6)) : "memory");
            __asm__ volatile("dc cvac, %0" : : "r"((const void *)(start + line_size * 7)) : "memory");
        }
        // tail: remaining lines
        for (; start < end; start += line_size) {
            __asm__ volatile("dc cvac, %0" : : "r"((const void *)start) : "memory");
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
    // range-based DC CIVAC with 8x loop unrolling (matching QCOM hex_l2flush pattern)
    if (size == 0) return;
    {
        const size_t line_size = 64;
        const char * start = (const char *)((uintptr_t)p & ~(line_size - 1));
        const char * end   = (const char *)p + size;
        // 8x unrolled: 8 cache lines per iteration
        for (; start + line_size * 8 <= end; start += line_size * 8) {
            __asm__ volatile("dc civac, %0" : : "r"((const void *)(start + line_size * 0)) : "memory");
            __asm__ volatile("dc civac, %0" : : "r"((const void *)(start + line_size * 1)) : "memory");
            __asm__ volatile("dc civac, %0" : : "r"((const void *)(start + line_size * 2)) : "memory");
            __asm__ volatile("dc civac, %0" : : "r"((const void *)(start + line_size * 3)) : "memory");
            __asm__ volatile("dc civac, %0" : : "r"((const void *)(start + line_size * 4)) : "memory");
            __asm__ volatile("dc civac, %0" : : "r"((const void *)(start + line_size * 5)) : "memory");
            __asm__ volatile("dc civac, %0" : : "r"((const void *)(start + line_size * 6)) : "memory");
            __asm__ volatile("dc civac, %0" : : "r"((const void *)(start + line_size * 7)) : "memory");
        }
        // tail: remaining lines
        for (; start < end; start += line_size) {
            __asm__ volatile("dc civac, %0" : : "r"((const void *)start) : "memory");
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

// True for metadata-only ops that never execute on CDSP.
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

// Returns a short description of mulmat_algotype. See ggml-hexagon.cfg for full details:
// algotype == 29 selects Qualcomm execute_op path (needs tile-based weight repack on AP);
// other values select self-built ggmlop_dsp_mulmat_* kernels (no repack, except algotype=30
// which uses x4x2 repack for faster HVX dequantization).
static const char * ggmlhexagon_get_mulmat_algotype_desc(int algotype) {
    switch (algotype) {
        case 0:  return "HVX multithread (default)";
        case 29: return "Qualcomm execute_op (tile-based repack)";
        case 30: return "HMX sync (x4x2 repack)";
        case 32: return "HMX pipeline (standard layout)";
        case 33: return "HVX multithread + VTCM";
        default: return "unknown";
    }
}

static inline bool ggml_hexagon_is_repack_type(enum ggml_type type) {
    return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 ||
           type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_NL ||
           type == GGML_TYPE_MXFP4;
}

static inline bool ggml_hexagon_is_hmx_weight_type(enum ggml_type type) {
    return type == GGML_TYPE_F16 || type == GGML_TYPE_F32 || ggml_hexagon_is_repack_type(type);
}

// =================================================================================================
//  section-6: CDSP helper functions
// =================================================================================================
static const char * ggmlhexagon_get_dsp_name(int domain_id) {
    (void)domain_id;
    return "Hexagon-cDSP";
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
        if (domain == CDSP_DOMAIN_ID) {
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

static bool ggmlhexagon_is_async_fastrpc_supported(int domain) {
    int hexagon_error = AEE_SUCCESS;
    if (remote_handle_control) {
        if (domain == CDSP_DOMAIN_ID) {
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
    (void)latency;

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
            GGMLHEXAGON_LOG_VERBOSE("set rpc qos %d (DSP default latency)", qos);
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
 * Unprivileged clients: 64 through 254 (CDSP only)
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

static int ggmlhexagon_get_hmx_support_info(int domain, uint32_t attr, uint32_t * capability) {
    int hexagon_error = AEE_SUCCESS;
    *capability = 0;

    if (attr != HMX_SUPPORT_SPATIAL && attr != HMX_SUPPORT_DEPTH) {
        hexagon_error = AEE_EBADPARM;
        GGMLHEXAGON_LOG_WARN("unsupported attr, only HMX_SUPPORT_SPATIAL and HMX_SUPPORT_DEPTH supported");
        goto bail;
    }

    if (remote_handle_control) {
        if (domain == CDSP_DOMAIN_ID) {
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

static int ggmlhexagon_get_hvx_support_info(int domain, uint32_t attr, uint32_t * capability) {
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
        if (domain == CDSP_DOMAIN_ID) {
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
        if (2 != g_hexagon_appcfg.offload_cgraph_type) {
            probe_slots.push_back(4096);
        }
    } else {
        probe_slots.push_back(1024+2048+200);
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
    GGMLHEXAGON_LOG_VERBOSE("capacity of rpc memory %d MiB", ctx->rpc_mempool_capacity / SIZE_IN_MB);

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
        if (cache_pool_size > 0 && ctx->rpc_mempool_len > cache_pool_size && g_hexagon_appcfg.mulmat_algotype == 32) {
            ctx->rpc_mempool_cache_offset = ctx->rpc_mempool_len - cache_pool_size;
        } else {
            ctx->rpc_mempool_cache_offset = 0;  // no cache
        }
        ctx->rpc_mempool_cache_budget = ctx->rpc_mempool_len - ctx->rpc_mempool_cache_offset;
        GGMLHEXAGON_LOG_ALWAYS("ION layout: total=%zuMB, cache_offset=%zuMB, cache_size=%zuMB, data_region=%zuMB",
                             ctx->rpc_mempool_len / SIZE_IN_MB,
                             ctx->rpc_mempool_cache_offset / SIZE_IN_MB,
                             (ctx->rpc_mempool_len - ctx->rpc_mempool_cache_offset) / SIZE_IN_MB,
                             ctx->rpc_mempool_cache_offset / SIZE_IN_MB);

        // Set up FP16 weight cache region on DSP side
        // Uses special batch_size=0xFFFF to signal cache setup mode
        if (ctx->rpc_mempool_cache_offset > 0) {
            uint32_t cache_offset_lo = (uint32_t)(ctx->rpc_mempool_cache_offset & 0xFFFFFFFF);
            int cache_err = ggmlop_dsp_execute_batch(ctx->ggmlop_handle, cache_offset_lo, 0xFFFF);
            if (cache_err == AEE_SUCCESS) {
                GGMLHEXAGON_LOG_ALWAYS("DSP FP16 weight cache set up: offset=%zuMB, size=%zuMB",
                                     ctx->rpc_mempool_cache_offset / SIZE_IN_MB,
                                     (ctx->rpc_mempool_len - ctx->rpc_mempool_cache_offset) / SIZE_IN_MB);
            } else {
                GGMLHEXAGON_LOG_ERROR("DSP FP16 weight cache setup failed: 0x%x", cache_err);
            }
        }

        // [ION-PROBE] Verify bidirectional ION shared memory access.
        // Call with batch_size=0 --> DSP enters probe mode: writes 0xAB at base+0,
        // 0xCD at base+64. AP then reads back to confirm DSP writes are visible.
        {
            int probe_err = hexagon_probe_invoke_timed(ctx);
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
                // Clean up probe patterns
                memset((void *)p, 0, 16);
                memset((void *)(p + 64), 0, 16);
            } else {
                GGMLHEXAGON_LOG_WARN("[AP-PROBE] dsp_execute_batch probe failed: 0x%x", probe_err);
            }

            // [ION-MULTI-INVOKE] Test: verify no repeated mmap/munmap on subsequent invokes.
            // Call dsp_execute_batch N times with different write patterns.
            // Check DSP log for "fastrpc_invoke_fd_mmap_create" — should NOT appear after 1st call.
            {
                const int N_ROUNDS = 5;
                bool multi_ok = true;

                for (int round = 0; round < N_ROUNDS; round++) {
                    uint8_t pattern = (uint8_t)(0xA0 + round);
                    // Write pattern from AP side first (AP --> DSP direction)
                    memset((void *)ctx->rpc_mempool, pattern, 16);
                    __builtin___clear_cache((char *)ctx->rpc_mempool,
                                            (char *)ctx->rpc_mempool + 16);

                    int err = hexagon_probe_invoke_timed(ctx);
                    if (err != AEE_SUCCESS) {
                        GGMLHEXAGON_LOG_ERROR("[MULTI-PROBE] round %d/%d invoke FAILED: 0x%x",
                                              round + 1, N_ROUNDS, err);
                        multi_ok = false;
                        break;
                    }

                    // Read back DSP-written data (DSP --> AP direction)
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

                    GGMLHEXAGON_LOG_ALWAYS("[MULTI-PROBE] round %d/%d PASS (invoke OK, data verified)",
                                         round + 1, N_ROUNDS);

                    // Clean up for next round
                    memset((void *)r, 0, 16);
                    memset((void *)(r + 64), 0, 16);
                }

                if (multi_ok) {
                    GGMLHEXAGON_LOG_ALWAYS("=== MULTI-INVOKE TEST PASSED: %d rounds, NO repeated mmap ===",
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
        GGMLHEXAGON_LOG_VERBOSE("dsp arch version 0x%x", dsp_version);
        //0x68 -> 68, 0x69 -> 69, 0x73 -> 73, 0x75 -> 75, 0x79 -> 79, 0x81 -> 81
        htp_arch = ggmlhexagon_htparch_hex_to_decimal(dsp_version);
        GGMLHEXAGON_LOG_DEBUG("dsp arch version %d", htp_arch);
        struct qcom_socinfo * socinfo = ggmlhexagon_get_socinfo_from_htparch(htp_arch);
        if (nullptr != socinfo) {
            ctx->socinfo = *socinfo;
            GGMLHEXAGON_LOG_VERBOSE("device info: %s, %s", socinfo->soc_desc, ggmlhexagon_get_htparch_desc(htp_arch));
        }
    } else {
        GGMLHEXAGON_LOG_VERBOSE("error: dsp arch version 0x%x is not supported", dsp_version);
    }

    uint32_t vtcm_count = 0;
    uint32_t vtcm_page  = 0;
    ggmlhexagon_get_vtcm_info(ctx->domain_id, VTCM_COUNT, &vtcm_count);
    ggmlhexagon_get_vtcm_info(ctx->domain_id, VTCM_PAGE, &vtcm_page);
    ctx->has_vtcm = (vtcm_count > 0 && vtcm_page > 0);

    uint32_t hmx_depth = 0;
    uint32_t hmx_spatial = 0;
    //FIXME: better approach to get correct/accurate info
    ggmlhexagon_get_hmx_support_info(ctx->domain_id, HMX_SUPPORT_DEPTH, &hmx_depth);
    ggmlhexagon_get_hmx_support_info(ctx->domain_id, HMX_SUPPORT_SPATIAL, &hmx_spatial);

    uint32_t hvx_support_128b = 0;
    ggmlhexagon_get_hvx_support_info(ctx->domain_id, HVX_SUPPORT_128B, &hvx_support_128b);
    ctx->has_hvx = (hvx_support_128b > 0);

    ctx->has_hmx = (hmx_depth > 0 || hmx_spatial > 0);
    // Fallback: DSPRPC_GET_DSP_INFO may not report HMX on some drivers
    // (returns EUNSUPPORTEDAPI, leaving capability 0). HMX is present on
    // V73+ (Snapdragon 8 Gen 2 and later); the DSP skel is built with -mhmx.
    if (!ctx->has_hmx && htp_arch >= V73) {
        ctx->has_hmx = true;
    }

    GGMLHEXAGON_LOG_VERBOSE("vtcm_count %d", vtcm_count);
    GGMLHEXAGON_LOG_VERBOSE("vtcm_page %d", vtcm_page);
    //FIXME: the log output is "hmx_depth 0 hmx_spatial 0", this is not correct
    //GGMLHEXAGON_LOG_VERBOSE("hmx_depth %d hmx_spatial %d", hmx_depth, hmx_spatial);
    GGMLHEXAGON_LOG_VERBOSE("hvx_support_128b %d", hvx_support_128b);
    GGMLHEXAGON_LOG_VERBOSE("unsigned pd supported %d", ggmlhexagon_is_unsignedpd_supported(ctx->domain_id));
    GGMLHEXAGON_LOG_VERBOSE("async fastrpc supported %d", ggmlhexagon_is_async_fastrpc_supported(ctx->domain_id));
    GGMLHEXAGON_LOG_VERBOSE("device %d caps: has_vtcm=%d has_hvx=%d has_hmx=%d", ctx->device, (int)ctx->has_vtcm, (int)ctx->has_hvx, (int)ctx->has_hmx);
    return htp_arch;
}

static void ggmlhexagon_deinit_cdsp(ggml_backend_hexagon_context * ctx) {
    int hexagon_error  = AEE_SUCCESS;
    GGMLHEXAGON_LOG_INFO("enter %s", __func__);
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
    ggmlhexagon_dump_perf_stats(ctx);
    ctx->domain_id             = -1;
    GGMLHEXAGON_LOG_INFO("leave %s", __func__);
}

static int ggmlhexagon_init_dsp(ggml_backend_hexagon_context * ctx) {
    int htp_arch                = 0;
    int hexagon_error           = AEE_SUCCESS;
    int domain_id               = CDSP_DOMAIN_ID;
    bool got_uri                = false;
    const char * uri            = NULL;
    domain * my_domain          = NULL;
    bool is_unsignedpd_enabled  = false;
    char final_uri[512];
    char ggmldsp_uri[256];

    if (nullptr == ctx)
        return 1;
    if (0 != ctx->ggmlop_handle) {
        GGMLHEXAGON_LOG_DEBUG("already init Hexagon CDSP with backend %d(%s)", ctx->device, ctx->name);
        return 0;
    }
    if (!remote_session_control) {
        GGMLHEXAGON_LOG_ERROR("remote_session_control not available");
        hexagon_error = AEE_EUNSUPPORTED;
        goto bail;
    }

    GGMLHEXAGON_LOG_DEBUG("init Hexagon CDSP with backend %d(%s)", ctx->device, ctx->name);
    ctx->ggmlop_handle = 0;
    my_domain = get_domain(domain_id);
    if (NULL == my_domain) {
        GGMLHEXAGON_LOG_ERROR("unable to get domain struct %d", domain_id);
        goto bail;
    }
    uri = my_domain->uri;
    GGMLHEXAGON_LOG_DEBUG("domain uri=%s", uri);
    // Reserve new FastRPC session (PD) for additional devices (dev_id > 0)
    // dev_id == 0 reuses the default CDSP PD (session_id=0)
    ctx->session_id = 0;
    if (ctx->device > 0) {
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
    }

    is_unsignedpd_enabled = ggmlhexagon_is_unsignedpd_supported(domain_id);
    if (!is_unsignedpd_enabled) {
        GGMLHEXAGON_LOG_ERROR("unsigned PD not allowed on domain %d, using signed offload", domain_id);
        goto bail;
    }

    ctx->domain_id = domain_id;
    GGMLHEXAGON_LOG_VERBOSE("using Hexagon domain %d(%s)", domain_id, ggmlhexagon_get_dsp_name(domain_id));
    GGMLHEXAGON_LOG_VERBOSE("unsignedpd_enabled %d", is_unsignedpd_enabled);
    if (is_unsignedpd_enabled) {
        struct remote_rpc_control_unsigned_module data;
        data.enable = 1;
        data.domain = domain_id;
        hexagon_error = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&data, sizeof(data));
        GGMLHEXAGON_LOG_DEBUG("remote_session_control returned %d for configuring unsigned PD", hexagon_error);
        if (AEE_SUCCESS != hexagon_error) {
            GGMLHEXAGON_LOG_ERROR("error 0x%x: remote_session_control failed", hexagon_error);
            goto bail;
        }
    }

    // Probe arch and build the versioned dsp skel URI
    htp_arch = ggmlhexagon_probe_dspinfo(ctx);
    GGML_ASSERT(htp_arch != 0);

    snprintf(ggmldsp_uri, sizeof(ggmldsp_uri),
             "file:///libggmldsp-skel-v%u.so?ggmldsp_skel_handle_invoke&_modver=1.0&_idlver=0.0.1",
             htp_arch);

    // Build the final URI for ggmlop_dsp_open.
    // session_id > 0: use FASTRPC_GET_URI to obtain the session-specific URI.
    // session_id == 0 (or FASTRPC_GET_URI failure): concatenate ggmldsp_uri + domain uri.
    if (ctx->session_id > 0) {
        struct remote_rpc_get_uri u = {};
        u.session_id      = ctx->session_id;
        u.domain_name     = const_cast<char *>(CDSP_DOMAIN_NAME);
        u.domain_name_len = strlen(CDSP_DOMAIN_NAME);
        u.module_uri      = const_cast<char *>(ggmldsp_uri);
        u.module_uri_len  = strlen(ggmldsp_uri);
        u.uri             = final_uri;
        u.uri_len         = sizeof(final_uri);
        int err = remote_session_control(FASTRPC_GET_URI, (void *) &u, sizeof(u));
        if (err == AEE_SUCCESS) {
            got_uri = true;
            GGMLHEXAGON_LOG_INFO("session URI for session_id=%d: %s", ctx->session_id, final_uri);
        } else {
            GGMLHEXAGON_LOG_WARN("FASTRPC_GET_URI failed for session_id=%d: error 0x%x, fallback to %s%s",
                                 ctx->session_id, err, ggmldsp_uri, uri);
        }
    }
    if (!got_uri) {
        snprintf(final_uri, sizeof(final_uri), "%s%s", ggmldsp_uri, uri);
    }

    GGMLHEXAGON_LOG_DEBUG("ggmlop domain uri: %s", final_uri);
    hexagon_error = ggmlop_dsp_open(final_uri, &ctx->ggmlop_handle);
    if (AEE_SUCCESS == hexagon_error) {
        GGMLHEXAGON_LOG_VERBOSE("succeed to open domain %d(%s)", domain_id, ggmlhexagon_get_dsp_name(domain_id));
        ggmlop_dsp_setclocks(ctx->ggmlop_handle, g_hexagon_appcfg.dump_diag_info, g_hexagon_appcfg.offload_cgraph_type, g_hexagon_appcfg.mulmat_algotype, g_hexagon_appcfg.thread_counts);
        ggmlhexagon_set_rpc_latency(ctx->ggmlop_handle, RPC_PM_QOS, 100);
        if (0 != ggmlhexagon_init_rpcmempool(ctx)) {
            GGMLHEXAGON_LOG_INFO("failed to init rpc mempool");
            goto bail;
        }
    } else {
        GGMLHEXAGON_LOG_INFO("error 0x%x: failed to open domain %d(%s)", hexagon_error, domain_id,
                             ggmlhexagon_get_dsp_name(domain_id));
        goto bail;
    }

    snprintf(ctx->name, sizeof(ctx->name), "Hexagon-cDSP%d", ctx->device);
    return 0;

bail:
    ggmlhexagon_deinit_cdsp(ctx);
    return -1;
}

// =================================================================================================
//  section-7: Qualcomm compatibility layer(ported from Qualcomm's ggml-hexagon for mulmat_algotype=29 path)
// =================================================================================================

// FA kernel selection: 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
// Controlled by ggml-hexagon.cfg: [cdsp] fa_select
static int ggml_hexagon_get_fa_select(void) {
    return g_hexagon_appcfg.fa_select;
}

// Precompute htp_fa_kernel_params on AP side for FLASH_ATTN_EXT.
// Ported from Qualcomm's ggml-hexagon::ggml_hexagon_precompute_flash_attn_params.
// Writes to kparams; caller casts from op.kernel_params or a stack local.
// Returns true if a valid kernel (HMX or HVX) was selected.
static bool ggml_hexagon_compute_fa_params(
    const ggml_backend_hexagon_context * ctx,
    const ggml_tensor * node,
    struct htp_fa_kernel_params * kparams
) {
    if (ggml_hexagon_get_fa_select() < 1) {
        return false;
    }

    memset(kparams, 0, sizeof(*kparams));

    const ggml_tensor * q    = node->src[0];
    const ggml_tensor * k    = node->src[1];
    const ggml_tensor * v    = node->src[2];
    const ggml_tensor * mask = node->src[3];
    const ggml_tensor * dst  = node;

    const uint32_t DK = (uint32_t) q->ne[0];
    const uint32_t DV = (uint32_t) v->ne[0];
    const uint32_t neq1 = (uint32_t) q->ne[1];
    const uint32_t nek1 = (uint32_t) k->ne[1];
    const uint32_t n_kv_heads = (uint32_t) k->ne[2];
    const uint32_t G = (uint32_t) q->ne[2] / n_kv_heads;

    float scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&scale,         &node->op_params[0], sizeof(float));
    memcpy(&max_bias,      &node->op_params[1], sizeof(float));
    memcpy(&logit_softcap, &node->op_params[2], sizeof(float));
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    kparams->scale         = scale;
    kparams->max_bias      = max_bias;
    kparams->logit_softcap = logit_softcap;
    kparams->is_q_fp32     = (q->type == GGML_TYPE_F32) ? 1 : 0;
    kparams->is_dst_fp32   = (dst->type == GGML_TYPE_F32) ? 1 : 0;
    kparams->G             = G;

    const uint32_t n_head = (uint32_t) q->ne[2];
    // largest power of 2 <= n_head
    uint32_t n_head_log2 = 1;
    while (n_head_log2 * 2u <= n_head) n_head_log2 *= 2;
    kparams->n_head_log2 = n_head_log2;
    // 2^x = exp(x * ln2), avoiding powf dependency
    const float ln2 = 0.6931471805599453f;
    kparams->m0 = expf(-ln2 * max_bias / (float) n_head_log2);
    kparams->m1 = expf(-ln2 * (max_bias * 0.5f) / (float) n_head_log2);

    // HMX eligibility
    bool hmx_eligible = false;
    if (ctx->has_hmx && ggml_hexagon_get_fa_select() >= 2 &&
        k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16) {
        if (DK % 64 == 0 && DV % 64 == 0 && !(DK <= 128 && neq1 < 5)) {
            hmx_eligible = true;
        }
    }

    if (hmx_eligible) {
        size_t Br = 0, Bc = 0;
        const size_t vtcm_budget = ctx->socinfo.vtcm_size_in_mb * 1024 * 1024;
        int ret = hmx_fa_find_chunk_size(&Br, &Bc, G, DK, DV, neq1, nek1,
                                         vtcm_budget, (size_t) ctx->n_threads);
        if (ret == 0) {
            kparams->kernel_type = HTP_FA_KERNEL_HMX;
            kparams->Br          = (uint16_t) Br;
            kparams->Bc          = (uint16_t) Bc;
            kparams->n_kv_blocks = (uint16_t)((nek1 + Bc - 1) / Bc);
            kparams->n_threads   = (kparams->n_kv_blocks >= 3 && ctx->n_threads >= 2)
                                   ? (uint8_t) ctx->n_threads : 1;
            kparams->u.hmx.g_br      = hex_align_up(G * Br, 32);
            kparams->u.hmx.pipeline  = (kparams->n_kv_blocks >= 3 && ctx->n_threads >= 2) ? 1 : 0;
            kparams->vtcm_size       = (uint32_t) hmx_fa_compute_vtcm_usage(
                G, DK, DV, Br, Bc, kparams->n_threads, kparams->u.hmx.pipeline != 0);

            const size_t row_vec_bytes = hex_align_up(Bc * sizeof(uint16_t), 256);
            kparams->u.hmx.row_buf_stride = row_vec_bytes / 128;
            const size_t m_line_bytes = hex_align_up(Bc * sizeof(uint16_t), 128);
            kparams->u.hmx.mask_buf_row_stride = m_line_bytes / sizeof(uint16_t);
            kparams->u.hmx.mask_broadcast = (mask && mask->ne[2] == 1) ? 1 : 0;
            kparams->u.hmx.div_G = init_fastdiv_values(G);
            if (mask) {
                kparams->src3_div2 = init_fastdiv_values((uint32_t) mask->ne[2]);
                kparams->src3_div3 = init_fastdiv_values((uint32_t) mask->ne[3]);
            }
            kparams->qrows = 0;
            kparams->qrows_per_thread = 0;
            return true;
        }
    }

    // Fallback to HVX
    kparams->kernel_type    = HTP_FA_KERNEL_HVX;
    kparams->Br             = 1;
    kparams->Bc             = 64;
    kparams->n_kv_blocks    = (uint16_t)((k->ne[1] + 64 - 1) / 64);
    kparams->n_threads      = (uint8_t) ctx->n_threads;
    kparams->vtcm_size      = (uint32_t) hvx_fa_compute_vtcm_usage(
        DK, DV, kparams->is_q_fp32 != 0, mask != nullptr, (size_t) ctx->n_threads);

    kparams->u.hvx.size_q_row_padded = hex_round_up((uint32_t)(q->ne[0] * (kparams->is_q_fp32 ? 4 : 2)), 128);
    kparams->u.hvx.size_k_row_padded = hex_round_up((uint32_t)(k->ne[0] * 2), 128);
    kparams->u.hvx.size_v_row_padded = hex_round_up((uint32_t)(v->ne[0] * 2), 128);
    kparams->u.hvx.src0_div21     = init_fastdiv_values((uint32_t)(q->ne[2] * q->ne[1]));
    kparams->u.hvx.src0_div1      = init_fastdiv_values((uint32_t) q->ne[1]);
    kparams->broadcast_rk2   = init_fastdiv_values((uint32_t)(q->ne[2] / k->ne[2]));
    kparams->broadcast_rk3   = init_fastdiv_values((uint32_t)(q->ne[3] / k->ne[3]));
    kparams->broadcast_rv2   = init_fastdiv_values((uint32_t)(q->ne[2] / v->ne[2]));
    kparams->broadcast_rv3   = init_fastdiv_values((uint32_t)(q->ne[3] / v->ne[3]));
    if (mask) {
        kparams->src3_div2 = init_fastdiv_values((uint32_t) mask->ne[2]);
        kparams->src3_div3 = init_fastdiv_values((uint32_t) mask->ne[3]);
    }
    kparams->qrows           = (uint32_t)(q->ne[1] * q->ne[2] * q->ne[3]);
    kparams->qrows_per_thread = (kparams->qrows + ctx->n_threads - 1) / ctx->n_threads;
    return true;
}

static bool ggml_hexagon_matmul_is_hmx_eligible(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    int ne01_padded,
    bool is_matmul_id,
    bool is_batched
) {
    const int ne00  = src0->ne[0];
    const int ne11  = src1->ne[1];
    const int ne12  = src1->ne[2];
    const int wtype = src0->type;

    if (ne01_padded % 32 != 0) {
        return false;
    }

    if (!ggml_hexagon_is_hmx_weight_type((ggml_type) wtype)) {
        return false;
    }

    if (ne00 % 32 != 0) {
        return false;
    }

    if (!is_matmul_id && is_batched && wtype != GGML_TYPE_F16) {
        return false;
    }

    if (src0->nb[0] > src0->nb[1] || src1->nb[0] > src1->nb[1]) {
        return false;
    }

    const int m = is_matmul_id ? ne12 : ne11;
    if (m <= HTP_MM_HMX_MIN_NROWS) {
        return false;
    }

    return true;
}

// Shared HMX eligibility check: computes standard params from src0/src1/dst
// and delegates to ggml_hexagon_matmul_is_hmx_eligible. Used by both
// mm_is_hmx_eligible (opfusion gate) and precompute_mm_params (HMX delegation)
// to keep the two decision points consistent.
static bool ggml_hexagon_mm_is_hmx_eligible_shared(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst
) {
    const int wtype = src0->type;
    const bool is_repack    = ggml_hexagon_is_repack_type((ggml_type) wtype);
    const bool is_matmul_id = (dst->op == GGML_OP_MUL_MAT_ID);
    const bool is_batched   = (src0->ne[2] * src0->ne[3] > 1 || src1->ne[2] * src1->ne[3] > 1);
    const int  ne01_padded  = is_repack ? (int) hex_round_up((uint32_t) src0->ne[1], 32) : src0->ne[1];

    return ggml_hexagon_matmul_is_hmx_eligible(src0, src1, dst, ne01_padded, is_matmul_id, is_batched);
}

// AP-side gate deciding whether a single MUL_MAT is a candidate for the HMX
// pipeline (Qualcomm execute_op's high-throughput path for large batches).
// Returns true when src0/src1 shapes/types are suitable for HMX.
//
// NOTE: this gate only controls QKV/FFN *fusion eligibility*, NOT the actual
// HMX dispatch. HMX dispatch is decided independently on the DSP side by
// build_mm_kernel_params() in entry.c (which tries HMX first, then falls
// back to HVX). mm_is_hmx_eligible is consulted only by is_mergeable_mul_mat
// to avoid merging MUL_MATs that would otherwise benefit from HMX.
//
// For a single MUL_MAT that is NOT part of a QKV/FFN pattern:
//   - fusion does not apply regardless of this gate
//   - dispatch still goes through execute_op -> op_matmul, and may use HMX
//     if build_mm_kernel_params on DSP side selects an HMX kernel
//
// For MUL_MATs that ARE part of a QKV/FFN pattern:
//   - if this gate returns true:  fusion is skipped, each MUL_MAT goes through
//                                 op_matmul (may use HMX)
//   - if this gate returns false: fusion is attempted -> op_matmul_qkv/ffn
//                                 (separate Qualcomm fused kernels, NOT HMX)
static bool mm_is_hmx_eligible(const ggml_backend_hexagon_context * ctx, const ggml_tensor * t) {
    if (!ctx->has_hmx) return false;
    return ggml_hexagon_mm_is_hmx_eligible_shared(t->src[0], t->src[1], t);
}

// A MUL_MAT is fusion-eligible when:
//   - src0 is quantized (Q4_0/Q8_0/etc.)
//   - src1 is F32 (fusion kernels read F32 activations)
//   - !mm_is_hmx_eligible (avoid merging MUL_MATs that would otherwise benefit
//     from HMX; fusion redirects them to op_matmul_qkv/ffn, which is a
//     separate path from the HMX pipeline)
// NOTE: this only affects whether fusion is *attempted*. MUL_MATs that do
// not match the QKV/FFN pattern are never fused regardless of this check.
static bool is_mergeable_mul_mat(const ggml_backend_hexagon_context * ctx, const ggml_tensor * t) {
    if (!t || t->op != GGML_OP_MUL_MAT)   return false;
    if (t->src[1]->type != GGML_TYPE_F32) return false;
    return ggml_is_quantized(t->src[0]->type) && !mm_is_hmx_eligible(ctx, t);
}

static bool is_mergeable_mul_mat_pair(const ggml_backend_hexagon_context * ctx, const ggml_tensor * n1, const ggml_tensor * n2) {
    if (!is_mergeable_mul_mat(ctx, n1) || !is_mergeable_mul_mat(ctx, n2)) {
        return false;
    }
    if (n1->src[1] != n2->src[1]) {
        return false;
    }
    if (n1->src[0]->ne[0] != n2->src[0]->ne[0] ||
        n1->src[0]->ne[1] != n2->src[0]->ne[1]) {
        return false;
    }
    if (n1->src[0]->type != n2->src[0]->type) {
        return false;
    }
    return true;
}

static bool is_qkv_mergeable(const ggml_backend_hexagon_context * ctx, const ggml_tensor * n_q, const ggml_tensor * n_k, const ggml_tensor * n_v) {
    if (!is_mergeable_mul_mat(ctx, n_q) || !is_mergeable_mul_mat(ctx, n_k) || !is_mergeable_mul_mat(ctx, n_v)) {
        return false;
    }
    if (n_q->src[1] != n_k->src[1] || n_q->src[1] != n_v->src[1]) {
        return false;
    }
    if (n_q->src[0]->type != n_k->src[0]->type || n_q->src[0]->type != n_v->src[0]->type) {
        return false;
    }
    if (n_k->src[0]->ne[0] != n_v->src[0]->ne[0] ||
        n_k->src[0]->ne[1] != n_v->src[0]->ne[1]) {
        return false;
    }
    if (n_q->src[0]->ne[0] != n_k->src[0]->ne[0]) {
        return false;
    }
    return true;
}

// Precompute htp_mm_kernel_params for fused QKV matmul (3 outputs: K, V, Q).
// Mirrors Qualcomm's ggml-hexagon: ggml_hexagon_precompute_fused_qkv_params.
// src0 = Wk (representative of K/V/Q weights), src1 = x (shared activation).
// DSP-side op_matmul_qkv expects src[0]=Wk, src[1]=x, src[2]=Wv, src[3]=Wq.
static void ggml_hexagon_precompute_fused_qkv_params(
    const ggml_backend_hexagon_context * ctx,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const int wtype = src0->type;
    const bool is_repack = ggml_hexagon_is_repack_type((ggml_type) wtype);

    const int ne10 = src1->ne[0];
    const int src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
    const size_t src1_row_size = (wtype == GGML_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
    const size_t src0_row_size = src0->nb[1];
    const size_t src0_row_size_padded = hex_round_up((uint32_t) src0_row_size, 128);

    size_t src0_sz_per_thread = 0;
    size_t src2_sz_per_thread = 0;
    size_t src3_sz_per_thread = 0;
    uint32_t best_n_prefetch = 16;

    const size_t vtcm_budget = ctx->socinfo.vtcm_size_in_mb * 1024 * 1024;
    size_t quant_scratch_size = hex_round_up((uint32_t)(ne10 * sizeof(float)), QK_Q8_0_TILED * sizeof(float)) * (uint32_t)ctx->n_threads;

    if (is_repack) {
        uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size(wtype);
        uint32_t n_k_tiles = hex_round_up((uint32_t) ne10, 32) / 32;
        uint32_t tile_row_size = n_k_tiles * aligned_tile_size;
        size_t src1_sz_per_thread = hex_round_up((uint32_t)(src1_row_size * src1_nrows), 128);
        size_t src1_sz = src1_sz_per_thread;

        const uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
        best_n_prefetch = 2;
        for (uint32_t d = max_prefetch; d >= 2; d /= 2) {
            size_t repacked_vtcm_size = hex_round_up(d * tile_row_size, 128);
            size_t src0_sz = repacked_vtcm_size * (uint32_t)ctx->n_threads;
            size_t src2_sz = hex_round_up(d * tile_row_size, 128) * (uint32_t)ctx->n_threads;
            size_t src3_sz = hex_round_up(d * tile_row_size, 128) * (uint32_t)ctx->n_threads;
            size_t tiled_vtcm_size = src0_sz + src1_sz + src2_sz + src3_sz + quant_scratch_size;

            if (tiled_vtcm_size <= vtcm_budget) {
                best_n_prefetch = d;
                src0_sz_per_thread = repacked_vtcm_size;
                src2_sz_per_thread = hex_round_up(d * tile_row_size, 128);
                src3_sz_per_thread = hex_round_up(d * tile_row_size, 128);
                break;
            }
        }
        if (best_n_prefetch == 2 && src0_sz_per_thread == 0) {
            size_t repacked_vtcm_size = hex_round_up(2 * tile_row_size, 128);
            src0_sz_per_thread = repacked_vtcm_size;
            src2_sz_per_thread = hex_round_up(2 * tile_row_size, 128);
            src3_sz_per_thread = hex_round_up(2 * tile_row_size, 128);
        }
    } else {
        best_n_prefetch = 16;
        src0_sz_per_thread = hex_round_up((uint32_t)(best_n_prefetch * src0_row_size_padded), 128);
        src2_sz_per_thread = hex_round_up((uint32_t)(best_n_prefetch * src0_row_size_padded), 128);
        src3_sz_per_thread = hex_round_up((uint32_t)(best_n_prefetch * src0_row_size_padded), 128);
    }

    size_t src1_sz_per_thread = hex_round_up((uint32_t)(src1_row_size * src1_nrows), 128);

    size_t src0_sz = src0_sz_per_thread * (uint32_t)ctx->n_threads;
    size_t src1_sz = src1_sz_per_thread;
    size_t src2_sz = src2_sz_per_thread * (uint32_t)ctx->n_threads;
    size_t src3_sz = src3_sz_per_thread * (uint32_t)ctx->n_threads;

    size_t tiled_vtcm_size = src0_sz + src1_sz + src2_sz + src3_sz + quant_scratch_size;
    bool try_tiled = (g_hexagon_appcfg.mulmat_algotype >= 29);
    if (try_tiled && tiled_vtcm_size <= vtcm_budget) {
        kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW;
        kparams->vtcm_src0_size = (int32_t) src0_sz;
        kparams->vtcm_src1_size = (int32_t) src1_sz;
        kparams->vtcm_src2_size = (int32_t) src2_sz;
        kparams->vtcm_src3_size = (int32_t) src3_sz;
        kparams->vtcm_dst_size  = (int32_t) quant_scratch_size;
        kparams->vtcm_size      = (int32_t) tiled_vtcm_size;
        kparams->n_prefetch     = (int32_t) best_n_prefetch;
    } else {
        kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT;
        size_t flat_src1_row_size = (wtype == GGML_TYPE_Q4_1) ? htp_mm_q8_1_flat_row_size(ne10) : htp_mm_q8_0_flat_row_size(ne10);
        size_t flat_src1_sz = hex_round_up((uint32_t)(flat_src1_row_size * src1_nrows), 128);
        kparams->vtcm_src0_size = (int32_t) src0_sz;
        kparams->vtcm_src1_size = (int32_t) flat_src1_sz;
        kparams->vtcm_src2_size = (int32_t) src2_sz;
        kparams->vtcm_src3_size = (int32_t) src3_sz;
        kparams->vtcm_dst_size  = (int32_t) quant_scratch_size;
        kparams->vtcm_size      = (int32_t)(src0_sz + flat_src1_sz + src2_sz + src3_sz + quant_scratch_size);
        kparams->n_prefetch     = (int32_t) best_n_prefetch;
    }
}

// Precompute htp_mm_kernel_params for fused FFN matmul (2 outputs: gate, up).
// Mirrors Qualcomm's ggml-hexagon:ggml_hexagon_precompute_fused_ffn_params.
// src0 = Wgate, src1 = y (shared activation).
// DSP-side op_matmul_ffn expects src[0]=Wgate, src[1]=y, src[2]=Wup.
static void ggml_hexagon_precompute_fused_ffn_params(
    const ggml_backend_hexagon_context * ctx,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const int wtype = src0->type;
    const bool is_repack = ggml_hexagon_is_repack_type((ggml_type) wtype);

    const int ne10 = src1->ne[0];
    const int src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
    const size_t src1_row_size = (wtype == GGML_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
    const size_t src0_row_size = src0->nb[1];
    const size_t src0_row_size_padded = hex_round_up((uint32_t) src0_row_size, 128);

    size_t src0_sz_per_thread = 0;
    size_t src2_sz_per_thread = 0;
    uint32_t best_n_prefetch = 16;

    const size_t vtcm_budget = ctx->socinfo.vtcm_size_in_mb * 1024 * 1024;
    size_t quant_scratch_size = hex_round_up((uint32_t)(ne10 * sizeof(float)), QK_Q8_0_TILED * sizeof(float)) * (uint32_t)ctx->n_threads;

    if (is_repack) {
        uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size(wtype);
        uint32_t n_k_tiles = hex_round_up((uint32_t) ne10, 32) / 32;
        uint32_t tile_row_size = n_k_tiles * aligned_tile_size;
        size_t src1_sz_per_thread = hex_round_up((uint32_t)(src1_row_size * src1_nrows), 128);
        size_t src1_sz = src1_sz_per_thread;

        const uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
        best_n_prefetch = 2;
        for (uint32_t d = max_prefetch; d >= 2; d /= 2) {
            size_t repacked_vtcm_size = hex_round_up(d * tile_row_size, 128);
            size_t src0_sz = repacked_vtcm_size * (uint32_t)ctx->n_threads;
            size_t src2_sz = hex_round_up(d * tile_row_size, 128) * (uint32_t)ctx->n_threads;
            size_t tiled_vtcm_size = src0_sz + src1_sz + src2_sz + quant_scratch_size;

            if (tiled_vtcm_size <= vtcm_budget) {
                best_n_prefetch = d;
                src0_sz_per_thread = repacked_vtcm_size;
                src2_sz_per_thread = hex_round_up(d * tile_row_size, 128);
                break;
            }
        }
        if (best_n_prefetch == 2 && src0_sz_per_thread == 0) {
            size_t repacked_vtcm_size = hex_round_up(2 * tile_row_size, 128);
            src0_sz_per_thread = repacked_vtcm_size;
            src2_sz_per_thread = hex_round_up(2 * tile_row_size, 128);
        }
    } else {
        best_n_prefetch = 16;
        src0_sz_per_thread = hex_round_up((uint32_t)(best_n_prefetch * src0_row_size_padded), 128);
        src2_sz_per_thread = hex_round_up((uint32_t)(best_n_prefetch * src0_row_size_padded), 128);
    }

    size_t src1_sz_per_thread = hex_round_up((uint32_t)(src1_row_size * src1_nrows), 128);

    size_t src0_sz = src0_sz_per_thread * (uint32_t)ctx->n_threads;
    size_t src1_sz = src1_sz_per_thread;
    size_t src2_sz = src2_sz_per_thread * (uint32_t)ctx->n_threads;

    size_t tiled_vtcm_size = src0_sz + src1_sz + src2_sz + quant_scratch_size;
    bool try_tiled = (g_hexagon_appcfg.mulmat_algotype >= 29);
    if (try_tiled && tiled_vtcm_size <= vtcm_budget) {
        kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW;
        kparams->vtcm_src0_size = (int32_t) src0_sz;
        kparams->vtcm_src1_size = (int32_t) src1_sz;
        kparams->vtcm_src2_size = (int32_t) src2_sz;
        kparams->vtcm_dst_size  = (int32_t) quant_scratch_size;
        kparams->vtcm_size      = (int32_t) tiled_vtcm_size;
        kparams->n_prefetch     = (int32_t) best_n_prefetch;
    } else {
        kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT;
        size_t flat_src1_row_size = (wtype == GGML_TYPE_Q4_1) ? htp_mm_q8_1_flat_row_size(ne10) : htp_mm_q8_0_flat_row_size(ne10);
        size_t flat_src1_sz = hex_round_up((uint32_t)(flat_src1_row_size * src1_nrows), 128);
        kparams->vtcm_src0_size = (int32_t) src0_sz;
        kparams->vtcm_src1_size = (int32_t) flat_src1_sz;
        kparams->vtcm_src2_size = (int32_t) src2_sz;
        kparams->vtcm_dst_size  = (int32_t) quant_scratch_size;
        kparams->vtcm_size      = (int32_t)(src0_sz + flat_src1_sz + src2_sz + quant_scratch_size);
        kparams->n_prefetch     = (int32_t) best_n_prefetch;
    }
}

// Precompute htp_mm_kernel_params on AP side for MUL_MAT in ION batch path.
// Mirrors build_mm_kernel_params in kernels/entry.c (F32/F16 HVX paths only).
// Writes directly to op.kernel_params; DSP side consumes via memcpy.
// For unsupported weight types (quant/HMX), leaves kernel_type=0 so DSP falls
// back to build_mm_kernel_params which emits the error.
// When is_matmul_id=false, the node is a plain MUL_MAT (not MUL_MAT_ID).
// =======================================================================
// Ported from Qualcomm's ggml-hexagon:
//   ggml_hexagon_precompute_hmx_mm_params
//   ggml_hexagon_precompute_hvx_mm_params
//   ggml_hexagon_precompute_matmul_params
//
// The HMX-first-then-HVX-fallback policy is preserved.
// =======================================================================
static bool ggml_hexagon_precompute_hmx_mm_params(
    const ggml_backend_hexagon_context * ctx,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    int wtype,
    int ne00_padded,
    int ne01_padded,
    int ne02,
    int ne11,
    int ne12,
    int ne11_padded,
    bool is_matmul_id,
    bool is_batched,
    size_t vtcm_budget,
    struct htp_mm_kernel_params * kparams
) {
    const int aligned_tile_size = htp_mm_get_weight_aligned_tile_size(wtype);
    const bool pipeline = is_matmul_id ? false : htp_mm_hmx_pipeline(ne11);
    const int n_threads = (int) ctx->n_threads;
    const int ne10 = src1->ne[0];

    const bool is_batched_val = is_matmul_id ? false : is_batched;
    const int group_size = (ne02 > 0 ? ne12 / ne02 : 1);

    size_t m_chunk = 0;
    size_t n_chunk = 0;
    size_t vtcm_size = 0;
    bool use_grouped = false;
    int act_threads_selected = 0;

    if (is_batched_val && wtype == GGML_TYPE_F16 && group_size > 1) {
        // Try grouped path first
        const bool use_dma_activation = (src1->nb[1]/sizeof(float) > (size_t)ne00_padded);
        size_t best_mblocks = SIZE_MAX;
        int best_act_threads = 0;
        size_t best_m_chunk = 0;
        size_t best_n_chunk = 0;
        size_t best_vtcm_size = 0;

        int act_threads = n_threads;
        while (act_threads >= 1) {
            const size_t f32_scratch_size = use_dma_activation
                ? hex_align_up((size_t)act_threads * HTP_MM_DMA_ACT_MULTIPLIER * ne00_padded * sizeof(float), HTP_MM_HMX_TILE_SIZE)
                : 0;
            size_t group_overhead = 256 + f32_scratch_size;
            size_t group_size_per_n, group_size_per_m, group_size_per_mn;
            htp_mm_hmx_get_batched_chunk_costs(ne00_padded, group_size,
                &group_size_per_n, &group_size_per_m, &group_size_per_mn);

            size_t m_chunk_candidate = 0;
            size_t n_chunk_candidate = 0;
            size_t vtcm_size_candidate = 0;

            if (htp_mm_hmx_compute_chunks(vtcm_budget, group_overhead,
                    group_size_per_n, group_size_per_m, group_size_per_mn,
                    (size_t)hex_align_up(ne11, 32), (size_t)ne01_padded,
                    (size_t) ne01_padded * HTP_MM_HMX_COST_W_DEQUANT,
                    (size_t) ne11 * HTP_MM_HMX_COST_A_CONVERT,
                    &m_chunk_candidate, &n_chunk_candidate,
                    &vtcm_size_candidate) == 0) {
                size_t exact_size = htp_mm_hmx_get_batched_vtcm_size(
                    wtype, ne00_padded, m_chunk_candidate, n_chunk_candidate,
                    group_size, use_dma_activation, pipeline, act_threads);
                if (exact_size <= vtcm_budget) {
                    size_t mblocks = ((size_t) ne11 + m_chunk_candidate - 1) / m_chunk_candidate;
                    if (mblocks < best_mblocks ||
                        (mblocks == best_mblocks && act_threads > best_act_threads)) {
                        best_mblocks = mblocks;
                        best_act_threads = act_threads;
                        best_m_chunk = m_chunk_candidate;
                        best_n_chunk = n_chunk_candidate;
                        best_vtcm_size = exact_size;
                    }
                }
            }
            if (act_threads == 1) {
                act_threads = 0;
            } else {
                act_threads /= 2;
            }
        }

        if (best_act_threads > 0) {
            m_chunk = best_m_chunk;
            n_chunk = best_n_chunk;
            vtcm_size = best_vtcm_size;
            act_threads_selected = best_act_threads;
            use_grouped = true;
        }
    }

    if (!use_grouped) {
        // Fallback to simple 2D path (group_size = 1)
        size_t best_mblocks = SIZE_MAX;
        int best_act_threads = 0;
        size_t best_m_chunk = 0;
        size_t best_n_chunk = 0;
        size_t best_vtcm_size = 0;

        // For MUL_MAT_ID the kernel runs one 2D matmul per expert, with M
        // equal to the number of rows routed to that expert. A single expert
        // can receive up to all routed rows (dst->ne[1]*dst->ne[2]), so size
        // the chunk search for that upper bound rather than ne12 (token
        // positions only). We recompute m_chunk per expert against the actual
        // count in the NPU kernel.
        const int m_id_rows    = (int) ((size_t) dst->ne[1] * dst->ne[2]);
        const int m_for_chunks = is_matmul_id ? hex_align_up(m_id_rows, 32) : ne11_padded;
        const int m_for_cost   = is_matmul_id ? m_id_rows : ne11;

        int act_threads = n_threads;
        while (act_threads >= 1) {
            const size_t act_f32_size = is_matmul_id
                ? 0
                : hex_align_up((size_t)act_threads * HTP_MM_DMA_ACT_MULTIPLIER * ne00_padded * sizeof(float), HTP_MM_HMX_TILE_SIZE);
            size_t simple_2d_overhead = 256 + act_f32_size;
            size_t simple_2d_size_per_n, simple_2d_size_per_m, simple_2d_size_per_mn;
            htp_mm_hmx_get_2d_chunk_costs(wtype, ne00_padded, pipeline,
                aligned_tile_size, &simple_2d_size_per_n,
                &simple_2d_size_per_m, &simple_2d_size_per_mn);

            size_t m_chunk_candidate = 0;
            size_t n_chunk_candidate = 0;
            size_t vtcm_size_candidate = 0;

            if (htp_mm_hmx_compute_chunks(vtcm_budget, simple_2d_overhead,
                    simple_2d_size_per_n, simple_2d_size_per_m,
                    simple_2d_size_per_mn, (size_t)m_for_chunks,
                    (size_t)ne01_padded,
                    (size_t) ne01_padded * HTP_MM_HMX_COST_W_DEQUANT,
                    (size_t) m_for_cost * HTP_MM_HMX_COST_A_CONVERT,
                    &m_chunk_candidate, &n_chunk_candidate,
                    &vtcm_size_candidate) == 0) {
                size_t exact_size = htp_mm_hmx_get_2d_vtcm_size(
                    wtype, ne00_padded, m_chunk_candidate, n_chunk_candidate,
                    pipeline, is_matmul_id ? 0 : act_threads, aligned_tile_size);
                if (exact_size <= vtcm_budget) {
                    size_t mblocks = ((size_t) m_for_cost + m_chunk_candidate - 1) / m_chunk_candidate;
                    if (mblocks < best_mblocks ||
                        (mblocks == best_mblocks && act_threads > best_act_threads)) {
                        best_mblocks = mblocks;
                        best_act_threads = act_threads;
                        best_m_chunk = m_chunk_candidate;
                        best_n_chunk = n_chunk_candidate;
                        best_vtcm_size = exact_size;
                    }
                }
            }
            if (act_threads == 1) {
                act_threads = 0;
            } else {
                act_threads /= 2;
            }
        }

        if (best_act_threads > 0) {
            m_chunk = best_m_chunk;
            n_chunk = best_n_chunk;
            vtcm_size = best_vtcm_size;
            act_threads_selected = best_act_threads;
        } else {
            return false;
        }
    }

    kparams->n_hmx = 1;
    kparams->pipeline = pipeline ? 1 : 0;
    kparams->m_chunk = (int32_t) m_chunk;
    kparams->n_chunk = (int32_t) n_chunk;
    kparams->n_threads = n_threads;
    kparams->n_act_threads = act_threads_selected;
    kparams->tile_size = (int32_t) htp_mm_get_weight_tile_size(wtype);
    kparams->aligned_tile_size = (int32_t) aligned_tile_size;
    kparams->src1_row_size = (int32_t)((wtype == GGML_TYPE_Q4_1)
        ? htp_mm_q8_1_tiled_row_size(ne10)
        : htp_mm_q8_0_tiled_row_size(ne10));
    kparams->vtcm_size = (int32_t) vtcm_size;
    kparams->vtcm_src0_size = 0;
    kparams->vtcm_src1_size = 0;
    kparams->vtcm_dst_size = 0;

    if (is_batched && !is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_F16_BATCHED;
    } else {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_2D;
    }
    return true;
}

static void ggml_hexagon_precompute_hvx_mm_params(
    const ggml_backend_hexagon_context * ctx,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    int wtype,
    int ne02,
    int ne03,
    int ne10,
    int ne11,
    int ne12,
    int ne13,
    bool is_matmul_id,
    size_t vtcm_budget,
    struct htp_mm_kernel_params * kparams
) {
    kparams->n_hmx = 0;

    const bool is_quant = (wtype != GGML_TYPE_F16 && wtype != GGML_TYPE_F32);
    const int src1_nrows = ne11 * ne12 * ne13;

    // opt_mm_select equivalent: algotype 29 means HMX-enabled (select >= 3),
    // algotype 29/32 both mean tiled HVX enabled (select >= 2)
    const bool opt_mm_tiled = (g_hexagon_appcfg.mulmat_algotype == 29 ||
                               g_hexagon_appcfg.mulmat_algotype == 32);

    if (is_quant) {
        // Quantized HVX
        kparams->tile_size = (int32_t) htp_mm_get_weight_tile_size(wtype);
        kparams->aligned_tile_size = (int32_t) htp_mm_get_weight_aligned_tile_size(wtype);

        const bool k_align = (ne10 % 32 == 0);

        if (is_matmul_id) {
            kparams->kernel_type   = (src1_nrows < (int) ctx->n_threads)
                ? HTP_MM_KERNEL_HVX_QUANT_BLOCK : HTP_MM_KERNEL_HVX_QUANT_ROW;
            kparams->src1_row_size = (int32_t)((wtype == GGML_TYPE_Q4_1)
                ? htp_mm_q8_1_tiled_row_size(ne10)
                : htp_mm_q8_0_tiled_row_size(ne10));

            size_t vtcm_src0_size = 0, vtcm_src1_size = 0, vtcm_dst_size = 0;
            uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
            uint32_t best_n_prefetch = 2;
            size_t total_size = 0;
            for (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                total_size = htp_mm_hvx_id_get_vtcm_sizes(
                    wtype, ne10, src1_nrows, (uint32_t)ctx->n_threads,
                    src0->nb[1], d,
                    &vtcm_src0_size, &vtcm_src1_size, &vtcm_dst_size
                );
                if (total_size <= vtcm_budget) {
                    best_n_prefetch = d;
                    break;
                }
            }
            if (best_n_prefetch == 2 && total_size > vtcm_budget) {
                total_size = htp_mm_hvx_id_get_vtcm_sizes(
                    wtype, ne10, src1_nrows, (uint32_t)ctx->n_threads,
                    src0->nb[1], 2,
                    &vtcm_src0_size, &vtcm_src1_size, &vtcm_dst_size
                );
            }
            kparams->n_prefetch = (int32_t) best_n_prefetch;
            kparams->vtcm_size      = (int32_t) total_size;
            kparams->vtcm_src0_size = (int32_t) vtcm_src0_size;
            kparams->vtcm_src1_size = (int32_t) vtcm_src1_size;
            kparams->vtcm_dst_size  = (int32_t) vtcm_dst_size;
        } else {
            bool try_tiled = (k_align && opt_mm_tiled);
            if (try_tiled) {
                kparams->src1_row_size = (int32_t)((wtype == GGML_TYPE_Q4_1)
                    ? htp_mm_q8_1_tiled_row_size(ne10)
                    : htp_mm_q8_0_tiled_row_size(ne10));
                if (src1_nrows < (int)ctx->n_threads) {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_BLOCK;
                } else {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW;
                }

                uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
                uint32_t best_n_prefetch = 2;
                size_t vtcm_src0_size = 0, vtcm_src1_size = 0, vtcm_dst_size = 0;
                size_t total_size = 0;
                for (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                    total_size = htp_mm_hvx_get_vtcm_sizes(
                        kparams->kernel_type, wtype, ne10, src1_nrows,
                        (uint32_t)ctx->n_threads,
                        dst->nb[1], src0->nb[1], src1->nb[1], d,
                        &vtcm_src0_size, &vtcm_src1_size, &vtcm_dst_size
                    );
                    if (total_size <= vtcm_budget) {
                        best_n_prefetch = d;
                        break;
                    }
                }
                if (best_n_prefetch == 2 && total_size > vtcm_budget) {
                    total_size = htp_mm_hvx_get_vtcm_sizes(
                        kparams->kernel_type, wtype, ne10, src1_nrows,
                        (uint32_t)ctx->n_threads,
                        dst->nb[1], src0->nb[1], src1->nb[1], 2,
                        &vtcm_src0_size, &vtcm_src1_size, &vtcm_dst_size
                    );
                }

                kparams->n_prefetch = (int32_t) best_n_prefetch;

                if (total_size <= vtcm_budget) {
                    kparams->vtcm_size = (int32_t) total_size;
                    kparams->vtcm_src0_size = (int32_t) vtcm_src0_size;
                    kparams->vtcm_src1_size = (int32_t) vtcm_src1_size;
                    kparams->vtcm_dst_size = (int32_t) vtcm_dst_size;
                    goto done_quant;
                }
                GGMLHEXAGON_LOG_DEBUG("precompute_hvx: tiled path VTCM too large "
                    "(need=%zu budget=%zu), falling back to flat",
                    total_size, vtcm_budget);
            }

            // Flat HVX fallback
            {
                kparams->src1_row_size = (int32_t)((wtype == GGML_TYPE_Q4_1)
                    ? htp_mm_q8_1_flat_row_size(ne10)
                    : htp_mm_q8_0_flat_row_size(ne10));
                kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT;

                size_t vtcm_src0_size = 0, vtcm_src1_size = 0, vtcm_dst_size = 0;
                size_t total_size = htp_mm_hvx_get_vtcm_sizes(
                    kparams->kernel_type, wtype, ne10, src1_nrows,
                    (uint32_t)ctx->n_threads,
                    dst->nb[1], src0->nb[1], src1->nb[1], 16,
                    &vtcm_src0_size, &vtcm_src1_size, &vtcm_dst_size
                );

                kparams->n_prefetch = 16;
                kparams->vtcm_size = (int32_t) total_size;
                kparams->vtcm_src0_size = (int32_t) vtcm_src0_size;
                kparams->vtcm_src1_size = (int32_t) vtcm_src1_size;
                kparams->vtcm_dst_size = (int32_t) vtcm_dst_size;
            }
        }

    done_quant:;
    } else if (wtype == GGML_TYPE_F16) {
        // F16 HVX
        const bool is_batched  = (ne02 > 1) || (ne03 > 1);
        const bool is_permuted = ggml_is_permuted(src0) || ggml_is_permuted(src1);

        size_t vtcm_src0_size = 0, vtcm_src1_size = 0, vtcm_dst_size = 0;
        size_t vtcm_size = htp_mm_hvx_get_vtcm_sizes(
            HTP_MM_KERNEL_HVX_F16_F16_VTCM, wtype, ne10, src1_nrows,
            (uint32_t)ctx->n_threads,
            dst->nb[1], src0->nb[1], src1->nb[1], 16,
            &vtcm_src0_size, &vtcm_src1_size, &vtcm_dst_size
        );

        if (!is_batched && !is_permuted && vtcm_size <= vtcm_budget) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F16_F16_VTCM;
            kparams->src1_row_size = (int32_t) hex_round_up(ne10 * 2, 128);
            kparams->vtcm_size = (int32_t) vtcm_size;
            kparams->vtcm_src0_size = (int32_t) vtcm_src0_size;
            kparams->vtcm_src1_size = (int32_t) vtcm_src1_size;
            kparams->vtcm_dst_size = (int32_t) vtcm_dst_size;
            kparams->n_prefetch = 16;
        } else {
            if (src1->type == GGML_TYPE_F32) {
                kparams->kernel_type = HTP_MM_KERNEL_HVX_F16_F32_DDR;
            } else {
                kparams->kernel_type = HTP_MM_KERNEL_HVX_F16_F16_DDR;
            }
            kparams->src1_row_size = (int32_t) src1->nb[1];
            size_t ddr_size = htp_mm_hvx_get_vtcm_sizes(
                kparams->kernel_type, wtype, ne10, src1_nrows,
                (uint32_t)ctx->n_threads,
                dst->nb[1], src0->nb[1], src1->nb[1], 16,
                &vtcm_src0_size, &vtcm_src1_size, &vtcm_dst_size
            );
            kparams->vtcm_size = (int32_t) ddr_size;
            kparams->vtcm_src0_size = (int32_t) vtcm_src0_size;
            kparams->vtcm_src1_size = (int32_t) vtcm_src1_size;
            kparams->vtcm_dst_size = (int32_t) vtcm_dst_size;
            kparams->n_prefetch = 16;
        }
    } else {
        // F32 HVX
        const bool is_batched  = (ne02 > 1) || (ne03 > 1);
        const bool is_permuted = ggml_is_permuted(src0) || ggml_is_permuted(src1);

        size_t vtcm_src0_size = 0, vtcm_src1_size = 0, vtcm_dst_size = 0;
        size_t vtcm_size = htp_mm_hvx_get_vtcm_sizes(
            HTP_MM_KERNEL_HVX_F32_F32_VTCM, wtype, ne10, src1_nrows,
            (uint32_t)ctx->n_threads,
            dst->nb[1], src0->nb[1], src1->nb[1], 16,
            &vtcm_src0_size, &vtcm_src1_size, &vtcm_dst_size
        );

        if (!is_batched && !is_permuted && vtcm_size <= vtcm_budget) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F32_F32_VTCM;
            kparams->src1_row_size = (int32_t) hex_round_up(ne10 * 4, 128);
            kparams->vtcm_size = (int32_t) vtcm_size;
            kparams->vtcm_src0_size = (int32_t) vtcm_src0_size;
            kparams->vtcm_src1_size = (int32_t) vtcm_src1_size;
            kparams->vtcm_dst_size = (int32_t) vtcm_dst_size;
            kparams->n_prefetch = 16;
        } else {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F32_F32_DDR;
            kparams->src1_row_size = (int32_t) src1->nb[1];
            size_t ddr_size = htp_mm_hvx_get_vtcm_sizes(
                kparams->kernel_type, wtype, ne10, src1_nrows,
                (uint32_t)ctx->n_threads,
                dst->nb[1], src0->nb[1], src1->nb[1], 16,
                &vtcm_src0_size, &vtcm_src1_size, &vtcm_dst_size
            );
            kparams->vtcm_size = (int32_t) ddr_size;
            kparams->vtcm_src0_size = (int32_t) vtcm_src0_size;
            kparams->vtcm_src1_size = (int32_t) vtcm_src1_size;
            kparams->vtcm_dst_size = (int32_t) vtcm_dst_size;
            kparams->n_prefetch = 16;
        }
    }
}

static void ggml_hexagon_precompute_mm_params(
    ggml_backend_hexagon_context * ctx,
    const ggml_tensor * node,
    hex_op_desc & op,
    bool is_matmul_id
) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    const ggml_tensor * dst  = node;

    struct htp_mm_kernel_params * kparams =
        (struct htp_mm_kernel_params *) op.kernel_params;
    memset(kparams, 0, sizeof(*kparams));

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    const int wtype = src0->type;
    const bool is_repack = ggml_hexagon_is_repack_type((ggml_type) wtype);
    const int ne00_padded = is_repack ? (int) hex_round_up((uint32_t) ne00, 32) : ne00;
    const int ne01_padded = is_repack ? (int) hex_round_up((uint32_t) ne01, 32) : ne01;
    const int ne11_padded = (int) hex_round_up((uint32_t) ne11, 32);

    const bool is_batched   = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    const size_t vtcm_budget = (size_t)ctx->socinfo.vtcm_size_in_mb * 1024 * 1024;

    // Cache key: weight data pointer (identifies the weight tensor) + ne11
    // (varies for PP, fixed for TG). is_batched is derived from ne02/ne12
    // and is the same for the same weight + same ne11, so it is captured
    // implicitly in the cached entry.
    const uintptr_t cache_key = (uintptr_t) src0->data ^ ((uintptr_t) ne11 << 32);
    auto it = ctx->mm_params_cache.find(cache_key);
    if (it != ctx->mm_params_cache.end()) {
        *kparams = it->second;
        return;
    }

    // HMX-first policy: try HMX precomputation if eligible, fall back to HVX.
    bool hmx_enabled = ctx->has_hmx && (g_hexagon_appcfg.mulmat_algotype == 29);
    if (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(
            src0, src1, dst, ne01_padded, is_matmul_id, is_batched)) {
        if (ggml_hexagon_precompute_hmx_mm_params(
                ctx, src0, src1, dst, wtype, ne00_padded, ne01_padded,
                ne02, ne11, ne12, ne11_padded, is_matmul_id, is_batched,
                vtcm_budget, kparams)) {
            goto finalize;
        }
    }

    // Fallback to HVX parameter computation
    ggml_hexagon_precompute_hvx_mm_params(
        ctx, src0, src1, dst, wtype,
        ne02, ne03, ne10, ne11, ne12, ne13,
        is_matmul_id, vtcm_budget, kparams);

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_values((uint32_t)(ne12 * ne11));
    kparams->div_ne1      = init_fastdiv_values((uint32_t) ne11);
    kparams->div_r2       = init_fastdiv_values(ne02 > 0 ? (uint32_t)(ne12 / ne02) : 1);
    kparams->div_r3       = init_fastdiv_values(ne03 > 0 ? (uint32_t)(ne13 / ne03) : 1);
    kparams->div_ne11     = init_fastdiv_values((uint32_t) ne11);

    // Intended for TG hot path (skip precompute on repeated calls). Disabled:
    // cache key (src0->data ^ ne11) collides on ION region reuse, returning
    // stale params. Observed: qwen3 ubatch=64 produced coherent but wrong text
    // (not caught by <unused> check). Trade-off: TG re-runs precompute/token.
    // TODO: include ggml_tensor* in key, or track gen counter on ION region.
    // ctx->mm_params_cache[cache_key] = *kparams;
}

// =================================================================================================
//  section-8: backend implementation
// =================================================================================================
ggml_backend_hexagon_context::ggml_backend_hexagon_context(int dev_id, ggml_backend_dev_t dev)
    : device(dev_id),
      backend(nullptr),
      socinfo{},
      n_threads(6),
      rpc_mempool_capacity(0),
      rpc_mempool_len(0),
      rpc_mempool_usage(0),
      rpc_mempool_cache_offset(0),
      rpc_mempool_cache_budget(0),
      rpc_mempool_cache_usage_est(0),
      weights_dirty(false),
      rpc_mempool(nullptr),
      rpc_mempool_handle(0),
      rpc_mempool_dsp_base(nullptr),
      ggmlop_handle(0),
      domain_id(CDSP_DOMAIN_ID),
      session_id(0),
      rpc_batch_call_count(0),
      cumulative_p7_us(0),
      cumulative_graph_us(0),
      last_graph_end_us(0),
      max_nodes_per_graph(0),
      min_nodes_per_graph(0),
      total_nodes_processed(0),
      min_graph_us(0),
      max_graph_us(0),
      max_graph_n_nodes(0),
      max_graph_n_ops(0),
      min_p7_us(0),
      max_p7_us(0),
      cum_p4_us(0),
      cum_p45_us(0),
      cum_p6_us(0),
      cum_p65_us(0),
      cum_p75_us(0),
      cum_p8_us(0),
      perf_hist_count(0),
      perf_hist_idx(0),
      cum_p7_rpc_setup_us(0),
      cum_p7_dsp_exec_us(0),
      cum_p7_civac_us(0),
      rpc_overhead_min_us(0),
      rpc_overhead_max_us(0),
      rpc_overhead_sum_us(0),
      rpc_overhead_count(0),
      buffer_type{},
      has_vtcm(false),
      has_hvx(false),
      has_hmx(false) {
    snprintf(name, sizeof(name), "Hexagon-cDSP%d", dev_id);
    snprintf(desc, sizeof(desc), "Qualcomm NPU(CDSP%d)", dev_id);
    snprintf(buft_name, sizeof(buft_name), "hexagon-ion-buffer-%s", name);
    snprintf(repack_buft_name, sizeof(repack_buft_name), "hexagon-ion-buffer-%s-REPACK", name);
    lib[0] = '\0';

    buffer_type.iface.get_name         = ggml_backend_hexagon_buffer_type_name;
    buffer_type.iface.alloc_buffer     = ggml_backend_hexagon_buffer_type_alloc_buffer;
    buffer_type.iface.get_alignment    = ggml_backend_hexagon_buffer_type_get_alignment;
    buffer_type.iface.get_max_size     = ggml_backend_hexagon_buffer_type_get_max_size;
    buffer_type.iface.get_alloc_size   = ggml_backend_hexagon_buffer_type_get_alloc_size;
    buffer_type.iface.is_host          = ggml_backend_hexagon_buffer_is_host;
    buffer_type.device  = dev;
    buffer_type.context = this;

    // Repack buffer type: same ION pool as buffer_type, but is_host=false
    repack_buffer_type.iface.get_name         = ggml_backend_hexagon_buffer_type_name;
    repack_buffer_type.iface.alloc_buffer     = ggml_backend_hexagon_buffer_type_alloc_buffer;
    repack_buffer_type.iface.get_alignment    = ggml_backend_hexagon_buffer_type_get_alignment;
    repack_buffer_type.iface.get_max_size     = ggml_backend_hexagon_buffer_type_get_max_size;
    repack_buffer_type.iface.get_alloc_size   = ggml_backend_hexagon_buffer_type_get_alloc_size;
    repack_buffer_type.iface.is_host          = ggml_backend_hexagon_repack_buffer_is_host;
    repack_buffer_type.device  = dev;
    repack_buffer_type.context = this;

    int result = ggmlhexagon_init_dsp(this);
    if (0 != result) {
        GGMLHEXAGON_LOG_ERROR("init hexagon dsp failure for device %d", dev_id);
    }
}

ggml_backend_hexagon_context::~ggml_backend_hexagon_context() {
    ggmlhexagon_deinit_cdsp(this);
    ggmlhexagon_print_running_timestamp(NULL);
}

static bool ggmlhexagon_compute_forward(ggml_backend_hexagon_context * ctx, struct ggml_tensor * op) {
    int hexagon_error               = AEE_SUCCESS;

    ggml_tensor * src0  = op->src[0];
    ggml_tensor * src1  = op->src[1];
    ggml_tensor * dst   = op;

    if (ggmlhexagon_k_op_caps[ggmlhexagon_get_op_index(op)].supported) {
        struct ggmlhexagon_task task;

        std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
        ggmlhexagon_task_init(&task);

        int ret = ggmlhexagon_task_add_op(&task, op->op, src0, src1, dst);
        if (ret != 0) {
            GGMLHEXAGON_LOG_WARN("failed to add op to task");
            return false;
        }
        std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<size_t, std::nano> duration = end_time - start_time;
        GGMLHEXAGON_LOG_DEBUG("pack duration %llu ns", duration.count());

        hexagon_error = ggmlhexagon_task_execute(ctx, &task);
        if (AEE_SUCCESS != hexagon_error) {
            GGMLHEXAGON_LOG_WARN("ggmlop %s computation fail on CDSP via dsp task", ggml_op_name(op->op));
        }
    } else {
        GGMLHEXAGON_LOG_DEBUG("op GGML_OP_%s not supported on CDSP", ggml_op_name(op->op));
        return false;
    }

    return true;
}

// Estimate the FP16 weight cache size that the DSP side would allocate
// for a quantized weight tensor. Mirrors the calculation in mulmat.c:
//   total_fp16_size = M_padded * K * sizeof(fp16)
// where M_padded is M rounded up to 32 (HMX_FP16_TILE_N_COLS).
// Returns 0 for non-quantized types.
static size_t ggmlhexagon_estimate_fp16_cache_size(const struct ggml_tensor * src0) {
    if (!ggml_is_quantized(src0->type)) {
        return 0;
    }
    const int64_t M = src0->ne[1];
    const int64_t K = src0->ne[0];
    if (M <= 0 || K <= 0) {
        return 0;
    }
    // Round M up to 32 (HMX_FP16_TILE_N_COLS) to match DSP-side padding
    const size_t M_padded = ((size_t)M + 31) / 32 * 32;
    return M_padded * (size_t)K * 2;  // 2 bytes per fp16 element
}

// Ref: function ggml_hexagon_supported_mul_mat in Qualcomm's ggml-hexagon
static bool ggmlhexagon_supported_mul_mat(const struct ggml_tensor * dst,
                                          ggml_backend_hexagon_context * ctx) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

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

    if (g_hexagon_appcfg.mulmat_algotype != 29) { // In algotype=29 path, bypass mulmat_min_n to reduce batch calls.
        if (g_hexagon_appcfg.mulmat_min_n >= n) {
            GGMLHEXAGON_LOG_DEBUG("MUL_MAT quantized N=%lld <= %d, keep on CPU\n", (long long)n, g_hexagon_appcfg.mulmat_min_n);
            return false;
        }
    }

    // GEMV offload debugging: when gemv_offload=0, keep N=1 MUL_MAT on CPU
    if (g_hexagon_appcfg.mulmat_algotype == 29 && g_hexagon_appcfg.gemv_offload == 0 && n == 1) {
        GGMLHEXAGON_LOG_DEBUG("MUL_MAT N=1 (GEMV): gemv_offload=0, keep on CPU\n");
        return false;
    }

    switch (src0->type) {
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_0:
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
        {
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

            // Quantized HVX kernels assume src0 is laid out contiguously in
            // row-major (ne[0] is innermost). Non-contiguous views (e.g. k_v
            // slices) cause wrong tile offsets -> silent numeric corruption.
            if (!ggml_is_contiguous(src0)) {
                GGMLHEXAGON_LOG_DEBUG("supported_mul_mat FAIL: src0 not contiguous (nb=[%lld,%lld,%lld,%lld] ne=[%lld,%lld,%lld,%lld])",
                                        (long long)src0->nb[0], (long long)src0->nb[1], (long long)src0->nb[2], (long long)src0->nb[3],
                                        (long long)src0->ne[0], (long long)src0->ne[1], (long long)src0->ne[2], (long long)src0->ne[3]);
                return false;
            }

            break;
        }

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
                //return false;
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


    if (g_hexagon_appcfg.mulmat_algotype == 29) {  // Qualcomm's mulmat
        // Precompute kernel params to get the actual VTCM size
        hex_op_desc tmp_op;
        memset(&tmp_op, 0, sizeof(tmp_op));
        tmp_op.opcode = dst->op;
        bool is_matmul_id = (dst->op == GGML_OP_MUL_MAT_ID);
        ggml_hexagon_precompute_mm_params(ctx, dst, tmp_op, is_matmul_id);
        const struct htp_mm_kernel_params * kparams =
            (const struct htp_mm_kernel_params *)tmp_op.kernel_params;

        const size_t vtcm_budget = (size_t)ctx->socinfo.vtcm_size_in_mb * 1024 * 1024;
        if ((size_t)kparams->vtcm_size > vtcm_budget) {
            GGMLHEXAGON_LOG_ALWAYS("MUL_MAT VTCM too small: needed=%d budget=%zu\n",
                                   kparams->vtcm_size, vtcm_budget);
            return false;
        }
    } else { //JZ's mulmat
        const size_t vtcm_budget = (size_t)ctx->socinfo.vtcm_size_in_mb * 1024 * 1024;
        const size_t src0_bytes  = (src0->type == GGML_TYPE_Q8_0)
                                      ? (size_t)m * (size_t)k
                                      : (size_t)m * (size_t)k / 2;
        const size_t src1_est    = (size_t)k * (size_t)n * 16;
        if (src0_bytes + src1_est >= vtcm_budget) {
            GGMLHEXAGON_LOG_ALWAYS("MUL_MAT quantized VTCM too small: src0=%zu + src1_est=%zu, budget=%zu\n", src0_bytes, src1_est, vtcm_budget);
            return false;
        }

        // algotype=32: track estimated FP16 weight cache usage to avoid DSP-side
        // ION cache overflow. precompute_mm_params gives more precise (smaller)
        // vtcm_size, which allows more MUL_MAT ops to pass VTCM check. But the
        // DSP-side ggmlop_cache_mempool_alloc allocates FP16 cache for each
        // offloaded quantized weight, and the cache region has limited capacity.
        // Reject ops whose cumulative cache usage would exceed the budget.
        if (g_hexagon_appcfg.mulmat_algotype == 32 && ctx->rpc_mempool_cache_budget > 0
            && ggml_is_quantized(src0->type)) {
            const size_t fp16_est = ggmlhexagon_estimate_fp16_cache_size(src0);
            // Only track weights above the DSP-side minimum cache size threshold (18 MB)
            const size_t FP16_CACHE_MIN = (size_t)18 * 1024 * 1024;
            if (fp16_est >= FP16_CACHE_MIN) {
                // Deduplicate: same weight data pointer may appear in multiple MUL_MAT ops
                if (ctx->rpc_mempool_cache_tracked.find(src0->data) ==
                    ctx->rpc_mempool_cache_tracked.end()) {
                    if (ctx->rpc_mempool_cache_usage_est + fp16_est > ctx->rpc_mempool_cache_budget) {
                        GGMLHEXAGON_LOG_ALWAYS("MUL_MAT ION cache full: fp16_est=%zuMB "
                                               "cumulative=%zuMB budget=%zuMB, keeping on CPU\n",
                                               fp16_est / (1024 * 1024),
                                               ctx->rpc_mempool_cache_usage_est / (1024 * 1024),
                                               ctx->rpc_mempool_cache_budget / (1024 * 1024));
                        return false;
                    }
                    ctx->rpc_mempool_cache_tracked.insert(src0->data);
                    ctx->rpc_mempool_cache_usage_est += fp16_est;
                }
            }
        }
    }

    return true;
}

// Decide whether a FLASH_ATTN_EXT node can be offloaded to the DSP.
// Ported from Qualcomm's ggml_hexagon_supported_flash_attn_ext:
// type/shape checks plus a precompute pass that verifies the selected kernel
// (HMX or HVX) fits the per-domain VTCM budget.
static bool ggmlhexagon_supported_flash_attn(
    const ggml_backend_hexagon_context * ctx, const struct ggml_tensor * dst) {
    const struct ggml_tensor * q     = dst->src[0];
    const struct ggml_tensor * k     = dst->src[1];
    const struct ggml_tensor * v     = dst->src[2];
    const struct ggml_tensor * mask  = dst->src[3];
    const struct ggml_tensor * sinks = dst->src[4];

    if (!q || !k || !v) {
        return false;
    }
    if ((q->type != GGML_TYPE_F16 && q->type != GGML_TYPE_F32) ||
        k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) {
        return false;
    }
    if (mask && mask->type != GGML_TYPE_F16) {
        return false;
    }
    if (sinks && sinks->type != GGML_TYPE_F32) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        return false;
    }
    if (dst->ne[3] != 1) {
        return false;
    }

    struct htp_fa_kernel_params kparams;
    if (!ggml_hexagon_compute_fa_params(ctx, dst, &kparams)) {
        return false;
    }
    const size_t vtcm_budget = (size_t)ctx->socinfo.vtcm_size_in_mb * 1024 * 1024;
    if ((size_t)kparams.vtcm_size > vtcm_budget) {
        return false;
    }
    return true;
}

static bool ggmlhexagon_can_handle_op_through_cdsp(ggml_backend_dev_t dev, const struct ggml_tensor * op_tensor) {
    // Session consistency gate (mirrors Qualcomm's ggml_backend_hexagon_device_supports_op):
    // all srcs & dsts of the op must be mapped to the same Hexagon session as
    // this device. Without this check, the scheduler can mis-assign an op to
    // a device whose tensors live in a different ION region, which would fault
    // on the DSP since ION mappings are not shared across separate FastRPC
    // sessions. Metadata-only ops (VIEW/RESHAPE/PERMUTE/...) have no buffers
    // and always pass.
    if (!ggmlhexagon_is_metadata_op(op_tensor->op)) {
        if (!ggmlhexagon_op_buffers_belong_to_dev(dev, op_tensor)) {
            return false;
        }
    }

    if (2 != g_hexagon_appcfg.offload_cgraph_type) { // FastRPC per-op mode
        if (ggmlhexagon_is_metadata_op(op_tensor->op)) {
            return true;
        }
        if (op_tensor->op != GGML_OP_MUL_MAT) { //limited to MUL_MAT in FastRPC per-op mode for debug only
            return false;
        }
        const ggml_tensor * src0 = op_tensor->src[0];
        const ggml_tensor * src1 = op_tensor->src[1];
        const int64_t ne00      = src0->ne[0];
        const int     src0_rank = ggml_n_dims(src0);
        int           src1_rank = ggml_n_dims(src1);
        if (ne00 < 1024) {
            return false;
        }
        if (src0_rank != src1_rank || src0_rank < 2) {
            return false;
        }
        return (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16
                || src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q8_0
                ) && (src1->type == GGML_TYPE_F32) && (op_tensor->type == GGML_TYPE_F32);
    } //end if (2 != g_hexagon_appcfg.offload_cgraph_type) { // FastRPC per-op mode

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
            // src0 (quantized weights that need tile repack) must reside in the
            // repack buft (is_host=false). That buft forces GGML core to route
            // data through set_tensor, which does the in-place tile repack. The
            // main buft (is_host=true) cannot host repacked weights, so the
            // model loader must allocate these quantized weights in the repack
            // buft; this check enforces that. Same gate that Qualcomm uses.
            // Only types handled by ggml_hexagon_is_repack_type are gated; other
            // quantized types (e.g. Q4_K) still use the main buft unchanged.
            const ggml_tensor * src0 = op_tensor->src[0];
            if (ggml_hexagon_is_repack_type(src0->type) && src0->buffer &&
                !ggml_backend_buffer_is_hexagon_repack(src0->buffer)) {
                return false;
            }
            return ggmlhexagon_supported_mul_mat(op_tensor, ctx);
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
        case GGML_OP_NORM:
        case GGML_OP_L2_NORM:
        {
            // Dispatched to op_unary (F32 only, same shape, dst contiguous)
            if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)
                return false;
            if (!ggml_are_same_shape(src0, dst))
                return false;
            if (!ggml_is_contiguous(dst))
                return false;
            return true;
        }
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        {
            // Element-wise unary: dispatch to op_unary (F32 only)
            if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)
                return false;
            if (!ggml_are_same_shape(src0, dst))
                return false;
            if (!ggml_is_contiguous(dst))
                return false;
            return true;
        }
        case GGML_OP_ROPE:
        {
            // op_rope only implements F32 path; src1 is I32 positions (newer GGML API)
            if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)
                return false;
            if (!src1 || src1->type != GGML_TYPE_I32)
                return false;
            // op_rope handles NORMAL(0), NEOX(2), MROPE(8), IMROPE(40);
            // VISION(24) is mishandled (treated as MROPE), keep on CPU
            const int32_t mode = op_tensor->op_params[2];
            if (mode == 24)
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
            // sinks (src2) participate in softmax normalization (max/sum),
            // see ggml_vec_soft_max_f32 in ggml-cpu/ops.cpp. Qualcomm's
            // op_softmax (htp/softmax-ops.c) ignores src2, so for now fall
            // back to CPU when sinks is present.
            if (op_tensor->src[2] != nullptr)
                return false;
            return true;
        }
        case GGML_OP_UNARY:
        {
            // Activations (SILU/GELU/GELU_QUICK) dispatch to op_activations.
            // Other unary ops (NEG/EXP/SIGMOID/SOFTPLUS/TANH) dispatch to op_unary.
            // Both DSP kernels only implement the F32 path.
            const int unary_op = (int)dst->op_params[0];
            switch (unary_op) {
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_QUICK:
                    // op_activations: requires contiguous src0 and dst
                    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)
                        return false;
                    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst))
                        return false;
                    return true;
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_TANH:
                    // op_unary: requires contiguous dst (src0 may be non-contiguous)
                    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)
                        return false;
                    if (!ggml_are_same_shape(src0, dst))
                        return false;
                    if (!ggml_is_contiguous(dst))
                        return false;
                    return true;
                default:
                    return false;
            }
        }
        case GGML_OP_GLU:
        {
            // GLU dispatches to op_activations (F32, contiguous), same as SILU/GELU.
            // op_params[0] selects the GLU sub-op; only SWIGLU/SWIGLU_OAI/GEGLU supported.
            if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)
                return false;
            if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst))
                return false;
            const int glu_op = (int)dst->op_params[0];
            switch (glu_op) {
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU:
                    return true;
                default:
                    return false;
            }
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
            // Qualcomm op_get_rows: F32 src0, F32 dst, I32/I64 src1
            if (!src1 || src1->type != GGML_TYPE_I32)
                return false;
            if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)
                return false;
            return true;
        }
        case GGML_OP_SET_ROWS:
        {
            // Qualcomm op_set_rows: F32 src0, F32/F16 dst, I32/I64 src1
            if (!src1 || src1->type != GGML_TYPE_I32)
                return false;
            if (src0->type != GGML_TYPE_F32)
                return false;
            if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16)
                return false;
            return true;
        }
        case GGML_OP_SUM_ROWS:
        {
            // Qualcomm op_sum_rows: F32 src0
            if (src0->type != GGML_TYPE_F32)
                return false;
            return true;
        }
        case GGML_OP_CONT:
        {
            // CONT maps to HTP_OP_CPY: F32/F16 src0 and dst
            if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16)
                return false;
            if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16)
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
            return ggmlhexagon_supported_flash_attn(ctx, op_tensor);
        }
        default:
            return true; // other ops in table: trust the table entry
    }
}

struct ggml_backend_hexagon_buffer_context {
    ~ggml_backend_hexagon_buffer_context() {
        if (buffer) {
            if (is_ion_buffer) {
                if (backend_ctx && backend_ctx->rpc_mempool) {
                    if (2 != g_hexagon_appcfg.offload_cgraph_type) {
                        backend_ctx->rpc_mempool_usage -= buffer_size;
                    } else {
                        // Mark the ION pool region as free so it can be reused.
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

static void repack_q4_0_q4x4x2(ggml_tensor * t, const void * data, size_t size, void * dst_buf = nullptr) {
    const int QK_Q4_0x4x2 = 256;

    int64_t nrows = ggml_nrows(t);

    size_t row_size    = ggml_row_size(t->type, t->ne[0]);
    size_t row_size_pd = ggml_row_size(t->type, hex_round_up((uint32_t)t->ne[0], (uint32_t)QK_Q4_0x4x2));
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

// ---- Tiled repack for HVX-quant MUL_MAT (mulmat_algotype=32) ----
// HVX-quant kernels (hvx_mm_2d_repacked_*_flat etc.) expect tile-based weight
// layout: each 32x32 tile is tile_size bytes, organized as (ct, kt) major with
// (cp, row) minor inside each tile. Standard GGML row-major layout must be
// converted before passing to DSP.

static void unpack_q4_0_quants(uint8_t * qs, const block_q4_0 * x) {
    for (unsigned int i = 0; i < QK4_0 / 2; ++i) {
        const int x0 = (x->qs[i] & 0x0F);
        const int x1 = (x->qs[i] >> 4);
        qs[i + 0]            = x0;
        qs[i + QK4_0 / 2]   = x1;
    }
}

static void unpack_q4_1_quants(uint8_t * qs, const block_q4_1 * x) {
    for (unsigned int i = 0; i < QK4_1 / 2; ++i) {
        const int x0 = (x->qs[i] & 0x0F);
        const int x1 = (x->qs[i] >> 4);
        qs[i + 0]            = x0;
        qs[i + QK4_1 / 2]   = x1;
    }
}

static void unpack_mxfp4_quants(uint8_t * qs, const block_mxfp4 * x) {
    for (unsigned int i = 0; i < QK_MXFP4 / 2; ++i) {
        const int x0 = (x->qs[i] & 0x0F);
        const int x1 = (x->qs[i] >> 4);
        qs[i + 0]            = x0;
        qs[i + QK_MXFP4 / 2] = x1;
    }
}

static void pack_q4_0_quants(block_q4_0 * x, const uint8_t * qs) {
    for (unsigned int i = 0; i < QK4_0 / 2; ++i) {
        x->qs[i] = qs[i] | (qs[i + QK4_0 / 2] << 4);
    }
}

static void pack_q4_1_quants(block_q4_1 * x, const uint8_t * qs) {
    for (unsigned int i = 0; i < QK4_1 / 2; ++i) {
        x->qs[i] = qs[i] | (qs[i + QK4_1 / 2] << 4);
    }
}

static size_t ggml_hexagon_repacked_size(enum ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    const uint32_t tile_size = htp_mm_get_weight_tile_size((int)type);
    if (tile_size == 0) return 0;
    const uint32_t ne0_p = hex_round_up((uint32_t)ne0, 32);
    const uint32_t ne1_p = hex_round_up((uint32_t)ne1, 32);
    return (size_t)(ne0_p / 32) * (ne1_p / 32) * tile_size * ne2 * ne3;
}

static void repack_q4_0_tiled_to_buf(const ggml_tensor * t, const void * data, void * dst_buf) {
    const block_q4_0 * src_matrix = (const block_q4_0 *) data;
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    const int n_col_tiles = hex_round_up((uint32_t)ne1, 32) / 32;
    const int n_k_tiles   = hex_round_up((uint32_t)ne0, 32) / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_0;
    const size_t matrix_size = (size_t)n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            const block_q4_0 * src_expert = src_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            uint8_t * matrix_dst = (uint8_t *) dst_buf + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;

                    uint8_t tile_quants[32][32];
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            unpack_q4_0_quants(tile_quants[row], &src_expert[r * (ne0 / 32) + kt]);
                        } else {
                            memset(tile_quants[row], 8, 32);
                        }
                    }

                    for (int cp = 0; cp < 16; cp++) {
                        for (int row = 0; row < 32; row++) {
                            tile_dst[cp * 32 + row] = (tile_quants[row][2 * cp + 1] << 4) | tile_quants[row][2 * cp];
                        }
                    }

                    ggml_half * scale_dst = (ggml_half *)(tile_dst + 512);
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        scale_dst[row] = (r < ne1 && kt < ne0 / 32) ? src_expert[r * (ne0 / 32) + kt].d : 0;
                    }
                }
            }
        }
    }
}

static void repack_q4_1_tiled_to_buf(const ggml_tensor * t, const void * data, void * dst_buf) {
    const block_q4_1 * src_matrix = (const block_q4_1 *) data;
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    const int n_col_tiles = hex_round_up((uint32_t)ne1, 32) / 32;
    const int n_k_tiles   = hex_round_up((uint32_t)ne0, 32) / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_1;
    const size_t matrix_size = (size_t)n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            const block_q4_1 * src_expert = src_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            uint8_t * matrix_dst = (uint8_t *) dst_buf + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;

                    uint8_t tile_quants[32][32];
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            unpack_q4_1_quants(tile_quants[row], &src_expert[r * (ne0 / 32) + kt]);
                        } else {
                            memset(tile_quants[row], 0, 32);
                        }
                    }

                    for (int cp = 0; cp < 16; cp++) {
                        for (int row = 0; row < 32; row++) {
                            tile_dst[cp * 32 + row] = (tile_quants[row][2 * cp + 1] << 4) | tile_quants[row][2 * cp];
                        }
                    }

                    ggml_half * scale_dst = (ggml_half *)(tile_dst + 512);
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            scale_dst[2 * row + 0] = src_expert[r * (ne0 / 32) + kt].d;
                            scale_dst[2 * row + 1] = src_expert[r * (ne0 / 32) + kt].m;
                        } else {
                            scale_dst[2 * row + 0] = 0;
                            scale_dst[2 * row + 1] = 0;
                        }
                    }
                }
            }
        }
    }
}

static void repack_q8_0_tiled_to_buf(const ggml_tensor * t, const void * data, void * dst_buf) {
    const block_q8_0 * src_matrix = (const block_q8_0 *) data;
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    const int n_col_tiles = hex_round_up((uint32_t)ne1, 32) / 32;
    const int n_k_tiles   = hex_round_up((uint32_t)ne0, 32) / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q8_0;
    const size_t matrix_size = (size_t)n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            const block_q8_0 * src_expert = src_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            uint8_t * matrix_dst = (uint8_t *) dst_buf + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;

                    for (int cp = 0; cp < 16; cp++) {
                        int col0 = cp * 2;
                        int col1 = col0 + 1;
                        for (int row = 0; row < 32; row++) {
                            int64_t r = ct * 32 + row;
                            const block_q8_0 * b = (r < ne1 && kt < ne0 / 32) ? &src_expert[r * (ne0 / 32) + kt] : NULL;
                            tile_dst[cp * 64 + 2 * row + 0] = b ? b->qs[col0] : 0;
                            tile_dst[cp * 64 + 2 * row + 1] = b ? b->qs[col1] : 0;
                        }
                    }

                    ggml_half * scale_dst = (ggml_half *)(tile_dst + 1024);
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        scale_dst[row] = (r < ne1 && kt < ne0 / 32) ? src_expert[r * (ne0 / 32) + kt].d : 0;
                    }
                }
            }
        }
    }
}

static void repack_mxfp4_tiled_to_buf(const ggml_tensor * t, const void * data, void * dst_buf) {
    const block_mxfp4 * src_matrix = (const block_mxfp4 *) data;
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    const int n_col_tiles = hex_round_up((uint32_t)ne1, 32) / 32;
    const int n_k_tiles   = hex_round_up((uint32_t)ne0, 32) / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_MXFP4;
    const size_t matrix_size = (size_t)n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            const block_mxfp4 * src_expert = src_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            uint8_t * matrix_dst = (uint8_t *) dst_buf + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;

                    uint8_t tile_quants[32][32];
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            unpack_mxfp4_quants(tile_quants[row], &src_expert[r * (ne0 / 32) + kt]);
                        } else {
                            memset(tile_quants[row], 0, 32);
                        }
                    }

                    for (int cp = 0; cp < 16; cp++) {
                        for (int row = 0; row < 32; row++) {
                            tile_dst[cp * 32 + row] = (tile_quants[row][2 * cp + 1] << 4) | tile_quants[row][2 * cp];
                        }
                    }

                    uint8_t * scale_dst = tile_dst + 512;
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        scale_dst[row] = (r < ne1 && kt < ne0 / 32) ? src_expert[r * (ne0 / 32) + kt].e : 0;
                    }
                }
            }
        }
    }
}

// Inverse repack: convert tiled layout back to canonical GGML layout.
// Used by get_tensor so CPU backends can read weight data in original format.

static void repack_tiled_q4_0_to_buf(void * dst_data, const ggml_tensor * t, size_t size) {
    block_q4_0 * dst_matrix = (block_q4_0 *) dst_data;
    int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up((uint32_t)ne0, 32);
    int64_t ne1_padded = hex_round_up((uint32_t)ne1, 32);
    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles   = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_0;
    const size_t matrix_size = (size_t)n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            block_q4_0 * dst_expert = dst_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            const uint8_t * matrix_src = (const uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    const uint8_t * tile_src = matrix_src + (ct * n_k_tiles + kt) * tile_size;

                    uint8_t tile_quants[32][32];
                    for (int cp = 0; cp < 16; cp++) {
                        for (int row = 0; row < 32; row++) {
                            uint8_t val = tile_src[cp * 32 + row];
                            tile_quants[row][2 * cp + 0] = val & 0x0F;
                            tile_quants[row][2 * cp + 1] = val >> 4;
                        }
                    }

                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            pack_q4_0_quants(&dst_expert[r * (ne0 / 32) + kt], tile_quants[row]);
                        }
                    }

                    const ggml_half * scale_src = (const ggml_half *)(tile_src + 512);
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            dst_expert[r * (ne0 / 32) + kt].d = scale_src[row];
                        }
                    }
                }
            }
        }
    }
    GGML_UNUSED(size);
}

static void repack_tiled_q4_1_to_buf(void * dst_data, const ggml_tensor * t, size_t size) {
    block_q4_1 * dst_matrix = (block_q4_1 *) dst_data;
    int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up((uint32_t)ne0, 32);
    int64_t ne1_padded = hex_round_up((uint32_t)ne1, 32);
    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles   = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_1;
    const size_t matrix_size = (size_t)n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            block_q4_1 * dst_expert = dst_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            const uint8_t * matrix_src = (const uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    const uint8_t * tile_src = matrix_src + (ct * n_k_tiles + kt) * tile_size;

                    uint8_t tile_quants[32][32];
                    for (int cp = 0; cp < 16; cp++) {
                        for (int row = 0; row < 32; row++) {
                            uint8_t val = tile_src[cp * 32 + row];
                            tile_quants[row][2 * cp + 0] = val & 0x0F;
                            tile_quants[row][2 * cp + 1] = val >> 4;
                        }
                    }

                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            pack_q4_1_quants(&dst_expert[r * (ne0 / 32) + kt], tile_quants[row]);
                        }
                    }

                    const ggml_half * scale_src = (const ggml_half *)(tile_src + 512);
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            dst_expert[r * (ne0 / 32) + kt].d = scale_src[2 * row + 0];
                            dst_expert[r * (ne0 / 32) + kt].m = scale_src[2 * row + 1];
                        }
                    }
                }
            }
        }
    }
    GGML_UNUSED(size);
}

static void ggml_backend_hexagon_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                               ggml_tensor * tensor, const void * data,
                                               size_t offset, size_t size) {
    // Repack buffer: data arrives in canonical GGML format (e.g. Q4_0 blocks),
    // set_tensor repacks it in-place into tiled (HMX) layout.
    static int set_tensor_call_count = 0;
    bool is_repack = ggml_backend_buffer_is_hexagon_repack(buffer);
    if (set_tensor_call_count < 10 || is_repack) {
        GGMLHEXAGON_LOG_ALWAYS("[SET_TENSOR] #%d name=%s type=%d ne=[%d,%d,%d,%d] nbytes=%zu is_repack=%d offset=%zu size=%zu\n",
                               set_tensor_call_count, tensor->name, (int)tensor->type,
                               (int)tensor->ne[0], (int)tensor->ne[1], (int)tensor->ne[2], (int)tensor->ne[3],
                               ggml_nbytes(tensor), (int)is_repack, offset, size);
    }
    set_tensor_call_count++;
    if (is_repack) {
        switch (tensor->type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_IQ4_NL:  // identical block layout to Q4_0
                GGML_ASSERT(offset == 0);
                GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
                repack_q4_0_tiled_to_buf(tensor, data, tensor->data);
                break;
            case GGML_TYPE_Q4_1:
                GGML_ASSERT(offset == 0);
                GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
                repack_q4_1_tiled_to_buf(tensor, data, tensor->data);
                break;
            case GGML_TYPE_Q8_0:
                GGML_ASSERT(offset == 0);
                GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
                repack_q8_0_tiled_to_buf(tensor, data, tensor->data);
                break;
            case GGML_TYPE_MXFP4:
                GGML_ASSERT(offset == 0);
                GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
                repack_mxfp4_tiled_to_buf(tensor, data, tensor->data);
                break;
            default:
                memcpy((char *)tensor->data + offset, data, size);
                break;
        }
    } else {
        // Main buffer: plain memcpy, no repack.
        memcpy((char *)tensor->data + offset, data, size);
    }

    // Mark weights dirty so Phase 6.5 flushes them on the next batch.
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
    // Repack buffer: un-repack tiled layout back to canonical GGML layout
    // so CPU backends see the original format. Only full-tensor reads
    // (offset==0, full size) are handled.
    if (ggml_backend_buffer_is_hexagon_repack(buffer)) {
        if (offset == 0 && size == ggml_nbytes(tensor)) {
            switch (tensor->type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_IQ4_NL:
                    repack_tiled_q4_0_to_buf(data, tensor, size);
                    return;
                case GGML_TYPE_Q4_1:
                    repack_tiled_q4_1_to_buf(data, tensor, size);
                    return;
                default:
                    break;
            }
        }
        // Partial reads on repack buffer are not supported; fall through
        // to memcpy (raw tiled data, caller beware).
    }
    memcpy(data, (const char *)tensor->data + offset, size);
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
        if (buft->iface.is_host == ggml_backend_hexagon_repack_buffer_is_host) {
            return ctx->repack_buft_name;
        }
        return ctx->buft_name;
    }
    return "hexagon-ion-buffer";
}

// For offload_cgraph_type !=2 path
static ggml_backend_buffer_t alloc_buffer_special(
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

    GGML_ASSERT(nullptr != ctx->rpc_mempool);

    // Bump allocate from the static ION mempool; fall back to heap when full.
    size_t aligned_offset = (ctx->rpc_mempool_usage + 127) / 128 * 128;
    if (aligned_offset + size_aligned <= ctx->rpc_mempool_len) {
        buffer_ctx->buffer      = (char *)ctx->rpc_mempool + aligned_offset;
        buffer_ctx->buffer_size = size_aligned;
        ctx->rpc_mempool_usage  = aligned_offset + size_aligned;
        GGMLHEXAGON_LOG_WARN("[ALLOC] device=%d ion pool: offset=%zu size=%zu",
                             ctx->device, aligned_offset, size_aligned);
    } else {
        GGMLHEXAGON_LOG_WARN("device=%d ion pool exhausted, falling back to heap", ctx->device);
        buffer_ctx->buffer = ggml_aligned_malloc(size_aligned);
        buffer_ctx->buffer_size = size_aligned;
        buffer_ctx->is_ion_buffer = false;
    }

    if (nullptr == buffer_ctx->buffer) {
        GGMLHEXAGON_LOG_WARN("%s: failed to allocate %d MiB\n", __func__, size / SIZE_IN_MB);
        delete buffer_ctx;
        return nullptr;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_hexagon_buffer_interface, buffer_ctx, size);
}

static ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer(
           ggml_backend_buffer_type_t buft, size_t size) {
    if (2 != g_hexagon_appcfg.offload_cgraph_type) {
        return alloc_buffer_special(buft, size);
    }

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
    GGMLHEXAGON_LOG_ALWAYS("device=%d size %ld(%d MiB), rpc_mempool_usage %ld(%d MiB), rpc_mempool_len %ld(%d MiB)",
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
        GGMLHEXAGON_LOG_ALWAYS("[ALLOC] device=%d reuse free region: offset=%zu size=%zu (requested=%zu, waste=%zu)",
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
            GGMLHEXAGON_LOG_ALWAYS("[ALLOC] device=%d new region: offset=%zu size=%zu", ctx->device, aligned_offset, size_aligned);
        } else {
            GGMLHEXAGON_LOG_ALWAYS("device=%d ion pool exhausted: needed %zu MiB, remaining %zu MiB -- falling back to system memory",
                                 ctx->device, size_aligned / SIZE_IN_MB,
                                 (data_limit - ctx->rpc_mempool_usage) / SIZE_IN_MB);
            buffer_ctx->buffer = ggml_aligned_malloc(size_aligned);
            buffer_ctx->buffer_size = size_aligned;
            buffer_ctx->is_ion_buffer = false;
        }
    }

    if (nullptr == buffer_ctx->buffer) {
        GGMLHEXAGON_LOG_ERROR("%s: failed to allocate %d MiB\n", __func__, size / SIZE_IN_MB);
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
        GGMLHEXAGON_LOG_ALWAYS("[ALLOC] device=%d LEAVE size=%zu (%.2f MiB) -> %s, pool_used=%zu/%zu (%.2f%%)",
                             ctx->device, size, (double)size / (1024.0 * 1024.0),
                             mem_type,
                             ctx->rpc_mempool_usage, ctx->rpc_mempool_len,
                             ctx->rpc_mempool_len > 0 ? (double)ctx->rpc_mempool_usage * 100.0 / ctx->rpc_mempool_len : 0.0);
    } else {
        GGMLHEXAGON_LOG_ALWAYS("[ALLOC] device=%d LEAVE size=%zu (%.2f MiB) -> heap", ctx->device, size, (double)size / (1024.0 * 1024.0));
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

static size_t ggml_backend_hexagon_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    // For quantized weight types that will be repacked to tile-based layout
    // in set_tensor, allocate enough space for the repacked data.
    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
            return ggml_hexagon_repacked_size(tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        default:
            return ggml_nbytes(tensor);
    }
}

static bool ggml_backend_buft_is_hexagon(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_hexagon_buffer_type_name;
}

// Repack buft is identified by its is_host discriminator (matches
// ggml_backend_buffer_is_hexagon_repack below). Both the main and repack bufts
// share the same get_name function in this implementation; only is_host
// differs. Used by supports_buft to allow GGML core to route quantized
// weights through set_tensor (which does the in-place tile repack).
static bool ggml_backend_buft_is_hexagon_repack(ggml_backend_buffer_type_t buft) {
    return buft->iface.is_host == ggml_backend_hexagon_repack_buffer_is_host;
}

static bool ggml_backend_hexagon_buffer_is_host(ggml_backend_buffer_type_t buft) {
    // Must return true: ION shared memory is system memory (DDR) that both AP
    // and DSP can access via their own VAs. Returning false would prevent the
    // scheduler from falling back unsupported ops (e.g. SET_ROWS on KV cache)
    // to CPU, causing "cannot run the operation" aborts.
    GGML_UNUSED(buft);
    return true;
}

// Repack buffer type: is_host=false forces GGML core to call set_tensor
// (which does the repack) instead of reading model data directly into
// tensor->data. Both buffer types manage the same ION shared memory pool.
static bool ggml_backend_hexagon_repack_buffer_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

// Returns true if this buffer was allocated from the repack buffer type.
// Repack buffers hold quantized weight data in tiled (HMX) layout and
// require set_tensor/get_tensor to repack/unrepack across the boundary.
static bool ggml_backend_buffer_is_hexagon_repack(const struct ggml_backend_buffer * b) {
    return b->buft->iface.is_host == ggml_backend_hexagon_repack_buffer_is_host;
}

// Session consistency check (mirrors Qualcomm's ggml_hexagon_supported_buffer):
//   - tensor is null:                  neutral, accept (compute-temporary-like)
//   - tensor has no buffer assigned:   neutral, accept (scheduler will route)
//   - buffer is hexagon (main or repack) on this device: accept
//   - buffer is hexagon on a different device: reject (wrong session)
//   - buffer is non-hexagon (e.g. CPU): reject (scheduler should keep on CPU)
//
// Both the main and repack bufts store the owning ggml_backend_hexagon_context
// pointer at buft->context (set in the constructor as `buffer_type.context = this`).
// The current device's context lives at dev->context.
static bool ggmlhexagon_tensor_buffer_is_owned_by(ggml_backend_dev_t dev, const struct ggml_tensor * t) {
    if (!t || !t->buffer) {
        return true;
    }
    ggml_backend_buffer_type_t buft = t->buffer->buft;
    if (!ggml_backend_buft_is_hexagon(buft) && !ggml_backend_buft_is_hexagon_repack(buft)) {
        return false;
    }
    ggml_backend_hexagon_context * dev_ctx  = (ggml_backend_hexagon_context *)dev->context;
    ggml_backend_hexagon_context * buft_ctx = (ggml_backend_hexagon_context *)buft->context;
    return buft_ctx == dev_ctx;
}

// All srcs and the dst of the op must be mapped to the same hexagon session
// (device). Tensors with no buffer are treated as neutral. Mirrors Qualcomm's
// ggml_hexagon_supported_buffers gate in ggml_backend_hexagon_device_supports_op:
// without this, the scheduler can incorrectly assign an op to a device whose
// tensors live in another device's ION region, which would fault on the DSP
// since ION mappings are not shared across separate FastRPC sessions.
static bool ggmlhexagon_op_buffers_belong_to_dev(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    if (!ggmlhexagon_tensor_buffer_is_owned_by(dev, op)) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (!ggmlhexagon_tensor_buffer_is_owned_by(dev, op->src[i])) {
            return false;
        }
    }
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
// MODE 0: per-op FastRPC call (debug only, limited to MUL_MAT)
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
        // Mode 0 only supports MUL_MAT for debugging
        if (node->op != GGML_OP_MUL_MAT) {
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
static enum ggml_status ggmlhexagon_backend_graph_compute_batch(ggml_backend_t backend, struct ggml_cgraph * cgraph) {

    enum ggml_status result         = GGML_STATUS_SUCCESS;
    ggml_backend_hexagon_context * ctx  = (ggml_backend_hexagon_context *)backend->context;
    int64_t begin_time = ggml_time_us();
    int64_t gap_from_prev = ctx->last_graph_end_us ? (begin_time - ctx->last_graph_end_us) : 0;

    // track per-graph node statistics (what ggml core assigned to this backend)
    uint32_t graph_n_nodes = (uint32_t)cgraph->n_nodes;
    ctx->total_nodes_processed += graph_n_nodes;
    if (ctx->min_nodes_per_graph == 0 || graph_n_nodes < ctx->min_nodes_per_graph) {
        ctx->min_nodes_per_graph = graph_n_nodes;
    }
    if (graph_n_nodes > ctx->max_nodes_per_graph) {
        ctx->max_nodes_per_graph = graph_n_nodes;
    }

    // record entry-side ring buffer samples (n_nodes, gap_from_prev) at the
    // earliest possible point so cold-cache first-call outliers are captured.
    {
        int hidx = ctx->perf_hist_idx;
        ctx->n_nodes_hist[hidx]       = (int32_t)graph_n_nodes;
        ctx->gap_from_prev_hist[hidx] = gap_from_prev;
    }

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

    std::vector<ggml_tensor *> tensor_src;
    std::vector<hex_op_desc> hex_ops;
    std::unordered_set<uint32_t> weight_indices;
    uint32_t n_tensors = 0;
    uint32_t n_ops = 0;

    // Phase timing: declare all timers here, used across the pipeline
    int64_t t_p1, t_p2, t_p25, t_p3, t_p4, t_p45, t_p5, t_p6, t_p65, t_p7, t_p75, t_p8;
    int64_t t_start = ggml_time_us();

    // ---- cgraph content-hash check for Phase 1/2/2.5 cache hit ----
    // Hash over each node's {op, ne[4], nb[4], src[0..2] ptr, data ptr}.
    // ~0.2us per node on ARM (FNV-1a: 1 xor + 1 mul per uint64). 17 nodes
    // = ~3us, dominated by the 17 cache-misses that miss this.
    //
    // cgraph pointer is NOT used: the scheduler rebuilds split->graph every
    // call (even when graph_reuse is on at the llama.cpp layer), so the
    // pointer churns. The content is stable.
    auto compute_content_hash = [&]() -> uint64_t {
        uint64_t h = 0xcbf29ce484222325ULL;  // FNV-1a 64-bit offset basis
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];
            if (!node) continue;
            h ^= (uint64_t)node->op; h *= 0x100000001b3ULL;
            for (int j = 0; j < 4; j++) { h ^= (uint64_t)node->ne[j]; h *= 0x100000001b3ULL; }
            for (int j = 0; j < 4; j++) { h ^= (uint64_t)node->nb[j]; h *= 0x100000001b3ULL; }
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                h ^= (uint64_t)(uintptr_t)node->src[j]; h *= 0x100000001b3ULL;
            }
            h ^= (uint64_t)(uintptr_t)node->data; h *= 0x100000001b3ULL;
        }
        return h;
    };
    const uint64_t content_hash = compute_content_hash();
    bool cache_hit = false;
    {
        auto it = ctx->cgraph_cache.find(content_hash);
        if (it != ctx->cgraph_cache.end() &&
            it->second.n_nodes == cgraph->n_nodes &&
            it->second.hex_ops.size() > 0) {
            // Hit. Restore cached state.
            tensor_src.assign(it->second.tensor_src.begin(), it->second.tensor_src.end());
            supported_nodes.assign(it->second.supported_nodes.begin(), it->second.supported_nodes.end());
            unsupported_nodes.assign(it->second.unsupported_nodes.begin(), it->second.unsupported_nodes.end());
            hex_ops.assign(it->second.hex_ops.begin(), it->second.hex_ops.end());
            weight_indices.clear();
            weight_indices.insert(it->second.weight_indices.begin(), it->second.weight_indices.end());
            n_tensors = (uint32_t)it->second.n_tensors;
            n_ops     = (uint32_t)it->second.n_ops;
            cache_hit = true;
            ctx->cgraph_cache_hits++;
        } else {
            ctx->cgraph_cache_misses++;
        }
    }

    // ---- Phase 1: collect unique tensor objects (per-tensor, not per-buffer) ----
    // Each tensor object gets its own descriptor with correct ne/nb,
    // even if multiple tensors share the same data buffer (in-place or buffer reuse).
    std::unordered_map<ggml_tensor *, int32_t> tensor_index_map;

    auto get_or_add_tensor_idx = [&](ggml_tensor * t) -> int32_t {
        if (!t) return -1;
        auto it = tensor_index_map.find(t);
        if (it != tensor_index_map.end()) return it->second;
        int32_t idx = (int32_t)tensor_src.size();
        tensor_index_map[t] = idx;
        tensor_src.push_back(t);
        return idx;
    };

    // Per-call ring buffer recorder. Writes `t_value` into the next slot of
    // `hist_arr`, and tags this slot's index in perf_hist_idx. The slot index
    // is shared across all phase histograms so a single dump can correlate
    // a specific call's phase breakdown (n_nodes, p1, p2, ..., graph_us).
    auto PERF_RECORD = [&](int64_t t_value, int64_t * hist_arr) {
        const int hidx = ctx->perf_hist_idx;
        hist_arr[hidx] = t_value;
    };

    t_p1 = t_start; t_start = ggml_time_us(); ctx->cum_p1_us += t_start - t_p1;
    PERF_RECORD(t_start - t_p1, ctx->p1_hist);

    // ---- Phase 2: build op descriptors ----
    if (!cache_hit) {
    for (auto * node : supported_nodes) {
        hex_op_desc op;
        memset(&op, 0, sizeof(op));
        for (int k = 0; k < 4; k++) op.dst_idx[k] = -1;
        for (int k = 0; k < 6; k++) op.src_idx[k] = -1;
        op.opcode   = node->op;
        memcpy(op.params, node->op_params, sizeof(op.params));
        if (node->op == GGML_OP_MUL_MAT) {
            ggml_hexagon_precompute_mm_params(ctx, node, op, false);

            // Diagnostic: log N=1 GEMV kernel params and tensor info for debugging
            // ION mirror precision issues. N=1 is the smallest batch and most
            // sensitive to data_len/data_offset mismatches.
            if (node->src[1]->ne[1] == 1) {
                const struct htp_mm_kernel_params * kp =
                    (const struct htp_mm_kernel_params *) op.kernel_params;
                const ggml_tensor * s0 = node->src[0];
                const ggml_tensor * s1 = node->src[1];
                GGMLHEXAGON_LOG_DEBUG("DIAG-N1 GEMV: kernel_type=%d n_prefetch=%d vtcm_size=%d "
                                     "src0[%s] ne=[%lld,%lld,%lld,%lld] nb=[%lld,%lld,%lld,%lld] nbytes=%zu "
                                     "src1[%s] ne=[%lld,%lld,%lld,%lld] nb=[%lld,%lld,%lld,%lld] nbytes=%zu "
                                     "dst[%s]  ne=[%lld,%lld,%lld,%lld] nb=[%lld,%lld,%lld,%lld] nbytes=%zu",
                                     kp->kernel_type, kp->n_prefetch, kp->vtcm_size,
                                     ggml_type_name(s0->type),
                                     (long long)s0->ne[0], (long long)s0->ne[1], (long long)s0->ne[2], (long long)s0->ne[3],
                                     (long long)s0->nb[0], (long long)s0->nb[1], (long long)s0->nb[2], (long long)s0->nb[3],
                                     (size_t)ggml_nbytes(s0),
                                     ggml_type_name(s1->type),
                                     (long long)s1->ne[0], (long long)s1->ne[1], (long long)s1->ne[2], (long long)s1->ne[3],
                                     (long long)s1->nb[0], (long long)s1->nb[1], (long long)s1->nb[2], (long long)s1->nb[3],
                                     (size_t)ggml_nbytes(s1),
                                     ggml_type_name(node->type),
                                     (long long)node->ne[0], (long long)node->ne[1], (long long)node->ne[2], (long long)node->ne[3],
                                     (long long)node->nb[0], (long long)node->nb[1], (long long)node->nb[2], (long long)node->nb[3],
                                     (size_t)ggml_nbytes(node));
            }
        } else if (node->op == GGML_OP_FLASH_ATTN_EXT) {
            ggml_hexagon_compute_fa_params(ctx, node,
                (struct htp_fa_kernel_params *) op.kernel_params);
        }
        op.src_idx[0] = get_or_add_tensor_idx(node->src[0]);
        op.src_idx[1] = (node->src[1]) ? get_or_add_tensor_idx(node->src[1]) : -1;
        op.src_idx[2] = (node->src[2]) ? get_or_add_tensor_idx(node->src[2]) : -1;
        op.src_idx[3] = (node->src[3]) ? get_or_add_tensor_idx(node->src[3]) : -1;
        op.dst_idx[0]  = get_or_add_tensor_idx(node);
        hex_ops.push_back(op);
    }

    n_tensors = (uint32_t)tensor_src.size();

    GGMLHEXAGON_LOG_DEBUG("ion-batch %zu ops, %u unique tensors", hex_ops.size(), n_tensors);
    for (size_t i = 0; i < hex_ops.size(); i++) {
        const hex_op_desc & o = hex_ops[i];
        GGMLHEXAGON_LOG_DEBUG("  ion-op[%zu] %s: src0[t%d] src1[t%d] src2[t%d] dst[t%d]",
                              i, ggml_op_name((ggml_op)o.opcode),
                              o.src_idx[0], o.src_idx[1], o.src_idx[2], o.dst_idx[0]);
    }

    // Identify weight tensors: src0 of MUL_MAT that is NOT dst of any op.
    // Weights are read-only across batches; AP never modifies them per batch,
    // so cache flush/invalidate can be skipped for them.
    std::unordered_set<uint32_t> dst_indices;   // indices of tensors that are dst of any op
    for (const auto & op : hex_ops) {
        dst_indices.insert(op.dst_idx[0]);
    }
    for (const auto & op : hex_ops) {
        if (op.opcode == GGML_OP_MUL_MAT) {
            if (dst_indices.find(op.src_idx[0]) == dst_indices.end()) {
                weight_indices.insert(op.src_idx[0]);
                GGMLHEXAGON_LOG_WARN("weight-cache: tensor[%d] identified as weight (type=%d)",
                                     op.src_idx[0], (int)tensor_src[op.src_idx[0]]->type);
            }
        }
    }
    }  // end if (!cache_hit) for Phase 2

    t_p2 = t_start; t_start = ggml_time_us(); ctx->cum_p2_us += t_start - t_p2;
    PERF_RECORD(t_start - t_p2, ctx->p2_hist);

    // ---- Phase 2.5: op fusion ----
    // Supported fusions:
    //   RMS_NORM + MUL      -> HTP_OP_RMS_NORM_MUL
    //   MUL_MAT + ADD       -> HTP_OP_MUL_MAT_ADD     (bias add inside kernel)
    //   3x MUL_MAT (Q,K,V)  -> HTP_OP_MUL_MAT_QKV     (algotype=29 only)
    //   2x MUL_MAT (gate,up)-> HTP_OP_MUL_MAT_FFN     (algotype=29 only)
    //
    // QKV/FFN fusion eligibility:
    //   quantized src0 + F32 src1 + !mm_is_hmx_eligible.
    //   HMX-eligible MUL_MATs are excluded: fusion redirects to HVX fused
    //   kernels, while HMX-eligible ops benefit more from the HMX pipeline.
    if (!cache_hit) {
    {
        // Count src usages of each tensor to ensure fused dst is single-use
        std::vector<int> src_use_count(n_tensors, 0);
        for (const auto & op : hex_ops) {
            for (int k = 0; k < 6; k++) {
                if (op.src_idx[k] >= 0 && op.src_idx[k] < (int)n_tensors) {
                    src_use_count[op.src_idx[k]]++;
                }
            }
        }

        std::vector<hex_op_desc> fused_ops;
        fused_ops.reserve(hex_ops.size());
        size_t n_rms_norm_mul = 0;
        size_t n_mul_mat_add  = 0;
        size_t n_mul_mat_qkv  = 0;
        size_t n_mul_mat_ffn  = 0;

        const size_t vtcm_budget = ctx->socinfo.vtcm_size_in_mb * 1024 * 1024;
        // QKV/FFN fusion only applies to algotype==29:
        //   - algotype==29 dispatches via Qualcomm execute_op, which provides
        //     op_matmul_qkv / op_matmul_ffn as dedicated fused kernels
        //     (in htp/*.c).
        //   - algotype!=29 uses self-built ggmlop_dsp_* path, which has no
        //     fused matmul kernels (only single ggmlop_dsp_mulmat).
        // htp_arch>=V73 is required because op_matmul_qkv/ffn use HMX instructions.
        bool qkv_ffn_enabled = (ctx->socinfo.htp_arch >= V73
                                && g_hexagon_appcfg.mulmat_algotype == 29
                                && g_hexagon_appcfg.enable_opfusion);

#if 0  // dbg: list all ops (kept for future fusion analysis)
        {
            size_t n_mulmat = 0;
            for (size_t i = 0; i < hex_ops.size(); i++) {
                const hex_op_desc & o = hex_ops[i];
                if (o.opcode == GGML_OP_MUL_MAT) n_mulmat++;
                const ggml_tensor * node = tensor_src[o.dst_idx[0]];
                const char * node_name = node ? ggml_get_name(node) : nullptr;
                GGMLHEXAGON_LOG_INFO("DBG op[%zu] %-12s node=%s dst[t%d]",
                                        i, ggml_op_name((ggml_op)o.opcode),
                                        node_name ? node_name : "(unnamed)",
                                        o.dst_idx[0]);
            }
            GGMLHEXAGON_LOG_VERBOSE("DBG batch: %zu ops, %zu MUL_MAT", hex_ops.size(), n_mulmat);
        }
#endif
#if 0  // dbg: list MUL_MAT ops for op-fusion analysis
        {
            size_t n_mulmat = 0;
            for (size_t i = 0; i < hex_ops.size(); i++) {
                const hex_op_desc & o = hex_ops[i];
                if (o.opcode != GGML_OP_MUL_MAT) continue;
                n_mulmat++;
                const ggml_tensor * node = tensor_src[o.dst_idx[0]];
                const ggml_tensor * w    = (o.src_idx[0] >= 0) ? tensor_src[o.src_idx[0]] : nullptr;
                const ggml_tensor * x    = (o.src_idx[1] >= 0) ? tensor_src[o.src_idx[1]] : nullptr;
                const char * node_name = node ? ggml_get_name(node) : nullptr;
                const char * w_name    = w    ? ggml_get_name(w)    : nullptr;
                const int    x_type    = x    ? (int)x->type         : -1;
                const int    hmx       = node ? (int)mm_is_hmx_eligible(node) : -1;
                GGMLHEXAGON_LOG_VERBOSE("DBG batch MUL_MAT[%zu] node=%s W=%s(src1.type=%d) hmx=%d",
                                        i,
                                        node_name ? node_name : "(unnamed)",
                                        w_name    ? w_name    : "(unnamed)",
                                        x_type, hmx);
            }
            GGMLHEXAGON_LOG_VERBOSE("DBG batch: %zu ops, %zu MUL_MAT", hex_ops.size(), n_mulmat);
        }
#endif

        for (size_t i = 0; i < hex_ops.size(); i++) {
            hex_op_desc op = hex_ops[i];

            // RMS_NORM + MUL -> RMS_NORM_MUL
            if (op.opcode == GGML_OP_RMS_NORM && i + 1 < hex_ops.size()) {
                const hex_op_desc & next = hex_ops[i + 1];
                if (next.opcode == GGML_OP_MUL &&
                    next.src_idx[0] == op.dst_idx[0] &&
                    src_use_count[op.dst_idx[0]] == 1) {
                    op.htp_opcode = HTP_OP_RMS_NORM_MUL;
                    op.src_idx[1] = next.src_idx[1];
                    op.dst_idx[0] = next.dst_idx[0];
                    fused_ops.push_back(op);
                    i++;
                    n_rms_norm_mul++;
                    continue;
                }
            }

            // QKV/FFN fusion: only applies to algotype==29 (Qualcomm execute_op).
            if (qkv_ffn_enabled && op.opcode == GGML_OP_MUL_MAT) {
                // QKV fusion: 3 MUL_MAT (Q,K,V) -> HTP_OP_MUL_MAT_QKV.
                // The Q/K/V MUL_MATs may appear in either Q,K,V or Q,V,K order
                // depending on the model (e.g. Gemma4/Llama3 uses Q,K,V, Qwen3 uses Q,V,K).
                // Detect the actual order from tensor names and map src/dst accordingly.
                // DSP-side expects: src[0]=Wk, src[1]=x, src[2]=Wv, src[3]=Wq; dst[0]=K, dst[1]=V, dst[2]=Q.
                if (i + 2 < hex_ops.size()) {
                    const hex_op_desc & next1 = hex_ops[i + 1];
                    const hex_op_desc & next2 = hex_ops[i + 2];
                    if (next1.opcode == GGML_OP_MUL_MAT && next2.opcode == GGML_OP_MUL_MAT) {
                        const ggml_tensor * n_q = tensor_src[op.dst_idx[0]];
                        const ggml_tensor * n1  = tensor_src[next1.dst_idx[0]];
                        const ggml_tensor * n2  = tensor_src[next2.dst_idx[0]];
                        if (is_qkv_mergeable(ctx, n_q, n1, n2)) {
                            // Determine which of n1/n2 is K and which is V by tensor name.
                            // Models name their Q/K/V projection outputs as Qcur-* / Kcur-* /Vcur-*.
                            auto is_k = [](const ggml_tensor * t) { return t && t->name && strstr(t->name, "Kcur"); };
                            auto is_v = [](const ggml_tensor * t) { return t && t->name && strstr(t->name, "Vcur"); };

                            const ggml_tensor * n_k;
                            const ggml_tensor * n_v;
                            const hex_op_desc * op_k;
                            const hex_op_desc * op_v;
                            if (is_k(n1)) {
                                // Q, K, V order (Gemma4, Llama3)
                                n_k = n1; op_k = &next1;
                                n_v = n2; op_v = &next2;
                            } else if (is_v(n1)) {
                                // Q, V, K order (Qwen3)
                                n_k = n2; op_k = &next2;
                                n_v = n1; op_v = &next1;
                            } else {
                                // Fallback: assume Q, K, V order
                                n_k = n1; op_k = &next1;
                                n_v = n2; op_v = &next2;
                            }

                            struct htp_mm_kernel_params kparams;
                            ggml_hexagon_precompute_fused_qkv_params(ctx, n_k->src[0], n_k->src[1], &kparams);
                            if ((size_t)kparams.vtcm_size <= vtcm_budget) {
                                int32_t wq_idx  = op.src_idx[0];
                                int32_t x_idx   = op.src_idx[1];
                                int32_t q_dst   = op.dst_idx[0];
                                op.htp_opcode   = HTP_OP_MUL_MAT_QKV;
                                op.src_idx[0]   = op_k->src_idx[0];  // Wk
                                op.src_idx[1]   = x_idx;             // x (shared)
                                op.src_idx[2]   = op_v->src_idx[0];  // Wv
                                op.src_idx[3]   = wq_idx;            // Wq
                                op.dst_idx[0]   = op_k->dst_idx[0];  // K
                                op.dst_idx[1]   = op_v->dst_idx[0];  // V
                                op.dst_idx[2]   = q_dst;             // Q
                                op.dst_idx[3]   = -1;
                                memcpy(op.kernel_params, &kparams, sizeof(kparams));
                                fused_ops.push_back(op);
                                i += 2;
                                n_mul_mat_qkv++;
                                GGMLHEXAGON_LOG_DEBUG("DBG QKV fusion: q=%s k=%s v=%s | Wq[t%d] Wk[t%d] Wv[t%d] x[t%d] | Q[t%d] K[t%d] V[t%d]",
                                                         n_q->name ? n_q->name : "?",
                                                         n_k->name ? n_k->name : "?",
                                                         n_v->name ? n_v->name : "?",
                                                         wq_idx, op_k->src_idx[0], op_v->src_idx[0], x_idx,
                                                         q_dst, op_k->dst_idx[0], op_v->dst_idx[0]);
                                continue;
                            } else {
                                GGMLHEXAGON_LOG_ALWAYS("skip QKV fusion: VTCM needed (%d) > budget (%zu)",
                                                      (int)kparams.vtcm_size, vtcm_budget);
                            }
                        }
                    }
                }

                // FFN fusion: 2 MUL_MAT (gate,up) -> HTP_OP_MUL_MAT_FFN.
                // Current op is gate, next is up.
                // src0=Wgate, src1=y, src2=Wup; dst[0]=gate, dst[1]=up.
                // Only triggers when is_mergeable_mul_mat returns true
                // (quantized src0 + F32 src1 + !mm_is_hmx_eligible).
                if (i + 1 < hex_ops.size()) {
                    const hex_op_desc & next = hex_ops[i + 1];
                    if (next.opcode == GGML_OP_MUL_MAT) {
                        const ggml_tensor * n_gate = tensor_src[op.dst_idx[0]];
                        const ggml_tensor * n_up   = tensor_src[next.dst_idx[0]];
                        if (is_mergeable_mul_mat_pair(ctx, n_gate, n_up)) {
                            struct htp_mm_kernel_params kparams;
                            ggml_hexagon_precompute_fused_ffn_params(ctx, n_gate->src[0], n_gate->src[1], &kparams);
                            if ((size_t)kparams.vtcm_size <= vtcm_budget) {
                                op.htp_opcode = HTP_OP_MUL_MAT_FFN;
                                // src0=Wgate (keep), src1=y (keep)
                                op.src_idx[2] = next.src_idx[0];   // Wup
                                op.src_idx[3] = -1;
                                // dst[0]=gate (keep)
                                op.dst_idx[1] = next.dst_idx[0];   // up
                                op.dst_idx[2] = -1;
                                op.dst_idx[3] = -1;
                                memcpy(op.kernel_params, &kparams, sizeof(kparams));
                                fused_ops.push_back(op);
                                i += 1;
                                n_mul_mat_ffn++;
                                GGMLHEXAGON_LOG_DEBUG("DBG FFN fusion: gate=%s up=%s | Wgate[t%d] y[t%d] Wup[t%d] | gate[t%d] up[t%d]",
                                                         n_gate->name ? n_gate->name : "?",
                                                         n_up->name ? n_up->name : "?",
                                                         op.src_idx[0], op.src_idx[1], next.src_idx[0],
                                                         op.dst_idx[0], next.dst_idx[0]);
                                continue;
                            } else {
                                GGMLHEXAGON_LOG_DEBUG("skip FFN fusion: VTCM needed (%d) > budget (%zu)",
                                                      (int)kparams.vtcm_size, vtcm_budget);
                            }
                        }
                    }
                }
            }

            // MUL_MAT + ADD -> MUL_MAT_ADD (bias add inside matmul kernel)
            // Only applies to pre-norm models where MUL_MAT (down_proj)
            // is immediately followed by residual ADD. Gemma uses post-norm
            // (MUL_MAT -> RMS_NORM -> MUL -> ADD), so this won't trigger there.
            if (op.opcode == GGML_OP_MUL_MAT && i + 1 < hex_ops.size()) {
                const hex_op_desc & next = hex_ops[i + 1];
                if (next.opcode == GGML_OP_ADD &&
                    src_use_count[op.dst_idx[0]] == 1 &&
                    (next.src_idx[0] == op.dst_idx[0] || next.src_idx[1] == op.dst_idx[0])) {
                    int32_t bias_idx = -1;
                    if (next.src_idx[0] == op.dst_idx[0]) {
                        bias_idx = next.src_idx[1];
                    } else if (next.src_idx[1] == op.dst_idx[0]) {
                        bias_idx = next.src_idx[0];
                    }
                    if (bias_idx >= 0) {
                        op.htp_opcode = HTP_OP_MUL_MAT_ADD;
                        op.src_idx[2]   = bias_idx;
                        op.dst_idx[0]    = next.dst_idx[0];
                        fused_ops.push_back(op);
                        i++;
                        n_mul_mat_add++;
                        continue;
                    }
                }
            }

            fused_ops.push_back(op);
        }

        if (n_rms_norm_mul + n_mul_mat_add + n_mul_mat_qkv + n_mul_mat_ffn > 0) {
            GGMLHEXAGON_LOG_DEBUG("op-fusion: %zu ops -> %zu ops (%zu RMS_NORM_MUL, %zu MUL_MAT_ADD, %zu MUL_MAT_QKV, %zu MUL_MAT_FFN)",
                                    hex_ops.size(), fused_ops.size(),
                                    n_rms_norm_mul, n_mul_mat_add, n_mul_mat_qkv, n_mul_mat_ffn);
            hex_ops = std::move(fused_ops);
        }
    }
    }  // end if (!cache_hit) for Phase 2.5

    n_ops = (uint32_t)hex_ops.size();

    // ---- Cache save: store Phase 1/2/2.5 result keyed by content_hash ----
    // Only on miss. operator[] safely creates entry if absent; on hit we
    // already restored from cache, so skip the assign work entirely.
    if (!cache_hit) {
        auto & entry = ctx->cgraph_cache[content_hash];
        entry.content_hash = content_hash;
        entry.n_nodes   = cgraph->n_nodes;
        entry.n_tensors = (int)n_tensors;
        entry.n_ops     = (int)n_ops;
        entry.tensor_src.assign(tensor_src.begin(), tensor_src.end());
        entry.supported_nodes.assign(supported_nodes.begin(), supported_nodes.end());
        entry.unsupported_nodes.assign(unsupported_nodes.begin(), unsupported_nodes.end());
        entry.hex_ops.assign(hex_ops.begin(), hex_ops.end());
        entry.weight_indices.assign(weight_indices.begin(), weight_indices.end());
    }

    t_p25 = t_start; t_start = ggml_time_us(); ctx->cum_p25_us += t_start - t_p25;
    PERF_RECORD(t_start - t_p25, ctx->p25_hist);

    // ---- Phase 3: compute layout sizes ----
    const uint32_t hdr_size      = (uint32_t)sizeof(hex_batch_hdr);          // ~24 bytes
    const uint32_t ops_region    = (uint32_t)(n_ops * sizeof(hex_op_desc));  // ~96*N
    const uint32_t tens_region   = (uint32_t)(n_tensors * sizeof(hex_tensor_desc)); // ~104*M
    // align ops/tensors regions
    const uint32_t ops_offset    = hdr_size;
    const uint32_t tensors_offset = ops_offset + ((ops_region + HEX_OP_ALIGN - 1) & ~(HEX_OP_ALIGN - 1));
    const uint32_t total_desc_size = tensors_offset + tens_region;

    t_p3 = t_start; t_start = ggml_time_us(); ctx->cum_p3_us += t_start - t_p3;
    PERF_RECORD(t_start - t_p3, ctx->p3_hist);

    // ---- Phase 4: handle heap tensors -> mirror into ION ----
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

#if 0
    // ---- VERIFY (temporary): dump cgraph structure for first 30 calls ----
    // Helps diagnose ubatch-related correctness issues by exposing the
    // actual shape contract that the cgraph provides to Phase 4/4.5/6.
    static int s_verify_call_count = 0;
    if (s_verify_call_count < 30) {
        int vc = s_verify_call_count;
        GGMLHEXAGON_LOG_ALWAYS("[VERIFY-#%d] n_tensors=%u n_ops=%u cgraph_n_nodes=%d",
                               vc, n_tensors, n_ops, cgraph->n_nodes);
        for (uint32_t i = 0; i < n_tensors && i < 64; i++) {
            ggml_tensor * t = tensor_src[i];
            if (!t) continue;
            const char * dp = (const char *)t->data;
            const char * loc = (dp && dp >= ion_base && dp < ion_base + (ptrdiff_t)ion_size) ? "ION" : "HEAP";
            GGMLHEXAGON_LOG_ALWAYS("[VERIFY-#%d] tsrc[%u] type=%d ne=[%lld,%lld,%lld,%lld] nb=[%lld,%lld,%lld,%lld] nbytes=%zu loc=%s",
                                   vc, i, (int)t->type,
                                   t->ne[0], t->ne[1], t->ne[2], t->ne[3],
                                   t->nb[0], t->nb[1], t->nb[2], t->nb[3],
                                   (size_t)ggml_nbytes(t), loc);
        }
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];
            if (!node) continue;
            char srcbuf[320] = {0};
            int  spos = 0;
            for (int j = 0; j < GGML_MAX_SRC && node->src[j]; j++) {
                ggml_tensor * s = node->src[j];
                spos += snprintf(srcbuf + spos, sizeof(srcbuf) - spos,
                                 "src%d=[%lld,%lld,%lld,%lld] ", j,
                                 s->ne[0], s->ne[1], s->ne[2], s->ne[3]);
            }
            GGMLHEXAGON_LOG_ALWAYS("[VERIFY-#%d] node[%d] op=%s ne=[%lld,%lld,%lld,%lld] %s",
                                   vc, i, ggml_op_name(node->op),
                                   node->ne[0], node->ne[1], node->ne[2], node->ne[3], srcbuf);
        }
        s_verify_call_count++;
    }
#endif

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
        // For quantized weights repacked in-place by set_tensor,
        // the actual data on heap is the repacked (larger) layout.
        // The mirror must copy the full repacked data for DSP access.
        bool is_quant_weight = t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16;
        if (is_quant_weight) {
            size_t repacked = ggml_hexagon_repacked_size(t->type, t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
            if (repacked > 0) t_size = (uint32_t)repacked;
        }
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

    // ---- Phase 4.5: track ION offsets for repacked quantized weights ----
    t_p4 = ggml_time_us() - t_prev; t_prev = ggml_time_us();
    PERF_RECORD(t_p4, ctx->p4_hist);
    // For mulmat_algotype=29: weights are already repacked to tile-based layout
    //   by set_tensor during model loading (via repack buffer type, is_host=false).
    //   Phase 4.5 only tracks ION offsets for DSP descriptor updates in Phase 6.
    // For mulmat_algotype=30: repack Q4_0 to x4x2 format in a separate ION region.
    std::vector<std::pair<uint32_t, uint32_t>> repacked_ion_weights; // (offset, length)
    static std::unordered_map<const void *, uint32_t> g_x4x2_ion_offsets;
    static std::unordered_map<const void *, uint32_t> g_tiled_ion_offsets;
    if (g_hexagon_appcfg.mulmat_algotype == 30) { // 30 = HMX sync + x4x2 repack
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
    } else if (g_hexagon_appcfg.mulmat_algotype == 29) { // 29 = Qualcomm execute_op
        // Quantized weights (Q4_0 / Q4_1 / Q8_0 / IQ4_NL / MXFP4) are repacked
        // to tile-based (HMX) layout in set_tensor during model loading,
        // because the model loader routes them through the repack buft
        // (is_host=false). By the time graph_compute_batch runs, every
        // quantized weight's data at t->data is already in tiled layout, so
        // Phase 4.5 does NO repack work here.
        //
        // The only thing Phase 4.5 still needs to do for algotype=29 is
        // record the ION offset of each repacked weight in g_tiled_ion_offsets
        // so Phase 7 can build the DSP descriptor with the correct
        // data_offset. Any quantized weight that somehow lives outside the
        // repack buft is logged as a one-shot warning (should not happen with
        // the current model loader) but its ION offset is still recorded if
        // we can find one, so the DSP descriptor remains well-formed.
        static std::unordered_set<const void *> s_warned_non_repack;
        for (uint32_t i = 0; i < n_tensors; i++) {
            ggml_tensor * t = tensor_src[i];
            if (!t || !t->data) continue;
            bool is_quant_weight = weight_indices.count(i) && t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16;
            if (!is_quant_weight) continue;
            if (t->type != GGML_TYPE_Q4_0 && t->type != GGML_TYPE_Q4_1 &&
                t->type != GGML_TYPE_Q8_0 && t->type != GGML_TYPE_IQ4_NL &&
                t->type != GGML_TYPE_MXFP4) continue;
            const int32_t K = t->ne[0];
            if (K % 32 != 0 || K <= 0) continue;

            if (!t->buffer || !ggml_backend_buffer_is_hexagon_repack(t->buffer)) {
                if (s_warned_non_repack.insert(t->data).second) {
                    GGMLHEXAGON_LOG_WARN("tiled: weight %s (data=%p) not in repack buft; "
                                         "assuming set_tensor already repacked it",
                                         t->name, t->data);
                }
            }

            if (g_tiled_ion_offsets.find(t->data) != g_tiled_ion_offsets.end()) {
                continue;  // already recorded on a prior graph_compute call
            }

            // Record the ION offset for Phase 7. The data is in ION either
            // directly (repack buft) or via the Phase 4 heap->ION mirror.
            const char * dp = (const char *)t->data;
            if (dp >= ion_base && dp < ion_base + (ptrdiff_t)ion_size) {
                g_tiled_ion_offsets[t->data] = (uint32_t)(dp - ion_base);
            } else {
                auto bmi = buffer_mirrors_map.find(t->data);
                if (bmi != buffer_mirrors_map.end() && bmi->second.allocated) {
                    g_tiled_ion_offsets[t->data] = bmi->second.mirror_offset;
                }
            }
        }
    }

    // ---- Phase 5: allocate batch descriptor region in ION mempool ----
    t_p45 = ggml_time_us() - t_prev; t_prev = ggml_time_us();
    PERF_RECORD(t_p45, ctx->p45_hist);
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

    t_p5 = t_prev; t_prev = ggml_time_us(); ctx->cum_p5_us += t_prev - t_p5;
    PERF_RECORD(t_prev - t_p5, ctx->p5_hist);

    // ---- Phase 6: build descriptors in local buffer, then memcpy to ION ----
    t_prev = ggml_time_us();
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
        if (is_quant_weight && ctx->rpc_mempool_cache_offset > 0 && g_hexagon_appcfg.mulmat_algotype == 32) { // 32 = HMX pipeline
            const int32_t M = t->ne[1];  // weight columns
            const int32_t K = t->ne[0];  // inner dimension
            if (K % 32 == 0 && M > 0) {
                td->op_params[0] = 1;  // request DSP-side FP16 cache
            }
        }

        // x4x2: mark tensor descriptor as x4x2 format and use repacked ION offset
        // (repack done in Phase 4.5, data is in ION mirror or separate ION region)
        if (is_quant_weight && g_hexagon_appcfg.mulmat_algotype == 30 && t->type == GGML_TYPE_Q4_0) { // 30 = HMX sync + x4x2 repack
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

        // tiled: update descriptor to match tile-based repacked layout
        // (repack done in set_tensor during model loading via repack buffer type)
        if (is_quant_weight && g_hexagon_appcfg.mulmat_algotype == 29) { // 29 = Qualcomm execute_op
            auto it = g_tiled_ion_offsets.find(t->data);
            if (it != g_tiled_ion_offsets.end()) {
                const int32_t ne0_p = (int32_t)hex_round_up((uint32_t)t->ne[0], 32);
                const int32_t ne1_p = (int32_t)hex_round_up((uint32_t)t->ne[1], 32);
                td->ne[0] = ne0_p;
                td->ne[1] = ne1_p;
                // nb[1] is used by DSP DMA as weight_stride = weight + nc * nb[1].
                // In the tiled layout, tiles are stored as (ne1_p/32) x (ne0_p/32)
                // tiles of tile_size bytes each. The byte offset to column nc is
                // (nc/32) * (ne0_p/32) * tile_size, so nb[1] = (ne0_p/32) * tile_size / 32.
                // ggml_row_size gives the original (non-tiled) stride, which is wrong here.
                td->nb[1] = (int32_t)((ne0_p / 32) * htp_mm_get_weight_tile_size((int)t->type) / 32);
                td->nb[2] = td->nb[1] * ne1_p;
                td->nb[3] = td->nb[2] * (int32_t)t->ne[2];
                td->data_len = (uint32_t)ggml_hexagon_repacked_size(t->type, t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
                td->data_offset = it->second;
                td->flags = 1;  // mirrored (needs cache flush)
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

    // ion_sync_mode controls which cache coherency mechanism to use:
    //   0 = both DC CVAC/CIVAC + DMA_BUF_IOCTL_SYNC (default, safest)
    //   1 = ion_sync only (skip manual DC CVAC/CIVAC, rely on kernel DMA_BUF_IOCTL_SYNC)
    //   2 = DC CVAC/CIVAC only (skip ion_sync, manual cache maintenance only)
    const bool do_dc_cvac  = (g_hexagon_appcfg.ion_sync_mode != 1);
    const bool do_ion_sync = (g_hexagon_appcfg.ion_sync_mode != 2);

    // ---- Phase 6.5: AP -> DSP cache coherency ----
    t_p6 = ggml_time_us() - t_prev; t_prev = ggml_time_us();
    PERF_RECORD(t_p6, ctx->p6_hist);
    // Flush CPU cache to DRAM so DSP can read AP-written data.
    {
        // Collect per-tensor dirty ranges and flush them individually (merged).
        // A single continuous [min, max] range would also flush the holes
        // between low-offset activations and high-offset weights (~940MB of
        // wasted cache-line writes when only a few MB are actually dirty).
        std::vector<std::pair<uint32_t, uint32_t>> ranges;
        ranges.reserve(n_tensors + mirrors.size() + repacked_ion_weights.size() + (size_t)cgraph->n_nodes + 4);

        // Diagnostic counters (per-call, reset every batch) so we can see which
        // source is dominating the flush cost. Logged once per call.
        uint64_t dbg_bytes_tensor = 0, dbg_bytes_mirror = 0;
        uint64_t dbg_bytes_repack_ion = 0, dbg_bytes_batch = 0, dbg_bytes_cgraph = 0;
        uint32_t dbg_ranges_tensor = 0, dbg_ranges_cgraph = 0;

        auto add_range = [&](uint32_t off, uint32_t len) {
            if (len > 0) ranges.push_back({off, off + len});
        };

        // ion_sync_mode=1 path: skip the per-tensor/cgraph range scans
        // entirely; the DMA_BUF_IOCTL_SYNC below handles cache coherency
        // for the whole ION pool. The scan work is pure overhead in this
        // mode (the collected ranges are never used to drive a DC CVAC).
        if (!do_dc_cvac) {
            if (do_ion_sync) {
                int ion_fd = ctx->rpc_mempool_handle;
                if (ion_fd > 0) ion_sync_for_direction(ion_fd, 1);
            }
            int was_weights_dirty = ctx->weights_dirty ? 1 : 0;
            ctx->weights_dirty = false;
            GGMLHEXAGON_LOG_WARN("ion-batch: phase6.5 skipped (ion_sync_mode=%d) dirty=%d",
                                  g_hexagon_appcfg.ion_sync_mode, was_weights_dirty);
        } else {

        for (uint32_t i = 0; i < n_tensors; i++) {
            ggml_tensor * t = tensor_src[i];
            if (!t || !t->data) continue;
            if (weight_indices.count(i) && !ctx->weights_dirty) continue;
            const char * dp = (const char *)t->data;
            if (dp >= ion_base && dp < ion_base + (ptrdiff_t)ion_size) {
                // For quantized weights repacked in-place by set_tensor,
                // use the repacked size (larger than ggml_nbytes).
                bool is_quant_weight = weight_indices.count(i) && t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16;
                size_t flush_size = is_quant_weight ? ggml_hexagon_repacked_size(t->type, t->ne[0], t->ne[1], t->ne[2], t->ne[3]) : ggml_nbytes(t);
                if (flush_size == 0) flush_size = ggml_nbytes(t);
                add_range((uint32_t)(dp - ion_base), (uint32_t)flush_size);
                dbg_bytes_tensor += flush_size;
                dbg_ranges_tensor++;
            }
        }
        for (const auto & m : mirrors) {
            add_range(m.mirror_offset, m.data_len);
            dbg_bytes_mirror += m.data_len;
        }
        for (const auto & rw : repacked_ion_weights) {
            add_range(rw.first, rw.second);
            dbg_bytes_repack_ion += rw.second;
        }
        add_range(batch_offset, total_desc_size);
        dbg_bytes_batch += total_desc_size;

        // Also flush non-op tensors in cgraph not in tensor_src (e.g., test sentinels).
        // Without this, Phase 7.5 DC CIVAC can invalidate cache lines containing
        // unflushed sentinel data, causing sentinel mismatch.
        //
        // CRITICAL: skip repack-buft weights when weights_dirty is false.
        // The per-tensor loop above already guards against re-flushing clean
        // weights via weight_indices; this loop did not, which caused every
        // graph_compute call to re-flush the entire repack weight region
        // (~1.5 GB for gemma4 9B) even though no repack had happened. That
        // single oversight was responsible for ~22 s of the 34 s total
        // graph_compute time in the algotype=29 path.
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * t = cgraph->nodes[i];
            if (!t || !t->data) continue;
            if (!ctx->weights_dirty &&
                t->buffer && ggml_backend_buffer_is_hexagon_repack(t->buffer)) {
                continue;  // repack-buft weight, cache already coherent
            }
            const char * dp = (const char *)t->data;
            if (dp >= ion_base && dp < ion_base + (ptrdiff_t)ion_size) {
                size_t sz = ggml_nbytes(t);
                add_range((uint32_t)(dp - ion_base), (uint32_t)sz);
                dbg_bytes_cgraph += sz;
                dbg_ranges_cgraph++;
            }
        }

        uint32_t flush_bytes = 0;
        uint32_t n_flush     = 0;
        if (do_dc_cvac && !ranges.empty()) {
            std::sort(ranges.begin(), ranges.end());
            // Merge overlapping/adjacent ranges. Merge gap = 1 cache line (64B):
            // flushing a tiny gap is cheaper than issuing a second flush call.
            const uint32_t merge_gap = 64;
            uint32_t cur_start = ranges[0].first;
            uint32_t cur_end   = ranges[0].second;
            for (size_t i = 1; i < ranges.size(); i++) {
                if (ranges[i].first <= cur_end + merge_gap) {
                    if (ranges[i].second > cur_end) cur_end = ranges[i].second;
                } else {
                    cpu_dcache_flush_range(ctx, 0,
                        (char *)ctx->rpc_mempool + cur_start, cur_end - cur_start);
                    flush_bytes += cur_end - cur_start;
                    n_flush++;
                    cur_start = ranges[i].first;
                    cur_end   = ranges[i].second;
                }
            }
            cpu_dcache_flush_range(ctx, 0,
                (char *)ctx->rpc_mempool + cur_start, cur_end - cur_start);
            flush_bytes += cur_end - cur_start;
            n_flush++;
        }
        // Also try DMA_BUF_IOCTL_SYNC as extra safeguard
        if (do_ion_sync) {
            int ion_fd = ctx->rpc_mempool_handle;
            if (ion_fd > 0) ion_sync_for_direction(ion_fd, 1);
        }

        int was_weights_dirty = ctx->weights_dirty ? 1 : 0;
        ctx->weights_dirty = false;
        GGMLHEXAGON_LOG_WARN("ion-batch: phase6.5 DC CVAC %u ranges, %u bytes flushed "
                              "(t=%u/%llu m=%llu ri=%llu b=%llu c=%u/%llu dirty=%d)",
                              n_flush, flush_bytes,
                              dbg_ranges_tensor, (unsigned long long)dbg_bytes_tensor,
                              (unsigned long long)dbg_bytes_mirror,
                              (unsigned long long)dbg_bytes_repack_ion,
                              (unsigned long long)dbg_bytes_batch,
                              dbg_ranges_cgraph, (unsigned long long)dbg_bytes_cgraph,
                              was_weights_dirty);
        }  // end else (do_dc_cvac)
    }

    // AP-side PRE-CALL diagnostic: log first op's src0 first 4 floats after DC CVAC.
    // Compare with [DSP-DIAG] POST-INVAL to pinpoint cache coherency issues.
    if (n_ops > 0) {
        const hex_op_desc & first_op = hex_ops[0];
        uint32_t s0_idx = first_op.src_idx[0];
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
    // 3-way split for fine-grained perf:
    //   rpc_setup: AP-side work between Phase 6.5 end and invoke() entry
    //   dsp_exec:  the synchronous invoke() call itself (RPC round-trip
    //              + DSP-side work + DSP->AP reply)
    //   civac:     AP-side cache invalidate after invoke() returns
    //               (measured in Phase 7.5 below, written into p7_civac_hist)
    t_p65 = ggml_time_us() - t_prev; t_prev = ggml_time_us();
    PERF_RECORD(t_p65, ctx->p65_hist);
    ctx->rpc_batch_call_count++;
    GGMLHEXAGON_LOG_WARN("batch_call #%llu n_ops=%u", ctx->rpc_batch_call_count, n_ops);

    // n_ops_hist: record at FastRPC dispatch time (most relevant for p7 breakdown)
    ctx->n_ops_hist[ctx->perf_hist_idx] = (int32_t)n_ops;

    int64_t t_p7_pre = ggml_time_us();
    int hexagon_error = ggmlop_dsp_execute_batch(ctx->ggmlop_handle, batch_offset, total_desc_size);
    int64_t t_p7_post = ggml_time_us();

    if (AEE_SUCCESS != hexagon_error) {
        GGMLHEXAGON_LOG_WARN("ggmlop_dsp_execute_batch failed: 0x%x", hexagon_error);
    }

    // t_p7 captures the entire synchronous invoke (== old p7 minus civac)
    t_p7 = t_p7_post - t_p7_pre;
    ctx->cumulative_p7_us += t_p7;
    PERF_RECORD(t_p7, ctx->p7_hist);
    ctx->cum_p7_dsp_exec_us  += t_p7;
    ctx->p7_dsp_exec_hist[ctx->perf_hist_idx] = t_p7;
    // rpc_setup = AP-side cost between Phase 6.5 end and the invoke entry
    int64_t p7_rpc_setup = t_p7_pre - t_prev;
    ctx->cum_p7_rpc_setup_us  += p7_rpc_setup;
    ctx->p7_rpc_setup_hist[ctx->perf_hist_idx] = p7_rpc_setup;
    t_prev = ggml_time_us();

    // ---- Phase 7.3: Post-invoke AP-side verification ----
    if (hexagon_error == AEE_SUCCESS && n_ops > 0) {
        // Log LAST op's dst tensor for general verification
        const hex_op_desc & last_op = hex_ops[n_ops - 1];
        uint32_t last_dst_idx = last_op.dst_idx[0];
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
    // civac is now tracked separately via t_civac so the AP-side cache-coherency
    // cost is broken out from p7 (sync invoke) and from p75 (verify+copy-back).
    int64_t t_civac = ggml_time_us();  // civac start
    // DSP writes results to DRAM via ION buffer, but CPU cache may still hold
    // stale data.  DC CIVAC + ion_sync controlled by ion_sync_mode (see Phase 6.5).
    if (hexagon_error == AEE_SUCCESS) {
        if (!do_dc_cvac) {
            // ion_sync_mode=1: rely solely on DMA_BUF_IOCTL_SYNC.
            if (do_ion_sync) {
                int ion_fd = ctx->rpc_mempool_handle;
                if (ion_fd > 0) ion_sync_for_direction(ion_fd, 0);
            }
        } else {
        uint32_t inval_min = ~0u, inval_max = 0;
        for (uint32_t oi = 0; oi < n_ops; oi++) {
            const hex_op_desc & cur_op = hex_ops[oi];
            uint32_t dst_idx = cur_op.dst_idx[0];
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
        if (do_ion_sync) {
            int ion_fd = ctx->rpc_mempool_handle;
            if (ion_fd > 0) ion_sync_for_direction(ion_fd, 0);
        }
        }  // end else (do_dc_cvac)
    }

    // record civac time (Phase 7.5 only). Cum + hist use a separate field.
    {
        int64_t civac_us = ggml_time_us() - t_civac;
        ctx->cum_p7_civac_us  += civac_us;
        ctx->p7_civac_hist[ctx->perf_hist_idx] = civac_us;
    }

    // ---- Phase 7.6: Post-CIVAC verification ----
    // Read dst AFTER DC CIVAC to see what the test framework will actually read.
    // Compare with [AP-POST] (pre-CIVAC, AP cache) and DSP-DIAG dst to pinpoint issues.
    if (hexagon_error == AEE_SUCCESS && n_ops > 0) {
        const hex_op_desc & last_op = hex_ops[n_ops - 1];
        uint32_t last_dst_idx = last_op.dst_idx[0];
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
    // t_p75 = Phase 7.6 verify + Phase 8 copy-back (civac is now its own field)
    t_p75 = ggml_time_us() - t_prev; t_prev = ggml_time_us();
    ctx->cum_p75_us += t_p75;
    PERF_RECORD(t_p75, ctx->p75_hist);
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
            uint32_t last_dst_idx = last_op.dst_idx[0];
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
    PERF_RECORD(t_p8, ctx->p8_hist);
    int64_t end_time = ggml_time_us();
    int64_t graph_dur = end_time - begin_time;
    // (cumulative_p7_us is already updated at the end of the Phase 7 invoke()
    //  block; do not add t_p7 a second time here.)
    ctx->cumulative_graph_us += graph_dur;
    ctx->last_graph_end_us   = end_time;
    // record total graph_us + advance ring buffer slot
    ctx->graph_us_hist[ctx->perf_hist_idx] = graph_dur;
    ctx->perf_hist_idx = (ctx->perf_hist_idx + 1) % ctx->PERF_HIST_CAP;
    if (ctx->perf_hist_count < ctx->PERF_HIST_CAP) ctx->perf_hist_count++;
    // per-phase cumulative time (cum_p75_us / cum_p7_civac_us already
    //  accumulated at the end of their respective phase; p4..p8 + p65 still
    //  use the trailing accumulator pattern.)
    ctx->cum_p4_us  += t_p4;
    ctx->cum_p45_us += t_p45;
    ctx->cum_p6_us  += t_p6;
    ctx->cum_p65_us += t_p65;
    ctx->cum_p8_us  += t_p8;
    // per-call min/max
    if (ctx->min_p7_us == 0 || t_p7 < ctx->min_p7_us)    ctx->min_p7_us = t_p7;
    if (t_p7 > ctx->max_p7_us)                            ctx->max_p7_us = t_p7;
    if (ctx->min_graph_us == 0 || graph_dur < ctx->min_graph_us) ctx->min_graph_us = graph_dur;
    if (graph_dur > ctx->max_graph_us) {
        ctx->max_graph_us     = graph_dur;
        ctx->max_graph_n_nodes = graph_n_nodes;
        ctx->max_graph_n_ops   = n_ops;
        GGMLHEXAGON_LOG_WARN("new max graph_dur=%lld us (n_nodes=%u n_ops=%u p7=%lld p6.5=%lld p7.5=%lld)",
                             (long long)graph_dur, graph_n_nodes, n_ops,
                             (long long)t_p7, (long long)t_p65, (long long)t_p75);
    }
    GGMLHEXAGON_LOG_WARN("ion-batch timing: p4=%lld p4.5=%lld p6=%lld p6.5=%lld p7=%lld p7.5=%lld p8=%lld (us) ops=%u",
                         (long long)t_p4, (long long)t_p45, (long long)t_p6, (long long)t_p65,
                         (long long)t_p7, (long long)t_p75, (long long)t_p8, n_ops);
    GGMLHEXAGON_LOG_WARN("graph supported_nodes   %d", supported_nodes.size());
    GGMLHEXAGON_LOG_WARN("graph inference duration %lld microseconds (gap_from_prev=%lld us)", (long long)graph_dur, (long long)gap_from_prev);
    GGMLHEXAGON_LOG_WARN("rpc stats: batch_calls=%llu cum_p7=%lld us cum_graph=%lld us avg_p7=%lld us avg_graph=%lld us",
                         (unsigned long long)ctx->rpc_batch_call_count,
                         (long long)ctx->cumulative_p7_us, (long long)ctx->cumulative_graph_us,
                         ctx->rpc_batch_call_count ? (long long)(ctx->cumulative_p7_us / (int64_t)ctx->rpc_batch_call_count) : 0,
                         ctx->rpc_batch_call_count ? (long long)(ctx->cumulative_graph_us / (int64_t)ctx->rpc_batch_call_count) : 0);

    // Reset per-graph FP16 cache tracking for the next graph
    ctx->rpc_mempool_cache_usage_est = 0;
    ctx->rpc_mempool_cache_tracked.clear();

    return result;
}

// Reorder cgraph nodes to improve DSP VTCM cache locality.
// Stack MUL_MAT ops sharing the same src1 (input activation) so the DSP can
// reuse VTCM-resident dynamically quantized src1 across consecutive matmuls.
// Matches htp_opnode::stackable() + reorder logic in Qualcomm's ggml-hexagon
//
// Fusion pairs recognized by Phase 2.5 inline fusion in graph_compute_batch
// (RMS_NORM+MUL, MUL_MAT+ADD) are kept adjacent so the inline fusion still
// triggers. Only independent MUL_MAT groups (single node, quantized src0) are
// eligible for reordering.
static void ggml_backend_hexagon_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * gf) {
    GGML_ASSERT(backend);
    GGML_ASSERT(gf);

    const int n = gf->n_nodes;
    if (n < 2) {
        return;
    }

    // Step 1: mark fusion pairs (Phase 2.5 patterns). Nodes sharing group_id
    // must stay adjacent and in order so Phase 2.5 can still detect (i, i+1).
    std::vector<int> group_id(n, -1);
    int next_group = 0;
    for (int i = 0; i < n; i++) {
        if (group_id[i] != -1) {
            continue;
        }
        struct ggml_tensor * node = gf->nodes[i];

        if (node->op == GGML_OP_RMS_NORM && i + 1 < n) {
            struct ggml_tensor * next = gf->nodes[i + 1];
            if (next->op == GGML_OP_MUL && next->src[0] == node) {
                group_id[i]     = next_group;
                group_id[i + 1] = next_group;
                next_group++;
                i++;
                continue;
            }
        }

        if (node->op == GGML_OP_MUL_MAT && i + 1 < n) {
            struct ggml_tensor * next = gf->nodes[i + 1];
            if (next->op == GGML_OP_ADD &&
                (next->src[0] == node || next->src[1] == node)) {
                group_id[i]     = next_group;
                group_id[i + 1] = next_group;
                next_group++;
                i++;
                continue;
            }
        }
    }

    // Step 2: build group list (each group is 1 or 2 contiguous node indices).
    std::vector<std::vector<int>> groups;
    {
        std::vector<bool> visited(n, false);
        for (int i = 0; i < n; i++) {
            if (visited[i]) {
                continue;
            }
            std::vector<int> g;
            if (group_id[i] != -1) {
                for (int j = i; j < n; j++) {
                    if (group_id[j] == group_id[i]) {
                        g.push_back(j);
                        visited[j] = true;
                    }
                }
            } else {
                g.push_back(i);
                visited[i] = true;
            }
            groups.push_back(std::move(g));
        }
    }

    // Step 3: reorder. Move stackable MUL_MAT groups with the same src1 close
    // together via a forward 16-group window. Non-stackable groups stay put.
    auto is_stackable_mul_mat = [](const struct ggml_tensor * node) -> bool {
        if (node == nullptr) {
            return false;
        }
        if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID) {
            return false;
        }
        return node->src[0] && ggml_is_quantized(node->src[0]->type);
    };

    auto same_src1 = [](const struct ggml_tensor * a, const struct ggml_tensor * b) -> bool {
        return a->src[1] != nullptr && a->src[1] == b->src[1];
    };

    std::vector<int> new_node_order;
    new_node_order.reserve(n);
    std::vector<bool> group_used(groups.size(), false);
    constexpr int N_FORWARD = 16;

    for (size_t g0 = 0; g0 < groups.size(); g0++) {
        if (group_used[g0]) {
            continue;
        }
        group_used[g0] = true;
        for (int idx : groups[g0]) {
            new_node_order.push_back(idx);
        }

        if (groups[g0].size() != 1) {
            continue;
        }
        const struct ggml_tensor * node0 = gf->nodes[groups[g0][0]];
        if (!is_stackable_mul_mat(node0)) {
            continue;
        }

        for (size_t g1 = g0 + 1; g1 < groups.size() && g1 <= g0 + N_FORWARD; g1++) {
            if (group_used[g1] || groups[g1].size() != 1) {
                continue;
            }
            const struct ggml_tensor * node1 = gf->nodes[groups[g1][0]];
            if (!is_stackable_mul_mat(node1) || !same_src1(node0, node1)) {
                continue;
            }
            group_used[g1] = true;
            for (int idx : groups[g1]) {
                new_node_order.push_back(idx);
            }
        }
    }

    // Step 4: write back reordered nodes. Only order changes; tensor pointers
    // remain valid, so all src/dst links stay intact.
    std::vector<struct ggml_tensor *> new_nodes(n);
    for (int i = 0; i < n; i++) {
        new_nodes[i] = gf->nodes[new_node_order[i]];
    }
    for (int i = 0; i < n; i++) {
        gf->nodes[i] = new_nodes[i];
    }
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

    // ggml backend has domain_id == -1 (not a real CDSP PD)
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
    props->device_id   = nullptr;  // no PCI bus id for Hexagon CDSP devices
    ggml_backend_hexagon_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
            /* .async                 = */ false,
            /* .host_buffer           = */ false,
            /* .buffer_from_host_ptr  = */ false,
            /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_hexagon_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGMLHEXAGON_LOG_ALWAYS("enter %s\n", __func__);
    int dev_index = 0;

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
        GGMLHEXAGON_LOG_ALWAYS("program specified param is not nullptr");
        //user's program calling ggml_backend_hexagon_device_init_backend directly
        dev_index = (int)(intptr_t)params;
        if (dev_index < 0) {
            GGMLHEXAGON_LOG_ALWAYS("it shouldn't happend\n");
            dev_index = 0;
        }
        GGMLHEXAGON_LOG_ALWAYS("program specified dev_index %d\n", dev_index);
    }
    if (dev_index >= GGML_HEXAGON_MAX_DEVICES) {
        GGMLHEXAGON_LOG_ERROR("invalid dev_index %d", dev_index);
        return nullptr;
    }
    GGMLHEXAGON_LOG_ALWAYS("dev_index=%d", dev_index);
    ggml_backend_t hexagon_backend = ggml_backend_hexagon_init(dev_index, g_hexagon_appcfg.runtime_libpath);
    GGMLHEXAGON_LOG_ALWAYS("leave %s\n", __func__);

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

static ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_repack_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_hexagon_context * ctx = (ggml_backend_hexagon_context *)dev->context;
    return &ctx->repack_buffer_type;
}

static ggml_backend_buffer_type_t * ggml_backend_hexagon_device_get_extra_buffers_type(ggml_backend_dev_t dev) {
    static ggml_backend_buffer_type_t bufts[2];
    bufts[0] = ggml_backend_hexagon_device_get_repack_buffer_type(dev);
    bufts[1] = NULL;
    return bufts;
}

static bool ggml_backend_hexagon_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    // Both main (is_host=true) and repack (is_host=false) bufts share the same
    // ION mempool. Accept either; the repack buft is needed so that GGML core
    // routes quantized weights through set_tensor (which does the in-place
    // tile repack), instead of writing canonical Q4_0/Q8_0 blocks directly.
    if (ggml_backend_buft_is_hexagon(buft) || ggml_backend_buft_is_hexagon_repack(buft)) {
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
        /* .supports_op          = */ ggmlhexagon_can_handle_op_through_cdsp,
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
        /* .graph_optimize          = */ ggml_backend_hexagon_graph_optimize,
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
    // Clamp to actual CDSP thread count: callers (e.g. test-backend-ops) pass
    // host CPU count, but kernel_params precompute must match DSP-side reality.
    ctx->n_threads = (n_threads < g_hexagon_appcfg.thread_counts)
                   ? n_threads : g_hexagon_appcfg.thread_counts;
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

    if (0 == strcmp(name, "ggml_backend_set_n_threads")) {
        return (void *)ggml_backend_hexagon_set_n_threads;
    }
    if (0 == strcmp(name, "ggml_backend_dev_get_extra_bufts")) {
        return (void *)ggml_backend_hexagon_device_get_extra_buffers_type;
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
    GGMLHEXAGON_LOG_ALWAYS("enter %s", __func__);

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

            int ndev = g_hexagon_appcfg.ndev;
            GGMLHEXAGON_LOG_ALWAYS("registering %d Hexagon device(s), ndev=%d", ndev, g_hexagon_appcfg.ndev);

            for (int i = 0; i < ndev; i++) {
                if (i >= GGML_HEXAGON_MAX_DEVICES) {
                    GGMLHEXAGON_LOG_ALWAYS("ndev=%d exceeds GGML_HEXAGON_MAX_DEVICES=%d, only %d devices registered",
                                         ndev, GGML_HEXAGON_MAX_DEVICES, i);
                    break;
                }

                GGMLHEXAGON_LOG_ALWAYS("create backend device for device %d", i);
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
                    GGMLHEXAGON_LOG_ALWAYS("init hexagon dsp failure for device %d", i);
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
    GGMLHEXAGON_LOG_ALWAYS("leave ggml_backend_hexagon_reg");

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
    GGMLHEXAGON_LOG_ALWAYS("enter %s", __func__);
    if (nullptr == runtime_libpath)
        return nullptr;

    //case-3: calling ggml_backend_hexagon_init() directly in user's code
    ggmlhexagon_load_cfg();
    if (!ggmlhexagon_check_valid_appcfg()) {
        return nullptr;
    }

    GGMLHEXAGON_LOG_ALWAYS("device %d", device);
    GGMLHEXAGON_LOG_ALWAYS("runtime libpath %s", runtime_libpath);
    if (device >= GGML_HEXAGON_MAX_DEVICES) {
        GGMLHEXAGON_LOG_ERROR("invalid device %d", device);
        return nullptr;
    }

    if (0 != memcmp(runtime_libpath, g_hexagon_appcfg.runtime_libpath, strlen(g_hexagon_appcfg.runtime_libpath))) {
        //re-setting runtime libpath
        ggmlhexagon_set_runtime_path(device, runtime_libpath);
    }

    if (nullptr != g_hexagon_mgr[device] && nullptr != g_hexagon_mgr[device]->backend) {
        GGMLHEXAGON_LOG_ALWAYS("backend %d(%s) already loaded", device,
                         ggml_backend_hexagon_get_devname(device));
        GGMLHEXAGON_LOG_ALWAYS("leave %s", __func__);
        return g_hexagon_mgr[device]->backend;
    }

    if (2 == g_hexagon_appcfg.offload_cgraph_type) {
        GGMLHEXAGON_LOG_ALWAYS("using ggmlhexagon_backend_graph_compute_batch (ION-based op-batch)");
        ggml_backend_hexagon_interface.graph_compute = ggmlhexagon_backend_graph_compute_batch;
    } else {
        GGMLHEXAGON_LOG_ALWAYS("using ggmlhexagon_backend_graph_compute_general (per-op)");
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
    GGMLHEXAGON_LOG_ALWAYS("leave %s", __func__);

    return hexagon_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)
