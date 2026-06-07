/*
 * Copyright (c) zhouwg(https://github.com/zhouwg)
 * Copyright (c) 2024-2025 The ggml authors
 *
 * implementation of self-made Android command line tool for benchmark of ggml-hexagon backend
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include <stddef.h>
#include <inttypes.h>
#if defined(__ANDROID__) || defined(__linux__)
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <limits.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/types.h>
#endif

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
#include <iomanip>
#include <sstream>
#include <chrono>
#include <memory>
#include <regex>
#include <random>
#include <functional>
#include <unordered_map>
#include <condition_variable>
#include <cassert>
#include <unordered_set>
#include <utility>
#include <algorithm>

#include "gguf.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hexagon.h"

static void tensor_dump(const ggml_tensor * tensor, const char * name);

#define TENSOR_DUMP(tensor, bdump) tensor_dump(tensor, #tensor, bdump)
#define TMPBUF_LEN                 256

static bool ggml_graph_compute_helper(
        struct ggml_backend * backend,
        struct ggml_cgraph * graph,
        std::vector<uint8_t> & buf,
        int n_threads,
        ggml_abort_callback abort_callback,
        void * abort_callback_data) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads, NULL);

    plan.abort_callback = abort_callback;
    plan.abort_callback_data = abort_callback_data;

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    if (nullptr != backend)
        return ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    else
        return ggml_graph_compute(graph, &plan);
}


static void tensor_dump_elements(const ggml_tensor * tensor) {
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
                    if (strlen(tmposs.str().c_str()) <= (4096 - 96)) {
                        printf("%s\n", tmposs.str().c_str());
                    }
                    tmposs.clear();
                    tmposs.str("");
                }
            }
        }
    }

    printf("\n");
}


static void tensor_dump(const ggml_tensor * tensor, const char * name, int bdump) {
    printf("dump ggml tensor %s(%s)\n", name, tensor->name);
    printf("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64", nb = (%5zi, %5zi, %5zi, %5zi)\n",
          name,
          tensor->type, ggml_type_name(tensor->type),
          tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
          tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[2]);
    if (1 == bdump)
        tensor_dump_elements(tensor);

    printf("\n");
}


static uint32_t get_tensor_rank(const ggml_tensor * tensor) {
    uint32_t rank = 0;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if ((0 != tensor->ne[i]) && (1 != tensor->ne[i])) {
            rank++;
        }
    }
    return rank;
}


static uint32_t get_tensor_data_size(const ggml_tensor * tensor) {
    return ggml_nbytes(tensor);
}


//ref: https://github.com/ggerganov/llama.cpp/blob/master/tests/test-backend-ops.cpp#L20
static void init_tensor_uniform(ggml_tensor * tensor, float min = -1.0f, float max = 1.0f) {
    // static RNG initialization (revisit if n_threads stops being constant)
    static const size_t n_threads = std::thread::hardware_concurrency();
    static std::vector<std::default_random_engine> generators = []() {
        std::random_device rd;
        std::vector<std::default_random_engine> vec;
        vec.reserve(n_threads);
        //for (size_t i = 0; i < n_threads; i++) { vec.emplace_back(1234 + i); } // fixed seed
        for (size_t i = 0; i < n_threads; i++) { vec.emplace_back(rd()); }
        return vec;
    }();

    size_t size = ggml_nelements(tensor);
    std::vector<float> data(size);

    auto init_thread = [&](size_t ith, size_t start, size_t end) {
        std::uniform_real_distribution<float> distribution(min, max);
        for (size_t i = start; i < end; i++) {
            data[i] = distribution(generators[ith]);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (size_t i = 0; i < n_threads; i++) {
        size_t start =     i*size/n_threads;
        size_t end   = (i+1)*size/n_threads;
        threads.emplace_back(init_thread, i, start, end);
    }
    for (auto & t : threads) {
        t.join();
    }
    if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_I32) {
        ggml_backend_tensor_set(tensor, data.data(), 0, size * sizeof(float));
    } else if (ggml_is_quantized(tensor->type) || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        GGML_ASSERT(size % ggml_blck_size(tensor->type) == 0);
        std::vector<uint8_t> dataq(ggml_row_size(tensor->type, size));
        std::vector<float> imatrix(tensor->ne[0], 1.0f); // dummy importance matrix
        const float * im = imatrix.data();
        if (!ggml_quantize_requires_imatrix(tensor->type)) {
            // when the imatrix is optional, we want to test both quantization with and without imatrix
            // use one of the random numbers to decide
            if (data[0] > 0.5f*(min + max)) {
                im = nullptr;
            }
        }
        ggml_quantize_chunk(tensor->type, data.data(), dataq.data(), 0, size/tensor->ne[0], tensor->ne[0], im);
        GGML_ASSERT(ggml_validate_row_data(tensor->type, dataq.data(), dataq.size()));
        ggml_backend_tensor_set(tensor, dataq.data(), 0, dataq.size());
    } else if (tensor->type == GGML_TYPE_I8 || tensor->type == GGML_TYPE_I16 || tensor->type == GGML_TYPE_I32) {
        // This is going to create some weird integers though.
        ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ASSERT(false);
    }
}


//ref: https://github.com/ggerganov/llama.cpp/blob/master/tests/test-backend-ops.cpp#L310
static void initialize_tensors(ggml_context * ctx) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        init_tensor_uniform(t);
    }
}


static void show_usage() {
    printf(" " \
        "\nUsage: ggmlhexagon-benchmark [options]\n" \
        "\n" \
        "Options:\n" \
        " -t ADD / MUL_MAT \n" \
        " -b 0(QNN_CPU) 1(QNN_GPU) 2(QNN_NPU) 3(Hexagon-cDSP) 4(ggml)\n" \
        " -m row\n" \
        " -n col\n" \
        " ?/h print usage information\n\n"
    );
}


static void get_timestring(char * p_currenttime) {
    if (nullptr == p_currenttime)
        return;


    auto time_to_string = [](const std::chrono::system_clock::time_point & tp)->std::string {
        auto as_time_t = std::chrono::system_clock::to_time_t(tp);
        struct tm tm;

        localtime_r(&as_time_t, &tm);

        std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch());
        char buf[TMPBUF_LEN];
        memset(buf, 0, TMPBUF_LEN);
        snprintf(buf, sizeof(buf), "%04d-%02d-%02d,%02d:%02d:%02d",
                 tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
        GGML_UNUSED(ms);
        return buf;
    };

    std::chrono::system_clock::time_point tp = std::chrono::system_clock::now();
    snprintf(p_currenttime, TMPBUF_LEN, "%s", time_to_string(tp).c_str());
}


int main(int argc, char * argv[]) {
    int64_t n_begin_time        = 0LL;
    int64_t n_end_time          = 0LL;
    int64_t n_duration          = 0LL;
    size_t  ctx_size            = 0;
    int     sizex               = 4096;
    int     sizey               = 4096;
    //int     sizez               = 4096;

    int n_backend_type          = HEXAGON_BACKEND_QNNNPU;
    int n_ggml_op_type          = GGML_OP_ADD;
    int n_mulmat_algotype       = 0;

    struct ggml_context * ctx   = nullptr;
    struct ggml_cgraph  * gf    = nullptr;
    struct ggml_tensor  * src0  = nullptr;
    struct ggml_tensor  * src1  = nullptr;
    //struct ggml_tensor  * src2  = nullptr;
    struct ggml_tensor  * dst   = nullptr;
    ggml_backend_t backend      = nullptr;
    ggml_backend_buffer_t buffer= nullptr;
    ggml_type qtype             = GGML_TYPE_F32;
    //ggml_type qtype           = GGML_TYPE_Q4_0;
    std::vector<uint8_t> work_buffer;

    for (int i = 1; i < argc; i++) {
        if (0 == strcmp(argv[i], "-t")) {
            if (i + 1 < argc) {
                if (0 == memcmp(argv[i + 1], "ADD", 3)) {
                    n_ggml_op_type = GGML_OP_ADD;
                } else if (0 == memcmp(argv[i + 1], "MUL_MAT", 7)) {
                    n_ggml_op_type = GGML_OP_MUL_MAT;
                } else {
                    show_usage();
                    return 1;
                }
                i++;
            }
        } else if (0 == strcmp(argv[i], "-b")) {
            if (i + 1 < argc) {
                int backend = atoi(argv[i + 1]);
                if (backend <= HEXAGON_BACKEND_GGML)
                {
                    n_backend_type     = backend;
                    printf("backend_type %d\n", backend);
                }
                else {
                    show_usage();
                    return 2;
                }
                i++;
            }
        } else if (0 == strcmp(argv[i], "-m")) {
            if (i + 1 < argc) {
                sizex = atoi(argv[i+1]);
                i++;
            }
        } else if (0 == strcmp(argv[i], "-n")) {
            if (i + 1 < argc) {
                sizey = atoi(argv[i+1]);
                i++;
            }
        } else if (0 == strcmp(argv[i], "-a")) {
            if (i + 1 < argc) {
                n_mulmat_algotype = atoi(argv[i+1]);
                i++;
            }
        } else {
            show_usage();
            return 3;
        }
    }

    printf("init backend %d\n", n_backend_type);

#ifdef GGML_USE_HEXAGON
    //avoid manually modify ggml-hexagon.cfg
    if (n_backend_type >= HEXAGON_BACKEND_CDSP) {
        ggml_backend_hexagon_set_cfg(n_backend_type, HWACCEL_CDSP);
    }
    if (n_backend_type < HEXAGON_BACKEND_CDSP) {
        ggml_backend_hexagon_set_cfg(n_backend_type, HWACCEL_QNN);
    }
    ggml_backend_hexagon_set_mulmat_algotype(n_mulmat_algotype);
#endif

    srand(time(NULL));

    ctx_size += 4096 * 4096 * 64;
    ctx_size += 4096 * 4096 * 64;
    printf("Allocating Memory of size %ld bytes, %ld MB\n", ctx_size, (ctx_size / 1024 / 1024));

    struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /* no_alloc   =*/ 0
    };

#ifdef GGML_USE_HEXAGON
    if (n_backend_type != HEXAGON_BACKEND_GGML) {
        params.no_alloc = true;
    }
#endif

    ctx = ggml_init(params);
    if (!ctx) {
        printf("ggml_init failure\n");
        return 4;
    }

    printf("creating new tensors\n");
    printf("ggml_blck_size(%s) %ld\n", ggml_type_name(qtype), ggml_blck_size(qtype));
    printf("ggml_type_size(%s) %ld\n", ggml_type_name(qtype), ggml_type_size(qtype));
    if (qtype != GGML_TYPE_F32) {
        sizex = ggml_blck_size(qtype);
    }

    printf("ggml op:%d(%s)", n_ggml_op_type, ggml_op_name((enum ggml_op) n_ggml_op_type));
    src0 = ggml_new_tensor_2d(ctx, qtype,         sizey, sizex);
    src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sizey, sizex);

    ggml_set_input(src0);
    ggml_set_input(src1);
    switch (n_ggml_op_type) {
        case GGML_OP_ADD:
            dst = ggml_add(ctx, src0, src1);
            break;
        case GGML_OP_MUL_MAT:
            dst = ggml_mul_mat(ctx, src0, src1);
            break;
        default:
            printf("ggml op %d(%s) not supported", n_ggml_op_type,
                  ggml_op_name((enum ggml_op) n_ggml_op_type));
            ggml_free(ctx);
            return 5;
    }

    ggml_set_output(dst);

#ifdef GGML_USE_HEXAGON
    if (n_backend_type != HEXAGON_BACKEND_GGML) {
        printf("init backend %d\n", n_backend_type);
        backend = ggml_backend_hexagon_init(n_backend_type, "/data/local/tmp/");
        if (nullptr == backend) {
            printf("create  backend %d(%s) failed\n", n_backend_type, ggml_backend_hexagon_get_devname(n_backend_type));
            ggml_free(ctx);
            return 6;
        } else {
            printf("create  backend %d(%s) succeed\n", n_backend_type, ggml_backend_hexagon_get_devname(n_backend_type));
        }

        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (nullptr == buffer) {
            printf("%s: failed to allocate backend buffer\n", __func__);
            ggml_free(ctx);
            ggml_backend_free(backend);
            return 7;
        }
    } else {
        printf("init default cpu backend\n");
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
#else
    backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
#endif
    GGML_ASSERT(backend != nullptr);

    printf("creating compute graph\n");
    gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    if (qtype == GGML_TYPE_F32) {
        ggml_set_f32(src0, 1.0f);
        ggml_set_f32(src1, 2.0f);
        ggml_set_f32(dst, 0.0f);
    } else {
        initialize_tensors(ctx);
    }

    n_begin_time = ggml_time_us();
    ggml_backend_graph_compute(backend, gf);
    n_end_time = ggml_time_us();
    n_duration = (n_end_time - n_begin_time) / 1000;
    if (get_tensor_data_size(dst) < (256 * 256)) {
        printf("dump result tensors:\n");
        TENSOR_DUMP(src0, 1);
        TENSOR_DUMP(src1, 1);
        TENSOR_DUMP(dst, 1);
    } else {
        if (get_tensor_data_size(dst) < (512 * 512)) {
            TENSOR_DUMP(dst, 1);
        }
        printf("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
              src0->name,
              src0->type, ggml_type_name(src0->type), src0->ne[0], src0->ne[1], src0->ne[2],
              src0->nb[0], src0->nb[1], src0->nb[2]);
        printf("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
              src1->name,
              src1->type, ggml_type_name(src1->type), src1->ne[0], src1->ne[1], src1->ne[2],
              src1->nb[0], src1->nb[1], src1->nb[2]);
        printf("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
              dst->name,
              dst->type, ggml_type_name(dst->type), dst->ne[0], dst->ne[1], dst->ne[2], dst->nb[0],
              dst->nb[1], dst->nb[2]);

    }

    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);

    char currenttime_string[TMPBUF_LEN];
    get_timestring(currenttime_string);

#ifdef GGML_USE_HEXAGON
    if ((n_backend_type == HEXAGON_BACKEND_CDSP) && (n_ggml_op_type == GGML_OP_MUL_MAT)) {
        printf("[%s] duration of ut GGML_OP_%s with backend %s(algo type:%d): %ld milliseconds\n", currenttime_string, ggml_op_name((enum ggml_op)n_ggml_op_type), ggml_backend_hexagon_get_devname(n_backend_type), ggml_backend_hexagon_get_mulmat_algotype(), n_duration);
    } else {
        printf("[%s] duration of ut GGML_OP_%s with backend %s: %ld milliseconds\n", currenttime_string, ggml_op_name((enum ggml_op)n_ggml_op_type), ggml_backend_hexagon_get_devname(n_backend_type), n_duration);
    }
#else
    printf("[%s] duration of ut GGML_OP_%s with the default ggml backend: %ld milliseconds\n", currenttime_string, ggml_op_name((enum ggml_op)n_ggml_op_type), n_duration);
#endif

    return 0;
}
