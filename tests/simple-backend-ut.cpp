/*
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stddef.h>
#include <unistd.h>
#include <inttypes.h>
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

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#define LOG_BUF_LEN         4096
#define TENSOR_DUMP(tensor) tensor_dump(tensor, #tensor)


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
                    if (strlen(tmposs.str().c_str()) <= (LOG_BUF_LEN - 96)) {
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


static void tensor_dump(const ggml_tensor * tensor, const char * name) {
    printf("dump ggml tensor %s(%s)\n", name, tensor->name);
    printf("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
          name,
          tensor->type, ggml_type_name(tensor->type),
          tensor->ne[0], tensor->ne[1], tensor->ne[2],
          tensor->nb[0], tensor->nb[1], tensor->nb[2]);
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
    size_t data_size = ggml_row_size(tensor->type, tensor->ne[0]);
    size_t n_dims = get_tensor_rank(tensor);
    for (size_t i = 1; i < n_dims; i++) {
        data_size *= tensor->ne[i];
    }
    printf("get_tensor_data_size %ld", data_size);
    printf("ggml_nbytes(tensor) %ld", ggml_nbytes(tensor));

    return ggml_nbytes(tensor);
}


static void show_usage() {
    printf(" " \
        "\nUsage: simple-backend-ut [options]\n" \
        "\n" \
        "Options:\n" \
        " -t GGML_OP_ADD / GGML_OP_MUL / GGML_OP_MULMAT\n" \
        " ?/h print usage information\n\n"
    );
}


int main(int argc, char * argv[]) {
    size_t  ctx_size            = 0;
    int     sizey               = 4;
    int     sizex               = 4;
    int num_threads             = 4;
    int n_ggml_op_type          = GGML_OP_ADD;

    struct ggml_context * ctx   = nullptr;
    struct ggml_cgraph  * gf    = nullptr;
    struct ggml_tensor  * src0  = nullptr;
    struct ggml_tensor  * src1  = nullptr;
    struct ggml_tensor  * dst   = nullptr;
    ggml_backend_t backend      = nullptr;
    ggml_backend_buffer_t buffer= nullptr;
    ggml_type qtype             = GGML_TYPE_F32;
    std::vector<uint8_t> work_buffer;

    for (int i = 1; i < argc; i++) {
        if (0 == strcmp(argv[i], "-t")) {
            if (i + 1 < argc) {
                if (0 == memcmp(argv[i + 1], "GGML_OP_ADD", 11)) {
                    n_ggml_op_type = GGML_OP_ADD;
                } else if (0 == memcmp(argv[i + 1], "GGML_OP_MUL_MAT", 15)) {
                    n_ggml_op_type = GGML_OP_MUL_MAT;
                } else if (0 == memcmp(argv[i + 1], "GGML_OP_MUL", 11)) {
                    n_ggml_op_type = GGML_OP_MUL;
                } else {
                    show_usage();
                    return 1;
                }
                i++;
            }
        } else {
            show_usage();
            return 1;
        }
    }

    printf("Testing %zu devices\n\n", ggml_backend_dev_count());
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);

        printf("Backend %zu/%zu: %s\n", i + 1, ggml_backend_dev_count(),
               ggml_backend_dev_name(dev));

        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            printf("  Skipping CPU backend\n");
            continue;
        }

        backend = ggml_backend_dev_init(dev, reinterpret_cast<const char *>(i));
        GGML_ASSERT(backend != NULL);
        if (backend != nullptr) {
            printf("%s: initialize %s backend\n", __func__, ggml_backend_dev_name(dev));
        }

        printf("  Device description: %s\n", ggml_backend_dev_description(dev));
        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);
        printf("  Device memory: %zu MB (%zu MB free)\n", total / 1024 / 1024, free / 1024 / 1024);
        printf("\n");
    }

    ggml_backend_t backend_cpu = nullptr;
    backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (nullptr == backend_cpu) {
        printf("failed to initialize cpu backend\n");
        exit(1);
    } else {
        printf("succeed to initialize cpu backend\n");
    }

    printf("ggml op:%d(%s)", n_ggml_op_type, ggml_op_name((enum ggml_op) n_ggml_op_type));

    ctx_size += 1024 * 1024 * 32;
    printf("allocating Memory of size %zi bytes, %zi MB\n", ctx_size,
                    (ctx_size / 1024 / 1024));

    struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /* no_alloc   =*/ 0
    };

    ctx = ggml_init(params);
    if (!ctx) {
        printf("ggml_init() failed\n");
        return 2;
    }

    if (qtype != GGML_TYPE_F32) {
        sizex = ggml_blck_size(qtype);
    }

    printf("creating new tensors\n");
    src0 = ggml_new_tensor_2d(ctx, qtype, sizey, sizex);
    src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sizey, sizex);

    ggml_set_input(src0);
    ggml_set_input(src1);
    switch (n_ggml_op_type) {
        case GGML_OP_ADD:
            dst = ggml_add(ctx, src0, src1);
            break;
        case GGML_OP_MUL:
            dst = ggml_mul(ctx, src0, src1);
            break;
        case GGML_OP_MUL_MAT:
            dst = ggml_mul_mat(ctx, src0, src1);
            break;
        default:
            printf("ggml op %d(%s) not supported", n_ggml_op_type,
                  ggml_op_name((enum ggml_op) n_ggml_op_type));
            ggml_free(ctx);
            ggml_backend_free(backend);
            ggml_backend_free(backend_cpu);
            return 3;
    }

    ggml_set_output(dst);

    printf("creating compute graph\n");
    gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_set_f32(src0, 1.0f);
    ggml_set_f32(src1, 2.0f);
    ggml_set_f32(dst,  0.0f);

    ggml_graph_compute_helper(backend, gf, work_buffer, num_threads, nullptr, nullptr);
    if (get_tensor_data_size(dst) < (100 * 100)) {
        printf("dump result tensors:\n");
        TENSOR_DUMP(src0);
        TENSOR_DUMP(src1);
        TENSOR_DUMP(dst);
    } else {
        printf("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi, %5zi)\n",
              src0->name,
              src0->type, ggml_type_name(src0->type), src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
              src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
        printf("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi, %5zi)\n",
              src1->name,
              src1->type, ggml_type_name(src1->type), src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
              src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
        printf("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi, %5zi)\n",
              dst->name,
              dst->type, ggml_type_name(dst->type), dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], dst->nb[0],
              dst->nb[1], dst->nb[2], dst->nb[3]);
    }
    TENSOR_DUMP(dst);

    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_backend_free(backend_cpu);

    return 0;
}
