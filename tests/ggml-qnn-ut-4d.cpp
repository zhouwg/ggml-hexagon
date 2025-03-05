// This file defines tests for 4d mulmat with QNN backend, modified from test-backend-ops.cpp.

// this file has three sections: Section 1 does general setup, section 2 defines the GGML ops to be tested,
// and section 3 defines which tests to run.
// Quick start for adding a new GGML op: Go to section 2 and create a struct that inherits from test_case,
// then go to section 3 and add an instantiation of your struct.


// ##############################
// ## Section 1: General Setup ##
// ##############################


#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <random>
#include <regex>
#include <string>
#include <thread>
#include <vector>
#if (defined __ANDROID__) || (defined ANDROID)
#include "android/log.h"
#endif


#define GGMLQNN_DEBUG                           1  // for troubleshooting QNN backend
#define GGML_QNN_LOGBUF_LEN                     4096

#define GGMLQNN_LOG_ERROR(...) ggmlqnn_log_internal(GGML_LOG_LEVEL_ERROR, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLQNN_LOG_WARN(...)  ggmlqnn_log_internal(GGML_LOG_LEVEL_WARN , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLQNN_LOG_INFO(...)  ggmlqnn_log_internal(GGML_LOG_LEVEL_INFO , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

#if GGMLQNN_DEBUG
#define GGMLQNN_LOG_DEBUG(...) ggmlqnn_log_internal(GGML_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#else
#define GGMLQNN_LOG_DEBUG(...)
#endif
static void ggmlqnn_log_internal(ggml_log_level level, const char * file, const char * func, int line, const char * format, ...) {
    static std::mutex ggmlqnn_log_internal_mutex;
    static char s_ggmlqnn_log_internal_buf[GGML_QNN_LOGBUF_LEN];

    GGML_UNUSED(file);
    {
        std::lock_guard<std::mutex> lock(ggmlqnn_log_internal_mutex);
        va_list args;
        va_start(args, format);
        int len_prefix = snprintf(s_ggmlqnn_log_internal_buf, GGML_QNN_LOGBUF_LEN, "[%s, %d]: ", func, line);
        int len = vsnprintf(s_ggmlqnn_log_internal_buf + len_prefix, GGML_QNN_LOGBUF_LEN - len_prefix, format, args);
        if (len < (GGML_QNN_LOGBUF_LEN - len_prefix)) {
#if (defined __ANDROID__) || (defined ANDROID)
            //for Android application(standard APP or command line tool)
            //modify from "ggml-qnn" to "kantv" to make AI happy
            __android_log_print(ANDROID_LOG_INFO, "kantv", "%s\n", s_ggmlqnn_log_internal_buf);
            if (GGML_LOG_LEVEL_INFO == level) {
                printf("%s\n", s_ggmlqnn_log_internal_buf);
            }
#else
            //for Snapdragon based WoA(Windows on ARM) device or Linux
            printf("%s\n", s_ggmlqnn_log_internal_buf);
#endif
        }
        va_end(args);
    }
}

static void init_tensor_uniform(ggml_tensor * tensor, float min = -1.0f, float max = 1.0f) {
    size_t nels = ggml_nelements(tensor);
    std::vector<float> data(nels);
    {
        // parallel initialization
        static const size_t n_threads = std::thread::hardware_concurrency();
        // static RNG initialization (revisit if n_threads stops being constant)
        static std::vector<std::default_random_engine> generators = []() {
            std::random_device rd;
            std::vector<std::default_random_engine> vec;
            vec.reserve(n_threads);
            //for (size_t i = 0; i < n_threads; i++) { vec.emplace_back(1234 + i); } // fixed seed
            for (size_t i = 0; i < n_threads; i++) { vec.emplace_back(rd()); }
            return vec;
        }();

        auto init_thread = [&](size_t ith, size_t start, size_t end) {
            std::uniform_real_distribution<float> distribution(min, max);
            auto & gen = generators[ith];
            for (size_t i = start; i < end; i++) {
                data[i] = distribution(gen);
            }
        };

        std::vector<std::future<void>> tasks;
        tasks.reserve(n_threads);
        for (size_t i = 0; i < n_threads; i++) {
            size_t start =     i*nels/n_threads;
            size_t end   = (i+1)*nels/n_threads;
            tasks.push_back(std::async(std::launch::async, init_thread, i, start, end));
        }
        for (auto & t : tasks) {
            t.get();
        }
    }

    if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_I32) {
        ggml_backend_tensor_set(tensor, data.data(), 0, nels * sizeof(float));
    } else if (ggml_is_quantized(tensor->type) || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        GGML_ASSERT(nels % ggml_blck_size(tensor->type) == 0);

         // dummy importance matrix
        std::vector<float> imatrix(tensor->ne[0], 1.0f);
        const float * im = imatrix.data();
        if (!ggml_quantize_requires_imatrix(tensor->type)) {
            // when the imatrix is optional, we want to test both quantization with and without imatrix
            // use one of the random numbers to decide
            if (data[0] > 0.5f*(min + max)) {
                im = nullptr;
            }
        }

        std::vector<uint8_t> dataq(ggml_row_size(tensor->type, nels));
        {
            // parallel quantization by block
            size_t blck_size = ggml_blck_size(tensor->type);
            size_t n_blocks = nels / blck_size;

            auto quantize_thread = [&](size_t start, size_t end) {
                ggml_quantize_chunk(tensor->type, data.data(), dataq.data(),
                    start * blck_size, end - start, blck_size, im);
            };

            const size_t min_blocks_per_thread = 1;
            const size_t n_threads = std::min<size_t>(std::thread::hardware_concurrency()/2,
                                                      std::max<size_t>(1, n_blocks / min_blocks_per_thread));
            std::vector<std::future<void>> tasks;
            tasks.reserve(n_threads);
            for (size_t i = 0; i < n_threads; i++) {
                size_t start =     i*n_blocks/n_threads;
                size_t end   = (i+1)*n_blocks/n_threads;
                tasks.push_back(std::async(std::launch::async, quantize_thread, start, end));
            }
            for (auto & t : tasks) {
                t.get();
            }
        }
        ggml_backend_tensor_set(tensor, dataq.data(), 0, dataq.size());
    } else if (tensor->type == GGML_TYPE_I8 || tensor->type == GGML_TYPE_I16 || tensor->type == GGML_TYPE_I32) {
        // This is going to create some weird integers though.
        ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_I64) {
        // Integers with a size of 8 bytes can be set by mirroring the float data, the specific values are again not really meaningful.
        const size_t nbytes_half = ggml_nbytes(tensor)/2;
        ggml_backend_tensor_set(tensor, data.data(), 0*nbytes_half, nbytes_half);
        ggml_backend_tensor_set(tensor, data.data(), 1*nbytes_half, nbytes_half);
    } else {
        GGML_ABORT("fatal error");
    }
}

static std::vector<float> tensor_to_float(const ggml_tensor * t) {
    std::vector<float> tv;
    tv.reserve(ggml_nelements(t));

    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));

    const auto * tt = ggml_get_type_traits(t->type);
    size_t bs = ggml_blck_size(t->type);
    std::vector<float> vq(ggml_blck_size(t->type));
    bool quantized = ggml_is_quantized(t->type);

    // access elements by index to avoid gaps in views
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < t->ne[0]; i0 += bs) {
                    size_t i = i3*t->nb[3] + i2*t->nb[2] + i1*t->nb[1] + i0/bs*t->nb[0];
                    if (t->type == GGML_TYPE_F16) {
                        tv.push_back(ggml_fp16_to_fp32(*(ggml_fp16_t*)&buf[i]));
                    } else if (t->type == GGML_TYPE_BF16) {
                        tv.push_back(ggml_bf16_to_fp32(*(ggml_bf16_t*)&buf[i]));
                    } else if (t->type == GGML_TYPE_F32) {
                        tv.push_back(*(float *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I64) {
                        tv.push_back((float)*(int64_t *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I32) {
                        tv.push_back((float)*(int32_t *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I16) {
                        tv.push_back((float)*(int16_t *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I8) {
                        tv.push_back((float)*(int8_t *) &buf[i]);
                    } else if (quantized) {
                        tt->to_float(&buf[i], vq.data(), bs);
                        tv.insert(tv.end(), vq.begin(), vq.end());
                    } else {
                        GGML_ABORT("fatal error");
                    }
                }
            }
        }
    }

    return tv;
}

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const float * a, const float * b, size_t n) {
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < n; i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

// maximum absolute asymmetry between a and b
// asymmetry: (a - b) / (a + b)
// This is more stable than relative error if one of the values fluctuates towards zero.
// n: number of values to compare.
// expected_vals: optional vector of expected values for a. If expected_vals is not empty, filter out all comparisons where
//     a does not match any of the expected values. Needed for noncontinuous gradients where the numerical calculation can fail.
static double mean_abs_asymm(const float * a, const float * b, const size_t n, const std::vector<float> & expected_vals) {
    double sum = 0.0f;

    size_t nvalid = 0;
    for (size_t i = 0; i < n; i++) {
        if (!expected_vals.empty()) {
            bool matches_any = false;
            for (const float & ev : expected_vals) {
                if (fabsf(a[i] - ev) < 1e-3f) {
                    matches_any = true;
                    break;
                }
            }
            if (!matches_any) {
                continue;
            }
        }

        const float asymm = (a[i] - b[i]) / (a[i] + b[i]);

        sum += fabsf(asymm);
        nvalid++;
    }

    return sum/nvalid;
}

// utils for printing the variables of the test cases

template<typename T>
static std::string var_to_str(const T & x) {
    return std::to_string(x);
}

template<typename T, size_t N>
static std::string var_to_str(const T (&x)[N]) {
    std::string s = "[";
    for (size_t i = 0; i < N; i++) {
        if (i > 0) {
            s += ",";
        }
        s += var_to_str(x[i]);
    }
    s += "]";
    return s;
}

template<typename T, size_t N>
static std::string var_to_str(const std::array<T, N> & x) {
    std::string s = "[";
    for (size_t i = 0; i < N; i++) {
        if (i > 0) {
            s += ",";
        }
        s += var_to_str(x[i]);
    }
    s += "]";
    return s;
}

static std::string var_to_str(ggml_type type) {
    return ggml_type_name(type);
}

static std::string var_to_str(ggml_op_pool pool) {
    switch (pool) {
        case GGML_OP_POOL_AVG:  return "avg";
        case GGML_OP_POOL_MAX:  return "max";
        default:                return std::to_string(pool);
    }
}

#define VAR_TO_STR(x) (#x "=" + var_to_str(x))

#define VARS_TO_STR1(a) VAR_TO_STR(a)
#define VARS_TO_STR2(a, b) VAR_TO_STR(a) + "," + VAR_TO_STR(b)
#define VARS_TO_STR3(a, b, c) VAR_TO_STR(a) + "," + VARS_TO_STR2(b, c)
#define VARS_TO_STR4(a, b, c, d) VAR_TO_STR(a) + "," + VARS_TO_STR3(b, c, d)
#define VARS_TO_STR5(a, b, c, d, e) VAR_TO_STR(a) + "," + VARS_TO_STR4(b, c, d, e)
#define VARS_TO_STR6(a, b, c, d, e, f) VAR_TO_STR(a) + "," + VARS_TO_STR5(b, c, d, e, f)
#define VARS_TO_STR7(a, b, c, d, e, f, g) VAR_TO_STR(a) + "," + VARS_TO_STR6(b, c, d, e, f, g)
#define VARS_TO_STR8(a, b, c, d, e, f, g, h) VAR_TO_STR(a) + "," + VARS_TO_STR7(b, c, d, e, f, g, h)
#define VARS_TO_STR9(a, b, c, d, e, f, g, h, i) VAR_TO_STR(a) + "," + VARS_TO_STR8(b, c, d, e, f, g, h, i)
#define VARS_TO_STR10(a, b, c, d, e, f, g, h, i, j) VAR_TO_STR(a) + "," + VARS_TO_STR9(b, c, d, e, f, g, h, i, j)
#define VARS_TO_STR11(a, b, c, d, e, f, g, h, i, j, k) VAR_TO_STR(a) + "," + VARS_TO_STR10(b, c, d, e, f, g, h, i, j, k)
#define VARS_TO_STR12(a, b, c, d, e, f, g, h, i, j, k, l) VAR_TO_STR(a) + "," + VARS_TO_STR11(b, c, d, e, f, g, h, i, j, k, l)

#ifdef GGML_USE_SYCL
static bool inline _isinf(float f) {
    return (*(uint32_t *)&f & 0x7fffffff) == 0x7f800000;
}
#else
static bool inline _isinf(float f) { return std::isinf(f); }
#endif

// accept FLT_MAX as infinity
static bool isinf_or_max(float f) {
    return _isinf(f) || f == FLT_MAX || f == -FLT_MAX;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

enum test_mode {
    MODE_TEST,
    MODE_PERF,
    MODE_GRAD,
};

struct test_case {
    virtual ~test_case() {}

    virtual std::string op_desc(ggml_tensor * t) {
        return ggml_op_desc(t);
    }

    virtual std::string vars() {
        return "";
    }

    virtual ggml_tensor * build_graph(ggml_context * ctx) = 0;

    virtual double max_nmse_err() {
        return 1e-7;
    }

    virtual double max_maa_err() {
        return 1e-4;
    }

    virtual float grad_eps() {
        return 1e-1f;
    }

    // If false, estimate gradient with 2 points, neglects 3rd order derivative and higher.
    // If true,  estimate gradient with 4 points, neglects 5th order derivative and higher.
    virtual bool grad_precise() {
        return false;
    }

    // Skip gradient checks if total number of gradients to be checked is larger than this (to speed up the tests).
    virtual int64_t grad_nmax() {
        return 10000;
    }

    // No effect if empty.
    // If not empty, skip all gradient checks where the numerical result does not match any of the values.
    // Needed for dealing with noncontinuous gradients (e.g. ReLU) where estimation using finite differences is unreliable.
    virtual std::vector<float> grad_expect() {
        return {};
    }

    virtual void initialize_tensors(ggml_context * ctx) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t);
        }
    }

    virtual size_t op_size(ggml_tensor * t) {
        size_t size = ggml_nbytes(t);
        // add source tensors
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (t->src[i] != NULL) {
                size += ggml_nbytes(t->src[i]);
            }
        }
        return size;
    }

    virtual uint64_t op_flops(ggml_tensor * t) {
        GGML_UNUSED(t);
        return 0;
    }

    ggml_cgraph * gf = nullptr;
    ggml_cgraph * gb = nullptr;

    static const int sentinel_size = 1024;

    test_mode mode;

    std::vector<ggml_tensor *> sentinels;

    void add_sentinel(ggml_context * ctx) {
        if (mode == MODE_PERF || mode == MODE_GRAD) {
            return;
        }
        ggml_tensor * sentinel = ::ggml_new_tensor_1d(ctx, GGML_TYPE_F32, sentinel_size);
        ggml_format_name(sentinel, "sent_%zu", sentinels.size());
        sentinels.push_back(sentinel);
    }

    // hijack ggml_new_tensor to add sentinels after each tensor to check for overflows in the backend

    ggml_tensor * ggml_new_tensor(ggml_context * ctx, ggml_type type, int n_dims, const int64_t * ne) {
        ggml_tensor * t = ::ggml_new_tensor(ctx, type, n_dims, ne);
        add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_1d(ggml_context * ctx, ggml_type type, int64_t ne0) {
        ggml_tensor * t = ::ggml_new_tensor_1d(ctx, type, ne0);
        add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_2d(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1) {
        ggml_tensor * t = ::ggml_new_tensor_2d(ctx, type, ne0, ne1);
        add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_3d(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2) {
        ggml_tensor * t = ::ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
        add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_4d(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
        ggml_tensor * t = ::ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
        add_sentinel(ctx);
        return t;
    }

    bool ut_eval(ggml_backend_t backend1, ggml_backend_t backend2, const char * op_name) {
        mode = MODE_TEST;

        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead()*128 + ggml_graph_overhead(),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context * ctx = ggml_init(params);
        GGML_ASSERT(ctx);

        gf = ggml_new_graph(ctx);

        // pre-graph sentinel
        add_sentinel(ctx);

        ggml_tensor * out = build_graph(ctx);

        if (op_name != nullptr && op_desc(out) != op_name) {
            //printf("  %s: skipping\n", op_desc(out).c_str());
            ggml_free(ctx);
            return true;
        }

        GGMLQNN_LOG_INFO("  %s(%s): ", op_desc(out).c_str(), vars().c_str());
        fflush(stdout);

        // check if the backends support the ops
        bool supported = true;
        for (ggml_backend_t backend : {backend1, backend2}) {
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
                if (!ggml_backend_supports_op(backend, t)) {
                    printf("not supported [%s] ", ggml_backend_name(backend));
                    supported = false;
                    break;
                }
            }
        }
        if (!supported) {
            printf("\n");
            ggml_free(ctx);
            return true;
        }

        // post-graph sentinel
        add_sentinel(ctx);

        // allocate
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend1);

        if (buf == NULL) {
            printf("failed to allocate tensors [%s] ", ggml_backend_name(backend1));
            ggml_free(ctx);
            return false;
        }

        // build graph
        ggml_build_forward_expand(gf, out);

        // add sentinels as graph nodes so that they are checked in the callback
        for (ggml_tensor * sentinel : sentinels) {
            ggml_graph_add_node(gf, sentinel);
        }

        // randomize tensors
        initialize_tensors(ctx);

        // compare
        struct callback_userdata {
            bool   ok;
            double max_err;
            ggml_backend_t backend1;
            ggml_backend_t backend2;
        };

        callback_userdata ud {
            true,
            max_nmse_err(),
            backend1,
            backend2
        };


        auto callback = [](int index, ggml_tensor * t1, ggml_tensor * t2, void * user_data) -> bool {
            callback_userdata * ud = (callback_userdata *) user_data;
            const char * bn1 = ggml_backend_name(ud->backend1);
            const char * bn2 = ggml_backend_name(ud->backend2);

            std::vector<float> f1 = tensor_to_float(t1);

            if (strcmp(ggml_op_desc(t1), "MUL_MAT") == 0) {
                GGMLQNN_LOG_DEBUG("Default backend output shape: [%d, %d, %d, %d]\n", t1->ne[0], t1->ne[1], t1->ne[2], t1->ne[3]);
                for (int i = 0; i < std::min(50, (int)f1.size()); i++) {
                    GGMLQNN_LOG_DEBUG("default_dst[%d] = %f\n", i, f1[i]);
                }
            }

            // Log t2->data directly before tensor_to_float
            GGMLQNN_LOG_DEBUG("Log t2->data directly before tensor_to_float");
            if (strcmp(ggml_op_desc(t2), "MUL_MAT") == 0) {
                GGMLQNN_LOG_DEBUG("QNN backend t2 shape: [%d, %d, %d, %d]\n", t2->ne[0], t2->ne[1], t2->ne[2], t2->ne[3]);
                float * t2_data = (float *)t2->data;
                for (int i = 0; i < std::min(50, static_cast<int>(t2->ne[0] * t2->ne[1] * t2->ne[2] * t2->ne[3])); i++) {
                    GGMLQNN_LOG_DEBUG("t2_data[%d] = %f\n", i, t2_data[i]);
                }
            }

            std::vector<float> f2 = tensor_to_float(t2);
            GGMLQNN_LOG_DEBUG("after tensor_to_float(t2)");
            double err = nmse(f1.data(), f2.data(), f1.size());
            if (err > ud->max_err) {
                GGMLQNN_LOG_INFO("[%s] NMSE = %.9f > %.9f ", ggml_op_desc(t1), err, ud->max_err);
                for (int i = 0; i < std::min(50, (int)f1.size()); i++) {
                    if (f1[i] != f2[i]) {
                        GGMLQNN_LOG_DEBUG("Mismatch at index %d: default=%f, qnn=%f, diff=%f\n", i, f1[i], f2[i], f1[i] - f2[i]);
                    }
                }
                ud->ok = false;
            }
            return ud->ok;


            if (t1->op == GGML_OP_NONE) {
                // sentinels must be unchanged
                std::vector<uint8_t> t1_data(ggml_nbytes(t1));
                std::vector<uint8_t> t2_data(ggml_nbytes(t2));
                ggml_backend_tensor_get(t1, t1_data.data(), 0, ggml_nbytes(t1));
                ggml_backend_tensor_get(t2, t2_data.data(), 0, ggml_nbytes(t2));

                if (memcmp(t1_data.data(), t2_data.data(), ggml_nbytes(t1)) != 0) {
                    printf("sentinel mismatch: %s ", t1->name);
                    ud->ok = false;
                    return true;
                }
            }

           // std::vector<float> f1 = tensor_to_float(t1);
           // std::vector<float> f2 = tensor_to_float(t2);

            double err2 = nmse(f1.data(), f2.data(), f1.size());
            if (err2 > ud->max_err) {
                GGMLQNN_LOG_INFO("[%s] NMSE = %.9f > %.9f ", ggml_op_desc(t1), err2, ud->max_err);
                // Log first few mismatched elements
                for (int i = 0; i < std::min(50, (int)f1.size()); i++) {
                    if (f1[i] != f2[i]) {
                        GGMLQNN_LOG_DEBUG("Mismatch at index %d: default=%f, qnn=%f, diff=%f\n", i, f1[i], f2[i], f1[i] - f2[i]);
                    }
                }
                ud->ok = false;
            }

            for (size_t i = 0; i < f1.size(); i++) {
                // check for nans
                if (std::isnan(f1[i]) || std::isnan(f2[i])) {
                    printf("[%s] NaN at index %zu (%s=%f %s=%f) ", ggml_op_desc(t1), i, bn1, f1[i], bn2, f2[i]);
                    ud->ok = false;
                    return true;
                }
                // check for infs: both must be inf of the same sign, or both must be finite
                if (isinf_or_max(f1[i]) || isinf_or_max(f2[i])) {
                    if (isinf_or_max(f1[i]) && isinf_or_max(f2[i])) {
                        if (std::signbit(f1[i]) != std::signbit(f2[i])) {
                            printf("[%s] inf sign mismatch: %s=%f %s=%f ", ggml_op_desc(t1), bn1, f1[i], bn2, f2[i]);
                            ud->ok = false;
                            return true;
                        }
                    } else {
                        printf("[%s] inf mismatch: %s=%f %s=%f ", ggml_op_desc(t1), bn1, f1[i], bn2, f2[i]);
                        ud->ok = false;
                        return true;
                    }
                }
            }

            err = nmse(f1.data(), f2.data(), f1.size());
            if (err > ud->max_err) {
                GGMLQNN_LOG_INFO("[%s] NMSE = %.9f > %.9f ", ggml_op_desc(t1), err, ud->max_err);
                //for (int i = 0; i < (int) f1.size(); i++) {
                //    printf("%5d %9.6f %9.6f, diff = %9.6f\n", i, f1[i], f2[i], f1[i] - f2[i]);
                //}
                //printf("\n");
                //exit(1);
                ud->ok = false;
            }
            return true;

            GGML_UNUSED(index);
        };

        const bool cmp_ok = ggml_backend_compare_graph_backend(backend1, backend2, gf, callback, &ud);

        if (!cmp_ok) {
            GGMLQNN_LOG_INFO("compare failed ");
        }

        ggml_backend_buffer_free(buf);

        ggml_free(ctx);

        if (ud.ok && cmp_ok) {
            GGMLQNN_LOG_INFO("\033[1;32mOK\033[0m\n");
            return true;
        }

        GGMLQNN_LOG_INFO("\033[1;31mFAIL\033[0m\n");
        return false;
    }
};


// ###################################
// ## Section 2: GGML Op Defintions ##
// ###################################


// The following is an example showing the bare minimum for creating a test for a GGML op.

// GGML_OP_EXAMPLE
struct test_example : public test_case {
    // Always define these 2 or variants thereof:
    const ggml_type type; // The type of the input tensors.
    const std::array<int64_t, 4> ne; // The shape of the input tensors.
    // For some ops it's necessary to define multiple types or shapes for the inputs.
    // Or they may need additional parameters.

    // Put all parameters needed to fully define the test into one of the VARS_TO_STR macros.
    // In most cases these are just the properties of the struct that you defined above.
    // This is needed for info prints.
    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    // Define a constructor for the struct.
    // In most cases it will be sufficient to have the same arguments as the struct has properties
    // and just use initializer lists.
    test_example(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3})
        : type(type), ne(ne) {}

    // Define how a simple GGML compute graph can be constructed for the new GGML op.
    ggml_tensor * build_graph(ggml_context * ctx) override {
        // Step 1: create input tensors that don't depend on any other tensors:
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a"); // Setting names is optional but it's useful for debugging.

        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(b, "b");

        // Step 2: use the op that you want to test in the GGML compute graph.
        ggml_tensor * out = ggml_add(ctx, a, b); // For this example we're just doing a simple addition.
        ggml_set_name(out, "out");

        // Step 3: return the output tensor.
        return out;
    }
    // In order to also check the gradients for your op, add calls like ggml_set_param(ctx, a)
    // immediately after you create the tensors.
    // This is optional and only makes sense if a backward pass has actually been implemented for the new op.
};


// GGML_OP_MUL_MAT
struct test_mul_mat : public test_case {
    const ggml_type type_a;
    const ggml_type type_b;
    const int64_t m;
    const int64_t n;
    const int64_t k;
    const std::array<int64_t, 2> bs;  // dims 3 and 4
    const std::array<int64_t, 2> nr;  // repeat in dims 3 and 4
    const std::array<int64_t, 4> per; // permutation of dimensions

    std::string vars() override {
        return VARS_TO_STR8(type_a, type_b, m, n, k, bs, nr, per);
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    int64_t grad_nmax() override {
        return 20000;
    }

    uint64_t op_flops(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return 2 * m * n * k * bs[0] * nr[0] * bs[1] * nr[1];
    }

    test_mul_mat(ggml_type type_a = GGML_TYPE_F32, ggml_type type_b = GGML_TYPE_F32,
            int64_t m = 32, int64_t n = 32, int64_t k = 32,
            std::array<int64_t, 2> bs = {10, 10},
            std::array<int64_t, 2> nr = {2, 2},
            std::array<int64_t, 4> per = {0, 1, 2, 3})
        : type_a(type_a), type_b(type_b), m(m), n(n), k(k), bs(bs), nr(nr), per(per) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // C^T = A * B^T: (k, m) * (k, n) => (m, n)
        ggml_tensor * a;
        ggml_tensor * b;

        const int npermuted = (per[0] != 0) + (per[1] != 1) + (per[2] != 2) + (per[3] != 3);
        if (npermuted > 0) {
            GGML_ASSERT(npermuted == 2);
            GGML_ASSERT(!ggml_is_quantized(type_a) || per[0] == 0);
            GGML_ASSERT(!ggml_is_quantized(type_b) || per[0] == 0);

            // Create tensors with the permuted dimensions, then permute them back to the dimensions given by m,n,k.
            const int64_t ne_a[4] = {k, m, bs[0],       bs[1]};
            const int64_t ne_b[4] = {k, n, bs[0]*nr[0], bs[1]*nr[1]};

            a = ggml_new_tensor_4d(ctx, type_a, ne_a[per[0]], ne_a[per[1]], ne_a[per[2]], ne_a[per[3]]);
            b = ggml_new_tensor_4d(ctx, type_b, ne_b[per[0]], ne_b[per[1]], ne_b[per[2]], ne_b[per[3]]);
            if (!ggml_is_quantized(type_a)) {
                if (bs[1] == 1 && nr[1] == 1) {
                    ggml_set_param(ctx, a);
                }
                ggml_set_param(ctx, b);
            }
            ggml_set_name(a, "a");
            ggml_set_name(b, "b");

            a = ggml_permute(ctx, a, per[0], per[1], per[2], per[3]);
            b = ggml_permute(ctx, b, per[0], per[1], per[2], per[3]);
            ggml_set_name(a, "a_permuted");
            ggml_set_name(b, "b_permuted");
        } else {
            a = ggml_new_tensor_4d(ctx, type_a, k, m, bs[0],       bs[1]);
            b = ggml_new_tensor_4d(ctx, type_b, k, n, bs[0]*nr[0], bs[1]*nr[1]);
            if (!ggml_is_quantized(type_a)) {
                if (bs[1] == 1 && nr[1] == 1) {
                    ggml_set_param(ctx, a);
                }
                ggml_set_param(ctx, b);
            }
            ggml_set_name(a, "a");
            ggml_set_name(b, "b");
        }

        ggml_tensor * out = ggml_mul_mat(ctx, a, b);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_MUL_MAT_ID
struct test_mul_mat_id : public test_case {
    const ggml_type type_a;
    const ggml_type type_b;
    const int n_mats;
    const int n_used;
    const bool b; // brodcast b matrix
    const int64_t m;
    const int64_t n;
    const int64_t k;

    std::string vars() override {
        return VARS_TO_STR8(type_a, type_b, n_mats, n_used, b, m, n, k);
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    uint64_t op_flops(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return 2 * m * k * n * n_used;
    }

    test_mul_mat_id(ggml_type type_a = GGML_TYPE_F32, ggml_type type_b = GGML_TYPE_F32,
            int n_mats = 8, int n_used = 2, bool b = false,
            int64_t m = 32, int64_t n = 32, int64_t k = 32)
        : type_a(type_a), type_b(type_b), n_mats(n_mats), n_used(n_used), b(b),
            m(m), n(n), k(k) {
            GGML_ASSERT(n_used <= n_mats);
        }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // C^T = A * B^T: (k, m) * (k, n) => (m, n)
        ggml_tensor * as = ggml_new_tensor_3d(ctx, type_a, k, m, n_mats);
        ggml_set_name(as, "as");

        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_mats, n);
        ggml_set_name(ids, "ids");
        if (n_used != n_mats) {
            ids = ggml_view_2d(ctx, ids, n_used, n, ids->nb[1], 0);
            ggml_set_name(ids, "view_of_ids");
        }

        ggml_tensor * b = ggml_new_tensor_3d(ctx, type_b, k, this->b ? 1 : n_used, n);
        ggml_set_name(b, "b");

        ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::random_device rd;
        std::default_random_engine rng(rd());
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                if (ggml_is_view_op(t->op)) { continue; }
                // ids
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::vector<int32_t> data(t->ne[0]);
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = i % n_mats;
                    }
                    std::shuffle(data.begin(), data.end(), rng);
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));
                }
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// ###########################################
// ## Section 3: GGML Op Test Instantiation ##
// ###########################################
static const ggml_type all_types[] = {
    GGML_TYPE_F32
};

static const ggml_type base_types[] = {
    GGML_TYPE_F32
};

static const ggml_type other_types[] = {
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    // GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, // TODO: implement for all backends
    GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,
    GGML_TYPE_IQ4_NL, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS,
    GGML_TYPE_BF16,
};

// Test cases for evaluation: should try to cover edge cases while using small input sizes to keep the runtime low
static std::vector<std::unique_ptr<test_case>> make_test_cases_eval() {
    std::vector<std::unique_ptr<test_case>> test_cases;
    std::default_random_engine rng(0);

    // test cases without permutation
    const ggml_type type_a = GGML_TYPE_F32;
    const ggml_type type_b = GGML_TYPE_F32;
    /*
    test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, {1, 1}, {2, 1}));
    test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, {1, 1}, {1, 2}));
    test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, {3, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, {3, 1}, {2, 1}));
    test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, {3, 2}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, {3, 2}, {2, 1}));

    test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, {3, 2}, {1, 2}));
    */
    test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, {3, 2}, {2, 2}));

    return test_cases;
}

static bool test_backend(ggml_backend_t backend, test_mode mode, const char * op_name, const char * params_filter) {
    auto filter_test_cases = [](std::vector<std::unique_ptr<test_case>> & test_cases, const char * params_filter) {
        if (params_filter == nullptr) {
            return;
        }

        std::regex params_filter_regex(params_filter);

        for (auto it = test_cases.begin(); it != test_cases.end();) {
            if (!std::regex_search((*it)->vars(), params_filter_regex)) {
                it = test_cases.erase(it);
                continue;
            }

            it++;
        }
    };

    if (mode == MODE_TEST) {
        auto test_cases = make_test_cases_eval();
        filter_test_cases(test_cases, params_filter);
        ggml_backend_t backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
        if (backend_cpu == NULL) {
            printf("  Failed to initialize CPU backend\n");
            return false;
        }

        size_t n_ok = 0;
        for (auto & test : test_cases) {
            if (test->ut_eval(backend, backend_cpu, op_name)) {
                n_ok++;
            }
        }
        printf("  %zu/%zu tests passed\n", n_ok, test_cases.size());

        ggml_backend_free(backend_cpu);

        return n_ok == test_cases.size();
    }

    GGML_ABORT("fatal error");
}

static void usage(char ** argv) {
    printf("Usage: %s [mode] [-o <op>] [-b <backend>] [-p <params regex>]\n", argv[0]);
    printf("    valid modes:\n");
    printf("      - test (default, compare with CPU backend for correctness)\n");
    printf("    op names for -o are as given by ggml_op_desc() (e.g. ADD, MUL_MAT, etc)\n");
}

int main(int argc, char ** argv) {
    test_mode mode = MODE_TEST;
    const char * op_name_filter = nullptr;
    const char * backend_filter = nullptr;
    const char * params_filter = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "test") == 0) {
            mode = MODE_TEST;
        } else if (strcmp(argv[i], "-o") == 0) {
            if (i + 1 < argc) {
                op_name_filter = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0) {
            if (i + 1 < argc) {
                backend_filter = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-p") == 0) {
            if (i + 1 < argc) {
                params_filter = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else {
            usage(argv);
            return 1;
        }
    }

    // load and enumerate backends
    ggml_backend_load_all();

    printf("Testing %zu devices\n\n", ggml_backend_dev_count());

    size_t n_ok = 0;

    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);

        printf("Backend %zu/%zu: %s\n", i + 1, ggml_backend_dev_count(), ggml_backend_dev_name(dev));

        if (backend_filter != NULL && strcmp(backend_filter, ggml_backend_dev_name(dev)) != 0) {
            printf("  Skipping\n");
            n_ok++;
            continue;
        }

        if (backend_filter == NULL && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && mode != MODE_GRAD) {
            printf("  Skipping CPU backend\n");
            n_ok++;
            continue;
        }

#ifdef GGML_USE_QNN
        ggml_backend_t backend = ggml_backend_dev_init(dev, reinterpret_cast<const char *>(i));
#else
        ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
#endif
        GGML_ASSERT(backend != NULL);

        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (ggml_backend_set_n_threads_fn) {
            // TODO: better value for n_threads
            ggml_backend_set_n_threads_fn(backend, std::thread::hardware_concurrency());
        }

        printf("  Device description: %s\n", ggml_backend_dev_description(dev));
        size_t free, total; // NOLINT
        ggml_backend_dev_memory(dev, &free, &total);
        printf("  Device memory: %zu MB (%zu MB free)\n", total / 1024 / 1024, free / 1024 / 1024);
        printf("\n");

        bool ok = test_backend(backend, mode, op_name_filter, params_filter);

        printf("  Backend %s: ", ggml_backend_name(backend));
        if (ok) {
            printf("\033[1;32mOK\033[0m\n");
            n_ok++;
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }

        printf("\n");

        ggml_backend_free(backend);
    }

    ggml_quantize_free();

    printf("%zu/%zu backends passed\n", n_ok, ggml_backend_dev_count());

    if (n_ok != ggml_backend_dev_count()) {
        printf("\033[1;31mFAIL\033[0m\n");
        return 1;
    }

    printf("\033[1;32mOK\033[0m\n");
    return 0;
}
