#include "ggml-dsp.h"
#include "worker_pool.h"

typedef struct {
    const ggml_tensor * src0;
    const ggml_tensor * src1;
    ggml_tensor * dst;
    int64_t start_idx;
    int64_t end_idx;
    worker_synctoken_t *synctoken;
} add_thread_data_t;

static HVX_INLINE_ALWAYS void l2fetch(const void * p, uint32_t stride,
                           uint32_t width, uint32_t height,
                           uint32_t dir) {
    uint64_t control = HEXAGON_V64_CREATE_H(dir, stride, width, height);
    __asm__ __volatile__ (" l2fetch(%0,%1) " : :"r"(p),"r"(control));
}

static HVX_INLINE_ALWAYS int32_t is_aligned(void * addr, uint32_t align)
{
    return ((size_t) addr & (align - 1)) == 0;
}

static inline void ggml_add_f32_scale (const int n, float * z, const float * x, const float * y) {
    for (int i = 0; i < n; ++i) z[i]  = x[i] + y[i];
}


static inline void ggml_vec_add_f32_hvx(const int n, float * GGML_RESTRICT z, const float * GGML_RESTRICT x, const float * GGML_RESTRICT y) {
    HVX_Vector * va;
    HVX_Vector * vb;
    HVX_Vector * vc;
    HVX_Vector qf32;
    HVX_Vector qfa;
    HVX_Vector qfb;

    const size_t FLOATS_PER_VECTOR = 128 / sizeof(float);
    const size_t block  = n / FLOATS_PER_VECTOR;
    const size_t left   = n % FLOATS_PER_VECTOR;
    const size_t blocks = block * FLOATS_PER_VECTOR;

    if ((((uintptr_t)z | (uintptr_t)x | (uintptr_t)y) % ALIGN_128_BYTE) != 0) {
        #pragma unroll
        for (size_t i = 0; i < n; ++i)
            z[i] = x[i] + y[i];

        return;
    }
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );

    va = (HVX_Vector *)x;
    vb = (HVX_Vector *)y;
    vc = (HVX_Vector *)z;

    int fetch_counts = 1;

    GGMLHEXAGON_LOG_DEBUG("fetch_counts %d", fetch_counts);
    for (size_t i = 0; i < block; i+= fetch_counts) {
        l2fetch(va + VLEN * fetch_counts, VLEN * fetch_counts, VLEN * fetch_counts, 1, 0);
        l2fetch(vb + VLEN * fetch_counts, VLEN * fetch_counts, VLEN * fetch_counts, 1, 0);

        //qfa = Q6_Vqf32_vadd_VsfVsf(*va++, Q6_V_vzero());
        //qfb = Q6_Vqf32_vadd_VsfVsf(*vb++, Q6_V_vzero());
        //qf32 = Q6_Vqf32_vadd_Vqf32Vqf32(qfa, qfb);
        #pragma unroll
        for (size_t j = 0; j < fetch_counts; j++) {
            qf32 = Q6_Vqf32_vadd_VsfVsf(*va++, *vb++);
            *vc++ = Q6_Vsf_equals_Vqf32(qf32);
        }
    }

    if (left > 0) {
        #pragma unroll
        for (size_t i = 0; i < left; i++) {
            z[i + blocks] = x[i + blocks] + y[i + blocks];
        }
    }
}

static inline void ggml_add_f32_hvx(const int n, float * GGML_RESTRICT z, const float * GGML_RESTRICT x, const float * GGML_RESTRICT y) {
    HVX_Vector * va;
    HVX_Vector * vb;
    HVX_Vector * vc;
    HVX_Vector qf32;
    HVX_Vector qfa;
    HVX_Vector qfb;
    const size_t FLOATS_PER_VECTOR = 128 / sizeof(float);
    const size_t block  = n / FLOATS_PER_VECTOR;
    const size_t left   = n % FLOATS_PER_VECTOR;
    const size_t blocks = block * FLOATS_PER_VECTOR;

    if ((((uintptr_t)z | (uintptr_t)x | (uintptr_t)y) % ALIGN_128_BYTE) != 0) {
        #pragma unroll
        for (size_t i = 0; i < n; ++i)
            z[i] = x[i] + y[i];

        return;
    }
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );

    va = (HVX_Vector *)x;
    vb = (HVX_Vector *)y;
    vc = (HVX_Vector *)z;

    int fetch_counts = 1;
    if (0 == (n % (128 * 8)))
        fetch_counts = 8;
    else if (0 == (n % (128 * 4)))
        fetch_counts = 4;
    else if (0 == (n % (128 * 2)))
        fetch_counts = 2;
    else
        fetch_counts = 1;

    GGMLHEXAGON_LOG_DEBUG("fetch_counts %d", fetch_counts);
    for (size_t i = 0; i < block; i += fetch_counts) {
        l2fetch((void*)((uint8_t*)va + VLEN * fetch_counts), VLEN, VLEN * fetch_counts, 1, 0);
        l2fetch((void*)((uint8_t*)vb + VLEN * fetch_counts), VLEN, VLEN * fetch_counts, 1, 0);

        //qfa = Q6_Vqf32_vadd_VsfVsf(*va++, Q6_V_vzero());
        //qfb = Q6_Vqf32_vadd_VsfVsf(*vb++, Q6_V_vzero());
        //qf32 = Q6_Vqf32_vadd_Vqf32Vqf32(qfa, qfb);
        #pragma unroll
        for (size_t j = 0; j < fetch_counts; j++) {
            qf32 = Q6_Vqf32_vadd_VsfVsf(*va++, *vb++);
            *vc++ = Q6_Vsf_equals_Vqf32(qf32);
        }
    }

    if (left > 0) {
        #pragma unroll
        for (size_t i = 0; i < left; i++) {
            z[i + blocks] = x[i + blocks] + y[i + blocks];
        }
    }
}

static void ggml_compute_forward_add_f32_me(
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );

    ggmlhexagon_dump_tensor(src0, 1);
    ggmlhexagon_dump_tensor(src1, 1);
    ggmlhexagon_dump_tensor(dst, 1);

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    const int ith = 0;
    const int nth = 1;

    const int rank_0 = ggml_n_dims(src0);
    const int rank_1 = ggml_n_dims(src1);
    //GGMLHEXAGON_LOG_INFO("rank_0  %d", rank_0);
    //GGMLHEXAGON_LOG_INFO("rank_1  %d", rank_1);
    //GGMLHEXAGON_LOG_INFO("src0->ne[1]  %d", src0->ne[1]);
    //GGMLHEXAGON_LOG_INFO("src1->ne[1]  %d", src1->ne[1]);
    //if (rank_0 == 3 && rank_1 == 3)
     //   return;

    if (rank_0 != 2 && rank_1 != 2) {
        const int len = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
        const int ne00 = src0->ne[0];
        GGMLHEXAGON_LOG_DEBUG("len  %d", len);
        //FARF(ALWAYS, "src0->ne[0] %d", src0->ne[0]);
        //if ((1 == rank_0) && (0 == len % 4096)) {
        if ((1 == src0->ne[1]) && (1 == src1->ne[1]) && (ne00 >= 384)) {
            float * dst_ptr  = (float *) (dst->data);
            float * src0_ptr = (float *) (src0->data);
            float * src1_ptr = (float *) (src1->data);
            ggml_vec_add_f32_hvx(ne00, dst_ptr, src0_ptr, src1_ptr);
        } else {
            float * dst_ptr  = (float *) (dst->data);
            float * src0_ptr = (float *) (src0->data);
            float * src1_ptr = (float *) (src1->data);
            for (size_t i = 0; i < len; ++i)
                dst_ptr[i] = src0_ptr[i] + src1_ptr[i];
        }
        return;
    }


    const int nr  = ggml_nrows(src0);

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT( nb0 == sizeof(float));
    GGML_ASSERT(nb00 == sizeof(float));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (nb10 == sizeof(float)) {
        for (int ir = ir0; ir < ir1; ++ir) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int64_t i03 = ir/(ne02*ne01);
            const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;
            const int64_t nr0 = ne00 / ne10;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
            float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);

            for (int64_t r = 0; r < nr0; ++r) {
                ggml_add_f32_hvx(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
                //ggml_add_f32_scale(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
            }
        }
    } else {
        // src1 is not contiguous
        for (int ir = ir0; ir < ir1; ++ir) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int64_t i03 = ir/(ne02*ne01);
            const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                const int64_t i10 = i0 % ne10;
                float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + i10*nb10);

                dst_ptr[i0] = src0_ptr[i0] + *src1_ptr;
            }
        }
    }
}

static void ggml_compute_forward_add_f32_multi(
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );
    uint64_t start_time = ggml_time_us();

    ggmlhexagon_dump_tensor(src0, 0);
    ggmlhexagon_dump_tensor(src1, 0);
    ggmlhexagon_dump_tensor(dst, 0);

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    const int ith = 0;
    const int nth = 1;

#if 1
    const int rank_0 = ggml_n_dims(src0);
    const int rank_1 = ggml_n_dims(src1);
    //GGMLHEXAGON_LOG_INFO("rank_0  %d", rank_0);
    //GGMLHEXAGON_LOG_INFO("rank_1  %d", rank_1);
    //GGMLHEXAGON_LOG_INFO("src0->ne[1]  %d", src0->ne[1]);
    //GGMLHEXAGON_LOG_INFO("src1->ne[1]  %d", src1->ne[1]);

    if (rank_0 != 2 && rank_1 != 2) {
        const int len = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
        GGMLHEXAGON_LOG_DEBUG("len  %d", len);
        float * dst_ptr  = (float *) (dst->data);
        float * src0_ptr = (float *) (src0->data);
        float * src1_ptr = (float *) (src1->data);
        for (size_t i = 0; i < len; ++i)
            dst_ptr[i] = src0_ptr[i] + src1_ptr[i];
        return;
    }

    if ((1 == src0->ne[1]) && (1 == src1->ne[1])) {
        const int len = src0->ne[0];
        GGMLHEXAGON_LOG_DEBUG("len  %d", len);
        float * dst_ptr  = (float *) (dst->data);
        float * src0_ptr = (float *) (src0->data);
        float * src1_ptr = (float *) (src1->data);
        ggml_vec_add_f32_hvx(len, dst_ptr, src0_ptr, src1_ptr);
        //for (size_t i = 0; i < len; ++i)
        //    dst_ptr[i] = src0_ptr[i] + src1_ptr[i];
        return;
    }
    //2025/06/07(June 07),fix a issue
    if (src0->ne[1] <= 2) {
        const int len = src0->ne[0];
        GGMLHEXAGON_LOG_DEBUG("len  %d", len);
        float * dst_ptr  = (float *) (dst->data);
        float * src0_ptr = (float *) (src0->data);
        float * src1_ptr = (float *) (src1->data);
        ggml_vec_add_f32_hvx(len, dst_ptr, src0_ptr, src1_ptr);
        //for (size_t i = 0; i < len; ++i)
        //    dst_ptr[i] = src0_ptr[i] + src1_ptr[i];
        return;
    }
#endif

    const int nr  = ggml_nrows(src0);
    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT( nb0 == sizeof(float));
    GGML_ASSERT(nb00 == sizeof(float));

    const int dr = (nr + nth - 1)/nth;
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);
    if (nb10 == sizeof(float)) {
        for (int ir = ir0; ir < ir1; ++ir) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int32_t i03 = ir/(ne02*ne01);
            const int32_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int32_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int32_t i13 = i03 % ne13;
            const int32_t i12 = i02 % ne12;
            const int32_t i11 = i01 % ne11;
            const int32_t nr0 = ne00 / ne10;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
            float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);
            for (int32_t r = 0; r < nr0; ++r) {
                ggml_add_f32_hvx(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
            }
        }
    } else {
        // src1 is not contiguous
        for (int ir = ir0; ir < ir1; ++ir) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int32_t i03 = ir/(ne02*ne01);
            const int32_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int32_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int32_t i13 = i03 % ne13;
            const int32_t i12 = i02 % ne12;
            const int32_t i11 = i01 % ne11;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

            for (int32_t i0 = 0; i0 < ne0; ++i0) {
                const int32_t i10 = i0 % ne10;
                float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + i10*nb10);

                dst_ptr[i0] = src0_ptr[i0] + *src1_ptr;
            }
        }
    }

    uint64_t end_time = ggml_time_us();
    uint64_t duration = (end_time - start_time);
    GGMLHEXAGON_LOG_DEBUG("computation duration %llu us", duration);
#if !GGMLHEXAGON_DEBUG
    UNUSED(duration);
#endif

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
}

static void add_thread_func(void * data) {
    add_thread_data_t * tdata = (add_thread_data_t *) data;
    const ggml_tensor * src0 = tdata->src0;
    const ggml_tensor * src1 = tdata->src1;
    ggml_tensor * dst = tdata->dst;
    const int64_t start_idx = tdata->start_idx;
    const int64_t end_idx = tdata->end_idx;

    if (src0->type == GGML_TYPE_F16) {
        uint16_t * dst_ptr  = (uint16_t *)dst->data;
        uint16_t * src0_ptr = (uint16_t *)src0->data;
        uint16_t * src1_ptr = (uint16_t *)src1->data;

        for (int64_t i = start_idx; i < end_idx; ++i) {
            float f0 = ggml_compute_fp16_to_fp32(src0_ptr[i]);
            float f1 = ggml_compute_fp16_to_fp32(src1_ptr[i]);
            float f_result = f0 + f1;
            dst_ptr[i] = ggml_compute_fp32_to_fp16(f_result);
        }
    } else {
        float * dst_ptr  = (float *)dst->data;
        float * src0_ptr = (float *)src0->data;
        float * src1_ptr = (float *)src1->data;

        const int rank_0 = ggml_n_dims(src0);
        const int rank_1 = ggml_n_dims(src1);

        if (rank_0 != 2 && rank_1 != 2) {
            for (int64_t i = start_idx; i < end_idx; ++i) {
                dst_ptr[i] = src0_ptr[i] + src1_ptr[i];
            }
        } else {
            const int n = end_idx - start_idx;
            const float * src0_ptr_offset = src0_ptr + start_idx;
            const float * src1_ptr_offset = src1_ptr + start_idx;
            float * dst_ptr_offset = dst_ptr + start_idx;

            if ((((uintptr_t)dst_ptr_offset | (uintptr_t)src0_ptr_offset | (uintptr_t)src1_ptr_offset) % ALIGN_128_BYTE) != 0 || n < 384) {
                for (int64_t i = start_idx; i < end_idx; ++i) {
                    dst_ptr[i] = src0_ptr[i] + src1_ptr[i];
                }
            } else {
                ggml_add_f32_hvx(n, dst_ptr_offset, src0_ptr_offset, src1_ptr_offset);
            }
        }
    }

    if (tdata->synctoken != NULL) {
        worker_pool_synctoken_jobdone(tdata->synctoken);
    }
}

static void ggml_compute_forward_add_f32_ai(
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );
    uint64_t start_time = ggml_time_us();

    ggmlhexagon_dump_tensor(src0, 1);
    ggmlhexagon_dump_tensor(src1, 1);
    ggmlhexagon_dump_tensor(dst, 1);

    const int n = ggml_nelements(src0);

    if (src0->type == GGML_TYPE_F16) {
        uint16_t * dst_ptr  = (uint16_t *)dst->data;
        uint16_t * src0_ptr = (uint16_t *)src0->data;
        uint16_t * src1_ptr = (uint16_t *)src1->data;

        for (int i = 0; i < n; ++i) {
            float f0 = ggml_compute_fp16_to_fp32(src0_ptr[i]);
            float f1 = ggml_compute_fp16_to_fp32(src1_ptr[i]);
            float f_result = f0 + f1;
            dst_ptr[i] = ggml_compute_fp32_to_fp16(f_result);
        }
    } else {
        float * dst_ptr  = (float *)dst->data;
        float * src0_ptr = (float *)src0->data;
        float * src1_ptr = (float *)src1->data;

        const int rank_0 = ggml_n_dims(src0);
        const int rank_1 = ggml_n_dims(src1);

        if (rank_0 != 2 && rank_1 != 2) {
            for (size_t i = 0; i < n; ++i)
                dst_ptr[i] = src0_ptr[i] + src1_ptr[i];
        } else {
            ggml_add_f32_hvx(n, dst_ptr, src0_ptr, src1_ptr);
        }
    }

    uint64_t end_time = ggml_time_us();
    uint64_t duration = (end_time - start_time);
    GGMLHEXAGON_LOG_DEBUG("duration %llu us", duration);
#if !GGMLHEXAGON_DEBUG
    UNUSED(duration);
#endif

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
}

static int ggmlop_dsp_add_singlethread(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );
    ggml_compute_forward_add_f32_ai(src0, src1, dst);
    //ggml_compute_forward_add_f32_me(src0, src1, dst);
    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return 0;
}

static int ggmlop_dsp_add_multithread(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );

    const int64_t n = ggml_nelements(src0);
    const int num_threads = num_workers;

    if (num_threads <= 1 || n < num_threads) {
        return ggmlop_dsp_add_singlethread(h, src0, src1, dst);
    }

    worker_synctoken_t synctoken;
    worker_pool_synctoken_init(&synctoken, num_threads - 1);

    add_thread_data_t tdata[num_threads];
    const int64_t ne_per_thread = n / num_threads;
    int64_t start_idx = 0;

    for (int i = 0; i < num_threads - 1; ++i) {
        int64_t end_idx = start_idx + ne_per_thread;
        tdata[i].src0 = src0;
        tdata[i].src1 = src1;
        tdata[i].dst = dst;
        tdata[i].start_idx = start_idx;
        tdata[i].end_idx = end_idx;
        tdata[i].synctoken = &synctoken;

        worker_pool_job_t job;
        job.fptr = add_thread_func;
        job.dptr = &tdata[i];
        worker_pool_submit(NULL, job);

        start_idx = end_idx;
    }

    tdata[num_threads - 1].src0 = src0;
    tdata[num_threads - 1].src1 = src1;
    tdata[num_threads - 1].dst = dst;
    tdata[num_threads - 1].start_idx = start_idx;
    tdata[num_threads - 1].end_idx = n;
    tdata[num_threads - 1].synctoken = NULL;

    add_thread_func(&tdata[num_threads - 1]);

    worker_pool_synctoken_wait(&synctoken);

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return 0;
}

int ggmlop_dsp_add(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s\n", __func__);

    if (ggmlop_get_thread_counts() > 1) {
        ggmlop_dsp_add_multithread(h, src0, src1, dst);
    } else {
        ggmlop_dsp_add_singlethread(h, src0, src1, dst);
    }
    GGMLHEXAGON_LOG_DEBUG("leave %s\n", __func__);
    return 0;
}
