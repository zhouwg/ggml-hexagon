#ifndef GGML_HEXAGON_HMX_QUEUE_H
#define GGML_HEXAGON_HMX_QUEUE_H

// Async HMX queue for pipelining DMA / HVX / HMX execution.
// Ported from Qualcomm's htp/hmx-queue.{h,c} with trace instrumentation
// removed (we have no htp_thread_trace infrastructure) and a depth() bug fix.
//
// The queue runs a dedicated worker thread that holds the HMX hardware lock
// (HAP_compute_res_hmx_lock) for the lifetime of a batch of submitted tasks.
// Producers (the calling matmul thread) push descriptors and later pop them
// to wait for completion. While the HMX worker is running one tile, the
// producer can prepare the next tile (DMA + dequant) on the HVX worker pool,
// achieving DMA/HVX/HMX pipeline overlap.
//
// Note: GGMLHEXAGON_LOG_INFO/ERROR macros (from ggml-dsp.h) are used for
// logging. Include ggml-dsp.h before this header in each .c file.

#include <stdbool.h>
#include <stdint.h>
#include <stdatomic.h>

#include <hexagon_types.h>
#include <qurt_thread.h>
#include <qurt_futex.h>
#include <HAP_farf.h>

#include "hex-utils.h"  // hex_pause, hex_ceil_pow2

#ifdef __cplusplus
extern "C" {
#endif

#define HMX_QUEUE_THREAD_STACK_SIZE (16 * 1024)
#define HMX_QUEUE_POLL_COUNT        2000

typedef void (*hmx_queue_func)(void *);

// Dummy funcs used as signals
enum hmx_queue_signal {
    HMX_QUEUE_NOOP = 0, // aka NULL
    HMX_QUEUE_SUSPEND,
    HMX_QUEUE_KILL
};

struct hmx_queue_desc {
    hmx_queue_func   func;
    void *           data;
    atomic_uint      done;
};

struct hmx_queue {
    struct hmx_queue_desc * desc;
    atomic_uint      idx_write; // updated by producer (push)
    atomic_uint      idx_read;  // updated by consumer (process)
    unsigned int     idx_pop;   // updated by producer (pop)
    uint32_t         idx_mask;
    uint32_t         capacity;

    atomic_uint      seqn;      // incremented for all pushes, used with futex
    qurt_thread_t    thread;
    void *           stack;
    uint32_t         hap_rctx;
    bool             hmx_locked;
    volatile unsigned int worker_state; // 0=idle 1=locking 2=computing 3=done
};

struct hmx_queue * hmx_queue_create(size_t capacity, uint32_t hap_rctx);
void hmx_queue_delete(struct hmx_queue * q);

static inline struct hmx_queue_desc hmx_queue_make_desc(hmx_queue_func func, void * data) {
    struct hmx_queue_desc d = { func, data };
    return d;
}

static inline bool hmx_queue_push(struct hmx_queue * q, struct hmx_queue_desc d) {
    unsigned int ir = atomic_load(&q->idx_read);
    unsigned int iw = q->idx_write;

    if (((iw + 1) & q->idx_mask) == ir) {
        FARF(ALWAYS, "hmx-queue: push FAIL (queue full, iw=%u ir=%u)", iw, ir);
        return false;
    }

    atomic_store(&d.done, 0);

    // Write fields individually so the atomic_uint 'done' uses atomic_store
    // (a struct assignment like q->desc[iw] = d would bypass atomic semantics
    // and could leave a stale done=1 visible to the consumer on reuse)
    q->desc[iw].func = d.func;
    q->desc[iw].data = d.data;
    atomic_store(&q->desc[iw].done, 0);
    atomic_store(&q->idx_write, (iw + 1) & q->idx_mask);
    // wake up our thread
    atomic_fetch_add(&q->seqn, 1);
    qurt_futex_wake(&q->seqn, 1);

    return true;
}

static inline bool hmx_queue_signal(struct hmx_queue *q, enum hmx_queue_signal sig) {
    return hmx_queue_push(q, hmx_queue_make_desc((hmx_queue_func) sig, NULL));
}

static inline bool hmx_queue_empty(struct hmx_queue * q) {
    return q->idx_pop == q->idx_write;
}

static inline uint32_t hmx_queue_depth(struct hmx_queue * q) {
    // Fixed: was (idx_read - idx_read), which always returned 0.
    return (q->idx_write - q->idx_read) & q->idx_mask;
}

static inline uint32_t hmx_queue_capacity(struct hmx_queue * q) {
    return q->capacity;
}

static inline struct hmx_queue_desc hmx_queue_pop(struct hmx_queue * q) {
    unsigned int ip = q->idx_pop;
    unsigned int iw = q->idx_write;

    struct hmx_queue_desc rd = { NULL, NULL };
    if (ip == iw) {
        return rd;
    }

    // Wait for desc to complete
    struct hmx_queue_desc * d = &q->desc[ip];
    unsigned int wait_cnt = 0;
    while (!atomic_load(&d->done)) {
        if ((++wait_cnt % 1000000) == 0) {
            FARF(ALWAYS, "hmx-queue: pop STUCK ip=%u func=%p worker_state=%u hmx_locked=%d (waited %u iterations)",
                 ip, d->func, q->worker_state, q->hmx_locked, wait_cnt);
        }
        hex_pause();
    }

    rd = *d;
    q->idx_pop = (ip + 1) & q->idx_mask;

    return rd;
}

static inline void hmx_queue_flush(struct hmx_queue * q) {
    while (hmx_queue_pop(q).func != NULL) ;
}

static inline void hmx_queue_suspend(struct hmx_queue *q) {
    hmx_queue_signal(q, HMX_QUEUE_SUSPEND);
    hmx_queue_flush(q);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* GGML_HEXAGON_HMX_QUEUE_H */
