#ifndef ggmldsp_ops_h
#define ggmldsp_ops_h

#include <string.h>
#include <stdlib.h>
#include <remote.h>
#include <AEEStdDef.h>

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct dsptensor dsptensor;

struct dsptensor {
   int32_t type;
   int32_t ne[4];
   int32_t nb[4];
   int32_t op;
   int32_t op_params[16];
   int32_t flags;
   void * data;
   int data_len;
};

typedef struct dsp_op_desc dsp_op_desc;
struct dsp_op_desc {
   int32_t opcode;
   int32_t params[16];
   int32_t src0_idx;
   int32_t src1_idx;
   int32_t src2_idx;
   int32_t src3_idx;
   int32_t dst_idx;
};

typedef struct dsp_opbatch_req dsp_opbatch_req;
struct dsp_opbatch_req {
   int32_t n_tensors;
   int32_t n_ops;
   dsptensor* tensors;
   int tensors_len;
   dsp_op_desc* ops;
   int ops_len;
};

int ggmlop_dsp_sub(remote_handle64 _h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) ;
int gggmlop_dsp_mul(remote_handle64 _h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) ;
int gggmlop_dsp_div(remote_handle64 _h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) ;
int gggmlop_dsp_rmsnorm(remote_handle64 _h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) ;
int gggmlop_dsp_rope(remote_handle64 _h, const dsptensor* src0, const dsptensor* src1, const dsptensor* src2, dsptensor* dst) ;
int gggmlop_dsp_softmax(remote_handle64 _h, const dsptensor* src0, const dsptensor* src1, const dsptensor* src2, dsptensor* dst) ;
int gggmlop_dsp_silu(remote_handle64 _h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) ;
int gggmlop_dsp_scale(remote_handle64 _h, const dsptensor* src0, dsptensor* dst) ;
int gggmlop_dsp_cpy(remote_handle64 _h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) ;
int gggmlop_dsp_getrows(remote_handle64 _h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) ;
int gggmlop_dsp_concat(remote_handle64 _h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) ;
int gggmlop_dsp_repeat(remote_handle64 _h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) ;
int gggmlop_dsp_diag_mask_inf(remote_handle64 _h, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) ;
int ggmlop_dsp_flash_attn(remote_handle64 h, const dsptensor * q, const dsptensor * k, const dsptensor * v, const dsptensor * mask, dsptensor * dst);

#ifdef  __cplusplus
}
#endif

#endif /* ggmldsp_ops_h */
