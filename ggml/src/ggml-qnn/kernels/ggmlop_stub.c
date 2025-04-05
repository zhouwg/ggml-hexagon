#ifndef _GGMLOP_STUB_H
#define _GGMLOP_STUB_H
/// @file ggmlop.idl
///
//qidl copyright
//qidl nested=false
#include "ggmlop.h"
#include <string.h>
#ifndef _WIN32
#include "HAP_farf.h"
#include <inttypes.h>
#endif //_WIN32 for HAP_farf
#ifndef _ALLOCATOR_H
#define _ALLOCATOR_H

#include <stdlib.h>
#include <stdint.h>

typedef struct _heap _heap;
struct _heap {
   _heap* pPrev;
   const char* loc;
   uint64_t buf;
};

typedef struct _allocator {
   _heap* pheap;
   uint8_t* stack;
   uint8_t* stackEnd;
   int nSize;
} _allocator;

_ATTRIBUTE_UNUSED
static __inline int _heap_alloc(_heap** ppa, const char* loc, int size, void** ppbuf) {
   _heap* pn = 0;
   pn = MALLOC((size_t)size + sizeof(_heap) - sizeof(uint64_t));
   if(pn != 0) {
      pn->pPrev = *ppa;
      pn->loc = loc;
      *ppa = pn;
      *ppbuf = (void*)&(pn->buf);
      return 0;
   } else {
      return -1;
   }
}
#define _ALIGN_SIZE(x, y) (((x) + (y-1)) & ~(y-1))

_ATTRIBUTE_UNUSED
static __inline int _allocator_alloc(_allocator* me,
                                    const char* loc,
                                    int size,
                                    unsigned int al,
                                    void** ppbuf) {
   if(size < 0) {
      return -1;
   } else if (size == 0) {
      *ppbuf = 0;
      return 0;
   }
   if((_ALIGN_SIZE((uintptr_t)me->stackEnd, al) + (size_t)size) < (uintptr_t)me->stack + (size_t)me->nSize) {
      *ppbuf = (uint8_t*)_ALIGN_SIZE((uintptr_t)me->stackEnd, al);
      me->stackEnd = (uint8_t*)_ALIGN_SIZE((uintptr_t)me->stackEnd, al) + size;
      return 0;
   } else {
      return _heap_alloc(&me->pheap, loc, size, ppbuf);
   }
}

_ATTRIBUTE_UNUSED
static __inline void _allocator_deinit(_allocator* me) {
   _heap* pa = me->pheap;
   while(pa != 0) {
      _heap* pn = pa;
      const char* loc = pn->loc;
      (void)loc;
      pa = pn->pPrev;
      FREE(pn);
   }
}

_ATTRIBUTE_UNUSED
static __inline void _allocator_init(_allocator* me, uint8_t* stack, int stackSize) {
   me->stack =  stack;
   me->stackEnd =  stack + stackSize;
   me->nSize = stackSize;
   me->pheap = 0;
}


#endif // _ALLOCATOR_H

#ifndef SLIM_H
#define SLIM_H

#include <stdint.h>

//a C data structure for the idl types that can be used to implement
//static and dynamic language bindings fairly efficiently.
//
//the goal is to have a minimal ROM and RAM footprint and without
//doing too many allocations.  A good way to package these things seemed
//like the module boundary, so all the idls within  one module can share
//all the type references.


#define PARAMETER_IN       0x0
#define PARAMETER_OUT      0x1
#define PARAMETER_INOUT    0x2
#define PARAMETER_ROUT     0x3
#define PARAMETER_INROUT   0x4

//the types that we get from idl
#define TYPE_OBJECT             0x0
#define TYPE_INTERFACE          0x1
#define TYPE_PRIMITIVE          0x2
#define TYPE_ENUM               0x3
#define TYPE_STRING             0x4
#define TYPE_WSTRING            0x5
#define TYPE_STRUCTURE          0x6
#define TYPE_UNION              0x7
#define TYPE_ARRAY              0x8
#define TYPE_SEQUENCE           0x9

//these require the pack/unpack to recurse
//so it's a hint to those languages that can optimize in cases where
//recursion isn't necessary.
#define TYPE_COMPLEX_STRUCTURE  (0x10 | TYPE_STRUCTURE)
#define TYPE_COMPLEX_UNION      (0x10 | TYPE_UNION)
#define TYPE_COMPLEX_ARRAY      (0x10 | TYPE_ARRAY)
#define TYPE_COMPLEX_SEQUENCE   (0x10 | TYPE_SEQUENCE)


typedef struct Type Type;

#define INHERIT_TYPE\
   int32_t nativeSize;                /*in the simple case its the same as wire size and alignment*/\
   union {\
      struct {\
         const uintptr_t         p1;\
         const uintptr_t         p2;\
      } _cast;\
      struct {\
         uint32_t  iid;\
         uint32_t  bNotNil;\
      } object;\
      struct {\
         const Type  *arrayType;\
         int32_t      nItems;\
      } array;\
      struct {\
         const Type *seqType;\
         int32_t      nMaxLen;\
      } seqSimple; \
      struct {\
         uint32_t bFloating;\
         uint32_t bSigned;\
      } prim; \
      const SequenceType* seqComplex;\
      const UnionType  *unionType;\
      const StructType *structType;\
      int32_t         stringMaxLen;\
      uint8_t        bInterfaceNotNil;\
   } param;\
   uint8_t    type;\
   uint8_t    nativeAlignment\

typedef struct UnionType UnionType;
typedef struct StructType StructType;
typedef struct SequenceType SequenceType;
struct Type {
   INHERIT_TYPE;
};

struct SequenceType {
   const Type *         seqType;
   uint32_t               nMaxLen;
   uint32_t               inSize;
   uint32_t               routSizePrimIn;
   uint32_t               routSizePrimROut;
};

//byte offset from the start of the case values for
//this unions case value array.  it MUST be aligned
//at the alignment requrements for the descriptor
//
//if negative it means that the unions cases are
//simple enumerators, so the value read from the descriptor
//can be used directly to find the correct case
typedef union CaseValuePtr CaseValuePtr;
union CaseValuePtr {
   const uint8_t*   value8s;
   const uint16_t*  value16s;
   const uint32_t*  value32s;
   const uint64_t*  value64s;
};

//these are only used in complex cases
//so I pulled them out of the type definition as references to make
//the type smaller
struct UnionType {
   const Type           *descriptor;
   uint32_t               nCases;
   const CaseValuePtr   caseValues;
   const Type * const   *cases;
   int32_t               inSize;
   int32_t               routSizePrimIn;
   int32_t               routSizePrimROut;
   uint8_t                inAlignment;
   uint8_t                routAlignmentPrimIn;
   uint8_t                routAlignmentPrimROut;
   uint8_t                inCaseAlignment;
   uint8_t                routCaseAlignmentPrimIn;
   uint8_t                routCaseAlignmentPrimROut;
   uint8_t                nativeCaseAlignment;
   uint8_t              bDefaultCase;
};

struct StructType {
   uint32_t               nMembers;
   const Type * const   *members;
   int32_t               inSize;
   int32_t               routSizePrimIn;
   int32_t               routSizePrimROut;
   uint8_t                inAlignment;
   uint8_t                routAlignmentPrimIn;
   uint8_t                routAlignmentPrimROut;
};

typedef struct Parameter Parameter;
struct Parameter {
   INHERIT_TYPE;
   uint8_t    mode;
   uint8_t  bNotNil;
};

#define SLIM_IFPTR32(is32,is64) (sizeof(uintptr_t) == 4 ? (is32) : (is64))
#define SLIM_SCALARS_IS_DYNAMIC(u) (((u) & 0x00ffffff) == 0x00ffffff)

typedef struct Method Method;
struct Method {
   uint32_t                    uScalars;            //no method index
   int32_t                     primInSize;
   int32_t                     primROutSize;
   int                         maxArgs;
   int                         numParams;
   const Parameter * const     *params;
   uint8_t                       primInAlignment;
   uint8_t                       primROutAlignment;
};

typedef struct Interface Interface;

struct Interface {
   int                            nMethods;
   const Method  * const          *methodArray;
   int                            nIIds;
   const uint32_t                   *iids;
   const uint16_t*                  methodStringArray;
   const uint16_t*                  methodStrings;
   const char*                    strings;
};


#endif //SLIM_H


#ifndef _GGMLOP_SLIM_H
#define _GGMLOP_SLIM_H
#include <stdint.h>

#ifndef __QAIC_SLIM
#define __QAIC_SLIM(ff) ff
#endif
#ifndef __QAIC_SLIM_EXPORT
#define __QAIC_SLIM_EXPORT
#endif

static const Type types[5];
static const Type* const typeArrays[5] = {&(types[0]),&(types[0]),&(types[2]),&(types[2]),&(types[3])};
static const StructType structTypes[1] = {{0x5,&(typeArrays[0]),0x50,0x4,0x48,0x8,0x4,0x8}};
static const Type types[5] = {{0x20,{{(const uintptr_t)&(types[1]),(const uintptr_t)0x4}}, 8,0x8},{0x8,{{(const uintptr_t)0,(const uintptr_t)1}}, 2,0x8},{0x4,{{(const uintptr_t)0,(const uintptr_t)1}}, 2,0x4},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)&(types[4]),(const uintptr_t)0x0}}, 9,SLIM_IFPTR32(0x4,0x8)},{0x4,{{(const uintptr_t)0,(const uintptr_t)1}}, 2,0x4}};
static const Parameter parameters[5] = {{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)0x0,0}}, 4,SLIM_IFPTR32(0x4,0x8),0,0},{SLIM_IFPTR32(0x4,0x8),{{(const uintptr_t)0xdeadc0de,(const uintptr_t)0}}, 0,SLIM_IFPTR32(0x4,0x8),3,0},{SLIM_IFPTR32(0x4,0x8),{{(const uintptr_t)0xdeadc0de,(const uintptr_t)0}}, 0,SLIM_IFPTR32(0x4,0x8),0,0},{SLIM_IFPTR32(0x50,0x58),{{(const uintptr_t)&(structTypes[0]),0}}, 22,0x8,0,0},{SLIM_IFPTR32(0x50,0x58),{{(const uintptr_t)&(structTypes[0]),0}}, 22,0x8,3,0}};
static const Parameter* const parameterArrays[6] = {(&(parameters[3])),(&(parameters[3])),(&(parameters[4])),(&(parameters[0])),(&(parameters[1])),(&(parameters[2]))};
static const Method methods[3] = {{REMOTE_SCALARS_MAKEX(0,0,0x2,0x0,0x0,0x1),0x4,0x0,2,2,(&(parameterArrays[3])),0x4,0x1},{REMOTE_SCALARS_MAKEX(0,0,0x0,0x0,0x1,0x0),0x0,0x0,1,1,(&(parameterArrays[5])),0x1,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x3,0x2,0x0,0x0),0xa4,0x48,3,3,(&(parameterArrays[0])),0x8,0x8}};
static const Method* const methodArrays[4] = {&(methods[0]),&(methods[1]),&(methods[2]),&(methods[2])};
static const char strings[65] = "mulmat\0flags\0close\0src1\0data\0type\0src0\0open\0dst\0add\0uri\0nb\0ne\0h\0";
static const uint16_t methodStrings[43] = {0,34,59,56,7,29,24,19,59,56,7,29,24,44,59,56,7,29,24,48,34,59,56,7,29,24,19,59,56,7,29,24,44,59,56,7,29,24,39,52,62,13,62};
static const uint16_t methodStringsArrays[4] = {38,41,19,0};
__QAIC_SLIM_EXPORT const Interface __QAIC_SLIM(ggmlop_slim) = {4,&(methodArrays[0]),0,0,&(methodStringsArrays [0]),methodStrings,strings};
#endif //_GGMLOP_SLIM_H


#ifdef __cplusplus
extern "C" {
#endif
__QAIC_STUB_EXPORT int __QAIC_STUB(ggmlop_open)(const char* uri, remote_handle64* h) __QAIC_STUB_ATTRIBUTE {
   return __QAIC_REMOTE(remote_handle64_open)(uri, h);
}
__QAIC_STUB_EXPORT int __QAIC_STUB(ggmlop_close)(remote_handle64 h) __QAIC_STUB_ATTRIBUTE {
   return __QAIC_REMOTE(remote_handle64_close)(h);
}
static __inline int _stub_unpack(_ATTRIBUTE_UNUSED remote_arg* _praROutPost, _ATTRIBUTE_UNUSED remote_arg* _ppraROutPost[1], _ATTRIBUTE_UNUSED void* _primROut, _ATTRIBUTE_UNUSED uint64_t _rout0[4], _ATTRIBUTE_UNUSED uint64_t _rout1[4], _ATTRIBUTE_UNUSED uint32_t _rout2[1], _ATTRIBUTE_UNUSED uint32_t _rout3[1], _ATTRIBUTE_UNUSED char* _rout4[1], _ATTRIBUTE_UNUSED uint32_t _rout4Len[1]) {
   int _nErr = 0;
   remote_arg* _praROutPostStart = _praROutPost;
   remote_arg** _ppraROutPostStart = _ppraROutPost;
   _ppraROutPost = &_praROutPost;
   _COPY(_rout0, 0, _primROut, 0, 32);
   _COPY(_rout1, 0, _primROut, 32, 32);
   _COPY(_rout2, 0, _primROut, 64, 4);
   _COPY(_rout3, 0, _primROut, 68, 4);
   _ppraROutPostStart[0] += (_praROutPost - _praROutPostStart) +1;
   return _nErr;
}
static __inline int _stub_pack(_ATTRIBUTE_UNUSED _allocator* _al, _ATTRIBUTE_UNUSED remote_arg* _praIn, _ATTRIBUTE_UNUSED remote_arg* _ppraIn[1], _ATTRIBUTE_UNUSED remote_arg* _praROut, _ATTRIBUTE_UNUSED remote_arg* _ppraROut[1], _ATTRIBUTE_UNUSED remote_arg* _praHIn, _ATTRIBUTE_UNUSED remote_arg* _ppraHIn[1], _ATTRIBUTE_UNUSED remote_arg* _praHROut, _ATTRIBUTE_UNUSED remote_arg* _ppraHROut[1], _ATTRIBUTE_UNUSED void* _primIn, _ATTRIBUTE_UNUSED void* _primROut, _ATTRIBUTE_UNUSED uint64_t _rout0[4], _ATTRIBUTE_UNUSED uint64_t _rout1[4], _ATTRIBUTE_UNUSED uint32_t _rout2[1], _ATTRIBUTE_UNUSED uint32_t _rout3[1], _ATTRIBUTE_UNUSED char* _rout4[1], _ATTRIBUTE_UNUSED uint32_t _rout4Len[1]) {
   int _nErr = 0;
   remote_arg* _praInStart = _praIn;
   remote_arg** _ppraInStart = _ppraIn;
   remote_arg* _praROutStart = _praROut;
   remote_arg** _ppraROutStart = _ppraROut;
   _ppraIn = &_praIn;
   _ppraROut = &_praROut;
   _COPY(_primIn, 0, _rout4Len, 0, 4);
   _praROut[0].buf.pv = _rout4[0];
   _praROut[0].buf.nLen = (4 * _rout4Len[0]);
   _ppraInStart[0] += (_praIn - _praInStart) + 0;
   _ppraROutStart[0] += (_praROut - _praROutStart) +1;
   return _nErr;
}
static __inline int _stub_pack_1(_ATTRIBUTE_UNUSED _allocator* _al, _ATTRIBUTE_UNUSED remote_arg* _praIn, _ATTRIBUTE_UNUSED remote_arg* _ppraIn[1], _ATTRIBUTE_UNUSED remote_arg* _praROut, _ATTRIBUTE_UNUSED remote_arg* _ppraROut[1], _ATTRIBUTE_UNUSED remote_arg* _praHIn, _ATTRIBUTE_UNUSED remote_arg* _ppraHIn[1], _ATTRIBUTE_UNUSED remote_arg* _praHROut, _ATTRIBUTE_UNUSED remote_arg* _ppraHROut[1], _ATTRIBUTE_UNUSED void* _primIn, _ATTRIBUTE_UNUSED void* _primROut, _ATTRIBUTE_UNUSED uint64_t _in0[4], _ATTRIBUTE_UNUSED uint64_t _in1[4], _ATTRIBUTE_UNUSED uint32_t _in2[1], _ATTRIBUTE_UNUSED uint32_t _in3[1], _ATTRIBUTE_UNUSED char* _in4[1], _ATTRIBUTE_UNUSED uint32_t _in4Len[1]) {
   int _nErr = 0;
   remote_arg* _praInStart = _praIn;
   remote_arg** _ppraInStart = _ppraIn;
   remote_arg* _praROutStart = _praROut;
   remote_arg** _ppraROutStart = _ppraROut;
   _ppraIn = &_praIn;
   _ppraROut = &_praROut;
   _COPY(_primIn, 0, _in0, 0, 32);
   _COPY(_primIn, 32, _in1, 0, 32);
   _COPY(_primIn, 64, _in2, 0, 4);
   _COPY(_primIn, 68, _in3, 0, 4);
   _COPY(_primIn, 72, _in4Len, 0, 4);
   _praIn[0].buf.pv = (void*) _in4[0];
   _praIn[0].buf.nLen = (4 * _in4Len[0]);
   _ppraInStart[0] += (_praIn - _praInStart) + 1;
   _ppraROutStart[0] += (_praROut - _praROutStart) +0;
   return _nErr;
}
static __inline void _count(int _numIn[1], int _numROut[1], int _numInH[1], int _numROutH[1], _ATTRIBUTE_UNUSED uint64_t _rout0[4], _ATTRIBUTE_UNUSED uint64_t _rout1[4], _ATTRIBUTE_UNUSED uint32_t _rout2[1], _ATTRIBUTE_UNUSED uint32_t _rout3[1], _ATTRIBUTE_UNUSED char* _rout4[1], _ATTRIBUTE_UNUSED uint32_t _rout4Len[1]) {
   _numIn[0] += 0;
   _numROut[0] += 1;
   _numInH[0] += 0;
   _numROutH[0] += 0;
}
static __inline void _count_1(int _numIn[1], int _numROut[1], int _numInH[1], int _numROutH[1], _ATTRIBUTE_UNUSED uint64_t _in0[4], _ATTRIBUTE_UNUSED uint64_t _in1[4], _ATTRIBUTE_UNUSED uint32_t _in2[1], _ATTRIBUTE_UNUSED uint32_t _in3[1], _ATTRIBUTE_UNUSED char* _in4[1], _ATTRIBUTE_UNUSED uint32_t _in4Len[1]) {
   _numIn[0] += 1;
   _numROut[0] += 0;
   _numInH[0] += 0;
   _numROutH[0] += 0;
}
static __inline int _stub_method(remote_handle64 _handle, uint32_t _mid, uint64_t _in0[SLIM_IFPTR32(10, 11)], uint64_t _in1[SLIM_IFPTR32(10, 11)], uint64_t _rout2[SLIM_IFPTR32(10, 11)]) {
   remote_arg* _pra = 0;
   int _numIn[1] = {0};
   int _numROut[1] = {0};
   int _numInH[1] = {0};
   int _numROutH[1] = {0};
   _allocator _al[1] = {{0}};
   uint64_t _primIn[21]= {0};
   uint64_t _primROut[9]= {0};
   remote_arg* _praIn = 0;
   remote_arg* _praROut = 0;
   remote_arg* _praROutPost = 0;
   remote_arg** _ppraROutPost = &_praROutPost;
   remote_arg** _ppraIn = &_praIn;
   remote_arg** _ppraROut = &_praROut;
   remote_arg* _praHIn = 0;
   remote_arg** _ppraHIn = &_praHIn;
   remote_arg* _praHROut = 0;
   remote_arg** _ppraHROut = &_praHROut;
   int _nErr = 0;
   _numIn[0] = 0;
   _numROut[0] = 0;
   _numInH[0] = 0;
   _numROutH[0] = 0;
   _count_1(_numIn, _numROut, _numInH, _numROutH, (uint64_t*)&(((uint64_t*)_in0)[0]), (uint64_t*)&(((uint64_t*)_in0)[4]), (uint32_t*)&(((uint32_t*)_in0)[16]), (uint32_t*)&(((uint32_t*)_in0)[17]), SLIM_IFPTR32((char**)&(((uint32_t*)_in0)[18]), (char**)&(((uint64_t*)_in0)[9])), SLIM_IFPTR32((uint32_t*)&(((uint32_t*)_in0)[19]), (uint32_t*)&(((uint32_t*)_in0)[20])));
   _count_1(_numIn, _numROut, _numInH, _numROutH, (uint64_t*)&(((uint64_t*)_in1)[0]), (uint64_t*)&(((uint64_t*)_in1)[4]), (uint32_t*)&(((uint32_t*)_in1)[16]), (uint32_t*)&(((uint32_t*)_in1)[17]), SLIM_IFPTR32((char**)&(((uint32_t*)_in1)[18]), (char**)&(((uint64_t*)_in1)[9])), SLIM_IFPTR32((uint32_t*)&(((uint32_t*)_in1)[19]), (uint32_t*)&(((uint32_t*)_in1)[20])));
   _count(_numIn, _numROut, _numInH, _numROutH, (uint64_t*)&(((uint64_t*)_rout2)[0]), (uint64_t*)&(((uint64_t*)_rout2)[4]), (uint32_t*)&(((uint32_t*)_rout2)[16]), (uint32_t*)&(((uint32_t*)_rout2)[17]), SLIM_IFPTR32((char**)&(((uint32_t*)_rout2)[18]), (char**)&(((uint64_t*)_rout2)[9])), SLIM_IFPTR32((uint32_t*)&(((uint32_t*)_rout2)[19]), (uint32_t*)&(((uint32_t*)_rout2)[20])));
   if(_numIn[0]>=255){
          _QAIC_FARF(RUNTIME_ERROR, "ERROR: Unsupported number of input buffers\n");
          return AEE_EUNSUPPORTED;
   }
   if(_numROut[0]>=255){
          _QAIC_FARF(RUNTIME_ERROR, "ERROR: Unsupported number of output buffers\n");
          return AEE_EUNSUPPORTED;
   }
   _allocator_init(_al, 0, 0);
   _QAIC_ALLOCATE(_nErr, _al, ((((((((_numIn[0] + _numROut[0]) + _numInH[0]) + _numROutH[0]) + 1) + 1) + 0) + 0) * sizeof(_pra[0])), 4, _pra);
   _QAIC_ASSERT(_nErr, _pra);
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _pra[(_numIn[0] + 1)].buf.pv = (void*)_primROut;
   _pra[(_numIn[0] + 1)].buf.nLen = sizeof(_primROut);
   _praIn = (_pra + 1);
   _praROut = (_praIn + _numIn[0] + 1);
   _praROutPost = _praROut;
   if(_praHIn == 0)
   {
      _praHIn = ((_praROut + _numROut[0]) + 1);
   }
   if(_praHROut == 0)
      (_praHROut = _praHIn + _numInH[0] + 0);
   _TRY(_nErr, _stub_pack_1(_al, (_praIn + 0), _ppraIn, (_praROut + 0), _ppraROut, _praHIn, _ppraHIn, _praHROut, _ppraHROut, ((char*)_primIn + 0), 0, (uint64_t*)&(((uint64_t*)_in0)[0]), (uint64_t*)&(((uint64_t*)_in0)[4]), (uint32_t*)&(((uint32_t*)_in0)[16]), (uint32_t*)&(((uint32_t*)_in0)[17]), SLIM_IFPTR32((char**)&(((uint32_t*)_in0)[18]), (char**)&(((uint64_t*)_in0)[9])), SLIM_IFPTR32((uint32_t*)&(((uint32_t*)_in0)[19]), (uint32_t*)&(((uint32_t*)_in0)[20]))));
   _TRY(_nErr, _stub_pack_1(_al, (_praIn + 0), _ppraIn, (_praROut + 0), _ppraROut, _praHIn, _ppraHIn, _praHROut, _ppraHROut, ((char*)_primIn + 80), 0, (uint64_t*)&(((uint64_t*)_in1)[0]), (uint64_t*)&(((uint64_t*)_in1)[4]), (uint32_t*)&(((uint32_t*)_in1)[16]), (uint32_t*)&(((uint32_t*)_in1)[17]), SLIM_IFPTR32((char**)&(((uint32_t*)_in1)[18]), (char**)&(((uint64_t*)_in1)[9])), SLIM_IFPTR32((uint32_t*)&(((uint32_t*)_in1)[19]), (uint32_t*)&(((uint32_t*)_in1)[20]))));
   _TRY(_nErr, _stub_pack(_al, (_praIn + 0), _ppraIn, (_praROut + 0), _ppraROut, _praHIn, _ppraHIn, _praHROut, _ppraHROut, ((char*)_primIn + 160), ((char*)_primROut + 0), (uint64_t*)&(((uint64_t*)_rout2)[0]), (uint64_t*)&(((uint64_t*)_rout2)[4]), (uint32_t*)&(((uint32_t*)_rout2)[16]), (uint32_t*)&(((uint32_t*)_rout2)[17]), SLIM_IFPTR32((char**)&(((uint32_t*)_rout2)[18]), (char**)&(((uint64_t*)_rout2)[9])), SLIM_IFPTR32((uint32_t*)&(((uint32_t*)_rout2)[19]), (uint32_t*)&(((uint32_t*)_rout2)[20]))));
   _QAIC_ASSERT(_nErr, (_numInH[0] + 0) <= 15);
   _QAIC_ASSERT(_nErr, (_numROutH[0] + 0) <= 15);
   _TRY_FARF(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, (_numIn[0] + 1), (_numROut[0] + 1), (_numInH[0] + 0), (_numROutH[0] + 0)), _pra));
   _TRY(_nErr, _stub_unpack((_praROutPost + 0), _ppraROutPost, ((char*)_primROut + 0), (uint64_t*)&(((uint64_t*)_rout2)[0]), (uint64_t*)&(((uint64_t*)_rout2)[4]), (uint32_t*)&(((uint32_t*)_rout2)[16]), (uint32_t*)&(((uint32_t*)_rout2)[17]), SLIM_IFPTR32((char**)&(((uint32_t*)_rout2)[18]), (char**)&(((uint64_t*)_rout2)[9])), SLIM_IFPTR32((uint32_t*)&(((uint32_t*)_rout2)[19]), (uint32_t*)&(((uint32_t*)_rout2)[20]))));
   _QAIC_CATCH(_nErr) {}
   _CATCH_FARF(_nErr) {
      _QAIC_FARF(RUNTIME_ERROR, "ERROR 0x%x: handle=0x%"PRIx64", scalar=0x%x, method ID=%d: %s failed\n", _nErr , _handle, REMOTE_SCALARS_MAKEX(0, _mid, (_numIn[0] + 1), (_numROut[0] + 1), (_numInH[0] + 0), (_numROutH[0] + 0)), _mid, __func__);
   }
   _allocator_deinit(_al);
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(ggmlop_add)(remote_handle64 _handle, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 2;
   return _stub_method(_handle, _mid, (uint64_t*)src0, (uint64_t*)src1, (uint64_t*)dst);
}
__QAIC_STUB_EXPORT int __QAIC_STUB(ggmlop_mulmat)(remote_handle64 _handle, const dsptensor* src0, const dsptensor* src1, dsptensor* dst) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 3;
   return _stub_method(_handle, _mid, (uint64_t*)src0, (uint64_t*)src1, (uint64_t*)dst);
}
#ifdef __cplusplus
}
#endif
#endif //_GGMLOP_STUB_H
