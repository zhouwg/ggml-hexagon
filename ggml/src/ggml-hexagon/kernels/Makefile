#following vars already defined in CMakeLists.txt
#HTP_ARCH_VERSION=v79
#DEBUG_FLAG=-DNDEBUG -Wall

HEXAGON_COMPUTE=compute${HTP_ARCH_VERSION}
HEXAGON_CC=${HEXAGON_TOOLS_PATH}/Tools/bin/hexagon-clang
HEXAGON_CXX=${HEXAGON_TOOLS_PATH}/Tools/bin/hexagon-clang

TARGET=libggmldsp-skel.so

$(info HEXAGON_SDK_PATH:${HEXAGON_SDK_PATH})
$(info HEXAGON_TOOLS_PATH:${HEXAGON_TOOLS_PATH})
$(info HTP_ARCH_VERSION:${HTP_ARCH_VERSION})
$(info DEBUG_FLAG:${DEBUG_FLAG})
$(info HEXAGON_COMPUTE:${HEXAGON_COMPUTE})

INCS=-I${HEXAGON_SDK_PATH}/incs -I${HEXAGON_SDK_PATH}/libs/qprintf/inc -I${HEXAGON_SDK_PATH}/incs/stddef -I${HEXAGON_SDK_PATH}/ipc/fastrpc/incs -I${HEXAGON_SDK_PATH}/ipc/fastrpc/rpcmem/inc -I${HEXAGON_SDK_PATH}/ipc/fastrpc/rtld/ship/inc -I${HEXAGON_SDK_PATH}/libs/atomic/inc -I${HEXAGON_SDK_PATH}/utils/sim_utils/inc -I${HEXAGON_SDK_PATH}/utils/sim_utils/inc -I${HEXAGON_SDK_PATH}/rtos/qurt/${HEXAGON_COMPUTE}/include/posix -I${HEXAGON_SDK_PATH}/rtos/qurt/${HEXAGON_COMPUTE}/include/qurt/ -I../htp -I../../ -I../../../include -I${HEXAGON_SDK_PATH}/tools/HEXAGON_Tools/8.8.06/Tools/target/hexagon/include

CFLAGS=-m${HTP_ARCH_VERSION} -c -Ofast -Wall -Wstrict-prototypes -fno-zero-initialized-in-bss -fdata-sections -fpic -D__V_DYNAMIC__ -mhvx -mhvx-length=128B -mhmx ${INCS} -fno-finite-math-only -fno-vectorize
CFLAGS+=-DGGMLDSP_VERSION="0.99.3"

LDFLAGS=-m${HTP_ARCH_VERSION} -Wl,--defsym=ISDB_TRUSTED_FLAG=2 -Wl,--defsym=ISDB_SECURE_FLAG=2 -Wl,--no-threads -fpic -shared -Wl,-Bsymbolic -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=free -Wl,--wrap=realloc -Wl,--wrap=memalign -lc -Wl,-soname=${TARGET}

# vpath: search ../htp for .c files not in kernels/
vpath %.c ../htp

# Only entry.c/ggml-dsp.c/skel.c from kernels/; all ops come from ../htp/
SRCS = ggml-dsp.c skel.c entry.c worker-pool.c hex-dma.c hmx-queue.c binary-ops.c unary-ops.c sum-rows-ops.c softmax-ops.c act-ops.c rope-ops.c set-rows-ops.c get-rows-ops.c cpy-ops.c repeat-ops.c argsort-ops.c ssm-conv.c cumsum-ops.c fill-ops.c concat-ops.c diag-ops.c solve-tri-ops.c gated-delta-net-ops.c pad-ops.c matmul-ops.c
OBJS = $(SRCS:.c=.o)

ALL:$(OBJS)
		${HEXAGON_CC} ${LDFLAGS} -o ${TARGET} -Wl,--start-group ${OBJS} -Wl,--end-group
		@ls -l ${TARGET}

%.o:%.c
		@echo "${HEXAGON_CC} ${CFLAGS} ${DEBUG_FLAG} -D__FILENAME__=\"$<\" -o $@ -c $<"
		${HEXAGON_CC} ${CFLAGS} ${DEBUG_FLAG} -D__FILENAME__=\"$<\" -o $@ -c $<
		@echo "\n"

%.o:%.S
		@echo "${HEXAGON_CC} ${CFLAGS} ${DEBUG_FLAG} -D__FILENAME__=\"$<\" -o $@ -c $<"
		${HEXAGON_CC} ${CFLAGS} ${DEBUG_FLAG} -D__FILENAME__=\"$<\" -o $@ -c $<
		@echo "\n"

%.o:%.cpp
		@echo "${HEXAGON_CC} ${CFLAGS} ${DEBUG_FLAG} -D__FILENAME__=\"$<\" -o $@ -c $<"
		${HEXAGON_CC} ${CFLAGS} ${DEBUG_FLAG} -D__FILENAME__=\"$<\" -o $@ -c $<
		@echo "\n"

clean:
	rm -f *.o
	/bin/rm -f *.so
