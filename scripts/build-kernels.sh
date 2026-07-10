#!/usr/bin/env bash

# build DSP kernel of JZ's ggml-hexagon independently for purpose of simplify work flow

PROJECT_HOME_PATH=`pwd`
PROJECT_ROOT_PATH=${PROJECT_HOME_PATH}
LOCAL_BUILD_DIR=${PROJECT_ROOT_PATH}/out/ggmlhexagon-android
REMOTE_PATH=/data/local/tmp

TOOLCHAIN_PATH=${PROJECT_ROOT_PATH}/prebuilts
#build all supported DSP skel versions: v75, v79, v81
#v75 uses qf32 accumulator fallbacks (HVX_V*_F32 macros) for __HVX_ARCH__ < 79
#HTP_ARCH_VERSIONS="v75 v79 v81"
HTP_ARCH_VERSIONS="v79"
HEXAGON_SDK_VERSION=6.6.0.0
HEXAGON_TOOLS_VERSION=19.0.07
HEXAGON_SDK_PATH=${TOOLCHAIN_PATH}/Hexagon_SDK/${HEXAGON_SDK_VERSION}
HEXAGON_TOOLS_PATH=${HEXAGON_SDK_PATH}/tools/HEXAGON_Tools/${HEXAGON_TOOLS_VERSION}
DEBUG_FLAG="-DNDEBUG -Wall"
#DEBUG_FLAG=-g

# ==============================================================================
# Return codes:
#    0 = NO changes
#    1 = FILE CHANGED
# ==============================================================================
function is_so_file_changed() {
    set +e
    local so_file="$1"
    local md5_file="${so_file}.md5"

    # check if .so exists
    if [ ! -f "$so_file" ]; then
        echo "ERROR: File not found: $so_file"
        return 1
    fi

    # get current MD5
    local current_md5
    current_md5=$(md5sum "$so_file" | awk '{print $1}')

    # FIRST RUN: no MD5 file → save it, return CHANGED
    if [ ! -f "$md5_file" ]; then
        echo "$current_md5" > "$md5_file"
        echo "Initialized MD5 for $so_file"
        return 1
    fi

    # read previous MD5
    local last_md5
    last_md5=$(cat "$md5_file")

    # compare
    if [ "$current_md5" = "$last_md5" ]; then
        # NO CHANGE
        return 0
    else
        # CHANGED → update MD5
        echo "$current_md5" > "$md5_file"
        return 1
    fi
}


cd ${PROJECT_ROOT_PATH}
echo "${HEXAGON_SDK_PATH}/ipc/fastrpc/qaic/bin/qaic -mdll -o ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/kernels -I${HEXAGON_SDK_PATH}/incs -I${HEXAGON_SDK_PATH}/incs/stddef -I${HEXAGON_SDK_PATH}/ipc/fastrpc/incs ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/kernels/ggml_dsp.idl"
${HEXAGON_SDK_PATH}/ipc/fastrpc/qaic/bin/qaic -mdll -o ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/kernels -I${HEXAGON_SDK_PATH}/incs -I${HEXAGON_SDK_PATH}/incs/stddef -I${HEXAGON_SDK_PATH}/ipc/fastrpc/incs ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/kernels/ggml_dsp.idl

for HTP_ARCH_VERSION in ${HTP_ARCH_VERSIONS}; do
    TARGET=${LOCAL_BUILD_DIR}/bin/libggmldsp-skel-${HTP_ARCH_VERSION}.so
    printf "\n========== build libggmldsp-skel-${HTP_ARCH_VERSION}.so ==========\n"
    cd ${PROJECT_ROOT_PATH} && make -C ggml/src/ggml-hexagon/kernels/ clean && make -C ggml/src/ggml-hexagon/kernels/ HTP_ARCH_VERSION=${HTP_ARCH_VERSION} HEXAGON_SDK_PATH=${HEXAGON_SDK_PATH} HEXAGON_TOOLS_PATH=${HEXAGON_TOOLS_PATH} DEBUG_FLAG=${DEBUG_FLAG}
    /bin/cp -fv ggml/src/ggml-hexagon/kernels/libggmldsp-skel.so  ${TARGET}
    if [  -f ${TARGET} ]; then
        is_so_file_changed ${TARGET}
        if [ $? -eq 0 ]; then
            printf "${TARGET} not changed\n\n"
        else
            printf "${TARGET} has changed or first check\n\n"
            echo "adb push ${LOCAL_BUILD_DIR}/bin/libggmldsp-skel-${HTP_ARCH_VERSION}.so ${REMOTE_PATH}/libggmldsp-skel-${HTP_ARCH_VERSION}.so"
            adb push ${LOCAL_BUILD_DIR}/bin/libggmldsp-skel-${HTP_ARCH_VERSION}.so ${REMOTE_PATH}/libggmldsp-skel-${HTP_ARCH_VERSION}.so
        fi
    fi
done

cd ${PROJECT_ROOT_PATH}
