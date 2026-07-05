#!/usr/bin/env bash
#
# this self-contained file is part of JZ's ggml-hexagon:
#
# this script will setup local dev envs automatically and docker is not needed for purpose of simplify workflow.
#
# this script is AI Agent friendly and verified with Trae AI Agent.
#
# 1. build&verify llama.cpp + JZ's ggml-hexagon backend(libggmldsp-skel.so) on Linux for Android phone equipped with Qualcomm Snapdragon mobile SoC(8Elite is recommended)
#
# 2. build&verify llama.cpp + Qualcomm's ggml-hexagon backend(libggml-htp.so) on Linux for Android phone equipped with Qualcomm Snapdragon mobile SoC(8Elite is recommended)
#
# 3. performance comparison of Qualcomm's ggml-hexagon and JZ's ggml-hexagon on Android phone equipped with Qualcomm Snapdragon mobile SoC(8Elite is recommended)
#
# Jeff Zhou - zhouwg2000@gmail.com
# GitHub:   - https://github.com/zhouwg/ggml-hexagon
#
set -e

######## part-1: public macros & vars ########

PWD=`pwd`
PROJECT_HOME_PATH=`pwd`
PROJECT_ROOT_PATH=${PROJECT_HOME_PATH}
HOST_CPU_COUNTS=`cat /proc/cpuinfo | grep "processor" | wc | awk '{print int($1)}'`

VERBOSE=OFF
VERBOSE=ON
default_mulmat_algotype=32

#running path on Android phone
REMOTE_PATH=/data/local/tmp

#path of built artifacts
LOCAL_BUILD_DIR=${PROJECT_ROOT_PATH}/out/ggmlhexagon-android

#path of toolchain, for purpose of share same toolchain in multiple instance of JZ's ggml-hexagon
TOOLCHAIN_PATH=${PROJECT_ROOT_PATH}/prebuilts
#TOOLCHAIN_PATH=/home/zhouwg/develop/ggml-hexagon/prebuilts

#Android NDK can be found at:
#https://developer.android.com/ndk/downloads
ANDROID_PLATFORM=android-34
ANDROID_NDK_VERSION=r28
ANDROID_NDK_NAME=android-ndk-${ANDROID_NDK_VERSION}
ANDROID_NDK_FULLNAME=${ANDROID_NDK_NAME}-linux.zip
ANDROID_NDK=${TOOLCHAIN_PATH}/${ANDROID_NDK_NAME}

# --- Define NDK paths based on the absolute SDK path ---
NDK_TOOLCHAIN_SYSROOT_INCLUDE_PATH="${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include"
NDK_TOOLCHAIN_SYSROOT_ARM64_LIB_PATH="${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android"

#OpenCL Headers can be found at:
#https://https://github.com/KhronosGroup/OpenCL-Headers
OPENCL_SDK_URL=https://github.com/KhronosGroup/OpenCL-Headers
OPENCL_SDK_PATH=${TOOLCHAIN_PATH}/OpenCL_SDK
OPENCL_HEADERS_PATH=${OPENCL_SDK_PATH}/OpenCL-Headers

#fully Qualcomm Hexagon SDK can be found at https://developer.qualcomm.com/software/hexagon-dsp-sdk/tools.
#fully Hexagon SDK must be obtained with Qualcomm Developer Account and follow PKLA&ECA.
HEXAGON_SDK_VERSION=6.6.0.0
HEXAGON_TOOLS_VERSION=19.0.07
HEXAGON_SDK_PATH=${TOOLCHAIN_PATH}/Hexagon_SDK/${HEXAGON_SDK_VERSION}
HEXAGON_TOOLS_PATH=${HEXAGON_SDK_PATH}/tools/HEXAGON_Tools/${HEXAGON_TOOLS_VERSION}

#supported htp arch version:
#v75 --- Snapdragon 8 Gen3
#v79 --- Snapdragon 8 Elite(aka 8 Gen4)
#v81 --- Snapdragon 8 Elite Gen5(aka 8 Gen5)

#Qualcomm Snapdragon 8Elite based Android phone is strongly recommended because:
#1. sometimes the same dsp codes can got the best performance on Snapdragon 8Elite based phone.
#2. DSP clock rate on 8Gen3 is slower than DSP clock rate on 8Elite.
#3. 8Elite support for LP-DDR5x memory, up to 5300 MHz; 8Gen3 support for LP-DDR5x memory, up to 4800 MHz.

#modify the following two lines to adapt to test phone
HTP_ARCH_VERSION=v79
HTP_ARCH_VERSION_a=V79
#all DSP skel versions to build and deploy (AP-side lib built once with HTP_ARCH_VERSION, extra DSP skels built via make)
#HTP_ARCH_VERSIONS="v75 v79 v81"
HTP_ARCH_VERSIONS="v79"

######## part-2: prompt and LLM models ########

#the following LLM models has verified(works fine) with the JZ's ggml-hexagon backend on a Snapdragon 8Elite based Android phone
#1.12 GiB, will be downloadded automatically via this script when running this script at the first time
GGUF_MODEL_NAME=/sdcard/qwen1_5-1_8b-chat-q4_0.gguf

#610 MB, download manually
GGUF_MODEL_NAME=/sdcard/Qwen3-0.6B-Q8_0.gguf

#1.2 GiB, will be downloadded automatically via this script when running this script at the first time
GGUF_MODEL_NAME=/sdcard/Qwen3.5-2B-Q4_0.gguf

GGUF_MODEL_NAME=/sdcard/llama-3.2-1B-Q4_0.gguf

#2.9 GiB, will be downloadded automatically via this script when running this script at the first time
GGUF_MODEL_NAME=/sdcard/gemma-4-E2B-it-Q4_0.gguf

# Model aliases for quick testing of multiple models
# Usage: ./scripts/build-run-ggmlhexagon-android.sh run_llamacli <alias>
#   qwen3   -> Qwen3-0.6B-Q8_0.gguf
#   gemma4  -> gemma-4-E2B-it-Q4_0.gguf
#   qwen1   -> qwen1_5-1_8b-chat-q4_0.gguf
#   llama3  -> llama-3.2-1B-Q4_0.gguf
#   (default) -> gemma-4-E2B-it-Q4_0.gguf
function resolve_model_name()
{
    case "$1" in
        qwen3)  echo "/sdcard/Qwen3-0.6B-Q8_0.gguf" ;;
        gemma4) echo "/sdcard/gemma-4-E2B-it-Q4_0.gguf" ;;
        qwen1)  echo "/sdcard/qwen1_5-1_8b-chat-q4_0.gguf" ;;
        llama3) echo "/sdcard/llama-3.2-1B-Q4_0.gguf" ;;
        *)      echo "" ; return 1 ;;
    esac
}

PROMPT_STRING="Hello, good morning, you are a powerful domain expert and know many things, now pls help to introduce the movie Once Upon a Time in America briefly, pls pay attention short then 1000 words\n"

#running_params=" -ngl 99 -t 6 -n 256 --no-warmup --no-mmap --poll 1000 --cpu-mask 0xfc --cpu-strict 1 --ctx-size 8192 --ubatch-size 1024 -fa on"
#running_params=" -ngl 99 -t 6 -n 256 --ctx-size 8192 --ubatch-size 32 --poll 1000 --no-warmup --no-mmap -fa on"
# Qualcomm recommended (docs/backend/snapdragon/developer.md): --ctx-size 8192
# Note: -ctk q8_0 -ctv q8_0 causes garbled output with our FLASH_ATTN_EXT
# --ubatch-size 32 caps PP batch so MUL_MAT passes mulmat_min_n=30 check at
# graph compute time (actual n=32 > 30). Graph build calls supports_op with
# max ubatch=512; VTCM fit is decided later by ggml_hexagon_ion_precompute_mm_params
# (4-level fallback: tiled -> n_prefetch 16/8/4/2 -> flat DDR kernel). Raise
# if PP throughput matters (n=64/128/256 also pass; n<=30 stays on CPU).
running_params=" -ngl 99 -t 6 -n 256 --ctx-size 8192 --ubatch-size 32 --poll 1000 --no-warmup --no-mmap -fa on"
#running_params=" -ngl 99 -t 6 -n 256 --no-warmup --no-mmap --poll 1000 --device Hexagon-cDSP0,Hexagon-cDSP1"

######## part-3: utilities and functions ########

function dump_vars()
{
    echo -e "ANDROID_NDK:          ${ANDROID_NDK}"
    echo -e "HEXAGON_SDK_PATH:     ${HEXAGON_SDK_PATH}"
}


function show_pwd()
{
    echo -e "current working path:$(pwd)\n"
}


function check_command_in_host()
{
    set +e
    cmd=$1
    if command -v ${cmd} > /dev/null 2>&1; then
        printf "${cmd} is available on host machine\n"
        echo ""
    else
        printf "${cmd} not exist on host machine, pls install command line utility ${cmd} firstly and accordingly\n"
        exit 1
    fi
    set -e
}


function check_commands_in_host()
{
    check_command_in_host wget
    check_command_in_host xzcat
    check_command_in_host adb
    check_command_in_host md5sum
    check_command_in_host ninja
}


function check_android_phone()
{
    local device_raw
    device_raw=$(adb devices 2>/dev/null | grep -v "List of devices" | awk 'NF>0')

    if [[ -z "$device_raw" ]]; then
        adb kill-server >/dev/null 2>&1
        sleep 0.1
        adb start-server >/dev/null 2>&1
        device_raw=$(adb devices 2>/dev/null | grep -v "List of devices" | awk 'NF>0')
        if [[ -z "$device_raw" ]]; then
            echo "No Android device detected."
            echo "Please check if phone is connected properly.Exiting"
            exit 1
        fi
    fi

    if echo "$device_raw" | grep -q "no permissions"; then
        echo "Device detected but has NO PERMISSIONS."
        echo "Please check if phone is connected properly.Exiting"
        exit 1
    fi

    if echo "$device_raw" | grep -q "unauthorized"; then
        echo "Device detected but UNAUTHORIZED."
        echo "Please check if phone is connected properly.Exiting"
        exit 1
    fi

    if echo "$device_raw" | grep -q "offline"; then
        echo "Device is OFFLINE."
        echo "Please check if phone is connected properly.Exiting"
        exit 1
    fi

    if echo "$device_raw" | awk '{print $2}' | grep -qx "device"; then
        local sn=$(echo "$device_raw" | awk '{print $1}')
        echo "Android device connected successfully: $sn"
        return 0
    fi

    echo "Unknown device error."
    echo "Please check if phone is connected properly.Exiting"
    exit 1
}


function check_and_download_hexagon_sdk()
{
    is_hexagon_llvm_exist=1
    if [ ! -f ${TOOLCHAIN_PATH}/Hexagon_SDK/${HEXAGON_SDK_VERSION}/tools/HEXAGON_Tools/${HEXAGON_TOOLS_VERSION}/NOTICE.txt ]; then
        echo -e "${TEXT_RED}minimal-hexagon-sdk not exist...${TEXT_RESET}\n"
        is_hexagon_llvm_exist=0
    fi

    if [ ${is_hexagon_llvm_exist} -eq 0 ]; then
        if [ -f ${TOOLCHAIN_PATH}/Hexagon_SDK/hexagon-sdk-v6.6.0.0-amd64-lnx.tar.xz ]; then
            echo -e "hexagon-sdk-v6.6.0.0-amd64-lnx.tar.xz already exist\n"
        else
            echo -e "begin downloading hexagon-sdk-v6.6.0.0-amd64-lnx.tar.xz \n"
            wget --no-config --quiet --show-progress -O ${TOOLCHAIN_PATH}/Hexagon_SDK/hexagon-sdk-v6.6.0.0-amd64-lnx.tar.xz https://github.com/snapdragon-toolchain/hexagon-sdk/releases/download/v6.6.0.0/hexagon-sdk-v6.6.0.0-amd64-lnx.tar.xz
            if [ $? -ne 0 ]; then
                printf "failed to download hexagon-sdk-v6.6.0.0-amd64-lnx.tar.xz\n"
                exit 1
            fi
        fi

        echo -e "begin decompressing hexagon-sdk-v6.6.0.0-amd64-lnx.tar.xz \n"
        xzcat ${TOOLCHAIN_PATH}/Hexagon_SDK/hexagon-sdk-v6.6.0.0-amd64-lnx.tar.xz | tar -C ${TOOLCHAIN_PATH}/Hexagon_SDK/ -xf -
        if [ $? -ne 0 ]; then
            printf "failed to decompress hexagon-sdk-v6.6.0.0-amd64-lnx.tar.xz\n"
            exit 1
        fi
        printf "install minimal-hexagon-sdk successfully\n\n"
    fi

    if [ ! -d ${HEXAGON_SDK_PATH} ]; then
        echo -e "HEXAGON_SDK_PATH ${HEXAGON_SDK_PATH} not exist, pls install it accordingly...\n"
        exit 0
    else
        printf "Qualcomm Hexagon SDK already exist:${HEXAGON_SDK_PATH} \n\n"
    fi
}


function check_and_download_opencl_sdk()
{
    is_opencl_sdk_exist=1

    if [ ! -d ${OPENCL_SDK_PATH} ]; then
        echo -e "OPENCL_SDK_PATH ${OPENCL_SDK_PATH} not exist, download it from ${OPENCL_SDK_URL}...\n"
        is_opencl_sdk_exist=0
    fi
    if [ ! -f ${NDK_TOOLCHAIN_SYSROOT_ARM64_LIB_PATH}/libOpenCL.so ]; then
        echo -e "${NDK_TOOLCHAIN_SYSROOT_ARM64_LIB_PATH}/libOpenCL.so not exist...\n"
        is_opencl_sdk_exist=0
    fi

    if [ ${is_opencl_sdk_exist} -eq 0 ]; then
        mkdir -p ${OPENCL_SDK_PATH}
        cd ${OPENCL_SDK_PATH}

        if [ ! -d OpenCL-Headers ]; then
            echo "Cloning OpenCL-Headers..."
            git clone https://github.com/KhronosGroup/OpenCL-Headers
            if [ $? -ne 0 ]; then
                printf "failed to download OpenCL-Headers to %s \n" "${OPENCL_SDK_PATH}"
                exit 1
            fi
        fi
        cd ${TOOLCHAIN_PATH}/OpenCL_SDK/OpenCL-Headers
        printf "Copying OpenCL Headers to Android NDK sysroot include: ${NDK_TOOLCHAIN_SYSROOT_INCLUDE_PATH}"
        mkdir -p ${NDK_TOOLCHAIN_SYSROOT_INCLUDE_PATH}
        /bin/cp -r -fv CL ${NDK_TOOLCHAIN_SYSROOT_INCLUDE_PATH}

        cd ${TOOLCHAIN_PATH}/OpenCL_SDK
        if [ ! -d OpenCL-ICD-Loader ]; then
            echo "Cloning OpenCL-ICD-Loader..."
            git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader
            if [ $? -ne 0 ]; then
                printf "failed to download OpenCL-ICD-Loader to %s \n" "${OPENCL_SDK_PATH}"
                exit 1
            fi
        fi
        cd ${TOOLCHAIN_PATH}/OpenCL_SDK/OpenCL-ICD-Loader
        mkdir -p build
        cd build
        cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DANDROID_STL=c++_shared -DOPENCL_ICD_LOADER_HEADERS_DIR=${NDK_TOOLCHAIN_SYSROOT_INCLUDE_PATH}
        echo "Building OpenCL-ICD-Loader with ninjia..."
        ninja
        if [ $? -ne 0 ]; then
            printf "failed to build OpenCL-ICD-Loader\n"
            exit 1
        fi
        mkdir -p ${NDK_TOOLCHAIN_SYSROOT_ARM64_LIB_PATH}
        /bin/cp -fv libOpenCL.so ${NDK_TOOLCHAIN_SYSROOT_ARM64_LIB_PATH}

        echo "OpenCL components setup complete"
        echo "OpenCL Headers are in: ${NDK_TOOLCHAIN_SYSROOT_INCLUDE_PATH}/CL"
        echo "libOpenCL.so is in:    ${NDK_TOOLCHAIN_SYSROOT_ARM64_LIB_PATH}/libOpenCL.so"

        cd ${PROJECT_ROOT_PATH}
    else
        printf "OpenCL SDK already exist:    ${OPENCL_SDK_PATH} \n\n"
    fi
}


function check_and_download_ndk()
{
    is_android_ndk_exist=1

    if [ ! -d ${ANDROID_NDK} ]; then
        is_android_ndk_exist=0
    fi

    if [ ! -f ${ANDROID_NDK}/build/cmake/android.toolchain.cmake ]; then
        is_android_ndk_exist=0
    fi

    if [ ${is_android_ndk_exist} -eq 0 ]; then

        if [ ! -f ${TOOLCHAIN_PATH}/${ANDROID_NDK_FULLNAME} ]; then
            wget --no-config --quiet --show-progress -O ${TOOLCHAIN_PATH}/${ANDROID_NDK_FULLNAME} https://dl.google.com/android/repository/${ANDROID_NDK_FULLNAME}
        fi

        cd ${TOOLCHAIN_PATH}
        unzip ${ANDROID_NDK_FULLNAME}

        if [ $? -ne 0 ]; then
            printf "failed to download Android NDK to %s \n" "${ANDROID_NDK}"
            exit 1
        fi
        cd ${PROJECT_ROOT_PATH}

        printf "Android NDK saved to ${ANDROID_NDK} \n\n"
    else
        printf "Android NDK already exist:         ${ANDROID_NDK} \n\n"
    fi
}


function build_idl()
{
    echo "build idl"
    #not acutually used at the moment
    #if [ -f ${HEXAGON_SDK_PATH}/ipc/fastrpc/qaic/bin/qaic ]; then
    #    ${HEXAGON_SDK_PATH}/ipc/fastrpc/qaic/bin/qaic -mdll -o ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/kernels -I${HEXAGON_SDK_PATH}/incs -I${HEXAGON_SDK_PATH}/incs/stddef -I${HEXAGON_SDK_PATH}/ipc/fastrpc/incs ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/kernels/ggmlop.idl
    #fi
}


#build extra DSP skels for versions other than the default HTP_ARCH_VERSION
#$1 = "debug" for debug build, anything else for release build
function build_extra_dsp_skels()
{
    local dsp_debug_flag
    if [ "$1" == "debug" ]; then
        dsp_debug_flag="-DDEBUG -Wall"
    else
        dsp_debug_flag="-DNDEBUG -Wall"
    fi

    for extra_ver in ${HTP_ARCH_VERSIONS}; do
        if [ "${extra_ver}" != "${HTP_ARCH_VERSION}" ]; then
            printf "\n========== build extra DSP skel: libggmldsp-skel-${extra_ver}.so ==========\n"
            make -C ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/kernels/ clean
            make -C ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/kernels/ HTP_ARCH_VERSION=${extra_ver} HEXAGON_SDK_PATH=${HEXAGON_SDK_PATH} HEXAGON_TOOLS_PATH=${HEXAGON_TOOLS_PATH} DEBUG_FLAG="${dsp_debug_flag}"
            /bin/cp -fv ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/kernels/libggmldsp-skel.so ${LOCAL_BUILD_DIR}/bin/libggmldsp-skel-${extra_ver}.so
        fi
    done
}


function build_arm64
{
    build_idl

    #make AI Agent happy
    export CCACHE_DIR=${PROJECT_ROOT_PATH}/.ccache

    cmake -H. -B${LOCAL_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DGGML_CCACHE=ON -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DGGML_HEXAGON=ON -DLLAMA_CURL=OFF -DGGML_LLAMAFILE=ON -DHEXAGON_SDK_PATH=${HEXAGON_SDK_PATH} -DHEXAGON_TOOLS_PATH=${HEXAGON_TOOLS_PATH} -DHTP_ARCH_VERSION=${HTP_ARCH_VERSION} -DCMAKE_VERBOSE_MAKEFILE:BOOL=${VERBOSE} -DGGML_USE_HEXAGON=ON
    cd ${LOCAL_BUILD_DIR}
    make -j${HOST_CPU_COUNTS}
    #cmake POST_BUILD already built libggmldsp-skel-${HTP_ARCH_VERSION}.so, build the rest
    build_extra_dsp_skels
    #upload the new libggmldsp-skel.so on device side
    prepare_ggmldsp
    show_pwd

    cd -
}


function build_arm64_debug
{
    build_idl

    #make AI Agent happy
    export CCACHE_DIR=${PROJECT_ROOT_PATH}/.ccache

    cmake -H. -B${LOCAL_BUILD_DIR} -DCMAKE_BUILD_TYPE=Debug -DGGML_OPENMP=OFF -DGGML_CCACHE=ON -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DGGML_HEXAGON=ON -DLLAMA_CURL=OFF -DGGML_LLAMAFILE=ON -DHEXAGON_SDK_PATH=${HEXAGON_SDK_PATH} -DHEXAGON_TOOLS_PATH=${HEXAGON_TOOLS_PATH} -DHTP_ARCH_VERSION=${HTP_ARCH_VERSION} -DCMAKE_VERBOSE_MAKEFILE:BOOL=${VERBOSE} -DGGML_USE_HEXAGON=ON
    cd ${LOCAL_BUILD_DIR}
    make -j${HOST_CPU_COUNTS}
    #cmake POST_BUILD already built libggmldsp-skel-${HTP_ARCH_VERSION}.so, build the rest
    build_extra_dsp_skels debug
    #upload the new libggmldsp-skel.so on device side
    prepare_ggmldsp
    show_pwd

    cd -
}

#build Qualcomm's ggml-hexagon backend for performance comparison
function build_arm64_qcom
{
    #make AI Agent happy
    export CCACHE_DIR=${PROJECT_ROOT_PATH}/.ccache_qcom

    echo "before build_qcom(build_arm64_qcom), prepare files"
    /bin/cp -fv ${PROJECT_ROOT_PATH}/docs/backend/snapdragon/CMakeUserPresets.json .
    /bin/cp -fv ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/ggml-hexagon.cpp      ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/ggml-hexagon.cpp.me
    /bin/cp -fv ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/CMakeLists.txt        ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/CMakeLists.txt.me
    /bin/cp -fv ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/ggml-hexagon-qcom.cpp ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/ggml-hexagon.cpp
    /bin/cp -fv ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/CMakeLists-qcom.txt   ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/CMakeLists.txt

    cmake -H. -B${LOCAL_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DGGML_OPENCL=OFF -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DGGML_HEXAGON=ON -DLLAMA_CURL=OFF -DGGML_LLAMAFILE=ON -DHEXAGON_SDK_PATH=${HEXAGON_SDK_PATH} -DHTP_ARCH_VERSION=${HTP_ARCH_VERSION} -DHEXAGON_SDK_ROOT=${HEXAGON_SDK_PATH} -DHEXAGON_TOOLS_ROOT=${HEXAGON_TOOLS_PATH} --preset arm64-android-snapdragon-release -DCMAKE_VERBOSE_MAKEFILE:BOOL=${VERBOSE}
    cmake --build ${LOCAL_BUILD_DIR}
    #upload the new libggml-htps.so on device side
    prepare_ggmlhtp
    show_pwd

    echo "after build_qcom(build_arm64_qcom), restore files"
    /bin/rm -f CMakeUserPresets.json
    /bin/cp -fv ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/ggml-hexagon.cpp.me   ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/ggml-hexagon.cpp
    /bin/cp -fv ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/CMakeLists.txt.me     ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/CMakeLists.txt

    echo "run following command to see the performance of qualcomm's official ggml-hexagon backend"
    echo "./scripts/build-run-android.sh run_testop MUL_MAT"
    echo "./scripts/build-run-android.sh run_testops"
    echo "./scripts/build-run-android.sh run_llamacli"
    echo "./scripts/build-run-android.sh run_llamabench"
}


function remove_temp_dir()
{
    if [ -d ${LOCAL_BUILD_DIR} ]; then
        echo "remove ${LOCAL_BUILD_DIR} directory"
        rm -rf ${LOCAL_BUILD_DIR}
    fi
}


function update_cfg()
{
    adb push ${PROJECT_ROOT_PATH}/scripts/ggml-hexagon.cfg ${REMOTE_PATH}/
}


function build_ggml_hexagon()
{
    show_pwd
    check_and_download_ndk
    check_and_download_opencl_sdk
    check_and_download_hexagon_sdk
    dump_vars
    remove_temp_dir
    build_arm64
}


function build_ggml_hexagon_debug()
{
    show_pwd
    check_and_download_ndk
    check_and_download_opencl_sdk
    check_and_download_hexagon_sdk
    dump_vars
    remove_temp_dir
    build_arm64_debug
}


function build_ggml_hexagon_qcom()
{
    show_pwd
    check_and_download_ndk
    check_and_download_opencl_sdk
    check_and_download_hexagon_sdk
    dump_vars
    remove_temp_dir
    build_arm64_qcom
}


#for Qualcomm's open-source ggml-hexagon backend in branch self-build-jz
function prepare_ggmlhtp()
{
    echo "adb push ${LOCAL_BUILD_DIR}/ggml/src/ggml-hexagon/libggml-htp-${HTP_ARCH_VERSION}.so ${REMOTE_PATH}/libggml-htp-${HTP_ARCH_VERSION}.so"
case "$HTP_ARCH_VERSION" in
    v75)
        adb push ${LOCAL_BUILD_DIR}/ggml/src/ggml-hexagon/libggml-htp-${HTP_ARCH_VERSION}.so ${REMOTE_PATH}/libggml-htp-${HTP_ARCH_VERSION}.so
    ;;

    v79)
        adb push ${LOCAL_BUILD_DIR}/ggml/src/ggml-hexagon/libggml-htp-${HTP_ARCH_VERSION}.so ${REMOTE_PATH}/libggml-htp-${HTP_ARCH_VERSION}.so
    ;;

    *)
        show_usage
        exit 1
    ;;
esac
}


#for JZ's open-source ggml-hexagon backend in branch self-build-jz
function prepare_ggmldsp()
{
    adb push ${PROJECT_ROOT_PATH}/scripts/ggml-hexagon.cfg ${REMOTE_PATH}/ggml-hexagon.cfg
    for ver in ${HTP_ARCH_VERSIONS}; do
        if [ -f ${LOCAL_BUILD_DIR}/bin/libggmldsp-skel-${ver}.so ]; then
            echo "adb push ${LOCAL_BUILD_DIR}/bin/libggmldsp-skel-${ver}.so ${REMOTE_PATH}/libggmldsp-skel-${ver}.so"
            adb push ${LOCAL_BUILD_DIR}/bin/libggmldsp-skel-${ver}.so ${REMOTE_PATH}/libggmldsp-skel-${ver}.so
        fi
    done
}


function check_and_download_model()
{
    set +e

    model_name=$1
    model_url=$2

    adb shell ls /sdcard/${model_name}
    if [ $? -eq 0 ]; then
        printf "the prebuild LLM model ${model_name} already exist on Android phone\n"
    else
        printf "the prebuild LLM model ${model_name} not exist on Android phone\n"
        printf "downloading from ${model_url}\n"
        wget --no-config --quiet --show-progress -O ${PROJECT_ROOT_PATH}/models/${model_name} ${model_url}
        adb push ${PROJECT_ROOT_PATH}/models/${model_name} /sdcard/
    fi

    set -e
}


function check_prebuilt_models()
{
    set +e

    adb shell ls /sdcard/t5-very-small-random-F32.gguf
    if [ $? -eq 0 ]; then
        printf "the prebuild LLM model t5-very-small-random-F32.gguf already exist on Android phone\n"
    else
        printf "the prebuild LLM model t5-very-small-random-F32.gguf not exist on Android phone\n"
        adb push ${PROJECT_ROOT_PATH}/models/t5-very-small-random-F32.gguf /sdcard/
    fi

    #1.12 GiB
    #check_and_download_model qwen1_5-1_8b-chat-q4_0.gguf  https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat-GGUF/resolve/main/qwen1_5-1_8b-chat-q4_0.gguf

    #1.2 GiB
    #check_and_download_model Qwen3.5-2B-Q4_0.gguf         https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_0.gguf

    #2.9 GiB
    check_and_download_model gemma-4-E2B-it-Q4_0.gguf     https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_0.gguf

    set -e
}


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


function update_ggml_libs()
{
    #adb push ${LOCAL_BUILD_DIR}/bin/*.so ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libggml-base.so                 ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libggml-cpu.so                  ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so              ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libggml.so                      ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libllama-common.so              ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libllama-completion-impl.so     ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libllama-bench-impl.so          ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libllama.so                     ${REMOTE_PATH}/
}


function prepare_run_on_phone()
{
    if [ $# != 1 ]; then
        print "invalid param"
        return
    fi
    program=$1

    update_cfg

    check_prebuilt_models

    is_so_file_changed ${LOCAL_BUILD_DIR}/bin/libggml-cpu.so
    if [ $? -eq 0 ]; then
        printf "${LOCAL_BUILD_DIR}/bin/libggml-cpu.so not changed\n\n"
        #reuse cached/uploaded ggml runtime libs on device side to avoid time-consuming task on host side
    else
        printf "${LOCAL_BUILD_DIR}/bin/libggml-cpu.so has changed or first check\n\n"
        #upload ggml runtime libs to Android phone
        update_ggml_libs
    fi

    #for verify JZ's open-source ggml-hexagon backend(libggmldsp-skel.so) which generated from source codes in this branch
    #this is default behaviour(it will report libggmldsp-skel.so can't found when exec UT after build_qcom), but Qualcomm's backend so already updated on device side when running build_qcom
    prepare_ggmldsp

    #for verify Qualcomm's open-source ggml-hexagon backend(libggml-htp.so) which generated from source codes in this branch
    #this is non-default behaviour, but JZ's backend so already updated on device side when running build
    #prepare_ggmlhtp

    adb push ${LOCAL_BUILD_DIR}/bin/${program} ${REMOTE_PATH}/

    adb shell ls -l ${REMOTE_PATH}/libggml-*.so

    adb shell chmod +x ${REMOTE_PATH}/${program}

    #configuration for cDSP's logcat
    adb shell "rm /data/local/tmp/${program}.farf"
    adb shell "touch /data/local/tmp/${program}.farf"
    adb shell "echo 0x1f > /data/local/tmp/${program}.farf"
    #observe cDSP's log with debug build:./scripts/build-run-android.sh build_debug
    #adb logcat  | grep -iE "CDSP0"
}


function run_llamacli()
{
    local model_name=""
    local model_path=""

    if [ $# -ge 1 ]; then
        model_name="$1"
        model_path=$(resolve_model_name "$model_name")
        if [ -z "$model_path" ]; then
            echo "ERROR: unknown model alias '$model_name'. Valid aliases: qwen3, gemma4, qwen1, llama3"
            exit 1
        fi
    else
        model_path="${GGUF_MODEL_NAME}"
    fi

    prepare_run_on_phone llama-completion

    echo "${REMOTE_PATH}/llama-completion ${running_params} --mulmat-algotype ${mulmat_algotype} -st -no-cnv -m ${model_path} -p \"${PROMPT_STRING}\""
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/llama-completion ${running_params} --mulmat-algotype ${mulmat_algotype} -st -no-cnv -m ${model_path} -p \"${PROMPT_STRING}\""

}


function run_llamabench()
{
    prepare_run_on_phone llama-bench

    echo "adb shell \"cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/llama-bench --mulmat-algotype ${mulmat_algotype} -t 6 --poll 1000 -fa 1 --ubatch-size 1024 -p 200,512,800,1024 -m ${GGUF_MODEL_NAME}\""

    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/llama-bench --mulmat-algotype ${mulmat_algotype} -t 6 --poll 1000 -fa 1 --ubatch-size 1024 -p 200,512,800,1024 -m ${GGUF_MODEL_NAME}"
}


function run_llamacli_all()
{
    local models=("gemma4" "qwen3" "qwen1" "llama3")
    #local algotypes=(29 30 32)
    local algotypes=(29 33)

    local total=$(( ${#models[@]} * ${#algotypes[@]} ))
    local count=0

    echo "=============================================="
    echo "  Batch inference test: ${#models[@]} models x ${#algotypes[@]} algotypes = ${total} tests"
    echo "=============================================="

    for model in "${models[@]}"; do
        for algotype in "${algotypes[@]}"; do
            count=$(( count + 1 ))
            echo ""
            echo "--- [${count}/${total}] model=${model} algotype=${algotype} ---"
            mulmat_algotype=${algotype}
            run_llamacli "${model}"
        done
    done

    echo ""
    echo "=============================================="
    echo "  Batch inference test complete: ${total} tests done"
    echo "=============================================="
}


function run_threadsafety()
{
    prepare_run_on_phone test-thread-safety

    echo "${REMOTE_PATH}/test-thread-safety -np 2 -m ${GGUF_MODEL_NAME} -p \"hello,world\" -n 256 -ngl 99 "
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/test-thread-safety -np 1 -m ${GGUF_MODEL_NAME} -p \"hello,world\" -n 256 -ngl 99 "

}


function run_test-ops()
{
    prog_name=test-backend-ops
    prepare_run_on_phone ${prog_name}

    echo "adb shell \"cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/${prog_name} -a ${mulmat_algotype} test\""


    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/${prog_name} -a ${mulmat_algotype} test"

}


function check_mulmat_algotype
{
    printf "mulmat_algotype ${mulmat_algotype} \n"
    if [[ ${mulmat_algotype} != 0 ]] && [[ ${mulmat_algotype} != 1 ]] && [[ ${mulmat_algotype} != 2 ]] && [[ ${mulmat_algotype} != 3 ]] && [[ ${mulmat_algotype} != 4 ]] && [[ ${mulmat_algotype} != 5 ]] && [[ ${mulmat_algotype} != 6 ]] && [[ ${mulmat_algotype} != 30 ]] && [[ ${mulmat_algotype} != 31 ]] && [[ ${mulmat_algotype} != 32 ]] && [[ ${mulmat_algotype} != 33 ]] && [[ ${mulmat_algotype} != 29 ]]; then
        printf "invalid mulmat algotype\n"
        printf "valid mulmat algotype: 0, 1, 2, 3, 4, 5, 6, 29, 30, 31, 32, 33 \n"
        exit 1
    fi
}


function run_test-op()
{
    prog_name=test-backend-ops
    prog_param="-o ${opname} -a ${mulmat_algotype}"
    prepare_run_on_phone ${prog_name}

    check_mulmat_algotype

    echo "adb shell cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/${prog_name} test ${prog_param}"

    echo "\n"
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/${prog_name} test ${prog_param}"

}


function run_perf-op()
{
    prog_name=test-backend-ops
    prepare_run_on_phone ${prog_name}

    check_mulmat_algotype

    echo "adb shell cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/${prog_name} perf -o ${opname} -a ${mulmat_algotype}"

    echo "\n"
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/${prog_name} perf -o ${opname} -a ${mulmat_algotype}"

}


function print_oplist()
{
oplist="DUP
    ADD
    ADD1
    ACC
    SUB
    MUL
    DIV
    SQR
    SQRT
    LOG
    SIN
    COS
    SUM
    SUM_ROWS
    MEAN
    ARGMAX
    COUNT_EQUAL
    REPEAT
    REPEAT_BACK
    CONCAT
    SILU_BACK
    NORM
    RMS_NORM
    RMS_NORM_BACK
    GROUP_NORM

    MUL_MAT
    MUL_MAT_ID
    OUT_PROD

    SCALE
    SET
    CPY
    CONT
    RESHAPE
    VIEW
    PERMUTE
    TRANSPOSE
    GET_ROWS
    GET_ROWS_BACK
    DIAG
    DIAG_MASK_INF
    DIAG_MASK_ZERO
    SOFT_MAX
    SOFT_MAX_BACK
    ROPE
    ROPE_BACK
    CLAMP
    CONV_TRANSPOSE_1D
    IM2COL
    IM2COL_BACK
    CONV_TRANSPOSE_2D
    POOL_1D
    POOL_2D
    POOL_2D_BACK
    UPSCALE
    PAD
    PAD_REFLECT_1D
    ARANGE
    TIMESTEP_EMBEDDING
    ARGSORT
    LEAKY_RELU

    FLASH_ATTN_EXT
    FLASH_ATTN_BACK
    SSM_CONV
    SSM_SCAN
    WIN_PART
    WIN_UNPART
    GET_REL_POS
    ADD_REL_POS
    RWKV_WKV6
    GATED_LINEAR_ATTN"

echo "opname list: "
echo ${oplist}
}


function show_usage()
{
    echo -e "\n\n\n"
    echo "Usage:"
    echo "  $0 help"
    echo "  $0 print_oplist"
    echo "  $0 update_ggml_libs"
    echo -e "\n"
    echo "  $0 build (build JZ's ggml-hexagon backend)"
    echo "  $0 build_debug (build JZ's ggml-hexagon backend in debug mode)"
    echo "  $0 build_qcom (build Qualcomm's ggml-hexagon backend for performance comparison)"
    echo "  $0 clean"
    echo -e "\n"
    echo "  $0 run_testops    [mulmat_algotype]"
    echo "  $0 run_llamabench [mulmat_algotype]"
    echo -e "\n"
    echo "  $0 run_llamacli   [model_alias] [mulmat_algotype]"
    echo "  Model aliases for run_llamacli:"
    echo "    qwen3   -> Qwen3-0.6B-Q8_0.gguf"
    echo "    gemma4  -> gemma-4-E2B-it-Q4_0.gguf"
    echo "    qwen1   -> qwen1_5-1_8b-chat-q4_0.gguf"
    echo "    llama3  -> llama-3.2-1B-Q4_0.gguf"
    echo "    (default) -> gemma-4-E2B-it-Q4_0.gguf"
    echo "  Examples:"
    echo "    $0 run_llamacli qwen3        # test qwen3 with default algotype"
    echo "    $0 run_llamacli qwen3 32     # test qwen3 with algotype=32 (HMX pipeline)"
    echo "    $0 run_llamacli gemma4 29    # test gemma4 with algotype=29 (Qualcomm execute_op)"
    echo -e "\n"
    echo "  $0 run_llamacli_all            # batch test 4 models x 2 algotypes (29,33) = 8 tests"
    echo "    Log capture example:"
    echo "      $0 run_llamacli_all 2>&1 | tee log_ci_\$(date +%y%m%d-%H%M%S).txt"
    echo -e "\n"
    echo "  $0 run_testop     ADD/MUL_MAT/FLASH_ATTN_EXT [mulmat_algotype] (verify accuracy    of ADD/MUL_MAT)"
    echo "  $0 run_perfop     ADD/MUL_MAT/FLASH_ATTN_EXT [mulmat_algotype] (verify performance of ADD/MUL_MAT)"
    echo -e "\n\n\n"
}


######## part-4: entry point  ########

show_pwd

check_commands_in_host
check_android_phone
check_and_download_ndk
check_and_download_opencl_sdk
check_and_download_hexagon_sdk
check_prebuilt_models

if [ $# == 0 ]; then
    show_usage
    exit 1
elif [ $# == 1 ]; then
    if [ "$1" == "-h" ]; then
        show_usage
        exit 1
    elif [ "$1" == "help" ]; then
        show_usage
        exit 1
    elif [ "$1" == "print_oplist" ]; then
        print_oplist
        exit 1
    elif [ "$1" == "update_ggml_libs" ]; then
        update_ggml_libs
        exit 1
    elif [ "$1" == "build" ]; then
        build_ggml_hexagon
        exit 0
    elif [ "$1" == "build_debug" ]; then
        build_ggml_hexagon_debug
        exit 0
    elif [ "$1" == "build_qcom" ]; then
        build_ggml_hexagon_qcom
        exit 0
    elif [ "$1" == "clean" ]; then
        remove_temp_dir
        exit 0
    elif [ "$1" == "run_testops" ]; then
        mulmat_algotype=${default_mulmat_algotype}
        run_test-ops
        exit 0
    elif [ "$1" == "run_llamacli" ]; then
        mulmat_algotype=${default_mulmat_algotype}
        run_llamacli
        exit 0
    elif [ "$1" == "run_llamabench" ]; then
        mulmat_algotype=${default_mulmat_algotype}
        run_llamabench
        exit 0
    elif [ "$1" == "run_llamacli_all" ]; then
        run_llamacli_all
        exit 0
    else
        show_usage
        exit 1
    fi
elif [ $# == 2 ]; then
#TODO: check opname in oplist
#opname can be found via print_oplist:

    if [ "$1" == "run_testop" ]; then
        opname=$2
        mulmat_algotype=${default_mulmat_algotype}
        run_test-op
        exit 0
    elif [ "$1" == "run_perfop" ]; then
        opname=$2
        mulmat_algotype=${default_mulmat_algotype}
        run_perf-op
        exit 0
    elif [ "$1" == "run_llamacli" ]; then
        mulmat_algotype=${default_mulmat_algotype}
        # If second arg is numeric, treat as algotype (backward compatibility)
        # Otherwise, treat as model name alias
        if [[ "$2" =~ ^[0-9]+$ ]]; then
            mulmat_algotype=$2
            check_mulmat_algotype
            run_llamacli
        else
            if [ -z "$(resolve_model_name "$2")" ]; then
                echo "ERROR: unknown model alias '$2'. Valid aliases: qwen3, gemma4, qwen1, llama3"
                show_usage
                exit 1
            fi
            run_llamacli "$2"
        fi
        exit 0
    elif [ "$1" == "run_llamabench" ]; then
        mulmat_algotype=$2
        check_mulmat_algotype
        run_llamabench
        exit 0
    elif [ "$1" == "run_threadsafety" ]; then
        mulmat_algotype=${default_mulmat_algotype}
        run_threadsafety
        exit 0
    else
        show_usage
        exit 1
    fi
elif [ $# == 3 ]; then
    if [ "$1" == "run_perfop" ]; then
        opname=$2
        mulmat_algotype=$3
        check_mulmat_algotype
        run_perf-op
        exit 0
    elif [ "$1" == "run_testop" ]; then
        opname=$2
        mulmat_algotype=$3
        check_mulmat_algotype
        run_test-op
        exit 0
    elif [ "$1" == "run_llamacli" ]; then
        if [ -z "$(resolve_model_name "$2")" ]; then
            echo "ERROR: unknown model alias '$2'. Valid aliases: qwen3, gemma4, qwen1, llama3"
            show_usage
            exit 1
        fi
        mulmat_algotype=$3
        check_mulmat_algotype
        run_llamacli "$2"
        exit 0
    else
        show_usage
        exit 1
    fi
else
    show_usage
    exit 1
fi
