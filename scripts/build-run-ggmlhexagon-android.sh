#!/usr/bin/env bash
#
# This single-source file is part of JZ's ggml-hexagon.
# 2024--2026 The ggml authors
# GitHub:  https://github.com/zhouwg/ggml-hexagon
#
# this script will setup local dev envs automatically and docker is not needed for purpose of simplify workflow.
#
# this script is AI Agent friendly and verified with Trae AI Agent.
#
# 1. build&verify llama.cpp + JZ's ggml-hexagon backend(libggmldsp-skel.so) on Linux for Android phone equipped with Qualcomm Snapdragon mobile SoC(8Elite is recommended)
#
# 2. build&verify llama.cpp + Qualcomm's ggml-hexagon backend(libggml-htp.so) on Linux for Android phone equipped with Qualcomm Snapdragon mobile SoC(8Elite is recommended)
#
# 3. performance comparison of Qualcomm's ggml-hexagon and JZ's ggml-hexagon on Android phone equipped with Qualcomm Snapdragon mobile SoC(8Elite is recommended & verified)
#
#
set -e

######## part-1: public macros & vars ########

PWD=`pwd`
PROJECT_HOME_PATH=`pwd`
PROJECT_ROOT_PATH=${PROJECT_HOME_PATH}
HOST_CPU_COUNTS=`cat /proc/cpuinfo | grep "processor" | wc | awk '{print int($1)}'`

VERBOSE=OFF
VERBOSE=ON

#running path on Android phone
REMOTE_PATH=/data/local/tmp

#path of built artifacts
LOCAL_BUILD_DIR=${PROJECT_ROOT_PATH}/out/ggmlhexagon-android

TOOLCHAIN_PATH=${PROJECT_ROOT_PATH}/prebuilts

#Android NDK can be found at:
#https://developer.android.com/ndk/downloads
ANDROID_PLATFORM=android-34
ANDROID_NDK_VERSION=r29
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
#v73 --- Snapdragon 8 Gen2
#v75 --- Snapdragon 8 Gen3
#v79 --- Snapdragon 8 Elite(aka 8 Gen4)
#v81 --- Snapdragon 8 Elite Gen5(aka 8 Gen5)

#Qualcomm Snapdragon 8Elite based Android phone is strongly recommended because:
#1. sometimes the same dsp codes can got the best performance on Snapdragon 8Elite based phone.
#2. DSP clock rate on 8Gen3 is slower than DSP clock rate on 8Elite.
#3. 8Elite support for LP-DDR5x memory, up to 5300 MHz; 8Gen3 support for LP-DDR5x memory, up to 4800 MHz.

#HTP_ARCH_VERSIONS="v79"                         # 8 Elite
#HTP_ARCH_VERSIONS="v79 v75"                     # 8 Elite + 8Gen3
HTP_ARCH_VERSIONS="v73 v75 v79 v81"              # all

# default HTP_ARCH
HTP_ARCH_VERSION=${HTP_ARCH_VERSIONS%% *}

######## part-2: prompt and LLM models ########
#supported models will be downloadded automatically in check_prebuilt_models() when running this script at the first time

#2.9 GiB, default model, will be downloadded automatically via this script when running this script at the first time
GGUF_MODEL_NAME=/sdcard/gemma-4-E2B-it-Q4_0.gguf

# Model aliases for quick testing of multiple models
# Usage: ./scripts/build-run-ggmlhexagon-android.sh run_llamacli <alias>
#   qwen3-2b            -> Qwen3.5-2B-Q4_0.gguf
#   qwen3-9b            -> Qwen3.5-9B-Q4_0.gguf
#   gemma4-e2b          -> gemma-4-E2B-it-Q4_0.gguf (2.9 GiB, fits entirely in ION mempool)
#   gemma4-e4b          -> gemma-4-E4B_q4_0-it.gguf (4.9 GiB, triggers mirror/eviction for stress testing)
#   qwen1               -> qwen1_5-1_8b-chat-q4_0.gguf
#   llama3              -> Llama-3.2-1B-Instruct-Q4_0.gguf
#   nanbeige-3b         -> Nanbeige_Nanbeige4.2-3B-Q4_0.gguf
#   minicpm5-1b         -> minicpm5-1b-q4_0.gguf
#   (default)           -> gemma-4-E2B-it-Q4_0.gguf
#   nanbeige-3b-q80     -> Nanbeige_Nanbeige4.2-3B-Q8_0.gguf
#   minicpm5-1b-q80     -> MiniCPM5-1B-Q8_0.gguf
function resolve_model_name()
{
    case "$1" in
        qwen3-2b)           echo "/sdcard/Qwen3.5-2B-Q4_0.gguf" ;;
        qwen3-9b)           echo "/sdcard/Qwen3.5-9B-Q4_0.gguf" ;;
        gemma4-e2b)         echo "/sdcard/gemma-4-E2B-it-Q4_0.gguf" ;;
        gemma4-e4b)         echo "/sdcard/gemma-4-E4B_q4_0-it.gguf" ;;
        qwen1)              echo "/sdcard/qwen1_5-1_8b-chat-q4_0.gguf" ;;
        llama3)             echo "/sdcard/Llama-3.2-1B-Instruct-Q4_0.gguf" ;;
        nanbeige-3b)        echo "/sdcard/Nanbeige_Nanbeige4.2-3B-Q4_0.gguf";;
        nanbeige-3b-q80)    echo "/sdcard/Nanbeige_Nanbeige4.2-3B-Q8_0.gguf";;
        minicpm5-1b)        echo "/sdcard/minicpm5-1b-q4_0.gguf";;
        minicpm5-1b-q80)    echo "/sdcard/MiniCPM5-1B-Q8_0.gguf";;
        *)                  echo "" ; return 1 ;;
    esac
}

PROMPT_STRING="Hello, good morning, you are a powerful domain expert and know many things, now pls help to introduce the movie Once Upon a Time in America briefly, pls pay attention short then 1000 words\n"

#unified command-line parameters used during inference testing for fair performance comparison of PP and TG across Qualcomm's ggml-hexagon and JZ's ggml-hexagon
#running_params=" -ngl 99 -t 6 -n 256 --no-warmup --load-mode none --poll 1000 --cpu-mask 0xfc --cpu-strict 1 --ctx-size 8192 --ubatch-size 1024 -fa on"
running_params=" -ngl 99 -t 6 -n 256 --ctx-size 8192 --ubatch-size 64 --poll 1000 --no-warmup --load-mode none -fa on --jinja -st"

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


#TODO:refine this function
function check_and_download_hexagon_sdk()
{
    is_hexagon_llvm_exist=1
    if [ ! -f ${TOOLCHAIN_PATH}/Hexagon_SDK/${HEXAGON_SDK_VERSION}/tools/HEXAGON_Tools/${HEXAGON_TOOLS_VERSION}/NOTICE.txt ]; then
        echo -e "${TEXT_RED}minimal-hexagon-sdk not exist...${TEXT_RESET}\n"
        is_hexagon_llvm_exist=0
    fi

    if [ ${is_hexagon_llvm_exist} -eq 0 ]; then
        mkdir -p ${TOOLCHAIN_PATH}/Hexagon_SDK/
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
    if [ -f ${HEXAGON_SDK_PATH}/ipc/fastrpc/qaic/bin/qaic ]; then
        ${HEXAGON_SDK_PATH}/ipc/fastrpc/qaic/bin/qaic -mdll -o ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/kernels -I${HEXAGON_SDK_PATH}/incs -I${HEXAGON_SDK_PATH}/incs/stddef -I${HEXAGON_SDK_PATH}/ipc/fastrpc/incs ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/kernels/ggml_dsp.idl
    fi
}


#build extra DSP skels (all HTP_ARCH_VERSIONS except the first/default)
#$1 = "debug" for debug build, anything else for release build
function build_extra_dsp_skels()
{
    local dsp_debug_flag
    if [ "$1" == "debug" ]; then
        dsp_debug_flag="-DDEBUG -Wall"
    else
        dsp_debug_flag="-DNDEBUG -Wall"
    fi

    # extras = HTP_ARCH_VERSIONS minus the first element (default)
    for extra_ver in ${HTP_ARCH_VERSIONS#* }; do
        printf "\n========== build extra DSP skel: libggmldsp-skel-${extra_ver}.so ==========\n"
        build_idl
        make -C ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/kernels/ clean
        make -C ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/kernels/ HTP_ARCH_VERSION=${extra_ver} HEXAGON_SDK_PATH=${HEXAGON_SDK_PATH} HEXAGON_TOOLS_PATH=${HEXAGON_TOOLS_PATH} DEBUG_FLAG="${dsp_debug_flag}"
        /bin/cp -fv ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/kernels/libggmldsp-skel.so ${LOCAL_BUILD_DIR}/bin/libggmldsp-skel-${extra_ver}.so
    done
}


#build JZ's ggml-hexagon backend for performance comparison
function build_arm64
{
    build_idl

    #make AI Agent happy
    export CCACHE_DIR=${PROJECT_ROOT_PATH}/.ccache

    #ARMv8.7a+i8mm CPU tuning flags, moved here from CMakeLists.txt to keep it aligned with upstream master
    local arm_cpu_flags="-march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE"

    # --- PGO support (Profile-Guided Optimization) ---
    # 2-stage build: generate profiles on device, then rebuild with -fprofile-use.
    # Step 1: PGO_GENERATE=1 build (run workload on device, profiles written to PGO_DIR on device)
    # PGO_GENERATE=1  ./scripts/build-run-ggmlhexagon-android.sh build
    # Step 2: Pull profiles from device to host, then PGO_USE=1 rebuild
    # adb pull /data/local/tmp/pgo ./pgo-data
    # PGO_USE=1 ./scripts/build-run-android.sh build
    local pgo_flags=""
    if [ "${PGO_GENERATE}" = "1" ]; then
        PGO_DIR="${PGO_DIR:-/data/local/tmp/pgo}"
        pgo_flags="-fprofile-generate=${PGO_DIR} -fno-profile-use"
        echo "[PGO] Instrumented build: profiles will be written to ${PGO_DIR} on device"
        echo "[PGO] After running workload, pull profiles: adb pull ${PGO_DIR} <host_dir>"
        echo "[PGO] Then rebuild with PGO_USE=1 and -DPGO_HOST_DIR=<host_dir>"
    elif [ "${PGO_USE}" = "1" ]; then
        PGO_HOST_DIR="${PGO_HOST_DIR:-${PROJECT_ROOT_PATH}/pgo-data}"
        local PROFDATA_FILE="${PGO_HOST_DIR}/default.profdata"
        local llvm_profdata="${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-profdata"

        # Auto-merge raw profiles if .profdata doesn't exist
        if [ ! -f "${PROFDATA_FILE}" ]; then
            echo "[PGO] Merging raw profiles from ${PGO_HOST_DIR}..."
            if ls ${PGO_HOST_DIR}/*.profraw 1>/dev/null 2>&1; then
                ${llvm_profdata} merge ${PGO_HOST_DIR}/*.profraw -o "${PROFDATA_FILE}"
                echo "[PGO] Merged to ${PROFDATA_FILE} ($(du -h "${PROFDATA_FILE}" | cut -f1))"
                #./prebuilts/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-profdata show ./pgo-data/default.profdata -topn=30
                #./prebuilts/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-profdata show ./pgo-data/default.profdata -topn=50 2>&1 | head -60
                #./prebuilts/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-profdata show ./pgo-data/default.profdata -all-functions
            else
                echo "[PGO] ERROR: No .profraw files found in ${PGO_HOST_DIR}"
                echo "[PGO] Run PGO_GENERATE=1 build first, then adb pull the profiles"
                exit 1
            fi
        else
            echo "[PGO] Using existing merged profile: ${PROFDATA_FILE}"
        fi
        pgo_flags="-fprofile-use=${PROFDATA_FILE}"
    fi

    local extra_flags="${arm_cpu_flags} ${pgo_flags}"

    cmake -H. -B${LOCAL_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DGGML_OPENCL=OFF -DGGML_CCACHE=ON -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DGGML_HEXAGON=ON -DLLAMA_CURL=OFF -DGGML_LLAMAFILE=ON -DGGML_HEXAGON_JZ=ON -DHEXAGON_SDK_ROOT=${HEXAGON_SDK_PATH} -DHEXAGON_TOOLS_ROOT=${HEXAGON_TOOLS_PATH} -DHTP_ARCH_VERSION=${HTP_ARCH_VERSION} -DCMAKE_C_FLAGS="${extra_flags}" -DCMAKE_CXX_FLAGS="${extra_flags}" -DCMAKE_VERBOSE_MAKEFILE:BOOL=${VERBOSE} -DGGML_USE_HEXAGON=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=ON -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF
    cd ${LOCAL_BUILD_DIR}
    make -j${HOST_CPU_COUNTS}
    #cmake POST_BUILD now builds all 4 DSP skels (v73/v75/v79/v81) in one pass; no script-side extras needed
    #build_extra_dsp_skels
    #upload the new libggmldsp-skel.so on device side
    prepare_ggmldsp
    #push AP-side libs too: libggml-hexagon.so embeds the regenerated FastRPC stub
    #which MUST stay in sync with the DSP skel signature, otherwise FastRPC args
    #get misaligned (root cause of the 8gen3 garbled-output regression).
    update_ggml_libs
    commit_so_file_md5 ${LOCAL_BUILD_DIR}/bin/libggml-cpu.so
    if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so ]; then
        commit_so_file_md5 ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so
        # backup for AB testing: JZ's AP-side libs + DSP skels
        # libggml-hexagon.so leaks different libc++ symbols between JZ/QCOM, so all
        # transitively-linked libs (libggml, libllama, libllama-common, *-impl) must
        # be swapped together to avoid symbol lookup failures at runtime
        mkdir -p ${PROJECT_ROOT_PATH}/out/ab-test
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so          ${PROJECT_ROOT_PATH}/out/ab-test/libggml-hexagon-jz.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libggml.so                  ${PROJECT_ROOT_PATH}/out/ab-test/libggml-jz.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama.so                 ${PROJECT_ROOT_PATH}/out/ab-test/libllama-jz.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-common.so          ${PROJECT_ROOT_PATH}/out/ab-test/libllama-common-jz.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-completion-impl.so ${PROJECT_ROOT_PATH}/out/ab-test/libllama-completion-impl-jz.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-server-impl.so     ${PROJECT_ROOT_PATH}/out/ab-test/libllama-server-impl-jz.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libmtmd.so                  ${PROJECT_ROOT_PATH}/out/ab-test/libmtmd-jz.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-bench-impl.so      ${PROJECT_ROOT_PATH}/out/ab-test/libllama-bench-impl-jz.so
        for skel in ${LOCAL_BUILD_DIR}/bin/libggmldsp-skel-v*.so; do
            [ -f "$skel" ] || continue
            /bin/cp -fv "$skel" ${PROJECT_ROOT_PATH}/out/ab-test/
        done
        # libggml-opencl.so is optional (GGML_OPENCL=OFF by default); back up if present
        if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-opencl.so ]; then
            /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libggml-opencl.so ${PROJECT_ROOT_PATH}/out/ab-test/libggml-opencl-jz.so
        fi
    fi
    show_pwd

    cd -
}


#build Qualcomm's ggml-hexagon backend for performance comparison
function build_arm64_qcom
{
    #make AI Agent happy
    export CCACHE_DIR=${PROJECT_ROOT_PATH}/.ccache_qcom

    rm -f ${LOCAL_BUILD_DIR}/.ab_test_runtime

    /bin/cp -fv ${PROJECT_ROOT_PATH}/docs/backend/snapdragon/CMakeUserPresets.json .

    cmake -H. -B${LOCAL_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DGGML_OPENCL=OFF -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DGGML_HEXAGON=ON -DLLAMA_CURL=OFF -DGGML_LLAMAFILE=ON -DHEXAGON_SDK_ROOT=${HEXAGON_SDK_PATH} -DHEXAGON_TOOLS_ROOT=${HEXAGON_TOOLS_PATH} --preset arm64-android-snapdragon-release -DCMAKE_VERBOSE_MAKEFILE:BOOL=${VERBOSE}
    cmake --build ${LOCAL_BUILD_DIR}
    #upload the new libggml-htps.so on device side
    prepare_ggmlhtp
    #push AP-side libs too: QCOM build also needs to sync runtime libs
    update_ggml_libs
    # backup for AB testing: QCOM's AP-side libs + DSP skels
    if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so ]; then
        mkdir -p ${PROJECT_ROOT_PATH}/out/ab-test
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so          ${PROJECT_ROOT_PATH}/out/ab-test/libggml-hexagon-qcom.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libggml.so                  ${PROJECT_ROOT_PATH}/out/ab-test/libggml-qcom.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama.so                 ${PROJECT_ROOT_PATH}/out/ab-test/libllama-qcom.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-common.so          ${PROJECT_ROOT_PATH}/out/ab-test/libllama-common-qcom.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-completion-impl.so ${PROJECT_ROOT_PATH}/out/ab-test/libllama-completion-impl-qcom.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-server-impl.so     ${PROJECT_ROOT_PATH}/out/ab-test/libllama-server-impl-qcom.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libmtmd.so                  ${PROJECT_ROOT_PATH}/out/ab-test/libmtmd-qcom.so
        /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-bench-impl.so      ${PROJECT_ROOT_PATH}/out/ab-test/libllama-bench-impl-qcom.so
        for skel in ${LOCAL_BUILD_DIR}/ggml/src/ggml-hexagon/libggml-htp-v*.so; do
            [ -f "$skel" ] || continue
            /bin/cp -fv "$skel" ${PROJECT_ROOT_PATH}/out/ab-test/
        done
        # libggml-opencl.so is optional (GGML_OPENCL=OFF by default); back up if present
        if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-opencl.so ]; then
            /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libggml-opencl.so ${PROJECT_ROOT_PATH}/out/ab-test/libggml-opencl-qcom.so
        fi
    fi
    show_pwd

    /bin/rm -f CMakeUserPresets.json

    echo "run following command to see the performance of qualcomm's official ggml-hexagon backend"
    echo "./scripts/build-run-android.sh run_llamacli"
    echo "./scripts/build-run-android.sh run_llamabench"
}


#build Android CPU-only reference (no ggml-hexagon) for correctness check and troubleshooting trick issues
function build_armcpu()
{
    export CCACHE_DIR=${PROJECT_ROOT_PATH}/.ccache_cpu

    #ARMv8.7a+i8mm CPU tuning flags, moved here from CMakeLists.txt to keep it aligned with upstream master
    local arm_cpu_flags="-march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE"

    cmake -H. -B${LOCAL_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DGGML_CCACHE=ON -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DGGML_HEXAGON=OFF -DLLAMA_CURL=OFF -DGGML_LLAMAFILE=ON -DCMAKE_C_FLAGS="${arm_cpu_flags}" -DCMAKE_CXX_FLAGS="${arm_cpu_flags}" -DCMAKE_VERBOSE_MAKEFILE:BOOL=${VERBOSE}
    cd ${LOCAL_BUILD_DIR}
    # use cmake --build so it matches whatever generator the cache was configured with
    cmake --build ${LOCAL_BUILD_DIR} -j${HOST_CPU_COUNTS}
    #remove stale hexagon artifacts from previous hexagon builds to ensure CPU-only runtime
    rm -f ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so
    rm -f ${LOCAL_BUILD_DIR}/bin/libggmldsp-skel-*.so
    # also clear QCOM skels left in the source dir by a prior build_qcom, else
    # detect_build_type() falls back to them and misreports hexagon-qcom
    rm -f ${LOCAL_BUILD_DIR}/ggml/src/ggml-hexagon/libggml-htp-*.so
    # backup CPU-only AP libs for AB switching (symmetric with build/build_qcom)
    # note: do NOT use "-cpu" suffix - libggml-cpu.so is the CPU backend impl, a
    # different lib from libggml.so (the core). use "-cpuonly" to avoid collision.
    mkdir -p ${PROJECT_ROOT_PATH}/out/ab-test
    /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libggml.so                  ${PROJECT_ROOT_PATH}/out/ab-test/libggml-cpuonly.so
    /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama.so                 ${PROJECT_ROOT_PATH}/out/ab-test/libllama-cpuonly.so
    /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-common.so          ${PROJECT_ROOT_PATH}/out/ab-test/libllama-common-cpuonly.so
    /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-completion-impl.so ${PROJECT_ROOT_PATH}/out/ab-test/libllama-completion-impl-cpuonly.so
    /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-server-impl.so     ${PROJECT_ROOT_PATH}/out/ab-test/libllama-server-impl-cpuonly.so
    /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libmtmd.so                  ${PROJECT_ROOT_PATH}/out/ab-test/libmtmd-cpuonly.so
    /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/libllama-bench-impl.so      ${PROJECT_ROOT_PATH}/out/ab-test/libllama-bench-impl-cpuonly.so

    # fix issue CANNOT LINK EXECUTABLE "/data/local/tmp/llama-completion": library "libggml-hexagon.so" not found: needed by main executable
    /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/llama-bench                 ${PROJECT_ROOT_PATH}/out/ab-test/llama-bench-cpuonly
    /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/llama-completion            ${PROJECT_ROOT_PATH}/out/ab-test/llama-completion-cpuonly
    /bin/cp -fv ${LOCAL_BUILD_DIR}/bin/llama-server                ${PROJECT_ROOT_PATH}/out/ab-test/llama-server-cpuonly

    show_pwd
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
    if [ -f ${PROJECT_ROOT_PATH}/scripts/ggml-hexagon.cfg ]; then
        adb push ${PROJECT_ROOT_PATH}/scripts/ggml-hexagon.cfg ${REMOTE_PATH}/
    fi
}


function build_ggml_hexagon()
{
    show_pwd
    check_and_download_ndk
    check_and_download_opencl_sdk
    check_and_download_hexagon_sdk
    dump_vars
    remove_temp_dir
    rm -f ${LOCAL_BUILD_DIR}/.ab_test_runtime
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
    for ver in ${HTP_ARCH_VERSIONS}; do
        case "$ver" in
            v73 | v75 | v79 | v81)
                echo "adb push ${LOCAL_BUILD_DIR}/ggml/src/ggml-hexagon/libggml-htp-${ver}.so ${REMOTE_PATH}/libggml-htp-${ver}.so"
                adb push ${LOCAL_BUILD_DIR}/ggml/src/ggml-hexagon/libggml-htp-${ver}.so ${REMOTE_PATH}/libggml-htp-${ver}.so
            ;;
            *)
                show_usage
                exit 1
            ;;
        esac
    done
}


#for JZ's open-source ggml-hexagon backend in branch self-build-jz
function prepare_ggmldsp()
{
    if [ -f ${PROJECT_ROOT_PATH}/scripts/ggml-hexagon.cfg ]; then
        adb push ${PROJECT_ROOT_PATH}/scripts/ggml-hexagon.cfg ${REMOTE_PATH}/ggml-hexagon.cfg
    fi
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

    #1.12 GiB
    check_and_download_model qwen1_5-1_8b-chat-q4_0.gguf  https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat-GGUF/resolve/main/qwen1_5-1_8b-chat-q4_0.gguf

    #1.2 GiB
    check_and_download_model Qwen3.5-2B-Q4_0.gguf         https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_0.gguf

    #5.1 GiB
    check_and_download_model Qwen3.5-9B-Q4_0.gguf         https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_0.gguf

    #2.9 GiB
    check_and_download_model gemma-4-E2B-it-Q4_0.gguf     https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_0.gguf

    # gemma-4-E4B_q4_0-it.gguf (4.9 GiB) is a stress-test model that triggers mirror/eviction in the 4GB ION mempool.
    check_and_download_model gemma-4-E4B_q4_0-it.gguf     https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/resolve/main/gemma-4-E4B_q4_0-it.gguf

    #737 MiB
    check_and_download_model Llama-3.2-1B-Instruct-Q4_0.gguf     https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf

    #2.4 GiB
    check_and_download_model Nanbeige_Nanbeige4.2-3B-Q4_0.gguf   https://huggingface.com/bartowski/Nanbeige_Nanbeige4.2-3B-GGUF/resolve/main/Nanbeige_Nanbeige4.2-3B-Q4_0.gguf

    #4.2 GiB
    #check_and_download_model Nanbeige_Nanbeige4.2-3B-Q8_0.gguf   https://huggingface.co/bartowski/Nanbeige_Nanbeige4.2-3B-GGUF/resolve/main/Nanbeige_Nanbeige4.2-3B-Q8_0.gguf

    #1.1 GiB
    #check_and_download_model MiniCPM5-1B-Q8_0.gguf               https://huggingface.co/openbmb/MiniCPM5-1B-GGUF/resolve/main/MiniCPM5-1B-Q8_0.gguf

    #635 MiB
    check_and_download_model minicpm5-1b-q4_0.gguf               https://huggingface.co/Elmermoreno/MiniCPM5-1B-Q4_0-GGUF/resolve/main/minicpm5-1b-q4_0.gguf
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

    # FIRST RUN: no MD5 file -> save it, return CHANGED
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
        # CHANGED -> update MD5
        echo "$current_md5" > "$md5_file"
        return 1
    fi
}

# Persist the current MD5 of a .so to its .md5 cache file. Call this ONLY after
# the file has been successfully pushed to the device.
function commit_so_file_md5() {
    local so_file="$1"
    local md5_file="${so_file}.md5"
    if [ -f "$so_file" ]; then
        md5sum "$so_file" | awk '{print $1}' > "$md5_file"
    fi
}


# Push AP-side libs (libggml-*.so, libllama-*.so) from bin/ to device.
# AP-only: does NOT push DSP skels (libggmldsp-skel-*.so / libggml-htp-*.so).
# Does NOT switch backend - DSP skels already on device stay as-is.
# Use update_jz_libs / update_qcom_libs for a full backend switch (AP + DSP).
# Gotcha: bin/ reflects the last build (JZ or QCOM); pushing JZ AP libs while
# QCOM DSP skels are still on device leaves an AP/DSP mismatch.
function update_ggml_libs()
{
    #adb push ${LOCAL_BUILD_DIR}/bin/*.so ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libggml-base.so                 ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libggml-cpu.so                  ${REMOTE_PATH}/
    #libggml-hexagon.so only exists in hexagon builds, not in CPU-only builds
    if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so ]; then
        adb push ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so          ${REMOTE_PATH}/
    fi
    #libggml-opencl.so is optional (GGML_OPENCL=OFF by default)
    if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-opencl.so ]; then
        adb push ${LOCAL_BUILD_DIR}/bin/libggml-opencl.so          ${REMOTE_PATH}/
    fi
    adb push ${LOCAL_BUILD_DIR}/bin/libggml.so                      ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libllama-common.so              ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libllama-completion-impl.so     ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libllama-server-impl.so         ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libmtmd.so                      ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libllama-bench-impl.so          ${REMOTE_PATH}/
    adb push ${LOCAL_BUILD_DIR}/bin/libllama.so                     ${REMOTE_PATH}/
}


#push JZ runtime .so from out/ab-test/ to device, renaming *-jz.so to canonical names
function update_jz_libs()
{
    local ab_test_dir=${PROJECT_ROOT_PATH}/out/ab-test
    if [ ! -f ${ab_test_dir}/libggml-hexagon-jz.so ]; then
        echo "ERROR: ${ab_test_dir}/libggml-hexagon-jz.so not found."
        echo "Run '$0 build' first to populate AB test backups."
        exit 1
    fi
    adb push ${ab_test_dir}/libggml-hexagon-jz.so          ${REMOTE_PATH}/libggml-hexagon.so
    adb push ${ab_test_dir}/libggml-jz.so                  ${REMOTE_PATH}/libggml.so
    adb push ${ab_test_dir}/libllama-jz.so                 ${REMOTE_PATH}/libllama.so
    adb push ${ab_test_dir}/libllama-common-jz.so          ${REMOTE_PATH}/libllama-common.so
    adb push ${ab_test_dir}/libllama-completion-impl-jz.so ${REMOTE_PATH}/libllama-completion-impl.so
    adb push ${ab_test_dir}/libllama-server-impl-jz.so     ${REMOTE_PATH}/libllama-server-impl.so
    adb push ${ab_test_dir}/libmtmd-jz.so                  ${REMOTE_PATH}/libmtmd.so
    adb push ${ab_test_dir}/libllama-bench-impl-jz.so      ${REMOTE_PATH}/libllama-bench-impl.so
    for skel in ${ab_test_dir}/libggmldsp-skel-*.so; do
        [ -f "$skel" ] && adb push "$skel" ${REMOTE_PATH}/
    done
    # libggml-opencl.so is optional (GGML_OPENCL=OFF by default)
    if [ -f ${ab_test_dir}/libggml-opencl-jz.so ]; then
        adb push ${ab_test_dir}/libggml-opencl-jz.so ${REMOTE_PATH}/libggml-opencl.so
    else
        adb shell "rm -f ${REMOTE_PATH}/libggml-opencl.so"
        adb shell "rm -f ${REMOTE_PATH}/libggml-vulkan.so"
    fi
    adb shell "rm -f ${REMOTE_PATH}/libggml-htp-*.so"
    echo "jz" > ${LOCAL_BUILD_DIR}/.ab_test_runtime
    echo "JZ runtime .so pushed to device."
}


#push QCOM runtime .so from out/ab-test/ to device, renaming *-qcom.so to canonical names
function update_qcom_libs()
{
    local ab_test_dir=${PROJECT_ROOT_PATH}/out/ab-test
    if [ ! -f ${ab_test_dir}/libggml-hexagon-qcom.so ]; then
        echo "ERROR: ${ab_test_dir}/libggml-hexagon-qcom.so not found."
        echo "Run '$0 build_qcom' first to populate AB test backups."
        exit 1
    fi
    adb push ${ab_test_dir}/libggml-hexagon-qcom.so          ${REMOTE_PATH}/libggml-hexagon.so
    adb push ${ab_test_dir}/libggml-qcom.so                  ${REMOTE_PATH}/libggml.so
    adb push ${ab_test_dir}/libllama-qcom.so                 ${REMOTE_PATH}/libllama.so
    adb push ${ab_test_dir}/libllama-common-qcom.so          ${REMOTE_PATH}/libllama-common.so
    adb push ${ab_test_dir}/libllama-completion-impl-qcom.so ${REMOTE_PATH}/libllama-completion-impl.so
    adb push ${ab_test_dir}/libllama-server-impl-qcom.so     ${REMOTE_PATH}/libllama-server-impl.so
    adb push ${ab_test_dir}/libmtmd-qcom.so                  ${REMOTE_PATH}/libmtmd.so
    adb push ${ab_test_dir}/libllama-bench-impl-qcom.so      ${REMOTE_PATH}/libllama-bench-impl.so
    for skel in ${ab_test_dir}/libggml-htp-*.so; do
        [ -f "$skel" ] && adb push "$skel" ${REMOTE_PATH}/
    done
    # libggml-opencl.so is optional (GGML_OPENCL=OFF by default)
    if [ -f ${ab_test_dir}/libggml-opencl-qcom.so ]; then
        adb push ${ab_test_dir}/libggml-opencl-qcom.so ${REMOTE_PATH}/libggml-opencl.so
    else
        adb shell "rm -f ${REMOTE_PATH}/libggml-opencl.so"
        adb shell "rm -f ${REMOTE_PATH}/libggml-vulkan.so"
    fi
    adb shell "rm -f ${REMOTE_PATH}/libggmldsp-skel-*.so"
    echo "qcom" > ${LOCAL_BUILD_DIR}/.ab_test_runtime
    echo "QCOM runtime .so pushed to device."
}


#push CPU-only runtime .so from out/ab-test/ to device, renaming *-cpuonly.so to canonical names
function update_cpu_libs()
{
    local ab_test_dir=${PROJECT_ROOT_PATH}/out/ab-test
    if [ ! -f ${ab_test_dir}/libggml-cpuonly.so ]; then
        echo "ERROR: ${ab_test_dir}/libggml-cpuonly.so not found."
        echo "Run '$0 build_armcpu' first to populate AB test backups."
        exit 1
    fi
    adb push ${ab_test_dir}/libggml-cpuonly.so                  ${REMOTE_PATH}/libggml.so
    adb push ${ab_test_dir}/libllama-cpuonly.so                 ${REMOTE_PATH}/libllama.so
    adb push ${ab_test_dir}/libllama-common-cpuonly.so          ${REMOTE_PATH}/libllama-common.so
    adb push ${ab_test_dir}/libllama-completion-impl-cpuonly.so ${REMOTE_PATH}/libllama-completion-impl.so
    adb push ${ab_test_dir}/libllama-server-impl-cpuonly.so     ${REMOTE_PATH}/libllama-server-impl.so
    adb push ${ab_test_dir}/libmtmd-cpuonly.so                  ${REMOTE_PATH}/libmtmd.so
    adb push ${ab_test_dir}/libllama-bench-impl-cpuonly.so      ${REMOTE_PATH}/libllama-bench-impl.so

    # fix issue CANNOT LINK EXECUTABLE "/data/local/tmp/llama-completion": library "libggml-hexagon.so" not found: needed by main executable
    # because there is a different linker procedure for Android‑CPU‑only builds
    /bin/cp -f ${ab_test_dir}/llama-bench-cpuonly               ${LOCAL_BUILD_DIR}/bin/llama-bench
    /bin/cp -f ${ab_test_dir}/llama-completion-cpuonly          ${LOCAL_BUILD_DIR}/bin/llama-completion
    /bin/cp -f ${ab_test_dir}/llama-server-cpuonly              ${LOCAL_BUILD_DIR}/bin/llama-server

    adb push ${ab_test_dir}/llama-bench-cpuonly                 ${REMOTE_PATH}/llama-bench
    adb push ${ab_test_dir}/llama-completion-cpuonly            ${REMOTE_PATH}/llama-completion
    adb push ${ab_test_dir}/llama-server-cpuonly                ${REMOTE_PATH}/llama-server

    # libggml-base.so / libggml-cpu.so are shared across builds, device-side kept as-is
    # libggml-opencl.so is optional (GGML_OPENCL=OFF by default)
    adb shell "rm -f ${REMOTE_PATH}/libggml-opencl.so"
    adb shell "rm -f ${REMOTE_PATH}/libggml-vulkan.so"
    adb shell "rm -f ${REMOTE_PATH}/libggml-hexagon.so"
    adb shell "rm -f ${REMOTE_PATH}/libggmldsp-skel-*.so"
    adb shell "rm -f ${REMOTE_PATH}/libggml-htp-*.so"
    echo "cpu" > ${LOCAL_BUILD_DIR}/.ab_test_runtime
    echo "CPU-only runtime .so pushed to device."
}


#detect build type from build output: hexagon-jz, hexagon-qcom, or cpu-only
function detect_build_type()
{
    if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so ]; then
        if ls ${LOCAL_BUILD_DIR}/bin/libggmldsp-skel-*.so 1>/dev/null 2>&1; then
            echo "hexagon-jz"
        else
            echo "hexagon-qcom"
        fi
    elif ls ${LOCAL_BUILD_DIR}/ggml/src/ggml-hexagon/libggml-htp-*.so 1>/dev/null 2>&1; then
        echo "hexagon-qcom"
    else
        echo "cpu-only"
    fi
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

    # AB test mode: if update_jz_libs/update_qcom_libs/update_cpu_libs set the marker,
    # skip lib/skel push - user has manually set up the runtime.
    local ab_test_marker="${LOCAL_BUILD_DIR}/.ab_test_runtime"
    if [ -f "${ab_test_marker}" ]; then
        local ab_runtime=$(cat "${ab_test_marker}")
        printf "AB test mode active: runtime='%s' (skipping lib/skel push)\n\n" "${ab_runtime}"
        adb push ${LOCAL_BUILD_DIR}/bin/${program} ${REMOTE_PATH}/
        adb shell ls -l ${REMOTE_PATH}/libggml-*.so
        adb shell chmod +x ${REMOTE_PATH}/${program}
        adb shell "rm -f /data/local/tmp/${program}.farf"
        adb shell "touch /data/local/tmp/${program}.farf"
        adb shell "echo 0x1c > /data/local/tmp/${program}.farf"
        return
    fi

    local current_build_type
    current_build_type=$(detect_build_type)

    local last_build_type_file="${LOCAL_BUILD_DIR}/.last_deployed_build_type"
    local last_build_type=""
    if [ -f "${last_build_type_file}" ]; then
        last_build_type=$(cat "${last_build_type_file}")
    fi

    if [ "${current_build_type}" != "${last_build_type}" ]; then
        printf "build type changed: '%s' -> '%s', force update ggml libs\n\n" "${last_build_type}" "${current_build_type}"
        update_ggml_libs
        commit_so_file_md5 ${LOCAL_BUILD_DIR}/bin/libggml-cpu.so
        if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so ]; then
            commit_so_file_md5 ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so
        fi
        echo "${current_build_type}" > "${last_build_type_file}"
    else
        local need_update=0
        is_so_file_changed ${LOCAL_BUILD_DIR}/bin/libggml-cpu.so
        if [ $? -eq 0 ]; then
            printf "${LOCAL_BUILD_DIR}/bin/libggml-cpu.so not changed\n"
            #reuse cached/uploaded ggml runtime libs on device side to avoid time-consuming task on host side
        else
            printf "${LOCAL_BUILD_DIR}/bin/libggml-cpu.so has changed or first check\n"
            need_update=1
        fi
        if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so ]; then
            is_so_file_changed ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so
            if [ $? -ne 0 ]; then
                printf "${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so has changed or first check\n"
                need_update=1
            else
                printf "${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so not changed\n"
            fi
        fi
        if [ ${need_update} -eq 0 ]; then
            #host-side MD5 matches cache, but verify libs actually exist on device
            #(user may have wiped /data/local/tmp manually, leaving host-side MD5 cache stale)
            if ! adb shell ls ${REMOTE_PATH}/libggml-cpu.so >/dev/null 2>&1; then
                printf "device-side libggml-cpu.so missing (maybe /data/local/tmp was wiped), force update ggml libs\n\n"
                need_update=1
            fi
        fi
        if [ ${need_update} -eq 0 ]; then
            printf "reuse cached/uploaded ggml runtime libs on device side\n\n"
        else
            #upload ggml runtime libs to Android phone
            update_ggml_libs
            commit_so_file_md5 ${LOCAL_BUILD_DIR}/bin/libggml-cpu.so
            if [ -f ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so ]; then
                commit_so_file_md5 ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so
            fi
        fi
    fi

    #deploy/cleanup backend-specific libs per build type
    case "${current_build_type}" in
        hexagon-jz)
            prepare_ggmldsp
            adb shell rm -f ${REMOTE_PATH}/libggml-htp-*.so
            ;;
        hexagon-qcom)
            prepare_ggmlhtp
            adb shell rm -f ${REMOTE_PATH}/libggmldsp-skel-*.so
            ;;
        cpu-only)
            adb shell rm -f ${REMOTE_PATH}/libggml-hexagon.so
            adb shell rm -f ${REMOTE_PATH}/libggmldsp-skel-*.so
            adb shell rm -f ${REMOTE_PATH}/libggml-htp-*.so
            adb shell rm -f ${REMOTE_PATH}/libggml-opencl.so
            adb shell rm -f ${REMOTE_PATH}/libggml-vulkan.so
            ;;
    esac

    adb push ${LOCAL_BUILD_DIR}/bin/${program} ${REMOTE_PATH}/

    adb shell ls -l ${REMOTE_PATH}/libggml-*.so

    adb shell chmod +x ${REMOTE_PATH}/${program}

    # configuration for cDSP's logcat
    # FARF bits: 0x01=LOW 0x02=MEDIUM 0x04=HIGH 0x08=ERROR 0x10=FATAL
    # 0x1f = ALL (default; chatty: HAP_compute_res logs ~6x per RPC, 26K+/test)
    # 0x1c = HIGH+ERROR+FATAL (drop LOW+MEDIUM verbose spam; keep diag)
    # 0x18 = ERROR+FATAL  (only errors)
    # 0x00 = silent
    adb shell "rm -f /data/local/tmp/${program}.farf"
    adb shell "touch /data/local/tmp/${program}.farf"
    adb shell "echo 0x1c > /data/local/tmp/${program}.farf"
    #observe cDSP's log
    #adb logcat  | grep "CDSP0"
}


function run_llamaversion()
{
    prepare_run_on_phone llama-cli

    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && export GGML_HEXAGON_OPPOLL=1 \
               && ${REMOTE_PATH}/llama-completion --version"
}


function run_llamacli()
{
    local model_name=""
    local model_path=""

    if [ $# -ge 1 ]; then
        model_name="$1"
        model_path=$(resolve_model_name "$model_name")
        if [ -z "$model_path" ]; then
            echo "ERROR: unknown model alias '$model_name'. Valid aliases: qwen3-2b, qwen3-9b, gemma4-e2b, gemma4-e4b, qwen1, llama3"
            exit 1
        fi
    else
        model_path="${GGUF_MODEL_NAME}"
    fi

    prepare_run_on_phone llama-completion

    #GGML_HEXAGON_OPPOLL is only effective for Qualcomm's ggml-hexagon, doesn't apply to JZ's ggml-hexagon
    echo "adb shell \"cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && export GGML_HEXAGON_OPPOLL=1 \
               && ${REMOTE_PATH}/llama-completion ${running_params} -m ${model_path} -p \"${PROMPT_STRING}\""
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && export GGML_HEXAGON_OPPOLL=1 \
               && ${REMOTE_PATH}/llama-completion ${running_params} -m ${model_path} -p \"${PROMPT_STRING}\""

}


#running llama-server on Snapdragon-based Android phone
function run_llamaserver()
{
    local model_name=""
    local model_path=""
    local server_running_params=" -ngl 99 -t 6 -n 256 --ctx-size 8192 --ubatch-size 64 --poll 1000 --no-warmup --load-mode none -fa on -np 1 --host 0.0.0.0"

    if [ $# -ge 1 ]; then
        model_name="$1"
        model_path=$(resolve_model_name "$model_name")
        if [ -z "$model_path" ]; then
            echo "ERROR: unknown model alias '$model_name'. Valid aliases: qwen3-2b, qwen3-9b, gemma4-e2b, gemma4-e4b, qwen1, llama3"
            exit 1
        fi
    else
        model_path="${GGUF_MODEL_NAME}"
    fi

    prepare_run_on_phone llama-server

    #GGML_HEXAGON_OPPOLL is only effective for Qualcomm's ggml-hexagon, doesn't apply to JZ's ggml-hexagon
    echo "adb shell \"cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && export GGML_HEXAGON_OPPOLL=1 \
               && ${REMOTE_PATH}/llama-server ${server_running_params} -m ${model_path} \""
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && export GGML_HEXAGON_OPPOLL=1 \
               && ${REMOTE_PATH}/llama-server ${server_running_params} -m ${model_path} "
}


#running llama-server-for-pi on Snapdragon-based Android phone
#TODO: llama_server: failed to initialize router models: subprocess is not enabled on this build
#root-cause:
# line 104 in the top-level CMakeLists.txt
# subprocess spawning isn't a supported/sandbox-friendly operation on mobile OSes or in WASM
function run_llamaserver_for_pi()
{
    local server_for_pi_running_params=" --models-dir /sdcard/ --no-models-autoload -ngl 999 -t 6 -n 256 --ctx-size 8192 --ubatch-size 64 --poll 1000 -fa on -np 1 --jinja --host 0.0.0.0 --api-key my-local-llama-key"
    prepare_run_on_phone llama-server
    #GGML_HEXAGON_OPPOLL is only effective for Qualcomm's ggml-hexagon, doesn't apply to JZ's ggml-hexagon
    echo "adb shell \"cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
           && export GGML_HEXAGON_OPPOLL=1 \
               && ${REMOTE_PATH}/llama-server ${server_for_pi_running_params} \""
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && export GGML_HEXAGON_OPPOLL=1 \
               && ${REMOTE_PATH}/llama-server ${server_for_pi_running_params} "
}


function run_llamabench()
{
    prepare_run_on_phone llama-bench

    #GGML_HEXAGON_OPPOLL is only effective for Qualcomm's ggml-hexagon, doesn't apply to JZ's ggml-hexagon
    echo "adb shell \"cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && export GGML_HEXAGON_OPPOLL=1 \
               && ${REMOTE_PATH}/llama-bench -t 6 --poll 1000 -ngl 99 -fa 1 --ubatch-size 1024 -p 200,500,800,1024 -n 128 -m ${GGUF_MODEL_NAME}\""

    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && export GGML_HEXAGON_OPPOLL=1 \
               && ${REMOTE_PATH}/llama-bench -t 6 --poll 1000 -ngl 99 -fa 1 --ubatch-size 1024 -p 200,500,800,1024 -n 128 -m ${GGUF_MODEL_NAME}"
}


function run_llamacli_all()
{
    local models=("qwen1" "minicpm5-1b" "llama3" "qwen3-2b" "gemma4-e2b" "nanbeige-3b" "gemma4-e4b" "qwen3-9b")

    local total=${#models[@]}
    local count=0

    echo "=============================================="
    echo "  Batch inference test: ${#models[@]} models = ${total} tests"
    echo "=============================================="

    for model in "${models[@]}"; do
        count=$(( count + 1 ))
        echo ""
        echo "--- [${count}/${total}] model=${model} ---"
        run_llamacli "${model}"
    done

    echo ""
    echo "=============================================="
    echo "  Batch inference test complete: ${total} tests done"
    echo "=============================================="
}


# Long-prompt stress test (~5500 tokens) with gemma4-e4b (4.9 GiB) to fill KV cache,
# trigger mirror/eviction in the 4GB ION mempool, and exercise split/coalesce paths.
function run_stresstest()
{
    local prompt_file="${PROJECT_ROOT_PATH}/scripts/prompt_long_cinema.txt"
    local remote_prompt="/data/local/tmp/prompt_long.txt"
    local model_path="/sdcard/gemma-4-E4B_q4_0-it.gguf"

    if [ ! -f "${prompt_file}" ]; then
        echo "ERROR: prompt file not found: ${prompt_file}"
        exit 1
    fi

    prepare_run_on_phone llama-completion

    echo "--- Pushing long prompt (~5500 tokens) to device ---"
    adb push "${prompt_file}" "${remote_prompt}"

    echo "adb shell \"cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && export GGML_HEXAGON_OPPOLL=1 \
               && ${REMOTE_PATH}/llama-completion ${running_params} -m ${model_path} -f ${remote_prompt}\""
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && export GGML_HEXAGON_OPPOLL=1 \
               && ${REMOTE_PATH}/llama-completion ${running_params} -m ${model_path} -f ${remote_prompt}"
}


function run_abtest()
{
    # JZ vs QCOM performance comparison test.
    # Requires out/ab-test/ populated (run 'build' then 'build_qcom' first).
    # Usage: run_abtest [rounds] [model_alias]
    #   rounds:      default 3
    #   model_alias: default gemma4-e2b
    #
    # Example:
    #   $0 run_abtest
    #   $0 run_abtest 5
    #   $0 run_abtest 3 qwen3-2b
    #   $0 run_abtest 2>&1 | tee log_abtest_$(date +%Y%m%d-%H%M%S).txt

    local rounds=3
    local model_path="${GGUF_MODEL_NAME}"
    local ab_test_dir=${PROJECT_ROOT_PATH}/out/ab-test

    if [ $# -ge 1 ]; then
        rounds=$1
    fi
    if [ $# -ge 2 ]; then
        local model_alias="$2"
        model_path=$(resolve_model_name "$model_alias")
        if [ -z "$model_path" ]; then
            echo "ERROR: unknown model alias '$model_alias'. Valid aliases: qwen3-2b, qwen3-9b, gemma4-e2b, gemma4-e4b, qwen1, llama3"
            exit 1
        fi
    fi

    # sanity check: verify out/ab-test/ has all required .so from both builds
    local missing=""
    local jz_libs="libggml-hexagon-jz.so libggml-jz.so libllama-jz.so libllama-common-jz.so libllama-completion-impl-jz.so libllama-server-impl-jz.so libmtmd-jz.so libllama-bench-impl-jz.so"
    local qcom_libs="libggml-hexagon-qcom.so libggml-qcom.so libllama-qcom.so libllama-common-qcom.so libllama-completion-impl-qcom.so libllama-server-impl-qcom.so libmtmd-qcom.so libllama-bench-impl-qcom.so"
    for f in ${jz_libs}; do
        [ ! -f ${ab_test_dir}/${f} ] && missing="${missing} ${f}"
    done
    for f in ${qcom_libs}; do
        [ ! -f ${ab_test_dir}/${f} ] && missing="${missing} ${f}"
    done
    if [ -n "${missing}" ]; then
        echo "ERROR: AB test backups incomplete, missing:${missing}"
        echo ""
        echo "Run these two commands first to populate ${ab_test_dir}:"
        echo "  $0 build        # builds JZ ggml-hexagon, backs up *-jz.so"
        echo "  $0 build_qcom   # builds QCOM ggml-hexagon, backs up *-qcom.so"
        exit 1
    fi
    # check DSP skels exist for at least one HTP arch version
    local jz_skels=$(ls ${ab_test_dir}/libggmldsp-skel-*.so 2>/dev/null | wc -l)
    local qcom_skels=$(ls ${ab_test_dir}/libggml-htp-*.so 2>/dev/null | wc -l)
    if [ ${jz_skels} -eq 0 ] || [ ${qcom_skels} -eq 0 ]; then
        echo "ERROR: DSP skels missing in ${ab_test_dir}"
        echo "  JZ skels (libggmldsp-skel-*.so): ${jz_skels} found"
        echo "  QCOM skels (libggml-htp-*.so): ${qcom_skels} found"
        echo ""
        echo "Run these two commands first:"
        echo "  $0 build        # builds JZ DSP skels"
        echo "  $0 build_qcom   # builds QCOM DSP skels"
        exit 1
    fi

    echo "=============================================="
    echo "  AB test: JZ vs QCOM, ${rounds} rounds each"
    echo "  model: ${model_path}"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================="

    # --- JZ phase ---
    echo ""
    echo "=== [$(date '+%H:%M:%S')] Switching to JZ ==="
    adb push ${ab_test_dir}/libggml-hexagon-jz.so           ${REMOTE_PATH}/libggml-hexagon.so
    adb push ${ab_test_dir}/libggml-jz.so                   ${REMOTE_PATH}/libggml.so
    adb push ${ab_test_dir}/libllama-jz.so                  ${REMOTE_PATH}/libllama.so
    adb push ${ab_test_dir}/libllama-common-jz.so           ${REMOTE_PATH}/libllama-common.so
    adb push ${ab_test_dir}/libllama-completion-impl-jz.so  ${REMOTE_PATH}/libllama-completion-impl.so
    adb push ${ab_test_dir}/libllama-server-impl-jz.so      ${REMOTE_PATH}/libllama-server-impl.so
    adb push ${ab_test_dir}/libmtmd-jz.so                   ${REMOTE_PATH}/libmtmd.so
    adb push ${ab_test_dir}/libllama-bench-impl-jz.so       ${REMOTE_PATH}/libllama-bench-impl.so
    for skel in ${ab_test_dir}/libggmldsp-skel-*.so; do
        [ -f "$skel" ] && adb push "$skel" ${REMOTE_PATH}/
    done
    # libggml-opencl.so is optional (GGML_OPENCL=OFF by default)
    if [ -f ${ab_test_dir}/libggml-opencl-jz.so ]; then
        adb push ${ab_test_dir}/libggml-opencl-jz.so ${REMOTE_PATH}/libggml-opencl.so
    else
        adb shell "rm -f ${REMOTE_PATH}/libggml-opencl.so"
        adb shell "rm -f ${REMOTE_PATH}/libggml-vulkan.so"
    fi
    adb shell "rm -f ${REMOTE_PATH}/libggml-htp-*.so"

    echo ""
    echo "========================================"
    echo "  JZ test (${rounds} runs)"
    echo "========================================"
    for i in $(seq 1 ${rounds}); do
        echo ""
        echo "-------- JZ run ${i}/${rounds} $(date '+%H:%M:%S') --------"
        adb shell "cd ${REMOTE_PATH} && export LD_LIBRARY_PATH=${REMOTE_PATH} && ${REMOTE_PATH}/llama-completion ${running_params} -m ${model_path} -p \"${PROMPT_STRING}\""
        echo "-------- JZ run ${i} END --------"
    done

    # --- QCOM phase ---
    echo ""
    echo "=== [$(date '+%H:%M:%S')] Switching to QCOM ==="
    adb push ${ab_test_dir}/libggml-hexagon-qcom.so          ${REMOTE_PATH}/libggml-hexagon.so
    adb push ${ab_test_dir}/libggml-qcom.so                  ${REMOTE_PATH}/libggml.so
    adb push ${ab_test_dir}/libllama-qcom.so                 ${REMOTE_PATH}/libllama.so
    adb push ${ab_test_dir}/libllama-common-qcom.so          ${REMOTE_PATH}/libllama-common.so
    adb push ${ab_test_dir}/libllama-completion-impl-qcom.so ${REMOTE_PATH}/libllama-completion-impl.so
    adb push ${ab_test_dir}/libllama-server-impl-qcom.so     ${REMOTE_PATH}/libllama-server-impl.so
    adb push ${ab_test_dir}/libmtmd-qcom.so                  ${REMOTE_PATH}/libmtmd.so
    adb push ${ab_test_dir}/libllama-bench-impl-qcom.so      ${REMOTE_PATH}/libllama-bench-impl.so
    for skel in ${ab_test_dir}/libggml-htp-*.so; do
        [ -f "$skel" ] && adb push "$skel" ${REMOTE_PATH}/
    done
    # libggml-opencl.so is optional (GGML_OPENCL=OFF by default)
    if [ -f ${ab_test_dir}/libggml-opencl-qcom.so ]; then
        adb push ${ab_test_dir}/libggml-opencl-qcom.so ${REMOTE_PATH}/libggml-opencl.so
    else
        adb shell "rm -f ${REMOTE_PATH}/libggml-opencl.so"
        adb shell "rm -f ${REMOTE_PATH}/libggml-vulkan.so"
    fi
    adb shell "rm -f ${REMOTE_PATH}/libggmldsp-skel-*.so"

    echo ""
    echo "========================================"
    echo "  QCOM test (${rounds} runs)"
    echo "========================================"
    for i in $(seq 1 ${rounds}); do
        echo ""
        echo "-------- QCOM run ${i}/${rounds} $(date '+%H:%M:%S') --------"
        adb shell "cd ${REMOTE_PATH} && export LD_LIBRARY_PATH=${REMOTE_PATH} && ${REMOTE_PATH}/llama-completion ${running_params} -m ${model_path} -p \"${PROMPT_STRING}\""
        echo "-------- QCOM run ${i} END --------"
    done

    echo ""
    echo "=============================================="
    echo "  AB test complete $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================="

    # Restore current build type libs after AB test.
    # AB test leaves QCOM libs on device (QCOM is the last phase).
    # Without this, subsequent run_llamabench would skip pushing
    # libggml-hexagon.so (MD5 matches local JZ build), leaving device
    # in a mixed state: QCOM AP lib + JZ DSP skels -> error 0x80000406.
    local restore_type=""
    local last_bt_file="${LOCAL_BUILD_DIR}/.last_deployed_build_type"
    if [ -f "${last_bt_file}" ]; then
        restore_type=$(cat "${last_bt_file}")
    fi
    if [ -z "${restore_type}" ]; then
        restore_type="hexagon-jz"
    fi
    echo ""
    echo "=== [$(date '+%H:%M:%S')] Restoring build type: ${restore_type} ==="
    case "${restore_type}" in
        hexagon-jz)
            update_jz_libs
            ;;
        hexagon-qcom)
            update_qcom_libs
            ;;
        cpu-only)
            update_cpu_libs
            ;;
    esac
}


function run_abtest_all()
{
    # Run AB test across all 8 supported models.
    # Usage: run_abtest_all [rounds]
    #   rounds: default 3 (per model); qwen3-9b is hard-capped to 1 (slow + high power, phone gets hot)
    #
    # Example:
    #   $0 run_abtest_all
    #   $0 run_abtest_all 3
    #   $0 run_abtest_all 2>&1 | tee log_abtest_all_$(date +%Y%m%d-%H%M%S).txt

    local rounds=3
    if [ $# -ge 1 ]; then
        rounds=$1
    fi

    #local all_models="qwen1 minicpm5-1b llama3 qwen3-2b gemma4-e2b nanbeige-3b gemma4-e4b qwen3-9b"
    local all_models="gemma4-e2b gemma4-e4b qwen3-2b nanbeige-3b qwen1 minicpm5-1b llama3 qwen3-9b"
    local total=8
    local idx=0

    for model_alias in ${all_models}; do
        idx=$((idx + 1))
        # qwen3-9b inference is slow + high power + phone gets hot; cap to 1 round (vs ${rounds} for other models)
        local model_rounds=${rounds}
        if [ "${model_alias}" = "qwen3-9b" ]; then
            model_rounds=1
            echo "  NOTE: qwen3-9b -> rounds=1 (slow inference, high power, phone gets hot)"
        fi
        echo ""
        echo "##############################################"
        echo "  AB test ${idx}/${total}: ${model_alias} (rounds=${model_rounds})"
        echo "  $(date '+%Y-%m-%d %H:%M:%S')"
        echo "##############################################"
        run_abtest ${model_rounds} ${model_alias}
    done

    echo ""
    echo "##############################################"
    echo "  All AB tests complete $(date '+%Y-%m-%d %H:%M:%S')"
    echo "##############################################"
}


function run_ubatchtest()
{
    # Update ubatch-size values for a given model.
    # Usage: run_ubatchtest [model_alias] [ubatch_sizes_csv]
    #   model_alias    default: gemma4-e2b
    #   ubatch_sizes   default: 32,64,128,512,1024
    local model_alias="gemma4-e2b"
    local ubatch_sizes=(8 16 22 32 64 128 512 1024)
    local save_params="${running_params}"

    if [ $# -ge 1 ] && [ "$1" != "help" ] && [ "$1" != "-h" ]; then
        if [ -n "$(resolve_model_name "$1")" ]; then
            model_alias="$1"
        else
            echo "ERROR: unknown model alias '$1'. Valid: qwen3-2b, qwen3-9b, gemma4-e2b, gemma4-e4b, qwen1, llama3"
            return 1
        fi
    fi
    if [ $# -ge 2 ] && [ -n "$2" ]; then
        IFS=',' read -ra ubatch_sizes <<< "$2"
    fi

    local model_path=$(resolve_model_name "${model_alias}")
    [ -z "${model_path}" ] && { echo "ERROR: bad model"; return 1; }

    # ensure libs are on phone
    prepare_run_on_phone llama-completion

    local stamp=$(date +%Y%m%d-%H%M%S)
    local combined_log="ubatchtest_${model_alias}_${stamp}.log"

    echo "==================================================" | tee "${combined_log}"
    echo "  ubatch sweep: model=${model_alias}" | tee -a "${combined_log}"
    echo "  sizes: ${ubatch_sizes[*]}" | tee -a "${combined_log}"
    echo "  combined log: ${combined_log}" | tee -a "${combined_log}"
    echo "==================================================" | tee -a "${combined_log}"

    # per-ubatch metrics, accumulated in-memory
    declare -A pp_tps tg_tps tot_ms graphs unused_cnt
    local total=${#ubatch_sizes[@]}
    local count=0

    # temp file for parsing: stream via tee so user sees output in real-time,
    # then read the temp file after the stream ends to extract metrics
    local tmpf
    tmpf=$(mktemp /tmp/ubatchtest.XXXXXX.log)
    # shellcheck disable=SC2064
    trap "rm -f '${tmpf}'" RETURN

    # ---------- Phase 1: raw runs (stream to terminal + combined + temp) ----------
    # grep/awk in command substitutions may return non-zero on no-match;
    # with set -e + declare -A array assignment this would exit the script early
    set +e
    for ub in "${ubatch_sizes[@]}"; do
        count=$(( count + 1 ))
        echo "" | tee -a "${combined_log}"
        echo "--- [${count}/${total}] ubatch=${ub} ---" | tee -a "${combined_log}"

        # override --ubatch-size: strip old, append new
        local new_params
        new_params=$(echo "${save_params}" | sed -E 's/--ubatch-size[[:space:]]+[0-9]+//')
        new_params="${new_params} --ubatch-size ${ub}"
        running_params="${new_params}"

        # print the actual command for reproducibility (captured by tee)
        echo "CMD: cd ${REMOTE_PATH} && export LD_LIBRARY_PATH=${REMOTE_PATH} && ${REMOTE_PATH}/llama-completion ${running_params} -m ${model_path} -p \"${PROMPT_STRING}\"" | tee -a "${combined_log}"

        # stream adb output in real-time: terminal <- tmpf <- combined_log
        # First tee writes to tmpf (full capture for later parsing) and forwards to stdout.
        # Second tee appends to combined_log and forwards to stdout.
        # Both tees must keep their stdout -> terminal sees the stream live.
        # Do NOT redirect the second tee's stdout to /dev/null: that would
        # swallow the terminal stream (the bug we just fixed).
        adb shell "cd ${REMOTE_PATH} \
                  && export LD_LIBRARY_PATH=${REMOTE_PATH} \
                  && ${REMOTE_PATH}/llama-completion ${running_params} -m ${model_path} -p \"${PROMPT_STRING}\"" 2>&1 \
            | tee "${tmpf}" \
            | tee -a "${combined_log}"

        # parse from temp file (after stream ends; full content available)
        # Anchor to "common_perf_print:" prefix to avoid matching model output
        pp_tps[$ub]=$(grep "common_perf_print:.*prompt eval time" "${tmpf}" | tail -1 \
            | grep -oE '[0-9.]+ tokens per second' | head -1 | awk '{print $1}')
        tg_tps[$ub]=$(grep "common_perf_print:.*eval time" "${tmpf}" | grep "runs" | tail -1 \
            | grep -oE '[0-9.]+ tokens per second' | head -1 | awk '{print $1}')
        tot_ms[$ub]=$(grep "common_perf_print:.*total time" "${tmpf}" | tail -1 \
            | sed -E 's/.*=\s*([0-9.]+) ms.*/\1/')
        unused_cnt[$ub]=$(grep -c '<unused' "${tmpf}")
        # line format: "...I common_perf_print:    graphs reused =        253"
        # field 5="reused", field 6="=", field 7="253" -> want $(i+2)
        graphs[$ub]=$(awk '/common_perf_print:.*graphs reused/ { for (i=1; i<=NF; i++) if ($i == "reused") { print $(i+2); exit } }' "${tmpf}")

        # truncate temp for next iteration (so grep on next ubatch is clean)
        : > "${tmpf}"
    done
    set -e

    # trap will clean up tmpf on function return
    running_params="${save_params}"

    # ---------- Phase 2: print summary from in-memory arrays ----------
    echo "" | tee -a "${combined_log}"
    echo "==================================================" | tee -a "${combined_log}"
    echo "  SUMMARY (parsed in-place during phase 1)" | tee -a "${combined_log}"
    echo "==================================================" | tee -a "${combined_log}"
    printf "%-7s  %-8s  %-8s  %-10s  %-7s  %-7s\n" \
        "ubatch" "PP_tps" "TG_tps" "total_ms" "unused" "graphs" | tee -a "${combined_log}"

    for ub in "${ubatch_sizes[@]}"; do
        printf "%-7s  %-8s  %-8s  %-10s  %-7s  %-7s\n" \
            "${ub}" "${pp_tps[$ub]:-N/A}" "${tg_tps[$ub]:-N/A}" "${tot_ms[$ub]:-N/A}" \
            "${unused_cnt[$ub]:-0}" "${graphs[$ub]:-N/A}" | tee -a "${combined_log}"
    done

    echo "==================================================" | tee -a "${combined_log}"
    echo "  done. combined log: ${combined_log}" | tee -a "${combined_log}"
    echo "==================================================" | tee -a "${combined_log}"
}


function run_ubatchtest_all()
{
    # Batch ubatch sweep across 5 models x 8 ubatches = 40 tests.
    # No arguments. Similar in spirit to run_llamacli_all.
    local models=("gemma4-e2b" "qwen3-2b" "qwen1" "llama3" "gemma4-e4b")
    local ubatch_sizes=(8 16 22 32 64 128 512 1024)

    # re-join ubatch sizes back into a comma-separated string for run_ubatchtest
    local ubatch_csv
    ubatch_csv=$(IFS=,; echo "${ubatch_sizes[*]}")

    local total_models=${#models[@]}
    local total_ubatches=${#ubatch_sizes[@]}
    local total_tests=$(( total_models * total_ubatches ))
    local count=0

    echo "=============================================="
    echo "  Batch ubatch sweep:"
    echo "    ${total_models} models x ${total_ubatches} ubatches = ${total_tests} tests"
    echo "    models:    ${models[*]}"
    echo "    ubatches:  ${ubatch_sizes[*]}"
    echo "  Log capture example:"
    echo "    $0 run_ubatchtest_all 2>&1 | tee log_ci_\$(date +%Y%m%d-%H%M%S).txt"
    echo "=============================================="

    for model in "${models[@]}"; do
        count=$(( count + 1 ))
        echo ""
        echo "=== [${count}/${total_models}] model=${model} ==="
        run_ubatchtest "${model}" "${ubatch_csv}"
    done

    echo ""
    echo "=============================================="
    echo "  Batch ubatch sweep complete: ${total_tests} tests done"
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
               && ${REMOTE_PATH}/${prog_name} test\""


    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/${prog_name} test"

}


function run_test-op()
{
    prog_name=test-backend-ops
    prog_param="-o ${opname}"
    prepare_run_on_phone ${prog_name}

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

    echo "adb shell cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/${prog_name} perf -o ${opname}"

    echo "\n"
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/${prog_name} perf -o ${opname}"

}


function show_usage()
{
    echo -e "\n"
    echo "Usage:"
    echo "  $0 help"

    echo "  $0 update_jz_libs   (push JZ runtime .so from out/ab-test/ to device, for build)"
    echo "  $0 update_qcom_libs (push QCOM runtime .so from out/ab-test/ to device, for build_qcom)"
    echo "  $0 update_cpu_libs  (push CPU-only runtime .so from out/ab-test/ to device, for build_armcpu)"
    echo "  $0 update_ggml_libs (incremental: push AP-side libs from bin/ to device only; keep DSP skels as-is)"

    echo "  $0 build            (build JZ's ggml-hexagon backend)"
    echo "  $0 build_qcom       (build Qualcomm's ggml-hexagon backend for performance comparison)"
    echo "  $0 build_armcpu     (build Android CPU-only reference for correctness check and troulbeshooting trick issues)"
    echo "  $0 clean"

    echo "  $0 run_testops"
    echo "  $0 run_testop     ADD/MUL_MAT/FLASH_ATTN_EXT (verify accuracy    of ADD/MUL_MAT)"
    echo "  $0 run_perfop     ADD/MUL_MAT/FLASH_ATTN_EXT (verify performance of ADD/MUL_MAT)"
    echo "  $0 run_llamacli"
    echo "  $0 run_llamabench"
    echo "  $0 run_llamaversion"

    echo "  $0 run_llamaserver"
    echo "  In a disconnected environment, download the pre-built UI from a llama.cpp
    release at https://github.com/ggml-org/llama.cpp/releases and extract to tools/ui/dist."

    echo "  $0 run_llamaserver_for_pi"

    echo "  $0 run_llamacli_all     (batch test 8 models = 8 tests)"
    echo "    Log capture example:"
    echo "      $0 run_llamacli_all 2>&1 | tee log_ci_\$(date +%Y%m%d-%H%M%S).txt"
    echo -e "\n"

    echo "  $0 run_abtest  [rounds] [model_alias]"
    echo "    JZ vs QCOM performance comparison (requires 'build' then 'build_qcom' first)."
    echo "    rounds:       default 3"
    echo "    model_alias:  default gemma4-e2b"
    echo "    Examples:"
    echo "      $0 run_abtest                      # 3 rounds, gemma4-e2b"
    echo "      $0 run_abtest 5                    # 5 rounds, gemma4-e2b"
    echo "      $0 run_abtest 3 qwen3-2b           # 3 rounds, qwen3-2b"
    echo "    Log capture example:"
    echo "      $0 run_abtest 2>&1 | tee log_abtest_\$(date +%Y%m%d-%H%M%S).txt"
    echo -e "\n"

    echo "  $0 run_abtest_all [rounds]"
    echo "    Batch AB test across all 8 models (qwen1 minicpm5-1b llama3 qwen3-2b gemma4-e2b nanbeige-3b gemma4-e4b qwen3-9b)."
    echo "    rounds: default 3"
    echo "    Log capture example:"
    echo "      $0 run_abtest_all 2>&1 | tee log_abtest_all_\$(date +%Y%m%d-%H%M%S).txt"
    echo -e "\n"

    echo "  $0 run_llamacli   [model_alias]"
    echo "  Model aliases for run_llamacli:"
    echo "    qwen3-2b      -> Qwen3.5-2B-Q4_0.gguf"
    echo "    qwen3-9b      -> Qwen3.5-9B-Q4_0.gguf"
    echo "    gemma4-e2b    -> gemma-4-E2B-it-Q4_0.gguf (2.9 GiB)"
    echo "    gemma4-e4b    -> gemma-4-E4B_q4_0-it.gguf (4.9 GiB, stress test for mirror/eviction)"
    echo "    qwen1         -> qwen1_5-1_8b-chat-q4_0.gguf"
    echo "    llama3        -> Llama-3.2-1B-Instruct-Q4_0.gguf"
    echo "    (default)     -> gemma-4-E2B-it-Q4_0.gguf"
    echo "  Examples:"
    echo "    $0 run_llamacli qwen3-2b     # test qwen3-2b"
    echo "    $0 run_llamacli gemma4-e2b   # test gemma4-e2b"
    echo "    $0 run_llamacli gemma4-e4b   # test gemma4-e4b (mirror stress test)"
    echo -e "\n"

    echo "  $0 run_stresstest"
    echo "    Long-context stress test with gemma4-E4B (~5500 token prompt)."
    echo "    Fills KV cache to trigger mirror/eviction, exercising allocator"
    echo "    split/coalesce paths. Requires gemma4-E4B model on device."
    echo "    Log capture example:"
    echo "      $0 run_stresstest 2>&1 | tee log_stress_\$(date +%Y%m%d-%H%M%S).txt"
    echo -e "\n"

    echo "  $0 run_ubatchtest  [model_alias] [ubatch_csv]"
    echo "    Sweep --ubatch-size values, dump raw per-ubatch logs (no in-shell parsing)."
    echo "    model_alias:  gemma4-e2b (default) | qwen3-2b | qwen3-9b | gemma4-e4b | qwen1 | llama3"
    echo "    ubatch_csv:   8,16,22,32,64,128,512,1024 (default)"
    echo "    Examples:"
    echo "      $0 run_ubatchtest                          # gemma4-e2b + 8/16/22/32/64/128/512/1024"
    echo "      $0 run_ubatchtest qwen3-2b                 # qwen3-2b   + 8/16/22/32/64/128/512/1024"
    echo "      $0 run_ubatchtest gemma4-e4b               # gemma4-e4b + 8/16/22/32/64/128/512/1024"
    echo "    Log capture example:"
    echo "      $0 run_ubatchtest 2>&1 | tee log_ci_\$(date +%Y%m%d-%H%M%S).txt"
    echo -e "\n"

    echo "  $0 run_ubatchtest_all"
    echo "    Batch ubatch sweep across 5 models x 8 ubatches = 40 tests."
    echo "    models:    gemma4-e2b, qwen3-2b, qwen1, llama3, gemma4-e4b"
    echo "    ubatches:  8, 16, 22, 32, 64, 128, 512, 1024"
    echo "    Log capture example:"
    echo "      $0 run_ubatchtest_all 2>&1 | tee log_ci_\$(date +%Y%m%d-%H%M%S).txt"
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
    elif [ "$1" == "update_ggml_libs" ]; then
        update_ggml_libs
        exit 1
    elif [ "$1" == "update_jz_libs" ]; then
        update_jz_libs
        exit 0
    elif [ "$1" == "update_qcom_libs" ]; then
        update_qcom_libs
        exit 0
    elif [ "$1" == "update_cpu_libs" ]; then
        update_cpu_libs
        exit 0
    elif [ "$1" == "build" ]; then
        build_ggml_hexagon
        exit 0
    elif [ "$1" == "build_debug" ]; then
        build_ggml_hexagon_debug
        exit 0
    elif [ "$1" == "build_qcom" ]; then
        build_ggml_hexagon_qcom
        exit 0
    elif [ "$1" == "build_armcpu" ]; then
        build_armcpu
        exit 0
    elif [ "$1" == "clean" ]; then
        remove_temp_dir
        exit 0
    elif [ "$1" == "run_testops" ]; then
        run_test-ops
        exit 0
    elif [ "$1" == "run_llamacli" ]; then
        run_llamacli
        exit 0
    elif [ "$1" == "run_llamaversion" ]; then
        run_llamaversion
        exit 0
    elif [ "$1" == "run_llamaserver" ]; then
        run_llamaserver
        exit 0
    elif [ "$1" == "run_llamaserver_for_pi" ]; then
        run_llamaserver_for_pi
        exit 0
    elif [ "$1" == "run_stresstest" ]; then
        run_stresstest
        exit 0
    elif [ "$1" == "run_llamabench" ]; then
        run_llamabench
        exit 0
    elif [ "$1" == "run_llamacli_all" ]; then
        run_llamacli_all
        exit 0
    elif [ "$1" == "run_abtest" ]; then
        run_abtest
        exit 0
    elif [ "$1" == "run_abtest_all" ]; then
        run_abtest_all
        exit 0
    elif [ "$1" == "run_ubatchtest" ]; then
        run_ubatchtest
        exit 0
    elif [ "$1" == "run_ubatchtest_all" ]; then
        run_ubatchtest_all
        exit 0
    else
        show_usage
        exit 1
    fi
elif [ $# == 2 ]; then
    if [ "$1" == "run_testop" ]; then
        opname=$2
        run_test-op
        exit 0
    elif [ "$1" == "run_perfop" ]; then
        opname=$2
        run_perf-op
        exit 0
    elif [ "$1" == "run_llamacli" ]; then
        if [ -z "$(resolve_model_name "$2")" ]; then
            echo "ERROR: unknown model alias '$2'. Valid aliases: qwen3-2b, qwen3-9b, gemma4-e2b, gemma4-e4b, qwen1, llama3"
            show_usage
            exit 1
        fi
        run_llamacli "$2"
        exit 0
    elif [ "$1" == "run_threadsafety" ]; then
        run_threadsafety
        exit 0
    elif [ "$1" == "run_ubatchtest" ]; then
        run_ubatchtest "$2"
        exit 0
    elif [ "$1" == "run_abtest" ]; then
        run_abtest "$2"
        exit 0
    elif [ "$1" == "run_abtest_all" ]; then
        run_abtest_all "$2"
        exit 0
    elif [ "$1" == "run_llamaserver" ]; then
        if [ -z "$(resolve_model_name "$2")" ]; then
            echo "ERROR: unknown model alias '$2'. Valid aliases: qwen3-2b, qwen3-9b, gemma4-e2b, gemma4-e4b, qwen1, llama3"
            show_usage
            exit 1
        fi
        run_llamaserver "$2"
        exit 0
    else
        show_usage
        exit 1
    fi
elif [ $# == 3 ]; then
    if [ "$1" == "run_perfop" ]; then
        opname=$2
        run_perf-op
        exit 0
    elif [ "$1" == "run_testop" ]; then
        opname=$2
        run_test-op
        exit 0
    elif [ "$1" == "run_llamacli" ]; then
        if [ -z "$(resolve_model_name "$2")" ]; then
            echo "ERROR: unknown model alias '$2'. Valid aliases: qwen3-2b, qwen3-9b, gemma4-e2b, gemma4-e4b, qwen1, llama3"
            show_usage
            exit 1
        fi
        run_llamacli "$2"
        exit 0
    elif [ "$1" == "run_ubatchtest" ]; then
        run_ubatchtest "$2" "$3"
        exit 0
    elif [ "$1" == "run_abtest" ]; then
        run_abtest "$2" "$3"
        exit 0
    else
        show_usage
        exit 1
    fi
elif [ $# == 4 ]; then
    show_usage
    exit 1
else
    show_usage
    exit 1
fi
