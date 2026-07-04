#!/usr/bin/env bash
#
# build llama.cpp + ggml-hexagon backend(Qualcomm's official version) on Linux for Android phone equipped with Qualcomm Snapdragon mobile SoC
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
LOCAL_BUILD_DIR=/tmp/ggmlhexagon-android
LOCAL_BUILD_DIR=${PROJECT_ROOT_PATH}/out/ggmlhexagon-android

TOOLCHAIN_PATH=${PROJECT_ROOT_PATH}/prebuilts

#Android NDK can be found at:
#https://developer.android.com/ndk/downloads
ANDROID_PLATFORM=android-34
ANDROID_NDK_VERSION=r28
ANDROID_NDK_NAME=android-ndk-${ANDROID_NDK_VERSION}
ANDROID_NDK_FULLNAME=${ANDROID_NDK_NAME}-linux.zip
ANDROID_NDK=${PROJECT_ROOT_PATH}/prebuilts/${ANDROID_NDK_NAME}

# --- Define NDK paths based on the absolute SDK path ---
NDK_TOOLCHAIN_SYSROOT_INCLUDE_PATH="${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include"
NDK_TOOLCHAIN_SYSROOT_ARM64_LIB_PATH="${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android"

#OpenCL Headers can be found at:
#https://https://github.com/KhronosGroup/OpenCL-Headers
OPENCL_SDK_URL=https://github.com/KhronosGroup/OpenCL-Headers
OPENCL_SDK_PATH=${PROJECT_ROOT_PATH}/prebuilts/OpenCL_SDK
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
#4. 8Elite Gen 5 is better.

#modify the following two lines to adapt to test phone
HTP_ARCH_VERSION=v79
HTP_ARCH_VERSION_a=V79

######## part-2: prompt and LLM models ########

#the following LLM models has verified(works fine) with the official ggml-hexagon backend on a Snapdragon 8Elite based Android phone
#1.12 GiB, will be downloadded automatically via this script when running this script at the first time
GGUF_MODEL_NAME=/sdcard/qwen1_5-1_8b-chat-q4_0.gguf

#1.2 GiB, will be downloadded automatically via this script when running this script at the first time
GGUF_MODEL_NAME=/sdcard/Qwen3.5-2B-Q4_0.gguf

#2.9 GiB, will be downloadded automatically via this script when running this script at the first time
GGUF_MODEL_NAME=/sdcard/gemma-4-E2B-it-Q4_0.gguf

PROMPT_STRING="introduce the movie Once Upon a Time in America briefly.\n"

#running_params=" -ngl 99 -t 6 -n 256 --no-warmup -fa 1 "
running_params=" -ngl 99 -t 6 -n 256 --no-warmup "

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
        cd ${PROJECT_ROOT_PATH}/prebuilts/OpenCL_SDK/OpenCL-Headers
        printf "Copying OpenCL Headers to Android NDK sysroot include: ${NDK_TOOLCHAIN_SYSROOT_INCLUDE_PATH}"
        mkdir -p ${NDK_TOOLCHAIN_SYSROOT_INCLUDE_PATH}
        /bin/cp -r -fv CL ${NDK_TOOLCHAIN_SYSROOT_INCLUDE_PATH}

        cd ${PROJECT_ROOT_PATH}/prebuilts/OpenCL_SDK
        if [ ! -d OpenCL-ICD-Loader ]; then
            echo "Cloning OpenCL-ICD-Loader..."
            git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader
            if [ $? -ne 0 ]; then
                printf "failed to download OpenCL-ICD-Loader to %s \n" "${OPENCL_SDK_PATH}"
                exit 1
            fi
        fi
        cd ${PROJECT_ROOT_PATH}/prebuilts/OpenCL_SDK/OpenCL-ICD-Loader
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

        if [ ! -f ${PROJECT_ROOT_PATH}/prebuilts/${ANDROID_NDK_FULLNAME} ]; then
            wget --no-config --quiet --show-progress -O ${PROJECT_ROOT_PATH}/prebuilts/${ANDROID_NDK_FULLNAME} https://dl.google.com/android/repository/${ANDROID_NDK_FULLNAME}
        fi

        cd ${PROJECT_ROOT_PATH}/prebuilts/
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


function build_arm64
{
    /bin/cp -fv ${PROJECT_ROOT_PATH}/docs/backend/snapdragon/CMakeUserPresets.json .

    cmake -H. -B${LOCAL_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DGGML_HEXAGON=ON -DLLAMA_CURL=OFF -DGGML_LLAMAFILE=ON -DHEXAGON_SDK_PATH=${HEXAGON_SDK_PATH} -DHTP_ARCH_VERSION=${HTP_ARCH_VERSION} -DHEXAGON_SDK_ROOT=${HEXAGON_SDK_PATH} -DHEXAGON_TOOLS_ROOT=${HEXAGON_TOOLS_PATH} --preset arm64-android-snapdragon-release -DCMAKE_VERBOSE_MAKEFILE:BOOL=${VERBOSE}
    cmake --build ${LOCAL_BUILD_DIR}
    show_pwd
    /bin/rm -f CMakeUserPresets.json
}


function build_arm64_debug
{
    /bin/cp -fv ${PROJECT_ROOT_PATH}/docs/backend/snapdragon/CMakeUserPresets.json .

    cmake -H. -B${LOCAL_BUILD_DIR} -DCMAKE_BUILD_TYPE=Debug -DGGML_OPENMP=OFF -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DGGML_HEXAGON=ON -DLLAMA_CURL=OFF -DGGML_LLAMAFILE=ON -DHEXAGON_SDK_PATH=${HEXAGON_SDK_PATH} -DHTP_ARCH_VERSION=${HTP_ARCH_VERSION} -DHEXAGON_SDK_ROOT=${HEXAGON_SDK_PATH} -DHEXAGON_TOOLS_ROOT=${HEXAGON_TOOLS_PATH} --preset arm64-android-snapdragon-debug -DCMAKE_VERBOSE_MAKEFILE:BOOL=${VERBOSE}
    cmake --build ${LOCAL_BUILD_DIR}
    show_pwd
    /bin/rm -f CMakeUserPresets.json
}


function remove_temp_dir()
{
    if [ -d ${LOCAL_BUILD_DIR} ]; then
        echo "remove ${LOCAL_BUILD_DIR} directory"
        rm -rf ${LOCAL_BUILD_DIR}
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


#for Qualcomm's ggml-hexagon backend
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

    # FIRST RUN: no MD5 file --> save it, return CHANGED
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
        # CHANGED --> update MD5
        echo "$current_md5" > "$md5_file"
        return 1
    fi
}


function update_ggml_libs()
{
    adb push ${LOCAL_BUILD_DIR}/bin/*.so ${REMOTE_PATH}/
}


function prepare_run_on_phone()
{
    if [ $# != 1 ]; then
        print "invalid param"
        return
    fi
    program=$1

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

    #for Qualcomm's ggml-hexagon backend
    prepare_ggmlhtp

    adb push ${LOCAL_BUILD_DIR}/bin/${program} ${REMOTE_PATH}/

    adb shell ls -l ${REMOTE_PATH}/libggml-*.so

    adb shell chmod +x ${REMOTE_PATH}/${program}
}


function run_llamacli()
{
    prepare_run_on_phone llama-completion

    echo "${REMOTE_PATH}/llama-completion ${running_params} -st -no-cnv -m ${GGUF_MODEL_NAME} -p \"${PROMPT_STRING}\""
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/llama-completion ${running_params} -st -no-cnv -m ${GGUF_MODEL_NAME} -p \"${PROMPT_STRING}\""

}


function run_llamabench()
{
    prepare_run_on_phone llama-bench

    echo "adb shell \"cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/llama-bench -t 6 --poll 1000 -fa 1 --ubatch-size 1024 -p 200,512,800 -m ${GGUF_MODEL_NAME},/sdcard/qwen1_5-1_8b-chat-q4_0.gguf\""

    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/llama-bench -t 6 --poll 1000 -fa 1 --ubatch-size 1024 -p 200,512,800 -m ${GGUF_MODEL_NAME},/sdcard/qwen1_5-1_8b-chat-q4_0.gguf"
}


function run_threadsafety()
{
    prepare_run_on_phone test-thread-safety

    echo "${REMOTE_PATH}/test-thread-safety -np 2 -m ${GGUF_MODEL_NAME} -p \"hello,world\" -n 256 -ngl 99 "
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/test-thread-safety -np 1  -m ${GGUF_MODEL_NAME} -p \"hello,world\" -n 256 -ngl 99 "

}


function run_test-ops()
{
    prepare_run_on_phone test-backend-ops

    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/test-backend-ops test"

}


function run_test-op()
{
    prepare_run_on_phone test-backend-ops

    echo "adb shell cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/test-backend-ops test -o ${opname}"

    echo "\n"
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/test-backend-ops test -o ${opname}"

}


function run_perf-op()
{
    prepare_run_on_phone test-backend-ops

    echo "adb shell cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/test-backend-ops perf -o ${opname}"

    echo "\n"
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/test-backend-ops perf -o ${opname}"

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
    echo "  $0 build"
    echo "  $0 build_debug (enable debug log for developers on ARM-AP side and cDSP side)"
    echo -e "\n"

    echo "  $0 run_testops"
    echo "  $0 run_llamacli "
    echo "  $0 run_llamabench"
    echo "  $0 run_threadsafety"
    echo "  $0 run_testop     ADD/MUL_MAT                                                    (verify accuracy    of ADD/MUL_MAT)"
    echo "  $0 run_perfop     ADD/MUL_MAT                                                    (verify performance of ADD/MUL_MAT)"

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
    elif [ "$1" == "build" ]; then
        build_ggml_hexagon
        exit 0
    elif [ "$1" == "build_debug" ]; then
        build_ggml_hexagon_debug
        exit 0
    elif [ "$1" == "run_testops" ]; then
        run_test-ops
        exit 0
    elif [ "$1" == "run_llamacli" ]; then
        run_llamacli
        exit 0
    elif [ "$1" == "run_llamabench" ]; then
        run_llamabench
        exit 0
    elif [ "$1" == "run_threadsafety" ]; then
        run_threadsafety
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
    else
        show_usage
        exit 1
    fi
else
    show_usage
    exit 1
fi
