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

#Qualcomm QNN SDK can be found at:
#https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
QNN_SDK_URL=https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
QNN_SDK_VERSION=2.32.0.250228
QNN_SDK_VERSION=2.33.0.250327
QNN_SDK_VERSION=2.34.0.250424
QNN_SDK_VERSION=2.35.0.250530
QNN_SDK_VERSION=2.36.0.250627
QNN_SDK_VERSION=2.46.0.260424
#fully official QNN SDK, will be downloaded automatically via this script
QNN_SDK_PATH=${PROJECT_ROOT_PATH}/prebuilts/QNN_SDK/qairt/${QNN_SDK_VERSION}/

#fully Qualcomm Hexagon SDK can be found at https://developer.qualcomm.com/software/hexagon-dsp-sdk/tools.
#fully Hexagon SDK must be obtained with Qualcomm Developer Account and follow PKLA&ECA.
#only for purpose of test/development
#HEXAGON_SDK_VERSION=6.2.0.1
#HEXAGON SDK 6.3.0.0 is required for htp v81 but skipped in this project due to Qualcomm's IPR policy(Product Kit License Agreement)
#HEXAGON_SDK_VERSION=6.3.0.0
#HEXAGON_SDK_PATH=/opt/qcom/Hexagon_SDK/${HEXAGON_SDK_VERSION}

#the official Qualcomm Hexagon SDK tech docs can be found at:
#https://docs.qualcomm.com/bundle/publicresource/topics/80-77512-1/hexagon-dsp-sdk-collection-landing-page.html?product=1601111740010422
#customized/tailored Hexagon SDK for simplify workflow and can be downloaded via this script
#this highly tailored minimal-hexagon-sdk should comply with Qualcomm's IPR policy.
#actually used in this project
HEXAGON_SDK_PATH=${PROJECT_ROOT_PATH}/prebuilts/Hexagon_SDK/6.2.0.1

HEXAGON_TOOLS_PATH=${HEXAGON_SDK_PATH}/tools/HEXAGON_Tools/8.8.06
HEXAGON_PRESET_PATH=${PROJECT_ROOT_PATH}/docs/backend/snapdragon

#supported htp arch version:
#v68 --- Snapdragon 888
#v69 --- Snapdragon 8 Gen1
#v73 --- Snapdragon 8 Gen2
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

#for jz's ggml-hexagon backend
#available prebuilt libs can be found at prebuilts/ggml-dsp
GGMLDSP_RELEASE_DATE=20250531
GGMLDSP_RELEASE_DATE=20250609
GGMLDSP_RELEASE_DATE=20250625
GGMLDSP_RELEASE_DATE=20250627
GGMLDSP_RELEASE_DATE=20250710

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
    echo -e "QNN_SDK_PATH:         ${QNN_SDK_PATH}"
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
    if [ ! -f ${PROJECT_ROOT_PATH}/prebuilts/Hexagon_SDK/6.2.0.1/tools/HEXAGON_Tools/8.8.06/NOTICE.txt ]; then
        echo -e "${TEXT_RED}minimal-hexagon-sdk not exist...${TEXT_RESET}\n"
        is_hexagon_llvm_exist=0
    fi

    if [ ${is_hexagon_llvm_exist} -eq 0 ]; then
        if [ -f ${PROJECT_ROOT_PATH}/prebuilts/Hexagon_SDK/minimal-hexagon-sdk-6.2.0.1.xz ]; then
            echo -e "minimal-hexagon-sdk-6.2.0.1.xz already exist\n"
        else
            echo -e "begin downloading minimal-hexagon-sdk-6.2.0.1.xz \n"
            wget --no-config --quiet --show-progress -O ${PROJECT_ROOT_PATH}/prebuilts/Hexagon_SDK/minimal-hexagon-sdk-6.2.0.1.xz https://github.com/zhouwg/toolchain/raw/refs/heads/master/minimal-hexagon-sdk-6.2.0.1.xz
            if [ $? -ne 0 ]; then
                printf "failed to download minimal-hexagon-sdk-6.2.0.1.xz\n"
                exit 1
            fi
        fi

        echo -e "begin decompressing minimal-hexagon-sdk-6.2.0.1.xz \n"
        xzcat ${PROJECT_ROOT_PATH}/prebuilts/Hexagon_SDK/minimal-hexagon-sdk-6.2.0.1.xz | tar -C ${PROJECT_ROOT_PATH}/prebuilts/Hexagon_SDK/ -xf -
        if [ $? -ne 0 ]; then
            printf "failed to decompress minimal-hexagon-sdk-6.2.0.1.xz\n"
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


function check_and_download_qnn_sdk()
{
    is_qnn_sdk_exist=1

    if [ ! -d ${QNN_SDK_PATH} ]; then
        echo -e "QNN_SDK_PATH ${QNN_SDK_PATH} not exist\n"
        is_qnn_sdk_exist=0
    fi

    if [ ! -f ${QNN_SDK_PATH}/NOTICE.txt ]; then
        echo -e "${TEXT_RED}${QNN_SDK_PATH}/NOTICE.txt not exist${TEXT_RESET}\n"
        is_qnn_sdk_exist=0
    fi

    if [ ${is_qnn_sdk_exist} -eq 0 ]; then
        if [ ! -f ${PROJECT_ROOT_PATH}/prebuilts/QNN_SDK/v${QNN_SDK_VERSION}.zip ]; then
            echo -e "QNN SDK not exist, download it from ${QNN_SDK_URL}...\n"
            wget --no-config --quiet --show-progress -O ${PROJECT_ROOT_PATH}/prebuilts/QNN_SDK/v${QNN_SDK_VERSION}.zip https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/${QNN_SDK_VERSION}/v${QNN_SDK_VERSION}.zip
        fi
        if [ $? -ne 0 ]; then
            printf "failed to download Qualcomm QNN SDK to %s \n" "${QNN_SDK_PATH}"
            exit 1
        fi
        cd ${PROJECT_ROOT_PATH}/prebuilts/QNN_SDK/
        unzip v${QNN_SDK_VERSION}.zip
        if [ $? -ne 0 ]; then
            printf "failed to decompress Qualcomm QNN SDK to %s \n" "${QNN_SDK_PATH}"
            exit 1
        fi
        printf "Qualcomm QNN SDK saved to ${QNN_SDK_PATH} \n\n"
        cd ${PROJECT_ROOT_PATH}
    else
        printf "Qualcomm QNN SDK already exist:    ${QNN_SDK_PATH} \n\n"
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
    #not acutually used at the moment, just for AI experts add other AI operators in the future
    #if [ -f ${HEXAGON_SDK_PATH}/ipc/fastrpc/qaic/bin/qaic ]; then
    #    ${HEXAGON_SDK_PATH}/ipc/fastrpc/qaic/bin/qaic -mdll -o ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/kernels -I${HEXAGON_SDK_PATH}/incs -I${HEXAGON_SDK_PATH}/incs/stddef -I${HEXAGON_SDK_PATH}/ipc/fastrpc/incs ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/kernels/ggmlop.idl
    #fi

    /bin/cp -fv ${HEXAGON_PRESET_PATH}/CMakeUserPresets.json .
    cmake -H. -B${LOCAL_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DGGML_HEXAGON=ON -DLLAMA_CURL=OFF -DGGML_LLAMAFILE=ON -DQNN_SDK_PATH=${QNN_SDK_PATH} -DHEXAGON_SDK_PATH=${HEXAGON_SDK_PATH} -DHTP_ARCH_VERSION=${HTP_ARCH_VERSION} -DHEXAGON_SDK_ROOT=${HEXAGON_SDK_PATH} -DHEXAGON_TOOLS_ROOT=${HEXAGON_TOOLS_PATH} --preset arm64-android-snapdragon-release -DCMAKE_VERBOSE_MAKEFILE:BOOL=${VERBOSE}
    cmake --build ${LOCAL_BUILD_DIR}
    show_pwd
    /bin/rm -f CMakeUserPresets.json
}


function build_arm64_debug
{
    /bin/cp -fv ${HEXAGON_PRESET_PATH}/CMakeUserPresets.json .

    cmake -H. -B${LOCAL_BUILD_DIR} -DCMAKE_BUILD_TYPE=Debug -DGGML_OPENMP=OFF -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DGGML_HEXAGON=ON -DLLAMA_CURL=OFF -DGGML_LLAMAFILE=ON -DQNN_SDK_PATH=${QNN_SDK_PATH} -DHEXAGON_SDK_PATH=${HEXAGON_SDK_PATH} -DHTP_ARCH_VERSION=${HTP_ARCH_VERSION} -DHEXAGON_SDK_ROOT=${HEXAGON_SDK_PATH} -DHEXAGON_TOOLS_ROOT=${HEXAGON_TOOLS_PATH} --preset arm64-android-snapdragon-debug -DCMAKE_VERBOSE_MAKEFILE:BOOL=${VERBOSE}
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


function check_qnn_libs()
{
    set +e

    #reuse the cached qnn libs on Android phone
    adb shell ls ${REMOTE_PATH}/libQnnCpu.so
    adb shell ls ${REMOTE_PATH}/libQnnGpu.so
    adb shell ls ${REMOTE_PATH}/libQnnHtp.so
    if [ $? -eq 0 ]; then
        printf "QNN runtime libs already exist on Android phone\n"
    else
        printf "QNN runtime libs not exist on Android phone\n"
        update_qnn_libs
    fi
    update_qnn_cfg

    set -e
}


function update_qnn_libs()
{
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnSystem.so              ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnCpu.so                 ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnGpu.so                 ${REMOTE_PATH}/

    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtp.so                 ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtpNetRunExtensions.so ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtpPrepare.so          ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtp${HTP_ARCH_VERSION_a}Stub.so          ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/hexagon-${HTP_ARCH_VERSION}/unsigned/libQnnHtp${HTP_ARCH_VERSION_a}Skel.so     ${REMOTE_PATH}/
}


function update_qnn_cfg()
{
    adb push ./scripts/ggml-hexagon.cfg ${REMOTE_PATH}/
}


function build_ggml_hexagon()
{
    show_pwd
    check_and_download_ndk
    check_and_download_opencl_sdk
    check_and_download_qnn_sdk
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
    check_and_download_qnn_sdk
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


#for jz's ggml-hexagon backend
function prepare_ggmldsp()
{
    adb push ./scripts/ggml-hexagon-for-binary-lib.cfg ${REMOTE_PATH}/ggml-hexagon.cfg
    echo "adb push ${PROJECT_ROOT_PATH}/prebuilts/ggml-dsp/${GGMLDSP_RELEASE_DATE}/libggmldsp-skel${HTP_ARCH_VERSION}.so ${REMOTE_PATH}/libggmldsp-skel.so"
case "$HTP_ARCH_VERSION" in
    v69)
        adb push ${PROJECT_ROOT_PATH}/prebuilts/ggml-dsp/${GGMLDSP_RELEASE_DATE}/libggmldsp-skel${HTP_ARCH_VERSION}.so ${REMOTE_PATH}/libggmldsp-skel.so
    ;;
    v73)
        adb push ${PROJECT_ROOT_PATH}/prebuilts/ggml-dsp/${GGMLDSP_RELEASE_DATE}/libggmldsp-skel${HTP_ARCH_VERSION}.so ${REMOTE_PATH}/libggmldsp-skel.so
    ;;
    v75)
        adb push ${PROJECT_ROOT_PATH}/prebuilts/ggml-dsp/${GGMLDSP_RELEASE_DATE}/libggmldsp-skel${HTP_ARCH_VERSION}.so ${REMOTE_PATH}/libggmldsp-skel.so
    ;;
    v79)
        adb push ${PROJECT_ROOT_PATH}/prebuilts/ggml-dsp/${GGMLDSP_RELEASE_DATE}/libggmldsp-skel${HTP_ARCH_VERSION}.so ${REMOTE_PATH}/libggmldsp-skel.so
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
    check_and_download_model qwen1_5-1_8b-chat-q4_0.gguf  https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat-GGUF/resolve/main/qwen1_5-1_8b-chat-q4_0.gguf

    #1.2 GiB
    check_and_download_model Qwen3.5-2B-Q4_0.gguf         https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_0.gguf

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
    #for branch self-build
    adb push ${LOCAL_BUILD_DIR}/bin/*.so ${REMOTE_PATH}/
    #for branch self-build-jz
    #adb push ${LOCAL_BUILD_DIR}/bin/libggml-base.so                 ${REMOTE_PATH}/
    #adb push ${LOCAL_BUILD_DIR}/bin/libggml-cpu.so                  ${REMOTE_PATH}/
    #adb push ${LOCAL_BUILD_DIR}/bin/libggml-hexagon.so              ${REMOTE_PATH}/
    #adb push ${LOCAL_BUILD_DIR}/bin/libggml.so                      ${REMOTE_PATH}/
    #adb push ${LOCAL_BUILD_DIR}/bin/libllama-common.so              ${REMOTE_PATH}/
    #adb push ${LOCAL_BUILD_DIR}/bin/libllama-completion-impl.so     ${REMOTE_PATH}/
    #adb push ${LOCAL_BUILD_DIR}/bin/libllama-bench-impl.so          ${REMOTE_PATH}/
    #adb push ${LOCAL_BUILD_DIR}/bin/libllama.so                     ${REMOTE_PATH}/
}


function prepare_run_on_phone()
{
    if [ $# != 1 ]; then
        print "invalid param"
        return
    fi
    program=$1

    check_qnn_libs

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

    #for jz's ggml-hexagon backend
    #comment this line when build library on Hexagon cDSP from the reference/self-develop source codes in this project
    #prepare_ggmldsp
    #un-comment this line when build library on Hexagon cDSP from the reference/self-develop source codes in this project
    #adb push ./scripts/ggml-hexagon.cfg ${REMOTE_PATH}/ggml-hexagon.cfg

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


function check_hexagon_backend
{
    if [[ ${hexagon_backend} != 0 ]] && [[ ${hexagon_backend} != 1 ]] && [[ ${hexagon_backend} != 2 ]] && [[ ${hexagon_backend} != 3 ]] && [[ ${hexagon_backend} != 4 ]] ; then
        printf "invalid hexagon backend\n"
        printf "valid hexagon backend: 0(QNN_CPU), 1(QNN_GPU), 2(QNN_NPU), 3(cDSP), 4(ggml)\n"
        exit 1
    fi
}


function check_mulmat_algotype
{
    printf "mulmat_algotype ${mulmat_algotype} \n"
    if [[ ${mulmat_algotype} != 0 ]] && [[ ${mulmat_algotype} != 1 ]] && [[ ${mulmat_algotype} != 2 ]] && [[ ${mulmat_algotype} != 3 ]] && [[ ${mulmat_algotype} != 4 ]] && [[ ${mulmat_algotype} != 5 ]] && [[ ${mulmat_algotype} != 6 ]] && [[ ${mulmat_algotype} != 32 ]] && [[ ${mulmat_algotype} != 33 ]]; then
        printf "invalid mulmat algotype\n"
        printf "valid mulmat algotype: 0, 1, 2, 3, 4, 5, 6, 32, 33 \n"
        exit 1
    fi
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


function run_benchmark()
{
    prepare_run_on_phone ggmlhexagon-benchmark

    check_mulmat_algotype

    echo "${REMOTE_PATH}/ggmlhexagon-benchmark -t ${opname} -b ${hexagon_backend} -m ${row} -n ${col} -a ${mulmat_algotype}"
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/ggmlhexagon-benchmark -t ${opname} -b ${hexagon_backend} -m ${row} -n ${col} -a ${mulmat_algotype}"

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
    echo "  $0 updateqnnlib"
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
check_and_download_qnn_sdk
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
    elif [ "$1" == "updateqnnlib" ]; then
        update_qnn_libs
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
        mulmat_algotype=32
        hexagon_backend=3
        run_test-op
        exit 0
    elif [ "$1" == "run_perfop" ]; then
        opname=$2
        mulmat_algotype=32
        hexagon_backend=3
        check_hexagon_backend
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
