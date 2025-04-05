#!/bin/bash
# build llama.cpp + ggml-hexagon for Snapdragon mobile SoC equipped Android phone on Linux

set -e

PWD=`pwd`
ANDROID_PLATFORM=android-34
ANDROID_NDK=${PWD}/android-ndk-r26c
REMOTE_PATH=/data/local/tmp/
GGUF_MODEL_NAME=/sdcard/gemma-3-4b-it-Q8_0.gguf
GGUF_MODEL_NAME=/sdcard/qwen1_5-1_8b-chat-q4_0.gguf

#QNN SDK could be found at:
#https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
#https://developer.qualcomm.com/software/hexagon-dsp-sdk/tools
QNN_SDK_URL=https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
QNN_SDK_INSTALL_PATH=/opt/qcom/aistack/qairt/
QNN_SDK_VERSION=2.32.0.250228
QNN_SDK_PATH=${QNN_SDK_INSTALL_PATH}/${QNN_SDK_VERSION}

HEXAGON_SDK_PATH=/opt/qcom/Hexagon_SDK/6.2.0.1
#available htp arch version:
#v68 --- Snapdragon 888
#v69 --- Snapdragon 8 Gen1
#v73 --- Snapdragon 8 Gen2
#v75 --- Snapdragon 8 Gen3
#v79 --- Snapdragon 8 Elite(aka Gen4)
HTP_ARCH_VERSION=v75

#running_params=" -mg 2 -ngl 99 "
#running_params=" -mg 2 -ngl 99 -t 8 -fa 1 "
running_params=" -mg 2 -ngl 99 -t 8 "

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


function check_hexagon_sdk()
{
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
        echo -e "QNN_SDK_PATH ${QNN_SDK_PATH} not exist, download it from ${QNN_SDK_URL}...\n"
        is_qnn_sdk_exist=0
    fi

    if [ ! -f ${QNN_SDK_PATH}/sdk.yaml ]; then
        is_qnn_sdk_exist=0
    fi

    if [ ${is_qnn_sdk_exist} -eq 0 ]; then
        echo "sudo mkdir -p ${QNN_SDK_INSTALL_PATH}"
        sudo mkdir -p ${QNN_SDK_INSTALL_PATH}
        if [ ! -f v${QNN_SDK_VERSION}.zip ]; then
            wget --no-config --quiet --show-progress -O v${QNN_SDK_VERSION}.zip https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/${QNN_SDK_VERSION}/v${QNN_SDK_VERSION}.zip
        fi
        unzip v${QNN_SDK_VERSION}.zip
        if [ $? -ne 0 ]; then
            printf "failed to download Qualcomm QNN SDK to %s \n" "${QNN_SDK_PATH}"
            exit 1
        fi
        sudo mv qairt/${QNN_SDK_VERSION} ${QNN_SDK_INSTALL_PATH}/
        printf "Qualcomm QNN SDK saved to ${QNN_SDK_PATH} \n\n"
        sudo rm -rf qairt
    else
        printf "Qualcomm QNN SDK already exist:${QNN_SDK_PATH} \n\n"
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

        if [ ! -f android-ndk-r26c-linux.zip ]; then
            wget --no-config --quiet --show-progress -O android-ndk-r26c-linux.zip  https://dl.google.com/android/repository/android-ndk-r26c-linux.zip
        fi

        unzip android-ndk-r26c-linux.zip

        if [ $? -ne 0 ]; then
            printf "failed to download android ndk to %s \n" "${ANDROID_NDK}"
            exit 1
        fi

        printf "android ndk saved to ${ANDROID_NDK} \n\n"
    else
        printf "android ndk already exist:${ANDROID_NDK} \n\n"
    fi
}


function build_arm64
{
    cmake -H. -B./out/android -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DCMAKE_C_FLAGS=-march=armv8.7-a -DGGML_HEXAGON=ON -DQNN_SDK_PATH=${QNN_SDK_PATH} -DHEXAGON_SDK_PATH=${HEXAGON_SDK_PATH} -DHTP_ARCH_VERSION=${HTP_ARCH_VERSION}
    cd out/android
    make -j16
    show_pwd

    cd -
}


function remove_temp_dir()
{
    if [ -d out/android ]; then
        echo "remove out/android directory in `pwd`"
        rm -rf out/android
    fi
}


function check_qnn_libs()
{
    #reuse the cached qnn libs on Android phone
    adb shell ls ${REMOTE_PATH}/libQnnCpu.so
    adb shell ls ${REMOTE_PATH}/libQnnGpu.so
    adb shell ls ${REMOTE_PATH}/libQnnHtp.so
    if [ $? -eq 0 ]; then
        printf "QNN libs already exist on Android phone\n"
    else
        update_qnn_libs
    fi
    update_qnn_cfg
}


function update_qnn_libs()
{
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnSystem.so              ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnCpu.so                 ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnGpu.so                 ${REMOTE_PATH}/

    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtp.so                 ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtpNetRunExtensions.so ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtpPrepare.so          ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtpV75Stub.so          ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so     ${REMOTE_PATH}/
}


function update_qnn_cfg()
{
    adb push ./scripts/ggml-hexagon.cfg ${REMOTE_PATH}/
}


function build_ggml_qnn()
{
    show_pwd
    check_and_download_ndk
    check_and_download_qnn_sdk
    check_hexagon_sdk
    dump_vars
    remove_temp_dir
    build_arm64
}


function prepare_run_on_phone()
{
    if [ $# != 1 ]; then
        print "invalid param"
        return
    fi
    program=$1

    check_qnn_libs

    if [ -f ./out/android/bin/libggml-cpu.so ]; then
        adb push ./out/android/bin/*.so ${REMOTE_PATH}/
    fi
    adb push ./out/android/bin/${program} ${REMOTE_PATH}/
    adb shell chmod +x ${REMOTE_PATH}/${program}
    adb push ggml/src/ggml-hexagon/kernels/libggmlop_skel${HTP_ARCH_VERSION}.so  ${REMOTE_PATH}/libggmlop_skel.so
}

function run_llamacli()
{
    prepare_run_on_phone llama-cli

    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/llama-cli ${running_params} -no-cnv -m ${GGUF_MODEL_NAME} -p \"introduce the movie Once Upon a Time in America briefly.\n\""

}


function run_llamabench()
{
    prepare_run_on_phone llama-bench

    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/llama-bench ${running_params} -m ${GGUF_MODEL_NAME}"

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
               && ${REMOTE_PATH}/test-backend-ops test -o $opname "

    echo "\n"
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/test-backend-ops test -o $opname "

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
    echo "Usage:"
    echo "  $0 help"
    echo "  $0 print_oplist"
    echo "  $0 build"
    echo "  $0 updateqnnlib"
    echo "  $0 run_testops"
    echo "  $0 run_testop          [ADD/MUL_MAT]"
    echo "  $0 run_llamacli"
    echo "  $0 run_llamabench"

    echo -e "\n\n\n"
}


show_pwd

check_and_download_ndk
check_and_download_qnn_sdk
check_hexagon_sdk

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
        build_ggml_qnn
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
    elif [ "$1" == "updateqnnlib" ]; then
        update_qnn_libs
        exit 0
    else
        show_usage
        exit 1
    fi
elif [ $# == 2 ]; then
    opname=$2
#TODO: check opname in oplist
#opname can be found via print_oplist:

    run_test-op
    exit 0
else
    show_usage
    exit 1
fi
