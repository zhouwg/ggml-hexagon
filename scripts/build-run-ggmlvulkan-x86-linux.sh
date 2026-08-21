#!/usr/bin/env bash
#
# build&verify ggml-vulkan on x86-linux
#
set -e

######## part-1: public macros & vars ########

PWD=`pwd`
PROJECT_HOME_PATH=`pwd`
PROJECT_ROOT_PATH=${PROJECT_HOME_PATH}
HOST_CPU_COUNTS=`cat /proc/cpuinfo | grep "processor" | wc | awk '{print int($1)}'`

VERBOSE=OFF
VERBOSE=ON

#path of built artifacts
LOCAL_BUILD_DIR=${PROJECT_ROOT_PATH}/out/ggmlvulkan-x86-linux

#running path on local x86-linux
LOCAL_PATH=${LOCAL_BUILD_DIR}/bin/

TOOLCHAIN_PATH=${PROJECT_ROOT_PATH}/prebuilts

VULKAN_SDK_VERSION=1.4.304.1
VULKAN_SDK_NAME=vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz
VULKAN_SDK_URL=https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/${VULKAN_SDK_NAME}
VULKAN_SDK_PATH=${TOOLCHAIN_PATH}/Vulkan_SDK

######## part-2: prompt and LLM models ########
#supported models will be downloadded automatically in check_prebuilt_models() when running this script at the first time

LLM_PATH=/media/zhouwg/0893c374-f64c-4121-9192-21e7fd97edef/LLM
GGUF_MODEL_NAME=${LLM_PATH}/gemma-4-E2B-it-Q4_0.gguf

# Model aliases for quick testing of multiple models
# Usage: ./scripts/build-run-ggmlvulkan-x86-linux.sh run_llamacli <alias>
#   qwen3-2b            -> Qwen3.5-2B-Q4_0.gguf
#   qwen3-9b            -> Qwen3.5-9B-Q4_0.gguf
#   gemma4-e2b          -> gemma-4-E2B-it-Q4_0.gguf
#   gemma4-e4b          -> gemma-4-E4B_q4_0-it.gguf
#   qwen1               -> qwen1_5-1_8b-chat-q4_0.gguf
#   llama3              -> Llama-3.2-1B-Instruct-Q4_0.gguf
#   nanbeige-3b         -> Nanbeige_Nanbeige4.2-3B-Q4_0.gguf
#   minicpm5-1b         -> minicpm5-1b-q4_0.gguf
#   (default)           -> gemma-4-E2B-it-Q4_0.gguf
#   nanbeige-3b-q80     -> Nanbeige_Nanbeige4.2-3B-Q8_0.gguf
#   minicpm5-1b-q80     -> MiniCPM5-1B-Q8_0.gguf
#   qwen3-27b           -> Qwen3.8-27B-Q4_0.gguf
function resolve_model_name()
{
    case "$1" in
        qwen3-2b)           echo "/${LLM_PATH}/Qwen3.5-2B-Q4_0.gguf" ;;
        qwen3-9b)           echo "/${LLM_PATH}/Qwen3.5-9B-Q4_0.gguf" ;;
        gemma4-e2b)         echo "/${LLM_PATH}/gemma-4-E2B-it-Q4_0.gguf" ;;
        gemma4-e4b)         echo "/${LLM_PATH}/gemma-4-E4B_q4_0-it.gguf" ;;
        qwen1)              echo "/${LLM_PATH}/qwen1_5-1_8b-chat-q4_0.gguf" ;;
        llama3)             echo "/${LLM_PATH}/Llama-3.2-1B-Instruct-Q4_0.gguf" ;;
        nanbeige-3b)        echo "/${LLM_PATH}/Nanbeige_Nanbeige4.2-3B-Q4_0.gguf";;
        nanbeige-3b-q80)    echo "/${LLM_PATH}/Nanbeige_Nanbeige4.2-3B-Q8_0.gguf";;
        minicpm5-1b)        echo "/${LLM_PATH}/minicpm5-1b-q4_0.gguf";;
        minicpm5-1b-q80)    echo "/${LLM_PATH}/MiniCPM5-1B-Q8_0.gguf";;
        qwen3-27b)          echo "/${LLM_PATH}/Qwen3.8-27B-Q4_0.gguf" ;;
        *)                  echo "" ; return 1 ;;
    esac
}

PROMPT_STRING="Hello, good morning, you are a powerful domain expert and know many things, now pls help to introduce the movie Once Upon a Time in America briefly, pls pay attention short then 1000 words\n"

#running_params=" -ngl 99 -t 6 -n 256 --no-warmup --load-mode none --poll 1000 --cpu-mask 0xfc --cpu-strict 1 --ctx-size 8192 --ubatch-size 1024 -fa on"
running_params=" -ngl 99 -t 6 -n 256 --ctx-size 8192 --ubatch-size 64 --poll 1000 --no-warmup --load-mode none -fa on --jinja -st"

######## part-3: utilities and functions ########

function dump_vars()
{
    echo -e "VULKAN_SDK:          ${VULKAN_SDK}"
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


function build_x86_linux
{
    #make AI Agent happy
    export CCACHE_DIR=${PROJECT_ROOT_PATH}/.ccache_vulkan_x86_linux
    cmake -H. -B${LOCAL_BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=1 -DLLAMA_CUDA=OFF -DGGML_OPENMP=OFF -DGGML_OPENCL=OFF -DGGML_CCACHE=ON -DLLAMA_CURL=OFF -DGGML_LLAMAFILE=ON -DCMAKE_C_FLAGS="${extra_flags}" -DCMAKE_CXX_FLAGS="${extra_flags}" -DCMAKE_VERBOSE_MAKEFILE:BOOL=${VERBOSE} -DGGML_USE_HEXAGON=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=ON -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF
    cd ${LOCAL_BUILD_DIR}
    make -j${HOST_CPU_COUNTS}
    show_pwd
    cd -
}


function remove_temp_dir()
{
    if [ -d ${LOCAL_BUILD_DIR} ]; then
        echo "remove ${LOCAL_BUILD_DIR} directory"
        rm -rf ${LOCAL_BUILD_DIR}
    fi
}


function build_ggml_vulkan_x86_linux()
{
    show_pwd
    dump_vars
    #remove_temp_dir
    build_x86_linux
}


function check_and_download_vulkan_sdk()
{
    is_vulkan_sdk_exist=1

    if [ ! -d ${VULKAN_SDK_PATH} ]; then
        echo -e "VULKAN_SDK_PATH ${VULKAN_SDK_PATH} not exist\n"
        is_vulkan_sdk_exist=0
    fi

    if [ ${is_vulkan_sdk_exist} -eq 0 ]; then
        mkdir -p ${VULKAN_SDK_PATH}
        cd ${VULKAN_SDK_PATH}

        if [ ! -f ${VULKAN_SDK_NAME} ]; then
            echo -e "download vulkan sdk from ${VULKAN_SDK_URL}...\n"
            wget --no-config --quiet --show-progress -O ${VULKAN_SDK_PATH}/${VULKAN_SDK_NAME} ${VULKAN_SDK_URL}
            if [ $? -ne 0 ]; then
                printf "failed to download Vulkan SDK to %s \n" "${VULKAN_SDK_PATH}"
                exit 1
            fi
        fi
        printf "decompress ${VULKAN_SDK_NAME}..."
        tar Jxf ${VULKAN_SDK_NAME}
    else
        printf "VULKAN SDK already exist:    ${VULKAN_SDK_PATH} \n\n"
    fi

    export VULKAN_SDK=${VULKAN_SDK_PATH}/${VULKAN_SDK_VERSION}/x86_64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${VULKAN_SDK}/lib
    export PATH=$PATH:${VULKAN_SDK}/bin
    export VK_LAYER_PATH=$VULKAN_SDK/share/vulkan/explicit_layer.d
    export VK_ADD_LAYER_PATH=$VULKAN_SDK/share/vulkan/explicit_layer.d
    export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$VULKAN_SDK/lib/pkgconfig/
    export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$VULKAN_SDK:$VULKAN_SDK/lib/VulkanLoader
}


function check_and_download_model()
{
    set +e

    model_name=$1
    model_url=$2

    ls ${LLM_PATH}/${model_name}
    if [ $? -eq 0 ]; then
        printf "the prebuild LLM model ${model_name} already exist\n"
    else
        printf "the prebuild LLM model ${model_name} not exist\n"
        printf "downloading from ${model_url}\n"
        wget --no-config --quiet --show-progress -O ${PROJECT_ROOT_PATH}/models/${model_name} ${model_url}
        /bin/cp -fv ${PROJECT_ROOT_PATH}/models/${model_name} ${LLM_PATH}
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

    local vulkan_running_envs="export GGML_VK_MEMORY_LOGGER=1 export GGML_VK_PERF_LOGGER=1"
    local vulkan_running_envs="export GGML_VK_ALLOW_SYSMEM_FALLBACK=1"
    echo "cd ${LOCAL_PATH} \
               && export LD_LIBRARY_PATH=${LOCAL_PATH} \
               && ${LOCAL_PATH}/llama-completion ${running_params} -m ${model_path} -p ${PROMPT_STRING@Q}"
    cd ${LOCAL_PATH} \
               && export LD_LIBRARY_PATH=${LOCAL_PATH} \
               && ${vulkan_running_envs} \
               && ${LOCAL_PATH}/llama-completion ${running_params} -m ${model_path} -p "${PROMPT_STRING}"

}


function run_llamabench()
{
    echo "cd ${LOCAL_PATH} \
               && export LD_LIBRARY_PATH=${LOCAL_PATH} \
               && ${LOCAL_PATH}/llama-bench -t 6 --poll 1000 -ngl 99 -fa 1 --ubatch-size 1024 -p 200,500,800,1024 -n 128 -m ${GGUF_MODEL_NAME}"

    cd ${LOCAL_PATH} \
               && export LD_LIBRARY_PATH=${LOCAL_PATH} \
               && ${LOCAL_PATH}/llama-bench -t 6 --poll 1000 -ngl 99 -fa 1 --ubatch-size 1024 -p 200,500,800,1024 -n 128 -m ${GGUF_MODEL_NAME}
}


function run_llamacli_all()
{
    local models=("qwen1" "minicpm5-1b" "llama3" "qwen3-2b" "gemma4-e2b" "nanbeige-3b" "gemma4-e4b" "qwen3-9b" "qwen3-27b")

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


function run_test-ops()
{
    prog_name=test-backend-ops

    echo "cd ${LOCAL_PATH} \
               && export LD_LIBRARY_PATH=${LOCAL_PATH} \
               && ${LOCAL_PATH}/${prog_name} test"


    cd ${LOCAL_PATH} \
               && export LD_LIBRARY_PATH=${LOCAL_PATH} \
               && ${LOCAL_PATH}/${prog_name} test

}


function run_test-op()
{
    prog_name=test-backend-ops
    prog_param="-o ${opname}"

    echo "cd ${LOCAL_PATH} \
               && export LD_LIBRARY_PATH=${LOCAL_PATH} \
               && ${LOCAL_PATH}/${prog_name} test ${prog_param}"

    echo "\n"
    cd ${LOCAL_PATH} \
               && export LD_LIBRARY_PATH=${LOCAL_PATH} \
               && ${LOCAL_PATH}/${prog_name} test ${prog_param}

}


function run_perf-op()
{
    prog_name=test-backend-ops

    echo "cd ${LOCAL_PATH} \
               && export LD_LIBRARY_PATH=${LOCAL_PATH} \
               && ${LOCAL_PATH}/${prog_name} perf -o ${opname}"

    echo "\n"
    cd ${LOCAL_PATH} \
               && export LD_LIBRARY_PATH=${LOCAL_PATH} \
               && ${LOCAL_PATH}/${prog_name} perf -o ${opname}

}


function show_usage()
{
    echo -e "\n"
    echo "Usage:"
    echo "  $0 help"

    echo "  $0 build"
    echo "  $0 clean"

    echo "  $0 run_testops"
    echo "  $0 run_testop     ADD/MUL_MAT/FLASH_ATTN_EXT (verify accuracy    of ADD/MUL_MAT)"
    echo "  $0 run_perfop     ADD/MUL_MAT/FLASH_ATTN_EXT (verify performance of ADD/MUL_MAT)"
    echo "  $0 run_llamacli"
    echo "  $0 run_llamabench"

    echo "  $0 run_llamacli_all     (batch test 8 models = 8 tests)"
    echo "    Log capture example:"
    echo "      $0 run_llamacli_all 2>&1 | tee log_ci_\$(date +%Y%m%d-%H%M%S).txt"
    echo -e "\n"

    echo "  $0 run_llamacli   [model_alias]"
    echo "  Model aliases for run_llamacli:"
    echo "    qwen3-2b      -> Qwen3.5-2B-Q4_0.gguf"
    echo "    qwen3-9b      -> Qwen3.5-9B-Q4_0.gguf"
    echo "    gemma4-e2b    -> gemma-4-E2B-it-Q4_0.gguf (2.9 GiB)"
    echo "    gemma4-e4b    -> gemma-4-E4B_q4_0-it.gguf (4.9 GiB)"
    echo "    qwen1         -> qwen1_5-1_8b-chat-q4_0.gguf"
    echo "    llama3        -> Llama-3.2-1B-Instruct-Q4_0.gguf"
    echo "    (default)     -> gemma-4-E2B-it-Q4_0.gguf"
    echo "    qwen3-27b     -> Qwen3.8-27B-Q4_0.gguf"
    echo "  Examples:"
    echo "    $0 run_llamacli qwen3-2b     # test qwen3-2b"
    echo "    $0 run_llamacli qwen3-9b     # test qwen3-9b"
    echo "    $0 run_llamacli qwen3-27b    # test qwen3-27b"
    echo "    $0 run_llamacli gemma4-e2b   # test gemma4-e2b"
    echo "    $0 run_llamacli gemma4-e4b   # test gemma4-e4b"
    echo -e "\n"
}


######## part-4: entry point  ########

show_pwd
check_commands_in_host
check_and_download_vulkan_sdk
#check_prebuilt_models

dump_vars

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
    elif [ "$1" == "build" ]; then
        build_ggml_vulkan_x86_linux
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
    elif [ "$1" == "run_llamabench" ]; then
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
