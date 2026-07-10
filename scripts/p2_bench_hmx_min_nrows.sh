#!/usr/bin/env bash
#
# P2 benchmark: HTP_MM_HMX_MIN_NROWS 4 vs 8 vs 16
#
# - Edits ggml/src/ggml-hexagon/htp/matmul-ops.h to set the constant
# - Incremental rebuild (AP + DSP skel)
# - Pushes libggml-hexagon.so + libggmldsp-skel-v79.so to /data/local/tmp
# - Runs 4-model CI (gemma4, qwen3, qwen1, llama3) with llama-bench -p 2048, 3 runs/model
# - Saves per-value log to out/p2_hmx_min_nrows_<VALUE>.log
#
# Usage:
#   ./scripts/p2_bench_hmx_min_nrows.sh <VALUE>
#   ./scripts/p2_bench_hmx_min_nrows.sh 4
#   ./scripts/p2_bench_hmx_min_nrows.sh 8
#   ./scripts/p2_bench_hmx_min_nrows.sh 16
#
# Requires:
#   - Android device connected (adb devices shows "device")
#   - Prebuilts at ~/develop/ggml-hexagon/prebuilts/{android-ndk-r28,Hexagon_SDK,OpenCL_SDK}
#   - gguf models on phone at /sdcard/gemma-4-E2B-it-Q4_0.gguf etc.
#
set -e

# trap to guarantee matmul-ops.h is restored even on failure
MATMUL_H=ggml/src/ggml-hexagon/htp/matmul-ops.h
trap 'if [ -f "$MATMUL_H.bak" ]; then mv -f "$MATMUL_H.bak" "$MATMUL_H" && echo "matmul-ops.h restored (trap)"; fi' EXIT

VALUE=${1:?usage: $0 <4|8|16>}
if [[ ! "$VALUE" =~ ^(4|8|16)$ ]]; then
    echo "ERROR: VALUE must be 4, 8, or 16, got '$VALUE'"
    exit 1
fi

PROJECT_ROOT=/home/zhouwg/develop/ggml-hexagon
cd "$PROJECT_ROOT"

# ---- paths (mirror scripts/build-run-ggmlhexagon-android.sh) ----
TOOLCHAIN_PATH=$PROJECT_ROOT/prebuilts
ANDROID_NDK=$TOOLCHAIN_PATH/android-ndk-r28
ANDROID_NDK_FULLNAME=android-ndk-r28-linux.zip
HEXAGON_SDK_VERSION=6.6.0.0
HEXAGON_TOOLS_VERSION=19.0.07
HEXAGON_SDK_PATH=$TOOLCHAIN_PATH/Hexagon_SDK/$HEXAGON_SDK_VERSION
HEXAGON_TOOLS_PATH=$HEXAGON_SDK_PATH/tools/HEXAGON_Tools/$HEXAGON_TOOLS_VERSION
HTP_ARCH_VERSION=v79
LOCAL_BUILD_DIR=$PROJECT_ROOT/out/ggmlhexagon-android
REMOTE_PATH=/data/local/tmp
NDK_TOOLCHAIN=$ANDROID_NDK/build/cmake/android.toolchain.cmake

LOG_DIR=$PROJECT_ROOT/out/p2_bench_hmx
mkdir -p "$LOG_DIR"
LOG_FILE=$LOG_DIR/value_${VALUE}.log

MATMUL_H=$PROJECT_ROOT/ggml/src/ggml-hexagon/htp/matmul-ops.h
RUNS_PER_MODEL=3
PROMPT_TOKENS=2048
GEN_TOKENS=32

# 1-model CI (gemma4 only). Phone too hot for llama3 too.
declare -A MODELS=(
    [gemma4]=/sdcard/gemma-4-E2B-it-Q4_0.gguf
)

echo "=========================================="
echo "P2 bench: HTP_MM_HMX_MIN_NROWS = $VALUE"
echo "log:     $LOG_FILE"
echo "=========================================="

# ---- adb check ----
adb get-state >/dev/null 2>&1 || { echo "ERROR: no adb device"; exit 1; }
adb shell "ls /sdcard/gemma-4-E2B-it-Q4_0.gguf /sdcard/Qwen3.5-2B-Q4_0.gguf /sdcard/qwen1_5-1_8b-chat-q4_0.gguf /sdcard/llama-3.2-1B-Q4_0.gguf" >/dev/null \
    || { echo "ERROR: missing models on phone"; exit 1; }

# ---- 1. edit matmul-ops.h ----
cp -f "$MATMUL_H" "$MATMUL_H.bak"
sed -i "s/^#define HTP_MM_HMX_MIN_NROWS[[:space:]]*[0-9]\+/#define HTP_MM_HMX_MIN_NROWS   $VALUE/" "$MATMUL_H"
echo "matmul-ops.h: HTP_MM_HMX_MIN_NROWS -> $VALUE"
grep "HTP_MM_HMX_MIN_NROWS" "$MATMUL_H" | head -1

# ---- 2. incremental rebuild ----
echo ""
echo "----- incremental rebuild -----"
export CCACHE_DIR=$PROJECT_ROOT/.ccache

cmake -H. -B"$LOCAL_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DGGML_CCACHE=ON \
    -DCMAKE_TOOLCHAIN_FILE=$NDK_TOOLCHAIN \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest \
    -DGGML_HEXAGON=ON -DLLAMA_CURL=OFF -DGGML_LLAMAFILE=ON \
    -DHEXAGON_SDK_PATH=$HEXAGON_SDK_PATH \
    -DHEXAGON_TOOLS_PATH=$HEXAGON_TOOLS_PATH \
    -DHTP_ARCH_VERSION=$HTP_ARCH_VERSION >/dev/null

cd "$LOCAL_BUILD_DIR"
make -j$(nproc) 2>&1 | tee "$LOG_DIR/build_value_${VALUE}.log" | tail -20
cd "$PROJECT_ROOT"

# restore matmul-ops.h immediately to keep working tree clean
mv -f "$MATMUL_H.bak" "$MATMUL_H"
echo "matmul-ops.h restored"

# ---- 3. push to phone ----
echo ""
echo "----- push to phone -----"
adb push $LOCAL_BUILD_DIR/bin/libggml-hexagon.so            $REMOTE_PATH/libggml-hexagon.so
adb push $LOCAL_BUILD_DIR/bin/libggmldsp-skel-v79.so        $REMOTE_PATH/libggmldsp-skel-v79.so
adb push $LOCAL_BUILD_DIR/bin/llama-bench                   $REMOTE_PATH/llama-bench
adb shell "chmod +x $REMOTE_PATH/llama-bench"

# reset farf bits
adb shell "echo 0x1c > $REMOTE_PATH/llama-bench.farf"

# ---- 4. 1-model CI ----
echo ""
echo "----- 1-model CI: $(printf '%s ' "${!MODELS[@]}") -----"
{
    echo "P2 bench: HTP_MM_HMX_MIN_NROWS=$VALUE"
    echo "device:   $(adb devices | awk 'NR==2{print $1}')"
    echo "build:    $(date -Iseconds)"
    echo "cmd:      llama-bench -p $PROMPT_TOKENS -n $GEN_TOKENS -t 6 --poll 1000 -fa 1 --ubatch-size 1024 --mulmat-algotype 29 -ngl 99 -m <MODEL>"
    echo "runs/model: $RUNS_PER_MODEL"
    echo ""
} | tee "$LOG_FILE"

# warmup: load one model to prime caches
adb shell "cd $REMOTE_PATH && export LD_LIBRARY_PATH=$REMOTE_PATH && $REMOTE_PATH/llama-bench -p 64 -n 4 -t 6 --poll 1000 -fa 1 --ubatch-size 1024 --mulmat-algotype 29 -ngl 99 -m /sdcard/gemma-4-E2B-it-Q4_0.gguf" >/dev/null 2>&1 || true

for alias in gemma4; do
    model_path=${MODELS[$alias]}
    echo "" | tee -a "$LOG_FILE"
    echo "### $alias ($model_path) ###" | tee -a "$LOG_FILE"
    for run in $(seq 1 $RUNS_PER_MODEL); do
        echo "--- run $run/$RUNS_PER_MODEL ---" | tee -a "$LOG_FILE"
        # disable the on-screen progress table by reducing it via -silent
        adb shell "cd $REMOTE_PATH && export LD_LIBRARY_PATH=$REMOTE_PATH && $REMOTE_PATH/llama-bench -p $PROMPT_TOKENS -n $GEN_TOKENS -t 6 --poll 1000 -fa 1 --ubatch-size 1024 --mulmat-algotype 29 -ngl 99 -m $model_path" 2>&1 \
            | tee -a "$LOG_FILE"
        # cool down between runs (thermal: longer sleep)
        sleep 20
    done
    # longer cool-down between models
    sleep 30
    echo "" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "DONE: $LOG_FILE"
echo ""
