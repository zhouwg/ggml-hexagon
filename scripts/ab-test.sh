#!/bin/bash

PROJECT_HOME_PATH=`pwd`
PROJECT_ROOT_PATH=${PROJECT_HOME_PATH}
AB_TEST_DIR=${PROJECT_ROOT_PATH}/out/ab-test
REMOTE_PATH=/data/local/tmp
MODEL=/sdcard/gemma-4-E2B-it-Q4_0.gguf
PROMPT_STRING="Hello, good morning, you are a powerful domain expert and know many things, now pls help to introduce the movie Once Upon a Time in America briefly, pls pay attention short then 1000 words\n"
PARAMS="-ngl 99 -t 6 -n 256 --ctx-size 8192 --ubatch-size 64 --poll 1000 --no-warmup --no-mmap -fa on --jinja -st"

switch_to_jz() {
    echo "=== [$(date '+%H:%M:%S')] Switching to JZ ==="
    adb push ${AB_TEST_DIR}/libggml-hexagon-jz.so           ${REMOTE_PATH}/libggml-hexagon.so
    adb push ${AB_TEST_DIR}/libggml-jz.so                   ${REMOTE_PATH}/libggml.so
    adb push ${AB_TEST_DIR}/libllama-jz.so                  ${REMOTE_PATH}/libllama.so
    adb push ${AB_TEST_DIR}/libllama-common-jz.so           ${REMOTE_PATH}/libllama-common.so
    adb push ${AB_TEST_DIR}/libllama-completion-impl-jz.so  ${REMOTE_PATH}/libllama-completion-impl.so
    adb push ${AB_TEST_DIR}/libllama-bench-impl-jz.so       ${REMOTE_PATH}/libllama-bench-impl.so
    adb push ${AB_TEST_DIR}/libggmldsp-skel-v75.so          ${REMOTE_PATH}/
    adb push ${AB_TEST_DIR}/libggmldsp-skel-v79.so          ${REMOTE_PATH}/
    adb shell rm -f ${REMOTE_PATH}/libggml-htp-v75.so ${REMOTE_PATH}/libggml-htp-v79.so
    echo "=== JZ .so deployed ==="
}

switch_to_qcom() {
    echo "=== [$(date '+%H:%M:%S')] Switching to QCOM ==="
    adb push ${AB_TEST_DIR}/libggml-hexagon-qcom.so          ${REMOTE_PATH}/libggml-hexagon.so
    adb push ${AB_TEST_DIR}/libggml-qcom.so                  ${REMOTE_PATH}/libggml.so
    adb push ${AB_TEST_DIR}/libllama-qcom.so                 ${REMOTE_PATH}/libllama.so
    adb push ${AB_TEST_DIR}/libllama-common-qcom.so          ${REMOTE_PATH}/libllama-common.so
    adb push ${AB_TEST_DIR}/libllama-completion-impl-qcom.so ${REMOTE_PATH}/libllama-completion-impl.so
    adb push ${AB_TEST_DIR}/libllama-bench-impl-qcom.so      ${REMOTE_PATH}/libllama-bench-impl.so
    adb push ${AB_TEST_DIR}/libggml-htp-v75.so               ${REMOTE_PATH}/
    adb push ${AB_TEST_DIR}/libggml-htp-v79.so               ${REMOTE_PATH}/
    adb shell rm -f ${REMOTE_PATH}/libggmldsp-skel-v75.so ${REMOTE_PATH}/libggmldsp-skel-v79.so
    echo "=== QCOM .so deployed ==="
}

# JZ is already on device from last test, but re-push to be sure
switch_to_jz

echo ""
echo "========================================"
echo "  JZ clean test (3 runs)"
echo "========================================"
for i in 1 2 3; do
    echo ""
    echo "-------- JZ run ${i} $(date '+%H:%M:%S') --------"
    adb shell "cd ${REMOTE_PATH} && export LD_LIBRARY_PATH=${REMOTE_PATH} && ${REMOTE_PATH}/llama-completion ${PARAMS} -m ${MODEL} -p \"${PROMPT_STRING}\"" 2>&1 | tee /tmp/jz-test.log
    echo "-------- JZ run ${i} END --------"
done

switch_to_qcom

echo ""
echo "========================================"
echo "  QCOM clean test (3 runs)"
echo "========================================"
for i in 1 2 3; do
    echo ""
    echo "-------- QCOM run ${i} $(date '+%H:%M:%S') --------"
    adb shell "cd ${REMOTE_PATH} && export LD_LIBRARY_PATH=${REMOTE_PATH} && ${REMOTE_PATH}/llama-completion ${PARAMS} -m ${MODEL} -p \"${PROMPT_STRING}\"" 2>&1 | tee /tmp/qcom-test.log
    echo "-------- QCOM run ${i} END --------"
done

echo ""
echo "========================================"
echo "  All tests complete $(date '+%H:%M:%S')"
echo "========================================"
