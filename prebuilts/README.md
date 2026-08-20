### Compliance Statement

Currently, Qualcomm's Hexagon SDK can't downloaded automatically via the script [build-run-ggmlhexagon-android.sh](https://github.com/zhouwg/ggml-hexagon/blob/self-build-jz/scripts/build-run-ggmlhexagon-android.sh) because of well-known and make-sense IPR policy. I provide a customized&tailored minimal Hexagon SDK to **simplify workflow** of build ggml-hexagon under the premise of **strictly abiding by Qualcomm's IPR policy.**

### Contents in this directory

- Hexagon_SDK: a [customized/tailored Qualcomm's Hexagon SDK](https://github.com/snapdragon-toolchain/hexagon-sdk/releases/download/v6.6.0.0/hexagon-sdk-v6.6.0.0-amd64-lnx.tar.xz) for build project ggml-hexagon conveniently and will be downloaded automatically via build-run-ggmlhexagon-android.sh. the fully Hexagon SDK could be found at Qualcomm's offcial website: https://developer.qualcomm.com/software/hexagon-dsp-sdk/tools. one more important thing, the fully Hexagon SDK must be obtained with a Qualcomm Developer Account.

- android-ndk-r29, will be downloaded automatically via [build-run-ggmlhexagon-android.sh](https://github.com/zhouwg/ggml-hexagon/blob/self-build-jz/scripts/build-run-ggmlhexagon-android.sh)

- OpenCL_SDK, will be downloaded automatically via [build-run-ggmlhexagon-android.sh](https://github.com/zhouwg/ggml-hexagon/blob/self-build-jz/scripts/build-run-ggmlhexagon-android.sh)

- Vulkan_SDK, will be downloaded automatically via [build-run-ggmlvulkan-x86-linux.sh](https://github.com/zhouwg/ggml-hexagon/blob/self-build-jz/scripts/build-run-ggmlvulkan-x86-linux.sh)
