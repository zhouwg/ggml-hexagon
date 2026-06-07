### Compliance Statement

Currently, Qualcomm's Hexagon SDK can't downloaded automatically via the script [build-run-ggmlhexagon-android.sh](https://github.com/zhouwg/ggml-hexagon/blob/self-build/scripts/build-run-ggmlhexagon-android.sh) because of well-known and make-sense IPR policy. I provide a customized&tailored minimal Hexagon SDK to **simplify workflow** of build ggml-hexagon under the premise of **strictly abiding by Qualcomm's IPR policy.**

### Contents in this directory

- QNN_SDK: the fully QNN SDK could be found at Qualcomm's offcial website: https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk, will be downloaded automatically via [build-run-ggmlhexagon-android.sh](https://github.com/zhouwg/ggml-hexagon/blob/self-build/scripts/build-run-ggmlhexagon-android.sh).

- Hexagon_SDK: a [customized/tailored Qualcomm's Hexagon SDK](https://github.com/zhouwg/toolchain/blob/main/minimal-hexagon-sdk-6.2.0.1.xz) for build project ggml-hexagon conveniently and will be downloaded automatically via build-run-ggmlhexagon-android.sh. the fully Hexagon SDK could be found at Qualcomm's offcial website: https://developer.qualcomm.com/software/hexagon-dsp-sdk/tools. one more important thing, the fully Hexagon SDK must be obtained with a Qualcomm Developer Account.

- [ggml-dsp](https://github.com/zhouwg/ggml-hexagon/tree/self-build/prebuilts/ggml-dsp): prebuilt libggmldsp-skel.so for Qualcomm Hexagon NPU on Android phone equipped with Qualcomm Snapdragon <b>high-end</b> mobile SoC.
