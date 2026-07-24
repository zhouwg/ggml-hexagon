## Overview

The steps here has verified on Ubuntu 20.04(EOL on 31 May 2025), Ubuntu 26.04:

Follow the steps below to build llama.cpp + ggml-hexagon backend(Qualcomm's official version) on Linux for Android phone equipped with Qualcomm Snapdragon 8Gen3 and 8Elite.

## Prerequisites

Ubuntu 20.04(EOL on 31 May 2025), Ubuntu 26.04 is recommended.

Upgrade cmake to cmake 4.2.3 when the host OS is Ubuntu 20.04.

```
sudo apt remove -y cmake

wget https://cmake.org/files/v4.2/cmake-4.2.3-linux-x86_64.tar.gz

tar -zxf cmake-4.2.3-linux-x86_64.tar.gz

sudo mv cmake-4.2.3-linux-x86_64 /opt/

sudo ln -sf /opt/cmake-4.2.3-linux-x86_64/bin/* /usr/bin/

cmake --version
```

## Fetch source codes
```
git clone https://github.com/zhouwg/ggml-hexagon.git

cd ggml-hexagon

git checkout self-build-jz
```

## Build
```

$ ./scripts/build-run-android.sh build

```


## How to do performance comparison of PP and TG between Qualcomm's ggml-hexagon and JZ's ggml-hexagon

for **fair performance comparison**, the same "running_params=" -ngl 99 -t 6 -n 256 --ctx-size 8192 --ubatch-size 64 --poll 1000 --no-warmup --no-mmap -fa on" "  and the same prompt and same LLM model file and same 8Elite phone would be used in both inference test.

### We can run automated AB tests on Snapdragon 8 Elite following the recommended steps below:

1. download codes

```
 git clone https://github.com/zhouwg/ggml-hexagon
 cd ggml-hexagon
 git checkout self-build-jz
```

2. build Qualcomm's ggml-hexagon in self-build-jz branch
```
./scripts/build-run-android.sh build_qcom
```

3. build JZ's ggml-hexagon in self-build-jz branch accordingly
```
./scripts/build-run-android.sh build
```
4. run automated AB test

```
./scripts/build-run-android.sh run_abtest 2>&1 | tee log_abtest_$(date +%Y%m%d-%H%M%S).txt
```

5. Analyze logs and generate a performance comparison table from `log_abtest_$(date +%Y%m%d-%H%M%S).txt`



### We can run non-automated AB tests on Snapdragon 8 Elite following the recommended steps below:

1. download codes

```
 git clone https://github.com/zhouwg/ggml-hexagon
 cd ggml-hexagon
 git checkout self-build-jz
```

2. build Qualcomm's ggml-hexagon in self-build-jz branch
```
./scripts/build-run-android.sh build_qcom
```

3. build JZ's ggml-hexagon in self-build-jz branch accordingly
```
./scripts/build-run-android.sh build
```

4. run llama-cli or llama-bench with Qualcomm ggml-hexagon

```
./scripts/build-run-android.sh update_qcom_libs
./scripts/build-run-android.sh run_llamacli
./scripts/build-run-android.sh run_llamabench
```

5. run llama-cli or llama-bench with JZ ggml-hexagon

```
./scripts/build-run-android.sh update_jz_libs
./scripts/build-run-android.sh run_llamacli
./scripts/build-run-android.sh run_llamabench
```
