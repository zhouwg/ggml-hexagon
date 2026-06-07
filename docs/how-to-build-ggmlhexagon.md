### Overview

The steps here has verified on Ubuntu 20.04(EOL on 31 May 2025), Ubuntu 26.04:

Follow the steps below to build llama.cpp + ggml-hexagon backend(Qualcomm's official version) on Linux for Android phone equipped with Qualcomm Snapdragon 8Gen3 and 8Elite.

### Prerequisites

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

### Fetch source codes
```
git clone https://github.com/zhouwg/ggml-hexagon.git

cd ggml-hexagon

git checkout self-build
```

### Build
```

$ ./scripts/build-run-android.sh build

```
