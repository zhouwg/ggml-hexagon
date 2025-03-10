@echo "build llama.cpp for Windows on ARM through llvm-mingw toolchain on x86-64 Windows"

set PATH=C:\Program Files\llvm-mingw-20250305-ggml-ucrt-x86_64\bin;C:\Program Files\llvm-mingw-20250305-ggml-ucrt-x86_64\Git\cmd;C:\Program Files\llvm-mingw-20250305-ggml-ucrt-x86_64\CMake\bin;%PATH%;
cmake --preset arm64-windows-llvm-release -D GGML_OPENMP=OFF -DGGML_QNN=ON -DCMAKE_CXX_FLAGS=-D_WIN32_WINNT=0x602 -DGGML_QNN_SDK_PATH="C:\\qairt\\2.32.0.250228"
cmake --build build-arm64-windows-llvm-release
