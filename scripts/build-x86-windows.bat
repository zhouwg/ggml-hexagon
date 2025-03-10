@echo "build llama.cpp for x86-64 Windows through llvm-mingw toolchain on x86-64 Windows"

set PATH=C:\Program Files\llvm-mingw-20250305-ggml-ucrt-x86_64\bin;C:\Program Files\llvm-mingw-20250305-ggml-ucrt-x86_64\Git\cmd;C:\Program Files\llvm-mingw-20250305-ggml-ucrt-x86_64\CMake\bin;%PATH%;
cmake --preset x64-windows-llvm-release -D GGML_OPENMP=OFF -DCMAKE_CXX_FLAGS=-D_WIN32_WINNT=0x602
cmake --build build-x64-windows-llvm-release
