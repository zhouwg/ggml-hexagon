# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# tools - only build what's needed for ggml-hexagon

if (EMSCRIPTEN)
else()
    add_subdirectory(llama-bench)
    add_subdirectory(completion)
endif()
