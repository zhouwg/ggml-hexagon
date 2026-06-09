# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# tools - keep core tools for ggml-hexagon development
# Removed: ui, mtmd, parser, tts, rpc, cvector-generator, export-lora, results, tokenize, imatrix, gguf-split

if (EMSCRIPTEN)
else()
    add_subdirectory(batched-bench)
    add_subdirectory(completion)
    add_subdirectory(fit-params)
    add_subdirectory(llama-bench)
    add_subdirectory(perplexity)
    add_subdirectory(quantize)
endif()
