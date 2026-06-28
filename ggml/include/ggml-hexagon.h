#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_HEXAGON_MAX_DEVICES    16
#define GGML_HEXAGON_BACKEND_NAME   "hexagon"

GGML_BACKEND_API ggml_backend_t     ggml_backend_hexagon_init(size_t dev_num, const char * runtime_libpath);

GGML_BACKEND_API bool               ggml_backend_is_hexagon(ggml_backend_t backend);

GGML_BACKEND_API int                ggml_backend_hexagon_get_device_count(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_hexagon_reg(void);

GGML_BACKEND_API const char *       ggml_backend_hexagon_get_devname(size_t dev_num);

GGML_BACKEND_API int                ggml_backend_hexagon_get_mulmat_algotype(void);

GGML_BACKEND_API void               ggml_backend_hexagon_set_mulmat_algotype(int new_mulmat_algotype);

#ifdef __cplusplus
}
#endif
