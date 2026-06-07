### About ggml-hexagon

ggml-hexagon backend is a specified backend for llama.cpp on Qualcomm Hexagon NPU.

details of ggml-hexagon can be found at: [about ggml-hexagon](https://github.com/zhouwg/ggml-hexagon/discussions/18)

ggml-hexagon backend consists of two parts:

 - codes on ARM AP side(libggml-hexagon.so), <b>fully</b> source code can be found at https://github.com/zhouwg/ggml-hexagon/blob/self-build/ggml/src/ggml-hexagon/ggml-hexagon.cpp

 - codes on cDSP side(libggmldsp-skel.so). <b>reference</b> source code can be found at https://github.com/zhouwg/ggml-hexagon/blob/self-build/ggml/src/ggml-hexagon/kernels, the prebuilt libggmldsp-skel.so can be found in this directory.

### Supported Qualcomm Snapdragon mobile SoC

```
#v68 --- Snapdragon 888
#v69 --- Snapdragon 8 Gen1
#v73 --- Snapdragon 8 Gen2
#v75 --- Snapdragon 8 Gen3
#v79 --- Snapdragon 8 Elite(aka Gen4)
```


### ChangeLog

- 20250531: ggml-hexagon.cpp v1.08 + ggml-dsp v0.96
- 20250609: ggml-hexagon.cpp v1.10 + ggml-dsp v0.97
- 20250625: ggml-hexagon.cpp v1.13 + ggml-dsp v0.98
- 20250627: ggml-hexagon.cpp v1.13 + ggml-dsp v0.98.8
- 20250710: ggml-hexagon.cpp v1.14 + ggml-dsp v0.98.9
