# Arhat LLM

Arhat LLM is a library of deep neural network primitives designed for implementation of LLM inference.
Arhat LLM can be used either in combination with [llama.cpp](https://github.com/ggml-org/llama.cpp) 
framework or as a standalone library.
Integration with llama.cpp is facilitated via a specialized Arhat backend for GGML.

Arhat LLM targets Intel Arc GPU hardware and is built on top of 
[oneAPI Deep Neural Network Library](https://github.com/uxlfoundation/oneDNN) (oneDNN).

Arhat LLM is implemented in C++ with computational kernels implemented in OpenCL.


## Requirements

Arhat LLM runs on Windows 11. Microsoft Visual C++ 2026 is required to build the software
from the source code.

Runtime requirements include: 

* [oneDNN](https://github.com/uxlfoundation/oneDNN) library v3.10.0 or higher 
with the GPU Engine enabled
* Khronos [OpenCL SDK](https://github.com/KhronosGroup/OpenCL-SDK)
* Intel Graphics Driver with support for OpenCL 3.0

Arhat LLM has been validated on integrated Intel Arc GPUs of Core Ultra 2 Series processors
(former Arrow Lake and Lunar Lake).


## Installing runtime dependencies

Get the recent release of **Khronos OpenCL SDK** from this
[GitHub repository](https://github.com/KhronosGroup/OpenCL-SDK).
You can build it from source or get pre-built binaries for Windows x64 from the 
["Releases"](https://github.com/KhronosGroup/OpenCL-SDK/releases) page.

Place the SDK components in the subdirectory `vendor/opencl`.
The subdirectory tree should have this structure:

```
vendor
    opencl
        bin
        include
        lib
        share
```

Download and build the **oneDNN library** as specified by the 
["Build from Source"](https://uxlfoundation.github.io/oneDNN/dev_guide_build.html) 
chapter of the oneDNN developer guide. 
Refer to the section "Build on Windows" for OS-specific details.

Set the CMake build options as described by the 
["Use Build Options"](https://uxlfoundation.github.io/oneDNN/dev_guide_build_options.html)
chapter of the oneDNN developer guide.

NOTE: Some build options may differ for various oneDNN versions. Make sure that you use
the developer guide corresponding to your version. (You can choose the version
number from the drop-down list located on the top of any developer guide page.)

Make sure that you specify the following build options:

* `ONEDNN_CPU_RUNTIME=OMP`
* `ONEDNN_GPU_RUNTIME=OCL`

Also, specify the path to the OpenCL SDK as described in the "GPU Options" section of the
build options guide corresponding to your oneDNN version.

Prior to v3.13 the path to the OpenCL SDK shall be specified using the `-DOPENCLROOT` 
CMake option in form of

```
-DOPENCLROOT=/path/to/opencl/sdk
```

Starting from v3.13, the location of OpenCL headers shall be specified using the option
`ONEDNN_OCL_INCLUDE_DIR`.

Place the oneDNN library components in the subdirectory `vendor/onednn`.
The subdirectory tree should have this structure:

```
vendor
    onednn
        bin
        include
        lib
        share
```


## Building from source

To build Arhat LLM, open the Visual Studio solution file `wks.sln` located in the subdirectory
`wks` using Visual C++ 2026 and build the project `build_all`. This will build all available
libraries and command line applications and place them in the subdirectories `lib` and `bin`
respectively.

The following command line applications will be built:

* `llama_cli.exe` - a CLI tool for running LLMs and experimenting with llama.cpp functionality
* `llama_bench.exe` - a tool for benchmarking the LLM inference performance
* `test_backend_ops.exe` - a tool for testing various llama.cpp backend operations
* `test_llama_arch.exe` - a tool for testing support of various LLM architectures

NOTE: You may need to adjust the compile time configuration for your GPU architecture
as described below in the section "Architecture-specific configuration".


## Getting started with `llama_cli`

To evaluate `llama_cli`, obtain a llama.cpp compatible model in GGUF format from 
[Hugging Face](https://huggingface.co/) platform. The following example will assume that
the downloaded models are placed in the `models` subdirectory.

Model collections at Hugging Face platform provided by 
[ggml-org](https://huggingface.co/ggml-org/collections) and 
[Unsloth AI](https://huggingface.co/unsloth/collections) can serve as a good source
for LLMs of various architectures.

For this example, we will use a GGUF representation of a relatively small Gemma 3 model 
with 1B parameters and Q4_K_M quantization 
[gemma-3-1b-it-Q4_K_M.gguf](https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/blob/main/gemma-3-1b-it-Q4_K_M.gguf).

Upon downloading the model, open the Windows command processor and start the `llama_cli` tool 
using this command line:

```
bin\release\llama_cli.exe -m .\models\gemma-3-1b-it-Q4_K_M.gguf
```

The tool will download the model in memory and run it in conversation mode.

Further examples of supported models are available in the Appendix B.


## Origin of OpenCL kernels

We use OpenCL 3.0 to implement GGML operations that are not natively supported by oneDNN.
The respective OpenCL kernels have been ported from various GGML backends like OpenCL,
CUDA, and Vulkan. For several computationally complex operations like matrix multiplication
or flash attention we have ported multiple kernels from different backends and compared
performance of the resulting code. In the most critical cases we implemented further
performance tuning specific for Intel GPU target platform. (Kernel tuning is an ongoing
effort as there is certainly more room for further optimization.)


## Architecture-specific configuration

Computational kernels and their launch parameters may depend on details of the target
GPU architecture. Currently, Arhat does not automatically detect the architecture.
Instead, the architecture shall be specified at compile time using the constant
`CONFIG_GPU_ARCH` specified in the header file 
[onednn/ocl/config_arch.hpp](./arhat/src/arhat/onednn/ocl/config_arch.hpp) as follows:

```
constexpr int GPU_ARCH_UHD = 0;
constexpr int GPU_ARCH_TGL = 1;
constexpr int GPU_ARCH_ARL = 2;
constexpr int GPU_ARCH_LNL = 3;

constexpr int CONFIG_GPU_ARCH = GPU_ARCH_LNL;
```

The options `ARL` and `LNL` correspond to the Arc iGPUs of the Arrow Lake and
Lunar Lake processor families. The Arrow Lake family includes Intel Core Ultra 255H and 285H
processors with Intel Arc 140T iGPU (Xe+ architecture). The Lunar Lake family includes
Intel Core Ultra 258V and 288V processors with Intel Arc 140V iGPU (Xe2 architecture).

The options `UHD` and `TGL` correspond respectively to Intel UHD graphics and Iris Xe iGPU 
of Tiger Lake processor family. They are considered legacy and retained mainly for backwards
compatibility. (But nevertheless, both Arhat and oneDNN still can run even on an older processor
with Intel UHD 630 graphics.)

We did not yet validate Arhat on discrete Intel GPUs. However, we believe that support for these
GPUs can be easily implemented by a modest update of the code. For the Arc Battlemage
architecture, the Lunar Lake option should be a good starting point. The code in these source
files may require moderate update:

* [onednn/gpu/context.cpp](./arhat/src/arhat/onednn/gpu/context.cpp) - function `Context::InitDeviceInfo`
* [onednn/ocl/device_info.cpp](./arhat/src/arhat/onednn/ocl/device_info.cpp)

Currently, the llama.cpp framework needs to know the size of available GPU memory to perform
tensor reservation. The Arhat backend for GGML does not automatically query the GPU memory size.
This size shall be configured at the compilation time within the function
`ggml_backend_arhat_device_get_memory` at the source file
[ggml-arhat/ggml-arhat.cpp](./ggml/src/ggml-arhat/ggml-arhat.cpp) as follows:

```
static void ggml_backend_arhat_device_get_memory(ggml_backend_dev_t dev, size_t *free, size_t *total) {
    GGML_UNUSED(dev);
    // TODO: Replace this stub with actual device query
    constexpr size_t KB = size_t(1024);
    size_t dummy = size_t(24) * KB * KB * KB;
    *free = dummy;
    *total = dummy;
}
```

The default size of 24 GB corresponds to our reference system of Arrow Lake or Lunar Lake
family with 32 GB of unified memory running Windows 11. (Note that Intel graphics driver
on Windows allows configuring the size of unified memory available for use by the GPU.)


## Contributing

We welcome your feedback in the form of questions, bug reports, and feature requests.
To submit the feedback, please use the Github issues page.

We do not accept pull requests. This public source is exported from our internal development
system which does not support the workflow for external pull requests.


## Appendix A. Supported GGML operations

Arhat LLM supports the following GGML operations:

* `ABS`
* `ADD`
* `ADD_ID`
* `ARGSORT`
* `CAST`
* `CEIL`
* `CLAMP`
* `CONCAT`
* `CONT`
* `CONV_2D`
* `CONV_2D_DW`
* `CONV_3D`
* `CPY`
* `CUM_SUM`
* `DIAG`
* `DIV`
* `DUP`
* `ELU`
* `EXP`
* `EXPM1`
* `FILL`
* `FLASH_ATTN_EXT`
* `FLOOR`
* `GATED_DELTA_NET`
* `GEGLU`
* `GEGLU_ERF`
* `GEGLU_QUICK`
* `GELU`
* `GELU_ERF`
* `GELU_QUICK`
* `GET_ROWS`
* `GROUP_NORM`
* `HARDSIGMOID`
* `HARDSWISH`
* `L2_NORM`
* `LEAKY_RELU`
* `LOG`
* `MEAN`
* `MUL`
* `MUL_MAT`
* `MUL_MAT_ID`
* `NEG`
* `NORM`
* `OUT_PROD`
* `PAD`
* `PERMUTE`
* `POOL_2D`
* `REGLU`
* `RELU`
* `REPEAT`
* `RESHAPE`
* `ROPE`
* `RMS_NORM`
* `ROUND`
* `SCALE`
* `SET`
* `SET_ROWS`
* `SGN`
* `SIGMOID`
* `SILU`
* `SOFT_MAX`
* `SOFTPLUS`
* `SOLVE_TRI`
* `SQR`
* `SQRT`
* `SSM_CONV`
* `STEP`
* `SUB`
* `SUM`
* `SUM_ROWS`
* `SWIGLU`
* `SWIGLU_OAI`
* `TANH`
* `TRANSPOSE`
* `TRI`
* `TRUNC`
* `UPSCALE`
* `VIEW`
* `XIELU`

Various operations may have individual restrictions on supported data types and layouts
of input tensors.

The operations `MUL_MAT` and `MUL_MAT_ID` support the following GGML quantization types:

* `Q4_0`
* `Q4_1`
* `Q5_0`
* `Q5_1`
* `Q8_0`
* `MXFP4`
* `Q2_K`
* `Q3_K`
* `Q4_K`
* `Q5_K`
* `Q6_K`


## Appendix B. Supported models

Following is a non-exhaustive list of models which we used to validate Arhat LLM.
We welcome user feedback on evaluation of further models. 

* Apertus
    * [Apertus-8B-Instruct-2509-Q4_K_M](https://huggingface.co/unsloth/Apertus-8B-Instruct-2509-GGUF/blob/main/Apertus-8B-Instruct-2509-Q4_K_M.gguf)
    * [Apertus-8B-Instruct-2509-Q6_K](https://huggingface.co/unsloth/Apertus-8B-Instruct-2509-GGUF/blob/main/Apertus-8B-Instruct-2509-Q6_K.gguf)

* DeepSeek-R1
    * [DeepSeek-R1-0528-Qwen3-8B-Q4_K_M](https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF/blob/main/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf)

* Falcon 3
    * [Falcon3-7B-Instruct-Q6_K](https://huggingface.co/bartowski/Falcon3-7B-Instruct-GGUF/blob/main/Falcon3-7B-Instruct-Q6_K.gguf)

* Gemma 3
    * [gemma-3-1b-it-Q4_K_M](https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/blob/main/gemma-3-1b-it-Q4_K_M.gguf)
    * [gemma-3-1b-it-Q8_0](https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/blob/main/gemma-3-1b-it-Q8_0.gguf)
    * [gemma-3-4b-it-Q4_K_M](https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/blob/main/gemma-3-4b-it-Q4_K_M.gguf)
    * [gemma-3-4b-it-Q8_0](https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/blob/main/gemma-3-4b-it-Q8_0.gguf)
    * [gemma-3-12b-it-Q4_K_M](https://huggingface.co/ggml-org/gemma-3-12b-it-GGUF/blob/main/gemma-3-12b-it-Q4_K_M.gguf)
    * [gemma-3-12b-it-Q8_0](https://huggingface.co/ggml-org/gemma-3-12b-it-GGUF/blob/main/gemma-3-12b-it-Q8_0.gguf)

* Gemma 4
    * [gemma-4-E4B-it-UD-Q4_K_XL](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/blob/main/gemma-4-E4B-it-UD-Q4_K_XL.gguf)
    * [gemma-4-12b-it-UD-Q4_K_XL](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/blob/main/gemma-4-12b-it-UD-Q4_K_XL.gguf)
    * [gemma-4-26B-A4B-it-Q4_0](https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF/blob/main/gemma-4-26B-A4B-it-Q4_0.gguf)

* GPT OSS
    * [gpt-oss-20b-UD-Q4_K_XL](https://huggingface.co/unsloth/gpt-oss-20b-GGUF/blob/main/gpt-oss-20b-UD-Q4_K_XL.gguf)
    * [gpt-oss-20b-MXFP4](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/blob/main/gpt-oss-20b-MXFP4.gguf)

* Granite 4.0
    * [granite-4.0-1b-Q4_K_M](https://huggingface.co/unsloth/granite-4.0-1b-GGUF/blob/main/granite-4.0-1b-Q4_K_M.gguf)
    * [granite-4.0-1b-Q6_K](https://huggingface.co/unsloth/granite-4.0-1b-GGUF/blob/main/granite-4.0-1b-Q6_K.gguf)
    * [granite-4.0-1b-Q8_0](https://huggingface.co/unsloth/granite-4.0-1b-GGUF/blob/main/granite-4.0-1b-Q8_0.gguf)

* Kimi-VL
    * [Kimi-VL-A3B-Thinking-2506-Q4_K_M](https://huggingface.co/ggml-org/Kimi-VL-A3B-Thinking-2506-GGUF/blob/main/Kimi-VL-A3B-Thinking-2506-Q4_K_M.gguf)

* Llama 3.1
    * [Llama-3.1-8B-Instruct-Q4_0](https://huggingface.co/unsloth/Llama-3.1-8B-Instruct-GGUF/blob/main/Llama-3.1-8B-Instruct-Q4_0.gguf)
    * [Llama-3.1-8B-Instruct-Q4_K_M](https://huggingface.co/unsloth/Llama-3.1-8B-Instruct-GGUF/blob/main/Llama-3.1-8B-Instruct-Q4_K_M.gguf)
    * [Llama-3.1-8B-Instruct-Q6_K](https://huggingface.co/unsloth/Llama-3.1-8B-Instruct-GGUF/blob/main/Llama-3.1-8B-Instruct-Q6_K.gguf)

* Ministral 3
    * [Ministral-3-3B-Reasoning-2512-Q8_0](https://huggingface.co/ggml-org/Ministral-3-3B-Reasoning-2512-GGUF/blob/main/Ministral-3-3B-Reasoning-2512-Q8_0.gguf)
    * [Ministral-3-8B-Reasoning-2512-Q8_0](https://huggingface.co/ggml-org/Ministral-3-8B-Reasoning-2512-GGUF/blob/main/Ministral-3-8B-Reasoning-2512-Q8_0.gguf)
    * [Ministral-3-14B-Reasoning-2512-Q8_0](https://huggingface.co/ggml-org/Ministral-3-14B-Reasoning-2512-GGUF/blob/main/Ministral-3-14B-Reasoning-2512-Q8_0.gguf)

* Mistral Small 3.2
    * [Mistral-Small-3.2-24B-Instruct-2506-Q2_K_L](https://huggingface.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF/blob/main/Mistral-Small-3.2-24B-Instruct-2506-Q2_K_L.gguf)
    * [Mistral-Small-3.2-24B-Instruct-2506-Q3_K_M](https://huggingface.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF/blob/main/Mistral-Small-3.2-24B-Instruct-2506-Q3_K_M.gguf)
    * [Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M](https://huggingface.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF/blob/main/Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf)

* Phi3
    * [Phi-3-mini-4k-instruct-q4](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/blob/main/Phi-3-mini-4k-instruct-q4.gguf)

* Phi4
    * [phi-4-Q4_K](https://huggingface.co/microsoft/phi-4-gguf/blob/main/phi-4-Q4_K.gguf)

* Qwen3
    * [Qwen3-8B-Q4_K_M](https://huggingface.co/ggml-org/Qwen3-8B-GGUF/blob/main/Qwen3-8B-Q4_K_M.gguf)
    * [Qwen3-8B-Q8_0](https://huggingface.co/ggml-org/Qwen3-8B-GGUF/blob/main/Qwen3-8B-Q8_0.gguf)

* Qwen3.5
    * [Qwen3.5-4B-Q4_K_M](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/blob/main/Qwen3.5-4B-Q4_K_M.gguf)
    * [Qwen3.5-9B-Q4_K_M](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/blob/main/Qwen3.5-9B-Q4_K_M.gguf)
    * [Qwen3.5-9B-Q6_K](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/blob/main/Qwen3.5-9B-Q6_K.gguf)
    * [Qwen3.5-9B-Q8_0](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/blob/main/Qwen3.5-9B-Q8_0.gguf)

* Qwen3.6
    * [Qwen3.6-35B-A3B-UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/blob/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf)

* TinyLlama
    * [TinyLlama-1.1B-Chat-v1.0-Q4_K_M](https://huggingface.co/andrijdavid/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf)
    * [TinyLlama-1.1B-Chat-v1.0-Q6_K](https://huggingface.co/andrijdavid/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/TinyLlama-1.1B-Chat-v1.0-Q6_K.gguf)
    * [TinyLlama-1.1B-Chat-v1.0-Q8_0](https://huggingface.co/andrijdavid/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/TinyLlama-1.1B-Chat-v1.0-Q8_0.gguf)


