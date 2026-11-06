# Sources and provenance

## Pinned experiment inputs

| Artifact | Identity |
|---|---|
| llama.cpp build | b12124 |
| llama.cpp revision | `8bed7cfb999cebc03d2cc5b78911026a8d57549a` |
| CPU ARM64 OCI manifest | `sha256:1440b21b6bd10dea81aafbe8bff3fca2d9b5cea7958bfab55e92e29c35d02d7f` |
| Vulkan ARM64 OCI manifest | `sha256:2c768718d72965c49e479355fadc5a0527a245ccce790c257418f40530df8ce8` |
| Example file | `LFM3-1.7B-Q4_K_M.gguf` |
| Example bytes | 1,099,627,808 |
| Example SHA-256 | `3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4` |

Machine-readable download identities are in [sources.json](../sources.json). The model URL follows upstream `main`, but the checksum pins the tested bytes; an upstream replacement causes verification to fail. The preparation script verifies the OCI manifest, image configuration, every layer, and the example model. It reconstructs root filesystems for the tested ARM64 server images without installing Docker on the Mac.

## Primary references

- [llama.cpp source at the tested revision](https://github.com/ggml-org/llama.cpp/tree/8bed7cfb999cebc03d2cc5b78911026a8d57549a)
- [llama.cpp Docker instructions at that revision](https://github.com/ggml-org/llama.cpp/blob/8bed7cfb999cebc03d2cc5b78911026a8d57549a/docs/docker.md)
- [Vulkan Dockerfile at that revision](https://github.com/ggml-org/llama.cpp/blob/8bed7cfb999cebc03d2cc5b78911026a8d57549a/.devops/vulkan.Dockerfile)
- [LiquidAI LFM3 GGUF model card](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF)
- [LiquidAI LFM3 ONNX model card](https://huggingface.co/LiquidAI/LFM3-1.7B-ONNX)
- [Houmo software stack overview](https://developer.houmoai.com/hmdoc/v3/software/software-manual/latest/getting-started/software_stack_architecture/index.html)
- [Houmo runtime SDK](https://developer.houmoai.com/hmdoc/v3/software/software-manual/latest/platform/runtime_user_guide/1_quickstart/runtime/linux.html)
- [Houmo compiler requirements](https://developer.houmoai.com/hmdoc/v3/software/software-manual/latest/platform/compilation_user_guide/quickstart/dependency.html)
- [Houmo LLM compilation](https://developer.houmoai.com/hmdoc/v3/software/software-manual/latest/platform/compilation_user_guide/1_user_guide/2_build/llm_build.html)

## Evidence and redistribution

Hardware identification, API behavior, throughput, buffer placement, OpenCL results, and the NPU memory test were observed locally. They are recorded in [the report](results.md) and the selected JSON measurement record. This repository omits machine-specific addresses, authentication material, full host inventories, and unrelated service configurations.

llama.cpp, its container dependencies, the model weights, and Tiiny/Houmo/Mali runtime libraries retain their upstream licenses. No model weights, container image layers, or vendor runtime libraries are committed here. The GPU profile uses the libraries already installed on the user's device.
