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

## NPU compiler investigation

- [Official compiler setup](https://developer.houmoai.com/hmdoc/v3/software/software-manual/latest/platform/compilation_user_guide/quickstart/setup.html) and [download portal](https://developer.houmoai.com/resources_v2).
- [Model zoo, pinned current revision](https://github.com/houmo-ai/postmo-modelzoo/tree/9121ddac32d0d1a7317103a592b22f955d656050); [historical control model that ran](https://github.com/houmo-ai/postmo-modelzoo/blob/8b1d5572081d1906322970acbdbfccd5c0d70f80/tools/hm_check/models/xh2_compute.hmm).
- [Quantization sources](https://github.com/houmo-ai/postmo-quantflow/tree/21e88eac9605d1de38f11cd3e0c10381fa4ee91a).
- [Houmo llama.cpp source](https://github.com/houmo-ai/llama.cpp/tree/df852f59f72293441b7f8ed5d464cd6cc9b07656) and [vLLM plugin](https://github.com/houmo-ai/vllm-postmo/tree/de45f66ac99118c395d3266bbf10a46dde4372fd).
- [Inspected LFM3 ONNX revision](https://huggingface.co/LiquidAI/LFM3-1.7B-ONNX/tree/57c125dfa29f2d39157e280b1d1f7480b8736d8d).

| NPU investigation artifact | Bytes | SHA-256 |
|---|---:|---|
| Current `xh2_compute.hmm`, rejected by installed runtime | 1,065,032 | `84d6632c7efd9c8c2e165e3f1f7a1af2f4187b570bfc4280c70088ae788e5c85` |
| Historical `xh2_compute.hmm`, loaded and executed | 978,240 | `a393993d900b9f73c78876a0b57394d44530171a4008c9440636baf3834f29ce` |
| LFM3 `model_fp16.onnx`, graph only | 270,733 | `d3093d69f6f954bd74f1dc9e59452ec5e9766e62d9c409456fb8ea5698a830fe` |

Git blob SHA-1 values from the official repository tree were also checked for both control binaries. These binaries and LFM3 external weights are not included in this repository. The [NPU record](../results/npu-compiler-investigation.json) separates inference evidence from structural graph inspection and compiler access findings.

## Evidence and redistribution

Hardware identification, API behavior, throughput, buffer placement, OpenCL results, NPU memory transfers, and the historical precompiled NPU model execution were observed locally. They are recorded in [the report](results.md) and the selected JSON measurement records. This repository omits machine-specific addresses, authentication material, full host inventories, and unrelated service configurations.

llama.cpp, its container dependencies, the model weights, and Tiiny/Houmo/Mali runtime libraries retain their upstream licenses. No model weights, container image layers, or vendor runtime libraries are committed here. The GPU profile uses the libraries already installed on the user's device.
