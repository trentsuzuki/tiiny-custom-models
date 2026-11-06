# How Tiiny's supported models are compiled and packaged

**The installed Qwen3-8B package contains precompiled Houmo/XH2 graphs, a shared weight bundle, and a small GGUF containing the token embedding.** This is direct evidence from this Tiiny's installed files. We did not recover Tiiny's complete internal build recipe, and we have not established that every supported model uses the same process.

## What was inspected on the device

The Qwen model directory was mounted read-only in a temporary container. Only file metadata, the weight manifest, and GGUF metadata/tensor descriptors were inspected. No model weights or proprietary model binaries are redistributed here. The selected evidence is in [native-model-package.json](../results/native-model-package.json).

| Installed file | Size | Observed role |
|---|---:|---|
| `modules/prefill.hmm` | 138,271,416 bytes | Compiled prompt-processing graph. |
| `modules/decode.hmm` | 121,994,800 bytes | Compiled autoregressive generation graph. |
| `weights/bundle.bin` | 4,180,459,072 bytes | Shared data referenced by the compiled modules. |
| `weights/manifest.json` | 559,706 bytes | 1,967 entries mapping hashes to bundle offsets/sizes; 1,820 list both source graphs. |
| `qwen3_8b.q8_0.gguf` | 667,158,464 bytes | Tokenizer/model metadata and exactly one tensor: `token_embd.weight`, shape `[4096, 151936]`. |

The manifest records these original build-output paths:

```text
../hmcc/work_dirs/qwen3_8b_64k_prefill.xh2/output.hmm
../hmcc/work_dirs/qwen3_8b_64k_decode.xh2/output.hmm
```

This is strong evidence that the graphs were compiled for XH2 before packaging, with common data extracted into the shared bundle. The `hmcc` path identifies a build directory; it does not establish an exact compiler version or command line.

The GGUF's filename and Q8_0 embedding do not establish the quantization of the rest of the model. Its metadata describes 36 transformer blocks, but its tensor directory contains only the embedding. The `622M` size label in that GGUF therefore must not be mistaken for the full model's parameter count. Likewise, the GGUF context metadata is 40,960 while the package/build paths and observed launcher request 65,536; this is another reason not to treat the GGUF alone as the complete deployment specification.

The earlier runtime inspection found the image `qwen3_8b_64k_houmo:20260119`, a modified `llama-server` reporting version `5926 (9248ffdae)`, and the libraries `libtcim_runtime_lite.so` and `libhal_xh2a.so`. Its entrypoint sets `HOUMO_LLM_PATH=../model`, `KV_CACHE_ALIGN=64`, and `NO_CPU_KV_CACHE=1`, then launches the embedding GGUF with a 64K context. This runtime evidence complements the current package inspection.

## Houmo's documented compilation pipeline

Houmo documents the general process as:

```text
Original model / architecture-specific export
    → calibration and quantization
    → quantized ONNX (HMONNX)
    → PostMo-Compiler / TCIM targeting XH2
    → executable .hmm / .hmms modules
    → matching PostMo/TCIM runtime + model-specific host application
```

The quantizer exports hardware-oriented low-precision graphs; the compiler performs graph/operator optimization and code generation; the runtime loads and schedules the resulting binaries on the NPU. [Houmo software-stack overview](https://developer.houmoai.com/hmdoc/v3/software/software-manual/latest/getting-started/quickstart_guide/intro/index.html).

For LLMs, the documented Python API `tcim.builder.api.build_from_hmonnx` compiles separate prefill and decode graphs. Context length, batch dimensions, chip/core allocation, and runtime compatibility are part of the deployment configuration. The documented compiler host is Linux x86_64. [LLM compiler guide](https://developer.houmoai.com/hmdoc/v3/software/software-manual/latest/platform/compilation_user_guide/1_user_guide/2_build/llm_build.html), [model-zoo source](https://github.com/houmo-ai/postmo-modelzoo/tree/9121ddac32d0d1a7317103a592b22f955d656050).

Houmo's public llama.cpp integration creates an `Xh2HalBackend` device manager and weight manager, then loads compiled prefill/decode modules through TCIM. Its newer HiModel packaging can store those compiled modules inside GGUF. That source explains a related public runtime route; it is not proof that Tiiny's older binary is an identical build. [Pinned Houmo loader source](https://github.com/houmo-ai/llama.cpp/blob/df852f59f72293441b7f8ed5d464cd6cc9b07656/src/houmo/houmo-llmodel.cpp).

Applying that documented pipeline to Tiiny's observed artifacts is an inference about the upstream build process. The device files directly establish the compiled modules and packaging, but not the original quantizer settings, calibration data, compiler revision, or every model-specific optimization. No evidence from this inspection establishes that this particular Qwen package uses TurboSparse or PowerInfer-specific transformations.

## Implication for custom models

The supported-model path distributes architecture-specific, prebuilt packages with a matching host engine/runtime. An ordinary GGUF file supplies neither the XH2 graphs nor the associated state/cache handling. Adding a model to a catalog entry alone is insufficient to reproduce this NPU path.

Reproducing it requires the matching compiler/quantizer and architecture integration, followed by numerical validation and runtime-version testing. Public precompiled Houmo models may bypass compilation, but still need a compatible runtime and host integration. See [the NPU investigation](npu.md) for the demonstrated old/new binary compatibility difference.
