# NPU investigation checkpoint

**Custom NPU inference is unfinished.** This file records the evidence needed to resume after the CPU/GPU workflow. It does not provide a working LFM3 NPU recipe.

## What was established

- The discrete PCIe accelerator uses the `houmo,xh2a` driver and `xh2a_drv` module: Houmo XH2 / M50-family software. The exact retail board SKU was not established.
- The cached image `z_image_turbo_houmo:20260115` includes a working runtime under `/opt/tcim_runtime_lite`, with `include/tcim`, `lib/libtcim_runtime_lite.so`, `lib/libhal_xh2a.so`, and Python bindings.
- `/opt/venv/bin/python3` imported `tcim_lite.runtime` and `tcim_lite.pytcim` with `PYTHONPATH=/opt/tcim_runtime_lite/python` and `LD_LIBRARY_PATH=/opt/tcim_runtime_lite/lib`.
- `TCIM_FORCE_BACKEND=Xh2HalBackend` and `HOUMO_TARGET=xh2` selected the installed backend. Runtime version was v2.5.0, commit `9fe0c2e0dcd4f82fc8196ae5633a2e05a395fd50`, built 2026-01-02.
- One NPU was enumerated. A device tensor allocation and 16 KiB host→device→host copy passed with zero error.

The memory probe used explicit device passthrough for the fast-memory, IPU, DDR/SPM allocators, transfer, RPMSG, and system nodes. It did not use privileged mode, the QSPI flash node, or the I2C node. These eight nodes were sufficient for the tested transfer; this does not establish the minimum device set or all requirements for compiled-model execution.

## What blocks LFM3 today

The existing Qwen image `qwen3_8b_64k_houmo:20260119` contains an older Houmo-enabled llama.cpp build, 5926 (`9248ffdae`). Its model assets include `prefill.hmm`, `decode.hmm`, `bundle.bin`, a manifest, and a GGUF. Its launch configuration sets `HOUMO_LLM_PATH` and Houmo-specific cache options.

A separate parser test with LFM3 failed with:

```text
error loading model architecture: unknown model architecture: 'lfm3'
```

This is an engine architecture failure before inference, not proof that the NPU hardware cannot execute LFM3 operations. Replacing a GGUF filename or setting `--n-gpu-layers` does not generate Houmo executable graphs.

The inspected runtime image's `tcim` import resolved to an empty namespace; `tcim.builder` was absent. Houmo documents the runtime SDK as an inference/deployment package, distinct from its compiler. [Runtime SDK documentation](https://developer.houmoai.com/hmdoc/v3/software/software-manual/latest/platform/runtime_user_guide/1_quickstart/runtime/linux.html)

## Next work

1. Obtain the Houmo compiler, quantizer, examples, and compatibility information for the installed Tiiny runtime/driver/firmware. Current compiler documentation requires Linux x86_64 for model compilation. [Compiler requirements](https://developer.houmoai.com/hmdoc/v3/software/software-manual/latest/platform/compilation_user_guide/quickstart/dependency.html)
2. First validate a tiny compiled model end to end before attempting the whole LFM3 network.
3. Adapt LFM3's supported operator graph, quantize/export HMONNX, and compile prefill/decode artifacts. Houmo documents `.hmm`/`.hmms` outputs and separate LLM prefill/decode compilation. [LLM compilation](https://developer.houmoai.com/hmdoc/v3/software/software-manual/latest/platform/compilation_user_guide/1_user_guide/2_build/llm_build.html)
4. Implement an inference loop using the runtime, or port the Houmo integration into a newer engine that understands LFM3. Validate numerical outputs, tokenization, cache updates, and CPU fallbacks before measuring speed.

Liquid provides an [LFM3 ONNX export](https://huggingface.co/LiquidAI/LFM3-1.7B-ONNX). It carries both convolution and attention caches and uses ONNX Runtime attention operators. Whether Houmo can compile those graphs directly is unverified; operator lowering and shape/cache adaptations may be required.

Current public Houmo documentation uses platform version 1.5.0, while the installed runtime reports v2.5.0. Those version numbers do not establish binary compatibility. No compiler, firmware, or driver upgrade was performed during this investigation.

The CIX SoC's integrated NPU is distinct from the discrete Houmo accelerator. No usable device interface/loaded driver for that integrated NPU was found in the inspected setup; it was not enabled or tested.
