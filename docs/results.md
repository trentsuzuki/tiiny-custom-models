# Investigation results — 2026-11-05

These findings come from one connected Tiiny z01. Local measurements take precedence over advertised capabilities. The selected measurement record is [results/2026-11-05.json](../results/2026-11-05.json).

## Connection and access

TiinyOS could not discover this older device in the initial Mac workflow. An isolated `tiiny-sdk==0.1.4` environment did connect, authenticate, identify firmware/service version **0.1.29**, and run the supported Qwen3-8B model. Native TiinyOS compatibility and firmware updates remain unresolved.

SSH login was not obtained. The usable route for custom workloads was the device's Docker Compose Manager on its USB network, port 8880. It could create and launch separate ARM64 containers. A Mac-hosted HTTP server supplied the image build context and model because the Tiiny had no working internet route in the tested setup.

The standard model catalog did not supply LFM3: attempted import endpoints returned 405, the catalog download returned application code 170004 (`No models can be downloaded.`), and an inference request for the unloaded model returned 503. These observations describe this firmware's tested endpoints, not a universal statement about all Tiiny versions. Custom containers provided a separate inference API.

## Hardware and software observed

| Item | Observation |
|---|---|
| Device | Tiiny z01 |
| Firmware/service | 0.1.29 |
| OS | Linux 6.6.89-generic, aarch64 |
| GPU | Mali-G720-Immortalis; `/dev/mali0`; `mali_kbase` r53p0 |
| GPU APIs | Vulkan worked; OpenCL 3.0 reported and custom kernel passed |
| GPU memory report | 31,797 MiB in the SoC shared-memory domain |
| Discrete NPU | PCI vendor 0x1f6b, device 0x0c00; `houmo,xh2a` driver; `xh2a_drv` module |
| NPU runtime | TCIM v2.5.0, built 2026-01-02; `Xh2HalBackend` enumerated one device |

The GPU's shared RAM and discrete NPU memory are separate execution domains. This work did not demonstrate pooling them for a single arbitrary model.

## Custom LFM3 CPU inference

The official ARM64 llama.cpp CPU server b12124 loaded `LFM3-1.7B-Q4_K_M.gguf` successfully. `/v1/models` reported 1,728,801,600 parameters and Q4_K - Medium. A math prompt returned `42`; a short explanation prompt completed after disabling reasoning and increasing the response budget to 256 tokens.

An earlier explanation request returned empty visible content after exhausting 128 generated tokens in reasoning. This was a request/template issue, not evidence that the device could not execute the model.

CPU configuration: 4 threads, 2,048-token context, one request at a time, no GPU layers, 6 GiB container cap, and RAM-backed model delivery. The cap is not a measured RAM requirement.

## Custom LFM3 Vulkan inference

The official ARM64 Vulkan engine at the same revision identified the Mali GPU and reported:

```text
offloaded 25/25 layers to GPU
Vulkan0 model buffer size = 1040.82 MiB
Vulkan_Host model buffer size = 205.08 MiB
```

GPU KV buffers were 24.00 MiB and 7.50 MiB; recurrent state was 0.61 MiB. These are logged buffer sizes, not a full process-memory or device-utilization measurement. The GPU container cap was 8 GiB.

The GPU answered the math prompt correctly and generated a complete explanation about leaves. All layers were offloaded, while host-side work and buffers remained part of inference.

## Short CPU/GPU comparison

Prompt: `Write three short sentences explaining why leaves appear green.`

Both requests used the same model/build, 4 CPU threads, context 2048, seed 42, temperature 0.1, top-k 50, repeat penalty 1.1, `enable_thinking: false`, `max_tokens: 256`, and `cache_prompt: false`.

| Measurement | CPU | Vulkan GPU, warm |
|---|---:|---:|
| Prompt tokens processed | 21 | 21 |
| Generated tokens | 95 | 100 |
| Prompt processing | 62.56 tokens/s | 8.97 tokens/s |
| Token generation | 27.86 tokens/s | 25.60 tokens/s |
| Wall time | 3.715 s | 6.229 s |

The GPU did not improve this workload. This was one short request per backend, not a statistically robust benchmark. Generated text differed slightly despite the same seed. No power, thermal, sustained-load, large-batch, long-context, or partial-offload study was performed. The rates do not establish how a different model will behave.

## Custom GPU computation

A small OpenCL C vector-add kernel was compiled and launched on `Mali-G720-Immortalis r0p0`. For 4,096 float32 elements, every output matched the expected result, with maximum absolute error **0.0**. This demonstrates custom GPU kernel execution beyond invoking the LLM server. Its sub-millisecond enqueue/read observation is not a meaningful GPU throughput benchmark.

## NPU checkpoint

The installed runtime's Python bindings imported successfully, enumerated one Houmo device, allocated a 4,096-element float32 device tensor, and copied 16,384 bytes from host to device and back with exact equality. This tests allocation and data transfer only.

A later experiment executed an older, precompiled Houmo convolution model through `Xh2HalBackend` in a separate container. Two single-round runs, using zero and constant inputs, took 8.408 and 8.327 ms and returned finite, different output arrays. The current public version of that control model failed the runtime's binary-format check; the historical version compiled by TCIM v2.4.0 loaded and ran on the installed v2.5.0 runtime. No independent numerical-reference comparison was performed.

A separate copy of Tiiny's bundled Houmo-enabled llama.cpp engine read LFM3 GGUF metadata, then rejected `lfm3` as an unknown architecture. That compatibility probe had no NPU device access; it failed before inference. No LFM3 NPU execution was achieved. The runtime image contained no usable `tcim.builder` compiler. The official compiler download requires a developer login; no package was obtained. Structural inspection of LFM3's FP16 ONNX graph found custom attention/rotary operators and 35 cache inputs that need conversion validation. See [the NPU investigation](npu.md) and [its measurement record](../results/npu-compiler-investigation.json).

## State left after the experiments

Temporary CPU/GPU model servers and the Mac asset server were stopped, with refused connections verified on their ports. One-shot inspection containers were removed. Existing firmware/driver installations were unchanged. The original Qwen3-8B service remained loaded and answered `3 + 4` with `7` after the accelerator tests.

Cached images, stopped experiment definitions, and local assets were retained for reuse. The repository workflow uses its own `tiiny-custom-cpu` and `tiiny-custom-gpu` names rather than the original experiment names.

## Limits still open

- Compatibility with other Tiiny firmware releases and models.
- Larger/split models, long context, multimodal inference, batching, and sustained throughput.
- GPU tuning or other inference engines beyond the tested llama.cpp/Vulkan and OpenCL kernel.
- Persistent model storage and integration into TiinyOS's native model catalog.
- New NPU model compilation, LFM3 operator/cache integration, numerical validation, and LFM3 NPU inference.
