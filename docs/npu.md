# Houmo NPU investigation — 2026-11-05

**A precompiled convolution model ran on the Tiiny's NPU from our own container. LFM3-1.7B did not run on the NPU, and no new model was compiled.** The compiler was identified, but its official download requires a Houmo login. The installed runtime also rejects the current public test binary's format. An older public binary worked without changing the firmware or drivers.

Measurements, source revisions, hashes, and the LFM3 graph inventory are in [the investigation record](../results/npu-compiler-investigation.json). The exact successful in-container test is [experiments/npu_compute.py](../experiments/npu_compute.py).

## Hardware and installed runtime

- Discrete PCIe accelerator: vendor `0x1f6b`, device `0x0c00`, driver `houmo,xh2a`, module `xh2a_drv`. This identifies the Houmo XH2 / M50 software family; the exact retail board SKU is unconfirmed.
- Cached image: `z_image_turbo_houmo:20260115`, ARM64 Linux.
- Runtime: `/opt/tcim_runtime_lite`; Python: `/opt/venv/bin/python3`.
- TCIM version: `v2.5.0`, commit `9fe0c2e0dcd4f82fc8196ae5633a2e05a395fd50`, built 2026-01-02.
- `Xh2HalBackend` enumerated one device. The earlier 16 KiB host→device→host copy passed exactly; that was a memory test, separate from the computation below.

The CIX SoC's integrated NPU is distinct from this discrete Houmo device. No usable interface/loaded driver for the integrated NPU was found in the inspected setup; it was not enabled or tested.

## Compiler found, download blocked

Houmo's compiler is **PostMo-Compiler / TCIM**, distributed in the Dadao conversion toolchain. The current official documentation names:

```text
Dadao-docker-xh2-1.5.0-ubuntu24.04-x86.64.tar
harbor.houmo.ai/toolchain/release:Dadao-docker-xh2-1.5.0-ubuntu24.04-x86.64
```

The conversion image includes quantization, compilation, and inference. Deployment/runtime packages provide inference only. Model compilation requires Linux x86_64; the LLM pipeline compiles separate prefill/decode HMONNX graphs through `tcim.builder.api.build_from_hmonnx`, producing `.hmm` or `.hmms` files. [Compiler setup](https://developer.houmoai.com/hmdoc/v3/software/software-manual/latest/platform/compilation_user_guide/quickstart/setup.html), [requirements](https://developer.houmoai.com/hmdoc/v3/software/software-manual/latest/platform/compilation_user_guide/quickstart/dependency.html), [LLM compilation](https://developer.houmoai.com/hmdoc/v3/software/software-manual/latest/platform/compilation_user_guide/1_user_guide/2_build/llm_build.html).

On the [official download portal](https://developer.houmoai.com/resources_v2), selecting an M50-family board and a software release displayed a login-required message. No developer account was available. The named registry hostname did not resolve from the tested Mac/network. The [Firefly distribution route](https://community.t-firefly.com/en/docs/products/accessories/HOUMO-LQ50/started) also directs users to request a resource link from sales. No compiler package was obtained or installed.

The public [model zoo](https://github.com/houmo-ai/postmo-modelzoo/tree/9121ddac32d0d1a7317103a592b22f955d656050) and [quantization sources](https://github.com/houmo-ai/postmo-quantflow/tree/21e88eac9605d1de38f11cd3e0c10381fa4ee91a) are useful conversion code, but do not supply the missing compiler binary. Their published LLM quantization workflow requires a GPU. The model-zoo README specifies Linux x64 for quantization/compilation. The current LLM compiler guide requests at least 64 GB available RAM for its Qwen example; this is not a measured LFM3 memory requirement.

The Tiiny runtime image's `tcim` import resolves to an empty namespace, with no `build_from_hmonnx` or usable `tcim.builder`. The documented `/usr/local/houmo`, `/usr/local/houmo-sdk`, and `/opt/tcim` paths were absent in that image. No cached image tag matched the inspected compiler/toolchain naming patterns. An ARM Mac or the Tiiny's ARM CPU is not a native host for the documented x86_64 compiler. Compiler emulation and GPU quantization on this Mac were not demonstrated.

## Actual NPU computation experiment

The control model is Houmo's `tools/hm_check/models/xh2_compute.hmm`, a precompiled convolution performance-test graph. We used the existing runtime and explicit device passthrough, without privileged mode. The original Qwen service was not stopped or reconfigured.

| Artifact | Result |
|---|---|
| Current public artifact at model-zoo commit `9121ddac32d0d1a7317103a592b22f955d656050`, 1,065,032 bytes | Rejected during load; no inference. Metadata reports compiler `v1.0.0`, commit `bd5271294`, `march=v2`. |
| Historical artifact at commit `8b1d5572081d1906322970acbdbfccd5c0d70f80`, 978,240 bytes | Loaded with `Status.OK` and executed twice. Metadata reports compiler `v2.4.0 2025-09-27 bed88f38d`; its build path refers to Dadao XH2 v0.4.0. |

The current artifact failed this exact runtime check:

```text
model_->header()->version() == tcim::binary::BinaryMagic_VERSION
RuntimeError: Module load error: Status.UNINITIALIZED
```

This is evidence of binary-format incompatibility for that artifact/runtime pair. Platform release numbers and TCIM's internal version numbers are not directly comparable; simply picking the highest number is insufficient. No version checks were patched or bypassed.

The compatible historical model had one float16 input and output, both shape `[1, 64, 64, 256]`. Its metadata reported 5,186,752 bytes total model memory. Each case called `model.run(True, rt.RunOption(1))` once, using `Xh2HalBackend` and device 0.

| Input | Synchronous run time | Output minimum | Output maximum | All finite |
|---|---:|---:|---:|---|
| All zeros | 8.408 ms | -357.75 | 394.25 | Yes |
| All values 0.03125 | 8.327 ms | -354.5 | 389.75 | Yes |

The output arrays differed. This proves compiled-model execution through the real NPU backend from a separate container. It does **not** validate numerical accuracy against an independent reference, demonstrate compilation of our own graph, or predict LFM3 throughput. These are two control runs, not a sustained benchmark.

### Tested container conditions

The temporary Compose service used the cached ARM64 image above, read-only root filesystem, 1 CPU, 1 GiB memory cap, 128 MiB `/tmp`, `cap_drop: [ALL]`, and `no-new-privileges`. It passed these eight device nodes read/write:

```text
/dev/xh2a_fast_memory_device0
/dev/xh2a_ipu0
/dev/xh2a_memory_allocator_device0_ddr
/dev/xh2a_memory_allocator_device0_spm0
/dev/xh2a_memory_allocator_device0_spm1
/dev/xh2a_memory_transfer_device0
/dev/xh2a_rpmsg_device0_0
/dev/xh2a_system_device0
```

Neither the QSPI flash nor I2C node was passed. These nodes were sufficient for this test; they are not claimed to be a minimal or universal set.

Runtime environment:

```text
PYTHONPATH=/opt/tcim_runtime_lite/python
LD_LIBRARY_PATH=/opt/tcim_runtime_lite/lib
TCIM_FORCE_BACKEND=Xh2HalBackend
HOUMO_TARGET=xh2
PYTHONDONTWRITEBYTECODE=1
```

To reuse the probe inside an equivalent container, download the [pinned historical binary](https://raw.githubusercontent.com/houmo-ai/postmo-modelzoo/8b1d5572081d1906322970acbdbfccd5c0d70f80/tools/hm_check/models/xh2_compute.hmm) to the Mac's asset directory as `xh2_compute_legacy.hmm`. Serve that directory over the USB interface and set `ASSET_URL` to its HTTP base URL. Execute `experiments/npu_compute.py` using `/opt/venv/bin/python3`. The script verifies SHA-256 `a393993d900b9f73c78876a0b57394d44530171a4008c9440636baf3834f29ce` before loading. The script runs **inside Tiiny**, and does not create/start the container or configure the Mac's server. The existing CPU/GPU launcher has no NPU mode.

In this investigation the Mac sent the probe through the Compose Manager, with a 120-second process alarm and temporary asset server. After collecting logs, the temporary service was scaled to zero and removed. The manager's misleading HTTP 500/status behavior also occurred here; model logs established the actual execution result.

## Why LFM3 still needs a port

### Existing and public inference engines

Tiiny's bundled Houmo llama.cpp build 5926 (`9248ffdae`) previously rejected the LFM3 GGUF with `unknown model architecture: 'lfm3'` before inference. Its Qwen deployment uses `prefill.hmm`, `decode.hmm`, `bundle.bin`, a manifest, and a GGUF.

The public [Houmo llama.cpp fork](https://github.com/houmo-ai/llama.cpp/tree/df852f59f72293441b7f8ed5d464cd6cc9b07656) does not remove the compilation requirement. Its [`HouMoLLModel` implementation](https://github.com/houmo-ai/llama.cpp/blob/df852f59f72293441b7f8ed5d464cd6cc9b07656/src/houmo/houmo-llmodel.cpp) loads precompiled prefill/decode graphs. Its [general GGML backend hooks](https://github.com/houmo-ai/llama.cpp/blob/df852f59f72293441b7f8ed5d464cd6cc9b07656/ggml/src/ggml-houmonpu/ggml-houmonpu.cpp#L360-L381) return `nullptr` for backend initialization and `false` for operation support. It is not a generic per-operator NPU backend into which an arbitrary GGUF can be loaded. The inspected architecture table contains LFM2, but not LFM3.

No LFM/Liquid-specific paths were found in the inspected model-zoo or quantflow trees. The public vLLM plugin also had no LFM3 registration in its inspected [model registry](https://github.com/houmo-ai/vllm-postmo/blob/de45f66ac99118c395d3266bbf10a46dde4372fd/vllm_houmo/models/__init__.py). These are scoped source findings, not proof that the hardware cannot implement LFM3.

### LFM3 ONNX graph inspection

We downloaded only the 270,733-byte FP16 graph from [LiquidAI's ONNX export](https://huggingface.co/LiquidAI/LFM3-1.7B-ONNX/tree/57c125dfa29f2d39157e280b1d1f7480b8736d8d), not its roughly 3.59 GB of external weights, and parsed it with the Tiiny image's ONNX package without loading the external tensors.

- 705 graph nodes; standard-domain opset 21 and `com.microsoft` opset 1.
- 154 `MatMul`, 13 `Conv`, and 119 `SimplifiedLayerNormalization` nodes.
- 11 `com.microsoft::GroupQueryAttention` and 22 `com.microsoft::RotaryEmbedding` nodes.
- 39 inputs: four request/control tensors, 13 convolution caches, and 22 key/value caches. There are 36 outputs: logits plus 35 updated caches.
- Symbolic batch/sequence/cache dimensions; convolution caches have shape `[batch, 2048, 7]`, and attention caches `[batch, 4, past_sequence_length, 128]`.

This was a structural inspection, **not a compile or inference attempt with ONNX weights**. Whether Houmo's compiler accepts these exact operators and shapes remains untested. Its HMONNX workflow may require exporting a Houmo-compatible graph, lowering the custom attention/rotary operators, and specializing shapes. Existing GGUF Q4_K_M quantization is not itself a Houmo executable or proof of a compatible quantization scheme. The runtime image's ONNX Runtime installation listed only Azure and CPU execution providers, with no Houmo provider.

## Concrete next requirements

1. Obtain a Linux x86_64 TCIM compiler/quantizer that produces binaries compatible with Tiiny's installed TCIM v2.5.0 runtime, or obtain a vendor-confirmed compatible runtime/compiler pair. Include the runtime commit and the successful v2.4.0 control-model metadata in that request. The latest public toolchain is not yet verified against this device.
2. Obtain a suitable Linux x86_64 build environment and the GPU resources required by the chosen LLM quantization workflow. First compile and run a tiny deterministic graph with an independent numerical reference.
3. Implement an LFM3 export/quantization adapter, including short-convolution state, attention/rotary operations, shapes, and prefill/decode cache semantics. Test one block before the full network.
4. Compile LFM3 artifacts and write a TCIM runtime token loop, or port LFM3 support into Houmo's inference-engine integration. Validate outputs and cache updates against the working CPU model before benchmarking.

No compiler installation, firmware update, driver replacement, or NPU binary-header workaround was performed. CPU and GPU remain the demonstrated LFM3 execution paths.
