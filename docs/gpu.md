# Run a custom model on the GPU

Complete [setup](setup.md). This uses the **Mali-G720-Immortalis GPU through Vulkan**. It does not use the Houmo NPU or CUDA.

## Driver interface found on this Tiiny

| Component | Tested value |
|---|---|
| Kernel device | `/dev/mali0` |
| Kernel module/version | `mali_kbase`, `r53p0-00eac0 (UK version 1.36)` |
| Userspace libraries | `/opt/cixgpu-pro/lib/aarch64-linux-gnu` |
| Vulkan ICD | `/etc/vulkan/icd.d/mali.json` |
| Extra library needed by the container | `/usr/lib/aarch64-linux-gnu/libwayland-server.so.0` |
| Engine | Official ARM64 llama.cpp `server-vulkan`, b12124 |

These paths are specific to the tested firmware. The profile mounts the existing driver stack instead of installing another one. `/dev/dri/renderD*` alone was not the Mali compute interface on this device. GPU-reported memory was about 31 GiB in the SoC shared-memory domain; the separate NPU memory is not added to that GPU pool.

## 1. Prepare and serve assets

```bash
.venv/bin/python scripts/prepare.py gpu
.venv/bin/python scripts/tiiny.py serve
```

The preparation command writes `.local/assets/gpu-engine.tar` and verifies/downloads the same example GGUF. If you already prepared the CPU example, the model is reused after its checksum is checked. Leave the file server running in terminal A.

## 2. Start and verify GPU offloading

In terminal B:

```bash
.venv/bin/python scripts/tiiny.py render gpu
.venv/bin/python scripts/tiiny.py start gpu
.venv/bin/python scripts/tiiny.py status gpu
.venv/bin/python scripts/tiiny.py logs gpu
```

This creates `tiiny-custom-gpu` using `tiiny-custom-models-gpu:b12124`. The profile passes `/dev/mali0`, mounts the driver libraries/ICD read-only, and sets:

```text
LD_LIBRARY_PATH=/vendor-gpu:/vendor-extra:/app
VK_ICD_FILENAMES=/vendor-icd/mali.json
```

`gpu_layers: 99` requests enough layers to offload the tested LFM3 fully. Confirm the log names the Mali device and includes:

```text
using device Vulkan0 (Mali-G720-Immortalis)
offloaded 25/25 layers to GPU
Vulkan0 model buffer size = 1040.82 MiB
```

An HTTP response alone does not establish GPU offloading. Check these backend/layer messages. All layers offloaded also does not mean every part of the application runs on the GPU; tokenization, sampling, and some host buffers remain on the CPU.

Defaults are 4 CPU threads, 2,048-token context, one concurrent request, 8 GiB container memory limit, and 1,536 MiB tmpfs capacity. The Vulkan engine and CPU engine are pinned to the same llama.cpp revision for comparison.

## 3. Run inference and stop

```bash
.venv/bin/python scripts/tiiny.py chat gpu \
  --prompt 'Write three short sentences explaining why leaves appear green.'
.venv/bin/python scripts/tiiny.py stop gpu
```

The default OpenAI-compatible API is `http://[fd80:7:7:7::1]:18082/v1`. Use the same request format as the [CPU example](cpu.md#4-run-inference), changing the port to 18082. Expect shutdown to report a refused connection on port 18082. Press Ctrl-C in the asset-server terminal when finished.

## Performance and other engines

In the short LFM3 comparison, Vulkan generated 25.6 tokens/s versus the CPU's 27.9 tokens/s, and GPU prompt processing was slower. Do not assume GPU offloading improves this workload; profile the model, context, and concurrency you actually need. Lowering `gpu_layers` permits partial offloading, but no partial-offload tuning was performed in the original experiment.

The same Mali stack successfully compiled and ran a custom OpenCL vector-add kernel with 4,096 exact results. That establishes a route for custom GPU code. Other inference engines need an ARM64 build and compatible Vulkan/OpenCL operators; this does not establish automatic PyTorch, ONNX Runtime GPU, or CUDA support.
