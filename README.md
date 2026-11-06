# tiiny-custom-models

Run a custom GGUF model on a Tiiny AI device using its ARM CPU or Mali GPU, from a Mac over USB. The tested example is **LiquidAI LFM3-1.7B Q4_K_M**, running in a separate llama.cpp container outside Tiiny's supported-model catalog.

**CPU and Vulkan GPU inference both worked on a Tiiny z01 with firmware/service version 0.1.29.** No SSH login, firmware update, or driver replacement was needed. This is an experimental workflow tested on one device, not an official Tiiny integration or a claim that every model or firmware version works.

## Start here

1. Follow [Mac and USB setup](docs/setup.md).
2. Run the example with the [CPU guide](docs/cpu.md) or [GPU guide](docs/gpu.md).
3. Substitute another compatible GGUF using [custom model instructions](docs/custom-models.md).
4. Read [results and limitations](docs/results.md), [troubleshooting](docs/troubleshooting.md), and the [NPU investigation checkpoint](docs/npu.md).

The Mac downloads and serves engine/model files. **Inference runs on Tiiny**, not on the Mac. The device does not need internet access during this workflow.

```text
Mac: download verified ARM64 engine + GGUF
  │
  ├── USB HTTP file server ──► Tiiny Docker image build / model download
  └── USB management API ────► Separate llama.cpp container
                                  ├── CPU: ARM64, GPU layers = 0
                                  └── GPU: Mali device + existing Vulkan driver
Mac client ◄─────────────────────── OpenAI-compatible inference API
```

## Results so far

Same model, llama.cpp b12124, 4 CPU threads, 2,048-token context, one short request per backend:

| Backend | Prompt processing | Token generation | Generated tokens | Request wall time |
|---|---:|---:|---:|---:|
| CPU | 62.6 tokens/s | 27.9 tokens/s | 95 | 3.72 s |
| Vulkan GPU, 25/25 layers offloaded | 9.0 tokens/s | 25.6 tokens/s | 100 | 6.23 s |

The GPU worked but was slower in this small test. Different output lengths, short prompts, and the absence of sustained-load or concurrency testing limit the comparison. These are engine-reported token rates, not hardware peak throughput.

A custom OpenCL vector-add kernel also passed on the GPU. Direct NPU allocation and a memory round trip passed, but **custom NPU inference and LFM3 on the NPU have not been achieved**.

## Repository contents

| File or directory | Purpose |
|---|---|
| `README.md` | Overview, measured results, and guide index. |
| `scripts/prepare.py` | Download and hash-check pinned official ARM64 engine layers and the example GGUF; create an offline build context. |
| `scripts/tiiny.py` | Serve assets, render profiles, start a model, inspect status/logs, run a prompt, and verify shutdown. |
| `profiles/cpu.json`, `profiles/gpu.json` | Reviewable Compose templates for each backend. |
| `config.example.json` | Connection addresses, model name, context, threads, and memory settings. |
| `Dockerfile`, `sources.json`, `requirements.txt` | Offline image reconstruction, pinned source hashes, and host dependencies. |
| `docs/setup.md` | Mac environment, USB addresses, and connection prerequisites. |
| `docs/cpu.md`, `docs/gpu.md` | Backend-specific launch, inference, verification, and shutdown instructions. |
| `docs/custom-models.md` | Model substitution and configuration/memory limits. |
| `docs/results.md` | Results and unresolved limits from the investigation. |
| `docs/troubleshooting.md` | Observed manager, archive, driver, and generation problems. |
| `docs/npu.md` | NPU evidence and next steps, without claiming custom NPU inference. |
| `docs/sources.md` | Artifact hashes, primary references, and provenance. |
| `results/2026-11-05.json` | Selected measurements and evidence from the original experiments. |
| `results/workflow-validation.json` | Live CPU/GPU checks of this repository's packaged workflow. |
| `.gitignore` | Exclude local assets, environments, credentials files, and binary model/image files. |
| `.local/` | Ignored local configuration, downloaded assets, and cache; created during setup. |

The scripts manage only services named `tiiny-custom-cpu` and `tiiny-custom-gpu` bearing this repository's ownership label. They do not manage the built-in Qwen service. CPU/GPU profiles bind the inference server to the configured Tiiny USB address. Model weights, container binaries, credentials, and machine-specific configuration are not committed.

Tiiny's manager can report `stopped` or return HTTP 500 while a model is running. The scripts check the actual inference endpoint and use Compose scale-to-zero for shutdown. Read [the lifecycle caveat](docs/troubleshooting.md#manager-status-is-unreliable) before substituting manual API commands.

## Upstream software and models

The engine is the official [llama.cpp](https://github.com/ggml-org/llama.cpp/tree/8bed7cfb999cebc03d2cc5b78911026a8d57549a) ARM64 build b12124. The example weights come from [LiquidAI/LFM3-1.7B-GGUF](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF). Their respective licenses apply; this repository downloads those artifacts rather than redistributing them. See [sources and provenance](docs/sources.md).
