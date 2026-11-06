# Substitute another model

The packaged workflow accepts a **single GGUF file supported by the pinned llama.cpp engine**. It is not an arbitrary PyTorch/ONNX loader. LFM3-1.7B Q4_K_M is the model actually tested.

1. Obtain the model's GGUF from a trusted source, check its license, and verify the publisher's checksum when available.
2. Stop the relevant custom service before changing its configuration.
3. Put the file in `.local/assets`, using a simple filename such as `my-model.gguf`.
4. Change `model_file` and `model_alias` in `.local/config.json`.
5. Check the memory, context, engine, and template requirements below.
6. Run the usual `serve`, `start cpu` or `start gpu`, `logs`, and `chat` commands.

To prepare only an engine when supplying your own model:

```bash
.venv/bin/python scripts/prepare.py cpu --skip-model
# or:
.venv/bin/python scripts/prepare.py gpu --skip-model
```

## Parameters to review

| Setting | Meaning |
|---|---|
| `model_file` | Filename in `.local/assets`; the file is downloaded to Tiiny container tmpfs on startup. |
| `model_alias` | Name accepted by the new inference API. It does not register the model in TiinyOS's catalog. |
| `context` | Context length allocated by llama.cpp. Start small; model-advertised maximum context is not a memory/performance guarantee. |
| `threads` | CPU threads and container CPU quota. |
| `gpu_layers` | GPU profile's requested offload count; verify actual placement in logs. Increase above 99 if a larger architecture needs it. |
| `tmpfs_mib` | Must exceed the model file size, with additional room for caches and temporary data. |
| `cpu_memory_gib`, `gpu_memory_gib` | Container memory cap, including RAM-backed files, model buffers, context caches, and runtime overhead. |
| `cpu_port`, `gpu_port` | Separate inference ports; do not reuse a port occupied by another service. |

The example uses RAM storage for model delivery. Larger models may need persistent disk-backed storage and a revised profile. That path has not been validated here. The tested 6/8 GiB caps are not estimates of a model's minimum memory requirement, and the NPU's separate memory cannot simply be assigned to the GPU container.

## Model/engine compatibility

- The pinned engine is b12124 at revision `8bed7cfb999cebc03d2cc5b78911026a8d57549a`. A newer architecture may require another engine build; changing only the filename will not add architecture support.
- `unknown model architecture` indicates an engine compatibility issue. An older Tiiny NPU-enabled llama.cpp build rejected LFM3 this way.
- The `chat` helper uses LFM3 sampling settings and `enable_thinking: false`. Other models may require different chat templates or sampling parameters; submit a model-appropriate request to `/v1/chat/completions`.
- Split GGUFs, multimodal projector files, remote tokenizer dependencies, long context, batching, and simultaneous CPU/GPU workloads are outside this workflow's validation.
- If updating engine digests in `sources.json`, also update profile image tags, verify Linux ARM64 configuration and Vulkan support, rebuild, and repeat inference/offload/shutdown checks. The preparation script is intended for the pinned server images, not as a general OCI image converter.

The model is available on the custom server's `/v1/models` endpoint. It need not appear in Tiiny's native model list or TiinyOS UI.
