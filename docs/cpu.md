# Run a custom model on the CPU

Complete [setup](setup.md) and run commands from the repository root. Start with the tested LFM3 example before changing model or memory settings.

## 1. Prepare the engine and model on the Mac

```bash
.venv/bin/python scripts/prepare.py cpu
```

This downloads the pinned official Linux ARM64 CPU image layers, verifies their SHA-256 digests, merges the image filesystem, and writes `.local/assets/cpu-engine.tar`. It also downloads and verifies `LFM3-1.7B-Q4_K_M.gguf` in `.local/assets`.

This is image reconstruction, not compilation from source. OCI image configuration is not implicitly carried across: `Dockerfile` and the Compose profile explicitly set the working directory, entrypoint, environment, and command used in the experiment.

## 2. Serve the assets

In terminal A, leave this running:

```bash
.venv/bin/python scripts/tiiny.py serve
```

## 3. Launch on Tiiny

In terminal B:

```bash
.venv/bin/python scripts/tiiny.py render cpu
.venv/bin/python scripts/tiiny.py start cpu
.venv/bin/python scripts/tiiny.py status cpu
.venv/bin/python scripts/tiiny.py logs cpu
```

`render` prints the exact Compose configuration before deployment. `start` creates or updates `tiiny-custom-cpu`, builds `tiiny-custom-models-cpu:b12124` on Tiiny from the Mac's HTTP context, and waits for `/health` to report `ok`. The initial build can take several minutes; increase `--timeout` if necessary, inspect logs, and stop a failed attempt before retrying.

The default profile uses 4 CPU threads, `--n-gpu-layers 0`, a 2,048-token context, one concurrent request, a 6 GiB container memory limit, and 1,536 MiB of tmpfs capacity. The memory limits are configured caps, not measured consumption. There is no GPU device passed into this container.

## 4. Run inference

```bash
.venv/bin/python scripts/tiiny.py chat cpu \
  --prompt 'What is 17 + 25? Reply with only the number.'
```

The helper prints the complete JSON response, including usage and timings. The default endpoint is `http://[fd80:7:7:7::1]:18080/v1`. A direct request looks like:

```bash
curl --noproxy '*' -g 'http://[fd80:7:7:7::1]:18080/v1/chat/completions' \
  -H 'Content-Type: application/json' \
  -d '{"model":"LiquidAI/LFM3-1.7B","messages":[{"role":"user","content":"Explain why leaves look green in three sentences."}],"max_tokens":256,"temperature":0.1,"top_k":50,"repeat_penalty":1.1,"chat_template_kwargs":{"enable_thinking":false}}'
```

Adjust the host and alias if you changed the config. Thinking is disabled in this LFM3 example because an earlier request spent its entire 128-token budget reasoning and returned empty visible content. This chat-template option is model-specific.

## 5. Stop and verify

```bash
.venv/bin/python scripts/tiiny.py stop cpu
```

Expect `Stopped; connection refused on model port 18080`. The helper obtains the running profile's address/port, applies `scale: 0`, checks that the TCP port refuses connections, then restores the saved profile without starting it. It deliberately does not trust the manager's status field or ordinary stop response.

Press Ctrl-C in terminal A to stop the Mac file server when finished. To run another model, follow [custom models](custom-models.md).
