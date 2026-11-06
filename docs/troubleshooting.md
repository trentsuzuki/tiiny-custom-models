# Troubleshooting

## Manager status is unreliable

On firmware/service 0.1.29, the Compose Manager sometimes returned HTTP 500 with `Service did not reach running state after 3s`, and its status endpoint showed `stopped` with no containers, even while llama.cpp served valid responses.

Use `/health`, inference responses, and engine logs to establish the actual state. `scripts/tiiny.py start` prints manager 5xx details and then checks health. A new GPU build also produced a container-name conflict response even though the model was healthy. A genuine build failure can produce the same HTTP status; if health never succeeds, inspect the printed error and logs instead of assuming every 500 is harmless.

The ordinary manager `/stop` endpoint also returned success without stopping a server. The provided `stop` command applies `scale: 0` to only the owned service, verifies a refused connection to its saved address/port, and restores its configuration without starting it. A network timeout is not accepted as proof of shutdown. Do not power-cycle or stop built-in model services as a substitute for checking this experiment's lifecycle.

## Model never becomes healthy

1. Check the Mac file-server terminal for engine/model requests.
2. Confirm `mac_host` is still the Mac's USB address and the file server is running.
3. Read `scripts/tiiny.py logs cpu` or `logs gpu`.
4. Check model file size against tmpfs and container memory caps.
5. Stop the experiment before retrying. A startup timeout can leave a container loading or running.

The scripts bypass HTTP proxy environment variables for local USB requests. If using `curl` manually, include `--noproxy '*'` and quote/bracket IPv6 URLs. Asset preparation still uses the Mac's ordinary internet connection.

## Remote build says `open context: no such file or directory`

During packaging validation, a remote build archive beginning with a PAX extended header produced this error before container startup. Recreating the same two-file context in USTAR format succeeded. `prepare.py` explicitly uses USTAR for the outer build archive. Preserve that setting if adapting the packaging code; the root filesystem remains a gzip-compressed tar inside it.

## GPU cannot be found

Check for `/dev/mali0`, matching vendor libraries, the Mali ICD, and the extra Wayland library listed in the GPU guide. Bind mounts use `create_host_path: false`, so missing firmware-specific paths fail instead of creating empty host directories. These recipes do not install or replace drivers.

Logs must identify `Mali-G720-Immortalis` and show actual offloaded layers. A software Vulkan device such as llvmpipe is not evidence of GPU acceleration.

## Empty visible answer or token limit reached

An early LFM3 request used its 128-token budget for reasoning. The helper uses a 256-token budget and disables thinking for the tested chat template. Inspect the response's content, reasoning fields, usage, and finish reason before treating this as a device failure.

## Download checksum failure

The script verifies manifests, image configuration, layers, and the example model. It leaves a failed `.partial` download in the ignored cache. Investigate a mismatch before changing the expected hash; the model URL uses upstream `main`, while the expected SHA-256 pins the tested bytes.

## Existing service or occupied port

The helper refuses to modify a service missing this repository's ownership label. It also refuses to start over a healthy inference endpoint. Inspect the existing service and choose another port if needed. No broad Docker deletion or pruning is required.
