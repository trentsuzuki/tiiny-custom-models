# Mac and USB setup

## Tested prerequisites

- Tiiny z01, firmware/service version **0.1.29**, Linux **6.6.89-generic**, ARM64.
- A Mac connected to the device's USB network interface; the tested interface was USB NCM.
- Python **3.11 or newer** on the Mac, plus `requests`.
- Internet access on the Mac for initial downloads. Allow several GB of free disk space for image layers, reconstructed filesystems, build contexts, and the 1.10 GB example model.
- Reachable Tiiny Compose Manager on port **8880**. This is the management interface observed on the tested firmware; its presence is not guaranteed on other releases.

Docker Desktop, SSH, and tiiny-sdk are not required for this path. Earlier SDK tests used `tiiny-sdk==0.1.4` successfully, but the custom-model workflow talks directly to Tiiny's container manager and the new inference endpoint.

## Prepare the checkout

```bash
git clone https://github.com/trentsuzuki/tiiny-custom-models.git
cd tiiny-custom-models
python3 --version
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
mkdir -p .local
cp config.example.json .local/config.json
```

Use a Python 3.11+ executable in place of `python3` if your default is older. The repository is private, so cloning requires GitHub access.

## Find the USB addresses

On the Mac:

```bash
networksetup -listallhardwareports
ifconfig
```

Find the USB network interface and its non-link-local IPv6 address. In the tested setup, Tiiny was `fd80:7:7:7::1` and the Mac had a different address in the same `fd80:7:7:7::/64` subnet. Interface numbers such as `en10` are machine-specific. Use the **Mac's address** for `mac_host`, not Tiiny's address or a `fe80::` link-local address.

Edit `.local/config.json`:

- Set `mac_host` to the Mac USB IPv6 address.
- Verify `tiiny_host` is the device address.
- Keep the other defaults for the first LFM3 test.

Check the manager without printing its service configurations:

```bash
curl --noproxy '*' -g -sS -o /dev/null -w '%{http_code}\n' \
  'http://[fd80:7:7:7::1]:8880/api/services'
```

The tested setup returned `200` without authentication on this local endpoint. If yours returns `401`/`403` or has no manager, stop here and establish the supported access method for that firmware; these scripts do not implement a login flow. If macOS asks for Local Network access for your terminal/Python process, allow it for the USB connection.

## What the workflow changes

It creates a dedicated Compose service and an image on Tiiny. The container runs as UID 1000 with a read-only root filesystem, temporary RAM storage, and an inference port bound to the USB address. The GPU profile additionally exposes `/dev/mali0` and mounts existing GPU libraries read-only. It does not modify firmware or host drivers.

The Mac's temporary file server exposes only `.local/assets` on the chosen USB address. Keep it running during image build and model startup. It can be stopped after the model is loaded, but must be restarted before a new launch because the model is downloaded into container tmpfs.

Stopping a model removes its running container through Compose scaling and verifies a refused TCP connection. The service definition, built image, and Mac cache remain for reuse. There is no automatic global Docker or download-cache pruning.
