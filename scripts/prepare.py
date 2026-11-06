import argparse
import hashlib
import json
import tarfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((ROOT / "sources.json").read_text())


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, destination, expected, headers=None):
    if destination.exists() and sha256(destination) == expected:
        print("Verified cached", destination.name, flush=True)
        return destination
    partial = destination.with_suffix(destination.suffix + ".partial")
    with requests.get(url, headers=headers, stream=True, timeout=(15, 90)) as response:
        response.raise_for_status()
        with partial.open("wb") as stream:
            for chunk in response.iter_content(4 * 1024 * 1024):
                stream.write(chunk)
    if sha256(partial) != expected:
        raise ValueError(f"SHA-256 mismatch: {destination.name}; partial file retained")
    partial.replace(destination)
    print("Downloaded and verified", destination.name, flush=True)
    return destination


def merge_layers(paths, destination):
    with ExitStack() as stack:
        members = {}
        for path in paths:
            archive = stack.enter_context(tarfile.open(path))
            current = archive.getmembers()
            for member in current:
                name = member.name.removeprefix("./").rstrip("/")
                if name.startswith("/") or ".." in Path(name).parts:
                    raise ValueError(f"Unexpected image archive path: {name}")
                basename = name.rsplit("/", 1)[-1]
                if not basename.startswith(".wh."):
                    continue
                parent = name.rsplit("/", 1)[0] + "/" if "/" in name else ""
                removed = parent + basename[4:]
                for old in list(members):
                    if (basename == ".wh..wh..opq" and old.startswith(parent)) or old == removed or old.startswith(removed + "/"):
                        del members[old]
            for member in current:
                name = member.name.removeprefix("./").rstrip("/")
                if name and not name.rsplit("/", 1)[-1].startswith(".wh."):
                    members[name] = (archive, member)
        with tarfile.open(destination, "w:gz", compresslevel=1) as merged:
            for name, (archive, member) in members.items():
                member.name = name
                merged.addfile(member, archive.extractfile(member) if member.isfile() else None)


def main():
    parser = argparse.ArgumentParser(description="Prepare the pinned ARM64 engine and example model on the Mac.")
    parser.add_argument("backend", choices=("cpu", "gpu"))
    parser.add_argument("--skip-model", action="store_true", help="Prepare only the engine for a separately supplied model")
    args = parser.parse_args()
    cache = ROOT / ".local" / "cache" / args.backend
    assets = ROOT / ".local" / "assets"
    cache.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)

    response = requests.get("https://ghcr.io/token", params={"service": "ghcr.io", "scope": "repository:ggml-org/llama.cpp:pull"}, timeout=30)
    response.raise_for_status()
    headers = {"Authorization": "Bearer " + response.json()["token"], "Accept": "application/vnd.oci.image.manifest.v1+json"}
    engine = SOURCES["engine"]
    manifest_path = download(engine["registry"] + "/manifests/" + engine[args.backend], cache / "manifest.json", engine[args.backend].split(":")[1], headers)
    manifest = json.loads(manifest_path.read_text())
    config_digest = manifest["config"]["digest"]
    config_path = download(engine["registry"] + "/blobs/" + config_digest, cache / "image-config.json", config_digest.split(":")[1], headers)
    config = json.loads(config_path.read_text())
    if (config["os"], config["architecture"]) != ("linux", "arm64"):
        raise ValueError("Expected a Linux ARM64 image")

    def fetch_layer(layer):
        digest = layer["digest"].split(":")[1]
        return download(engine["registry"] + "/blobs/" + layer["digest"], cache / (digest + ".tar.gz"), digest, headers)

    with ThreadPoolExecutor(max_workers=3) as pool:
        layers = list(pool.map(fetch_layer, manifest["layers"]))
    rootfs = cache / "rootfs.tar.gz"
    merge_layers(layers, rootfs)
    context = assets / (args.backend + "-engine.tar")
    # Tiiny's remote builder misidentifies archives beginning with a PAX header.
    with tarfile.open(context, "w", format=tarfile.USTAR_FORMAT) as archive:
        archive.add(ROOT / "Dockerfile", arcname="Dockerfile")
        archive.add(rootfs, arcname="rootfs.tar.gz")
    if not args.skip_model:
        model = SOURCES["model"]
        download(model["url"], assets / model["file"], model["sha256"])
    print("Ready:", context, flush=True)


if __name__ == "__main__":
    main()
