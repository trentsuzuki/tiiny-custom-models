import argparse
import copy
import functools
import http.server
import ipaddress
import json
import socket
import time
from pathlib import Path
from string import Template
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
OWNER_LABEL = "io.github.trentsuzuki.tiiny-custom-models"


def url(host, port):
    return f"http://[{host}]:{port}" if ":" in host else f"http://{host}:{port}"


def render(value, substitutions):
    if isinstance(value, str):
        return Template(value).substitute(substitutions)
    if isinstance(value, list):
        return [render(item, substitutions) for item in value]
    if isinstance(value, dict):
        return {key: render(item, substitutions) for key, item in value.items()}
    return value


def health(session, endpoint):
    try:
        response = session.get(endpoint + "/health", timeout=(2, 3))
        return response.status_code == 200 and response.json().get("status") == "ok"
    except (requests.RequestException, ValueError):
        return False


def apply(session, manager):
    try:
        response = session.post(manager + "/start", timeout=(3, 90))
        if response.status_code < 500:
            response.raise_for_status()
        else:
            print("Manager detail:", response.text, flush=True)
        print("Manager start HTTP", response.status_code, "— checking the actual model port", flush=True)
    except requests.Timeout:
        print("Manager request timed out; checking the actual model port", flush=True)


def owned_service(session, manager, backend):
    response = session.get(manager, timeout=(3, 15))
    if response.status_code == 404:
        return None
    response.raise_for_status()
    service = json.loads(response.json()["compose_file"])
    if service["services"]["llm"].get("labels", {}).get(OWNER_LABEL) != backend:
        raise ValueError("Refusing to change a service without this repository's ownership label")
    return service


def main():
    parser = argparse.ArgumentParser(description="Run a custom model on Tiiny over its USB network.")
    parser.add_argument("action", choices=("serve", "render", "start", "status", "logs", "chat", "stop"))
    parser.add_argument("backend", choices=("cpu", "gpu"), nargs="?")
    parser.add_argument("--config", type=Path, default=ROOT / ".local" / "config.json")
    parser.add_argument("--timeout", type=int, default=240, help="Seconds to wait for startup or shutdown verification")
    parser.add_argument("--prompt", default="What is 17 + 25? Reply with only the number.")
    args = parser.parse_args()
    if args.action != "serve" and args.backend is None:
        parser.error("backend is required for this action")
    config = json.loads(args.config.read_text())
    for key in ("mac_host", "tiiny_host"):
        if ipaddress.ip_address(config[key]).is_unspecified:
            raise ValueError(f"Set {key} to the specific USB interface address")
    model_file = config["model_file"]
    if Path(model_file).name != model_file or model_file in ("", ".", ".."):
        raise ValueError("model_file must be a filename inside .local/assets")
    assets = ROOT / ".local" / "assets"
    if args.action == "serve":
        if not assets.is_dir():
            raise ValueError("Run scripts/prepare.py before serving assets")
        family = socket.AF_INET6 if ":" in config["mac_host"] else socket.AF_INET
        class Server(http.server.ThreadingHTTPServer):
            address_family = family
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(assets))
        with Server((config["mac_host"], config["http_port"]), handler) as server:
            print("Serving model/image files at", url(config["mac_host"], config["http_port"]), "— Ctrl-C to stop", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
        return

    backend = args.backend
    port = config[backend + "_port"]
    endpoint = url(config["tiiny_host"], port)
    manager = url(config["tiiny_host"], 8880) + "/api/services/tiiny-custom-" + backend
    substitutions = {key.upper(): str(value) for key, value in config.items()}
    substitutions.update(ASSET_URL=url(config["mac_host"], config["http_port"]), MODEL_FILE=quote(model_file, safe=""), PORT=str(port), MEMORY_GIB=str(config[backend + "_memory_gib"]))
    compose = render(json.loads((ROOT / "profiles" / (backend + ".json")).read_text()), substitutions)
    compose["services"]["llm"]["build"]["dockerfile_inline"] = (ROOT / "Dockerfile").read_text()
    session = requests.Session()
    session.trust_env = False

    if args.action == "render":
        print(json.dumps(compose, indent=2))
    elif args.action == "status":
        response = session.get(manager + "/status", timeout=(3, 15))
        print(json.dumps({"endpoint": endpoint, "model_healthy": health(session, endpoint), "manager_http_status": response.status_code, "manager": response.json()}, indent=2))
    elif args.action == "logs":
        response = session.get(manager + "/logs", params={"tail": 500}, timeout=(3, 15))
        response.raise_for_status()
        print(response.json()["logs"])
    elif args.action == "chat":
        payload = {"model": config["model_alias"], "messages": [{"role": "user", "content": args.prompt}], "max_tokens": 256, "temperature": 0.1, "top_k": 50, "repeat_penalty": 1.1, "seed": 42, "chat_template_kwargs": {"enable_thinking": False}, "stream": False}
        response = session.post(endpoint + "/v1/chat/completions", json=payload, timeout=(3, 180))
        response.raise_for_status()
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    elif args.action == "start":
        if health(session, endpoint):
            raise RuntimeError("A model already serves this port. Stop it before changing the configuration.")
        for filename in (model_file, backend + "-engine.tar"):
            if not (assets / filename).is_file():
                raise FileNotFoundError(assets / filename)
            response = session.head(url(config["mac_host"], config["http_port"]) + "/" + quote(filename, safe=""), timeout=(3, 10))
            response.raise_for_status()
        if (assets / model_file).stat().st_size >= config["tmpfs_mib"] * 1024 * 1024:
            raise ValueError("Increase tmpfs_mib: it must be larger than the model file, with room for caches")
        existing = owned_service(session, manager, backend)
        if existing is None:
            response = session.post(url(config["tiiny_host"], 8880) + "/api/services", json={"id": "tiiny-custom-" + backend, "name": "Custom model (" + backend + ")", "description": "Managed by tiiny-custom-models", "compose_file": json.dumps(compose)}, timeout=(3, 15))
        else:
            response = session.put(manager, json={"compose_file": json.dumps(compose)}, timeout=(3, 15))
        response.raise_for_status()
        apply(session, manager)
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            if health(session, endpoint):
                print("Model ready:", endpoint + "/v1", flush=True)
                return
            time.sleep(2)
        raise TimeoutError("Model did not become healthy. Read logs; use stop before retrying.")
    elif args.action == "stop":
        existing = owned_service(session, manager, backend)
        if existing is None:
            raise RuntimeError("No service created by this repository exists for this backend")
        command = existing["services"]["llm"]["command"]
        actual_host = command[command.index("--host") + 1]
        actual_port = int(command[command.index("--port") + 1])
        stopped = copy.deepcopy(existing)
        stopped["services"]["llm"]["scale"] = 0
        response = session.put(manager, json={"compose_file": json.dumps(stopped)}, timeout=(3, 15))
        response.raise_for_status()
        apply(session, manager)
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((actual_host, actual_port), timeout=3):
                    pass
            except ConnectionRefusedError:
                response = session.put(manager, json={"compose_file": json.dumps(existing)}, timeout=(3, 15))
                response.raise_for_status()
                print("Stopped; connection refused on model port", actual_port, flush=True)
                return
            time.sleep(2)
        raise TimeoutError("Model port still accepts connections. Scale-zero config left in place.")


if __name__ == "__main__":
    main()
