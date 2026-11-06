import json
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / '.local/config.json').read_text())
host = CONFIG['tiiny_host']
BASE = 'http://' + ('[' + host + ']' if ':' in host else host) + ':8880/api/services'
ID = 'tiiny-gpu-signal-test'
local = ROOT / '.local'
source = (ROOT / 'experiments/gpu_signal.py').read_text()
command = ('import signal, time, traceback\nsignal.alarm(150)\ntry:\n'
           + '\n'.join('    ' + line for line in source.splitlines())
           + '\nexcept Exception:\n    traceback.print_exc()\nfinally:\n'
           + '    print("GPU_PROBE_FINISHED", flush=True)\n    time.sleep(30)\n')
probe = {
    'image': 'z_image_turbo_houmo:20260115', 'pull_policy': 'never', 'platform': 'linux/arm64',
    'user': '1000:1000', 'read_only': True, 'cap_drop': ['ALL'],
    'security_opt': ['no-new-privileges:true'], 'network_mode': 'none',
    'devices': ['/dev/mali0:/dev/mali0:rw'], 'cpus': 1, 'mem_limit': '1g',
    'pids_limit': 128, 'restart': 'no', 'tmpfs': ['/tmp:rw,size=128m,uid=1000,gid=1000,mode=1777'],
    'labels': {'io.github.trentsuzuki.tiiny-custom-models.gpu-signal': 'owned'},
    'entrypoint': ['/opt/venv/bin/python3', '-c'], 'command': [command],
    'environment': {'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
                    'LD_LIBRARY_PATH': '/vendor-gpu:/vendor-extra', 'XDG_CACHE_HOME': '/tmp/cache',
                    'XDG_RUNTIME_DIR': '/tmp', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'},
    'volumes': [
        {'type': 'bind', 'source': src, 'target': dest, 'read_only': True, 'bind': {'create_host_path': False}}
        for src, dest in [('/opt/cixgpu-pro/lib/aarch64-linux-gnu', '/vendor-gpu'),
                          ('/usr/lib/aarch64-linux-gnu/libwayland-server.so.0', '/vendor-extra/libwayland-server.so.0')]
    ],
}
compose = {'services': {'probe': probe}}
(local / 'gpu-signal-compose.json').write_text(json.dumps(compose, indent=2))
session = requests.Session()
session.trust_env = False
manager = BASE + '/' + ID
existing = session.get(manager, timeout=(3, 15))
if existing.status_code != 404:
    raise RuntimeError('Temporary service name already exists; refusing to replace it')
response = session.post(BASE, json={'id': ID, 'name': 'Python GPU signal experiment',
                                  'description': 'Temporary bounded experiment', 'compose_file': json.dumps(compose)}, timeout=(3, 15))
response.raise_for_status()
result = None
try:
    try:
        response = session.post(manager + '/start', timeout=(3, 45))
        print('Manager start HTTP', response.status_code, '— checking actual results', flush=True)
        if response.status_code < 500:
            response.raise_for_status()
    except requests.Timeout:
        print('Start timed out; checking actual results', flush=True)
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        response = session.get(manager + '/logs', params={'tail': 1000}, timeout=(3, 15))
        response.raise_for_status()
        logs = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', response.json()['logs'])
        (local / 'gpu-signal.log').write_text(logs)
        if 'GPU_PROBE_FINISHED' in logs:
            break
        time.sleep(1)
    else:
        raise TimeoutError('GPU probe did not finish within its deadline')
    for marker, filename in [('GPU_SIGNAL_RESULT_JSON', 'gpu-signal-result.json'), ('GPU_SIGNAL_PLOT_JSON', 'gpu-signal-plot-data.json')]:
        matching = [line.split(marker, 1)[1].strip() for line in logs.splitlines() if marker in line]
        if not matching:
            raise RuntimeError('Experiment failed; inspect .local/gpu-signal.log')
        data = json.loads(matching[-1])
        (local / filename).write_text(json.dumps(data, indent=2) + '\n')
        if marker == 'GPU_SIGNAL_RESULT_JSON':
            result = data
    assert result['passed'] and all(case['passed'] for case in result['cases'])
    print(json.dumps({'device': result['device']['name'], 'passed': result['passed'],
                      'cases': [{'batch': case['batch'], 'median_ms': case['median_ms'],
                                 'output_relative_l2_error': case['output_relative_l2_error']}
                                for case in result['cases']]}, indent=2), flush=True)
finally:
    probe['scale'] = 0
    session.put(manager, json={'compose_file': json.dumps(compose)}, timeout=(3, 15)).raise_for_status()
    response = session.post(manager + '/start', timeout=(3, 45))
    print('Scale-zero HTTP', response.status_code, flush=True)
    response = session.delete(manager, timeout=(3, 30))
    response.raise_for_status()
    if session.get(manager, timeout=(3, 15)).status_code != 404:
        raise RuntimeError('Temporary service still exists after cleanup')
    print('Cleanup verified: temporary service removed', flush=True)
