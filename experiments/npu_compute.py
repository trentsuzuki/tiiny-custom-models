import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

import numpy as np
import tcim_lite.runtime as rt

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(os.environ['ASSET_URL'] + '/xh2_compute_legacy.hmm', timeout=15) as response:
    content = response.read(2 * 1024 * 1024)
assert hashlib.sha256(content).hexdigest() == 'a393993d900b9f73c78876a0b57394d44530171a4008c9440636baf3834f29ce'
path = Path('/tmp/xh2_compute_legacy.hmm')
path.write_bytes(content)
print('COMPUTE_RUNTIME', rt.get_version(), flush=True)
print('MODEL_INFO', str(rt.Module.load_model_info(str(path))), flush=True)
print('MODEL_LOAD_BEGIN', flush=True)
model = rt.load(str(path), rt.Option(rt.DevManager(0, 'Xh2HalBackend')))
print('MODEL_LOAD_OK', model.init_status, model.get_model_version(), flush=True)
stream = rt.Stream(True)
model.set_stream(stream)
inputs = []
for i in range(model.get_num_inputs()):
    name = model.get_input_name(i)
    info = model.get_input_info(name)
    assert 0 < np.prod(info.shape) <= 16 * 1024 * 1024
    inputs.append((name, info))
    print('INPUT', name, info.shape, str(info.dtype), flush=True)
previous = []
for case, value in [('zero', 0), ('small_constant', 0.03125)]:
    for name, info in inputs:
        model.set_input(name, rt.Tensor(info.ascontiguous(), np.full(info.shape, value, dtype=info.dtype)))
    started = time.perf_counter()
    model.run(True, rt.RunOption(1))
    elapsed = time.perf_counter() - started
    outputs = []
    for i in range(model.get_num_outputs()):
        name = model.get_output_name(i)
        data = model.get_output(name).numpy().copy()
        item = {'name': name, 'shape': list(data.shape), 'dtype': str(data.dtype), 'finite': bool(np.isfinite(data).all()), 'minimum': float(np.min(data)), 'maximum': float(np.max(data)), 'sha256': hashlib.sha256(data.tobytes()).hexdigest()}
        if case != 'zero':
            item['changed_from_zero'] = not np.array_equal(data, previous[i])
        outputs.append(item)
        if case == 'zero':
            previous.append(data)
    print('COMPUTE_RESULT_JSON', json.dumps({'case': case, 'rounds': 1, 'elapsed_ms': elapsed * 1000, 'outputs': outputs}), flush=True)
print('NPU_COMPUTE_COMPLETED', flush=True)
