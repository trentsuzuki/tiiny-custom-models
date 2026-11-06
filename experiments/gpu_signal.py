import ctypes as C
import hashlib
import json
import platform
import statistics
import time

import numpy as np
import scipy
from scipy import fft

N, WG, FS = 2048, 128, 32_000_000
ROUNDS = 15
SOURCE = r'''
#define N 2048
#define LOGN 11
uint reverse_index(uint value) {
    uint result = 0;
    for (uint bit = 0; bit < LOGN; ++bit) {
        result = (result << 1) | (value & 1);
        value >>= 1;
    }
    return result;
}
void transform(__local float2 *scratch, int inverse) {
    uint lane = get_local_id(0), width = get_local_size(0);
    for (uint span = 2; span <= N; span <<= 1) {
        uint half_span = span >> 1;
        for (uint butterfly = lane; butterfly < N/2; butterfly += width) {
            uint j = butterfly % half_span;
            uint i = (butterfly / half_span) * span + j;
            float angle = (inverse ? 6.283185307179586f : -6.283185307179586f)
                          * (float)j / (float)span;
            float2 w = (float2)(cos(angle), sin(angle));
            float2 a = scratch[i], b = scratch[i + half_span];
            float2 t = (float2)(w.x*b.x - w.y*b.y, w.x*b.y + w.y*b.x);
            scratch[i] = a + t;
            scratch[i + half_span] = a - t;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
__kernel void window_fft(__global const float *input,
                         __global const float *window,
                         __global float2 *spectrum) {
    __local float2 scratch[N];
    uint base = get_group_id(0) * N;
    for (uint i = get_local_id(0); i < N; i += get_local_size(0))
        scratch[reverse_index(i)] = (float2)(input[base+i]*window[i], 0.0f);
    barrier(CLK_LOCAL_MEM_FENCE);
    transform(scratch, 0);
    for (uint i = get_local_id(0); i < N; i += get_local_size(0))
        spectrum[base+i] = scratch[i];
}
__kernel void bandpass_ifft(__global const float2 *spectrum,
                            __global const float *mask,
                            __global float *output) {
    __local float2 scratch[N];
    uint base = get_group_id(0) * N;
    for (uint i = get_local_id(0); i < N; i += get_local_size(0))
        scratch[reverse_index(i)] = spectrum[base+i] * mask[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    transform(scratch, 1);
    for (uint i = get_local_id(0); i < N; i += get_local_size(0))
        output[base+i] = scratch[i].x / (float)N;
}
'''

cl = C.CDLL('/vendor-gpu/libOpenCL.so')
P, U, I, Z, L = C.c_void_p, C.c_uint, C.c_int, C.c_size_t, C.c_ulong
signatures = {
    'clGetPlatformIDs': (I, [U, C.POINTER(P), C.POINTER(U)]),
    'clGetDeviceIDs': (I, [P, L, U, C.POINTER(P), C.POINTER(U)]),
    'clGetDeviceInfo': (I, [P, U, Z, P, C.POINTER(Z)]),
    'clCreateContext': (P, [P, U, C.POINTER(P), P, P, C.POINTER(I)]),
    'clCreateCommandQueue': (P, [P, P, L, C.POINTER(I)]),
    'clCreateBuffer': (P, [P, L, Z, P, C.POINTER(I)]),
    'clCreateProgramWithSource': (P, [P, U, C.POINTER(C.c_char_p), C.POINTER(Z), C.POINTER(I)]),
    'clBuildProgram': (I, [P, U, C.POINTER(P), C.c_char_p, P, P]),
    'clGetProgramBuildInfo': (I, [P, P, U, Z, P, C.POINTER(Z)]),
    'clCreateKernel': (P, [P, C.c_char_p, C.POINTER(I)]),
    'clSetKernelArg': (I, [P, U, Z, P]),
    'clEnqueueNDRangeKernel': (I, [P, P, U, C.POINTER(Z), C.POINTER(Z), C.POINTER(Z), U, P, C.POINTER(P)]),
    'clEnqueueWriteBuffer': (I, [P, P, U, Z, Z, P, U, P, C.POINTER(P)]),
    'clEnqueueReadBuffer': (I, [P, P, U, Z, Z, P, U, P, C.POINTER(P)]),
    'clGetEventProfilingInfo': (I, [P, U, Z, P, C.POINTER(Z)]),
    'clFinish': (I, [P]),
}
for name in ['clReleaseEvent', 'clReleaseMemObject', 'clReleaseKernel', 'clReleaseProgram', 'clReleaseCommandQueue', 'clReleaseContext']:
    signatures[name] = (I, [P])
for name, (restype, argtypes) in signatures.items():
    function = getattr(cl, name)
    function.restype, function.argtypes = restype, argtypes


def check(status):
    if status != 0:
        raise RuntimeError('OpenCL status ' + str(status))


platform_id, device, count, error = P(), P(), U(), I()
check(cl.clGetPlatformIDs(1, C.byref(platform_id), C.byref(count)))
check(cl.clGetDeviceIDs(platform_id, 4, 1, C.byref(device), C.byref(count)))
device_info = {}
for name, code, cls in [('name', 0x102B, None), ('version', 0x102F, None), ('type', 0x1000, L), ('compute_units', 0x1002, U), ('max_work_group_size', 0x1004, Z), ('local_memory_bytes', 0x1023, L)]:
    value = cls() if cls else C.create_string_buffer(1024)
    check(cl.clGetDeviceInfo(device, code, C.sizeof(value), C.byref(value), None))
    device_info[name] = value.value if cls else value.value.decode()
assert device_info['type'] == 4 and 'Mali' in device_info['name']
assert device_info['local_memory_bytes'] >= N * 8 and device_info['max_work_group_size'] >= WG
context = P(cl.clCreateContext(None, 1, C.byref(device), None, None, C.byref(error)))
check(error.value)
queue = P(cl.clCreateCommandQueue(context, device, 2, C.byref(error)))
check(error.value)
source = C.c_char_p(SOURCE.encode())
program = P(cl.clCreateProgramWithSource(context, 1, C.byref(source), None, C.byref(error)))
check(error.value)
started = time.perf_counter()
status = cl.clBuildProgram(program, 1, C.byref(device), b'-cl-std=CL1.2', None, None)
compile_ms = (time.perf_counter() - started) * 1000
if status:
    log = C.create_string_buffer(16384)
    check(cl.clGetProgramBuildInfo(program, device, 0x1183, len(log), log, None))
    raise RuntimeError(log.value.decode())
kernels = []
for name in [b'window_fft', b'bandpass_ifft']:
    kernels.append(P(cl.clCreateKernel(program, name, C.byref(error))))
    check(error.value)


def buffer(array, flags):
    handle = P(cl.clCreateBuffer(context, flags, array.nbytes, array.ctypes.data if flags & 32 else None, C.byref(error)))
    check(error.value)
    return handle


def event_ms(event):
    start, end = L(), L()
    check(cl.clGetEventProfilingInfo(event, 0x1282, C.sizeof(start), C.byref(start), None))
    check(cl.clGetEventProfilingInfo(event, 0x1283, C.sizeof(end), C.byref(end), None))
    return (end.value - start.value) / 1e6


window = np.hanning(N).astype(np.float32)
frequency = np.fft.fftfreq(N, 1 / FS)
mask = ((np.abs(frequency) >= 2e6) & (np.abs(frequency) <= 4e6)).astype(np.float32)
window_buffer, mask_buffer = buffer(window, 36), buffer(mask, 36)
results = []
for batch in [1, 256]:
    rng = np.random.default_rng(20261105)
    t = np.arange(N, dtype=np.float64) / FS
    phase = rng.uniform(-np.pi, np.pi, (batch, 1))
    data = (np.sin(2*np.pi*3e6*t + phase) + 0.6*np.sin(2*np.pi*8e6*t + phase/2)
            + 0.05*rng.standard_normal((batch, N))).astype(np.float32)
    spectrum = np.empty((batch, N), dtype=np.complex64)
    output = np.empty_like(data)
    input_buffer, spectrum_buffer, output_buffer = buffer(data, 4), buffer(spectrum, 1), buffer(output, 2)
    for kernel, args in zip(kernels, [(input_buffer, window_buffer, spectrum_buffer), (spectrum_buffer, mask_buffer, output_buffer)]):
        for index, arg in enumerate(args):
            check(cl.clSetKernelArg(kernel, index, C.sizeof(P), C.byref(arg)))
    global_size, local_size = Z(batch * WG), Z(WG)

    def gpu_run(transfer=True, read_spectrum=False):
        started = time.perf_counter()
        if transfer:
            check(cl.clEnqueueWriteBuffer(queue, input_buffer, 1, 0, data.nbytes, data.ctypes.data, 0, None, None))
        events = [P(), P()]
        for kernel, event in zip(kernels, events):
            check(cl.clEnqueueNDRangeKernel(queue, kernel, 1, None, C.byref(global_size), C.byref(local_size), 0, None, C.byref(event)))
        if transfer:
            check(cl.clEnqueueReadBuffer(queue, output_buffer, 1, 0, output.nbytes, output.ctypes.data, 0, None, None))
        else:
            check(cl.clFinish(queue))
        wall_ms = (time.perf_counter() - started) * 1000
        timings = [event_ms(event) for event in events]
        for event in events:
            check(cl.clReleaseEvent(event))
        if read_spectrum:
            check(cl.clEnqueueReadBuffer(queue, spectrum_buffer, 1, 0, spectrum.nbytes, spectrum.ctypes.data, 0, None, None))
        return wall_ms, sum(timings)

    gpu_run(read_spectrum=True)
    double_input = data.astype(np.float64) * window.astype(np.float64)
    reference_fft = np.fft.fft(double_input, axis=-1)
    reference_output = np.fft.ifft(reference_fft * mask, axis=-1).real
    fft_rel_l2 = float(np.linalg.norm(spectrum.astype(np.complex128) - reference_fft) / np.linalg.norm(reference_fft))
    output_rel_l2 = float(np.linalg.norm(output.astype(np.float64) - reference_output) / np.linalg.norm(reference_output))
    output_max_error = float(np.max(np.abs(output - reference_output)))
    assert np.isfinite(spectrum).all() and np.isfinite(output).all()
    assert fft_rel_l2 < 2e-5 and output_rel_l2 < 2e-5 and output_max_error < 2e-5
    mean_psd = np.mean(np.abs(spectrum[:, :N//2+1].astype(np.complex128))**2, axis=0)
    after_fft = np.fft.fft(output.astype(np.float64), axis=-1)
    tone_bins = [int(3e6 * N / FS), int(8e6 * N / FS)]
    gain_db = [float(20*np.log10(np.linalg.norm(after_fft[:, k]) / np.linalg.norm(reference_fft[:, k]))) for k in tone_bins]
    peak_candidates = np.flatnonzero((mean_psd[1:-1] > mean_psd[:-2]) & (mean_psd[1:-1] > mean_psd[2:])) + 1
    peaks = sorted((peak_candidates[np.argsort(mean_psd[peak_candidates])[-2:]] * FS / N).tolist())
    assert peaks == [3_000_000.0, 8_000_000.0]
    assert abs(gain_db[0]) < 0.001 and gain_db[1] < -80

    def cpu_run():
        transformed = fft.fft(data * window, axis=-1, workers=1)
        return fft.ifft(transformed * mask, axis=-1, workers=1).real

    cpu_output = cpu_run()
    assert cpu_output.dtype == np.float32
    assert float(np.linalg.norm(cpu_output.astype(np.float64) - reference_output) / np.linalg.norm(reference_output)) < 2e-5
    for _ in range(3):
        gpu_run()
        cpu_run()
    gpu_wall, gpu_kernel, resident_wall, cpu_wall = [], [], [], []
    for _ in range(ROUNDS):
        wall_ms, kernel_ms = gpu_run()
        gpu_wall.append(wall_ms)
        gpu_kernel.append(kernel_ms)
        resident_wall.append(gpu_run(transfer=False)[0])
        started = time.perf_counter()
        cpu_run()
        cpu_wall.append((time.perf_counter() - started) * 1000)
    result = {'batch': batch, 'samples': int(data.size), 'rounds': ROUNDS, 'fft_relative_l2_error': fft_rel_l2,
              'output_relative_l2_error': output_rel_l2, 'output_max_absolute_error': output_max_error,
              'dominant_frequencies_hz': peaks, 'gain_at_3mhz_db': gain_db[0], 'gain_at_8mhz_db': gain_db[1],
              'median_ms': {'gpu_host_to_host': statistics.median(gpu_wall), 'gpu_kernel_events': statistics.median(gpu_kernel),
                            'gpu_resident_python_wall': statistics.median(resident_wall), 'cpu_scipy_single_worker': statistics.median(cpu_wall)},
              'timings_ms': {'gpu_host_to_host': gpu_wall, 'gpu_kernel_events': gpu_kernel, 'gpu_resident_python_wall': resident_wall, 'cpu_scipy_single_worker': cpu_wall},
              'input_sha256': hashlib.sha256(data.tobytes()).hexdigest(), 'output_sha256': hashlib.sha256(output.tobytes()).hexdigest(), 'passed': True}
    results.append(result)
    print('GPU_SIGNAL_CASE_JSON', json.dumps(result), flush=True)
    if batch == 256:
        print('GPU_SIGNAL_PLOT_JSON', json.dumps({'sample_rate_hz': FS, 'nfft': N, 'input_first_128': data[0, :128].tolist(), 'output_first_128': output[0, :128].tolist(), 'reference_first_128': reference_output[0, :128].tolist(), 'before_power': mean_psd.tolist(), 'after_power': np.mean(np.abs(after_fft[:, :N//2+1])**2, axis=0).tolist()}), flush=True)
    for handle in [input_buffer, spectrum_buffer, output_buffer]:
        check(cl.clReleaseMemObject(handle))
print('GPU_SIGNAL_RESULT_JSON', json.dumps({'device': device_info, 'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'fft_length': N, 'work_group_size': WG, 'sample_rate_hz': FS, 'passband_hz': [2_000_000, 4_000_000], 'precision': 'float32/complex64', 'compile_ms': compile_ms, 'kernel_source_sha256': hashlib.sha256(SOURCE.encode()).hexdigest(), 'cases': results, 'passed': True}), flush=True)
for function_name, handles in [('clReleaseMemObject', [window_buffer, mask_buffer]), ('clReleaseKernel', kernels), ('clReleaseProgram', [program]), ('clReleaseCommandQueue', [queue]), ('clReleaseContext', [context])]:
    for handle in handles:
        check(getattr(cl, function_name)(handle))
