"""Untimed memory/environment diagnostics for the paper benchmark."""
import hashlib
import os
import platform
import subprocess
from pathlib import Path

import torch


def storage_bytes(tensors):
    """Count backing storage once, including capacity retained by tensor views."""
    seen = {}
    for tensor in tensors:
        if tensor is not None:
            storage = tensor.untyped_storage()
            seen[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(seen.values())


def cache_bytes(cache):
    if cache is None:
        return 0
    return storage_bytes(t for state in cache for t in (state.get('attn_state') or []))


def memory_snapshot(model, device, cache=None):
    body = model.model
    offloaders = list((getattr(body, '_offloader_cache', None) or {}).values())
    active = getattr(body, 'memory_offloader', None)
    if active is not None and all(active is not item for item in offloaders):
        offloaders.append(active)
    slots = [slot for off in offloaders for slot in off.slots]
    table = getattr(body, 'memory_table', None)
    pinned_table = [table] if table is not None and table.is_pinned() else []
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    return dict(
        gpu_allocated_bytes=torch.cuda.memory_allocated(device),
        gpu_reserved_bytes=torch.cuda.memory_reserved(device),
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        host_rss_bytes=int(status['VmRSS'].split()[0]) * 1024,
        host_process_high_water_bytes=int(status['VmHWM'].split()[0]) * 1024,
        host_high_water_scope='process lifetime, including model loading',
        raw_table_snapshot_bytes=storage_bytes(getattr(body, '_raw_m_proj_weights', None) or []),
        cpu_table_bytes=storage_bytes([getattr(body, 'memory_table', None)]),
        cpu_table_pinned_bytes=storage_bytes(pinned_table),
        offload_pinned_bytes=storage_bytes([s['host'] for s in slots] + pinned_table),
        offload_gpu_buffer_bytes=storage_bytes(s['gpu'] for s in slots),
        offloader_capacities=[dict(batch=off.batch, length=off.seq_len, policy=off.policy,
                                  host_bytes=storage_bytes(s['host'] for s in off.slots),
                                  gpu_bytes=storage_bytes(s['gpu'] for s in off.slots)) for off in offloaders],
        kv_cache_storage_bytes=cache_bytes(cache),
    )


def command_output(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=10)
        return dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.SubprocessError) as exc:
        return dict(error=str(exc))


def source_state():
    root = Path(os.environ["MA_SOURCE_ROOT"])
    paths = sorted(set((root / 'fla').rglob('*.py')) | set((root / 'profile').glob('*.py')) |
                   {p for p in (root / 'setup.py', root / 'pyproject.toml') if p.exists()})
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode() + b'\0' + path.read_bytes() + b'\0')
    return dict(source_sha256=digest.hexdigest(),
                git_commit=command_output(['git', '-C', str(root), 'rev-parse', 'HEAD']),
                git_status=command_output(['git', '-C', str(root), 'status', '--porcelain']),
                source_patch=command_output(['git', '-C', str(root), 'diff', 'HEAD', '--', 'fla', 'profile/*.py', 'setup.py', 'pyproject.toml']))


def environment_details():
    return dict(cpu=command_output(['lscpu']), cpu_affinity=sorted(os.sched_getaffinity(0)),
                platform=platform.platform(), torch_threads=torch.get_num_threads(),
                torch_interop_threads=torch.get_num_interop_threads(),
                thread_environment={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
                nvidia_smi=command_output(['nvidia-smi']),
                gpu_telemetry=command_output(['nvidia-smi', '--query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used', '--format=csv']))
