"""Storage-owned write-combined CUDA host buffers for streaming CPU writes."""
from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path
import weakref

import torch


@lru_cache(maxsize=1)
def _runtime():
    major = torch.version.cuda.split('.')[0]
    packages = Path(torch.__file__).resolve().parents[1]
    candidates = [packages / f'nvidia/cu{major}/lib/libcudart.so.{major}',
                  packages / f'nvidia/cuda_runtime/lib/libcudart.so.{major}']
    library = next((str(path) for path in candidates if path.exists()), f'libcudart.so.{major}')
    runtime = ctypes.CDLL(library)
    runtime.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
    runtime.cudaHostAlloc.restype = ctypes.c_int
    runtime.cudaFreeHost.argtypes = [ctypes.c_void_p]
    runtime.cudaFreeHost.restype = ctypes.c_int
    return runtime


def _release(runtime, pointer, device):
    # This storage is external to PyTorch's pinned caching allocator. Its event
    # tracking cannot protect us when the last tensor view is dropped mid-copy.
    with torch.cuda.device(device):
        torch.cuda.synchronize(device)
        status = runtime.cudaFreeHost(pointer)
    if status:
        raise RuntimeError(f'cudaFreeHost failed with CUDA status {status}')


def write_combined_empty(numel: int, dtype: torch.dtype, device) -> torch.Tensor:
    """Allocate a 1-D CPU tensor; storage retains the CUDA allocation owner."""
    if numel <= 0:
        raise ValueError('write-combined allocation size must be positive')
    device = torch.device(device)
    runtime = _runtime()
    size = numel * torch.empty((), dtype=dtype).element_size()
    pointer = ctypes.c_void_p()
    with torch.cuda.device(device):
        status = runtime.cudaHostAlloc(ctypes.byref(pointer), size, 0x04)
    if status:
        raise RuntimeError(f'cudaHostAllocWriteCombined failed with CUDA status {status}')
    buffer = (ctypes.c_ubyte * size).from_address(pointer.value)
    # torch.frombuffer holds a reference to its Python buffer owner for the
    # entire storage lifetime, including aliases after the original tensor dies.
    cleanup = weakref.finalize(buffer, _release, runtime, pointer, device)
    cleanup.atexit = False  # Process teardown releases CUDA resources itself.
    tensor = torch.frombuffer(buffer, dtype=dtype)
    if not tensor.is_pinned():
        raise RuntimeError('CUDA write-combined allocation was not recognized as pinned')
    return tensor
