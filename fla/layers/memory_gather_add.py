"""Exact gather/add for deduplicated, folded offloaded memory rows."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gather_add(K, M, I, V, N: tl.constexpr, D: tl.constexpr,
                MS: tl.constexpr, MC: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = x < N * D
    row = x // D
    col = x % D
    index = tl.load(I + row, row < N, other=0)
    k = tl.load(K + x, valid, other=0).to(tl.float32)
    m = tl.load(M + index * MS + col * MC, valid, other=0).to(tl.float32)
    tl.store(V + x, k + m, valid)


def gather_add(keys: torch.Tensor, rows: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
    if (not keys.is_cuda or rows.device != keys.device or inverse.device != keys.device
            or keys.dtype != rows.dtype or inverse.dtype != torch.long):
        raise ValueError("gather/add requires matching CUDA values and int64 inverse IDs")
    if (not keys.is_contiguous() or rows.ndim != 2 or inverse.ndim != 1
            or not inverse.is_contiguous() or keys.numel() != inverse.numel() * rows.shape[1]):
        raise ValueError("gather/add received incompatible shapes or key layout")
    output = torch.empty_like(keys)
    _gather_add[(triton.cdiv(keys.numel(), 1024),)](
        keys, rows, inverse, output, inverse.numel(), rows.shape[1],
        rows.stride(0), rows.stride(1), 1024, num_warps=4)
    return output
