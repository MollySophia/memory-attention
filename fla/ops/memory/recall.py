# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["ROWS"])
def _memory_recall_kernel(
    keys, ids, table, weight, output,
    ROWS, HEADS: tl.constexpr, DIM: tl.constexpr,
    VOCAB: tl.constexpr, EPS: tl.constexpr, FOLDED: tl.constexpr,
    HAS_WEIGHT: tl.constexpr, BLOCK_ROWS: tl.constexpr, BLOCK_DIM: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    dims = tl.arange(0, BLOCK_DIM)
    token = tl.load(ids + rows // HEADS, mask=rows < ROWS, other=0)
    valid = (token >= 0) & (token < VOCAB)
    offsets = (token[:, None] * HEADS + rows[:, None] % HEADS) * DIM + dims[None, :]
    mask = (rows[:, None] < ROWS) & (dims[None, :] < DIM)
    m = tl.load(table + offsets, mask=mask & valid[:, None], other=0).to(tl.float32)
    if not FOLDED:
        variance = tl.sum(m * m, axis=1) / DIM
        m = m * (1.0 / tl.sqrt(variance + EPS))[:, None]
        if HAS_WEIGHT:
            w = tl.load(weight + dims, mask=dims < DIM, other=0).to(tl.float32)
            m = m * w[None, :]
        # Preserve the stored RMSNorm result's rounding before the BF16/FP16
        # addition, matching the unfused implementation exactly.
        m = m.to(output.dtype.element_ty).to(tl.float32)
    k = tl.load(keys + rows[:, None] * DIM + dims[None, :], mask=mask, other=0).to(tl.float32)
    v = tl.where(valid[:, None], k + m, float('nan'))
    tl.store(output + rows[:, None] * DIM + dims[None, :], v, mask=mask)


def memory_recall(
    keys: torch.Tensor,
    ids: torch.Tensor,
    table: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    folded: bool = False,
) -> torch.Tensor:
    """Inference-only V reconstruction without materializing historical M.

    Inputs are contiguous GPU raw keys [batch, length, heads, dim], validated
    token IDs [batch, length] and an embedding table [vocab, heads * dim].
    """
    output = torch.empty_like(keys)
    heads, dim = keys.shape[-2:]
    rows = keys.numel() // dim
    _memory_recall_kernel[(triton.cdiv(rows, 16),)](
        keys, ids, table, weight, output,
        rows, heads, dim, table.shape[0], eps, folded, weight is not None,
        16, triton.next_power_of_2(dim), num_warps=4, enable_fp_fusion=False,
    )
    return output
