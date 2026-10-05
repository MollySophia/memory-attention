"""All-layer GPU gather from a pinned host table for small bulk inputs."""
from __future__ import annotations

import threading

import torch
import triton
import triton.language as tl

from fla.layers.memory_offload import PendingM, _BulkTicket


@triton.jit
def _mapped_bulk_gather(table, ids, output, WIDTH: tl.constexpr,
                        VOCAB: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    token = tl.load(ids + row)
    # GPU embedding checks IDs before the entry dependency. Mask additionally
    # so an invalid input cannot cause an out-of-bounds mapped-host read.
    values = tl.load(table + token * WIDTH + column,
                     mask=(column < WIDTH) & (token >= 0) & (token < VOCAB), other=0)
    tl.store(output + row * WIDTH + column, values, mask=column < WIDTH)


class MappedBulkMemoryTableOffloader:
    policy = "bulk"
    accepts_gpu_ids = True

    def __init__(self, weights, batch, seq_len, device):
        if weights.device.type != "cpu" or weights.ndim != 3 or not weights.is_pinned() or not weights.is_contiguous():
            raise ValueError("mapped table must be contiguous pinned CPU [vocab,layers,dim]")
        if min(batch, seq_len) < 1:
            raise ValueError("shapes must be positive")
        self.weights = weights
        self.batch, self.seq_len = batch, seq_len
        self.layers, self.dim = weights.shape[1:]
        self.group = self.layers
        self.device = torch.device(device)
        self.gpu = torch.empty((batch * seq_len, self.layers, self.dim),
                               dtype=weights.dtype, device=self.device)
        self.values = [self.gpu[:, i].view(batch, seq_len, self.dim) for i in range(self.layers)]
        self.host = torch.empty(0, dtype=weights.dtype, pin_memory=True)
        self.slots = [{"host": self.host, "gpu": self.gpu}]
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
        self.copied = torch.cuda.Event()
        self.consumed = torch.cuda.Event()
        self.started = self.closed = self.broken = False
        self.lock = threading.Lock()
        # Shape, dtype, device and table/output storage are fixed by this
        # owner. Contiguous int64 IDs can still have different pointer
        # alignment (for example an odd-offset slice), so specialize each.
        self._gather_runners = {}

    @property
    def group_size(self):
        return self.group

    @torch.inference_mode()
    def forward(self, ids, consume):
        if self.closed or self.broken:
            raise RuntimeError("offloader is closed or failed")
        if ids.dtype != torch.long or tuple(ids.shape) != (self.batch, self.seq_len):
            raise ValueError("IDs must be int64 with the preallocated shape")
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("concurrent forwards are unsupported")
        try:
            if ids.device.type == "cpu" and (bool((ids < 0).any()) or bool((ids >= self.weights.shape[0]).any())):
                raise IndexError("memory table index out of range")
            ids_gpu = ids.to(self.device, non_blocking=True).reshape(-1).contiguous()
            self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.device(self.device):
                if self.started:
                    self.copy_stream.wait_event(self.consumed)
                width = self.layers * self.dim
                alignment = ids_gpu.data_ptr() % 16
                runner = self._gather_runners.get(alignment)
                if runner is None:
                    grid = (self.batch * self.seq_len, triton.cdiv(width, 512), 1)
                    with torch.cuda.stream(self.copy_stream):
                        compiled = _mapped_bulk_gather[grid](
                            self.weights, ids_gpu, self.gpu, width,
                            self.weights.shape[0], 512, num_warps=4)
                    self._gather_runners[alignment] = compiled[grid]
                else:
                    # The compiled runner preserves Triton launch hooks. It
                    # reads fresh IDs/table values on the original copy stream.
                    runner(self.weights, ids_gpu, self.gpu, width,
                           self.weights.shape[0], 512,
                           stream=self.copy_stream.cuda_stream)
                self.copied.record(self.copy_stream)
            ticket = _BulkTicket(self)
            for layer in range(self.layers):
                if layer == 0 or layer == self.layers - 1:
                    handle = PendingM(ticket, layer)
                    consume(layer, handle)
                    if not handle.released:
                        raise RuntimeError("consumer did not release M")
                else:
                    consume(layer, self.values[layer])
            # Preserve fresh CPU-table mutation and IDs lifetime semantics.
            self.copy_stream.synchronize()
            self.started = True
        except BaseException:
            self.broken = True
            self.copy_stream.synchronize()
            raise
        finally:
            self.lock.release()

    def close(self):
        self.closed = True
        torch.cuda.synchronize(self.device)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
