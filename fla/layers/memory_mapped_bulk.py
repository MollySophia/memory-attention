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


class _CallerStreamTicket(_BulkTicket):
    def __init__(self, owner, stream):
        super().__init__(owner)
        self.stream = stream

    def acquire(self, offset):
        if offset == 0:
            current = torch.cuda.current_stream(self.owner.device)
            if current != self.stream:
                current.wait_event(self.owner.copied)
        return self.owner.values[offset]


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
        gather_stream = self.copy_stream
        gather_recorded = False
        try:
            if ids.device.type == "cpu" and (bool((ids < 0).any()) or bool((ids >= self.weights.shape[0]).any())):
                raise IndexError("memory table index out of range")
            ids_gpu = ids.to(self.device, non_blocking=True).reshape(-1).contiguous()
            if self.batch * self.seq_len == 1:
                gather_stream = torch.cuda.current_stream(self.device)
                if self.started:
                    gather_stream.wait_event(self.consumed)
                width = self.layers * self.dim
                _mapped_bulk_gather[(1, triton.cdiv(width, 512))](
                    self.weights, ids_gpu, self.gpu, width, self.weights.shape[0], 512, num_warps=4)
                self.copied.record(gather_stream)
                ticket = _CallerStreamTicket(self, gather_stream)
            else:
                self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.device(self.device), torch.cuda.stream(self.copy_stream):
                    if self.started:
                        self.copy_stream.wait_event(self.consumed)
                    width = self.layers * self.dim
                    _mapped_bulk_gather[(self.batch * self.seq_len, triton.cdiv(width, 512))](
                        self.weights, ids_gpu, self.gpu, width, self.weights.shape[0], 512, num_warps=4)
                    self.copied.record(self.copy_stream)
                ticket = _BulkTicket(self)
            gather_recorded = True
            for layer in range(self.layers):
                if layer == 0 or layer == self.layers - 1:
                    handle = PendingM(ticket, layer)
                    consume(layer, handle)
                    if not handle.released:
                        raise RuntimeError("consumer did not release M")
                else:
                    consume(layer, self.values[layer])
            # Preserve fresh CPU-table mutation and IDs lifetime semantics.
            if self.batch * self.seq_len == 1:
                self.copied.synchronize()
            else:
                self.copy_stream.synchronize()
            self.started = True
        except BaseException:
            self.broken = True
            if gather_recorded and self.batch * self.seq_len == 1:
                self.copied.synchronize()
            else:
                gather_stream.synchronize()
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
