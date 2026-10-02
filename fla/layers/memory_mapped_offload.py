"""Bounded GPU gathers from a page-locked, host-resident memory table."""
from __future__ import annotations
import threading
import torch
import triton
import triton.language as tl
from fla.layers.memory_offload import PendingM


@triton.jit(do_not_specialize=['start'])
def _mapped_gather(table, ids, output, start, TOKENS: tl.constexpr,
                   DIM: tl.constexpr, LAYERS: tl.constexpr, COUNT: tl.constexpr,
                   VOCAB: tl.constexpr, BLOCK: tl.constexpr):
    tiles = tl.cdiv(COUNT * DIM, BLOCK)
    for tile in range(tl.program_id(0), TOKENS * tiles, tl.num_programs(0)):
        row = tile // tiles
        column = (tile % tiles) * BLOCK + tl.arange(0, BLOCK)
        token = tl.load(ids + row)
        # The model embedding validates GPU IDs before this stream's entry
        # event. Mask anyway so invalid input can never read beyond host memory.
        value = tl.load(table + token * (LAYERS * DIM) + start * DIM + column,
                        mask=(column < COUNT * DIM) & (token >= 0) & (token < VOCAB), other=0)
        tl.store(output + row * (COUNT * DIM) + column, value, mask=column < COUNT * DIM)


class _MappedTicket:
    def __init__(self, owner, slot, start):
        self.owner, self.slot, self.start = owner, slot, start
        self.count = min(owner.group, owner.layers - start)
        self.waited = False
        self.release_count = 0

    def acquire(self, offset):
        if not self.waited:
            torch.cuda.current_stream(self.owner.device).wait_event(self.slot['copied'])
            self.waited = True
        return self.owner._view(self.slot['gpu'], self.start)[:, offset].view(
            self.owner.batch, self.owner.seq_len, self.owner.dim)

    def release(self, offset):
        if offset != self.release_count:
            raise RuntimeError('memory table slices must be consumed in layer order')
        self.release_count += 1
        if self.release_count == self.count:
            self.slot['consumed'].record(torch.cuda.current_stream(self.owner.device))


class MappedMemoryTableOffloader:
    """GPU input IDs select fresh rows from pinned CPU storage on every call.

    GPU IDs must be valid embedding indices; the model embedding checks them.
    CPU IDs are checked here. The final read-stream synchronization ensures the
    caller can update the CPU table after forward returns, even while model
    computation that no longer reads the table is still pending.
    """
    policy = 'pipeline'
    accepts_gpu_ids = True

    def __init__(self, weights, batch, seq_len, group_size=1, device='cuda:0', prefetch_depth=4):
        if weights.device.type != 'cpu' or weights.ndim != 3 or not weights.is_pinned() or not weights.is_contiguous():
            raise ValueError('mapped table must be contiguous pinned CPU [vocab,layers,dim]')
        if min(batch, seq_len, group_size, prefetch_depth) < 1:
            raise ValueError('shapes and prefetch depth must be positive')
        self.weights = weights
        self.batch, self.seq_len = batch, seq_len
        self.tokens = batch * seq_len
        self.layers, self.dim = weights.shape[1:]
        self.group = min(group_size, self.layers)
        self.groups = list(range(0, self.layers, self.group))
        self.device = torch.device(device)
        self.copy_stream = torch.cuda.Stream(device=self.device)
        capacity = self.tokens * self.group * self.dim
        self.slots = [dict(host=torch.empty(0, dtype=weights.dtype, pin_memory=True),
                           gpu=torch.empty(capacity, dtype=weights.dtype, device=self.device),
                           copied=torch.cuda.Event(), consumed=torch.cuda.Event(), used=False)
                      for _ in range(min(prefetch_depth, len(self.groups)))]
        self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
        self.lock = threading.Lock()
        self.closed = self.broken = False

    @property
    def group_size(self):
        return self.group

    def _view(self, flat, start):
        count = min(self.group, self.layers - start)
        return flat[:self.tokens * count * self.dim].view(self.tokens, count, self.dim)

    def _submit(self, ticket, ids):
        slot = ticket.slot
        with torch.cuda.device(self.device), torch.cuda.stream(self.copy_stream):
            if slot['used']:
                self.copy_stream.wait_event(slot['consumed'])
            grid = min(64, self.tokens * triton.cdiv(ticket.count * self.dim, 512))
            _mapped_gather[(grid,)](self.weights, ids, slot['gpu'], ticket.start,
                                    self.tokens, self.dim, self.layers, ticket.count,
                                    self.weights.shape[0], 512, num_warps=4)
            slot['copied'].record(self.copy_stream)
            slot['used'] = True

    @torch.inference_mode()
    def forward(self, ids, consume):
        if self.closed or self.broken:
            raise RuntimeError('offloader is closed or failed')
        if ids.dtype != torch.long or tuple(ids.shape) != (self.batch, self.seq_len):
            raise ValueError('IDs must be int64 with the preallocated shape')
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('concurrent forwards are unsupported')
        try:
            if ids.device.type == 'cpu' and (bool((ids < 0).any()) or bool((ids >= self.weights.shape[0]).any())):
                raise IndexError('memory table index out of range')
            ids_gpu = ids.to(self.device, non_blocking=True).reshape(-1).contiguous()
            entry = torch.cuda.Event()
            entry.record(torch.cuda.current_stream(self.device))
            self.copy_stream.wait_event(entry)
            tickets = [_MappedTicket(self, self.slots[i % len(self.slots)], start)
                       for i, start in enumerate(self.groups)]
            for ticket in tickets[:len(self.slots)]:
                self._submit(ticket, ids_gpu)
            for index, ticket in enumerate(tickets):
                for offset in range(ticket.count):
                    handle = PendingM(ticket, offset)
                    consume(ticket.start + offset, handle)
                    if not handle.released:
                        raise RuntimeError('consumer did not acquire/release its PendingM')
                following = index + len(self.slots)
                if following < len(tickets):
                    self._submit(tickets[following], ids_gpu)
            # CUDA kernels directly read host storage. Preserve the CPU-table
            # lifetime/update contract and ids_gpu lifetime on this stream.
            self.copy_stream.synchronize()
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
