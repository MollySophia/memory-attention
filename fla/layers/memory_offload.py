# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
"""CPU offload for the Memory Attention table.

The memory table ``m_proj`` is an ``nn.Embedding`` of shape
``[vocab_size, num_kv_heads * head_dim]`` per layer. At typical LM settings that
is the single largest tensor in the model, so it dominates GPU residency.

This module streams those rows from pinned host memory to the device on demand.
The design follows two rules that keep the transfer off the critical path:

* CPU readiness and CUDA completion are tracked separately. A ticket signals
  "the H2D copy is submitted" long before "the copy finished", so the block can
  keep queueing Q/K kernels instead of stalling on the producer.
* The transferred buffer is safe to overwrite as soon as ``k + m`` has been
  *enqueued* on the compute stream, because the copy already happened before
  that point. Releasing a slot therefore only needs an event record, not a
  device synchronize.

Everything here is inference-only. Training needs autograd through the table and
is intentionally unsupported.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import torch


class GroupTicket:
    """One layer-group transfer generation.

    Covers ``group`` consecutive layers that share a single gather + H2D. The
    caller acquires the handle per layer (in order) and releases it once that
    layer's ``k + m`` has been enqueued.
    """

    def __init__(self, owner: MemoryTableOffloader, slot: dict, start: int) -> None:
        self.owner = owner
        self.slot = slot
        self.start = start
        self.count = min(owner.group, owner.layers - start)
        self.ready = threading.Event()
        self.released = threading.Event()
        self.error: BaseException | None = None
        self.waited = False
        self.release_count = 0

    def acquire(self, offset: int) -> torch.Tensor:
        # CPU must have submitted the copy, then we wait for the GPU side of it.
        self.ready.wait()
        if self.error is not None:
            raise RuntimeError("memory table prefetch failed") from self.error
        if not self.waited:
            torch.cuda.current_stream(self.owner.device).wait_event(self.slot["copied"])
            self.waited = True
        values = self.owner._view(self.slot["gpu"], self.start)
        return values[:, offset].view(self.owner.batch, self.owner.seq_len, self.owner.dim)

    def release(self, offset: int) -> None:
        if offset != self.release_count:
            raise RuntimeError("memory table slices must be consumed in layer order")
        self.release_count += 1
        if self.release_count == self.count:
            # All reads of M are inside k + m. Later attention/MLP work uses the
            # newly allocated V instead, so the slot is free after that kernel.
            self.slot["consumed"].record(torch.cuda.current_stream(self.owner.device))
            self.released.set()


class PendingM:
    """Deferred dependency on a :class:`GroupTicket` slice."""

    def __init__(self, ticket: GroupTicket, offset: int) -> None:
        self.ticket = ticket
        self.offset = offset
        self.acquired = False
        self.released = False

    def acquire(self) -> torch.Tensor:
        if self.acquired:
            raise RuntimeError("M handle acquired twice")
        value = self.ticket.acquire(self.offset)
        self.acquired = True
        return value

    def release(self) -> None:
        if not self.acquired or self.released:
            raise RuntimeError("M handle must be acquired once before release")
        self.ticket.release(self.offset)
        self.released = True


class MemoryTableOffloader:
    """Prefetching producer: one CPU task per forward, bounded lookahead slots.

    The producer gathers the needed rows on a worker thread and submits the
    H2D on a dedicated copy stream, independently of block execution. Layers
    resolve their slice through :class:`PendingM` at the point of use.

    Args:
        weights: CPU tensor of shape ``[vocab_size, num_layers, dim]``, detached.
        batch: Batch size the buffers are preallocated for.
        seq_len: Sequence length the buffers are preallocated for.
        group_size: Layers transferred per gather + H2D.
        device: CUDA device.
        prefetch_depth: Number of buffered groups. Clamped to the group count.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        batch: int,
        seq_len: int,
        group_size: int = 1,
        device: torch.device | str = "cuda:0",
        prefetch_depth: int = 4,
    ) -> None:
        if weights.device.type != "cpu" or weights.ndim != 3:
            raise ValueError("weights must be a CPU tensor of shape [vocab, layers, dim]")
        if min(batch, seq_len, group_size, prefetch_depth) < 1:
            raise ValueError("shapes and prefetch depth must be positive")
        self.weights = weights
        self.batch = batch
        self.seq_len = seq_len
        self.tokens = batch * seq_len
        self.layers, self.dim = weights.shape[1:]
        self.group = min(group_size, self.layers)
        self.groups = list(range(0, self.layers, self.group))
        self.device = torch.device(device)
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ma-prefetch")
        self.lock = threading.Lock()
        self.closed = False
        self.broken = False
        capacity = self.tokens * self.group * self.dim
        self.slots = [
            dict(
                host=torch.empty(capacity, dtype=weights.dtype, pin_memory=True),
                gpu=torch.empty(capacity, dtype=weights.dtype, device=self.device),
                copied=torch.cuda.Event(),
                consumed=torch.cuda.Event(),
                previous=None,
            )
            for _ in range(min(prefetch_depth, len(self.groups)))
        ]
        for slot in self.slots:
            slot["host_alt"] = torch.empty(capacity, dtype=weights.dtype, pin_memory=True)
            slot["host_events"] = (slot["copied"], torch.cuda.Event())
            slot["host_used"] = [False, False]
            slot["host_turn"] = 0

    @property
    def policy(self) -> str:
        return "pipeline"

    @property
    def group_size(self) -> int:
        return self.group

    @property
    def extra_host_buffers(self):
        return [slot["host_alt"] for slot in self.slots]

    def _view(self, flat: torch.Tensor, start: int) -> torch.Tensor:
        count = min(self.group, self.layers - start)
        return flat[: self.tokens * count * self.dim].view(self.tokens, count, self.dim)

    @torch.inference_mode()
    def _produce(self, ids: torch.Tensor, tickets: list[GroupTicket], entry, cancel) -> None:
        try:
            # Current device/stream are thread-local, so select them explicitly.
            with torch.cuda.device(self.device), torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_event(entry)
                for ticket in tickets:
                    slot = ticket.slot
                    previous = slot["previous"]
                    if previous is not None:
                        while not previous.released.wait(timeout=0.05):
                            if cancel.is_set():
                                return
                        if cancel.is_set():
                            return
                    if cancel.is_set():
                        return
                    host_index = slot["host_turn"]
                    host = slot["host"] if host_index == 0 else slot["host_alt"]
                    host_event = slot["host_events"][host_index]
                    if slot["host_used"][host_index]:
                        # Only this host allocation's last DMA constrains CPU
                        # overwrite; the other host buffer may still be read.
                        host_event.synchronize()
                    source = self.weights[:, ticket.start: ticket.start + ticket.count]
                    torch.index_select(source, 0, ids, out=self._view(host, ticket.start))
                    if previous is not None:
                        # Snapshot before the event can be re-recorded.
                        self.copy_stream.wait_event(slot["consumed"])
                    self._view(slot["gpu"], ticket.start).copy_(
                        self._view(host, ticket.start), non_blocking=True
                    )
                    host_event.record(self.copy_stream)
                    # The previous ticket has fully released before changing
                    # this pointer, and readiness publishes it to this ticket.
                    slot["copied"] = host_event
                    slot["host_used"][host_index] = True
                    slot["host_turn"] = 1 - host_index
                    slot["previous"] = ticket
                    # "Ready" means submitted, not completed.
                    ticket.ready.set()
        except BaseException as exc:
            for ticket in tickets:
                if not ticket.ready.is_set():
                    ticket.error = exc
                    ticket.ready.set()
            raise

    @torch.inference_mode()
    def forward(self, ids_cpu: torch.Tensor, consume) -> None:
        """Drive ``consume(layer_index, handle_or_tensor)`` for every layer."""
        if self.closed or self.broken:
            raise RuntimeError("offloader is closed or failed; construct a new one")
        if ids_cpu.device.type != "cpu" or ids_cpu.dtype != torch.long:
            raise ValueError("ids must be CPU int64")
        if tuple(ids_cpu.shape) != (self.batch, self.seq_len):
            raise ValueError("IDs do not match the preallocated shapes")
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("concurrent forwards on one offloader are unsupported")
        cancel = threading.Event()
        future = None
        try:
            ids = ids_cpu.reshape(-1).contiguous()
            entry = torch.cuda.Event()
            entry.record(torch.cuda.current_stream(self.device))
            tickets = [
                GroupTicket(self, self.slots[i % len(self.slots)], start)
                for i, start in enumerate(self.groups)
            ]
            future = self.worker.submit(self._produce, ids, tickets, entry, cancel)
            for ticket in tickets:
                for offset in range(ticket.count):
                    handle = PendingM(ticket, offset)
                    consume(ticket.start + offset, handle)
                    if not handle.released:
                        raise RuntimeError("consumer did not acquire/release its PendingM")
            future.result()
        except BaseException:
            self.broken = True
            cancel.set()
            if future is not None:
                try:
                    future.result()
                except BaseException:
                    pass
            raise
        finally:
            self.lock.release()

    def close(self) -> None:
        self.closed = True
        self.worker.shutdown(wait=True)
        torch.cuda.synchronize(self.device)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class _BulkTicket:
    """Bulk path only needs a handle at the first and last layer."""

    def __init__(self, owner: "BulkMemoryTableOffloader") -> None:
        self.owner = owner

    def acquire(self, offset: int) -> torch.Tensor:
        if offset == 0:
            torch.cuda.current_stream(self.owner.device).wait_event(self.owner.copied)
        return self.owner.values[offset]

    def release(self, offset: int) -> None:
        if offset == self.owner.layers - 1:
            self.owner.consumed.record(torch.cuda.current_stream(self.owner.device))


class BulkMemoryTableOffloader:
    """Small-input path: one CPU gather and one H2D, no producer thread.

    Trades the pipeline's overlap for much less per-layer coordination, which
    wins when the transfer itself is small (single-token decode).
    """

    policy = "bulk"

    def __init__(self, weights: torch.Tensor, batch: int, seq_len: int, device) -> None:
        self.weights = weights
        self.batch = batch
        self.seq_len = seq_len
        self.layers, self.dim = weights.shape[1:]
        self.group = self.layers
        self.device = torch.device(device)
        shape = (batch * seq_len, self.layers, self.dim)
        self.host = torch.empty(shape, dtype=weights.dtype, pin_memory=True)
        self.gpu = torch.empty(shape, dtype=weights.dtype, device=self.device)
        self.values = [self.gpu[:, i].view(batch, seq_len, self.dim) for i in range(self.layers)]
        self.copy_stream = torch.cuda.Stream(device=self.device)
        # Establish allocation-stream ordering once, before async use.
        self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
        self.copied = torch.cuda.Event()
        self.consumed = torch.cuda.Event()
        self.slots = [{"host": self.host, "gpu": self.gpu}]
        self.started = False
        self.closed = False
        self.broken = False
        self.lock = threading.Lock()

    @property
    def group_size(self) -> int:
        return self.group

    @torch.inference_mode()
    def forward(self, ids_cpu: torch.Tensor, consume) -> None:
        if self.closed or self.broken:
            raise RuntimeError("offloader is closed or failed")
        if ids_cpu.device.type != "cpu" or ids_cpu.dtype != torch.long:
            raise ValueError("ids must be CPU int64")
        if tuple(ids_cpu.shape) != (self.batch, self.seq_len):
            raise ValueError("IDs do not match the preallocated shapes")
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("concurrent forwards are unsupported")
        try:
            if self.started and not self.copied.query():
                self.copied.synchronize()
            # No caching: every call performs a real lookup.
            torch.index_select(self.weights, 0, ids_cpu.reshape(-1), out=self.host)
            with torch.cuda.device(self.device), torch.cuda.stream(self.copy_stream):
                if self.started:
                    self.copy_stream.wait_event(self.consumed)
                self.gpu.copy_(self.host, non_blocking=True)
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
            self.started = True
        except BaseException:
            self.broken = True
            raise
        finally:
            self.lock.release()

    def close(self) -> None:
        self.closed = True
        torch.cuda.synchronize(self.device)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def fold_memory_table(
    norm,
    raw: torch.Tensor,
    chunk_size: int = 1024,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Fold head-wise RMSNorm (and its affine weight) into raw table rows.

    ``raw`` is ``[vocab, num_kv_heads, head_dim]``. The result is pre-normalized,
    so the attention layer can skip the per-token norm at runtime. Chunked to
    keep peak memory bounded for large vocabularies.
    """
    vocab = raw.shape[0]
    folded = torch.empty(vocab, raw.shape[1] * raw.shape[2], dtype=out_dtype, device=raw.device)
    for start in range(0, vocab, chunk_size):
        count = min(chunk_size, vocab - start)
        block = norm(raw[start:start + count])
        folded[start:start + count] = block.reshape(count, -1).to(out_dtype)
    return folded


def build_cpu_table(
    m_projs: list,
    norms: list,
    head_dim: int,
    chunk_size: int = 1024,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Stack per-layer ``m_proj`` weights into one CPU table with folded norm.

    Args:
        m_projs: Per-layer ``nn.Embedding`` (or equivalent) modules.
        norms: Matching per-layer ``m_norm`` modules.
        head_dim: Per-head dimension, so each layer's row splits into
            ``[vocab, num_kv_heads, head_dim]`` before normalization.
        chunk_size: Rows folded per step.

    Returns:
        CPU tensor of shape ``[vocab_size, num_layers, kv_dim]``.
    """
    if len(m_projs) != len(norms) or not m_projs:
        raise ValueError("m_projs and norms must be non-empty and the same length")
    if head_dim < 1:
        raise ValueError("head_dim must be positive")
    vocab, kv_dim = m_projs[0].weight.shape
    if kv_dim % head_dim:
        raise ValueError(f"kv_dim {kv_dim} is not divisible by head_dim {head_dim}")
    table = torch.empty(vocab, len(m_projs), kv_dim, dtype=dtype, device="cpu")
    for index, (m_proj, norm) in enumerate(zip(m_projs, norms)):
        # RMSNorm keeps its own dtype; match the module rather than forcing one.
        raw = m_proj.weight.detach().reshape(vocab, kv_dim // head_dim, head_dim).to(norm.weight.dtype)
        table[:, index] = fold_memory_table(norm, raw, chunk_size, dtype).cpu()
    return table
