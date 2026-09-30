"""Benchmark MemoryForCausalLM with the memory table resident vs CPU-offloaded.

Mirrors bmk.py's default configuration (24 layers, hidden 2048, 32 heads,
batch 8, seq/context 2048, BF16) so the numbers can be compared directly with
bmk.py's `ma_gpu` and `ma_offload` rows.

Timing scope matches bmk.py: input embedding, all blocks, final RMSNorm and
LM head. No loss, backward, optimizer or sampling.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from fla.models.memory.configuration_memory import MemoryConfig
from fla.models.memory.modeling_memory import MemoryForCausalLM


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--context-len", type=int, default=2048)
    p.add_argument("--hidden-size", type=int, default=2048)
    p.add_argument("--num-heads", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=24)
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--group-size", type=int, default=1)
    p.add_argument("--prefetch-depth", type=int, default=4)
    p.add_argument("--policy", choices=["auto", "pipeline", "bulk"], default="auto")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def build(args):
    torch.manual_seed(1234)
    config = MemoryConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_heads,
        vocab_size=args.vocab_size,
        use_cache=False,
        fuse_norm=False,
        memory_offload_group_size=args.group_size,
        memory_offload_prefetch_depth=args.prefetch_depth,
        memory_offload_policy=args.policy,
    )
    return MemoryForCausalLM(config).to(args.device, dtype=torch.bfloat16).eval()


def param_counts(model, offloaded):
    """GPU-resident parameters, with the table counted only when resident."""
    tensors = [p for p in model.parameters()]
    if not offloaded and model.model.memory_table is not None:
        tensors.append(model.model.memory_table)
    gpu = sum(t.numel() for t in tensors if t.device.type == "cuda")
    total = sum(t.numel() for t in tensors)
    return gpu, total


@torch.inference_mode()
def time_forward(model, ids, warmup, repeats):
    for _ in range(warmup):
        model(input_ids=ids, use_cache=False)
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        model(input_ids=ids, use_cache=False)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return samples


@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device(args.device)
    model = build(args)
    ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
    ids_cpu = ids.to("cpu")

    rows = []
    for label, offload in [("ma_gpu", False), ("ma_offload", True)]:
        model.close_memory_offload()
        torch.cuda.empty_cache()
        if offload:
            model.enable_memory_offload(device=device, dtype=torch.bfloat16, fold_norm=True)

        policy = "none"
        if offload:
            model.set_offload_offloader(args.batch_size, args.seq_len)
            policy = model.model.memory_offloader.policy

        gpu_params, total_params = param_counts(model, offload)
        table_mib = 0.0
        if model.model.memory_table is not None:
            t = model.model.memory_table
            table_mib = t.numel() * t.element_size() / 2**20

        # The offload producer gathers from CPU ids, so feed them on CPU.
        feed = ids_cpu if offload else ids
        samples = time_forward(model, feed, args.warmup, args.repeats)
        rows.append(dict(
            variant=label,
            policy=policy,
            mean_ms=statistics.mean(samples),
            median_ms=statistics.median(samples),
            min_ms=min(samples),
            max_ms=max(samples),
            gpu_params_m=gpu_params / 1e6,
            total_params_m=total_params / 1e6,
            table_mib=table_mib,
        ))

        print(f"{label}: {statistics.mean(samples):.3f} ms "
              f"({min(samples):.3f}..{max(samples):.3f}) policy={policy} "
              f"table={table_mib:.0f} MiB", flush=True)
        # Keep the CPU copy alive across both measurements.
        if not offload:
            pass

    base = rows[0]["mean_ms"]
    print()
    print(f"{'variant':<12} {'ms':>9} {'rel':>7} {'GPU params M':>13} {'table MiB':>10}")
    for r in rows:
        print(f"{r['variant']:<12} {r['mean_ms']:>9.3f} {r['mean_ms'] / base:>6.2f}x "
              f"{r['gpu_params_m']:>13.1f} {r['table_mib']:>10.0f}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(dict(config=vars(args), results=rows), indent=2, default=str))
        print(f"\nSaved {args.json}")

    model.close_memory_offload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
