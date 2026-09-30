"""Head-to-head: MemoryForCausalLM vs bmk.py, same configuration.

Runs the three table placements that both implementations support and checks
they agree numerically before comparing latency:

  ma_gpu_folded   table resident on GPU, m_norm folded in   <- bmk.py `ma_gpu`
  ma_gpu_unfolded table resident on GPU, norm per token    <- original fla path
  ma_offload      table streamed from pinned host memory   <- bmk.py `ma_offload`

Comparing a *folded* resident table against the offloaded one is what makes
the offload overhead meaningful: both already skip the per-token norm, so the
delta is the transfer and its coordination, not the folding win.

Every variant is timed with the same warmup/repeats and the same timing scope
bmk.py uses (embedding, all blocks, final norm, LM head).
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
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--rounds", type=int, default=5)
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
        memory_offload_policy="auto",
    )
    return MemoryForCausalLM(config).to(args.device, dtype=torch.bfloat16).eval()


@torch.inference_mode()
def time_forward(model, feed, warmup, repeats, rounds):
    for _ in range(warmup):
        model(input_ids=feed, use_cache=False)
    torch.cuda.synchronize()

    # Per-round means, then mean of rounds -- same shape as bmk.py's rows.
    round_means = []
    for _ in range(rounds):
        samples = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(input_ids=feed, use_cache=False)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1e3)
        round_means.append(statistics.mean(samples))
    return round_means


@torch.inference_mode()
def run_variant(model, name, args, ids, ids_cpu, offload, fold, reference):
    model.close_memory_offload()
    torch.cuda.empty_cache()
    model.model._restore_unfolded_table(args.device, torch.bfloat16)

    policy = "none"
    if fold:
        model.fold_memory_table_on_gpu(dtype=torch.bfloat16)
    feed = ids
    if offload:
        model.enable_memory_offload(device=args.device, dtype=torch.bfloat16, fold_norm=True)
        model.set_offload_offloader(args.batch_size, args.seq_len)
        policy = model.model.memory_offloader.policy
        feed = ids_cpu

    logits = model(input_ids=feed, use_cache=False).logits.float()
    if reference is None:
        ref = logits.clone()
        diff = 0.0
        agree = 100.0
    else:
        diff = (logits - reference).abs().max().item()
        agree = (logits.argmax(-1) == reference.argmax(-1)).float().mean().item() * 100

    gpu_m = sum(p.numel() for p in model.parameters() if p.device.type == "cuda") / 1e6
    rounds = time_forward(model, feed, args.warmup, args.repeats, args.rounds)

    return dict(
        variant=name,
        policy=policy,
        mean_ms=statistics.mean(rounds),
        best_round_ms=min(rounds),
        worst_round_ms=max(rounds),
        gpu_params_m=gpu_m,
        max_logit_diff=diff,
        argmax_agreement=agree,
        _logits=logits,
    )


@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device(args.device)
    model = build(args)
    ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
    ids_cpu = ids.to("cpu")

    plan = [
        ("ma_gpu_folded", False, True),
        ("ma_offload", True, True),
        ("ma_gpu_unfolded", False, False),
    ]
    rows = []
    # Reference is the folded resident table: bmk.py's `ma_gpu` and the
    # offloaded path must both reproduce it exactly, since both consume the
    # same folded table. The unfolded path differs only by BF16 rounding in
    # the table, which is reported separately.
    reference = None
    for name, offload, fold in plan:
        row = run_variant(model, name, args, ids, ids_cpu, offload, fold, reference)
        if reference is None:
            reference = row.pop("_logits")
        rows.append(row)
        print(f"{name:<16} {row['mean_ms']:>8.3f} ms  policy={row['policy']:<9} "
              f"gpu={row['gpu_params_m']:.1f}M  maxdiff={row['max_logit_diff']:.6f}", flush=True)
    for r in rows:
        r.pop("_logits", None)

    base = rows[0]["mean_ms"]  # ma_gpu_folded, the honest offload reference
    print()
    print(f"{'variant':<16} {'ms':>9} {'rel(folded)':>12} {'GPU params M':>13} {'max diff':>10}")
    for r in rows:
        print(f"{r['variant']:<16} {r['mean_ms']:>9.3f} {r['mean_ms'] / base:>11.2f}x "
              f"{r['gpu_params_m']:>13.1f} {r['max_logit_diff']:>10.6f}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(dict(config=vars(args), results=rows), indent=2, default=str))
        print(f"\nSaved {args.json}")

    model.close_memory_offload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
