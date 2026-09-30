"""Decode benchmark: resident vs offloaded memory table, single-token steps.

Mirrors bmk.py's decode methodology: build a real prefilled prefix at a fixed
context length, then time one-token forwards that overwrite the same next-token
slot. Only the decode step is timed -- the prefix build is excluded, as is in
bmk.py.

This is the case where offload has a chance to win: a single token touches one
table row per layer, so the transfer is tiny and the bulk policy's lower
per-layer coordination should pay off. Whether it actually does is the
question this measures.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from fla.models.memory.configuration_memory import MemoryConfig
from fla.models.memory.modeling_memory import MemoryForCausalLM
from fla.models.utils import Cache


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=8)
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
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--id-pool", type=int, default=16)
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
        use_cache=True,
        fuse_norm=False,
        memory_offload_group_size=args.group_size,
        memory_offload_prefetch_depth=args.prefetch_depth,
        memory_offload_policy=args.policy,
    )
    return MemoryForCausalLM(config).to(args.device, dtype=torch.bfloat16).eval()


@torch.inference_mode()
def prepare_prefix(model, args, device):
    """Build a real KV cache of `context_len` tokens, once per variant.

    Returns the cache object from the prefill call: re-wrapping it with
    Cache.from_legacy_cache() would reset the decode path, so the returned
    cache is threaded through every subsequent step.
    """
    torch.manual_seed(99)
    prefix = torch.randint(
        0, args.vocab_size, (args.batch_size, args.context_len), device=device
    )
    # Only the prefix build; not timed.
    out = model(
        input_ids=prefix, past_key_values=Cache.from_legacy_cache(None), use_cache=True
    )
    return prefix, out.past_key_values


def rollback(cache, cache_len):
    """Trim the cache back to `cache_len` tokens.

    fla/models/utils.py:241 concatenates the new KV onto the existing state
    when no sliding window is set, so an un-trimmed decode loop grows the
    context every step (and eventually trips the 2D mask path in
    fla/layers/utils.py:164). Trimming keeps the attended context identical
    across repetitions, which is what makes the timings comparable.

    Handles both Cache layouts: `states` (legacy fla cache) and `layers`
    (the transformers-backed cache this environment resolves to).
    """
    if hasattr(cache, "states"):
        holders = cache.states
    else:
        holders = [layer.state for layer in cache.layers]
    for state in holders:
        if isinstance(state, dict) and state.get("attn_state") is not None:
            state["attn_state"] = [t[:, :cache_len] for t in state["attn_state"]]
    cache._seen_tokens = cache_len


def time_decode(model, args, cache, cache_len, step_ids, device):
    """Time one-token steps at a fixed context length."""
    def one(token_ids):
        rollback(cache, cache_len)
        return model(
            input_ids=token_ids,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        ).logits

    for i in range(args.warmup):
        one(step_ids[i % args.id_pool])
    torch.cuda.synchronize(device)

    round_means = []
    for _ in range(args.rounds):
        samples = []
        for i in range(args.repeats):
            token = step_ids[i % args.id_pool]
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            one(token)
            torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - start) * 1e3)
        round_means.append(statistics.mean(samples))
    return round_means


def run_variant(model, name, args, step_ids, device, fold, offload, reference):
    model.close_memory_offload()
    torch.cuda.empty_cache()
    model.model._restore_unfolded_table(device, torch.bfloat16)

    policy = "none"
    if fold:
        model.fold_memory_table_on_gpu(dtype=torch.bfloat16)
    if offload:
        model.enable_memory_offload(device=device, dtype=torch.bfloat16, fold_norm=True)
        policy = "auto"

    prefix, cache = prepare_prefix(model, args, device)
    rounds = time_decode(model, args, cache, args.context_len, step_ids, device)

    rollback(cache, args.context_len)
    logits = model(
        input_ids=step_ids[0], past_key_values=cache, use_cache=True, logits_to_keep=1
    ).logits.float()
    if reference is None:
        ref, diff, agree = logits.clone(), 0.0, 100.0
    else:
        diff = (logits - reference).abs().max().item()
        agree = (logits.argmax(-1) == reference.argmax(-1)).float().mean().item() * 100

    gpu_m = sum(p.numel() for p in model.parameters() if p.device.type == "cuda") / 1e6
    del cache, prefix
    return dict(
        variant=name, policy=policy,
        mean_ms=statistics.mean(rounds), best_ms=min(rounds), worst_ms=max(rounds),
        gpu_params_m=gpu_m, max_logit_diff=diff, argmax_agreement=agree,
    )


@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device(args.device)
    model = build(args)
    torch.manual_seed(7)
    # [id_pool, batch, 1] so each repetition is a full [batch, 1] forward.
    # Indexing a [batch, 1] tensor would yield a 1-D id and 1-D hidden states.
    step_ids = torch.randint(
        0, args.vocab_size, (args.id_pool, args.batch_size, 1), device=device
    )

    plan = [
        ("ma_gpu_folded", True, False),
        ("ma_offload", True, True),
        ("ma_gpu_unfolded", False, False),
    ]
    rows = []
    reference = None
    for name, fold, offload in plan:
        row = run_variant(model, name, args, step_ids, device, fold, offload, reference)
        if reference is None:
            # Re-run to capture baseline logits, so all rows see equal timing
            # conditions after the cache has been built once.
            model.close_memory_offload()
            model.model._restore_unfolded_table(device, torch.bfloat16)
            model.fold_memory_table_on_gpu(dtype=torch.bfloat16)
            _, cache = prepare_prefix(model, args, device)
            rollback(cache, args.context_len)
            reference = model(
                input_ids=step_ids[0], past_key_values=cache,
                use_cache=True, logits_to_keep=1,
            ).logits.float().clone()
            del cache
            row = run_variant(model, name, args, step_ids, device, fold, offload, reference)
        rows.append(row)
        print(f"{name:<16} {row['mean_ms']:>8.3f} ms  policy={row['policy']:<9} "
              f"gpu={row['gpu_params_m']:.1f}M  maxdiff={row['max_logit_diff']:.6f}", flush=True)

    base = rows[0]["mean_ms"]
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
