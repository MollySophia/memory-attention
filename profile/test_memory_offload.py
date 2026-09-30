"""Correctness check: CPU-offloaded memory table vs GPU-resident baseline.

Runs the same weights twice through MemoryForCausalLM -- once with the table
resident on the GPU, once with it streamed from pinned host memory -- and
compares logits. The folded table must reproduce the per-token m_norm result,
so both paths should agree to within BF16 tolerance.
"""

import torch

from fla.models.memory.configuration_memory import MemoryConfig
from fla.models.memory.modeling_memory import MemoryForCausalLM


def build(seed=1234, layers=4, hidden=512, heads=8, vocab=2048):
    torch.manual_seed(seed)
    config = MemoryConfig(
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_heads=heads,
        num_kv_heads=heads,
        vocab_size=vocab,
        qk_norm=True,
        use_gate=True,
        fuse_norm=False,
        use_cache=False,
    )
    model = MemoryForCausalLM(config).to("cuda:0", dtype=torch.bfloat16).eval()
    return model, config


@torch.inference_mode()
def run_case(policy, batch, seq_len, model, config, ids, ref):
    model.close_memory_offload()
    model.enable_memory_offload(device="cuda:0", dtype=torch.bfloat16, fold_norm=True)
    model.config.memory_offload_policy = policy
    model.set_offload_offloader(batch, seq_len)

    out = model(input_ids=ids, use_cache=False).logits.float()
    diff = (out - ref).abs()
    agree = (out.argmax(-1) == ref.argmax(-1)).float().mean().item() * 100

    print(f"  policy={policy:<8} tokens={batch * seq_len:<6} "
          f"max diff={diff.max().item():.6f}  argmax agree={agree:.2f}%  "
          f"finite={bool(torch.isfinite(out).all())}")
    return diff.max().item(), agree


@torch.inference_mode()
def check_decode(model, config, device="cuda:0"):
    """Offloaded decode must match the resident path against a real KV cache.

    Decoding is where offload has the best chance of paying off: one token
    touches one table row per layer, so the transfer is tiny. That makes the
    shape plumbing worth testing -- prefill alone never exercises the
    seq_len=1 path or the per-shape offloader cache.
    """
    from fla.models.utils import Cache

    torch.manual_seed(11)
    batch, prefix_len = 2, 96
    prefix = torch.randint(0, config.vocab_size, (batch, prefix_len), device=device)
    step = torch.randint(0, config.vocab_size, (batch, 1), device=device)

    def rollback(cache, length):
        holders = cache.states if hasattr(cache, "states") else [l.state for l in cache.layers]
        for state in holders:
            if isinstance(state, dict) and state.get("attn_state") is not None:
                state["attn_state"] = [t[:, :length] for t in state["attn_state"]]
        cache._seen_tokens = length

    model.close_memory_offload()
    model.model._restore_unfolded_table(device, torch.bfloat16)
    model.fold_memory_table_on_gpu(dtype=torch.bfloat16)

    cache = Cache.from_legacy_cache(None)
    model(input_ids=prefix, past_key_values=cache, use_cache=True)
    rollback(cache, prefix_len)
    ref = model(
        input_ids=step, past_key_values=cache, use_cache=True, logits_to_keep=1
    ).logits.float().clone()

    # Same call, but streaming the table from host memory. Also drives the
    # per-shape offloader cache: prefill used a (batch, prefix_len) buffer and
    # decode needs its own (batch, 1) one.
    model.close_memory_offload()
    model.model._restore_unfolded_table(device, torch.bfloat16)
    model.enable_memory_offload(device=device, dtype=torch.bfloat16, fold_norm=True)

    cache2 = Cache.from_legacy_cache(None)
    model(input_ids=prefix, past_key_values=cache2, use_cache=True)
    rollback(cache2, prefix_len)
    out = model(
        input_ids=step, past_key_values=cache2, use_cache=True, logits_to_keep=1
    ).logits.float()

    diff = (out - ref).abs().max().item()
    agree = (out.argmax(-1) == ref.argmax(-1)).float().mean().item() * 100
    cached = len(getattr(model.model, "_offloader_cache", {}))
    policy = model.model.memory_offloader.policy if model.model.memory_offloader else "none"
    print(f"  decode           max diff={diff:.6f}  argmax agree={agree:.2f}%  "
          f"policy={policy}  cached offloaders={cached}")
    return diff


@torch.inference_mode()
def main():
    model, config = build()
    torch.manual_seed(7)

    print("=== correctness: offloaded table vs GPU-resident baseline ===")
    worst = 0.0
    for policy, batch, seq_len in [
        ("bulk", 2, 128),       # small input -> bulk
        ("pipeline", 2, 128),   # same input, pipeline path
        ("pipeline", 4, 512),   # larger -> exercises multiple groups
        ("pipeline", 1, 2048),  # long sequence
    ]:
        ids = torch.randint(0, config.vocab_size, (batch, seq_len), device="cuda:0")
        ref = model(input_ids=ids, use_cache=False).logits.float().clone()
        if policy == "bulk" and batch == 2 and seq_len == 128:
            print(f"  reference logits captured for max |logit| = {ref.abs().max().item():.3f}")
        d, _ = run_case(policy, batch, seq_len, model, config, ids, ref)
        worst = max(worst, d)

    print()
    print("=== decode: real KV cache, single-token steps ===")
    decode_diff = check_decode(model, config)
    worst = max(worst, decode_diff)

    print()
    print("=== memory accounting ===")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(0, config.vocab_size, (2, 128), device="cuda:0")
    model.close_memory_offload()
    model.enable_memory_offload(device="cuda:0", dtype=torch.bfloat16, fold_norm=True)
    resident = torch.cuda.memory_allocated()
    table_mib = model.model.memory_table.numel() * model.model.memory_table.element_size() / 2**20
    print(f"  folded CPU table   : {table_mib:.1f} MiB (pinned, {model.model.memory_table.shape})")
    print(f"  GPU after offload  : {resident / 2**20:.1f} MiB")

    ok = worst < 0.05
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} (worst max diff {worst:.6f}, tolerance 0.05)")

    model.close_memory_offload()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
