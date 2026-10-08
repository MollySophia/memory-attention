"""Shared builders and storage accounting for offload correctness tests."""
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
def resident_prefill(model, ids):
    """Capture an independent folded resident reference for every case."""
    model.close_memory_offload()
    model.fold_memory_table_on_gpu(dtype=torch.bfloat16)
    try:
        return model(input_ids=ids, use_cache=False).logits.float().clone()
    finally:
        model.close_memory_offload()


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


def storage_bytes(tensors):
    """Count backing storage once, including aliased staging views."""
    seen = {}
    for tensor in tensors:
        if tensor is not None:
            storage = tensor.untyped_storage()
            seen[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(seen.values())
