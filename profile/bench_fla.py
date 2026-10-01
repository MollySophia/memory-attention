"""Sweep-compatible single-run driver for MemoryForCausalLM.

Emits the same JSON schema bmk.py does, so sweep_bmk.py can drive this repo's
model without modification:

    python sweep_bmk.py --bmk bench_fla.py --output sweep_results

One process per (variant, mode, batch, length), matching the isolation the
sweep relies on. Variants mirror bmk.py's naming so summaries line up:

  standard         -- not available: MemoryAttention has no v_proj baseline
  ma_gpu           -- resident table, m_norm folded      (1.00x reference)
  ma_offload       -- streamed from pinned host memory
  ma_gpu_unfolded  -- resident table, norm per token    (original fla path)

Primary prefill constructs a KV cache and returns last-token logits. Historical
full-logits/no-cache prefill is selected explicitly with --prefill-workload.
Every raw sample is retained; median_ms is the median of round means. Round
spread describes variability and is not a significance threshold. Protocol is
still draft until memory/environment and growing-generation reporting land.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from fla.models.memory.configuration_memory import MemoryConfig
from fla.models.memory.modeling_memory import MemoryForCausalLM
from fla.models.utils import Cache


VARIANTS = ("ma_gpu", "ma_offload", "ma_gpu_unfolded")
MODES = ("prefill", "decode")
PROTOCOL_VERSION = "paper_v1_draft"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--variants", nargs="+", choices=VARIANTS, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--context-len", type=int, required=True)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--logits-to-keep", type=int, default=1)
    p.add_argument("--prefill-workload", choices=("inference", "historical"), default="inference")
    p.add_argument("--hidden-size", type=int, default=2048)
    p.add_argument("--num-heads", type=int, default=32)
    p.add_argument("--num-kv-heads", type=int, default=None)
    p.add_argument("--num-layers", type=int, default=24)
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--hidden-ratio", type=int, default=4)
    p.add_argument("--intermediate-size", type=int, default=None)
    p.add_argument("--group-size", type=int, default=1)
    p.add_argument("--prefetch-depth", type=int, default=4)
    p.add_argument("--policy", choices=["auto", "pipeline", "bulk"], default="auto")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--json", type=Path, required=True)
    args = p.parse_args()
    if min(args.batch_size, args.seq_len, args.context_len, args.repeats, args.rounds) <= 0 or args.warmup < 0:
        p.error("shapes, repeats and rounds must be positive; warmup must be nonnegative")
    if args.logits_to_keep < 0:
        p.error("logits-to-keep must be nonnegative")
    return args


def build(args):
    torch.manual_seed(args.seed)
    config = MemoryConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads if args.num_kv_heads else args.num_heads,
        vocab_size=args.vocab_size,
        hidden_ratio=args.hidden_ratio,
        intermediate_size=args.intermediate_size,
        use_cache=True,
        fuse_norm=False,
        memory_offload_group_size=args.group_size,
        memory_offload_prefetch_depth=args.prefetch_depth,
        memory_offload_policy=args.policy,
    )
    return MemoryForCausalLM(config).to(args.device, dtype=torch.bfloat16).eval()


def rollback(cache, length):
    """Trim the KV cache back to `length` tokens.

    fla/models/utils.py:241 concatenates instead of overwriting when no sliding
    window is set, so an un-trimmed decode loop would grow the context every
    step and measure something different each repetition.
    """
    holders = cache.states if hasattr(cache, "states") else [layer.state for layer in cache.layers]
    for state in holders:
        if isinstance(state, dict) and state.get("attn_state") is not None:
            state["attn_state"] = [t[:, :length] for t in state["attn_state"]]
    cache._seen_tokens = length


@torch.inference_mode()
def run_prefill(model, args, device):
    ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
    feed = ids
    historical = args.prefill_workload == "historical"

    def one():
        # A fresh cache on every prefill; construction is part of model work.
        return model(input_ids=feed, use_cache=not historical,
                     logits_to_keep=0 if historical else args.logits_to_keep)

    for _ in range(args.warmup):
        one()
    torch.cuda.synchronize(device)

    rounds = []
    raw_samples = []
    for _ in range(args.rounds):
        samples = []
        for _ in range(args.repeats):
            torch.cuda.synchronize(device)
            begin = time.perf_counter()
            one()
            torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - begin) * 1e3)
        raw_samples.append(samples)
        rounds.append(statistics.mean(samples))
    return {"round_ms": rounds, "samples_ms": raw_samples}


@torch.inference_mode()
def run_decode(model, args, device):
    prefix = torch.randint(
        0, args.vocab_size, (args.batch_size, args.context_len), device=device
    )
    steps = torch.randint(
        0, args.vocab_size, (args.repeats + args.warmup, args.batch_size, 1), device=device
    )
    # Prefix build is setup, not part of the measured step.
    out = model(
        input_ids=prefix, past_key_values=Cache.from_legacy_cache(None), use_cache=True, logits_to_keep=1
    )
    cache = out.past_key_values
    del out, prefix

    def one(token):
        return model(
            input_ids=token, past_key_values=cache, use_cache=True,
            logits_to_keep=args.logits_to_keep,
        ).logits

    for i in range(args.warmup):
        rollback(cache, args.context_len)
        one(steps[i])
    torch.cuda.synchronize(device)

    rounds = []
    raw_samples = []
    cursor = args.warmup
    for _ in range(args.rounds):
        samples = []
        for _ in range(args.repeats):
            # Restore the fixed context and select the test token before
            # timing. The model's own KV-cache update stays inside one().
            rollback(cache, args.context_len)
            token = steps[cursor % steps.shape[0]]
            torch.cuda.synchronize(device)
            begin = time.perf_counter()
            one(token)
            torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - begin) * 1e3)
            cursor += 1
        raw_samples.append(samples)
        rounds.append(statistics.mean(samples))
    del cache, steps
    return {"round_ms": rounds, "samples_ms": raw_samples}


@torch.inference_mode()
def measure(model, args, variant, device):
    model.close_memory_offload()
    torch.cuda.empty_cache()
    model.model._restore_unfolded_table(device, torch.bfloat16)

    offload = variant == "ma_offload"
    if offload:
        # enable_memory_offload() folds the table and frees the per-layer
        # m_proj, so folding separately is neither needed nor possible here.
        model.enable_memory_offload(device=device, dtype=torch.bfloat16, fold_norm=True)
    elif variant == "ma_gpu":
        model.fold_memory_table_on_gpu(dtype=torch.bfloat16)

    policy = "none"
    if offload:
        # Build for both shapes up front, so buffer allocation is not charged
        # to the first timed forward.
        model.set_offload_offloader(args.batch_size, args.seq_len)
        model.set_offload_offloader(args.batch_size, 1)
        # Report the policy of the shape this mode actually runs. prefill uses
        # (batch, seq_len); decode steps are (batch, 1), and auto policy picks
        # bulk there because a decode step touches few enough rows that the
        # pipeline's per-layer coordination would dominate.
        if args.mode == 'prefill':
            model.set_offload_offloader(args.batch_size, args.seq_len)
        else:
            model.set_offload_offloader(args.batch_size, 1)
        policy = model.model.memory_offloader.policy

    runner = run_prefill if args.mode == "prefill" else run_decode
    # Variant-independent inputs, even when several placements share a process.
    torch.manual_seed(args.seed + 1)
    measurements = runner(model, args, device)
    rounds = measurements["round_ms"]
    samples = [sample for run in measurements["samples_ms"] for sample in run]

    gpu_params = sum(p.numel() for p in model.parameters() if p.device.type == "cuda")
    table = model.model.memory_table
    row = dict(
        variant=variant,
        mode=args.mode,
        median_ms=statistics.median(rounds),
        mean_ms=statistics.mean(rounds),
        min_ms=min(rounds),
        max_ms=max(rounds),
        round_ms=rounds,
        samples_ms=measurements["samples_ms"],
        protocol_version=PROTOCOL_VERSION,
        estimator="median_of_round_means",
        sample_p50_ms=statistics.median(samples),
        sample_p95_ms=sorted(samples)[max(0, math.ceil(len(samples) * .95) - 1)],
        sample_percentile_method="nearest_rank_p95; median_p50",
        tokens_per_second=args.batch_size * (args.seq_len if args.mode == "prefill" else 1) * 1000 / statistics.median(rounds),
        output_scope="full_logits_no_cache" if args.mode == "prefill" and args.prefill_workload == "historical" else "cached_logits",
        logits_to_keep=0 if args.mode == "prefill" and args.prefill_workload == "historical" else args.logits_to_keep,
        # Round spread is descriptive, not a significance threshold.
        spread_pct=(max(rounds) - min(rounds)) / statistics.mean(rounds) * 100,
        seq_len=args.seq_len if args.mode == "prefill" else 1,
        context_len=args.context_len,
        group_size=args.group_size,
        offload_policy=policy,
        gpu_parameters=gpu_params,
        cpu_parameters=(table.numel() if table is not None else 0),
        gpu_parameter_mib=gpu_params * 2 / 2**20,
        cpu_parameter_mib=(table.numel() * table.element_size() / 2**20) if table is not None else 0,
        offload_gpu_buffer_mib=0.0,
        offload_pinned_mib=0.0,
        benchmark_scope="model",
    )
    if args.mode == "decode":
        row["decode_timing_protocol"] = "fixed_context_v2_rollback_excluded"
    if offload and model.model.memory_offloader is not None:
        off = model.model.memory_offloader
        row["offload_gpu_buffer_mib"] = sum(
            s["gpu"].numel() * s["gpu"].element_size() for s in off.slots
        ) / 2**20
        row["offload_pinned_mib"] = sum(
            s["host"].numel() * s["host"].element_size() for s in off.slots
        ) / 2**20
    return row


def env_fingerprint():
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    try:
        import flash_attn
        flash = flash_attn.__version__
    except Exception:
        flash = "none"
    import fla
    return {
        "git_commit": commit,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "flash_attn": flash,
        "fla": fla.__version__,
        "python": platform.python_version(),
        "gpu": gpu,
        "gpu_capability": f"sm_{cap[0]}{cap[1]}",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


@torch.inference_mode()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("this benchmark requires a CUDA device")

    model = build(args)
    results = []
    for variant in dict.fromkeys(args.variants):
        results.append(measure(model, args, variant, device))

    payload = dict(protocol_version=PROTOCOL_VERSION, config=vars(args), model_config=model.config.to_dict(), env=env_fingerprint(), results=results)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    for row in results:
        print(
            f"{row['variant']:<16} {row['mode']:<8} median={row['median_ms']:9.3f} ms  "
            f"min={row['min_ms']:9.3f}  max={row['max_ms']:9.3f}  "
            f"spread={row['spread_pct']:.2f}%  policy={row['offload_policy']}",
            flush=True,
        )
    model.close_memory_offload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
