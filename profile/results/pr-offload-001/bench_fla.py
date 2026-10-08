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
spread describes variability and is not a significance threshold. Timing scope offload_gap_v1 belongs to offload-gap-001. Sampling defaults
to screening (3/5/1); generation defaults to generation_validation (2/5/3).
Use --stage confirmation or full_validation for 10/10/3, or legacy for 30/30/5.
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
import traceback
from pathlib import Path

import torch

# Support direct CLI invocation and importlib-based regression tests.
sys.path.insert(0, os.environ["MA_SOURCE_ROOT"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_telemetry import memory_snapshot, environment_details, source_state
from sampling_plan import STAGES, sampling_plan

from fla.models.memory.configuration_memory import MemoryConfig
from fla.models.memory.modeling_memory import MemoryForCausalLM
from fla.models.utils import Cache


VARIANTS = ("ma_gpu", "ma_offload", "ma_gpu_unfolded")
MODES = ("prefill", "decode", "generation")
PROTOCOL_VERSION = "offload_gap_v1"
CAMPAIGN_ID = "pr-offload-001"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--variants", nargs="+", choices=VARIANTS, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--context-len", type=int, required=True)
    p.add_argument("--generation-steps", type=int, default=128)
    p.add_argument("--stage", choices=STAGES, help="default: screening; generation uses generation_validation")
    p.add_argument("--warmup", type=int, help="override stage warmup (recorded as custom plan)")
    p.add_argument("--repeats", type=int)
    p.add_argument("--rounds", type=int)
    p.add_argument("--logits-to-keep", type=int, default=1)
    p.add_argument("--prefill-workload", choices=("inference", "historical"), default="inference")
    p.add_argument("--hidden-size", type=int, default=2048)
    p.add_argument("--num-heads", type=int, default=32)
    p.add_argument("--num-kv-heads", type=int, default=None)
    p.add_argument("--num-layers", type=int, default=24)
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--hidden-ratio", type=int, default=4)
    p.add_argument("--intermediate-size", type=int, default=5632)
    p.add_argument("--group-size", type=int, default=1)
    p.add_argument("--prefetch-depth", type=int, default=4)
    p.add_argument("--policy", choices=["auto", "pipeline", "bulk"], default="auto")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--json", type=Path, required=True)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--estimate-setup-seconds", type=float, default=30)
    p.add_argument("--estimate-call-ms", type=float, default=1000)
    args = p.parse_args()
    if len(args.variants) != 1:
        p.error("PR measurements require one fresh model/placement per process")
    stage = args.stage or ('generation_validation' if args.mode == 'generation' else 'screening')
    try:
        plan = sampling_plan(stage, args.mode, args.warmup, args.repeats, args.rounds)
    except ValueError as exc:
        p.error(str(exc))
    for key, value in plan.items():
        setattr(args, key, value)
    if min(args.batch_size, args.seq_len, args.context_len, args.repeats, args.rounds, args.generation_steps) <= 0 or args.warmup < 0:
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

    memory_before = memory_snapshot(model, device) if str(device) != "cpu" else None
    if memory_before is not None:
        torch.cuda.reset_peak_memory_stats(device)
    memory_after = None
    rounds = []
    raw_samples = []
    for _ in range(args.rounds):
        samples = []
        for _ in range(args.repeats):
            torch.cuda.synchronize(device)
            begin = time.perf_counter()
            output = one()
            torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - begin) * 1e3)
            if memory_before is not None and len(raw_samples) == args.rounds - 1 and len(samples) == args.repeats:
                memory_after = memory_snapshot(model, device, getattr(output, "past_key_values", None))
            del output
        raw_samples.append(samples)
        rounds.append(statistics.mean(samples))
    return {"round_ms": rounds, "samples_ms": raw_samples, "memory_before": memory_before, "memory_after": memory_after}


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
        )

    for i in range(args.warmup):
        rollback(cache, args.context_len)
        one(steps[i])
    torch.cuda.synchronize(device)

    memory_before = memory_snapshot(model, device, cache) if str(device) != "cpu" else None
    if memory_before is not None:
        torch.cuda.reset_peak_memory_stats(device)
    memory_after = None
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
            output = one(token)
            torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - begin) * 1e3)
            if memory_before is not None and len(raw_samples) == args.rounds - 1 and len(samples) == args.repeats:
                memory_after = memory_snapshot(model, device, cache)
            del output
            cursor += 1
        raw_samples.append(samples)
        rounds.append(statistics.mean(samples))
    del cache, steps
    return {"round_ms": rounds, "samples_ms": raw_samples, "memory_before": memory_before, "memory_after": memory_after}


@torch.inference_mode()
def run_generation(model, args, device):
    prefix = torch.randint(0, args.vocab_size, (args.batch_size, args.context_len), device=device)
    # Predetermined GPU tokens; selection and allocation are excluded.
    tokens = torch.randint(0, args.vocab_size, (args.generation_steps, args.batch_size, 1), device=device).unbind(0)

    def one():
        output = model(input_ids=prefix, use_cache=True, logits_to_keep=args.logits_to_keep)
        for token in tokens:
            cache = output.past_key_values
            del output
            output = model(input_ids=token, past_key_values=cache, use_cache=True,
                           logits_to_keep=args.logits_to_keep)
        return output

    for _ in range(args.warmup):
        one()
    torch.cuda.synchronize(device)
    before = memory_snapshot(model, device) if str(device) != "cpu" else None
    if before is not None:
        torch.cuda.reset_peak_memory_stats(device)
    raw, after = [], None
    for round_index in range(args.rounds):
        samples = []
        for sample_index in range(args.repeats):
            torch.cuda.synchronize(device)
            begin = time.perf_counter()
            output = one()
            torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - begin) * 1000)
            if before is not None and round_index == args.rounds - 1 and sample_index == args.repeats - 1:
                after = memory_snapshot(model, device, output.past_key_values)
            del output
        raw.append(samples)
    return dict(round_ms=[statistics.mean(samples) for samples in raw], samples_ms=raw,
                memory_before=before, memory_after=after)


@torch.inference_mode()
def measure(model, args, variant, device):
    # Each isolated process constructs fresh weights; no restore API is needed.
    torch.cuda.empty_cache()

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
        model.set_offload_offloader(args.batch_size, args.seq_len if args.mode == "prefill" else args.context_len)
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

    runner = {"prefill": run_prefill, "decode": run_decode, "generation": run_generation}[args.mode]
    # Variant-independent inputs, even when several placements share a process.
    torch.manual_seed(args.seed + 1)
    measurements = runner(model, args, device)
    rounds = measurements["round_ms"]
    samples = [sample for run in measurements["samples_ms"] for sample in run]

    gpu_params = sum(p.numel() for p in model.parameters() if p.device.type == "cuda")
    table = getattr(model.model, "memory_table", None)
    row = dict(
        variant=variant,
        mode=args.mode,
        median_ms=statistics.median(rounds),
        mean_ms=statistics.mean(rounds),
        min_ms=min(rounds),
        max_ms=max(rounds),
        round_ms=rounds,
        samples_ms=measurements["samples_ms"],
        memory_before=measurements["memory_before"],
        memory_after=measurements["memory_after"],
        protocol_version=PROTOCOL_VERSION,
        sampling_plan=sampling_plan(args.stage, args.mode, args.warmup, args.repeats, args.rounds),
        estimator="median_of_round_means",
        sample_p50_ms=statistics.median(samples),
        sample_p95_ms=sorted(samples)[max(0, math.ceil(len(samples) * .95) - 1)],
        sample_percentile_method="nearest_rank_p95; median_p50",
        tokens_per_second=args.batch_size * (args.seq_len if args.mode == "prefill" else args.context_len + args.generation_steps if args.mode == "generation" else 1) * 1000 / statistics.median(rounds),
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
    if args.mode == "generation":
        row["generation_steps"] = args.generation_steps
        row["generation_timing_protocol"] = "prefill_plus_growing_decode_predetermined_tokens_no_sampling"
        row["throughput_token_scope"] = "prefix_plus_decode_tokens"
    if args.mode == "decode":
        row["decode_timing_protocol"] = "fixed_context_v2_rollback_excluded"
    memory = measurements["memory_after"]
    row["offload_gpu_buffer_mib"] = memory["offload_gpu_buffer_bytes"] / 2**20
    row["offload_pinned_mib"] = memory["offload_pinned_bytes"] / 2**20
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
        "model_module": sys.modules[MemoryForCausalLM.__module__].__file__,
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
    if getattr(args, "plan_only", False):
        calls = args.warmup + args.repeats * args.rounds
        calls *= 1 + args.generation_steps if args.mode == "generation" else 1
        print(json.dumps(dict(campaign_id=CAMPAIGN_ID, protocol_version=PROTOCOL_VERSION, config=vars(args), jobs=len(args.variants), model_calls_per_job=calls, estimated_seconds=len(args.variants)*(args.estimate_setup_seconds + calls*args.estimate_call_ms/1000), estimate_note="Planning estimates; replace setup/call estimates with measured values.", command=sys.argv), default=str, indent=2))
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("this benchmark requires a CUDA device")

    environment_before = environment_details()
    sources = source_state()
    model = None
    results = []
    status, failure = "completed", None
    try:
        model = build(args)
        for variant in dict.fromkeys(args.variants):
            results.append(measure(model, args, variant, device))
    except Exception as exc:
        status = "oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "benchmark_failed"
        failure = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        print(failure["traceback"], file=sys.stderr, flush=True)

    payload = dict(campaign_id=CAMPAIGN_ID, protocol_version=PROTOCOL_VERSION, stage=args.stage, measurement_plan_id=args.measurement_plan_id, config=vars(args), status=status, failure=failure, model_config=model.config.to_dict() if model is not None else None, env=env_fingerprint(), source=sources, environment_before=environment_before, environment_after=environment_details(), command=sys.argv, results=results)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    for row in results:
        print(
            f"{row['variant']:<16} {row['mode']:<8} median={row['median_ms']:9.3f} ms  "
            f"min={row['min_ms']:9.3f}  max={row['max_ms']:9.3f}  "
            f"spread={row['spread_pct']:.2f}%  policy={row['offload_policy']}",
            flush=True,
        )
    if model is not None and hasattr(model, "close_memory_offload"):
        model.close_memory_offload()
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
