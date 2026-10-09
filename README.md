# Memory Attention

An experimental implementation of Memory Attention built on Flash Linear Attention (FLA). It introduces a learnable memory table indexed by token IDs into the attention value computation and provides a Hugging Face-style causal language model interface.

The core implementation lives in [`fla/layers/memory_attn.py`](fla/layers/memory_attn.py) and [`fla/models/memory/`](fla/models/memory/). This repository retains FLA's other models and operators. The Python import name remains `fla`, and the package name remains `flash-linear-attention`.

## How It Works

For each layer's hidden states `X` and original token IDs `I`:

```text
Q  = q_proj(X)
K₀ = k_proj(X)
M  = m_proj(I)
V  = K₀ + RMSNorm(M)

Q, K = RoPE(optional_qk_norm(Q, K₀))
O    = causal_attention(Q, K, V)
Y    = o_proj(O)
```

Each attention layer has its own learnable embedding table, `m_proj`, with shape `[vocab_size, num_kv_heads × head_dim]`. Memory vectors are RMS-normalized per head. Values use `K₀` before QK normalization and RoPE, with no separate `v_proj`. When `use_gate` is enabled, a sigmoid gate computed from the hidden states is applied to `O` before the output projection.

The attention operation uses FlashAttention to compute causal softmax attention. Full attention has quadratic compute complexity in sequence length. Here, memory refers to a learned token lookup table stored in the model parameters; it is distinct from external retrieval and the KV cache.

## Installation

See the [Flash Linear Attention repository](https://github.com/fla-org/flash-linear-attention) for installation instructions.

## Performance Profiling

The [`profile/`](profile/) directory contains standalone inference experiments comparing standard attention (`standard`), GPU-resident memory tables (`ma_gpu`), and CPU-offloaded memory tables (`ma_offload`). `MemoryForCausalLM` also supports inference-only CPU table offload; see below.

Run the prefill and decode benchmark with the default configuration and save the results:

```bash
python profile/bmk.py --mode both --json bmk_results.json
```

Append `--profile-dir profiles` to capture stage timelines and `--kernel-trace` to save Chrome traces. Once a timeline has been captured, render it as an image:

```bash
python -m pip install matplotlib
python profile/ma_profile.py profiles/ma_offload_decode.timeline.json
```

Benchmark timings include the input embedding, all blocks, final RMSNorm, and LM head. CPU and GPU stage timelines use separate time origins. GPU ranges measure elapsed stream time, not kernel busy time. Report the device, model dimensions, context length, and timing configuration alongside performance results.

## Repository Structure

```text
fla/layers/memory_attn.py  Memory Attention layer
fla/models/memory/        Configuration, backbone, and causal language model
fla/modules/              RMSNorm, RoPE, MLP, and other components
fla/ops/                  FLA Triton operators
profile/                  Inference comparisons, CPU offloading, and profiling
benchmarks/               Benchmarks for other models and operators
evals/                    Language model evaluation entry points
legacy/training/          Legacy training code and examples
```

[`eval.sh`](eval.sh) demonstrates the `evals.harness` entry point, which requires the additional `lm-eval` dependency. Replace its local model path and device settings before use. Historical results in its comments are not performance claims for the current implementation. See [`legacy/training/README.md`](legacy/training/README.md) for the legacy training instructions.

## Acknowledgments and License

This project builds on Flash Linear Attention and retains its modules, operators, and source attribution. See [`CITATION.cff`](CITATION.cff) for upstream citation information.

The repository's [`LICENSE`](LICENSE) contains Apache License 2.0, while some license metadata in `pyproject.toml` and `setup.py` still specifies MIT. These declarations have not yet been reconciled.

## Model memory-table offload

Memory Attention supports BF16 inference with CPU-resident memory tables on CUDA.
Start with an evaluated CUDA model; offload folds table normalization and streams
lookups through pinned host memory. Disable offload before saving a checkpoint; training with offload is unsupported.

```python
model.eval()
with torch.inference_mode():
    model.enable_memory_offload()
    outputs = model(input_ids=input_ids, use_cache=True)
model.close_memory_offload()
```

Correctness tests require CUDA and FlashAttention:

```sh
python -m pytest tests/models/test_memory_offload.py -q
```

## Memory recall without RoPE

Set `rope_theta=None` to disable rotary positions. Set `memory_recall=True` to
cache raw keys and token IDs instead of full K/V pairs:

```python
config = MemoryConfig(rope_theta=None, memory_recall=True)
model = MemoryForCausalLM(config).to("cuda", dtype=torch.bfloat16).eval()
with torch.inference_mode():
    prefix = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    step = model(input_ids=next_token_ids, past_key_values=prefix.past_key_values,
                 use_cache=True, logits_to_keep=1)
```

Each layer stores `(K_raw, token_ids)` in its cache `attn_state`. On decode it
looks up the historical IDs and reconstructs `V = K_raw + RMSNorm(M)` before
attention. Optional QK normalization is applied after reconstruction, so the
cached keys remain raw. GPU-folded memory tables also work. With
`memory_recall=False` and `rope_theta=None`, the same no-RoPE model uses ordinary
K/V caching as a reference. Use a fresh cache when changing cache modes.

This initial implementation requires GPU-resident tables and does not support
CPU table offload or cached packed sequences. It materializes historical M and V
on every decode step; it reduces persistent cache storage, not necessarily peak
GPU memory or latency. For BF16 and a per-layer key width of `D`, persistent cache
storage per token changes from `4D` bytes to `2D + 8` bytes. Token IDs are currently
stored separately in each layer. Combining V reconstruction with a tiled
attention kernel is future work.

Disabling RoPE changes the model computation; this is not an equivalent conversion
of a checkpoint trained with RoPE. Existing configurations keep RoPE and full K/V
caching by default.

Validation covers explicit prefill/decode calls, including 128 decode steps.
With the tested Transformers 5.3.0 installation, `.generate()` encounters an
existing cache/input slicing error, also reproduced on the unchanged full-KV
model with RoPE. Use the explicit calls above for this prototype.

```sh
python -m pytest tests/models/test_memory_recall.py -q
```
