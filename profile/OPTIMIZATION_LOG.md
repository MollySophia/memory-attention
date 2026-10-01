# Optimization log

Running record for Memory Attention experiments. One entry per optimization
step, newest last. Each entry must be reproducible from the stated commit.

## How to read an entry

- **commit** — the exact tree that produced the numbers. Re-check out and rerun.
- **env** — from `bench_fla.py`'s fingerprint. Numbers are not comparable across
  different `torch` / `flash_attn` / GPU rows.
- **prefill / decode** — median ms at 24 layers / hidden 2048 / 32 heads /
  batch 8 / seq 2048, BF16, against the `ma_gpu` reference (1.00x).
- **spread** — (max−min)/mean across the 5 per-round means. This is the noise
  floor. A change smaller than the spread is not a result.
- **correctness** — gate status from `profile/test_memory_offload.py`. Prefill,
  decode and left-padded batches, each against the resident folded path.

## Gate

```bash
python profile/test_memory_offload.py          # must print PASS, exit 0
```

Covers: prefill (bulk + pipeline), decode with a real KV cache, and a
left-padded batch that exercises the unpad/varlen attention branch.

## Benchmark

```bash
# single configuration
python profile/bench_fla.py --mode prefill --variants ma_gpu ma_offload \
    --batch-size 8 --seq-len 2048 --context-len 2048 --json out.json

# sweep (isolated process per variant/mode/shape)
python profile/sweep_bmk.py --bmk bench_fla.py --output sweep_results \
    --reference ma_gpu --batch-sizes 8 16 32 --lengths 2048 4096
```

`--reference ma_gpu` is required for this model: it has no `standard` variant
(no `v_proj`), so `ma_gpu` — the resident, norm-folded table — is the baseline
where the only difference is the transfer itself.

### Environment convention

Each sweep writes `<output>/env.json` recording the git commit, whether the
tree was dirty, torch / torch_cuda / flash-attn / fla versions, python, GPU
name, capability and count. On a later run against the same directory:

- any difference in those fields is **rejected** with a diff of what changed,
  because a different GPU or torch build moves timings by more than any
  optimization under test;
- the environment is folded into every job signature, so `--resume` cannot
  reuse rows measured elsewhere;
- pass `--allow-env-change` to deliberately re-measure in a new environment.

**One sweep directory = one environment = one experiment.** When comparing
across optimization steps, compare the summary CSVs and check that their
`env.json` files agree; if the commit differs, the numbers are from a different
tree and are not directly comparable. Record the commit in the log entry.

## Established baseline

Recorded on the environment below, at commit `300ebb9` (tree dirty: the
sweep harness changes were uncommitted during the run). See `env.json` in
`results/sweep_baseline/` for the machine-readable form.

| env | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, sm_120, 32 GiB |
| torch | 2.9.0+cu130 |
| flash-attn | 2.8.3 |
| fla | 0.4.2 |
| python | 3.12.2 |

Sweep over batch size at seq/context 2048, 24 layers / hidden 2048 / 32 heads,
BF16. `ma_gpu` is the 1.00x reference. Raw data:
`results/sweep_baseline/summary.csv` (12/12 runs, no OOM).

| mode | batch | ma_gpu (ms) | ma_offload (ms) | rel | offload spread |
|---|---|---|---|---|---|
| prefill | 4 | 110.159 | 113.277 | 0.97x | 0.81% |
| prefill | 8 | 217.244 | 220.549 | 0.99x | 0.93% |
| prefill | 16 | 438.525 | 446.003 | 0.98x | 1.42% |
| decode | 4 | 7.277 | 7.370 | 0.99x | 0.20% |
| decode | 8 | 11.004 | 11.124 | 0.99x | 0.88% |
| decode | 16 | 20.423 | 20.394 | **1.00x** | 0.57% |

Offload overhead: ~1–3% at prefill, ≤1% at decode, and at batch 16 decode it
falls inside the noise floor. Reference spread is 0.05–1.93%; treat anything
under ~2% at prefill and ~0.9% at decode as unresolved at this sample size.

Throughput at batch 16 decode: 784 tok/s offloaded vs 784 resident.

An earlier run of this same sweep gave 216.9 / 220.6 ms at bs8 prefill against
217.2 / 220.5 here — agreement to 0.5%. That is the spread column doing its job:
run-to-run variation is comparable to the effects being measured, so no single
point should be quoted as evidence on its own.

Memory: GPU-resident parameters drop 2836.5M → 1263.6M, and the 3000 MiB table
moves to pinned host memory. That is the actual trade — the offload is
essentially free at decode and costs a few percent at prefill.

Numeric agreement: the offloaded table is bitwise identical to the resident
folded table. The ~0.25 max logit delta between offloaded and resident is BF16
rounding in `k + m` (relative 2⁻⁸, accumulated over 24 layers), not a logic
difference. Unfolded vs folded is exact. All three gate cases pass bit-exact.

### Reading these numbers

The interesting result is that offload is close to free at decode. A decode
step touches one table row per layer (~2 MiB), so the transfer hides behind the
attention work; a prefill touches `batch × seq` rows and must actually move
them. If the goal is a large model on a small GPU, the prefill cost is the
number to argue about — at batch 16 that is 7.3 ms of 446 ms.

## Entries

One per optimization step, newest last. Copy this template:

```markdown
### <short title> — <commit>

**Change.** What changed and why.
**Correctness.** `PASS` / failure, and anything the gate does not cover.
**Data.** `<output dir>` — prefill median vs baseline, decode median vs
baseline, spread, GPU parameters.
**Verdict.** Better / worse / within noise, against the spread column.
```

_(none yet — append one per optimization step)_
