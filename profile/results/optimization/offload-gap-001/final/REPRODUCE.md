# Reproduce offload-gap-001

Use the recorded RTX 5090 environment: Python 3.12 environment at the path below, torch 2.9 CUDA 13, FlashAttention 2.8.3, driver 610.57.04, Ryzen 9 9950X; torch intra-op 16 and inter-op 32. Exact versions, affinity, thread variables, GPU temperature/clocks and competing processes are in each raw payload. Do not compare a different environment as matching confirmation. Run GPU experiments sequentially. The recorded idle TRM process consumed approximately 654 MiB GPU memory.

Run from the campaign repository. Use existing frozen worktrees or create them at these exact commits; commands below use the existing recorded paths. The campaign helper scripts live in the campaign branch, independently of the frozen model checkouts. Source hashes cover model and immediate profile Python files. Fresh output directories are mandatory.

```bash
cd /home/molly/workspace-memory-attn/memory-attention
PY=/home/molly/miniconda3/envs/fla-bench/bin/python
C="$PWD/profile/results/optimization/offload-gap-001"
B=/home/molly/workspace-memory-attn/offload-gap-001-A0000
F=/home/molly/workspace-memory-attn/offload-gap-001-A0001
# Only if these worktrees do not already exist:
# git worktree add --detach "$B" 62942a0387608fe21baaeb2dce9ef3b0947dde4d
# git worktree add --detach "$F" 942a2a52d94cf6dbe6e753f7ac291daf8b0699fb
OUT=$(mktemp -d /tmp/offload-gap-reproduce.XXXXXX)
```

Correctness gates (run in each frozen checkout; retained source has 43 combined regression/sampling tests):

```bash
(cd "$F" && "$PY" -m pytest -q tests/test_memory_offload_regressions.py)
(cd "$F" && "$PY" -m pytest -q tests/test_decode_benchmark.py tests/test_sampling_plans.py)
(cd "$F" && "$PY" profile/test_memory_offload.py)
for batch in 1 8; do
  for seed in 1234 4321; do
    "$PY" "$C/full_correctness.py" --source-root "$B" --variant ma_gpu --batch-size "$batch" --seed "$seed" --output "$OUT/reference-b$batch-s$seed.json"
    "$PY" "$C/full_correctness.py" --source-root "$F" --variant ma_offload --batch-size "$batch" --seed "$seed" --output "$OUT/candidate-b$batch-s$seed.json"
    "$PY" "$C/compare_correctness.py" "$OUT/reference-b$batch-s$seed.json" "$OUT/candidate-b$batch-s$seed.json" --output "$OUT/comparison-b$batch-s$seed.json"
  done
done
```

Fresh primary confirmation has 48 isolated processes: four workloads × two sources × two placements × three independently balanced blocks. Each process uses 10 warmups, 10 samples per round, three rounds. Plan first in a different output directory; the actual controller writes its own immutable plan and raw results. Estimates are planning information, not live-job deadlines.

```bash
"$PY" "$C/run_paired.py" --baseline-root "$B" --candidate-root "$F" --stage confirmation --screen-manifest "$C/A0000/R01/manifest.json" --output "$OUT/confirmation-plan" --plan-only
"$PY" "$C/run_paired.py" --baseline-root "$B" --candidate-root "$F" --stage confirmation --screen-manifest "$C/A0000/R01/manifest.json" --output "$OUT/confirmation"
"$PY" "$C/analyze_paired.py" "$OUT/confirmation/manifest.json" --output "$OUT/confirmation-analysis.json"
```

Full validation contains 84 unique points. Reusing the fresh confirmation supplies 16 primary points with all three independent blocks, leaving 68 new processes. Generation contains 12 points, each with two complete warmup trajectories and five trajectories per round × three rounds. Generation includes the prefix plus 128 predetermined-token calls; selection is outside timing.

```bash
"$PY" "$C/run_paired.py" --baseline-root "$B" --candidate-root "$F" --stage full_validation --screen-manifest "$C/A0000/R01/manifest.json" --reuse-confirmation "$OUT/confirmation/manifest.json" --output "$OUT/full-plan" --plan-only
"$PY" "$C/run_paired.py" --baseline-root "$B" --candidate-root "$F" --stage full_validation --screen-manifest "$C/A0000/R01/manifest.json" --reuse-confirmation "$OUT/confirmation/manifest.json" --output "$OUT/full"
"$PY" "$C/run_paired.py" --baseline-root "$B" --candidate-root "$F" --stage generation_validation --screen-manifest "$C/A0000/R01/manifest.json" --output "$OUT/generation-plan" --plan-only
"$PY" "$C/run_paired.py" --baseline-root "$B" --candidate-root "$F" --stage generation_validation --screen-manifest "$C/A0000/R01/manifest.json" --output "$OUT/generation"
"$PY" "$C/collect_matrix.py" "$OUT/full/manifest.json" --output "$OUT/full-summary"
"$PY" "$C/collect_matrix.py" "$OUT/generation/manifest.json" --output "$OUT/generation-summary"
```

These commands reproduce the base validation protocol. The historical secondary investigations additionally used predeclared R07 and R08 manifests, preserving original pairs and collecting exactly two more balanced pairs for the specified points. Their exact commands, environment, estimates and original/new source references are in those manifests. Do not substitute the invalid A0001/R02 design for eligible R03 confirmation, or extend sampling until a favorable result appears.

Regenerate final plots from the stored, fully audited evidence without GPU benchmarking:

```bash
MPLCONFIGDIR=/tmp/offload-gap-mpl "$PY" "$C/plot_validation.py" --matrix "$C/A0001/full-audited-summary.json" --generation "$C/A0001/generation-audited-summary.json" --output "$OUT/validation" --status-label 'Frozen baseline A0000; final accepted step 1: A0001'
MPLCONFIGDIR=/tmp/offload-gap-mpl "$PY" "$C/plot_history.py" --campaign "$C" --output "$OUT/history"
"$PY" "$C/audit_final.py" --repository "$PWD" --campaign "$C" --artifacts "$C/final" --output "$OUT/core-audit.json"
```

All original per-job commands are retained in their manifests. Raw result paths refer to the recorded absolute workspace; after relocating the repository, explicitly remap these prefixes in copies of the manifests, preserving the originals and source checksums. Python helper command lines are also discoverable with `--help`. The complete attempt ledger links each frozen candidate source, correctness output, screen, confirmation, profiles and verdict.
