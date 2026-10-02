# Reproduce continuation 02

Use the recorded RTX 5090 / driver 610.57.04, Ryzen 9950X, Python 3.12.2, torch 2.9.0+cu130, FlashAttention 2.8.3 environment. Torch intra-op/inter-op threads are 16/32; affinity covers 32 CPUs. Exact environment and source hashes accompany every raw process result. Run GPU experiments sequentially. Use fresh output directories; estimates are planning inputs, not deadlines.

```bash
cd /home/molly/workspace-memory-attn/memory-attention
PY=/home/molly/miniconda3/envs/fla-bench/bin/python
C="$PWD/profile/results/optimization/offload-gap-001"
B=/home/molly/workspace-memory-attn/offload-gap-001-A0000
P=/home/molly/workspace-memory-attn/offload-gap-001-A0001
F=/home/molly/workspace-memory-attn/offload-gap-001-A0016
OUT=$(mktemp -d /tmp/offload-gap-cont02.XXXXXX)
```

Frozen baseline A0000 is `62942a0387608fe21baaeb2dce9ef3b0947dde4d`; parent A0001 is `942a2a52d94cf6dbe6e753f7ac291daf8b0699fb`. Retained A0016 is `0ce6645eb4c4bc3992258ad0878c72ea3c37a64e`. If a worktree is absent, create it using `git worktree add --detach PATH SHA`. Candidate worktrees follow the same `offload-gap-001-A00NN` naming. Never substitute retained code for a rejected candidate.

| Attempt | Parent | Exact source commit |
|---|---|---|
| A0011 | A0001 | `812ac2decafaf2d9919028a7bbd0c740db07ba1a` |
| A0012 | A0001 | `817cc6b5081acdcc7d50935f98e0a4b8c0b71b3a` |
| A0013 | A0001 | `5d5eb4858f5102b5e7c0c004e359304b13ac38c8` |
| A0014 | A0001 | `b9938ebc46d7a92fc07898a09c72edc547f5f8f3` |
| A0015 | A0001 | `0436370506e82e3a32d6ccbc6bb5217bc0e84637` |
| A0016 | A0001 | `0ce6645eb4c4bc3992258ad0878c72ea3c37a64e` |
| A0017 | A0016 | `7eb99060e4ac0125bb93487abc85f139f1e73c8f` |
| A0018 | A0016 | `6089489be39a59a31360b8f89add2042cc0a7c8b` |
| A0019 | A0016 | `f59cdf0c8d7fa32f98b6947c0ba784210f0e7437` |
| A0020 | A0016 | `41869553c1df91e5c8ae4cda16cd4c6b1c0ab44b` |

## Correctness and screening

Example for A0016; substitute a candidate worktree to reproduce another attempt. Original per-attempt correctness logs and screen manifests retain their exact commands.

```bash
(cd "$F" && "$PY" -m pytest -q tests/test_memory_offload_regressions.py)
(cd "$F" && "$PY" profile/test_memory_offload.py)
"$PY" "$F/profile/run_paper_matrix.py" --stage screening --include-batch16 --estimate-setup-seconds 21 --estimate-prefill-ms 218 --estimate-decode-ms 12 --plan-only --output "$OUT/screen-plan"
"$PY" "$F/profile/run_paper_matrix.py" --stage screening --include-batch16 --estimate-setup-seconds 21 --estimate-prefill-ms 218 --estimate-decode-ms 12 --output "$OUT/screen"
"$PY" "$C/summarize_screen.py" "$OUT/screen" --output "$OUT/screen-summary"
"$PY" "$C/compare_screens.py" --candidate "$OUT/screen-summary.json" --references "$C/A0000/R01-summary.json" "$C/A0001/R01-summary.json" --output "$OUT/screen-comparison"
```

Screening contains 12 processes: batches 1/8/16, prefill/decode, both placements; each has 3 warmups and 5 samples. Screens are descriptive and cannot establish an incremental gain. For A0017–A0020, use the A0016 screen as the parent reference.

## Independent confirmations and retained validation

Run the following stages only when their preceding correctness and acceptance gates pass. A0011, A0012, A0014, A0015, A0016, A0017 and A0020 received parent confirmation. Only A0016 advanced through the entire sequence. Each confirmation has 48 processes, three balanced independent blocks across all four primary workloads and both placements, 10 warmups and 3 rounds of 10 samples.

```bash
"$PY" "$C/run_paired.py" --baseline-root "$P" --candidate-root "$F" --stage confirmation --screen-manifest "$OUT/screen/manifest.json" --output "$OUT/parent-plan" --plan-only
"$PY" "$C/run_paired.py" --baseline-root "$P" --candidate-root "$F" --stage confirmation --screen-manifest "$OUT/screen/manifest.json" --output "$OUT/parent"
"$PY" "$C/analyze_paired.py" "$OUT/parent/manifest.json" --output "$OUT/parent-analysis.json"
"$PY" "$C/run_paired.py" --baseline-root "$B" --candidate-root "$F" --stage confirmation --screen-manifest "$OUT/screen/manifest.json" --output "$OUT/baseline-plan" --plan-only
"$PY" "$C/run_paired.py" --baseline-root "$B" --candidate-root "$F" --stage confirmation --screen-manifest "$OUT/screen/manifest.json" --output "$OUT/baseline"
"$PY" "$C/analyze_paired.py" "$OUT/baseline/manifest.json" --output "$OUT/baseline-analysis.json"
for batch in 1 8; do
  for seed in 1234 4321; do
    "$PY" "$C/full_correctness.py" --source-root "$B" --variant ma_gpu --batch-size "$batch" --seed "$seed" --output "$OUT/reference-b${batch}-s${seed}.json"
    "$PY" "$C/full_correctness.py" --source-root "$F" --variant ma_offload --batch-size "$batch" --seed "$seed" --output "$OUT/candidate-b${batch}-s${seed}.json"
    "$PY" "$C/compare_correctness.py" "$OUT/reference-b${batch}-s${seed}.json" "$OUT/candidate-b${batch}-s${seed}.json" --output "$OUT/correctness-b${batch}-s${seed}.json"
  done
done
for stage in full_validation generation_validation; do
  reuse=()
  if [ "$stage" = full_validation ]; then reuse=(--reuse-confirmation "$OUT/baseline/manifest.json"); fi
  "$PY" "$C/run_paired.py" --baseline-root "$B" --candidate-root "$F" --stage "$stage" --screen-manifest "$OUT/screen/manifest.json" "${reuse[@]}" --output "$OUT/${stage}-plan" --plan-only
  "$PY" "$C/run_paired.py" --baseline-root "$B" --candidate-root "$F" --stage "$stage" --screen-manifest "$OUT/screen/manifest.json" "${reuse[@]}" --output "$OUT/$stage"
done
```

Use Bash for the array-based block. Full validation covers 84 points, with 68 new processes and 16 primary points backed by the 48 matching confirmation processes. Generation adds 12 processes. The recorded run reused four validated resident correctness fingerprints from A0001 and generated four new A0016 trajectories; the commands above instead compute fresh references. Reuse is valid only after the strict checks in `reuse_correctness.py`.

[Secondary policy](../../A0016/secondary-regression-policy.json) was fixed before inspecting those checks: any adverse original offload or resident direction at a nonprimary workload triggers exactly two additional balanced pairs, retaining the original. [Selection and runner](../../A0016/run_secondary_regressions.py), [matrix plan](../../A0016/R07-regression-plan/manifest.json), and [generation plan](../../A0016/R08-regression-plan/manifest.json) preserve the exact selected commands. For fresh evidence, use `select_regression_checks.py`, `extend_pairs.py`, `analyze_regression.py`, `merge_validation_evidence.py` and `collect_matrix.py` with fresh manifests/outputs and the same fixed rule. Do not extend inconclusive primary confirmations. Stored derived summaries retain all 200 final raw references; original summaries remain intact.

## Regenerate final artifacts without GPU measurements

```bash
MPLCONFIGDIR=/tmp/offload-gap-mpl "$PY" "$C/plot_validation.py" --matrix "$C/A0016/full-audited-summary.json" --generation "$C/A0016/generation-audited-summary.json" --output "$OUT/validation" --status-label 'After 20 attempts: retained A0016; frozen baseline A0000'
MPLCONFIGDIR=/tmp/offload-gap-mpl "$PY" "$C/plot_history.py" --campaign "$C" --output "$OUT/history"
"$PY" "$C/audit_final.py" --repository "$PWD" --campaign "$C" --continuation continuation-02 --artifacts "$C/continuation-02/final" --output "$OUT/core-audit.json"
"$PY" "$C/audit_continuation.py" --campaign "$C" --continuation continuation-02 --output "$OUT/additional-raw-audit.json"
"$PY" -m pytest -q "$C"/test_*.py
```

`continuation-02/build_ledger.py` regenerates the final ledger in its established destination. Raw paths are absolute; relocation requires remapping prefixes in copies while preserving original evidence and exact source hashes. Final execution files equal the verified A0016 source, so publication reused its full matrix, generation and correctness evidence without repeating identical GPU measurements. The two earlier final artifact directories remain unchanged.
