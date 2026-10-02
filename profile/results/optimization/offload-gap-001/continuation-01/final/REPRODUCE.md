# Reproduce continuation 01

Use the same recorded environment: RTX 5090, driver610.57.04, Python3.12.2, torch2.9.0+cu130, FlashAttention2.8.3, Ryzen9950X; torch intra-op16/inter-op32. Exact affinity, thread environment, source hashes and GPU telemetry accompany every raw result. Run GPU workloads sequentially. New output directories are mandatory; estimates are not live-job deadlines.

```bash
cd /home/molly/workspace-memory-attn/memory-attention
PY=/home/molly/miniconda3/envs/fla-bench/bin/python
C="$PWD/profile/results/optimization/offload-gap-001"
P=/home/molly/workspace-memory-attn/offload-gap-001-A0001
OUT=$(mktemp -d /tmp/offload-gap-continuation.XXXXXX)
```

Frozen candidate sources (all have accepted parent A0001, source942a2a52d94cf6dbe6e753f7ac291daf8b0699fb):

| Attempt | Source commit |
|---|---|
| A0006 | b58cdb8640fb2d6a57631e0c26fc6f9a72761f74 |
| A0007 | 2bcbe8d197ef3244af1f51beb1214e4f3ecf6ffc |
| A0008 | fb17afb2a97b68ec61ef517622707f68ce33e051 |
| A0009 | 0ac06a0f7ce6c3ce77fd5a878c4a34c991527ec2 |
| A0010 | b22f8f616ee933ddc4ad2dd09c3eb91e60ef477b |

Existing frozen worktrees use `/home/molly/workspace-memory-attn/offload-gap-001-A000N`. If absent, create a detached worktree at the exact SHA using `git worktree add --detach PATH SHA`. Do not substitute the current retained model for a rejected candidate.

Example A0006 correctness and fresh screen; substitute another candidate's worktree to reproduce its attempt. Every attempt's original screen manifest retains all exact per-process commands and setup estimates.

```bash
F=/home/molly/workspace-memory-attn/offload-gap-001-A0006
(cd "$F" && "$PY" -m pytest -q tests/test_memory_offload_regressions.py)
(cd "$F" && "$PY" -m pytest -q tests/test_decode_benchmark.py tests/test_sampling_plans.py)
(cd "$F" && "$PY" profile/test_memory_offload.py)
"$PY" "$F/profile/run_paper_matrix.py" --include-batch16 --estimate-setup-seconds 20 --estimate-prefill-ms 420 --estimate-decode-ms 21 --output "$OUT/screen-plan" --plan-only
"$PY" "$F/profile/run_paper_matrix.py" --include-batch16 --estimate-setup-seconds 20 --estimate-prefill-ms 420 --estimate-decode-ms 21 --output "$OUT/screen"
"$PY" "$C/summarize_screen.py" "$OUT/screen" --output "$OUT/screen-summary"
"$PY" "$C/compare_screens.py" --candidate "$OUT/screen-summary.json" --references "$C/A0000/R01-summary.json" "$C/A0001/R01-summary.json" --output "$OUT/screen-comparison"
```

This is descriptive screening, 3warmups/5samples/1round; stored screens cannot confirm an incremental gain. A0008's original estimate adds10seconds per process for new JIT/pinning setup. All candidates include batches1/8/16, prefill/decode, both offload and folded resident placements.

A0006 received the predeclared independent parent confirmation below: 48processes, all four primary workloads, both placements, three balanced blocks, 10warmups/10samples/3rounds. It failed the required dual latency/gap criterion. Other candidates stopped at screening; no additional effort was spent to chase a favorable result.

```bash
"$PY" "$C/run_paired.py" --baseline-root "$P" --candidate-root "$F" --stage confirmation --screen-manifest "$OUT/screen/manifest.json" --output "$OUT/parent-plan" --plan-only
"$PY" "$C/run_paired.py" --baseline-root "$P" --candidate-root "$F" --stage confirmation --screen-manifest "$OUT/screen/manifest.json" --output "$OUT/parent"
"$PY" "$C/analyze_paired.py" "$OUT/parent/manifest.json" --output "$OUT/parent-analysis.json"
```

Profiles must run separately from all model timing:

```bash
"$PY" "$C/diagnose_pipeline_coordination.py" --source-root "$P" --batch 1 --output "$OUT/coordination-b1.json"
"$PY" "$C/diagnose_pipeline_coordination.py" --source-root "$P" --batch 8 --output "$OUT/coordination-b8.json"
"$PY" "$C/diagnose_transfer_alternatives.py" --source-root "$P" --output "$OUT/transfer-alternatives.json"
"$PY" "$C/diagnose_transfer_alternatives.py" --source-root "$P" --mapped-ctas 64 128 --output "$OUT/transfer-bounded.json"
```

These are instrumented/component diagnostics, not model speedups. The transfer prototype pins a3000MiB CPU table and checks exact output every trial, using fresh identical IDs across methods; it does not test model compute overlap. The original helper revisions, commands, hashes and compiler logs are committed alongside output.

Regenerate continuation figures and audits without redoing GPU measurements:

```bash
MPLCONFIGDIR=/tmp/offload-gap-mpl "$PY" "$C/plot_validation.py" --matrix "$C/A0001/full-audited-summary.json" --generation "$C/A0001/generation-audited-summary.json" --output "$OUT/validation" --status-label 'After 10 attempts: retained A0001; frozen baseline A0000'
MPLCONFIGDIR=/tmp/offload-gap-mpl "$PY" "$C/plot_history.py" --campaign "$C" --output "$OUT/history"
"$PY" "$C/audit_final.py" --repository "$PWD" --campaign "$C" --continuation continuation-01 --artifacts "$C/continuation-01/final" --output "$OUT/core-audit.json"
"$PY" "$C/audit_continuation.py" --campaign "$C" --continuation continuation-01 --output "$OUT/additional-raw-audit.json"
```

The retained implementation is unchanged A0001; its independent full-size correctness, full matrix and generation evidence remain valid and are reused. [Original reproduction commands](../../final/REPRODUCE.md) document a fresh A0000/A0001 confirmation, full validation and generation. For the current extended ledger, use the explicit continuation audit above rather than the original audit invocation. Raw source-result paths are absolute: after relocation, remap prefixes in copies while preserving originals and exact source hashes. The first-tranche final report remains an immutable historical artifact at commitb687044b4179e09ead630d0cc132f787b100772b.
