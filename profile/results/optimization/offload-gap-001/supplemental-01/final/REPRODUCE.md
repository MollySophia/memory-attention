# Reproduce the supplemental shape survey

Use the same recorded environment as the campaign (RTX5090 32GiB, driver610.57.04, Ryzen9950X, Python3.12.2, torch2.9.0+cu130, FlashAttention2.8.3, torch threads16/32, 32-CPU affinity). Exact environment, source SHA/hash, command and sample plan accompany each raw result. Run GPU jobs sequentially. Existing frozen worktrees `offload-gap-001-A0000` through `A0020` are required at their authoritative record SHAs; create missing ones with `git worktree add --detach PATH SHA`. Do not substitute current A0016 for a historical rejected candidate.

Initial driver/plans were committed at `58db344` before the first screen. Independent confirmation rules and wrapper were committed at `f6edc1b` while the first screen was running. Historical outcome and acceptance records remain unchanged. [Plan](../PLAN.md) defines branch-aware exclusions, the sign-based nomination rule, fixed three-block confirmation and Holm family. [Coverage](../coverage.json) explicitly records unmeasured cells.

For a fresh reproduction, use a new sibling directory under the campaign root. Never rerun `prepare` or a controller inside the completed evidence directory. The following commands create new output and preserve the existing evidence:

```bash
cd /home/molly/workspace-memory-attn/memory-attention
PY=/home/molly/miniconda3/envs/fla-bench/bin/python
C="$PWD/profile/results/optimization/offload-gap-001"
D="$C/supplemental-01"
NEW=$(mktemp -d "$C/supplemental-reproduction.XXXXXX")
cp "$D/study.py" "$D/confirm.py" "$D/continue_after_screen.py" "$D/report.py" "$D/audit.py" "$D/test_study.py" "$NEW/"
"$PY" -m pytest -q "$NEW/test_study.py"
"$PY" "$NEW/study.py" prepare
"$PY" "$NEW/study.py" screen > "$NEW/screen-controller.txt" 2>&1
"$PY" "$NEW/continue_after_screen.py" > "$NEW/workflow-controller.txt" 2>&1
```

The last command observes the completed original460-process screen, runs the predeclared12-process A0011 bulk addendum, freezes all nominees and potential parent/current hypotheses, then runs the independent confirmations. `confirm.py prepare` therefore includes GPU timing for that addendum and must run sequentially. In the original run the wrapper was launched while screening was active, and waited on its exact PID/starttime. It never launches concurrent GPU timing or restarts an interrupted controller. If a controller fails, inspect its manifest and logs before taking any action; completed results are not silently overwritten or treated as missing zeroes.

Each selected workload gets12 new parent-comparison processes (three balanced blocks×two implementations×two placements). A separate current-A0016 comparison runs only for parent workloads passing the predeclared nominal local gate. All potential current hypotheses, including unrun slots with p=1, remain in the frozen multiple-testing family. Generation uses full128-step trajectories with2 warmups and3×5 measured trajectories; it is more expensive than a simple prefill/decode screen. No automatic acceptance or source integration occurs.

To regenerate tables, figures and audits from the **existing completed** evidence, without GPU measurements:

```bash
MPLCONFIGDIR=/tmp/offload-gap-mpl "$PY" "$D/report.py"
"$PY" "$D/audit.py"
```

For a fresh reproduction, replace `D` with `NEW` after its workflow completes. Those commands overwrite only derived files in that reproduction's `final/` directory. Original raw samples and previous final reports are preserved. Raw paths are absolute; relocation requires remapping prefixes in copies, retaining originals and exact source hashes. The original campaign model and timing scope are unchanged: seeded random weights, GPU IDs, BF16, last-token logits, real KV, and predetermined-token generation excluding sampling. These measurements do not establish model quality or full serving latency.
