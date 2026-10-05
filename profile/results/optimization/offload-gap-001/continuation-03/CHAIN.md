# Verified optimization chain through A0023

This evidence index records the verified implementation through accepted step 3.
Later candidates do not change this chain until their complete retention gates
pass. The overall all-workload near-GPU objective remains unmet.

| Accepted step | Implementation | Parent | Change | Independent evidence |
| --- | --- | --- | --- | --- |
| 0 | A0000 | — | Frozen campaign baseline | [Record](../A0000/record.json) |
| 1 | A0001 | A0000 | Lower automatic bulk cutoff to 1024 tokens | [Confirmation](../A0001/validation-v2-confirmation-analysis.json), [verdict and validation links](../A0001/record.json) |
| 2 | A0016 | A0001 | Use one automatic pipeline slot through 2048 input tokens | [Incremental confirmation](../A0016/parent-confirmation-analysis.json), [cumulative A0000 confirmation](../A0016/validation-v3-confirmation-analysis.json), [verdict](../A0016/record.json) |
| 3 | A0023 | A0016 | Read mapped host tables for bulk inputs of 1–16 tokens | [Incremental confirmation](../A0023/R02-parent-confirmation/analysis.json), [all-workload guards](../A0023/full-parent-analysis.json), [integrity and correctness audit](../A0023/final-evidence-audit.json), [verdict](../A0023/record.json) |

Exact implementation commits are in each record. The current verified A0023
source is `5f3d5958fea25f953bb91f5ef6f35880ff37fcfa`. It retains the earlier
cutoff and selective single-slot policy; mapped lookup pins the complete
3000 MiB host table, which is explicitly counted in the memory evidence.

## How recovered local gains connect to the chain

The [supplemental survey](../supplemental-01/final/REPORT.md) preserves the
historical verdicts and distinguishes gains against historical parents from
gains against the then-current A0016. It motivated two integration paths:

| Mechanism | Recovery evidence | Integration outcome through A0026 |
| --- | --- | --- |
| Mapped host lookup for small bulk inputs | A0013 improved batch-8 generation against A0016 in the supplemental survey. [A0022](../A0022/record.json) confirmed this mechanism on A0016 but regressed batch-1 generation. | [A0023](../A0023/record.json) extended the mapped range to include batch 1 and passed all retention gates. This recovery is now in accepted step 3. |
| One pipeline slot for short prefill | A0014 improved batch-8/512 prefill in the supplemental survey. [A0021](../A0021/record.json) confirmed a local gain on A0016 but regressed the batch-1 decode gap. | [A0024](../A0024/record.json) confirmed the short-prefill gain on A0023, but its batch-1 prefill latency regression failed the fixed retention gate. This recovery remains unintegrated. |
| One host staging buffer while retaining multiple GPU slots | [A0026](../A0026/record.json) separated the host working-set hypothesis from GPU lookahead. Its [independent confirmation](../A0026/R02-parent-confirmation/analysis.json) found gains at batch-4/2048 and batch-8/512 prefill against A0023. | A batch-1 generation offload slowdown failed the fixed retention gate, although the shared-host branch is inactive at batch 1. Both local gains remain preserved for a separately registered integration; A0026 is not in the accepted chain. |

A0015's nominal corrected batch-16 prefill gain failed its resident-reference
guard in the supplemental survey. It is not an accepted offload improvement.

[A0025](../A0025/record.json), which precomputed pipeline tensor views, had no
corrected confirmed gain and failed the batch-1 decode resident guard. It was
reverted. A0026's retained local findings are offload reductions of 2.8812 ms
(95% CI 2.1768–3.5856 ms) at batch-4/2048 prefill and 2.4720 ms
(2.2863–2.6578 ms) at batch-8/512 prefill. Their matched gap reductions are
2.8457 and 2.6696 ms; Holm-adjusted joint p-values are 0.01817 and 0.01371
over the frozen 12-workload family. Batch-1 generation offload instead slowed
by 7.8733 ms (3.4524–12.2942 ms); its gap and resident shifts were unresolved.
These comparisons identify an integration opportunity, not its cause or a
retained cumulative benefit. The [raw-evidence audit](../A0026/evidence-integrity.json)
checks all 64 screen and 144 independent confirmation processes.

A0023's independently confirmed incremental offload reductions are 27.9865 ms
for batch-1 generation, 30.8881 ms for batch-8 generation, and 0.1010 ms for
batch-16/context-2048 decode. These refer to matched A0016 comparisons; the
linked confirmation contains paired intervals, gap reductions and corrected
decisions. Generation uses a 2048-token prefix plus 128 predetermined decode
tokens. These effects must not be added to results measured at other workloads.

## Figure inputs and remaining evidence

The [complete ledger through A0026](ledger-through-A0026/LEDGER.md) preserves
all 26 finalized attempts and the A0000 baseline, with full record snapshots
in JSON and an index in CSV. Active candidates remain outside that finalized
snapshot until their verdict is recorded.

Use accepted steps 0–3 for the cumulative implementation figure. Use separate
workload panels for the recovered, nonretained local gains, retaining their
named comparator, intervals and adverse findings. Do not splice the fastest
points from different candidates into a final implementation.

Continuation findings through A0026 are now exported as paired
[local-gain panels](local-findings-through-A0026/local-gains.pdf) and
[blocking-guard panels](local-findings-through-A0026/blocking-guards.pdf), with
PNG/SVG companions. The complete 55 nominated workload comparisons, including
within-noise results, are in [CSV](local-findings-through-A0026/findings.csv) and
[JSON](local-findings-through-A0026/findings.json). Regenerate with
`python profile/results/optimization/offload-gap-001/continuation-03/plot_local_findings.py --through A0026`.
The script recomputes paired intervals and checks the frozen Holm adjustment;
it does not merge evidence from different comparators or revise verdicts.

[A0023 gap diagnostics](../A0023/gap-diagnostics/gaps.json) and the accompanying
[CSV](../A0023/gap-diagnostics/gaps.csv) and
[plot](../A0023/gap-diagnostics/gaps.pdf) cover all 16 workloads from its formal
parent-paired validation. Regenerate them with [plot_candidate_gaps.py](plot_candidate_gaps.py).
They reuse candidate-selection evidence and therefore do not establish the
final independent near-GPU acceptance required by [GOAL.md](../../../../../GOAL.md).

Final deliverables still require the complete attempt ledger, the remaining
scaling/throughput/memory/history figures, fresh cumulative A0000 comparisons,
and independent all-16 confirmation of the user-approved latency tolerance for
the final retained source. This index is not a final publication or completion
claim.
