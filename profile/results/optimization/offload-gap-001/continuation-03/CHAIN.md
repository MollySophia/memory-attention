# Verified optimization chain through A0028

This evidence index records the verified implementation through accepted step 4.
Later candidates do not change this chain until their complete retention gates
pass. The overall all-workload near-GPU objective remains unmet.

| Accepted step | Implementation | Parent | Change | Independent evidence |
| --- | --- | --- | --- | --- |
| 0 | A0000 | — | Frozen campaign baseline | [Record](../A0000/record.json) |
| 1 | A0001 | A0000 | Lower automatic bulk cutoff to 1024 tokens | [Confirmation](../A0001/validation-v2-confirmation-analysis.json), [verdict and validation links](../A0001/record.json) |
| 2 | A0016 | A0001 | Use one automatic pipeline slot through 2048 input tokens | [Incremental confirmation](../A0016/parent-confirmation-analysis.json), [cumulative A0000 confirmation](../A0016/validation-v3-confirmation-analysis.json), [verdict](../A0016/record.json) |
| 3 | A0023 | A0016 | Read mapped host tables for bulk inputs of 1–16 tokens | [Incremental confirmation](../A0023/R02-parent-confirmation/analysis.json), [all-workload guards](../A0023/full-parent-analysis.json), [integrity and correctness audit](../A0023/final-evidence-audit.json), [verdict](../A0023/record.json) |
| 4 | A0028 | A0023 | Share one host staging allocation at 4096–16384 input tokens, preserving four GPU slots | [Incremental confirmation](../A0028/R02-parent-confirmation/analysis.json), [all-workload guards](../A0028/full-parent-analysis.json), [integrity and correctness audit](../A0028/final-evidence-audit.json), [verdict](../A0028/record.json) |

Exact implementation commits are in each record. The current verified A0028
source is `6cc9354760ebde3bab9a90996a7713a936dfb439`. It retains the earlier
cutoff, selective single-slot and mapped policies, adding shared host staging
without reducing GPU lookahead. Mapped lookup pins the complete
3000 MiB host table, which is explicitly counted in the memory evidence.

## How recovered local gains connect to the chain

The [supplemental survey](../supplemental-01/final/REPORT.md) preserves the
historical verdicts and distinguishes gains against historical parents from
gains against the then-current A0016. It motivated two integration paths:

| Mechanism | Recovery evidence | Integration outcome through A0028 |
| --- | --- | --- |
| Mapped host lookup for small bulk inputs | A0013 improved batch-8 generation against A0016 in the supplemental survey. [A0022](../A0022/record.json) confirmed this mechanism on A0016 but regressed batch-1 generation. | [A0023](../A0023/record.json) extended the mapped range to include batch 1 and passed all retention gates. This recovery is now in accepted step 3. |
| One pipeline slot for short prefill | A0014 improved batch-8/512 prefill in the supplemental survey. [A0021](../A0021/record.json) confirmed a local gain on A0016 but regressed the batch-1 decode gap. | [A0024](../A0024/record.json) confirmed the short-prefill gain on A0023, but its batch-1 prefill latency regression failed the fixed retention gate. The short-prefill opportunity is now recovered through A0028 shared host staging; the one-GPU-slot mechanism itself remains unretained. |
| One host staging buffer while retaining multiple GPU slots | [A0026](../A0026/record.json) separated the host working-set hypothesis from GPU lookahead. Its [independent confirmation](../A0026/R02-parent-confirmation/analysis.json) found gains at batch-4/2048 and batch-8/512 prefill against A0023. | A batch-1 generation offload slowdown failed the fixed retention gate, although the shared-host branch is inactive at batch 1. A0028 separately extended the shared-host range and passed all gates, confirming the short-prefill gain as step 4. Its batch-4 gain remains nominal after correction. A0026 itself is not in the accepted chain. |

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

[A0027](../A0027/record.json) tested caller-stream mapped gather for one token.
Its 64-process screen and 84-process independent confirmation found no
corrected gain. Batch-8 generation slowed by 0.2363 ms (95% CI
0.0727–0.4000 ms), failing the signed regression guard; the gap and resident
changes remained unresolved. This does not identify the cause: the one-token
branch is inactive at batch 8. The candidate was reverted after its
[148-result audit](../A0027/evidence-integrity.json). [A0028](../A0028/record.json) separately extended A0026's shared-host range
to 16384 tokens and is now retained after all 288 performance results and six
bit-exact full-model comparisons passed audit. Its batch-8/512 prefill offload
reduction is 2.4604 ms (95% CI 2.2925–2.6282); gap reduction is 2.3024 ms
(1.7967–2.8080), with Holm10 adjusted p=0.01298 against A0023. No resolved
regression or memory loss was found across all 16 workloads.

A0023's independently confirmed incremental offload reductions are 27.9865 ms
for batch-1 generation, 30.8881 ms for batch-8 generation, and 0.1010 ms for
batch-16/context-2048 decode. These refer to matched A0016 comparisons; the
linked confirmation contains paired intervals, gap reductions and corrected
decisions. Generation uses a 2048-token prefix plus 128 predetermined decode
tokens. These effects must not be added to results measured at other workloads.

## Latest findings and figures

[A0030](../A0030/record.json) is in progress on A0028, testing per-forward
resolution of offload device, offloader and layer metadata plus closure-local
callback state. Source `a9e5bda` passed 56 regression tests and the standalone
zero-difference correctness check. Its committed screen covers all16 workloads
in64 fresh parent/candidate placement processes. Six independent confirmation
blocks were predeclared before measurements, motivated by the post-A0029
between-process generation variance; the inner sampling and gain/regression
gates are unchanged. This is not a retained step or a confirmed gain. The
A0029 verdict stays fixed. [Helper tests](A0030-helper-tests.txt) and
[historical recomputation audit](A0030-helper-history-audit.json) verify that
variable block counts preserve all prior continuation statistics. Scaling
exports now derive sample count and degrees of freedom from the frozen plan;
[plot-data tests](A0030-plot-data-tests.txt) cover six-block completeness and
unchanged A0023/A0028 source data. No new figures are rendered during timing.

[A0029](../A0029/record.json) tested alignment-specialized compiled mapped
launchers on A0028. All208 screen/confirmation results were audited; no gain
survived fixed Holm12 correction, and no regression was resolved. Batch-1
generation offload reduction4.4820ms had95%CI[-11.7042,20.6683]; the positive
gap interval alone is insufficient. Batch-4 decode's nominal dual gain failed
correction (p=0.08269). The verdict is within_noise; source commitf4cead8 was
reverted separately byf06bfdf. The verified implementation remains A0028.

The [complete ledger through A0029](ledger-through-A0029/LEDGER.md) preserves
all29 finalized attempts and A0000, with exact record snapshots in JSON and an
index in CSV. There are four retained optimization steps.

The [attempt-history figure](history-through-A0029/attempt-history-screen_v1_w3_n5_r1.pdf)
shows30 baseline/candidate records on four primary-workload panels, including
A0029 as within_noise. Its [CSV](history-through-A0029/history-source.csv) and
[JSON](history-through-A0029/history-source.json) contain120 measured points,
sample standard deviations and the accepted incumbent. The incumbent after
A0028 remains A0028; screening points are not confirmed cumulative gains.
Reproduce with `python profile/results/optimization/offload-gap-001/plot_history.py --output profile/results/optimization/offload-gap-001/continuation-03/history-through-A0029 --through A0029 --history-only`.

Continuation findings through A0029 are exported as
[confirmed local gains](local-findings-through-A0029/local-gains.pdf) and
[blocking guards](local-findings-through-A0029/blocking-guards.pdf), with PNG/SVG
companions. All84 nominated workload comparisons, including within-noise and
adverse evidence, are in [CSV](local-findings-through-A0029/findings.csv) and
[JSON](local-findings-through-A0029/findings.json). Nine confirmed local gains
include A0028's integrated short-prefill improvement. A0029 has no corrected
gain; its nominal effects remain in the data, not the confirmed-gain panels.
The standalone exporter recomputes paired intervals and fixed Holm corrections:
`python profile/results/optimization/offload-gap-001/continuation-03/plot_local_findings.py --through A0029`.
Effects have explicitly named comparators and must not be added across sources.

Current retained A0028 has complete [latency scaling](../A0028/workload-diagnostics/latency_ms-scaling.pdf),
[throughput scaling](../A0028/workload-diagnostics/tokens_per_second-scaling.pdf),
[generation](../A0028/workload-diagnostics/generation.pdf), and
[GPU/host memory](../A0028/workload-diagnostics/memory.pdf) figures with PNG/SVG
companions and [CSV](../A0028/workload-diagnostics/workloads.csv) /
[JSON](../A0028/workload-diagnostics/workloads.json) sources. These32 placement
rows use96 audited formal processes. Throughput is derived per process before
estimating uncertainty; generation counts output tokens and includes prefix
time. Host RSS includes runtime/loading state; pinned storage is separate.
Reproduce with `python profile/results/optimization/offload-gap-001/continuation-03/plot_workload_scaling.py --attempt A0028`.

[A0028 absolute gap](../A0028/gap-diagnostics/gaps.pdf) and
[relative overhead](../A0028/gap-diagnostics/relative-overhead.pdf) figures cover
all16 workloads with paired intervals and tolerance markers, plus PNG/SVG,
[CSV](../A0028/gap-diagnostics/gaps.csv) and
[JSON](../A0028/gap-diagnostics/gaps.json). Only6/16 diagnostic simultaneous
residual bounds meet tolerance despite15/16 favorable means. All7 prefill
points, batch-1/batch-4 decode and batch-1 generation remain unproven. Reproduce
with `python profile/results/optimization/offload-gap-001/continuation-03/plot_candidate_gaps.py --attempt A0028`;
`--data-only` defers rendering while timing is live. These plots reuse retention
data and do not establish final independent acceptance.

Previous A0023 and through-A0026 snapshots remain preserved in their original
directories. Every latest figure was visually checked; local-gain labels were
split across lines to avoid overlap as new attempts were added.

## Remaining final evidence

Use accepted steps0–4 for the cumulative implementation figure, with a common
frozen baseline and comparable protocols. Do not splice different candidates'
best points into a fictional implementation. The final publication still needs
fresh cumulative A0000 comparisons, updated final-source cumulative figures,
and independent confirmation of the user-approved tolerance at every workload.

The [independent final verification workflow](FINAL_VERIFICATION.md) has a
standalone runner/auditor and12 CPU-tested statistical/evidence guards. No real
final source/count plan has been frozen or executed by that helper yet. The
all-workload near-GPU goal remains active and unproven.
