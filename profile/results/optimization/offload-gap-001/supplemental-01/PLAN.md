# Supplemental shape survey 01

Authorized by the user's request to recheck optimization opportunities missed by the primary-shape screen. This is a supplementary evaluation of 18 existing frozen implementations, not new optimization attempts. Historical A0001/A0016 acceptance and all earlier reports remain unchanged. Current execution source remains A0016.

The exact source SHAs, per-attempt shapes, process commands and cost estimates are frozen in `plan.json` and each `A00NN/screen/manifest.json` before measurement. The 115 workload comparisons require 460 isolated sequential processes, estimated 10724 seconds (about three hours) from previous measured screen setup and latency, with larger lengths extrapolated for planning. Estimates are not timeouts.

## Coverage

- Pipeline candidates: prefill b4/2048, b8/512, b8/4096 and b8/8192. For dedup/fused-dedup/write-combined candidates A0009/A0015/A0017, b8/512 is below their 8192-token activation threshold and excluded.
- Pipeline candidates also receive a b8/8192 decode control, because cached prefix buffers can indirectly affect decode. They receive b1/b8 generation; A0009/A0015/A0017 receive b8 generation only, since their new path is inactive for b1.
- Tiny bulk candidates A0002/A0005: decode b4/2048 and b8 at 512/4096/8192, plus b1/b8 generation.
- Mapped bulk A0013: decode b8 at 512/4096/8192 and b8 generation; b4/b1 lie below its activation threshold.
- Existing batch16 screens are reviewed for newly eligible local gains. Active-path b16 prefill is eligible for pipeline candidates; b16 decode is eligible for A0011 (ID staging) and A0013 (mapped bulk). Those observations nominate fresh confirmation only and do not count as independent confirmation evidence.
- A0001 and A0016 already have complete accepted-source matrix/generation evidence and are not rerun as new candidates. Parent/current references are nevertheless freshly measured for the supplemental comparisons.

This targets previously untested changed paths within the established workload matrix; it is not exhaustive over arbitrary models, hardware, shapes, or combinations of rejected changes. Unchanged-path matrix cells omitted from the survey are not asserted to be equivalent. No quantization, altered tokens, cached lookups or model arithmetic changes are introduced.

## Fixed measurement and selection rules

All model dimensions, BF16, seeded random weights, input generation, output scope, real KV behavior, GPU input location, sequential GPU execution and raw telemetry match the original campaign. Prefill/decode screening uses 3 warmups and 5 measurements. Generation uses its established 2 warmups and 3 rounds of 5 full trajectories (prefix2048 +128 predetermined token steps, no sampling).

Every comparison measures parent/candidate and offload/resident afresh, with balanced order across workloads. Each candidate is the original frozen implementation, compared first with its actual historical parent. Source hashes, configuration, samples, environment and memory accounting are audited; failure/OOM is preserved and stops the controller for review rather than being converted to zero or silently skipped.

Every new screen point with positive mean offload reduction AND positive mean gap reduction, with positive candidate GPU savings, is nominated. The same sign rule applies to the eligible historical batch16 points above. There is no top-K limit, minimum speedup threshold, second screen or extension until favorable. No statistically confirmed claim is made from a screen, even if a generation screen contains multiple within-process rounds.

Nominees receive exactly three fresh balanced independent process blocks for both placements (12 processes per workload), excluding all screen observations. For an original parent other than A0016, a separate three-block comparison against current A0016 is also required before calling the result useful for the current implementation. Formal prefill/decode uses10 warmups/3×10 measurements; generation retains2 warmups/3×5 trajectories. Each stage's command manifest and measured-cost plan precede execution.

Report every selected point, including unfavorable and inconclusive confirmations. Report paired Student-t 95% intervals over the three independent block differences. For multiplicity, use a one-sided conjunction p-value max(p_offload,p_gap) per point/comparator and Holm correction at0.05 over the entire preselected confirmation family. Also require both nominal two-sided95% lower bounds >0, no resolved resident slowdown and positive GPU savings. n=3 and the unverified symmetric-difference assumption remain explicit limitations. Multiplicity correction does not remedy measurement or distribution assumptions.

A confirmed local performance result is not automatic retention: full applicable correctness, workload regressions and integration with A0016 would still be required to change the retained implementation. A0014's established primary regression remains disclosed regardless of any local gain.

## Execution

Use `/home/molly/miniconda3/envs/fla-bench/bin/python`. `study.py prepare` was run once to create immutable initial plans; do not rerun it in this directory. After committing the driver and plans, `study.py screen` runs all18 plans sequentially and emits per-candidate audited summaries. Never restart the running controller. Further independent-confirmation commands will be recorded alongside their manifests. Historical artifacts and frozen worktrees must remain unchanged.

Confirmation execution detail, fixed while the first candidate's screen is still running: `confirm.py prepare` freezes every nominee and all potential parent/current hypotheses only after the complete screen. `confirm.py run` measures all nominated parent comparisons. It runs a current-A0016 comparison only for a workload whose parent comparison passes the nominal dual95% intervals, no resolved resident slowdown and memory gate. Unrun current slots remain in the originally fixed multiplicity family with p=1, so this gate cannot shrink the correction denominator. The separate runner excludes all screen data, including generation rounds, from inferential confirmation. `continue_after_screen.py` watches the exact current screen PID/starttime, never restarts it, and executes those two stages sequentially only after successful completion. Six driver tests passed before this wrapper was started.

## Coverage correction before confirmation: A0011 bulk decode

During source/coverage review while the original screen was still running, A0011 was found to affect bulk ID staging as well as pipeline staging. The original460-process plan had only its b8/8192 decode control. `bulk-coverage-addendum.json` freezes the other three previously unmeasured decode cells (b4/2048, b8/512, b8/4096):12 new paired processes, about four minutes of additional setup/timing. This is a source-coverage correction, not selection from observed results at those cells. Original plans and observations are preserved.

The original screen controller continues unchanged. After it completes, `confirm.py prepare` first executes this committed addendum sequentially, then freezes nominations from the complete118-point/472-process supplemental screen plus eligible historical b16 evidence. It now includes this GPU measurement stage; do not run it concurrently with another GPU workload. Nomination, independent-block counts, gates and the full-family multiplicity rule remain unchanged. The final report and audit explicitly distinguish the original plan from this addendum.

## Confirmation freeze and preparation recovery

All472 fresh screen processes completed successfully. Confirmation preparation encountered a metadata-path assertion: A0001's historical b16 reference was originally executed in the main repository at the correct recorded SHA/hash, before its frozen worktree was created. The strict module-path audit now compares with the recorded command/cwd path while still requiring exact frozen source SHA/hash, configuration, samples and memory. Eight driver tests pass, including rejection of an unrelated loaded module path; all historical b16 references pass. No model timing failed or was repeated. The initial preparation log and its unexecuted A0002 plans are preserved under `preparation-failure-01/`.

The recovered preparation freezes66 nominated workloads and112 potential parent/current hypotheses. There are792 fresh parent-confirmation processes, estimated6.2494hours; up to552 current-A0016 processes are conditional on the fixed nominal parent gate. Unrun current slots remain p=1 in the112-member family. This larger effort follows the original rule to confirm every positive dual-sign nominee without a top-K cutoff. `resume_confirmation.py` launches only the unstarted confirmation phase and preserves the earlier controller record. All screening data, nomination/statistical rules and sample counts remain unchanged.
