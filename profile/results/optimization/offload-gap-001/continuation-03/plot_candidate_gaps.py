"""Export complete-matrix gap diagnostics from audited independent process blocks.

Usage: python continuation-03/plot_candidate_gaps.py --attempt A0023
Existing selection/validation data are diagnostic, not independent final acceptance.
"""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from scipy.stats import t

import driver as d


def interval(values):
    n = len(values)
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(n)
    delta = float(t.ppf(.975, n - 1)) * se
    return mean, mean - delta, mean + delta, se


def export(attempt, data_only=False):
    directory = d.C / attempt
    analysis = d.read(directory / 'full-parent-analysis.json')
    assert analysis['status'] == 'audited'
    assert len(analysis['rows']) == 16
    assert {(r['mode'], r['batch'], r['length']) for r in analysis['rows']} == set(d.WORKLOADS)
    rows = []
    for r in analysis['rows']:
        pairs = r['latencies']
        assert len(pairs) == 3
        gpu = [p['candidate_gpu_ms'] for p in pairs]
        offload = [p['candidate_offload_ms'] for p in pairs]
        assert all(math.isfinite(x) and x > 0 for x in gpu + offload)
        gaps = [o - g for o, g in zip(offload, gpu)]
        overheads = [o / g - 1 for o, g in zip(offload, gpu)]
        residuals = [o - g - max(.1, .01 * g) for o, g in zip(offload, gpu)]
        gap, low, high, _ = interval(gaps)
        overhead, olow, ohigh, _ = interval(overheads)
        residual, rlow, rhigh, se = interval(residuals)
        upper = residual + float(t.ppf(1 - .05 / 16, 2)) * se
        rows.append(dict(
            mode=r['mode'], batch=r['batch'], length=r['length'],
            gpu_ms=statistics.mean(gpu), offload_ms=statistics.mean(offload),
            gap_ms=gap, gap_lower_95_ms=low, gap_upper_95_ms=high,
            relative_overhead=overhead, relative_overhead_lower_95=olow,
            relative_overhead_upper_95=ohigh,
            tolerance_at_mean_gpu_ms=max(.1, .01 * statistics.mean(gpu)),
            residual_ms=residual, residual_lower_95_ms=rlow,
            residual_upper_95_ms=rhigh,
            residual_upper_simultaneous_95_ms=upper,
            diagnostic_bound_within_target=upper <= 0,
            independent_blocks=3, source_manifest=r['source_manifest']))
    rows.sort(key=lambda r: (['prefill', 'decode', 'generation'].index(r['mode']), r['batch'], r['length']))
    out = directory / 'gap-diagnostics'
    out.mkdir(exist_ok=True)
    note = ('Diagnostic reuse of retention data, not independent final goal acceptance. '
            'Error bars: paired process-block two-sided 95% t intervals (n=3). '
            'Target residual is computed per pair as offload-GPU-max(0.1ms,0.01*GPU); '
            'its one-sided simultaneous 95% upper bound uses Bonferroni over all16 workloads. '
            'Parametric t intervals assume independent approximately normal block differences.')
    (out / 'gaps.json').write_text(json.dumps(dict(attempt=attempt, status='diagnostic_only',
        goal_accepted=False, note=note, rows=rows), indent=2) + '\n')
    with (out / 'gaps.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(attempt=attempt, output=str(out),
        diagnostic_bounds_within=sum(r['diagnostic_bound_within_target'] for r in rows),
        not_within=[(r['mode'], r['batch'], r['length']) for r in rows if not r['diagnostic_bound_within_target']],
        goal_accepted=False)
    if data_only:
        print(json.dumps(result))
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name, value, low, high, scale, ylabel in (
            ('gaps', 'gap_ms', 'gap_lower_95_ms', 'gap_upper_95_ms', 1,
             'Latency gap (ms); lower is better'),
            ('relative-overhead', 'relative_overhead', 'relative_overhead_lower_95',
             'relative_overhead_upper_95', 100, 'Relative overhead (%); lower is better')):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [3, 3, 1.5]})
        for ax, mode in zip(axes, ('prefill', 'decode', 'generation')):
            subset = [r for r in rows if r['mode'] == mode]
            x = list(range(len(subset)))
            ax.errorbar(x, [scale*r[value] for r in subset], yerr=[
                [scale*(r[value]-r[low]) for r in subset],
                [scale*(r[high]-r[value]) for r in subset]],
                fmt='o', capsize=3, color='#20639b', label='Offload vs GPU (95% CI)')
            tolerances = [r['tolerance_at_mean_gpu_ms'] if name == 'gaps' else
                          100*r['tolerance_at_mean_gpu_ms']/r['gpu_ms'] for r in subset]
            ax.scatter(x, tolerances, marker='_', s=180,
                       color='#c44432', label='Tolerance at mean GPU')
            ax.axhline(0, color='grey', linewidth=.8)
            ax.set_xticks(x, [f"{r['batch']}/{r['length']}" for r in subset], rotation=55, ha='right')
            ax.set_title(mode.capitalize() + (' (prefix + 128 steps)' if mode == 'generation' else ''))
            ax.set_xlabel('Batch / context length')
            ax.set_ylabel(ylabel)
            ax.grid(axis='y', alpha=.25)
        axes[0].legend(fontsize=8)
        fig.suptitle(f'{attempt}: matched offload versus folded all-GPU, complete 16-workload matrix')
        fig.text(.5, .015, '2.836B BF16 · RTX 5090 · last-token logits + real KV cache · random weights/tokens\n'
                 'Three independent paired blocks; screening excluded. Diagnostic retention data; final goal not accepted.',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .09, 1, .93))
        for extension in ('png', 'svg', 'pdf'):
            fig.savefig(out / f'{name}.{extension}', dpi=180)
        plt.close(fig)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--data-only', action='store_true',
                        help='Export CSV/JSON now; defer plot rendering during live timing')
    args = parser.parse_args()
    export(args.attempt, data_only=args.data_only)
