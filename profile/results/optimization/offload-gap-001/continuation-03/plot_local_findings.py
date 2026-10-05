"""Export finalized continuation findings without treating local gains as retention.

python continuation-03/plot_local_findings.py --through A0026
Reads independent R02 process blocks only; never executes GPU work.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import driver as d
from confirm import holm

METRICS = ('offload_reduction_ms', 'gap_reduction_ms', 'gpu_reduction_ms')
LABELS = ('Offload reduction', 'Gap reduction', 'Resident reduction')
COLORS = ('#20639b', '#c44432', '#777777')


def collect(through):
    rows = []
    for number in range(21, int(through[1:]) + 1):
        attempt = f'A{number:04d}'
        record = d.record(attempt)
        assert record['status'] in ('accepted', 'rejected'), (attempt, record['status'])
        source = d.C / attempt / 'R02-parent-confirmation/analysis.json'
        result = d.read(source)
        assert result['status'] == 'completed'
        assert result['fixed_gain_family'] == len(result['rows'])
        adjusted = holm([r['joint_one_sided_p'] for r in result['rows']])
        for r, p in zip(result['rows'], adjusted):
            assert math.isclose(p, r['holm_adjusted_p'], rel_tol=1e-12, abs_tol=1e-12)
            pairs = r['latencies']
            assert len(pairs) == 3
            off = [v['baseline_offload_ms'] - v['candidate_offload_ms'] for v in pairs]
            gpu = [v['baseline_gpu_ms'] - v['candidate_gpu_ms'] for v in pairs]
            values = (off, [o - g for o, g in zip(off, gpu)], gpu)
            out = dict(attempt=attempt, comparator=result['baseline'],
                       candidate_status=record['status'], mode=r['mode'],
                       batch=r['batch'], length=r['length'],
                       confirmed_local_gain=r['confirmed_local_gain'],
                       holm_adjusted_p=p, fixed_gain_family=result['fixed_gain_family'],
                       memory_savings_preserved=r['memory_savings_preserved'],
                       resolved_offload_regression=r['resolved_offload_regression'],
                       resolved_gap_regression=r['resolved_gap_regression'],
                       resolved_resident_slowdown=r['resolved_resident_slowdown'],
                       source_analysis=str(source.relative_to(d.C)),
                       source_manifest=str(Path(r['source_manifest']).relative_to(d.C)))
            for metric, samples in zip(METRICS, values):
                recomputed = d.interval(samples)
                for field in ('mean', 'lower_95', 'upper_95'):
                    assert math.isclose(recomputed[field], r['statistics'][metric][field],
                                        rel_tol=1e-10, abs_tol=1e-10)
                    out[f'{metric}_{field}'] = recomputed[field]
            rows.append(out)
    return rows


def adverse(row):
    return any(row[k] for k in ('resolved_offload_regression',
                               'resolved_gap_regression', 'resolved_resident_slowdown'))


def figures(rows, directory):
    gains = [r for r in rows if r['confirmed_local_gain']]
    guards = [r for r in rows if adverse(r)]
    for name, selected, metric_count in [('local-gains', gains, 2), ('blocking-guards', guards, 3)]:
        shapes = sorted({(r['mode'], r['batch'], r['length']) for r in selected})
        nrows = max(1, math.ceil(len(shapes) / 2))
        fig, axes = plt.subplots(nrows, 2, figsize=(13, nrows * 3.5 + 1), squeeze=False)
        for ax, shape in zip(axes.flat, shapes):
            subset = [r for r in selected if (r['mode'], r['batch'], r['length']) == shape]
            for mi in range(metric_count):
                metric = METRICS[mi]
                x = [i + (mi - (metric_count - 1) / 2) * .18 for i in range(len(subset))]
                y = [r[f'{metric}_mean'] for r in subset]
                ax.errorbar(x, y, yerr=[
                    [r[f'{metric}_mean'] - r[f'{metric}_lower_95'] for r in subset],
                    [r[f'{metric}_upper_95'] - r[f'{metric}_mean'] for r in subset]],
                    fmt='o', markersize=4, capsize=3, color=COLORS[mi], label=LABELS[mi])
            ax.axhline(0, color='black', linewidth=.7)
            labels = [f"{r['attempt']} vs {r['comparator']}\n{r['candidate_status']}" +
                      (f" · Holm p={r['holm_adjusted_p']:.4f}" if name == 'local-gains' else '')
                      for r in subset]
            ax.set_xticks(range(len(subset)), labels, fontsize=8)
            ax.set_xlim(-.55, len(subset) - .45)
            mode, batch, length = shape
            ax.set_title(f'{mode.capitalize()} · batch {batch} · context {length}' +
                         (' + 128 steps' if mode == 'generation' else ''), fontsize=10)
            ax.set_ylabel('Reduction vs named comparator (ms)')
            ax.grid(axis='y', alpha=.2)
        for ax in list(axes.flat)[len(shapes):]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        title = ('Confirmed local gains; retention is a separate decision' if name == 'local-gains'
                 else 'Blocking guards preserved alongside local gains')
        fig.suptitle(title)
        fig.text(.5, .02, '2.836B BF16 · RTX 5090 · last-token logits + real KV cache · random weights/tokens\n'
                 'Independent paired process blocks (n=3), two-sided 95% t intervals; positive = improvement.\n'
                 'Screening excluded. Fixed Holm family per attempt for gains; guards use signed intervals. '
                 'These are not cumulative speedups or final target acceptance.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .095, 1, .965))
        for extension in ('png', 'svg', 'pdf'):
            fig.savefig(directory / f'{name}.{extension}', dpi=180)
        plt.close(fig)


def main(through):
    rows = collect(through)
    directory = d.D / f'local-findings-through-{through}'
    directory.mkdir(exist_ok=True)
    d.save(directory / 'findings.json', dict(status='historical_findings', through=through,
           final_goal_accepted=False, rows=rows,
           note='All R02 nominated workloads included, including within-noise and adverse results. '
                'Plots select confirmed gains and blocking guards; CSV/JSON retain the complete nominated evidence. '
                'Comparators and confirmation families differ; effects must not be added. '
                'Parametric intervals assume independent approximately normal paired block differences.'))
    with (directory / 'findings.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figures(rows, directory)
    print(json.dumps(dict(output=str(directory), rows=len(rows),
                          local_gains=sum(r['confirmed_local_gain'] for r in rows),
                          blocking_guard_workloads=sum(adverse(r) for r in rows))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--through', required=True)
    main(parser.parse_args().through)
