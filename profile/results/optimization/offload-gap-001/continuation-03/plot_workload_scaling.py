"""Plot an audited retained candidate's complete workload matrix from raw results.

python continuation-03/plot_workload_scaling.py --attempt A0023
No measurements are launched. These retention data are not final goal acceptance.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import driver as d

VARIANTS = ('ma_gpu', 'ma_offload')
COLORS = ('#777777', '#20639b')
LABELS = ('Folded all-GPU', 'Offload')
MEMORY = ('gpu_peak_allocated_bytes', 'host_rss_bytes', 'offload_pinned_bytes',
          'cpu_table_bytes', 'cpu_table_pinned_bytes', 'offload_gpu_buffer_bytes',
          'kv_cache_storage_bytes')
CAPTION = ('2.836B BF16 · RTX 5090 · last-token logits + real KV cache · random weights/tokens\n'
           'Three independent process blocks; two-sided 95% t intervals (df=2). Screening excluded.\n'
           'Retained-source diagnostic data; not independent final target acceptance.')


def load(attempt):
    analysis = d.read(d.C / attempt / 'full-parent-analysis.json')
    assert analysis['status'] == 'audited'
    assert d.record(attempt)['status'] == 'accepted'
    assert {(r['mode'], r['batch'], r['length']) for r in analysis['rows']} == set(d.WORKLOADS)
    assert len(analysis['rows']) == 16
    rows, raw, envs = [], [], set()
    for r in analysis['rows']:
        shape = (r['mode'], r['batch'], r['length'])
        manifest_path = Path(r['source_manifest'])
        manifest = d.read(manifest_path)
        assert manifest['status'] == 'completed'
        for variant in VARIANTS:
            jobs = sorted([j for j in manifest['jobs'] if j['implementation'] == 'candidate'
                           and j['variant'] == variant and
                           (j['mode'], j['batch'], j['length']) == shape], key=lambda j:j['block'])
            assert [j['block'] for j in jobs] == [1, 2, 3]
            values = {k: [] for k in ('latency_ms', 'tokens_per_second', *MEMORY)}
            for job in jobs:
                assert job['status'] == 'completed' and job['attempt'] == attempt
                path = manifest_path.parent / (job['name'] + '.json')
                env, result = d.audit_job(job, path)
                envs.add(env)
                mean = result['mean_ms']
                column = 'candidate_gpu_ms' if variant == 'ma_gpu' else 'candidate_offload_ms'
                assert mean == r['latencies'][job['block']-1][column]
                # Generation reports output-token throughput, including prefix
                # time in its denominator; prefill reports input-token throughput.
                tokens = shape[1] * (shape[2] if shape[0] == 'prefill' else
                                    128 if shape[0] == 'generation' else 1)
                values['latency_ms'].append(mean)
                values['tokens_per_second'].append(tokens * 1000 / mean)
                for key in MEMORY:
                    values[key].append(result['memory_after'][key])
                raw.append(dict(mode=shape[0], batch=shape[1], length=shape[2],
                                variant=variant, block=job['block'],
                                raw_path=str(path.relative_to(d.C)),
                                latency_ms=mean, memory=result['memory_after']))
            out = dict(attempt=attempt, mode=shape[0], batch=shape[1], length=shape[2],
                       variant=variant, blocks=3, source_manifest=str(manifest_path.relative_to(d.C)))
            for key, samples in values.items():
                interval = d.interval(samples)
                for field in ('mean', 'lower_95', 'upper_95'):
                    out[f'{key}_{field}'] = interval[field]
            rows.append(out)
    assert len(envs) == 1 and len(raw) == 96
    return rows, raw


def save(fig, directory, name, note=''):
    fig.text(.5, .015, CAPTION + ('\n' + note if note else ''), ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .13, 1, .94))
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(directory / f'{name}.{extension}', dpi=180)
    plt.close(fig)


def errorbars(ax, subset, key, x, color, label, scale=1):
    y = [r[key+'_mean']/scale for r in subset]
    lower = [(r[key+'_mean']-r[key+'_lower_95'])/scale for r in subset]
    upper = [(r[key+'_upper_95']-r[key+'_mean'])/scale for r in subset]
    ax.errorbar(x, y, yerr=[lower, upper], marker='o', markersize=4,
                capsize=3, color=color, label=label)
    ax.grid(axis='y', alpha=.2)


def plots(rows, directory, attempt):
    for metric, ylabel in [('latency_ms', 'Latency (ms)'), ('tokens_per_second', 'Tokens / second')]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for mi, mode in enumerate(('prefill', 'decode')):
            for si, sweep in enumerate(('batch', 'length')):
                ax = axes[mi, si]
                for variant, color, label in zip(VARIANTS, COLORS, LABELS):
                    subset = sorted([r for r in rows if r['mode'] == mode and r['variant'] == variant
                                     and (r['length'] == 2048 if sweep == 'batch' else r['batch'] == 8)],
                                    key=lambda r:r[sweep])
                    assert len(subset) == 4
                    x = [r[sweep] for r in subset]
                    errorbars(ax, subset, metric, x, color, label)
                ax.set_xlabel('Batch size' if sweep == 'batch' else 'Context length')
                ax.set_ylabel(ylabel)
                ax.set_xticks(x)
                ax.set_title(f'{mode.capitalize()} · ' + ('context 2048' if sweep == 'batch' else 'batch 8'))
        axes[0, 0].legend()
        fig.suptitle(f'{attempt}: workload scaling')
        save(fig, directory, metric+'-scaling')
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, metric, ylabel in zip(axes, ('latency_ms', 'tokens_per_second'),
                                  ('Total generation latency (ms)', 'Output tokens / second')):
        for variant, color, label in zip(VARIANTS, COLORS, LABELS):
            subset = sorted([r for r in rows if r['mode']=='generation' and r['variant']==variant], key=lambda r:r['batch'])
            assert len(subset) == 2
            errorbars(ax, subset, metric, [r['batch'] for r in subset], color, label)
        ax.set_xticks([1, 8]); ax.set_xlabel('Batch size'); ax.set_ylabel(ylabel)
    axes[0].legend()
    fig.suptitle(f'{attempt}: generation, 2048-token prefix + 128 growing decode steps')
    save(fig, directory, 'generation', 'Output-token throughput includes prefix time; predetermined tokens, not a quality evaluation.')
    metrics = [('gpu_peak_allocated_bytes', 'Peak allocated GPU memory'),
               ('host_rss_bytes', 'Host process RSS after workload'),
               ('offload_pinned_bytes', 'Pinned host table + staging buffers')]
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    order = [(m,b,l) for m in ('prefill','decode','generation')
             for b,l in (d.SHAPES if m!='generation' else [(1,2048),(8,2048)])]
    for ax, (metric, title) in zip(axes, metrics):
        for variant, color, label in zip(VARIANTS, COLORS, LABELS):
            indexed = {(r['mode'],r['batch'],r['length']):r for r in rows if r['variant']==variant}
            subset = [indexed[w] for w in order]
            errorbars(ax, subset, metric, list(range(16)), color, label, scale=2**20)
        ax.set_xticks(range(16), [f'{m[:3]}\n{b}/{l}' for m,b,l in order], fontsize=8)
        ax.set_ylabel('MiB'); ax.set_title(title)
    axes[0].legend()
    fig.suptitle(f'{attempt}: complete-matrix GPU and host memory')
    save(fig, directory, 'memory', 'Host RSS includes runtime and retained loading state. Pinned table is counted once; not added to RSS.')


def main(attempt):
    rows, raw = load(attempt)
    out = d.C / attempt / 'workload-diagnostics'
    out.mkdir(exist_ok=True)
    d.save(out / 'workloads.json', dict(status='retained_source_diagnostics', attempt=attempt,
           final_goal_accepted=False, rows=rows, raw_processes=raw,
           note=CAPTION+' Throughput is computed per process block before summarizing; generation counts 128 output tokens per sequence.'))
    with (out / 'workloads.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    plots(rows, out, attempt)
    print(f'Exported {len(rows)} placement/workload rows from {len(raw)} audited processes to {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--attempt', required=True)
    main(parser.parse_args().attempt)
