"""Validate raw matrix results and export CSV/JSON and standalone figures.

Incomplete points are retained with null measurements. Error bars span round
means (descriptive range, not confidence intervals). Never infer speedup from
unpaired placements. Re-run after the matrix finishes for complete figures.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def collect(run):
    manifest = json.loads((run/'manifest.json').read_text())
    rows = []
    for job in manifest['jobs']:
        row = {k: job[k] for k in ('name', 'mode', 'variant', 'batch', 'length', 'status')}
        row.update(median_ms=None, round_min_ms=None, round_max_ms=None,
                   tokens_per_second=None, gpu_peak_allocated_gib=None,
                   gpu_peak_reserved_gib=None, host_rss_gib=None, pinned_mib=None,
                   kv_gib=None, sample_count=None)
        if job['status'] == 'completed':
            data = json.loads((run/(job['name']+'.json')).read_text())
            assert data['status'] == 'completed', job['name']
            assert data['protocol_version'] == 'paper_v1', job['name']
            assert data['source']['git_commit']['stdout'].strip() == manifest['candidate_sha'], job['name']
            config = data['config']
            assert (config['warmup'], config['rounds'], config['repeats']) == (30, 5, 30), job['name']
            assert (config['batch_size'], config['seq_len'], config['context_len']) == (job['batch'], job['length'], job['length'])
            expected_config = dict(num_layers=24, hidden_size=2048, num_heads=32,
                                   num_kv_heads=32, intermediate_size=5632, vocab_size=32000,
                                   seed=1234, logits_to_keep=1, prefill_workload='inference')
            assert all(config[k] == v for k,v in expected_config.items()), job['name']
            assert data['env']['gpu'] == 'NVIDIA GeForce RTX 5090', job['name']
            assert len(data['results']) == 1
            result = data['results'][0]
            assert (result['mode'], result['variant']) == (job['mode'], job['variant'])
            samples = result['samples_ms']
            assert len(samples) == 5 and all(len(r) == 30 for r in samples), job['name']
            assert all(math.isfinite(x) and x > 0 for r in samples for x in r)
            means = [statistics.mean(r) for r in samples]
            assert means == result['round_ms']
            assert statistics.median(means) == result['median_ms']
            work_tokens = job['batch'] * (job['length'] if job['mode']=='prefill' else
                                         job['length']+128 if job['mode']=='generation' else 1)
            assert math.isclose(result['tokens_per_second'], work_tokens*1000/result['median_ms'])
            mem = result['memory_after']
            row.update(median_ms=result['median_ms'], round_min_ms=min(means), round_max_ms=max(means),
                       tokens_per_second=result['tokens_per_second'],
                       gpu_peak_allocated_gib=mem['gpu_peak_allocated_bytes']/2**30,
                       gpu_peak_reserved_gib=mem['gpu_peak_reserved_bytes']/2**30,
                       host_rss_gib=mem['host_rss_bytes']/2**30,
                       pinned_mib=mem['offload_pinned_bytes']/2**20,
                       kv_gib=mem['kv_cache_storage_bytes']/2**30, sample_count=150)
        rows.append(row)
    return manifest, rows


def plot(rows, output, partial):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    variants = ('ma_offload', 'ma_gpu', 'ma_gpu_unfolded')
    colors = ('#0072B2', '#D55E00', '#009E73')
    caption = ('2.836B random parameters · BF16 · RTX 5090 · paper_v1\n'
               'Last-token logits + KV cache; fixed-context decode. Round-mean range, not CI.\n'
               'Baseline placement: ma_offload; resident/folding references shown separately. No quality claim.')

    def finish(fig, name):
        fig.suptitle(('PARTIAL — ' if partial else '')+name.replace('_', ' '))
        fig.text(.5, .01, caption.replace('fixed-context decode', 'growing-context decode') if name == 'growing_generation' else caption, ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .12, 1, .95))
        for extension in ('png', 'svg', 'pdf'):
            fig.savefig(output/f'{name}.{extension}', dpi=160)
        plt.close(fig)

    def series(ax, mode, sweep, metric):
        axis_key = 'batch' if sweep == 'batch' else 'length'
        expected = (1,4,8,16) if sweep == 'batch' else (512,2048,4096,8192)
        for index, (variant, color) in enumerate(zip(variants, colors)):
            selected = [r for r in rows if r['mode']==mode and r['variant']==variant and
                        (r['length']==2048 if sweep=='batch' else r['batch']==8)]
            by_x = {r[axis_key]:r for r in selected}
            values = [by_x[x][metric] if by_x[x]['status']=='completed' else math.nan for x in expected]
            ax.plot(expected, values, 'o-', color=color, label=variant)
            for x in expected:
                row = by_x[x]
                if row['status'] != 'completed':
                    ax.text(x, .03+.08*index, row['status'], transform=ax.get_xaxis_transform(),
                            color=color, fontsize=6, va='bottom', ha='center')
                elif metric == 'median_ms':
                    y = row[metric]
                    ax.errorbar(x, y, yerr=[[y-row['round_min_ms']], [row['round_max_ms']-y]], color=color, capsize=3)
                elif metric == 'tokens_per_second':
                    y=row[metric]
                    work=y*row['median_ms']
                    ax.errorbar(x, y, yerr=[[y-work/row['round_max_ms']], [work/row['round_min_ms']-y]], color=color, capsize=3)
        ax.set_xscale('log', base=2)
        ax.set_xticks(expected, [str(x) for x in expected])
        ax.set_xlim(expected[0]/1.2, expected[-1]*1.2)
        relevant = [r for r in rows if r['mode']==mode and (r['length']==2048 if sweep=='batch' else r['batch']==8)]
        if not any(r['status']=='completed' for r in relevant):
            ax.set_yticks([])
            ax.text(.5,.6,'No completed measurements',transform=ax.transAxes,ha='center')
        ax.set_xlabel('Batch (length=2048)' if sweep=='batch' else 'Context/prefill length (batch=8)')
        ax.set_ylabel({'median_ms': 'Median of round means (ms)', 'tokens_per_second': 'Tokens / second', 'gpu_peak_allocated_gib': 'Peak GPU allocated (GiB)', 'gpu_peak_reserved_gib': 'Peak GPU reserved (GiB)', 'host_rss_gib': 'Host RSS (GiB)'}[metric])
        ax.set_title(mode)
        ax.grid(alpha=.2)

    for metric in ('median_ms', 'tokens_per_second', 'gpu_peak_allocated_gib', 'gpu_peak_reserved_gib', 'host_rss_gib'):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for i, mode in enumerate(('prefill', 'decode')):
            for j, sweep in enumerate(('batch','length')):
                series(axes[i,j], mode, sweep, metric)
        axes[0,0].legend(fontsize=8)
        finish(fig, metric)
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, (variant, color) in enumerate(zip(variants, colors)):
        selected = [r for r in rows if r['mode']=='generation' and r['variant']==variant]
        for row in selected:
            x = row['batch'] + (index-1)*.15
            if row['status']=='completed':
                y=row['median_ms']
                ax.errorbar(x,y,yerr=[[y-row['round_min_ms']], [row['round_max_ms']-y]],fmt='o',color=color,
                            label=variant if row['batch']==1 else None,capsize=4)
            else:
                ax.text(x,.04+.12*index,variant+': '+row['status'],transform=ax.get_xaxis_transform(),fontsize=7,rotation=45)
    ax.set_xticks([1,8]); ax.set_xlim(.3,8.7)
    ax.set_xlabel('Batch; 2048-token prefix + 128 predetermined decode steps (no sampling)')
    ax.set_ylabel('Generation latency (ms, prefix included)')
    handles, _ = ax.get_legend_handles_labels()
    if handles: ax.legend()
    finish(fig,'growing_generation')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest, rows=collect(args.run)
    args.output.mkdir(parents=True,exist_ok=False)
    counts={status:sum(r['status']==status for r in rows) for status in sorted({r['status'] for r in rows})}
    payload=dict(candidate_sha=manifest['candidate_sha'],run=str(args.run.resolve()),status_counts=counts,
                 uncertainty='range of round means; descriptive, not confidence interval', rows=rows)
    (args.output/'source.json').write_text(json.dumps(payload,indent=2)+'\n')
    with (args.output/'source.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plot(rows,args.output,any(r['status'] in ('pending','running') for r in rows))
    print(json.dumps(counts))


if __name__=='__main__':
    main()
