"""Render audited baseline/candidate scaling, only after timing exits.

Single-process round ranges are descriptive. Independent primary and followup
confidence intervals remain in their paired reports, not inferred here.
"""
import argparse
import csv
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try: cmd=proc.read_bytes().split(b'\0')
        except (OSError,PermissionError): continue
        if any(Path(x.decode(errors='replace')).name in ('run_confirmation.py','run_full_validation.py','run_small_batch_check.py','run_screening.py','run_regression_followup.py') for x in cmd):
            raise RuntimeError(f'Timing controller still live: {proc.parent.name}')
    data=json.loads(args.audit.read_text())
    assert not data['partial'] and data['controller_status']=='complete'
    rows=data['rows'];assert len(rows)==96 and all(r['status']=='completed' for r in rows)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'source.json').write_text(json.dumps(data,indent=2)+'\n')
    with (args.output/'source.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(ma_offload='#0072B2',ma_gpu='#D55E00',ma_gpu_unfolded='#009E73')
    captions=('2.836B random parameters · BF16 · RTX 5090 · paper_v1 · no quality claim\n'
              'Frozen A0000 source vs A0002; matched sampling plans. Bars: round-mean range, not CI.\n'
              'Prefill: last-token logits + KV; decode: fixed context, rollback excluded.')
    def finish(fig,name,generation=False):
        fig.suptitle('A0002 vs frozen baseline — '+name.replace('_',' '))
        caption=captions if not generation else captions.rsplit('\n',1)[0]+'\n2048-token prefix + 128 predetermined decode calls; sampling excluded; full-trajectory2/5/3.'
        fig.text(.5,.015,caption,ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.13,1,.94))
        for ext in ('png','pdf','svg'):fig.savefig(args.output/f'{name}.{ext}',dpi=160)
        plt.close(fig)
    metrics={'median_ms':'Latency (ms)','tokens_per_second':'Tokens / second',
             'gpu_peak_allocated_gib':'Peak GPU allocated (GiB)',
             'gpu_peak_reserved_gib':'Peak GPU reserved (GiB)','host_rss_gib':'Host RSS (GiB)'}
    def line(ax,selected,metric,xkey,color,side,label):
        selected=sorted(selected,key=lambda r:r[xkey]);xs=[r[xkey] for r in selected];ys=[r[metric] for r in selected]
        ax.plot(xs,ys,'o--' if side=='baseline' else 's-',color=color,label=label,alpha=.8)
        if metric in ('median_ms','tokens_per_second'):
            if metric=='median_ms':lo=[r['round_min_ms'] for r in selected];hi=[r['round_max_ms'] for r in selected]
            else:
                lo=[r[metric]*r['median_ms']/r['round_max_ms'] for r in selected]
                hi=[r[metric]*r['median_ms']/r['round_min_ms'] for r in selected]
            ax.errorbar(xs,ys,yerr=[[y-l for y,l in zip(ys,lo)],[h-y for y,h in zip(ys,hi)]],fmt='none',color=color,capsize=3)
    for metric,title in metrics.items():
        fig,axes=plt.subplots(2,2,figsize=(12,9))
        for i,mode in enumerate(('prefill','decode')):
            for j,sweep in enumerate(('batch','length')):
                ax=axes[i,j];xs=(1,4,8,16) if sweep=='batch' else (512,2048,4096,8192)
                for variant,color in colors.items():
                    for side in ('baseline','candidate'):
                        selected=[r for r in rows if r['mode']==mode and r['variant']==variant and r['side']==side and (r['length']==2048 if sweep=='batch' else r['batch']==8)]
                        line(ax,selected,metric,sweep,color,side,f'{variant} / {side}')
                ax.set_xscale('log',base=2);ax.set_xticks(xs,[str(x) for x in xs]);ax.grid(alpha=.2)
                ax.set_xlabel('Batch (length2048)' if sweep=='batch' else 'Length (batch8)');ax.set_ylabel(title);ax.set_title(mode)
        axes[0,0].legend(fontsize=7);finish(fig,metric)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,metric in zip(axes,('median_ms','gpu_peak_allocated_gib')):
        for variant,color in colors.items():
            for side in ('baseline','candidate'):
                selected=[r for r in rows if r['mode']=='generation' and r['variant']==variant and r['side']==side]
                line(ax,selected,metric,'batch',color,side,f'{variant} / {side}')
        ax.set_xticks([1,8]);ax.set_xlabel('Batch');ax.set_ylabel(metrics[metric]);ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);finish(fig,'growing_generation',True)
    print('Exported six figures in PNG/PDF/SVG and audited source CSV/JSON.')


if __name__=='__main__':main()
