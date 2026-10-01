"""Plot comparable formal attempt history and accepted steps from stored evidence.

Run only after timing controllers exit. Original A0000 legacy measurements stay
untouched; history uses its first matched formal remeasurement, explicitly labeled.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def geometric(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def collect(root):
    attempts=[]
    for directory in sorted(root.glob('A[0-9][0-9][0-9][0-9]')):
        meta_path=directory/'attempt.json'
        if not meta_path.exists():continue
        meta=json.loads(meta_path.read_text())
        summary_path=directory/'confirmation-summary.json'
        summary=json.loads(summary_path.read_text()) if summary_path.exists() else None
        attempts.append((directory,meta,summary))
    reference=next((a for a in attempts if a[2] is not None),None)
    assert reference is not None, 'No matched formal evidence for history'
    rows=[];incumbent={};accepted_rows=[]
    for directory,meta,summary in attempts:
        status=meta.get('status') or meta.get('phase','pending')
        is_baseline=meta['attempt_id']=='A0000'
        evidence=reference[2] if is_baseline else summary
        metrics={}
        for mode in ('prefill','decode'):
            row=dict(attempt_id=meta['attempt_id'],status=status,accepted_step=meta.get('accepted_step'),
                     mode=mode,variant='ma_offload',batch=8,length=2048,label=meta.get('label'),reason=meta.get('reason'),latency_ms=None,
                     process_min_ms=None,process_max_ms=None,paired_speedup=None,ci95_low=None,ci95_high=None,
                     incumbent_attempt=None,incumbent_latency_ms=None,evidence=None)
            if evidence is not None:
                record=next(r for r in evidence['rows'] if r['variant']=='ma_offload' and r['mode']==mode)
                values=[r['baseline_ms' if is_baseline else 'candidate_ms'] for r in record['pairs']]
                assert len(values)==3 and all(math.isfinite(v) and v>0 for v in values)
                row.update(latency_ms=geometric(values),process_min_ms=min(values),process_max_ms=max(values),
                           evidence=str((reference[0] if is_baseline else directory)/'confirmation-summary.json'))
                if not is_baseline:
                    row.update(paired_speedup=record['geometric_mean_speedup'],
                               ci95_low=record['paired_speedup_ci95'][0],ci95_high=record['paired_speedup_ci95'][1])
                metrics[mode]=row['latency_ms']
            rows.append(row)
        if status=='accepted':
            assert set(metrics)=={'prefill','decode'}, 'Accepted source needs both primary measurements'
            incumbent=dict(attempt_id=meta['attempt_id'],**metrics)
            for row in rows[-2:]:
                accepted_rows.append(dict(row,paired_speedup=1.0 if is_baseline else row['paired_speedup'],
                                           ci95_low=1.0 if is_baseline else row['ci95_low'],
                                           ci95_high=1.0 if is_baseline else row['ci95_high']))
        for row in rows[-2:]:
            row.update(incumbent_attempt=incumbent.get('attempt_id'),incumbent_latency_ms=incumbent.get(row['mode']))
    return rows,accepted_rows,reference[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    # A fresh manifest status is necessary but not sufficient: check own known
    # timing controller command lines too, so a stale record cannot permit plots.
    for path in root.glob('A*/R*/manifest.json'):
        manifest=json.loads(path.read_text());pid=manifest.get('controller_pid')
        proc=Path(f'/proc/{pid}/cmdline')
        if manifest.get('status')=='running' and proc.exists():
            cmd=proc.read_bytes()
            assert not any(name in cmd for name in (b'run_confirmation.py',b'run_full_validation.py',b'run_small_batch_check.py',b'run_screening.py')), f'Live timing controller: {pid}'
    rows,accepted,reference=collect(root)
    args.output.mkdir(parents=True,exist_ok=False)
    payload=dict(scope='paper_v1; primary batch8/context2048; ma_offload; formal10/10/3; random weights',
                 latency_estimator='Geometric mean of three independent process estimates; each process estimate is median of round means.',
                 latency_uncertainty='Range of the three process estimates, descriptive only.',
                 speedup_uncertainty='95% Student-t CI on three paired log ratios, df2; each accepted attempt uses its own matched frozen-baseline processes.',
                 baseline_history_reference=str(reference/'confirmation-summary.json'),
                 baseline_note='A0000 history point uses the first matched formal remeasurement, not the original30/30/5 record. Original baseline data are preserved separately.',
                 rows=rows,accepted_steps=accepted)
    (args.output/'source.json').write_text(json.dumps(payload,indent=2)+'\n')
    for name,items in [('history',rows),('accepted_steps',accepted)]:
        with (args.output/(name+'.csv')).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(items)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    caption='2.836B random parameters · BF16 · RTX5090 · batch8/context2048 · last-token logits + KV\nNo quality claim. Same formal10/10/3 plan. Latency bars: process range, not CI.'
    def save(fig,name,caption_text=caption):
        fig.text(.5,.015,caption_text,ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.12,1,.94))
        for ext in ('png','pdf','svg'):fig.savefig(args.output/f'{name}.{ext}',dpi=170)
        plt.close(fig)
    ids=list(dict.fromkeys(r['attempt_id'] for r in rows))
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for ax,mode in zip(axes,('prefill','decode')):
        selected=[r for r in rows if r['mode']==mode]
        for i,row in enumerate(selected):
            if row['latency_ms'] is None:
                ax.text(i,.1,row['status']+'\nno comparable timing',transform=ax.get_xaxis_transform(),ha='center',fontsize=7)
                continue
            accepted_point=row['status']=='accepted'
            rejected=row['status'] in ('rejected','correctness_failed','benchmark_failed','oom','unsupported')
            color='#D55E00' if rejected else '#009E73' if accepted_point else '#0072B2'
            marker='x' if rejected else 'o'
            y=row['latency_ms']
            ax.errorbar(i,y,yerr=[[y-row['process_min_ms']],[row['process_max_ms']-y]],fmt=marker,color=color,capsize=4,
                        markerfacecolor=color if accepted_point else 'none')
            label=row['status'] if accepted_point or rejected else 'pending validation'
            if row['attempt_id']=='A0001' and rejected:label+='\nbatch1 regression'
            ax.annotate(label,(i,y),xytext=(0,12),textcoords='offset points',ha='center',fontsize=7)
        ax.step(range(len(ids)),[r['incumbent_latency_ms'] for r in selected],where='post',color='black',linestyle='--',label='incumbent accepted implementation')
        ax.set_xticks(range(len(ids)),ids);ax.set_xlim(-.4,len(ids)-.6);ax.set_title(mode);ax.set_ylabel('Latency (ms)');ax.grid(alpha=.2)
    axes[1].legend(fontsize=7);fig.suptitle('Attempt history — failed attempts retained')
    save(fig,'attempt_history')
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for ax,mode in zip(axes,('prefill','decode')):
        selected=sorted((r for r in accepted if r['mode']==mode),key=lambda r:r['accepted_step'])
        xs=[r['accepted_step'] for r in selected];ys=[r['paired_speedup'] for r in selected]
        ax.errorbar(xs,ys,yerr=[[r['paired_speedup']-r['ci95_low'] for r in selected],
                               [r['ci95_high']-r['paired_speedup'] for r in selected]],fmt='o-',capsize=4,color='#0072B2')
        ax.axhline(1,color='gray',linestyle=':');ax.set_xticks(xs,['baseline' if x==0 else f'+step{x}\n{r["attempt_id"]}' for x,r in zip(xs,selected)])
        ax.set_title(mode);ax.set_ylabel('Speedup vs matched frozen baseline');ax.grid(alpha=.2)
    fig.suptitle('Cumulative accepted steps only')
    save(fig,'accepted_steps',caption.replace('Latency bars: process range, not CI.','Speedup bars:95% paired-log t CI (3 independent pairs).'))


if __name__=='__main__':main()
