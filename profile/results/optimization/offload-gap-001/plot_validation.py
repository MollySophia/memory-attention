"""Standalone validation figures from collect_matrix.py JSON exports.

Writes PDF, SVG, PNG, and the exact source JSON/CSV used by every figure.
No screening numbers are accepted as headline validation data.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'ma_offload':'#cc5533','ma_gpu':'#28689b','ma_gpu_unfolded':'#7a6396'}
VARIANTS=('ma_offload','ma_gpu','ma_gpu_unfolded')
STATUS_LABEL = ''
CAPTION='2.836B parameters · BF16 · RTX 5090 · seeded random weights (no quality claim)\nCached last-token logits. Bars: process 95% CI where n=3; otherwise descriptive round range, not CI.'


def save(fig,output,name,caption=CAPTION,top=.96):
    fig.text(.01,.015,caption + ('\n' + STATUS_LABEL if STATUS_LABEL else ''),fontsize=8,va='bottom')
    fig.tight_layout(rect=(0,.095,1,top))
    for extension in ('pdf','svg','png'):
        fig.savefig(output/(name+'.'+extension),dpi=180)
    plt.close(fig)


def errorbar(ax,x,y,low,high,**kwargs):
    ax.errorbar(x,y,yerr=[[max(0,y-lo) if math.isfinite(y) and math.isfinite(lo) else math.nan for y,lo in zip(y,low)],
                         [max(0,hi-y) if math.isfinite(y) and math.isfinite(hi) else math.nan for y,hi in zip(y,high)]],capsize=2,**kwargs)


def series(points,implementation,variant,mode,sweep):
    rows=[r for r in points if r['implementation']==implementation and r['variant']==variant
          and r['mode']==mode and (r['length']==2048 if sweep=='batch' else r['batch']==8)]
    key='batch' if sweep=='batch' else 'length'
    return sorted(rows,key=lambda r:r[key]),key


def draw_series(ax,points,mode,sweep,value,lower,upper,scale=1):
    for index,(impl,variant) in enumerate((i,v) for i in ('baseline','candidate') for v in VARIANTS):
        rows,key=series(points,impl,variant,mode,sweep)
        valid=[r for r in rows if r.get(value) is not None]
        for r in rows:
            if r.get(value) is None:
                ax.text(r[key],.97-index*.045,f'{impl}/{variant}: {r["status"]}',
                        transform=ax.get_xaxis_transform(),fontsize=6,rotation=15)
        if not valid:continue
        x=[r[key] for r in rows]
        y=[r[value]*scale if r.get(value) is not None else math.nan for r in rows]
        lo=[r.get(lower,r.get(value))*scale if r.get(lower,r.get(value)) is not None else math.nan for r in rows]
        hi=[r.get(upper,r.get(value))*scale if r.get(upper,r.get(value)) is not None else math.nan for r in rows]
        errorbar(ax,x,y,lo,hi,label=f'{impl} {variant}',color=COLORS[variant],
                 linestyle='--' if impl=='baseline' else '-',marker='o' if impl=='candidate' else 's',
                 markersize=3,alpha=.6 if impl=='baseline' else 1)
    ax.set_xlabel('Batch size (length/context 2048)' if sweep=='batch' else 'Length/context (batch 8)')
    ax.grid(alpha=.2)


def scaling(points,output,sweep):
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,mode in enumerate(('prefill','decode')):
        draw_series(axes[0,col],points,mode,sweep,'latency_ms','latency_low_ms','latency_high_ms')
        draw_series(axes[1,col],points,mode,sweep,'tokens_per_second','throughput_low','throughput_high')
        axes[0,col].set_title(mode.title());axes[0,col].set_ylabel('Mean model-call latency (ms)')
        axes[1,col].set_ylabel('Tokens / second')
    axes[0,0].legend(fontsize=7,ncol=2)
    fig.suptitle('Latency and throughput vs '+('batch size' if sweep=='batch' else 'context length'))
    save(fig,output,'scaling-'+sweep)


def memory(points,output,sweep):
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,mode in enumerate(('prefill','decode')):
        for row,field in enumerate(('gpu_peak_allocated_bytes','host_rss_bytes')):
            draw_series(axes[row,col],points,mode,sweep,field,field+'_min',field+'_max',1/2**30)
            axes[row,col].set_ylabel('GPU peak allocated (GiB)' if row==0 else 'Host RSS at measured output (GiB)')
        axes[0,col].set_title(mode.title())
    axes[0,0].legend(fontsize=7,ncol=2)
    fig.suptitle('Memory vs '+('batch size' if sweep=='batch' else 'context length'))
    save(fig,output,'memory-'+sweep,CAPTION.split('\n')[0]+'\nCached last-token logits and KV.\nGPU includes KV, activations, logits and cached buffers; host includes tables/raw restore snapshots. Bars show process range. Reserved/pinned bytes are in source CSV.')


def derived_gap(off,gpu,relative):
    a,b=off['latency_ms'],gpu['latency_ms']
    center=(a/b-1)*100 if relative else a-b
    av,bv=off['process_mean_ms'],gpu['process_mean_ms']
    if len(av)==len(bv)==3:
        values=[(x/y-1)*100 if relative else x-y for x,y in zip(av,bv)]
        center=statistics.mean(values)
        half=4.302652729911275*statistics.stdev(values)/(3**.5)
        return center,center-half,center+half
    lo=(off['latency_low_ms']/gpu['latency_high_ms']-1)*100 if relative else off['latency_low_ms']-gpu['latency_high_ms']
    hi=(off['latency_high_ms']/gpu['latency_low_ms']-1)*100 if relative else off['latency_high_ms']-gpu['latency_low_ms']
    return center,lo,hi


def gaps(points,output,relative):
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for row,sweep in enumerate(('batch','length')):
        for col,mode in enumerate(('prefill','decode')):
            ax=axes[row,col]
            for impl,color in [('baseline','#777777'),('candidate','#cc5533')]:
                off,key=series(points,impl,'ma_offload',mode,sweep)
                gpu,_=series(points,impl,'ma_gpu',mode,sweep)
                gpu={r[key]:r for r in gpu}
                values=[]
                for r in off:
                    other=gpu.get(r[key])
                    if r.get('latency_ms') is None or not other or other.get('latency_ms') is None:
                        ax.text(r[key],.95,impl+': missing/OOM',transform=ax.get_xaxis_transform(),fontsize=7)
                        values.append((r[key],math.nan,math.nan,math.nan));continue
                    values.append((r[key],*derived_gap(r,other,relative)))
                if values:
                    x,y,lo,hi=zip(*values)
                    errorbar(ax,x,y,lo,hi,label=impl,color=color,marker='o',linestyle='--' if impl=='baseline' else '-')
            ax.axhline(0,color='black',linewidth=.6);ax.grid(alpha=.2)
            ax.set_title(mode.title());ax.set_xlabel('Batch size (context 2048)' if sweep=='batch' else 'Length/context (batch 8)')
            ax.set_ylabel('Signed relative overhead (%)' if relative else 'Signed offload − resident gap (ms)')
    axes[0,0].legend();fig.suptitle('Offload overhead against folded resident placement')
    save(fig,output,'relative-overhead' if relative else 'absolute-gap',CAPTION.split('\n')[0]+'\nCached last-token logits and KV.\nBars: paired process-block 95% CI where n=3; otherwise round-range envelope, not CI. Signed values retained.')


def generation(points,output):
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    for ax,batch in zip(axes,(1,8)):
        rows=sorted([r for r in points if r['mode']=='generation' and r['batch']==batch],key=lambda r:(r['implementation'],r['variant']))
        labels=[]
        for i,r in enumerate(rows):
            labels.append(r['implementation']+'\n'+r['variant'])
            if r.get('latency_ms') is None:
                ax.text(i,.9,r['status'],transform=ax.get_xaxis_transform(),rotation=90);continue
            y=r['latency_ms'];ax.bar(i,y,color=COLORS[r['variant']],alpha=.5 if r['implementation']=='baseline' else 1)
            errorbar(ax,[i],[y],[r['latency_low_ms']],[r['latency_high_ms']],color='black',linestyle='none')
        ax.set_xticks(range(len(rows)),labels,rotation=25,fontsize=7);ax.set_title(f'Batch {batch}')
        ax.set_ylabel('Prefix + 128 decode calls (ms)');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Growing-cache generation latency')
    save(fig,output,'generation',CAPTION.split('\n')[0]+'\nCached last-token logits and KV.\n2048-token prefix + 128 predetermined GPU-token decode calls. Sampling excluded; not serving latency. Bars: process 95% CI where n=3; otherwise trajectory round range, not CI.')


def main():
    global STATUS_LABEL
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix',type=Path,required=True)
    p.add_argument('--generation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--status-label', default='', help='Explicit candidate verdict or pending status printed on every figure')
    args=p.parse_args();STATUS_LABEL=args.status_label;args.output.mkdir(parents=True,exist_ok=True)
    matrix=json.loads(args.matrix.read_text());gen=json.loads(args.generation.read_text())
    assert matrix['stage']=='full_validation' and gen['stage']=='generation_validation'
    all_points=matrix['points']+gen['points']
    (args.output/'source.json').write_text(json.dumps(dict(matrix=matrix,generation=gen,status_label=STATUS_LABEL),indent=2)+'\n')
    fields=sorted({k for r in all_points for k in r})
    with (args.output/'source.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in all_points:writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    for sweep in ('batch','length'):
        scaling(matrix['points'],args.output,sweep);memory(matrix['points'],args.output,sweep)
    for relative in (False,True):gaps(matrix['points'],args.output,relative)
    generation(gen['points'],args.output)
    print('Exported seven validation figure sets (PDF/SVG/PNG) plus source CSV/JSON.')

if __name__=='__main__':main()
