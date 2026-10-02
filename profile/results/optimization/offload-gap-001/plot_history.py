"""Attempt history and accepted-step figures with separate primary workloads.

History uses matched screening rows and is explicitly provisional. Accepted-step
figures require accepted records plus the corrected independent confirmation
manifest; they never substitute screening latencies for confirmed evidence.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_validation import save,CAPTION

WORKLOADS=[('prefill',1),('decode',1),('prefill',8),('decode',8)]
STYLE={'baseline':('o','#28689b'),'accepted':('o','#218358'),'rejected':('X','#c64740'),'within_noise':('s','#777777'),
       'correctness_failed':('v','#c64740'),'benchmark_failed':('v','#b97824'),
       'oom':('v','#333333'),'unsupported':('v','#333333'),'interrupted':('s','#777777'),
       'pending':('D','#c39225')}


def attempts(root):
    found=[]
    for directory in sorted(root.glob('A[0-9][0-9][0-9][0-9]')):
        record_path=directory/'record.json'
        if not record_path.exists():continue
        record=json.loads(record_path.read_text())
        assert record['campaign_id']=='offload-gap-001'
        summary=directory/record.get('screen_summary','R01-summary.json')
        rows=json.loads(summary.read_text())['rows'] if summary.exists() else []
        rows={(r['mode'],r['batch']):r for r in rows if r['length']==2048 and r['batch'] in (1,8)}
        # A valid offload timing belongs in history even if its resident
        # partner failed and therefore no matched gap summary was possible.
        manifest_path=directory/record.get('screen_manifest','R01/manifest.json')
        if manifest_path.exists():
            manifest=json.loads(manifest_path.read_text())
            for job in manifest['jobs']:
                if job['status']!='completed' or job['variant']!='ma_offload' or job['batch'] not in (1,8) or job['length']!=2048:
                    continue
                raw=manifest_path.parent/(job['name']+'.json')
                payload=json.loads(raw.read_text())
                assert payload['protocol_version']=='offload_gap_v1' and payload['stage']=='screening'
                timing=payload['results'][0]
                samples=[x for block in timing['samples_ms'] for x in block]
                rows[(job['mode'],job['batch'])]=dict(measurement_plan_id=payload['measurement_plan_id'],
                    offload_ms=timing['mean_ms'],offload_sample_sd_ms=statistics.stdev(samples) if len(samples)>1 else None)
        found.append(dict(directory=directory,record=record,rows=rows))
    return found


def history(entries,output):
    plans={r['measurement_plan_id'] for entry in entries for r in entry['rows'].values()}
    exported=[]
    for plan in sorted(plans):
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for ax,key in zip(axes.flat,WORKLOADS):
            incumbent=None;incumbent_attempt=None;step=[];x=[]
            for index,entry in enumerate(entries):
                record=entry['record'];row=entry['rows'].get(key)
                if row is not None and row['measurement_plan_id']!=plan:row=None
                status='baseline' if record['attempt_id']=='A0000' else record.get('status') or 'pending'
                if status not in STYLE:status='pending'
                if record.get('accepted_step') is not None and record.get('status')=='accepted':
                    incumbent=row['offload_ms'] if row else None
                    incumbent_attempt=record['attempt_id']
                x.append(index);step.append(float('nan') if incumbent is None else incumbent)
                if row:
                    marker,color=STYLE[status]
                    ax.errorbar(index,row['offload_ms'],yerr=row['offload_sample_sd_ms'],
                                marker=marker,color=color,capsize=3,linestyle='none')
                else:
                    label='other plan / not measured' if entry['rows'].get(key) else status+'; no timing'
                    ax.text(index,.94,label,transform=ax.get_xaxis_transform(),rotation=70,fontsize=7,ha='center',va='top')
                exported.append(dict(attempt=record['attempt_id'],label=record['label'],status=status,
                                     mode=key[0],batch=key[1],length=2048,measurement_plan_id=plan,
                                     measured_ms=row['offload_ms'] if row else None,
                                     sample_sd_ms=row['offload_sample_sd_ms'] if row else None,
                                     incumbent_attempt=incumbent_attempt,incumbent_ms=incumbent))
            ax.step(x,step,where='post',color='#222222',linestyle='--',linewidth=1,label='Incumbent accepted implementation')
            ax.set_xticks(x,[e['record']['attempt_id'] for e in entries],rotation=30)
            ax.set_title(f'{key[0].title()}, batch {key[1]}, length/context 2048')
            ax.set_ylabel('Offloaded latency (ms)');ax.grid(alpha=.2)
        handles=[plt.Line2D([],[],color=c,marker=m,linestyle='none',label=s) for s,(m,c) in STYLE.items() if s in {'baseline' if e['record']['attempt_id']=='A0000' else e['record'].get('status') or 'pending' for e in entries}]
        handles.append(plt.Line2D([],[],color='#222222',linestyle='--',label='Accepted incumbent (one model, all workloads)'))
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.955),ncol=4,fontsize=8)
        fig.suptitle('Optimization attempts — provisional screening measurements',y=.995)
        save(fig,output,'attempt-history-'+plan,CAPTION.split('\n')[0]+'\nCached last-token logits and KV; frozen baseline A0000.\nHistory uses matched screening only; bars are sample SD, not confidence intervals. Failures have no numerical point. Incumbent is never a per-metric minimum.',top=.88)
    return exported


def interval(values):
    assert len(values)==3
    center=statistics.mean(values);half=4.302652729911275*statistics.stdev(values)/(3**.5)
    return center,center-half,center+half


def confirmation(entry):
    record=entry['record']
    name=record.get('confirmation_manifest')
    if not name:raise ValueError('Accepted record must explicitly identify its eligible confirmation manifest')
    path=Path(name)
    if not path.is_absolute():path=entry['directory']/path
    manifest=json.loads(path.read_text())
    assert manifest['status']=='completed' and manifest['pairing_plan_id']=='balanced_order_v2'
    from run_paired import validate_balanced_order
    validate_balanced_order(manifest['jobs'])
    data={}
    for job in manifest['jobs']:
        assert job['status']=='completed'
        payload=json.loads((path.parent/(job['name']+'.json')).read_text())
        assert payload['protocol_version']=='offload_gap_v1'
        assert payload['measurement_plan_id']=='formal_v1_w10_n10_r3'
        data[(job['block'],job['mode'],job['batch'],job['implementation'],job['variant'])]=payload['results'][0]['mean_ms']
    return data


def accepted_steps(entries,output):
    retained=[e for e in entries if e['record'].get('status')=='accepted' and (e['record'].get('accepted_step') or 0)>0]
    retained.sort(key=lambda e:e['record']['accepted_step'])
    if not retained:return []
    assert [e['record']['accepted_step'] for e in retained]==list(range(1,len(retained)+1))
    data=[confirmation(e) for e in retained]
    labels=['baseline']+['+'+e['record']['label'] for e in retained]
    metrics={name:{key:[] for key in WORKLOADS} for name in ('latency','speedup','absolute-gap','relative-overhead')}
    exported=[]
    for step in range(len(retained)+1):
        source=data[max(0,step-1)]
        impl='baseline' if step==0 else 'candidate'
        for mode,batch in WORKLOADS:
            values={name:[] for name in metrics}
            for block in (1,2,3):
                off=source[(block,mode,batch,impl,'ma_offload')]
                gpu=source[(block,mode,batch,impl,'ma_gpu')]
                baseline=source[(block,mode,batch,'baseline','ma_offload')]
                values['latency'].append(off)
                values['speedup'].append(baseline/off)
                values['absolute-gap'].append(off-gpu)
                values['relative-overhead'].append((off/gpu-1)*100)
            for name,array in values.items():
                center,low,high=interval(array)
                metrics[name][(mode,batch)].append((center,low,high))
                exported.append(dict(accepted_step=step,label=labels[step],attempt='A0000' if step==0 else retained[step-1]['record']['attempt_id'],mode=mode,batch=batch,length=2048,metric=name,mean=center,lower_95=low,upper_95=high,process_values=array))
    for name,workloads in metrics.items():
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for ax,key in zip(axes.flat,WORKLOADS):
            center,low,high=zip(*workloads[key]);x=list(range(len(center)))
            ax.errorbar(x,center,yerr=[[max(0,c-l) for c,l in zip(center,low)],[max(0,h-c) for c,h in zip(center,high)]],marker='o',capsize=3,color='#218358')
            ax.set_xticks(x,labels,rotation=22,ha='right',fontsize=7)
            ax.set_title(f'{key[0].title()}, batch {key[1]}, length/context 2048')
            ax.set_ylabel({'latency':'Offloaded latency (ms)','speedup':'Offload speedup vs frozen A0000','absolute-gap':'Signed offload − resident gap (ms)','relative-overhead':'Signed relative overhead (%)'}[name])
            ax.grid(alpha=.2)
        fig.suptitle('Accepted cumulative implementations — '+name)
        save(fig,output,'accepted-steps-'+name,CAPTION.split('\n')[0]+'\nCached last-token logits and KV; frozen baseline A0000.\nThree independent balanced process blocks; 95% Student-t intervals (df=2). Baseline source is frozen A0000. Implementation history, not an additive decomposition.')
    return exported


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    entries=attempts(args.campaign)
    sources=dict(history=history(entries,args.output),accepted_steps=accepted_steps(entries,args.output),
                 records=[e['record'] for e in entries])
    (args.output/'history-source.json').write_text(json.dumps(sources,indent=2)+'\n')
    for name in ('history','accepted_steps'):
        rows=sources[name]
        with (args.output/(name+'-source.csv')).open('w') as f:
            if rows:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Exported history for',len(entries),'attempts including baseline;',len(sources['accepted_steps'])//16,'accepted implementations including baseline if available.')

if __name__=='__main__':main()
