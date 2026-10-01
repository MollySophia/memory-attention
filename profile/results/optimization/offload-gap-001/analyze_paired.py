"""Paired primary confirmation, using independent process blocks (not rounds).

95% Student-t intervals over three independent paired blocks (df=2). These
intervals assume approximately symmetric block differences; with n=3 they are
necessarily imprecise. Passing confirmation never constitutes acceptance.
"""
import argparse
import json
from pathlib import Path
import statistics


def interval(values):
    assert len(values)==3, 'The frozen confirmation plan requires exactly three blocks'
    mean=statistics.mean(values)
    half=4.302652729911275*statistics.stdev(values)/(len(values)**.5)
    return dict(values=values,mean=mean,lower_95=mean-half,upper_95=mean+half,
                method='paired process-block mean +/- t(df=2,0.975)*SE',n=3)


def analyze(path):
    manifest=json.loads(path.read_text())
    assert manifest['status']=='completed'
    assert manifest['stage']=='confirmation'
    from run_paired import validate_balanced_order,PAIRING_PLAN_ID
    assert manifest['pairing_plan_id']==PAIRING_PLAN_ID
    validate_balanced_order(manifest['jobs'])
    assert len(manifest['jobs'])==48
    data={}
    for job in manifest['jobs']:
        assert job['status']=='completed', 'Missing/failed points cannot pass confirmation'
        payload=json.loads((path.parent/(job['name']+'.json')).read_text())
        assert payload['protocol_version']=='offload_gap_v1'
        assert payload['measurement_plan_id']=='formal_v1_w10_n10_r3'
        assert payload['config']['seed']==1234
        row=payload['results'][0]
        data[(job['block'],job['batch'],job['mode'],job['implementation'],job['variant'])]=row
    report=[]
    for batch in (1,8):
        for mode in ('prefill','decode'):
            values={k:[] for k in ('offload_reduction_ms','gpu_reduction_ms','gap_reduction_ms','offload_speedup','baseline_gap_ms','candidate_gap_ms','baseline_relative_overhead','candidate_relative_overhead')}
            memory=[]
            for block in (1,2,3):
                rows={impl:{variant:data[(block,batch,mode,impl,variant)] for variant in ('ma_offload','ma_gpu')} for impl in ('baseline','candidate')}
                bo,bg,co,cg=[rows[i][v]['mean_ms'] for i,v in [('baseline','ma_offload'),('baseline','ma_gpu'),('candidate','ma_offload'),('candidate','ma_gpu')]]
                v=dict(offload_reduction_ms=bo-co,gpu_reduction_ms=bg-cg,gap_reduction_ms=(bo-bg)-(co-cg),offload_speedup=bo/co,baseline_gap_ms=bo-bg,candidate_gap_ms=co-cg,baseline_relative_overhead=bo/bg-1,candidate_relative_overhead=co/cg-1)
                for k,x in v.items():values[k].append(x)
                memory.append({impl:{variant:rows[impl][variant]['memory_after'] for variant in rows[impl]} for impl in rows})
            stats={k:interval(v) for k,v in values.items()}
            retained_savings=all(m[impl]['ma_offload']['gpu_peak_allocated_bytes'] < m[impl]['ma_gpu']['gpu_peak_allocated_bytes'] for m in memory for impl in ('baseline','candidate'))
            report.append(dict(batch=batch,mode=mode,length=2048,statistics=stats,memory=memory,memory_savings_preserved=retained_savings,
                               repeatable_offload_and_gap_reduction=stats['offload_reduction_ms']['lower_95']>0 and stats['gap_reduction_ms']['lower_95']>0,
                               resolved_offload_regression=stats['offload_reduction_ms']['upper_95']<0,
                               resolved_resident_slowdown=stats['gpu_reduction_ms']['upper_95']<0))
    promising=any(r['repeatable_offload_and_gap_reduction'] for r in report) and not any(r['resolved_offload_regression'] or r['resolved_resident_slowdown'] or not r['memory_savings_preserved'] for r in report)
    return dict(campaign_id='offload-gap-001',source_manifest=str(path),rows=report,confirmation_promising=promising,
                accepted=False,remaining_gates=['full matrix regression and placement/folding references','growing generation performance','full model independent-reference correctness at primary shapes','source/resident-behavior audit'],
                uncertainty_note='Intervals use independent process blocks; not within-process rounds. n=3, normality assumption unverified; no multiple-comparison correction. All four prespecified workloads are reported.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=analyze(args.manifest)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for r in result['rows']:
        print(r['mode'],r['batch'],json.dumps({k:r['statistics'][k] for k in ('offload_reduction_ms','gap_reduction_ms')}))
    print('confirmation_promising',result['confirmation_promising'],'accepted',False)

if __name__=='__main__':main()
