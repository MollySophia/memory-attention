"""Export complete matched validation evidence, retaining failures and reuse.

Point estimates are means of process means. Three independent process blocks
use a Student-t 95% interval. A single process gets only its descriptive range
of round means, explicitly not a confidence interval. No acceptance decisions
are made from that within-process range.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics

MEMORY_FIELDS=('gpu_allocated_bytes','gpu_reserved_bytes','gpu_peak_allocated_bytes',
               'gpu_peak_reserved_bytes','host_rss_bytes','host_process_high_water_bytes',
               'raw_table_snapshot_bytes','cpu_table_bytes','offload_pinned_bytes',
               'offload_gpu_buffer_bytes','kv_cache_storage_bytes')


def summarize_rows(rows,mode,batch,length):
    means=[r['mean_ms'] for r in rows]
    center=statistics.mean(means)
    if len(means)==3:
        half=4.302652729911275*statistics.stdev(means)/(3**.5)
        low,high=center-half,center+half
        uncertainty='95% Student-t interval over 3 independent process means (df=2)'
    elif len(means)==1:
        low,high=min(rows[0]['round_ms']),max(rows[0]['round_ms'])
        uncertainty='descriptive round-mean range within one process; NOT a confidence interval'
    else:raise ValueError('Unexpected independent-process count')
    tokens=batch*(length if mode=='prefill' else length+128 if mode=='generation' else 1)
    result=dict(latency_ms=center,latency_low_ms=low,latency_high_ms=high,
                uncertainty_kind=uncertainty,independent_processes=len(rows),
                tokens_per_second=tokens*1000/center,
                throughput_low=tokens*1000/high,
                throughput_high=tokens*1000/low if low>0 else None,
                process_mean_ms=means,raw_sample_count=sum(len(s) for r in rows for s in r['samples_ms']),
                estimator='mean of process means; each process mean averages all raw samples',
                gpu_parameter_mib=rows[0]['gpu_parameter_mib'],cpu_parameter_mib=rows[0]['cpu_parameter_mib'])
    for field in MEMORY_FIELDS:
        values=[r['memory_after'][field] for r in rows]
        result[field]=statistics.mean(values)
        result[field+'_min']=min(values);result[field+'_max']=max(values)
    return result


def collect(path):
    manifest=json.loads(path.read_text())
    assert manifest['status']=='completed'
    points=[]
    for job in manifest['jobs']:
        point={k:job[k] for k in ('implementation','mode','variant','batch','length','candidate_sha')}
        point['status']=job['status']
        sources=[Path(x) for x in job.get('reused_results',[])] or [path.parent/(job['name']+'.json')]
        point['source_results']=[str(p) for p in sources]
        point['protocol_version']='offload_gap_v1'
        if job['status'] not in ('completed','reused'):
            point.update(latency_ms=None,tokens_per_second=None,reason='Preserved failed/OOM/unsupported job; inspect raw log and payload')
            points.append(point);continue
        payloads=[json.loads(p.read_text()) for p in sources]
        plans={p['measurement_plan_id'] for p in payloads}
        assert len(plans)==1
        expected_plan='generation_v1_w2_n5_r3' if job['mode']=='generation' else 'formal_v1_w10_n10_r3'
        assert plans=={expected_plan}
        for payload in payloads:
            assert payload['status']=='completed' and payload['protocol_version']=='offload_gap_v1'
            assert payload['source']['git_commit']['stdout'].strip()==job['candidate_sha']
            assert payload['config']['seed']==1234
            assert payload['config']['batch_size']==job['batch']
            assert payload['results'][0]['variant']==job['variant']
            assert payload['results'][0]['output_scope']=='cached_logits'
            assert payload['results'][0]['logits_to_keep']==1
        point.update(measurement_plan_id=next(iter(plans)),original_stages=[p['stage'] for p in payloads])
        point.update(summarize_rows([p['results'][0] for p in payloads],job['mode'],job['batch'],job['length']))
        points.append(point)
    return dict(campaign_id='offload-gap-001',source_manifest=str(path),stage=manifest['stage'],
                points=points,figure_caption_context='2,836,499,968 parameters; BF16; RTX 5090; seed-1234 random weights (no quality claim). Prefill includes embedding, 24 layers, norm, last-token logits and KV construction; decode uses real fixed-context KV. Generation includes prefix and 128 predetermined-token calls, excludes sampling; not serving latency.',
                uncertainty_note='Use the explicit uncertainty_kind per point. Do not interpret round spread as independent confirmation.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=collect(args.manifest)
    args.output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    fields=sorted({k for row in result['points'] for k in row})
    with args.output.with_suffix('.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for point in result['points']:
            w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in point.items()})
    print('Exported',len(result['points']),'points; failures remain explicit.')

if __name__=='__main__':main()
