"""Audit matched full-matrix raw data, including explicitly reused primary runs.

This script does not accept a candidate or infer confidence from round ranges.
Use --partial while the controller is live; plot only after GPU timing ends.
"""
import argparse
import csv
import json
import io
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'profile'))
from report_paper_matrix import plot


def collect(run, partial=False):
    manifest=json.loads((run/'manifest.json').read_text())
    if not partial:
        assert manifest['status'] in ('complete','complete_with_failures')
    rows=[];records={};hashes={'baseline':set(),'candidate':set()}
    reference_environment=None
    assert len(manifest['jobs'])==96
    for job in manifest['jobs']:
        row={k:job[k] for k in ('name','side','mode','variant','batch','length','status')}
        plan=job['sampling_plan'];row.update(stage=plan['stage'],measurement_plan_id=plan['measurement_plan_id'],
            reused=job['status']=='reused',source_result=job['result'],median_ms=None,round_min_ms=None,round_max_ms=None,
            tokens_per_second=None,gpu_peak_allocated_gib=None,gpu_peak_reserved_gib=None,host_rss_gib=None,
            pinned_mib=None,kv_gib=None,sample_count=None)
        if job['status'] not in ('completed','reused'):
            rows.append(row);continue
        data=json.loads(Path(job['result']).read_text());config=data['config']
        gpu=next(csv.DictReader(io.StringIO(data['environment_before']['gpu_telemetry']['stdout'].strip()),skipinitialspace=True))
        environment=dict(versions={k:v for k,v in data['env'].items() if k!='git_commit'},
                         driver=gpu['driver_version'],
                         settings={k:data['environment_before'][k] for k in ('torch_threads','torch_interop_threads','thread_environment','cpu_affinity','platform')})
        if reference_environment is None:reference_environment=environment
        assert environment==reference_environment, ('matrix/reused environment mismatch',job['name'])
        assert data['status']=='completed' and data['protocol_version']=='paper_v1',job['name']
        assert data['source']['git_commit']['stdout'].strip()==job['expected_commit'],job['name']
        assert data['source']['source_sha256']==job['expected_source_sha256'],job['name']
        assert not data['source']['source_patch']['stdout'],job['name']
        hashes[job['side']].add(data['source']['source_sha256'])
        checkout=ROOT if job['side']=='candidate' else ROOT.parent/'baseline-paper-001'
        assert Path(job['verified_cache_module']).resolve()==checkout/'fla/models/utils.py'
        expected=dict(batch_size=job['batch'],seq_len=job['length'],context_len=job['length'],
                      mode=job['mode'],variants=[job['variant']],num_layers=24,hidden_size=2048,
                      num_heads=32,num_kv_heads=32,intermediate_size=5632,vocab_size=32000,
                      seed=1234,logits_to_keep=1,prefill_workload='inference',generation_steps=128)
        assert all(config[k]==v for k,v in expected.items()),job['name']
        assert all(config[k]==plan[k] for k in ('warmup','repeats','rounds')),job['name']
        if job['side']=='candidate':
            assert data['measurement_plan_id']==plan['measurement_plan_id']
            assert data['stage']==plan['stage']
        assert len(data['results'])==1
        r=data['results'][0]
        assert (r['variant'],r['mode'])==(job['variant'],job['mode'])
        if job['side']=='candidate':assert r['sampling_plan']==plan
        samples=r['samples_ms']
        assert len(samples)==plan['rounds'] and all(len(s)==plan['repeats'] for s in samples)
        assert all(math.isfinite(x) and x>0 for s in samples for x in s)
        means=[statistics.mean(s) for s in samples]
        assert means==r['round_ms'] and statistics.median(means)==r['median_ms']
        work=job['batch']*(job['length'] if job['mode']=='prefill' else job['length']+128 if job['mode']=='generation' else 1)
        assert math.isclose(r['tokens_per_second'],work*1000/r['median_ms'])
        mem=r['memory_after']
        row.update(status='completed',median_ms=r['median_ms'],round_min_ms=min(means),round_max_ms=max(means),
                   tokens_per_second=r['tokens_per_second'],gpu_peak_allocated_gib=mem['gpu_peak_allocated_bytes']/2**30,
                   gpu_peak_reserved_gib=mem['gpu_peak_reserved_bytes']/2**30,host_rss_gib=mem['host_rss_bytes']/2**30,
                   pinned_mib=mem['offload_pinned_bytes']/2**20,kv_gib=mem['kv_cache_storage_bytes']/2**30,
                   sample_count=sum(map(len,samples)))
        key=(job['mode'],job['variant'],job['batch'],job['length'])
        records[(*key,job['side'])]=data
        rows.append(row)
    assert all(len(h)<=1 for h in hashes.values())
    assert len({(r['side'],r['mode'],r['variant'],r['batch'],r['length']) for r in rows})==96
    formal=json.loads((run.parent/'confirmation-summary.json').read_text())
    primary_verdicts={(r['mode'],r['variant'],8,2048):r['verdict'] for r in formal['rows']}
    small=json.loads((run.parent/'small-batch-summary.json').read_text())
    primary_verdicts.update({(r['mode'],r['variant'],r['batch'],r['context']):r['verdict'] for r in small['rows']})
    comparisons=[]
    for row in rows:
        if row['side']!='candidate':continue
        key=tuple(row[k] for k in ('mode','variant','batch','length'))
        base=next(r for r in rows if r['side']=='baseline' and tuple(r[k] for k in ('mode','variant','batch','length'))==key)
        cmp={k:row[k] for k in ('mode','variant','batch','length')}
        cmp.update(baseline_status=base['status'],candidate_status=row['status'],baseline_ms=base['median_ms'],
                   candidate_ms=row['median_ms'],speedup=None,round_extreme_ratio_low=None,round_extreme_ratio_high=None,
                   regression_signal=None)
        if row['status']=='completed' and base['status']=='completed':
            b,c=records[(*key,'baseline')],records[(*key,'candidate')]
            assert b['model_config']==c['model_config'],key
            assert {k:v for k,v in b['env'].items() if k!='git_commit'}=={k:v for k,v in c['env'].items() if k!='git_commit'},key
            for k in ('torch_threads','torch_interop_threads','thread_environment','cpu_affinity','platform'):
                assert b['environment_before'][k]==c['environment_before'][k],(key,k)
            for k,v in b['config'].items():
                if k!='json':assert c['config'][k]==v,(key,k)
            cmp.update(speedup=base['median_ms']/row['median_ms'],
                       round_extreme_ratio_low=base['round_min_ms']/row['round_max_ms'],
                       round_extreme_ratio_high=base['round_max_ms']/row['round_min_ms'],
                       regression_signal=row['round_min_ms']>base['round_max_ms'])
        primary_verdict=primary_verdicts.get((row['mode'],row['variant'],row['batch'],row['length']))
        cmp['independent_confirmation_verdict']=primary_verdict
        cmp['requires_investigation']=(cmp['regression_signal'] is True and primary_verdict is None) or (base['status']=='completed' and row['status'] in ('oom','unsupported','benchmark_failed'))
        comparisons.append(cmp)
    return manifest,rows,comparisons


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--partial',action='store_true')
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    manifest,rows,comparisons=collect(args.run,args.partial)
    assert not args.plot or not args.partial, 'Do not plot during GPU timing'
    args.output.mkdir(parents=True,exist_ok=False)
    counts={status:sum(r['status']==status for r in rows) for status in sorted({r['status'] for r in rows})}
    payload=dict(run=str(args.run.resolve()),controller_status=manifest['status'],partial=args.partial,
                 status_counts=counts,sample_count=sum(r['sample_count'] or 0 for r in rows),
                 accepted=False,uncertainty='Within-process round ranges are descriptive, not confidence intervals. Primary process-pair CI is in confirmation-summary.json. Non-primary single-pair signals require investigation before acceptance.',
                 rows=rows,comparisons=comparisons)
    (args.output/'source.json').write_text(json.dumps(payload,indent=2)+'\n')
    for name,items in [('source',rows),('comparisons',comparisons)]:
        with (args.output/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    if args.plot:
        for side in ('baseline','candidate'):
            out=args.output/side;out.mkdir();plot([r for r in rows if r['side']==side],out,False)
    print(json.dumps(dict(status_counts=counts,samples=payload['sample_count'],regression_signals=[r for r in comparisons if r['regression_signal']]),indent=2))


if __name__=='__main__':main()
