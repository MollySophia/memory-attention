"""Audit A0002's primary and small-copy screening; not an acceptance test."""
import csv
import json
import math
from pathlib import Path
import statistics
import sys


def main():
    run=Path(sys.argv[1]).resolve();m=json.loads((run/'manifest.json').read_text())
    assert m['status']=='complete' and len(m['jobs'])==16
    records={};hashes={'baseline':set(),'candidate':set()};pids=set()
    for j in m['jobs']:
        assert j['status']=='complete' and j['returncode']==0
        assert j['pid'] not in pids;pids.add(j['pid'])
        d=json.loads((run/(j['name']+'.json')).read_text());c=d['config'];r=d['results'][0]
        assert d['status']=='completed' and d['protocol_version']=='paper_v1'
        assert d['source']['git_commit']['stdout'].strip()==j['expected_commit']
        assert not d['source']['source_patch']['stdout']
        hashes[j['side']].add(d['source']['source_sha256'])
        assert (c['warmup'],c['repeats'],c['rounds'])==(3,5,1)
        assert (c['batch_size'],c['seq_len'],c['context_len'])==(j['batch'],j['length'],j['length'])
        assert (r['mode'],r['variant'])==(j['mode'],j['variant'])
        assert c['prefill_workload']=='inference' and c['logits_to_keep']==1
        root=Path(__file__).resolve().parents[4]
        checkout=root.parent/'baseline-paper-001' if j['side']=='baseline' else root
        assert Path(j['verified_cache_module']).resolve()==checkout/'fla/models/utils.py'
        if j['side']=='candidate':assert d['measurement_plan_id']=='screen_v1_w3_n5_r1'
        samples=r['samples_ms'];assert len(samples)==1 and len(samples[0])==5
        assert all(math.isfinite(x) and x>0 for x in samples[0])
        assert statistics.mean(samples[0])==r['round_ms'][0]==r['median_ms']
        records[(j['mode'],j['variant'],j['batch'],j['length'],j['side'])]=d
    assert all(len(v)==1 for v in hashes.values())
    rows=[]
    for mode,variant,batch,length in sorted({k[:4] for k in records}):
        b,c=(records[(mode,variant,batch,length,s)] for s in ('baseline','candidate'))
        assert b['model_config']==c['model_config']
        assert {k:v for k,v in b['env'].items() if k!='git_commit'}=={k:v for k,v in c['env'].items() if k!='git_commit'}
        for k,v in b['config'].items():
            if k!='json':assert c['config'][k]==v,k
        for k in ('torch_threads','torch_interop_threads','thread_environment','cpu_affinity','platform'):
            assert b['environment_before'][k]==c['environment_before'][k],k
        br,cr=b['results'][0],c['results'][0];bm,cm=br['median_ms'],cr['median_ms']
        rows.append(dict(mode=mode,variant=variant,batch=batch,length=length,baseline_ms=bm,candidate_ms=cm,
                         provisional_speedup=bm/cm,latency_change_pct=(cm/bm-1)*100,
                         baseline_gpu_peak_gib=br['memory_after']['gpu_peak_allocated_bytes']/2**30,
                         candidate_gpu_peak_gib=cr['memory_after']['gpu_peak_allocated_bytes']/2**30,
                         baseline_kv_capacity_gib=br['memory_after']['kv_cache_storage_bytes']/2**30,
                         candidate_kv_capacity_gib=cr['memory_after']['kv_cache_storage_bytes']/2**30))
    summary=dict(run=run.name,stage='screening',accepted=False,raw_sample_count=80,
                 audit='16 completed independent processes; expected source commits/hashes, actual imported modules, paired model/config/environment, sampling counts and recomputed means all pass.',
                 interpretation='Single short process per side/point; provisional ratios only. No repeatable gain or regression claim.',
                 elapsed_minutes=(m['ended']-m['started'])/60,source_hashes={k:sorted(v) for k,v in hashes.items()},rows=rows)
    (run.parent/'screening-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (run.parent/'screening-summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
