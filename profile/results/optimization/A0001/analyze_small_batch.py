"""Audit and analyze three independent batch1 decode pairs per placement."""
import csv
import json
import math
from pathlib import Path
import statistics
import sys


def main():
    run=Path(sys.argv[1]).resolve();manifest=json.loads((run/'manifest.json').read_text())
    assert manifest['status']=='complete'
    records={};hashes={'baseline':set(),'candidate':set()};pids=set()
    for job in manifest['jobs']:
        assert job['status'] in ('completed','reused')
        assert job['pid'] not in pids;pids.add(job['pid'])
        d=json.loads(Path(job['result']).read_text());c=d['config'];r=d['results'][0]
        assert d['status']=='completed' and d['protocol_version']=='paper_v1'
        assert d['source']['git_commit']['stdout'].strip()==job['expected_commit']
        assert not d['source']['source_patch']['stdout']
        hashes[job['side']].add(d['source']['source_sha256'])
        assert (c['mode'],c['batch_size'],c['context_len'],c['warmup'],c['repeats'],c['rounds'])==('decode',1,2048,10,10,3)
        assert (r['variant'],r['mode'])==(job['variant'],'decode')
        samples=r['samples_ms'];assert len(samples)==3 and all(len(x)==10 for x in samples)
        assert all(math.isfinite(v) and v>0 for x in samples for v in x)
        means=[statistics.mean(x) for x in samples]
        assert means==r['round_ms'] and statistics.median(means)==r['median_ms']
        assert job['verified_cache_module'].endswith(('/baseline-paper-001' if job['side']=='baseline' else '/memory-attention')+'/fla/models/utils.py')
        records[(job['variant'],job['pair'],job['side'])]=d
    assert all(len(v)==1 for v in hashes.values())
    rows=[];pairs=[]
    for variant in ('ma_offload','ma_gpu'):
        ratios=[]
        for pair in (1,2,3):
            b,c=(records[(variant,pair,s)] for s in ('baseline','candidate'))
            assert b['model_config']==c['model_config']
            assert {k:v for k,v in b['env'].items() if k!='git_commit'}=={k:v for k,v in c['env'].items() if k!='git_commit'}
            for k,v in b['config'].items():
                if k!='json':assert c['config'][k]==v,k
            for k in ('torch_threads','torch_interop_threads','thread_environment','cpu_affinity','platform'):
                assert b['environment_before'][k]==c['environment_before'][k]
            bm,cm=(d['results'][0]['median_ms'] for d in (b,c));ratios.append(bm/cm)
            pairs.append(dict(variant=variant,pair=pair,baseline_ms=bm,candidate_ms=cm,speedup=bm/cm))
        x=[math.log(r) for r in ratios];center=statistics.mean(x);half=4.302652729911275*statistics.stdev(x)/math.sqrt(3)
        low,high=math.exp(center-half),math.exp(center+half)
        rows.append(dict(variant=variant,mode='decode',batch=1,context=2048,geometric_mean_speedup=math.exp(center),ci95=[low,high],
                         geometric_latency_increase_pct=(math.exp(-center)-1)*100,
                         verdict='resolved_regression' if high<1 else 'repeatable_improvement' if low>1 else 'within_noise'))
    summary=dict(run=run.name,rows=rows,pairs=pairs,sample_count=360,independent_pairs_per_placement=3,
                 estimator='Median of round means per process; geometric mean of paired speedups.',
                 uncertainty='Predeclared95% Student-t CI on paired log ratios, df2; assumes approximately normal paired log ratios.',
                 source_hashes={k:sorted(v) for k,v in hashes.items()},
                 audit='Raw samples, round/run estimates, source SHA/hash/patch, scope/counts, actual module paths, model/config/environment and independent process IDs verified.',
                 regression_confirmed=any(r['verdict']=='resolved_regression' for r in rows),accepted=False)
    (run.parent/'small-batch-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (run.parent/'small-batch-pairs.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
