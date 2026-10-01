"""Audit predeclared fresh process pairs for full-matrix regression signals."""
import csv
import json
import math
from pathlib import Path
import statistics
import sys

ATTEMPT=Path(__file__).resolve().parent
ROOT=ATTEMPT.parents[3]
sys.path.insert(0,str(ATTEMPT))
from run_full_validation import environment_signature


def main():
    run=Path(sys.argv[1]).resolve();m=json.loads((run/'manifest.json').read_text())
    assert m['status']=='complete'
    hashes=json.loads((ATTEMPT/'confirmation-summary.json').read_text())['source_hashes']
    records={};pids=set();samples_total=0
    for j in m['jobs']:
        assert j['status']=='completed' and j['returncode']==0
        assert j['pid'] not in pids;pids.add(j['pid'])
        d=json.loads((run/(j['name']+'.json')).read_text());c=d['config'];r=d['results'][0]
        assert d['status']=='completed' and d['protocol_version']=='paper_v1'
        assert d['source']['git_commit']['stdout'].strip()==j['expected_commit']
        assert d['source']['source_sha256'] in hashes[j['side']]
        assert not d['source']['source_patch']['stdout']
        checkout=ROOT.parent/'baseline-paper-001' if j['side']=='baseline' else ROOT
        assert Path(j['verified_cache_module']).resolve()==checkout/'fla/models/utils.py'
        plan=j['sampling_plan']
        assert all(c[k]==plan[k] for k in ('warmup','repeats','rounds'))
        assert (c['mode'],c['batch_size'],c['context_len'],c['seq_len'])==(j['mode'],j['batch'],j['length'],j['length'])
        assert (r['mode'],r['variant'])==(j['mode'],j['variant'])
        if j['side']=='candidate':assert r['sampling_plan']==plan
        samples=r['samples_ms'];assert len(samples)==plan['rounds'] and all(len(s)==plan['repeats'] for s in samples)
        assert all(math.isfinite(x) and x>0 for s in samples for x in s)
        means=[statistics.mean(s) for s in samples]
        assert means==r['round_ms'] and statistics.median(means)==r['median_ms']
        samples_total+=sum(map(len,samples));records[(j['case'],j['pair'],j['side'])]=d
    summaries=[];pairs=[]
    for number,original in enumerate(m['cases'],1):
        ratios=[]
        for pair in (1,2,3):
            b,c=(records[(number,pair,s)] for s in ('baseline','candidate'))
            assert b['model_config']==c['model_config']
            assert environment_signature(b)==environment_signature(c)
            for k,v in b['config'].items():
                if k!='json':assert c['config'][k]==v,k
            bm,cm=(d['results'][0]['median_ms'] for d in (b,c));ratios.append(bm/cm)
            pairs.append(dict(case=number,pair=pair,**{k:original[k] for k in ('mode','variant','batch','length')},
                              baseline_ms=bm,candidate_ms=cm,speedup=bm/cm))
        logs=list(map(math.log,ratios));center=statistics.mean(logs);half=4.302652729911275*statistics.stdev(logs)/math.sqrt(3)
        low,high=math.exp(center-half),math.exp(center+half)
        summaries.append(dict(case=number,**{k:original[k] for k in ('mode','variant','batch','length')},
                              geometric_mean_speedup=math.exp(center),ci95=[low,high],
                              verdict='resolved_regression' if high<1 else 'repeatable_improvement' if low>1 else 'within_noise',
                              original_single_pair_speedup=original['speedup']))
    payload=dict(run=run.name,job_count=len(pids),sample_count=samples_total,rows=summaries,pairs=pairs,
                 unresolved_failed_points=m['unresolved_failed_points'],
                 any_resolved_regression=any(r['verdict']=='resolved_regression' for r in summaries),
                 accepted=False,original_points_pooled=False,
                 uncertainty='95% Student-t interval on three fresh paired log speedups,df2. Small-n normal-log assumption. within_noise does not establish equivalence.',
                 audit='Unique process IDs, exact source commits/hashes/imports, clean source patches, paired config/model/environment, sampling counts, finite samples and recomputed estimators verified.')
    (run.parent/'regression-followup-summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    with (run.parent/'regression-followup-pairs.csv').open('w') as f:
        if pairs:
            w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    print(json.dumps(payload,indent=2))


if __name__=='__main__':main()
