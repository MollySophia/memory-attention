"""Fresh all16 A0000 comparison for a retained implementation's cumulative figures.

This does not replace independent final offload-versus-GPU acceptance.
Prepare and commit the plan before running; never reuse or extend selection data.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import driver as d
from final_target import committed_plan, digest


def location(attempt):
    return d.C / attempt / 'cumulative-A0000'


def helper_hashes():
    paths=[Path(__file__),d.D/'driver.py',d.D/'final_target.py',d.C/'run_paired.py',
           d.C/'analyze_paired.py',d.C/'audit_continuation.py']
    return {str(p.relative_to(d.C)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def prepare(attempt):
    assert attempt!='A0000' and d.record(attempt)['status']=='accepted', 'Choose a retained optimized source'
    directory=location(attempt)
    assert not directory.exists(), 'Never replace or extend an existing cumulative plan'
    p=d.plan(attempt,d.WORKLOADS,'A0000',True,directory)
    p.update(stage='cumulative_baseline',purpose='all16_fresh_A0000_comparison',
             created_unix=time.time(),helper_hashes=helper_hashes(),
             stopping_rule='Exactly the predeclared independent blocks; no optional extension or selection-data reuse')
    directory.mkdir()
    d.save(directory/'plan.json',p)
    manifest=copy.deepcopy(p);manifest['plan_sha256']=digest(p)
    d.save(directory/'manifest.json',manifest)
    print(json.dumps(p['planned_work']))
    return p


def validate(plan,manifest):
    attempt=plan['attempt'];directory=location(attempt)
    assert attempt!='A0000' and d.record(attempt)['status']=='accepted'
    assert plan['purpose']=='all16_fresh_A0000_comparison'
    assert plan['stage']==manifest['stage']=='cumulative_baseline'
    assert plan['baseline']==manifest['baseline']=='A0000'
    assert manifest['attempt']==attempt and manifest['formal'] and plan['formal']
    assert manifest['plan_sha256']==digest(plan)
    assert plan['helper_hashes']==helper_hashes(), 'Runner or audit changed after freeze'
    regenerated=d.plan(attempt,d.WORKLOADS,'A0000',True,directory)
    assert plan['source_signatures']==manifest['source_signatures']==regenerated['source_signatures']
    assert plan['jobs']==regenerated['jobs'], 'Changed planned commands, source, coverage or order'
    blocks=d.planned_blocks(plan)
    assert d.planned_blocks(manifest)==blocks
    assert len(manifest['jobs'])==len(plan['jobs'])==64*blocks
    for expected,actual in zip(plan['jobs'],manifest['jobs']):
        assert all(actual[k]==v for k,v in expected.items() if k!='status'), 'Changed job identity or order'
    d.validate_balanced_order(manifest['jobs'],blocks)
    return blocks


def run(attempt):
    directory=location(attempt)
    plan,manifest=[d.read(directory/name) for name in ('plan.json','manifest.json')]
    validate(plan,manifest)
    assert manifest['status']=='planned' and all(j['status']=='pending' for j in manifest['jobs'])
    assert all(not (directory/(j['name']+ext)).exists() for j in plan['jobs'] for ext in ('.json','.txt'))
    manifest['frozen_plan_commit']=committed_plan(directory)
    d.save(directory/'manifest.json',manifest)
    d.run(directory)
    return audit(attempt)


def audit(attempt):
    directory=location(attempt)
    plan,manifest=[d.read(directory/name) for name in ('plan.json','manifest.json')]
    blocks=validate(plan,manifest)
    assert manifest['status']=='completed'
    assert manifest['started_unix']>=plan['created_unix']
    assert manifest['frozen_plan_commit']==committed_plan(directory)
    for j in manifest['jobs']:
        assert j['status']=='completed' and j['returncode']==0
        assert j['finished_unix']>=j['started_unix']>=manifest['started_unix']
        _,row=d.audit_job(j,directory/(j['name']+'.json'))
        mean=statistics.mean(statistics.mean(xs) for xs in row['samples_ms'])
        assert math.isclose(row['mean_ms'],mean,rel_tol=1e-12,abs_tol=1e-12)
    summary=d.summarize(directory)
    assert len(summary['rows'])==16
    assert {(r['mode'],r['batch'],r['length']) for r in summary['rows']}==set(d.WORKLOADS)
    rows=[]
    for r in summary['rows']:
        latencies=r['latencies']
        out=dict(mode=r['mode'],batch=r['batch'],length=r['length'],independent_blocks=blocks,
                 baseline='A0000',candidate=attempt,source_manifest=r['source_manifest'],
                 memory_savings_preserved=r['memory_savings_preserved'])
        for k in ('baseline_offload_ms','baseline_gpu_ms','candidate_offload_ms','candidate_gpu_ms'):
            out[k]=statistics.mean(l[k] for l in latencies)
        for metric in ('offload_speedup','offload_reduction_ms','gap_reduction_ms'):
            for field in ('mean','lower_95','upper_95'):
                out[metric+'_'+field]=r['statistics'][metric][field]
        rows.append(out)
    result=dict(status='audited',attempt=attempt,baseline='A0000',independent_blocks=blocks,
                checked_processes=len(manifest['jobs']),rows=rows,plan_sha256=digest(plan),
                source_signatures=plan['source_signatures'],goal_accepted=False,
                note='Cumulative A0000 comparisons with descriptive paired95% intervals. No corrected local-gain claim or final near-GPU acceptance; final target requires separate independent evidence.')
    d.save(directory/'analysis.json',result)
    with (directory/'summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(attempt=attempt,checked_processes=len(manifest['jobs']),workloads=len(rows),goal_accepted=False)))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','audit'));parser.add_argument('--attempt',required=True)
    args=parser.parse_args()
    {'prepare':prepare,'run':run,'audit':audit}[args.action](args.attempt)
