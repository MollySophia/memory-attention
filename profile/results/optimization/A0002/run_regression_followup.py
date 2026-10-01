"""Predeclared three fresh pairs for completed full-matrix regression signals.

Requires the full audit first. Original single-pair points are retained, not
pooled with fresh confirmation pairs (some original baselines were reused).
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ATTEMPT=Path(__file__).resolve().parent
ROOT=ATTEMPT.parents[3]
BASELINE=ROOT.parent/'baseline-paper-001'
sys.path.insert(0,str(ROOT/'profile'))
from run_paper_matrix import command
from sampling_plan import sampling_plan
sys.path.insert(0,str(ATTEMPT))
from run_full_validation import source_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args();audit=json.loads(args.audit.read_text())
    assert not audit['partial'] and audit['controller_status'] in ('complete','complete_with_failures')
    cases=[r for r in audit['comparisons'] if r['requires_investigation'] and r['baseline_status']==r['candidate_status']=='completed']
    failures=[r for r in audit['comparisons'] if r['candidate_status']!='completed' or r['baseline_status']!='completed']
    expected={};hashes=json.loads((ATTEMPT/'confirmation-summary.json').read_text())['source_hashes']
    for side,checkout in [('baseline',BASELINE),('candidate',ROOT)]:
        assert not subprocess.check_output(['git','status','--porcelain','--','fla','profile/*.py'],cwd=checkout,text=True).strip()
        expected[side]=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()
        assert hashes[side]==[source_hash(checkout)]
    output=args.output.resolve();jobs=[]
    for number,case in enumerate(cases,1):
        point={k:case[k] for k in ('mode','variant','batch','length')}
        plan=sampling_plan('full_validation',point['mode'])
        for pair in (1,2,3):
            for side in (('baseline','candidate') if pair % 2 else ('candidate','baseline')):
                name=f'C{number:02d}-p{pair}-{side}'
                cmd=command(point,output/(name+'.json'),'full_validation')
                if side=='baseline':
                    cmd[1]=str(BASELINE/'profile/bench_fla.py');i=cmd.index('--stage');del cmd[i:i+2]
                jobs.append(dict(**point,name=name,case=number,pair=pair,side=side,command=cmd,
                                 sampling_plan=plan,expected_commit=expected[side],status='pending'))
    estimated_seconds=sum(19+(case['baseline_ms']+case['candidate_ms'])/2/1000*(2+5*3 if case['mode']=='generation' else 40) for case in cases for _ in range(6))
    manifest=dict(estimated_minutes=estimated_seconds/60,cost_assumptions='19s setup/import per process plus measured point latency times planned warmup/sample count; generation units are full trajectories.',stage='matrix_regression_confirmation',status='planned',controller_pid=os.getpid(),
                  original_audit=str(args.audit.resolve()),cases=cases,unresolved_failed_points=failures,
                  job_count=len(jobs),jobs=jobs,started=time.time(),
                  analysis_plan='Three fresh independent alternating pairs per flagged completed point. Do not pool original points. Geometric paired speedup and95% Student-t CI on log ratios (df2). Upper bound below1: resolved regression; otherwise within_noise unless lower bound above1. No optional extension for favorable results.',
                  acceptance_note='Failed/OOM/unsupported points remain unresolved separately; this followup cannot silently clear them.')
    output.mkdir(parents=True,exist_ok=False)
    def save():
        temp=output/'manifest.tmp';temp.write_text(json.dumps(manifest,indent=2)+'\n');temp.replace(output/'manifest.json')
    save()
    if args.plan_only:
        print(f'{len(cases)} completed signals, {len(jobs)} new processes, {len(failures)} other points needing review');return
    manifest['status']='running';save()
    for job in jobs:
        checkout=BASELINE if job['side']=='baseline' else ROOT
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()==job['expected_commit']
        env=dict(os.environ,PYTHONPATH=str(checkout))
        probe=subprocess.check_output([sys.executable,'-c','import fla.models.utils as u; print(u.__file__)'],cwd=checkout,env=env,text=True).strip()
        assert Path(probe).resolve()==checkout/'fla/models/utils.py'
        job.update(status='running',started=time.time(),verified_cache_module=probe)
        with (output/(job['name']+'.txt')).open('w') as log:
            child=subprocess.Popen(job['command'],cwd=checkout,env=env,stdout=log,stderr=subprocess.STDOUT)
            job['pid']=child.pid;save();job['returncode']=child.wait()
        path=output/(job['name']+'.json');status=json.loads(path.read_text()).get('status') if path.exists() else 'missing'
        job.update(result_status=status,status='completed' if status=='completed' and job['returncode']==0 else 'failed',ended=time.time());save()
        print(job['name'],job['status'],flush=True)
        if job['status']=='failed':
            manifest.update(status='failed',ended=time.time());save();raise SystemExit(1)
    manifest.update(status='complete',ended=time.time());save()


if __name__=='__main__':main()
