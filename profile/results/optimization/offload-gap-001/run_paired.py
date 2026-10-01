"""Independent balanced baseline/candidate processes for confirmation/validation.

Each job writes raw results even on benchmark failure. Never resumes/restarts
jobs implicitly. --plan-only writes exact commands and measured-cost estimates.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def save(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--candidate-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=('confirmation','full_validation','generation_validation'),required=True)
    parser.add_argument('--screen-manifest',type=Path,required=True,help='Measured baseline process costs and timings for planning')
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    roots={k:getattr(args,k+'_root').resolve() for k in ('baseline','candidate')}
    commits={k:subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip() for k,root in roots.items()}
    for root in roots.values():
        dirty=subprocess.check_output(['git','status','--porcelain','--','fla','profile/*.py'],cwd=root,text=True)
        if dirty and not args.plan_only: raise RuntimeError('Uncommitted implementation: '+dirty)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(roots['candidate']/'profile'))
    from sampling_plan import sampling_plan
    from run_paper_matrix import command
    import run_paper_matrix as matrix
    baseline_screen=json.loads(args.screen_manifest.read_text())
    costs={}
    for job in baseline_screen['jobs']:
        path=args.screen_manifest.parent/(job['name']+'.json')
        if job['status']=='completed' and path.exists():
            row=json.loads(path.read_text())['results'][0]
            measured_calls=8
            setup=max(0,job['finished_unix']-job['started_unix']-row['mean_ms']*measured_calls/1000)
            costs[(job['mode'],job['batch'],job['length'],job['variant'])]=(setup,row['mean_ms'])
    shapes=[(1,2048),(8,2048)] if args.stage!='full_validation' else [(1,2048),(4,2048),(8,2048),(16,2048),(8,512),(8,4096),(8,8192)]
    modes=('generation',) if args.stage=='generation_validation' else ('prefill','decode')
    variants=('ma_offload','ma_gpu') if args.stage=='confirmation' else ('ma_offload','ma_gpu','ma_gpu_unfolded')
    blocks=3 if args.stage=='confirmation' else 1
    jobs=[]
    for block in range(blocks):
        workloads=[(b,l,m) for b,l in shapes for m in modes]
        if block%2: workloads.reverse()
        for wi,(batch,length,mode) in enumerate(workloads):
            order=list(variants)
            if (wi+block)%2: order.reverse()
            for vi,variant in enumerate(order):
                implementations=['baseline','candidate'] if (block+wi+vi)%2==0 else ['candidate','baseline']
                for impl in implementations:
                    job=dict(mode=mode,variant=variant,batch=batch,length=length)
                    name=f'B{block+1}-J{len(jobs)+1:03d}-{impl}-{mode}-{variant}-b{batch}-l{length}'
                    matrix.ROOT=roots[impl]
                    cmd=command(job,output/(name+'.json'),args.stage)
                    sample=sampling_plan(args.stage,mode)
                    reference_variant='ma_gpu' if variant=='ma_gpu_unfolded' else variant
                    # Interpolate planning costs only; never synthesize measurements.
                    def estimate(m):
                        exact=costs.get((m,batch,length,reference_variant))
                        if exact: return exact
                        nearby=costs.get((m,batch,2048,reference_variant),costs.get((m,8,2048,reference_variant),(17,1000)))
                        return nearby[0],nearby[1]*(length/2048 if m=='prefill' else max(1,length/2048))
                    setup,latency=estimate('prefill' if mode=='generation' else mode)
                    if mode=='generation':latency+=128*estimate('decode')[1]
                    seconds=setup+(sample['warmup']+sample['repeats']*sample['rounds'])*latency/1000
                    jobs.append(dict(**job,name=name,implementation=impl,block=block+1,command=cmd,cwd=str(roots[impl]),candidate_sha=commits[impl],sampling_plan=sample,estimated_seconds=seconds,status='pending'))
    manifest=dict(campaign_id='offload-gap-001',protocol_version='offload_gap_v1',stage=args.stage,controller_pid=os.getpid(),source_commits=commits,status='planned',jobs=jobs,planned_work=dict(jobs=len(jobs),estimated_seconds=sum(j['estimated_seconds'] for j in jobs),cost_source=str(args.screen_manifest),note='Setup and call costs measured in screen; other lengths/unfolded/generation extrapolated for planning only. Full validation and generation are separate invocations.'))
    path=output/'manifest.json';save(path,manifest)
    print(json.dumps(manifest['planned_work']),flush=True)
    if args.plan_only:return
    manifest['status']='running';save(path,manifest)
    for job in jobs:
        print('START '+job['name'],flush=True)
        job['started_unix']=time.time()
        with (output/(job['name']+'.txt')).open('w') as log:
            process=subprocess.Popen(job['command'],cwd=job['cwd'],stdout=log,stderr=subprocess.STDOUT)
            job.update(pid=process.pid,status='running');save(path,manifest)
            job['returncode']=process.wait()
        result=output/(job['name']+'.json')
        job['status']=json.loads(result.read_text()).get('status','benchmark_failed') if result.exists() else 'benchmark_failed'
        if job['returncode'] and job['status']=='completed':job['status']='benchmark_failed'
        job['finished_unix']=time.time();save(path,manifest)
        print('END '+job['name']+' '+job['status'],flush=True)
    manifest['status']='completed';save(path,manifest)

if __name__=='__main__':main()
