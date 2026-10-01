"""Independent balanced baseline/candidate processes for confirmation/validation.

Each job writes raw results even on benchmark failure. Never resumes/restarts
jobs implicitly. --plan-only writes exact commands and measured-cost estimates.
"""
import argparse
import hashlib
import importlib.metadata
import platform
import statistics
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def save(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)



def source_hash(root):
    paths=sorted(set((root/'fla').rglob('*.py')) | set((root/'profile').glob('*.py')) |
                 {p for p in (root/'setup.py',root/'pyproject.toml') if p.exists()})
    digest=hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode()+b'\0'+path.read_bytes()+b'\0')
    return digest.hexdigest()


def validate_reuse_payload(payload, job, commit, digest, expected_env):
    assert payload['status']=='completed'
    assert payload['campaign_id']=='offload-gap-001'
    assert payload['protocol_version']=='offload_gap_v1'
    assert payload['measurement_plan_id']=='formal_v1_w10_n10_r3'
    assert payload['source']['git_commit']['stdout'].strip()==commit
    assert payload['source']['source_sha256']==digest
    config=payload['config']
    for key,value in dict(seed=1234,batch_size=job['batch'],seq_len=job['length'],
                          context_len=job['length'],logits_to_keep=1,prefill_workload='inference',
                          warmup=10,repeats=10,rounds=3,hidden_size=2048,num_layers=24,
                          num_heads=32,num_kv_heads=32,intermediate_size=5632,vocab_size=32000,
                          policy='auto',group_size=1,prefetch_depth=4).items():
        assert config[key]==value,(key,config[key],value)
    assert config['mode']==job['mode'] and config['variants']==[job['variant']]
    for key,value in dict(hidden_size=2048,num_hidden_layers=24,num_heads=32,num_kv_heads=32,
                          intermediate_size=5632,vocab_size=32000,qk_norm=False,use_gate=False,
                          fuse_norm=False,tie_word_embeddings=False).items():
        assert payload['model_config'][key]==value
    assert len(payload['results'])==1
    row=payload['results'][0]
    assert row['output_scope']=='cached_logits' and row['logits_to_keep']==1
    assert len(row['samples_ms'])==3 and all(len(r)==10 for r in row['samples_ms'])
    for key,value in expected_env.items():
        assert payload['env'][key]==value,(key,payload['env'][key],value)
    return True


def reuse_index(path, roots, commits):
    manifest=json.loads(path.read_text())
    assert manifest['status']=='completed' and manifest['stage']=='confirmation'
    assert len(manifest['jobs'])==48 and manifest['source_commits']==commits
    expected_env=dict(torch=importlib.metadata.version('torch'),
                      flash_attn=importlib.metadata.version('flash_attn'),
                      python=platform.python_version(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES',''))
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip().split(', ')
    assert len(gpu)==2, 'Reuse currently requires one visible physical GPU'
    expected_env['gpu']=gpu[0]
    hashes={impl:source_hash(root) for impl,root in roots.items()}
    index={}
    for job in manifest['jobs']:
        assert job['status']=='completed'
        result=path.parent/(job['name']+'.json')
        payload=json.loads(result.read_text())
        validate_reuse_payload(payload,job,commits[job['implementation']],hashes[job['implementation']],expected_env)
        assert Path(payload['env']['model_module']).resolve().is_relative_to(roots[job['implementation']])
        before=payload['environment_before']
        recorded_gpu=before['gpu_telemetry']['stdout'].splitlines()[1].split(', ')
        assert recorded_gpu[:2]==gpu
        assert before['cpu_affinity']==sorted(os.sched_getaffinity(0))
        assert before['thread_environment']=={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}
        key=(job['implementation'],job['mode'],job['variant'],job['batch'],job['length'])
        index.setdefault(key,[]).append(str(result.resolve()))
    assert len(index)==16 and all(len(v)==3 for v in index.values())
    return index


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--candidate-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=('confirmation','full_validation','generation_validation'),required=True)
    parser.add_argument('--screen-manifest',type=Path,required=True,help='Measured baseline process costs and timings for planning')
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--reuse-confirmation',type=Path,help='Reuse all three matching independent primary blocks in full validation')
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
    paused_pids=set()
    pause=args.screen_manifest.parent.parent/'profile-pause.json'
    if pause.exists():
        paused_pids={j.get('pid') for j in json.loads(pause.read_text())['manifest_snapshot']['jobs'] if j['status']=='running'}
    for job in baseline_screen['jobs']:
        path=args.screen_manifest.parent/(job['name']+'.json')
        if job['status']=='completed' and path.exists():
            row=json.loads(path.read_text())['results'][0]
            measured_calls=8
            setup=max(0,job['finished_unix']-job['started_unix']-row['mean_ms']*measured_calls/1000)
            costs[(job['mode'],job['batch'],job['length'],job['variant'])]=(None if job.get('pid') in paused_pids else setup,row['mean_ms'])
    for key,(setup,latency) in list(costs.items()):
        if setup is None:
            costs[key]=(statistics.median(s for k,(s,_) in costs.items() if s is not None and k[3]==key[3]),latency)
    reused={}
    if args.reuse_confirmation:
        assert args.stage=='full_validation'
        reused=reuse_index(args.reuse_confirmation,roots,commits)
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
    for job in jobs:
        refs=reused.get((job['implementation'],job['mode'],job['variant'],job['batch'],job['length']))
        if refs:
            job.update(reused_results=refs,status='reused',estimated_seconds=0,planned_command_not_executed=job.pop('command'))
    manifest=dict(campaign_id='offload-gap-001',protocol_version='offload_gap_v1',stage=args.stage,controller_pid=os.getpid(),source_commits=commits,status='planned',jobs=jobs,planned_work=dict(jobs=len(jobs),new_process_jobs=sum(j['status']=='pending' for j in jobs),reused_points=sum(j['status']=='reused' for j in jobs),estimated_seconds=sum(j['estimated_seconds'] for j in jobs),cost_source=str(args.screen_manifest),note='Setup and call costs measured in screen; other lengths/unfolded/generation extrapolated for planning only. Full validation and generation are separate invocations. Deliberately paused setup durations replaced with median unpaused setup for the placement. Reused confirmation keeps original files/stage and all three process blocks.'))
    path=output/'manifest.json';save(path,manifest)
    print(json.dumps(manifest['planned_work']),flush=True)
    if args.plan_only:return
    manifest['status']='running';save(path,manifest)
    for job in jobs:
        if job['status']=='reused':
            print('REUSE '+job['name'],flush=True)
            continue
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
