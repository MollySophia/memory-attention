"""Verify raw evidence and memory bookkeeping for an additional-attempt tranche."""
import argparse,json,math,re,subprocess
from pathlib import Path
from run_paired import source_hash,validate_balanced_order


def read(path):return json.loads(path.read_text())


def validate_offload_memory(attempt, memory):
    """Count every retained allocation, including transport metadata and tables."""
    caps = memory['offloader_capacities']
    assert memory['cpu_table_bytes'] == 3000 * 2**20
    assert memory['offload_gpu_buffer_bytes'] == sum(c['gpu_bytes'] for c in caps)
    table_pinned = memory.get('cpu_table_pinned_bytes', 0)
    if attempt in ('A0008', 'A0013'):
        assert table_pinned == 3000 * 2**20
    else:
        assert table_pinned == 0
    assert memory['offload_pinned_bytes'] == table_pinned + sum(c['host_bytes'] for c in caps)
    for cap in caps:
        tokens = cap['batch'] * cap['length']
        host, gpu = cap['host_bytes'], cap['gpu_bytes']
        if attempt == 'A0010' and cap['policy'] == 'pipeline':
            assert host == 2 * gpu
        if attempt in ('A0009', 'A0015') and cap['policy'] == 'pipeline':
            inverse = tokens * 8 if tokens >= 8192 else 0
            assert gpu == host + inverse
        if attempt == 'A0011':
            assert host == gpu + tokens * 8
        if attempt == 'A0013':
            if cap['policy'] == 'bulk' and 8 <= tokens <= 16:
                assert host == 0 and gpu == tokens * 24 * 2048 * 2
            else:
                assert host == gpu
        if attempt in ('A0014', 'A0016'):
            depth = 1 if attempt == 'A0014' or tokens <= 2048 else 4
            expected = tokens * 2048 * 2 * (depth if cap['policy'] == 'pipeline' else 24)
            assert host == gpu == expected


def audit(campaign,continuation):
    tranche=read(campaign/continuation/'manifest.json')
    start=int(tranche['first_new_attempt'][1:])
    paths=[p for p in sorted(campaign.glob('A[0-9][0-9][0-9][0-9]/record.json')) if int(p.parent.name[1:])>=start]
    assert tranche['minimum_additional_attempts']<=len(paths)<=tranche['maximum_additional_attempts']
    checked=[];environments=set()
    def raw(path,sha,root,stage):
        payload=read(path)
        assert payload['status']=='completed' and payload['protocol_version']=='offload_gap_v1'
        assert payload['source']['git_commit']['stdout'].strip()==sha
        assert payload['source']['source_sha256']==source_hash(root)
        assert payload['stage']==stage
        config=payload['config'];row=payload['results'][0]
        assert config['seed']==1234 and config['logits_to_keep']==1 and row['output_scope']=='cached_logits'
        assert config['hidden_size']==2048 and config['num_layers']==24 and config['num_heads']==config['num_kv_heads']==32
        assert config['intermediate_size']==5632 and config['vocab_size']==32000
        rounds,repeats,warmup=(1,5,3) if stage=='screening' else (3,10,10)
        assert config['warmup']==warmup and len(row['samples_ms'])==rounds
        assert all(len(s)==repeats and all(math.isfinite(v) and v>0 for v in s) for s in row['samples_ms'])
        assert payload['measurement_plan_id']==('screen_v1_w3_n5_r1' if stage=='screening' else 'formal_v1_w10_n10_r3')
        env=payload['env'];before=payload['environment_before']
        environments.add((env['torch'],env['torch_cuda'],env['flash_attn'],env['gpu'],env['python'],env['cuda_visible_devices'],before['gpu_telemetry']['stdout'].splitlines()[1].split(', ')[1],tuple(before['cpu_affinity']),before['torch_threads'],before['torch_interop_threads'],json.dumps(before['thread_environment'],sort_keys=True)))
        checked.append(str(path));return row
    records=[]
    for path in paths:
        r=read(path);d=path.parent
        assert r['continuation_id']==continuation and r['workflow_stage']=='final_verdict'
        assert r['status'] in ('accepted','rejected','within_noise')
        sha=r['candidate_sha'];root=Path(r['frozen_checkout'])
        assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==sha
        # A real model implementation must distinguish every counted attempt.
        changed=subprocess.check_output(['git','-C',str(root),'diff','--name-only',r['parent_source_sha'],sha,'--','fla'],text=True).splitlines()
        assert changed and r['hypothesis'] and r['profile_evidence']
        tests=(d/'correctness-tests.txt').read_text();standalone=(d/'standalone-correctness.txt').read_text()
        assert re.search(r'=+\s+\d+ passed in',tests) and not re.search(r'\d+ failed[, ]',tests)
        assert 'RESULT: PASS (worst max diff 0.000000' in standalone
        screen=read(d/r['screen_manifest']);assert screen['status']=='completed' and len(screen['jobs'])==12
        expected={(mode,batch,variant) for mode in ('prefill','decode') for batch in (1,8,16) for variant in ('ma_offload','ma_gpu')}
        assert {(j['mode'],j['batch'],j['variant']) for j in screen['jobs']}==expected
        for job in screen['jobs']:
            assert job['status']=='completed' and job['length']==2048
            row=raw(d/'R01'/(job['name']+'.json'),sha,root,'screening')
            memory=row['memory_after'];caps=memory['offloader_capacities']
            if job['variant']=='ma_offload':
                validate_offload_memory(r['attempt_id'], memory)
        confirmation=r.get('parent_confirmation_manifest')
        if confirmation:
            manifest=read(d/confirmation);assert manifest['status']=='completed' and len(manifest['jobs'])==48
            assert manifest['pairing_plan_id']=='balanced_order_v2';validate_balanced_order(manifest['jobs'])
            parent=read(campaign/r['parent_attempt_id']/'record.json')
            assert manifest['source_commits']==dict(baseline=parent['candidate_sha'],candidate=sha)
            for job in manifest['jobs']:
                impl=job['implementation'];sr=parent if impl=='baseline' else r
                assert job['status']=='completed'
                raw((d/confirmation).parent/(job['name']+'.json'),sr['candidate_sha'],Path(sr['frozen_checkout']),'confirmation')
        records.append(dict(attempt=r['attempt_id'],status=r['status'],source_sha=sha,changed_model_files=changed,screen_jobs=12,parent_confirmation_jobs=48 if confirmation else 0))
    assert len(environments)==1
    return dict(campaign_id='offload-gap-001',continuation=continuation,status='additional_raw_evidence_checks_passed',attempts=records,checked_raw_results=len(checked),source_results=checked,limitations='Does not substitute for retention gates, report/figure review, or Git publication verification.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--continuation',default='continuation-01');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.campaign,a.continuation);a.output.write_text(json.dumps(result,indent=2)+'\n');print(result['status'],result['checked_raw_results'])
