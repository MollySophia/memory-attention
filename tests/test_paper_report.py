"""Reject corrupt timing records instead of silently drawing misleading charts."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('paper_report', ROOT/'profile/report_paper_matrix.py')
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def test_pending_and_failures_remain_null(tmp_path):
    jobs = [dict(name=str(i), mode='prefill', variant='ma_offload', batch=8, length=2048, status=status)
            for i,status in enumerate(('pending','running','oom','benchmark_failed','interrupted'))]
    (tmp_path/'manifest.json').write_text(json.dumps(dict(jobs=jobs)))
    _, rows = REPORT.collect(tmp_path)
    assert len(rows) == 5
    assert all(r['median_ms'] is None and r['sample_count'] is None for r in rows)


@pytest.mark.parametrize("stage,mode", [(None,"prefill"),("screening","prefill"),("full_validation","prefill"),("full_validation","generation")])
def test_raw_samples_and_source_are_audited(tmp_path, stage, mode):
    job=dict(name='job',mode='prefill',variant='ma_offload',batch=8,length=2048,status='completed')
    (tmp_path/'manifest.json').write_text(json.dumps(dict(jobs=[job],candidate_sha='frozen')))
    data=dict(status='completed',protocol_version='paper_v1',source=dict(git_commit=dict(stdout='frozen\n')),
              config=dict(warmup=30,rounds=5,repeats=30,batch_size=8,seq_len=2048,context_len=2048,
                          num_layers=24,hidden_size=2048,num_heads=32,num_kv_heads=32,intermediate_size=5632,
                          vocab_size=32000,seed=1234,logits_to_keep=1,prefill_workload='inference'),
              env=dict(gpu='NVIDIA GeForce RTX 5090'),
              results=[dict(mode='prefill',variant='ma_offload',samples_ms=[[2.]*30]*5,
                            round_ms=[2.]*5,median_ms=2.,tokens_per_second=8192000.,
                            memory_after=dict(gpu_peak_allocated_bytes=1024,gpu_peak_reserved_bytes=2048,
                                              host_rss_bytes=4096,offload_pinned_bytes=512,kv_cache_storage_bytes=256))])
    expected_count = 150
    if stage is not None:
        from sampling_plan import sampling_plan
        plan = sampling_plan(stage, mode)
        job.update(mode=mode, sampling_plan=plan)
        (tmp_path/'manifest.json').write_text(json.dumps(dict(jobs=[job],candidate_sha='frozen')))
        data.update(stage=stage, measurement_plan_id=plan['measurement_plan_id'])
        data['config'].update(plan)
        data['config']['generation_steps'] = 128
        data['results'][0].update(mode=mode, sampling_plan=plan,
            samples_ms=[[2.]*plan['repeats'] for _ in range(plan['rounds'])],
            round_ms=[2.]*plan['rounds'],
            tokens_per_second=8*(2176 if mode=='generation' else 2048)*500.)
        expected_count = plan['repeats']*plan['rounds']
    path=tmp_path/'job.json'
    path.write_text(json.dumps(data))
    assert REPORT.collect(tmp_path)[1][0]['sample_count'] == expected_count
    data['results'][0]['median_ms']=1.
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError): REPORT.collect(tmp_path)
    data['results'][0]['median_ms']=2.
    data['source']['git_commit']['stdout']='other_source'
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError): REPORT.collect(tmp_path)
