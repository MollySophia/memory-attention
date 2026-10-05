import copy
import pytest
import driver as d


def test_configured_mapped_range_and_strict_memory_totals():
    path = d.C/'A0023/R01-complete-screen/manifest.json'
    manifest = d.read(path)
    job = manifest['jobs'][1]
    raw = path.parent/(job['name']+'.json')
    # Real expanded-range payload must pass: prefix also retains mapped1 decode buffer.
    d.audit_job(job, raw)
    original = d.read(raw)
    old_read = d.read
    for field in ('offload_pinned_bytes','offload_gpu_buffer_bytes','cpu_table_pinned_bytes'):
        bad = copy.deepcopy(original)
        bad['results'][0]['memory_after'][field] += 1
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(d,'read',lambda p: bad if p==raw else old_read(p))
            with pytest.raises(AssertionError):
                d.audit_job(job,raw)
    bad = copy.deepcopy(original)
    bad['model_config']['memory_offload_mapped_bulk_min_tokens'] = 8
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(d,'read',lambda p: bad if p==raw else old_read(p))
        with pytest.raises(AssertionError):
            d.audit_job(job,raw)


def test_recovery_never_reruns_completed_jobs_and_rejects_live_manifest(tmp_path, monkeypatch):
    # A recovery with all jobs already complete must not create any child process.
    monkeypatch.setattr(d,'signature',lambda a: dict(attempt=a))
    monkeypatch.setattr(d,'audit_job',lambda *args: None)
    monkeypatch.setattr(d.subprocess,'Popen',lambda *args,**kwargs: pytest.fail('Completed job rerun'))
    p=dict(formal=False,status='stopped_needs_review',controller_pid=123,source_signatures={'candidate':dict(attempt='A0023')},jobs=[dict(status='completed',returncode=0,name='done')])
    d.save(tmp_path/'manifest.json',p)
    d.run(tmp_path,resume=True)
    final=d.read(tmp_path/'manifest.json')
    assert final['status']=='completed' and final['jobs']==p['jobs']
    assert len(final['recovery_history'])==1
    for status in ('running','completed','planned'):
        p['status']=status;d.save(tmp_path/'manifest.json',p)
        with pytest.raises(AssertionError):d.run(tmp_path,resume=True)
    p['status']='stopped_needs_review';p['jobs'][0]['status']='failed';d.save(tmp_path/'manifest.json',p)
    with pytest.raises(AssertionError):d.run(tmp_path,resume=True)
