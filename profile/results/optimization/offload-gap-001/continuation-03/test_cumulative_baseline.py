"""CPU-only cumulative plan and shared-runner tests; no GPU measurements."""
import copy
from pathlib import Path
import pytest
import cumulative_baseline as c


@pytest.fixture(params=[3,6])
def prepared(tmp_path,monkeypatch,request):
    attempt='A9999';directory=tmp_path/'cumulative';blocks=request.param
    monkeypatch.setattr(c,'location',lambda a:directory)
    monkeypatch.setattr(c.d,'signature',lambda a:dict(attempt=a,root=str(tmp_path/a),commit='sha-'+a,source_sha256='hash-'+a))
    monkeypatch.setattr(c.d,'record',lambda a:dict(status='accepted',confirmation_blocks=blocks))
    monkeypatch.setattr(c.d,'estimate',lambda *args:1.)
    plan=c.prepare(attempt)
    return attempt,directory,plan,c.d.read(directory/'manifest.json'),blocks


def test_frozen_complete_A0000_plan_and_mutation_guards(prepared):
    attempt,directory,plan,manifest,blocks=prepared
    assert c.validate(plan,manifest)==blocks
    assert len(plan['jobs'])==64*blocks
    with pytest.raises(AssertionError):c.prepare(attempt)
    for mutation in ('missing','parent','command','count'):
        bad=copy.deepcopy(manifest)
        if mutation=='missing':bad['jobs'].pop()
        elif mutation=='parent':bad['baseline']='A0028'
        elif mutation=='command':bad['jobs'][0]['command'][-1]='other.json'
        else:bad['independent_blocks']=blocks+1
        with pytest.raises(AssertionError):c.validate(plan,bad)


def test_cumulative_real_driver_handoff_and_exact_A0000_ratios(prepared,monkeypatch):
    attempt,directory,plan,manifest,blocks=prepared
    monkeypatch.setattr(c,'committed_plan',lambda path:'frozen-plan-commit')
    launches=[]
    class Child:
        def __init__(self,command,**kwargs):
            launches.append(command);self.pid=12300+len(launches)
            self.path=Path(command[command.index('--json')+1])
        def wait(self):
            c.d.save(self.path,dict(status='completed'));return 0
    monkeypatch.setattr(c.d.subprocess,'Popen',Child)
    def raw(job,path):
        baseline=job['implementation']=='baseline';gpu=job['variant']=='ma_gpu'
        mean=(100. if baseline else 80.) if gpu else (120. if baseline else 90.)
        return ('same-environment',),dict(mean_ms=mean,samples_ms=[[mean]*10 for _ in range(3)],
                                         memory_after=dict(gpu_peak_allocated_bytes=200 if gpu else 100))
    monkeypatch.setattr(c.d,'audit_job',raw)
    result=c.run(attempt)
    assert len(launches)==64*blocks
    assert result['baseline']=='A0000' and len(result['rows'])==16 and not result['goal_accepted']
    for row in result['rows']:
        assert row['independent_blocks']==blocks
        assert row['offload_speedup_mean']==pytest.approx(120./90.)
        assert row['offload_reduction_ms_mean']==30.
        assert row['gap_reduction_ms_mean']==10.
    with pytest.raises(AssertionError):c.run(attempt)
    assert len(launches)==64*blocks


def test_nonretained_source_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(c.d,'record',lambda a:dict(status='testing'))
    monkeypatch.setattr(c,'location',lambda a:tmp_path/'unused')
    with pytest.raises(AssertionError):c.prepare('A9999')
    assert not (tmp_path/'unused').exists()


def test_preexisting_raw_result_prevents_launch(prepared,monkeypatch):
    attempt,directory,plan,_,_=prepared
    (directory/(plan['jobs'][0]['name']+'.json')).write_text('{}')
    monkeypatch.setattr(c.d,'run',lambda *args:pytest.fail('Must not launch'))
    with pytest.raises(AssertionError):c.run(attempt)
