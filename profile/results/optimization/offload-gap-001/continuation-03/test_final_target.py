"""CPU-only checks of final acceptance statistics and immutable fresh sampling."""
import copy
import math
import subprocess

import pytest
from scipy.stats import t
import final_target as f


def test_mean_below_tolerance_is_not_simultaneous_acceptance():
    result = f.assess([100.,100.,100.], [98.,99.,100.])
    assert result['residual_ms']['mean'] == -2
    # An ordinary one-sided95% bound would pass; family16 must not.
    assert -2 + t.ppf(.95,2)/math.sqrt(3) < 0
    assert result['residual_upper_simultaneous_95_ms'] > 0
    assert not result['within_target']


def test_signed_residual_and_pair_specific_piecewise_tolerance():
    result = f.assess([5.,20.,5.,20.], [5.05,20.15,5.05,20.15])
    assert result['tolerance_ms'] == [.1,.2,.1,.2]
    assert result['residual_ms']['mean'] == pytest.approx(-.05)
    assert result['within_target']
    assert f.assess([100.]*4,[99.]*4)['within_target']
    assert not f.assess([100.]*4,[102.]*4)['within_target']


@pytest.mark.parametrize('gpu,off', [([1.,1.],[1.,1.]),([1.,1.,1.],[1.,1.]),
                                     ([1.,float('nan'),1.],[1.,1.,1.]),([1.,1.,1.],[0.,1.,1.])])
def test_invalid_pair_evidence_rejected(gpu, off):
    with pytest.raises(AssertionError):
        f.assess(gpu,off)


def test_block_plan_requires_exact_all16_and_three_independent_pairs():
    assert len(f.block_counts(3)) == 16
    with pytest.raises(AssertionError):
        f.block_counts(2)
    rows=[dict(mode=m,batch=b,length=l,blocks=4) for m,b,l in f.d.WORKLOADS]
    with pytest.raises(AssertionError):
        f.block_counts(rows=rows[:-1])
    with pytest.raises(AssertionError):
        f.block_counts(rows=rows+[rows[0]])


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    attempt='A9999'
    directory=tmp_path/'fresh-target'
    sig=dict(attempt=attempt,commit='frozen-source',source_sha256='frozen-hash',root=str(f.d.root(attempt)))
    monkeypatch.setattr(f,'location',lambda a:directory)
    monkeypatch.setattr(f.d,'signature',lambda a:sig)
    monkeypatch.setattr(f.d,'record',lambda a:dict(status='accepted'))
    monkeypatch.setattr(f.d,'estimate',lambda *a:1.)
    f.prepare(attempt,f.block_counts(3))
    plan=f.d.read(directory/'plan.json');manifest=f.d.read(directory/'manifest.json')
    return attempt,directory,plan,manifest


def test_frozen_plan_is_complete_balanced_and_not_replaceable(prepared):
    attempt,_,plan,manifest=prepared
    assert len(f.validate(plan,manifest)) == 16
    assert len(manifest['jobs']) == 96
    with pytest.raises(AssertionError,match='replace or extend'):
        f.prepare(attempt,f.block_counts(4))
    changed=copy.deepcopy(manifest)
    changed['jobs'][0]['command'][-1]='reused-result.json'
    with pytest.raises(AssertionError,match='Changed planned job'):
        f.validate(plan,changed)


def test_even_refrozen_plan_cannot_reuse_output_names(prepared):
    _,_,plan,manifest=prepared
    plan['jobs'][2]['name']=plan['jobs'][0]['name']
    manifest['jobs'][2]['name']=plan['jobs'][0]['name']
    manifest['plan_sha256']=f.digest(plan)
    with pytest.raises(AssertionError,match='Repeated/invalid output name'):
        f.validate(plan,manifest)


def test_runner_rejects_preexisting_raw_evidence(prepared,monkeypatch):
    attempt,directory,plan,_=prepared
    (directory/(plan['jobs'][0]['name']+'.json')).write_text('{}')
    monkeypatch.setattr(f.d,'run',lambda *a:pytest.fail('Must not launch a benchmark'))
    with pytest.raises(AssertionError):
        f.run(attempt)


def test_plan_must_be_committed_byte_for_byte(tmp_path,monkeypatch):
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    directory=tmp_path/'target';directory.mkdir()
    path=directory/'plan.json';path.write_text('{"frozen":true}\n')
    monkeypatch.setattr(f.d,'D',tmp_path)
    with pytest.raises(subprocess.CalledProcessError):
        f.committed_plan(directory)
    subprocess.run(['git','add','target/plan.json'],cwd=tmp_path,check=True)
    subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.invalid',
                    'commit','-qm','freeze plan'],cwd=tmp_path,check=True)
    assert len(f.committed_plan(directory)) == 40
    path.write_text('{"frozen":false}\n')
    with pytest.raises(AssertionError,match='Commit the exact frozen plan'):
        f.committed_plan(directory)


def test_completed_audit_uses_process_pairs_and_requires_every_workload(prepared,monkeypatch):
    attempt,directory,plan,manifest=prepared
    manifest.update(status='completed',started_unix=plan['created_unix']+1,frozen_plan_commit='plan-commit')
    for j in manifest['jobs']:
        j.update(status='completed',returncode=0,started_unix=manifest['started_unix']+1,
                 finished_unix=manifest['started_unix']+2)
        (directory/(j['name']+'.json')).write_text('{}')
    f.d.save(directory/'manifest.json',manifest)
    monkeypatch.setattr(f,'committed_plan',lambda path:'plan-commit')
    def raw(job,path):
        shape=tuple(job[k] for k in ('mode','batch','length'))
        gpu=job['variant']=='ma_gpu'
        mean=100. if gpu else (97.+job['block'] if shape==f.d.WORKLOADS[0] else 99.)
        return ('same-environment',),dict(mean_ms=mean,samples_ms=[[mean]*10 for _ in range(3)],
                                         memory_after=dict(gpu_peak_allocated_bytes=200 if gpu else 100))
    monkeypatch.setattr(f.d,'audit_job',raw)
    result=f.audit(attempt)
    assert not result['all_workloads_within_target']
    assert not result['goal_accepted']
    assert sum(r['within_target'] for r in result['rows']) == 15
    assert all(r['independent_pairs']==3 for r in result['rows'])
    assert result['all_memory_savings_preserved']
    manifest['jobs'].pop()
    f.d.save(directory/'manifest.json',manifest)
    with pytest.raises(AssertionError):
        f.audit(attempt)
