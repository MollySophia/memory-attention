"""History must track one accepted model, never independent per-metric minima."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('history_plot',ROOT/'plot_history.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def entry(number,status,step,prefill=None,decode=None):
    rows={}
    if prefill is not None:
        for mode,batch in module.WORKLOADS:
            rows[(mode,batch)]=dict(measurement_plan_id='unit_test_fixture',offload_ms=prefill if mode=='prefill' else decode,offload_sample_sd_ms=.1)
    return dict(directory=Path('/unused-test-fixture'),record=dict(attempt_id=f'A{number:04d}',campaign_id='offload-gap-001',label='unit test fixture',status=status,accepted_step=step),rows=rows)


def test_incumbent_is_whole_model_and_failure_has_no_numeric_point(tmp_path):
    entries=[entry(0,'accepted',0,10,20),entry(1,'accepted',1,8,21),
             entry(2,'rejected',None,7,19),entry(3,'correctness_failed',None)]
    records=module.history(entries,tmp_path)
    for row in records:
        if row['attempt'] in ('A0002','A0003'):
            assert row['incumbent_attempt']=='A0001'
            assert row['incumbent_ms']==(8 if row['mode']=='prefill' else 21)
        if row['attempt']=='A0003':assert row['measured_ms'] is None
    for suffix in ('pdf','svg','png'):
        assert (tmp_path/f'attempt-history-unit_test_fixture.{suffix}').exists()


def test_accepted_step_requires_explicit_eligible_confirmation(tmp_path):
    with pytest.raises(ValueError,match='explicitly identify'):
        module.accepted_steps([entry(0,'accepted',0,10,20),entry(1,'accepted',1,8,21)],tmp_path)


def test_valid_offload_point_survives_failed_resident_partner(tmp_path):
    import json
    directory=tmp_path/'A0001';run=directory/'R01';run.mkdir(parents=True)
    (directory/'record.json').write_text(json.dumps(dict(campaign_id='offload-gap-001',attempt_id='A0001',label='test fixture',status='benchmark_failed',accepted_step=None)))
    jobs=[dict(name='offload',mode='prefill',variant='ma_offload',batch=1,length=2048,status='completed'),dict(name='resident',mode='prefill',variant='ma_gpu',batch=1,length=2048,status='oom')]
    (run/'manifest.json').write_text(json.dumps(dict(jobs=jobs)))
    (run/'offload.json').write_text(json.dumps(dict(protocol_version='offload_gap_v1',stage='screening',measurement_plan_id='unit_test_fixture',results=[dict(mean_ms=10,samples_ms=[[9,10,11]])])))
    loaded=module.attempts(tmp_path)
    assert loaded[0]['rows'][('prefill',1)]['offload_ms']==10


@pytest.mark.parametrize('reverse', [False, True])
def test_paired_screen_history_uses_candidate_not_parent_or_generation(tmp_path, reverse):
    import json
    directory=tmp_path/'A0021';run=directory/'R01-complete-screen';run.mkdir(parents=True)
    (directory/'record.json').write_text(json.dumps(dict(campaign_id='offload-gap-001',
        attempt_id='A0021',label='test fixture',status='rejected',accepted_step=None,
        screen_manifest='R01-complete-screen/manifest.json',screen_summary='R01-complete-screen/summary.json')))
    # Modern summaries have paired statistics rather than a single offload_ms.
    (run/'summary.json').write_text(json.dumps(dict(rows=[dict(mode='prefill',batch=1,length=2048,statistics={})])))
    jobs=[]
    for impl,attempt,mean in [('baseline','A0016',20),('candidate','A0021',10)]:
        jobs.append(dict(name=impl,attempt=attempt,implementation=impl,mode='prefill',
                         variant='ma_offload',batch=1,length=2048,status='completed'))
        (run/(impl+'.json')).write_text(json.dumps(dict(protocol_version='offload_gap_v1',
            stage='screening',measurement_plan_id='screen_v1_w3_n5_r1',
            results=[dict(mean_ms=mean,samples_ms=[[mean-1,mean,mean+1]])])))
    jobs.append(dict(name='generation',attempt='A0021',implementation='candidate',mode='generation',
                     variant='ma_offload',batch=1,length=2048,status='completed'))
    # No generation file: a primary prefill/decode history must not read it.
    (run/'manifest.json').write_text(json.dumps(dict(jobs=jobs[::-1] if reverse else jobs)))
    loaded=module.attempts(tmp_path)
    assert list(loaded[0]['rows'])==[('prefill',1)]
    assert loaded[0]['rows'][('prefill',1)]['offload_ms']==10
