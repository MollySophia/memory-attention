"""Prevent scope/protocol/source mismatches from entering reused validation."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('paired_driver',ROOT/'run_paired.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


def evidence():
    directory=ROOT/'A0001/R02-confirmation'
    manifest=json.loads((directory/'manifest.json').read_text())
    job=manifest['jobs'][0]
    payload=json.loads((directory/(job['name']+'.json')).read_text())
    commit=manifest['source_commits']['baseline']
    digest=driver.source_hash(Path('/home/molly/workspace-memory-attn/offload-gap-001-A0000'))
    env={k:payload['env'][k] for k in ('torch','flash_attn','python','gpu','cuda_visible_devices')}
    return payload,job,commit,digest,env


def test_valid_evidence():
    assert driver.validate_reuse_payload(*evidence())


@pytest.mark.parametrize('path,value',[
    (('status',),'oom'),(('protocol_version',),'wrong'),
    (('measurement_plan_id',),'screen_v1_w3_n5_r1'),
    (('config','logits_to_keep'),0),(('config','seed'),4321),
    (('config','hidden_size'),1024),(('model_config','qk_norm'),True),
    (('config','batch_size'),8),(('source','source_sha256'),'wrong'),
    (('env','torch'),'wrong'),(('results',0,'output_scope'),'full_logits_no_cache'),
    (('results',0,'samples_ms'),[[1.]*10]*2),
])
def test_incompatible_evidence_rejected(path,value):
    payload,job,commit,digest,env=evidence()
    target=payload
    for key in path[:-1]:target=target[key]
    target[path[-1]]=value
    with pytest.raises(AssertionError):driver.validate_reuse_payload(payload,job,commit,digest,env)


def test_original_confirmation_order_is_rejected():
    manifest=json.loads((ROOT/'A0001/R02-confirmation/manifest.json').read_text())
    with pytest.raises(AssertionError):driver.validate_balanced_order(manifest['jobs'])


def test_new_plan_alternates_each_workload_and_placement(tmp_path):
    import subprocess,sys
    destination=tmp_path/'plan'
    subprocess.run([sys.executable,str(ROOT/'run_paired.py'),
        '--baseline-root','/home/molly/workspace-memory-attn/offload-gap-001-A0000',
        '--candidate-root','/home/molly/workspace-memory-attn/offload-gap-001-A0001',
        '--screen-manifest',str(ROOT/'A0000/R01/manifest.json'),'--stage','confirmation',
        '--plan-only','--output',str(destination)],check=True,capture_output=True,text=True)
    manifest=json.loads((destination/'manifest.json').read_text())
    assert len(manifest['jobs'])==48 and manifest['pairing_plan_id']=='balanced_order_v2'
    assert driver.validate_balanced_order(manifest['jobs'])
    for mode in ('prefill','decode'):
        for batch in (1,8):
            for variant in ('ma_gpu','ma_offload'):
                firsts=[next(j['implementation'] for j in manifest['jobs'] if j['block']==block and j['mode']==mode and j['batch']==batch and j['variant']==variant) for block in (1,2,3)]
                assert firsts[0]!=firsts[1] and firsts[1]!=firsts[2]
