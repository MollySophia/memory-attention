"""Do not satisfy an additional-attempt request using the old tranche."""
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import audit_final


def fixture(root,count):
    for n in range(count+1):
        d=root/f'A{n:04d}';d.mkdir()
        record=dict(campaign_id='offload-gap-001',attempt_id=f'A{n:04d}',parent_attempt_id='A0000',
                    candidate_sha='0'*40,status='within_noise',workflow_stage='final_verdict',
                    accepted_step=None,hypothesis='test hypothesis',profile_evidence='test evidence',decision='test verdict')
        if n>5:record['continuation_id']='continuation-01'
        (d/'record.json').write_text(json.dumps(record))
    d=root/'continuation-01';d.mkdir()
    (d/'manifest.json').write_text(json.dumps(dict(campaign_id='offload-gap-001',start_after_attempt='A0005',first_new_attempt='A0006',minimum_additional_attempts=5,maximum_additional_attempts=10)))


def test_old_tranche_cannot_satisfy_additional_count(tmp_path):
    fixture(tmp_path,9)
    with pytest.raises(AssertionError,match='Additional attempt count'):
        audit_final.audit(tmp_path,tmp_path,tmp_path,'continuation-01')
    with pytest.raises(AssertionError,match='Explicit --continuation'):
        audit_final.audit(tmp_path,tmp_path,tmp_path)


def test_registration_count_does_not_replace_final_verdicts(tmp_path,monkeypatch):
    fixture(tmp_path,10)
    p=tmp_path/'A0010/record.json';r=json.loads(p.read_text());r['workflow_stage']='parent_confirmation';p.write_text(json.dumps(r))
    monkeypatch.setattr(audit_final.subprocess,'check_output',lambda *a,**kw:'commit\n')
    with pytest.raises(AssertionError,match='A0010'):
        audit_final.audit(tmp_path,tmp_path,tmp_path,'continuation-01')
