"""Accepted intermediate implementations must keep complete evidence."""
import json
from pathlib import Path
import pytest
import audit_final

ROOT = Path(__file__).resolve().parent


def records():
    return {path.parent.name:json.loads(path.read_text()) for path in ROOT.glob('A????/record.json')}


def test_existing_accepted_source_all_raw_stages_validate():
    data=records()
    result=audit_final.audit_retained_evidence(ROOT,data['A0001'],data)
    assert len(result['source_results'])==160
    assert len(result['correctness_reports'])==4


@pytest.mark.parametrize('field', ['full_summary','generation_summary','correctness_directory'])
def test_missing_intermediate_stage_fails(field):
    data=records();data['A0001'][field]='missing-required-evidence'
    with pytest.raises(FileNotFoundError):
        audit_final.audit_retained_evidence(ROOT,data['A0001'],data)


def test_incremental_gain_must_be_in_declared_target():
    record=dict(continuation_id='continuation-02',eligible_gain_workloads=[dict(mode='prefill',batch=1)])
    result=dict(rows=[dict(mode='prefill',batch=1,repeatable_offload_and_gap_reduction=False),
                      dict(mode='prefill',batch=8,repeatable_offload_and_gap_reduction=True)])
    with pytest.raises(AssertionError,match='predeclared changed primary'):
        audit_final.require_eligible_gain(result,record)
    result['rows'][0]['repeatable_offload_and_gap_reduction']=True
    audit_final.require_eligible_gain(result,record)
