import importlib.util,json
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('supplemental_study',Path(__file__).with_name('study.py'));s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)

def test_scope_covers_all_nonretained_sources_and_appropriate_branches():
 assert len(s.ATTEMPTS)==18 and len(set(s.ATTEMPTS))==18
 assert s.shapes('A0002')==s.DECODES+s.GEN
 assert ('prefill',8,512) not in s.shapes('A0015')
 assert ('generation',1,2048) not in s.shapes('A0013')
 assert ('decode',8,8192) in s.shapes('A0014')
 assert all(('prefill',8,8192) in s.shapes(a) for a in s.ATTEMPTS if a not in ('A0002','A0005','A0013'))

def test_formal_plan_uses_fresh_alternating_blocks_and_unchanged_generation_scope(tmp_path):
 p=s.plan('A0020',[('prefill',8,8192),('generation',1,2048)],'A0016',True,tmp_path)
 assert len(p['jobs'])==24 and s.validate_balanced_order(p['jobs'])
 for j in p['jobs']:
  cmd=j['command'];assert cmd[cmd.index('--rounds')+1]=='3'
  assert cmd[cmd.index('--warmup')+1]==('2' if j['mode']=='generation' else '10')
  assert not j.get('reused_results')

@pytest.mark.parametrize('field,value',[('candidate_sha','bad'),('source_sha256','bad'),('batch',16)])
def test_raw_audit_rejects_wrong_provenance_or_shape(field,value):
 d=s.C/'A0020/R01';p=s.read(d/'manifest.json');j=p['jobs'][0].copy()
 j.update(attempt='A0020',candidate_sha=s.record('A0020')['candidate_sha'],source_sha256=s.source_hash(s.root('A0020')))
 s.audit_job(j,d/(j['name']+'.json'))
 j[field]=value
 with pytest.raises(AssertionError):s.audit_job(j,d/(j['name']+'.json'))
