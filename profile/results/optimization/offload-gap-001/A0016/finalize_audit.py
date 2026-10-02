"""Audit A0016's fixed evidence before publishing its retained-step verdict."""
import json
from pathlib import Path
import subprocess
import sys

here=Path(__file__).resolve().parent
root=here.parent
repo=root.parents[3]
sys.path.insert(0,str(root))
from audit_final import audit_retained_evidence,require_eligible_gain
from audit_continuation import validate_offload_memory
from analyze_paired import analyze
from analyze_regression import analyze as regression
from select_regression_checks import select
from run_paired import source_hash

def read(p):return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
records={p.parent.name:read(p) for p in root.glob('A[0-9][0-9][0-9][0-9]/record.json')}
r=records['A0016'].copy()
assert read(here/'secondary-regressions-controller.json')['status']=='completed_needs_analysis_and_audit'
for manifest in ('R02-parent-confirmation/manifest.json','R03-baseline-confirmation/manifest.json'):
 result=analyze(here/manifest)
 assert result['confirmation_promising']
 if manifest.startswith('R02'):require_eligible_gain(result,r)
for prefix,stage,run in [('full','full_validation','R07-regression'),('generation','generation_validation','R08-regression')]:
 expected=select(read(here/(prefix+'-summary.json')),stage)
 stored=read(here/(prefix+'-regression-selection.json'))
 assert all(stored[k]==v for k,v in expected.items())
 result=regression(here/run/'manifest.json')
 prior=read(here/(prefix+'-regression-analysis.json'))
 assert result['rows']==prior['rows'] and result['regression_gate_passed']
 actual={f"{row['mode']}:{row['batch']}:{row['length']}" for row in result['rows']}
 assert actual==set(expected['selected'])
 jobs=read(here/run/'manifest.json')['jobs']
 assert len(jobs)==12*len(actual)
 assert sum(j['status']=='reused' for j in jobs)==4*len(actual)
r.update(status='accepted',workflow_stage='final_verdict',accepted_step=2,
         validation_matrix_manifest='full-evidence-manifest.json',generation_manifest='generation-evidence-manifest.json',
         full_summary='full-audited-summary.json',generation_summary='generation-audited-summary.json',
         regression_analyses=['full-regression-analysis.json','generation-regression-analysis.json'],
         source_audit='source-correctness-audit.json')
assert source_hash(repo)==source_hash(Path(r['frozen_checkout']))
# Recompute raw correctness comparisons, source/environment/config/sample and GPU-saving checks.
audit=audit_retained_evidence(root,r,records)
for path in audit['source_results']:
 p=read(Path(path))
 if p['source']['git_commit']['stdout'].strip()==r['candidate_sha'] and p['results'][0]['variant']=='ma_offload':
  validate_offload_memory('A0016',p['results'][0]['memory_after'])
source=read(here/'source-review.json')
patch=subprocess.check_output(['git','diff',r['parent_source_sha'],r['candidate_sha'],'--','fla'],cwd=repo,text=True)
assert patch==source['parent_source_patch']
benchmark_files=[str(p.relative_to(repo)) for p in (repo/'profile').glob('*.py')]
assert not subprocess.check_output(['git','diff',r['parent_source_sha'],r['candidate_sha'],'--',*benchmark_files],cwd=repo)
assert '31 passed' in (here/'correctness-tests.txt').read_text()
assert 'RESULT: PASS (worst max diff 0.000000' in (here/'standalone-correctness.txt').read_text()
source.update(accepted=True,acceptance=True,remaining_gate=None,
              regression_evidence=r['regression_analyses'],raw_evidence_audit='retained-evidence-audit.json')
audit.update(campaign_id='offload-gap-001',status='retention_evidence_checks_passed',checked_raw_results=len(audit['source_results']),
             limitations='Manual source review remains recorded in source-correctness-audit.json; three-block uncertainty and adverse directions retained in record.json.')
r.update(not_run={},validation_progress='All fixed primary, exactness, full/generation, bounded regression and source/memory gates completed.',
 decision='Accepted: changed b1 prefill independently reduces offload latency and gap against verified A0001; cumulative A0000 confirmation also passes. Full exactness, all84 matrix/12 generation points, fixed secondary regression investigations, source/environment and memory gates pass. Retain all adverse observations and uncertainty; no broad decode/generation gain claim.',
 limitations=[
 'Incremental b1 prefill offload reduction0.430878 ms CI[0.153321,0.708434], gap0.451613 CI[0.304395,0.598832]. Cumulative A0000 reduction14.021136 ms includes A0001 and is not this step alone.',
 'Parent-relative b8 decode slower in all3 pairs, mean reduction-0.038070 ms CI[-0.100821,0.024682]; unresolved.',
 'A0000-relative b16 decode slower in all3 secondary pairs: reduction-0.033485 ms CI[-0.082574,0.015604]. b8/4096 prefill slower in all3: reduction-0.937146 ms CI[-2.972030,1.097739]. No resolved regression under the predeclared rule; adverse directions retained.',
 'b8/8192 prefill reduction-4.475341 ms CI[-15.449950,6.499268] remains uncertain. No additional extension.',
 'b8 generation reduction0.438272 ms CI[-4.831173,5.707717]; no generation gain claim. b1 generation has only the original single pair.',
 'Three paired blocks yield wide intervals; n=3 symmetry assumption unverified. Seeded random weights test equivalence/performance, not language quality.'
 ])
save(here/'retained-evidence-audit.json',audit)
save(here/'source-correctness-audit.json',source)
save(here/'record.json',r)
print('A0016 accepted after complete audit:',len(audit['source_results']),'raw results;',len(audit['correctness_reports']),'full-model comparisons')
