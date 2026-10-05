"""Audit completed all-workload paired evidence; no retention or target verdict."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / 'continuation-03'))
import driver as d
import validation

rows = []
envs = set()
raw_files = []
all_cells = []
for stage, count in [('R02-parent-confirmation', 240), ('R03-remaining-confirmation', 144)]:
    folder = BASE / stage
    manifest = d.read(folder / 'manifest.json')
    assert manifest['status'] == 'completed'
    assert d.planned_blocks(manifest) == 6
    assert len(manifest['jobs']) == count
    assert d.validate_balanced_order(manifest['jobs'], 6)
    for sig in manifest['source_signatures'].values():
        assert d.signature(sig['attempt']) == sig
    for job in manifest['jobs']:
        assert job['status'] == 'completed' and job['returncode'] == 0
        path = folder / (job['name'] + '.json')
        env, row = d.audit_job(job, path)
        envs.add(env)
        assert math.isclose(row['mean_ms'], statistics.mean(map(statistics.mean, row['samples_ms'])), rel_tol=1e-12, abs_tol=1e-12)
        all_cells.append(tuple(job[k] for k in ('mode', 'batch', 'length', 'block', 'implementation', 'variant')))
        raw_files.append(dict(path=str(path.relative_to(BASE)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
assert len(envs) == 1
expected = {(*w, b, i, v) for w in d.WORKLOADS for b in range(1, 7) for i in ('baseline', 'candidate') for v in ('ma_offload', 'ma_gpu')}
assert len(all_cells) == len(set(all_cells)) == 384 and set(all_cells) == expected
old = d.read(BASE / 'full-parent-analysis.json')
new = validation.paired_audit('A0030')
assert json.loads(json.dumps(new)) == old
assert not new['resolved_regressions']
assert all(r['memory_savings_preserved'] for r in new['rows'])
result = dict(status='passed', attempt='A0030', workloads=16, independent_blocks=6, checked_raw_results=384, complete_unique_coverage=True, balanced_order=True, one_environment=True, source_signatures_verified=True, raw_means_recomputed=True, analysis_recomputed_identical=True, resolved_regressions=[], memory_savings_preserved=True, raw_files=raw_files, accepted=False, note='Full parent-paired validation only. R02 fixed-family gain decisions remain authoritative. Folding and full-model correctness remain required; these are not independent final-target data.')
d.save(BASE / 'remaining-audit.json', result)
print(json.dumps({k: v for k, v in result.items() if k != 'raw_files'}, indent=2))
