"""Run after summarize.py; correct a stale pre-campaign note from raw snapshots.

The frozen harness and manifest are retained unchanged. This postprocessor
records the evidence for the correction and refreshes the artifact hash audit.
"""
import hashlib
import json
import re
from pathlib import Path

D = Path(__file__).resolve().parent
manifest = json.loads((D / 'manifest.json').read_text())
assert manifest['status'] == 'completed'
snapshots = []
for job in manifest['jobs']:
    path = D / (job['name'] + '.json')
    if job['kind'] != 'performance' or not path.exists():
        continue
    raw = json.loads(path.read_text())
    for point in ('environment_before', 'environment_after'):
        snapshot = raw[point]
        text = snapshot['nvidia_smi']['stdout']
        match = re.findall(r'3625831.*?(\d+)MiB', text)
        assert len(match) == 1, (job['name'], point, match)
        snapshots.append(dict(job=job['name'], point=point,
                              external_process_mib=int(match[0]),
                              gpu_telemetry=snapshot['gpu_telemetry']['stdout']))
values = sorted({s['external_process_mib'] for s in snapshots})
assert values == [654], values
note = ('Environment correction: the 6994 MiB external allocation noted in the frozen '
        'manifest and baseline register was observed during earlier diagnostics. All '
        f'{len(snapshots)} recorded before/after performance snapshots in R0002 show '
        '654 MiB for the unrelated process (PID3625831), matching the original baseline '
        'background allocation. No unrelated process was stopped. These are snapshots, '
        'not continuous monitoring. All timing controls were freshly measured. '
        'Full-KV batch64 OOMs and skipped identical repeats remain in the matrix.')
(D / 'environment-audit.json').write_text(json.dumps(dict(
    status='passed', correction=note, external_process_mib_values=values,
    snapshots=snapshots), indent=2) + '\n')
report = D / 'REPORT.md'
old = ('An unrelated GPU process holds about6994 MiB, versus about654 MiB during the '
       'original baseline collection. It was not stopped. We remeasured all controls; '
       'do not subtract old timings as though they were paired, or interpret a new OOM '
       'as a capacity regression without accounting for external allocation. Identical '
       'OOM repeats are skipped, never recorded as zero.')
content = report.read_text()
assert old in content or note in content
report.write_text(content.replace(old, note + ' See environment-audit.json.'))
audit_path = D / 'integrity-audit.json'
audit = json.loads(audit_path.read_text())
audit['environment_note_corrected_from_raw_snapshots'] = True
audit['sha256'] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                   for f in D.iterdir()
                   if f.suffix in ('.py', '.json', '.txt', '.md', '.csv')
                   and f.name != 'integrity-audit.json'}
audit_path.write_text(json.dumps(audit, indent=2) + '\n')
print(note)
