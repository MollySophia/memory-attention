"""Compare each screen with frozen baseline and accepted parent, without claims."""
import argparse
import csv
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--candidate', type=Path, required=True)
p.add_argument('--references', type=Path, nargs='+', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
candidate = json.loads(a.candidate.read_text())
rows = []
for reference_path in a.references:
    reference = json.loads(reference_path.read_text())
    indexed = {(r['mode'], r['batch'], r['length']): r for r in reference['rows']}
    for c in candidate['rows']:
        key = (c['mode'], c['batch'], c['length'])
        b = indexed.get(key)
        if b is None:
            rows.append(dict(reference=str(reference_path), mode=key[0], batch=key[1], length=key[2],
                             status='unmatched', offload_speedup=None))
            continue
        assert b['protocol_version'] == c['protocol_version']
        assert b['measurement_plan_id'] == c['measurement_plan_id'] == 'screen_v1_w3_n5_r1'
        row = dict(reference=str(reference_path), mode=key[0], batch=key[1], length=key[2], status='screening_only',
                   offload_speedup=b['offload_ms']/c['offload_ms'],
                   offload_reduction_ms=b['offload_ms']-c['offload_ms'],
                   gap_reduction_ms=b['absolute_gap_ms']-c['absolute_gap_ms'],
                   candidate_gpu_memory_saving_bytes=c['gpu_gpu_peak_allocated_bytes']-c['offload_gpu_peak_allocated_bytes'])
        for name in ('offload_ms', 'gpu_ms', 'absolute_gap_ms', 'relative_overhead', 'offload_sample_sd_ms', 'gpu_sample_sd_ms'):
            row['reference_'+name] = b[name]
            row['candidate_'+name] = c[name]
        rows.append(row)
report = dict(campaign_id='offload-gap-001', candidate=str(a.candidate), rows=rows,
              failed_or_unmatched=candidate['failed_or_unmatched'],
              note='Short screens are descriptive only. No accepted gain or confidence interval. Incremental attribution uses accepted parent, cumulative comparison uses A0000.')
a.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
with a.output.with_suffix('.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
    writer.writeheader(); writer.writerows(rows)
for r in rows:
    print(Path(r['reference']).parent.name, r['mode'], r['batch'],
          'offload reduction', r.get('offload_reduction_ms'), 'gap reduction', r.get('gap_reduction_ms'))
