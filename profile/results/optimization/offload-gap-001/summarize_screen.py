"""Summarize isolated screening jobs; no inferential gain or acceptance claims.

Usage: python summarize_screen.py ATTEMPT/RUN --output ATTEMPT/RUN-summary
Preserves signed gaps. Sample dispersion is descriptive; independent paired
confirmation is needed for uncertainty on accepted gap reductions.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics


def summarize(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    points = {}
    failures = []
    for job in manifest['jobs']:
        path = directory / (job['name'] + '.json')
        if job['status'] != 'completed' or not path.exists():
            failures.append(dict(job=job, performance=None))
            continue
        payload = json.loads(path.read_text())
        assert payload['campaign_id'] == 'offload-gap-001'
        assert payload['protocol_version'] == 'offload_gap_v1'
        row = payload['results'][0]
        key = (job['mode'], job['batch'], job['length'])
        points.setdefault(key, {})[job['variant']] = (payload, row)
    rows = []
    for (mode, batch, length), placements in points.items():
        if not {'ma_offload', 'ma_gpu'} <= placements.keys():
            failures.append(dict(mode=mode,batch=batch,length=length,reason='unmatched placement',performance=None))
            continue
        off_payload, off = placements['ma_offload']
        gpu_payload, gpu = placements['ma_gpu']
        assert off_payload['measurement_plan_id'] == gpu_payload['measurement_plan_id']
        for field in ('torch', 'torch_cuda', 'flash_attn', 'gpu', 'python'):
            assert off_payload['env'][field] == gpu_payload['env'][field]
        samples = {name: [v for block in row['samples_ms'] for v in block]
                   for name, row in [('offload', off), ('gpu', gpu)]}
        a,b = off['mean_ms'],gpu['mean_ms']
        result = dict(mode=mode,batch=batch,length=length,
                      protocol_version=off_payload['protocol_version'],
                      measurement_plan_id=off_payload['measurement_plan_id'],
                      offload_ms=a,gpu_ms=b,absolute_gap_ms=a-b,relative_overhead=a/b-1,
                      offload_tokens_per_second=off['tokens_per_second'],gpu_tokens_per_second=gpu['tokens_per_second'],
                      uncertainty='screening: sample dispersion only; no independent-block confidence interval')
        for name, row in [('offload', off), ('gpu', gpu)]:
            result[name+'_sample_sd_ms'] = statistics.stdev(samples[name]) if len(samples[name]) > 1 else None
            result[name+'_sample_min_ms'] = min(samples[name])
            result[name+'_sample_max_ms'] = max(samples[name])
            for field in ('gpu_peak_allocated_bytes','gpu_peak_reserved_bytes','host_rss_bytes','offload_pinned_bytes','offload_gpu_buffer_bytes','kv_cache_storage_bytes'):
                result[name+'_'+field] = row['memory_after'][field]
        rows.append(result)
    return dict(campaign_id='offload-gap-001',source_manifest=str(directory/'manifest.json'),
                candidate_sha=manifest['candidate_sha'],status=manifest['status'],
                rows=rows,failed_or_unmatched=failures,
                note='Provisional screening, not formal confirmation. No accepted gain claims.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    report=summarize(args.run)
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    with args.output.with_suffix('.csv').open('w') as f:
        if report['rows']:
            writer=csv.DictWriter(f,fieldnames=list(report['rows'][0]))
            writer.writeheader();writer.writerows(report['rows'])
    for r in report['rows']:
        print(f"{r['mode']} b{r['batch']} l{r['length']}: offload={r['offload_ms']:.3f} gpu={r['gpu_ms']:.3f} gap={r['absolute_gap_ms']:+.3f} ms overhead={r['relative_overhead']:+.1%}")

if __name__=='__main__':
    main()
