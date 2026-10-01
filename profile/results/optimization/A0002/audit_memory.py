"""Audit placement and measured memory savings for every completed matrix point."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--partial',action='store_true')
    args=p.parse_args();m=json.loads((args.run/'manifest.json').read_text())
    if not args.partial:assert m['status'] in ('complete','complete_with_failures')
    records={};checks=[]
    for job in m['jobs']:
        if job['status'] not in ('completed','reused'):continue
        d=json.loads(Path(job['result']).read_text());r=d['results'][0];mem=r['memory_after']
        assert r['cpu_parameters']+r['gpu_parameters']==2836499968
        if job['variant']=='ma_offload':
            assert r['cpu_parameters']==1572864000 and r['gpu_parameters']==1263635968
            assert mem['cpu_table_bytes']==3145728000
            assert mem['offload_pinned_bytes']>0 and mem['offload_gpu_buffer_bytes']>0
            assert mem['offload_pinned_bytes']==sum(c['host_bytes'] for c in mem['offloader_capacities'])
            assert mem['offload_gpu_buffer_bytes']==sum(c['gpu_bytes'] for c in mem['offloader_capacities'])
        else:
            assert r['cpu_parameters']==0 and r['gpu_parameters']==2836499968
            assert mem['cpu_table_bytes']==0 and mem['offload_pinned_bytes']==0 and mem['offload_gpu_buffer_bytes']==0
        assert mem['kv_cache_storage_bytes']>0
        assert mem['gpu_peak_allocated_bytes']>=mem['gpu_allocated_bytes']
        assert mem['gpu_peak_reserved_bytes']>=mem['gpu_peak_allocated_bytes']
        assert mem['host_rss_bytes']>0
        key=(job['side'],job['mode'],job['batch'],job['length'])
        records[(*key,job['variant'])]=mem
        checks.append(dict(name=job['name'],result=job['result'],parameter_placement='verified',
                           kv_storage_bytes=mem['kv_cache_storage_bytes'],
                           host_rss_bytes=mem['host_rss_bytes'],raw_snapshot_bytes=mem['raw_table_snapshot_bytes'],
                           pinned_bytes=mem['offload_pinned_bytes'],gpu_buffers_bytes=mem['offload_gpu_buffer_bytes']))
    savings=[]
    for key in sorted({k[:4] for k in records}):
        off=records.get((*key,'ma_offload'));resident=records.get((*key,'ma_gpu'))
        if off is None or resident is None:continue
        saved=resident['gpu_peak_allocated_bytes']-off['gpu_peak_allocated_bytes']
        savings.append(dict(side=key[0],mode=key[1],batch=key[2],length=key[3],
                            gpu_peak_saved_bytes=saved,gpu_peak_saved_gib=saved/2**30,
                            offload_host_rss_bytes=off['host_rss_bytes'],resident_host_rss_bytes=resident['host_rss_bytes'],
                            offload_raw_snapshot_bytes=off['raw_table_snapshot_bytes'],
                            note='Host RSS includes model setup/allocator effects; not a claim of host-memory saving.'))
    observed=[r for r in savings if r['side']=='candidate']
    expected_points={tuple(j[k] for k in ('mode','batch','length')) for j in m['jobs'] if j['side']=='candidate'}
    payload=dict(expected_candidate_placement_pairs=len(expected_points),observed_candidate_placement_pairs=len(observed),controller_status=m['status'],partial=args.partial,checked_results=len(checks),checks=checks,savings=savings,
                 candidate_all_observed_savings_positive=all(r['gpu_peak_saved_bytes']>0 for r in observed) if observed else None,
                 scope='Per completed workload: parameter placement, CPU table, all cached pinned/GPU offloader capacities, backing KV storage, GPU allocated/reserved peaks, RSS and raw snapshots.',accepted=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+'\n')
    candidate=[r['gpu_peak_saved_gib'] for r in savings if r['side']=='candidate']
    print(json.dumps(dict(checked_results=len(checks),candidate_complete_placement_pairs=len(candidate),candidate_min_saving_gib=min(candidate) if candidate else None,candidate_max_saving_gib=max(candidate) if candidate else None)))


if __name__=='__main__':main()
