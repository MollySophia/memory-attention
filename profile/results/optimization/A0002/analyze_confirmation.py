"""Audit independent runs and apply the predeclared paired-log analysis."""
import csv
import json
import math
from pathlib import Path
import statistics
import sys


def main():
    run = Path(sys.argv[1]).resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    records = {}
    pids = set()
    hashes = {'baseline': set(), 'candidate': set()}
    for job in manifest['jobs']:
        assert job['status'] == 'complete' and job['returncode'] == 0
        assert job['pid'] not in pids
        pids.add(job['pid'])
        assert Path(job['verified_cache_module']) == Path(job['pythonpath']) / 'fla/models/utils.py'
        data = json.loads((run / (job['name'] + '.json')).read_text())
        assert data['status'] == 'completed' and data['protocol_version'] == 'paper_v1'
        assert data['source']['git_commit']['stdout'].strip() == job['expected_commit']
        assert not data['source']['source_patch']['stdout']
        hashes[job['side']].add(data['source']['source_sha256'])
        config = data['config']
        assert (config['warmup'], config['repeats'], config['rounds']) == (10, 10, 3)
        assert (config['batch_size'], config['seq_len'], config['context_len']) == (8, 2048, 2048)
        assert config['prefill_workload'] == 'inference' and config['logits_to_keep'] == 1
        row = data['results'][0]
        assert row['variant'] == job['variant'] and row['mode'] == job['mode']
        assert len(row['samples_ms']) == 3
        means = []
        for samples in row['samples_ms']:
            assert len(samples) == 10 and all(math.isfinite(x) and x > 0 for x in samples)
            means.append(statistics.mean(samples))
        assert all(abs(x-y) < 1e-8 for x, y in zip(means, row['round_ms']))
        assert abs(statistics.median(means) - row['median_ms']) < 1e-8
        records[(job['variant'], job['mode'], job['pair'], job['side'])] = data
    assert all(len(values) == 1 for values in hashes.values())
    summaries = []
    for variant in ('ma_offload', 'ma_gpu'):
        for mode in ('prefill', 'decode'):
            paired = []
            for pair in range(1, 4):
                b = records[(variant, mode, pair, 'baseline')]
                c = records[(variant, mode, pair, 'candidate')]
                assert b['model_config'] == c['model_config']
                for key in ('torch_threads', 'torch_interop_threads', 'thread_environment', 'cpu_affinity', 'platform'):
                    assert b['environment_before'][key] == c['environment_before'][key], key
                assert {k:v for k,v in b['env'].items() if k!='git_commit'} == {k:v for k,v in c['env'].items() if k!='git_commit'}
                for k, v in b['config'].items():
                    if k != 'json':
                        assert c['config'][k] == v, (k, v, c['config'].get(k))
                br, cr = b['results'][0], c['results'][0]
                for key in ('cpu_parameters', 'gpu_parameters'):
                    assert br[key] == cr[key], key
                if variant == 'ma_offload':
                    assert cr['cpu_parameters'] == 1572864000
                    assert cr['memory_after']['cpu_table_bytes'] == 3145728000
                paired.append(dict(pair=pair, baseline_ms=br['median_ms'], candidate_ms=cr['median_ms'],
                                   speedup=br['median_ms']/cr['median_ms'],
                                   baseline_peak_gpu_gib=br['memory_after']['gpu_peak_allocated_bytes']/2**30,
                                   candidate_peak_gpu_gib=cr['memory_after']['gpu_peak_allocated_bytes']/2**30,
                                   baseline_host_rss_gib=br['memory_after']['host_rss_bytes']/2**30,
                                   candidate_host_rss_gib=cr['memory_after']['host_rss_bytes']/2**30))
            logs = [math.log(x['speedup']) for x in paired]
            center = statistics.mean(logs)
            half = 4.302652729911275 * statistics.stdev(logs) / math.sqrt(3)
            low, high = math.exp(center-half), math.exp(center+half)
            verdict = 'repeatable_improvement' if low > 1 else 'resolved_regression' if high < 1 else 'within_noise'
            summaries.append(dict(variant=variant, mode=mode, pairs=paired,
                                  geometric_mean_speedup=math.exp(center), paired_speedup_ci95=[low, high],
                                  verdict=verdict))
    by = {(r['variant'], r['mode']):r for r in summaries}
    memory = []
    for mode in ('prefill', 'decode'):
        for off, gpu in zip(by[('ma_offload',mode)]['pairs'], by[('ma_gpu',mode)]['pairs']):
            memory.append(dict(mode=mode, pair=off['pair'],
                               baseline_offload_saving_gib=gpu['baseline_peak_gpu_gib']-off['baseline_peak_gpu_gib'],
                               candidate_offload_saving_gib=gpu['candidate_peak_gpu_gib']-off['candidate_peak_gpu_gib'],
                               candidate_offload_peak_increase_mib=(off['candidate_peak_gpu_gib']-off['baseline_peak_gpu_gib'])*1024))
    # Preserve the actual table placement (asserted above) and a measured GPU
    # peak saving. Report its exact size/change without a post-hoc tolerance.
    saving_preserved = all(x['candidate_offload_saving_gib'] > 0 for x in memory)
    primary = {r['mode']:r for r in summaries if r['variant']=='ma_offload'}
    small = json.loads((run.parent/'small-batch-summary.json').read_text())
    assert small['source_hashes'] == {k:sorted(v) for k,v in hashes.items()}, 'Different source from batch1 gate'
    nominee = not small['regression_confirmed'] and saving_preserved and primary['decode']['verdict']=='repeatable_improvement' and primary['prefill']['verdict']!='resolved_regression'
    output = dict(stage='confirmation', run=run.name, job_count=len(pids), sample_count=30*len(pids),
                  audit='All actual import paths, expected commits, source patches/hashes, paired model/config/environment, 720 finite raw samples, and recomputed round/run estimators passed.',
                  estimator='median of round means per process; geometric mean of paired baseline/candidate ratios',
                  uncertainty='95% Student-t interval on three independent paired log ratios, df=2. Small-n interval assumes approximately normal paired log ratios; within-process rounds are not independent process pairs.',
                  source_hashes={k:sorted(v) for k,v in hashes.items()}, rows=summaries,
                  environment_note='Raw before/after GPU clocks, temperature, utilization and process listings retained per job. A separate trm-mcp Python process holds 654 MiB; snapshots are not continuous utilization monitoring.',
                  memory_comparison=memory, offload_memory_saving_preserved=saving_preserved,
                  elapsed_minutes=(manifest['ended']-manifest['started'])/60,
                  nominated_for_full_validation=nominee, accepted=False,
                  next_action='Full matrix and growing generation validation' if nominee else 'Investigate resolved primary regression or record within_noise; no acceptance.')
    (run.parent/'confirmation-summary.json').write_text(json.dumps(output,indent=2)+'\n')
    with (run.parent/'confirmation-pairs.csv').open('w') as f:
        fields=['variant','mode',*summaries[0]['pairs'][0].keys()]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for item in summaries:
            for pair in item['pairs']:
                writer.writerow(dict(variant=item['variant'],mode=item['mode'],**pair))
    print(json.dumps(output,indent=2))


if __name__ == '__main__':
    main()
