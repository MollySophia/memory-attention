"""Audit predeclared reuse-one/add-two regression blocks without dropping rows."""
import argparse
import json
from pathlib import Path
from analyze_paired import interval
from run_paired import validate_balanced_order


def analyze(path):
    manifest = json.loads(path.read_text())
    assert manifest['status'] == 'completed'
    assert manifest['measurement_plan_id'] == 'regression_reuse1_add2_balanced_v1'
    validate_balanced_order(manifest['jobs'])
    data = {}
    for job in manifest['jobs']:
        assert job['status'] in ('completed', 'reused')
        source = Path(job['source_result']) if job['status'] == 'reused' else path.parent / (job['name'] + '.json')
        payload = json.loads(source.read_text())
        assert payload['status'] == 'completed'
        assert payload['protocol_version'] == 'offload_gap_v1'
        assert payload['source']['git_commit']['stdout'].strip() == job['candidate_sha']
        expected = 'generation_v1_w2_n5_r3' if job['mode'] == 'generation' else 'formal_v1_w10_n10_r3'
        assert payload['measurement_plan_id'] == expected
        key = tuple(job[k] for k in ('mode', 'batch', 'length', 'block', 'implementation', 'variant'))
        assert key not in data
        data[key] = payload['results'][0]
    report = []
    for mode, batch, length in sorted({k[:3] for k in data}):
        values = {k: [] for k in ('offload_reduction_ms', 'gpu_reduction_ms', 'gap_reduction_ms')}
        savings = []
        for block in (1, 2, 3):
            rows = [data[(mode, batch, length, block, impl, variant)] for impl, variant in
                    [('baseline', 'ma_offload'), ('baseline', 'ma_gpu'), ('candidate', 'ma_offload'), ('candidate', 'ma_gpu')]]
            bo, bg, co, cg = [r['mean_ms'] for r in rows]
            for name, value in zip(values, (bo-co, bg-cg, (bo-bg)-(co-cg))):
                values[name].append(value)
            savings.append(rows[3]['memory_after']['gpu_peak_allocated_bytes'] - rows[2]['memory_after']['gpu_peak_allocated_bytes'])
        stats = {k: interval(v) for k, v in values.items()}
        report.append(dict(mode=mode, batch=batch, length=length, statistics=stats,
                           resolved_offload_regression=stats['offload_reduction_ms']['upper_95'] < 0,
                           resolved_resident_slowdown=stats['gpu_reduction_ms']['upper_95'] < 0,
                           memory_savings_preserved=min(savings) > 0, gpu_savings_bytes=savings))
    return dict(campaign_id='offload-gap-001', source_manifest=str(path), rows=report,
                regression_gate_passed=not any(r['resolved_offload_regression'] or r['resolved_resident_slowdown'] or not r['memory_savings_preserved'] for r in report),
                uncertainty_note='95% Student-t intervals over exactly three independent paired blocks, including original pair. n=3; symmetry assumption unverified. No samples discarded; not a new improvement claim.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.manifest)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
