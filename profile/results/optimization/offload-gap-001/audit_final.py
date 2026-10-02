"""Check final campaign evidence. Passing does not replace visual/source review."""
import argparse
import json
import math
from pathlib import Path
import subprocess

from analyze_paired import analyze
from run_paired import source_hash

FINAL_STATUSES = {'accepted', 'rejected', 'within_noise', 'correctness_failed',
                  'oom', 'unsupported', 'benchmark_failed', 'interrupted'}
SHAPES = {(b, 2048) for b in (1, 4, 8, 16)} | {(8, length) for length in (512, 4096, 8192)}
VARIANTS = ('ma_offload', 'ma_gpu', 'ma_gpu_unfolded')


def read(path):
    return json.loads(path.read_text())



def audit_retained_evidence(root, final, records):
    assert final['status'] == 'accepted' and final['workflow_stage'] == 'final_verdict'
    directory = root / final['attempt_id']
    matrix = read(directory / final['full_summary'])
    generation = read(directory / final['generation_summary'])
    checked_sources = set()
    environment_signatures = set()
    baseline_checkout = records['A0000'].get('frozen_checkout') or read(root / 'manifest.json')['frozen_checkout']
    hashes = {'baseline': source_hash(Path(baseline_checkout)),
              'candidate': source_hash(Path(final['frozen_checkout']))}
    for summary, modes, shapes, count in [(matrix, ('prefill', 'decode'), SHAPES, 84),
                                          (generation, ('generation',), {(1, 2048), (8, 2048)}, 12)]:
        points = summary['points']
        expected = {(impl, mode, variant, batch, length) for impl in ('baseline', 'candidate')
                    for mode in modes for variant in VARIANTS for batch, length in shapes}
        assert len(points) == count
        actual = {tuple(p[k] for k in ('implementation', 'mode', 'variant', 'batch', 'length')) for p in points}
        assert actual == expected
        for point in points:
            assert point['status'] in ('completed', 'reused') and math.isfinite(point['latency_ms']) and point['latency_ms'] > 0
            sha = records['A0000']['candidate_sha'] if point['implementation'] == 'baseline' else final['candidate_sha']
            assert point['candidate_sha'] == sha
            for source in point['source_results']:
                payload = read(Path(source))
                assert payload['status'] == 'completed' and payload['protocol_version'] == 'offload_gap_v1'
                assert payload['source']['git_commit']['stdout'].strip() == sha
                assert payload['source']['source_sha256'] == hashes[point['implementation']]
                config = payload['config']; row = payload['results'][0]
                assert config['mode'] == point['mode'] and config['batch_size'] == point['batch']
                assert config['seq_len'] == config['context_len'] == point['length']
                assert config['variants'] == [point['variant']]
                assert config['seed'] == 1234 and config['logits_to_keep'] == 1
                for key, value in dict(num_layers=24, hidden_size=2048, num_heads=32, num_kv_heads=32,
                                       intermediate_size=5632, vocab_size=32000).items():
                    assert config[key] == value
                for key, value in dict(num_hidden_layers=24, hidden_size=2048, num_heads=32, num_kv_heads=32,
                                       intermediate_size=5632, vocab_size=32000, qk_norm=False,
                                       use_gate=False, fuse_norm=False, tie_word_embeddings=False).items():
                    assert payload['model_config'][key] == value
                env = payload['env']; before = payload['environment_before']
                signature = tuple(env[k] for k in ('torch', 'torch_cuda', 'flash_attn', 'gpu', 'python', 'cuda_visible_devices'))
                signature += (tuple(before['cpu_affinity']), before['torch_threads'], before['torch_interop_threads'],
                              json.dumps(before['thread_environment'], sort_keys=True),
                              before['gpu_telemetry']['stdout'].splitlines()[1].split(', ')[1])
                environment_signatures.add(signature)
                assert row['gpu_parameters'] + row['cpu_parameters'] == 2836499968
                assert row['output_scope'] == 'cached_logits' and row['logits_to_keep'] == 1
                repeats = 5 if point['mode'] == 'generation' else 10
                assert config['warmup'] == (2 if repeats == 5 else 10)
                assert len(row['samples_ms']) == 3 and all(len(block) == repeats for block in row['samples_ms'])
                assert all(math.isfinite(value) and value > 0 for block in row['samples_ms'] for value in block)
                if point['mode'] == 'generation':
                    assert config['generation_steps'] == 128
                for when in ('memory_before', 'memory_after'):
                    assert all(k in row[when] for k in ('gpu_allocated_bytes', 'gpu_reserved_bytes', 'host_rss_bytes', 'offload_pinned_bytes', 'kv_cache_storage_bytes'))
                checked_sources.add(source)
        by_key = {(r['implementation'], r['mode'], r['batch'], r['length'], r['variant']): r for r in points}
        for r in points:
            if r['variant'] == 'ma_offload':
                gpu = by_key[(r['implementation'], r['mode'], r['batch'], r['length'], 'ma_gpu')]
                assert r['gpu_peak_allocated_bytes_max'] < gpu['gpu_peak_allocated_bytes_min']
    assert len(environment_signatures) == 1, 'Final comparisons mix software/device/thread environments'
    from compare_correctness import compare
    from reuse_correctness import validate_reference, ENV_KEYS, RUNTIME_KEYS
    correctness = []
    reference_paths = {}
    reuse_path = directory / final.get('correctness_reference_reuse', 'correctness-reference-reuse.json')
    if reuse_path.exists():
        for item in read(reuse_path)['references']:
            reference_paths[(item['batch'], item['seed'])] = Path(item['path'])
    baseline_point = next(point for point in matrix['points'] if point['implementation']=='baseline'
                          and point['variant']=='ma_gpu' and point['mode']=='prefill' and point['batch']==1)
    current = read(Path(baseline_point['source_results'][0]))
    for batch in (1, 8):
        for seed in (1234, 4321):
            path = directory / final['correctness_directory'] / f'comparison-b{batch}-s{seed}.json'
            result = read(path)
            assert result['status'] == 'completed' and result['bit_exact'] and not result['mismatches']
            assert result['max_absolute_error'] == 0 and result['argmax_agreement'] == 1
            assert result['candidate_source']['git_commit']['stdout'].strip() == final['candidate_sha']
            candidate_path = path.parent / f'correctness-b{batch}-s{seed}-candidate.json'
            reference_path = reference_paths.get((batch,seed), path.parent / f'correctness-b{batch}-s{seed}-baseline.json')
            candidate = read(candidate_path)
            reference = read(reference_path)
            validate_reference(reference, current, Path(baseline_checkout), records['A0000']['candidate_sha'], hashes['baseline'], batch, seed)
            assert candidate['source']['git_commit']['stdout'].strip() == final['candidate_sha']
            assert candidate['source']['source_sha256'] == hashes['candidate']
            assert Path(candidate['env']['model_module']).resolve().is_relative_to(Path(final['frozen_checkout']).resolve())
            for key in ENV_KEYS:
                assert candidate['env'][key] == reference['env'][key]
            for key in RUNTIME_KEYS:
                assert candidate['environment'][key] == reference['environment'][key]
            assert compare(reference, candidate) == result, 'Correctness report no longer matches its raw fingerprints'
            correctness.append(str(path))
    return dict(attempt=final['attempt_id'], source_results=sorted(checked_sources),
                environment_signature=next(iter(environment_signatures)), correctness_reports=correctness)


def require_eligible_gain(result, record):
    eligible = record.get('eligible_gain_workloads')
    if record.get('continuation_id') == 'continuation-02':
        assert eligible, 'New accepted step needs its predeclared eligible gain workloads'
    if eligible:
        keys = {(item['mode'], item['batch']) for item in eligible}
        assert any((row['mode'], row['batch']) in keys and row['repeatable_offload_and_gap_reduction']
                   for row in result['rows']), 'No confirmed gain in the predeclared changed primary path'


def audit(root, repository, artifacts, continuation=None):
    records = {p.parent.name: read(p) for p in sorted(root.glob('A[0-9][0-9][0-9][0-9]/record.json'))}
    attempts = {k: v for k, v in records.items() if k != 'A0000'}
    if continuation is None:
        assert not any(r.get('continuation_id') for r in attempts.values()), 'Explicit --continuation required for an additional-attempt campaign'
        assert 5 <= len(attempts) <= 10, 'Five to ten actual attempts required'
    else:
        tranche = read(root / continuation / 'manifest.json')
        assert tranche['campaign_id'] == 'offload-gap-001'
        previous_count = int(tranche['start_after_attempt'][1:])
        additional = len(attempts) - previous_count
        assert tranche['minimum_additional_attempts'] <= additional <= tranche['maximum_additional_attempts'], 'Additional attempt count does not satisfy user request'
        assert tranche['first_new_attempt'] == f'A{previous_count+1:04d}'
    assert list(attempts) == [f'A{i:04d}' for i in range(1, len(attempts)+1)]
    for name, record in attempts.items():
        assert record['campaign_id'] == 'offload-gap-001'
        assert record.get('status') in FINAL_STATUSES and record['workflow_stage'] == 'final_verdict', name
        sha = record['candidate_sha']
        assert len(sha) == 40
        assert subprocess.check_output(['git', 'cat-file', '-t', sha], cwd=repository, text=True).strip() == 'commit'
        assert record['parent_attempt_id'] in records
        assert record['hypothesis'] and record['profile_evidence'] and record['decision']
        if record['status'] != 'accepted':
            assert record['accepted_step'] is None
    retained = sorted([v for v in attempts.values() if v['status'] == 'accepted'], key=lambda r: r['accepted_step'])
    assert retained and [r['accepted_step'] for r in retained] == list(range(1, len(retained)+1))
    confirmations = []
    for record in retained:
        directory = root / record['attempt_id']
        path = directory / record['confirmation_manifest']
        manifest = read(path)
        assert manifest['source_commits'] == {'baseline': records['A0000']['candidate_sha'], 'candidate': record['candidate_sha']}
        result = analyze(path)
        assert result['confirmation_promising'], record['attempt_id']
        confirmations.append(dict(attempt=record['attempt_id'], manifest=str(path), primary_rows=result['rows']))
        if record['parent_attempt_id'] != 'A0000':
            parent_path = directory / record['parent_confirmation_manifest']
            assert read(parent_path)['source_commits']['baseline'] == records[record['parent_attempt_id']]['candidate_sha']
            parent_result = analyze(parent_path)
            assert parent_result['confirmation_promising'], 'No incremental gain against accepted parent'
            require_eligible_gain(parent_result, record)
        else:
            require_eligible_gain(result, record)
    final = retained[-1]
    assert source_hash(repository) == source_hash(Path(final['frozen_checkout'])), 'Current implementation differs from verified final source'
    retained_evidence = [audit_retained_evidence(root, record, records) for record in retained]
    assert len({json.dumps(item['environment_signature']) for item in retained_evidence}) == 1, 'Accepted history mixes environments'
    checked_sources = set(retained_evidence[-1]['source_results'])
    correctness = retained_evidence[-1]['correctness_reports']
    for name in ('scaling-batch', 'scaling-length', 'memory-batch', 'memory-length', 'absolute-gap', 'relative-overhead', 'generation'):
        for extension in ('pdf', 'svg', 'png'):
            assert (artifacts / 'validation' / f'{name}.{extension}').stat().st_size > 0
    for name in ('attempt-history-screen_v1_w3_n5_r1', 'accepted-steps-latency', 'accepted-steps-speedup', 'accepted-steps-absolute-gap', 'accepted-steps-relative-overhead'):
        for extension in ('pdf', 'svg', 'png'):
            assert (artifacts / 'history' / f'{name}.{extension}').stat().st_size > 0
    for name in ('validation/source.csv', 'validation/source.json', 'history/history-source.json',
                 'history/history-source.csv', 'history/accepted_steps-source.csv', 'REPRODUCE.md', 'REPORT.md'):
        assert (artifacts / name).stat().st_size > 0
    return dict(campaign_id='offload-gap-001', status='core_evidence_checks_passed', attempts=len(attempts),
                accepted_steps=len(retained), continuation=continuation, final_attempt=final['attempt_id'], final_source_sha=final['candidate_sha'],
                source_hash=source_hash(repository), checked_raw_sources=len(checked_sources),
                confirmations=confirmations, full_correctness_reports=correctness,
                retained_evidence_checks=retained_evidence,
                remaining_manual_audit=['Inspect final rendered figures and report/commands',
                                        'Review secondary regression investigations and all attempt verdicts',
                                        'Verify committed/published audit trail and final worktree state'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--continuation', help='Audit an explicitly requested additional-attempt tranche')
    args = parser.parse_args()
    result = audit(args.campaign, args.repository, args.artifacts, args.continuation)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['status'], result['attempts'], 'attempts;', result['accepted_steps'], 'accepted steps')
