"""Read-only final evidence audit; never accepts a candidate or launches GPU work."""
import argparse
import json
from pathlib import Path
import driver as d
from compare_correctness import compare


def audit_manifests(a):
    parent = d.record(a)['parent_attempt_id']
    signatures = {i: d.signature(x) for i, x in [('baseline', parent), ('candidate', a)]}
    environments = set()
    counts = {}
    formal_cells = set()
    for stage in ('R01-complete-screen', 'R02-parent-confirmation',
                  'R03-remaining-confirmation', 'R04-unfolded-reference'):
        directory = d.C / a / stage
        manifest = d.read(directory / 'manifest.json')
        assert manifest['status'] == 'completed', stage
        assert manifest['source_signatures'] == signatures, (stage, 'source signatures')
        assert manifest['attempt'] == a and manifest['baseline'] == parent
        cells = set()
        formal = stage in ('R02-parent-confirmation', 'R03-remaining-confirmation')
        if formal:
            assert manifest['formal']
            if manifest['jobs']:
                d.validate_balanced_order(manifest['jobs'])
        for job in manifest['jobs']:
            assert job['status'] == 'completed' and job['returncode'] == 0
            sig = signatures[job['implementation']]
            assert (job['attempt'], job['candidate_sha'], job['source_sha256']) == (
                sig['attempt'], sig['commit'], sig['source_sha256'])
            cell = tuple(job[k] for k in ('mode', 'batch', 'length', 'block', 'implementation', 'variant'))
            assert cell not in cells, (stage, 'duplicate cell', cell)
            cells.add(cell)
            env, _ = d.audit_job(job, directory / (job['name'] + '.json'))
            environments.add(env)
        if formal:
            assert not formal_cells.intersection(cells), 'Repeated R02/R03 evidence'
            formal_cells.update(cells)
        else:
            variants = ('ma_gpu_unfolded',) if stage == 'R04-unfolded-reference' else ('ma_offload', 'ma_gpu')
            expected = {(*w, 1, i, v) for w in d.WORKLOADS
                        for i in signatures for v in variants}
            assert cells == expected, (stage, 'coverage')
        counts[stage] = len(cells)
    expected = {(*w, b, i, v) for w in d.WORKLOADS for b in (1, 2, 3)
                for i in signatures for v in ('ma_offload', 'ma_gpu')}
    assert formal_cells == expected, 'All16 fresh three-block coverage required'
    assert len(environments) == 1, 'Cross-stage environment mismatch'
    return counts


def audit_correctness(a):
    directory = d.C / a / 'R05-full-correctness'
    sigs = {i: d.signature(x) for i, x in [('baseline', 'A0000'), ('candidate', a)]}
    checked = []
    for batch, length in ((1, 2048), (8, 2048), (8, 512)):
        for seed in (1234, 4321):
            values = {}
            for impl, sig in sigs.items():
                p = d.read(directory / f'correctness-b{batch}-l{length}-s{seed}-{impl}.json')
                assert (p['batch_size'], p['prefix_length'], p['seed']) == (batch, length, seed)
                assert p['source']['git_commit']['stdout'].strip() == sig['commit']
                assert p['source']['source_sha256'] == sig['source_sha256']
                assert Path(p['env']['model_module']).resolve() == Path(sig['root']) / 'fla/models/memory/modeling_memory.py'
                for field, value in dict(hidden_size=2048, num_hidden_layers=24, num_heads=32,
                                         num_kv_heads=32, intermediate_size=5632, vocab_size=32000,
                                         qk_norm=False, use_gate=False, fuse_norm=False).items():
                    assert p['model_config'][field] == value
                assert [c['step'] for c in p['checkpoints']] == list(range(129))
                for c in p['checkpoints']:
                    assert c['context_length'] == length + c['step']
                    if c['step'] in (0, 1, 2, 128):
                        assert len(c['hidden_states']) == 25 and len(c['kv_states']) == 24
                values[impl] = p
            result = compare(values['baseline'], values['candidate'])
            assert result['bit_exact']
            assert result == d.read(directory / f'comparison-b{batch}-l{length}-s{seed}.json')
            checked.append(dict(batch=batch, length=length, seed=seed))
    return checked


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    counts = audit_manifests(args.attempt)
    correctness = audit_correctness(args.attempt)
    print(json.dumps(dict(attempt=args.attempt, evidence_integrity='passed',
                          process_counts=counts, correctness_pairs=correctness,
                          accepted=False, note='Integrity audit only; review fixed gain/regression gates, memory tradeoffs and goal evidence separately.'), indent=2))
