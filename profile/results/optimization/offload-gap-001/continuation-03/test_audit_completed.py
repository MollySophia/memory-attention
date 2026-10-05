import copy
import pytest
import driver as d
import audit_completed as audit


@pytest.mark.parametrize('blocks', [3, 6])
def test_cross_stage_audit_requires_complete_unique_matching_evidence(tmp_path, monkeypatch, blocks):
    monkeypatch.setattr(d, 'C', tmp_path)
    monkeypatch.setattr(d, 'record', lambda a: dict(parent_attempt_id='A0016', confirmation_blocks=blocks))
    monkeypatch.setattr(d, 'signature', lambda a: dict(attempt=a, commit='sha-'+a,
                      source_sha256='hash-'+a, root=str(d.root(a))))
    monkeypatch.setattr(d, 'estimate', lambda *args: 1.)
    monkeypatch.setattr(d, 'audit_job', lambda j, path: (('same environment',), {}))
    plans = {}
    for stage, workloads, formal in [
        ('R01-complete-screen', d.WORKLOADS, False),
        ('R02-parent-confirmation', d.WORKLOADS[:9], True),
        ('R03-remaining-confirmation', d.WORKLOADS[9:], True),
        ('R04-unfolded-reference', d.WORKLOADS, False),
    ]:
        directory = tmp_path / 'A0022' / stage
        directory.mkdir(parents=True)
        p = d.plan('A0022', workloads, 'A0016', formal, directory)
        if stage == 'R04-unfolded-reference':
            p['jobs'] = [j for j in p['jobs'] if j['variant'] == 'ma_gpu']
            for j in p['jobs']:
                j['variant'] = 'ma_gpu_unfolded'
        p['status'] = 'completed'
        for j in p['jobs']:
            j.update(status='completed', returncode=0)
        plans[stage] = p
        d.save(directory / 'manifest.json', p)
    assert sum(audit.audit_manifests('A0022').values()) == 96+64*blocks
    path = tmp_path / 'A0022/R03-remaining-confirmation/manifest.json'
    for mutation in ('duplicate', 'missing', 'source', 'unfinished'):
        p = copy.deepcopy(plans['R03-remaining-confirmation'])
        if mutation == 'duplicate':
            p['jobs'].append(copy.deepcopy(p['jobs'][0]))
        elif mutation == 'missing':
            p['jobs'] = p['jobs'][:-1]
        elif mutation == 'source':
            p['jobs'][0]['source_sha256'] = 'other source'
        else:
            p['status'] = 'running'
        d.save(path, p)
        with pytest.raises(AssertionError):
            audit.audit_manifests('A0022')
    d.save(path, plans['R03-remaining-confirmation'])
    monkeypatch.setattr(d, 'audit_job', lambda j, path: ((path.parent.name,), {}))
    with pytest.raises(AssertionError, match='Cross-stage environment mismatch'):
        audit.audit_manifests('A0022')


def test_correctness_audit_checks_source_scope_and_recomputes_comparison(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'C', tmp_path)
    monkeypatch.setattr(d, 'signature', lambda a: dict(attempt=a, commit='sha-'+a,
                      source_sha256='hash-'+a, root=str(d.root(a))))
    directory = tmp_path / 'A0022/R05-full-correctness'
    directory.mkdir(parents=True)
    fingerprint = dict(shape=[1], dtype='torch.bfloat16', sha256='same bytes', finite=True)
    for batch, length in ((1, 2048), (8, 2048), (8, 512)):
        for seed in (1234, 4321):
            values = {}
            for impl, aid, variant in [('baseline', 'A0000', 'ma_gpu'), ('candidate', 'A0022', 'ma_offload')]:
                p = dict(batch_size=batch, prefix_length=length, seed=seed, status='completed',
                         scope='fixture scope', variant=variant,
                         source=dict(git_commit=dict(stdout='sha-'+aid), source_sha256='hash-'+aid),
                         env=dict(model_module=str(d.root(aid)/'fla/models/memory/modeling_memory.py')),
                         model_config=dict(hidden_size=2048, num_hidden_layers=24, num_heads=32,
                                           num_kv_heads=32, intermediate_size=5632, vocab_size=32000,
                                           qk_norm=False, use_gate=False, fuse_norm=False), checkpoints=[])
                for step in range(129):
                    c = dict(step=step, context_length=length+step, logits=fingerprint, argmax=[[0]])
                    if step in (0, 1, 2, 128):
                        c.update(hidden_states=[fingerprint]*25, kv_states=[[fingerprint]*2]*24)
                    p['checkpoints'].append(c)
                values[impl] = p
                d.save(directory/f'correctness-b{batch}-l{length}-s{seed}-{impl}.json', p)
            d.save(directory/f'comparison-b{batch}-l{length}-s{seed}.json',
                   audit.compare(values['baseline'], values['candidate']))
    assert len(audit.audit_correctness('A0022')) == 6
    path = directory/'correctness-b8-l512-s4321-candidate.json'
    original = d.read(path)
    for mutation in ('source', 'context', 'hidden', 'nonfinite', 'bytes'):
        p = copy.deepcopy(original)
        if mutation == 'source':
            p['source']['source_sha256'] = 'other source'
        elif mutation == 'context':
            p['checkpoints'][128]['context_length'] -= 1
        elif mutation == 'hidden':
            p['checkpoints'][0]['hidden_states'].pop()
        elif mutation == 'nonfinite':
            p['checkpoints'][63]['logits']['finite'] = False
        else:
            p['checkpoints'][77]['logits']['sha256'] = 'different bytes'
        d.save(path, p)
        with pytest.raises(AssertionError):
            audit.audit_correctness('A0022')
