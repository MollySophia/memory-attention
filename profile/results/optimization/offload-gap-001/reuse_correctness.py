"""Strict reuse of frozen resident full-model byte fingerprints, never timings."""
import json
import re
from pathlib import Path
import subprocess
from run_paired import source_hash

SCOPE = 'logits every step; all hidden and KV states at prefix and decode steps 1,2,128; unpadded full model'
ENV_KEYS = ('torch','torch_cuda','flash_attn','python','gpu','gpu_capability','cuda_visible_devices')
RUNTIME_KEYS = ('torch_threads','torch_interop_threads','cpu_affinity','thread_environment')


def validate_reference(payload, current, root, sha, digest, batch, seed):
    assert payload['campaign_id'] == 'offload-gap-001'
    assert payload['status'] == 'completed' and payload['variant'] == 'ma_gpu'
    assert payload['batch_size'] == batch and payload['seed'] == seed
    assert payload['scope'] == SCOPE
    source = payload['source']
    assert source['git_commit']['stdout'].strip() == sha
    assert source['source_sha256'] == digest
    assert not source['git_status']['stdout'].strip()
    assert not source['source_patch']['stdout'].strip()
    assert Path(payload['env']['model_module']).resolve().is_relative_to(root.resolve())
    for name in ENV_KEYS:
        assert payload['env'][name] == current['env'][name], name
    for name in RUNTIME_KEYS:
        assert payload['environment'][name] == current['environment_before'][name], name
    old_gpu = payload['environment']['gpu_telemetry']['stdout'].splitlines()[1].split(', ')[:2]
    new_gpu = current['environment_before']['gpu_telemetry']['stdout'].splitlines()[1].split(', ')[:2]
    assert old_gpu == new_gpu
    config = payload['model_config']
    for name, value in dict(hidden_size=2048,num_hidden_layers=24,num_heads=32,num_kv_heads=32,
                            intermediate_size=5632,vocab_size=32000,qk_norm=False,use_gate=False,
                            fuse_norm=False,tie_word_embeddings=False).items():
        assert config[name] == value, name
    def fingerprint(value, shape):
        assert value['shape'] == shape and value['dtype'] == 'torch.bfloat16'
        assert value['finite'] is True
        assert re.fullmatch('[0-9a-f]{64}', value['sha256'])
    checkpoints = payload['checkpoints']
    assert len(checkpoints) == 129
    for step, point in enumerate(checkpoints):
        assert point['step'] == step and point['context_length'] == 2048 + step
        fingerprint(point['logits'], [batch,1,32000])
        assert len(point['argmax']) == batch
        assert all(len(row)==1 and isinstance(row[0],int) and 0<=row[0]<32000 for row in point['argmax'])
        if step in (0,1,2,128):
            assert len(point['hidden_states']) == 25 and len(point['kv_states']) == 24
            for value in point['hidden_states']:
                fingerprint(value, [batch,2048 if step==0 else 1,2048])
            for layer in point['kv_states']:
                assert len(layer) == 2
                for value in layer:
                    fingerprint(value, [batch,2048+step,2048])
    return True


def reference_index(directory, baseline_root, current):
    sha = subprocess.check_output(['git','-C',str(baseline_root),'rev-parse','HEAD'],text=True).strip()
    digest = source_hash(baseline_root)
    assert current['source']['git_commit']['stdout'].strip() == sha
    assert current['source']['source_sha256'] == digest
    result = {}
    for batch in (1,8):
        for seed in (1234,4321):
            path = directory / f'correctness-b{batch}-s{seed}-baseline.json'
            validate_reference(json.loads(path.read_text()),current,baseline_root,sha,digest,batch,seed)
            result[(batch,seed)] = path.resolve()
    return result
