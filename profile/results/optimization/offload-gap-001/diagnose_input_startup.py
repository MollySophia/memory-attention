"""Instrument ID staging and embedding startup separately from model timing."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-root', type=Path, required=True)
p.add_argument('--batch', type=int, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
sys.path[:0] = [str(a.source_root), str(a.source_root / 'profile')]
import torch
from bench_fla import build, env_fingerprint
from benchmark_telemetry import source_state, environment_details

config = SimpleNamespace(seed=1234,hidden_size=2048,num_layers=24,num_heads=32,
    num_kv_heads=32,vocab_size=32000,hidden_ratio=4,intermediate_size=5632,
    group_size=1,prefetch_depth=4,policy='auto',device='cuda:0')
with torch.inference_mode():
    model = build(config)
    model.enable_memory_offload()
    model.set_offload_offloader(a.batch, 2048)
    torch.manual_seed(1235)
    ids = torch.randint(0,32000,(a.batch,2048),device='cuda')
    def one():
        return model(input_ids=ids,use_cache=True,logits_to_keep=1)
    for _ in range(10):
        one()
    torch.cuda.synchronize()
    records = []
    events = []
    original_to = torch.Tensor.to
    def traced_to(tensor, *args, **kwargs):
        target = args[0] if args else kwargs.get('device')
        selected = (tensor.is_cuda and tensor.dtype == torch.long
                    and tuple(tensor.shape) == (a.batch,2048)
                    and isinstance(target,(str,torch.device))
                    and torch.device(target).type == 'cpu')
        if not selected:
            return original_to(tensor,*args,**kwargs)
        start = time.perf_counter_ns()
        value = original_to(tensor,*args,**kwargs)
        records.append(dict(operation='ID_to_CPU',cpu_ms=(time.perf_counter_ns()-start)/1e6))
        return value
    def embedding_before(module, args):
        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        events.append((start,end,time.perf_counter_ns()))
    def embedding_after(module, args, output):
        start,end,cpu_start = events[-1]
        cpu_ms = (time.perf_counter_ns()-cpu_start)/1e6
        end.record()
        records.append(dict(operation='embedding_dispatch',cpu_ms=cpu_ms))
    handles = [model.model.embeddings.register_forward_pre_hook(embedding_before),
               model.model.embeddings.register_forward_hook(embedding_after)]
    torch.Tensor.to = traced_to
    try:
        start = time.perf_counter_ns()
        output = one()
        torch.cuda.synchronize()
        wall = (time.perf_counter_ns()-start)/1e6
    finally:
        torch.Tensor.to = original_to
        for handle in handles:
            handle.remove()
    assert len(events)==1 and sum(r['operation']=='ID_to_CPU' for r in records)==1
    gpu_ms = events[0][0].elapsed_time(events[0][1])
    assert gpu_ms>0
    payload = dict(campaign_id='offload-gap-001',purpose=__doc__,source=source_state(),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),command=sys.argv,
        environment=environment_details(),env=env_fingerprint(),batch=a.batch,length=2048,
        warmup=10,records=records,embedding_cuda_interval_ms=gpu_ms,instrumented_model_ms=wall,
        limitation='One instrumented call, not acceptance timing. CPU dispatch and CUDA stream spans overlap; CUDA events may include dispatch gaps and contention. No sum is claimed as removable latency.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(dict(records=records,embedding_cuda_interval_ms=gpu_ms)))
    model.close_memory_offload()
