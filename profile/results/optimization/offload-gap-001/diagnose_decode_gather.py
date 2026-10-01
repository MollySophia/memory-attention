"""Instrument CPU index_select wall/thread-CPU time across a 10/10/3 decode run.
Diagnostic only; wrapper timing overhead invalidates headline latency claims.
"""
import argparse
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();root=a.source_root.resolve()
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'profile'))
import torch
from bench_fla import build,measure,env_fingerprint
from benchmark_telemetry import source_state,environment_details
args=SimpleNamespace(seed=1234,hidden_size=2048,num_layers=24,num_heads=32,num_kv_heads=32,
                     vocab_size=32000,hidden_ratio=4,intermediate_size=5632,group_size=1,
                     prefetch_depth=4,policy='auto',device='cuda:0',batch_size=1,seq_len=2048,
                     context_len=2048,mode='decode',stage='confirmation',warmup=10,repeats=10,
                     rounds=3,logits_to_keep=1,prefill_workload='inference',generation_steps=128)
records=[];original=torch.index_select

def index(source,dim,indices,*pos,**kw):
    if source.device.type!='cpu':return original(source,dim,indices,*pos,**kw)
    start=time.perf_counter_ns();cpu=time.thread_time_ns()
    result=original(source,dim,indices,*pos,**kw)
    cpu=time.thread_time_ns()-cpu;wall=time.perf_counter_ns()-start
    records.append(dict(ids=indices.numel(),source_shape=list(source.shape),wall_ms=wall/1e6,
                        caller_thread_cpu_ms=cpu/1e6,thread=threading.current_thread().name))
    return result

with torch.inference_mode():
    model=build(args)
    try:
        torch.index_select=index
        row=measure(model,args,'ma_offload',torch.device(args.device))
    finally:
        torch.index_select=original
        model.close_memory_offload()
payload=dict(campaign_id='offload-gap-001',purpose='diagnostic only; instrumented, not headline timing',
             command=sys.argv,source=source_state(),env=env_fingerprint(),environment=environment_details(),
             config=vars(args),records=records,model_measurement=row,
             note='For ids=1 the first 10 gathers are warmup and the following 30 match model samples in order. Caller thread CPU excludes worker CPU; wall/CPU differences can indicate waiting but do not uniquely identify its cause.')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(payload,indent=2)+'\n')
print('saved',a.output)
