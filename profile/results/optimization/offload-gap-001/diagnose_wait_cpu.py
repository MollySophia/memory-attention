"""Separate producer wait wall time from thread CPU consumption; diagnostic only."""
import argparse,hashlib,json,sys,threading,time
from pathlib import Path
from types import SimpleNamespace
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-root',type=Path,required=True);p.add_argument('--batch',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path[:0]=[str(a.source_root),str(a.source_root/'profile')]
import torch
from bench_fla import build,env_fingerprint
from benchmark_telemetry import source_state,environment_details
config=SimpleNamespace(seed=1234,hidden_size=2048,num_layers=24,num_heads=32,num_kv_heads=32,vocab_size=32000,hidden_ratio=4,intermediate_size=5632,group_size=1,prefetch_depth=4,policy='auto',device='cuda:0')
with torch.inference_mode():
 model=build(config);model.enable_memory_offload();model.set_offload_offloader(a.batch,2048)
 torch.manual_seed(1235);ids=torch.randint(0,32000,(a.batch,2048),device='cuda')
 def one():return model(input_ids=ids,use_cache=True,logits_to_keep=1)
 for _ in range(10):one()
 torch.cuda.synchronize();records=[]
 original=torch.cuda.Event.synchronize
 def traced(event):
  if not threading.current_thread().name.startswith('ma-prefetch'):return original(event)
  wall=time.perf_counter_ns();cpu=time.thread_time_ns();result=original(event)
  records.append(dict(wall_ms=(time.perf_counter_ns()-wall)/1e6,thread_cpu_ms=(time.thread_time_ns()-cpu)/1e6))
  return result
 torch.cuda.Event.synchronize=traced
 try:
  start=time.perf_counter_ns();one();torch.cuda.synchronize();wall=(time.perf_counter_ns()-start)/1e6
 finally:torch.cuda.Event.synchronize=original
 payload=dict(campaign_id='offload-gap-001',purpose=__doc__,source=source_state(),helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),command=sys.argv,environment=environment_details(),env=env_fingerprint(),batch=a.batch,length=2048,warmup=10,instrumented_model_wall_ms=wall,records=records,summary=dict(waits=len(records),wait_wall_ms=sum(r['wall_ms'] for r in records),wait_thread_cpu_ms=sum(r['thread_cpu_ms'] for r in records)),limitations='Instrumented overlapping waits are not removable model latency; CPU time can include driver work and does not itself prove busy-spinning.')
 a.output.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary']))
 model.close_memory_offload()
