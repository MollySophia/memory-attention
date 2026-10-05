"""Untimed-for-claims generation diagnostic: CPU submission, process CPU and GC.
Preserves asynchronous model execution; synchronization only around trajectory.
"""
import argparse,gc,json,sys,time,hashlib
from pathlib import Path
from types import SimpleNamespace
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--mapped-min-tokens',type=int,default=None)
p.add_argument('--profile',action='store_true')
a=p.parse_args();root=a.source_root.resolve();sys.path[:0]=[str(root/'profile'),str(root)]
import torch
from bench_fla import build,env_fingerprint
from benchmark_telemetry import source_state,environment_details
args=SimpleNamespace(seed=1234,hidden_size=2048,num_layers=24,num_heads=32,num_kv_heads=32,vocab_size=32000,hidden_ratio=4,intermediate_size=5632,group_size=1,prefetch_depth=4,policy='auto',device='cuda:0')
with torch.inference_mode():
 model=build(args)
 if a.mapped_min_tokens is not None:model.config.memory_offload_mapped_bulk_min_tokens=a.mapped_min_tokens
 model.enable_memory_offload();model.set_offload_offloader(1,2048);model.set_offload_offloader(1,1)
 torch.manual_seed(1235);prefix=torch.randint(0,32000,(1,2048),device='cuda');tokens=torch.randint(0,32000,(128,1,1),device='cuda').unbind()
 def trajectory(wall=None,cpu=None):
  output=None
  for i,token in enumerate((prefix,*tokens)):
   cache=output.past_key_values if output is not None else None
   del output
   t=time.perf_counter();c=time.process_time()
   output=model(input_ids=token,past_key_values=cache,use_cache=True,logits_to_keep=1)
   if wall is not None:wall[i]=(time.perf_counter()-t)*1000;cpu[i]=(time.process_time()-c)*1000
  return output
 for _ in range(2):trajectory()
 torch.cuda.synchronize()
 wall=[0.]*129;cpu=[0.]*129;events=[]
 def callback(phase,info):events.append((time.perf_counter(),phase,dict(info)))
 gc.callbacks.append(callback)
 start=time.perf_counter();output=trajectory(wall,cpu);torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
 gc.callbacks.remove(callback)
 payload=dict(campaign_id='offload-gap-001',purpose='diagnostic only; CPU submission includes stream waits; not isolated per-token GPU time',command=sys.argv,source=source_state(),env=env_fingerprint(),environment=environment_details(),helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model_config=model.config.to_dict(),warmup_trajectories=2,measured_trajectories=1,wall_ms=elapsed,call_wall_ms=wall,call_process_cpu_ms=cpu,gc_events=[dict(relative_ms=(t-start)*1000,phase=phase,info=info) for t,phase,info in events])
 a.output.parent.mkdir(parents=True,exist_ok=True)
 if a.profile:
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
   for token in tokens[:4]:
    output=model(input_ids=token,past_key_values=output.past_key_values,use_cache=True,logits_to_keep=1)
   torch.cuda.synchronize()
  trace=a.output.with_suffix('.trace.json');prof.export_chrome_trace(str(trace))
  payload['profile_trace']=str(trace)
  payload['profile_events']=[dict(key=e.key,count=e.count,self_cpu_time_total_us=e.self_cpu_time_total,cpu_time_total_us=e.cpu_time_total) for e in prof.key_averages()]
  payload['profile_scope']='Four additional growing decode calls after diagnostic trajectory; profiler timing excluded from wall_ms'
 a.output.write_text(json.dumps(payload,indent=2)+'\n');model.close_memory_offload()
