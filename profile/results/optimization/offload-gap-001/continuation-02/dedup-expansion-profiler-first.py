"""Profile frozen within-call dedup GPU expansion; no headline timing."""
import argparse,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--batch',type=int,default=8);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
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
 torch.cuda.synchronize();original=torch.index_select;count=[0]
 def traced(source,dim,index,*args,**kwargs):
  if source.is_cuda and source.ndim==2 and source.shape[1]==2048 and index.numel()==a.batch*2048:
   count[0]+=1
   with torch.profiler.record_function('offload_inverse_expansion'):
    return original(source,dim,index,*args,**kwargs)
  return original(source,dim,index,*args,**kwargs)
 torch.index_select=traced
 try:
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
   one();torch.cuda.synchronize()
 finally:torch.index_select=original
 rows=[dict(name=e.key,count=e.count,cpu_total_us=e.cpu_time_total,device_total_us=e.device_time_total,self_device_us=e.self_device_time_total) for e in prof.key_averages()]
 target=[r for r in rows if r['name']=='offload_inverse_expansion']
 assert count[0]==24 and len(target)==1 and target[0]['device_total_us']>0,(count,target)
 trace=a.output.with_suffix('.trace.json');prof.export_chrome_trace(str(trace))
 payload=dict(campaign_id='offload-gap-001',purpose=__doc__,source=source_state(),helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),command=sys.argv,environment=environment_details(),env=env_fingerprint(),batch=a.batch,length=2048,warmup=10,expansion_calls=count[0],expansion=target[0],trace=str(trace),events=rows,limitations='Instrumented single model call. GPU expansion work is not automatically removable latency; a fused indirect add must be measured and pass exactness/retention gates.')
 a.output.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(target[0]))
 model.close_memory_offload()
