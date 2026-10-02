"""Profile frozen within-call dedup GPU expansion; no headline timing."""
import argparse,hashlib,json,sys,time
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
 torch.cuda.synchronize();original=torch.index_select;records=[]
 def traced(source,dim,index,*args,**kwargs):
  if source.is_cuda and source.ndim==2 and source.shape[1]==2048 and index.numel()==a.batch*2048:
   begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
   begin.record();cpu=time.perf_counter_ns()
   value=original(source,dim,index,*args,**kwargs)
   elapsed=(time.perf_counter_ns()-cpu)/1e6;end.record()
   records.append((begin,end,elapsed));return value
  return original(source,dim,index,*args,**kwargs)
 torch.index_select=traced
 try:
  start=time.perf_counter_ns();one();torch.cuda.synchronize();wall=(time.perf_counter_ns()-start)/1e6
 finally:torch.index_select=original
 rows=[dict(cpu_ms=cpu,cuda_stream_ms=begin.elapsed_time(end)) for begin,end,cpu in records]
 assert len(rows)==24 and all(r['cuda_stream_ms']>0 for r in rows),rows
 summary=dict(count=len(rows),cpu_total_ms=sum(r['cpu_ms'] for r in rows),cuda_stream_total_ms=sum(r['cuda_stream_ms'] for r in rows))
 payload=dict(campaign_id='offload-gap-001',purpose=__doc__,source=source_state(),helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),command=sys.argv,environment=environment_details(),env=env_fingerprint(),batch=a.batch,length=2048,warmup=10,instrumented_model_wall_ms=wall,expansion=summary,records=rows,limitations='Instrumented single call. CUDA-event intervals bracket each expansion on its compute stream and can include dispatch gaps/resource contention; they are not pure kernel duration or guaranteed removable model latency. CPU and CUDA spans overlap.')
 a.output.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(summary))
 model.close_memory_offload()
