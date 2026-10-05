"""Instrument frozen-source pipeline coordination; never use as headline timing."""
import argparse,hashlib,json,sys,time,threading
from pathlib import Path
from types import SimpleNamespace
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-root',type=Path,required=True);p.add_argument('--batch',type=int,required=True)
p.add_argument('--length',type=int,default=2048)
p.add_argument('--single-host-max-tokens',type=int,default=None)
p.add_argument('--dependency-events',action='store_true',help='Instrument compute-stream intervals around ticket acquisition; diagnostic perturbation only')
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists(), 'Preserve earlier diagnostics under distinct output names'
from diagnose_layer_working_set import require_idle
require_idle()
sys.path[:0]=[str(a.source_root),str(a.source_root/'profile')]
import torch
from bench_fla import build,env_fingerprint
from benchmark_telemetry import source_state,environment_details
from fla.layers.memory_offload import MemoryTableOffloader,GroupTicket
config=SimpleNamespace(seed=1234,hidden_size=2048,num_layers=24,num_heads=32,num_kv_heads=32,vocab_size=32000,hidden_ratio=4,intermediate_size=5632,group_size=1,prefetch_depth=4,policy='auto',device='cuda:0')
with torch.inference_mode():
 model=build(config)
 if a.single_host_max_tokens is not None:
  assert hasattr(model.config,'memory_offload_single_host_max_tokens')
  model.config.memory_offload_single_host_max_tokens=a.single_host_max_tokens
 model.enable_memory_offload();model.set_offload_offloader(a.batch,a.length)
 torch.manual_seed(1235);ids=torch.randint(0,32000,(a.batch,a.length),device='cuda')
 def one():return model(input_ids=ids,use_cache=True,logits_to_keep=1)
 for _ in range(10):one()
 torch.cuda.synchronize();records=[];originals=[];dependency_events=[]
 def wrap(cls,name,label,worker_only=False):
  original=getattr(cls,name);originals.append((cls,name,original))
  def traced(*args,**kw):
   worker=threading.current_thread().name.startswith('ma-prefetch')
   if worker_only and not worker:return original(*args,**kw)
   start=time.perf_counter_ns();r=original(*args,**kw)
   records.append(dict(label=label,thread=threading.current_thread().name,cpu_ms=(time.perf_counter_ns()-start)/1e6));return r
  setattr(cls,name,traced)
 wrap(threading.Event,'wait','producer_release_wait',True)
 wrap(torch.cuda.Event,'synchronize','producer_dma_sync',True)
 wrap(MemoryTableOffloader,'_view','buffer_view')
 wrap(GroupTicket,'__init__','ticket_setup')
 wrap(torch,'index_select','gather',True)
 if a.dependency_events:
  original_acquire=GroupTicket.acquire;originals.append((GroupTicket,'acquire',original_acquire))
  def traced_acquire(ticket,offset):
   stream=torch.cuda.current_stream(ticket.owner.device)
   begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
   begin.record(stream);cpu_begin=time.perf_counter_ns()
   result=original_acquire(ticket,offset)
   cpu_ms=(time.perf_counter_ns()-cpu_begin)/1e6;end.record(stream)
   dependency_events.append((ticket.start+offset,begin,end,cpu_ms))
   return result
  GroupTicket.acquire=traced_acquire
 try:
  start=time.perf_counter();out=one();torch.cuda.synchronize();wall=(time.perf_counter()-start)*1000
 finally:
  for cls,name,original in reversed(originals):setattr(cls,name,original)
 payload=dict(campaign_id='offload-gap-001',purpose='diagnostic only; instrumented overlapping CPU spans, not latency acceptance',source=source_state(),environment=environment_details(),env=env_fingerprint(),batch=a.batch,length=a.length,command=sys.argv,helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model_config=model.config.to_dict(),warmup=10,wall_ms=wall,records=records,summary={label:dict(count=sum(r['label']==label for r in records),sum_ms=sum(r['cpu_ms'] for r in records if r['label']==label)) for label in sorted({r['label'] for r in records})})
 if a.dependency_events:
  assert len(dependency_events)==24
  payload['compute_dependency_intervals']=[dict(layer=layer,cuda_interval_ms=begin.elapsed_time(end),cpu_acquire_ms=cpu) for layer,begin,end,cpu in dependency_events]
  payload['dependency_limitations']='CUDA intervals bracket host-ready wait and copied-event dependency on compute stream, including instrumentation and submission gaps. They do not isolate DMA duration or establish removable end-to-end latency.'
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary']))
 model.close_memory_offload()
