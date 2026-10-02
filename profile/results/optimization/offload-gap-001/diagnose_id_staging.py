"""GPU-to-CPU ID staging variants, fresh inputs; isolated diagnostics only."""
import argparse,hashlib,json,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path[:0]=[str(a.source_root),str(a.source_root/'profile')]
import torch
from benchmark_telemetry import environment_details,source_state
results=[]
with torch.inference_mode():
 for shape in ((1,1),(8,1),(1,2048),(8,2048),(16,2048)):
  host=torch.empty(shape,dtype=torch.long,pin_memory=True);event=torch.cuda.Event()
  samples={k:[] for k in ('to_cpu','pinned_copy_blocking','pinned_copy_event')}
  for trial in range(30):
   original=torch.randint(0,32000,shape);gpu=original.cuda();torch.cuda.synchronize()
   methods=list(samples);methods=methods[trial%3:]+methods[:trial%3]
   for name in methods:
    t=time.perf_counter_ns()
    if name=='to_cpu':out=gpu.to('cpu')
    elif name=='pinned_copy_blocking':host.copy_(gpu);out=host
    else:
     host.copy_(gpu,non_blocking=True);event.record();event.synchronize();out=host
    ms=(time.perf_counter_ns()-t)/1e6
    torch.testing.assert_close(out,original,rtol=0,atol=0)
    if trial>=10:samples[name].append(ms)
  results.append(dict(shape=shape,samples_ms=samples,mean_ms={k:sum(v)/len(v) for k,v in samples.items()}))
 payload=dict(campaign_id='offload-gap-001',purpose=__doc__,source=source_state(),helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),command=sys.argv,environment=environment_details(),warmup=10,trials=20,exactness='All outputs exact with changing IDs',results=results)
 a.output.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items() if k!='samples_ms'} for r in results]))
