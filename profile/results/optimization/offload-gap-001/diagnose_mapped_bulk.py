"""Mapped-host reads of all layers for tiny bulk inputs; standalone diagnostic."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import torch
import triton
from diagnose_transfer_alternatives import mapped_rows
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path[:0]=[str(a.source_root),str(a.source_root/'profile')]
from benchmark_telemetry import environment_details,source_state
with torch.inference_mode():
 torch.manual_seed(1234);table=torch.empty(32000,24,2048,dtype=torch.bfloat16,pin_memory=True).normal_();results=[]
 for batch in (1,8,16):
  host=torch.empty(batch,24,2048,dtype=table.dtype,pin_memory=True);gpu=torch.empty_like(host,device='cuda')
  samples={name:[] for name in ('d2h_gather_h2d','mapped_bulk')}
  for trial in range(30):
   ids=torch.randint(0,32000,(batch,),device='cuda');expected=table.index_select(0,ids.cpu())
   for name in (list(samples) if trial%2==0 else list(reversed(samples))):
    torch.cuda.synchronize();t=time.perf_counter_ns()
    if name=='d2h_gather_h2d':
     torch.index_select(table,0,ids.cpu(),out=host);gpu.copy_(host,non_blocking=True)
    else:mapped_rows[(batch,triton.cdiv(24*2048,512))](table,ids,gpu,batch,24*2048,1,32000,512,num_warps=4)
    torch.cuda.synchronize();ms=(time.perf_counter_ns()-t)/1e6
    torch.testing.assert_close(gpu.cpu(),expected,rtol=0,atol=0)
    if trial>=10:samples[name].append(ms)
  results.append(dict(batch=batch,samples_ms=samples,mean_ms={k:sum(v)/len(v) for k,v in samples.items()}))
 payload=dict(campaign_id='offload-gap-001',purpose=__doc__,source=source_state(),helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),kernel_helper_sha256=hashlib.sha256(Path(__file__).with_name('diagnose_transfer_alternatives.py').read_bytes()).hexdigest(),command=sys.argv,environment=environment_details(),warmup=10,trials=20,exactness='All changing-ID outputs exact',pinned_table_bytes=table.numel()*table.element_size(),results=results,limitations='All24-layer bulk lookup including D2H for baseline; no model compute overlap or accepted-speedup claim. Whole table pinned for mapped addressability.')
 a.output.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items() if k!='samples_ms'} for r in results]))
