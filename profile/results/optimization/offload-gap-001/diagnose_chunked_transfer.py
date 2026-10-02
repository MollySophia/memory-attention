"""Overlap chunked CPU gather and H2D; component timing, not model performance."""
import argparse,hashlib,json,sys,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path[:0]=[str(a.source_root),str(a.source_root/'profile')]
import torch
from benchmark_telemetry import source_state,environment_details
with torch.inference_mode():
 torch.manual_seed(1234);table=torch.randn(32000,24,2048,dtype=torch.bfloat16);results=[]
 for tokens in (2048,16384,32768):
  host=torch.empty(tokens,1,2048,dtype=table.dtype,pin_memory=True);gpu=torch.empty_like(host,device='cuda');samples={k:[] for k in ('whole','chunk1024','chunk512')}
  for trial in range(20):
   ids=torch.randint(0,32000,(tokens,));layer=trial%24;source=table[:,layer:layer+1];expected=source.index_select(0,ids)
   methods=list(samples);methods=methods[trial%3:]+methods[:trial%3]
   for name in methods:
    torch.cuda.synchronize();start=time.perf_counter_ns()
    if name=='whole':
     torch.index_select(source,0,ids,out=host);gpu.copy_(host,non_blocking=True)
    else:
     chunk=1024 if name=='chunk1024' else 512
     for offset in range(0,tokens,chunk):
      h=host[offset:offset+chunk];g=gpu[offset:offset+chunk]
      torch.index_select(source,0,ids[offset:offset+chunk],out=h);g.copy_(h,non_blocking=True)
    torch.cuda.synchronize();ms=(time.perf_counter_ns()-start)/1e6
    torch.testing.assert_close(gpu.cpu(),expected,rtol=0,atol=0)
    if trial>=10:samples[name].append(ms)
  results.append(dict(tokens=tokens,samples_ms=samples,mean_ms={k:sum(v)/len(v) for k,v in samples.items()}))
 payload=dict(campaign_id='offload-gap-001',purpose=__doc__,source=source_state(),helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),command=sys.argv,environment=environment_details(),warmup=10,trials=10,seed=1234,exactness='All fresh-ID outputs exact',results=results,limitations='One layer only. CPU IDs available before timing for all methods; table ordinary CPU, gathered output pinned. CUDA synchronized around each component trial; does not measure overlap with model compute.')
 a.output.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items() if k!='samples_ms'} for r in results]))
