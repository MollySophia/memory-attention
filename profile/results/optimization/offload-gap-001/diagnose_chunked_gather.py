"""CPU gather chunk sizes: fresh IDs, exactness, no model-speedup claim."""
import argparse,hashlib,json,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path[:0]=[str(a.source_root),str(a.source_root/'profile')]
import torch
from benchmark_telemetry import environment_details,source_state
with torch.inference_mode():
 torch.manual_seed(1234);weights=torch.randn(32000,24,2048,dtype=torch.bfloat16)
 results=[]
 for tokens in (2048,16384,32768):
  host=torch.empty(tokens,1,2048,dtype=torch.bfloat16,pin_memory=True)
  methods=[tokens,128,512,1024];methods=list(dict.fromkeys(methods));samples={str(k):[] for k in methods}
  for trial in range(20):
   ids=torch.randint(0,32000,(tokens,));layer=trial%24;source=weights[:,layer:layer+1]
   expected=source.index_select(0,ids)
   order=methods[trial%len(methods):]+methods[:trial%len(methods)]
   for size in order:
    t=time.perf_counter_ns()
    if size==tokens:torch.index_select(source,0,ids,out=host)
    else:
     for start in range(0,tokens,size):torch.index_select(source,0,ids[start:start+size],out=host[start:start+size])
    ms=(time.perf_counter_ns()-t)/1e6
    torch.testing.assert_close(host,expected,rtol=0,atol=0)
    if trial>=10:samples[str(size)].append(ms)
  results.append(dict(tokens=tokens,samples_ms=samples,mean_ms={k:sum(v)/len(v) for k,v in samples.items()}))
 payload=dict(campaign_id='offload-gap-001',purpose=__doc__,source=source_state(),helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),command=sys.argv,environment=environment_details(),warmup=10,trials=10,exactness='All fresh-ID outputs exact',results=results)
 a.output.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items() if k!='samples_ms'} for r in results]))
