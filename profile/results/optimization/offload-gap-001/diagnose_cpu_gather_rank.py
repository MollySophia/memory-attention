"""Fresh-ID CPU gather rank diagnostic; not an end-to-end speedup claim."""
import argparse, hashlib, json, sys, time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
sys.path[:0]=[str(a.source_root),str(a.source_root/'profile')]
import torch
from benchmark_telemetry import environment_details, source_state
with torch.inference_mode():
 torch.manual_seed(1234)
 base=torch.randn(32000,24,2048,dtype=torch.bfloat16)
 layouts={'rank3':base,'rank2':base}
 results=[]
 for tokens in (1,8,2048,16384,32768):
  for group in ((24,) if tokens<=8 else (1,)):
   output=torch.empty(tokens,group,2048,dtype=torch.bfloat16,pin_memory=True)
   records={name:[] for name in layouts}
   for trial in range(20):
    ids=torch.randint(0,32000,(tokens,))
    start=(trial*7)%(25-group)
    expected=base[:,start:start+group].index_select(0,ids)
    for name in (list(layouts) if trial%2==0 else list(reversed(layouts))):
     source=layouts[name][:,start:start+group]
     t=time.perf_counter_ns()
     if name=='rank2':torch.index_select(source.view(32000,group*2048),0,ids,out=output.view(tokens,group*2048))
     else:torch.index_select(source,0,ids,out=output)
     ms=(time.perf_counter_ns()-t)/1e6
     torch.testing.assert_close(output,expected,rtol=0,atol=0)
     if trial>=10:records[name].append(ms)
   results.append(dict(tokens=tokens,group=group,samples_ms=records,mean_ms={k:sum(v)/len(v) for k,v in records.items()}))
 payload=dict(campaign_id='offload-gap-001',purpose=__doc__,source=source_state(),helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),command=sys.argv,environment=environment_details(),warmup=10,trials=10,seed=1234,table_shape=list(base.shape),layouts={k:dict(stride=list(v.stride()),bytes=v.numel()*v.element_size()) for k,v in layouts.items()},exactness='All fresh IDs/layouts/trials bit-exact',results=results)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(payload,indent=2)+'\n')
 print(json.dumps(results))
