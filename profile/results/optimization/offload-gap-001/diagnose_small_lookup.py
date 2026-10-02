"""Compare serial byte-copy lookup with torch bulk gather; diagnostic only."""
import json,sys,time,statistics
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, '/home/molly/workspace-memory-attn/offload-gap-001-A0000/profile')
from benchmark_telemetry import environment_details
records=[]
with torch.inference_mode():
 weights=torch.randn(32000,24,2048,dtype=torch.bfloat16)
 source=weights.view(torch.uint8).numpy().reshape(32000,-1)
 for tokens in (1,8):
  ids=torch.randint(0,32000,(tokens,));host=torch.empty((tokens,24,2048),dtype=weights.dtype,pin_memory=True)
  dest=host.view(torch.uint8).numpy().reshape(tokens,-1)
  def serial():
   for row,index in enumerate(ids.tolist()):
    if index<0 or index>=len(source): raise IndexError(index)
    np.copyto(dest[row],source[index])
  expected=weights.index_select(0,ids);serial();assert torch.equal(host,expected)
  samples={k:[] for k in ('torch','serial_bytes')};cpu={k:[] for k in samples}
  for i in range(110):
   order=[('torch',lambda:torch.index_select(weights,0,ids,out=host)),('serial_bytes',serial)]
   if i%2:order.reverse()
   for label,fn in order:
    start=time.perf_counter_ns();c=time.process_time_ns();fn();c=time.process_time_ns()-c;dt=time.perf_counter_ns()-start
    if i>=10:samples[label].append(dt/1e6);cpu[label].append(c/1e6)
  records.append(dict(tokens=tokens,exact=True,samples_ms=samples,process_cpu_ms=cpu,mean_ms={k:statistics.mean(v) for k,v in samples.items()}))
out=Path('profile/results/optimization/offload-gap-001/A0001/small-lookup-diagnostic.json')
out.write_text(json.dumps(dict(campaign_id='offload-gap-001',purpose='CPU-only diagnostic, not end-to-end performance or independent confirmation',command=sys.argv,environment=environment_details(),records=records,limitations='Hot repeated indices may understate model lookup costs. CPU totals include all process threads. No model change or optimization verdict.'),indent=2)+'\n')
for r in records:print(r['tokens'],r['mean_ms'])
