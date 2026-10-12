"""Diagnostic reconstruction latency, not an end-to-end acceptance benchmark."""
import json,sys,time,statistics
from pathlib import Path
sys.path.insert(0,'/home/molly/workspace-memory-attn/memory-attention-recall')
import torch
from fla.modules import RMSNorm
from fla.ops.memory.recall import memory_recall
out=[]
with torch.inference_mode():
 torch.manual_seed(1234)
 table=torch.randn(32000,2048,device='cuda',dtype=torch.bfloat16)
 norm=RMSNorm(64).to('cuda',dtype=torch.bfloat16)
 norm.weight.copy_(torch.linspace(.5,1.5,64,device='cuda'))
 for b,l in [(1,2048),(8,2048),(32,2048),(8,8192)]:
  ids=torch.randint(0,32000,(b,l),device='cuda');k=torch.randn(b,l,32,64,device='cuda',dtype=torch.bfloat16)
  def original():return k+norm(torch.nn.functional.embedding(ids,table).view(b,l,32,64))
  def fused():return memory_recall(k,ids,table,norm.weight,norm.eps)
  torch.testing.assert_close(fused(),original(),rtol=0,atol=0)
  m=torch.nn.functional.embedding(ids,table).view_as(k);normalized=norm(m)
  def measure(fn):
   for _ in range(10):fn()
   samples=[]
   for _ in range(30):
    torch.cuda.synchronize();start=time.perf_counter();fn();torch.cuda.synchronize();samples.append((time.perf_counter()-start)*1000)
   return statistics.mean(samples)
  row=dict(batch=b,length=l,lookup_ms=measure(lambda:torch.nn.functional.embedding(ids,table)),norm_ms=measure(lambda:norm(m)),add_ms=measure(lambda:k+normalized),original_ms=measure(original),fused_ms=measure(fused))
  out.append(row);print(json.dumps(row),flush=True)
  del k,ids,m,normalized
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
