"""CPU-only offload dispatch diagnostic with stub layers; no model speedup claim."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import timeit

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();root=a.source.resolve();sys.path.insert(0,str(root))
import torch
from fla.models.memory.modeling_memory import MemoryModel
assert Path(sys.modules[MemoryModel.__module__].__file__).resolve()==root/'fla/models/memory/modeling_memory.py'
torch.set_num_threads(16)

class Block(torch.nn.Module):
    def __init__(self,increment=1):
        super().__init__();self.increment=increment
    def forward(self,hidden,**kwargs):
        past=kwargs['past_key_values']
        return hidden,(past or 0)+self.increment

class Offloader:
    def forward(self,ids,consume):
        for i in range(24):consume(i,None)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embeddings=torch.nn.Embedding(16,8)
        self.layers=torch.nn.ModuleList([Block() for _ in range(24)])
        self.norm=torch.nn.Identity();self.offloader=Offloader()
    def _offloader_for(self,batch,length):
        return self.offloader

model=Model();hidden=torch.randn(1,1,8);ids=torch.zeros((1,1),dtype=torch.long)

def baseline(output_hidden=False,use_cache=True):
    # Same entry discovery as MemoryModel.forward, then its actual offload loop.
    device=next(model.parameters()).device
    offloader=model._offloader_for(*ids.shape)
    assert device.type=='cpu' and offloader is model.offloader
    return MemoryModel._forward_offloaded(model,hidden,ids,None,0,use_cache,output_hidden,False)

def resolved_metadata(output_hidden=False,use_cache=True):
    # Candidate hypothesis only in this stub diagnostic, not execution source.
    device=model.embeddings.weight.device
    offloader=model._offloader_for(*ids.shape)
    assert device.type=='cpu' and offloader is model.offloader
    layers=tuple(model.layers)
    state_hidden=hidden;past=0
    all_hidden=() if output_hidden else None
    def consume(index,memory_table):
        nonlocal state_hidden,past,all_hidden
        if output_hidden:all_hidden+=(state_hidden,)
        outputs=layers[index](state_hidden,attention_mask=None,past_key_values=past,
                              output_attentions=False,use_cache=use_cache,memory_table=memory_table)
        state_hidden=outputs[0]
        if use_cache:past=outputs[1]
    offloader.forward(ids,consume)
    state_hidden=model.norm(state_hidden)
    if output_hidden:all_hidden+=(state_hidden,)
    return tuple(v for v in (state_hidden,past,all_hidden) if v is not None)

def equal(left,right):
    if isinstance(left,torch.Tensor):return torch.equal(left,right)
    if isinstance(left,tuple):return len(left)==len(right) and all(equal(x,y) for x,y in zip(left,right))
    return left==right

with torch.inference_mode():
    for collect in (False,True):
        for cache in (False,True):assert equal(baseline(collect,cache),resolved_metadata(collect,cache))
    model.layers[23]=Block(increment=3)
    assert equal(baseline(True),resolved_metadata(True)) and resolved_metadata()[1]==26
    rows=[]
    for block in range(7):
        methods=[('actual_indexed_dict_loop',baseline),('resolved_per_call_metadata',resolved_metadata)]
        if block%2:methods.reverse()
        for name,fn in methods:
            elapsed=timeit.timeit(fn,number=10000)
            rows.append(dict(block=block+1,method=name,iterations=10000,us_per_call=elapsed*100))
result=dict(status='cpu_dispatch_diagnostic_only',source=str(root),
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            modeling_sha256=hashlib.sha256((root/'fla/models/memory/modeling_memory.py').read_bytes()).hexdigest(),
            helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),torch=torch.__version__,rows=rows,
            exactness='Stub output/cache/hidden-state results match; replacing a layer between forwards is visible.',
            limitations=['Stub layers and offloader; no table lookup, CUDA work or whole-model performance evidence.',
                         'Measures combined per-call metadata discovery and callback bookkeeping hypothesis only.',
                         'Layer references are rebuilt every forward; no cross-call output/token cache.',
                         'Actual frozen offload loop is the comparator; production source is unchanged.'])
a.output.write_text(json.dumps(result,indent=2)+'\n')
for method in ('actual_indexed_dict_loop','resolved_per_call_metadata'):
    print(method,statistics.mean(r['us_per_call'] for r in rows if r['method']==method))
