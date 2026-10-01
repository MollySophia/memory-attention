"""Exact asynchronous scheduling gate against a frozen, independent forward."""
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]

def load(path):
    spec=importlib.util.spec_from_file_location(path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

@pytest.mark.parametrize('policy',['bulk','pipeline'])
@pytest.mark.parametrize('batch,kv_heads,padded',[(1,2,False),(2,2,True),(1,1,True),(2,1,False)])
def test_deferred_acquire_exact_128_steps(policy,batch,kv_heads,padded):
    if not torch.cuda.is_available():pytest.skip('requires CUDA')
    gate=load(ROOT/'profile/test_memory_offload.py')
    ref=load(ROOT/'profile/results/optimization/A0003/reference.py')
    with torch.inference_mode():
        model,_=gate.build(seed=781+batch+kv_heads,layers=3,hidden=128,heads=2,vocab=128,kv_heads=kv_heads)
        try:
            for layer in model.model.layers:
                layer.attn.m_norm.weight.copy_(torch.linspace(.5,1.5,64,device='cuda',dtype=torch.bfloat16))
            model.config.memory_offload_policy=policy
            model.config.memory_offload_group_size=2
            model.config.memory_offload_prefetch_depth=1
            model.enable_memory_offload()
            prefix=torch.randint(0,128,(batch,16),device='cuda')
            tokens=torch.randint(0,128,(128,batch,1),device='cuda')
            def capture():
                cache=None;values=[]
                for i,ids in enumerate([prefix,*tokens.unbind(0)]):
                    mask=None
                    if padded:
                        mask=torch.ones((batch,16+i),device='cuda',dtype=torch.long);mask[0,:3]=0
                    out=model(input_ids=ids,attention_mask=mask,past_key_values=cache,
                              use_cache=True,logits_to_keep=1,output_hidden_states=True)
                    cache=out.past_key_values
                    values.append([out.logits.clone(),*[h.clone() for h in out.hidden_states],
                                   *[t.clone() for state in cache for t in state['attn_state']]])
                torch.cuda.synchronize()
                return values
            with ref.frozen_attention():expected=capture()
            actual=capture()
            for got,want in zip(actual,expected):
                for x,y in zip(got,want):torch.testing.assert_close(x,y,rtol=0,atol=0)
        finally:model.close_memory_offload()
