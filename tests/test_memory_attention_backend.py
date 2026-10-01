"""Predeclared changed-reduction gate against independent A0002 attention."""
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
def test_decode_backend_128_steps(policy,batch,kv_heads,padded):
    if not torch.cuda.is_available():pytest.skip('requires CUDA')
    gate=load(ROOT/'profile/test_memory_offload.py')
    ref=load(ROOT/'profile/results/optimization/A0003/reference.py')
    numerics=load(ROOT/'profile/results/optimization/A0006/numerics.py')
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
                    # The frozen rotary kernel does not write negative-position
                    # padded K entries. Compare every defined K and all V, plus
                    # all logits/hidden states (including padded hidden states).
                    valid = mask.bool() if mask is not None else None
                    values.append(dict(logits=out.logits.clone(),
                        **{f'hidden_{j}':h.clone() for j,h in enumerate(out.hidden_states)},
                        **{f'cache_{j}_{n}':(t[valid] if valid is not None and n == 0 else t).clone()
                           for j,state in enumerate(cache) for n,t in enumerate(state['attn_state'])}))
                torch.cuda.synchronize()
                return values
            with ref.frozen_attention():expected=capture()
            actual=capture()
            rows=[]
            for step,(got,want) in enumerate(zip(actual,expected)):
                for name,x in got.items():
                    rows.append(dict(step=step,**numerics.compare(x,want[name],name,exact=step==0 or padded)))
            logits=[r for r in rows if r['name']=='logits']
            agreement=sum(r['argmax_matches'] for r in logits)/sum(r['argmax_count'] for r in logits)
            import json
            out=ROOT/f'profile/results/optimization/A0006/numerics-{policy}-b{batch}-kv{kv_heads}-pad{padded}.json'
            out.write_text(json.dumps(dict(contract=numerics.CONTRACT,rows=rows,argmax_agreement=agreement),indent=2)+'\n')
            assert agreement>=.99,agreement
        finally:model.close_memory_offload()

@torch.inference_mode()
def test_decode_backend_reads_cache_without_second_update():
    if not torch.cuda.is_available():pytest.skip('requires CUDA')
    from unittest.mock import patch
    import fla.layers.memory_attn as attention
    gate=load(ROOT/'profile/test_memory_offload.py')
    model,_=gate.build(layers=3,hidden=128,heads=2,vocab=128)
    original=attention.flash_attn_with_kvcache;calls=[]
    def checked(*args,**kwargs):
        assert len(args)==3 and 'k' not in kwargs and 'v' not in kwargs
        k,v=args[1],args[2];before_k,before_v=k.clone(),v.clone()
        out=original(*args,**kwargs)
        assert torch.equal(k,before_k) and torch.equal(v,before_v)
        calls.append(k.shape[1]);return out
    try:
        model.fold_memory_table_on_gpu()
        prefix=torch.randint(0,128,(2,4),device='cuda');token=prefix[:,:1]
        with patch.object(attention,'flash_attn_with_kvcache',checked):
            cache=model(input_ids=prefix,use_cache=True).past_key_values
            assert not calls
            model(input_ids=token,past_key_values=cache,use_cache=True)
            assert calls==[5,5,5]
            model(input_ids=token,attention_mask=torch.ones(2,6,device='cuda',dtype=torch.long),past_key_values=cache,use_cache=True)
            assert calls==[5,5,5]
    finally:model.close_memory_offload()

@pytest.mark.parametrize('batch,hq,hk,length',[(1,2,2,2048),(8,32,32,2049),(8,32,8,8192)])
@torch.inference_mode()
def test_long_context_backend_numerics_and_read_only(batch,hq,hk,length):
    if not torch.cuda.is_available():pytest.skip('requires CUDA')
    from flash_attn import flash_attn_func,flash_attn_with_kvcache
    numerics=load(ROOT/'profile/results/optimization/A0006/numerics.py')
    torch.manual_seed(532)
    q=torch.randn(batch,1,hq,64,device='cuda',dtype=torch.bfloat16)
    # Logical view backed by larger capacity exercises the production layout.
    k=torch.randn(batch,length+127,hk,64,device='cuda',dtype=torch.bfloat16)[:,:length]
    v=torch.randn_like(k);saved_k=k.clone();saved_v=v.clone()
    expected=flash_attn_func(q,k,v,causal=True)
    actual=flash_attn_with_kvcache(q,k,v,causal=True,num_splits=0)
    assert torch.equal(k,saved_k) and torch.equal(v,saved_v)
    report=numerics.compare(actual,expected,'attention')
    import json
    path=ROOT/f'profile/results/optimization/A0006/long-context-b{batch}-hq{hq}-hk{hk}-l{length}.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
