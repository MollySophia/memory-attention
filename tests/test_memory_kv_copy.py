"""Paired append must write exact bytes only at the requested cache position."""
import pytest
import torch
from fla.ops.utils.memory_cache import append_kv_pair

@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16,torch.float32])
@pytest.mark.parametrize('batch,width,offset',[(2,17,0),(3,65,127),(8,2048,2048)])
@torch.inference_mode()
def test_strided_pair_copy_and_sentinels(dtype,batch,width,offset):
    if not torch.cuda.is_available():pytest.skip('requires CUDA')
    torch.manual_seed(238)
    # Different source and destination strides for K and V.
    k=torch.randn(batch,1,width*2,device='cuda',dtype=dtype)[...,::2]
    v=torch.randn(1,batch,width,device='cuda',dtype=dtype).transpose(0,1)
    bk=torch.full((batch,offset+3,width*2),37,device='cuda',dtype=dtype)[...,::2]
    bv=torch.full((offset+3,batch,width),-19,device='cuda',dtype=dtype).transpose(0,1)
    expected_k,expected_v=bk.clone(),bv.clone()
    expected_k[:,offset:offset+1].copy_(k);expected_v[:,offset:offset+1].copy_(v)
    append_kv_pair((bk,bv),(k,v),offset)
    assert torch.equal(bk,expected_k)
    assert torch.equal(bv,expected_v)

@pytest.mark.parametrize('shape',[(1,1,17),(3,1,1)])
@torch.inference_mode()
def test_broadcast_input_keeps_torch_copy_fallback(shape):
    if not torch.cuda.is_available():pytest.skip('requires CUDA')
    from fla.models.utils import FLALayer
    layer=FLALayer()
    initial=tuple(torch.randn(3,5,17,device='cuda',dtype=torch.bfloat16) for _ in range(2))
    layer.update(attn_state=initial,cache_kwargs={'memory_append':True})
    incoming=tuple(torch.randn(*shape,device='cuda',dtype=torch.bfloat16) for _ in range(2))
    got=layer.update(attn_state=incoming,cache_kwargs={'memory_append':True})['attn_state']
    for out,old,x in zip(got,initial,incoming):
        want=torch.cat((old,x.expand(3,1,17)),dim=1)
        assert torch.equal(out,want)

@torch.inference_mode()
def test_copy_preserves_bfloat16_special_value_bits():
    if not torch.cuda.is_available():pytest.skip('requires CUDA')
    bits=torch.tensor([0,-32768,32640,-128,32704,1,-1],device='cuda',dtype=torch.int16)
    values=bits.view(torch.bfloat16).view(1,1,-1).expand(2,1,-1)
    buffers=tuple(torch.zeros((2,4,7),device='cuda',dtype=torch.bfloat16) for _ in range(2))
    append_kv_pair(buffers,(values,values),2)
    for buf in buffers:assert torch.equal(buf[:,2:3].view(torch.int16),values.view(torch.int16))
