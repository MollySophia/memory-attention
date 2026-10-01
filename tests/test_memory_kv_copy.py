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
