"""Memory-only paired decode RoPE must equal the generic separate calls."""
from unittest.mock import patch
import pytest
import torch
from fla.modules.rotary import RotaryEmbedding

@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16,torch.float32])
@pytest.mark.parametrize('batch,hq,hk,dim,offset',[(1,2,2,64,0),(2,4,1,64,127),(8,32,32,64,2048)])
@torch.inference_mode()
def test_pair_exact(dtype,batch,hq,hk,dim,offset):
    if not torch.cuda.is_available():pytest.skip('requires CUDA')
    torch.manual_seed(527)
    rotary=RotaryEmbedding(dim=dim).cuda()
    q=torch.randn(batch,1,hq,dim,device='cuda',dtype=dtype)
    k=torch.randn(batch,1,hk,dim,device='cuda',dtype=dtype)
    expected=rotary(q,k,seqlen_offset=offset,max_seqlen=offset+1)
    actual=rotary(q,k,seqlen_offset=offset,max_seqlen=offset+1,memory_pair=True)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)

@pytest.mark.parametrize('kind',['prefill','tensor_offset','noncontiguous','interleaved','xpos','partial_rotary','grad'])
def test_unsupported_pair_uses_original(kind):
    if not torch.cuda.is_available():pytest.skip('requires CUDA')
    context=torch.enable_grad() if kind=='grad' else torch.inference_mode()
    with context:
        rotary=RotaryEmbedding(dim=32 if kind=='partial_rotary' else 64,
                               interleaved=kind=='interleaved',scale_base=512 if kind=='xpos' else None).cuda()
        length=4 if kind=='prefill' else 1
        q=torch.randn(2,length,2,128 if kind=='noncontiguous' else 64,device='cuda',dtype=torch.bfloat16)
        if kind=='noncontiguous':q=q[...,::2]
        if kind=='grad':q.requires_grad_()
        k=q.clone()
        offset=torch.tensor([0,1],device='cuda') if kind=='tensor_offset' else 0
        expected=rotary(q,k,seqlen_offset=offset,max_seqlen=length+1)
        with patch('fla.modules.rotary._memory_rotary_pair',side_effect=AssertionError('must use generic path')):
            actual=rotary(q,k,seqlen_offset=offset,max_seqlen=length+1,memory_pair=True)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
        if kind=='grad':sum(x.float().sum() for x in actual).backward();assert q.grad is not None
