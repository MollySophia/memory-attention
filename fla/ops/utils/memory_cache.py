"""Exact paired single-token writes into cache-owned inference buffers."""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['offset'])
def _append_kv(K, V, BK, BV, offset,
               N: tl.constexpr, D: tl.constexpr,
               KS0: tl.constexpr, KS2: tl.constexpr, VS0: tl.constexpr, VS2: tl.constexpr,
               BKS0: tl.constexpr, BKS1: tl.constexpr, BKS2: tl.constexpr,
               BVS0: tl.constexpr, BVS1: tl.constexpr, BVS2: tl.constexpr,
               BLOCK: tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    b,d=i//D,i%D
    k=tl.load(K+b*KS0+d*KS2,i<N)
    v=tl.load(V+b*VS0+d*VS2,i<N)
    tl.store(BK+b*BKS0+offset*BKS1+d*BKS2,k,i<N)
    tl.store(BV+b*BVS0+offset*BVS1+d*BVS2,v,i<N)


def append_kv_pair(buffers, incoming, offset):
    k,v=incoming;bk,bv=buffers
    with torch.cuda.device(k.device):
        _append_kv[(triton.cdiv(k.numel(),256),)](
            k,v,bk,bv,offset,k.numel(),k.shape[2],
            k.stride(0),k.stride(2),v.stride(0),v.stride(2),
            *bk.stride(),*bv.stride(),BLOCK=256,num_warps=4)
