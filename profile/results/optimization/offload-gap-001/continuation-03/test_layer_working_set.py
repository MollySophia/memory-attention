"""Small CPU semantic checks; never exercise the real diagnostic workload here."""
import torch
import diagnose_layer_working_set as d


def test_full_walk_visits_every_layer_and_fresh_rows():
    original=torch.arange(11*4*8).reshape(11,4,8).float()
    layered=original.permute(1,0,2).contiguous().permute(1,0,2)
    assert not layered.is_contiguous()
    ids=torch.tensor([2,7,2]);out=torch.empty(3,1,8)
    d.verify(torch,original,layered,ids)
    seen=[]
    class Operations:
        @staticmethod
        def index_select(source,dim,indices,*,out):
            seen.append(source[0,0,0].item())
            return torch.index_select(source,dim,indices,out=out)
    d.walk(Operations,original,ids,out)
    assert seen==[0.,8.,16.,24.]
    assert torch.equal(out,original[:,-1:].index_select(0,ids))
    original[7,1,3]=-123;layered[7,1,3]=-123
    ids=torch.tensor([7,1,0]);d.verify(torch,original,layered,ids)
    samples=d.measure(torch,original,layered,ids,out,1,3)
    assert all(len(v)==3 and all(x>0 for x in v) for v in samples.values())
