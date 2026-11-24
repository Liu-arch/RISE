import torch
from torch.nn import functional as F
def pool_historical(x,slots):
 n=x.shape[0];assert n>0
 if n<slots:return x[torch.linspace(0,n-1,slots,device=x.device).round().long()]
 edges=torch.linspace(0,n,slots+1,device=x.device).floor().long()
 return torch.stack([x[int(edges[i]):max(int(edges[i])+1,int(edges[i+1]))].mean(0) for i in range(slots)])

def hidden_distance(s,t):return (1-F.cosine_similarity(s.float(),t.detach().float(),dim=-1)).mean()+.1*F.smooth_l1_loss(s.float(),t.detach().float())

def token_kl(s,t,w,temp=2.):return ((F.kl_div(F.log_softmax(s.float()/temp,-1),F.softmax(t.detach().float()/temp,-1),reduction='none').sum(-1)*temp**2)*w).sum()/w.sum()
