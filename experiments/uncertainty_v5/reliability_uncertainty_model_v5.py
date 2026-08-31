from __future__ import annotations
import torch
import torch.nn as nn
try:
    from mamba_ssm import Mamba
    HAS_MAMBA=True
except Exception:
    Mamba=None; HAS_MAMBA=False

class ReliabilityUncertaintyV5(nn.Module):
    def __init__(self,input_dim,hidden_dim=96,layers=2,dropout=.1,horizon=3):
        super().__init__(); self.horizon=horizon; self.backend="mamba" if HAS_MAMBA else "gru"
        self.proj=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.LayerNorm(hidden_dim),nn.SiLU())
        if HAS_MAMBA:
            self.blocks=nn.ModuleList([Mamba(d_model=hidden_dim,d_state=16,d_conv=4,expand=2) for _ in range(layers)])
            self.norms=nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)]); self.gru=None
        else:
            self.blocks=None; self.norms=None
            self.gru=nn.GRU(hidden_dim,hidden_dim,layers,batch_first=True,dropout=dropout if layers>1 else 0)
        self.head=nn.Linear(hidden_dim,2*horizon)
    def forward(self,x):
        h=self.proj(x)
        if self.gru is not None: h,_=self.gru(h)
        else:
            for b,n in zip(self.blocks,self.norms): h=n(h+b(h))
        raw=self.head(h[:,-1]).view(-1,self.horizon,2)
        ev=torch.nn.functional.softplus(raw)
        a,b=ev[...,0]+1,ev[...,1]+1
        r=a/(a+b); u=torch.clamp(2/(a+b),0,1)
        return r,u,a,b

def evidential_loss(y,r,a,b,epoch,anneal_epochs=20):
    y=torch.clamp(y,1e-5,1-1e-5)
    logp=(a-1)*torch.log(y)+(b-1)*torch.log(1-y)-torch.lgamma(a)-torch.lgamma(b)+torch.lgamma(a+b)
    nll=-logp.mean()
    reg=(torch.abs(y-r).detach()*(a+b-2)).mean()
    return nll+0.01*min(1,(epoch+1)/anneal_epochs)*reg
