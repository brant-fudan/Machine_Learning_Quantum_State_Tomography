"""Direct predictors. GRU reuses the recurrent-inductive-bias of legacy rnn.py.

Not a drop-in import of the old POVM generator: its autoregressive likelihood
models p(outcomes), rather than the supervised target N(rho).
"""
from __future__ import annotations
import torch
from torch import nn

class PropertyNet(nn.Module):
    def __init__(self, kind: str='mlp', hidden: int=48):
        super().__init__()
        self.kind=kind
        if kind=='mlp':
            self.encoder=nn.Sequential(nn.Linear(45,hidden),nn.ReLU(),
                                       nn.Linear(hidden,hidden),nn.ReLU())
        elif kind=='gru':
            self.embed=nn.Linear(18,hidden)
            self.gru=nn.GRU(hidden,hidden,num_layers=1,batch_first=True)
        else:
            raise ValueError(kind)
        self.head=nn.Linear(hidden,2)

    def forward(self, features):
        if self.kind=='mlp':
            h=self.encoder(features)
        else:
            # Interleave (mu,mask,shot noise, observable id) along Pauli tokens.
            x=features.reshape(-1,3,15).transpose(1,2)
            identities=torch.eye(15,device=x.device).expand(len(x),-1,-1)
            token=torch.cat((x,identities),dim=-1)
            h=self.gru(torch.relu(self.embed(token)))[0][:,-1]
        o=self.head(h)
        return .5*torch.sigmoid(o[:,0]),o[:,1]


def fit_network(features, targets, *, kind='mlp', seed=42, epochs=30,
                batch_size=128, lr=.002):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model=PropertyNet(kind)
    rng=torch.Generator().manual_seed(seed)
    x=torch.as_tensor(features,dtype=torch.float32)
    y=torch.as_tensor(targets,dtype=torch.float32)
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    bce=nn.BCEWithLogitsLoss()
    model.train()
    history=[]
    for ep in range(epochs):
        perm=torch.randperm(len(x),generator=rng)
        losses=[]
        for indices in perm.split(batch_size):
            pred, logits=model(x[indices]); labels=(y[indices]>1e-8).float()
            # Binary decision and regression have distinct objectives.
            loss=(pred-y[indices]).square().mean()*15.+.35*bce(logits,labels)
            opt.zero_grad();loss.backward();opt.step()
            losses.append(loss.detach().item())
        history.append(float(sum(losses)/len(losses)))
    model.eval()
    return model,history

@torch.no_grad()
def predict(model, features):
    x=torch.as_tensor(features,dtype=torch.float32)
    a=[];b=[]
    for batch in x.split(512):
        y,z=model(batch)
        a.append(y.cpu().numpy());b.append(torch.sigmoid(z).cpu().numpy())
    import numpy as np
    return np.concatenate(a),np.concatenate(b)
