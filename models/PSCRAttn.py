# Autism spectrum disorder diagnosis using graph attention network based on spatial-constrained sparse functional brain networks
# attention-based PSCR adjacency + dense GAT for fMRI-based classification

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

def corrcoef_batch_time(x):
    B, T, N = x.shape
    xm = x - x.mean(dim=1, keepdim=True)
    cov = xm.transpose(1, 2) @ xm / (T - 1 + 1e-8)            
    var = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(1e-8)
    std = var.sqrt()
    denom = std.unsqueeze(-1) * std.unsqueeze(-2)
    R = cov / denom
    R = torch.nan_to_num(R, 0.0, 0.0, 0.0)
    R = 0.5 * (R + R.transpose(1,2))
    eye = torch.eye(N, device=x.device).unsqueeze(0)
    R = R * (1 - eye) + eye
    return R

def spatial_kernel(coords, sigma=50.0):
    N = coords.shape[0]
    P = coords.to(dtype=torch.float32)
    d2 = (P.unsqueeze(1) - P.unsqueeze(0)).pow(2).sum(-1)      
    K = torch.exp(-d2 / (2.0 * (sigma ** 2) + 1e-8))
    K.fill_diagonal_(1.0)
    K = 0.5 * (K + K.t())
    return K

def pscr_adjacency(x_enc, coords=None, sigma=50.0, top_ratio=0.2, use_abs=True):
    B, T, N = x_enc.shape
    R = corrcoef_batch_time(x_enc)                              
    if coords is None:
        K = torch.ones(N, N, device=x_enc.device, dtype=x_enc.dtype)
    else:
        K = spatial_kernel(coords, sigma=sigma).to(x_enc.device, x_enc.dtype)

    Score = (R.abs() if use_abs else R) * K       
    iu = torch.triu_indices(N, N, offset=1, device=x_enc.device)
    E = N * (N - 1) // 2
    k = max(1, int(math.ceil(top_ratio * E)))

    A = torch.zeros_like(R)
    for b in range(B):
        svec = Score[b, iu[0], iu[1]]                          
        topv, idx = torch.topk(svec, k, largest=True)
        mask = torch.zeros(E, device=x_enc.device, dtype=R.dtype)
        mask.scatter_(0, idx, 1.0)
        M = torch.zeros(N, N, device=x_enc.device, dtype=R.dtype)
        M[iu[0], iu[1]] = mask
        M = M + M.t()
        Ab = R[b] * M
        Ab.fill_diagonal_(1.0)
        A[b] = 0.5 * (Ab + Ab.t())
    return A

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1, concat=True, leaky_slope=0.2, add_self_loop=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.add_self_loop = add_self_loop

        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, out_dim))
        self.a_dst = nn.Parameter(torch.randn(heads, out_dim))
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

        self.leaky = nn.LeakyReLU(leaky_slope)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, A):
        B, N, _ = X.shape
        if self.add_self_loop:
            eye = torch.eye(N, device=A.device).unsqueeze(0)
            A = A * 0 + (A + eye)
        mask = (A > 0).unsqueeze(1)                             

        H = self.lin(X).view(B, N, self.heads, self.out_dim)    
        H = H.permute(0, 2, 1, 3)                               

        e_src = (H * self.a_src.view(1, self.heads, 1, self.out_dim)).sum(-1)  
        e_dst = (H * self.a_dst.view(1, self.heads, 1, self.out_dim)).sum(-1)  
        e = self.leaky(e_src.unsqueeze(-1) + e_dst.unsqueeze(-2))              

        e = e + torch.log(A.clamp_min(1e-8)).unsqueeze(1)                       

        e = e.masked_fill(~mask, float('-inf'))
        alpha = torch.softmax(e, dim=-1)                                        
        alpha = self.dropout(alpha)

        out = alpha @ H                                                        
        out = out.permute(0, 2, 1, 3)                                           
        if self.concat:
            out = out.reshape(B, N, self.heads * self.out_dim)                  
        else:
            out = out.mean(dim=2)                                               
        return out

class AttnReadout(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim * 2, feat_dim)
        self.fc2 = nn.Linear(feat_dim, 1)

    def forward(self, H):
        B, N, F = H.shape
        g = H.mean(dim=1, keepdim=True).expand(-1, N, -1)
        a = torch.tanh(self.fc1(torch.cat([H, g], dim=-1)))     
        a = self.fc2(a)                                         
        w = torch.softmax(a.squeeze(-1), dim=1).unsqueeze(-1)
        z = (w * H).sum(dim=1)                                
        return z, w
    
# ---------------------------
# PSCRAttn
# ---------------------------

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.N = configs.channel
        self.classes = configs.classes
        self.top_ratio = 0.4
        self.sigma = 50.0
        self.use_abs = False
        self.coords = None

        Fin = self.N
        hid = configs.d_model
        heads = 8
        dropout = configs.dropout
        layers = configs.layer

        self.backbone = nn.ModuleList()
        for i in range(layers):
            in_dim = Fin if i == 0 else hid * (heads if True else 1)
            self.backbone.append(GATLayer(
                in_dim=in_dim, out_dim=hid, heads=heads, dropout=dropout, concat=True
            ))

        self.readout = AttnReadout(hid * heads)
        self.out_dim = 1 if self.classes == 2 else self.classes
        self.cls = nn.Linear(hid * heads, self.out_dim)

    def forward(self, x_enc):
        B, T, N = x_enc.shape
        assert N == self.N, f"N mismatch: {N} vs {self.N}"
        A = pscr_adjacency(
            x_enc, coords=self.coords, sigma=self.sigma,
            top_ratio=self.top_ratio, use_abs=self.use_abs
        )                                                       
        X = A.clone()                                          

        H = X
        for gat in self.backbone:
            H = gat(H, A)                                       

        z, _ = self.readout(H)                                 
        out = self.cls(z)                                       
        if self.out_dim == 1:
            out = torch.sigmoid(out)
        return out
