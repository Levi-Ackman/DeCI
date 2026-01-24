# Learning Dynamic Graph Representation of Brain Connectome with Spatio-Temporal Attention
# STAGIN / Attention-based Spatio-Temporal Graph Isomorphism Network

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

def sliding_windows(x, win, step):
    B, T, N = x.shape
    idx = [(s, s+win) for s in range(0, T - win + 1, step)]
    return torch.stack([x[:, s:e, :] for (s, e) in idx], dim=1)

def corrcoef_batch_time(x):
    B, L, N = x.shape
    xm = x - x.mean(dim=1, keepdim=True)
    cov = xm.transpose(1, 2) @ xm / (L - 1 + 1e-8)             # [B,N,N]
    var = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(1e-8)
    std = var.sqrt()
    denom = std.unsqueeze(-1) * std.unsqueeze(-2)
    corr = cov / denom
    corr = torch.nan_to_num(corr, 0.0, 0.0, 0.0)
    eye = torch.eye(N, device=x.device).unsqueeze(0).expand(B, -1, -1)
    corr = corr * (1 - eye) + eye
    return corr

def sparsify_topk(A, topk):
    if topk is None or topk <= 0 or topk >= A.shape[-1]:
        return 0.5 * (A + A.transpose(1, 2))
    B, N, _ = A.shape
    vals, idx = torch.topk(A.abs(), k=topk, dim=-1)            
    mask = torch.zeros_like(A, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    A_top = A * mask
    A_sym = torch.max(A_top, A_top.transpose(1, 2))
    return A_sym

def build_dynamic_graphs(x, win=30, step=2, topk=20):
    xw = sliding_windows(x, win, step)                           
    B, Wn, L, N = xw.shape

    A_list, X_list = [], []
    for w in range(Wn):
        Aw = corrcoef_batch_time(xw[:, w])                       
        Aw = sparsify_topk(Aw, topk=topk)
        eye = torch.eye(N, device=x.device).unsqueeze(0).expand(B, -1, -1)
        Aw = Aw * (1 - eye) + eye
        mean = xw[:, w].mean(dim=1)                              
        stdv = xw[:, w].std(dim=1).clamp_min(1e-8)               
        deg  = Aw.sum(dim=-1) - 1.0                              
        Xw = torch.stack([mean, stdv, deg], dim=-1)              

        A_list.append(Aw)
        X_list.append(Xw)

    A_seq = torch.stack(A_list, dim=1)                           
    X_seq = torch.stack(X_list, dim=1)                           
    return A_seq, X_seq

class GINConv(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=None, eps_init=0.0, dropout=0.0):
        super().__init__()
        hidden = hidden or out_dim
        self.eps = nn.Parameter(torch.tensor(eps_init, dtype=torch.float32))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, A):
        agg = torch.matmul(A, h)                                  
        out = self.mlp((1.0 + self.eps) * h + agg)
        return self.dropout(out)

class NodeAttnReadout(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, in_dim)
        self.u = nn.Parameter(torch.randn(in_dim))

    def forward(self, H):
        S = torch.tanh(self.lin(H))                               
        scores = torch.matmul(S, self.u)                          
        alpha = F.softmax(scores, dim=1)                         
        g = torch.sum(alpha.unsqueeze(-1) * H, dim=1)          
        return g, alpha

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        S, B, F = x.shape
        return x + self.pe[:S, :F].unsqueeze(1)

class TemporalAttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.lin = nn.Linear(d_model, d_model)
        self.v = nn.Parameter(torch.randn(d_model))

    def forward(self, G_seq):
        S = torch.tanh(self.lin(G_seq))                         
        scores = torch.matmul(S, self.v)                        
        beta = F.softmax(scores, dim=1)                           
        z = torch.sum(beta.unsqueeze(-1) * G_seq, dim=1)         
        return z, beta

# ---------------------------
# STAGIN
# ---------------------------

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.N = configs.channel
        self.win = 30
        self.step = 2
        self.topk = 20
        self.use_norm = configs.use_norm

        in_node = 3  
        self.node_norm = nn.LayerNorm(in_node) if self.use_norm else nn.Identity()

        self.gin_layers = nn.ModuleList()
        for i in range(configs.layer):
            fin = in_node if i == 0 else configs.d_model
            self.gin_layers.append(GINConv(fin, configs.d_model, dropout=configs.dropout))

        self.readout = NodeAttnReadout(configs.d_model)
        self.posenc = PositionalEncoding(configs.d_model, max_len=2000)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=configs.d_model,
            nhead=8,
            dim_feedforward= 2 * configs.d_model,
            dropout=configs.dropout,
            batch_first=False, 
            activation='gelu'
        )
        self.temporal = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )
        self.temporal_pool = TemporalAttentionPool(configs.d_model)

        self.out_dim = 1 if configs.classes == 2 else configs.classes
        self.classifier = nn.Linear(configs.d_model, self.out_dim)

    def forward(self, x_enc):
        B, T, N = x_enc.shape
        assert N == self.N, f"N mismatch: got {N}, expected {self.N}"

        A_seq, X_seq = build_dynamic_graphs(x_enc, self.win, self.step, self.topk)  
        B, S, N, _ = A_seq.shape
        X_seq = self.node_norm(X_seq)
        G_list, spatial_attn = [], []
        for s in range(S):
            H = X_seq[:, s]                                      
            A = A_seq[:, s]                                     
            for gin in self.gin_layers:
                H = gin(H, A)                                   
            g, alpha = self.readout(H)                           
            G_list.append(g)
            spatial_attn.append(alpha)

        G_seq = torch.stack(G_list, dim=1)                       
        spatial_attn = torch.stack(spatial_attn, dim=1) 
        G_seq_t = G_seq.transpose(0, 1)                          
        G_seq_t = self.posenc(G_seq_t)
        Z_seq = self.temporal(G_seq_t).transpose(0, 1)  
        z, beta = self.temporal_pool(Z_seq)         
        out = self.classifier(z)
        if self.out_dim == 1:
            out = torch.sigmoid(out)
        return out
