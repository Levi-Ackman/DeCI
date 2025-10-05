# Multi-scale dynamic graph learning for brain disorder detection with functional MRI.
# MDGLMultiScale / MDGL  

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
    cov = xm.transpose(1, 2) @ xm / (L - 1 + 1e-8)              # [B,N,N]
    var = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(1e-8) # [B,N]
    std = var.sqrt()
    corr = cov / (std.unsqueeze(-1) * std.unsqueeze(-2))
    corr = torch.nan_to_num(corr, 0.0, 0.0, 0.0)
    eye = torch.eye(N, device=x.device).unsqueeze(0).expand(B, -1, -1)
    return corr * (1 - eye) + eye

def sparsify_topk(A, topk):
    if topk is None or topk <= 0 or topk >= A.shape[-1]:
        A_sym = 0.5 * (A + A.transpose(1, 2))
    else:
        B, N, _ = A.shape
        vals, idx = torch.topk(A.abs(), k=topk, dim=-1)
        mask = torch.zeros_like(A, dtype=torch.bool)
        mask.scatter_(-1, idx, True)
        A_row = A * mask
        A_sym = torch.max(A_row, A_row.transpose(1, 2))
    eye = torch.eye(A.shape[-1], device=A.device).unsqueeze(0).expand(A.shape[0], -1, -1)
    return A_sym * (1 - eye) + eye

def binarize_top_percent(A, keep_ratio=0.3):
    B, N, _ = A.shape
    iu = torch.triu_indices(N, N, offset=1, device=A.device)
    Au = A[:, iu[0], iu[1]].abs()                                
    k = max(1, int(round(Au.shape[1] * keep_ratio)))
    vals, _ = torch.topk(Au, k=k, dim=1)
    thr = vals[:, -1]                                            
    mask_u = (Au >= thr.unsqueeze(1))
    A_bin = torch.zeros_like(A)
    A_bin[:, iu[0], iu[1]] = mask_u.float()
    A_bin[:, iu[1], iu[0]] = mask_u.float()
    eye = torch.eye(N, device=A.device).unsqueeze(0).expand(B, -1, -1)
    return A_bin * (1 - eye) + eye

def build_dynamic_graphs(x, win, step, topk=None, keep_ratio=0.3):
    xw = sliding_windows(x, win, step)                           
    B, S, L, N = xw.shape
    A_list, X_list = [], []
    for s in range(S):
        Aw = corrcoef_batch_time(xw[:, s, :, :])                  
        if topk is not None:
            Aw = sparsify_topk(Aw, topk=topk)
        else:
            Aw = binarize_top_percent(Aw, keep_ratio=keep_ratio)
        deg = (Aw.sum(dim=-1) - 1.0)                              
        mean = xw[:, s].mean(dim=1)
        stdv = xw[:, s].std(dim=1).clamp_min(1e-8)
        Xw = torch.stack([mean, stdv, deg], dim=-1)            
        A_list.append(Aw); X_list.append(Xw)
    return torch.stack(A_list, dim=1), torch.stack(X_list, dim=1)

class GINConv(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=None, eps_init=0.0, dropout=0.0):
        super().__init__()
        hidden = hidden or out_dim
        self.eps = nn.Parameter(torch.tensor(eps_init, dtype=torch.float32))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )
        self.drop = nn.Dropout(dropout)
    def forward(self, h, A):
        return self.drop(self.mlp((1.0 + self.eps) * h + A @ h))

class NodeAttnReadout(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, in_dim)
        self.u = nn.Parameter(torch.randn(in_dim))
    def forward(self, H):
        S = torch.tanh(self.lin(H))                              
        score = S @ self.u                                        
        alpha = F.softmax(score, dim=1)
        g = (alpha.unsqueeze(-1) * H).sum(dim=1)                 
        return g, alpha

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
    def forward(self, x):  
        S,B,F = x.shape
        return x + self.pe[:S, :F].unsqueeze(1)

class TemporalAttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.lin = nn.Linear(d_model, d_model)
        self.v = nn.Parameter(torch.randn(d_model))
    def forward(self, G_seq):  
        S = torch.tanh(self.lin(G_seq))
        score = S @ self.v                                      
        beta = F.softmax(score, dim=1)
        z = (beta.unsqueeze(-1) * G_seq).sum(dim=1)             
        return z, beta

# ---------------------------
# MDGL
# ---------------------------

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.N = configs.channel
        self.use_norm = configs.use_norm
        self.ms_wins  = [30, 60]
        self.ms_steps = [2,  2 ]
        if len(self.ms_steps) < len(self.ms_wins):
            self.ms_steps = self.ms_steps + [self.ms_steps[-1]] * (len(self.ms_wins) - len(self.ms_steps))
        self.topk = None             
        self.keep_ratio = 0.3   

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
            dim_feedforward=2 * configs.d_model,
            dropout=configs.dropout,
            batch_first=False,
            activation='gelu'
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.temporal_pool = TemporalAttentionPool(configs.d_model)
        self.out_dim = 1 if configs.classes == 2 else configs.classes
        self.classifier = nn.Linear(configs.d_model * len(self.ms_wins), self.out_dim)

    def _encode_one_scale(self, x_enc, win, step):
        A_seq, X_seq = build_dynamic_graphs(
            x_enc, win=win, step=step, topk=self.topk, keep_ratio=self.keep_ratio
        )  
        B, S, N, _ = A_seq.shape
        X_seq = self.node_norm(X_seq)
        G_list = []
        for s in range(S):
            H = X_seq[:, s]                                    
            A = A_seq[:, s]                                   
            for gin in self.gin_layers:
                H = gin(H, A)                                
            g, _ = self.readout(H)                             
            G_list.append(g)
        G_seq = torch.stack(G_list, dim=1)
        G_seq_t = self.posenc(G_seq.transpose(0, 1))           
        Z_seq = self.temporal(G_seq_t).transpose(0, 1)          
        z, _ = self.temporal_pool(Z_seq)                       
        return z

    def forward(self, x_enc):
        B, T, N = x_enc.shape
        assert N == self.N, f"N mismatch: got {N}, expected {self.N}"
        zs = []
        for w, s in zip(self.ms_wins, self.ms_steps):
            zs.append(self._encode_one_scale(x_enc, win=w, step=s))
        z_ms = torch.cat(zs, dim=-1)                           
        out = self.classifier(z_ms)
        if self.out_dim == 1:
            out = torch.sigmoid(out)
        return out
