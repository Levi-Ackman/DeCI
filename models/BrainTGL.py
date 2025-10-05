# BrainTGL: A dynamic graph representation learning model for brain network analysis
# BrainTGL
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

# ---------------------------
# Utils: windows & correlations (PCC)
# ---------------------------

def sliding_windows(x, win, step):
    B, T, N = x.shape
    idx = [(s, s+win) for s in range(0, T - win + 1, step)]
    return torch.stack([x[:, s:e, :] for (s, e) in idx], dim=1)

def corrcoef_batch_time(x):
    B,L,N = x.shape
    xm = x - x.mean(dim=1, keepdim=True)
    cov = xm.transpose(1,2) @ xm / (L - 1 + 1e-8)              # [B,N,N]
    var = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(1e-8)
    std = var.sqrt()
    corr = cov / (std.unsqueeze(-1) * std.unsqueeze(-2))
    corr = torch.nan_to_num(0.5*(corr + corr.transpose(1,2)), 0.0, 0.0, 0.0)
    eye = torch.eye(N, device=x.device).unsqueeze(0)
    return corr*(1-eye) + eye

def build_dynamic_fc(x, win=30, step=None):
    if step is None: step = win
    xw = sliding_windows(x, win, step)
    B,Wn,L,N = xw.shape
    A_list = [corrcoef_batch_time(xw[:, w, :, :]) for w in range(Wn)]
    return torch.stack(A_list, dim=1), xw

# ---------------------------
# S-RL: signal-level temporal conv (per ROI)
# ---------------------------

class SignalRep(nn.Module):
    def __init__(self, out_dim=64, ch=32, k=5, layers=2, dropout=0.1):
        super().__init__()
        mods = []
        in_ch = 1
        for i in range(layers):
            mods += [nn.Conv1d(in_ch, ch, k, padding=k//2), nn.ReLU(True)]
            in_ch = ch
        self.conv = nn.Sequential(*mods)
        self.proj = nn.Linear(ch, out_dim)
        self.drop = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(self, xw):
        B,L,N = xw.shape
        seq = xw.transpose(1,2).contiguous().view(B*N, 1, L)  
        h = self.conv(seq).mean(-1)                          
        h = self.proj(h)                                      
        h = self.drop(h).view(B, N, -1)                      
        return h

# ---------------------------
# Graph basics: GCN & Attentional pooling (coarsening)
# ---------------------------

class BatchedGCN(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    @staticmethod
    def norm_adj(A):
        B,N,_ = A.shape
        I = torch.eye(N, device=A.device).unsqueeze(0)
        Ahat = A + I
        d = Ahat.sum(-1).clamp_min(1e-8)
        Dinv = torch.diag_embed(d.pow(-0.5))
        return Dinv @ Ahat @ Dinv

    def forward(self, X, A):
        A_n = self.norm_adj(A)
        return A_n @ self.lin(X)

class AttnGraphPooling(nn.Module):
    def __init__(self, in_dim, n_clusters):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(n_clusters, in_dim))
        nn.init.xavier_uniform_(self.prototypes)
        self.n_clusters = n_clusters

    def forward(self, H, A):
        B,N,F = H.shape
        P = self.prototypes                               
        scores = H @ P.t()                                
        S = torch.softmax(scores, dim=-1)                 
        Hc = torch.einsum('bnk,bnf->bkf', S, H)           
        Ac = torch.einsum('bnk,bnm,bml->bkl', S, A, S) 
        Ac = 0.5*(Ac + Ac.transpose(1,2))
        return Hc, Ac, S

# ---------------------------
# TG-RL: Graph-convolutional LSTM cell (with multi-skip)
# ---------------------------

class GCLSTMCell(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.gxi = BatchedGCN(in_dim,  hid_dim)
        self.ghi = BatchedGCN(hid_dim, hid_dim)
        self.gxf = BatchedGCN(in_dim,  hid_dim)
        self.ghf = BatchedGCN(hid_dim, hid_dim)
        self.gxo = BatchedGCN(in_dim,  hid_dim)
        self.gho = BatchedGCN(hid_dim, hid_dim)
        self.gxc = BatchedGCN(in_dim,  hid_dim)
        self.ghc = BatchedGCN(hid_dim, hid_dim)

    def forward(self, X, A, H_prev, C_prev):
        i = torch.sigmoid(self.gxi(X, A) + self.ghi(H_prev, A))
        f = torch.sigmoid(self.gxf(X, A) + self.ghf(H_prev, A))
        o = torch.sigmoid(self.gxo(X, A) + self.gho(H_prev, A))
        g = torch.tanh(   self.gxc(X, A) + self.ghc(H_prev, A))
        C = f * C_prev + i * g
        H = o * torch.tanh(C)
        return H, C

class TGRL(nn.Module):
    def __init__(self, in_dim, hid_dim, p_list=(1,2), dropout=0.1):
        super().__init__()
        self.cell = GCLSTMCell(in_dim, hid_dim)
        self.p_list = sorted(set(p_list))
        self.drop = nn.Dropout(dropout)

    def forward(self, X_seq, A_seq):
        B,Wn,Nc,F = X_seq.shape
        H = torch.zeros(B, Nc, self.cell.ghi.lin.out_features, device=X_seq.device)
        C = torch.zeros_like(H)
        H_hist = []
        for t in range(Wn):
            H_prev = H.clone()
            for p in self.p_list:
                if len(H_hist) >= p:
                    H_prev = H_prev + H_hist[-p]
            H_prev = H_prev / (1 + len([p for p in self.p_list if len(H_hist)>=p]))
            H, C = self.cell(X_seq[:, t], A_seq[:, t], H_prev, C)
            H = self.drop(H)
            H_hist.append(H)
        H_seq = torch.stack(H_hist, dim=1) 
        return H_seq

# ---------------------------
# Temporal attention pooling
# ---------------------------

class TemporalAttnPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_model))
    def forward(self, Z): 
        q = self.q / (self.q.norm() + 1e-8)
        logits = (Z * q.view(1,1,-1)).sum(-1)          
        w = torch.softmax(logits, dim=1).unsqueeze(-1)
        return (w * Z).sum(1), w.squeeze(-1)           

# ---------------------------
# BrainTGL
# ---------------------------

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.N = configs.channel
        self.win = 30
        self.step = None
        self.out_dim = 1 if configs.classes == 2 else configs.classes

        self.srl = SignalRep(
            out_dim=configs.d_model,
            ch=32,
            k=5,
            layers=configs.layer,
            dropout=configs.dropout
        )
        self.n_clusters = 32
        self.pool = AttnGraphPooling(in_dim=configs.d_model, n_clusters=self.n_clusters)
        self.tgrl = TGRL(
            in_dim=configs.d_model,
            hid_dim=configs.d_model*2,
            p_list=(1,2),
            dropout=configs.dropout
        )
        self.readout_t = TemporalAttnPool(configs.d_model*2)
        self.readout_s = TemporalAttnPool(configs.d_model)
        self.use_signal_branch = True

        fused_dim = configs.d_model*2 + (configs.d_model if self.use_signal_branch else 0)
        self.cls = nn.Linear(fused_dim, self.out_dim)

    def forward(self, x_enc):
        """
        x_enc: [B,T,N]
        """
        B,T,N = x_enc.shape
        assert N == self.N, f"N mismatch: {N} vs {self.N}"
        A_seq, Xw = build_dynamic_fc(x_enc, self.win, self.step) 
        Wn = A_seq.shape[1]
        Xc_list, Ac_list, Hs_list = [], [], []
        for w in range(Wn):
            H = self.srl(Xw[:, w, :, :])                                   
            Hs_list.append(H)
            Xc, Ac, S = self.pool(H, A_seq[:, w, :, :])                    
            Xc_list.append(Xc); Ac_list.append(Ac)

        Xc_seq = torch.stack(Xc_list, dim=1)                                
        Ac_seq = torch.stack(Ac_list, dim=1)                           
        H_seq = self.tgrl(Xc_seq, Ac_seq)                                  
        Zg_seq = H_seq.mean(dim=2)                                          
        z_graph, w_tg = self.readout_t(Zg_seq)                              

        z = z_graph
        aux = {'temporal_alpha_graph': w_tg}
        if self.use_signal_branch:
            Zs_seq = torch.stack([Hs.mean(dim=1) for Hs in Hs_list], dim=1) 
            z_signal, w_ts = self.readout_s(Zs_seq)                         
            z = torch.cat([z_graph, z_signal], dim=-1)
            aux['temporal_alpha_signal'] = w_ts

        out = self.cls(z)                                                   
        if self.out_dim == 1:
            out = torch.sigmoid(out)
        return out

