# Constructing Multi-View High-Order Functional Connectivity Networks for Diagnosis of Autism Spectrum Disorder
# Ho-FCN / multi-view GCN for fMRI-based classification
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

def sliding_windows(x, win, step):
    B, T, N = x.shape
    idx = []
    for s in range(0, T - win + 1, step):
        idx.append((s, s + win))
    chunks = [x[:, s:e, :] for (s, e) in idx] 
    return torch.stack(chunks, dim=1)  


def corrcoef_batch_time(x):
    B, L, N = x.shape
    xm = x - x.mean(dim=1, keepdim=True)
    cov = torch.matmul(xm.transpose(1, 2), xm) / (L - 1 + 1e-8)
    var = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(1e-8)  
    std = var.sqrt()
    denom = torch.matmul(std.unsqueeze(-1), std.unsqueeze(-2))  
    corr = cov / denom
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    eye = torch.eye(N, device=x.device).unsqueeze(0).expand(B, -1, -1)
    corr = corr * (1 - eye) + eye
    return corr


def build_lo_d_fcn(x, win=30, step=2):
    B, T, N = x.shape
    xw = sliding_windows(x, win, step)  
    B, Wn, L, N = xw.shape
    corr_list = []
    for w in range(Wn):
        corr_w = corrcoef_batch_time(xw[:, w, :, :])
        corr_list.append(corr_w)
    lo_d_fcn = torch.stack(corr_list, dim=1) 
    return lo_d_fcn


def central_moment(lo_d_fcn, order=2):
    if order == 1:
        cm = lo_d_fcn.mean(dim=1)  
    else:
        mean = lo_d_fcn.mean(dim=1, keepdim=True)
        centered = lo_d_fcn - mean
        cm = (centered ** order).mean(dim=1)
    cm = torch.nan_to_num(cm, nan=0.0, posinf=0.0, neginf=0.0)
    return cm


def rowwise_corr_batch(M):
    B, N, F = M.shape
    X = M - M.mean(dim=-1, keepdim=True) 
    num = torch.matmul(X, X.transpose(1, 2))
    denom = (X.pow(2).sum(dim=-1).clamp_min(1e-8)).sqrt()
    den = torch.matmul(denom.unsqueeze(-1), denom.unsqueeze(-2))
    corr = num / den
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    eye = torch.eye(N, device=M.device).unsqueeze(0).expand(B, -1, -1)
    corr = corr * (1 - eye) + eye
    return corr

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    @staticmethod
    def normalize_adj(adj):
        B, N, _ = adj.shape
        I = torch.eye(N, device=adj.device).unsqueeze(0).expand(B, -1, -1)
        A_hat = adj + I
        d = A_hat.sum(dim=-1) 
        d_inv_sqrt = (1.0 / d.clamp_min(1e-8)).sqrt()
        D_inv_sqrt = torch.diag_embed(d_inv_sqrt) 
        return D_inv_sqrt @ A_hat @ D_inv_sqrt

    def forward(self, x, adj):
        A_norm = self.normalize_adj(adj)
        support = self.linear(x)
        out = torch.matmul(A_norm, support)
        out = self.relu(out)
        out = self.dropout(out)
        return out

# ---------------------------
# MVHO
# ---------------------------

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.use_norm = configs.use_norm
        self.win = 30
        self.step = 2
        self.moment_orders = [2,4,8]
        self.adj_agg = 'mean'

        self.N = configs.channel
        self.num_views = len(self.moment_orders)
        in_dim = self.N * self.num_views

        self.gcn_layers = nn.ModuleList()
        for i in range(configs.layer):
            fin = in_dim if i == 0 else configs.d_model
            self.gcn_layers.append(GCNLayer(fin, configs.d_model, configs.dropout))

        self.out_dim = 1 if configs.classes == 2 else configs.classes
        self.classifier = nn.Linear(configs.d_model, self.out_dim)
        self.norm = nn.LayerNorm(in_dim) if self.use_norm else nn.Identity()

    @staticmethod
    def _aggregate_adjs(adj_list, mode='mean'):
        A = torch.stack(adj_list, dim=1)
        if mode == 'max':
            A = A.max(dim=1).values
        else:
            A = A.mean(dim=1)
        A = 0.5 * (A + A.transpose(1, 2))
        return A

    def forward(self, x_enc):
        B, T, N = x_enc.shape
        assert N == self.N, f"Input N={N} vs configs.channel={self.N}"
        lo = build_lo_d_fcn(x_enc, self.win, self.step)  

        cm_list, ho_list = [], []
        for d in self.moment_orders:
            cm = central_moment(lo, d)        
            ho = rowwise_corr_batch(cm)       
            cm_list.append(cm)
            ho_list.append(ho)
        X = torch.cat(cm_list, dim=-1) 
        X = self.norm(X)
        A = self._aggregate_adjs(ho_list, mode=self.adj_agg)
        H = X
        for layer in self.gcn_layers:
            H = layer(H, A)
        g = H.mean(dim=1)
        out = self.classifier(g)        
        if self.out_dim == 1:
            out = torch.sigmoid(out)
        return out
