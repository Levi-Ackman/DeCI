# Similarity-guided multi-view functional brain network fusion
# SimMV_FBNFusion / multi-view FBN fusion + GCN for fMRI-based classification

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

def _safe_corrcoef(x):
    B, L, N = x.shape
    xm = x - x.mean(dim=1, keepdim=True)
    cov = xm.transpose(1, 2) @ xm / (L - 1 + 1e-8)
    var = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(1e-8)
    std = var.sqrt()
    denom = std.unsqueeze(-1) * std.unsqueeze(-2)
    corr = cov / denom
    eye = torch.eye(N, device=x.device).unsqueeze(0).expand(B, -1, -1)
    corr = torch.nan_to_num(corr, 0.0, 0.0, 0.0)
    corr = corr * (1 - eye) + eye
    return corr

def _spearman_corr(x):
    B, L, N = x.shape
    ranks = torch.zeros_like(x)
    idx = torch.argsort(x, dim=1)
    inv = torch.argsort(idx, dim=1)
    ranks = inv.float() + 1.0
    return _safe_corrcoef(ranks)

def _partial_corr(x): 
    B, L, N = x.shape
    xm = x - x.mean(dim=1, keepdim=True)
    cov = xm.transpose(1, 2) @ xm / (L - 1 + 1e-8)
    eye = torch.eye(N, device=x.device).unsqueeze(0)
    cov = cov + 1e-3 * eye
    prec = torch.linalg.pinv(cov)                          
    d = torch.diagonal(prec, dim1=-2, dim2=-1).clamp_min(1e-8)
    denom = torch.sqrt(d).unsqueeze(-1) * torch.sqrt(d).unsqueeze(-2)
    pc = -prec / denom
    eyeB = eye.expand(B, -1, -1)
    pc = pc * (1 - eyeB) + eyeB
    pc = torch.tanh(pc)
    return pc

def _normalize_adj(adj):
    B, N, _ = adj.shape
    eye = torch.eye(N, device=adj.device).unsqueeze(0).expand(B, -1, -1)
    A = (adj + adj.transpose(1,2)) * 0.5
    A = A - torch.diagonal(A, dim1=-2, dim2=-1).unsqueeze(-1) * torch.eye(N, device=adj.device)
    A = A + eye
    d = A.sum(-1).clamp_min(1e-8)
    Dinv2 = torch.diag_embed(d.pow(-0.5))
    return Dinv2 @ A @ Dinv2

def _upper_vec(A):
    B, N, _ = A.shape
    iu = torch.triu_indices(N, N, offset=1, device=A.device)
    return A[:, iu[0], iu[1]]

def _diffusion(A, K=3):
    P = _normalize_adj(A) 
    acc = torch.zeros_like(P)
    Pk = P
    for _ in range(K):
        acc = acc + Pk
        Pk = Pk @ P
    return acc / K

def build_views(x_enc, views=('pearson','spearman','partial')):
    outs = {}
    if 'pearson' in views:
        outs['pearson'] = _safe_corrcoef(x_enc)
    if 'spearman' in views:
        outs['spearman'] = _spearman_corr(x_enc)
    if 'partial' in views:
        outs['partial']  = _partial_corr(x_enc)
    return outs

class SimilarityGuidedFusion(nn.Module):
    def __init__(self, gamma=4.0, use_abs=False):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        self.use_abs = use_abs

    def forward(self, A_dict):
        names = list(A_dict.keys())
        A_stack = torch.stack([A_dict[k] for k in names], dim=1)
        if self.use_abs:
            A_stack = A_stack.abs()

        B, V, N, _ = A_stack.shape
        vecs = _upper_vec(A_stack.view(B*V, N, N)).view(B, V, -1)
        sims = []
        for v in range(V):
            others = torch.mean(torch.stack([vecs[:,u,:] for u in range(V) if u!=v], dim=1), dim=1)
            vvec = vecs[:, v, :]
            num = (vvec * others).sum(-1)
            den = vvec.norm(dim=-1).clamp_min(1e-8) * others.norm(dim=-1).clamp_min(1e-8)
            sims.append(num / den)
        sims = torch.stack(sims, dim=1) 
        alphas = F.softmax(self.gamma * sims, dim=1) 

        A_fused = (alphas.view(B, V, 1, 1) * A_stack).sum(dim=1)
        A_fused = 0.5 * (A_fused + A_fused.transpose(1,2))
        eye = torch.eye(N, device=A_fused.device).unsqueeze(0).expand(B, -1, -1)
        A_fused = A_fused * (1 - eye) + eye
        return A_fused, alphas, A_stack

    @staticmethod
    def losses(A_stack, A_fused, K=3, lam_manifold=1.0, lam_pair=0.5):
        B, V, N, _ = A_stack.shape
        D_views = _diffusion(A_stack.view(B*V, N, N), K=K).view(B, V, N, N)
        D_fused = _diffusion(A_fused, K=K) 

        D_mean = D_views.mean(dim=1)
        L_manifold = ((D_fused - D_mean)**2).mean()
        dv = _upper_vec(D_views.view(B*V, N, N)).view(B, V, -1)
        df = _upper_vec(D_fused).unsqueeze(1) 
        num = (dv * df).sum(-1)
        den = dv.norm(dim=-1).clamp_min(1e-8) * df.norm(dim=-1).clamp_min(1e-8)
        cos_sim = (num / den).mean()
        L_pair = -cos_sim

        return lam_manifold * L_manifold + lam_pair * L_pair, {'L_manifold': L_manifold.item(), 'L_pair': L_pair.item()}

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        A = _normalize_adj(adj)     
        h = self.linear(x)         
        h = A @ h
        return self.dropout(self.relu(h))

# ---------------------------
# SimMVF
# ---------------------------

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.views = ['pearson','spearman','partial']
        self.N = configs.channel
        self.use_norm = configs.use_norm

        self.fuser = SimilarityGuidedFusion(4.0, use_abs=False)

        in_dim = self.N * len(self.views)
        self.norm = nn.LayerNorm(in_dim) if self.use_norm else nn.Identity()

        self.gcn_layers = nn.ModuleList()
        for i in range(configs.layer):
            fin = in_dim if i == 0 else configs.d_model
            self.gcn_layers.append(GCNLayer(fin, configs.d_model, configs.dropout))

        self.out_dim = 1 if configs.classes == 2 else configs.classes
        self.classifier = nn.Linear(configs.d_model, self.out_dim)
        self.K = 3
        self.lam_manifold = 0.1
        self.lam_pair     = 0.05

    def forward(self, x_enc):
        B, T, N = x_enc.shape
        assert N == self.N
        A_dict = build_views(x_enc, self.views)
        A_fused, alphas, A_stack = self.fuser(A_dict)
        X = torch.cat([A_dict[k] for k in self.views], dim=-1)   
        X = self.norm(X)
        H = X
        for layer in self.gcn_layers:
            H = layer(H, A_fused)                             
        g = H.mean(dim=1)                                       

        logits = self.classifier(g)                                
        out = torch.sigmoid(logits) if self.out_dim == 1 else logits

        aux = {}
        if (self.lam_manifold > 0.0) or (self.lam_pair > 0.0):
            reg, parts = self.fuser.losses(
                A_stack, A_fused, K=self.K,
                lam_manifold=self.lam_manifold, lam_pair=self.lam_pair
            )
            aux['reg_loss'] = reg
            aux.update(parts)
        return out
