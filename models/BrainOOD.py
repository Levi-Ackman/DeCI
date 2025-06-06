import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, adj, dropout):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.register_buffer('adj_norm', self.normalize_adj(adj))
    
    def normalize_adj(self, adj):
        device = adj.device
        I = torch.eye(adj.size(0), device=device)
        A_hat = adj + I
        d = A_hat.sum(1)
        D_inv_sqrt = torch.diag(torch.pow(d, -0.5))
        return D_inv_sqrt @ A_hat @ D_inv_sqrt

    def forward(self, x):
        support = self.linear(x)
        out = torch.matmul(self.adj_norm, support)
        out = self.relu(out)
        out = self.dropout(out)
        return out

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.use_norm = configs.use_norm
        self.gcn_layers = nn.ModuleList()
        for i in range(configs.layer):
            in_dim = configs.channel if i == 0 else configs.d_model
            self.gcn_layers.append(GCNLayer(in_dim, configs.d_model, torch.ones([configs.channel,configs.channel]), configs.dropout))
        self.out_dim = 1 if configs.classes == 2 else configs.classes
        self.classifier = nn.Linear(configs.d_model, self.out_dim)

    def forward(self, x_enc):
        B,T,N=x_enc.shape
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        x_fc = torch.nan_to_num(
            torch.stack([torch.corrcoef(x.T) for x in x_enc]),
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )
        out=x_fc
        for layer in self.gcn_layers:
            out = layer(out)
        out = out.mean(dim=1)
        out = self.classifier(out)
        if self.out_dim == 1:
            out = torch.sigmoid(out)
        return out
