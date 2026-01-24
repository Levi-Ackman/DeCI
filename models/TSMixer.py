import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, configs):
        super(ResBlock, self).__init__()

        self.temporal = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.seq_len),
            nn.Dropout(configs.dropout)
        )

        self.channel = nn.Sequential(
            nn.Linear(configs.channel, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.channel),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x):
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.use_norm = configs.use_norm
        self.layer = configs.layer
        self.model = nn.ModuleList([ResBlock(configs)
                                    for _ in range(configs.layer)])
        
        self.out_dim=1 if configs.classes==2 else configs.classes
        self.classifier=nn.Linear(configs.seq_len,self.out_dim)

    def forward(self, x_enc):
        B,T,N=x_enc.shape
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        
        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
        x_enc=x_enc.mean(dim=-1)
        y_hat=self.classifier(x_enc)
        if self.out_dim==1: y_hat=torch.sigmoid(y_hat)
        return y_hat