import torch
import torch.nn as nn
from layers.DeCI_Layer import DeCI_Block

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.use_norm = configs.use_norm
        # Embedding
        self.Variate_Embedding = nn.Linear(configs.seq_len, configs.d_model)
        # LiNo Block
        self.deci_blocks=nn.ModuleList([DeCI_Block(configs) for _ in range(configs.layer)])  
        
        self.out_dim=1 if configs.classes==2 else configs.classes
    def forward(self, x_enc):
        B,T,N=x_enc.shape
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        x_enc=x_enc.transpose(1,2)
        x_embed = self.Variate_Embedding (x_enc)
        res=x_embed
        trends=[]
        seasonals=[]
        for deci_block in self.deci_blocks:
            trend,seasonal,res=deci_block(res)
            trends.append(trend)
            seasonals.append(seasonal)
        y_hat=sum(trends)+sum(seasonals)
        if self.out_dim==1: y_hat=torch.sigmoid(y_hat)
        return y_hat