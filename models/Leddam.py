import torch
import torch.nn as nn
from layers.Leddam_Layer import Leddam

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.use_norm = configs.use_norm
        self.leddam=Leddam(configs.channel,configs.seq_len,configs.d_model,
                       configs.dropout,pe_type='zeros',kernel_size=25,n_layers=configs.layer)
        self.out_dim=1 if configs.classes==2 else configs.classes
        self.res_classifier=nn.Linear(configs.d_model,self.out_dim)
        self.main_classifier=nn.Linear(configs.d_model,self.out_dim)
    def forward(self, x_enc):
        B,T,N=x_enc.shape
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        res,main=self.leddam(x_enc)
        res=res.mean(dim=-1)
        main=main.mean(dim=-1)
        y_hat=self.res_classifier(res)+self.main_classifier(main)
        if self.out_dim==1: y_hat=torch.sigmoid(y_hat)
        return y_hat