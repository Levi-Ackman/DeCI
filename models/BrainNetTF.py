import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.use_norm = configs.use_norm
        # Embedding
        self.Variate_Embedding = nn.Linear(configs.channel, configs.d_model)
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(attention_dropout=configs.dropout), configs.d_model, configs.n_head),
                    d_model=configs.d_model,
                    d_ff=configs.d_model,
                    dropout=configs.dropout,
                ) for _ in range(configs.layer)
            ]
        )
        self.out_dim=1 if configs.classes==2 else configs.classes
        self.classifier=nn.Linear(configs.d_model,self.out_dim)

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
        x_emb = self.Variate_Embedding(x_fc)
        x_emb = self.encoder(x_emb)
        x_emb=x_emb.mean(dim=1)
        y_hat=self.classifier(x_emb)

        if self.out_dim==1: y_hat=torch.sigmoid(y_hat)
        return y_hat