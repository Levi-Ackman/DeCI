import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Medformer_Layer import ListPatchEmbedding,Encoder,EncoderLayer,MedformerLayer

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2405.19363
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.use_norm = configs.use_norm
        self.channel = configs.channel
        self.single_channel = configs.single_channel
        # Embedding
        patch_len_list = list(map(int, configs.patch_len_list.split(",")))
        stride_list = patch_len_list
        seq_len = configs.seq_len
        patch_num_list = [
            int((seq_len - patch_len) / stride + 2)
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]

        self.enc_embedding = ListPatchEmbedding(
            configs.channel,
            configs.d_model,
            configs.seq_len,
            patch_len_list,
            stride_list,
            configs.dropout,
            configs.single_channel,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    MedformerLayer(
                        len(patch_len_list),
                        configs.d_model,
                        configs.n_head,
                        configs.dropout,
                        configs.output_attention,
                        configs.no_inter_attn,
                    ),
                    configs.d_model,
                    configs.d_model,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.layer)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
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
        # Embedding
        x_emb = self.enc_embedding(x_enc)
        x_emb, attns = self.encoder(x_emb)
        if self.single_channel:
            x_emb = torch.reshape(x_emb, (-1, self.channel, *x_emb.shape[-2:]))

        # Output
        x_emb=x_emb.mean(dim=1)
        y_hat=self.classifier(x_emb)

        if self.out_dim==1: y_hat=torch.sigmoid(y_hat)
        return y_hat