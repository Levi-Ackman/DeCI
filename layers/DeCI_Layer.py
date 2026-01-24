import torch
import torch.nn as nn
import torch.nn.functional as F

class DeCI_Block(nn.Module):
    def __init__(self, configs):
        super(DeCI_Block, self).__init__()
        self.trend_ext=Trend_ext(configs.channel,configs.d_model,configs.dropout)
        self.seasonal_ext = Seasonal_ext(configs.d_model, dropout=configs.dropout)
        
        self.out_dim=1 if configs.classes==2 else configs.classes
        self.trend_classifier=nn.Linear(configs.d_model,self.out_dim)
        self.seasonal_classifier=nn.Linear(configs.d_model,self.out_dim)

    def forward(self, inp):
        # inp: [B, N, D]
        trend=self.trend_ext(inp)
        res=inp-trend
        seasonal=self.seasonal_ext(res)
        res=res-seasonal
        
        cls_trend=self.trend_classifier(trend.mean(dim=1)) # [B, N, D] -> [B, C]
        cls_seasonal=self.seasonal_classifier(seasonal.mean(dim=1)) # [B, N, D] -> [B, C]
        
        return cls_trend,cls_seasonal,res

class Trend_ext(nn.Module):
    def __init__(self, channel,kernel_size=256,dropout=0.):
        super(Trend_ext, self).__init__()
        self.kernel_size=kernel_size
        self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size, stride=1, padding=0, bias=True,groups=channel) 
        weights = torch.ones(channel, 1, kernel_size)
        self.conv.weight.data = F.softmax(weights, dim=-1)
        self.conv.bias.data.fill_(0.0)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, inp):
        B,C,D=inp.shape
        front = torch.zeros([B,C,self.kernel_size-1]).to(inp.device)
        inp = torch.cat([front, inp], dim=-1)
        out = self.conv(inp)
        return self.dropout(out)


class Seasonal_ext(nn.Module):
    def __init__(self,d_model,dropout):
        super(Seasonal_ext, self).__init__()
        self.Gate = nn.Sequential(
                nn.Linear(d_model, d_model*2),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(d_model*2, d_model),
                nn.Sigmoid()
        )
        self.Gate_out = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.MLP = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(d_model*2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self, inp):
        
        gate_weight = self.Gate(inp)
        gate_out = self.Gate_out((inp *(gate_weight)))
        emb1=self.norm1(gate_out+inp)
        emb2 = self.MLP(emb1)
        out = self.norm2(emb2+emb1)
        return out   