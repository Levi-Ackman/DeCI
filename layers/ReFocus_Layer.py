import torch
import torch.nn as nn
import torch.nn.functional as F
        
class FLinear(nn.Module):
    def __init__(self, inp, out):
        super(FLinear, self).__init__()
        self.inp_size = inp // 2 + 1
        self.out_size = out // 2 + 1
        self.proj = nn.Linear(self.inp_size, self.out_size).to(torch.cfloat)
        
    def forward(self, x):
        return torch.fft.irfft(self.proj(torch.fft.rfft(x, dim=-1)), dim=-1)

    def initial(self):
        init_value = 1 / self.inp_size
        real_part = torch.full((self.out_size, self.inp_size), init_value)
        imaginary_part = torch.full((self.out_size, self.inp_size), init_value)
        complex_weights = torch.complex(real_part, imaginary_part)
        self.proj.weight = nn.Parameter(complex_weights)
        
class Filter(nn.Module):
    def __init__(self,channel=1,kernel_size=25):
        super(Filter, self).__init__()
        self.kernel_size=kernel_size
        self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size, stride=1, 
                              padding=int(kernel_size//2), padding_mode='replicate', bias=True,groups=channel)
        self.conv.weight = nn.Parameter(
                (1 / kernel_size) * torch.ones([channel, 1, kernel_size]))
    def forward(self, inp):
        out = self.conv(inp.transpose(1,2)).transpose(1,2)
        return out


class Encoder(nn.Module):
    def __init__(self, configs):
        super(Encoder, self).__init__()
        self.fc1 = FLinear(configs.d_model, configs.d_model)
        self.fc2 = FLinear(configs.d_model, configs.d_model)
        self.fc_core = FLinear(configs.d_model, configs.d_model)
        self.fc_ori = FLinear(configs.d_model, configs.d_model)
        
        self.fc3 = FLinear(configs.d_model, configs.d_model)
        self.fc4 = FLinear(configs.d_model, configs.d_model)
        self.fc5 = FLinear(configs.d_model, configs.d_model)
        self.fc6 = FLinear(configs.d_model, configs.d_model)
        self.norm1 = nn.LayerNorm(configs.d_model)
        self.norm2 = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, inp):
        batch_size, channels, d_model = inp.shape
        # set FFN
        core = F.gelu(self.fc1(inp))
        core = self.fc2(core)

        # stochastic pooling
        core_fft=torch.fft.rfft(core, dim=-1)
        energy = torch.abs(core_fft).pow(2)
        ratio = F.softmax(energy, dim=1)
        ratio = ratio.permute(0, 2, 1)
        ratio = ratio.reshape(-1, channels)
        indices = torch.multinomial(ratio, 1)
        indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)
        core = torch.fft.irfft(torch.gather(core_fft, 1, indices),dim=-1).repeat(1, channels, 1)
        ## mlp fusion
        core = F.gelu(self.fc3(self.fc_core(core)+self.fc_ori(inp)))
        core = self.fc4(core)
        res = self.norm1(inp + self.dropout(core))
        output1 = res 
        output1 = self.dropout(F.gelu(self.fc5(output1)))
        output2 = self.dropout(self.fc6(output1))
        
        return self.norm2(res + output2)