import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C3k


class CDFE(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(ch, ch, 1)

        self.h_conv = nn.Conv2d(ch, ch, (1, 7), 1, (0, 3), groups=ch, bias=False)
        self.v_conv = nn.Conv2d(ch, ch, (7, 1), 1, (3, 0), groups=ch, bias=False)
        self.conv2 = Conv(ch, ch, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x_p = self.conv1(self.avg_pool(x))
        attn = self.v_conv(self.h_conv(x_p))
        return self.act(self.conv2(attn))


class SpatialAdaDRF_V2(nn.Module):

    def __init__(self, ch):
        super().__init__()

        self.conv_std = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.conv_dilated = nn.Conv2d(ch, ch, 3, padding=3, dilation=3, groups=ch, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.SiLU()

        self.spatial_weight_gen = nn.Sequential(
            nn.Conv2d(ch, ch // 4, 1, bias=False),            
            nn.Conv2d(ch // 4, 2, 3, padding=1, bias=False),  
            nn.Sigmoid()                                      
        )

    def forward(self, x):
        w = self.spatial_weight_gen(x)
        x1 = self.conv_std(x)
        x2 = self.conv_dilated(x)
        out = x1 * w[:, 0:1, :, :] + x2 * w[:, 1:2, :, :]
        return self.act(self.bn(out))


class channel_shuffle(nn.Module):
    def __init__(self, groups=2):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()
        g = self.groups
        x = x.view(b, g, c // g, h, w).transpose(1, 2).contiguous().view(b, c, h, w)
        return x


class ADGM(nn.Module):
    def __init__(self, dim, num_heads=None, area=1):
        super().__init__()
        self.c_half = dim // 2
        
        self.pe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim, bias=False)
        self.shuffle = channel_shuffle(groups=2)
  
        self.data_conv = SpatialAdaDRF_V2(self.c_half)

        self.caa_gate = CDFE(self.c_half)

        self.proj = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):

        shortcut = self.pe(x)
        
        x_shuffled = self.shuffle(x)
        x_data, x_gate = x_shuffled.chunk(2, dim=1)

        f_data = self.data_conv(x_data)

        gate_mask = self.caa_gate(x_gate)

        x_fused = f_data * gate_mask 

        out = torch.cat([x_fused, x_gate], dim=1)
        return self.proj(out) + shortcut


class ADGBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__()

        self.attn = ADGM(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, dim, 1, bias=False)
        )

    def forward(self, x):
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f_ADG(nn.Module):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)  

        self.c = self.c + (32 - self.c % 32) if self.c % 32 != 0 else self.c

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        
        self.m = nn.ModuleList(
            nn.Sequential(*(ADGBlock(self.c, self.c // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(self.c, self.c, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, len(self.gamma), 1, 1) * out
        return out