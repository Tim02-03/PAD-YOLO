import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv, DWConv

class GSConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, act=True): 
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k=k, s=s, act=act)
        self.cv2 = Conv(c_, c_, k=5, s=1, g=c_, act=act) 
        
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        b, c, h, w = x2.data.size()
        x2 = x2.view(b, 2, c // 2, h, w).permute(0, 2, 1, 3, 4).contiguous()
        return x2.view(b, c, h, w)


class MCI(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, 1, c)
        y = self.conv1d(y).view(b, c, 1, 1)
        return x * self.sigmoid(y)


class PRFE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw_1 = DWConv(dim, dim, k=3)
        self.dw_2 = DWConv(dim, dim, k=3)
        self.dw_3 = DWConv(dim, dim, k=3)
        self.conv_out = Conv(dim, dim, k=1)

    def forward(self, x):
        out3 = self.dw_1(x)              
        out5 = self.dw_2(out3 + x)       
        out7 = self.dw_3(out5 + x)     
        
        return self.conv_out(out3 + out5 + out7)


class PMI_Block(nn.Module): 
    def __init__(self, dim, e=0.5): 
        super().__init__()
        self.dim_split = int(dim * e)
        self.dim_identity = dim - self.dim_split

        self.conv_in = GSConv(dim, dim)

        self.core_branch = nn.Sequential(
            PRFE(self.dim_split), 
            MCI(self.dim_split)  
        )

        self.conv_out = GSConv(dim, dim)

    def forward(self, x):
        x_processed = self.conv_in(x)
        
        core_x, identity_x = torch.split(x_processed, [self.dim_split, self.dim_identity], dim=1)
        
        core_out = self.core_branch(core_x)
        
        return self.conv_out(torch.cat((core_out, identity_x), 1))

class CASD(nn.Module):
    def __init__(self, inc, ouc):
        super().__init__()
        
        self.spatial_calib = Conv(inc, inc, k=3, s=1, g=inc) 

        self.fusion = Conv(inc * 4, ouc, k=1)

    def forward(self, x):
        x = self.spatial_calib(x)
        x_spd = torch.cat([
            x[..., ::2, ::2], 
            x[..., 1::2, ::2], 
            x[..., ::2, 1::2], 
            x[..., 1::2, 1::2]
        ], 1)

        return self.fusion(x_spd)