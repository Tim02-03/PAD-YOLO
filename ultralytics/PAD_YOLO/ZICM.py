import torch
import torch.nn as nn


class ZICM(nn.Module):
    def __init__(self, c1, c2, heads=4): 
        super().__init__()
        
        self.c1_in, self.c2_in = c1[0], c1[1]
        self.dim = c2 
        
        self.hf_explicit_guide = nn.Sequential(
            nn.Conv2d(self.c2_in, self.c2_in, kernel_size=3, padding=1, groups=self.c2_in, bias=False),
            nn.BatchNorm2d(self.c2_in),
            nn.Conv2d(self.c2_in, 1, 1, bias=False), 
            nn.Sigmoid()
        )
        

        mid_channels = max(self.c1_in // 4, 8)
        self.semantic_aware_mask = nn.Sequential(
            nn.Conv2d(self.c1_in, mid_channels, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, self.c2_in, 1, bias=False), 
            nn.Sigmoid()
        )
        

        self.non_destructive_aggregator = nn.Sequential(
            nn.Conv2d(self.c1_in + self.c2_in, self.dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        y1, y2 = x[0], x[1] 
        
        s_mask = self.hf_explicit_guide(y2)
        
        c_mask = self.semantic_aware_mask(y1.mean(dim=(2, 3), keepdim=True))
        y1_out = y1 * (1.0 + s_mask) 

        y2_out = y2 * (1.0 + c_mask)

        out = torch.cat([y1_out, y2_out], dim=1)
        return self.non_destructive_aggregator(out)