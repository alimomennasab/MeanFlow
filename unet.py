# 2d unet backbone 

# resolution: 64 -> 32 -> 16 -> 8 -> 4
# residual blocks, one per resolution

import torch.nn as nn
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super.__init__()
        self.norm1 = nn.BatchNorm2d()
        self.act1 = nn.LeakyRelu()
        self.conv1 = nn.Conv2d(in_ch, out_ch)
        self.norm2 = nn.BatchNorm2d()
        self.act2 = nn.LeakyRelu()
        self.conv2 = nn.Conv2d(out_ch, out_ch)
        self.skip = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        f = x
        f = self.norm1(f)
        f = self.act1(f)
        f = self.conv1(f)
        f = self.norm2(f)
        f = self.act2(f)
        f = self.conv2(f)
        return f + self.skip(x)
