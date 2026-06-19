# 2d unet backbone with residual blocks consisting of 2 conv operations, injected with r & t timesteps
# unet encoder resolution: 64 -> 32 -> 16 -> 8

import math
import torch
import torch.nn as nn

# timestep sampling and embedding helpers
def sample_t_r(batch_size, percent_unequal=0.25):
    # percent_unequal: fraction of batch where r != t
    t = torch.rand(batch_size)
    r = torch.rand(batch_size) * t  # r in [0, t], so r <= t

    num_unequal = max(int(percent_unequal * batch_size), 1)

    unequal_mask = torch.zeros(batch_size, dtype=torch.bool)
    unequal_indices = torch.randperm(batch_size)[:num_unequal] # randomly choose percent_unequal% of (r, t) to be unequal
    unequal_mask[unequal_indices] = True

    r = torch.where(unequal_mask, r, t)

    return t, r

def timestep_emb(timesteps, d_emb):
    half_dim = d_emb // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half_dim) / (half_dim - 1)
        # torch.arange(half_dim) = [0, 1, 2, ..., half_dim]
        # divide by half_dim - 1 because the array begins at zero, and we want the final elem (half_dim), with index 
        # half_dim - 1, to be perfectly divided to become 1 
        # range of freqs: [1, 1/10000] (which is why we have negative log) 
    )

    # apply frequency to each timestep (outer product)
    args = timesteps[:, None].float() * freqs[None, :] # [batch, half_dim]

    # interleave sin & cos
    emb = torch.zeros(len(timesteps), d_emb) # 2d because we are returning embeddings for a batch

    emb[:, 0::2] = torch.sin(args) # even indexes
    emb[:, 1::2] = torch.cos(args) # odd indexes
    

    return emb

def create_timestep_embs(r, t, d_emb):
    """
    Embedding (t,t-r), that is, time and interval, achieves the best result, while directly embedding (r,t) 
    performs almost as well. Notably, even embedding only the interval t-r yields reasonable results.
    """
    t_embs = timestep_emb(t, d_emb) # t embedding
    tr_embs = timestep_emb(t-r, d_emb) # interval embedding

    return t_embs, tr_embs 

# model 
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, d_emb=256):
        super().__init__()
        self.proj_embs = nn.Linear(d_emb, out_ch)
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act1 = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x, tembs):
        # project timestep embeddings (r and t) to inject inside res block
        tembs = self.proj_embs(tembs)
        tembs = tembs[:, :, None, None] # (B, out_ch, 1, 1)

        f = x
        f = self.norm1(f)
        f = self.act1(f)
        f = self.conv1(f)
        f = f + tembs
        f = self.norm2(f)
        f = self.act2(f)
        f = self.conv2(f)
        return f + self.skip(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, ch=(64, 128, 256, 512), d_emb=256):
        super().__init__()
        self.d_emb = d_emb

        # embedding MLPs (sinusoidal dim -> d_emb, matching MeanFlow TimestepEmbedder)
        self.mlp_t = nn.Sequential(
            nn.Linear(d_emb, d_emb),
            nn.SiLU(),
            nn.Linear(d_emb, d_emb),
        )
        self.mlp_tr = nn.Sequential(
            nn.Linear(d_emb, d_emb),
            nn.SiLU(),
            nn.Linear(d_emb, d_emb),
        )

        # encoder
        self.in_project = nn.Conv2d(in_ch, ch[0], kernel_size=3, padding=1)
        self.enc1 = ResBlock(ch[0], ch[1])
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResBlock(ch[1], ch[2])
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ResBlock(ch[2], ch[3])
        self.pool3 = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = ResBlock(ch[3], ch[3])

        # decoder
        self.up1 = nn.ConvTranspose2d(in_channels=ch[3], out_channels=ch[3], kernel_size=2, stride=2)
        self.dec1 = ResBlock(ch[3] * 2, ch[2])

        self.up2 = nn.ConvTranspose2d(in_channels=ch[2], out_channels=ch[2], kernel_size=2, stride=2)
        self.dec2 = ResBlock(ch[2] * 2, ch[1])

        self.up3 = nn.ConvTranspose2d(in_channels=ch[1], out_channels=ch[1], kernel_size=2, stride=2)
        self.dec3 = ResBlock(ch[1] * 2, ch[0])
        self.out_proj = nn.Conv2d(ch[0], in_ch, kernel_size=3, padding=1)

    def forward(self, x, r, t):
        t_embs, tr_embs = create_timestep_embs(r, t, self.d_emb)
        t_embs = self.mlp_t(t_embs)
        tr_embs = self.mlp_tr(tr_embs)
        tembs = t_embs + tr_embs

        # x: (B, 3, 64, 64)

        # encoder
        x = self.in_project(x)          # (B, 64, 64, 64)
        x = self.enc1(x, tembs)         # (B, 128, 64, 64)
        skip1 = x
        x = self.pool1(x)               # (B, 128, 32, 32)

        x = self.enc2(x, tembs)         # (B, 256, 32, 32)
        skip2 = x
        x = self.pool2(x)               # (B, 256, 16, 16)

        x = self.enc3(x, tembs)         # (B, 512, 16, 16)
        skip3 = x
        x = self.pool3(x)               # (B, 512, 8, 8)  

        # bottleneck
        x = self.bottleneck(x, tembs)   # (B, 512, 8, 8) 

        # decoder
        x = self.up1(x)                             # (B, 512, 16, 16)
        x = torch.cat([x, skip3], dim=1)            # (B, 1024, 16, 16)
        x = self.dec1(x, tembs)                     # (B, 256, 16, 16)

        x = self.up2(x)                             # (B, 256, 32, 32)
        x = torch.cat([x, skip2], dim=1)            # (B, 512, 32, 32)
        x = self.dec2(x, tembs)                     # (B, 128, 32, 32)

        x = self.up3(x)                             # (B, 128, 64, 64)
        x = torch.cat([x, skip1], dim=1)            # (B, 256, 64, 64)
        x = self.dec3(x, tembs)                     # (B, 64, 64, 64)
        x = self.out_proj(x)                        # (B, 3, 64, 64)

        return x

    def u_fn(self, z, r, t):
        # used by torch.func.jvp during training
        return self.forward(z, r, t)
