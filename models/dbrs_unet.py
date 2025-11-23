import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=k//2)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.b1 = ConvBlock(ch, ch)
        self.b2 = ConvBlock(ch, ch, act=False)
    def forward(self, x):
        return F.relu(x + self.b2(self.b1(x)), inplace=True)

class DBRSUNet(nn.Module):
    """Lightweight refinement UNet predicting residual.
    Args:
      in_channels: channels of MRDNet final output (usually 3)
      cond_channels: list of channels for conditioning tensors (e.g. multi-scale outputs)
      base_channels: internal width
      steps: kept for future diffusion sampler integration; unused now
    Forward:
      x: image from MRDNet (full resolution)
      cond: list/tuple of conditioning tensors (e.g. MRDNet outputs)
    Returns refined image = x + residual.
    """
    def __init__(self, in_channels=3, cond_channels=(3,3,3), base_channels=48, steps=8):
        super().__init__()
        self.steps = steps
        total_in = in_channels + sum(cond_channels)
        self.enc1 = ConvBlock(total_in, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels*2)
        self.enc3 = ConvBlock(base_channels*2, base_channels*2)
        self.res_blocks = nn.Sequential(*[ResidualBlock(base_channels*2) for _ in range(4)])
        self.dec2 = ConvBlock(base_channels*2, base_channels)
        self.dec1 = ConvBlock(base_channels, base_channels)
        self.out = nn.Conv2d(base_channels, in_channels, 3, padding=1)
    def forward(self, x, cond=None):
        cond_list = []
        if cond is not None:
            if isinstance(cond, (list, tuple)):
                for t in cond:
                    # resize to x resolution if mismatch
                    if t.shape[-2:] != x.shape[-2:]:
                        t = F.interpolate(t, size=x.shape[-2:], mode='bilinear', align_corners=False)
                    cond_list.append(t)
            else:
                t = cond
                if t.shape[-2:] != x.shape[-2:]:
                    t = F.interpolate(t, size=x.shape[-2:], mode='bilinear', align_corners=False)
                cond_list.append(t)
        merged = torch.cat([x] + cond_list, dim=1) if cond_list else x
        e1 = self.enc1(merged)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        mid = self.res_blocks(e3)
        d2 = self.dec2(F.interpolate(mid, scale_factor=2, mode='bilinear')) + e2
        d1 = self.dec1(F.interpolate(d2, scale_factor=2, mode='bilinear')) + e1
        residual = self.out(d1)
        return torch.clamp(x + residual, 0.0, 1.0)
