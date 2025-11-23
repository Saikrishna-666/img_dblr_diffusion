import torch
import torch.nn as nn
from models.MRDNet import MRDNet, MRDNetPlus
from models.dbrs_unet import DBRSUNet

class FullModel(nn.Module):
    """Wrapper combining MRDNet* backbone and DBRS refinement.
    Args:
        model_name: 'MRDNet' or 'MRDNetPlus'
        use_dfd: enable DFD inside MRDNet
        dbrs_depth: base channels for DBRS (passed as base_channels)
        dbrs_steps: sampling steps placeholder for future diffusion sampler
    Forward returns (mrd_outs, refined) where mrd_outs is list of multi-scale outputs.
    """
    def __init__(self, model_name='MRDNet', use_dfd=False, dbrs_depth=48, dbrs_steps=4):
        super().__init__()
        if model_name == 'MRDNetPlus':
            self.mrd = MRDNetPlus()
        else:
            self.mrd = MRDNet(use_dfd=use_dfd)
        # Determine conditioning channel sizes from MRDNet outputs (all are RGB 3)
        self.dbrs = DBRSUNet(in_channels=3, cond_channels=(3,3,3), base_channels=dbrs_depth, steps=dbrs_steps)

    def forward(self, x):
        mrd_outs = self.mrd(x)  # list [low, mid, high]
        refined = self.dbrs(mrd_outs[-1], cond=mrd_outs)
        return mrd_outs, refined

__all__ = ["FullModel"]import torch
import torch.nn as nn
from models.MRDNet import MRDNet
from models.dbrs_unet import DBRSUNet

class FullModel(nn.Module):
    """Wrapper combining MRDNet and a lightweight DBRS refinement UNet.
    Args:
      use_dbrs: enable refinement stage
      use_dfd: pass through to MRDNet (optional frequency decomposition)
      refine_channels: list of channels for conditioning (auto inferred from MRDNet outputs at runtime)
    Forward returns (mrd_outputs_list, refined_image)
    """
    def __init__(self, use_dbrs=True, use_dfd=False):
        super().__init__()
        self.mrd = MRDNet(use_dfd=use_dfd)
        self.use_dbrs = use_dbrs
        if use_dbrs:
            # We don't know per-scale channels until first forward; create placeholder and rebuild later if needed.
            # For simplicity keep fixed cond channel counts assuming RGB outputs.
            self.dbrs = DBRSUNet(in_channels=3, cond_channels=(3,3,3), base_channels=48)
        else:
            self.dbrs = None

    def forward(self, x):
        mrd_outs = self.mrd(x)  # list of 3 tensors [low, mid, full]
        final = mrd_outs[-1]
        if self.dbrs is not None:
            refined = self.dbrs(final, cond=mrd_outs)
        else:
            refined = final
        return mrd_outs, refined
