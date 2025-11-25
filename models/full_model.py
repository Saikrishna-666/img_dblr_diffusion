import torch
import torch.nn as nn

from models.MRDNet import MRDNet, MRDNetPlus
from models.dbrs_unet import DBRSUNet


class FullModel(nn.Module):
    """Wrap MRDNet-style backbone with an optional DBRS refinement stage."""

    def __init__(
        self,
        model_name: str = 'MRDNet',
        use_dbrs: bool = True,
        use_dfd: bool = False,
        dbrs_depth: int = 48,
        dbrs_steps: int = 4,
    ) -> None:
        super().__init__()
        if model_name == 'MRDNetPlus':
            self.mrd = MRDNetPlus()
        else:
            self.mrd = MRDNet(use_dfd=use_dfd)
        self.use_dbrs = use_dbrs
        if use_dbrs:
            # DBRS expects RGB inputs; future diffusion sampler can use steps parameter.
            self.dbrs = DBRSUNet(in_channels=3, cond_channels=(3, 3, 3), base_channels=dbrs_depth, steps=dbrs_steps)
        else:
            self.dbrs = None

    def forward(self, x: torch.Tensor):
        mrd_outs = self.mrd(x)
        if self.dbrs is not None:
            refined = self.dbrs(mrd_outs[-1], cond=mrd_outs)
        else:
            refined = mrd_outs[-1]
        return mrd_outs, refined


__all__ = ['FullModel']
