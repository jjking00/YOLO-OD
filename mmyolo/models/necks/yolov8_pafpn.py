# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import List, Union

from mmyolo.models.layers.yolo_bricks import CSPLayerWithTwoConv
from mmyolo.models.necks.yolov5_pafpn import YOLOv5PAFPN
from mmyolo.models.utils.misc import make_divisible, make_round
import torch
import torch.nn as nn
from mmdet.utils import ConfigType, OptMultiConfig
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmyolo.registry import MODELS
from mmcv.cnn import build_activation_layer
import torch.nn.functional as F

class LearnableActivation(nn.Module):
    def __init__(self):
        super(LearnableActivation, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))  

    def forward(self, x):
        return self.alpha * torch.nn.functional.silu(x)  


class LearnableBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super(LearnableBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)  # Use non-fixed parameters
        self.gamma = nn.Parameter(torch.ones(num_features, requires_grad=True))  # Learnable scaling parameter γ
        self.beta = nn.Parameter(torch.zeros(num_features, requires_grad=True))  # Learnable offset parameter β
        self.delta_mu = nn.Parameter(torch.zeros(num_features, requires_grad=True))  # Learnable mean adjustment parameter Δμ
        self.delta_sigma = nn.Parameter(torch.zeros(num_features, requires_grad=True))  # Learnable variance adjustment parameter Δσ

    def forward(self, x):

        # print(f"Input shape: {x.shape}")
        # print(f"BatchNorm running mean shape: {self.bn.running_mean.shape}")
        
        # Calculate standardized output.
        x_hat = (x - self.bn.running_mean.view(1, -1, 1, 1) + self.delta_mu.view(1, -1, 1, 1)) / \
                (torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.delta_sigma.view(1, -1, 1, 1) + 1e-5))
        
        # Apply learnable scaling and offset
        out = self.gamma.view(1, -1, 1, 1) * x_hat + self.beta.view(1, -1, 1, 1)
        return out



class MixedConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MixedConv, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)

    def forward(self, x):
        out1 = self.conv1x1(x)
        out2 = self.conv3x3(x)
        out3 = self.conv5x5(x)
        # return out2
        # return torch.cat([out1,out2,out3],dim=1)
        return out1 + out2 + out3  

class CSPLayerWithLearnableActivation(nn.Module):  
    def __init__(self, in_channels, out_channels, num_blocks, add_identity, norm_cfg, act_cfg):
        super(CSPLayerWithLearnableActivation, self).__init__()
        self.add_identity = add_identity
        self.mixed_conv1 = MixedConv(in_channels, out_channels)
        self.mixed_conv2 = MixedConv(out_channels, out_channels)
        self.norm1 = LearnableBatchNorm2d(out_channels)
        self.activation1 = LearnableActivation()
        self.norm2 = LearnableBatchNorm2d(out_channels)
        self.activation2 = LearnableActivation()

    def forward(self, x):
        identity = x
        out = self.mixed_conv1(x)
        out = self.norm1(out)
        out = self.activation1(out)
        out = self.mixed_conv2(out)
        out = self.norm2(out)
        out = self.activation2(out)
        if self.add_identity:
            out += identity
        return out
@MODELS.register_module()
class YOLOv8PAFPN(YOLOv5PAFPN):
    """Path Aggregation Network used in YOLOv8.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Defaults to 1.
        freeze_all(bool): Whether to freeze the model
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            num_csp_blocks=num_csp_blocks,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        return nn.Identity()

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        return CSPLayerWithLearnableActivation(
            make_divisible((self.in_channels[idx - 1] + self.in_channels[idx]),
                           self.widen_factor),
            make_divisible(self.out_channels[idx - 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        return CSPLayerWithLearnableActivation(
            make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]),
                self.widen_factor),
            make_divisible(self.out_channels[idx + 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
