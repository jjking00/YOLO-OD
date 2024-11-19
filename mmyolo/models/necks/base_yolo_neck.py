# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import List, Union

import cv2
import torch
import torch.nn as nn
from mmdet.utils import ConfigType, OptMultiConfig
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmyolo.registry import MODELS
from mmcv.cnn import build_activation_layer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

@MODELS.register_module()
class BaseYOLONeck(BaseModule, metaclass=ABCMeta):


    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[int, List[int]],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 upsample_feats_cat_first: bool = True,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = None,
                 act_cfg: ConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 **kwargs):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor
        self.upsample_feats_cat_first = upsample_feats_cat_first
        self.freeze_all = freeze_all
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.reduce_layers = nn.ModuleList()
        for idx in range(len(in_channels)):
            self.reduce_layers.append(self.build_reduce_layer(idx))

        # build top-down blocks
        self.upsample_layers = nn.ModuleList()
        self.top_down_layers = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.upsample_layers.append(self.build_upsample_layer(idx))
            self.top_down_layers.append(self.build_top_down_layer(idx))

        # build bottom-up blocks
        self.downsample_layers = nn.ModuleList()
        self.bottom_up_layers = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsample_layers.append(self.build_downsample_layer(idx))
            self.bottom_up_layers.append(self.build_bottom_up_layer(idx))

        self.out_layers = nn.ModuleList()
        for idx in range(len(in_channels)):
            self.out_layers.append(self.build_out_layer(idx))

        self.scale_aware1 = nn.Sequential(
        nn.AdaptiveAvgPool2d(1), nn.Conv2d(128, 1, 1),
        nn.ReLU(inplace=True), build_activation_layer(dict(type='HSigmoid', bias=3.0, divisor=6.0)))
        self.scale_aware2 = nn.Sequential(
        nn.AdaptiveAvgPool2d(1), nn.Conv2d(256, 1, 1),
        nn.ReLU(inplace=True), build_activation_layer(dict(type='HSigmoid', bias=3.0, divisor=6.0)))            

        self.scale_aware3 = nn.Sequential(
        nn.AdaptiveAvgPool2d(1), nn.Conv2d(512, 1, 1),
        nn.ReLU(inplace=True), build_activation_layer(dict(type='HSigmoid', bias=3.0, divisor=6.0)))

        self.norm1=nn.BatchNorm2d(128)
        self.norm2=nn.BatchNorm2d(256)

        self.norm3=nn.BatchNorm2d(512)

    @abstractmethod
    def build_reduce_layer(self, idx: int):
        """build reduce layer."""
        pass

    @abstractmethod
    def build_upsample_layer(self, idx: int):
        """build upsample layer."""
        pass

    @abstractmethod
    def build_top_down_layer(self, idx: int):
        """build top down layer."""
        pass

    @abstractmethod
    def build_downsample_layer(self, idx: int):
        """build downsample layer."""
        pass

    @abstractmethod
    def build_bottom_up_layer(self, idx: int):
        """build bottom up layer."""
        pass

    @abstractmethod
    def build_out_layer(self, idx: int):
        """build out layer."""
        pass

    def _freeze_all(self):
        """Freeze the model."""
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep the normalization
        layer freezed."""
        super().train(mode)
        if self.freeze_all:
            self._freeze_all()

    def forward(self, inputs: List[torch.Tensor]) -> tuple:
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        inputs=list(inputs)

        # # for idx, i in enumerate(inputs):
        for idx,i in enumerate(inputs):

             if idx == 0:
                # The first scale-aware processing and residual connection
                scale_aware_output = self.scale_aware1(inputs[idx])
                inputs[idx] = scale_aware_output * inputs[idx] + inputs[idx]
                inputs[idx]=self.norm1(inputs[idx])
                # # The second scale-aware processing and residual connection
                # scale_aware_output = self.scale_aware1(inputs[idx])
                # inputs[idx] = scale_aware_output * inputs[idx] + inputs[idx]

             elif idx == 1:
                # The first scale-aware processing and residual connection
                scale_aware_output = self.scale_aware2(inputs[idx])
                inputs[idx] = scale_aware_output * inputs[idx] + inputs[idx]
                inputs[idx]=self.norm2(inputs[idx])
                # # The second scale-aware processing and residual connection
                # scale_aware_output = self.scale_aware2(inputs[idx])
                # inputs[idx] = scale_aware_output * inputs[idx] + inputs[idx]
        
    
             elif idx == 2:
                # The first scale-aware processing and residual connection
                scale_aware_output = self.scale_aware3(inputs[idx])
                inputs[idx] = scale_aware_output * inputs[idx] + inputs[idx]
                inputs[idx]=self.norm3(inputs[idx])
                # # The second scale-aware processing and residual connection
                # scale_aware_output = self.scale_aware3(inputs[idx])
                # inputs[idx] = scale_aware_output * inputs[idx] + inputs[idx]



        inputs=tuple(inputs)
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](inputs[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 -
                                                 idx](
                                                     feat_high)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs)
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](
                torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        return tuple(results)
