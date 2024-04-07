"""
Description: 
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-05-25 00:45:19
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2023-05-29 18:48:33
"""

from typing import List

from .layers import MetaConv2d
from torch import nn
import torch.nn.functional as F
import torch

__all__ = ["MetaConvCombined"]


class MetaConvCombined(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, path_multiplier, path_depth, device, padding=1):
        super(MetaConvCombined, self).__init__()
        self.metaconv_layer = MetaConv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            path_multiplier=path_multiplier,
            path_depth=path_depth,
            w_bit=32,
            in_bit=32,
            bias=True,
            device=device,
            with_cp=True,
        )
        # The last conv layer after your custom layer
        self.conv2 = nn.Conv2d(out_channel, 64, kernel_size=3, padding=padding)  # Example parameters

        # An adaptive avg pool layer to reduce the spatial dimensions to 1x1
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

        # Dense layer (fully connected) to output 10 classes for MNIST
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.metaconv_layer(x)  # Pass input through metacov layer
        # print(x.shape)
        x = F.relu(self.conv2(F.relu(x)))  # Activation after the last conv layer
        x = self.adaptive_pool(x)  # Reduce spatial dimensions for the dense layer
        x = torch.flatten(x, 1)  # Flatten the output for the dense layer
        x = self.fc(x)  # Final dense layer
        return x