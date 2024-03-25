"""
Date: 2024-03-24 15:35:03
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-03-24 16:11:00
FilePath: /SparseTeMPO/unitest/test_model.py
"""

import torch
from core.models import TeMPO_ResNet20, TeMPO_CNN, TeMPO_VGG8


def test_cnn():
    device = "cuda:0"
    model = TeMPO_CNN(
        img_height=32,
        img_width=32,
        in_channels=3,
        num_classes=10,
        kernel_list=[16, 16],
        kernel_size_list=[3, 3],
        stride_list=[1, 1],
        padding_list=[1, 1],
        dilation_list=[1, 1],
        groups=1,
        pool_out_size=2,
        hidden_list=[32],
        conv_cfg=dict(type="TeMPOBlockConv2d", miniblock=(8, 8), mode="weight"),
        linear_cfg=dict(type="TeMPOBlockLinear", miniblock=(8, 8), mode="weight"),
        norm_cfg=dict(type="BN", affine=True),
        act_cfg=dict(type="ReLU", inplace=True),
        device=device,
    ).to(device)
    model.set_weight_bitwidth(8)
    model.set_input_bitwidth(8)
    x = torch.randn(1, 3, 32, 32, device=device)
    y = model(x)
    y.mean().backward()
    print(y)


def test_resnet():
    device = "cuda:0"
    model = TeMPO_ResNet20(
        img_height=32,
        img_width=32,
        in_channels=3,
        num_classes=10,
        conv_cfg=dict(type="TeMPOBlockConv2d", miniblock=(8, 8), mode="weight"),
        linear_cfg=dict(type="TeMPOBlockLinear", miniblock=(8, 8), mode="weight"),
        norm_cfg=dict(type="BN", affine=True),
        act_cfg=dict(type="ReLU", inplace=True),
        device=device,
    ).to(device)
    model.set_weight_bitwidth(8)
    model.set_input_bitwidth(8)
    x = torch.randn(1, 3, 32, 32, device=device)
    y = model(x)
    y.mean().backward()
    print(y)


def test_vgg():
    device = "cuda:0"
    model = TeMPO_VGG8(
        img_height=32,
        img_width=32,
        in_channels=3,
        num_classes=10,
        conv_cfg=dict(type="TeMPOBlockConv2d", miniblock=(8, 8), mode="weight"),
        linear_cfg=dict(type="TeMPOBlockLinear", miniblock=(8, 8), mode="weight"),
        norm_cfg=dict(type="BN", affine=True),
        act_cfg=dict(type="ReLU", inplace=True),
        device=device,
    ).to(device)
    model.set_weight_bitwidth(8)
    model.set_input_bitwidth(8)
    x = torch.randn(1, 3, 32, 32, device=device)
    y = model(x)
    y.mean().backward()
    print(y)


if __name__ == "__main__":
    test_cnn()
    test_vgg()
    test_resnet()
    print("Pass")
