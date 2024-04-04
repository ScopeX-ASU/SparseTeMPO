'''
Date: 2024-04-04 14:00:00
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-04-04 14:00:00
FilePath: /SparseTeMPO/unitest/test_layer.py
'''
import torch
from core.models.layers.tempo_conv2d import TeMPOBlockConv2d
from core.models.layers.tempo_linear import TeMPOBlockLinear


def test_conv():
    device = "cuda:0"
    layer = TeMPOBlockConv2d(4, 4, 3, miniblock=(4, 4, 2, 3), device=device)
    layer.set_input_bitwidth(8)
    layer.set_weight_bitwidth(8)
    x = torch.randn(1, 4, 8, 8, device=device)
    y = layer(x)
    y.mean().backward()
    print(y)


def test_linear():
    device = "cuda:0"
    layer = TeMPOBlockLinear(4, 4, miniblock=(4, 4, 2, 3), device=device)
    layer.set_input_bitwidth(8)
    layer.set_weight_bitwidth(8)
    x = torch.randn(1, 4, device=device)
    y = layer(x)
    y.mean().backward()
    print(y)


if __name__ == "__main__":
    test_conv()
    test_linear()
    print("Pass")
