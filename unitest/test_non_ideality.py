'''
Date: 2024-03-23 14:04:00
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-03-24 15:48:25
FilePath: /SparseTeMPO/unitest/test_non_ideality.py
'''
"""
Description:    
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 2023-10-03 23:49:27
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-03-23 14:01:47
"""

import torch
from core.models.layers.tempo_linear import TeMPOBlockLinear
from core.models.layers.tempo_conv2d import TeMPOBlockConv2d
from core.models.layers.utils import CrosstalkScheduler


def test_mode_switch():
    device = "cuda:0"
    layer = TeMPOBlockLinear(4, 4, device=device)
    weight = layer.weight.clone()
    print(layer.weight)
    layer.sync_parameters(src="weight")
    print(layer.phase)
    layer.sync_parameters(src="phase")
    print(layer.weight)
    assert torch.allclose(layer.weight, weight)


def test_crosstalk():
    device = "cuda:0"
    k = 0.05
    layer = TeMPOBlockLinear(4, 4, miniblock=[2,3], device=device)
    crosstalk_scheduler = CrosstalkScheduler(
        crosstalk_coupling_factor=k, interv_h=100, interv_v=100, device=device
    )
    layer.crosstalk_scheduler = crosstalk_scheduler
    weight = layer.build_weight(enable_noise=False, enable_ste=True)
    
    layer.set_crosstalk_noise(True)
    weight_noisy = layer.build_weight(enable_noise=True, enable_ste=True)
    print(weight)
    print(weight_noisy)
    nmae = torch.norm(weight_noisy - weight, p=1) / torch.norm(weight, p=1)
    print(f"crosstalk coefficient k: {k}, N-MAE: {nmae}")


test_mode_switch()
test_crosstalk()
