'''
Date: 2024-03-23 14:04:00
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-04-04 14:24:22
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
from core.models.layers.utils import CrosstalkScheduler, SparsityEnergyScheduler


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
    layer = TeMPOBlockLinear(16, 16, miniblock=[4, 4, 4, 4], device=device)
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


def test_layer_power_calculator():
    device = "cuda:0"
    core_size = [8, 8]
    layer = TeMPOBlockLinear(4, 4, miniblock=[2,3], device=device)
    power_calculator = SparsityEnergyScheduler(core_size=[8, 8], threshold=15, pi_shift_power=30.0, device=device)
    layer.switch_power_scheduler = power_calculator
    weight0 = layer.build_weight(enable_noise=False, enable_ste=True)
    weight1 = layer.weight
    layer.weight.data.fill_(0)
    weight2 = layer.weight
    print(weight1)
    print(weight2)
    # layer.set_switch_power_count(True)
    # power = layer.cal_switch_power(weight=None, src="phase")
    # print(power)

# test_mode_switch()
test_crosstalk()
# test_layer_power_calculator()
