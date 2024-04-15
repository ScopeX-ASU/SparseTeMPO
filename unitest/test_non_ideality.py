'''
Date: 2024-03-23 14:04:00
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-04-04 18:49:20
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
from core.models.dst import MultiMask
from core import builder
from pyutils.config import configs
from core.utils import get_parameter_group, register_hidden_hooks

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

def test_output_noise():
    device = "cuda:0"
    layer = TeMPOBlockLinear(32, 32, miniblock=[4, 4, 4, 4], device=device)
    x = torch.randn(1, 32, device=device)
    y = layer(x)
    layer.set_output_noise(0.001)
    layer.set_noise_flag(True)
    y2 = layer(x)
    nmae = torch.norm(y2 - y, p=1) / torch.norm(y, p=1)
    print(f"N-MAE: {nmae}")

    mask = MultiMask({"row_mask": [2,2,4,1,4,1], "col_mask": [2,2,1,4,1,4]}, device=device)
    layer.prune_mask = mask
    layer.set_light_redist(True)
    y3 = layer(x)
    nmae = torch.norm(y3 - y, p=1) / torch.norm(y, p=1)
    print(f"N-MAE: {nmae}")

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

def test_DST_scheduler():
    device = "cuda:0"
    configs.load("./configs/dst_test_config/train/sparse_train.yml", recursive=True)
    model = builder.make_model(
        device,
        model_cfg=configs.model,
        random_state=(
            int(configs.run.random_state) if int(configs.run.deterministic) else None
        ),
    )
    optimizer = builder.make_optimizer(get_parameter_group(model, weight_decay=float(configs.optimizer.weight_decay)),
        name=configs.optimizer.name,
        configs=configs.optimizer,)
    train_loader, validation_loader, test_loader = builder.make_dataloader(
        splits=["train", "valid", "test"]
    )
    dst_scheduler = builder.make_dst_scheduler(optimizer, model, train_loader, configs)
    


# test_mode_switch()
# test_output_noise()
# test_crosstalk()
# test_layer_power_calculator()
test_DST_scheduler()
