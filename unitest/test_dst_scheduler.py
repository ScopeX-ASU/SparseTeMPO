'''
Date: 2024-04-14 17:40:57
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-04-15 00:45:24
FilePath: /SparseTeMPO/unitest/test_dst_scheduler.py
'''

import torch
from core.models.layers.tempo_linear import TeMPOBlockLinear
from core.models.layers.tempo_conv2d import TeMPOBlockConv2d
from core.models.layers.utils import CrosstalkScheduler, SparsityEnergyScheduler
from core.models.dst import MultiMask
from core import builder
from pyutils.config import configs
from core.utils import get_parameter_group, register_hidden_hooks
from pyutils.torch_train import set_torch_deterministic

def test_DST_scheduler(pruning_type="structure_row", death_mode="magnitude", growth_mode="gradient"):
    set_torch_deterministic(0)
    device = "cuda:0"
    configs.load("./configs/dst_test_config/train/sparse_train2.yml", recursive=True)
    
    train_loader, validation_loader, test_loader = builder.make_dataloader(
        splits=["train", "valid", "test"]
    )
    configs.dataset.in_channels = 16
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
    
    configs.dst_scheduler.pruning_type = pruning_type
    configs.dst_scheduler.death_mode = death_mode
    configs.dst_scheduler.growth_mode = growth_mode
    
    dst_scheduler = builder.make_dst_scheduler(optimizer, model, train_loader, configs)

    dst_scheduler.set_magnitude_based_flag(False)
    dst_scheduler.set_gradient_based_flag(False)
    print(dst_scheduler)
    dst_scheduler.apply_mask()
    x = torch.randn(1, 16, 8, 8, device=device)
    y = model(x)
    y.square().sum().backward()
    # for name, mask in dst_scheduler.masks.items():
    #     weight = dst_scheduler.params[name]
    #     # # print(weight)
    #     # print(weight.shape)
    #     # print(name)
    #     print(mask["row_mask"])
    #     new_mask = dst_scheduler.row_only_magnitude_death(mask, weight, name)
    #     print(new_mask["row_mask"])
    #     break
    pruning_type = dst_scheduler.pruning_type
    death_mode = dst_scheduler.death_mode
    growth_mode = dst_scheduler.growth_mode
    dst_scheduler.plot_mask(filename=f"{pruning_type}_D-{death_mode}_G-{growth_mode}_before_mask", save_fig=True)
    dst_scheduler.update_death_mask()
    dst_scheduler.plot_mask(filename=f"{pruning_type}_D-{death_mode}_G-{growth_mode}_after_death_mask", save_fig=True)
    dst_scheduler.update_growth_mask()
    dst_scheduler.apply_mask()
    dst_scheduler.plot_mask(filename=f"{pruning_type}_D-{death_mode}_G-{growth_mode}_after_growth_mask", save_fig=True)
        
if __name__ == "__main__":

    test_DST_scheduler(
        pruning_type="structure_row",
        death_mode="magnitude_crosstalk_power",
        growth_mode="gradient_crosstalk_power",
    )

    test_DST_scheduler(
        pruning_type="structure_row",
        death_mode="magnitude_power_crosstalk",
        growth_mode="gradient_power_crosstalk",
    )

    test_DST_scheduler(
        pruning_type="structure_col",
        death_mode="magnitude_crosstalk_power",
        growth_mode="gradient_crosstalk_power",
    )

    test_DST_scheduler(
        pruning_type="structure_col",
        death_mode="magnitude_power_crosstalk",
        growth_mode="gradient_power_crosstalk",
    )

    