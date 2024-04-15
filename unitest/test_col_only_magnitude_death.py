'''
Date: 2024-04-14 17:40:57
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-04-14 23:09:31
FilePath: /SparseTeMPO/unitest/test_col_only_magnitude_death.py
'''

import torch
from core.models.layers.tempo_linear import TeMPOBlockLinear
from core.models.layers.tempo_conv2d import TeMPOBlockConv2d
from core.models.layers.utils import CrosstalkScheduler, SparsityEnergyScheduler
from core.models.dst import MultiMask
from core import builder
from pyutils.config import configs
from core.utils import get_parameter_group, register_hidden_hooks

def test_DST_scheduler():
    device = "cuda:0"
    configs.load("./configs/dst_test_config/train/sparse_train2.yml", recursive=True)
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

    dst_scheduler.set_magnitude_based_flag(False)
    dst_scheduler.set_gradient_based_flag(False)
    print(dst_scheduler)
    dst_scheduler.apply_mask()
    x = torch.randn(1, 1, 8, 8, device=device)
    y = model(x)
    y.mean().backward()
    # for name, mask in dst_scheduler.masks.items():
    #     weight = dst_scheduler.params[name]
    #     # # print(weight)
    #     # print(weight.shape)
    #     # print(name)
    #     print(mask["row_mask"])
    #     new_mask = dst_scheduler.row_only_magnitude_death(mask, weight, name)
    #     print(new_mask["row_mask"])
    #     break
    dst_scheduler.plot_mask(filename="before_mask", save_fig=True)
    dst_scheduler.update_and_apply_mask()
    dst_scheduler.plot_mask(filename="after_mask", save_fig=True)
        

test_DST_scheduler()