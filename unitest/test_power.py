'''
Date: 2024-04-23 02:19:51
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-04-23 02:19:51
FilePath: /SparseTeMPO/unitest/test_power.py
'''
"""
Date: 2024-04-14 17:40:57
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-04-19 22:08:52
FilePath: /SparseTeMPO/unitest/test_power.py
"""

import torch
from core.models.layers.tempo_linear import TeMPOBlockLinear
from core.models.layers.tempo_conv2d import TeMPOBlockConv2d
from core.models.layers.utils import CrosstalkScheduler, SparsityEnergyScheduler
from core.models.dst import MultiMask
from core import builder
from pyutils.config import configs
from core.utils import get_parameter_group, register_hidden_hooks
from pyutils.torch_train import set_torch_deterministic


def test_power(
    pruning_type="structure_row", death_mode="magnitude", growth_mode="gradient"
):
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
    optimizer = builder.make_optimizer(
        get_parameter_group(model, weight_decay=float(configs.optimizer.weight_decay)),
        name=configs.optimizer.name,
        configs=configs.optimizer,
    )

    configs.dst_scheduler.pruning_type = pruning_type
    configs.dst_scheduler.death_mode = death_mode
    configs.dst_scheduler.growth_mode = growth_mode

    dst_scheduler = builder.make_dst_scheduler(optimizer, model, train_loader, configs)
    mask = torch.tensor([[1, 0, 1, 1, 0, 0, 1, 0]])
    print(dst_scheduler.cal_ports_power(mask))


if __name__ == "__main__":
    test_power()
