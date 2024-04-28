'''
Date: 2024-04-27 18:51:19
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-04-27 18:51:19
FilePath: /SparseTeMPO/unitest/test_energy.py
'''
"""
Date: 2024-04-23 02:19:51
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-04-27 18:41:38
FilePath: /SparseTeMPO/unitest/test_energy.py
"""

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


def test_energy(
    pruning_type="structure_row", death_mode="magnitude", growth_mode="gradient"
):
    set_torch_deterministic(0)
    device = "cuda:0"
    configs.load("./configs/fmnist/cnn/train/sparse_train.yml", recursive=True)

    model = builder.make_model(
        device,
        model_cfg=configs.model,
        random_state=(
            int(configs.run.random_state) if int(configs.run.deterministic) else None
        ),
    )
    freq = 1  # GHz
    (
        total_energy,  # mJ
        energy_dict,  # layer-wise energy breakdown
        total_cycles,  # total cycles
        cycle_dict,  # layer-wise cycle breakdown
        avg_power,  # average power mW
        power_dict,  # layer-wise power breakdown
    ) = model.calc_weight_MZI_energy([1, 1, 28, 28], R=8, C=8, freq=freq)
    print(f"total weight MZI energy: {total_energy:.4f} mJ")
    print(f"total cycles: {total_cycles} cycles")
    print(f"cycle breakdown:", cycle_dict)
    print(f"average weight MZI power: {avg_power:.4f} mW")
    print(f"weight MZI energy breakdown:", energy_dict)


if __name__ == "__main__":
    test_energy()
