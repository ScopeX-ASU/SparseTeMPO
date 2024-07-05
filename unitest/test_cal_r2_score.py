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
from core.models.layers.utils import MZIPowerEvaluator
from core.models.dst import MultiMask
from core import builder
from pyutils.config import configs
from core.utils import get_parameter_group, register_hidden_hooks
from pyutils.torch_train import set_torch_deterministic


def test_power():
    power_fit = MZIPowerEvaluator()
    power_fit._fit_MZI_power_interp()


if __name__ == "__main__":
    test_power()
