'''
Date: 2024-03-24 16:56:03
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-03-24 17:03:01
FilePath: /SparseTeMPO/core/utils.py
'''
"""
Date: 2024-03-24 16:43:34
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-03-24 16:43:34
FilePath: /SparseTeMPO/core/models/utils.py
"""

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

def hidden_register_hook(m, input, output):
    m._recorded_hidden = output


def register_hidden_hooks(model):
    for name, m in model.named_modules():
        if isinstance(m, (nn.ReLU, nn.GELU, nn.Hardswish, nn.ReLU6)):
            m.register_forward_hook(hidden_register_hook)


def get_parameter_group(model, weight_decay=0.0):
    """set weigh_decay to Normalization layers to 0"""
    all_parameters = set(model.parameters())
    group_no_decay = set()

    for m in model.modules():
        if isinstance(m, _BatchNorm):
            if m.weight is not None:
                group_no_decay.add(m.weight)
            if m.bias is not None:
                group_no_decay.add(m.bias)
    group_decay = all_parameters - group_no_decay

    return [
        {"params": list(group_no_decay), "weight_decay": 0.0},
        {"params": list(group_decay), "weight_decay": weight_decay},
    ]