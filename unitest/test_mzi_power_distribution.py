'''
Date: 2024-04-04 14:00:00
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-04-04 14:00:00
FilePath: /SparseTeMPO/unitest/test_layer.py
'''
import torch
from core.models.layers.tempo_conv2d import TeMPOBlockConv2d
from core.models.layers.tempo_linear import TeMPOBlockLinear
import torch.nn.init as init
from core.models.layers.utils import CrosstalkScheduler

def test_conv():
    device = "cuda:0"

    crosstalk_scheduler = CrosstalkScheduler(
        interv_h=20,
        interv_v=120,
        interv_s=10,
    )
    layer = TeMPOBlockConv2d(64, 64, 3, miniblock=(2, 2, 8, 8), crosstalk_scheduler=crosstalk_scheduler, device=device)
    layer.reset_parameters()
    layer.set_input_bitwidth(8)
    layer.set_weight_bitwidth(8)
    # init.normal_(layer.weight, mean=0.0, std=1.0)
    weight = layer.weight_quantizer(layer.weight.data)
    layer.build_phase_from_weight_(weight)
    average_abs_phase = torch.mean(torch.abs(layer.phase))
    # mzi_power = layer.calc_weight_MZI_power()
    print(average_abs_phase)



if __name__ == "__main__":
    test_conv()
    # test_linear()
    print("Pass")
