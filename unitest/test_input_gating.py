import matplotlib.pyplot as plt
import torch
from pyutils.config import configs

from core import builder
from core.models.dst import MultiMask
from core.models.layers.tempo_conv2d import TeMPOBlockConv2d
from core.models.layers.tempo_linear import TeMPOBlockLinear
from core.models.dst2 import MultiMask
from core.models.layers.utils import CrosstalkScheduler, SparsityEnergyScheduler
from core.utils import get_parameter_group, register_hidden_hooks
from pyutils.plot import batch_plot
import numpy as np
from pyutils.torch_train import set_torch_deterministic

def test_input_gating():
    device = "cuda:0"
    set_torch_deterministic(0)
    sparsity_minmax = (0.1, 1) # density
    sparsity_range = np.arange(*sparsity_minmax, 0.1)
    layer = TeMPOBlockConv2d(64, 64, 3, miniblock=[2, 2, 16, 16], device=device)
    mask = MultiMask(
            {"row_mask": [layer.weight.shape[0], layer.weight.shape[1], layer.weight.shape[2], 1, layer.weight.shape[4], 1], 
            "col_mask": [layer.weight.shape[0], layer.weight.shape[1], 1, layer.weight.shape[3], 1, layer.weight.shape[5]]}, device=device
        )
    
    sparsity = 0.6
    mask["col_mask"].bernoulli_(sparsity)
    mask["row_mask"].bernoulli_(1)
    crosstalk_scheduler = CrosstalkScheduler(
        crosstalk_coupling_factor=[
            3.55117528e-07,
            -1.55789201e-05,
            -8.29631681e-06,
            9.89616761e-03,
            -1.76013871e-01,
            1,
        ],  # y=p1*x^5+p2*x^4+p3*x^3+p4*x^2+p5*x+p6
        crosstalk_exp_coupling_factor=[
            0.2167267,
            -0.12747211,
        ],  # a * exp(b*x)
        interv_h=7+6+1,
        interv_v=120,
        interv_s=7,
        device=device,
    )
    layer.crosstalk_scheduler = crosstalk_scheduler
    layer.set_crosstalk_noise(True)
    layer.set_output_noise(0.01)
    layer.set_output_power_gating(True)
    layer.prune_mask = mask
    layer.weight.data *= mask.data
    set_torch_deterministic(0)
    x = torch.randn(1, 64, 32, 32, device=device) * 10
    

    layer.set_noise_flag(False)
    y = layer(x)

    layer.set_noise_flag(True)
    layer.set_input_power_gating(False)
    """ after prune, weights
    0.9 0 1.4
    0.1 0 4.1
    1.3 0 1.1
    """

    """ after prune and crosstalk
    1.3 0.0001 1.7
    1.1 0.0001 3.6
    0.6 0.0001 1.6
    """

    """ after prune and crosstalk with input gating ER=10
    1.3 0.00001 1.7
    1.1 0.00001 3.6
    0.6 0.00001 1.6
    """ 
    # "0" weight noises reduced by 10x
    ## input gating has no impact on output PD noises.
    
    set_torch_deterministic(0)
    y1 = layer(x)
    nmae = torch.norm(y1 - y, p=1) / torch.norm(y, p=1)

    print(f"no gating sparsity {sparsity:.3f} N-MAE: {nmae}")

    # layer.set_output_power_gating(True)
    layer.set_input_power_gating(True)
    set_torch_deterministic(0)
    y2 = layer(x)
    nmae = torch.norm(y2 - y, p=1) / torch.norm(y, p=1)
    
    print(f"with gating sparsity {sparsity:.3f} N-MAE: {nmae}")

    layer.set_light_redist(True)
    set_torch_deterministic(0)
    y3 = layer(x)
    nmae = torch.norm(y3 - y, p=1) / torch.norm(y, p=1)
    
    print(f"with gating with redist sparsity {sparsity:.3f} N-MAE: {nmae}")
       

if __name__ == "__main__":
    test_input_gating()