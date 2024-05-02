import matplotlib.pyplot as plt
import torch
from pyutils.config import configs

from core import builder
from core.models.dst import MultiMask
from core.models.layers.tempo_conv2d import TeMPOBlockConv2d
from core.models.layers.tempo_linear import TeMPOBlockLinear
from core.models.dst2 import MultiMask
from torch import Tensor
from core.models.layers.utils import CrosstalkScheduler, SparsityEnergyScheduler
from core.utils import get_parameter_group, register_hidden_hooks
from pyutils.plot import batch_plot
from pyutils.general import ensure_dir
import numpy as np
from pyutils.torch_train import set_torch_deterministic
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from matplotlib.ticker import NullFormatter
from pyparsing import line
from pyutils.config import configs
from pyutils.general import ensure_dir
from pyutils.torch_train import set_torch_deterministic
from pyutils.plot import batch_plot, pdf_crop, set_axes_size_ratio, set_ms
from sklearn.manifold import TSNE

from core import builder
from core.models.dst import MultiMask
from core.models.layers.tempo_conv2d import TeMPOBlockConv2d
from core.models.layers.tempo_linear import TeMPOBlockLinear
from core.models.layers.utils import CrosstalkScheduler
from core.utils import get_parameter_group, register_hidden_hooks

set_ms()
color_dict = {
    "black": "#000000",
    "red": "#de425b",  # red
    "blue": "#1F77B4",  # blue
    "orange": "#f58055",  # orange
    "yellow": "#f6df7f",  # yellow
    "green": "#2a9a2a",  # green
    "grey": "#979797",  # grey
    "purple": "#AF69C5",  # purple,
    "mitred": "#A31F34",  # mit red
    "pink": "#CDA2BE",
}

def generate_interleave_mask(mask: MultiMask, sparsity: float = 0.5, is_col: bool=False, device="cuda:0"):
    p, q, r, _, k1, _ = mask["row_mask"].shape
    _, _, _, c, _, k2 = mask["col_mask"].shape
    
    total_element = mask.data.numel()
    row_turn_off_number = int(round((total_element - total_element * sparsity) / (c * k2) / (p * q)))
    row_interleave_index = torch.tensor(list(range(0, r*k1, 2)) + list(range(1, r*k1, 2)))[:row_turn_off_number]
    col_turn_off_number = int(round((total_element - total_element * sparsity) / (r * k1) / (p * q)))
    col_interleave_index = torch.tensor(list(range(0, c*k2, 2)) + list(range(1, c*k2, 2)))[:col_turn_off_number]
    row_mask = torch.ones(r*k1, dtype=bool)
    col_mask = torch.ones(c*k2, dtype=bool)
    row_mask[row_interleave_index] = 0
    col_mask[col_interleave_index] = 0
    row_mask = row_mask.reshape(r, 1, k1, 1).expand(p, q, -1, -1, -1, -1)
    col_mask = col_mask.reshape(1, c, 1, k2).expand(p, q, -1, -1, -1, -1)
    # if is_col:
    #     mask["col_mask"] = col_mask.to(device)
    # else:
    #     mask["row_mask"] = row_mask.to(device)
    return col_mask.to(device) if is_col else row_mask.to(device)


def plot_sparsity_interleave():
    device = "cuda:0"
    set_torch_deterministic(0)
    sparsity_minmax = (0.1, 1) # density
    sparsity_range = np.arange(*sparsity_minmax, 0.1)
    layer = TeMPOBlockConv2d(1024, 1024, 1, miniblock=[2, 2, 16, 16], device=device)
    mask = MultiMask(
            {"row_mask": [layer.weight.shape[0], layer.weight.shape[1], layer.weight.shape[2], 1, layer.weight.shape[4], 1], 
            "col_mask": [layer.weight.shape[0], layer.weight.shape[1], 1, layer.weight.shape[3], 1, layer.weight.shape[5]]}, device=device
        )
    # mask["col_mask"].bernoulli_(1)
    # sparsity = 0.6
    # mask["row_mask"].bernoulli_(sparsity)
    # print(row_mask.cpu().numpy().tolist())
    # p, q, r, c, k1, k2 = mask.data.shape
    # print(mask.data.permute(0, 1, 2, 4, 3, 5).reshape(p*q, r*k1, c*k2))
    # return
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
        interv_h=9+6+5,
        interv_v=120,
        interv_s=9,
        device=device,
    )

    crosstalk_scores = []
    # weight_nmaes_std = []
    # phase_nmaes_mean = []
    # phase_nmaes_std = []
    layer.crosstalk_scheduler = crosstalk_scheduler
    # layer.set_crosstalk_noise(True)
    # layer.set_output_noise(0.00)
    # layer.set_output_power_gating(False)
    # layer.prune_mask = mask
    # layer.weight.data *= mask.data
    # set_torch_deterministic(0)
    # x = torch.randn(1, 64, 32, 32, device=device)
    # mask = generate_interleave_mask(mask, 0.7, False, device)
    sparsity_range = np.arange(0, 1.1, 0.1)
    is_col = False
    with torch.no_grad():
        for sparsity in sparsity_range:
            # print(sparsity)
            mask_new = generate_interleave_mask(mask, sparsity, is_col, device)
            # print(mask_new)
            crosstalk_scores.append(layer.crosstalk_scheduler.calc_crosstalk_scores(mask_new[..., 0], is_col=is_col).sum().item())

    # print(crosstalk_scores)

    # exit(0)

    # exit(0)
    fig, ax = None, None
    name = "Crosstalk Scores with Interleave Patterns"
    fig, ax, _ = batch_plot(
        "line",
        raw_data={
            "x": sparsity_range,
            "y": crosstalk_scores,
            # "yerror": weight_nmaes_std,
        },
        name=name,
        xlabel="Sparsity",
        ylabel="Crosstalk Score",
        fig=fig,
        ax=ax,
        xrange=[0, 1.1, 0.2],
        xlimit=[0.1, 0.9],
        yrange=[-150, 10, 30],
        xformat="%.1f",
        yformat="%.1f",
        figscale=[0.65, 0.65 * 9.1 / 8],
        fontsize=10,
        linewidth=1,
        gridwidth=0.5,
        ieee=True,
        trace_marker=".",
        trace_markersize=4,
        # trace_label="Weight",
        # trace_color=color_dict["blue"],
        legend=True,
    )
    ensure_dir(f"./figs")
    fig.savefig(f"./figs/{name}.png")
    fig.savefig(f"./figs/{name}.pdf")
    pdf_crop(f"./figs/{name}.pdf", f"./figs/{name}.pdf")
    # layer.set_noise_flag(False)
    # y = layer(x)

    # layer.set_noise_flag(True)
    # layer.set_output_power_gating(False)
    # set_torch_deterministic(0)
    # y1 = layer(x)
    # nmae = torch.norm(y1 - y, p=1) / torch.norm(y, p=1)

    # print(f"no gating sparsity {sparsity:.3f} N-MAE: {nmae}")

    # layer.set_output_power_gating(True)
    # set_torch_deterministic(0)
    # y2 = layer(x)
    # nmae = torch.norm(y2 - y, p=1) / torch.norm(y, p=1)
    
    # print(f"with gating sparsity {sparsity:.3f} N-MAE: {nmae}")
       

if __name__ == "__main__":
    plot_sparsity_interleave()