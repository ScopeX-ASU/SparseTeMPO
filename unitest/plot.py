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


def plot_sparsity(sp_mode, sa_mode):
    set_ms()
    csv_file = f"./Experiment/log/cifar10/SparseBP_MRR_VGG8/sparsity_mode-{sp_mode}_salience_mode-{sa_mode}.csv"
    data = np.loadtxt(csv_file, delimiter=",")
    # [step, cycle, mae, acc]
    steps = data[:, ::4].T
    cycles = data[:, 1::4].T
    maes = data[:, 2::4].T
    accs = data[:, 3::4].T
    sparsitys = np.arange(0.1, 1.01, 0.1)
    print(accs.shape, sparsitys.shape)

    fig, ax = plt.subplots(1, 1)

    name = f"{sp_mode}_{sa_mode}"
    ensure_dir(f"./figures/sparsity")
    # cmap = mpl.colormaps['viridis'].resampled(8)
    cmap = mpl.colormaps["coolwarm"].resampled(8)
    for i, (cycle, acc, sparsity) in enumerate(zip(cycles, accs, sparsitys)):
        fig, ax, _ = batch_plot(
            "line",
            raw_data={"x": cycle / 1000, "y": acc},
            name=name,
            xlabel="Cycles (K)",
            ylabel="Test Acc (%)",
            fig=fig,
            ax=ax,
            xrange=[0, 250.1, 50],
            yrange=[50, 100.1, 10],
            xformat="%d",
            yformat="%d",
            figscale=[0.65, 0.65 * 9.1 / 8],
            fontsize=9,
            linewidth=1,
            gridwidth=0.5,
            trace_color=cmap(sparsity),
            alpha=1,
            trace_label=f"{sparsity}",
            linestyle="-",
            ieee=True,
        )

        set_axes_size_ratio(0.4, 0.5, fig, ax)

        plt.savefig(f"./figures/sparsity/{name}.png", dpi=300)
        plt.savefig(f"./figures/sparsity/{name}.pdf", dpi=300)


def plot_lowrank_scanning2d_resnet():
    fig, ax = None, None

    Bc = np.arange(1, 9, 1)  ## X-axis Col
    Bi = np.arange(1, 9, 1)  ## Y-axis Row
    acc = np.array(
        [
            [82.62, 87.40, 88.80, 89.97, 90.27, 90.36, 90.73, 91.33],
            [83.75, 88.07, 89.09, 89.58, 91.09, 90.92, 91.00, 91.52],
            [84.18, 87.86, 89.08, 89.91, 90.48, 91.02, 91.47, 91.57],
            [83.95, 88.12, 89.37, 90.78, 91.10, 90.99, 91.10, 91.42],
            [83.93, 87.94, 89.75, 90.01, 90.62, 90.73, 91.53, 91.30],
            [84.07, 88.72, 90.20, 90.49, 91.27, 91.28, 91.27, 91.26],
            [84.69, 88.54, 90.16, 90.32, 91.13, 91.08, 91.52, 91.54],
            [85.99, 88.85, 89.89, 90.90, 91.26, 91.18, 91.03, 91.47],
        ]
    )

    name = "Scan2d"
    fig, ax, _ = batch_plot(
        "mesh2d",
        raw_data={"x": Bc, "y": Bi, "z": acc},
        name=name,
        xlabel="Bc",
        ylabel="Bi",
        fig=fig,
        ax=ax,
        xrange=[1, 8.01, 1],
        yrange=[1, 8.01, 1],
        xformat="%.0f",
        yformat="%.0f",
        figscale=[0.65, 0.65 * 9.1 / 8],
        fontsize=10,
        linewidth=1,
        gridwidth=0.5,
        ieee=True,
    )
    X, Y = np.meshgrid(Bc, Bi)
    ct = ax.contour(X, Y, acc, 5, colors="k", linewidths=0.5)
    ax.clabel(ct, fontsize=10, colors="k")

    # ct = ax.contour(X, Y, acc,5,colors='b', linewidths=0.5)
    # ax.clabel(ct,fontsize=10,colors='b')

    fig.savefig(f"{name}.png")
    fig.savefig(f"{name}.pdf")


# plot_lowrank_scanning2d_resnet()


def plot_crosstalk():
    device = "cuda:0"
    layer = TeMPOBlockLinear(8000, 8, miniblock=[1, 1, 8, 8], device=device)

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
        interv_h=25,
        interv_v=120,
        interv_s=9,
        device=device,
    )
    layer.crosstalk_scheduler = crosstalk_scheduler
    # layer.weight.data.fill_(-1)
    weight = layer.build_weight(enable_noise=False, enable_ste=True).detach()
    phase, _ = layer.build_phase_from_weight(weight.data)
    phase = phase.clone()

    weight_nmaes_mean = []
    weight_nmaes_std = []
    phase_nmaes_mean = []
    phase_nmaes_std = []
    layer.set_crosstalk_noise(True)
    interv_h_minmax = crosstalk_scheduler.ps_width + crosstalk_scheduler.interv_s, 41
    interv_h_range = np.arange(*interv_h_minmax, 2)
    with torch.no_grad():
        for interv_h in interv_h_range:
            layer.crosstalk_scheduler.set_spacing(interv_h=interv_h)
            weight_noisy = layer.build_weight(
                enable_noise=True, enable_ste=True
            ).detach()
            phase_noisy = layer.noisy_phase
            print(phase_noisy[0, 0, 0, 0])
            print(phase[0, 0, 0, 0])

            weight_nmae = torch.norm(
                weight_noisy - weight, p=2, dim=(-2, -1)
            ) / weight.norm(1, dim=(-2, -1))
            phase_nmae = torch.norm(
                phase_noisy - phase, p=1, dim=(-2, -1)
            ) / phase.norm(1, dim=(-2, -1))
            weight_nmaes_mean.append(weight_nmae.mean().item())
            weight_nmaes_std.append(weight_nmae.std().item())
            phase_nmaes_mean.append(phase_nmae.mean().item())
            phase_nmaes_std.append(phase_nmae.mean().item())
            print(
                f"interv_h: {interv_h}, N-MAE: weight={weight_nmae.mean().item():.5f} phase={phase_nmae.mean().item():.5f}"
            )
    fig, ax = None, None

    name = "CrosstalkNMAE"
    fig, ax, _ = batch_plot(
        "errorbar",
        raw_data={
            "x": interv_h_range,
            "y": weight_nmaes_mean,
            "yerror": weight_nmaes_std,
        },
        name=name,
        xlabel="lh (um)",
        ylabel="N-MAE",
        fig=fig,
        ax=ax,
        xrange=[interv_h_minmax[0], interv_h_minmax[1] + 0.1, 5],
        xlimit=[interv_h_minmax[0] - 1, interv_h_minmax[1]],
        yrange=[0, 0.25, 0.1],
        xformat="%.0f",
        yformat="%.1f",
        # yscale="log",
        figscale=[0.65, 0.65 * 9.1 / 8],
        fontsize=10,
        linewidth=1,
        gridwidth=0.5,
        ieee=True,
        trace_label="Weight",
        trace_color=color_dict["blue"],
        legend=True,
    )

    fig, ax, _ = batch_plot(
        "errorbar",
        raw_data={
            "x": interv_h_range,
            "y": phase_nmaes_mean,
            "yerror": phase_nmaes_std,
        },
        name=name,
        xlabel="lh (um)",
        ylabel="N-MAE",
        fig=fig,
        ax=ax,
        xrange=[interv_h_minmax[0], interv_h_minmax[1] + 0.1, 5],
        xlimit=[interv_h_minmax[0] - 1, interv_h_minmax[1]],
        yrange=[0, 0.25, 0.1],
        xformat="%.0f",
        yformat="%.1f",
        # yscale="log",
        figscale=[0.65, 0.65 * 9.1 / 8],
        fontsize=10,
        linewidth=1,
        gridwidth=0.5,
        ieee=True,
        trace_label="Phase",
        trace_color=color_dict["mitred"],
        legend=True,
    )
    set_ms()
    ensure_dir(f"./figs")
    fig.savefig(f"./figs/{name}.png")
    fig.savefig(f"./figs/{name}.pdf")
    pdf_crop(f"./figs/{name}.pdf", f"./figs/{name}.pdf")


def plot_spacing():
    device = "cuda:0"
    layer = TeMPOBlockLinear(8000, 8, miniblock=[1, 1, 8, 8], device=device)
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
        interv_h=25,
        interv_v=1200,
        interv_s=10,
        device=device,
    )
    layer.crosstalk_scheduler = crosstalk_scheduler
    # layer.weight.data.fill_(-1)
    weight = layer.build_weight(enable_noise=False, enable_ste=True).detach()
    phase, _ = layer.build_phase_from_weight(weight.data)
    phase = phase.clone()

    weight_nmaes_mean = []
    phase_nmaes_mean = []
    power_mean = []
    layer.set_crosstalk_noise(True)
    interv_s_range = np.arange(7, 25)
    interv_h_range = np.arange(16, 41)
    interv_s_mixmax = (7, 10)
    interv_h_mixmax = (16, 30)
    interv_s_range = np.arange(*interv_s_mixmax)
    interv_h_range = np.arange(*interv_h_mixmax)
    with torch.no_grad():
        for interv_s in interv_s_range:
            weight_nmaes_mean_tmp = []
            for interv_h in interv_h_range:
                try:
                    layer.crosstalk_scheduler.set_spacing(
                        interv_h=interv_h, interv_s=interv_s
                    )
                except:
                    weight_nmaes_mean_tmp.append(0)
                    continue
                weight_noisy = layer.build_weight(
                    enable_noise=True, enable_ste=True
                ).detach()

                # print(phase_noisy[0, 0, 0, 0])
                # print(phase[0, 0, 0, 0])

                weight_nmae = torch.norm(
                    weight_noisy - weight, p=2, dim=(-2, -1)
                ) / weight.norm(1, dim=(-2, -1))
                weight_nmaes_mean_tmp.append(weight_nmae.mean().item())

                print(
                    f"interv_h: {interv_h}, N-MAE: weight={weight_nmae.mean().item():.5f}"
                )
            weight_nmaes_mean.append(weight_nmaes_mean_tmp)
            power = (
                layer.calc_weight_MZI_power(reduction="none")
                .sum(dim=[-4, -3, -2, -1])
                .mean()
                .item()
            )
            power_mean.append(power)
    weight_nmaes_mean = np.array(weight_nmaes_mean)
    print(weight_nmaes_mean.shape)
    print(interv_s_range.shape)
    print(interv_h_range.shape)
    fig, ax = plt.subplots(1, 1)

    name = "PowerCrosstalkSpacing"
    im = ax.pcolormesh(
        interv_s_range,
        interv_h_range,
        weight_nmaes_mean,
        vmin=np.min(weight_nmaes_mean),
        vmax=np.max(weight_nmaes_mean),
        shading="nearest",
        cmap=plt.cm.RdYlGn,
    )
    fig.colorbar(im, ax=ax)

    set_ms()
    ensure_dir(f"./figs")
    fig.savefig(f"./figs/{name}.png")
    fig.savefig(f"./figs/{name}.pdf")
    pdf_crop(f"./figs/{name}.pdf", f"./figs/{name}.pdf")

def plot_light_redist():
    device = "cuda:0"
    sparsity_minmax = (0.1, 1) # density
    sparsity_range = np.arange(*sparsity_minmax, 0.1)
    layer = TeMPOBlockConv2d(64, 64, 3, miniblock=[2, 2, 16, 16], device=device)
    mask = MultiMask(
            {"row_mask": [layer.weight.shape[0], layer.weight.shape[1], layer.weight.shape[2], 1, layer.weight.shape[4], 1], 
            "col_mask": [layer.weight.shape[0], layer.weight.shape[1], 1, layer.weight.shape[3], 1, layer.weight.shape[5]]}, device=device
        )
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
    layer.crosstalk_scheduler = crosstalk_scheduler
    layer.set_crosstalk_noise(True)
    layer.set_output_noise(0.01)
    layer.prune_mask = mask
    nmae_nodist = []
    nmae_dist = []
    for sparsity in sparsity_range:
        nmae_nodist_tmp = []
        nmae_dist_tmp = []
        for random_seed in range(10):
            set_torch_deterministic(random_seed)
            layer.reset_parameters()
            x = torch.randn(1, 64, 32, 32, device=device)*10

            mask["col_mask"].bernoulli_(sparsity**0.5)
            mask["row_mask"].bernoulli_(sparsity**0.5)
            layer.weight.data *= mask.data

            layer.set_noise_flag(False)
            y = layer(x)

            layer.set_noise_flag(True)
            layer.set_light_redist(False)
            set_torch_deterministic(0)
            y1 = layer(x)
            nmae = torch.norm(y1 - y, p=1) / torch.norm(y, p=1)
            nmae_nodist_tmp.append(nmae.item())
            print(f"no redistribuion sparsity {sparsity:.3f} N-MAE: {nmae}")

            layer.set_light_redist(True)
            set_torch_deterministic(0)
            y2 = layer(x)
            nmae = torch.norm(y2 - y, p=1) / torch.norm(y, p=1)
            nmae_dist_tmp.append(nmae.item())
            print(f"with redistribuion sparsity {sparsity:.3f} N-MAE: {nmae}")
        nmae_nodist.append((np.mean(nmae_nodist_tmp), np.std(nmae_nodist_tmp)))
        nmae_dist.append((np.mean(nmae_dist_tmp), np.std(nmae_dist_tmp)))
    nmae_nodist = np.array(nmae_nodist)
    nmae_dist = np.array(nmae_dist)
    print(nmae_nodist)
    print(nmae_dist)


    fig, ax = None, None
    name = "LightRefocusingNoise"

    fig, ax, _ = batch_plot(
        "errorbar",
        raw_data={
            "x": sparsity_range,
            "y": nmae_nodist[:, 0],
            "yerror": nmae_nodist[:, 1],
        },
        name=name,
        xlabel="Sparsity",
        ylabel="N-MAE",
        fig=fig,
        ax=ax,
        xrange=[sparsity_range[0], sparsity_range[-1] + 0.01, 0.2],
        xlimit=[sparsity_range[0]-0.01, sparsity_range[-1]+0.01],
        yrange=[0.05, 0.151, 0.05],
        ylimit=[0.05, 0.16],
        xformat="%.1f",
        yformat="%.2f",
        figscale=[0.65, 0.45 * 9.1 / 8],
        fontsize=10,
        linewidth=1,
        gridwidth=0.5,
        ieee=True,
        trace_label="Prune",
        trace_color=color_dict["mitred"],
        # legend=True,
        # legend_loc="upper right",
    )

    fig, ax, _ = batch_plot(
        "errorbar",
        raw_data={
            "x": sparsity_range,
            "y": nmae_dist[:, 0],
            "yerror": nmae_dist[:, 1],
        },
        name=name,
        xlabel="Sparsity",
        ylabel="N-MAE",
        fig=fig,
        ax=ax,
        xrange=[sparsity_range[0], sparsity_range[-1] + 0.01, 0.2],
        xlimit=[sparsity_range[0]-0.01, sparsity_range[-1]+0.01],
        yrange=[0.05, 0.151, 0.05],
        ylimit=[0.05, 0.16],
        xformat="%.1f",
        yformat="%.2f",
        figscale=[0.65, 0.45 * 9.1 / 8],
        fontsize=10,
        linewidth=1,
        gridwidth=0.5,
        ieee=True,
        trace_label="Prune+Refocusing",
        trace_color=color_dict["blue"],
        # legend=True,
        # legend_loc="upper right",
    )
    set_ms()
    ensure_dir(f"./figs")
    fig.savefig(f"./figs/{name}.png")
    fig.savefig(f"./figs/{name}.pdf")
    pdf_crop(f"./figs/{name}.pdf", f"./figs/{name}.pdf")

if __name__ == "__main__":
    # for sp_mode in ["uniform", "topk", "IS"]:
    #     for sa_mode in ["first_grad", "second_grad"]:
    #         plot_sparsity(sp_mode=sp_mode, sa_mode=sa_mode)
    plot_crosstalk()
    # plot_spacing()
    # plot_light_redist()
