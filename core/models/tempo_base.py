"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-10-24 16:23:19
LastEditors: ScopeX-ASU jiaqigu@asu.edu
LastEditTime: 2023-10-28 22:57:34
"""

import inspect
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn.bricks import build_activation_layer, build_conv_layer, build_norm_layer
from mmengine.registry import MODELS
from pyutils.general import TimerCtx, logger
from pyutils.torch_train import set_torch_deterministic
from torch import Tensor, nn  # , set_deterministic
from torch.types import Device, _size
from torchonn.op.mrr_op import *

__all__ = [
    "LinearBlock",
    "ConvBlock",
    "TeMPO_Base",
]


def build_linear_layer(cfg: Optional[Dict], *args, **kwargs) -> nn.Module:
    if cfg is None:
        cfg_ = dict(type="Linear")
    else:
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be a dict")
        if "type" not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if inspect.isclass(layer_type):
        return layer_type(*args, **kwargs, **cfg_)  # type: ignore
    # Switch registry to the target scope. If `linear_layer` cannot be found
    # in the registry, fallback to search `linear_layer` in the
    # mmengine.MODELS.
    with MODELS.switch_scope_and_registry(None) as registry:
        linear_layer = registry.get(layer_type)
    if linear_layer is None:
        raise KeyError(
            f"Cannot find {linear_layer} in registry under scope "
            f"name {registry.scope}"
        )
    layer = linear_layer(*args, **kwargs, **cfg_)

    return layer


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        linear_cfg: dict = dict(type="TeMPOBlockLinear"),
        norm_cfg: dict | None = None,
        act_cfg: dict | None = dict(type="ReLU", inplace=True),
        dropout: float = 0.0,
        device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=False) if dropout > 0 else None
        if linear_cfg["type"] not in {"Linear", None}:
            linear_cfg.update({"device": device})
        self.linear = build_linear_layer(
            linear_cfg,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, out_features)
        else:
            self.norm = None

        if act_cfg is not None:
            self.activation = build_activation_layer(act_cfg)
        else:
            self.activation = None

    def forward(self, x: Tensor) -> Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: Union[int, _size] = 1,
        padding: Union[int, _size] = 0,
        dilation: Union[int, _size] = 1,
        groups: int = 1,
        bias: bool = False,
        conv_cfg: dict = dict(type="TeMPOBlockConv2d"),
        norm_cfg: dict | None = dict(type="BN", affine=True),
        act_cfg: dict | None = dict(type="ReLU", inplace=True),
        device: Device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ) -> None:
        super().__init__()
        conv_cfg = conv_cfg.copy()
        if conv_cfg["type"] not in {"Conv2d", None}:
            conv_cfg.update({"device": device})
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if norm_cfg is not None:
            _, self.norm = build_norm_layer(norm_cfg, out_channels)
        else:
            self.norm = None

        if act_cfg is not None:
            self.activation = build_activation_layer(act_cfg)
        else:
            self.activation = None

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class TeMPO_Base(nn.Module):
    def __init__(
        self,
        *args,
        conv_cfg=dict(type="TeMPOBlockConv2d"),
        linear_cfg=dict(type="TeMPOBlockLinear"),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        with MODELS.switch_scope_and_registry(None) as registry:
            self._conv = (registry.get(conv_cfg["type"]),)
            self._linear = (registry.get(linear_cfg["type"]),)
            self._conv_linear = self._conv + self._linear

    def reset_parameters(self, random_state: int = None) -> None:
        for name, m in self.named_modules():
            if isinstance(m, self._conv_linear):
                if random_state is not None:
                    # deterministic seed, but different for different layer, and controllable by random_state
                    set_torch_deterministic(random_state + sum(map(ord, name)))
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def backup_phases(self) -> None:
        self.phase_backup = {}
        for layer_name, layer in self.fc_layers.items():
            self.phase_backup[layer_name] = {
                "weight": (
                    layer.weight.data.clone() if layer.weight is not None else None
                ),
            }

    def restore_phases(self) -> None:
        for layer_name, layer in self.fc_layers.items():
            backup = self.phase_backup[layer_name]
            for param_name, param_src in backup.items():
                param_dst = getattr(layer, param_name)
                if param_src is not None and param_dst is not None:
                    param_dst.data.copy_(param_src.data)

    def set_phase_variation(self, flag: bool = True):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_phase_variation"
            ):
                layer.set_phase_variation(flag)

    def set_output_noise(self, noise_std: float = 0.0):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_output_noise"
            ):
                layer.set_output_noise(noise_std)

    def set_global_temp_drift(self, flag: bool = True):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_global_temp_drift"
            ):
                layer.set_global_temp_drift(flag)

    def set_light_redist(self, flag: bool = True):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_light_redist"
            ):
                layer.set_light_redist(flag)

    def set_input_power_gating(self, flag: bool = True, ER: float = 6) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_input_power_gating"
            ):
                layer.set_input_power_gating(flag, ER)
    
    def set_output_power_gating(self, flag: bool = True) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_output_power_gating"
            ):
                layer.set_output_power_gating(flag)

    def set_gamma_noise(
        self, noise_std: float = 0.0, random_state: Optional[int] = None
    ) -> None:
        self.gamma_noise_std = noise_std
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_gamma_noise"
            ):
                layer.set_gamma_noise(noise_std, random_state=random_state)

    def set_crosstalk_noise(self, flag: bool = True) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_crosstalk_noise"
            ):
                layer.set_crosstalk_noise(flag)

    def set_weight_noise(self, noise_std: float = 0.0) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_weight_noise"
            ):
                layer.set_weight_noise(noise_std)

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_weight_bitwidth"
            ):
                layer.set_weight_bitwidth(w_bit)

    def set_input_bitwidth(self, in_bit: int) -> None:
        self.in_bit = in_bit
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "set_input_bitwidth"
            ):
                layer.set_input_bitwidth(in_bit)

    def load_parameters(self, param_dict: Dict[str, Dict[str, Tensor]]) -> None:
        """
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
        """
        for layer_name, layer_param_dict in param_dict.items():
            self.layers[layer_name].load_parameters(layer_param_dict)

    def build_obj_fn(self, X: Tensor, y: Tensor, criterion: Callable) -> Callable:
        def obj_fn(X_cur=None, y_cur=None, param_dict=None):
            if param_dict is not None:
                self.load_parameters(param_dict)
            if X_cur is None or y_cur is None:
                data, target = X, y
            else:
                data, target = X_cur, y_cur
            pred = self.forward(data)
            return criterion(pred, target)

        return obj_fn

    def enable_fast_forward(self) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "enable_fast_forward"
            ):
                layer.enable_fast_forward()

    def disable_fast_forward(self) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "disable_fast_forward"
            ):
                layer.disable_fast_forward()

    def sync_parameters(self, src: str = "weight") -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "sync_parameters"
            ):
                layer.sync_parameters(src=src)

    def build_weight(self) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(layer, "build_weight"):
                layer.build_weight()

    def print_parameters(self) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "print_parameters"
            ):
                layer.print_parameters()

    def gen_mixedtraining_mask(
        self,
        sparsity: float,
        prefer_small: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Dict[str, Tensor]]:
        return {
            layer_name: layer.gen_mixedtraining_mask(
                sparsity, prefer_small, random_state
            )
            for layer_name, layer in self.named_modules()
            if isinstance(layer, self._conv_linear)
        }

    def switch_mode_to(self, mode: str) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear) and hasattr(
                layer, "switch_mode_to"
            ):
                layer.switch_mode_to(mode)

    def get_power(self, mixedtraining_mask: Optional[Tensor] = None) -> float:
        power = sum(
            layer.get_power(mixedtraining_mask[layer_name])
            for layer_name, layer in self.fc_layers.items()
            if hasattr(layer, "get_power")
        )
        return power

    def set_noise_schedulers(
        self,
        scheduler_dict={
            "phase_variation_scheduler": None,
            "global_temp_scheduler": None,
            "crosstalk_scheduler": None,
        },
    ):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                for scheduler_name, scheduler in scheduler_dict.items():
                    setattr(layer, scheduler_name, scheduler)

        for scheduler_name, scheduler in scheduler_dict.items():
            setattr(self, scheduler_name, scheduler)

    def reset_noise_schedulers(self):
        self.phase_variation_scheduler.reset()
        self.global_temp_scheduler.reset()
        self.crosstalk_scheduler.reset()

    def step_noise_scheduler(self, T=1):
        if self.phase_variation_scheduler is not None:
            for _ in range(T):
                self.phase_variation_scheduler.step()

        if self.global_temp_scheduler is not None:
            for _ in range(T):
                self.global_temp_scheduler.step()

    def backup_ideal_weights(self):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer._ideal_weight = layer.weight.detach().clone()

    def cycles(self, x_size, R: int = 8, C: int = 8) -> float:
        x = torch.randn(x_size, device=self.device)
        self.eval()

        def hook(m, inp):
            m._input_shape = inp[0].shape

        handles = []
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                handle = layer.register_forward_pre_hook(hook)
                handles.append(handle)
        with torch.no_grad():
            self.forward(x)
        cycles = {} # name: (cycles_per_block, total_cycles)
        for name, layer in self.named_modules():
            if isinstance(layer, self._conv_linear):
                cycles[name] = layer.cycles(layer._input_shape, R=R, C=C)
        for handle in handles:
            handle.remove()
        return np.sum(list(cycles.values())), cycles

    def gen_sparsity_mask(self, sparsity=1.0, mode="topk"):
        # top sparsity% will be calibrated
        ## this sparsity is for each layer, every layer, there are sparsity% blocks are trained. not selected from the whole network.
        ## return [P,Q] boolean mask, 1 represents the kxk block is important, 0 means not important.
        R, C, k = 4, 4, 8  # torch.ones([4,4,8,8]).size[0:3]
        self._sparsity = sparsity
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                # [p,q,k,k] # larger in absolute values mean more important
                salience = layer.weight._salience
                P, Q = salience.shape[0:2]
                shape = int(np.ceil(P / R) * R), int(
                    np.ceil(Q / C) * C
                )  # [P_pad, Q_pad]
                salience = torch.nn.functional.pad(
                    salience,
                    (0, 0, 0, 0, 0, shape[1] - Q, 0, shape[0] - P),
                    mode="constant",
                    value=0,
                )  # [P_pad, Q_pad, k, k]
                salience = (
                    salience.reshape(shape[0] // R, R, shape[1] // C, C, k, k)
                    .permute(0, 2, 1, 3, 4, 5)
                    .flatten(2)
                    .norm(1, dim=-1)
                )  # [shape[0]//R, shape[1]//C]

                if mode == "topk":
                    threshold = torch.quantile(
                        salience.flatten(), q=1 - sparsity, dim=0
                    )  # [1]
                    # 1 means the Rk x Ck block is important, 0 means not important, # [shape[0]//R, shape[1]//C]
                    mask = salience >= threshold

                elif mode == "IS":
                    mask = torch.zeros_like(salience.flatten())
                    sample_IS = np.random.choice(
                        a=len(list(salience.cpu().detach().numpy().flatten())),
                        size=round(max(1, salience.numel() * sparsity)),
                        replace=False,
                        p=salience.cpu().detach().numpy().flatten()
                        / salience.cpu()
                        .detach()
                        .numpy()
                        .flatten()
                        .sum(),  # / salience.cpu().detach().numpy().flatten().sum()]
                    )

                    for i in range(len(sample_IS)):
                        mask[sample_IS[i]] = 1
                    mask = mask.view_as(salience)

                elif mode == "uniform":
                    mask = torch.zeros_like(salience.flatten())
                    sample_IS = np.random.choice(
                        a=len(list(salience.cpu().detach().numpy().flatten())),
                        size=round(max(1, salience.numel() * sparsity)),
                        replace=False,
                        p=np.ones_like(salience.cpu().detach().numpy().flatten())
                        / len(
                            salience.cpu().detach().numpy().flatten()
                        ),  # / salience.cpu().detach().numpy().flatten().sum()]
                    )

                    for i in range(len(sample_IS)):
                        mask[sample_IS[i]] = 1
                    mask = mask.view_as(salience)

                # [P, Q], 1 represents the kxk block is important
                mask = (
                    mask[:, None, :, None]
                    .repeat(1, R, 1, C)
                    .reshape(shape[0], shape[1])[:P, :Q]
                )
                layer.weight._sparsity_mask = mask
                layer.weight._sparsity = sparsity

    def gen_weight_salience(self, mode="first_grad"):
        assert mode in {"magnitude", "first_grad", "second_grad"}
        ## need a dictionary, key is layer, value is salience scores
        ## assume salience scores are of the same shape as weight, i.e., [p,q,k,k]
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                if mode == "magnitude":
                    layer.weight._salience = layer.weight.abs()
                elif mode == "first_grad":
                    layer.weight._salience = layer.weight._first_grad.abs()
                elif mode == "second_grad":
                    layer.weight._salience = layer.weight._second_grad.abs()
                else:
                    raise ValueError(f"Unknown mode {mode}")

    def map_to_hardware(
        self,
        input_shape,
        lr=1e-2,
        num_steps=100,
        stop_thres=None,
        average_times=1,
        criterion="nmae",
        verbose: bool = True,
        sparsity: float = 1.0,
        sparsity_mode: str = "uniform",
        validation_callback=None,
    ):
        ## solve min \sum |E[W]-W'| to calibrate latent self.weight, such that E[noisy_weight = self.build_weight()] is close to self._ideal_weight
        ## self.build_weight() only requires input identity matrix to probe the circuit transfer matrix, which is much cheaper then conv2d layers.
        ## therefore, we count the forward times, and only step() when the forward times equal to the one inference.
        ## min_{W} |E[W_noisy] - W_ideal|. We can only get dL/E[dW_noisy], we use this as an approximation of dL/dW, and use the gradient to update W.
        ## every times you build_weight and accumulate it to average_weight, it takes k+1 cycles
        ## average_times forward takes average_times * (k-1) cycles
        ## if no randomness, e.g., only have static crosstalk/temperature drift, average_times=1
        assert criterion in {
            "mse",
            "mae",
            "nmse",
            "nmae",
            "first-order",
            "second-order",
        }
        if criterion == "mse":
            loss_fn = torch.nn.functional.mse_loss
            one_prob_cycles = self.probe_cycles() * average_times
        elif criterion == "mae":
            loss_fn = torch.nn.functional.l1_loss
            one_prob_cycles = self.probe_cycles() * average_times
        elif criterion == "nmse":
            loss_fn = (
                lambda x, target: x.sub(target).square().sum() / target.square().sum()
            )
            one_prob_cycles = self.probe_cycles() * average_times
        elif criterion == "nmae":
            loss_fn = lambda x, target: x.sub(target).norm(p=1) / target.norm(p=1)
            one_prob_cycles = self.probe_cycles() * average_times
        elif criterion == "first-order":
            loss_fn = lambda x, target, grad: (
                grad * (x - target)
            ).sum().abs() + torch.nn.functional.l1_loss(x, target)
            one_prob_cycles = (
                self.probe_cycles(num_vectors=1) + self.probe_cycles() * average_times
            )  # extra 2 cycle
        elif criterion == "second-order":

            def loss_fn(x, target, grad, second_grad):
                error = x - target
                return (
                    (grad * error).sum() + 0.5 * (error.square() * second_grad).sum()
                ).abs() + +torch.nn.functional.l1_loss(x, target)

            ## every window, it inceases 1 cycle
            ## #block * 1 + #block * probe_vectors * average_times
            one_prob_cycles = (
                self.probe_cycles(num_vectors=2) + self.probe_cycles() * average_times
            )  # extra 2 cycle
        one_prob_cycles = int(round(one_prob_cycles * getattr(self, "_sparsity", 1)))
        metric_fn = lambda x, target: x.sub(target).norm(p=1) / target.norm(p=1)
        self.backup_ideal_weights()
        if verbose:
            logger.info(f"Mapping ideal weights to noisy hardware")
            logger.info(f"Backup ideal weights...")
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, num_steps, eta_min=0.5 * lr
        )
        one_inference_cycles = self.cycles(input_shape)
        cycle_count = 0
        total_cycle = 0
        total_steps = 0
        if verbose:
            logger.info(
                f"lr: {lr:.2e} average_times: {average_times} #Cycles per inference: {one_inference_cycles:5d}. #Cycles per probe: {one_prob_cycles:5d}."
            )

        for step in range(num_steps):
            loss = []
            metric = []
            self.gen_sparsity_mask(sparsity, sparsity_mode)
            for layer in self.modules():
                if isinstance(layer, self._conv_linear):
                    noisy_weight = (
                        sum(
                            [
                                layer.build_weight(enable_ste=True, enable_remap=True)
                                for _ in range(average_times)
                            ]
                        )
                        / average_times
                    )
                    if hasattr(layer.weight, "_sparsity_mask"):
                        sparsity_mask = (
                            layer.weight._sparsity_mask.flatten().nonzero().flatten()
                        )  # [nonzero]
                        noisy_weight = noisy_weight.flatten(0, 1)[
                            sparsity_mask
                        ]  # [nonzero, k, k]
                        ideal_weight = layer._ideal_weight.flatten(0, 1)[sparsity_mask]
                        # print(layer.weight.shape, layer.weight.shape[0] * layer.weight.shape[1], sparsity_mask)
                    else:
                        sparsity_mask = None
                        ideal_weight = layer._ideal_weight

                    if criterion == "first-order":
                        grad = (
                            layer.weight._first_grad
                        )  # need to be preloaded and stored
                        if sparsity_mask is not None:
                            grad = grad.flatten(0, 1)[sparsity_mask]
                        loss.append(loss_fn(noisy_weight, ideal_weight, grad))
                    elif criterion == "second-order":
                        grad = layer.weight._first_grad
                        second_grad = layer.weight._second_grad
                        if sparsity_mask is not None:
                            grad = grad.flatten(0, 1)[sparsity_mask]
                            second_grad = second_grad.flatten(0, 1)[sparsity_mask]
                        loss.append(
                            loss_fn(noisy_weight, ideal_weight, grad, second_grad)
                        )
                    else:
                        loss.append(loss_fn(noisy_weight, ideal_weight))

                    metric.append(
                        metric_fn(noisy_weight.detach(), ideal_weight).cpu().numpy()
                    )
                    # loss = loss + sum([torch.nn.functional.l1_loss(layer._ideal_weight, layer.build_weight(enable_ste=True)) for _ in range(average_times)]) / average_times
            cycle_count += one_prob_cycles
            total_cycle += one_prob_cycles
            if cycle_count >= one_inference_cycles:
                T = int(cycle_count // one_inference_cycles)
                self.step_noise_scheduler(T)
                total_steps += T
                cycle_count -= T * one_inference_cycles
            loss = sum(loss) / len(loss)
            if validation_callback is not None:
                logger.info(
                    f"Step: {step}, #Cycle: {total_cycle}, {criterion} loss: {loss.item():.2e}"
                )
                validation_callback(step, total_cycle, loss.item())

            if stop_thres is not None and loss.mean() < stop_thres:
                break

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step()

            if verbose and (step % 5 == 0 or step == num_steps - 1):
                logger.info(
                    f"Step: {step:4d} cycle: {total_cycle:.3e}({cycle_count:4d} / {one_inference_cycles:4d}, {total_steps:3d} noise steps) {criterion} loss: {loss.item():.2e} NMAE: (mean) {np.mean(metric):.2e} (std) {np.std(metric):.2e}"
                )

        optimizer.zero_grad()
        self.sync_parameters(src="weight")
        ## remember to record last temperature
        self.global_temp_scheduler.record_current_temp()
        if verbose:
            logger.info(
                f"Finish calibration, current temp: {self.global_temp_scheduler.T} K"
            )

    def remap(
        self,
        input_shape,
        flag: bool,
        alg: str = "LAP",
        salience_mode: str = "heuristic",
        average_times: int = 1,
        tolerance: float = 1,
        verbose: bool = True,
        enable_step: bool = True,
    ):
        total_cycles = 0
        cycle_count = 0
        one_inference_cycles = self.cycles(input_shape)
        if flag:
            for layer in self.modules():
                if isinstance(layer, self._conv_linear):
                    _, _, cycles = layer.remap_intra_tile(
                        alg=alg,
                        salience_mode=salience_mode,
                        average_times=average_times,
                        tolerance=tolerance,
                    )
                    cycle_count += cycles
                    total_cycles += cycles
                    if cycle_count >= one_inference_cycles:
                        T = int(cycle_count // one_inference_cycles)
                        if enable_step:
                            self.step_noise_scheduler(T)
                        cycle_count -= T * one_inference_cycles

            self.global_temp_scheduler.record_current_temp()
            if verbose:
                logger.info(
                    f"Finish remapping, total cycles: {total_cycles}, current temp: {self.global_temp_scheduler.T} K"
                )

        return total_cycles

        # perform intra tile remapping in layer here
        # input: layer.weight partitioned into (bs_x, bs_y) batches, with each batch taking (R,C,K,K) weights and remapping them on tile
        # For each (R,C,K,K) weight bank, we only change the R indexs, e.g. (1,2,3,4) -> (3,2,1,4) to minimize the E|W_tilde - layer.weight|

    def is_map_from_temp(self) -> bool:
        is_remap = False
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                temp_drift = (
                    layer.global_temp_scheduler.get_global_temp()
                    - layer.global_temp_scheduler.T0
                )
                if temp_drift > 0.2:
                    is_remap = True
        return is_remap

    def probe_weight_error(self) -> Tensor:
        self.backup_ideal_weights()
        sum_error = 0.0
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer_error = (layer.build_weight() - layer._ideal_weight).norm(p=1)
                sum_error += layer_error
        return sum_error

    def set_enable_ste(self, enable_ste: bool) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer._enable_ste = enable_ste

    def set_noise_flag(self, noise_flag: bool) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer._noise_flag = noise_flag

    def set_enable_remap(self, enable_remap: bool) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.set_enable_remap(enable_remap)

    def calc_weight_MZI_energy(
        self,
        input_size=[1, 3, 32, 32],
        R: int = 8,
        C: int = 8,
        freq: float=1.0, # GHz
    ) -> None:
        ## return total energy in mJ and power mW breakdown

        total_cycles, cycle_dict = self.cycles(input_size, R=R, C=C)
        power_dict = {}
        energy_dict = {}
        with torch.no_grad():
            for name, layer in self.named_modules():
                if isinstance(layer, self._conv_linear) and hasattr(
                    layer, "calc_weight_MZI_power"
                ):
                    power = layer.calc_weight_MZI_power(
                        src="weight", reduction="none"
                    )  # [p,q,r,c,k1,k1] -> 1 # mW

                    ## calculate energy
                    ## (P1*cyc_per_clk + P2*cyc_per_clk + ... + P_{RC} * cyc_per_clk) / freq
                    ## (P1+P2+P3+...+P_{RC}) * cyc_per_clock / freq
                    ## sum(P) * cyc_per_clock / freq
                    power = power.sum().item()
                    power_dict[name] = power
                    cycles_per_block = cycle_dict[name][0]
                    energy_dict[name] = power * cycles_per_block / freq / 1e9 # mJ
        total_energy = np.sum(list(energy_dict.values()))
        avg_power = total_energy / (total_cycles / freq / 1e9)
        return (
            total_energy, #mJ
            energy_dict, # layer-wise energy breakdown
            total_cycles, # total cycles
            cycle_dict, # layer-wise cycle breakdown
            avg_power, # average power mW
            power_dict, # layer-wise power breakdown
        )

    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
