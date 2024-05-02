"""
Description: 
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 2023-11-14 16:53:34
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2023-11-17 15:34:49
"""

"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-08 18:55:05
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-08 18:55:05
"""
import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from pyutils.general import print_stat
from pyutils.compute import gen_gaussian_noise, add_gaussian_noise
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.nn import Parameter, init
from torch.types import Device
from einops import einsum
from .utils import STE, mzi_out_diff_to_phase, mzi_phase_to_out_diff, partition_chunks

__all__ = ["ONNBaseLayer"]


class ONNBaseLayer(nn.Module):
    def __init__(self, *args, device: Device = torch.device("cpu"), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # cuda or cpu, defaults to cpu
        self.device = device

    def build_parameters(self, mode: str = "weight") -> None:
        ## weight mode
        phase = torch.empty(
            self.grid_dim_y,
            self.grid_dim_x,
            *self.miniblock,
            device=self.device,
        )
        weight = torch.empty(
            self.grid_dim_y,
            self.grid_dim_x,
            *self.miniblock,
            device=self.device,
        )
        # TIA gain
        S_scale = torch.ones(
            size=list(weight.shape[:-2]) + [1], device=self.device, dtype=torch.float32
        )

        if mode == "weight":
            self.weight = Parameter(weight)
        elif mode == "phase":
            self.phase = Parameter(phase)
            self.S_scale = Parameter(S_scale)
        elif mode == "voltage":
            raise NotImplementedError
        else:
            raise NotImplementedError

        for p_name, p in {
            "weight": weight,
            "phase": phase,
            "S_scale": S_scale,
        }.items():
            if not hasattr(self, p_name):
                self.register_buffer(p_name, p)

    def reset_parameters(self, mode=None) -> None:
        mode = mode or self.mode
        if mode in {"weight"}:
            if hasattr(self, "kernel_size"):  # for conv2d
                weight = nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                    bias=False,
                ).weight.data
                weight = weight.flatten(1)
                in_channels_pad = self.in_channels_pad - weight.shape[1]
                out_channels_pad = self.out_channels_pad - weight.shape[0]
            elif hasattr(self, "in_features"):  # for linear
                weight = nn.Linear(
                    self.in_features, self.out_features, bias=False
                ).weight.data
                in_channels_pad = self.in_features_pad - weight.shape[1]
                out_channels_pad = self.out_features_pad - weight.shape[0]
            weight = torch.nn.functional.pad(
                weight,
                (0, in_channels_pad, 0, out_channels_pad),
                mode="constant",
                value=0,
            )
            self.weight.data.copy_(
                partition_chunks(weight, out_shape=self.weight.shape).to(
                    self.weight.device
                )
            )

        elif mode in {"phase"}:
            self.reset_parameters(mode="weight")
            scale = self.weight.data.abs().flatten(-2, -1).max(dim=-1, keepdim=True)[0]
            self.S_scale.data.copy_(scale)
            self.phase.data.copy_(
                mzi_out_diff_to_phase(self.weight.data.div(scale[..., None]))
            )
        else:
            raise NotImplementedError

        if self.bias is not None:
            init.uniform_(self.bias, 0, 0)

    @classmethod
    def from_layer(cls, layer: nn.Module, *args, **kwargs) -> nn.Module:
        raise NotImplementedError

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def enable_fast_forward(self) -> None:
        self.fast_forward_flag = True

    def disable_fast_forward(self) -> None:
        self.fast_forward_flag = False

    def set_phase_variation(self, flag: bool = False) -> None:
        self._enable_phase_variation = flag

    def set_weight_noise(self, noise_std: float = 0.0) -> None:
        self.weight_noise_std = noise_std

    def set_output_noise(self, noise_std: float = 0.0) -> None:
        self.output_noise_std = noise_std

    # tenperature drift changes, new added
    def set_global_temp_drift(self, flag: bool = False) -> None:
        self._enable_global_temp_drift = flag

    def set_gamma_noise(
        self, noise_std: float, random_state: Optional[int] = None
    ) -> None:
        self.gamma_noise_std = noise_std

    # crosstalk changes
    def set_crosstalk_noise(self, flag: bool = False) -> None:
        self._enable_crosstalk = flag

    def set_switch_power_count(self, flag: bool = False) -> None:
        self._enable_power_count = flag

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit
        self.phase_quantizer.set_bit(w_bit)
        self.weight_quantizer.set_bit(w_bit)

    def set_input_bitwidth(self, in_bit: int) -> None:
        self.in_bit = in_bit
        self.input_quantizer.set_bit(in_bit)

    def load_parameters(self, param_dict: Dict[str, Any]) -> None:
        """
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
        """
        if self.mode == "phase":
            self.build_weight(update_list=param_dict)

    def switch_mode_to(self, mode: str) -> None:
        self.mode = mode

    def set_enable_ste(self, enable_ste: bool) -> None:
        self._enable_ste = enable_ste

    def set_enable_remap(self, enable_remap: bool) -> None:
        self._enable_remap = enable_remap

    def set_noise_flag(self, noise_flag: bool) -> None:
        self._noise_flag = noise_flag

    def _add_phase_variation(
        self, x, src: float = "weight", enable_remap: bool = False
    ) -> None:
        # Gaussian blur to process the noise distribution from
        # do not do inplace tensor modification to x, this is dynamic noise injection in every forward pass
        # this function can handle both phase noise injection to phase tensors and weight tensors
        if (not self._enable_phase_variation) or self.phase_variation_scheduler is None:
            return x  #  no noise injected

        if src == "weight":
            phase, S_scale = self.build_phase_from_weight_(x)
            # we need to use this func to update S_scale and keep track of S_scale if weight gets updated.
        elif src == "phase":
            phase = x
        else:
            raise NotImplementedError

        ## we need to remap noise distribution instead of phase or noises!
        # because multiple weights can mapto the same tile.
        # because even when they are mapped to the same tile, they cannot share noises, they share noise distribution.
        ## then we do not need to unapply_remap.
        noise = self.phase_variation_scheduler.sample_noise(
            size=phase.shape, enable_remap=enable_remap, col_ind=self.col_ind
        )

        phase = phase + noise

        if src == "weight":
            x = self.mrr_tr_to_weight(self.mrr_roundtrip_phase_to_tr(phase)).mul(
                S_scale[..., None]
            )  # do not modify weight inplace! we cannot call build_weight_from_phase here because it will update weight.data
        else:
            x = phase

        return x

    def _add_global_temp_drift(
        self, x, src: float = "weight", enable_remap: bool = False
    ) -> None:
        if (not self._enable_global_temp_drift) or self.global_temp_scheduler is None:
            return x  #  no noise injected

        if src == "weight":
            phase, S_scale = self.build_phase_from_weight_(x)
            # we need to use this func to update S_scale and keep track of S_scale if weight gets updated.
        elif src == "phase":
            phase = x
        else:
            raise NotImplementedError

        T = self.global_temp_scheduler.get_global_temp()
        noise = self.global_temp_scheduler.get_phase_drift(
            phase, T, enable_remap=enable_remap, col_ind=self.col_ind
        )
        phase = phase + noise

        if src == "weight":
            x = self.mrr_tr_to_weight(self.mrr_roundtrip_phase_to_tr(phase)).mul(
                S_scale[..., None]
            )  # do not modify weight inplace! we cannot call build_weight_from_phase here because it will update weight.data
        else:
            x = phase

        return x

    def _add_crosstalk_noise(self, x, src: str = "weight") -> None:
        if (not self._enable_crosstalk) or self.crosstalk_scheduler is None:
            return x  #  no noise injected

        if src == "weight":
            phase, S_scale = self.build_phase_from_weight_(
                x
            )  # we need to use this func to update S_scale and keep track of S_scale if weight gets updated.
        elif src == "phase":
            phase = x
        else:
            raise NotImplementedError

        # crosstalk_coupling_matrix = self.crosstalk_scheduler.get_crosstalk_matrix(self.phase)
        crosstalk_coupling_matrix = self.crosstalk_scheduler.get_crosstalk_matrix(phase)
        # print("coupling", crosstalk_coupling_matrix)
        phase = self.crosstalk_scheduler.apply_crosstalk(
            phase, crosstalk_coupling_matrix
        )

        if src == "weight":
            x = mzi_phase_to_out_diff(phase).mul(
                S_scale[..., None]
            )  # do not modify weight inplace! we cannot call build_weight_from_phase here because it will update weight.data
        elif src == "phase":
            x = phase
        else:
            raise NotImplementedError

        return x

    def set_light_redist(self, flag: bool = False) -> None:
        ## enable or disable light redistribution if prune_mask is available
        self._enable_light_redist = flag

    def set_input_power_gating(self, flag: bool = False, ER: float = 6) -> None:
        ## enable or disable power gating for light shutdown if prune_mask is available
        ## ER 6dB SL-MZM,
        self._enable_input_power_gating = flag
        self._input_modulator_ER = ER
    
    def set_output_power_gating(self, flag: bool = False) -> None:
        ## enable or disable power gating for TIA/ADC shutdown if prune_mask is available
        self._enable_output_power_gating = flag

    def _add_output_noise(self, x) -> None:
        if self.output_noise_std > 1e-6:
            if (self._enable_light_redist or self._enable_output_power_gating) and self.prune_mask is not None:
                r, k1, k2 = self.miniblock[0], self.miniblock[-2], self.miniblock[-1]
                p, q, c = self.weight.shape[0], self.weight.shape[1], self.weight.shape[3]  # q*c
                if self._enable_light_redist:
                    col_mask = self.prune_mask["col_mask"]  # [p,q,1,c,1,k2]
                    col_nonzeros = col_mask.sum(-1).squeeze(-1)  # [p,q,1,c]
                    factor = col_nonzeros / k2  # [p,q,1,c]
                else:
                    factor = torch.ones([p,q,1,c], device=self.device) # [p,q,1,c]

                print(factor.mean())
                if self._enable_output_power_gating:
                    row_mask = self.prune_mask["row_mask"]  # [p,q,r,1,k1,1]
                    row_mask = row_mask[..., 0, :, :].flatten(2, 3)  # [p,q,r*k1, 1]
                    factor = factor * row_mask  # [p,q,r*k1, c]
                else:
                    factor = factor.expand(-1, -1, r*k1, -1) # [p,q,r*k1, c]

                factor = factor.permute(0, 2, 1, 3).flatten(0, 1)[
                    : x.shape[1]
                ]  # [p*r*k1, q, c] -> [out_c, q, c]
                print(row_mask.sum()/row_mask.numel())
                print(factor.sum()/factor.numel())

                std = factor.mul(k2**0.5).square().sum([-2, -1]).sqrt()

                std *= self.output_noise_std # [out_c]
                print("no light dist no out gating: std: ", np.sqrt(np.prod(self.weight.shape[1::2])) * self.output_noise_std)
                print("w/ light dist w/ out gating: std: ", std.mean().item(), std.min().item(), std.max().item())

                noise = torch.randn_like(x)  # [bs, out_c, h, w] or [bs, out_c, q, c]

                if noise.dim() == 4:
                    noise = noise * std[..., None, None]
                elif noise.dim() == 2:
                    noise = noise * std
                else:
                    raise NotImplementedError
                # if noise.dim() == 6:
                #     noise = torch.einsum(
                #         "bohwqc,oqc -> bohw", noise, factor
                #     )  # [bs, out_c, h, w]
                # elif noise.dim() == 4:
                #     # noise = torch.einsum("boqc,oqc -> bo", noise, factor)  # [bs, out_c]
                #     noise = einsum(noise, factor, "b o q c, o q c -> b o")
                # else:
                #     raise NotImplementedError
                # print_stat(x, message="x: ")
                # print_stat(noise, message="noise: ")
                x = x + noise
            else:
                vector_len = np.prod(self.weight.shape[1::2])  # q*c*k2
                noise = gen_gaussian_noise(
                    x,
                    noise_mean=0,
                    noise_std=np.sqrt(vector_len) * self.output_noise_std,
                )
                # print_stat(x, message="x: ")
                # print_stat(noise, message="noise: ")
                x = x + noise
        return x

    def calc_weight_MZI_power(
        self, weight=None, src: str = "weight", reduction: str = "none"
    ) -> None:

        weight = weight if weight is not None else self.weight

        if src == "weight":
            ## no inplace modification here.
            phase, _ = self.build_phase_from_weight(
                weight
            )  # we need to use this func to update S_scale and keep track of S_scale if weight gets updated.
        elif src == "phase":
            phase = weight
        else:
            raise NotImplementedError

        return self.crosstalk_scheduler.calc_MZI_power(
            phase, reduction=reduction
        )  # [p,q,r,c,k1,k2]

    def build_weight_from_phase(self, phases: Tensor) -> Tensor:
        ## inplace operation: not differentiable operation using copy_
        self.weight.data.copy_(
            mzi_phase_to_out_diff(phases).mul(self.S_scale.data[..., None])
        )
        return self.weight

    def build_weight_from_voltage(
        self,
        voltage: Tensor,
        S_scale: Tensor,
    ) -> Tensor:
        return self.build_weight_from_phase(
            *self.build_phase_from_voltage(voltage, S_scale)
        )

    def build_phase_from_weight_(self, weight: Tensor) -> Tuple[Tensor, Tensor]:
        ## inplace operation: not differentiable operation using copy_
        phase, S_scale = self.build_phase_from_weight(weight)
        self.phase.data.copy_(phase)
        self.S_scale.data.copy_(S_scale)
        return self.phase, self.S_scale

    def build_phase_from_weight(self, weight: Tensor) -> Tuple[Tensor, Tensor]:
        ## inplace operation: not differentiable operation using copy_
        S_scale = (
            weight.data.abs().flatten(-2, -1).max(dim=-1, keepdim=True)[0]
        )  # block-wise abs_max as scale factor

        weight = torch.where(
            S_scale[..., None] > 1e-8,
            weight.data.div(S_scale[..., None]),
            torch.zeros_like(weight.data),
        )
        phase = mzi_out_diff_to_phase(weight)
        return phase, S_scale

    def build_voltage_from_phase(
        self,
        phase: Tensor,
        S_scale: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def build_phase_from_voltage(
        self,
        voltage: Tensor,
        S_scale: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def build_voltage_from_weight(self, weight: Tensor) -> Tuple[Tensor, Tensor]:
        return self.build_voltage_from_phase(*self.build_phase_from_weight_(weight))

    def sync_parameters(self, src: str = "weight") -> None:
        """
        description: synchronize all parameters from the source parameters
        """
        if src == "weight":
            self.build_phase_from_weight_(self.weight)
        elif src == "phase":
            self.build_weight_from_phase(self.phase)
        elif src == "voltage":
            NotImplementedError
        else:
            raise NotImplementedError

    def print_parameters(self):
        print(self.phase) if self.mode == "phase" else print(self.weight)

    def build_weight(
        self,
        weight=None,
        enable_noise: bool = True,
        enable_ste: bool = False,
    ) -> Tensor:
        if self.mode == "weight":
            weight = weight if weight is not None else self.weight
            if self.w_bit < 16:
                weight = self.weight_quantizer(weight)

            ## TODO
            ## can apply pruning to weight(phase) here
            ## e.g., weight = self.prune(weight) make sure it is differentiable, and not inplace

            if enable_noise:
                ## use auto-diff from torch
                if enable_ste:
                    weight_tmp = weight.detach()
                else:
                    weight_tmp = weight

                phase, S_scale = self.build_phase_from_weight_(weight_tmp)
                ## step 1 add random phase variation
                phase = self._add_phase_variation(phase, src="phase")

                ## step 2 add thermal crosstalk
                phase = self._add_crosstalk_noise(phase, src="phase")

                ## reconstruct noisy weight
                weight_noisy = mzi_phase_to_out_diff(phase).mul(S_scale[..., None])
                if self._enable_output_power_gating and self.prune_mask is not None:
                    print("no output gating")
                    print_stat((weight_noisy - weight.data).abs())
                    weight_noisy = (
                        weight_noisy * self.prune_mask["row_mask"]
                    )  ## reapply mask to shutdown nonzero weights due to crosstalk, but gradient will still flow through the mask due to STE
                    print("w/ output gating")
                    print_stat((weight_noisy - weight.data).abs())
                if self._enable_input_power_gating and self.prune_mask is not None:
                    ratio = 1/10**(self._input_modulator_ER / 10)
                    print("no input gating")
                    print_stat((weight_noisy - weight.data).abs())
                    weight_noisy = weight_noisy * self.prune_mask["col_mask"].float().add(ratio).clamp(max=1)
                    print(f"w/ input gating {ratio}")
                    print_stat((weight_noisy - weight.data).abs())

                if enable_ste:
                    weight = STE.apply(
                        weight, weight_noisy
                    )  # cut off gradient for weight_noisy, only flow through weight
                else:
                    weight = weight_noisy
                self.noisy_phase = phase  # TODO: to DEBUG

        elif self.mode == "phase":
            if self.w_bit < 16:
                phase = self.phase_quantizer(self.phase)
            else:
                phase = self.phase

            if self.phase_noise_std > 1e-5:
                ### phase_S is assumed to be protected
                phase = add_gaussian_noise(
                    phase,
                    0,
                    self.phase_noise_std,
                    trunc_range=(-2 * self.phase_noise_std, 2 * self.phase_noise_std),
                )

            weight = self.build_weight_from_phase(phase)
        elif self.mode == "voltage":
            raise NotImplementedError
        else:
            raise NotImplementedError
        if self.weight_noise_std > 1e-6:
            weight = weight * (1 + torch.randn_like(weight) * self.weight_noise_std)
        return weight

    def layer_weight_partition_chunk(
        self, X: Tensor, require_size: torch.Size = [4, 4, 8, 8], complex: bool = False
    ) -> Tensor:
        """this function is used to partition layer weight into our required [R,C,K,K] size"""
        if isinstance(X, torch.Tensor):
            R, C = require_size[0], require_size[1]
            P, Q, k = X.shape[0:3]
            shape = int(np.ceil(P / R) * R), int(np.ceil(Q / C) * C)  # [P_pad, Q_pad]
            X = torch.nn.functional.pad(
                X,
                (0, 0, 0, 0, 0, shape[1] - Q, 0, shape[0] - P),
                mode="constant",
                value=0,
            )  # [P_pad, Q_pad, k, k]
            X = X.reshape(shape[0] // R, R, shape[1] // C, C, k, k).permute(
                0, 2, 1, 3, 4, 5
            )  # [b0, b1, R,C,K,K]

            return X

        # elif isinstance(x, np.ndarray):
        #     P, Q, K = self.weight.data.shape[0], self.weight.data.shape[1], self.weight.data.shape[2]
        #     R, C = require_size[0], require_size[1]
        #     bc_x, bc_y = P // R, Q // C
        #     if not complex:
        #         x = np.reshape(x, [bc_x, R, bc_y, C, K, K])
        #         x = np.transpose(x, [0, 2, 1, 3, 4, 5])
        #     else:
        #         x = np.reshape(x, [bc_x, R, bc_y, C, K, K, 2])
        #         x = np.transpose(x, [0, 2, 1, 3, 4, 5, 2])
        else:
            raise NotImplementedError

    def layer_weight_merge_chunk(self, x: Tensor, complex: bool = False) -> Tensor:
        # x = [bc_x, bc_y, R, C, K, K]
        if not complex:
            bc_x, bc_y, R, C, K = x.size(0), x.size(1), x.size(2), x.size(3), x.size(4)
            x = x.permute(0, 2, 1, 3, 4, 5)  # [bc_x, R, bc_y, C, K, K]
            x = x.reshape(bc_x * R, bc_y * C, K, K)  # [P_pad, Q_pad, K, K]
        else:
            bc_x, bc_y, R, C, K = x.size(0), x.size(1), x.size(2), x.size(3), x.size(4)
            x = x.permute(0, 2, 1, 3, 4, 5, 6)  # [bc_x, R, bc_y, C, K, K]
            x = x.reshape(bc_x * R, bc_y * C, K, K, 2)  # [P_pad, Q_pad, K, K]

        return x

    def remap_intra_tile(
        self,
        alg: str = "LAP",
        salience_mode="first_grad",
        average_times: int = 1,
        tolerance: float = 1,
    ):
        """Remap for [R,C,K,K] cores, map noise
        This function only solves row_ind and col_ind, it will not apply those indices to weights.
        """
        assert alg in {"LAP", "heuristic"}
        assert salience_mode in {"first_grad", "second_grad", "none"}
        self.row_ind, self.col_ind = [], []
        layer_weight = self._ideal_weight
        size = self.phase_variation_scheduler.size
        self.batch_weight_cores = self.layer_weight_partition_chunk(
            self.weight.data, require_size=size  # [b0, b1, R, C, k, k]
        )  # a tensor, with padding 0

        self.batch_ideal_weight_core = self.layer_weight_partition_chunk(
            layer_weight, require_size=size
        )  # [b0, b1, R, C, k, k]
        salience = (
            self.layer_weight_partition_chunk(self.weight._salience, require_size=size)
            if salience_mode in {"first_grad", "second_grad"}
            else None
        )
        if salience_mode == "none":
            first_salience = second_salience = None
        elif salience_mode == "first_grad":
            first_salience = self.layer_weight_partition_chunk(
                self.weight._first_grad, require_size=size
            )
            second_salience = None
        elif salience_mode == "second_grad":
            first_salience = self.layer_weight_partition_chunk(
                self.weight._first_grad, require_size=size
            )
            second_salience = self.layer_weight_partition_chunk(
                self.weight._second_grad, require_size=size
            )
        else:
            raise NotImplementedError
        # [b0, b1, R, C, k, k]
        # print(self.weight -layer_weight)
        # [4,4,8,8] PV noise for each batch

        ## generate epsilon matrix [b0, b1, R, R]
        ## we only need shift weight row-wise (R) and parallel prob
        ## here we assume average times = 1 based on our previuos experiments
        all_weights = []
        all_first_salience = []
        all_second_salience = []
        ## [W1, W2, ..., WR]
        ## [W2, W3, ..., W1]
        ## [WR, WR-1, ..., W1]
        for r in range(size[0]):
            shifted_weights = self.layer_weight_merge_chunk(
                self.batch_weight_cores.data.roll(-r, dims=2)
            )[
                : self.grid_dim_y, : self.grid_dim_x
            ]  # [b0, b1, <R>, C, K, k] -> shift and merge -> [P, Q, K, K]
            shifted_weights = self.build_weight(
                weight=shifted_weights,
                flag=True,
                enable_ste=False,
                enable_remap=False,
            )
            shifted_weights = self.layer_weight_partition_chunk(
                shifted_weights, require_size=size
            ).roll(
                r, dims=2
            )  # [b0, b1, R, C, k, k]
            all_weights.append(shifted_weights)  # [b0, b1, R, C, k, k]
            if first_salience is not None:
                all_first_salience.append(first_salience.roll(-r, dims=2))
            if second_salience is not None:
                all_second_salience.append(second_salience.roll(-r, dims=2))
        all_weights = torch.stack(all_weights, dim=2)  # [b0, b1, <R>, R, C, k, k]

        if len(all_first_salience) > 0:
            all_first_salience = torch.stack(
                all_first_salience, dim=2
            )  # [b0, b1, <R>, R, C, k, k]
        if len(all_second_salience) > 0:
            all_second_salience = torch.stack(
                all_second_salience, dim=2
            )  # [b0, b1, <R>, R, C, k, k]

        # print(all_weights.shape, self.batch_ideal_weight_core.shape)
        if salience_mode == "none":
            epsilon_matrix = all_weights.sub(
                self.batch_ideal_weight_core.unsqueeze(2)
            ).norm(
                p=1, dim=(-3, -2, -1)
            )  # [b0, b1, <R>, R]
        elif salience_mode == "first_grad":
            epsilon_matrix = (
                all_weights.sub(self.batch_ideal_weight_core.unsqueeze(2))
                .mul(all_first_salience)
                .sum(dim=[-1, -2, -3])
                .abs()
            )  # [b0, b1, <R>, R]
        elif salience_mode == "second_grad":
            err = all_weights.sub(self.batch_ideal_weight_core.unsqueeze(2))
            epsilon_matrix = (
                err.mul(all_first_salience).sum(dim=[-1, -2, -3])
                + 0.5 * err.square().mul(all_second_salience).sum(dim=[-1, -2, -3])
            ).abs()
            # [b0, b1, <R>, R]
        # if salience is not None:
        #     epsilon_matrix.mul_(all_salience)
        # print(epsilon_matrix.shape)
        # exit(0)

        ## merge parallel probing results into epsilon_matrix
        ##[e11, e22, e33, e44]
        ##[e21, e32, e43, e14]
        ##[e31, e42, e13, e24]
        ##[e41, e12, e23, e34]
        ## ->
        ##[e11, e12, e13, e14]
        ##[e21, e22, e23, e24]
        ##[e31, e32, e33, e34]
        ##[e41, e42, e43, e44]
        ## just roll columns
        for col in range(1, epsilon_matrix.shape[-1]):
            epsilon_matrix[..., col] = epsilon_matrix[..., col].roll(col, dims=-1)
        epsilon_matrix = epsilon_matrix.data
        ## now we have [b0, b1, R, R] epsilon_matrix
        # print(epsilon_matrix)

        ## apply threshold based on tolerance and regenerate epsilon_matrix and tile_indices
        tile_errors = epsilon_matrix.mean(dim=-2)  # [b0, b1, R]
        tile_min_errors = tile_errors.min(dim=-1)[0]  # [b0, b1]
        tile_indices = torch.zeros_like(
            tile_errors
        )  # [b0, b1, R], important, used to reinterprete tile remapping
        max_workload_assigned = torch.zeros_like(tile_min_errors).to(
            torch.int32
        )  # [b1, b0]
        for b0 in range(epsilon_matrix.shape[0]):
            for b1 in range(epsilon_matrix.shape[1]):
                tile_err = tile_errors[
                    b0, b1
                ]  # e.g., R=5, [0.3, 0.02, 0.05, 0.2, 0.03]
                tile_mask = tile_err <= max(
                    tolerance, tile_min_errors[b0, b1]
                )  # e.g., [0, 1, 1, 0, 1], at least there is one tile
                good_tiles = torch.nonzero(tile_mask)[:, 0]  # e.g., [1, 2, 4]
                ## we need to duplicate good tiles to fill the whole rows
                tile_times, workload_left = divmod(
                    size[0], len(good_tiles)
                )  # e.g., 5 // 3 = 1, 5 % 3 = 2
                ## every tile will do tile_times by default
                ## then the rest workload_left will be spread to lowest error tiles.
                selected_tiles = torch.argsort(tile_err[good_tiles])[
                    :workload_left
                ]  # [0.02, 0.05, 0.03] -> [0, 2, 1] -> [0, 2]
                workload_assigned = torch.tensor(
                    [tile_times] * len(good_tiles), device=self.device
                )
                workload_assigned[
                    selected_tiles
                ] += 1  # [1, 1, 1] + [1, 0, 1] -> [2, 1, 2] total workload
                max_workload_assigned[b0, b1] = workload_assigned.max()
                tile_index = []
                ## [1, 2, 4] repeat [2, 1, 2] -> [1, 1, 2, 4, 4]
                for idx, assigned in zip(good_tiles, workload_assigned):
                    tile_index.extend([idx] * assigned)
                tile_indices[b0, b1] = torch.tensor(tile_index, device=self.device)
                epsilon_matrix[b0, b1].copy_(epsilon_matrix[b0, b1, :, tile_index])
        epsilon_matrix = epsilon_matrix.cpu().numpy()
        self.max_workload_assigned = max_workload_assigned

        # print(tile_indices)
        ## we need to solve b0 x b1 linear assignment problem
        self.row_ind = []
        for b0 in range(epsilon_matrix.shape[0]):
            row_ind_list, col_ind_list = [], []
            for b1 in range(epsilon_matrix.shape[1]):
                row_ind, col_ind = linear_sum_assignment(epsilon_matrix[b0, b1])
                row_ind_list.append(torch.from_numpy(row_ind).to(self.device))

                ## need to reinterprete col_ind based on tile_indices
                ## e.g., tile_indices [1, 1, 2, 4, 4]
                ## col_ind            [0, 3, 2, 4, 1]
                ## actual col_ind     [1, 4, 2, 4, 1]
                col_ind = tile_indices[b0, b1][col_ind]
                col_ind_list.append(col_ind)
            self.row_ind.append(torch.stack(row_ind_list))
            self.col_ind.append(torch.stack(col_ind_list))

        self.row_ind = torch.stack(self.row_ind).long()  # [b0, b1, R]
        self.col_ind = torch.stack(self.col_ind).long()  # [b0, b1, R]

        cycles = (
            (size[0] * size[-1] + size[0] ** 3)
            * epsilon_matrix.shape[0]
            * epsilon_matrix.shape[1]
        )  # Rk + R^3
        return self.row_ind, self.col_ind, cycles
        
    def forward(self, x):
        raise NotImplementedError

    def extra_repr(self) -> str:
        return ""
