"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 23:37:55
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 23:37:55
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.registry import MODELS
from pyutils.compute import add_gaussian_noise, gen_gaussian_noise, merge_chunks
from pyutils.general import logger, print_stat
from pyutils.quant.lsq import ActQuantizer_LSQ, WeightQuantizer_LSQ
from torch import Tensor, nn
from torch.nn import Parameter, init
from torch.nn.modules.utils import _pair
from torch.types import Device, _size

from .base_layer import ONNBaseLayer
from .utils import CrosstalkScheduler, PhaseVariationScheduler, SparsityEnergyScheduler

__all__ = [
    "TeMPOBlockConv2d",
]


@MODELS.register_module()
class TeMPOBlockConv2d(ONNBaseLayer):
    """
    blocking Conv2d layer constructed by cascaded TeMPOs.
    """

    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "padding_mode",
        "output_padding",
        "in_channels",
        "out_channels",
        "kernel_size",
        "miniblock",
    ]
    __annotations__ = {"bias": Optional[torch.Tensor]}

    _in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Tuple[int, ...]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]
    miniblock: int
    mode: str

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size,
        stride: _size = 1,
        padding: _size = 0,
        dilation: _size = 1,
        groups: int = 1,
        bias: bool = True,
        miniblock: _size = (4, 4),  # dim_y, dim_x, i.e., #cols, #rows
        mode: str = "weight",
        w_bit: int = 32,
        in_bit: int = 32,
        phase_variation_scheduler: PhaseVariationScheduler = None,
        crosstalk_scheduler: CrosstalkScheduler = None,
        switch_power_scheduler: SparsityEnergyScheduler = None,
        device: Device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ):
        super().__init__(device=device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        assert (
            groups == 1
        ), f"Currently group convolution is not supported, but got group: {groups}"
        self.mode = mode
        assert mode in {"weight", "phase", "voltage"}, logger.error(
            f"Mode not supported. Expected one from (weight, phase, voltage) but got {mode}."
        )
        self.miniblock = miniblock
        self.in_channels_flat = (
            self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        )
        self.grid_dim_x = int(np.ceil(self.in_channels_flat / miniblock[1]))
        self.grid_dim_y = int(np.ceil(self.out_channels / miniblock[0]))
        self.in_channels_pad = self.grid_dim_x * miniblock[1]
        self.out_channels_pad = self.grid_dim_y * miniblock[0]

        self.v_max = 10.8
        self.v_pi = 4.36
        self.gamma = np.pi / self.v_pi**2
        self.w_bit = w_bit
        self.in_bit = in_bit
        self.phase_noise_std = 1e-5
        ### build trainable parameters
        self.build_parameters(mode)
        ### quantization tool
        self.input_quantizer = ActQuantizer_LSQ(
            None, device=device, nbits=self.in_bit, offset=True, mode="tensor_wise"
        )
        self.weight_quantizer = WeightQuantizer_LSQ(
            None,
            device=device,
            nbits=self.w_bit,
            offset=False,
            signed=True,
            mode="tensor_wise",
        )
        self.phase_quantizer = WeightQuantizer_LSQ(
            None,
            device=device,
            nbits=self.w_bit,
            offset=False,
            signed=True,
            mode="tensor_wise",
        )

        ### default set to slow forward
        self.disable_fast_forward()
        ### default set no phase variation
        self.set_phase_variation(False)
        self.set_global_temp_drift(False)
        self.set_crosstalk_noise(False)
        self.set_weight_noise(0)
        self.set_enable_ste(True)
        self.set_noise_flag(True)
        self.set_enable_remap(False)
        self.phase_variation_scheduler = phase_variation_scheduler
        self.crosstalk_scheduler = crosstalk_scheduler
        self.switch_power_scheduler = switch_power_scheduler

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels).to(self.device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    @classmethod
    def from_layer(
        cls,
        layer: nn.Conv2d,
        mode: str = "weight",
    ) -> nn.Module:
        """Initialize from a nn.Conv2d layer. Weight mapping will be performed

        Args:
            mode (str, optional): parametrization mode. Defaults to "weight".
            decompose_alg (str, optional): decomposition algorithm. Defaults to "clements".
            photodetect (bool, optional): whether to use photodetect. Defaults to True.

        Returns:
            Module: a converted TeMPOConv2d module
        """
        assert isinstance(
            layer, nn.Conv2d
        ), f"The conversion target must be nn.Conv2d, but got {type(layer)}."
        in_channels = layer.in_channels
        out_channels = layer.out_channels
        kernel_size = layer.kernel_size
        stride = layer.stride
        padding = layer.padding
        dilation = layer.dilation
        groups = layer.groups
        bias = layer.bias is not None
        device = layer.weight.data.device
        instance = cls(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            mode=mode,
            device=device,
        ).to(device)
        instance.weight.data.copy_(layer.weight)
        instance.sync_parameters(src="weight")
        if bias:
            instance.bias.data.copy_(layer.bias)

        return instance

    def MAC(self, x: Tensor) -> int:
        input_H, input_W = x.size(0), x.size(1)
        output_H = ((input_H - self.kernel_size + 2 * self.padding) // self.stride) + 1
        output_W = ((input_W - self.kernel_size + 2 * self.padding) // self.stride) + 1
        MAC = (
            self.in_channels_pad
            * self.out_channels_pad
            * self.kernel_size**2
            * output_H
            * output_W
        )
        return MAC

    def cycles(self, x_size=None, probe: bool = True, num_vectors=None) -> int:
        if num_vectors is None:
            if probe:
                num_vectors = self.miniblock
            else:
                input_H, input_W = x_size[-2], x_size[-1]
                output_H = (
                    (input_H - self.kernel_size[0] + 2 * self.padding[0])
                    // self.stride[0]
                ) + 1
                output_W = (
                    (input_W - self.kernel_size[1] + 2 * self.padding[1])
                    // self.stride[1]
                ) + 1
                num_vectors = output_H * output_W

        R, C, _, _ = self.phase_variation_scheduler.size
        P, Q = self.grid_dim_y, self.grid_dim_x
        if self._enable_remap and hasattr(self, "max_workload_assigned"):
            ## same times the accelerator needs multiple cycles to finish the workload
            cycles = self.max_workload_assigned.sum().item() * num_vectors
        else:
            cycles = int(np.ceil(P / R) * np.ceil(Q / C) * num_vectors)
        return cycles

    def get_output_dim(self, img_height: int, img_width: int) -> _size:
        h_out = (
            img_height
            - self.dilation[0] * (self.kernel_size[0] - 1)
            - 1
            + 2 * self.padding[0]
        ) / self.stride[0] + 1
        w_out = (
            img_width
            - self.dilation[1] * (self.kernel_size[1] - 1)
            - 1
            + 2 * self.padding[1]
        ) / self.stride[1] + 1
        return int(h_out), int(w_out)

    def forward(self, x: Tensor) -> Tensor:
        if self.in_bit < 16:
            x = self.input_quantizer(x)
        if not self.fast_forward_flag or self.weight is None:
            weight = self.build_weight(
                enable_noise=self._noise_flag,
                enable_ste=self._enable_ste,
            )  # [p, q, k, k]
        else:
            weight = self.weight
        weight = merge_chunks(weight)[
            : self.out_channels, : self.in_channels_flat
        ].view(-1, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        x = F.conv2d(
            x,
            weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return x

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.miniblock is not None:
            s += ', miniblock={miniblock}'
        if self.mode is not None:
            s += ', mode={mode}'
        return s.format(**self.__dict__)