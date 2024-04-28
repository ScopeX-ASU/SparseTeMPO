"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 23:37:55
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 23:37:55
"""

from ast import Not
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.registry import MODELS
from pyutils.general import logger, print_stat
from pyutils.quant.lsq import ActQuantizer_LSQ
from torch import Tensor, nn
from torch.nn import Parameter
from torch.types import Device

from .base_layer import ONNBaseLayer
from .utils import (
    CrosstalkScheduler,
    PhaseVariationScheduler,
    SparsityEnergyScheduler,
    WeightQuantizer_LSQ,
    merge_chunks,
)

__all__ = [
    "TeMPOBlockLinear",
]

MODELS.register_module(name="Linear", module=nn.Linear)


@MODELS.register_module()
class TeMPOBlockLinear(ONNBaseLayer):
    """
    blocking Linear layer constructed by cascaded TeMPOs.
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    miniblock: Tuple[int, int, int, int]
    weight: Tensor
    mode: str
    __annotations__ = {"bias": Optional[Tensor]}

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        miniblock: Tuple[int, int, int, int] = [4, 4, 4, 4],  # [r, c, dim_y, dim_x]
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
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        assert mode in {"weight", "phase", "voltage"}, logger.error(
            f"Mode not supported. Expected one from (weight, usv, phase, voltage) but got {mode}."
        )
        self.miniblock = miniblock
        self.grid_dim_x = int(np.ceil(self.in_features / miniblock[1] / miniblock[3]))
        self.grid_dim_y = int(np.ceil(self.out_features / miniblock[0] / miniblock[2]))
        self.in_features_pad = self.grid_dim_x * miniblock[1] * miniblock[3]
        self.out_features_pad = self.grid_dim_y * miniblock[0] * miniblock[2]

        self.v_max = 10.8
        self.v_pi = 4.36
        self.gamma = np.pi / self.v_pi**2
        self.w_bit = w_bit
        self.in_bit = in_bit
        self.phase_noise_std = 0

        self.weight_rank = []

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
        self.set_phase_variation(False)
        self.set_crosstalk_noise(False)
        self.set_weight_noise(0)
        self.set_output_noise(0)
        self.set_enable_ste(True)
        self.set_noise_flag(True)
        self.set_enable_remap(False)
        self.set_light_redist(False)
        self.set_power_gating(False)
        self.phase_variation_scheduler = phase_variation_scheduler
        self.crosstalk_scheduler = crosstalk_scheduler
        self.switch_power_scheduler = switch_power_scheduler

        if bias:
            self.bias = Parameter(torch.Tensor(out_features).to(self.device))
        else:
            self.register_parameter("bias", None)

        self.prune_mask = None
        self.reset_parameters()

    @classmethod
    def from_layer(
        cls,
        layer: nn.Linear,
        mode: str = "weight",
    ) -> nn.Module:
        """Initialize from a nn.Linear layer. Weight mapping will be performed

        Args:
            mode (str, optional): parametrization mode. Defaults to "weight".
            decompose_alg (str, optional): decomposition algorithm. Defaults to "clements".
            photodetect (bool, optional): whether to use photodetect. Defaults to True.

        Returns:
            Module: a converted TeMPOLinear module
        """
        assert isinstance(
            layer, nn.Linear
        ), f"The conversion target must be nn.Linear, but got {type(layer)}."
        in_features = layer.in_features
        out_features = layer.out_features
        bias = layer.bias is not None
        device = layer.weight.data.device
        instance = cls(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            mode=mode,
            device=device,
        ).to(device)
        instance.weight.data.copy_(layer.weight)
        instance.sync_parameters(src="weight")
        if bias:
            instance.bias.data.copy_(layer.bias)

        return instance

    def MAC(self) -> int:
        # MAC for single-batch inference
        MAC = self.in_features_pad * self.out_features_pad
        return MAC

    def cycles(self, x_size=None, R: int = 8, C: int = 8) -> int:
        bs = x_size[0]

        num_cycles_per_block = bs

        k1, k2 = self.miniblock[-2:]
        num_blocks = int(
            np.ceil(self.out_features / (R * k1)) * np.ceil(self.in_features / (C * k2))
        )
        return num_cycles_per_block, num_blocks * num_cycles_per_block

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
        weight = merge_chunks(weight)[: self.out_features, : self.in_features]
        x = F.linear(
            x,
            weight,
            bias=self.bias,
        )

        if self._noise_flag:
            x = self._add_output_noise(x)

        return x

    def extra_repr(self):
        s = "{in_features}, {out_features}"
        if self.bias is None:
            s += ", bias=False"
        if self.miniblock is not None:
            s += ", miniblock={miniblock}"
        if self.mode is not None:
            s += ", mode={mode}"
        return s.format(**self.__dict__)
