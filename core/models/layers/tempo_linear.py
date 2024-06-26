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
from pyutils.compute import add_gaussian_noise, gen_gaussian_noise, merge_chunks
from pyutils.general import logger, print_stat
from pyutils.quant.lsq import ActQuantizer_LSQ, WeightQuantizer_LSQ
from torch import Tensor, nn
from torch.nn import Parameter, init
from torch.types import Device

from .base_layer import ONNBaseLayer
from .utils import CrosstalkScheduler, PhaseVariationScheduler

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
    miniblock: int
    weight: Tensor
    mode: str
    __annotations__ = {"bias": Optional[Tensor]}

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        miniblock: Tuple[int, int] = [4, 4],  # dim_y, dim_x, i.e., #cols, #rows
        mode: str = "weight",
        w_bit: int = 32,
        in_bit: int = 32,
        phase_variation_scheduler: PhaseVariationScheduler = None,
        crosstalk_scheduler: CrosstalkScheduler = None,
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
        self.grid_dim_x = int(np.ceil(self.in_features / miniblock[1]))
        self.grid_dim_y = int(np.ceil(self.out_features / miniblock[0]))
        self.in_features_pad = self.grid_dim_x * miniblock[1]
        self.out_features_pad = self.grid_dim_y * miniblock[0]

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
        self.set_enable_ste(True)
        self.set_noise_flag(True)
        self.set_enable_remap(False)
        self.phase_variation_scheduler = phase_variation_scheduler
        self.crosstalk_scheduler = crosstalk_scheduler

        if bias:
            self.bias = Parameter(torch.Tensor(out_features).to(self.device))
        else:
            self.register_parameter("bias", None)

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

    def cycles(self, x_size=None, probe: bool = True, num_vectors=None) -> int:
        if num_vectors is None:
            if probe:
                num_vectors = self.miniblock
            else:
                num_vectors = 1
        R, C, _, _ = self.phase_variation_scheduler.size
        P, Q = self.grid_dim_y, self.grid_dim_x

        if self._enable_remap and hasattr(self, "max_workload_assigned"):
            ## same times the accelerator needs multiple cycles to finish the workload
            cycles = self.max_workload_assigned.sum().item() * num_vectors
        else:
            cycles = int(np.ceil(P / R) * np.ceil(Q / C) * num_vectors)

        return cycles

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
