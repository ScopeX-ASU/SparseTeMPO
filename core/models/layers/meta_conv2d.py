"""
Description: 
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-05-25 22:19:52
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-05-26 01:02:27
"""

from typing import Tuple, Optional, Union
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.autograd import Function
from torch.types import Device
from core.models.layers.utils import hard_diff_round
import torch.nn.functional as F
from mmengine.registry import MODELS
from core.models.layers.meta_layer_base import Meta_Layer_BASE
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
import torch.utils.checkpoint as cp
from functools import lru_cache

__all__ = ["MetaConv2d"]


class _MetaConv2dMultiPath(Meta_Layer_BASE):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        n_pads: int = 5,
        bias: bool = False,
        w_bit: int = 16,
        in_bit: int = 16,
        input_uncertainty: float = 0,
        pad_noise_std: float = 0,
        dpe=None,
        pad_max: float = 1.0,
        sigma_trainable: str = "row_col",
        mode: str = "usv",
        lambda_=0.532,  # um as unit distance
        path_multiplier: int = 2,
        path_depth: int = 2,
        unfolding: bool = True,
        device: Device = torch.device("cuda"),
        verbose: bool = False,
        with_cp: bool = False,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            n_pads=n_pads,
            kernel_size=kernel_size,
            w_bit=w_bit,
            in_bit=in_bit,
            input_uncertainty=input_uncertainty,
            pad_noise_std=pad_noise_std,
            dpe=dpe,
            pad_max=pad_max,
            mode=mode,
            sigma_trainable=sigma_trainable,
            device=device,
            verbose=verbose,
            with_cp=with_cp,
        )
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        # allocate parameters
        self.weight = None
        self.path_weight = None
        self.x_zero_pad = None
        self.polarization = None

        self.lambda_ = lambda_
        self.wv = 2 * np.pi / lambda_
        self.sigma_trainable = sigma_trainable
        self.path_multiplier = path_multiplier
        self.pixel_size = nn.Parameter(
            torch.empty(1, device=self.device).fill_(0.4),
            requires_grad=False,  # learned value 0.5, very similar to the 0.4 value in paper
        )

        self.delta_z = nn.Parameter(
            torch.ones(1, device=self.device).fill_(8.42),
            requires_grad=False,  # learned value, very similar to the 8.42 in paper
        )
        self.path_depth = path_depth
        self.unfolding = unfolding

        self.build_parameters(bias=bias)
        self.reset_parameters()

    def build_parameters(self, bias: bool) -> None:
        self.in_channels_flat = self.in_channels // self.groups

        self.weight = nn.Parameter(
            torch.randn(
                self.path_multiplier,
                self.path_depth,
                2,
                *self.kernel_size,
                device=self.device
            )
        )

        """
        For path weight, if use Gumbel Softmax approximation method, the weight for the path should be 
        outc * inc * d * (path_multiplier + 1)
        """
        # Initialize path weights
        self.path_weight = nn.Parameter(
            torch.randn(
                self.path_depth,
                self.path_multiplier,
                self.path_multiplier,
                device=self.device,
            ),
            requires_grad=False,
        )

        self.alpha = nn.Parameter(
            torch.randn(self.in_channels, device=self.device), requires_grad=True
        )

        self.beta = nn.Parameter(torch.ones(1, device=self.device), requires_grad=True)
        # self.polarization = nn.Parameter(
        #     torch.rand(
        #         self.parallel_path,
        #         device=self.device
        #     ) * np.pi
        # )

        self.alm_multiplier = nn.Parameter(
            torch.empty(self.path_depth, 2, self.path_multiplier, device=self.device),
            requires_grad=False,
        )

        if bias:
            self.bias = Parameter(torch.zeros(self.out_channels, device=self.device))
        else:
            self.register_parameter("bias", None)

    def build_path_weight(self):
        # Ensure normalization across each n x n tensor
        path_weight = self.path_weight
        path_weight = path_weight.abs()
        # print(path_weight.shape)
        path_weight = path_weight / path_weight.data.sum(
            dim=1, keepdim=True
        )  # Sum over rows
        # print(path_weight.shape)
        path_weight = path_weight / path_weight.data.sum(
            dim=2, keepdim=True
        )  # Sum over columns

        with torch.no_grad():
            perm_loss = path_weight.data.norm(p=1, dim=1).sub(
                path_weight.data.norm(p=2, dim=1)
            ).mean(dim=-1) + (1 - path_weight.data.norm(p=2, dim=2)).mean(dim=(-1))

        # print(perm_loss.shape)

        for i in range(perm_loss.shape[0]):
            if perm_loss[i] < 0.05:
                path_weight[i] = hard_diff_round(path_weight[i])

        return path_weight

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.constant_(self.alm_multiplier.data, 0)
        nn.init.constant_(self.alpha.data, 0.002)
        nn.init.constant_(self.beta.data, 1)

    def get_perm_loss(self):
        path_weight = self.build_path_weight()
        loss = path_weight.data.norm(p=1, dim=1).sub(
            path_weight.data.norm(p=2, dim=1)
        ).mean(dim=-1) + (1 - path_weight.data.norm(p=2, dim=2)).mean(dim=(-1))
        # print(loss.shape)
        return loss

    def get_alm_perm_loss(self, rho: float = 0.1):
        ## quadratic tern is also controlled multiplier
        path_weight = self.build_path_weight()
        d_path_weight_r = path_weight.norm(p=1, dim=1).sub(path_weight.norm(p=2, dim=1))
        # d_weight_c = weight.norm(p=1, dim=1).sub(weight.norm(p=2, dim=1))
        d_path_weight_c = 1 - path_weight.norm(p=2, dim=2)
        loss = torch.zeros(path_weight.shape[0])
        d_path_weight_r_square = d_path_weight_r.square()
        d_path_weight_c_square = d_path_weight_c.square()

        for i in range(path_weight.shape[0]):
            loss_r = self.alm_multiplier[i, 0].dot(
                d_path_weight_r[i] + rho / 2 * d_path_weight_r_square[i]
            )
            loss_c = self.alm_multiplier[i, 1].dot(
                d_path_weight_c[i] + rho / 2 * d_path_weight_c_square[i]
            )
            loss[i] = loss_r + loss_c
        # loss = self.alm_multiplier[:, 0].unsqueeze(1).dot(d_path_weight_r + rho / 2 * d_path_weight_r.square()) + \
        #        self.alm_multiplier[:, 1].unsqueeze(1).dot(d_path_weight_c + rho / 2 * d_path_weight_c.square())

        print(loss.shape)
        return loss

    def update_alm_multiplier(
        self, rho: float = 0.1, max_lambda: Optional[float] = None
    ):
        with torch.no_grad():
            path_weight = self.build_path_weight().detach()
            d_path_weight_r = path_weight.norm(p=1, dim=1).sub(
                path_weight.norm(p=2, dim=1)
            )
            d_path_weight_c = path_weight.norm(p=1, dim=2).sub(
                path_weight.norm(p=2, dim=2)
            )
            d_path_weight_r_square = d_path_weight_r.square()
            d_path_weight_c_square = d_path_weight_c.square()
            for i in range(path_weight.shape[0]):
                self.alm_multiplier[i, 0].add_(
                    d_path_weight_r[i] + rho / 2 * d_path_weight_r_square[i]
                )
                self.alm_multiplier[i, 1].add_(
                    d_path_weight_c[i] + rho / 2 * d_path_weight_c_square[i]
                )
            if max_lambda is not None:
                self.alm_multiplier.data.clamp_max_(max_lambda)

    def path_generation(self, path_weight):
        path_before_transpose = torch.argmax(path_weight, dim=-1)
        path_after_transpose = torch.transpose(path_before_transpose, 0, 1)
        full_repetitions, remainder = divmod(
            self.in_channels, path_after_transpose.size(0)
        )
        repeated_a = path_after_transpose.repeat(full_repetitions, 1)
        if remainder:
            repeated_a = torch.cat(
                (repeated_a, path_after_transpose[:remainder]), dim=0
            )

        return repeated_a.unsqueeze(0).expand(self.out_channels, -1, -1)

    def build_initial_path(self):
        # [[0000],[1111], [2222]]
        # Generate a tensor of size (m, 1) ranging from 0 to m-1
        row_values = torch.arange(self.path_multiplier).view(self.path_multiplier, 1)

        # Expand the tensor to size (m, n) by repeating the columns
        init_path = row_values.expand(-1, self.path_depth)

        full_repetitions, remainder = divmod(self.in_channels, init_path.size(0))
        repeated_a = init_path.repeat(full_repetitions, 1)
        if remainder:
            repeated_a = torch.cat((repeated_a, init_path[:remainder]), dim=0)

        return repeated_a.unsqueeze(0).expand(self.out_channels, -1, -1)

    def build_alpha(self) -> Tensor:
        alpha = self.alpha
        alpha = alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return alpha

    def build_beta(self) -> Tensor:
        beta = self.beta
        beta = torch.sigmoid(beta)
        # beta = torch.clamp(beta, min=0, max=1) # Add sigmoid or tanh
        # beta = beta.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4)
        return beta

    def build_weight(self) -> Tensor:
        # return [outc, inc, d, s, s] complex
        # from weight and path_weight, we build weight and path
        # if self.reinforce_training:
        #     path_weight = self.build_path_weight()
        #     paths = self.path_generation(path_weight)
        #     # In init part
        # else:
        paths = self.build_initial_path()
        
        # Check if paths are constant
        # print(paths)

        weight = super()._weight  # [m, d, s, s, 2] real, [outc, inc, d] long
        # print(weight.shape)

        # Weight Initialization Check
        # print(weight)
        # return

        weight = weight % (2 * torch.pi)
        # print(weight)
        # return

        # print("yes")/
        # print(f"First time : {weight.shape}")
        # print(f"path : {paths}")

        # [m, d, s, s]

        weight = torch.exp(1j * weight)

        # if self.reinforce_training:
        #     magnitude = torch.abs(weight)
        #     # Ae^(j phi)
        #
        #     # Not knowing the exact phase lut yet
        #     phase = PhiPLUT(self.polarization, self.lut, self.functions)
        #     # Might need to reshape to [m, d, s, s]
        #
        #     weight = torch.exp(1j * phase)

        # print(f"Second time : {weight.shape}")

        # add 1 padding to identity phase mask
        # [m + 1(padding 1), d, s, s]
        # print(weight.shape)
        # weight = torch.nn.functional.pad(
        #     weight, pad=[0] * int((weight.dim() - 1) * 2) + [0, 1], value=1
        # )
        # print(weight)
        # print(weight.shape)
        # return
        # print(f"Third time : {weight.shape}")

        # for parallel implementation, we will use paths to extract all phase masks for all images
        weight = weight[
            paths, torch.arange(weight.shape[1])
        ]  # [outc, inc, d, s, s] complex
        # print(weight)
        # print(weight.shape)
        # return
        # print(f"Fourth time : {weight.shape}")
        # print(f"Weight: {weight}")

        return weight

    @lru_cache(maxsize=32)
    def construct_positions(self, k):
        # Generate a grid of x and y coordinates
        x = torch.arange(k, device=self.device)
        y = torch.arange(k, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

        # Flatten the grid to create a list of positions
        positions = torch.stack(
            [grid_x.flatten(), grid_y.flatten()], dim=1
        )  # Convert to float for distance calculations

        return positions

    def build_diffraction_matrix(self):
        k = self.kernel_size[0]

        # Coordinates for the first matrix

        coords2 = coords1 = self.construct_positions(k)

        coords1_expanded = coords1[:, None, :]  # Shape becomes [k, 1, 2]
        coords2_expanded = coords2[None, :, :]  # Shape becomes [1, k, 2]

        # Compute the differences in positions using broadcasting
        # Notice this is meta-atoms pixel positions, not actual distance.
        # we define meta-atom pixel dimension is 0.1 um, keep th same unit as wavelength 1.55 um
        # print(coords1_expanded)
        # print(coords2_expanded)
        # print(coords1_expanded- coords2_expanded)
        # exit(0)
        pixel_size = self.pixel_size.abs()  # um
        coords_differences = pixel_size * (
            coords1_expanded - coords2_expanded
        )  # Resulting shape will be [k, k, 2]

        delta_z = self.delta_z.abs()  # um
        # Calculate the squared differences in the x and y coordinates, summing over the last dimension, and add 1 for the z-distance squared
        squared_distances = coords_differences.square().sum(2) + delta_z**2

        # Take the square root to get the Euclidean distances
        distances_efficient = torch.sqrt(squared_distances)

        self.transfer_matrix = (
            (1 / (2 * np.pi))
            * (delta_z / squared_distances)
            * (1 / distances_efficient - self.wv * 1j)
            * (torch.exp(1j * self.wv * distances_efficient))
        )

        # print(self.transfer_matrix)
        # print(self.transfer_matrix.norm(p=2, dim=0))
        # print(self.transfer_matrix.shape)
        # exit(0)
        # self.H_matrix = torch.zeros(*self.kernel_size)
        # matrix1 = np.indices(*self.kernel_size).T.reshape(-1, 2)
        # for i in self.H_matrix.shape[0]:
        #     for j in self.H_matrix.shape[1]:
        #         distances = np.sqrt((matrix1[:, 0] - i) ** 2 + (matrix1[:, 1] - j) ** 2 + 1 ** 2)
        #         self.H_matrix[i, j] =

    def _forward_impl(self, x: Tensor) -> Tensor:        
        # assert (
        #     x.size(-1) == self.in_channels
        # ), f"[E] Input dimension does not match the weight size {self.out_channels, self.in_channels}, but got input size ({tuple(x.size())}))"

        ## TODO: not support group convolution
        # modulation
        # x: [bs, inc, h, w] real
        # x = self.add_input_noise(x)

        # x :[bs, inc, 2, h, w]

        ##################
        # Begin MetaConv #
        ##################

        weight = self.build_weight()  # [outc, inc, d, 2, s, s]
        beta = self.build_beta()
        alpha = self.build_alpha()

        ################
        # End MetaConv #
        ################

        ###################
        # Begin Benchmark #
        ###################

        # weight = self.build_weight()  # [outc, inc, d, 2, s, s]
        # beta = self.build_beta()
        # alpha = self.build_alpha()

        #################
        # End Benchmark #
        #################

        # Setup x, y wise ratio:

        # each metasurface is (M, F)
        # (F), M, F, M, F, M, F, total d masks for each image
        #      ----  ----  ----

        # first scale input size to match metasurface size via zoom lens
        input_size = x.shape[-2:]
        # print(f"First: {x}")
        if x.shape[-1] != self.kernel_size[0]:
            x = torch.nn.functional.interpolate(
                x, size=self.kernel_size, mode="nearest"
            )  # this one has problem
        
        # # first diffraction on x can be shared across out_channels
        # x = torch.fft.fft2(x, norm="ortho")  # [bs, inc, s, s] complex
            
        x = x.unsqueeze(1).unsqueeze(3)  # [bs, inc, s, s] -> [bs, 1, inc, 2, s, s]
        x = x.repeat(1, 1, 1, 2, 1, 1)

        for i in range(self.path_depth):
            kernel = weight[None, :, :, i]  # [1, outc, inc, 2, s, s]
            # input [bs, 1 or outc, inc, 2, s, s]
            # kernel [1, outc, inc, 2, s, s]
            x = x * kernel  # [bs, outc, inc, 2, s, s]

            bs_s, outc_s, inc_s, xy_s, ker_s, _ = x.shape
            #Real diffraction function
            self.build_diffraction_matrix()
            x = torch.matmul(x.flatten(-2, -1), self.transfer_matrix) # [bs, outc, inc, 2, s^2]

            # For testing diffraction matrix function
            # x = torch.fft.fft(x, dim=-1)

            x = x.view(
                bs_s, outc_s, inc_s, xy_s, ker_s, ker_s
            )  # [bs, outc, inc, 2, s, s]
            # print(f"Loop 2, Norm {i}:  {torch.norm(x.detach()).item()}")
        # rescale it back via zoom lens
        x = x.flatten(0, 1)  # [bs*outc, inc, 2, s, s]

        # [bs*outc, inc, s, s]
        x_wise_real = x[:, :, 0, :, :].real
        x_wise_imag = x[:, :, 0, :, :].imag
        y_wise_real = x[:, :, 1, :, :].real
        y_wise_imag = x[:, :, 1, :, :].imag

        if input_size[-1] != x_wise_real.shape[-1]:
            # [bs*outc, inc, h, w]
            x_wise_real = torch.nn.functional.interpolate(
                x_wise_real, size=input_size, mode="nearest"
            )
            x_wise_imag = torch.nn.functional.interpolate(
                x_wise_imag, size=input_size, mode="nearest"
            )

            # [bs*outc, inc, h, w]

            y_wise_real = torch.nn.functional.interpolate(
                y_wise_real, size=input_size, mode="nearest"
            )
            y_wise_imag = torch.nn.functional.interpolate(
                y_wise_imag, size=input_size, mode="nearest"
            )

        # [bs*outc, inc, h, w]real

        x_wise = (
            x_wise_real.square() + x_wise_imag.square()
        ) * beta  # x_wise detection
        y_wise = (y_wise_real.square() + y_wise_imag.square()) * (
            1 - beta
        )  # y_wise detection

        x = x_wise - y_wise  # [bs*outc, inc, h, w] real

        ##########################################
        #Comment out if you want to run benchmark#
        ##########################################

        x = x * alpha

        ##########################################
        #Comment out if you want to run benchmark#
        ##########################################

        # x = self.add_detection_noise(x)
        #
        # sum over all input channels
        x = x.sum(dim=1)  # [bs*outc, h, w] real
        x = x.view(-1, self.out_channels, *input_size)

        if self.bias is not None:
            x = x + self.bias[None, :, None, None]
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

@MODELS.register_module()
class MetaConv2d(Meta_Layer_BASE):
    _conv_types = _MetaConv2dMultiPath

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        n_pads: int = 5,
        bias: bool = True,
        w_bit: int = 16,
        in_bit: int = 16,
        # constant scaling factor from intensity to detected voltages
        input_uncertainty: float = 0,
        pad_noise_std: float = 0,
        dpe=None,
        pad_max: float = 1.0,
        sigma_trainable: str = "row_col",
        mode: str = "usv",
        # reinforce_training: bool = False,
        path_multiplier: int = 2,
        # parallel_path: int = 1,
        path_depth: int = 2,
        unfolding: bool = False,
        device: Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        verbose: bool = False,
        with_cp: bool = False,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            n_pads=n_pads,
            kernel_size=kernel_size,
            w_bit=w_bit,
            in_bit=in_bit,
            input_uncertainty=input_uncertainty,
            pad_noise_std=pad_noise_std,
            dpe=dpe,
            pad_max=pad_max,
            mode=mode,
            # reinforce_training=reinforce_training,
            # parallel_path=parallel_path,
            path_multiplier=path_multiplier,
            path_deth=path_depth,
            unfolding=unfolding,
            device=device,
            verbose=verbose,
            with_cp=with_cp,
        )

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.in_channels_pos = self.in_channels
        self.in_channels_neg = 0 if unfolding else self.in_channels
        self._conv_pos = _MetaConv2dMultiPath(
            in_channels=self.in_channels_pos,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            n_pads=n_pads,
            bias=False,
            w_bit=w_bit,
            in_bit=in_bit,
            input_uncertainty=input_uncertainty,
            pad_noise_std=pad_noise_std,
            dpe=dpe,
            pad_max=pad_max,
            sigma_trainable=sigma_trainable,
            mode=mode,
            # reinforce_training=reinforce_training,
            # parallel_path=parallel_path,
            path_multiplier=path_multiplier,
            path_depth=path_depth,
            unfolding=unfolding,
            device=device,
            verbose=verbose,
            with_cp=with_cp,
        )
        self._conv_neg = None
        #     (
        #     _MetaConv2dMultiPath(
        #         in_channels=self.in_channels_pos,
        #         out_channels=out_channels,
        #         kernel_size=kernel_size,
        #         stride=stride,
        #         padding=padding,
        #         dilation=dilation,
        #         groups=groups,
        #         n_pads=n_pads,
        #         bias=False,
        #         w_bit=w_bit,
        #         in_bit=in_bit,
        #         input_uncertainty=input_uncertainty,
        #         pad_noise_std=pad_noise_std,
        #         dpe=dpe,
        #         pad_max=pad_max,
        #         sigma_trainable=sigma_trainable,
        #         mode=mode,
        #         # reinforce_training=reinforce_training,
        #         # parallel_path=parallel_path,
        #         path_multiplier=path_multiplier,
        #         path_depth=path_depth,
        #         unfolding=unfolding,
        #         device=device,
        #         verbose=verbose,
        #         with_cp=with_cp,
        #     )
        #     if self.in_channels_neg > 0
        #     else None
        # )

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels).to(self.device))
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, self._conv_types):
                m.reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()

    def requires_grad_Meta(self, mode: bool = True):
        self._requires_grad_Meta = mode
        for m in self.modules():
            if isinstance(m, self._conv_types):
                m.requires_grad_Meta(mode)

    def get_perm_loss(self):
        for m in self.modules():
            if isinstance(m, self._conv_types):
                m.get_perm_loss()

    def get_alm_perm_loss(self):
        for m in self.modules():
            if isinstance(m, self._conv_types):
                m.get_alm_perm_loss()

    def set_input_er(self, er: float = 0, x_max: float = 6.0) -> None:
        ## extinction ratio of input modulator
        self.input_er = er
        self.input_max = x_max
        for m in self.modules():
            if isinstance(m, self._conv_types):
                m.set_input_er(er, x_max)

    def set_input_snr(self, snr: float = 0) -> None:
        self.input_snr = snr
        for m in self.modules():
            if isinstance(m, self._conv_types):
                m.set_input_snr(snr)

    def set_detection_snr(self, snr: float = 0) -> None:
        self.detection_snr = snr
        for m in self.modules():
            if isinstance(m, self._conv_types):
                m.set_detection_snr(snr)

    @property
    def _weight(self):
        # control pads to complex transfer matrix
        # [p, q, n_pads] real -> [p, q, k, k] complex
        weights = []
        for m in self.modules():
            if isinstance(m, self._conv_types):
                weights.append(m._build_weight())
        return weights

    @property
    def _weight_unroll(self):
        weights = []
        for m in self.modules():
            if isinstance(m, self._conv_types):
                weights.append(m.build_weight_unroll(m._build_weight()))
        return weights

    @property
    def _weight_complex(self):
        weights = []
        for m in self.modules():
            if isinstance(m, self._conv_types):
                weights.append(m.build_weight(m._build_weight()))
        return weights

    def _forward_impl(self, x):
        y = self._conv_pos(x)
        if self._conv_neg is not None:
            y_neg = self._conv_neg(x)
            y = y - y_neg

        if self.bias is not None:
            y = y + self.bias[None, :, None, None]
        return y

    def get_output_dim(self, img_height: int, img_width: int) -> Tuple[int, int]:
        h_out = (img_height - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[
            0
        ] + 1
        w_out = (img_width - self.kernel_size[1] + 2 * self.padding[1]) / self.stride[
            1
        ] + 1
        return (int(h_out), int(w_out))

    def forward(self, x):
        if self.in_bit <= 8:
            x = self.input_quantizer(x)
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self._forward_impl, x)
        else:
            out = self._forward_impl(x)

        # print(f"Final3333, Norm: {torch.norm(out.detach()).item()}")
        return out


# class PhiPLUT(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, P, lut, approximate_derivative_fn):
#         # Forward: Use exact values from LUT
#         phi = lut[P]
#         # Save tensor for backward pass
#         ctx.save_for_backward(P, phi)
#         ctx.approximated_derivative_fn = approximate_derivative_fn
#
#         return phi
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         P, phi = ctx.saved_tensors
#
#         # Backward: Use approximate function to compute gradient
#         delta_P = ctx.approximate_derivative_fn(phi)
#         grad_P = delta_P * grad_output
#
#         return grad_P, None, None
