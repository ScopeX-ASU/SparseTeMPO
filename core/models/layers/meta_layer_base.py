from typing import Optional
import numpy as np
from pyutils.torch_train import set_torch_deterministic
import torch

from pyutils.compute import gen_gaussian_noise
from pyutils.general import logger
from pyutils.quantize import input_quantize_fn, weight_quantize_fn
from torch import Tensor, nn
from torch.types import Device

from .utils import pad_quantize_fn
from pyutils.general import print_stat

__all__ = ["Meta_Layer_BASE"]


class Meta_Layer_BASE(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_pads: int = 0,  # zero control pads for metasurface by default. By default is passive metasuface
            w_bit: int = 16,
            in_bit: int = 16,
            # constant scaling factor from intensity to detected voltages
            input_uncertainty: float = 0,
            pad_noise_std: float = 0,
            Meta: Optional[nn.Module] = None,
            pad_max: float = 1.0,
            mode: str = "usv",
            path_multiplier: int = 2,  # how many metalens in parallel
            path_depth: int = 2,  # how may metalens cascaded
            unfolding: bool = False,
            sigma_trainable: str = "row_col",
            device: Device = torch.device("cuda"),
            verbose: bool = False,
            with_cp: bool = False,
            gumbel_temperature: float = 2.0,
            **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_pads = n_pads
        self.w_bit = w_bit
        self.in_bit = in_bit
        self.pad_max = pad_max
        # constant scaling factor from intensity to detected voltages
        self.input_uncertainty = input_uncertainty
        self.pad_noise_std = pad_noise_std
        self.mode = mode
        self.path_multiplier = path_multiplier
        self.path_depth = path_depth
        self.unfolding = unfolding
        self.sigma_trainable = sigma_trainable
        self.gumbel_temperature = gumbel_temperature

        # data Meta
        self.Meta = Meta

        self.verbose = verbose
        self.with_cp = with_cp
        self.device = device

        # allocate parameters
        self.weight = None
        self.path_weight = None
        self.sigma = None
        self.x_zero_pad = None

        # quantization tool
        self.pad_quantizer = pad_quantize_fn(max(2, self.w_bit), v_max=pad_max, quant_ratio=1)
        self.sigma_quantizer = weight_quantize_fn(w_bit=self.w_bit, alg="dorefa_sym")
        self.input_quantizer = input_quantize_fn(in_bit=in_bit, device=device, alg="normal")

        self._requires_grad_Meta = True
        self.input_er = 0
        self.input_max = 6
        self.input_snr = 0
        self.detection_snr = 0
        self.pad_noise_std = 0

    def build_parameters(self, bias: bool):
        raise NotImplementedError

    def reset_parameters(self, fan_in=None):
        # nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        # the fourier-domain convolution is equivalent to h x w kernel-size convolution
        if hasattr(self, "in_channels_flat"):
            in_channels = self.in_channels_flat
        else:
            in_channels = self.in_channels
        if hasattr(self, "kernel_size"):
            fan_in = np.prod(self.kernel_size) * in_channels
        std = (1 / fan_in) ** 0.5

        if self.weight is not None:
            # self.weight.data[..., 0].normal_(0, std / self.path_depth)
            # self.weight.data[..., 0].fill_(0) # zero phase initialization
            # self.weight.data[..., 1].fill_(0)
            nn.init.uniform_(self.weight, 0, 0)

        if self.bias is not None:
            nn.init.uniform_(self.bias, 0, 0)

    def requires_grad_Meta(self, mode: bool = True):
        self._requires_grad_Meta = mode

    def set_input_er(self, er: float = 0, x_max: float = 6.0) -> None:
        ## extinction ratio of input modulator
        self.input_er = er
        self.input_max = x_max

    def set_input_snr(self, snr: float = 0) -> None:
        self.input_snr = snr

    def set_detection_snr(self, snr: float = 0) -> None:
        self.detection_snr = snr

    def add_input_noise(self, x: Tensor) -> Tensor:
        if 1e-5 < self.input_er < 80:
            x_max = self.input_max
            x_min = x_max / 10 ** (self.input_er / 10)
            x = x.mul((x_max - x_min) / x_max).add(x_min)
        if 1e-5 < self.input_snr < 80:
            avg_noise_power = 1 / 10 ** (self.input_snr / 10)
            noise = gen_gaussian_noise(x, 1, noise_std=avg_noise_power ** 0.5)
            return x.mul(noise)
        return x

    def add_detection_noise(self, x: Tensor) -> Tensor:
        if 1e-5 < self.detection_snr < 80:
            avg_noise_power = 0.5 / 10 ** (self.detection_snr / 10)
            noise = gen_gaussian_noise(x, 0, noise_std=avg_noise_power ** 0.5)
            return x.add(noise)
        return x

    def set_gumbel_temperature(self, T: float = 5.0):
        self.gumbel_temperature = T

    # def path_generation(self, hard: bool = False):
    #     logits = self.path_weight
    #     # print(self.path_weight.shape)
    #     path_prob = torch.nn.functional.gumbel_softmax(
    #             logits,
    #             tau=self.gumbel_temperature,
    #             hard=hard,
    #             dim=-1,
    #         )
    #     # print(f"path prob weight: {path_prob}")
    #     selected_path = torch.argmax(path_prob, dim=-1)
    #     # print(f"selected path weight: {selected_path}")
    #     return selected_path

    # def path_generation(self, path_weight):
    #     '''
    #     Original version of path_generator
    #     '''
    #     """TODO, randomly generate some path
    #     This is a random path generation with fixed random seed.
    #     path contention is not resolved, so it most likely will be mapped to hardware one path at a time
    #     If contention can be fully addressed, e.g.,
    #     [0, 0, 0, 0]
    #     [1, 2, 1, 3]
    #     [2, 1, 3, 1]
    #     [3, 3, 2, 2]
    #     i.e., each column is a permutation of [0,1,2,3], without overlapped indices.
    #     Then we can parallelly map 4 paths on the hardware at one shot
    #     This permutation matrix corresponds to an all-to-all routing,
    #     can be differentiably learned as ADEPT [Gu+, DAC'22]
    #     """
    #     paths = []
    #     for o in range(self.out_channels):
    #         paths_out = []
    #         for i in range(self.in_channels):
    #             paths_in = []
    #             for d in range(self.path_depth):
    #                 random_state = o + i + d
    #                 set_torch_deterministic(random_state)
    #                 p = np.random.choice(self.path_multiplier + 1)
    #                 if p < self.path_multiplier:
    #                     paths_in.append(p)
    #                 else:
    #                     break
    #             paths_in.extend([self.path_multiplier] * int(
    #                 self.path_depth - len(paths_in)))  # pad self.path_depth for identity phase mask
    #             paths_out.append(paths_in)
    #         paths.append(paths_out)
    #     paths = torch.tensor(paths, dtype=torch.long)  # tensorize, no need on CUDA.
    #     print(paths)
    #     return paths

    # def path_generation(self, path_weight):
    #   """
    #   Version 1, able to get rid of one of the inner loop, still ensured the reproducibility
    #   """
    #     paths = torch.empty(self.out_channels, self.in_channels, self.path_depth, dtype=torch.long)
    #
    #     for o in range(self.out_channels):
    #         for i in range(self.in_channels):
    #             random_state = o + i
    #             set_torch_deterministic(random_state)
    #             path = torch.randint(0, self.path_multiplier + 1, (self.path_depth,))
    #             '''
    #             The code below corresponds to the path_depth loop
    #             '''
    #
    #             # Instead of padding the rest into self.path_multiplier,
    #             # The cumulative_mask will directly change every value
    #             # after the (depth - len(path_in)) into self.path_multiplier
    #             # Achieving the same goal.
    #             mask = path == self.path_multiplier
    #             cumulative_mask = mask.cumsum(dim=0)
    #             path[mask] = self.path_multiplier
    #             path[cumulative_mask > 1] = self.path_multiplier
    #             paths[o, i] = path
    #     return paths

    # def path_generation(self, path_weight):
    #     """
    #     Version 2, get rid of all nest loop, but no unique sequence for each o and i loop,
    #     but deterministic behavior remains
    #     """
    #     set_torch_deterministic(0)
    #     paths = torch.randint(0, self.path_multiplier + 1, (self.out_channels, self.in_channels, self.path_depth))
    #
    #     # Adjust using o and i for "unique" paths
    #     o_adjustment = torch.arange(self.out_channels).view(-1, 1, 1)
    #     i_adjustment = torch.arange(self.in_channels).view(1, -1, 1)
    #     paths = (paths + o_adjustment + i_adjustment) % (self.path_multiplier + 1)
    #
    #     # Handle early termination and padding
    #     mask = paths == self.path_multiplier
    #     cumulative_mask = mask.cumsum(dim=-1)
    #     paths[mask] = self.path_multiplier
    #     paths[cumulative_mask > 1] = self.path_multiplier
    #
    #     return paths

    # def generate_permutation_paths(self, path_weight):
    #     if self.in_channels != self.path_depth:
    #         raise ValueError("path_depth must be equal to out_channels for permutation generation!")
    #
    #     # Ensure that in_channels doesn't exceed out_channels
    #     if self.out_channels > self.in_channels:
    #         raise ValueError(
    #             f"Number of in_channels {self.out_channels} cannot exceed out_channels "
    #             f"{self.in_channels} for unique permutation generation.")
    #
    #     # Generate a base permutation tensor
    #     all_matrices = []
    #
    #     for i in range(self.out_channels):
    #
    #         # Ensure deterministic behavior
    #         set_torch_deterministic(self.out_channels + i)
    #
    #         # Generate a base permutation tensor
    #         base_permutation = torch.randperm(self.in_channels)
    #
    #         # Container for all permutations
    #         all_permutations = [base_permutation]
    #
    #         # Generate the remaining permutations by shifting the base permutation
    #         for _ in range(1, self.in_channels):
    #             base_permutation = torch.roll(base_permutation, shifts=1, dims=0)
    #             all_permutations.append(base_permutation)
    #
    #         # Stack all permutations into a matrix
    #         permutation_matrix = torch.stack(all_permutations, dim=0)
    #         all_matrices.append(permutation_matrix)
    #
    #     # Stack all matrices to form the final 3D tensor
    #     permutation_tensor = torch.stack(all_matrices, dim=1)
    #
    #     return permutation_tensor


    @property
    def _weight(self):
        # control pads to complex transfer matrix
        # [m, d, s, s] complex metasurface masks + 
        # return [m, d, s, s] quantized metasurface phase profile/mag+phase profile, and [c_out, c_in] paths 

        # mask: quantize the metasurface mask, if one channel used, it means phase, otherwise is complex value for
        # mag/phase
        if self.w_bit < 16:
            weight = self.sigma_quantizer(self.weight)  # [m, d, s, s]
        else:
            weight = self.weight
        # path:
        # paths = self.path_generation(True)

        # test_path = self.path_generator(True)

        return weight

    def _forward_impl(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self._forward_impl(x)
