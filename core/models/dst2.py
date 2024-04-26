from __future__ import print_function

import copy
import math
import random
from itertools import combinations
from typing import Callable, Dict, List, Optional, Tuple, Union
from functools import lru_cache
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pyutils.general import logger, ensure_dir
from torch import Tensor, nn
import matplotlib.pyplot as plt

__all__ = ["DSTScheduler2", "CosineDecay", "LinearDecay"]

DEBUG = False


class CosineDecay(object):
    def __init__(self, death_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.sgd = optim.SGD(
            torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate
        )
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.sgd, T_max, eta_min, last_epoch
        )

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self, death_rate):
        return self.sgd.param_groups[0]["lr"]


class LinearDecay(object):
    def __init__(self, death_rate, factor=0.99, frequency=600):
        self.factor = factor
        self.steps = 0
        self.frequency = frequency

    def step(self):
        self.steps += 1

    def get_dr(self, death_rate):
        if self.steps > 0 and self.steps % self.frequency == 0:
            return death_rate * self.factor
        else:
            return death_rate


def parameters_distribution(model):

    emb_all = 0
    mlp_all = 0
    att_mlp_all = 0
    att_qkv_all = 0
    others = 0
    for name, tensor in model.named_parameters():
        if "embed.proj" in name:
            emb_all += tensor.numel()
        elif "attn.proj" in name:
            att_mlp_all += tensor.numel()
        elif "attn.qkv" in name:
            att_qkv_all += tensor.numel()
        elif "mlp" in name:
            mlp_all += tensor.numel()
        else:
            others += tensor.numel()
    total = emb_all + att_mlp_all + att_qkv_all + mlp_all + others
    print("all:{}".format(total))
    print("embeding:{} /{:.2f}".format(emb_all, emb_all / total))
    print("attn mlp:{} /{:.2f}".format(att_mlp_all, att_mlp_all / total))
    print("attn qkv:{} /{:.2f}".format(att_qkv_all, att_qkv_all / total))
    print("mlp all :{} /{:.2f}".format(mlp_all, mlp_all / total))
    print("others  :{} /{:.2f}".format(others, others / total))


class MultiMask(object):
    def __init__(
        self,
        mask_cfg={"row_mask": [4, 4, 4, 1, 4, 1], "col_mask": [4, 4, 1, 4, 1, 4]},
        device="cuda:0",
    ) -> None:
        self.mask_cfg = mask_cfg
        self._masks = {
            name: (
                torch.ones(cfg, device=device, dtype=torch.bool)
                if not torch.is_tensor(cfg)
                else cfg.to(device)
            )
            for name, cfg in mask_cfg.items()
        }

        try:
            mask = self.data
        except:
            raise ValueError("mask shapes should be able to multiplied together.")

        self.total_elem = mask.numel()
        self.shape = mask.shape

    def __getitem__(self, key):
        return self._masks[key]

    def __setitem__(self, key, value):
        self._masks[key] = value

    @property
    def data(self):
        masks = list(self._masks.values())
        out = masks[0]
        for mask in masks[1:]:
            out = out & mask
        return out

    def size(self):
        return self.shape

    def numel(self):
        return self.total_elem

    def num_nonzeros(self):
        return self.sum().item()

    def num_zeros(self):
        return self.numel() - self.num_nonzeros()

    def get_density(self):
        return self.num_nonzeros() / self.numel()

    def sum(self):
        return self.data.sum()

    def __mul__(self, other):
        if isinstance(other, MultiMask):
            new_mask = self.clone()
            new_mask._masks = {
                name: mask * other[name] for name, mask in new_mask._masks.items()
            }
            return new_mask
        return self.data * other

    def __rmul__(self, other):
        if isinstance(other, MultiMask):
            new_mask = self.clone()
            new_mask._masks = {
                name: mask * other[name] for name, mask in new_mask._masks.items()
            }
            return new_mask

        return self.data * other

    def __eq__(self, other):
        if isinstance(other, MultiMask):
            return self.data == other.data
        return self.data == other

    def __invert__(self):
        return ~self.data

    def __and__(self, other):
        if isinstance(other, MultiMask):
            new_mask = self.clone()
            new_mask._masks = {
                name: mask & other[name] for name, mask in new_mask._masks.items()
            }
            return new_mask
        return self.data & other

    def __rand__(self, other):
        return self.__and__(other)

    def __or__(self, other):
        if isinstance(other, MultiMask):
            new_mask = self.clone()
            new_mask._masks = {
                name: mask | other[name] for name, mask in new_mask._masks.items()
            }
            return new_mask
        return self.data | other

    def __ror__(self, other):
        return self.__or__(other)

    def __xor__(self, other):
        if isinstance(other, MultiMask):
            new_mask = self.clone()
            new_mask._masks = {
                name: mask ^ other[name] for name, mask in new_mask._masks.items()
            }
            return new_mask
        return self.data ^ other

    def __rxor__(self, other):
        return self.__xor__(other)

    def clone(self):
        return copy.deepcopy(self)


class DSTScheduler2(nn.Module):
    _death_modes = {
        "magnitude",
        "random",
        "magnitude_power",
        "magnitude_crosstalk",
        "magnitude_power_crosstalk",
        "magnitude_crosstalk_power",
    }
    _growth_modes = {
        "random",
        "gradient",
        "gradient_power",
        "gradient_crosstalk",
        "gradient_crosstalk_power",
        "gradient_power_crosstalk",
    }
    _pruning_types = {
        "unstructure",
        "structure_row",
        "structure_col",
        "structure_row_col",
    }

    def __init__(
        self,
        optimizer,
        death_rate: float = 0.3,
        growth_death_ratio: float = 1.0,
        death_rate_decay=None,
        death_mode: str = "magnitude",
        growth_mode: str = "gradient",
        redistribution_mode: str = "momentum",
        args=None,
        spe_initial=None,
        train_loader=None,
        pruning_type: str = "structure_row",
        pi_shift_power: float = 30,
        power_choice_margin: int = 2,
        ADC_power: float = 7.5,
        TIA_power: float = 3,
        HDAC_power: float = 6.43,
        first_layer_pruning=True,
        update_frequency: int = 100,
        T_max: int = 10000,
        group: str = "layer",  # layer, block wise magnitude sorting
        splitter_biases: float | List[int] = 90,
        max_combinations: int = 100,  # set a maximum combinations to enumerate. otherwise it might have too many combinations
        device="cuda:0",
    ) -> None:
        ## Dynamic Sparse Training Scheduler
        super().__init__()
        self.args = args
        self.loader = train_loader
        self.modules = []
        self.optimizer = optimizer

        if pruning_type not in self._pruning_types:
            raise ValueError(
                f"pruning_type expects {self._pruning_types}, but got {pruning_type}."
            )
        self.pruning_type = pruning_type

        self.growth_death_ratio = growth_death_ratio
        if growth_mode not in self._growth_modes:
            raise ValueError(
                f"Growth mode expects {self._growth_modes}, but got {growth_mode}."
            )

        self.growth_mode = growth_mode  # gradient
        opt = [m for m in self.growth_mode.split("_")[1:]]
        if self.pruning_type == "structure_row":
            opt = [m for m in opt if m != "power"]
            logger.info(
                f"structure_row does not support power optimization, growth_opts reduced to {opt}"
            )
        self.growth_opts = opt

        self.redistribution_mode = redistribution_mode  # momentum
        self.spe_initial = spe_initial  # initial masks made by SNIP
        self.snip_masks = None  # masks made by SNIP during training
        self.nonzeros_index = None

        self.update_frequency = update_frequency
        self.T_max = T_max
        self.group = group
        self.max_combinations = max_combinations
        self.first_layer_pruning = first_layer_pruning
        self.steps = 0
        self.device = device

        self.names = []
        self.masks = {}
        self.atten_masks = {}
        self.other_masks = {}
        self.newly_masks = {}
        # death
        self.death_mode = death_mode  # magnitude
        if death_mode not in self._death_modes:
            raise ValueError(
                f"Death mode expects {self._death_modes}, but got {death_mode}."
            )

        opt = [m for m in self.death_mode.split("_")[1:]]
        if self.pruning_type == "structure_row":
            opt = [m for m in opt if m != "power"]
            logger.info(
                f"structure_row does not support power optimization, death_mode reduced to {opt}"
            )
        self.death_opts = opt

        self.death_rate = death_rate
        self.death_rate_decay = death_rate_decay
        self.name2death_rate = {}
        self.splitter_biases = splitter_biases

        # Power exploration.
        # Default, no exploration on power
        self.set_death_power_exploration(False)
        self.set_grow_power_exploration(False)
        self.pi_shift_power = pi_shift_power
        self.power_choice_margin = power_choice_margin
        self.ADC_power = ADC_power
        self.TIA_power = TIA_power
        self.HDAC_power = HDAC_power

        # stats
        self.name2zeros = {}
        self.name2nonzeros = {}
        self.nonzeros_attn_heads = {}
        self.survival = {}
        self.pruned_number = {}
        self.params = {}

    """
    Basic
    """

    def add_module(
        self,
        module,
        density: float,
        init_mode: str = "uniform",
        mask_path=None,
        pruning_type: str | None = None,
    ):
        pruning_type = pruning_type or self.pruning_type
        if pruning_type in {"unstructure"}:
            self.modules.append(module)
            self.set_splitter_bias(biases=self.splitter_biases)
            index = len(self.masks)
            for name, m in module.named_modules():
                if isinstance(m, module._conv):  # no last fc layer
                    name_cur = name + "_" + str(index)
                    index += 1
                    self.names.append(name_cur)
                    self.params[name_cur] = m.weight  # [p, q, k, k]
                    self.masks[name_cur] = MultiMask(
                        {"elem_mask": m.weight.shape}, device=self.device
                    )
                    m.prune_mask = self.masks[
                        name_cur
                    ]  # the layer needs the mask to perform forward computation, pruning the weight is not enough.
            logger.info(f"created pruning mask.")
            self.unstructure_init(mode=init_mode, density=density, mask_file=mask_path)
            logger.info(f"initialized pruning mask.")
        elif pruning_type in {"structure_row", "structure_col", "structure_row_col"}:

            self.modules.append(module)
            self.set_splitter_bias(biases=self.splitter_biases)
            index = len(self.masks)
            for name, m in module.named_modules():
                if isinstance(m, module._conv):
                    name_cur = name + "_" + str(index)
                    index += 1
                    self.names.append(name_cur)
                    self.params[name_cur] = m.weight  # [p, q, r, c, k1, k2]
                    shape = list(m.weight.shape)
                    row_shape = copy.deepcopy(shape)
                    col_shape = copy.deepcopy(shape)
                    row_shape[-3] = 1
                    row_shape[-1] = 1
                    col_shape[-4] = 1
                    col_shape[-2] = 1
                    self.masks[name_cur] = MultiMask(
                        mask_cfg={"row_mask": row_shape, "col_mask": col_shape},
                        device=self.device,
                    )
                    m.prune_mask = self.masks[
                        name_cur
                    ]  # the layer needs the mask to perform forward computation, pruning the weight is not enough.
            logger.info(f"created pruning mask.")
            self.structure_init(mode=init_mode, density=density, mask_file=mask_path)
            logger.info(f"initialized pruning mask.")

        else:
            raise ValueError("unrecognize pruning type")

    def step(self, pruning_type: str | None = None):
        pruning_type = pruning_type or self.pruning_type
        ## apply pruning mask (inplace weight tensor modification) and update death rate
        self.apply_mask(pruning_type=pruning_type)
        self.death_rate_decay.step()
        for name in self.name2death_rate:
            if self.args.death_rate_decay == "cosine":
                self.name2death_rate[name] = self.death_rate_decay.get_dr(
                    self.name2death_rate[name]
                )
            elif self.args.death_rate_decay == "constant":
                self.name2death_rate[name] = self.args.death_rate
            self.death_rate = self.name2death_rate[name]
        self.steps += 1

        if self.steps % self.update_frequency == 0 and self.steps < self.T_max:
            self.at_end_of_epoch(pruning_type)

    def at_end_of_epoch(
        self, indicator_list=None, pruning_type: str | None = None
    ) -> None:
        pruning_type = pruning_type or self.pruning_type
        if pruning_type == "unstructure":
            self.update_and_apply_mask()
            _, _ = self.update_fired_masks()
            self.print_nonzero_counts()
        elif pruning_type in {"structure_row", "structure_col", "structure_row_col"}:
            self.update_and_apply_mask(pruning_type, indicator_list)
            _, _ = self.update_fired_masks(pruning_type="structure")

        else:
            raise ValueError(f"Unrecognized Pruning Type {pruning_type}")

    def resume(self, checkpoint, pruning_type, density):

        if pruning_type == "unstructure":
            print("loading masks from checkpoint")
            self.masks = checkpoint["mask"]

            self.apply_mask(pruning_type=pruning_type)
            total_size = 0
            for name, weight in self.masks.items():
                total_size += weight.numel()
            print("Total Model parameters:", total_size)

            sparse_size = 0
            for name, weight in self.masks.items():
                sparse_size += (weight != 0).sum().int().item()
            print(
                "Total parameters under density level of {0}: {1}".format(
                    density, sparse_size / total_size
                )
            )

        else:
            print("loading other_mask and atten_mask from checkpoint")
            self.other_masks = checkpoint["other_mask"]
            self.atten_masks = checkpoint["atten_mask"]
            self.apply_mask(pruning_type=pruning_type)
            self.print_structure_mask()

        if "mask_step" in checkpoint.keys():
            print("resume death rate with steps = ", checkpoint["mask_step"])
            self.steps = checkpoint["mask_step"]
            for _ in range(self.steps):
                self.death_rate_decay.step()
            for name in self.name2death_rate:
                if self.args.decay_schedule == "cosine":
                    self.name2death_rate[name] = self.death_rate_decay.get_dr(
                        self.name2death_rate[name]
                    )
                elif self.args.decay_schedule == "constant":
                    self.name2death_rate[name] = self.args.death_rate
                self.death_rate = self.name2death_rate[name]

        if "fired_mask" in checkpoint.keys():
            print("loading fired mask for calculation RS")
            self.fired_masks = checkpoint["fired_mask"]

    """
    Basic Utility
    """

    def set_death_power_exploration(self, flag: bool = False) -> None:
        self.death_power_flag = flag

    def set_grow_power_exploration(self, flag: bool = False) -> None:
        self.grow_power_flag = flag

    def set_magnitude_based_flag(self, flag: bool = False) -> None:
        self.magnitude_based_flag = flag

    def set_gradient_based_flag(self, flag: bool = False) -> None:
        self.gradient_based_flag = flag

    def set_splitter_bias(self, biases: float | List[int] = 90) -> None:
        ## cos^2((delta_phi + bias)/2) = ratio
        if isinstance(biases, (int, float)):
            biases = [biases] * int(np.log2(self.modules[0].conv_cfg["miniblock"][-1]))
        self.splitter_biases = [b / 180 * np.pi for b in biases]

    def cal_ports_power(self, ports_array: Tensor) -> Tensor:
        ## ports_array: [#combinations, array_length] bool mask representing the sparsity pattern
        ## return: [#combinations] power of each sparsity pattern
        ## first fold the posrt_array to tensors [#combinations, 2, 2, ..., 2]
        n_levels = int(np.log2(ports_array.shape[1]))
        ports_array = ports_array.view([-1] + [2] * n_levels)
        power = 0
        # if DEBUG:
        #     print("ports_array shape:", ports_array.shape)
        #     print("n_levels:", n_levels)
        #     print(ports_array)
        for level in range(n_levels):
            ## e.g., k=8, n_levels=3
            ## L0: [..., 2, <2, 2>] sum[-2, -1],
            ## L1: [..., 2, 2, <2>] sum[-1],
            ## L2: [..., 2, 2, 2]   sum[],
            sum_dims = list(range(level - n_levels + 1, 0, 1))
            # if DEBUG:
            #     print(f"---------------- level {level}")
            #     print(f"sum_dims", sum_dims)
            if len(sum_dims) > 0:
                ports_sum = ports_array.sum(dim=sum_dims)
            else:
                ports_sum = ports_array

            ratios = ports_sum[..., 0:1] / ports_sum.sum(-1, keepdim=True)
            ## note that if the ratio is inf, i.e., two port of MZI are all 0
            ## then its ratio should be set to 50% to minimize power.
            ratios[torch.isnan(ratios)] = 0.5
            # print(ratios)

            ## L0: [..., <2>]       sum[-1],
            ## L1: [..., <2, 2>]    sum[-2, -1],
            ## L2: [..., <2, 2, 2>] sum[-3, -2, -1],
            sum_dims = list(range(-1 - level, 0, 1))
            ## cos^2((delta_phi + pi/2)/2) = ratio
            ## (1-sin(delta_phi))/2 = ratio
            ## |delta_phi| = |arccos(sqrt(ratio)) * 2 - pi/2|
            # delta_phi = (ratios.sqrt_()
            #     .acos()
            #     .mul_(2)
            #     .sub_(np.pi / 2))
            # print(delta_phi)
            p = (
                ratios.sqrt_()
                .acos()
                .mul_(2)
                .sub_(self.splitter_biases[level])
                .abs_()
                .mul_(self.pi_shift_power / np.pi)
                .sum(dim=sum_dims)
            )  # [#combinations]
            # p = (
            #     ratios.mul_(-2).add_(1)
            #     .asin_()
            #     .abs_()
            #     .mul_(self.pi_shift_power / np.pi)
            #     .sum(dim=sum_dims)
            # )  # [#combinations]
            # print("ratio", ratios)
            # print("p", p)
            power += p

        # print(power)

        return power  # [#combinations]

    @lru_cache(maxsize=32)
    def find_sparsity_patterns(self, array_length: int, num_zeros: int) -> Tensor:
        # Ensure that the number of zeros does not exceed the array length
        if num_zeros > array_length:
            raise ValueError(
                "The number of zeros cannot exceed the total array length."
            )

        # Generate all possible positions for zeros in the array
        patterns = []

        ## [::-1] means prefer pruning right/bottom side, which are paddings.
        for i, zero_indices in enumerate(
            combinations(range(array_length)[::-1], num_zeros)
        ):
            if i >= self.max_combinations:
                break
            array = torch.ones(array_length, dtype=torch.bool, device=self.device)
            if len(zero_indices) != 0:
                array[torch.tensor(zero_indices)] = 0
            patterns.append(array)

        return torch.stack(patterns)  # [#combinations, array_length] bool mask

    def magnitude_based_col_sparsity_patterns(
        self, remain_length, num_of_zeros, fixed_index
    ):
        fixed_index = np.array(fixed_index).reshape(-1)
        indices = np.arange(fixed_index.size)
        fixed_index = fixed_index - indices
        possible_patterns = self.col_sparsity_patterns(remain_length, num_of_zeros)
        if fixed_index.size != 0:
            possible_patterns = np.insert(possible_patterns, fixed_index, 1, axis=1)
        powers = self.cal_ports_power(possible_patterns)
        possible_patterns, _ = self.find_minimal_power_pattern(
            possible_patterns, powers
        )
        return self.find_least_crosstalk_patterns(possible_patterns, is_col=True)

    def gradient_based_col_sparsity_patterns(
        self, remain_length, num_of_zeros, fixed_index
    ):
        fixed_index = np.array(fixed_index).reshape(-1)
        indices = np.arange(fixed_index.size)
        fixed_index = fixed_index - indices
        possible_patterns = self.row_sparsity_patterns(remain_length, num_of_zeros)
        if fixed_index.size != 0:
            possible_patterns = np.insert(possible_patterns, fixed_index, 1, axis=1)
        powers = self.cal_ports_power(possible_patterns)
        possible_patterns, _ = self.find_minimal_power_pattern(
            possible_patterns, powers
        )
        return self.find_least_crosstalk_patterns(possible_patterns, is_col=True)

    def magnitude_based_row_sparsity_patterns(
        self, remain_length, num_of_zeros, fixed_index
    ):
        fixed_index = np.array(fixed_index).reshape(-1)
        indices = np.arange(fixed_index.size)
        fixed_index = fixed_index - indices
        possible_patterns = self.row_sparsity_patterns(remain_length, num_of_zeros)
        if fixed_index.size != 0:
            possible_patterns = np.insert(possible_patterns, fixed_index, 1, axis=1)
        return self.find_least_crosstalk_patterns(possible_patterns, is_col=False)

    def find_minimal_power_pattern(
        self, patterns: Tensor, pattern_powers: Tensor
    ) -> Tuple[Tensor, float]:
        ## given sparsity mask and powers, find the subsets with minimal power. some times not unique.
        lowest_power = pattern_powers.min()
        return patterns[pattern_powers == lowest_power], lowest_power.item()

    # def _possible_patterns(self, num_zeros, array_length):
    #     if num_zeros > array_length:
    #         return "The number of zeros cannot exceed the total array length."

    #     # Generate all possible positions for zeros in the array
    #     indices = range(array_length)
    #     zero_positions = list(combinations(indices, num_zeros))
    #     patterns = []
    #     for positions in zero_positions:
    #         # Initialize the array with all ones

    #         array = [1] * array_length
    #         # Place zeros in the specified positions

    #         for pos in positions:
    #             array[pos] = 0

    #         patterns.append(array)

    #     return patterns

    # def _sparsity_pattern_power_dictionary(self, num_zeros, array_length):
    #     # Ensure that the number of zeros does not exceed the array length

    #     if num_zeros > array_length:
    #         return "The number of zeros cannot exceed the total array length."

    #     # Generate all possible positions for zeros in the array
    #     indices = range(array_length)
    #     zero_positions = list(combinations(indices, num_zeros))
    #     power_dict = defaultdict(list)

    #     for positions in zero_positions:
    #         # Initialize the array with all ones

    #         array = [1] * array_length
    #         # Place zeros in the specified positions

    #         for pos in positions:
    #             array[pos] = 0

    #         # Calculate the power for this array configuration
    #         power = self._calc_total_and_upper_ports(array)

    #         # Add the power and array pair to the dictionary
    #         power_dict[power].append(array)

    #     return {key: value for key, value in sorted(power_dict.items())}

    def find_least_crosstalk_patterns(
        self, masks: Tensor, is_col: bool = True
    ) -> Tuple[Tensor, float]:
        ## among the possible patterns with min power, find the one with least crosstalk
        ## masks: [#combinations, array_length]
        ## best_patterns [#num, array_length], best_score: float
        if masks.shape[0] == 1:
            return masks, self.calc_crosstalk_score(masks[0], is_col)

        # If it is col pattern, just return the first pattern,
        # no need to calculate the crosstalk based on current design
        ### If later on need to calculate the col crosstalk pattern,
        ### You can delete this part of code
        if is_col:
            return masks, 0.0

        scores = []
        for mask in masks:
            score = self.calc_crosstalk_score(mask, is_col)
            scores.append(score)
        scores = torch.tensor(scores)
        max_score = scores.max().item()
        if DEBUG:
            print(masks.shape, masks)
            print("crosstalk_scores:", scores)
            print("Max score:", max_score)

        return masks[scores == max_score], max_score

    def calc_crosstalk_score(self, mask: Tensor, is_col: bool = True) -> float:
        ## given a sparsity mask, find its crosstalk score, higher score means less crosstalk
        ## the crosstalk is the sum of the negative exp distance between active elements, sum(exp(-d_ij))
        # active_indices = torch.nonzero(mask).squeeze()  # e.g., [0, 1, 3, 5]
        # num_active = active_indices.numel()
        # if num_active < 2:
        #     # Not enough active elements to calc separation or density.
        #     # should be higher than the best case with two actives
        #     ## treated as 2 actives with distance of k
        #     active_indices = torch.tensor([0, mask.shape[0]])
        #     num_active = 2

        # # calc distances between consecutive active elements

        # dist = (
        #     active_indices[None, ...] - active_indices[..., None]
        # ).float().abs() * self.node_spacing[int(is_col)]
        # total_crosstalk = torch.exp(-0.1 * dist).sum().item() - num_active

        # return -total_crosstalk
        return self.modules[0].crosstalk_scheduler.calc_crosstalk_score(mask, is_col)

    def calc_crosstalk_scores(self, masks: Tensor, is_col: bool = True) -> Tensor:
        ## given a sparsity mask, find its crosstalk score, higher score means less crosstalk
        ## the crosstalk is the sum of the negative exp distance between active elements, sum(exp(-d_ij))
        ## mask: [num_combinations, array_length]
        ## return: [num_combinations]
        # shape = mask.shape[:-1]
        # mask = mask.flatten(0, -2)
        # total_crosstalk = []
        # for m in mask:
        #     total_crosstalk.append(self.calc_crosstalk_score(m, is_col))
        # return torch.tensor(total_crosstalk, device=self.device).view(shape)
        return self.modules[0].crosstalk_scheduler.calc_crosstalk_scores(masks, is_col)

    def calc_TIA_ADC_powers(self, mask: Tensor) -> Tensor:
        ## given a sparsity mask, find its corresponding TIA and ADC power, please provide row mask only
        shape = mask.shape[:-1]
        mask = mask.flatten(0, -2)
        total_power = []
        for m in mask:
            array_length = m.shape[0]
            empty_rows = array_length - m.sum(-1)
            # print(total_power)
            total_power.append(
                self.calc_TIA_ADC_power(
                    array_length, empty_rows, self.TIA_power, self.ADC_power
                )
            )
        # print(total_power)
        return torch.tensor(total_power, device=self.device).view(shape)

    def calc_HDAC_powers(self, mask: Tensor) -> Tensor:
        ## given a sparsity mask, find its corresponding HDAC power, please provide column mask only
        shape = mask.shape[:-1]
        mask = mask.flatten(0, -2)
        total_power = []
        for m in mask:
            array_length = m.shape[0]
            empty_cols = array_length - m.sum(-1)
            total_power.append(
                self.calc_HDAC_power(array_length, empty_cols, self.HDAC_power)
            )
        return torch.tensor(total_power, device=self.device).view(shape)

    def calc_TIA_ADC_power(
        self,
        mask_length: int | Tensor,
        empty_rows: int | Tensor,
        TIA_power: float,
        ADC_power: float,
    ) -> float:
        return (mask_length - empty_rows) * (TIA_power + ADC_power)

    def calc_HDAC_power(
        self, mask_length: int, empty_cols: int, HDAC_power: float
    ) -> float:
        return (mask_length - empty_cols) * (HDAC_power)

    # def find_max_min_power_from_mask(self, mode="structure" TIA_power, ADC_power):
    #     if mode != "structure":
    #         raise ValueError("Not structure pruning, can't calc power from here")

    #     for name, mask in self.masks.items():

    def init_death_rate(self, death_rate, pruning_type="unstructure"):
        if pruning_type == "unstructure":
            for name in self.masks:
                self.name2death_rate[name] = death_rate
        elif pruning_type == "structure":
            for name in self.masks:
                self.name2death_rate[name] = death_rate
        else:
            raise ValueError("Unrecognized Pruning Type !")

    # init masks for unstructure pruning
    def unstructure_init(
        self, mode="ER", density=0.05, erk_power_scale=1.0, mask_file=None
    ):
        self.sparsity = density
        if mode == "uniform":
            for mask in self.masks.values():
                mask["elem_mask"].bernoulli_(p=density)
        elif mode == "custom":
            custom_mask = torch.load(mask_file, map_location=self.device)
            for name, mask in self.masks.items():
                mask["elem_mask"] = custom_mask[name.removeprefix("module.") + "_mask"]
        elif mode == "fixed_ERK":
            total_params = sum([m.numel() for m in self.masks.values()])
            is_epsilon_valid = False
            # # The following loop will terminate worst case when all masks are in the
            # custom_sparsity_map. This should probably never happen though, since once
            # we have a single variable or more with the same constant, we have a valid
            # epsilon. Note that for each iteration we add at least one variable to the
            # custom_sparsity_map and therefore this while loop should terminate.
            dense_layers = set()
            while not is_epsilon_valid:
                # We will start with all layers and try to find right epsilon. However if
                # any probablity exceeds 1, we will make that layer dense and repeat the
                # process (finding epsilon) with the non-dense layers.
                # We want the total number of connections to be the same. Let say we have
                # for layers with N_1, ..., N_4 parameters each. Let say after some
                # iterations probability of some dense layers (3, 4) exceeded 1 and
                # therefore we added them to the dense_layers set. Those layers will not
                # scale with erdos_renyi, however we need to count them so that target
                # paratemeter count is achieved. See below.
                # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
                #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
                # eps * (p_1 * N_1 + p_2 * N_2) =
                #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
                # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name, mask in self.masks.items():
                    n_param = mask.numel()
                    n_zeros = n_param * (1 - density)  # 0.95
                    n_ones = n_param * density  # 0.05

                    if name in dense_layers:
                        # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                        rhs -= n_zeros

                    else:
                        # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                        # equation above.
                        rhs += n_ones
                        # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                        if len(mask.shape) == 6:
                            mask_shape = [
                                mask.shape[0] * mask.shape[2],
                                mask.shape[1] * mask.shape[3],
                                mask.shape[4],
                                mask.shape[5],
                            ]
                        else:
                            mask_shape = mask.shape
                        raw_probabilities[name] = (
                            np.sum(mask_shape) / n_param
                        ) ** erk_power_scale
                        # Note that raw_probabilities[mask] * n_param gives the individual
                        # elements of the divisor.
                        divisor += raw_probabilities[name] * n_param
                # By multipliying individual probabilites with epsilon, we should get the
                # number of parameters per layer correctly.
                epsilon = rhs / divisor

                # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
                # mask to 0., so they become part of dense_layers sets.
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            print(
                                f"Sparsity of var:{mask_name} had to be set to 0, i.e., dense layer."
                            )
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            density_dict = {}
            total_nonzero = 0.0
            # With the valid epsilon, we can set sparsities of the remaining layers.
            for name, mask in self.masks.items():
                n_param = mask.numel()
                if name in dense_layers:
                    density_dict[name] = 1.0
                else:
                    probability_one = epsilon * raw_probabilities[name]
                    density_dict[name] = probability_one
                logger.info(
                    f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
                )
                mask["elem_mask"].bernoulli_(p=density_dict[name])

                total_nonzero += density_dict[name] * n_param
            logger.info(f"Overall Density {total_nonzero / total_params}")

        elif mode == "ER":
            # initialization used in sparse evolutionary training
            ## here we use tensorized 4D tensor [p,q,k,k] and its epsilon-rank CP decomposition as parameter estimation
            epsilon = (
                sum([m.numel() for m in self.masks.values()])
                * density
                / sum(sum(m.shape) for m in self.masks.values())
            )
            for name, mask in self.masks.items():
                growth = epsilon * sum(mask.shape)
                prob = growth / mask.numel()
                mask["elem_mask"].bernoulli_(p=prob)
        else:
            raise ValueError("Unrecognized Init Mode !")

        self.apply_mask()
        self.fired_masks = {
            name: m.data.clone() for name, m in self.masks.items()
        }  # used for over-paremeters
        self.init_death_rate(self.death_rate)

        total_size = sum([m.numel() for m in self.masks.values()])
        logger.info(f"Total Model parameters: {total_size}")

        total_nonzeros = sum([m.sum().item() for m in self.masks.values()])

        logger.info(
            "Total parameters under density level of {0}: {1}".format(
                density, total_nonzeros / total_size
            )
        )

        self.gather_statistics()
        logger.info(
            "Scale up initialized weights by (weight_count/nonzeros) to maintain the same variance"
        )
        for name in self.masks:
            self.params[name].data.mul_(
                self.params[name].numel() / self.name2nonzeros[name]
            )

        params = {name: m.numel() for name, m in self.masks.items()}
        logger.info(f"Zero counts:\n\t{self.name2zeros}")
        logger.info(f"Nonzero counts:\n\t{self.name2nonzeros}")
        logger.info(f"Param counts:\n\t{params}")

    def _structure_init_random(self, density: float = 0.05) -> None:
        if self.pruning_type == "structure_row":
            for mask in self.masks.values():
                mask["row_mask"].bernoulli_(p=density)

        elif self.pruning_type == "structure_col":
            for mask in self.masks.values():
                mask["col_mask"].bernoulli_(p=density)
        elif self.pruning_type == "structure_row_col":
            for mask in self.masks.values():
                mask["row_mask"].bernoulli_(p=density**0.5)
                mask["col_mask"].bernoulli_(p=density**0.5)
        else:
            raise ValueError(f"Unrecognized Pruning Type {self.pruning_type}")

    def find_least_switch_power_patterns(
        self, patterns: Tensor
    ) -> Tuple[Tensor, float]:
        powers = self.cal_ports_power(patterns)  # [#combinations]
        if DEBUG:
            print(f"calc power", powers.shape, powers)
        min_power_patterns, min_power = self.find_minimal_power_pattern(
            patterns, powers
        )
        return min_power_patterns, min_power

    def _structure_init_power_crosstalk(
        self, density: float = 0.05, opts: List = ["power", "crosstalk"]
    ) -> None:
        if self.pruning_type == "structure_row":
            for name, mask in self.masks.items():
                row_num = k1 = self.params[name].shape[-2]
                empty_row_num = int(round(row_num * (1 - density)))
                patterns = self.find_sparsity_patterns(
                    row_num, empty_row_num
                )  # [#combinations, col_num]
                for opt in opts:
                    if opt == "crosstalk":
                        patterns, _ = self.find_least_crosstalk_patterns(
                            patterns, is_col=False
                        )
                mask["row_mask"][..., :, 0] = patterns[0]
        elif self.pruning_type == "structure_col":
            for name, mask in self.masks.items():
                ## assume all blocks have the same initial sparsity mask with min power
                ## just select k2' from k2 with min power
                col_num = k2 = self.params[name].shape[-1]
                empty_col_num = int(round(col_num * (1 - density)))
                patterns = self.find_sparsity_patterns(
                    col_num, empty_col_num
                )  # [#combinations, col_num]
                if DEBUG:
                    print(f"select {empty_col_num} from {col_num}")
                    print("patterns", patterns.shape, patterns)
                for opt in opts:
                    if opt == "power":
                        patterns, _ = self.find_least_switch_power_patterns(patterns)
                        if DEBUG:
                            print(f"after power opt", patterns.shape, patterns)
                    elif opt == "crosstalk":
                        patterns, _ = self.find_least_crosstalk_patterns(
                            patterns, is_col=True
                        )
                    else:
                        raise NotImplementedError
                mask["col_mask"][..., :] = patterns[0]
        elif self.pruning_type == "structure_row_col":
            for name, mask in self.masks.items():
                row_num, col_num = k1, k2 = self.params[name].shape[-2:]
                max_empty_col_num = int(round(col_num * (1 - density)))

                best_score = float("-inf")  # higher the better score
                best_row_patterns = best_col_patterns = None

                # find the integer solution (col, row) of `(k2-col) * (k1-row) = k1 * k2 * density`
                for empty_col_num in range(max_empty_col_num + 1):
                    empty_row_num = int(
                        round(k1 - (k1 * k2 * density) / (k2 - empty_col_num))
                    )

                    col_patterns = self.find_sparsity_patterns(col_num, empty_col_num)
                    row_patterns = self.find_sparsity_patterns(row_num, empty_row_num)
                    if DEBUG:
                        print("col_pattern:", col_patterns)
                        print("row_pattern:", row_patterns)
                    score = {}
                    for opt in opts:
                        if opt == "power":
                            col_patterns, switch_power = (
                                self.find_least_switch_power_patterns(col_patterns)
                            )
                            TIA_ADC_power = self.calc_TIA_ADC_power(
                                row_num, empty_row_num, self.TIA_power, self.ADC_power
                            )
                            HDAC_power = self.calc_HDAC_power(
                                col_num, empty_col_num, self.HDAC_power
                            )
                            score["power"] = -(
                                switch_power + TIA_ADC_power + HDAC_power
                            )
                        elif opt == "crosstalk":
                            col_patterns, col_crosstalk = (
                                self.find_least_crosstalk_patterns(
                                    col_patterns, is_col=True
                                )
                            )
                            row_patterns, row_crosstalk = (
                                self.find_least_crosstalk_patterns(
                                    row_patterns, is_col=False
                                )
                            )

                            score["crosstalk"] = col_crosstalk + row_crosstalk
                        else:
                            raise NotImplementedError

                    # prioritize the first opt
                    if score[opts[0]] > best_score:
                        best_score = score[opts[0]]
                        best_row_patterns = row_patterns
                        best_col_patterns = col_patterns

                self.masks[name]["col_mask"][..., :] = best_col_patterns[0]
                self.masks[name]["row_mask"][..., :, 0] = best_row_patterns[0]
        else:
            raise ValueError(f"Unrecognized Pruning Type {self.pruning_type}")

    def structure_init(
        self, mode="fixedERK", density=0.05, erk_power_scale=1.0, mask_file=None
    ) -> None:
        assert mode in {
            "fixedERK",
            "uniform",
            "uniform_power",
            "uniform_power_crosstalk",
            "uniform_crosstalk",
            "uniform_crosstalk_power",
        }, f"Unrecognized Init Mode {mode}"
        opts = mode.split("_")[1:]
        if self.pruning_type == "structure_row":
            opts = [m for m in opts if m != "power"]
            if len(opts) == 0:
                opts = ["uniform"]
            logger.info(
                f"{self.pruning_type} not support power, init mode reduced to {opts}"
            )

        if mode == "fixedERK":
            raise NotImplementedError
        elif mode == "uniform":
            self._structure_init_random(density)
        elif mode in {
            "uniform_power",
            "uniform_power_crosstalk",
            "uniform_crosstalk",
            "uniform_crosstalk_power",
        }:
            self._structure_init_power_crosstalk(density, opts)
        else:
            raise ValueError(f"Unrecognized Init Mode {mode}")

        self.apply_mask()
        self.fired_masks = {
            name: m.data.clone() for name, m in self.masks.items()
        }  # used for over-paremeters
        self.init_death_rate(self.death_rate)

        total_size = sum([m.numel() for m in self.masks.values()])
        logger.info(f"Total Model parameters: {total_size}")

        total_nonzeros = sum([m.sum().item() for m in self.masks.values()])

        logger.info(
            "Total parameters under density level of {0}: {1}".format(
                density, total_nonzeros / total_size
            )
        )

        self.gather_statistics()
        logger.info(
            "Scale up initialized weights by (weight_count/nonzeros) to maintain the same variance"
        )
        for name in self.masks:
            self.params[name].data.mul_(
                self.params[name].numel() / self.name2nonzeros[name]
            )

        params = {name: m.numel() for name, m in self.masks.items()}
        logger.info(f"Zero counts:\n\t{self.name2zeros}")
        logger.info(f"Nonzero counts:\n\t{self.name2nonzeros}")
        logger.info(f"Param counts:\n\t{params}")

    # multiple mask for paramenters and momentum in optimizers
    def apply_mask(self, pruning_type: str | None = None) -> None:
        pruning_type = pruning_type or self.pruning_type
        if pruning_type in {
            "unstructure",
            "structure_row",
            "structure_col",
            "structure_row_col",
        }:
            for name in self.masks:
                if not self.first_layer_pruning and "conv1" in name:
                    continue
                mask = self.masks[name]
                weight = self.params[name]
                weight.data *= mask
                state = self.optimizer.state[weight]
                if "momentum_buffer" in state:
                    state["momentum_buffer"] *= mask
        else:
            raise ValueError(f"Unrecognized Pruning Type {pruning_type}")

    def gather_statistics(self, pruning_type: str | None = None) -> None:
        pruning_type = pruning_type or self.pruning_type
        if pruning_type in {
            "unstructure",
            "structure_row",
            "structure_col",
            "structure_row_col",
        }:
            self.name2nonzeros = {
                name: mask.num_nonzeros() for name, mask in self.masks.items()
            }
            self.name2zeros = {
                name: mask.numel() - self.name2nonzeros[name]
                for name, mask in self.masks.items()
            }
        else:
            raise ValueError("Unrecognized Pruning Type !")

    def update_death_mask(self, pruning_type: str | None = None) -> None:
        # update pruning and growth masks
        pruning_type = pruning_type or self.pruning_type

        self.gather_statistics()  # count each of module's zeros and non-zeros
        # update pruning mask
        for name, mask in self.masks.items():
            weight = self.params[name]
            if self.death_mode == "magnitude" and pruning_type == "unstructure":
                new_mask = self.magnitude_death(mask, weight, name)
            elif self.death_mode.startswith("magnitude"):
                if pruning_type == "structure_row":
                    new_mask = self.row_only_magnitude_death(mask, weight, name)
                elif pruning_type == "structure_col":
                    new_mask = self.col_only_magnitude_death(mask, weight, name)
                elif pruning_type == "structure_row_col":
                    new_mask = self.row_col_magnitude_death(mask, weight, name)
                else:
                    raise ValueError(f"Unrecognized Pruning Type {pruning_type}")
            elif self.death_mode == "SET":
                new_mask = self.magnitude_and_negativity_death(mask, weight, name)
            elif self.death_mode == "threshold":
                new_mask = self.threshold_death(mask, weight, name)

            self.pruned_number[name] = int(new_mask.num_zeros() - self.name2zeros[name])
            assert (
                self.pruned_number[name] >= 0
            ), f"{new_mask.num_nonzeros()} must >= {self.name2nonzeros[name]}"
            self.masks[name] = new_mask  # update new mask

    def update_growth_mask(self, pruning_type: str | None = None) -> None:
        # update pruning mask with growing
        for name, mask in self.masks.items():
            weight = self.params[name]
            if self.growth_mode == "random":
                new_mask = self.random_growth(
                    name, mask, self.pruned_number[name], weight
                )
            elif self.growth_mode == "gradient" and self.pruning_type == "unstructure":
                new_mask = self.gradient_growth(
                    name, mask, self.pruned_number[name], weight
                )
            elif self.growth_mode.startswith("gradient"):
                if self.pruning_type == "structure_row":
                    new_mask = self.row_only_gradient_growth(
                        name, mask, self.pruned_number[name], weight
                    )
                elif self.pruning_type == "structure_col":
                    new_mask = self.col_only_gradient_growth(
                        name, mask, self.pruned_number[name], weight
                    )
                elif self.pruning_type == "structure_row_col":
                    new_mask = self.row_col_gradient_growth(
                        name, mask, self.pruned_number[name], weight
                    )
                else:
                    raise ValueError(f"Unrecognized Pruning Type {pruning_type}")
            self.masks[name] = new_mask

    def update_and_apply_mask(
        self, pruning_type: str | None = None, indicator_list=None
    ) -> None:
        # update pruning and growth masks
        pruning_type = pruning_type or self.pruning_type

        self.update_death_mask(pruning_type)
        self.update_growth_mask(pruning_type)
        self.apply_mask()

    # remove part mask
    def remove_weight_partial_name(self, partial_name):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:
                print(
                    "Removing {0} of size {1} with {2} parameters...".format(
                        name, self.masks[name].shape, np.prod(self.masks[name].shape)
                    )
                )
                removed.add(name)
                self.masks.pop(name)
        print("Removed {0} layers.".format(len(removed)))

        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed:
                self.names.pop(i)
            else:
                i += 1

    def remove_type(self, nn_type):
        index = 0
        for module in self.modules:
            for name, module in module.named_modules():
                print(name)
                if isinstance(module, nn_type):
                    self.remove_weight(name, index)
                index += 1

    def remove_weight(self, name, index):
        if name in self.masks:
            print(
                "Removing {0} of size {1} = {2} parameters.".format(
                    name, self.masks[name].shape, self.masks[name].numel()
                )
            )

    """
                DEATH
    """

    def CS_death(self, mask, snip_mask):
        # calc scores for all weights
        # note that the gradients are from the last iteration, which are not very accurate
        # but in another perspective, we can understand the weights are from the next iterations, the differences are not very large.
        """
        grad = self.get_gradient_for_weights(weight)
        scores = torch.abs(grad * weight * (mask == 0).float())
        norm_factor = torch.sum(scores)
        scores.div_(norm_factor)
        x, idx = torch.sort(scores.data.view(-1))

        num_remove = math.ceil(self.name2death_rate[name]*self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)
        if num_remove == 0.0: return weight.data != 0.0

        mask.data.view(-1)[idx[:k]] = 0.0
        """

        assert snip_mask.shape == mask.shape

        return snip_mask

    def threshold_death(self, mask, weight, name):
        return torch.abs(weight.data) > self.threshold

    def col_only_magnitude_death(
        self, mask: MultiMask, weight: Tensor, name: str
    ) -> MultiMask:
        # mask here is col mask [p, q, r, 1, k1, 1] and row mask [p, q, 1, c, 1, k2]
        if mask["col_mask"].sum().item() == 0:
            return mask

        death_rate = self.name2death_rate[name]
        num_remove = math.ceil(death_rate * self.name2nonzeros[name])

        if num_remove == 0.0:
            return mask

        mask = self.col_only_magnitude_select(
            mask=mask,
            weight=weight,
            name=name,
            death=True,
            num_select=num_remove,
        )

        return mask

    def row_only_magnitude_death(
        self, mask: MultiMask, weight: Tensor, name: str
    ) -> MultiMask:
        # mask here is row mask [p, q, r, 1, k1, 1] and row mask [p, q, 1, c, 1, k2]
        if mask.num_nonzeros() == mask.numel():
            return mask

        death_rate = self.name2death_rate[name]
        num_remove = math.ceil(death_rate * self.name2nonzeros[name])

        if num_remove == 0.0:
            return mask

        mask = self.row_only_magnitude_select(
            mask=mask,
            weight=weight,
            name=name,
            death=True,
            num_select=num_remove,
        )

        return mask

    def row_col_magnitude_death(
        self, mask: MultiMask, weight: Tensor, name: str
    ) -> MultiMask:
        # mask here is row mask [p, q, r, 1, k1, 1] and row mask [p, q, 1, c, 1, k2]
        if mask.num_nonzeros() == mask.numel():
            return mask

        death_rate = self.name2death_rate[name]
        num_remove = math.ceil(death_rate * self.name2nonzeros[name])

        mask = self.row_col_magnitude_select(
            mask=mask,
            weight=weight,
            name=name,
            death=True,
            num_select=num_remove,
        )

        return mask

    def magnitude_death(self, mask, weight, name):

        if mask.sum().item() == mask.numel():
            return mask

        death_rate = self.name2death_rate[name]

        num_remove = math.ceil(
            death_rate * self.name2nonzeros[name]
        )  # pruning nonzeros
        if num_remove == 0.0:
            return mask
        # num_remove = math.ceil(self.name2death_rate[name]*self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]

        x, idx = torch.sort(torch.abs(weight.data.view(-1)))
        n = idx.shape[0]

        k = math.ceil(num_zeros + num_remove)
        threshold = x[k - 1].item()

        return torch.abs(weight.data) > threshold

    def magnitude_and_negativity_death(self, mask, weight, name):
        num_remove = math.ceil(self.name2death_rate[name] * self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]

        # find magnitude threshold
        # remove all weights which absolute value is smaller than threshold
        x, idx = torch.sort(weight[weight > 0.0].data.view(-1))
        k = math.ceil(num_remove / 2.0)
        if k >= x.shape[0]:
            k = x.shape[0]

        threshold_magnitude = x[k - 1].item()

        # find negativity threshold
        # remove all weights which are smaller than threshold
        x, idx = torch.sort(weight[weight < 0.0].view(-1))
        k = math.ceil(num_remove / 2.0)
        if k >= x.shape[0]:
            k = x.shape[0]
        threshold_negativity = x[k - 1].item()

        pos_mask = (weight.data > threshold_magnitude) & (weight.data > 0.0)
        neg_mask = (weight.data < threshold_negativity) & (weight.data < 0.0)

        new_mask = pos_mask | neg_mask
        return new_mask

    """
                GROWTH
    """

    def random_growth(self, name, new_mask, total_regrowth, weight):
        n = new_mask.numel() - new_mask.sum().item()
        if n == 0:
            return new_mask
        expected_growth_probability = total_regrowth / n
        if self.pruning_type == "unstructure":
            new_weights = new_mask.clone()
            new_weights["elem_mask"].bernoulli_(p=expected_growth_probability)
        elif self.pruning_type in {
            "structure",
            "structure_row",
            "structure_col",
            "structure_row_col",
        }:
            new_weights = new_mask.clone()
            new_weights["row_mask"].bernoulli_(p=expected_growth_probability)
            new_weights["col_mask"].bernoulli_(p=expected_growth_probability)
        else:
            raise ValueError("Unrecognized Pruning Type !")

        return new_mask | new_weights

    def gradient_growth(self, name, new_mask, total_regrowth, weight):
        if total_regrowth == 0:
            return new_mask
        grad = self.get_gradient_for_weights(weight)
        grad = grad * (~new_mask)
        if self.pruning_type == "unstructure":
            y, idx = torch.sort(grad.abs().flatten(), descending=True)
            new_mask.data.view(-1)[idx[:total_regrowth]] = 1
        elif self.pruning_type == "structure":
            raise NotImplementedError
        else:
            raise ValueError(f"Unrecognized Pruning Type {self.pruning_type}")
        return new_mask

    def col_only_gradient_growth(
        self, name: str, new_mask: MultiMask, total_regrowth: int, weight: Tensor
    ) -> MultiMask:
        if total_regrowth == 0:
            return new_mask

        grad = self.get_gradient_for_weights(weight)
        grad = grad * (~new_mask)

        new_mask = self.col_only_magnitude_select(
            mask=new_mask,
            weight=grad,
            name=name,
            death=False,
            num_select=total_regrowth,
        )

        return new_mask

    def row_only_gradient_growth(
        self, name: str, new_mask: MultiMask, total_regrowth: int, weight: Tensor
    ) -> MultiMask:
        if total_regrowth == 0:
            return new_mask
        grad = self.get_gradient_for_weights(weight)
        grad = grad * (~new_mask)

        new_mask = self.row_only_magnitude_select(
            mask=new_mask,
            weight=grad,
            name=name,
            death=False,
            num_select=total_regrowth,
        )

        return new_mask

    def row_col_gradient_growth(
        self, name: str, new_mask: MultiMask, total_regrowth: int, weight: Tensor
    ) -> MultiMask:
        if total_regrowth == 0:
            return new_mask
        grad = self.get_gradient_for_weights(weight)
        grad = grad * (~new_mask)

        new_mask = self.row_col_magnitude_select(
            mask=new_mask,
            weight=grad,
            name=name,
            death=False,
            num_select=total_regrowth,
        )

        return new_mask

    def row_only_magnitude_select(
        self,
        mask: MultiMask,
        weight: Tensor,
        name: str,
        death: bool = True,
        row_col: bool = False,
        row_elements_average: float = 1.0,
        num_select: int = 0,
    ) -> MultiMask:
        # mask here is row mask [p, q, r, 1, k1, 1] and row mask [p, q, 1, c, 1, k2]
        # weight here is [p, q, r, c, k1, k2]
        p, q, r, c, k1, k2 = weight.shape
        # num of rows to prune out of total p*q*r*k1 rows
        if row_col:
            num_row_select = int(round(num_select / row_elements_average))
        else:
            num_row_select = int(round(num_select / (c * k2)))

        if num_row_select == 0:
            return mask
        opts = self.death_opts if death else self.growth_opts

        if self.group == "layer":  # sort row magnitude per layer
            # [p, q, r, k1]
            # if sorting globally in this layer, there might have many combinations, not tractable if with large p,q
            row_magnitude = weight.data.norm(
                p=2, dim=(3, 5), keepdim=True
            )  # [p, q, r, 1, k1, 1]
            ## to make sure pruned magnitude is always smaller than unpruned vectors (which can also have 0 magnitude)
            ## we set pruned magnitude to -1
            if death:
                ## to make sure the pruned weight always have larger magnitude than unpruned one
                ## we set pruned weight magnitude to 1e8
                row_magnitude[~mask["row_mask"]] = 1e8
            else:
                ## to make sure the pruned grad always have larger magnitude than unpruned one (which can also have 0 magnitude)
                ## we set unpruned grad magnitude to -1
                row_magnitude[mask["row_mask"]] = -1
            row_magnitude = row_magnitude.flatten()  # [p*q*r*k1]
            # index in dim p, index in dim q, index in dim r, index in dim k1
            if death:
                num_to_select_rows = mask["row_mask"].sum().item()
            else:  # growth
                num_to_select_rows = (
                    mask["row_mask"].numel() - mask["row_mask"].sum().item()
                )

            margin = 0 if len(opts) == 0 else self.power_choice_margin
            num_row_select_candidates = min(num_row_select + margin, num_to_select_rows)

            ## select a slightly larger candidates pool
            selected_row_indices_flat = torch.argsort(
                row_magnitude, descending=False if death else True
            )[:num_row_select_candidates]

            ## convert from flattened indices to high-dimensional indices to match row_mask
            selected_row_indices = torch.unravel_index(
                selected_row_indices_flat, mask["row_mask"].shape
            )  # tuple of indices in each dimension of row_mask

            if DEBUG:
                print(f"row_mag", row_magnitude)
                print(f"num_row_select", num_row_select)
                print(f"num_row_select_candidates", num_row_select_candidates)
                print(f"selected_row_indices", selected_row_indices_flat)
                print(f"selected_row_indices unraveled", selected_row_indices)
                print(mask["row_mask"].shape)

            # only magnitude sorting, no power or crosstalk optimization
            if len(opts) == 0 or num_row_select_candidates == num_row_select:
                mask["row_mask"][selected_row_indices] = 0 if death else 1
                return mask

            ## till this point, we know self.death_opts = ["crosstalk"]
            best_crosstalk_gain = float("-inf")
            best_row_mask = None
            for i, indices in enumerate(
                combinations(range(num_row_select_candidates), num_row_select)
            ):
                if i >= self.max_combinations:
                    break

                indices = torch.tensor(indices)
                selected_row_indices_cand = tuple(
                    row_index[indices] for row_index in selected_row_indices
                )

                if DEBUG:
                    print(f"-----------------\niter {i}")
                    print(f"selected indices", indices)
                    print(f"selected row indices", selected_row_indices_cand)
                ## check crosstalk score for this combination
                ## crosstalk can be calculated based on its index along k1 dimension

                ## first try to prune rows in a cloned row_mask
                row_mask = mask["row_mask"].clone()
                row_mask[selected_row_indices_cand] = 0 if death else 1

                total_crosstalk_gain = 0
                ## after pruning, calculate crosstalk gain on affected row_masks
                affected_mask_indices = set()  # use set to avoid duplicate indices
                for row_id in range(num_row_select):
                    affected_mask_indices.add(
                        (
                            selected_row_indices_cand[0][row_id].item(),  # p
                            selected_row_indices_cand[1][row_id].item(),  # q
                            selected_row_indices_cand[2][row_id].item(),  # r
                        )
                    )

                for mask_indices in affected_mask_indices:
                    gain = self.calc_crosstalk_score(
                        row_mask[
                            mask_indices[0],
                            mask_indices[1],
                            mask_indices[2],
                            0,
                            :,  # k1
                            0,
                        ],
                        is_col=False,
                    ) - self.calc_crosstalk_score(
                        mask["row_mask"][
                            mask_indices[0],
                            mask_indices[1],
                            mask_indices[2],
                            0,
                            :,  # k1
                            0,
                        ],
                        is_col=False,
                    )
                    total_crosstalk_gain += gain
                    if DEBUG:
                        print(
                            "crosstalk gain:",
                            gain,
                            "from",
                            mask["row_mask"][
                                mask_indices[0],
                                mask_indices[1],
                                mask_indices[2],
                                0,
                                :,  # k1
                                0,
                            ].long(),
                            "to",
                            row_mask[
                                mask_indices[0],
                                mask_indices[1],
                                mask_indices[2],
                                0,
                                :,  # k1
                                0,
                            ].long(),
                        )

                if total_crosstalk_gain > best_crosstalk_gain:
                    best_crosstalk_gain = total_crosstalk_gain
                    best_row_mask = row_mask
                if DEBUG:
                    print(f"affected_mask_indices", affected_mask_indices)
                    print(f"total_crosstalk_gain", total_crosstalk_gain)
                    print(f"best_crosstalk_gain", best_crosstalk_gain)

            mask["row_mask"] = best_row_mask
            return mask
        elif self.group == "block":
            ## we can maintain uniform sparsity in each [rk1, ck2] block, then the row combinations are limited to rk1.
            ## through this will limit accuracy, but it will faster.
            raise NotImplementedError
        else:
            raise NotImplementedError

    def col_only_magnitude_select(
        self,
        mask: MultiMask,
        weight: Tensor,
        name: str,
        death: bool = True,
        row_col: bool = False,
        col_elements_average: float = 1.0,
        num_select: int = 0,
    ) -> MultiMask:
        # mask here is col mask [p, q, r, 1, k1, 1] and row mask [p, q, 1, c, 1, k2]
        # weight here is [p, q, r, c, k1, k2]
        p, q, r, c, k1, k2 = weight.shape

        if row_col:
            num_col_select = int(
                round(num_select / col_elements_average)
            )  # num of col p*q*c*k2
        else:
            num_col_select = int(round(num_select / (r * k1)))  # num of col p*q*c*k2

        if num_col_select == 0:
            return mask

        opts = self.death_opts if death else self.growth_opts

        if self.group == "layer":  # sort col magnitude per layer
            # [p, q, r, k1]
            # if sorting globally in this layer, there might have many combinations, not tractable if with large p,q
            col_magnitude = weight.data.norm(
                p=2, dim=(2, 4), keepdim=True
            )  # [p, q, 1, c, 1, k2]

            if death:
                ## to make sure the pruned weight always have larger magnitude than unpruned one
                ## we set pruned weight magnitude to 1e8
                col_magnitude[~mask["col_mask"]] = 1e8
            else:
                ## to make sure the pruned grad always have larger magnitude than unpruned one (which can also have 0 magnitude)
                ## we set unpruned grad magnitude to -1
                col_magnitude[mask["col_mask"]] = -1
            col_magnitude = col_magnitude.flatten()  # [p*q*c*k2]
            # index in dim p, index in dim q, index in dim r, index in dim k1
            if death:
                num_to_select_cols = mask["col_mask"].sum().item()
            else:  # growth
                num_to_select_cols = (
                    mask["col_mask"].numel() - mask["col_mask"].sum().item()
                )
            margin = 0 if len(opts) == 0 else self.power_choice_margin
            num_col_select_candidates = min(num_col_select + margin, num_to_select_cols)

            ## select a slightly larger candidates pool
            selected_col_indices = torch.argsort(
                col_magnitude, descending=False if death else True
            )[:num_col_select_candidates]
            ## convert from flattened indices to high-dimensional indices to match col_mask
            selected_col_indices = torch.unravel_index(
                selected_col_indices, mask["col_mask"].shape
            )  # tuple of indices in each dimension of row_mask

            # only magnitude sorting, no power or crosstalk optimization
            if len(opts) == 0 or num_col_select_candidates == num_col_select:
                mask["col_mask"][selected_col_indices] = 0 if death else 1
                return mask

            ## till this point, we know self.death_opts might contain power or crosstalk optimization

            ## we perform coordinate ascent to find the best combination in each optimization metrics
            best_gain = float("-inf")
            search_range = list(
                combinations(range(num_col_select_candidates), num_col_select)
            )

            def obj_fn(
                opt: str, col_mask: Tensor, affected_mask_indices: Tuple
            ) -> float:
                # affected_mask_indices: [(p, q, c), (p,q,c), ..., (p,q,c)]
                if opt == "crosstalk":
                    gain = sum(
                        self.calc_crosstalk_score(
                            col_mask[p, q, 0, c, 0, :], is_col=True
                        )
                        - self.calc_crosstalk_score(
                            mask["col_mask"][p, q, 0, c, 0, :], is_col=True
                        )
                        for (p, q, c) in affected_mask_indices
                    )
                elif opt == "power":
                    if DEBUG:
                        print(affected_mask_indices)
                    ps, qs, cs = zip(*affected_mask_indices)
                    gain = (
                        self.cal_ports_power(
                            mask["col_mask"][ps, qs, 0, cs, 0, :]
                        ).sum()
                        - self.cal_ports_power(col_mask[ps, qs, 0, cs, 0, :]).sum()
                    ).item()
                else:
                    gain = 0
                return gain  # higher the better

            for opt in opts:  # search for each optimization metric
                if len(search_range) == 1:
                    break  # only solution left, no need to search
                best_gain = float("-inf")
                selected_range = []
                best_col_masks = []  # can have multiple best solutions

                # search in the current search range
                for i, indices in enumerate(search_range):
                    if i >= self.max_combinations:
                        break
                    indices = torch.tensor(indices)
                    selected_col_indices_cand = tuple(
                        col_index[indices] for col_index in selected_col_indices
                    )
                    ## check score for this combination
                    ## crosstalk can be calculated based on its index along k2 dimension

                    ## first try to prune cols in a cloned col_mask
                    col_mask = mask["col_mask"].clone()
                    col_mask[selected_col_indices_cand] = 0 if death else 1

                    ## after pruning, calculate gain on affected row_masks
                    affected_mask_indices = set()  # use set to avoid duplicate indices
                    for col_id in range(num_col_select):
                        affected_mask_indices.add(
                            (
                                selected_col_indices_cand[0][col_id].item(),  # p
                                selected_col_indices_cand[1][col_id].item(),  # q
                                selected_col_indices_cand[3][col_id].item(),  # c
                            )
                        )
                    affected_mask_indices = tuple(affected_mask_indices)
                    gain = obj_fn(
                        opt=opt,
                        col_mask=col_mask,
                        affected_mask_indices=affected_mask_indices,
                    )

                    if gain > best_gain:
                        best_gain = gain
                        best_col_masks = [col_mask]
                        selected_range = [indices]
                    elif gain == best_gain:
                        best_col_masks.append(col_mask)
                        selected_range.append(indices)
                    if DEBUG:
                        print(f"-----------------\niter {i} {opt}")
                        print(f"selected indices", indices)
                        print(f"selected col indices", selected_col_indices_cand)
                        print(f"gain", gain)
                        print(f"best gain", best_gain)
                        print(f"best col masks", len(best_col_masks))

                # shrink the search range to the selected range
                search_range = selected_range
                if DEBUG:
                    print(f"best gain {opt}", best_gain, len(best_col_masks))
                    print(f"search_range", search_range)
            if DEBUG:
                print(
                    f"best col masks final",
                    len(best_col_masks),
                    type(best_col_masks[0]),
                )
            mask["col_mask"] = best_col_masks[0]
            return mask
        elif self.group == "block":
            ## we can maintain uniform sparsity in each [rk1, ck2] block, then the row combinations are limited to rk1.
            ## through this will limit accuracy, but it will faster.
            raise NotImplementedError
        else:
            raise NotImplementedError

    def calculate_pruned_elements(
        self,
        row_elements: int,
        col_elements: int,
        rows_pruned: int,
        cols_pruned: int,
    ):
        # Calculate the number of elements pruned when rows and columns are pruned
        return row_elements * rows_pruned + (col_elements - rows_pruned) * cols_pruned

    def helper_for_find_row_col_pruning_combination(
        self,
        depth: int,  # Current slide of mask
        current_pruned: int,  # Number of elements been pruned away at current slide and slides befroe
        upper_limit_pruning_number: int,
        lower_limit_pruning_number: int,
        max_depth: int,  # The maxiumn depth of slides
        row_elements: Tensor,  # Number of elements would be turned off by one row; [depth] shape is expected
        col_elements: Tensor,  # Number of elements would be turned off by one col; [depth] shape is expected
        active_rows: Tensor,  # Number of rows are on in the current mask; [depth] shape is expectd
        active_cols: Tensor,  # Number of cols are on in the current mask; [depth] shape is expectd
        path: list,  # Possible solution, but will not be final solution unless is been appended into result
        results: list,  # Final possible results
        pruning_order: str = "col_row",  # Either "col_row" pruning order or "row_col" pruning order
    ):

        if lower_limit_pruning_number <= current_pruned <= upper_limit_pruning_number:
            results.append(path.copy())  # Append a copy of the current path to results
            return
        if current_pruned > upper_limit_pruning_number or depth == max_depth:
            return  # Stop if we have pruned too many elements or reached max depth

        # Explore the number of rows and columns that can be pruned at the current depth
        for r in range(active_rows[depth]):
            for c in range(active_cols[depth]):
                # Get the number of elements pruned away by this current choice
                pruned_by_this_choice = self.calculate_pruned_elements(
                    row_elements[depth].item(),
                    col_elements[depth].item(),
                    r,
                    c,
                    pruning_order,
                )
                print("Current choice pruned away elements:", pruned_by_this_choice)
                print(
                    "Current total pruned away elements:",
                    current_pruned + pruned_by_this_choice,
                )
                if current_pruned + pruned_by_this_choice < lower_limit_pruning_number:
                    path.append(
                        [r, c]
                    )  # Choose to prune r rows and c columns at this depth
                    self.helper_for_find_row_col_pruning_combination(
                        depth + 1,
                        current_pruned + pruned_by_this_choice,
                        upper_limit_pruning_number,
                        lower_limit_pruning_number,
                        max_depth,
                        row_elements,
                        col_elements,
                        active_rows,
                        active_cols,
                        path,
                        results,
                        pruning_order,
                    )
                    path.pop()  # Backtrack

    def find_all_pruning_combinations(
        self,
        max_depth: int,  # Should be p*q total loading orders
        row_elements: Tensor,  # Number of elements would be turned off by one row; [depth] shape is expected
        col_elements: Tensor,  # Number of elements would be turned off by one col; [depth] shape is expected
        active_rows: Tensor,  # Number of rows are on in the current mask; [depth] shape is expectd
        active_cols: Tensor,  # Number of cols are on in the current mask; [depth] shape is expectd
        target_pruned: int,  # The total number would be pruned off by the target, expect the number from num_remove
        pruning_order: str = "col_row",  # Either "col_row" pruning order or "row_col" pruning order
        range_percentage: float = 0.1,  # Set a range between for the target_pruned, (target_pruned +- range_percentage * target_pruned)
    ):

        results = []
        stack = [(0, 0, [])]  # (current depth, current pruned, path)

        upper_limit_pruning_number = int(
            round(target_pruned + target_pruned * range_percentage)
        )
        lower_limit_pruning_number = int(
            round(target_pruned - target_pruned * range_percentage)
        )

        if DEBUG:
            print("Upper limit:", upper_limit_pruning_number)
            print("Lower limit:", lower_limit_pruning_number)

        while stack:
            depth, current_pruned, path = stack.pop()

            if (
                lower_limit_pruning_number
                <= current_pruned
                <= upper_limit_pruning_number
            ):
                results.append(path.copy())
                continue

            if current_pruned > upper_limit_pruning_number or depth == max_depth:
                continue

            for r in range(active_rows[depth] + 1):
                for c in range(active_cols[depth] + 1):
                    pruned_by_this_choice = self.calculate_pruned_elements(
                        row_elements[depth].item(),
                        col_elements[depth].item(),
                        r,
                        c,
                        pruning_order,
                    )
                    new_pruned = current_pruned + pruned_by_this_choice
                    if new_pruned <= upper_limit_pruning_number:
                        stack.append((depth + 1, new_pruned, path + [[r, c]]))

        return results
        # self.helper_for_find_row_col_pruning_combination(
        #     0,
        #     0,
        #     upper_limit_pruning_number,
        #     lower_limit_pruning_number,
        #     max_depth,
        #     row_elements,
        #     col_elements,
        #     active_rows,
        #     active_cols,
        #     [],
        #     results,
        #     pruning_order)

        return results

    def find_all_pruning_combinations_dp(
        self,
        max_depth: int,  # Should be p*q total loading orders
        row_elements: Tensor,  # Number of elements would be turned off by one row; [depth] shape is expected
        col_elements: Tensor,  # Number of elements would be turned off by one col; [depth] shape is expected
        active_rows: Tensor,  # Number of rows are on in the current mask; [depth] shape is expectd
        active_cols: Tensor,  # Number of cols are on in the current mask; [depth] shape is expectd
        target_pruned: int,  # The total number would be pruned off by the target, expect the number from num_remove
        range_percentage: float = 0.1,  # Set a range between for the target_pruned, (target_pruned +- range_percentage * target_pruned)
    ):

        upper_limit = int(target_pruned * (1 + range_percentage))
        lower_limit = int(target_pruned * (1 - range_percentage))

        dp = [{} for _ in range(max_depth + 1)]
        dp[0][0] = [[]]  # Starting point: no pruning at depth 0

        for depth in range(max_depth):
            for pruned_so_far, paths in dp[depth].items():
                for r in range(active_rows[depth].item() + 1):
                    for c in range(active_cols[depth].item() + 1):
                        pruned_by_this = self.calculate_pruned_elements(
                            row_elements[depth].item(), col_elements[depth].item(), r, c
                        )

                        new_pruned = pruned_so_far + pruned_by_this
                        if new_pruned <= upper_limit:
                            if new_pruned not in dp[depth + 1]:
                                dp[depth + 1][new_pruned] = []
                            # Add new paths to the dp structure
                            for path in paths:
                                dp[depth + 1][new_pruned].append(path + [[r, c]])

        # Collect all valid paths
        results = []
        for pruned_count, paths in dp[max_depth].items():
            if lower_limit <= pruned_count <= upper_limit:
                results.extend(paths)

        return results

    def row_col_magnitude_select(
        self,
        mask: MultiMask,
        weight: Tensor,
        name: str,
        death: bool = True,
        num_select: int = 0,
    ) -> MultiMask:

        p, q, r, c, k1, k2 = weight.shape

        # print(death)
        if death:
            # Get full mask
            full_mask = mask.data
            # print(full_mask)

        else:
            full_mask = mask.data
            full_mask = ~full_mask

        full_mask_reshape = full_mask.permute(0, 1, 2, 4, 3, 5).reshape(
            p * q, r * k1, c * k2
        )
        col_turn_off = torch.sum(
            full_mask_reshape, dim=1
        )  # Sum over k1, result is (depth, k2)

        row_turn_off = torch.sum(
            full_mask_reshape, dim=2
        )  # Sum over k2, result is (depth, k1)

        # For each slides,
        # Number of elements turned off by one column; [depth] shape
        col_turn_off_elements = torch.max(col_turn_off, dim=1)[0]

        # For each slides,
        # Number of elements turned off by one row; [depth] shape
        row_turn_off_elements = torch.max(row_turn_off, dim=1)[0]

        col_turn_off_average = col_turn_off_elements.sum().divide(
            col_turn_off_elements.shape[0]
        ).item()

        row_turn_off_average = row_turn_off_elements.sum().divide(
            row_turn_off_elements.shape[0]
        ).item()

        col_total_required_elements = (
            self.HDAC_power / (self.ADC_power + self.TIA_power + self.HDAC_power)
        ) * num_select

        mask_temp = mask.clone()

        mask_temp = self.col_only_magnitude_select(
            mask_temp,
            weight,
            name,
            death,
            row_col=True,
            col_elements_average=col_turn_off_average,
            num_select=col_total_required_elements,
        )

        real_turned_off_by_col = (((mask["col_mask"] == 1) & (mask_temp["col_mask"] == 0)).reshape(p*q, c*k2).sum(-1) * col_turn_off_elements).sum().item()

        row_total_required_elements = num_select - real_turned_off_by_col
        row_required_weight = weight.data * mask_temp

        mask_temp = self.row_only_magnitude_select(
            mask_temp,
            row_required_weight,
            name,
            death,
            row_col=True,
            row_elements_average=row_turn_off_average,
            num_select=row_total_required_elements,
        )

        return mask_temp

    def row_col_magnitude_select_old(
        self,
        mask: MultiMask,
        weight: Tensor,
        name: str,
        death: bool = True,
        num_select: int = 0,
    ) -> MultiMask:

        p, q, r, c, k1, k2 = weight.shape

        # print(death)
        if death:
            # Get full mask
            full_mask = mask.data
            # print(full_mask)

        else:
            full_mask = mask.data
            full_mask = ~full_mask
        # Reshape the mask into the proper number of slides, each slides the k1' and k2'
        # Easy to calculate each column or row can turn off how many elements based on curretn situation
        full_mask_reshape = full_mask.permute(0, 1, 2, 4, 3, 5).reshape(
            p * q, r * k1, c * k2
        )

        # row_magnitude = weight.data.norm(p=2, dim=(3, 5)) # [p,q,r,k1]
        # col_magnitude = weight.data.norm(p=2, dim=(2, 4)) # [p,q,c,k2]
        # row_magnitude_flat = row_magnitude.flatten() # [pqrk1]
        # col_magnitude_flat = col_magnitude.flatten() # [pqck2]
        # total_mag = torch.cat([row_magnitude, col_magnitude], 0) # [pqrk1 + pqck2]
        # sorted_mag, sorted_idx = torch.sort(total_mag)
        # percent = 0.6
        # sorted_idx = sorted_idx[:int(percent * sorted_idx.shape[0])]
        # sorted_idx_row = torch.unravel_index(sorted_idx[sorted_idx < row_magnitude_flat.numel()], row_magnitude.shape) # [p,q,r,k1]
        # sorted_idx_col = torch.unravel_index(sorted_idx[sorted_idx >= row_magnitude_flat.numel()], col_magnitude.shape) # [p,q,c,k2]

        # Row magnitude average
        # Col magnitude average

        col_turn_off = torch.sum(
            full_mask_reshape, dim=1
        )  # Sum over k1, result is (depth, k2)
        row_turn_off = torch.sum(
            full_mask_reshape, dim=2
        )  # Sum over k2, result is (depth, k1)

        # For each slides,
        # Number of elements turned off by one column; [depth] shape
        col_turn_off_elements = torch.max(col_turn_off, dim=1)[0]

        # For each slides,
        # Number of elements turned off by one row; [depth] shape
        row_turn_off_elements = torch.max(row_turn_off, dim=1)[0]

        # For each slides,
        # Number of columns can be turned off; [depth] shape
        col_on_number = torch.sum(mask["col_mask"].reshape(p * q, c * k2), dim=-1)
        # For each slides,
        # Number of rows can be turned off; [depth] shape
        row_on_number = torch.sum(mask["row_mask"].reshape(p * q, r * k1), dim=-1)
        # print("number of elements turned off by row:", row_turn_off_elements)
        # print("number of elements turned off by col:", col_turn_off_elements)
        # print("Turning on rows:", row_on_number)
        # print("Turning on cols:", col_on_number)

        # [Number of different combinations, loading sequece(p*q), 2(row col turning off combination)]
        results = torch.tensor(
            self.find_all_pruning_combinations_dp(
                max_depth=p * q,
                row_elements=row_turn_off_elements,
                col_elements=col_turn_off_elements,
                active_rows=row_on_number,
                active_cols=col_on_number,
                target_pruned=num_select,
                range_percentage=0.2,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        if results.shape[0] == 0:
            return mask

        # All [pq]
        row_wise_magnitude = weight.data.norm(p=2, dim=(3, 5)).reshape(p * q, r * k1)
        row_wise_magnitude = row_wise_magnitude.sum(-1) / (r * k1)
        col_wise_magnitude = weight.data.norm(p=2, dim=(2, 4)).reshape(p * q, c * k2)
        col_wise_magnitude = col_wise_magnitude.sum(-1) / (c * k2)

        print(row_wise_magnitude.shape)
        print(col_wise_magnitude.shape)
        print(results.shape)

        ## [NC, P*Q] x [P*Q] + [NC, P*Q] x [P*Q] = [NC]
        result_mag = results.sum(dim=-1).matmul(row_wise_magnitude) + results.sum(
            dim=-1
        ).matmul(col_wise_magnitude)

        # exit(0)
        # block_wise_magnitude = weight.data.norm(p=2, dim=(2, 3, 4, 5)).flatten()

        # # [NC, P*Q] x [P*Q] = [NC]
        # # print(results.cpu().numpy().tolist())
        # result_mag = results.sum(dim=-1).matmul(block_wise_magnitude)
        # print(results.shape)
        # exit(0)
        # # First filter 2 times max combination
        # # Rough filter
        results = results[torch.argsort(result_mag)[: self.max_combinations * 5]].int()
        # print(results.cpu().numpy().tolist())
        # exit(0)
        # results = results.int()

        if self.group == "layer":  # sort col magnitude per layer
            col_magnitude = weight.data.norm(
                p=2, dim=(2, 4), keepdim=True
            ).square_()  # [p, q, 1, c, 1, k2]
            # col_magnitude = weight.permute(0, 1, 2, 4, 3, 5).reshape(p*q, r*k1, c*k2).norm(p=2, dim=1)
            # Shape [p*q, c*k2]
            if death:
                col_magnitude[~mask["col_mask"]] = 1e8
            else:
                col_magnitude[mask["col_mask"]] = -1

            # [p*q, c*k2]
            col_magnitude = col_magnitude.reshape(p * q, c * k2)
            # Now the shape is [Number of slides(p*q), columns(c*k2)]

            # [p*q, c*k2]
            _, col_indices = torch.sort(
                col_magnitude, dim=-1, descending=False if death else True
            )
            # if DEBUG:
            # print("This is col_indices:", col_indices)
            # print("Col indices shape:", col_indices.shape)

            output_masks = []
            for i in range(results.shape[0]):  # [p*q, c*k2]
                # col_temp_mask = torch.zeros_like(col_indices, dtype=torch.bool)
                # for j in range(p * q):
                #     # print("col_indices shape:", col_indices.shape)
                #     # print("col temp mask shape:", col_temp_mask.shape)
                #     num = results[
                #         i, j, 1
                #     ].item()  # Curent combinnation, current slide, column turning off number
                #     # print(j, num)
                #     # print("These are the indices:", col_indices[j, : num])
                #     col_temp_mask[j, col_indices[j, :num]] = True
                # [p*q]
                num_ones = results[i, :, 1]
                # if torch.any(num_ones > 1):
                #     print(num_ones)
                #     exit(0)
                # [p*q]
                num_zeros = c * k2 - num_ones
                # [p*q*2]
                num_ones_zeros = torch.stack([num_ones, num_zeros], -1).flatten()
                col_temp_mask = torch.repeat_interleave(
                    torch.stack(
                        [
                            torch.ones(p * q, device=self.device, dtype=torch.bool),
                            torch.zeros(p * q, device=self.device, dtype=torch.bool),
                        ],
                        -1,
                    ).flatten(),
                    num_ones_zeros,
                ).reshape(p * q, c * k2)
                col_temp_mask[
                    torch.arange(p * q)[..., None].expand(-1, c * k2), col_indices
                ] = col_temp_mask.clone()
                # print("col_indices:", col_indices)
                # print("num ones:", num_ones)
                # print("col temp mask:", col_temp_mask)
                # print(col_temp_mask)
                col_mags_sum = col_magnitude[col_temp_mask].sum()
                # [p, q, r, c, k1, k2] * [p, q, 1, c, 1, k2]
                # [p, q, r, 1, k1, 1]

                pruned_row_magnitude = (
                    (weight.data * col_temp_mask.reshape(p, q, 1, c, 1, k2))
                    .norm(p=2, dim=(3, 5), keepdim=True)
                    .square_()
                )

                if death:
                    pruned_row_magnitude[~mask["row_mask"]] = 1e8
                else:
                    pruned_row_magnitude[mask["row_mask"]] = -1

                # [p*q, r*k1]
                pruned_row_magnitude = pruned_row_magnitude.reshape(p * q, r * k1)

                # [p*q, r*k1]
                _, row_indices = torch.sort(
                    pruned_row_magnitude, dim=-1, descending=False if death else True
                )

                # print("row_indices shape:", row_indices.shape)

                # row_temp_mask = torch.zeros_like(row_indices, dtype=torch.bool)

                # print("row_tem mask shape:", row_temp_mask.shape)

                # for z in range(p * q):

                #     num = results[
                #         i, z, 0
                #     ].item()  # Curent combinnation, current slide, row turning off number
                #     # Access current slide's which row to be turned on
                #     row_temp_mask[z, row_indices[z, :num]] = True
                row_temp_mask = torch.repeat_interleave(
                    torch.stack(
                        [
                            torch.ones(p * q, device=self.device, dtype=torch.bool),
                            torch.zeros(p * q, device=self.device, dtype=torch.bool),
                        ],
                        -1,
                    ).flatten(),
                    num_ones_zeros,
                ).reshape(p * q, r * k1)
                row_temp_mask[
                    torch.arange(p * q)[..., None].expand(-1, r * k1), row_indices
                ] = row_temp_mask.clone()
                row_mags_sum = pruned_row_magnitude[row_temp_mask].sum()
                # print("row_indices:", row_indices)
                ## True: selected, False: not selected
                col_temp_mask = col_temp_mask.reshape(p, q, 1, c, 1, k2)
                row_temp_mask = row_temp_mask.reshape(p, q, r, 1, k1, 1)

                # new_combo_mask = MultiMask(
                #     {"row_mask": [p, q, r, 1, k1, 1], "col_mask": [p, q, 1, c, 1, k2]}
                # )
                # new_combo_mask["row_mask"] = row_temp_mask
                # new_combo_mask["col_mask"] = col_temp_mask
                ## TODO: row_temp_mask
                output_mask = MultiMask(
                    {
                        "row_mask": (
                            (mask["row_mask"] & ~row_temp_mask)
                            if death
                            else (mask["row_mask"] | row_temp_mask)
                        ),
                        "col_mask": (
                            (mask["col_mask"] & ~col_temp_mask)
                            if death
                            else (mask["col_mask"] | col_temp_mask)
                        ),
                    }
                )
                output_masks.append(
                    (
                        (row_mags_sum + col_mags_sum).item(),
                        output_mask,
                    )
                )

            ## [max_comb, multimask]
            output_masks = [m[1] for m in sorted(output_masks, key=lambda x: x[0])][
                : self.max_combinations
            ]

            opts = self.death_opts if death else self.growth_opts

            def obj_fn(opt: str, mask: MultiMask) -> float:
                if opt == "crosstalk":
                    gain = (
                        (
                            self.calc_crosstalk_scores(mask["col_mask"], is_col=True)
                            + self.calc_crosstalk_scores(
                                mask["row_mask"][..., 0], is_col=False
                            )
                        )
                        .sum()
                        .item()
                    )

                elif opt == "power":
                    gain = -self.get_power_from_mask(mask)[0]
                else:
                    raise NotImplementedError
                return gain  # higher the better

            for opt in opts:
                best_score = float("-inf")
                selected_masks = []
                for mask_temp in output_masks:
                    score = obj_fn(opt, mask_temp)
                    # print(score)
                    # print(mask_temp.data)
                    if score > best_score:
                        selected_masks = [mask_temp]
                        best_score = score
                    elif score == best_score:
                        selected_masks.append(mask_temp)
                output_masks = selected_masks

        if DEBUG:
            print(full_mask)
            print(full_mask.permute(0, 1, 2, 4, 3, 5).reshape(p * q, r * k1, c * k2))
            print("Total number need to be pruned:", num_select)
            print("number of elements turned off by row:", row_turn_off_elements)
            print("number of elements turned off by col:", col_turn_off_elements)
            print("Turning on rows:", row_on_number)
            print("Turning on cols:", col_on_number)
            # print("Reshaped col magnitude:", col_magnitude)
            # print("And its shape:", col_magnitude.shape)
            # print("Length of the results:", type(results[0]))
            # print("Pairs of each P, Q loading sequence, number of elements turned off by one column and by one row under current state:", col_row_possible_turned_off_elements_number_pairs)
            # print("Pairs of each P, Q loading sequence, number of columns and rows on under current state:", current_on_col_row_number_pairs)
            print("Possible combinations:", results)
            print("And it's shape:", results.shape)
        # max_num_col_selected = int(round(num_select / (r * k1)))

        return output_masks[0]

    def momentum_neuron_growth(self, name, new_mask, total_regrowth, weight):
        grad = self.get_momentum_for_weight(weight)

        M = torch.abs(grad)
        if len(M.shape) == 2:
            sum_dim = [1]
        elif len(M.shape) == 4:
            sum_dim = [1, 2, 3]

        v = M.mean(sum_dim).data
        v /= v.sum()

        slots_per_neuron = (new_mask == 0).sum(sum_dim)

        M = M * (new_mask == 0).float()
        for i, fraction in enumerate(v):
            neuron_regrowth = math.floor(fraction.item() * total_regrowth)
            available = slots_per_neuron[i].item()

            y, idx = torch.sort(M[i].flatten())
            if neuron_regrowth > available:
                neuron_regrowth = available
            threshold = y[-(neuron_regrowth)].item()
            if threshold == 0.0:
                continue
            if neuron_regrowth < 10:
                continue
            new_mask[i] = new_mask[i] | (M[i] > threshold)

        return new_mask

    """
                UTILITY
    """

    def get_gradient_for_weights(self, weight):
        grad = weight.grad
        return grad

    def print_nonzero_counts(self):
        for name, mask in self.masks.items():
            num_nonzeros = mask.sum().item()
            val = "{0}: nonzeros={1}->{2}, density: {3:.3f}".format(
                name,
                self.name2nonzeros[name],
                num_nonzeros,
                num_nonzeros / mask.numel(),
            )
            logger.info(val)
        logger.info("Death rate: {0}\n".format(self.death_rate))

    def print_structure_mask(self):
        mlp_total_size = 0
        att_total_size = 0
        mlp_sparse_size = 0
        att_sparse_size = 0

        for name, weight in self.other_masks.items():
            mlp_total_size += weight.numel()
            mlp_sparse_size += (weight != 0).sum().int().item()

        for name, weight in self.atten_masks.items():
            print(
                "{} | {}/{} | shape:{}".format(
                    name, (weight != 0).sum().int().item(), weight.numel(), weight.shape
                )
            )
            att_total_size += weight.numel()
            att_sparse_size += (weight != 0).sum().int().item()

        logger.info(
            "* (Total parameters under density level of mlp [{}/{:.4f}] att [{}/{:.4f}])".format(
                self.args.other_density,
                mlp_sparse_size / mlp_total_size,
                self.args.atten_density,
                att_sparse_size / att_total_size,
            )
        )

    def get_power_from_mask(self, mask: MultiMask) -> Tuple[float, dict]:
        # print("This is col_mask:", mask["col_mask"].shape)
        switch_power = (
            self.cal_ports_power(mask["col_mask"].flatten(0, -2)).sum().item()
        )
        TIA_ADC_power = self.calc_TIA_ADC_powers(mask["row_mask"][..., 0]).sum().item()
        HDAC_power = self.calc_HDAC_powers(mask["col_mask"]).sum().item()

        powers = {
            "switch_power": switch_power,
            "TIA_ADC_power": TIA_ADC_power,
            "HDAC_power": HDAC_power,
        }
        total_power = switch_power + TIA_ADC_power + HDAC_power
        return total_power, powers

    def get_total_power(self):
        total_power = 0
        powers = {}
        for name, mask in self.masks.items():
            power, powers_tmp = self.get_power_from_mask(mask)
            powers[name] = powers_tmp
            total_power += power

        return total_power, powers

    def get_total_crosstalk(self):
        total_crosstalk = 0
        crosstalks = {}
        for name, mask in self.masks.items():

            crosstalk = (
                self.calc_crosstalk_scores(mask["col_mask"], is_col=True).sum().item()
                + self.calc_crosstalk_scores(mask["row_mask"][..., 0], is_col=False)
                .sum()
                .item()
            )
            crosstalks[name] = crosstalk
            total_crosstalk += crosstalk

        return total_crosstalk, crosstalks

    def plot_mask(self, filename: str, save_fig=False):
        num_masks = len(self.masks)
        fig, axes = plt.subplots(3, num_masks, figsize=(4 * num_masks, 10))
        if num_masks == 1:
            axes = [[ax] for ax in axes]
        total_power, powers = self.get_total_power()
        total_crosstalk, crosstalks = self.get_total_crosstalk()
        for i, (name, mask) in enumerate(self.masks.items()):
            k1, k2 = mask["row_mask"].shape[-2], mask["col_mask"].shape[-1]
            mask = mask.data.permute(0, 2, 4, 1, 3, 5).flatten(0, 2).flatten(1)
            axes[0][i].imshow(mask.data.cpu().numpy(), cmap="gray")
            axes[0][i].set_title(
                name
                + " Mask"
                + f"\nTotal Power {total_power:.2f}, Total Crosstalk {total_crosstalk:.3f}"
            )

            weight = (
                self.params[name]
                .data.permute(0, 2, 4, 1, 3, 5)
                .flatten(0, 2)
                .flatten(1)
            )
            axes[1][i].imshow(weight.data.abs().cpu().numpy(), cmap="gray")
            axes[1][i].set_title(name + " abs(Weight)")

            grad = (
                self.params[name]
                .grad.permute(0, 2, 4, 1, 3, 5)
                .flatten(0, 2)
                .flatten(1)
            )
            axes[2][i].imshow(grad.data.abs().cpu().numpy(), cmap="gray")
            axes[2][i].set_title(name + " abs(Grad)")

            for j in range(len(axes)):
                for y in np.arange(-0.5, mask.shape[0] - 0.4, k1):
                    axes[j][i].plot([-0.5, mask.shape[1] - 0.5], [y, y], "b")
                for x in np.arange(-0.5, mask.shape[1] - 0.4, k2):
                    axes[j][i].plot([x, x], [-0.5, mask.shape[0] - 0.5], "b")

        if save_fig:
            ensure_dir("./unitest/figs")
            plt.savefig(f"./unitest/figs/{filename}.png", dpi=300)

    def update_fired_masks(self, pruning_type="unstructure"):
        if pruning_type in {"unstructure", "structure"}:
            ntotal_fired_weights = 0.0
            ntotal_weights = sum(m.numel() for m in self.masks.values())
            layer_fired_weights = {}
            for name, mask in self.masks.items():
                fired_mask = self.fired_masks[name] | mask
                added_fired_weights = fired_mask.sum().item()
                ntotal_fired_weights += added_fired_weights
                layer_fired_weights[name] = added_fired_weights / mask.numel()
            total_fired_weights = ntotal_fired_weights / ntotal_weights
            logger.info(
                f"The percentage of the total fired weights is: {total_fired_weights}"
            )
            return layer_fired_weights, total_fired_weights
        else:
            raise ValueError("Unrecognized Pruning Type !")

    def extra_repr(self) -> str:
        s = f"pruning_type={self.pruning_type}, "
        s += f"death_rate={self.death_rate}, "
        s += f"death_mode={self.death_mode}, "
        s += f"growth_mode={self.growth_mode}, "
        s += f"power_choice_margin={self.power_choice_margin}, "
        s += f"max_combinations={self.max_combinations}, "
        s += f"group={self.group}"

        return s
