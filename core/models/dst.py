from __future__ import print_function

import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pyutils.general import logger

__all__ = ["DSTScheduler", "CosineDecay", "LinearDecay"]


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
            name: torch.ones(shape, device=device, dtype=torch.bool)
            for name, shape in mask_cfg.items()
        }

        try:
            mask = self.data
        except:
            raise ValueError("mask shapes should be able to multiplied together.")

        self.total_elem = mask.numel()
        self.shape = mask.shape

    def __getitem__(self, key):
        return self._masks[key]

    @property
    def data(self):
        out = 1
        for mask in self._masks.values():
            out = out * mask
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
        return self.data * other

    def __rmul__(self, other):
        return self.data * other

    def __eq__(self, other):
        return self.data == other

    def __invert__(self):
        return ~self.data

    def __and__(self, other):
        return self.data & other

    def __or__(self, other):
        return self.data | other

    def __xor__(self, other):
        return self.data ^ other


class DSTScheduler(object):
    def __init__(
        self,
        optimizer,
        death_rate=0.3,
        growth_death_ratio=1.0,
        death_rate_decay=None,
        death_mode="magnitude",
        growth_mode="momentum",
        redistribution_mode="momentum",
        args=None,
        spe_initial=None,
        train_loader=None,
        pruning_type="structure",
        update_frequency: int = 100,
        T_max: int = 10000,
        device="cuda:0",
    ):
        ## Dynamic Sparse Training Scheduler

        growth_modes = ["random", "momentum", "momentum_neuron", "gradient"]
        if growth_mode not in growth_modes:
            raise ValueError(
                f"Growth mode expects {growth_modes}, but got {growth_mode}."
            )

        self.args = args
        self.loader = train_loader
        self.modules = []
        self.optimizer = optimizer

        self.growth_death_ratio = growth_death_ratio
        self.growth_mode = growth_mode  # gradient
        self.redistribution_mode = redistribution_mode  # momentum
        self.spe_initial = spe_initial  # initial masks made by SNIP
        self.snip_masks = None  # masks made by SNIP during training
        self.nonzeros_index = None
        self.pruning_type = pruning_type
        self.update_frequency = update_frequency
        self.T_max = T_max

        self.steps = 0
        self.device = device

        self.names = []
        self.masks = {}
        self.atten_masks = {}
        self.other_masks = {}
        self.newly_masks = {}
        # death
        self.death_mode = death_mode  # magnitude
        self.death_rate = death_rate
        self.death_rate_decay = death_rate_decay
        self.name2death_rate = {}

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
        init_mode: str = "fixed_ERK",
        pruning_type="unstructure",
        mask_path=None,
    ):

        if pruning_type in {"unstructure"}:
            self.modules.append(module)
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
        elif pruning_type in {"structure"}:
            self.modules.append(module)
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

    def step(self, pruning_type=None):
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

    def at_end_of_epoch(self, pruning_type="unstructure", indicator_list=None):
        if pruning_type == "unstructure":
            self.update_and_apply_mask()
            _, _ = self.update_fired_masks()
            self.print_nonzero_counts()
        elif pruning_type == "structure":
            self.update_and_apply_mask(pruning_type, indicator_list)
            _, _ = self.update_fired_masks(pruning_type="structure")
        elif pruning_type == "structure_new":
            self.update_and_apply_mask(pruning_type)
            _, _ = self.update_fired_masks(pruning_type="structure_new")
        else:
            raise ValueError("Unrecognized Pruning Type !")

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
                        raw_probabilities[name] = (
                            np.sum(mask.shape) / n_param
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

    def structure_init(self, mode="fixed_ERK", density=0.05, erk_power_scale=1.0):
        raise NotImplementedError

    # multiple mask for paramenters and momentum in optimizers
    def apply_mask(self, pruning_type="unstructure"):
        if pruning_type in {"unstructure", "structure"}:
            for name in self.masks:
                mask = self.masks[name]
                weight = self.params[name]
                weight.data *= mask
                state = self.optimizer.state[weight]
                if "momentum_buffer" in state:
                    state["momentum_buffer"] *= mask
        else:
            raise ValueError("Unrecognized Pruning Type !")

    def gather_statistics(self, pruning_type="unstructure"):
        if pruning_type in {"unstructure", "structure"}:
            self.name2nonzeros = {
                name: mask.sum().item() for name, mask in self.masks.items()
            }
            self.name2zeros = {
                name: mask.numel() - self.name2nonzeros[name]
                for name, mask in self.masks.items()
            }
        else:
            raise ValueError("Unrecognized Pruning Type !")

    def update_and_apply_mask(self, pruning_type="unstructure", indicator_list=None):
        # update pruning and growth masks
        if pruning_type in {"unstructure", "structure"}:
            self.gather_statistics()  # count each of module's zeros and non-zeros
            # update pruning mask
            for name, mask in self.masks.items():
                weight = self.params[name]
                if self.death_mode == "magnitude":
                    new_mask = self.magnitude_death(mask, weight, name)
                elif self.death_mode == "SET":
                    new_mask = self.magnitude_and_negativity_death(mask, weight, name)
                elif self.death_mode == "threshold":
                    new_mask = self.threshold_death(mask, weight, name)
                self.pruned_number[name] = int(
                    self.name2nonzeros[name] - new_mask.sum().item()
                )
                self.masks[name] = new_mask  # update new mask

            # update pruning mask with growing
            for name, mask in self.masks.items():
                weight = self.params[name]
                if self.growth_mode == "random":
                    new_mask = self.random_growth(
                        name, mask, self.pruned_number[name], weight
                    )
                elif self.growth_mode == "momentum":
                    new_mask = self.momentum_growth(
                        name, mask, self.pruned_number[name], weight
                    )
                elif self.growth_mode == "gradient":
                    new_mask = self.gradient_growth(
                        name, mask, self.pruned_number[name], weight
                    )
                self.masks[name] = new_mask

            self.apply_mask()
        else:
            raise ValueError("Unrecognized Pruning Type !")

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
        # calculate scores for all weights
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
            new_weights = new_mask.new_zeros().bernoulli_(p=expected_growth_probability)
        elif self.pruning_type == "structure":
            raise NotImplementedError
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
            raise ValueError("Unrecognized Pruning Type !")
        return new_mask

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
        grad = weight.grad.clone()
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
