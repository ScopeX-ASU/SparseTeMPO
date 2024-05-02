#!/usr/bin/env python
# coding=UTF-8
import argparse
import os
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import mlflow
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from pyutils.config import configs, Config
from pyutils.general import AverageMeter
from pyutils.general import logger as lg
from pyutils.torch_train import (
    BestKModelSaver,
    count_parameters,
    get_learning_rate,
    load_model,
    set_torch_deterministic,
)
from pyutils.typing import Criterion, DataLoader, Optimizer, Scheduler
from core.models.dst2 import MultiMask
from hardware.photonic_crossbar import PhotonicCrossbar
from core import builder
from core.datasets.mixup import Mixup, MixupAll
from core.utils import get_parameter_group, register_hidden_hooks


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: Scheduler,
    epoch: int,
    criterion: Criterion,
    aux_criterions: Dict,
    mixup_fn: Callable = None,
    device: torch.device = torch.device("cuda:0"),
    grad_scaler: Optional[Callable] = None,
    teacher: Optional[nn.Module] = None,
    dst_scheduler: Optional[Callable] = None,
) -> None:
    model.train()
    step = epoch * len(train_loader)

    class_meter = AverageMeter("ce")
    aux_meters = {name: AverageMeter(name) for name in aux_criterions}
    aux_output_weight = getattr(configs.criterion, "aux_output_weight", 0)

    data_counter = 0
    correct = 0
    total_data = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        data_counter += data.shape[0]

        target = target.to(device, non_blocking=True)
        if mixup_fn is not None:
            data, target = mixup_fn(data, target)

        with amp.autocast(enabled=grad_scaler._enabled):
            output = model(data)
            class_loss = criterion(output, target)
            class_meter.update(class_loss.item())
            loss = class_loss

            for name, config in aux_criterions.items():
                aux_criterion, weight = config
                aux_loss = 0
                if name in {"kd", "dkd"} and teacher is not None:
                    with torch.no_grad():
                        teacher_scores = teacher(data).data.detach()
                    aux_loss = weight * aux_criterion(output, teacher_scores, target)
                elif name == "mse_distill" and teacher is not None:
                    with torch.no_grad():
                        teacher(data).data.detach()
                    teacher_hiddens = [
                        m._recorded_hidden
                        for m in teacher.modules()
                        if hasattr(m, "_recorded_hidden")
                    ]
                    student_hiddens = [
                        m._recorded_hidden
                        for m in model.modules()
                        if hasattr(m, "_recorded_hidden")
                    ]

                    aux_loss = weight * sum(
                        F.mse_loss(h1, h2)
                        for h1, h2 in zip(teacher_hiddens, student_hiddens)
                    )
                loss = loss + aux_loss
                aux_meters[name].update(aux_loss)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).sum().item()

        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        if configs.run.grad_clip:
            torch.nn.utils.clip_grad_value_(
                [p for p in model.parameters() if p.requires_grad],
                float(configs.run.max_grad_value),
            )
        grad_scaler.step(optimizer)
        grad_scaler.update()
        step += 1

        if dst_scheduler is not None:
            dst_scheduler.step()  # apply pruning mask and update rate

        if batch_idx % int(configs.run.log_interval) == 0:
            log = "Train Epoch: {} [{:7d}/{:7d} ({:3.0f}%)] Loss: {:.4e} class Loss: {:.4e}".format(
                epoch,
                data_counter,
                total_data,
                100.0 * data_counter / total_data,
                loss.data.item(),
                class_loss.data.item(),
            )
            for name, aux_meter in aux_meters.items():
                log += f" {name}: {aux_meter.val:.4e}"
            lg.info(log)

            mlflow.log_metrics({"train_loss": loss.item()}, step=step)

    scheduler.step()
    avg_class_loss = class_meter.avg
    accuracy = 100.0 * correct / total_data
    lg.info(
        f"Train class Loss: {avg_class_loss:.4e}, Accuracy: {correct}/{total_data} ({accuracy:.2f}%)"
    )
    mlflow.log_metrics(
        {
            "train_class": avg_class_loss,
            "train_acc": accuracy,
            "lr": get_learning_rate(optimizer),
        },
        step=epoch,
    )
    if dst_scheduler is not None:
        lg.info(f"Crosstalk value:{dst_scheduler.get_total_crosstalk()}")
        lg.info(f"Power:{dst_scheduler.get_total_power()}")


def validate(
    model: nn.Module,
    validation_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    fp16: bool = False,
) -> None:
    model.eval()
    val_loss = 0
    correct = 0
    class_meter = AverageMeter("ce")
    with amp.autocast(enabled=fp16):
        with torch.no_grad():
            for i, (data, target) in enumerate(validation_loader):
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                if mixup_fn is not None:
                    data, target = mixup_fn(data, target, random_state=i, vflip=False)

                output = model(data)

                val_loss = criterion(output, target)
                class_meter.update(val_loss.item())
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

    loss_vector.append(class_meter.avg)
    accuracy = 100.0 * correct / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)
    lg.info(
        f"\nValidation set: Average loss: {class_meter.avg:.4e}, Accuracy: {correct}/{len(validation_loader.dataset)} ({accuracy:.2f}%)\n"
    )
    mlflow.log_metrics({"val_loss": class_meter.avg, "val_acc": accuracy}, step=epoch)


def test(
    model: nn.Module,
    test_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    fp16: bool = False,
) -> None:
    model.eval()
    val_loss = 0
    correct = 0
    class_meter = AverageMeter("ce")
    with amp.autocast(enabled=fp16):
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                if mixup_fn is not None:
                    data, target = mixup_fn(
                        data, target, random_state=i + 10000, vflip=False
                    )

                output = model(data)

                val_loss = criterion(output, target)
                class_meter.update(val_loss.item())
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

    loss_vector.append(class_meter.avg)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    accuracy_vector.append(accuracy)

    # lg.info(
    #     f"\nTest set: Average loss: {class_meter.avg:.4e}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    # )

    # mlflow.log_metrics(
    #     {"test_loss": class_meter.avg, "test_acc": accuracy}, step=epoch
    # )
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    # parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    # parser.add_argument('--pdb', action='store_true', help='pdb')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)
    arch_config = Config()
    arch_config.load("./configs/hardware/arch_config.yaml")
    configs.update({"arch": arch_config.dict()})
    lg.info(configs)
    configs.arch.core.precision.in_bit = in_bit = configs.model.conv_cfg.in_bit
    configs.arch.core.precision.w_bit = w_bit = configs.model.conv_cfg.w_bit
    configs.arch.core.precision.act_bit = act_bit = configs.model.conv_cfg.w_bit
    r ,c, k1, k2 = configs.model.conv_cfg.miniblock
    configs.arch.core.width = k2
    configs.arch.core.height = k1
    configs.arch.arch.r = r
    configs.arch.arch.c = c
    work_freq = configs.arch.core.work_freq
    hw = PhotonicCrossbar(k2, k1, 1, in_bit, w_bit, act_bit, w_bit, configs.arch)
    if torch.cuda.is_available() and int(configs.run.use_cuda):
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device("cuda:" + str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    if int(configs.run.deterministic) == True:
        set_torch_deterministic()

    train_loader, validation_loader, test_loader = builder.make_dataloader(
        splits=["train", "valid", "test"]
    )

    if (
        configs.run.do_distill
        and configs.teacher is not None
        and os.path.exists(configs.teacher.checkpoint)
    ):
        teacher = builder.make_model(device, model_cfg=configs.teacher)
        load_model(teacher, path=configs.teacher.checkpoint)
        teacher.eval()
        lg.info(f"Load teacher model from {configs.teacher.checkpoint}")
    else:
        teacher = None

    model = builder.make_model(
        device,
        model_cfg=configs.model,
        random_state=(
            int(configs.run.random_state) if int(configs.run.deterministic) else None
        ),
    )
    ## dummy forward to initialize quantizer
    model(next(iter(test_loader))[0].to(device))

    optimizer = builder.make_optimizer(
        get_parameter_group(model, weight_decay=float(configs.optimizer.weight_decay)),
        name=configs.optimizer.name,
        configs=configs.optimizer,
    )
    scheduler = builder.make_scheduler(optimizer)
    criterion = builder.make_criterion(configs.criterion.name, configs.criterion).to(
        device
    )

    aux_criterions = dict()
    if configs.aux_criterion is not None:
        for name, config in configs.aux_criterion.items():
            if float(config.weight) > 0:
                try:
                    fn = builder.make_criterion(name, cfg=config)
                except NotImplementedError:
                    fn = name
                aux_criterions[name] = [fn, float(config.weight)]
    print(aux_criterions)
    if "mse_distill" in aux_criterions and teacher is not None:
        ## register hooks for teacher and student
        register_hidden_hooks(teacher)
        register_hidden_hooks(model)
        print(len([m for m in teacher.modules() if hasattr(m, "_recorded_hidden")]))
        print(len([m for m in teacher.modules() if hasattr(m, "_recorded_hidden")]))
        print(f"Register hidden state hooks for teacher and students")

    if configs.dst_scheduler is not None:
        dst_scheduler = builder.make_dst_scheduler(
            optimizer, model, train_loader, configs
        )
    else:
        dst_scheduler = None
    lg.info(model)
    mixup_config = configs.dataset.augment
    mixup_fn = MixupAll(**mixup_config) if mixup_config is not None else None
    test_mixup_fn = (
        MixupAll(**configs.dataset.test_augment) if mixup_config is not None else None
    )
    saver = BestKModelSaver(
        k=int(configs.checkpoint.save_best_model_k),
        descend=True,
        truncate=2,
        metric_name="acc",
        format="{:.2f}",
    )
    grad_scaler = amp.GradScaler(enabled=getattr(configs.run, "fp16", False))
    lg.info(f"Number of parameters: {count_parameters(model)}")

    model_name = f"{configs.model.name}"
    checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.pt"

    lg.info(f"Current checkpoint: {checkpoint}")

    mlflow.set_experiment(configs.run.experiment)
    experiment = mlflow.get_experiment_by_name(configs.run.experiment)

    # run_id_prefix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    mlflow.start_run(run_name=model_name)
    mlflow.log_params(
        {
            "exp_name": configs.run.experiment,
            "exp_id": experiment.experiment_id,
            "run_id": mlflow.active_run().info.run_id,
            "init_lr": configs.optimizer.lr,
            "checkpoint": checkpoint,
            "restore_checkpoint": configs.checkpoint.restore_checkpoint,
            "pid": os.getpid(),
        }
    )

    lossv, accv = [0], [0]
    epoch = 0

    try:
        lg.info(
            f"Experiment {configs.run.experiment} ({experiment.experiment_id}) starts. Run ID: ({mlflow.active_run().info.run_id}). PID: ({os.getpid()}). PPID: ({os.getppid()}). Host: ({os.uname()[1]})"
        )
        if (
            int(configs.checkpoint.resume)
            and len(configs.checkpoint.restore_checkpoint) > 0
        ):
            load_model(
                model,
                configs.checkpoint.restore_checkpoint,
                ignore_size_mismatch=int(configs.checkpoint.no_linear),
            )
            for name, m in model.named_modules():
                if isinstance(m, model._conv_linear):  # no last fc layer
                    if hasattr(m, "row_prune_mask") and m.row_prune_mask is not None and hasattr(m, "col_prune_mask") and m.col_prune_mask is not None:
                        m.prune_mask = MultiMask({"row_mask": m.row_prune_mask, "col_mask": m.col_prune_mask})
                        percent = m.row_prune_mask.sum() / m.row_prune_mask.numel()
                        print(percent)
            lg.info("Validate resumed model...")
            acc = test(
                model,
                validation_loader,
                0,
                criterion,
                lossv,
                accv,
                device,
                fp16=grad_scaler._enabled,
            )
            print(f"Validate loaded checkpoint validation acc: {acc}")
        if teacher is not None:
            test(
                teacher,
                validation_loader,
                0,
                criterion,
                [],
                [],
                device,
                fp16=grad_scaler._enabled,
            )
            lg.info("Map teacher to student...")
            if hasattr(model, "load_from_teacher"):
                with amp.autocast(grad_scaler._enabled):
                    model.load_from_teacher(teacher)

        ## compile models
        if getattr(configs.run, "compile", False):
            model = torch.compile(model)
            if teacher is not None:
                teacher = torch.compile(teacher)
        
        model.set_noise_flag(True)
        model.set_crosstalk_noise(True)
        model.set_output_noise(configs.noise.output_noise_std)
        model.set_light_redist(configs.noise.light_redist)
        model.set_input_power_gating(configs.noise.input_power_gating, configs.noise.input_modulation_ER)
        model.set_output_power_gating(configs.noise.output_power_gating)

        interv_s_minax = [7, 12]
        interv_s_range = np.arange(interv_s_minax[0], interv_s_minax[1] + 0.1, 2)

        interv_h_s_minax = [1, 6]
        interv_h_s_range = np.arange(interv_h_s_minax[0], interv_h_s_minax[1] + 0.1, 2)

        R = configs.arch.arch.num_tiles
        C = configs.arch.arch.num_pe_per_tile

        mzi_total_energy, mzi_energy_dict, _, cycle_dict, _, _ = model.calc_weight_MZI_energy(next(iter(test_loader))[0].shape, R=R, C=C, freq=work_freq)

        total_cycles = 0
        for key, value in cycle_dict.items():
            total_cycles += value[0]  

        layer_energy, layer_energy_breakdown, newtwork_energy_breakdown, total_energy = hw.calc_total_energy(cycle_dict, dst_scheduler, model)
        
        acc_list = []
        avg_power_list = []
        for interv_s in interv_s_range:
            for interv_h_s in interv_h_s_range:
                interv_h = interv_h_s + interv_s + model.crosstalk_scheduler.ps_width
                model.crosstalk_scheduler.set_spacing(
                    interv_s=interv_s, interv_h=interv_h
                )
                if configs.noise.output_noise_std > 0:
                    N = 3
                else:
                    N = 1
                accs = []
                for i in range(N):
                    acc = test(
                        model,
                        test_loader,
                        0,
                        criterion,
                        [],
                        [],
                        device,
                        mixup_fn=None,
                        fp16=False,
                    )
                    accs.append(acc)
                acc = np.mean(accs)
                acc_list.append((interv_s, interv_h_s, acc))
                print(f"interv_s: {interv_s}, interv_h: {interv_h}, acc: {acc}")
            # next(iter(test_loader))[0].shape
            mzi_total_energy, mzi_energy_dict, _, _, _, _ = model.calc_weight_MZI_energy(next(iter(test_loader))[0].shape, R=R, C=C, freq=work_freq)
            for key in layer_energy:
                layer_energy[key] += mzi_energy_dict[key]
                layer_energy_breakdown[key]["MZI Power"] = mzi_energy_dict[key]
                newtwork_energy_breakdown["MZI Power"] = mzi_total_energy
                total_energy += mzi_total_energy

            


        acc_list = np.array(acc_list)
        avg_power_list = np.array(avg_power_list)
        print(acc_list.tolist())
        print(avg_power_list.tolist())

        np.savetxt(f"./log/fmnist/cnn/test_structural_pruning_without_optimization/{configs.loginfo}.csv",  acc_list, delimiter=",", fmt="%.2f")
        np.savetxt(f"./log/fmnist/cnn/test_structural_pruning_without_optimization/{configs.loginfo}_acc_matrix.csv",  acc_list[:, -1].reshape([-1, interv_h_s_range.shape[0]]), delimiter=",", fmt="%.2f")
        np.savetxt(f"./log/fmnist/cnn/test_structural_pruning_without_optimization/{configs.loginfo}_avgpower_list.csv",  avg_power_list, delimiter=",", fmt="%.2f")

    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")


if __name__ == "__main__":
    main()
