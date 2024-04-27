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
from pyutils.config import configs
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
    lg.info(configs)

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
    lg.info(model)

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

            lg.info("Validate resumed model...")
            test(
                model,
                validation_loader,
                0,
                criterion,
                lossv,
                accv,
                device,
                fp16=grad_scaler._enabled,
            )
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

        interv_s_minax = [7, 25]
        interv_s_range = np.arange(interv_s_minax[0], interv_s_minax[1] + 0.1, 2)

        interv_h_s_minax = [1, 25]
        interv_h_s_range = np.arange(interv_h_s_minax[0], interv_h_s_minax[1] + 0.1, 2)

        acc_list = []
        for interv_s in interv_s_range:
            for interv_h_s in interv_h_s_range:
                interv_h = interv_h_s + interv_s + model.crosstalk_scheduler.ps_width
                model.crosstalk_scheduler.set_spacing(
                    interv_s=interv_s, interv_h=interv_h
                )
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
                acc_list.append((interv_s, interv_h_s, acc))
                print(f"interv_s: {interv_s}, interv_h: {interv_h}, acc: {acc}")
        acc_list = np.array(acc_list)
        print(acc_list)

        np.savetxt(f"./log/fmnist/cnn/crosstalk_spacing/crosstalk_spacing_acc_list.csv",  acc_list, delimiter=",", fmt="%.2f")
        """
[[ 7.,    1.,   85.89],
 [ 7.,    3.,   89.87],
 [ 7.,    5.,   90.98],
 [ 7.,    7.,   91.27],
 [ 7.,    9.,   91.4 ],
 [ 7.,   11.,   91.49],
 [ 7.,   13.,   91.42],
 [ 7.,   15.,   91.52],
 [ 7.,   17.,   91.52],
 [ 7.,   19.,   91.54],
 [ 7.,   21.,   91.57],
 [ 7.,   23.,   91.58],
 [ 7.,   25.,   91.56],
 [ 9.,    1.,   85.53],
 [ 9.,    3.,   89.75],
 [ 9.,    5.,   90.87],
 [ 9.,    7.,   91.19],
 [ 9.,    9.,   91.36],
 [ 9.,   11.,   91.44],
 [ 9.,   13.,   91.44],
 [ 9.,   15.,   91.52],
 [ 9.,   17.,   91.56],
 [ 9.,   19.,   91.56],
 [ 9.,   21.,   91.53],
 [ 9.,   23.,   91.54],
 [ 9.,   25.,   91.56],
 [11.,    1.,   85.26],
 [11.,    3.,   89.59],
 [11.,    5.,   90.81],
 [11.,    7.,   91.21],
 [11.,    9.,   91.37],
 [11.,   11.,   91.43],
 [11.,   13.,   91.47],
 [11.,   15.,   91.54],
 [11.,   17.,   91.52],
 [11.,   19.,   91.55],
 [11.,   21.,   91.53],
 [11.,   23.,   91.53],
 [11.,   25.,   91.59],
 [13.,    1.,   84.96],
 [13.,    3.,   89.45],
 [13.,    5.,   90.78],
 [13.,    7.,   91.19],
 [13.,    9.,   91.33],
 [13.,   11.,   91.41],
 [13.,   13.,   91.47],
 [13.,   15.,   91.52],
 [13.,   17.,   91.52],
 [13.,   19.,   91.52],
 [13.,   21.,   91.55],
 [13.,   23.,   91.54],
 [13.,   25.,   91.58],
 [15.,    1.,   84.82],
 [15.,    3.,   89.45],
 [15.,    5.,   90.76],
 [15.,    7.,   91.18],
 [15.,    9.,   91.34],
 [15.,   11.,   91.39],
 [15.,   13.,   91.47],
 [15.,   15.,   91.51],
 [15.,   17.,   91.54],
 [15.,   19.,   91.57],
 [15.,   21.,   91.57],
 [15.,   23.,   91.56],
 [15.,   25.,   91.53],
 [17.,    1.,   84.86],
 [17.,    3.,   89.42],
 [17.,    5.,   90.8 ],
 [17.,    7.,   91.18],
 [17.,    9.,   91.36],
 [17.,   11.,   91.41],
 [17.,   13.,   91.46],
 [17.,   15.,   91.52],
 [17.,   17.,   91.52],
 [17.,   19.,   91.53],
 [17.,   21.,   91.58],
 [17.,   23.,   91.55],
 [17.,   25.,   91.59],
 [19.,    1.,   84.83],
 [19.,    3.,   89.32],
 [19.,    5.,   90.77],
 [19.,    7.,   91.21],
 [19.,    9.,   91.35],
 [19.,   11.,   91.38],
 [19.,   13.,   91.43],
 [19.,   15.,   91.5 ],
 [19.,   17.,   91.53],
 [19.,   19.,   91.55],
 [19.,   21.,   91.56],
 [19.,   23.,   91.54],
 [19.,   25.,   91.56],
 [21.,    1.,   84.82],
 [21.,    3.,   89.32],
 [21.,    5.,   90.76],
 [21.,    7.,   91.2 ],
 [21.,    9.,   91.35],
 [21.,   11.,   91.4 ],
 [21.,   13.,   91.46],
 [21.,   15.,   91.51],
 [21.,   17.,   91.51],
 [21.,   19.,   91.55],
 [21.,   21.,   91.56],
 [21.,   23.,   91.54],
 [21.,   25.,   91.58],
 [23.,    1.,   84.87],
 [23.,    3.,   89.25],
 [23.,    5.,   90.81],
 [23.,    7.,   91.21],
 [23.,    9.,   91.34],
 [23.,   11.,   91.4 ],
 [23.,   13.,   91.47],
 [23.,   15.,   91.54],
 [23.,   17.,   91.51],
 [23.,   19.,   91.56],
 [23.,   21.,   91.57],
 [23.,   23.,   91.54],
 [23.,   25.,   91.54],
 [25.,    1.,   84.82],
 [25.,    3.,   89.26],
 [25.,    5.,   90.82],
 [25.,    7.,   91.22],
 [25.,    9.,   91.28],
 [25.,   11.,   91.4 ],
 [25.,   13.,   91.45],
 [25.,   15.,   91.49],
 [25.,   17.,   91.5 ],
 [25.,   19.,   91.58],
 [25.,   21.,   91.59],
 [25.,   23.,   91.53],
 [25.,   25.,   91.56]]"""

    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")


if __name__ == "__main__":
    main()
