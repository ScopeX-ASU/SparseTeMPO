"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-03-31 17:48:41
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-09-26 00:51:50
"""

#########
import os
from tkinter import NS
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from pyutils.config import configs
from pyutils.datasets import get_dataset
from pyutils.loss import AdaptiveLossSoft, KDLoss, DKDLoss, CrossEntropyLossSmooth
from pyutils.lr_scheduler.warmup_cosine_restart import CosineAnnealingWarmupRestarts
from pyutils.optimizer.sam import SAM
from pyutils.optimizer.dadapt_adam import DAdaptAdam
from pyutils.optimizer.dadapt_sgd import DAdaptSGD
from pyutils.optimizer.prodigy import Prodigy
from pyutils.typing import DataLoader, Optimizer, Scheduler
from torch.types import Device

from core.datasets import (
    CIFAR10Dataset,
    CIFAR100Dataset,
    FashionMNISTDataset,
    MNISTDataset,
    SVHNDataset,
)

from core.models import *

__all__ = [
    "make_dataloader",
    "make_model",
    "make_optimizer",
    "make_scheduler",
    "make_criterion",
]


def make_dataloader(
    cfg: dict = None, splits=["train", "valid", "test"]
) -> Tuple[DataLoader, DataLoader]:
    cfg = cfg or configs.dataset
    name = cfg.name.lower()
    if name == "mnist":
        train_dataset, validation_dataset, test_dataset = (
            MNISTDataset(
                root=cfg.root,
                split=split,
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                center_crop=cfg.center_crop,
                resize=cfg.img_height,
                resize_mode=cfg.resize_mode,
                binarize=False,
                binarize_threshold=0.273,
                digits_of_interest=list(range(10)),
                n_test_samples=cfg.n_test_samples,
                n_valid_samples=cfg.n_valid_samples,
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "fmnist":
        train_dataset, validation_dataset, test_dataset = (
            FashionMNISTDataset(
                root=cfg.root,
                split=split,
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                center_crop=cfg.center_crop,
                resize=cfg.img_height,
                resize_mode=cfg.resize_mode,
                binarize=False,
                n_test_samples=cfg.n_test_samples,
                n_valid_samples=cfg.n_valid_samples,
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "cifar10":
        train_dataset, validation_dataset, test_dataset = (
            CIFAR10Dataset(
                root=cfg.root,
                split=split,
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                center_crop=cfg.center_crop,
                resize=cfg.img_height,
                resize_mode=cfg.resize_mode,
                binarize=False,
                grayscale=False,
                n_test_samples=cfg.n_test_samples,
                n_valid_samples=cfg.n_valid_samples,
                augment=cfg.transform != "basic",
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "cifar100":
        train_dataset, validation_dataset, test_dataset = (
            CIFAR100Dataset(
                root=cfg.root,
                split=split,
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                center_crop=cfg.center_crop,
                resize=cfg.img_height,
                resize_mode=cfg.resize_mode,
                binarize=False,
                grayscale=False,
                n_test_samples=cfg.n_test_samples,
                n_valid_samples=cfg.n_valid_samples,
                augment=cfg.transform != "basic",
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "svhn":
        train_dataset, validation_dataset, test_dataset = (
            SVHNDataset(
                root=cfg.root,
                split=split,
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                center_crop=cfg.center_crop,
                resize=cfg.img_height,
                resize_mode=cfg.resize_mode,
                binarize=False,
                grayscale=False,
                n_test_samples=cfg.n_test_samples,
                n_valid_samples=cfg.n_valid_samples,
                augment=cfg.transform != "basic",
            )
            for split in ["train", "valid", "test"]
        )
    else:
        train_dataset, test_dataset = get_dataset(
            name,
            cfg.img_height,
            cfg.img_width,
            dataset_dir=cfg.root,
            transform=cfg.transform,
        )
        validation_dataset = None

    train_loader = (
        torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=configs.run.batch_size,
            shuffle=int(cfg.shuffle),
            pin_memory=True,
            num_workers=cfg.num_workers,
        )
        if train_dataset is not None
        else None
    )

    validation_loader = (
        torch.utils.data.DataLoader(
            dataset=validation_dataset,
            batch_size=configs.run.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=cfg.num_workers,
        )
        if validation_dataset is not None
        else None
    )

    test_loader = (
        torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=configs.run.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=cfg.num_workers,
        )
        if test_dataset is not None
        else None
    )

    return train_loader, validation_loader, test_loader


def make_model(
    device: Device, model_cfg: Optional[str] = None, random_state: int = None, **kwargs
) -> nn.Module:
    model_cfg = model_cfg or configs.model
    name = model_cfg.name
    if "cnn" in model_cfg.name.lower():
        model = eval(model_cfg.name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channels=configs.dataset.in_channels,
            num_classes=configs.dataset.num_classes,
            kernel_list=model_cfg.kernel_list,
            kernel_size_list=model_cfg.kernel_size_list,
            pool_out_size=model_cfg.pool_out_size,
            stride_list=model_cfg.stride_list,
            padding_list=model_cfg.padding_list,
            hidden_list=model_cfg.hidden_list,
            conv_cfg=model_cfg.conv_cfg,
            linear_cfg=model_cfg.linear_cfg,
            norm_cfg=model_cfg.norm_cfg,
            act_cfg=model_cfg.act_cfg,
            device=device,
        ).to(device)
        model.reset_parameters(random_state)
    elif "vgg" in model_cfg.name.lower():
        model = eval(model_cfg.name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channels=configs.dataset.in_channels,
            num_classes=configs.dataset.num_classes,
            conv_cfg=model_cfg.conv_cfg,
            linear_cfg=model_cfg.linear_cfg,
            norm_cfg=model_cfg.norm_cfg,
            act_cfg=model_cfg.act_cfg,
            device=device,
        ).to(device)
        model.reset_parameters(random_state)
    elif "resnet" in name.lower():
        model = eval(name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channels=configs.dataset.in_channels,
            num_classes=configs.dataset.num_classes,
            conv_cfg=model_cfg.conv_cfg,
            linear_cfg=model_cfg.linear_cfg,
            norm_cfg=model_cfg.norm_cfg,
            act_cfg=model_cfg.act_cfg,
            device=device,
        ).to(device)

    else:
        model = None
        raise NotImplementedError(f"Not supported model name: {model_cfg.name}")
    return model


def make_optimizer(params, name: str = None, configs=None) -> Optimizer:
    if name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=configs.lr,
            momentum=configs.momentum,
            weight_decay=configs.weight_decay,
            nesterov=True,
        )
    elif name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=configs.lr,
            weight_decay=configs.weight_decay,
            betas=getattr(configs, "betas", (0.9, 0.999)),
        )
    elif name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
    elif name == "sam_sgd":
        base_optimizer = torch.optim.SGD
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(configs, "rho", 0.5),
            adaptive=getattr(configs, "adaptive", True),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
            momenum=0.9,
        )
    elif name == "sam_adam":
        base_optimizer = torch.optim.Adam
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(configs, "rho", 0.001),
            adaptive=getattr(configs, "adaptive", True),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
    elif name == "dadapt_adam":
        optimizer = DAdaptAdam(
            params,
            lr=configs.lr,
            betas=getattr(configs, "betas", (0.9, 0.999)),
            weight_decay=configs.weight_decay,
        )
    elif name == "dadapt_sgd":
        optimizer = DAdaptSGD(
            params,
            lr=configs.lr,
            momentum=configs.momentum,
            weight_decay=configs.weight_decay,
        )
    elif name == "prodigy":
        optimizer = Prodigy(
            params,
            lr=configs.lr,
            betas=getattr(configs, "betas", (0.9, 0.999)),
            weight_decay=configs.weight_decay,
        )
    else:
        raise NotImplementedError(name)

    return optimizer


def make_scheduler(optimizer: Optimizer, name: str = None) -> Scheduler:
    name = (name or configs.scheduler.name).lower()
    if name == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1
        )
    elif name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(configs.run.n_epochs),
            eta_min=float(configs.scheduler.lr_min),
        )
    elif name == "cosine_warmup":
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=configs.run.n_epochs,
            max_lr=configs.optimizer.lr,
            min_lr=configs.scheduler.lr_min,
            warmup_steps=int(configs.scheduler.warmup_steps),
        )
    elif name == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=configs.scheduler.lr_gamma
        )
    else:
        raise NotImplementedError(name)

    return scheduler


def make_criterion(name: str = None, cfg=None) -> nn.Module:
    name = (name or configs.criterion.name).lower()
    # cfg = cfg or configs.criterion
    if name == "nll":
        criterion = nn.NLLLoss()
    elif name == "mse":
        criterion = nn.MSELoss()
    elif name == "mae":
        criterion = nn.L1Loss()
    elif name == "ce":
        criterion = nn.CrossEntropyLoss()
    elif name == "ce_smooth":
        criterion = CrossEntropyLossSmooth(label_smoothing=getattr(cfg, "label_smoothing", 0.1))
    elif name == "adaptive":
        criterion = AdaptiveLossSoft(alpha_min=-1.0, alpha_max=1.0)
    elif name == "kd":
        criterion = KDLoss(
            T=getattr(cfg, "T", 2),
            ce_weight=getattr(cfg, "ce_weight", 0),
            kd_weight=getattr(cfg, "kd_weight", 0.9),
            logit_stand=getattr(cfg, "logit_stand", False),
        )
    elif name == "dkd":
        criterion = DKDLoss(
            T=getattr(cfg, "T", 2),
            ce_weight=getattr(cfg, "ce_weight", 0),
            kd_alpha=getattr(cfg, "kd_alpha", 1),
            kd_beta=getattr(cfg, "kd_beta", 1),
            logit_stand=getattr(cfg, "logit_stand", False),
        )
    else:
        raise NotImplementedError(name)
    return criterion

