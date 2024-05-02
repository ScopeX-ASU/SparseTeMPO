"""
Date: 2024-03-25 00:24:29
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-03-26 14:23:01
FilePath: /SparseTeMPO/scripts/cifar10/vgg8/train.py
"""

import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

dataset = "cifar10"
model = "vgg8"
root = f"log/{dataset}/{model}/DensePTC"
script = "train.py"
config_file = f"configs/{dataset}/{model}/train/sparse_train.yml"
configs.load(config_file, recursive=True)


def task_launcher(args):
    lr, w_bit, in_bit, conv_block, gpu_id, id= args
    pres = [f"export CUDA_VISIBLE_DEVICES={gpu_id};", "python3", script, config_file]
    with open(
        os.path.join(
            root, f"{model}_{dataset}_lr-{lr:.3f}_wb-{w_bit}_ib-{in_bit}_run-{id}_cb-{conv_block}.log"
        ),
        "w",
    ) as wfid:
        exp = [
            f"--optimizer.lr={lr}",
            f"--run.random_state={42}",
            f"--model.conv_cfg.w_bit={w_bit}",
            f"--model.linear_cfg.w_bit={w_bit}",
            f"--model.conv_cfg.in_bit={in_bit}",
            f"--model.linear_cfg.in_bit={in_bit}",
            f"--model.conv_cfg.miniblock=[{','.join([str(i) for i in conv_block])}]",
            f"--model.linear_cfg.miniblock=[{','.join([str(i) for i in conv_block])}]",
            f"--checkpoint.model_comment=lr-{lr:.4f}_wb-{w_bit}_ib-{in_bit}_cb-[{','.join([str(i) for i in conv_block])}]_run-{id}",
        ]
        cmd = " ".join(pres + exp)
        logger.info(f"running command:\n\t{cmd}")
        subprocess.call(cmd, stderr=wfid, stdout=wfid, shell=True)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        (0.002, 8, 6, [1, 1, 16, 16], 1, 0),
        (0.002, 8, 6, [4, 4, 16, 16], 1, 0),
    ]
    with Pool(2) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
