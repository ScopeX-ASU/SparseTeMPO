'''
Date: 2024-03-27 22:40:49
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-04-04 13:17:18
FilePath: /SparseTeMPO/scripts/cifar10/resnet20/sparse_train.py
'''
"""
Date: 2024-03-27 22:39:38
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-03-27 22:39:38
FilePath: /SparseTeMPO/scripts/cifar10/resnet20/train.py
"""

import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

dataset = "cifar10"
model = "resnet20"
experiment = "sparse_train"
root = f"log/{dataset}/{model}/{experiment}"
script = "sparse_train.py"
config_file = f"configs/{dataset}/{model}/train/sparse_train.yml"
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ["python3", script, config_file]
    lr, w_bit, in_bit, id = args
    with open(
        os.path.join(
            root, f"{model}_{dataset}_lr-{lr:.3f}_wb-{w_bit}_ib-{in_bit}_run-{id}.log"
        ),
        "w",
    ) as wfid:
        exp = [
            f"--optimizer.lr={lr}",
            f"--run.random_state={41+id}",
            f"--model.conv_cfg.w_bit={w_bit}",
            f"--model.linear_cfg.w_bit={w_bit}",
            f"--model.conv_cfg.in_bit={in_bit}",
            f"--model.linear_cfg.in_bit={in_bit}",
            f"--checkpoint.model_comment=lr-{lr:.4f}",
        ]
        cmd = " ".join(pres + exp)
        logger.info(f"running command:\n\t{cmd}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        # (1, 8, 8, 1),
        (1, 8, 8, 3),
    ]
    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
