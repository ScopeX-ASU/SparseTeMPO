"""
Date: 2024-03-24 20:20:29
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-03-24 20:20:29
FilePath: /SparseTeMPO/script/cifar10/resnet18/train_trans.py
"""

import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

dataset = "cifar100"
model = "resnet18"
root = f"log/{dataset}/{model}"
script = "train.py"
config_file = f"configs/{dataset}/{model}/train/train.yml"
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ["python3", script, config_file]
    lr, id = args
    with open(
        os.path.join(root, f"{model}_{dataset}_lr-{lr:.3f}_run-{id}.log"), "w"
    ) as wfid:
        exp = [
            f"--optimizer.lr={lr}",
            f"--run.random_state={41+id}",
            f"--checkpoint.model_comment=lr-{lr:.4f}",
        ]
        cmd = " ".join(pres + exp)
        logger.info(f"running command:\n\t{cmd}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        (1, 1),
    ]
    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
