"""
Date: 2024-03-26 14:23:47
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-03-26 14:23:47
FilePath: /SparseTeMPO/scripts/fmnist/cnn/train/train.py
"""

import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

dataset = "fmnist"
model = "cnn"
root = f"log/{dataset}/{model}"
script = "train.py"
file_id = "sparse_train_64_4_[4, 4, 8, 8]row"
config_file = f"configs/{dataset}/{model}/train/{file_id}.yml"
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
            f"--run.random_state={42}",
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
        (1, 8, 8, 3),
    ]
    with Pool(4) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
