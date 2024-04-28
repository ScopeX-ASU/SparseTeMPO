"""
Date: 2024-04-27 15:21:22
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-04-27 15:27:05
FilePath: /SparseTeMPO/scripts/fmnist/cnn/train/test_crosstalk.py
"""

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
root = f"log/{dataset}/{model}/crosstalk_spacing"
script = "crosstalk_spacing.py"
config_file = f"configs/{dataset}/{model}/train/sparse_train.yml"
configs.load(config_file, recursive=True)


def task_launcher(args):
    lr, w_bit, in_bit, death_mode, growth_mode, init_mode, conv_block, id, gpu_id = args
    pres = [f"export CUDA_VISIBLE_DEVICES={gpu_id};", "python3", script, config_file]
    with open(
        os.path.join(
            root,
            f"{model}_{dataset}_lr-{lr:.3f}_wb-{w_bit}_ib-{in_bit}_cb-{conv_block}_run-{id}.log",
        ),
        "w",
    ) as wfid:
        exp = [
            f"--optimizer.lr={lr}",
            f"--run.random_state={42}",
            f"--model.conv_cfg.w_bit={w_bit}",
            f"--model.linear_cfg.w_bit={w_bit}",
            f"--model.dst_scheduler={None}",
            f"--dst_scheduler={None}",
            f"--model.conv_cfg.miniblock=[{','.join([str(i) for i in conv_block])}]",
            f"--model.conv_cfg.in_bit={in_bit}",
            f"--model.linear_cfg.in_bit={in_bit}",
            f"--checkpoint.resume={True}",
            f"--checkpoint.restore_checkpoint=./checkpoint/fmnist/cnn/train/TeMPO_CNN_pretrain_lr-0.0020_wb-8_ib-8_cb-[8,8,8,8]_run-2_acc-92.00_epoch-18.pt",
            f"--checkpoint.model_comment=pretrain_lr-{lr:.4f}_wb-{w_bit}_ib-{in_bit}_cb-[{','.join([str(i) for i in conv_block])}]_run-{id}",
        ]
        cmd = " ".join(pres + exp)
        logger.info(f"running command:\n\t{cmd}")
        # subprocess.call(pres + exp, stderr=wfid, stdout=wfid, shell=True)
        subprocess.call(cmd, stderr=wfid, stdout=wfid, shell=True)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        (0.002, 8, 8, "magnitude", "gradient", "uniform", [8, 8, 8, 8], 1, 1),  # adam
    ]
    with Pool(10) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
