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
root = f"log/{dataset}/{model}/DensePTC"
script = "sparse_train.py"
# file_id = "sparse_train_64_4_[4, 4, 8, 8]row"
config_file = f"configs/{dataset}/{model}/train/train.yml"
configs.load(config_file, recursive=True)


def task_launcher(args):
    lr, density, w_bit, in_bit, conv_block, id, gpu_id = args
    pres = [
        f"export CUDA_VISIBLE_DEVICES={gpu_id};",
        "python3", script, config_file]
    with open(
        os.path.join(
            root, f"{model}_{dataset}_lr-{lr:.3f}_wb-{w_bit}_ib-{in_bit}_cb-{conv_block}_den-{density}_run-{id}.log"
        ),
        "w",
    ) as wfid:
        exp = [
            f"--optimizer.lr={lr}",
            f"--run.random_state={42}",
            f"--model.conv_cfg.w_bit={w_bit}",
            f"--model.linear_cfg.w_bit={w_bit}",
            # f"--model.dst_scheduler.death_mode={death_mode}",
            # f"--model.dst_scheduler.growth_mode={growth_mode}",
            # f"--model.dst_scheduler.init_mode={init_mode}",
            # f"--model.dst_scheduler.density={density}",
            f"--model.conv_cfg.miniblock=[{','.join([str(i) for i in conv_block])}]",
            f"--model.linear_cfg.miniblock=[{','.join([str(i) for i in conv_block])}]",
            f"--model.conv_cfg.in_bit={in_bit}",
            f"--model.linear_cfg.in_bit={in_bit}",
            f"--checkpoint.model_comment=lr-{lr:.4f}_wb-{w_bit}_ib-{in_bit}_cb-[{','.join([str(i) for i in conv_block])}]_run-{id}",
        ]
        cmd = " ".join(pres + exp)
        logger.info(f"running command:\n\t{cmd}")
        # subprocess.call(pres + exp, stderr=wfid, stdout=wfid, shell=True)
        subprocess.call(cmd, stderr=wfid, stdout=wfid, shell=True)
     


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [

        #(0.002, 1, 8, 6, [1, 1, 16, 16], 1, 2),
        (0.002, 1, 8, 6, [4, 4, 16, 16], 1, 2),
        # (0.002, 0.5, 8, 8, "magnitude_power", "gradient_power", "uniform_power", [2, 2, 4, 4], 2, 1),
        # (0.002, 0.5, 8, 8, "magnitude_crosstalk", "gradient_crosstalk", "uniform_crosstalk", [2, 2, 4, 4], 3, 1),
        # (0.002, 0.5, 8, 8, "magnitude_power_crosstalk", "gradient_power_crosstalk", "uniform_power_crosstalk", [2, 2, 4, 4], 4, 1),
        # (0.002, 0.5, 8, 8, "magnitude_crosstalk_power", "gradient_crosstalk_power", "uniform_crosstalk_power", [2, 2, 4, 4], 5, 1),
        # (0.002, 0.5, 8, 8, "magnitude", "gradient", "uniform", [2, 2, 8, 8], 1, 2),
        # (0.002, 0.5, 8, 8, "magnitude_power", "gradient_power", "uniform_power", [2, 2, 8, 8], 2, 2),
        # (0.002, 0.5, 8, 8, "magnitude_crosstalk", "gradient_crosstalk", "uniform_crosstalk", [2, 2, 8, 8], 3, 2),
        # (0.002, 0.5, 8, 8, "magnitude_power_crosstalk", "gradient_power_crosstalk", "uniform_power_crosstalk", [2, 2, 8, 8], 4, 2),
        # (0.002, 0.5, 8, 8, "magnitude_crosstalk_power", "gradient_crosstalk_power", "uniform_crosstalk_power", [2, 2, 8, 8], 5, 2),
    ]
    with Pool(10) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
