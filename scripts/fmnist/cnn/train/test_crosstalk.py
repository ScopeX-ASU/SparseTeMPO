'''
Date: 2024-04-27 23:07:24
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-04-29 14:07:34
FilePath: /SparseTeMPO/scripts/fmnist/cnn/train/test_crosstalk.py
'''
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
    lr, w_bit, in_bit, death_mode, growth_mode, init_mode, conv_block, ckpt, id, gpu_id = args
    pres = [f"export CUDA_VISIBLE_DEVICES={gpu_id};", "python3", script, config_file]
    with open(
        os.path.join(
            root,
            f"{model}_{dataset}_lr-{lr:.3f}_wb-{w_bit}_ib-{in_bit}_cb-{conv_block}_run-{id}.log",
        ),
        "w",
    ) as wfid:
        exp = [
            f"--run.batch_size=200",
            f"--optimizer.lr={lr}",
            f"--run.random_state={42}",
            f"--model.conv_cfg.w_bit={w_bit}",
            f"--model.linear_cfg.w_bit={w_bit}",
            f"--model.dst_scheduler={None}",
            f"--dst_scheduler={None}",
            f"--model.conv_cfg.miniblock=[{','.join([str(i) for i in conv_block])}]",
            f"--model.linear_cfg.miniblock=[{','.join([str(i) for i in conv_block])}]",
            f"--model.conv_cfg.in_bit={in_bit}",
            f"--model.linear_cfg.in_bit={in_bit}",
            f"--checkpoint.resume={True}",
            f"--checkpoint.restore_checkpoint={ckpt}",
            f"--noise.output_noise_std=0.01",
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
        (0.002, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "./checkpoint/fmnist/cnn/train/TeMPO_CNN_pretrain_lr-0.0020_wb-8_ib-6_cb-[1,1,16,16]_crstlk-0_run-4_acc-92.15_epoch-18.pt", 4, 3),  # adam
    ]
    with Pool(10) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
