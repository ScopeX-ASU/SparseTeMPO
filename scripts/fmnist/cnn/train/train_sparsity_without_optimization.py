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
root = f"log/{dataset}/{model}/train_sparsity_exploration_crosstalk_power_optimization/row_only_interleave_nochange"
script = "sparse_train.py"
# file_id = "sparse_train_64_4_[4, 4, 8, 8]row"
config_file = f"configs/{dataset}/{model}/train/sparse_train.yml"
configs.load(config_file, recursive=True)
keep_same = True

def task_launcher(args):
    lr, density, w_bit, in_bit, death_mode, growth_mode, init_mode, conv_block, row_col, crosstalk, redist, input_gate, output_gate, interv_s, interv_h, id, gpu_id  = args
    pres = [
        f"export CUDA_VISIBLE_DEVICES={gpu_id};",
        "python3", script, config_file]
    with open(
        os.path.join(
            root, f"{model}_{dataset}_lr-{lr:.3f}_wb-{w_bit}_ib-{in_bit}_dm-{death_mode}_gm-{growth_mode}_im-{init_mode}_cb-{conv_block}_{row_col}-without-opt_den-{density}_ls-{interv_s}_lh-{interv_h}_run-{id}.log"
        ),
        "w",
    ) as wfid:
        exp = [
            f"--optimizer.lr={lr}",
            f"--run.random_state={42}",
            f"--model.conv_cfg.w_bit={w_bit}",
            f"--model.linear_cfg.w_bit={w_bit}",
            f"--dst_scheduler.death_mode={death_mode}",
            f"--dst_scheduler.growth_mode={growth_mode}",
            f"--dst_scheduler.init_mode={init_mode}",
            f"--dst_scheduler.density={density}",
            f"--dst_scheduler.keep_same={keep_same}",
            f"--dst_scheduler.pruning_type={row_col}",
            f"--noise.crosstalk_scheduler.interv_s={interv_s}",
            f"--noise.crosstalk_scheduler.interv_h={interv_h}",
            f"--noise.noise_flag={crosstalk}",
            f"--noise.crosstalk_flag={crosstalk}",
            f"--model.conv_cfg.miniblock=[{','.join([str(i) for i in conv_block])}]",
            f"--model.linear_cfg.miniblock=[{','.join([str(i) for i in conv_block])}]",
            f"--noise.light_redist={redist}",
            f"--noise.input_power_gating={input_gate}",
            f"--noise.output_power_gating={output_gate}",
            f"--model.conv_cfg.in_bit={in_bit}",
            f"--model.linear_cfg.in_bit={in_bit}",
            f"--checkpoint.model_comment={row_col}-only-without-opt_lr-{lr:.4f}_wb-{w_bit}_ib-{in_bit}_dm-{death_mode}_gm-{growth_mode}_im-{init_mode}_cb-[{','.join([str(i) for i in conv_block])}]_density-{density}_ls-{interv_s}_lh-{interv_h}_run-{id}",
        ]
        cmd = " ".join(pres + exp)
        logger.info(f"running command:\n\t{cmd}")
        # subprocess.call(pres + exp, stderr=wfid, stdout=wfid, shell=True)
        subprocess.call(cmd, stderr=wfid, stdout=wfid, shell=True)
     


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        (0.002, 0.4, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "structure_row",  0, 1, 1, 1, 9, 120, 4, 1),
        # (0.002, 0.6, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "structure_row_col",  0, 9, 120, 4, 1),
        # (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "structure_row_col",  0, 9, 120, 4, 1),
        # (0.002, 0.4, 8, 6, "magnitude", "gradient", "uniform", [2, 2, 16, 16], "structure_row_col",  0, 9, 120, 4, 1),
        # (0.002, 0.6, 8, 6, "magnitude", "gradient", "uniform", [2, 2, 16, 16], "structure_row_col",  0, 9, 120, 4, 1),
        # (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [2, 2, 16, 16], "structure_row_col",  0, 9, 120, 4, 1),
        # (0.002, 0.4, 8, 6, "magnitude", "gradient", "uniform", [4, 4, 16, 16], "structure_row_col",  0, 9, 120, 4, 1),
        # (0.002, 0.6, 8, 6, "magnitude", "gradient", "uniform", [4, 4, 16, 16], "structure_row_col",  0, 9, 120, 4, 1),
        # (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [4, 4, 16, 16], "structure_row_col",  0, 9, 120, 4, 1),
        # (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "structure_row",  0, 10, 120, 4, 2),
        # (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "structure_row",  0, 10, 120, 4, 2),
        # (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "structure_row",  0, 10, 120, 4, 2),
        # (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "structure_row",  0, 10, 120, 4, 2),
        # for gpu_id, s in enumerate(range(7, 16, 2)) for h_s in range(1, 8, 2)
        # (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "structure_row",  0, 7, 22, 4, 1),
        # (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "structure_row",  0, 15, 24, 4, 1),
        # (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "structure_row",  0, 15, 26, 4, 1),
        # (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "structure_row",  0, 15, 28, 4, 1),
        # (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "structure_row",  0, 11, 20, 4, 1),
        # (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "structure_row",  0, 7, 16, 4, 1),
        # (0.002, 0.6, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "structure_row",  0, 13, 24, 4, 1),
        # (0.002, 0.6, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "structure_row",  0, 9, 18, 4, 1),


    ]
    with Pool(9) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
