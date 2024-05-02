'''
Date: 2024-04-27 23:07:24
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-04-27 23:07:32
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
root = f"log/{dataset}/{model}/test_structural_pruning_without_optimization"
script = "crosstalk_spacing.py"
config_file = f"configs/{dataset}/{model}/train/sparse_train.yml"
configs.load(config_file, recursive=True)


def task_launcher(args):
    lr, density, w_bit, in_bit, death_mode, growth_mode, init_mode, conv_block, row_col, crosstalk, interv_s, interv_h, redist, input_gate, output_gate, out_noise_std, ckpt, id, gpu_id = args
    pres = [f"export CUDA_VISIBLE_DEVICES={gpu_id};", "python3", script, config_file]
    with open(
        os.path.join(
            root,
            f"{model}_{dataset}_lr-{lr:.3f}_wb-{w_bit}_ib-{in_bit}_dm-{death_mode}_gm-{growth_mode}_im-{init_mode}_cb-{conv_block}_{row_col}-without-opt_den-{density}_ls-{interv_s}_lh-{interv_h}_rd-{int(redist)}_ig-{int(input_gate)}_og-{int(output_gate)}_run-{id}.log",
        ),
        "w",
    ) as wfid:
        exp = [
            f"--run.batch_size=200",
            f"--optimizer.lr={lr}",
            f"--run.random_state={42}",
            f"--model.conv_cfg.w_bit={w_bit}",
            f"--model.linear_cfg.w_bit={w_bit}",
            f"--dst_scheduler.death_mode={death_mode}",
            f"--dst_scheduler.growth_mode={growth_mode}",
            f"--dst_scheduler.init_mode={init_mode}",
            f"--dst_scheduler.density={density}",
            f"--dst_scheduler.pruning_type={row_col}",
            # f"--noise.crosstalk_scheduler.interv_s={interv_s}",
            # f"--noise.crosstalk_scheduler.interv_h={interv_h}",
            f"--noise.noise_flag={crosstalk}",
            f"--noise.crosstalk_flag={crosstalk}",
            f"--noise.output_noise_std={out_noise_std}",
            f"--loginfo=rd-{int(redist)}_ig-{int(input_gate)}_og-{int(output_gate)}_id-{id}",
            f"--model.conv_cfg.miniblock=[{','.join([str(i) for i in conv_block])}]",
            f"--model.linear_cfg.miniblock=[{','.join([str(i) for i in conv_block])}]",
            f"--model.conv_cfg.in_bit={in_bit}",
            f"--model.linear_cfg.in_bit={in_bit}",
            f"--noise.light_redist={redist}",
            f"--noise.input_power_gating={input_gate}",
            f"--noise.output_power_gating={output_gate}",
            f"--checkpoint.resume={True}",
            f"--checkpoint.restore_checkpoint={ckpt}",
            # f"--checkpoint.restore_checkpoint=./checkpoint/fmnist/cnn/train/TeMPO_CNN_structure_col-only-without-opt_lr-0.0020_wb-8_ib-6_dm-magnitude_gm-gradient_im-uniform_cb-[1,1,16,16]_density-0.4_ls-10_lh-120_run-4_acc-91.85_epoch-39.pt",
            f"--checkpoint.model_comment=pretrain_lr-{lr:.4f}_wb-{w_bit}_ib-{in_bit}_cb-[{','.join([str(i) for i in conv_block])}]_run-{id}",
        ]
        cmd = " ".join(pres + exp)
        logger.info(f"running command:\n\t{cmd}")
        # subprocess.call(pres + exp, stderr=wfid, stdout=wfid, shell=True)
        subprocess.call(cmd, stderr=wfid, stdout=wfid, shell=True)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    # TeMPO_CNN_structure_row-only-without-opt_lr-0.0020_wb-8_ib-6_dm-magnitude_gm-gradient_im-uniform_cb-[1,1,16,16]_density-0.6_ls-10_lh-120_run-4_acc-92.10_epoch-30.pt
    # TeMPO_CNN_structure_row-only-without-opt_lr-0.0020_wb-8_ib-6_dm-magnitude_gm-gradient_im-uniform_cb-[1,1,16,16]_density-0.6_ls-10_lh-120_run-4_acc-92.07_epoch-24.pt
    tasks = [
        # (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "structure_row", 0, 7, 14, 0, 0, 0, 0.01, "./checkpoint/fmnist/cnn/train/TeMPO_CNN_structure_row-only-without-opt_lr-0.0020_wb-8_ib-6_dm-magnitude_gm-gradient_im-uniform_cb-[1,1,16,16]_density-0.6_ls-10_lh-120_run-4_acc-92.10_epoch-30.pt", 6, 0), 
        # (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "structure_row", 0, 7, 14, 1, 1, 1, 0.01, "./checkpoint/fmnist/cnn/train/TeMPO_CNN_structure_row-only-without-opt_lr-0.0020_wb-8_ib-6_dm-magnitude_gm-gradient_im-uniform_cb-[1,1,16,16]_density-0.6_ls-10_lh-120_run-4_acc-92.10_epoch-30.pt", 6, 0),
        # (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "structure_row", 0, 7, 14, 1, 1, 1, 0.01, "./checkpoint/fmnist/cnn/train/TeMPO_CNN_structure_row-only-without-opt_lr-0.0020_wb-8_ib-6_dm-magnitude_gm-gradient_im-uniform_cb-[1,1,16,16]_density-0.6_ls-10_lh-120_run-4_acc-92.07_epoch-24.pt", 6, 0), 
        # (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "structure_row", 0, 7, 14, 1, 1, 1, 0.01, "./checkpoint/fmnist/cnn/train/TeMPO_CNN_structure_row-only-without-opt_lr-0.0020_wb-8_ib-6_dm-magnitude_gm-gradient_im-uniform_cb-[1,1,16,16]_density-0.6_ls-10_lh-120_run-4_acc-92.07_epoch-24.pt", 6, 0),
        (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "structure_row", 0, 7, 14, 0, 0, 0, 0.01, "./checkpoint/fmnist/cnn/train/TeMPO_CNN_structure_row-only-without-opt_lr-0.0020_wb-8_ib-6_dm-magnitude_gm-gradient_im-uniform_cb-[1,1,16,16]_density-0.4_ls-10_lh-120_run-4_acc-91.97_epoch-40.pt", 6, 0),
        (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "structure_row", 0, 7, 14, 1, 1, 1, 0.01, "./checkpoint/fmnist/cnn/train/TeMPO_CNN_structure_row-only-without-opt_lr-0.0020_wb-8_ib-6_dm-magnitude_gm-gradient_im-uniform_cb-[1,1,16,16]_density-0.4_ls-10_lh-120_run-4_acc-91.97_epoch-40.pt", 6, 0),  
        # (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "structure_row", 0, 7, 16, 2, 0), 
        # (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "structure_row", 0, 7, 18, 92.00, 26, 2, 0), 
        # (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "structure_row", 0, 7, 20, 91.90, 43, 2, 0),
        # (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "structure_row", 0, 7, 20, 91.90, 43, 2, 0), 
        # (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "structure_row", 0, 7, 20, 91.90, 43, 2, 0), 
        # (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "structure_row", 0, 7, 20, 91.90, 43, 2, 0), 

        # (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], 0, 7, 14, 2, 0), 
        # (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], 0, 7, 14, 2, 0), 
        # (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], 0, 7, 14, 2, 0), 
        # (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], 0, 7, 14, 2, 0), 
        # (0.002, 0.8, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], 0, 7, 14, 2, 0), 


    ]
    with Pool(4) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
