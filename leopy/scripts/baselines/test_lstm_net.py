#!/usr/bin/env python

import numpy as np
import math
import os

import hydra
import glob

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from lstm_utils import Nav2dFixDataset, Push2dDataset, LSTMPoseSeqNet

import matplotlib.pyplot as plt
# from leopy.eval import quant_metrics
# from leopy.utils import vis_utils

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
CONFIG_PATH = os.path.join(BASE_PATH, "python/config/baselines/lstm_net_test.yaml")

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

def wrap_to_pi(arr):
    arr_wrap = (arr + math.pi) % (2 * math.pi) - math.pi
    return arr_wrap


def init_plots(n_figs=1, figsize=(12, 8), interactive=True):
    if interactive:
        plt.ion()

    plt.close('all')
    figs = []
    for fid in range(0, n_figs):
        figs.append(plt.figure(constrained_layout=True, figsize=figsize))

    return figs

def traj_error(xyh_est, xyh_gt, err_type="rmse"):

    if (err_type == "rmse"):
        diff = xyh_gt - xyh_est
        diff[:, 2] = wrap_to_pi(diff[:, 2])
        diff_sq = diff**2

        rmse_trans = np.sqrt(np.mean(diff_sq[:, 0:2].flatten()))
        rmse_rot = np.sqrt(np.mean(diff_sq[:, 2].flatten()))
        error = (rmse_trans, rmse_rot)

    elif (err_type == "ate"):
        pass

    return error

def test_model(cfg):

    figs = init_plots(n_figs=1, figsize=(8,6))

    # Load model checkpoint
    checkpoint_dir = f"{BASE_PATH}/local/checkpoints/{cfg.checkpoint_dir}"
    checkpoint_file = sorted(glob.glob(f"{checkpoint_dir}/*.ckpt"), reverse=True)[0] # latest ckpt

    model = LSTMPoseSeqNet.load_from_checkpoint(checkpoint_file)
    model.eval()

    # Test data
    dataset_dir = f"{BASE_PATH}/local/datasets/{cfg.test_dataset_dir}"
    dataset_files = sorted(glob.glob(f"{dataset_dir}/*.json"), reverse=False)
    dataset_type = cfg.test_dataset_dir.partition('/')[0] # sim, real

    print(f"Running model checkpoint {checkpoint_file} on dataset {dataset_dir}")

    err_trans_test, err_rot_test = np.zeros((len(dataset_files), 1)), np.zeros((len(dataset_files), 1))
    for ds_idx, dataset_file in enumerate(dataset_files):
      
      if dataset_type == "sim":
        test_dataset = Nav2dFixDataset(dataset_file, 1, device)
      elif dataset_type == "real":
        test_dataset = Push2dDataset(dataset_file, 1, device)

      with torch.no_grad():
        y_pred = model.forward(test_dataset.get_input_sequence().unsqueeze(0))
        y_gt = test_dataset.get_output_sequence().unsqueeze(0)
        test_loss = model.loss(y_pred, y_gt)

      y_pred_np = (y_pred.squeeze(0)).detach().cpu().numpy()
      y_gt_np = (y_gt.squeeze(0)).detach().cpu().numpy()
      # np.set_printoptions(threshold=5000)
      # print(y_pred_np)
      # print(y_gt_np)

      err_trans, err_rot = traj_error(xyh_est=y_pred_np, xyh_gt=y_gt_np)
      err_trans_test[ds_idx, :] = err_trans
      err_rot_test[ds_idx, :] = err_rot

      if cfg.verbose:
        print(f"File: {dataset_file}\n loss: {test_loss}, err_trans: {err_trans}, err_rot: {err_rot}")
        # torch.set_printoptions(threshold=5000)
        # print(y_test_pred)
        # print(y_gt)
      
      if cfg.show_plot:
        plt.cla()

        plt.plot(y_gt_np[:, 0], y_gt_np[:, 1], linewidth=3, linestyle='--', color='tab:grey')
        plt.plot(y_pred_np[:, 0], y_pred_np[:, 1], linewidth=3, linestyle='-', color='#d95f02', alpha=0.8)

        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.axis('off')

        plt.show()
        plt.pause(1e-1)

    print(f"*** Final error statistics for dataset {cfg.test_dataset_dir} ***")
    print(f"err_trans_mean: {np.mean(err_trans_test)}, err_trans_stdev: {np.std(err_trans_test)}")
    print(f"err_rot_mean: {np.mean(err_rot_test)}, err_rot_stdev: {np.std(err_rot_test)}")

@hydra.main(config_path=CONFIG_PATH)
def main(cfg):

    test_model(cfg)

if __name__ == '__main__':
    main()
