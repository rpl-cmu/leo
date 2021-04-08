#!/usr/bin/env python

import math
import numpy as np

import torch

from logopy.utils import tf_utils, dir_utils, vis_utils

def wrap_to_pi(arr):
    arr_wrap = (arr + math.pi) % (2 * math.pi) - math.pi
    return arr_wrap


# def traj_loss(x_opt, x_gt, device=None, return_tensor=True):
#     # x_gt: n x 3, x_opt: n x 3
#     x_gt = x_gt if torch.is_tensor(x_opt) else torch.tensor(x_gt, device=device)
#     x_opt = x_opt if torch.is_tensor(x_opt) else torch.tensor(x_opt, device=device)
#     x_rel = tf_utils.tf2d_between(x_gt, x_opt, device=device)
#     loss = torch.linalg.norm(x_rel, dim=0)
#     loss = loss if return_tensor else loss.detach().cpu().numpy()

#     return loss

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

def get_traj_error_step(params, tstep, logger):
    if (params.dataio.dataset_type == "nav2d"):
        poses_obj_gt = logger.data[tstep]["gt/poses2d"]
        poses_obj_graph = logger.data[tstep]["graph/poses2d"]
    elif (params.dataio.dataset_type == "push2d"):
        poses_obj_gt = logger.data[tstep]["gt/obj_poses2d"]
        poses_obj_graph = logger.data[tstep]["graph/obj_poses2d"]
    else:
        print("[quant_metrics::compute_traj_error_step] logger poses not found")

    error = traj_error(poses_obj_gt, poses_obj_graph, err_type="rmse")

    return error
