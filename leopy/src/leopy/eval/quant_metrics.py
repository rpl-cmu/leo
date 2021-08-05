#!/usr/bin/env python

import math
import numpy as np

def wrap_to_pi(arr):
    arr_wrap = (arr + math.pi) % (2 * math.pi) - math.pi
    return arr_wrap

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
