
#!/usr/bin/env python

import os
import json
import hydra

import numpy as np
from scipy.optimize import minimize
import cma

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from leopy.optim import cost, gtsamopt, sampler
from leopy.algo.leo_obs_models import *

from leopy.eval import quant_metrics
from leopy.utils import tf_utils, dir_utils, vis_utils

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
CONFIG_PATH = os.path.join(BASE_PATH, "python/config/leo_nav2d.yaml")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FVAL_CALLS = 0
log_dict = {'itr': [], 'traj_err_trans_train': [], 'traj_err_rot_train': [], 'traj_err_trans_test': [], 'traj_err_rot_test': []}

def load_dataset(params, idx, dataset_mode="train"):
    
    idx_offset = 0 if (dataset_mode == "train") else params.leo.n_data_train # params.leo.n_data_train, 30

    filename = "{0}/{1}/{2}/{3}/{4:04d}.json".format(
        params.BASE_PATH, params.dataio.srcdir_dataset, params.dataio.dataset_name, dataset_mode, idx+idx_offset)

    dataset = dir_utils.read_file_json(filename, verbose=False)

    return dataset

def optimizer_soln(theta, params, data_idx, dataset_mode="train"):

    # load data
    data = load_dataset(params, data_idx, dataset_mode)
    params.dataio.data_idx = "{0:04d}".format(data_idx) 

    cost_obj = cost.Cost(data, theta, params=params, device=device)
    mean, cov = gtsamopt.run_optimizer(cost_obj, params=params)

    # x_opt: n_samples x n_x x dim_x (sampler=True), x_opt: n_x x dim_x (sampler=False)
    x_opt = sampler.sampler_gaussian(mean, cov, n_samples=params.leo.n_samples)

    return x_opt, data

def groundtruth_poses(data, params):
    if (params.dataio.dataset_type == "nav2d"):
        x_gt = np.array(data['poses_gt'][0:params.optim.nsteps])
    
    return x_gt

def loss_nav2d(x_opt, x_gt):
    err_trans, err_rot = quant_metrics.traj_error(xyh_est=x_opt, xyh_gt=x_gt)
    loss = err_trans + err_rot
    return loss

def cost_fn_nav2d(theta_vals, params):

    global FVAL_CALLS
    FVAL_CALLS = FVAL_CALLS + 1
    print("[cost_fn_nav2d] FVAL_CALLS: {0}".format(FVAL_CALLS))

    theta = theta_vals_to_obj(theta_vals, params)
    n_data = params.leo.n_data_train

    # parallelized optimizer run
    pool = mp.Pool(processes=params.leo.pool_processes)
    optimizer_soln_fn = partial(optimizer_soln, theta, params)
    data_idxs = np.arange(0, n_data)
    result_opt = pool.map(optimizer_soln_fn, data_idxs)
    pool.close()
    pool.join()

    loss = 0.0
    for data_idx in range(0, n_data):
        # x_opt, data = optimizer_soln(theta, params, data_idx) # serial run
        x_opt, data = result_opt[data_idx][0], result_opt[data_idx][1]
        x_gt = groundtruth_poses(data, params)
        
        loss = loss + loss_nav2d(x_opt, x_gt)

    loss = loss / n_data

    return loss

def traj_error_final(theta_vals, params, dataset_mode="train"):    
    theta = theta_vals_to_obj(theta_vals, params)

    n_data = params.leo.n_data_train if (dataset_mode == "train") else params.leo.n_data_test
    traj_err_trans, traj_err_rot = 0., 0. 

    for data_idx in range(0, n_data):
        x_opt, data = optimizer_soln(theta, params, data_idx, dataset_mode)
        x_gt = groundtruth_poses(data, params)

        err_trans, err_rot = quant_metrics.traj_error(xyh_est=x_opt, xyh_gt=x_gt)
        traj_err_trans = traj_err_trans + err_trans
        traj_err_rot = traj_err_rot + err_rot

    traj_err_trans = 1/n_data * traj_err_trans
    traj_err_rot = 1/n_data * traj_err_rot

    return traj_err_trans, traj_err_rot

def theta_vals_to_obj(theta_vals, params):
    if (params.dataio.dataset_type == "nav2d"):
        if (params.dataio.model_type == "fixed_cov"):
            theta = ThetaNav2dFixedCov(
                sigma_inv_odom_vals=theta_vals[0:3], sigma_inv_gps_vals=theta_vals[3:6])
        elif (params.dataio.model_type == "varying_cov"):
            theta = ThetaNav2dVaryingCov(sigma_inv_odom0_vals=theta_vals[0:3], sigma_inv_gps0_vals=theta_vals[3:6],
                                         sigma_inv_odom1_vals=theta_vals[6:9], sigma_inv_gps1_vals=theta_vals[9:12])
    return theta

def init_theta_vals(params):
    if (params.dataio.dataset_type == "nav2d"):
        if (params.dataio.model_type == "fixed_cov"):
            theta_vals = np.array(params.theta_init.sigma_inv_odom_vals + params.theta_init.sigma_inv_gps_vals)
        elif (params.dataio.model_type == "varying_cov"):
            theta_vals = np.array(params.theta_init.sigma_inv_odom0_vals + params.theta_init.sigma_inv_gps0_vals +
                                  params.theta_init.sigma_inv_odom1_vals + params.theta_init.sigma_inv_gps1_vals)

    return theta_vals

def callback_scipyopt(xk, params):
    global FVAL_CALLS

    theta_vals_curr = xk
    print("theta_vals_curr: {0}".format(theta_vals_curr))

    # train data rmse errors
    traj_err_trans_train, traj_err_rot_train = traj_error_final(theta_vals_curr, params, dataset_mode="train")
    print("[baselines::train] fval calls {0}/{1}, traj_err_trans_train: {2}, traj_err_rot_train: {3}".format(
        FVAL_CALLS, params.baselines.max_fval_calls - 1, traj_err_trans_train, traj_err_rot_train))

    # test data rmse errors
    traj_err_trans_test, traj_err_rot_test = traj_error_final(theta_vals_curr, params, dataset_mode="test")
    print("[baselines::test] fval calls {0}/{1}, traj_err_trans_test: {2}, traj_err_rot_test: {3}".format(
        FVAL_CALLS, params.baselines.max_fval_calls - 1, traj_err_trans_test, traj_err_rot_test))
    
    # log values
    log_dict['itr'].append(FVAL_CALLS)
    log_dict['traj_err_trans_train'].append(traj_err_trans_train)
    log_dict['traj_err_rot_train'].append(traj_err_rot_train)
    log_dict['traj_err_trans_test'].append(traj_err_trans_test)
    log_dict['traj_err_rot_test'].append(traj_err_rot_test)
    filename = "{0}/{1}/{2}/{3}_{4}_errors.csv".format(
        params.BASE_PATH, params.plot.dstdir, params.dataio.dataset_name, params.dataio.prefix, params.baselines.method)
    dir_utils.write_dict_of_lists_to_csv(filename, log_dict)

    # terminaton criteria
    if (FVAL_CALLS > params.baselines.max_fval_calls):
        print("*** Terminating *** \n FVAL_CALLS {0} > params.baselines.max_fval_calls {1}".format(FVAL_CALLS, params.baselines.max_fval_calls))
        assert False
    if (traj_err_trans_train < params.leo.eps_traj_err_trans) & (traj_err_rot_train < params.leo.eps_traj_err_rot):
        print("*** Terminating *** \n traj_err_trans {0} < eps_traj_err_trans {1} & traj_err_rot {2} < eps_traj_err_rot {3} ".format(
            traj_err_trans_train, params.leo.eps_traj_err_trans, traj_err_rot_train, params.leo.eps_traj_err_rot))
        assert False

def run(params):

    params.optim.save_fig = False
    mp.set_start_method('spawn')
    params.dataio.prefix = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

    # initialize cost function
    if (params.dataio.dataset_type == "nav2d"):
        cost_fn = lambda theta : cost_fn_nav2d(theta, params)

    # initialize theta params to be optimized
    theta_vals_init = init_theta_vals(params)

    # call black-box optimizer
    print("Running optimizer {0}".format(params.baselines.method))
    def callback_fn(x): return callback_scipyopt(x, params)

    if (params.baselines.method == "CMAES"):
        xopt, es = cma.fmin2(cost_fn, theta_vals_init, 2., {'maxfevals': params.baselines.max_fval_calls, 'verb_disp': 1, 'bounds': [0.1, 1e6]})
        callback_fn(xopt)
    else:
        result_optimizer = minimize(
            cost_fn, x0=theta_vals_init, method=params.baselines.method, callback=callback_fn)


    # print final errors
    theta_vals_final = result_optimizer.x    
    err_trans, err_rot = traj_error_final(theta_vals_final, params)
    print("theta_final: {0}, err_trans: {1}, err_rot: {2}".format(theta_vals_final, err_trans, err_rot))


@hydra.main(config_name=CONFIG_PATH)
def main(cfg):

    print(cfg.pretty())

    cfg.BASE_PATH = BASE_PATH
    run(cfg)

if __name__ == '__main__':
    main()




