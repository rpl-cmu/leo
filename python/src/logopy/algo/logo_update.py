import sys
sys.path.append("/usr/local/cython/")

import math
import numpy as np

import os
import json
import hydra
from datetime import datetime
from attrdict import AttrDict
from tqdm import tqdm
import copy
from functools import partial

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

import gtsam

from logopy.optim import cost, gtsamopt, sampler
from logopy.algo.logo_obs_models import *

from logopy.utils import tf_utils, dir_utils, vis_utils
from logopy.eval import quant_metrics

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_dataset(params, idx, dataset_mode="train"):
    
    idx_offset = 0 if (dataset_mode == "train") else params.logo.test_idx_offset

    filename = "{0}/{1}/{2}/{3}/{4:04d}.json".format(
        params.BASE_PATH, params.dataio.srcdir_dataset, params.dataio.dataset_name, dataset_mode, idx+idx_offset)

    dataset = dir_utils.read_file_json(filename, verbose=False)

    return dataset

def optimizer_soln(theta, params, data_idx, dataset_mode="train"):

    # load data
    data = load_dataset(params, data_idx, dataset_mode)
    params.dataio.data_idx = "{0:04d}".format(data_idx) # todo: switch to integer

    cost_obj = cost.Cost(data, theta, params=params, device=device)
    mean, cov = gtsamopt.run_optimizer(cost_obj, params=params)

    # x_samples: n_samples x n_x x dim_x
    x_opt = sampler.sampler_gaussian(mean, cov=None)

    # x_opt: n_x x dim_x
    x_samples = sampler.sampler_gaussian(mean, cov, n_samples=params.logo.n_samples)

    return x_opt, data, x_samples

def cost_optimizer(x_samples, cost_obj, params=None):
    
    # get cost values
    if params.logo.sampler:
        cost_opt = torch.tensor([0.], requires_grad=True, device=device)
        for sidx in range(0, x_samples.shape[0]):
            cost_opt = cost_opt + cost_obj.costfn(x_samples[sidx])
        cost_opt = cost_opt / x_samples.shape[0]
    else: 
        cost_opt = cost_obj.costfn(x_samples)

    return cost_opt

def cost_expert(x_exp, cost_obj, params=None):

    cost_exp = cost_obj.costfn(x_exp)

    return cost_exp

# def cost_debug(x, cost_obj, params=None):
#     cost = cost_obj.costfn_debug(x)
#     return cost

def get_exp_traj_realizable(data, theta_exp, params):

    # expert values x_exp

    cost_obj_exp = cost.Cost(data, theta_exp, params=params, device=device)
    mean, cov = gtsamopt.run_optimizer(cost_obj_exp, params=params)
    x_samples = sampler.sampler_gaussian(mean, cov=None)
    x_exp = torch.tensor(x_samples, requires_grad=True, dtype=torch.float32, device=device)

    return x_exp

def get_exp_traj_groundtruth(data, params):

    # expert values x_exp
    if (params.dataio.dataset_type == "push2d"):
        obj_poses_gt = np.array(data['obj_poses_gt'][0:params.optim.nsteps])
        ee_poses_gt = np.array(data['ee_poses_gt'][0:params.optim.nsteps])
        x_gt = np.vstack((obj_poses_gt, ee_poses_gt))
        x_exp = torch.tensor(x_gt, requires_grad=True, dtype=torch.float32, device=device)
        vis_utils.vis_expert_push2d(obj_poses_gt, ee_poses_gt, params=params)
    elif (params.dataio.dataset_type == "nav2d"):
        x_gt = np.array(data['poses_gt'][0:params.optim.nsteps])
        x_exp = torch.tensor(x_gt, requires_grad=True, dtype=torch.float32, device=device)
    else:
        print("[logo_update::get_exp_traj_groundtruth] x_exp not found for {0}".format(
            params.dataio.dataset_type))

    return x_exp

def optimizer_update(optimizer, output):
    # clear, backprop and apply new gradients
    optimizer.zero_grad()
    output.backward()
    optimizer.step()

def scheduler_update(scheduler):
    scheduler.step()

def eval_learnt_params(theta, theta_exp, params):

    params.optim.save_fig = False

    n_data = params.logo.n_data_test
    traj_err_trans, traj_err_rot = 0., 0.

    for data_idx in range(0, n_data):

        x_opt, data, _ = optimizer_soln(theta, params, data_idx)
        x_opt = torch.tensor(x_opt, requires_grad=True, dtype=torch.float32, device=device)

        x_exp_realizable = get_exp_traj_realizable(data, theta_exp, params)
        # x_exp_groundtruth = get_exp_traj_groundtruth(data, params)
        
        # x_diff = x_exp_groundtruth - x_exp_realizable
        # x_diff[:, 2] = tf_utils.wrap_to_pi(x_diff[:, 2])
        # x_exp = x_exp_realizable + params.logo.realizability_coeff * x_diff
        # x_exp[:, 2] = tf_utils.wrap_to_pi(x_exp[:, 2])

        x_exp = x_exp_realizable

        # traj errors
        err_trans, err_rot = quant_metrics.traj_error(
            xyh_est=x_opt.detach().cpu().numpy(), xyh_gt=x_exp.detach().cpu().numpy())
        traj_err_trans = traj_err_trans + err_trans
        traj_err_rot = traj_err_rot + err_rot

    # avg errors across data points
    traj_err_trans = 1/n_data * traj_err_trans
    traj_err_rot = 1/n_data * traj_err_rot

    return traj_err_trans, traj_err_rot

def run(params):

    # figs = vis_utils.init_plots(n_figs=1, interactive=params.optim.show_fig)
    print("[logo_update::run] Using device: {0}".format(device))
    params.dataio.prefix = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

    if params.random_seed is not None:
        np.random.seed(params.random_seed)
        torch.manual_seed(params.random_seed)

    # save config params
    dir_cfg = "{0}/{1}/{2}/{3}".format(params.BASE_PATH, params.plot.dstdir, params.dataio.dataset_name, params.dataio.prefix)
    dir_utils.make_dir(dir_cfg)
    print(params.pretty())
    with open("{0}/{1}_config.txt".format(dir_cfg, params.dataio.prefix), "w") as f:
        print(params.pretty(), file=f)

    # init tensorboard visualizer
    if params.logo.tb_flag:
        tb_dir = "{0}".format(params.dataio.prefix)
        os.system('mkdir -p {0}/runs/{1}'.format(params.BASE_PATH, tb_dir))
        tb_writer = SummaryWriter("{0}/runs/{1}".format(params.BASE_PATH, tb_dir))

    # for printing cost grad, vals every logo iteration
    if (params.dataio.dataset_type == "push2d"):
        print_list = ["tactile", "qs"]
    elif (params.dataio.dataset_type == "nav2d"):
        print_list = ["odom", "gps"]
    
    # init theta params
    theta, theta_exp = init_theta(params)
    print(" ************** [logo_update::theta_init] ************** ")
    for name, param in theta.named_parameters():
        print('name: ', name)
        print(type(param))
        print('param.shape: ', param.shape)
        print('param.requires_grad: ', param.requires_grad)
        print(param)
        print('=====')

    # init logo update params
    max_iters = params.logo.max_iters
    n_data = params.logo.n_data_train
    
    # logo loss optimizer
    params_optimize = filter(lambda p: p.requires_grad, theta.parameters())
    optimizer = optim.Adam(params_optimize, lr=params.logo.lr, weight_decay=params.logo.lmd)
    if params.logo.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # init multiprocess params
    mp.set_start_method('spawn')

    # init clipper, gradient hook
    for name, param in theta.named_parameters():
        if (param.requires_grad == True):
        # if any(word in name for word in print_list):
            param.register_hook(lambda grad: print("[logo_update::run] GRAD : {}".format(grad)))

    # collect expert trajectories
    x_exp_all = []
    for data_idx in range(0, n_data):
        data = load_dataset(params, data_idx)

        params.logo.itr = 0
        params.dataio.data_idx = "{0:04d}".format(data_idx)
        params.optim.save_fig = False

        x_exp_realizable = get_exp_traj_realizable(data, theta_exp, params)
        # x_exp_groundtruth = get_exp_traj_groundtruth(data, params)

        # x_diff = x_exp_groundtruth - x_exp_realizable
        # x_diff[:, 2] = tf_utils.wrap_to_pi(x_diff[:, 2])
        # x_exp = x_exp_realizable + params.logo.realizability_coeff * x_diff
        # x_exp[:, 2] = tf_utils.wrap_to_pi(x_exp[:, 2])

        x_exp = x_exp_realizable

        x_exp_all.append(x_exp)

    # update theta loop
    itr = 0
    log_dict = {'itr': [], 'loss': [], 'traj_err_trans_train': [
    ], 'traj_err_rot_train': [], 'traj_err_trans_test': [], 'traj_err_rot_test': []}
    cost_opt = torch.tensor([0.], requires_grad=True, device=device)
    cost_exp = torch.tensor([0.], requires_grad=True, device=device)
    while (itr < max_iters):

        traj_err_trans_prev, traj_err_rot_prev = 0.0, 0.0
        traj_err_trans, traj_err_rot = 0.0, 0.0

        loss = torch.tensor([0.], requires_grad=True, device=device)

        # set config params
        params.logo.itr = itr
        params.optim.save_fig = True

        # logger
        if (params.optim.save_logger):
            logger_dir = "{0}/local/logger/{1}/logo_itr_{2:04d}".format(
                params.BASE_PATH, params.dataio.dataset_name, params.logo.itr)
            dir_utils.make_dir(logger_dir)

        # parallelized optimizer run
        pool = mp.Pool(processes=params.logo.pool_processes)
        optimizer_soln_fn = partial(optimizer_soln, copy.deepcopy(theta), params)
        data_idxs = np.arange(0, n_data)
        result_opt = pool.map(optimizer_soln_fn, data_idxs)
        pool.close()
        pool.join()

        for data_idx in range(0, n_data):

            # expert, optim trajs for current data idx
            x_exp = x_exp_all[data_idx]
            x_opt = result_opt[data_idx][0]
            data = result_opt[data_idx][1]
            x_samples = result_opt[data_idx][2]

            x_opt = torch.tensor(x_opt, requires_grad=True,
                                 dtype=torch.float32, device=device)
            x_samples = torch.tensor(x_samples, requires_grad=True,
                                 dtype=torch.float32, device=device)

            # optim, expert costs for current data idx
            cost_obj = cost.Cost(data, theta, params=params, device=device)
            cost_opt = cost_optimizer(x_samples, cost_obj, params=params)
            cost_exp = cost_expert(x_exp, cost_obj, params=None)

            # sum up costs over data idxs
            cost_opt = cost_opt + cost_opt
            cost_exp = cost_exp + cost_exp

            # traj errors
            err_trans, err_rot = quant_metrics.traj_error(
                xyh_est=x_opt[0:params.optim.nsteps, :].detach().cpu().numpy(), xyh_gt=x_exp[0:params.optim.nsteps, :].detach().cpu().numpy())
            traj_err_trans = traj_err_trans + err_trans
            traj_err_rot = traj_err_rot + err_rot

        # logo loss
        loss = 1/n_data * (cost_exp - cost_opt)

        # trajectory errors
        traj_err_trans = 1/n_data * traj_err_trans
        traj_err_rot = 1/n_data * traj_err_rot
        traj_err_trans_test, traj_err_rot_test = eval_learnt_params(theta, theta_exp, params)

        log_dict['itr'].append(itr)
        log_dict['loss'].append(loss.item())
        log_dict['traj_err_trans_train'].append(traj_err_trans)
        log_dict['traj_err_rot_train'].append(traj_err_rot)
        log_dict['traj_err_trans_test'].append(traj_err_trans_test)
        log_dict['traj_err_rot_test'].append(traj_err_rot_test)
        filename = "{0}/{1}/{2}/{3}/{4}_logo_errors.csv".format(
            params.BASE_PATH, params.plot.dstdir, params.dataio.dataset_name, params.dataio.prefix, params.dataio.prefix)
        dir_utils.write_dict_of_lists_to_csv(filename, log_dict)

        if params.logo.tb_flag:
            tb_writer.add_scalar("train/loss", loss.item(), itr)
            tb_writer.add_scalar("traj_errors/trans", traj_err_trans, itr)
            tb_writer.add_scalar("traj_errors/rot", traj_err_rot, itr)
        
        for name, param in theta.named_parameters():
            # if any(word in name for word in print_list):
            if (param.requires_grad == True):
                print("[logo_update::train] iteration {0}/{1} VALUE {2}: {3}".format(itr, max_iters-1, name, param.data))
        print("[logo_update::train] iteration {0}/{1}, loss: {2}, cost_opt: {3}, cost_exp: {4}".format(
            itr, max_iters-1, loss.item(), cost_opt.item(), cost_exp.item()))
        print("[logo_update::train] iteration {0}/{1}, traj_err_trans: {2}, traj_err_rot: {3}".format(
            itr, max_iters-1, traj_err_trans, traj_err_rot))
        print("[logo_update::test] fevals: {0}, traj_err_trans_test: {1}, traj_err_rot_test: {2}".format(
            itr, traj_err_trans_test, traj_err_rot_test))

        # traj convergence criteria
        if (params.logo.use_traj_convergence):
            # if ((traj_err_trans < params.logo.eps_traj_err_trans) & (traj_err_rot < params.logo.eps_traj_err_rot)):
                # break
            diff_traj_err_trans = np.absolute(traj_err_trans - traj_err_trans_prev)
            diff_traj_err_rot = np.absolute(traj_err_rot - traj_err_rot_prev)
            print("[logo_update::train] iteration {0}/{1}, diff_traj_err_trans: {2}, diff_traj_err_rot: {3}".format(
                itr, max_iters-1, diff_traj_err_trans, diff_traj_err_rot))

            if ((diff_traj_err_trans < params.logo.eps_diff_traj_err_trans) & (diff_traj_err_rot < params.logo.eps_diff_traj_err_rot)):
                break

        traj_err_trans_prev, traj_err_rot_prev = traj_err_trans, traj_err_rot

        optimizer_update(optimizer, loss)
        theta.min_clip()

        if params.logo.lr_scheduler:
            scheduler_update(scheduler)

        itr = itr + 1

    # plotting
    if (params.logo.save_video):     
        for idx in range(0, n_data):
            dataset_name_idx = "{0}_{1:04d}".format(params.dataio.dataset_name, idx)
            imgdir = "{0}/{1}/{2}".format(params.BASE_PATH, params.plot.dstdir, dataset_name_idx)
            vis_utils.write_video_ffmpeg(imgdir, "{0}/{1}".format(imgdir, "optimized_isam2"))
