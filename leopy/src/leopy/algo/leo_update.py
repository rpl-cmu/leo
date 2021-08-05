import numpy as np

import os
from datetime import datetime
import copy
from functools import partial
import pandas as pd

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from leopy.optim import cost, gtsamopt, sampler
from leopy.algo.leo_obs_models import *

from leopy.utils import dir_utils, vis_utils
from leopy.eval import quant_metrics

import logging
log = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_dataset(params, idx, dataset_mode="train"):
    
    idx_offset = 0 if (dataset_mode == "train") else params.leo.test_idx_offset

    filename = "{0}/{1}/{2}/{3}/{4:04d}.json".format(
        params.BASE_PATH, params.dataio.srcdir_dataset, params.dataio.dataset_name, dataset_mode, idx+idx_offset)

    dataset = dir_utils.read_file_json(filename, verbose=False)

    return dataset

def eval_learnt_params(theta, theta_exp, params, dataframe):

    params.optim.save_fig = False

    n_data = params.leo.n_data_test
    traj_err_trans_test, traj_err_rot_test = np.zeros((n_data, 1)), np.zeros((n_data, 1))

    for data_idx in range(0, n_data):

        # optimizer trajectory
        x_opt, data, _, _ = optimizer_soln(theta, params, data_idx, dataset_mode="test")
        x_opt = torch.tensor(x_opt, requires_grad=True, dtype=torch.float32, device=device)

        # expert trajectory
        x_exp = get_exp_traj(data, theta_exp, params)

        # traj errors
        traj_err_trans_test[data_idx, :], traj_err_rot_test[data_idx, :] = quant_metrics.traj_error(
            xyh_est=x_opt.detach().cpu().numpy(), xyh_gt=x_exp.detach().cpu().numpy())
        
        dataframe.loc[(data_idx+params.leo.test_idx_offset, params.optim.nsteps-1), 'test/err/tracking/trans'] = traj_err_trans_test[data_idx, :]
        dataframe.loc[(data_idx+params.leo.test_idx_offset, params.optim.nsteps-1), 'test/err/tracking/rot'] = traj_err_rot_test[data_idx, :]

    return traj_err_trans_test, traj_err_rot_test, dataframe

def add_tracking_errors_to_dataframe(df, x_opt, x_exp, params=None):

    nsteps = int(0.5 * x_opt.shape[0]) if (params.dataio.dataset_type == 'push2d') else x_opt.shape[0]

    x_opt_np = x_opt.detach().cpu().numpy()
    x_exp_np = x_exp.detach().cpu().numpy()

    for tstep in range(1, nsteps):
        err_trans, err_rot = quant_metrics.traj_error(xyh_est=x_opt_np[0:tstep, :], xyh_gt=x_exp_np[0:tstep, :])
        df.loc[tstep, 'train/err/tracking/trans'] = err_trans
        df.loc[tstep, 'train/err/tracking/rot'] = err_rot

    return df

def check_traj_convergence(traj_err_trans, traj_err_rot, traj_err_trans_prev, traj_err_rot_prev, params):

    # if ((traj_err_trans < params.leo.eps_traj_err_trans) & (traj_err_rot < params.leo.eps_traj_err_rot)):
        # return True

    diff_traj_err_trans = np.absolute(traj_err_trans - traj_err_trans_prev)
    diff_traj_err_rot = np.absolute(traj_err_rot - traj_err_rot_prev)

    # print("[leo_update::train] iteration {0}/{1}, diff_traj_err_trans: {2}, diff_traj_err_rot: {3}".format(
    #     params.leo.itr, params.leo.max_iters-1, diff_traj_err_trans, diff_traj_err_rot))

    if ((diff_traj_err_trans < params.leo.eps_diff_traj_err_trans) & (diff_traj_err_rot < params.leo.eps_diff_traj_err_rot)):
        return True
    
    return False

def optimizer_soln(theta, params, data_idx, dataset_mode="train"):

    # load data
    data = load_dataset(params, data_idx, dataset_mode)
    params.dataio.data_idx = "{0:04d}".format(data_idx) # todo: switch to integer

    cost_obj = cost.Cost(data, theta, params=params, device=device)
    mean, cov, dataframe = gtsamopt.run_optimizer(cost_obj, params=params)

    # x_opt: n_x x dim_x
    x_opt = sampler.sampler_gaussian(mean, cov=None)

    # x_samples: n_samples x n_x x dim_x    
    x_samples = sampler.sampler_gaussian(mean, cov, n_samples=params.leo.n_samples, temp=params.leo.temp)

    return x_opt, data, x_samples, dataframe

def cost_optimizer(x_samples, cost_obj, params=None):
    
    # get cost values
    if params.leo.sampler:
        cost_opt = torch.tensor([0.], requires_grad=True, device=device)
        for sidx in range(0, x_samples.shape[0]):
            cost_opt = cost_opt + cost_obj.costfn(x_samples[sidx], log=params.logger.cost_flag)
        cost_opt = cost_opt / x_samples.shape[0]
    else: 
        cost_opt = cost_obj.costfn(x_samples, log=params.logger.cost_flag)

    return cost_opt

def cost_expert(x_exp, cost_obj, params=None):

    cost_exp = cost_obj.costfn(x_exp)

    return cost_exp

def get_exp_traj_realizable(data, theta_exp, params):

    # expert values x_exp
    cost_obj_exp = cost.Cost(data, theta_exp, params=params, device=device)
    mean, _, _ = gtsamopt.run_optimizer(cost_obj_exp, params=params)
    x_samples = sampler.sampler_gaussian(mean, cov=None)
    x_exp = torch.tensor(x_samples, requires_grad=True, dtype=torch.float32, device=device)

    return x_exp

def get_exp_traj_groundtruth(data, params):

    # expert values x_exp
    if (params.dataio.dataset_type == "push2d"):
        obj_poses_gt = np.array(data['obj_poses_gt'][0:params.optim.nsteps])
        ee_poses_gt = np.array(data['ee_poses_gt'][0:params.optim.nsteps])
        x_gt = np.vstack((obj_poses_gt, ee_poses_gt))
    elif (params.dataio.dataset_type == "nav2d"):
        x_gt = np.array(data['poses_gt'][0:params.optim.nsteps])
    else:
        print(f"[leo_update::get_exp_traj_groundtruth] x_exp not found for {params.dataio.dataset_type}")
        return

    x_exp = torch.tensor(x_gt, requires_grad=True, dtype=torch.float32, device=device)

    return x_exp

def get_exp_traj(data, theta_exp, params):
    
    x_exp_realizable = get_exp_traj_realizable(data, theta_exp, params)
    x_exp_groundtruth = get_exp_traj_groundtruth(data, params)

    # debug: backward through graph a second time error
    # x_diff = x_exp_realizable - x_exp_groundtruth
    # x_diff[:, 2] = tf_utils.wrap_to_pi(x_diff[:, 2])
    # x_exp = x_exp_groundtruth + params.leo.realizability_coeff * x_diff
    # x_exp[:, 2] = tf_utils.wrap_to_pi(x_exp[:, 2])

    x_exp = x_exp_groundtruth

    return x_exp

def optimizer_update(optimizer, output):
    # clear, backprop and apply new gradients
    optimizer.zero_grad()
    output.backward()
    optimizer.step()

def run(params):

    # figs = vis_utils.init_plots(n_figs=1, interactive=params.optim.show_fig)
    print("[leo_update::run] Using device: {0}".format(device))
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
    if params.leo.tb_flag:
        tb_dir = "{0}".format(params.dataio.prefix)
        os.system('mkdir -p {0}/runs/{1}'.format(params.BASE_PATH, tb_dir))
        tb_writer = SummaryWriter("{0}/runs/{1}".format(params.BASE_PATH, tb_dir))

    # for printing cost grad, vals every leo iteration
    if (params.dataio.dataset_type == "push2d"):
        print_list = ["tactile", "qs"]
    elif (params.dataio.dataset_type == "nav2d"):
        print_list = ["odom", "gps"]
    
    # init theta params
    params = add_theta_exp_to_params(params)
    theta, theta_exp = init_theta(params)
    print(" ************** [leo_update::theta_init] ************** ")
    for name, param in theta.named_parameters():
        print('name: ', name)
        print(type(param))
        print('param.shape: ', param.shape)
        print('param.requires_grad: ', param.requires_grad)
        print(param)
        print('=====')

    # init leo update params
    max_iters = params.leo.max_iters
    n_data = params.leo.n_data_train
    
    # leo loss optimizer
    params_optimize = filter(lambda p: p.requires_grad, theta.parameters())
    optimizer = optim.Adam(params_optimize, lr=params.leo.lr, weight_decay=params.leo.lmd)
    if params.leo.lr_scheduler:
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5, verbose=True)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, verbose=True)

    # init multiprocess params
    mp.set_start_method('spawn')

    # init clipper, gradient hook
    for name, param in theta.named_parameters():
        if (param.requires_grad == True):
        # if any(word in name for word in print_list):
            param.register_hook(lambda grad: print("[leo_update::run] GRAD : {}".format(grad)))

    # collect expert trajectories
    x_exp_all = []
    for data_idx in range(0, n_data):
        data = load_dataset(params, data_idx)

        params.leo.itr = 0
        params.dataio.data_idx = "{0:04d}".format(data_idx)
        params.optim.save_fig = False

        # expert trajectory
        x_exp = get_exp_traj(data, theta_exp, params)

        x_exp_all.append(x_exp)
    
    # main leo loop, update theta
    df_leo_list = []
    itr = 0
    while (itr < max_iters):

        cost_opt = torch.tensor([0.], requires_grad=True, device=device)
        cost_exp = torch.tensor([0.], requires_grad=True, device=device)

        mean_traj_err_trans_prev, mean_traj_err_rot_prev = 0.0, 0.0
        loss = torch.tensor([0.], requires_grad=True, device=device)

        # set config params
        params.leo.itr = itr
        params.optim.save_fig = True

        # logger
        if (params.optim.save_logger):
            logger_dir = "{0}/local/logger/{1}/leo_itr_{2:04d}".format(
                params.BASE_PATH, params.dataio.dataset_name, params.leo.itr)
            dir_utils.make_dir(logger_dir)

        # parallelized optimizer run
        pool = mp.Pool(processes=params.leo.pool_processes)
        optimizer_soln_fn = partial(optimizer_soln, copy.deepcopy(theta), params)
        data_idxs = np.arange(0, n_data)
        result_opt = pool.map(optimizer_soln_fn, data_idxs)
        pool.close()
        pool.join()

        traj_err_trans_train, traj_err_rot_train = np.zeros((n_data, 1)), np.zeros((n_data, 1))
        df_data_list = []
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
            cost_opt_curr = cost_optimizer(x_samples, cost_obj, params=params)
            cost_exp_curr = cost_expert(x_exp, cost_obj, params=None)

            # create a common data frame
            if params.logger.enable:
                df_opt = result_opt[data_idx][3]
                df_cost = cost_obj.get_dataframe()
                df = pd.concat([df_cost, df_opt], axis=1)

                df = add_tracking_errors_to_dataframe(df, x_opt, x_exp, params)
                df = df.assign(leo_loss= (cost_exp - cost_opt).item())
                df = pd.concat({data_idx: df}, names=['data_idx'])
                df_data_list.append(df)

            # sum up costs over data idxs
            cost_opt = cost_opt + cost_opt_curr
            cost_exp = cost_exp + cost_exp_curr

            # traj errors
            traj_err_trans_train[data_idx, :], traj_err_rot_train[data_idx, :] = quant_metrics.traj_error(
                xyh_est=x_opt[0:params.optim.nsteps, :].detach().cpu().numpy(), xyh_gt=x_exp[0:params.optim.nsteps, :].detach().cpu().numpy())
                    
        # leo loss
        loss = 1/n_data * (cost_exp - cost_opt)

        # test trajectory errors
        # traj_err_trans_test, traj_err_rot_test, df = eval_learnt_params(theta, theta_exp, params=params, dataframe=df)

        # concat dataframes across data idxs
        if params.logger.enable:
            df = pd.concat(df_data_list)
            df = pd.concat({itr: df}, names=['leo_itr'])
            df_leo_list.append(df)

        print("[leo_update::train] iteration {0}/{1} VALUE {2}: {3}".format(itr, max_iters-1, name, param.data))

        mean_traj_err_trans, mean_traj_err_rot = np.mean(traj_err_trans_train), np.mean(traj_err_rot_train)
        if params.leo.tb_flag:
            tb_writer.add_scalar("loss", loss.item(), itr)
            tb_writer.add_scalar("err/tracking/trans", mean_traj_err_trans, itr)
            tb_writer.add_scalar("err/tracking/rot", mean_traj_err_rot, itr)
        
        for name, param in theta.named_parameters():
            # if any(word in name for word in print_list):
            if (param.requires_grad == True):
                print("[leo_update::train] iteration {0}/{1} VALUE {2}: {3}".format(itr, max_iters-1, name, param.data))
        print("[leo_update::train] iteration {0}/{1}, loss: {2}, cost_opt: {3}, cost_exp: {4}".format(
            itr, max_iters-1, loss.item(), cost_opt.item(), cost_exp.item()))
        print("[leo_update::train] iteration {0}/{1}, traj_err_trans_train: {2}, traj_err_rot_train: {3}".format(
            itr, max_iters-1, mean_traj_err_trans, mean_traj_err_rot))
        # print("[leo_update::test] fevals: {0}, traj_err_trans_test: ({1}, {2}), traj_err_rot_test: ({3}, {4})".format(
        #     itr, np.mean(traj_err_trans_test), np.std(traj_err_trans_test), np.mean(traj_err_rot_test), np.std(traj_err_rot_test)))
        
        if (params.leo.use_traj_convergence):
            converge_flag = check_traj_convergence(mean_traj_err_trans, mean_traj_err_rot, mean_traj_err_trans_prev, mean_traj_err_rot_prev, params)
            mean_traj_err_trans_prev, mean_traj_err_rot_prev = mean_traj_err_trans, mean_traj_err_rot
            if converge_flag: break

        optimizer_update(optimizer, loss)
        theta.min_clip()

        if params.leo.lr_scheduler:
            scheduler.step()
                
        # write dataframe to file
        if (params.logger.enable) & (params.logger.save_csv):
            dataframe = pd.concat(df_leo_list)
            logdir = f"{params.BASE_PATH}/local/datalogs/{params.dataio.dataset_name}/{params.dataio.prefix}"
            os.makedirs(logdir, exist_ok=True)
            dataframe.to_csv(f"{logdir}/datalog_{params.dataio.prefix}.csv")
            log.info(f"Saved logged data to {logdir}")

        itr = itr + 1

    # plotting
    if (params.leo.save_video):     
        for idx in range(0, n_data):
            dataset_name_idx = "{0}_{1:04d}".format(params.dataio.dataset_name, idx)
            imgdir = "{0}/{1}/{2}".format(params.BASE_PATH, params.plot.dstdir, dataset_name_idx)
            vis_utils.write_video_ffmpeg(imgdir, "{0}/{1}".format(imgdir, "optimized_isam2"))
