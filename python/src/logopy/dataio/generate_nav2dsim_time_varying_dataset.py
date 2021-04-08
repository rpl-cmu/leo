#!/usr/bin/env python

import sys
sys.path.append("/usr/local/cython/")

import numpy as np
import math

import os
import hydra
import json
import csv
from attrdict import AttrDict
from datetime import datetime

import gtsam
from logopy.utils import tf_utils, dir_utils

import matplotlib.pyplot as plt

BASE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../.."))
CONFIG_PATH = os.path.join(BASE_PATH, "python/config/dataio/nav2d.yaml")

def get_waypoints_gui(params, poses=None):

    class MouseEvents:
        def __init__(self, fig, line):
            self.path_start = False  # if true, capture data
            self.fig = fig
            self.line = line
            self.xs = list(line.get_xdata())
            self.ys = list(line.get_ydata())
            self.orientation = []

        def connect(self):
            self.a = self.fig.canvas.mpl_connect(
                'button_press_event', self.on_press)
            self.b = self.fig.canvas.mpl_connect(
                'motion_notify_event', self.on_motion)

        def on_press(self, event):
            print('Pressed', event.button, event.xdata, event.ydata)
            self.path_start = not self.path_start

        def on_motion(self, event):
            if self.path_start is True:
                if len(self.orientation) == 0:
                    self.orientation.append(0)
                else:
                    self.orientation.append(
                        np.pi/2 + np.arctan2((self.ys[-1] - event.ydata), (self.xs[-1] - event.xdata)))
                self.xs.append(event.xdata)
                self.ys.append(event.ydata)
                self.line.set_data(self.xs, self.ys)
                self.line.figure.canvas.draw()

    plt.ioff()
    plt.close('all')
    fig = plt.figure(figsize=(12, 8))
    plt.title(
        "Generate waypoints for nav2d/{0:04d}/{1:04d}.json dataset: \n Click and move pointer to draw trajectory. Close window once finished.".format(params.dataio.ds_idx, params.dataio.seq_idx))
    if poses is not None:
        plt.plot(poses[:, 0], poses[:, 1], 'o--', c='m')

    plt.xlim(params.env.area.xmin, params.env.area.xmax)
    plt.ylim(params.env.area.ymin, params.env.area.ymax)

    line, = plt.plot([], [])
    mouse = MouseEvents(fig, line)
    mouse.connect()

    plt.show()
    
    return np.hstack((np.array(mouse.xs)[:, None], np.array(mouse.ys)[:, None], np.array(mouse.orientation)[:, None]))[1:]

def plot_data(params, logger, plot_ori=False):
    plt.ion()
    plt.close('all')
    fig = plt.figure(figsize=(12, 8))

    poses_gt = np.asarray(logger.poses_gt)
    meas_odom = np.asarray(logger.meas_odom)
    meas_gps = np.asarray(logger.meas_gps)

    num_steps = params.num_steps
    poses_odom = np.zeros((num_steps, 3))
    poses_gps = np.zeros((num_steps, 3))
    poses_odom[0, :] = poses_gt[0, :]

    # compute poses
    for tstep in range(0, num_steps):

        if (tstep > 0):     
            poses_odom[tstep] = tf_utils.pose2_to_vec3(tf_utils.vec3_to_pose2(
                poses_odom[tstep-1, :]).compose(tf_utils.vec3_to_pose2(meas_odom[tstep-1, :])))

        poses_gps[tstep, :] = meas_gps[tstep, :]
    
    # plot poses
    for tstep in range(num_steps-1, num_steps):

        plt.cla()
        plt.xlim(params.env.area.xmin, params.env.area.xmax)
        plt.ylim(params.env.area.ymin, params.env.area.ymax)

        plt.scatter([0], [0], marker='*', c='k', s=20,
                    alpha=1.0, zorder=3, edgecolor='k')
        plt.scatter(poses_gt[tstep, 0], poses_gt[tstep, 1], marker=(3, 0, poses_gt[tstep, 2]/np.pi*180),
                    color='dimgray', s=300, alpha=0.25, zorder=3, edgecolor='dimgray')
        
        plt.plot(poses_gt[0:tstep, 0], poses_gt[0:tstep, 1], color=params.plot.colors[0], linewidth=2, label="groundtruth")
        plt.plot(poses_odom[0:tstep, 0], poses_odom[0:tstep, 1], color=params.plot.colors[1], linewidth=2, label="odom")
        plt.plot(poses_gps[0:tstep, 0], poses_gps[0:tstep, 1], color=params.plot.colors[2], linewidth=2, label="gps")

        # if plot_ori:
        #     ori = poses_gt[:, 2]
        #     sz_arw = 0.03
        #     (dx, dy) = (sz_arw * np.cos(ori), sz_arw * np.sin(ori))
        #     for i in range(0, num_steps):
        #         plt.arrow(poses_gt[i, 0], poses_gt[i, 1], dx[i], dy[i], linewidth=4,
        #                 head_width=0.01, color='black', head_length=0.1, fc='black', ec='black')

        plt.title("Logged dataset nav2d/{0:04d}/{1:04d}.json".format(params.dataio.ds_idx, params.dataio.seq_idx))
        plt.legend(loc='upper right')

        plt.show()
        plt.pause(1)

# def covariance_type(tfrac):

#     cov_type = None

#     if tfrac <= 0.25:
#         cov_type = 0
#     elif (tfrac > 0.25) & (tfrac <= 0.5):
#         cov_type = 1
#     elif (tfrac > 0.5) & (tfrac <= 0.75):
#         cov_type = 0
#     elif (tfrac > 0.75):
#         cov_type = 1

#     return cov_type

def covariance_type(tfrac):

    cov_type = None

    if tfrac <= 0.25:
        cov_type = 0
    elif (tfrac > 0.25) & (tfrac <= 0.75):
        cov_type = 1
    elif (tfrac >= 0.75):
        cov_type = 0

    return cov_type

def create_measurements(params, poses, covariances):
    
    # noise models
    odom_noise0 = gtsam.noiseModel_Diagonal.Sigmas(covariances.odom0)
    odom_noise1 = gtsam.noiseModel_Diagonal.Sigmas(covariances.odom1)
    gps_noise0 = gtsam.noiseModel_Diagonal.Sigmas(covariances.gps0)
    gps_noise1 = gtsam.noiseModel_Diagonal.Sigmas(covariances.gps1)

    # samplers
    sampler_odom_noise0 = gtsam.Sampler(odom_noise0, 0)
    sampler_odom_noise1 = gtsam.Sampler(odom_noise1, 0)
    sampler_gps_noise0 = gtsam.Sampler(gps_noise0, 0)
    sampler_gps_noise1 = gtsam.Sampler(gps_noise1, 0)

    # init measurements
    measurements = AttrDict()
    num_steps = params.num_steps
    measurements.odom = np.zeros((num_steps-1, 3))
    measurements.gps = np.zeros((num_steps, 3))
    measurements.cov_type = np.zeros((num_steps, 1))

    # add measurements
    for tstep in range(0, num_steps):

        cov_type = covariance_type(tstep / float(num_steps))
        sampler_odom_noise = sampler_odom_noise0 if (cov_type == 0) else sampler_odom_noise1
        sampler_gps_noise = sampler_gps_noise0 if (cov_type == 0) else sampler_gps_noise1

        measurements.cov_type[tstep] = cov_type

        # binary odom
        if (tstep > 0):
            prev_pose = tf_utils.vec3_to_pose2(poses[tstep-1])
            curr_pose = tf_utils.vec3_to_pose2(poses[tstep])
            delta_pose = prev_pose.between(curr_pose)
            delta_pose_noisy = tf_utils.add_gaussian_noise(delta_pose, sampler_odom_noise.sample())

            measurements.odom[tstep-1, :] = tf_utils.pose2_to_vec3(delta_pose_noisy)

        # unary gps
        curr_pose = tf_utils.vec3_to_pose2(poses[tstep])
        curr_pose_noisy = tf_utils.add_gaussian_noise(curr_pose, sampler_gps_noise.sample())

        measurements.gps[tstep, :] = tf_utils.pose2_to_vec3(curr_pose_noisy)

    return measurements

def log_data_v0(params, poses, measurements, save_file=False):
    logger = AttrDict()

    logger.poses_gt = poses.tolist()
    logger.meas_odom = measurements.odom.tolist()
    logger.meas_gps = measurements.gps.tolist()

    logger.sigma_odom = list(params.measurements.noise_models.odom)
    logger.sigma_gps = list(params.measurements.noise_models.gps)

    logger.env_xlim = [params.env.area.xmin, params.env.area.xmax]
    logger.env_ylim = [params.env.area.ymin, params.env.area.ymax]

    logger.logname = "{0}_{1}".format(params.dataio.dataset_name, datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))

    if save_file:
        filename = "{0}/{1}/{2}.json".format(BASE_PATH, params.dataio.dstdir_dataset, params.dataio.dataset_name)
        dir_utils.write_file_json(filename=filename, data=logger)

    return logger


def log_data(params, poses, measurements, save_file=False):

    # get data for logger
    sigma_mat_odom = np.diag(list(params.measurements.noise_models.odom))
    sigma_mat_odom = (np.reshape(
        sigma_mat_odom, (sigma_mat_odom.shape[0]*sigma_mat_odom.shape[1]))).tolist()
    sigma_mat_gps = np.diag(list(params.measurements.noise_models.gps))
    sigma_mat_gps = (np.reshape(
        sigma_mat_gps, (sigma_mat_gps.shape[0]*sigma_mat_gps.shape[1]))).tolist()
    
    factor_names, factor_keysyms, factor_keyids, factor_covs, factor_meas = ([] for i in range(5))
    num_steps = params.num_steps
    for tstep in range(0, num_steps):

        # odom
        if (tstep > 0):
            factor_names.append('odom')
            factor_keysyms.append(['x', 'x'])
            factor_keyids.append([tstep-1, tstep])
            factor_covs.append(sigma_mat_odom)
            factor_meas.append(measurements.odom[tstep-1].tolist() + measurements.cov_type[tstep-1].tolist())

        # gps
        factor_names.append('gps')
        factor_keysyms.append(['x'])
        factor_keyids.append([tstep])
        factor_covs.append(sigma_mat_gps)
        factor_meas.append(measurements.gps[tstep].tolist() + measurements.cov_type[tstep].tolist())

    # save to logger object
    logger = AttrDict()
    logger.poses_gt = poses.tolist()

    logger.factor_names = factor_names
    logger.factor_keysyms = factor_keysyms
    logger.factor_keyids = factor_keyids
    logger.factor_covs = factor_covs
    logger.factor_meas = factor_meas

    logger.logname = "{0}_{1}".format(
        params.dataio.dataset_name, datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))

    if save_file:
        seq_idx = params.dataio.seq_idx
        dataset_mode = "train" if (seq_idx < params.dataio.n_data_train) else "test"
        filename = "{0}/{1}/{2:04d}.json".format(params.dataio.dstdir_logger, dataset_mode, seq_idx)
        dir_utils.write_file_json(filename=filename, data=logger)

    return logger

def load_poses_file(params):
    filename = "{0}/{1}/{2}/poses/{3:04d}.json".format(
        BASE_PATH, params.dataio.dstdir_dataset, params.dataio.dataset_name, params.dataio.seq_idx)

    dataset = dir_utils.read_file_json(filename, verbose=False)
    poses = np.asarray(dataset['poses'])

    return poses

def save_poses_file(params, poses):
    filename = "{0}/{1}/{2}/poses/{3:04d}.json".format(
        BASE_PATH, params.dataio.dstdir_dataset, params.dataio.dataset_name, params.dataio.seq_idx)
 
    logger = AttrDict()
    logger.poses = poses.tolist()

    dir_utils.write_file_json(filename, data=logger)

def random_cov_sigmas(min_val=0., max_val=1., dim=3):
    sigmas = np.random.rand(dim) * (max_val - min_val) + min_val    
    return sigmas
    
def get_covariances(params):
    
    covariances = AttrDict()
    
    if (params.measurements.noise_models == "random"):
        covariances.odom1 = random_cov_sigmas(min_val=1e-2, max_val=1e-1, dim=3)
        covariances.gps1 = random_cov_sigmas(min_val=1e-1, max_val=1, dim=3)
    
        covariances.odom2 = random_cov_sigmas(min_val=1e-1, max_val=1e-1, dim=3)
        covariances.gps2 = random_cov_sigmas(min_val=1e-1, max_val=1, dim=3)

        return covariances

    covariances.odom0 = np.array(params.measurements.noise_models.odom0)
    covariances.gps0 = np.array(params.measurements.noise_models.gps0)

    covariances.odom1 = np.array(params.measurements.noise_models.odom1)
    covariances.gps1 = np.array(params.measurements.noise_models.gps1)

    return covariances

@hydra.main(config_path=CONFIG_PATH, strict=False)
def main(cfg):
        
    if cfg.options.random_seed is not None:
        np.random.seed(cfg.options.random_seed)

    # create logger dstdir
    cfg.dataio.dstdir_logger = "{0}/{1}/{2}/dataset_{3:04d}".format(
        BASE_PATH, cfg.dataio.dstdir_dataset, cfg.dataio.dataset_name, cfg.dataio.start_ds_idx)
    dir_utils.make_dir(cfg.dataio.dstdir_logger+"/train", clear=True)
    dir_utils.make_dir(cfg.dataio.dstdir_logger+"/test", clear=True)

    for ds_idx in range(cfg.dataio.start_ds_idx, cfg.dataio.n_datasets):
        cfg.dataio.ds_idx = ds_idx

        for seq_idx in range(cfg.dataio.start_seq_idx, cfg.dataio.n_seqs):
            cfg.dataio.seq_idx = seq_idx
            covariances = get_covariances(cfg)

            # load poses
            if (cfg.dataio.load_poses_file):
                poses = load_poses_file(cfg)
            else:
                poses = get_waypoints_gui(cfg, poses=None)
                if (cfg.dataio.save_poses_file):
                    save_poses_file(cfg, poses)

            # create measurements
            cfg.num_steps = int(np.minimum(poses.shape[0], cfg.measurements.num_steps_max))
            measurements = create_measurements(cfg, poses, covariances)

            cfg.dataio.dstdir_logger = "{0}/{1}/{2}/dataset_{3:04d}".format(
                BASE_PATH, cfg.dataio.dstdir_dataset, cfg.dataio.dataset_name, ds_idx)
            dir_utils.make_dir(cfg.dataio.dstdir_logger, clear=False)

            logger = log_data(cfg, poses, measurements, save_file=True)

            # todo: plot using updated logger
            logger_plot = log_data_v0(
                cfg, poses, measurements, save_file=False)
            plot_data(cfg, logger_plot, plot_ori=False)


if __name__ == '__main__':
    main()
