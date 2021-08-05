#!/usr/bin/env python

import sys
sys.path.append("/usr/local/cython/")

import numpy as np
import os

import pandas as pd
import glob
import hydra

import matplotlib.pyplot as plt
from matplotlib import cm

from leopy.utils import tf_utils

import logging
log = logging.getLogger(__name__)

plt.rcParams.update({'font.size': 18})

dataset_type = "nav2d" # "nav2d", "push2d"

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
CONFIG_PATH = os.path.join(BASE_PATH, f"python/config/leo_{dataset_type}.yaml")

def pandas_string_to_numpy(arr_str):
    arr_npy = np.fromstring(arr_str.replace('\n', '').replace('[', '').replace(']', '').replace('  ', ' '), sep=', ')
    return arr_npy

def compute_x_samples(mean, dx_samples):
    # mean: n_steps x 3
    # dx_samples: n_samples x n_steps x 3

    S, T, D = dx_samples.shape
    x_samples = np.zeros(dx_samples.shape)

    for sidx in range(0, S):
        for tidx in range(0, T):
            x_mean = tf_utils.vec3_to_pose2(mean[tidx, :])
            x_sample_pose = x_mean.retract(dx_samples[sidx, tidx, :])
            x_samples[sidx, tidx, :] = tf_utils.pose2_to_vec3(x_sample_pose)

    return x_samples

def traj_samples(logfiles):

    data_idx = 0
    nsteps = 499
    leo_itrs = 40
    n_samples = 40
    cov_scale = 10.

    plt.ion()
    fig = plt.figure(constrained_layout=True, figsize=(8, 4))
    ax1 = plt.gca()

    colors = np.vstack((
        # cm.Reds(np.linspace(0.2, 0.2, num=n_samples)),
        cm.Blues(np.linspace(0.4, 0.4, num=int(0.5*n_samples))),
        cm.Blues(np.linspace(0.4, 0.4, num=int(0.5*n_samples)))
    ))

    tstart = 25
    tend = 425

    for file_idx in range(0, len(logfiles)):

        print(f'Loading csv: {logfiles[file_idx]}')
        df = pd.read_csv(f'{logfiles[file_idx]}')

        row_filter_1 = (df.data_idx == data_idx) & (df.tstep == nsteps)
        
        for itr in range(0, leo_itrs):
            plt.cla()
            plt.xlim([-50, 50])
            plt.ylim([0, 60])
            plt.axis('off')

            row_filter_2 = row_filter_1 & (df.leo_itr == itr)

            mean = (df.loc[row_filter_2, ['opt/mean']]).iloc[0, 0]
            covariance = (df.loc[row_filter_2, ['opt/covariance']]).iloc[0, 0]
            poses_gt = (df.loc[row_filter_2, ['opt/gt/poses2d']]).iloc[0, 0]

            mean = pandas_string_to_numpy(mean)
            covariance = pandas_string_to_numpy(covariance)
            poses_gt = pandas_string_to_numpy(poses_gt)
                        
            covariance = covariance.reshape(mean.shape[0], mean.shape[0])
            covariance = cov_scale * covariance
            dx_samples = np.random.multivariate_normal(np.zeros(mean.shape[0]), covariance, n_samples)

            mean = mean.reshape(-1, 3)
            dx_samples = dx_samples.reshape(dx_samples.shape[0], -1, 3)
            x_samples = compute_x_samples(mean, dx_samples)            
            poses_gt = poses_gt.reshape(-1, 3)

            # plotting
            for sidx in range(0, n_samples):
                plt.plot(x_samples[sidx, tstart:tend, 0], x_samples[sidx, tstart:tend, 1], linewidth=4, linestyle='-', color=colors[sidx])

            plt.plot(poses_gt[tstart:tend, 0], poses_gt[tstart:tend, 1], linewidth=4, linestyle='--', color='tab:grey')
            plt.plot(mean[tstart:tend, 0], mean[tstart:tend, 1], linewidth=3, linestyle='-', color='tab:blue')

            plt.show()
            plt.pause(1e-2)
            plt.savefig(f"{BASE_PATH}/local/plots/traj_samples/{dataset_type}/leo_itr_{itr:04d}")

        import pdb; pdb.set_trace()

@hydra.main(config_name=CONFIG_PATH)
def main(cfg):
    
    srcdir = f"{BASE_PATH}/local/datalogs/{cfg.dataio.dataset_name}"
    logfiles = sorted(glob.glob(f"{srcdir}/**/*.csv"), reverse=True)

    traj_samples(logfiles)

if __name__ == '__main__':
    main()
