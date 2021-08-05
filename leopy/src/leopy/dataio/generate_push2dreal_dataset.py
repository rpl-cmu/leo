#!/usr/bin/env python

import sys
sys.path.append("/usr/local/cython/")

import numpy as np
import math

import os
import hydra
from attrdict import AttrDict
from datetime import datetime

from itertools import combinations, permutations, combinations_with_replacement
import random
from tqdm import tqdm

import gtsam
from leopy.utils import tf_utils, dir_utils
from leopy.dataio import data_process
from leopy.eval import quant_metrics

import matplotlib.pyplot as plt

BASE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../.."))
CONFIG_PATH = os.path.join(BASE_PATH, "python/config/dataio/push2d.yaml")

def create_cov_mat(sigmas, flatten=True):

    sigmas_sq = [sigma**2 for sigma in sigmas]
    cov_mat = np.diag(list(sigmas_sq))
    if flatten:
        cov_mat = np.reshape(cov_mat, (sigmas.shape[0]*sigmas.shape[1])).tolist()

    return cov_mat

def wrap_logger_angles_to_pi(logger, field_names):

    for field in field_names:
        field_arr = np.asarray(logger[field])
        field_arr[:, -1] = quant_metrics.wrap_to_pi(field_arr[:, -1]) # x, y, theta format
        logger[field] = field_arr.tolist()
    
    return logger

def log_data(params, dataset, ds_idxs, eps_idxs, seq_idx, save_file=False):

    # init noise models
    sigma_mat_ee = list(params.measurements.noise_models.ee_pose_prior)
    sigma_mat_qs = list(params.measurements.noise_models.qs_push_motion)
    sigma_mat_sdf = list(params.measurements.noise_models.sdf_intersection)
    sigma_mat_tactile = list(params.measurements.noise_models.tactile_rel_meas)
    sigma_mat_interseq = list(params.measurements.noise_models.binary_interseq_obj)

    ee_pose_prior_noise = gtsam.noiseModel_Diagonal.Sigmas(
        np.array(params.measurements.noise_models.ee_pose_prior))
    sampler_ee_pose_prior_noise = gtsam.Sampler(ee_pose_prior_noise, 0)

    # init other params
    factor_names, factor_keysyms, factor_keyids, factor_covs, factor_meas = ([] for i in range(5))
    num_steps = params.dataio.num_steps
    wmin, wmax = params.measurements.tactile.wmin, params.measurements.tactile.wmax
    wnum = 2

    meas_ee_prior = []
    meas_tactile_img_feats = []
    for tstep in range(0, num_steps):

        # ee_pose_prior
        ee_pose_noisy = tf_utils.add_gaussian_noise(tf_utils.vec3_to_pose2(
            dataset['ee_poses_2d'][ds_idxs[tstep]]), sampler_ee_pose_prior_noise.sample())
        factor_names.append('ee_pose_prior')
        factor_keysyms.append(['e'])
        factor_keyids.append([tstep])
        factor_covs.append(sigma_mat_ee)
        factor_meas.append(tf_utils.pose2_to_vec3(ee_pose_noisy))

        # qs_push_motion
        if (tstep > 0) & (eps_idxs[tstep-1] == eps_idxs[tstep]):
            factor_names.append('qs_push_motion')
            factor_keysyms.append(['o', 'o', 'e', 'e'])
            factor_keyids.append([tstep-1, tstep, tstep-1, tstep])
            factor_covs.append(sigma_mat_qs)
            factor_meas.append([0., 0., 0.])

        # sdf_intersection
        factor_names.append('sdf_intersection')
        factor_keysyms.append(['o', 'e'])
        factor_keyids.append([tstep, tstep])
        factor_covs.append(sigma_mat_sdf)
        factor_meas.append([0.])

        # tactile_rel
        if (tstep > wmin):

            # wmax_step = np.minimum(tstep, wmax)
            # wrange = np.linspace(wmin, wmax_curr_step, num=wnum).astype(np.int)
            # for w in wrange:
            
            wmax_step = np.minimum(tstep, wmax)
            for w in range(wmin, wmax_step):

                # skip inter episode factors
                if (eps_idxs[tstep-w] != eps_idxs[tstep]):
                    continue

                # print([tstep-w, tstep, tstep-w, tstep])

                factor_names.append('tactile_rel')
                factor_keysyms.append(['o', 'o', 'e', 'e'])
                factor_keyids.append([tstep-w, tstep, tstep-w, tstep])
                factor_covs.append(sigma_mat_tactile)
                factor_meas.append((dataset['img_feats'][ds_idxs[tstep-w]], dataset['img_feats'][ds_idxs[tstep]]))
        
        # interseq binary
        if (tstep > 0) & (eps_idxs[tstep-1] != eps_idxs[tstep]):
            factor_names.append('binary_interseq_obj')
            factor_keysyms.append(['o', 'o'])
            factor_keyids.append([tstep-1, tstep])
            factor_covs.append(sigma_mat_interseq)
            factor_meas.append([0., 0., 0.])
        
        # store external measurement separately
        meas_ee_prior.append(tf_utils.pose2_to_vec3(ee_pose_noisy))
        meas_tactile_img_feats.append(dataset['img_feats'][ds_idxs[tstep]])
    
    # save to logger object
    logger = AttrDict()
    logger.ee_poses_gt = [dataset['ee_poses_2d'][idx] for idx in ds_idxs]
    logger.obj_poses_gt = [dataset['obj_poses_2d'][idx] for idx in ds_idxs]
    logger.contact_episode = [dataset['contact_episode'][idx] for idx in ds_idxs]
    logger.contact_flag = [dataset['contact_flag'][idx] for idx in ds_idxs]

    logger.factor_names = factor_names
    logger.factor_keysyms = factor_keysyms
    logger.factor_keyids = factor_keyids
    logger.factor_covs = factor_covs
    logger.factor_meas = factor_meas

    logger.meas_ee_prior = meas_ee_prior
    logger.meas_tactile_img_feats = meas_tactile_img_feats
    logger = wrap_logger_angles_to_pi(logger, field_names=['ee_poses_gt', 'obj_poses_gt', 'meas_ee_prior'])

    logger.logname = "{0}_{1}".format(
        params.dataio.dataset_name, datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))

    if save_file:
        dataset_mode = "train" if (seq_idx < params.dataio.n_data_train) else "test"
        filename = "{0}/{1}/{2:04d}.json".format(params.dataio.dstdir_logger, dataset_mode, seq_idx)
        dir_utils.write_file_json(filename=filename, data=logger)

    return logger

def sequence_generator(contact_episodes, num_episodes_seq):

    episode_idxs = list(set(contact_episodes))
    episode_seq_list = list(permutations(episode_idxs, num_episodes_seq))

    random.shuffle(episode_seq_list)
    print("[sequence_generator] Generated {0} episode sequences".format(len(episode_seq_list)))

    return episode_seq_list


def sequence_logger(params, dataset, episode_seq, seq_idx, save_file=True):
    dataset_idxs = []
    episode_idxs = []
    episode_seq = list(episode_seq)
    logger = None
 
    # collect dataset idxs for episodes in episode_seq
    for episode in episode_seq:
        dataset_idxs_curr = [idx for idx, val in enumerate(
            dataset['contact_episode']) if (val == episode)]
        dataset_idxs.append(dataset_idxs_curr)
        episode_idxs.append([episode] * len(dataset_idxs_curr))

    # flatten into single list
    dataset_idxs = [item for sublist in dataset_idxs for item in sublist]
    episode_idxs = [item for sublist in episode_idxs for item in sublist]

    if (len(episode_idxs) < params.dataio.num_steps):
        return

    # cut down to num_steps entries
    dataset_idxs = dataset_idxs[0:params.dataio.num_steps]
    episode_idxs = episode_idxs[0:params.dataio.num_steps]

    print("Logging dataset {0}/{1} of length {2} and episode idxs {3}".format(
        seq_idx, params.dataio.num_seqs, len(episode_idxs), episode_seq))

    dataset_tf = data_process.transform_episodes_common_frame(
        episode_seq, dataset)

    logger = log_data(params, dataset_tf, dataset_idxs,
                    episode_idxs, seq_idx, save_file=save_file)

    return logger


@hydra.main(config_path=CONFIG_PATH)
def main(cfg):

    dataset_file = f"{BASE_PATH}/{cfg.dataio.srcdir_pushest}/{cfg.dataio.dataset_name}.json"
    dataset = dir_utils.read_file_json(dataset_file)
    
    # create logger dstdir
    cfg.dataio.dstdir_logger = "{0}/{1}/{2}".format(
        BASE_PATH, cfg.dataio.dstdir_dataset, cfg.dataio.dataset_name)
    dir_utils.make_dir(cfg.dataio.dstdir_logger+"/train", clear=True)
    dir_utils.make_dir(cfg.dataio.dstdir_logger+"/test", clear=True)

    # generate random episode combination sequences
    episode_seq_list = sequence_generator(
        dataset['contact_episode'], cfg.dataio.num_episodes_seq)
    
    # store episode combinations used for generating the dataset
    eps_seq_file = f"{BASE_PATH}/{cfg.dataio.dstdir_dataset}/episode_seq_{cfg.dataio.dataset_name}.txt"
    eps_seq_fid = open(eps_seq_file, 'w')

    seq_idx = 0
    num_seqs = np.minimum(len(episode_seq_list), cfg.dataio.num_seqs)
    for episode_seq in tqdm(episode_seq_list):

        logger = sequence_logger(cfg, dataset, episode_seq, seq_idx, save_file=True)

        if logger is not None:
            seq_idx = seq_idx + 1

        eps_seq_fid.write(f"{episode_seq}\n")
        # plot_logger_data(cfg, logger)

        if (seq_idx > num_seqs):
            eps_seq_fid.close()
            break

if __name__ == '__main__':
    main()
