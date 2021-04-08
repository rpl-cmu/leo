#!/usr/bin/env python

import sys
sys.path.append("/usr/local/cython/")

import math
import numpy as np

import torch
from torch.autograd import Variable

import gtsam
import logo

from logopy.utils import tf_utils, dir_utils, vis_utils

def get_noise_model(factor_cov):

    factor_noise_model = None

    if (factor_cov.shape[0] <= 3):  # cov sigmas
        factor_noise_model = gtsam.noiseModel_Diagonal.Sigmas(factor_cov)
    elif (factor_cov.shape[0] == 9):  # cov matrix
        factor_cov = np.reshape(factor_cov, (3, 3))
        factor_noise_model = gtsam.noiseModel_Gaussian.Covariance(factor_cov)

    return factor_noise_model


def add_unary_factor(graph, keys, factor_cov, factor_meas):

    factor_noise_model = get_noise_model(factor_cov)
    factor_meas_pose = tf_utils.vec3_to_pose2(factor_meas)
    factor = gtsam.PriorFactorPose2(
        keys[0], factor_meas_pose, factor_noise_model)

    graph.push_back(factor)

    return graph


def add_binary_odom_factor(graph, keys, factor_cov, factor_meas):

    factor_noise_model = get_noise_model(factor_cov)
    factor_meas_pose = tf_utils.vec3_to_pose2(factor_meas)
    factor = gtsam.BetweenFactorPose2(
        keys[0], keys[1], factor_meas_pose, factor_noise_model)

    graph.push_back(factor)

    return graph

def log_step_nav2d(tstep, logger, data, optimizer):

    num_poses = tstep + 1

    pose_vec_graph = np.zeros((num_poses, 3))
    pose_vec_gt = np.asarray(data['poses_gt'][0:num_poses])
    poses_graph = optimizer.calculateEstimate()

    for i in range(0, num_poses):
        key = gtsam.symbol(ord('x'), i)
        pose2d = poses_graph.atPose2(key)
        pose_vec_graph[i, :] = [pose2d.x(), pose2d.y(), pose2d.theta()]

    logger.log_step("graph/poses2d", pose_vec_graph, tstep)
    logger.log_step("gt/poses2d", pose_vec_gt, tstep)

    return logger
