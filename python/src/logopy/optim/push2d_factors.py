#!/usr/bin/env python

import sys
sys.path.append("/usr/local/cython/")

import math
import numpy as np
import json

import torch
from torch.autograd import Variable

import gtsam
import logo

from logopy.utils import tf_utils, dir_utils, vis_utils

class PlanarSDF():
    def __init__(self, sdf_file=None):
        self.sdf = None

        if sdf_file is not None:
            self.load_sdf_from_file_json(sdf_file)

    def load_sdf_from_file_json(self, sdf_file):

        with open(sdf_file) as f:
            sdf_data = json.load(f)

        cell_size = sdf_data['grid_res']
        sdf_cols = sdf_data['grid_size_x']
        sdf_rows = sdf_data['grid_size_y']
        sdf_data_vec = sdf_data['grid_data']
        sdf_origin_x = sdf_data['grid_origin_x']
        sdf_origin_y = sdf_data['grid_origin_y']

        origin = gtsam.Point2(sdf_origin_x, sdf_origin_y)

        sdf_data_mat = np.zeros((sdf_rows, sdf_cols))
        for i in range(sdf_data_mat.shape[0]):
            for j in range(sdf_data_mat.shape[1]):
                sdf_data_mat[i, j] = sdf_data_vec[i][j]

        self.sdf = logo.PlanarSDF(origin, cell_size, sdf_data_mat)

def get_noise_model(factor_cov):

    factor_noise_model = None

    if (factor_cov.shape[0] <= 3):  # cov sigmas
        factor_noise_model = gtsam.noiseModel_Diagonal.Sigmas(factor_cov)
    elif (factor_cov.shape[0] == 9):  # cov matrix
        factor_cov = np.reshape(factor_cov, (3, 3))
        factor_noise_model = gtsam.noiseModel_Gaussian.Covariance(factor_cov)

    return factor_noise_model

def add_binary_odom_factor(graph, keys, factor_cov, factor_meas):

    factor_noise_model = get_noise_model(factor_cov)
    factor_meas_pose = tf_utils.vec3_to_pose2(factor_meas)
    factor = gtsam.BetweenFactorPose2(
        keys[0], keys[1], factor_meas_pose, factor_noise_model)

    graph.push_back(factor)

    return graph
    
def add_qs_motion_factor(graph, keys, factor_cov, params=None):
    # keys: o_{t-1}, o_{t}, e_{t-1}, e_{t}

    factor_noise_model = get_noise_model(factor_cov)

    if (params.dataio.obj_shape == 'disc'):
        c_sq = math.pow(0.088 / 3, 2)
    elif (params.dataio.obj_shape == 'rect'):
        c_sq = math.pow(math.sqrt(0.2363**2 + 0.1579**2), 2)
    elif (params.dataio.obj_shape == 'ellip'):
        c_sq = (0.5 * (0.1638 + 0.2428)) ** 2
    else:
        print("object shape sdf not found")

    factor = logo.QSVelPushMotionRealObjEEFactor(
        keys[0], keys[1], keys[2], keys[3], c_sq, factor_noise_model)

    graph.push_back(factor)

    return graph


def add_sdf_intersection_factor(graph, keys, factor_cov, object_sdf):
    # keys: o_{t}, e_{t}

    factor_noise_model = get_noise_model(factor_cov)
    factor = logo.IntersectionPlanarSDFObjEEFactor(
        keys[0], keys[1], object_sdf, 0.0, factor_noise_model)

    graph.push_back(factor)

    return graph


def add_tactile_rel_meas_factor(graph, keys, factor_cov, tf_pred, params=None):
    # keys: o_{t-k}, o_{t}, e_{t-k}, e_{t}

    factor_noise_model = get_noise_model(factor_cov)
    factor = logo.TactileRelativeTfRegressionFactor(
        keys[0], keys[1], keys[2], keys[3], tf_pred, factor_noise_model)

    factor.setFlags(yawOnlyError=params.yaw_only_error,
                    constantModel=params.constant_model)
    factor.setLabel(classLabel=params.class_label,
                    numClasses=params.num_classes)

    graph.push_back(factor)

    return graph

def tactile_model_output(tactile_model, img_feats, params=None):
    img_feat_i = torch.tensor(img_feats[0]).view(1, -1)
    img_feat_j = torch.tensor(img_feats[1]).view(1, -1)

    if (params.norm_img_feat == True):
        img_feat_i = tf_utils.normalize_vector(
            img_feat_i, torch.tensor(params.mean_img_feat), torch.tensor(params.std_img_feat))
        img_feat_j = tf_utils.normalize_vector(
            img_feat_j, torch.tensor(params.mean_img_feat), torch.tensor(params.std_img_feat))

    class_label_vec = torch.nn.functional.one_hot(
        torch.tensor(params.class_label), params.num_classes)
    class_label_vec = class_label_vec.view(1, -1)

    tf_pred = tactile_model.forward(img_feat_i, img_feat_j, class_label_vec)
    tf_pred = (tf_pred.view(-1)).detach().cpu().numpy()

    return tf_pred

def tactile_oracle_output(data, key_ids):
    # keys: o_{t-k}, o_{t}, e_{t-k}, e_{t}

    obj_pose1 = torch.tensor(data['obj_poses_gt'][key_ids[0]]).view(1, -1)
    obj_pose2 = torch.tensor(data['obj_poses_gt'][key_ids[1]]).view(1, -1)
    ee_pose1 = torch.tensor(data['ee_poses_gt'][key_ids[2]]).view(1, -1)
    ee_pose2 = torch.tensor(data['ee_poses_gt'][key_ids[3]]).view(1, -1)

    ee_pose1__obj = tf_utils.tf2d_between(obj_pose1, ee_pose1) # n x 3
    ee_pose2__obj = tf_utils.tf2d_between(obj_pose2, ee_pose2)

    pose_rel_gt = tf_utils.tf2d_between(ee_pose1__obj, ee_pose2__obj) # n x 3

    yaw_only_error = True
    if yaw_only_error:
        tf_pred = np.array([0., 0., np.cos(pose_rel_gt[0, 2].data), np.sin(pose_rel_gt[0, 2])])
    
    return tf_pred

def debug_tactile_factor(data, key_ids, tf_pred_net):
    # keys: o_{t-k}, o_{t}, e_{t-k}, e_{t}

    tf_pred_net = torch.tensor(tf_pred_net)

    obj_pose1 = torch.tensor(data['obj_poses_gt'][key_ids[0]]).view(1, -1)
    obj_pose2 = torch.tensor(data['obj_poses_gt'][key_ids[1]]).view(1, -1)
    ee_pose1 = torch.tensor(data['ee_poses_gt'][key_ids[2]]).view(1, -1)
    ee_pose2 = torch.tensor(data['ee_poses_gt'][key_ids[3]]).view(1, -1)
    ee_pose1__obj = tf_utils.tf2d_between(obj_pose1, ee_pose1) # n x 3
    ee_pose2__obj = tf_utils.tf2d_between(obj_pose2, ee_pose2)
    pose_rel_gt = tf_utils.tf2d_between(ee_pose1__obj, ee_pose2__obj) # n x 3

    pose_rel_gt = torch.tensor([0., 0., pose_rel_gt[0, 2]]).view(1, -1)
    pose_rel_meas = torch.tensor([0., 0., torch.asin(tf_pred_net[3])]).view(1, -1) # n x 3
    diff_vec = tf_utils.tf2d_between(pose_rel_gt, pose_rel_meas) # n x 3

    print("pose_rel_gt: {0}, pose_rel_meas: {1}, pose_diff: {2}".format(
        pose_rel_gt, pose_rel_meas, diff_vec))

def log_step_push2d(tstep, logger, data, optimizer):

    num_poses = tstep + 1

    # log estimated poses
    poses_graph = optimizer.calculateEstimate()
    obj_pose_vec_graph = np.zeros((num_poses, 3))
    ee_pose_vec_graph = np.zeros((num_poses, 3))

    for i in range(0, num_poses):
        obj_key = gtsam.symbol(ord('o'), i)
        ee_key = gtsam.symbol(ord('e'), i)

        obj_pose2d = poses_graph.atPose2(obj_key)
        ee_pose2d = poses_graph.atPose2(ee_key)

        obj_pose_vec_graph[i, :] = [obj_pose2d.x(), obj_pose2d.y(), obj_pose2d.theta()]
        ee_pose_vec_graph[i, :] = [ee_pose2d.x(), ee_pose2d.y(), ee_pose2d.theta()]

    logger.log_step("graph/obj_poses2d", obj_pose_vec_graph, tstep)
    logger.log_step("graph/ee_poses2d", ee_pose_vec_graph, tstep)

    # log gt poses
    obj_pose_vec_gt = data['obj_poses_gt'][0:num_poses]
    ee_pose_vec_gt = data['ee_poses_gt'][0:num_poses]
    logger.log_step('gt/obj_poses2d', obj_pose_vec_gt, tstep)
    logger.log_step('gt/ee_poses2d', ee_pose_vec_gt, tstep)

    return logger