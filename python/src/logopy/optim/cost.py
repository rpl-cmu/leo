#!/usr/bin/env python

import sys
sys.path.append("/usr/local/cython/")

import math
import numpy as np
from scipy.optimize import minimize

import os
import json
import hydra
from datetime import datetime
from attrdict import AttrDict
from collections import namedtuple

import torch
from torch.autograd import Variable
import gtsam

from logopy.utils import tf_utils, dir_utils

## factor costs in pytorch ##

def unary_factor_cost(x, key_syms, key_ids, factor_inf, factor_meas, device=None, meta=None):

    key_id = key_ids[0]
    key_id = key_id + int(0.5 * x.shape[0]) if (key_syms[0] == 'e') else key_id

    est_pose = (x[key_id, :]).view(1, -1)
    meas_pose = factor_meas.view(1, -1)

    diff_val = tf_utils.tf2d_between(est_pose, meas_pose, device=device).view(-1, 1)
    # diff_val = (torch.sub(est_val, factor_meas)).view(-1, 1)

    err = torch.matmul(torch.matmul(diff_val.permute(1, 0), factor_inf), diff_val)

    err = err.view(-1)

    return err

def binary_odom_factor_cost(x, key_syms, key_ids, factor_inf, factor_meas, device=None, meta=None):
    
    p1 = (x[key_ids[0], :]).view(1, -1) # n x 3
    p2 = (x[key_ids[1], :]).view(1, -1) # n x 3

    est_val = (tf_utils.tf2d_between(p1, p2, device)).view(-1)

    diff_val = (torch.sub(est_val, factor_meas)).view(-1, 1) # 3 x 1
    err = torch.matmul(torch.matmul(diff_val.permute(1, 0), factor_inf), diff_val)

    err = err.view(-1)
            
    return err

def sdf_intersection_factor_cost(x, key_syms, key_ids, factor_inf, device=None, meta=None):

    err = torch.tensor([0.], device=device)

    return err

def qs_motion_factor_metadata(obj_shape):

    meta = AttrDict()

    if (obj_shape == 'disc'):
        meta.c_sq = math.pow(0.088 / 3, 2)
    elif (obj_shape == 'rect'):
        meta.c_sq = math.pow(math.sqrt(0.2363**2 + 0.1579**2), 2)
    elif (obj_shape == 'ellip'):
        meta.c_sq = (0.5 * (0.1638 + 0.2428)) ** 2
    else:
        print("object shape sdf not found")

    return meta

def qs_motion_factor_cost(x, key_syms, key_ids, factor_inf, device=None, meta=None):
    # keys: o_{t-1}, o_{t}, e_{t-1}, e_{t}
    
    offset_e = int(0.5 * x.shape[0])

    obj_pose0 = (x[key_ids[0], :]).view(1, -1) # 1 x 3
    obj_pose1 = (x[key_ids[1], :]).view(1, -1)
    ee_pose0 = (x[offset_e + key_ids[2], :]).view(1, -1) # 1 x 3
    ee_pose1 = (x[offset_e + key_ids[3], :]).view(1, -1)

    obj_ori1 = obj_pose1.clone(); obj_ori1[0, 0] = 0.; obj_ori1[0, 1] = 0.
    obj_pose_rel__world = tf_utils.tf2d_between(obj_pose0, obj_pose1, device=device)

    vel_obj__world = obj_pose1.clone() - obj_pose0.clone(); vel_obj__world[0, 2] = 0.
    vel_obj__obj = tf_utils.tf2d_between(obj_ori1, vel_obj__world, device=device)

    vel_contact__world = ee_pose1.clone() - ee_pose0.clone(); vel_contact__world[0, 2] = 0.
    vel_contact__obj = tf_utils.tf2d_between(obj_ori1, vel_contact__world, device=device)
    
    contact_point1 = ee_pose1.clone(); contact_point1[0, 2] = 0.
    contact_point__obj = tf_utils.tf2d_between(obj_pose1, contact_point1, device=device)

    # # D*V = Vp
    vx = vel_obj__obj[0, 0]
    vy = vel_obj__obj[0, 1]
    omega = obj_pose_rel__world[0, 2]

    vpx = vel_contact__obj[0, 0]
    vpy = vel_contact__obj[0, 1]

    px = contact_point__obj[0, 0]
    py = contact_point__obj[0, 1]

    D = torch.tensor([[1, 0, -py], [0, 1, px], [-py, px, -meta.c_sq]], device=device)
    V = torch.tensor([vx, vy, omega], device=device)
    Vp = torch.tensor([vpx, vpy, 0.], device=device)

    diff_vec = torch.sub(torch.matmul(D, V), Vp).view(1, -1) # 1 x 3
    err = torch.matmul(torch.matmul(diff_vec, factor_inf), diff_vec.permute(1, 0)) # 1 x 1

    err = err.view(-1)

    return err

def tactile_rel_meas_factor_cost(x, key_syms, key_ids, factor_inf, factor_meas, device=None, meta=None):
    # keys: o_{t-k}, o_{t}, e_{t-k}, e_{t}
        
    offset_e = int(0.5 * x.shape[0])

    obj_pose1 = (x[key_ids[0], :]).view(1, -1) # n x 3
    obj_pose2 = (x[key_ids[1], :]).view(1, -1)
    ee_pose1 = (x[offset_e + key_ids[2], :]).view(1, -1) # n x 3
    ee_pose2 = (x[offset_e + key_ids[3], :]).view(1, -1)

    ee_pose1__obj = tf_utils.tf2d_between(obj_pose1, ee_pose1, device=device) # n x 3
    ee_pose2__obj = tf_utils.tf2d_between(obj_pose2, ee_pose2, device=device)
    pose_rel_expect = tf_utils.tf2d_between(ee_pose1__obj, ee_pose2__obj, device=device) # n x 3

    yaw_only_error = True
    if yaw_only_error:
        pose_rel_meas = torch.tensor([0., 0., torch.asin(factor_meas[3])], device=device).view(1, -1) # n x 3
        pose_rel_expect = torch.tensor([0., 0., pose_rel_expect[0, 2]], device=device).view(1, -1)

    diff_vec = tf_utils.tf2d_between(pose_rel_expect, pose_rel_meas, device) # n x 3
    err = torch.matmul(torch.matmul(diff_vec, factor_inf), diff_vec.permute(1, 0)) # 1 x 1

    err = err.view(-1)

    return err

def tactile_model_output(tactile_model, img_feats, meta=None):
    img_feat_i = torch.tensor(img_feats[0]).view(1, -1)
    img_feat_j = torch.tensor(img_feats[1]).view(1, -1)

    if (meta.norm_img_feat == True):
        img_feat_i = tf_utils.normalize_vector(
            img_feat_i, torch.tensor(meta.mean_img_feat), torch.tensor(meta.std_img_feat))
        img_feat_j = tf_utils.normalize_vector(
            img_feat_j, torch.tensor(meta.mean_img_feat), torch.tensor(meta.std_img_feat))

    class_label_vec = torch.nn.functional.one_hot(
        torch.tensor(meta.class_label), meta.num_classes)
    class_label_vec = class_label_vec.view(1, -1)

    tf_pred = tactile_model.forward(img_feat_i, img_feat_j, class_label_vec)
    tf_pred = tf_pred.view(-1) # [tx, ty, cyaw, syaw]

    return tf_pred

def tactile_oracle_output(data, key_syms, key_ids, device=None, meta=None):
    # keys: o_{t-k}, o_{t}, e_{t-k}, e_{t}

    obj_pose_gt_i = torch.tensor(data['obj_poses_gt'][key_ids[0]], device=device).view(1, -1) # n x 3
    obj_pose_gt_j = torch.tensor(data['obj_poses_gt'][key_ids[1]], device=device).view(1, -1)
    ee_pose_gt_i = torch.tensor(data['ee_poses_gt'][key_ids[2]], device=device).view(1, -1)
    ee_pose_gt_j = torch.tensor(data['ee_poses_gt'][key_ids[3]], device=device).view(1, -1)

    ee_pose_gt_i__obj = tf_utils.tf2d_between(obj_pose_gt_i, ee_pose_gt_i, device=device) # n x 3
    ee_pose_gt_j__obj = tf_utils.tf2d_between(obj_pose_gt_j, ee_pose_gt_j, device=device)
    pose_rel_gt = tf_utils.tf2d_between(ee_pose_gt_i__obj, ee_pose_gt_j__obj, device=device) # n x 3

    yaw_only_error = True
    if yaw_only_error:
        tf_pred = torch.tensor([0., 0., torch.cos(pose_rel_gt[0, 2]), torch.sin(pose_rel_gt[0, 2])], device=device).view(-1)
    
    return tf_pred

class Cost():
    def __init__(self, data, theta, params=None, device=None):
        self.data = data
        self.theta = theta
        self.params = params
        self.device = device
    
    def costfn(self, x):

        dataset_type = self.params.dataio.dataset_type

        if (dataset_type == "push2d"):
            nsteps = np.minimum(self.params.optim.nsteps, len(self.data['obj_poses_gt']))
            tactile_model = torch.jit.load("{0}/local/models/{1}.pt".format(
                self.params.BASE_PATH, self.params.tactile_model.name))
        else:
            nsteps = np.minimum(self.params.optim.nsteps, len(self.data['poses_gt']))
        
        fval = torch.tensor([0.], requires_grad=True, device=self.device)
        
        for tstep in range(1, nsteps):

            # filter out curr step factors
            factor_idxs = [idxs for idxs, keys in enumerate(
                self.data['factor_keyids']) if (max(keys) == tstep)]

            factor_keysyms = [self.data['factor_keysyms'][idx]
                            for idx in factor_idxs]
            factor_keyids = [self.data['factor_keyids'][idx]
                            for idx in factor_idxs]
            factor_meas = [self.data['factor_meas'][idx] for idx in factor_idxs]
            factor_names = [self.data['factor_names'][idx] for idx in factor_idxs]

            # compute factor costs
            for idx in range(0, len(factor_idxs)):

                key_syms, key_ids = factor_keysyms[idx], factor_keyids[idx]
                factor_name = factor_names[idx]

                factor_cost = torch.tensor([0.], requires_grad=True, device=self.device)

                if (factor_name == 'gps'):
                    factor_meas_val = torch.tensor(factor_meas[idx], device=self.device)
                    factor_inf = torch.diag(self.theta.get_sigma_inv(factor_name, z=factor_meas_val)) ** 2
                    factor_cost = unary_factor_cost(x, key_syms, key_ids, factor_inf, factor_meas_val[0:3], device=self.device) if (factor_inf.requires_grad) else factor_cost
                elif (factor_name == 'odom'):
                    factor_meas_val = torch.tensor(factor_meas[idx], device=self.device)                    
                    factor_inf = torch.diag(self.theta.get_sigma_inv(factor_name, z=factor_meas_val)) ** 2
                    factor_cost = binary_odom_factor_cost(x, key_syms, key_ids, factor_inf, factor_meas_val[0:3], device=self.device) if (factor_inf.requires_grad) else factor_cost
                elif (factor_name == 'ee_pose_prior'):
                    factor_meas_val = torch.tensor(factor_meas[idx], device=self.device)                    
                    factor_inf = torch.diag(self.theta.get_sigma_inv(factor_name, z=factor_meas_val)) ** 2
                    factor_cost = unary_factor_cost(x, key_syms, key_ids, factor_inf, factor_meas_val, device=self.device) if (factor_inf.requires_grad) else factor_cost
                elif (factor_name == 'qs_push_motion'):
                    meta = qs_motion_factor_metadata(self.params.dataio.obj_shape)
                    factor_inf = torch.diag(self.theta.get_sigma_inv(factor_name, z=None)) ** 2
                    factor_cost = qs_motion_factor_cost(x, key_syms, key_ids, factor_inf, device=self.device, meta=meta) if (factor_inf.requires_grad) else factor_cost
                elif (factor_name == 'sdf_motion'):
                    factor_inf = torch.diag(self.theta.get_sigma_inv(factor_name, z=None)) ** 2
                    factor_cost = sdf_intersection_factor_cost(x, key_syms, key_ids, factor_inf) if (factor_inf.requires_grad) else factor_cost
                elif (factor_name == 'tactile_rel'):
                    if (self.params.tactile_model.oracle == True):
                        tf_pred = tactile_oracle_output(self.data, key_syms, key_ids, device=self.device)
                    else:
                        tf_pred = tactile_model_output(tactile_model, img_feats=factor_meas[idx], meta=self.params.tactile_model)
                    factor_inf = torch.diag(self.theta.get_sigma_inv(factor_name, z=None)) ** 2
                    factor_cost = tactile_rel_meas_factor_cost(x, key_syms, key_ids, factor_inf, tf_pred, device=self.device) if (factor_inf.requires_grad) else factor_cost

                fval = fval + 0.5 * factor_cost

        return fval

    def grad_theta_costfn(self, x):

        for name, param in self.theta.named_parameters():
            if param.data.grad is not None:
                param.data.grad = None
        
        fval = self.costfn(x)
        fval.backward()

        grad_theta = AttrDict()
        for name, param in self.theta.named_parameters():
            grad_theta[name] = param.data.grad

        return grad_theta
