import math
import numpy as np

import os
import json
import hydra
from datetime import datetime
from attrdict import AttrDict
from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from functools import partial

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_theta(params):
    
    if (params.dataio.dataset_type == "nav2d"):
        if (params.dataio.model_type == "fixed_cov"):
            sigma_noise = params.theta_init.sigma_noise if (params.theta_init.noisy == True) else 0. 
            sigma_inv_odom_vals = np.array(params.theta_init.sigma_inv_odom_vals) + sigma_noise * np.random.randn(3)
            sigma_inv_gps_vals = np.array(params.theta_init.sigma_inv_gps_vals) + 0. * np.random.randn(3)
            theta = ThetaNav2dFixedCov(sigma_inv_odom_vals=sigma_inv_odom_vals,
                                       sigma_inv_gps_vals=sigma_inv_gps_vals)
            theta_exp = ThetaNav2dFixedCov(sigma_inv_odom_vals=params.theta_exp.sigma_inv_odom_vals,
                                           sigma_inv_gps_vals=params.theta_exp.sigma_inv_gps_vals)
        elif (params.dataio.model_type == "varying_cov"):
            sigma_noise = params.theta_init.sigma_noise if (params.theta_init.noisy == True) else 0. 
            sigma_inv_odom0_vals = np.array(params.theta_init.sigma_inv_odom0_vals) + sigma_noise * np.random.randn(3)
            sigma_inv_gps0_vals = np.array(params.theta_init.sigma_inv_gps0_vals) + sigma_noise * np.random.randn(3)
            sigma_inv_odom1_vals = np.array(params.theta_init.sigma_inv_odom1_vals) + sigma_noise * np.random.randn(3)
            sigma_inv_gps1_vals = np.array(params.theta_init.sigma_inv_gps1_vals) + sigma_noise * np.random.randn(3)
            
            theta = ThetaNav2dVaryingCov(sigma_inv_odom0_vals=sigma_inv_odom0_vals,
                                         sigma_inv_gps0_vals=sigma_inv_gps0_vals,
                                         sigma_inv_odom1_vals=sigma_inv_odom1_vals,
                                         sigma_inv_gps1_vals=sigma_inv_gps1_vals)
            theta_exp = ThetaNav2dVaryingCov(sigma_inv_odom0_vals=params.theta_exp.sigma_inv_odom0_vals,
                                             sigma_inv_gps0_vals=params.theta_exp.sigma_inv_gps0_vals,
                                             sigma_inv_odom1_vals=params.theta_exp.sigma_inv_odom1_vals,
                                             sigma_inv_gps1_vals=params.theta_exp.sigma_inv_gps1_vals)

    elif (params.dataio.dataset_type == "push2d"):
        if (params.dataio.model_type == "fixed_cov"):
            theta = ThetaPush2dFixedCov(sigma_inv_tactile_rel_vals=params.theta_init.sigma_inv_tactile_rel_vals,
                                        sigma_inv_qs_push_motion_vals=params.theta_init.sigma_inv_qs_push_motion_vals,
                                        sigma_inv_ee_pose_prior_vals=params.theta_init.sigma_inv_ee_pose_prior_vals,
                                        sigma_inv_sdf_intersection_vals=params.theta_init.sigma_inv_sdf_intersection_vals,
                                        sigma_inv_binary_interseq_obj_vals=params.theta_init.sigma_inv_binary_interseq_obj_vals)
            theta_exp = ThetaPush2dFixedCov(sigma_inv_tactile_rel_vals=params.theta_exp.sigma_inv_tactile_rel_vals,
                                            sigma_inv_qs_push_motion_vals=params.theta_exp.sigma_inv_qs_push_motion_vals,
                                            sigma_inv_ee_pose_prior_vals=params.theta_exp.sigma_inv_ee_pose_prior_vals,
                                            sigma_inv_sdf_intersection_vals=params.theta_exp.sigma_inv_sdf_intersection_vals,
                                            sigma_inv_binary_interseq_obj_vals=params.theta_exp.sigma_inv_binary_interseq_obj_vals)
    else:
        print(
            "[logo_obs_models::init_theta] Observation model parameter class not found.")

    [print("[logo_obs_models::init_theta] theta: {0} {1}".format(
        name, param)) for name, param in theta.named_parameters()]
    [print("[logo_obs_models::init_theta] theta_exp: {0} {1}".format(
        name, param)) for name, param in theta_exp.named_parameters()]

    return theta, theta_exp


def min_clip(w):
    w_min = 1e-1 * torch.ones(w.shape, requires_grad=True, device=device)
    w = torch.max(w_min, w)
    return w


class ThetaNav2dFixedCov(nn.Module):
    def __init__(self, sigma_inv_odom_vals=None, sigma_inv_gps_vals=None):
        super().__init__()

        self.sigma_inv_odom = torch.nn.Parameter(torch.tensor(
            sigma_inv_odom_vals, dtype=torch.float32, device=device)) if sigma_inv_odom_vals is not None else torch.nn.Parameter(torch.tensor([10., 10., 10.], device=device))

        self.sigma_inv_gps = torch.nn.Parameter(torch.tensor(
            sigma_inv_gps_vals, dtype=torch.float32, device=device)) if sigma_inv_gps_vals is not None else torch.nn.Parameter(torch.tensor([10., 10., 10.], device=device))

        self.freeze_params()

    def freeze_params(self):
        pass

    def get_sigma_inv(self, factor_name, z=None):
        sigma_inv_val = getattr(self, "sigma_inv_{0}".format(factor_name))

        return sigma_inv_val

    def min_clip(self):
        self.sigma_inv_odom.data = min_clip(self.sigma_inv_odom.data)
        self.sigma_inv_gps.data = min_clip(self.sigma_inv_gps.data)


class ThetaNav2dVaryingCov(nn.Module):
    def __init__(self, sigma_inv_odom0_vals=None, sigma_inv_gps0_vals=None, sigma_inv_odom1_vals=None, sigma_inv_gps1_vals=None):
        super().__init__()

        self.sigma_inv_odom0 = torch.nn.Parameter(torch.tensor(
            sigma_inv_odom0_vals, dtype=torch.float32, device=device)) if sigma_inv_odom0_vals is not None else torch.nn.Parameter(torch.tensor([10., 10., 10.], device=device))
        self.sigma_inv_gps0 = torch.nn.Parameter(torch.tensor(
            sigma_inv_gps0_vals, dtype=torch.float32, device=device)) if sigma_inv_gps0_vals is not None else torch.nn.Parameter(torch.tensor([10., 10., 10.], device=device))

        self.sigma_inv_odom1 = torch.nn.Parameter(torch.tensor(
            sigma_inv_odom1_vals, dtype=torch.float32, device=device)) if sigma_inv_odom1_vals is not None else torch.nn.Parameter(torch.tensor([10., 10., 10.], device=device))
        self.sigma_inv_gps1 = torch.nn.Parameter(torch.tensor(
            sigma_inv_gps1_vals, dtype=torch.float32, device=device)) if sigma_inv_gps1_vals is not None else torch.nn.Parameter(torch.tensor([10., 10., 10.], device=device))

        self.freeze_params()

    def freeze_params(self):
        pass

    def get_sigma_inv(self, factor_name, z=None):
        sigma_inv_val0 = getattr(self, "sigma_inv_{0}0".format(factor_name))
        sigma_inv_val1 = getattr(self, "sigma_inv_{0}1".format(factor_name))

        cl = z[-1]
        sigma_inv_val = (1 - cl) * sigma_inv_val0 + cl * sigma_inv_val1

        return sigma_inv_val

    def min_clip(self):
        self.sigma_inv_odom0.data = min_clip(self.sigma_inv_odom0.data)
        self.sigma_inv_gps0.data = min_clip(self.sigma_inv_gps0.data)

        self.sigma_inv_odom1.data = min_clip(self.sigma_inv_odom1.data)
        self.sigma_inv_gps1.data = min_clip(self.sigma_inv_gps1.data)


class ThetaPush2dFixedCov(nn.Module):
    def __init__(self, sigma_inv_tactile_rel_vals=None, sigma_inv_qs_push_motion_vals=None,
                 sigma_inv_ee_pose_prior_vals=None, sigma_inv_sdf_intersection_vals=None, sigma_inv_binary_interseq_obj_vals=None):
        super().__init__()

        self.sigma_inv_tactile_rel = torch.nn.Parameter(torch.tensor(
            sigma_inv_tactile_rel_vals, device=device)) if sigma_inv_tactile_rel_vals is not None else torch.nn.Parameter(torch.tensor([10., 10., 10.], device=device))
        self.sigma_inv_qs_push_motion = torch.nn.Parameter(torch.tensor(
            sigma_inv_qs_push_motion_vals, device=device)) if sigma_inv_qs_push_motion_vals is not None else torch.nn.Parameter(torch.tensor([10., 10., 10.], device=device))

        self.sigma_inv_ee_pose_prior = torch.nn.Parameter(torch.tensor(
            sigma_inv_ee_pose_prior_vals, device=device)) if sigma_inv_ee_pose_prior_vals is not None else torch.nn.Parameter(torch.tensor([10., 10., 10.], device=device))
        self.sigma_inv_sdf_intersection = torch.nn.Parameter(torch.tensor(
            sigma_inv_sdf_intersection_vals, device=device)) if sigma_inv_sdf_intersection_vals is not None else torch.nn.Parameter(torch.tensor([10., 10., 10.], device=device))
        self.sigma_inv_binary_interseq_obj = torch.nn.Parameter(torch.tensor(
            sigma_inv_binary_interseq_obj_vals, device=device)) if sigma_inv_binary_interseq_obj_vals is not None else torch.nn.Parameter(torch.tensor([10., 10., 10.], device=device))

        self.freeze_params()

    def freeze_params(self):
        self.sigma_inv_ee_pose_prior.requires_grad = False
        self.sigma_inv_sdf_intersection.requires_grad = False
        self.sigma_inv_binary_interseq_obj.requires_grad = False

    def get_sigma_inv(self, factor_name, z=None):
        sigma_inv_val = getattr(self, "sigma_inv_{0}".format(factor_name))

        return sigma_inv_val

    def min_clip(self):
        self.sigma_inv_tactile_rel.data = min_clip(
            self.sigma_inv_tactile_rel.data)
        self.sigma_inv_qs_push_motion.data = min_clip(
            self.sigma_inv_qs_push_motion.data)
