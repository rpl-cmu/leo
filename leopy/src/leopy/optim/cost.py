import sys
sys.path.append("/usr/local/cython/")

import math
import numpy as np
from scipy.optimize import minimize

from attrdict import AttrDict

import torch
from torch.autograd import Variable
import collections
import pandas as pd

from leopy.utils import tf_utils, dir_utils
from leopy.algo.leo_obs_models import TactileModelNetwork

import logging
log = logging.getLogger(__name__)

## factor costs in pytorch ##

def whiten_factor_error(factor_error, factor_inf):
    # factor_error: n x 1, factor_inf: n x n

    factor_cost = torch.matmul(torch.matmul(
        factor_error.permute(1, 0), factor_inf), factor_error)
    factor_cost = factor_cost.view(-1)

    return factor_cost

def unary_factor_error(x, key_syms, key_ids, factor_inf, factor_meas, device=None, params=None):

    key_id = key_ids[0]
    key_id = key_id + int(0.5 * x.shape[0]) if (key_syms[0] == 'e') else key_id

    est_pose = (x[key_id, :]).view(1, -1)
    meas_pose = factor_meas.view(1, -1)

    err = tf_utils.tf2d_between(est_pose, meas_pose, device=device, requires_grad=True).view(-1, 1) # 3 x 1

    return err

def binary_odom_factor_error(x, key_syms, key_ids, factor_inf, factor_meas, device=None, params=None):
    
    p1 = (x[key_ids[0], :]).view(1, -1) # n x 3
    p2 = (x[key_ids[1], :]).view(1, -1) # n x 3

    est_val = (tf_utils.tf2d_between(p1, p2, device=device, requires_grad=True)).view(-1)

    # err = (torch.sub(est_val, factor_meas)).view(-1, 1) # 3 x 1
    est_val = est_val.view(1, -1)
    factor_meas = factor_meas.view(1, -1)
    err = (tf_utils.tf2d_between(factor_meas, est_val, device=device, requires_grad=True)).view(-1, 1)

    return err

def sdf_intersection_factor_error(x, key_syms, key_ids, factor_inf, device=None, params=None):

    err = torch.tensor([[0.]], device=device)

    return err

def qs_motion_factor_params(obj_shape):

    params = AttrDict()

    if (obj_shape == 'disc'):
        params.c_sq = math.pow(0.088 / 3, 2)
    elif (obj_shape == 'rect'):
        params.c_sq = math.pow(math.sqrt(0.2363**2 + 0.1579**2), 2)
    elif (obj_shape == 'ellip'):
        params.c_sq = (0.5 * (0.1638 + 0.2428)) ** 2
    else:
        print("object shape sdf not found")

    return params

def qs_motion_factor_error(x, key_syms, key_ids, factor_inf, device=None, params=None):
    # keys: o_{t-1}, o_{t}, e_{t-1}, e_{t}
    
    offset_e = int(0.5 * x.shape[0])

    obj_pose0 = (x[key_ids[0], :]).view(1, -1) # 1 x 3
    obj_pose1 = (x[key_ids[1], :]).view(1, -1)
    ee_pose0 = (x[offset_e + key_ids[2], :]).view(1, -1) # 1 x 3
    ee_pose1 = (x[offset_e + key_ids[3], :]).view(1, -1)

    obj_ori1 = obj_pose1.clone(); obj_ori1[0, 0] = 0.; obj_ori1[0, 1] = 0.
    obj_pose_rel__world = tf_utils.tf2d_between(obj_pose0, obj_pose1, device=device, requires_grad=True)

    vel_obj__world = obj_pose1.clone() - obj_pose0.clone(); vel_obj__world[0, 2] = 0.
    vel_obj__obj = tf_utils.tf2d_between(obj_ori1, vel_obj__world, device=device, requires_grad=True)

    vel_contact__world = ee_pose1.clone() - ee_pose0.clone(); vel_contact__world[0, 2] = 0.
    vel_contact__obj = tf_utils.tf2d_between(obj_ori1, vel_contact__world, device=device, requires_grad=True)
    
    contact_point1 = ee_pose1.clone(); contact_point1[0, 2] = 0.
    contact_point__obj = tf_utils.tf2d_between(obj_pose1, contact_point1, device=device, requires_grad=True)

    # # D*V = Vp
    vx = vel_obj__obj[0, 0]
    vy = vel_obj__obj[0, 1]
    omega = obj_pose_rel__world[0, 2]

    vpx = vel_contact__obj[0, 0]
    vpy = vel_contact__obj[0, 1]

    px = contact_point__obj[0, 0]
    py = contact_point__obj[0, 1]

    D = torch.tensor([[1, 0, -py], [0, 1, px], [-py, px, -params.c_sq]], device=device)
    V = torch.tensor([vx, vy, omega], device=device)
    Vp = torch.tensor([vpx, vpy, 0.], device=device)

    err = torch.sub(torch.matmul(D, V), Vp).view(-1, 1) # 3 x 1

    return err

def tactile_rel_meas_factor_error(x, key_syms, key_ids, factor_inf, factor_meas, device=None, params=None):
    # keys: o_{t-k}, o_{t}, e_{t-k}, e_{t}
        
    offset_e = int(0.5 * x.shape[0])

    obj_pose1 = (x[key_ids[0], :]).view(1, -1) # 1 x 3
    obj_pose2 = (x[key_ids[1], :]).view(1, -1)
    ee_pose1 = (x[offset_e + key_ids[2], :]).view(1, -1) # 1 x 3
    ee_pose2 = (x[offset_e + key_ids[3], :]).view(1, -1)

    ee_pose1__obj = tf_utils.tf2d_between(obj_pose1, ee_pose1, device=device, requires_grad=True) # 1 x 3
    ee_pose2__obj = tf_utils.tf2d_between(obj_pose2, ee_pose2, device=device, requires_grad=True) # 1 x 3
    pose_rel_expect = tf_utils.tf2d_between(ee_pose1__obj, ee_pose2__obj, device=device, requires_grad=True) # 1 x 3

    yaw_only_error = True
    if yaw_only_error:
        yaw_only_mask = torch.zeros(3, 4, device=device)
        yaw_only_mask[-1,-1] = 1.
        pose_rel_meas = torch.asin(torch.matmul(yaw_only_mask, factor_meas)).view(1, -1) # 1 x 3
        pose_rel_expect = torch.tensor([0., 0., pose_rel_expect[0, 2]], requires_grad=True, device=device).view(1, -1) # 1 x 3

    err = tf_utils.tf2d_between(pose_rel_expect, pose_rel_meas, device=device, requires_grad=True).view(-1, 1) # 3 x 1

    return err

def tactile_model_output_fixed(tactile_model, img_feats, device=None, params=None):
    img_feat_i = torch.tensor(img_feats[0], requires_grad=True, device=device).view(1, -1)
    img_feat_j = torch.tensor(img_feats[1], requires_grad=True, device=device).view(1, -1)

    if (params.norm_img_feat == True):
        mean_img_feat = torch.tensor(params.mean_img_feat, requires_grad=True, device=device)
        std_img_feat = torch.tensor(params.std_img_feat, requires_grad=True, device=device)

        img_feat_i = tf_utils.normalize_vector(img_feat_i, mean_img_feat, std_img_feat)
        img_feat_j = tf_utils.normalize_vector(img_feat_j, mean_img_feat, std_img_feat)

    class_label = torch.tensor(params.class_label, device=device)
    class_label_vec = torch.nn.functional.one_hot(class_label, params.num_classes)
    class_label_vec = class_label_vec.view(1, -1)

    tf_pred = tactile_model.forward(img_feat_i, img_feat_j, class_label_vec)
    tf_pred = tf_pred.view(-1) # [tx, ty, cyaw, syaw]

    return tf_pred

def tactile_oracle_output(data, key_syms, key_ids, device=None, params=None):
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

def init_tactile_model(filename=None, device=None):

    tactile_model = TactileModelNetwork(input_size=2*2*4, output_size=4)
    tactile_model = tactile_model.to(device)
    
    if filename is not None:
        model_saved = torch.jit.load(filename)
        state_dict_saved = model_saved.state_dict()

        # fix: saved dict key mismatch
        state_dict_new = collections.OrderedDict()
        state_dict_new['fc1.weight'] = state_dict_saved['model.fc1.weight']
        state_dict_new['fc1.bias'] = state_dict_saved['model.fc1.bias']

        tactile_model.load_state_dict(state_dict_new)
    
    return tactile_model

def update_data_dict(data_dict, factor_name, factor_cost, factor_inf):
    factor_key = f"err/factor/{factor_name}"

    if factor_key not in data_dict:
        data_dict[factor_key] = [factor_cost.item()]
    else:
        (data_dict[factor_key]).append(factor_cost.item())

    return data_dict

def dict_to_row(data_dict, step):

    for key, val in data_dict.items():
        data_dict[key] = sum(val) / len(val)

    # data_dict['tstep'] = step
    data_row = pd.DataFrame(data_dict, index=[step])
    data_row.index.name = 'tstep'

    return data_row
class Cost():
    def __init__(self, data, theta, params=None, device=None):
        self.data = data
        self.theta = theta
        self.params = params
        self.device = device

        self.dataframe = None
                
    def get_dataframe(self):
        return self.dataframe

    def costfn(self, x, log=False):

        dataset_type = self.params.dataio.dataset_type

        if (dataset_type == "push2d"):
            nsteps = np.minimum(self.params.optim.nsteps, len(self.data['obj_poses_gt']))
            tactile_model_filename = "{0}/local/models/{1}.pt".format(self.params.BASE_PATH, self.params.tactile_model.name)
            tactile_model = init_tactile_model(filename=tactile_model_filename, device=self.device)
        else:
            nsteps = np.minimum(self.params.optim.nsteps, len(self.data['poses_gt']))
        
        if log: self.dataframe = pd.DataFrame()

        fval = torch.tensor([0.], requires_grad=True, device=self.device)

        for tstep in range(1, nsteps):

            # filter out curr step factors
            factor_idxs = [idxs for idxs, keys in enumerate(
                self.data['factor_keyids']) if (max(keys) == tstep)]

            factor_keysyms = [self.data['factor_keysyms'][idx] for idx in factor_idxs]
            factor_keyids = [self.data['factor_keyids'][idx] for idx in factor_idxs]
            factor_meas = [self.data['factor_meas'][idx] for idx in factor_idxs]
            factor_names = [self.data['factor_names'][idx] for idx in factor_idxs]
            
            # compute factor costs
            data_dict = {}
            for idx in range(0, len(factor_idxs)):

                key_syms, key_ids = factor_keysyms[idx], factor_keyids[idx]
                factor_name = factor_names[idx]

                if (factor_name == 'gps'):
                    factor_meas_val = torch.tensor(factor_meas[idx], device=self.device)
                    factor_inf = torch.diag(self.theta.get_sigma_inv(factor_name, z=factor_meas_val)) ** 2
                    factor_error = unary_factor_error(x, key_syms, key_ids, factor_inf, factor_meas_val[0:3], device=self.device) if (factor_inf.requires_grad) else None
                elif (factor_name == 'odom'):
                    factor_meas_val = torch.tensor(factor_meas[idx], device=self.device)                    
                    factor_inf = torch.diag(self.theta.get_sigma_inv(factor_name, z=factor_meas_val)) ** 2
                    factor_error = binary_odom_factor_error(x, key_syms, key_ids, factor_inf, factor_meas_val[0:3], device=self.device) if (factor_inf.requires_grad) else None
                elif (factor_name == 'ee_pose_prior'):
                    factor_meas_val = torch.tensor(factor_meas[idx], device=self.device)                    
                    factor_inf = torch.diag(self.theta.get_sigma_inv(factor_name, z=factor_meas_val)) ** 2
                    factor_error = unary_factor_error(x, key_syms, key_ids, factor_inf, factor_meas_val, device=self.device) if (factor_inf.requires_grad) else None
                elif (factor_name == 'qs_push_motion'):
                    params_qs = qs_motion_factor_params(self.params.dataio.obj_shape)
                    factor_inf = torch.diag(self.theta.get_sigma_inv(factor_name, z=None)) ** 2
                    factor_error = qs_motion_factor_error(x, key_syms, key_ids, factor_inf, device=self.device, params=params_qs) if (factor_inf.requires_grad) else None
                elif (factor_name == 'sdf_motion'):
                    factor_inf = torch.diag(self.theta.get_sigma_inv(factor_name, z=None)) ** 2
                    factor_error = sdf_intersection_factor_error(x, key_syms, key_ids, factor_inf) if (factor_inf.requires_grad) else None
                elif (factor_name == 'tactile_rel'):
                    if (self.params.tactile_model.oracle == True):
                        tf_pred = tactile_oracle_output(self.data, key_syms, key_ids, device=self.device)
                    else:
                        if (self.params.dataio.model_type == "fixed_cov_varying_mean"):
                            tf_pred = self.theta.tactile_model_output(img_feats=factor_meas[idx], params=self.params.tactile_model) # learnable model weights
                        else:
                            tf_pred = tactile_model_output_fixed(tactile_model, img_feats=factor_meas[idx], device=self.device, params=self.params.tactile_model) # fixed model weights

                    factor_inf = torch.diag(self.theta.get_sigma_inv(factor_name, z=None)) ** 2
                    factor_error = tactile_rel_meas_factor_error(x, key_syms, key_ids, factor_inf, tf_pred, device=self.device) # if (factor_inf.requires_grad) else factor_error
                
                factor_cost = whiten_factor_error(factor_error, factor_inf) if (factor_error is not None) else torch.tensor([0.], requires_grad=True, device=self.device)
                fval = fval + 0.5 * factor_cost

                factor_cost_unwhtn = torch.matmul(factor_error.permute(1, 0), factor_error) if (factor_error is not None) else torch.tensor([0.], requires_grad=True, device=self.device)
                if log: data_dict = update_data_dict(data_dict, factor_name=factor_name, factor_cost=factor_cost_unwhtn, factor_inf=factor_inf)
            
            if log: data_row = dict_to_row(data_dict=data_dict, step=tstep)
            if log: self.dataframe = self.dataframe.append(data_row)
        
        fval = fval / self.params.leo.norm_loss

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
